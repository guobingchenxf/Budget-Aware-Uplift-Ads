"""模型：响应模型基线 + 经典 uplift 学习器（S/T/X-learner + 类别变换）。

统一接口：
    model.fit(X, t, y)
    model.predict(X) -> score
其中 score 的语义由 `score_kind` 决定：
    - "response"：预测处理后的转化概率 P(y=1|x, t=1)
    - "uplift"  ：预测增量 τ(x) = P(y=1|x,t=1) - P(y=1|x,t=0)

实现说明（避免"堆模型名"）：
- 全部使用 LightGBM（CPU 友好、表格数据强），没有任何 GPU 依赖。
- 所有学习器都显式写出其识别假设与局限，见各类 docstring。
- X-learner 的加权组合严格按 Künzel et al. (2019) 的 g(x)=propensity 形式实现；
  另提供 weight_mode="balanced" 的等权变体，便于对照。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import ModelConfig, get_logger

logger = get_logger()

try:  # 允许在缺少 lightgbm 时给出清晰错误，而不是 ImportError 堆栈
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False


def _require_lgb() -> None:
    if not _HAS_LGB:
        raise RuntimeError("需要 lightgbm。请使用项目 .venv 安装 requirements.txt")


class _LGBRegressor:
    """薄封装：统一训练参数、静默训练日志、固定随机种子。"""

    def __init__(self, cfg: ModelConfig, seed: int, objective: str = "regression"):
        _require_lgb()
        self.cfg = cfg
        self.seed = seed
        self.objective = objective
        self.model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X, y, categorical_feature=None):
        params: Dict[str, object] = dict(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            min_child_samples=self.cfg.min_child_samples,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            random_state=self.seed,
            n_jobs=self.cfg.n_jobs,
            verbose=-1,
        )
        if self.objective == "binary":
            self.model = lgb.LGBMClassifier(objective="binary", **params)
        else:
            self.model = lgb.LGBMRegressor(objective="regression", **params)
        self.model.fit(X, y, categorical_feature=categorical_feature or "auto")
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        if self.objective == "binary":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


class UpliftModel:
    """所有策略模型的基类。"""

    name = "base"
    score_kind = "uplift"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "UpliftModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # 便捷属性
    @property
    def is_uplift(self) -> bool:
        return self.score_kind == "uplift"


class RandomPolicy(UpliftModel):
    """随机分配：对照用的下界（不是模型）。"""

    name = "random"
    score_kind = "uplift"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, t, y):
        return self

    def predict(self, X) -> np.ndarray:
        return self.rng.random(len(X))


class ResponseModel(UpliftModel):
    """响应模型（response model）：只在处理组上训练 P(y=1|x, t=1)。

    这是工业界最常见的"按预测转化概率排序"策略。
    它**不考虑**"不投放会怎样"，因此在存在大量自然转化
    （即 P(y=1|x,t=0) 也很高）的人群上会做出错误决策——
    这正是本项目要对比的核心问题。
    """

    name = "response"
    score_kind = "response"

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._m = _LGBRegressor(cfg, seed, objective="binary")

    def fit(self, X, t, y):
        mask = np.asarray(t) == 1
        self._m.fit(X.loc[mask] if hasattr(X, "loc") else X[mask],
                    np.asarray(y)[mask])
        return self

    def predict(self, X) -> np.ndarray:
        return self._m.predict(X)


class SLearner(UpliftModel):
    """S-learner：单一模型把 treatment 当作普通特征，τ(x)=f(x,1)-f(x,0)。

    优点：样本利用率高、实现最简单。
    风险：treatment 只有一个二值特征，树模型容易低估其重要性，
          在 treatment 效应弱时会被其他特征"淹没"。
    """

    name = "s_learner"
    score_kind = "uplift"

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._m = _LGBRegressor(cfg, seed, objective="binary")

    @staticmethod
    def _with_t(X: pd.DataFrame, t) -> pd.DataFrame:
        Xt = X.copy()
        Xt["_treatment"] = np.asarray(t, dtype=float)
        return Xt

    def fit(self, X, t, y):
        self._m.fit(self._with_t(X, t), np.asarray(y))
        return self

    def predict(self, X) -> np.ndarray:
        ones = np.ones(len(X))
        zeros = np.zeros(len(X))
        return self._m.predict(self._with_t(X, ones)) - self._m.predict(self._with_t(X, zeros))


class TLearner(UpliftModel):
    """T-learner：处理组/对照组各训一个模型，τ(x)=μ1(x)-μ0(x)。

    优点：两端各自拟合，灵活。
    风险：两个模型误差不相消；当某一臂样本少时估计不稳。
    """

    name = "t_learner"
    score_kind = "uplift"

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._m1 = _LGBRegressor(cfg, seed, objective="binary")
        self._m0 = _LGBRegressor(cfg, seed + 1, objective="binary")

    def fit(self, X, t, y):
        t = np.asarray(t)
        y = np.asarray(y)
        idx = np.arange(len(t))
        self._m1.fit(X.loc[idx[t == 1]] if hasattr(X, "loc") else X[t == 1], y[t == 1])
        self._m0.fit(X.loc[idx[t == 0]] if hasattr(X, "loc") else X[t == 0], y[t == 0])
        return self

    def predict(self, X) -> np.ndarray:
        return self._m1.predict(X) - self._m0.predict(X)


class XLearner(UpliftModel):
    """X-learner（Künzel et al., 2019, PNAS）。

    四步：
      1) 拟合 μ0, μ1；
      2) 为处理组插补 D1 = y - μ0(x)，为对照组插补 D0 = μ1(x) - y；
      3) 分别拟合 τ1 = E[D1|x, t=1]（处理组）、τ0 = E[D0|x, t=0]（对照组）；
      4) 组合 τ(x) = g(x)·τ0(x) + (1-g(x))·τ1(x)，g(x) 取倾向得分。

    适用场景：两臂样本量相差较大时通常优于 T-learner。
    局限：仍然依赖"无未观测混杂"；插补误差会在第 3 步被放大。
    """

    name = "x_learner"
    score_kind = "uplift"

    def __init__(self, cfg: ModelConfig, seed: int = 0, weight_mode: str = "propensity"):
        self.cfg = cfg
        self.seed = seed
        self.weight_mode = weight_mode
        self._mu1 = _LGBRegressor(cfg, seed, objective="binary")
        self._mu0 = _LGBRegressor(cfg, seed + 1, objective="binary")
        self._tau1 = _LGBRegressor(cfg, seed + 2, objective="regression")
        self._tau0 = _LGBRegressor(cfg, seed + 3, objective="regression")
        self.propensity_: float = 0.5

    def fit(self, X, t, y):
        t = np.asarray(t)
        y = np.asarray(y)
        self.propensity_ = float(np.mean(t))

        idx = np.arange(len(t))
        Xtr1 = X.loc[idx[t == 1]] if hasattr(X, "loc") else X[t == 1]
        Xtr0 = X.loc[idx[t == 0]] if hasattr(X, "loc") else X[t == 0]

        self._mu1.fit(Xtr1, y[t == 1])
        self._mu0.fit(Xtr0, y[t == 0])

        d1 = y[t == 1] - self._mu0.predict(Xtr1)
        d0 = self._mu1.predict(Xtr0) - y[t == 0]
        self._tau1.fit(Xtr1, d1)
        self._tau0.fit(Xtr0, d0)
        return self

    def predict(self, X) -> np.ndarray:
        t1 = self._tau1.predict(X)
        t0 = self._tau0.predict(X)
        if self.weight_mode == "balanced":
            g = 0.5
        else:
            g = self.propensity_
        return g * t0 + (1.0 - g) * t1


class ClassTransform(UpliftModel):
    """类别变换法（Lai 等提出的 z-transformation 思路）。

    定义 z = 1 当 (t=1 且 y=1) 或 (t=0 且 y=0)，否则 z = 0。
    在**两臂等量**（P(t=1)=0.5）且随机化成立时：
        P(z=1|x) = 0.5·[P(y=1|x,t=1) + P(y=1|x,t=0)]
    于是 τ(x) = 2·P(z=1|x) - 1。

    ⚠️ 本实现在训练前会**对多数臂做下采样使两臂等量**，因为上述恒等式
    只在等量时成立；这是本项目的工程处理，已在实验报告中标注。
    """

    name = "class_transform"
    score_kind = "uplift"

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._m = _LGBRegressor(cfg, seed, objective="binary")
        self.balanced_n_: int = 0

    def fit(self, X, t, y):
        t = np.asarray(t).astype(int)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)

        idx1 = np.flatnonzero(t == 1)
        idx0 = np.flatnonzero(t == 0)
        n = min(len(idx1), len(idx0))
        sel1 = rng.choice(idx1, size=n, replace=False)
        sel0 = rng.choice(idx0, size=n, replace=False)
        sel = np.concatenate([sel1, sel0])
        rng.shuffle(sel)
        self.balanced_n_ = int(n)

        z = np.where((t[sel] == 1) & (y[sel] == 1), 1.0,
                     np.where((t[sel] == 0) & (y[sel] == 0), 1.0, 0.0))
        Xs = X.loc[sel] if hasattr(X, "loc") else X[sel]
        self._m.fit(Xs, z)
        return self

    def predict(self, X) -> np.ndarray:
        return 2.0 * self._m.predict(X) - 1.0


def build_model(name: str, cfg: ModelConfig, seed: int) -> UpliftModel:
    """工厂：按名称构建策略模型。"""
    name = name.lower()
    if name == "random":
        return RandomPolicy(seed)
    if name == "response":
        return ResponseModel(cfg, seed)
    if name == "s_learner":
        return SLearner(cfg, seed)
    if name == "t_learner":
        return TLearner(cfg, seed)
    if name == "x_learner":
        return XLearner(cfg, seed)
    if name == "class_transform":
        return ClassTransform(cfg, seed)
    raise ValueError(f"未知策略: {name}")


def expected_uplift_note() -> str:
    return ("τ(x) 是在无未观测混杂假设下的估计量；"
            "Hillstrom 为随机实验，该假设成立，但结果是"
            "'营销邮件触达'的增量，不等价于真实广告曝光效果。")
