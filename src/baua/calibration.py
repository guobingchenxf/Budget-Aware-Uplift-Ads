"""概率校准与阈值策略。

背景（为什么需要这个模块）：
    广告系统里预估值不只是用来排序，还要**乘钱**（eCPM、出价）或**过阈值**
    （"预测转化率低于 X% 的用户不投"）。这两类用法对"数值尺度"敏感，
    而排序对尺度不敏感——单调变换不改变顺序。

本模块要回答两个可检验的问题：
    Q1 校准能否改善**排序类**决策（top-k 预算分配）？
    Q2 校准能否改善**阈值类**决策（绝对阈值 → 触达规模）？

设计要点：
- 校准器**必须在独立的验证集上拟合**，绝不能用在评估集上拟合的校准器去评估——
  那是标签泄漏的一种。本模块通过 API 强制把 fit 与 apply 分开。
- 报告"校准前后 ECE"只是中间指标，**真正的结论要看策略价值**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import policy_gain


@dataclass
class IsotonicCalibrator:
    """单调校准器：把原始概率映射到校准后的概率。

    使用 sklearn 的 IsotonicRegression（分段单调），并用 out_of_bounds="clip"
    保证推理时不会因超出训练区间而报错。
    """

    model: Any = None
    n_fit: int = 0
    fitted: bool = False

    def fit(self, raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        raw = np.asarray(raw, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(raw) != len(y):
            raise ValueError("raw 与 y 长度不一致")
        if len(np.unique(y)) < 2:
            # 只有一个类别时无法拟合，退化为恒等映射（并记录 n_fit=0）
            self.model = None
            self.fitted = False
            self.n_fit = int(len(y))
            return self
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.clip(raw, 0.0, 1.0), y)
        self.model = iso
        self.fitted = True
        self.n_fit = int(len(y))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        if not self.fitted or self.model is None:
            return raw  # 未拟合时退化为恒等映射
        return np.asarray(self.model.predict(raw), dtype=float)


@dataclass
class SigmoidCalibrator:
    """Platt 校准：对分数的 logit 做一次线性变换，再过一个 sigmoid。

        p_cal = sigma(a * logit(p_raw) + b)

    与 isotonic 的关键差异（本项目的实测驱动，见实验报告 §4.6b）：
      - isotonic 是**分段常数**，会产生大量并列，破坏 top-k 排序；
      - Platt 是**严格单调**（当 a > 0 时），理论上**完全保留排序**，且不引入并列。
    因此它是"既要校准数值、又要保留排序分辨率"场景的正确选择。
    """

    coef_: float = 1.0
    intercept_: float = 0.0
    fitted: bool = False
    n_fit: int = 0
    eps: float = 1e-6

    def _logit(self, p: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=float), self.eps, 1.0 - self.eps)
        return np.log(q / (1.0 - q))

    def fit(self, raw: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        from sklearn.linear_model import LogisticRegression

        raw = np.asarray(raw, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(raw) != len(y):
            raise ValueError("raw 与 y 长度不一致")
        self.n_fit = int(len(y))
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        z = self._logit(raw).reshape(-1, 1)
        # C 取较大但有限值：接近最大似然，同时避免完全可分时系数发散
        lr = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        self.coef_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        z = self.coef_ * self._logit(raw) + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def threshold_policy_table(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """绝对阈值策略：选中 {i : score_i >= tau}，报告规模与该子集的增量收益。

    为什么绝对阈值有意义：业务规则通常是"预测转化率低于 8% 不投"这种**绝对值**，
    而不是"投前 5% 的人"。绝对阈值对校准敏感，排序对它不敏感。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    rows: list[dict[str, Any]] = []
    for name, sc in scores.items():
        sc = np.asarray(sc, dtype=float)
        if len(sc) != n:
            raise ValueError(f"策略 {name} 分数长度与样本数不一致")
        for tau in thresholds:
            sel = sc >= float(tau)
            g = policy_gain(y, t, sel)
            n_sel = int(sel.sum())
            rows.append({
                "strategy": name,
                "threshold": float(tau),
                "n_selected": n_sel,
                "selected_fraction": n_sel / n if n else float("nan"),
                "gain": g["gain"],
                "gain_per_selected": (g["gain"] / n_sel) if n_sel > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def compare_calibration_effect(
    raw: np.ndarray,
    calibrated: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """对比原始分数与校准分数在 top-k 选择上的差异。

    ⚠️ 实测发现（与"单调变换不改变排序"的朴素直觉**不完全一致**）：
    isotonic 是分段常数映射，会把一个区间内的所有原始分数压成**完全相同的值**。
    一旦这个平坦区间跨越 top-k 的截断边界，并列样本只能按索引任意打破，
    于是**校准后的 top-k 选择会与校准前不同**。

    这不是理论上的细枝末节：如果业务依赖"取前 N 个人"，
    校准会削弱截断点附近的区分度（代价是引入了并列，收益是数值可用）。
    本函数把这件事量化出来（`n_unique_*` 与 `overlap_ratio`），
    而不是假设它不发生。
    """
    from .metrics import policy_gain, select_top_k

    raw = np.asarray(raw, dtype=float)
    calibrated = np.asarray(calibrated, dtype=float)
    n = len(y)
    n_unique_raw = int(len(np.unique(raw)))
    n_unique_cal = int(len(np.unique(calibrated)))

    rows = []
    for k in (int(0.01 * n), int(0.05 * n), int(0.10 * n), int(0.20 * n)):
        if k <= 0:
            continue
        sel_raw = select_top_k(raw, k)
        sel_cal = select_top_k(calibrated, k)
        inter = int((sel_raw & sel_cal).sum())
        g_raw = policy_gain(y, t, sel_raw)["gain"]
        g_cal = policy_gain(y, t, sel_cal)["gain"]
        rows.append({
            "k": k,
            "overlap": inter,
            "overlap_ratio": inter / k,
            "identical": bool(inter == k),
            "gain_raw": g_raw,
            "gain_calibrated": g_cal,
            "gain_diff": g_cal - g_raw,
            "n_unique_raw": n_unique_raw,
            "n_unique_calibrated": n_unique_cal,
            "unique_ratio_calibrated": n_unique_cal / n,
            "ties_introduced": n_unique_raw - n_unique_cal,
        })
    return pd.DataFrame(rows)
