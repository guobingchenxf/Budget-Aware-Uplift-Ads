"""观测数据下的因果估计：IPS 与 DR（双重稳健）。

为什么需要这个模块：
    本项目的其余部分都建立在"数据来自随机实验"之上（Hillstrom 与默认半合成数据），
    此时 `policy_gain` 的 Qini 形式是无偏的。但**真实广告数据几乎都是观测数据**：
    曝光/投放由上一版策略决定，treatment 与用户特征强相关，于是

        Ĝ_naive(S) = Y_t(S) − Y_c(S)·N_t(S)/N_c(S)

    会有系统性偏差（处理组与对照组的基线本来就不同）。本模块回答一个问题：
    **换成 IPS / DR 之后，偏差能被纠正回多少？**

三种估计量（都对集合 S 的总增量 Σ_{i∈S} τ_i 做估计）：

    朴素     Ĝ_naive = Y_t(S) − Y_c(S)·N_t(S)/N_c(S)
    IPS      Ĝ_ips   = Σ_{i∈S} [ t_i y_i / ê_i − (1−t_i) y_i / (1−ê_i) ]
    DR       Ĝ_dr    = Σ_{i∈S} [ μ̂1(x_i) − μ̂0(x_i)
                                 + t_i (y_i − μ̂1(x_i)) / ê_i
                                 − (1−t_i)(y_i − μ̂0(x_i)) / (1−ê_i) ]

理论性质：
- IPS 在**倾向模型正确**时无偏；
- DR 在**倾向模型或结果模型之一正确**时无偏（双重稳健）；
- 两者都要求**重叠/正性**：0 < ê(x) < 1，且不能太接近 0/1，否则方差爆炸。

工程要点（本模块实现的关键细节）：
1. **交叉拟合（cross-fitting）**：倾向与结果模型都在 K 折中"用其他折训练、在本折预测"，
   避免用同一批数据既训练又评估导致的过拟合偏差；
2. **倾向截断**：把 ê 截断到 [clip_min, 1-clip_min]，并报告被截断的比例，
   因为极端倾向会让 IPS 方差失控——这是实践中必须监控的量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import policy_gain


@dataclass
class CausalEstimates:
    subset: str
    estimator: str
    value: float


def cross_fit_propensity(
    X: pd.DataFrame,
    t: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    clip_min: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """交叉拟合倾向得分 P(t=1|x)。返回 (ê, 诊断信息)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    t = np.asarray(t).astype(int)
    n = len(t)
    e_hat = np.zeros(n, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(np.zeros(n), t):
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        lr.fit(np.asarray(X.iloc[tr_idx]), t[tr_idx])
        e_hat[te_idx] = lr.predict_proba(np.asarray(X.iloc[te_idx]))[:, 1]

    n_clipped = int(((e_hat < clip_min) | (e_hat > 1 - clip_min)).sum())
    diag = {
        "propensity_mean": float(e_hat.mean()),
        "propensity_min": float(e_hat.min()),
        "propensity_max": float(e_hat.max()),
        "clipped_fraction": n_clipped / n,
    }
    return np.clip(e_hat, clip_min, 1.0 - clip_min), diag


def cross_fit_outcomes(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    n_estimators: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """交叉拟合两个结果模型 μ̂1(x)=E[y|x,t=1]、μ̂0(x)=E[y|x,t=0]。

    ⚠️ 实现要点（本项目踩过的坑，见实验报告 §5）：
    AIPW 需要在**每个样本**上同时取得 μ̂1(x_i) 与 μ̂0(x_i)（其中一个是反事实预测）。
    如果按"臂"划分交叉拟合、只填充本臂的位置，另一臂的位置会保持初始化值 0，
    于是 DR 估计量会用到大量恒为 0 的"预测"，结果严重有偏
    （本项目实测：DR 给出真值的一半，而 IPS 正常）。
    **正确做法是按行（而非按臂）做 K 折**：在每一折上，
    用其他折的处理组训练 μ̂1、其他折的对照组训练 μ̂0，
    再对**本折的全部样本**同时预测 μ̂1 与 μ̂0。
    """
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    Xarr = np.asarray(X)
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    n = len(t)
    mu1 = np.full(n, np.nan, dtype=float)
    mu0 = np.full(n, np.nan, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_rows, te_rows in kf.split(np.zeros(n)):
        for arm, out in ((1, mu1), (0, mu0)):
            fit_rows = tr_rows[t[tr_rows] == arm]
            if len(fit_rows) < 10:
                out[te_rows] = float(y[t == arm].mean()) if (t == arm).any() else 0.0
                continue
            m = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=0.05, num_leaves=15,
                min_child_samples=40, random_state=seed, n_jobs=2, verbose=-1)
            m.fit(Xarr[fit_rows], y[fit_rows])
            out[te_rows] = m.predict_proba(Xarr[te_rows])[:, 1]

    if np.isnan(mu1).any() or np.isnan(mu0).any():  # pragma: no cover - 兜底
        mu1 = np.nan_to_num(mu1, nan=float(y[t == 1].mean()))
        mu0 = np.nan_to_num(mu0, nan=float(y[t == 0].mean()))
    return mu1, mu0


def estimate_subset(
    y: np.ndarray,
    t: np.ndarray,
    selected: np.ndarray,
    e_hat: np.ndarray,
    mu1: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """对选中集合 S 估计其总增量收益 Σ_{i∈S} τ_i，并给出三种估计量。

    ⚠️ 口径说明（本项目实测发现的易混淆点）：
    教科书式的 Qini 估计量 Y_t(S) − Y_c(S)·N_t(S)/N_c(S) 的期望是
        n_t · mean_{i∈S}(τ_i)
    而不是 |S| · mean(τ_i)。在等量随机化下 n_t ≈ |S|/2，
    所以它大约是"对 S 中**全部**成员都投放"效果的一半。
    它估计的是"投放 S 中那些被处理者所贡献的增量"，在比较策略时是自洽的
    （共同因子相消），但与 IPS/DR 的口径不同，不能直接比大小。
    因此这里额外给出 `naive_full = |S|·(ȳ_t(S) − ȳ_c(S))`，
    它与 IPS/DR 一样以 Σ_{i∈S} τ_i 为目标，用于公平比较。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    sel = np.asarray(selected, dtype=bool)
    e = np.asarray(e_hat, dtype=float)

    qini_naive = policy_gain(y, t, sel)["gain"]

    ys, ts, es = y[sel], t[sel], e[sel]
    n_t, n_c = float(ts.sum()), float((1 - ts).sum())
    mean_t = float(ys[ts == 1].mean()) if n_t > 0 else float("nan")
    mean_c = float(ys[ts == 0].mean()) if n_c > 0 else float("nan")
    naive_full = float(len(ys)) * (mean_t - mean_c)

    ips = float((ts * ys / es - (1 - ts) * ys / (1 - es)).sum())

    out = {
        "n_selected": int(sel.sum()),
        "n_treated": int(n_t),
        "naive_qini": float(qini_naive),
        "naive_full": naive_full,
        "ips": ips,
        "dr": float("nan"),
    }
    if mu1 is not None and mu0 is not None:
        m1, m0 = np.asarray(mu1)[sel], np.asarray(mu0)[sel]
        dr = float((m1 - m0
                    + ts * (ys - m1) / es
                    - (1 - ts) * (ys - m0) / (1 - es)).sum())
        out["dr"] = dr
    return out


def compare_estimators_over_policies(
    y: np.ndarray,
    t: np.ndarray,
    true_cate: np.ndarray,
    scores: Dict[str, np.ndarray],
    budget_units: int,
    e_hat: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> pd.DataFrame:
    """对多个策略的 top-k 选择，比较三种估计量与**真值**的偏差。

    真值 G_true(S) = Σ_{i∈S} 真实 CATE —— 只有半合成数据能给出，
    这正是用它做验证的价值：可以量化每个估计量的绝对与相对偏差。
    """
    from .metrics import select_top_k

    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    true_cate = np.asarray(true_cate, dtype=float)
    rows: List[Dict[str, object]] = []
    for name, sc in scores.items():
        sel = select_top_k(np.asarray(sc, dtype=float), budget_units)
        est = estimate_subset(y, t, sel, e_hat, mu1, mu0)
        truth = float(true_cate[sel].sum())
        row = {"policy": name, "truth": truth, **est}
        # 只对"口径一致"的估计量计算偏差（Qini 式目标是 n_t·mean(tau)，不可直接比）
        for k in ("naive_full", "ips", "dr"):
            v = row[k]
            row[f"{k}_bias"] = float(v - truth)
            row[f"{k}_abs_err"] = abs(float(v - truth))
        row["naive_qini_ratio_to_truth"] = (
            float(row["naive_qini"] / truth) if abs(truth) > 1e-12 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def confounding_study(
    n: int = 40000,
    seed: int = 20260929,
    confounding_levels: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    budget_ratio: float = 0.05,
    uplift_strength: float = 3.0,
    mode: str = "conflicting",
) -> pd.DataFrame:
    """主实验：把选择偏差强度从 0（随机实验）逐步加到 2.0，观察三种估计量的偏差。

    用 uplift 模型（s_learner 的替身：真实 CATE 的带噪版本）与响应模型各选一次 top-k，
    分别报告三种估计量相对真值的偏差。
    """
    from .data import make_synthetic
    from .models import TLearner
    from .config import ModelConfig

    rows: List[Dict[str, object]] = []
    for gamma in confounding_levels:
        df, true_cate = make_synthetic(n=n, seed=seed, mode=mode,
                                       uplift_strength=uplift_strength,
                                       confounding=float(gamma))
        X = df[[c for c in df.columns if c.startswith("x")]]
        t = df["treatment"].values.astype(int)
        y = df["y"].values.astype(float)

        e_hat, e_diag = cross_fit_propensity(X, t, seed=seed)
        mu1, mu0 = cross_fit_outcomes(X, t, y, seed=seed)

        budget_units = int(n * budget_ratio)
        # 两个策略：按真实 CATE 排序（oracle）与按响应排序（用 x0 作为响应强弱的代理）
        scores = {
            "oracle_uplift": true_cate,
            "response_proxy": X["x0"].values.astype(float),
        }
        tab = compare_estimators_over_policies(
            y, t, true_cate, scores, budget_units, e_hat, mu1, mu0)
        tab.insert(0, "confounding", float(gamma))
        tab["propensity_clipped_fraction"] = e_diag["clipped_fraction"]
        tab["propensity_min"] = e_diag["propensity_min"]
        tab["propensity_max"] = e_diag["propensity_max"]
        rows.append(tab)

    return pd.concat(rows, ignore_index=True)
