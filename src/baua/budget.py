from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import auuc, policy_gain, qini_coefficient, select_top_k


def simulate_budget_allocation(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    budget_units: int,
    cost_per_treatment: float = 1.0,
    score_noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """对多个策略做预算约束分配并评估。

    参数
    ----
    scores : {策略名: 每个候选的分数}
    budget_units : 预算可覆盖的人数上限（= budget / cost_per_treatment 取整）
    score_noise : 分数噪声强度（乘以分数标准差），用于模拟"模型误差"；
                  None/0 表示不加噪声。**同一噪声会作用到所有策略**以保证公平。
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for name, sc in scores.items():
        sc = np.asarray(sc, dtype=float)
        if sc.shape[0] != len(y):
            raise ValueError(f"策略 {name} 的分数长度 {sc.shape[0]} != 样本数 {len(y)}")
        noisy = sc
        if score_noise and score_noise > 0:
            sd = float(np.std(sc))
            noisy = sc + rng.normal(0.0, score_noise * sd, size=sc.shape[0])

        sel = select_top_k(noisy, budget_units)
        g = policy_gain(y, t, sel)
        cost = g["n_selected"] * cost_per_treatment
        rows.append({
            "strategy": name,
            "n_selected": g["n_selected"],
            "cost": cost,
            "budget_units": int(budget_units),
            "gain": g["gain"],
            "gain_per_user": g["gain_per_user"],
            "gain_per_cost": (g["gain"] / cost) if cost > 0 else float("nan"),
            "qini": qini_coefficient(y, t, noisy),
            "auuc": auuc(y, t, noisy),
            "score_noise": score_noise,
        })
    out = pd.DataFrame(rows)
    # 相对随机策略的倍数：更直观地看"是否值得做增量排序"
    if "random" in set(out["strategy"]):
        base = float(out.loc[out["strategy"] == "random", "gain"].iloc[0])
        out["gain_vs_random"] = out["gain"] / base if abs(base) > 1e-12 else np.nan
    return out


def bootstrap_gain_ci(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    budget_units: int,
    cost_per_treatment: float = 1.0,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """对被选中集合的增量收益做自助法（bootstrap）区间估计。

    为什么必须做：top-k 集合的样本量小（如 480 人），
    G(S) 的抽样标准误可达 ±10，策略之间看似"差 2 倍"的差距
    完全可能是噪声。不报区间就下结论是常见的方法论错误。

    做法：对评估集做有放回重采样（用户级），在每次重采样上重新选 top-k 并计算 G(S)，
    得到 gain 的分布与分位数区间。**注意**：这度量的是"评估集抽样不确定性"，
    不包含模型重训带来的不确定性（后者需要多种子重训，见后续路线）。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    rows = []
    for name, sc in scores.items():
        sc = np.asarray(sc, dtype=float)
        if len(sc) != n:
            raise ValueError(f"策略 {name} 分数长度与样本数不一致")
        gains = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sel = select_top_k(sc[idx], budget_units)
            gains[b] = policy_gain(y[idx], t[idx], sel)["gain"]
        rows.append({
            "strategy": name,
            "n_selected": int(budget_units),
            "cost": float(budget_units * cost_per_treatment),
            "gain_mean": float(np.mean(gains)),
            "gain_std": float(np.std(gains, ddof=1)),
            "gain_lo": float(np.quantile(gains, alpha / 2)),
            "gain_hi": float(np.quantile(gains, 1 - alpha / 2)),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
        })
    out = pd.DataFrame(rows).sort_values("gain_mean", ascending=False).reset_index(drop=True)
    return out


def budget_units_for_ratio(n_candidates: int, budget_ratio: float,
                           cost_per_treatment: float = 1.0) -> int:
    """把"预算占全员触达成本的比例"换算为可触达人数上限。

    语义定义（务必与配置注释一致）：
        预算 = budget_ratio × (全员触达成本) = budget_ratio × N × cost
        可触达人数 k = 预算 / cost = budget_ratio × N
    注意：**k 与 cost_per_treatment 无关**（cost 在分子分母中抵消）。
    换句话说，budget_ratio 直接表示"能被触达的人口比例"。
    """
    if not 0.0 <= float(budget_ratio) <= 1.0:
        raise ValueError("budget_ratio 必须在 [0, 1] 内")
    if cost_per_treatment <= 0:
        raise ValueError("cost_per_treatment 必须为正")
    return int(np.floor(float(n_candidates) * float(budget_ratio)))


def sensitivity_over_budget(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    budget_ratios: list[float],
    cost_per_treatment: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    frames = []
    for r in budget_ratios:
        k = budget_units_for_ratio(len(y), r, cost_per_treatment)
        df = simulate_budget_allocation(scores, y, t, k, cost_per_treatment, 0.0, seed)
        df["budget_ratio"] = r
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def sensitivity_over_noise(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    budget_units: int,
    noise_levels: list[float],
    cost_per_treatment: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    frames = []
    for lvl in noise_levels:
        df = simulate_budget_allocation(scores, y, t, budget_units, cost_per_treatment, lvl, seed)
        df["noise_level"] = lvl
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
