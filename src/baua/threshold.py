"""阈值策略 vs 固定预算 top-k。

研究动机（对齐 UpliftBench 2026 的 "sign-threshold policy" 论点）：
    工业界最常见的落地形式是"固定预算、按分数取前 k 个"（top-k）。
    但真实决策常常是**阈值型**的："预估增量大于 0 才投"、
    "预估增量超过单次成本才投"。两者在预算紧张时几乎等价，
    但在预算**宽裕**时差异巨大：

        top-k 为了花完预算，会把钱投给**预估增量为负**的用户
        （典型的"睡狗"客户：本来就会转化，投了反而抑制转化）；
        阈值策略则宁可不花钱，也不投负增量用户。

本模块用一个可验证的方式量化这个差异：在半合成数据上（真实 CATE 已知），对比
    ① top-k（固定人数）
    ② 符号阈值（预测 tau > 0）
    ③ oracle 符号阈值（真实 tau > 0，上界）
并统计"被投给负增量用户的比例"这一浪费指标。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import policy_gain, select_top_k


def threshold_mask(score: np.ndarray, threshold: float) -> np.ndarray:
    """选中 score > threshold 的样本。"""
    return np.asarray(score, dtype=float) > float(threshold)


def evaluate_policy_masks(
    y: np.ndarray,
    t: np.ndarray,
    masks: Dict[str, np.ndarray],
    true_cate: Optional[np.ndarray] = None,
    cost_per_treatment: float = 1.0,
) -> pd.DataFrame:
    """对若干"选择掩码"评估规模、增量收益、单位成本收益与负增量用户占比。

    - 增量收益用 policy_gain（Qini 形式，随机实验下无偏）；
    - 若提供 true_cate（半合成数据），额外报告真实增量总和与
      **被投给负增量用户的比例**：这是 top-k 浪费预算的直接度量。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    rows: List[Dict[str, object]] = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        g = policy_gain(y, t, mask)
        n_sel = int(mask.sum())
        cost = n_sel * cost_per_treatment
        row: Dict[str, object] = {
            "policy": name,
            "n_selected": n_sel,
            "selected_fraction": n_sel / n if n else float("nan"),
            "cost": cost,
            "gain": g["gain"],
            "gain_per_cost": (g["gain"] / cost) if cost > 0 else float("nan"),
        }
        if true_cate is not None:
            tc = np.asarray(true_cate, dtype=float)
            row["true_gain"] = float(tc[mask].sum())
            row["true_gain_per_cost"] = (float(tc[mask].sum()) / cost) if cost > 0 else float("nan")
            row["n_negative_uplift_selected"] = int((tc[mask] <= 0).sum())
            row["neg_uplift_share"] = (float((tc[mask] <= 0).mean()) if n_sel else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_at_budget(
    score: np.ndarray,
    true_cate: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    budget_units: int,
    cost_per_treatment: float = 1.0,
) -> pd.DataFrame:
    """在给定预算下对比 top-k / 符号阈值 / oracle 符号阈值 / 全投。"""
    n = len(score)
    masks = {
        "topk_within_budget": select_top_k(np.asarray(score, dtype=float), budget_units),
        "sign_threshold_pred": threshold_mask(score, 0.0),
        "sign_threshold_oracle": threshold_mask(true_cate, 0.0),
        "treat_all": np.ones(n, dtype=bool),
    }
    return evaluate_policy_masks(y, t, masks, true_cate, cost_per_treatment)


def threshold_study(
    n: int = 40000,
    seed: int = 20260929,
    budget_ratios=(0.02, 0.05, 0.10, 0.20, 0.40, 0.80),
    mode: str = "conflicting",
    uplift_strength: float = 3.0,
    score_source: str = "model",
) -> pd.DataFrame:
    """扫描预算比例，观察 top-k 相对阈值策略何时开始恶化。

    score_source:
      - "model" ：用 T-learner 的预测 uplift 作为分数（更真实，含模型误差）
      - "oracle"：直接用真实 CATE（上界参照）
    """
    from .config import ModelConfig
    from .data import make_synthetic
    from .models import TLearner

    df, true_cate = make_synthetic(n=n, seed=seed, mode=mode,
                                   uplift_strength=uplift_strength)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    tc = np.asarray(true_cate, dtype=float)

    if score_source == "oracle":
        score = tc.copy()
    else:
        model = TLearner(ModelConfig(n_estimators=300, num_leaves=31), seed)
        model.fit(X, t, y)
        score = np.asarray(model.predict(X), dtype=float)

    frames = []
    for r in budget_ratios:
        k = int(n * float(r))
        tab = compare_at_budget(score, tc, y, t, k)
        tab.insert(0, "budget_ratio", float(r))
        tab.insert(1, "budget_units", k)
        tab.insert(2, "score_source", score_source)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True)
