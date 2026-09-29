"""阈值策略 vs 固定预算 top-k 的测试。"""

from __future__ import annotations

import numpy as np
from baua.data import make_synthetic
from baua.threshold import compare_at_budget, evaluate_policy_masks, threshold_mask, threshold_study


def test_threshold_mask_boundary():
    s = np.array([-0.5, 0.0, 0.1, 0.5])
    m = threshold_mask(s, 0.0)
    assert m.tolist() == [False, False, True, True], "严格大于 0 才算正增量"


def test_evaluate_policy_masks_reports_negative_uplift():
    y = np.array([1.0, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 1, 0, 0, 0])
    tc = np.array([0.5, -0.2, 0.3, -0.4, 0.1, -0.1])
    masks = {"all": np.ones(6, dtype=bool), "positive_only": tc > 0}
    tab = evaluate_policy_masks(y, t, masks, tc)
    assert tab.loc[0, "n_negative_uplift_selected"] == 3
    assert tab.loc[1, "n_negative_uplift_selected"] == 0
    assert tab.loc[1, "neg_uplift_share"] == 0.0
    assert tab.loc[1, "true_gain"] > tab.loc[0, "true_gain"]


def test_topk_wastes_budget_when_budget_exceeds_positive_uplift_users():
    """核心命题：预算宽裕时 top-k 会把钱投给负增量用户，阈值策略更优。"""
    n = 30000
    df, true_cate = make_synthetic(n=n, seed=11, mode="conflicting",
                                   uplift_strength=3.0)
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    tc = np.asarray(true_cate, dtype=float)
    n_pos = int((tc > 0).sum())
    assert n_pos < n, "conflicting 场景下应存在负增量用户"

    k = min(int(n_pos * 1.5), n)      # 预算远超正增量用户数
    tab = compare_at_budget(tc, tc, y, t, k)   # oracle 分数，隔离排序误差
    row_topk = tab[tab.policy == "topk_within_budget"].iloc[0]
    row_sign = tab[tab.policy == "sign_threshold_oracle"].iloc[0]

    assert row_topk["n_negative_uplift_selected"] > 0, "top-k 应选中负增量用户"
    assert row_sign["n_negative_uplift_selected"] == 0
    assert row_sign["true_gain"] > row_topk["true_gain"], (
        "oracle 阈值策略的真实增益应高于固定预算 top-k")


def test_threshold_study_shape():
    tab = threshold_study(n=8000, seed=12, budget_ratios=(0.05, 0.2),
                          mode="conflicting", score_source="model")
    assert set(tab["budget_ratio"]) == {0.05, 0.2}
    assert set(tab["policy"]) == {"topk_within_budget", "sign_threshold_pred",
                                  "sign_threshold_oracle", "treat_all"}
    assert (tab["n_selected"] >= 0).all()
    assert tab["selected_fraction"].between(0, 1).all()
