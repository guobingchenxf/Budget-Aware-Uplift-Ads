"""因果估计（IPS/DR）的测试：针对真实行为的断言。"""

from __future__ import annotations

import numpy as np
import pytest

from baua.causal import (compare_estimators_over_policies, cross_fit_outcomes,
                         cross_fit_propensity, estimate_subset)
from baua.data import make_synthetic
from baua.metrics import select_top_k


def test_confounding_changes_treatment_balance():
    """注入选择偏差后，处理组比例应偏离 0.5，且倾向得分依赖 x0。"""
    df_rct, _ = make_synthetic(n=20000, seed=1, confounding=0.0)
    df_cfd, _ = make_synthetic(n=20000, seed=1, confounding=2.0)

    assert abs(df_rct["treatment"].mean() - 0.5) < 0.02
    assert "propensity_true" in df_cfd.columns
    assert df_cfd["propensity_true"].std() > df_rct["propensity_true"].std() + 0.05
    corr = np.corrcoef(df_cfd["x0"], df_cfd["propensity_true"])[0, 1]
    assert corr > 0.5, f"倾向应与 x0 正相关，实际 {corr}"


def test_cross_fitted_propensity_recovers_truth():
    df, _ = make_synthetic(n=30000, seed=2, confounding=1.5)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    e_hat, diag = cross_fit_propensity(X, t, seed=2)
    err = float(np.abs(e_hat - df["propensity_true"].values).mean())
    assert err < 0.05, f"交叉拟合倾向应接近真值，平均绝对误差 {err}"
    assert 0.0 < diag["propensity_min"] < diag["propensity_max"] < 1.0
    assert diag["clipped_fraction"] <= 0.05


def test_propensity_is_constant_under_rct():
    df, _ = make_synthetic(n=20000, seed=3, confounding=0.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    e_hat, _ = cross_fit_propensity(X, t, seed=3)
    assert e_hat.std() < 0.02, "随机实验下倾向得分应近似常数"


def test_naive_biased_but_ips_dr_closer_under_confounding():
    """核心命题：有选择偏差时"假设无混杂"的朴素估计量有偏，IPS/DR 显著更接近真值。

    注意口径：Qini 式估计量（naive_qini）的目标是 n_t·mean(tau) 而非 sum(tau)，
    因此这里用与之口径一致的 naive_full 作为对照。
    """
    n = 40000
    df, true_cate = make_synthetic(n=n, seed=7, mode="conflicting",
                                   uplift_strength=3.0, confounding=2.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    tc = np.asarray(true_cate, dtype=float)

    e_hat, _ = cross_fit_propensity(X, t, seed=7)
    mu1, mu0 = cross_fit_outcomes(X, t, y, seed=7, n_estimators=150)

    sel = select_top_k(tc, int(0.05 * n))
    est = estimate_subset(y, t, sel, e_hat, mu1, mu0)
    truth = float(tc[sel].sum())

    err_naive = abs(est["naive_full"] - truth)
    err_ips = abs(est["ips"] - truth)
    err_dr = abs(est["dr"] - truth)

    assert err_naive > 0.04 * abs(truth), (
        f"混杂下朴素估计量应明显有偏: naive={est['naive_full']:.1f}, truth={truth:.1f}")
    assert err_ips < err_naive, f"IPS 应优于朴素: {err_ips:.1f} vs {err_naive:.1f}"
    assert err_dr < err_naive, f"DR 应优于朴素: {err_dr:.1f} vs {err_naive:.1f}"
    assert err_dr < 0.08 * abs(truth), f"DR 的相对误差应较小: {err_dr / abs(truth):.3f}"


def test_qini_estimator_targets_scaled_estimand():
    """口径检验：Qini 式估计量约为 sum(tau) 的 n_t/|S| 倍，而不是它的无偏估计。"""
    n = 40000
    df, true_cate = make_synthetic(n=n, seed=7, mode="conflicting",
                                   uplift_strength=3.0, confounding=0.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    tc = np.asarray(true_cate, dtype=float)
    sel = select_top_k(tc, 2000)
    est = estimate_subset(y, t, sel, np.full(n, 0.5))
    truth = float(tc[sel].sum())
    expected_ratio = est["n_treated"] / est["n_selected"]
    assert est["naive_qini"] / truth == pytest.approx(expected_ratio, abs=0.06), (
        f"Qini 估计量应约为真值的 n_t/|S| = {expected_ratio:.3f} 倍")


def test_estimators_agree_under_rct():
    """随机实验下三种同口径估计量应彼此接近（此时朴素估计量也是无偏的）。"""
    n = 40000
    df, true_cate = make_synthetic(n=n, seed=8, mode="conflicting",
                                   uplift_strength=3.0, confounding=0.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    tc = np.asarray(true_cate, dtype=float)

    e_hat, _ = cross_fit_propensity(X, t, seed=8)
    mu1, mu0 = cross_fit_outcomes(X, t, y, seed=8, n_estimators=150)
    sel = select_top_k(tc, int(0.05 * n))
    est = estimate_subset(y, t, sel, e_hat, mu1, mu0)
    truth = float(tc[sel].sum())

    for k in ("naive_full", "ips", "dr"):
        assert abs(est[k] - truth) < 0.10 * abs(truth), (
            f"随机实验下 {k} 应接近真值: {est[k]:.1f} vs {truth:.1f}")


def test_outcome_models_defined_for_every_unit():
    """回归测试：μ1/μ0 必须对**每个样本**都有预测值。

    曾经的 bug：按臂做交叉拟合、只填充本臂索引，另一臂位置保持初始化的 0，
    导致 DR 用到大量恒为 0 的"反事实预测"，估计值只有真值的一半。
    """
    n = 5000
    df, _ = make_synthetic(n=n, seed=21, mode="conflicting", confounding=1.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    mu1, mu0 = cross_fit_outcomes(X, t, y, seed=21, n_estimators=60)

    assert not np.isnan(mu1).any() and not np.isnan(mu0).any()
    assert (mu1 > 0).all() and (mu1 < 1).all(), "μ1 在任意样本上都应是有效概率"
    assert (mu0 > 0).all() and (mu0 < 1).all(), "μ0 在任意样本上都应是有效概率"
    # 两组预测都应有变化（若对照组位置恒为 0 会出现大量精确 0）
    for mu in (mu1, mu0):
        assert (mu > 1e-6).mean() > 0.9, "不应出现大量恒为 0 的预测"
        assert mu.std() > 0.01, "预测应有变异"


def test_compare_estimators_over_policies_columns():
    n = 12000
    df, true_cate = make_synthetic(n=n, seed=9, mode="conflicting",
                                   uplift_strength=3.0, confounding=1.0)
    X = df[[c for c in df.columns if c.startswith("x")]]
    t = df["treatment"].values.astype(int)
    y = df["y"].values.astype(float)
    e_hat, _ = cross_fit_propensity(X, t, seed=9)
    mu1, mu0 = cross_fit_outcomes(X, t, y, seed=9, n_estimators=100)
    tab = compare_estimators_over_policies(
        y, t, np.asarray(true_cate, dtype=float),
        {"oracle": true_cate, "response_x0": X["x0"].values.astype(float)},
        int(0.05 * n), e_hat, mu1, mu0)
    assert len(tab) == 2
    for col in ("truth", "naive_full", "ips", "dr", "naive_full_bias",
                "dr_abs_err", "naive_qini_ratio_to_truth"):
        assert col in tab.columns
    assert tab.loc[tab.policy == "oracle", "truth"].iloc[0] > \
        tab.loc[tab.policy == "response_x0", "truth"].iloc[0]
