"""指标模块测试
"""

from __future__ import annotations

import numpy as np
import pytest
from baua.data import make_synthetic
from baua.metrics import (
    auuc,
    calibration_report,
    cate_calibration,
    policy_gain,
    qini_coefficient,
    qini_curve,
    select_top_k,
)


@pytest.fixture(scope="module")
def synth():
    df, true_cate = make_synthetic(n=20000, seed=7, uplift_strength=2.5)
    return df, true_cate


def test_qini_perfect_ranking_beats_random(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values

    q_oracle = qini_coefficient(y, t, true_cate)
    rng = np.random.default_rng(0)
    q_random = qini_coefficient(y, t, rng.random(len(y)))

    assert q_oracle > q_random, "按真实 CATE 排序应当优于随机"
    assert q_oracle > 0.2, f"真实 CATE 的 Qini 过低: {q_oracle}"
    assert abs(q_random) < 0.1, f"随机策略的 Qini 应接近 0，实际 {q_random}"


def test_qini_inverted_ranking_is_negative(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    q_inv = qini_coefficient(y, t, -true_cate)
    assert q_inv < 0, f"反向排序的 Qini 应为负，实际 {q_inv}"


def test_qini_curve_shape_and_endpoint(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    cur = qini_curve(y, t, true_cate)
    n = len(y)
    assert len(cur["fractions"]) == n
    assert cur["fractions"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(cur["fractions"]) > 0), "前缀占比必须严格递增"
    # 曲线终值 = 总体增量收益估计 Y_t - Y_c * N_t/N_c
    n_t, n_c = t.sum(), (1 - t).sum()
    expected = y[t == 1].sum() - y[t == 0].sum() * n_t / n_c
    assert cur["qini"][-1] == pytest.approx(expected, rel=1e-6)


def test_auuc_sign_matches_qini(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    assert auuc(y, t, true_cate) > 0
    assert auuc(y, t, -true_cate) < 0


def test_policy_gain_exact_on_handmade_case():
    """手工算例：处理组 4 人中有 2 人转化，对照组 4 人中有 1 人转化。"""
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    sel = np.ones(8, dtype=bool)
    g = policy_gain(y, t, sel)
    # G(S) = Y_t - Y_c * N_t/N_c = 2 - 1 * (4/4) = 1
    assert g["n_selected"] == 8
    assert g["n_treated"] == 4 and g["n_control"] == 4
    assert g["gain"] == pytest.approx(1.0)
    assert g["gain_per_user"] == pytest.approx(1.0 / 8)


def test_policy_gain_empty_selection():
    y = np.array([1, 0, 1, 0], dtype=float)
    t = np.array([1, 0, 1, 0], dtype=int)
    g = policy_gain(y, t, np.zeros(4, dtype=bool))
    assert g["n_selected"] == 0 and g["gain"] == 0.0


def test_policy_gain_no_control_returns_nan():
    y = np.array([1.0, 0.0])
    t = np.array([1, 1])
    g = policy_gain(y, t, np.ones(2, dtype=bool))
    assert np.isnan(g["gain"]), "没有对照组时增量无法识别，必须返回 NaN 而不是编造数字"


def test_select_top_k_behaviour_and_ties():
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.2])
    m = select_top_k(s, 3)
    assert m.sum() == 3
    assert m[1] and m[3] and m[2], "应选出分数最高的三个（含并列）"
    assert select_top_k(s, 0).sum() == 0
    assert select_top_k(s, 99).sum() == 5, "k 超过样本数时应全部选中"
    # 可复现性：相同输入必须给出相同结果
    assert np.array_equal(select_top_k(s, 2), select_top_k(s, 2))


def test_calibration_report_detects_bias():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.5, size=20000)
    y = (rng.random(len(p_true)) < p_true).astype(float)

    good = calibration_report(y, p_true)
    biased = calibration_report(y, p_true * 0.5)

    assert good["ece"] < 0.02, f"良好校准的 ECE 应很小，实际 {good['ece']}"
    assert biased["ece"] > good["ece"], "系统性低估应被检出"


def test_cate_calibration_on_synthetic(synth):
    df, true_cate = synth
    rep = cate_calibration(true_cate, true_cate)
    assert rep["mae_binned"] == pytest.approx(0.0, abs=1e-9)
    assert rep["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert rep["bias"] == pytest.approx(0.0, abs=1e-9)
