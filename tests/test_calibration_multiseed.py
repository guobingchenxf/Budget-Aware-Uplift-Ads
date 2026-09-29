"""校准与多种子聚合的测试（针对真实行为，不做 mock 断言）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baua.calibration import (IsotonicCalibrator, compare_calibration_effect,
                              threshold_policy_table)
from baua.multiseed import aggregate_multiseed


@pytest.fixture(scope="module")
def biased_scores():
    """构造系统性低估的分数：真实概率是预测值的两倍。"""
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.5, size=30000)
    y = (rng.random(len(p_true)) < p_true).astype(float)
    p_raw = p_true * 0.5          # 系统性低估
    return p_raw, y, p_true


def test_isotonic_reduces_calibration_error(biased_scores):
    from baua.metrics import calibration_report
    p_raw, y, _ = biased_scores
    n = len(y)
    cut = n // 2
    cal = IsotonicCalibrator().fit(p_raw[:cut], y[:cut])
    assert cal.fitted and cal.n_fit == cut

    before = calibration_report(y[cut:], p_raw[cut:])["ece"]
    after = calibration_report(y[cut:], cal.transform(p_raw[cut:]))["ece"]
    assert after < before, f"校准后 ECE 应下降: before={before}, after={after}"
    assert after < 0.02, f"校准后 ECE 应很小，实际 {after}"


def test_isotonic_is_monotone_and_preserves_ranking(biased_scores):
    p_raw, y, _ = biased_scores
    cal = IsotonicCalibrator().fit(p_raw, y)
    out = cal.transform(p_raw)
    order_raw = np.argsort(p_raw, kind="mergesort")
    order_cal = np.argsort(out, kind="mergesort")
    # 单调映射下，原始分数不同的样本对顺序不应反转
    diff_pair = np.flatnonzero(np.diff(p_raw[order_raw]) > 0)
    a, b = order_raw[diff_pair], order_raw[diff_pair + 1]
    assert np.all(out[a] <= out[b] + 1e-12), "校准破坏了单调性"


def test_isotonic_unfitted_is_identity():
    cal = IsotonicCalibrator()
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal.transform(x), x)


def test_isotonic_single_class_degrades_gracefully():
    cal = IsotonicCalibrator().fit(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, 1.0]))
    assert not cal.fitted
    assert np.allclose(cal.transform(np.array([0.2])), np.array([0.2]))


def test_calibration_introduces_ties_and_may_change_topk():
    """实测结论：单调度只保证"不反转顺序"，但**会引入并列**。

    isotonic 是分段常数映射，一个平坦区间内的样本得到相同分数；
    若该区间跨越 top-k 的截断边界，并列只能按索引打破，
    于是校准前后的 top-k 选择会不同（实测 overlap 可低至 0.66）。
    该测试同时锁定这两个事实：①确实引入并列；②增量收益的变化量级很小。
    """
    rng = np.random.default_rng(3)
    n = 8000
    x = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x + 1.0 * t)))).astype(float)
    raw = 1 / (1 + np.exp(-x)) * 0.5      # 被压低的分数
    cal = IsotonicCalibrator().fit(raw, y)
    scores = cal.transform(raw)

    tab = compare_calibration_effect(raw, scores, y, t)
    assert (tab["ties_introduced"] > 0).all(), "isotonic 应当引入并列"
    assert (tab["n_unique_calibrated"] < tab["n_unique_raw"]).all()
    # 顺序不会被"反转"，但并列会改变边界处的选择
    assert (tab["overlap_ratio"] > 0.5).all(), "两种排序不应大幅背离"

    # 关键：排序质量的变化应当远小于"模型 vs 随机"的差距
    from baua.metrics import policy_gain, select_top_k
    k = int(0.05 * n)
    g_raw = policy_gain(y, t, select_top_k(raw, k))["gain"]
    g_cal = policy_gain(y, t, select_top_k(scores, k))["gain"]
    assert abs(g_cal - g_raw) < 0.5 * abs(g_raw), (
        f"校准带来的收益变化应远小于策略本身的收益: raw={g_raw}, cal={g_cal}")


def test_threshold_policy_table_shapes_and_monotonicity():
    rng = np.random.default_rng(5)
    n = 6000
    score = rng.random(n)
    t = rng.integers(0, 2, size=n)
    y = (rng.random(n) < 0.1 + 0.4 * score * t).astype(float)

    tab = threshold_policy_table({"m": score}, y, t, [0.2, 0.5, 0.8])
    assert len(tab) == 3
    # 阈值越高，选中人数应越少（单调性）
    counts = tab.sort_values("threshold")["n_selected"].tolist()
    assert counts[0] >= counts[1] >= counts[2], f"阈值与选中人数应单调: {counts}"
    # 构造的数据里增量随分数上升，所以高阈值子集的"人均增量"应更高
    per = tab.sort_values("threshold")["gain_per_selected"].tolist()
    assert per[-1] > per[0], f"高阈值子集的人均增量应更高: {per}"


def test_threshold_ordering_differs_between_raw_and_calibrated(biased_scores):
    """同一绝对阈值在原始分数与校准分数下的触达规模必须不同（这是校准的业务意义）。"""
    p_raw, y, _ = biased_scores
    t = np.ones(len(y), dtype=int)
    cal = IsotonicCalibrator().fit(p_raw, y)
    tab = threshold_policy_table({"raw": p_raw, "calibrated": cal.transform(p_raw)},
                                y, t, [0.15])
    raw_n = float(tab.loc[tab.strategy == "raw", "selected_fraction"].iloc[0])
    cal_n = float(tab.loc[tab.strategy == "calibrated", "selected_fraction"].iloc[0])
    assert cal_n > raw_n, (
        f"分数被低估时，校准应放大同一阈值下的触达人数: raw={raw_n}, cal={cal_n}")


# ---------------------------------------------------------------- 多种子聚合
def test_aggregate_multiseed_paired_diff():
    per_seed = pd.DataFrame([
        {"seed": 1, "strategy": "random", "gain": 10.0},
        {"seed": 1, "strategy": "a", "gain": 14.0},
        {"seed": 2, "strategy": "random", "gain": 20.0},
        {"seed": 2, "strategy": "a", "gain": 25.0},
        {"seed": 3, "strategy": "random", "gain": 30.0},
        {"seed": 3, "strategy": "a", "gain": 36.0},
    ])
    out = aggregate_multiseed(per_seed, baseline="random")
    s = out["summary"].set_index("strategy")
    assert s.loc["a", "n_seeds"] == 3
    assert s.loc["a", "gain_mean"] == pytest.approx((14 + 25 + 36) / 3)
    # 配对差值恒定 = 4/5/6 → 均值 5，标准差 1
    assert s.loc["a", "paired_diff_vs_random_mean"] == pytest.approx(5.0)
    assert s.loc["a", "paired_diff_vs_random_std"] == pytest.approx(1.0)
    assert bool(s.loc["a", "paired_diff_vs_random_significant"]) is True
    # 与自身配对时差值恒为 0，不应被判为显著
    assert bool(s.loc["random", "paired_diff_vs_random_significant"]) is False


def test_aggregate_multiseed_rejects_missing_columns():
    with pytest.raises(ValueError):
        aggregate_multiseed(pd.DataFrame({"seed": [1], "gain": [1.0]}))
