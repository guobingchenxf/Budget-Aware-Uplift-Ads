"""预算分配与节奏模拟测试：断言真实行为，不 mock。"""

from __future__ import annotations

import numpy as np
import pytest

from baua.budget import (budget_units_for_ratio, sensitivity_over_noise,
                         simulate_budget_allocation)
from baua.data import make_synthetic
from baua.pacing import (assign_slots, compare_pacing_strategies, simulate_pacing,
                         supply_profile, value_profile)


@pytest.fixture(scope="module")
def synth():
    df, true_cate = make_synthetic(n=20000, seed=11, uplift_strength=3.0)
    return df, true_cate


# ---------------------------------------------------------------- 预算分配
def test_budget_units_math():
    """语义：budget_ratio = 可被触达的人口比例，因此 k = ratio × N（与 unit cost 无关）。"""
    assert budget_units_for_ratio(1000, 0.05, 1.0) == 50
    assert budget_units_for_ratio(1000, 0.05, 2.0) == 50, "cost 在换算中被抵消"
    assert budget_units_for_ratio(1000, 1.0, 7.5) == 1000
    assert budget_units_for_ratio(10, 0.01, 1.0) == 0
    with pytest.raises(ValueError):
        budget_units_for_ratio(100, 1.5)
    with pytest.raises(ValueError):
        budget_units_for_ratio(100, 0.1, 0.0)


def test_uplift_ranking_beats_response_when_they_disagree(synth):
    """核心研究命题的行为测试：当"响应高"与"增量高"不一致时，uplift 排序应拿到更多增量。

    构造方式：用半合成数据的真实 CATE（uplift 正确排序）与
    P(y=1|t=1)（响应模型可达到的最优排序）做对比。
    由于生成机制中 baseline 与 CATE 由不同特征驱动，
    两者故意不同，因此"按响应排序"会把预算浪费在自然转化人群上。
    """
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    x0 = df["x0"].values
    # 响应规模：模拟 P(y=1|t=1) 的最优排序（用真实条件概率的代理）
    response_proxy = x0.copy()

    k = budget_units_for_ratio(len(y), 0.05)
    res = simulate_budget_allocation(
        {"response": response_proxy, "uplift": true_cate}, y, t, k, 1.0, 0.0, seed=3)
    gain = dict(zip(res["strategy"], res["gain"]))
    assert gain["uplift"] > gain["response"], (
        f"uplift 排序应优于响应排序: uplift={gain['uplift']}, response={gain['response']}")


def test_noise_does_not_improve_gain(synth):
    """分数加噪不应提升实际增量（否则说明评估逻辑有误）。"""
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    k = budget_units_for_ratio(len(y), 0.05)
    res = sensitivity_over_noise({"uplift": true_cate}, y, t, k,
                                [0.0, 0.5, 2.0], 1.0, seed=5)
    gains = res.sort_values("noise_level")["gain"].tolist()
    assert gains[0] >= gains[1] >= gains[2], f"加噪后增益不应上升: {gains}"


def test_budget_allocation_respects_budget(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    k = 137
    res = simulate_budget_allocation({"uplift": true_cate}, y, t, k, 2.5, 0.0, seed=1)
    row = res.iloc[0]
    assert row["n_selected"] == k
    assert row["cost"] == pytest.approx(k * 2.5)


def test_scores_length_mismatch_raises(synth):
    df, true_cate = synth
    y, t = df["y"].values, df["treatment"].values
    with pytest.raises(ValueError):
        simulate_budget_allocation({"bad": np.zeros(10)}, y, t, 5)


# ---------------------------------------------------------------- 节奏模拟
def test_value_and_supply_profiles_normalised():
    v = value_profile(24, peak_boost=1.8)
    assert len(v) == 24 and v.min() > 0
    assert v.mean() == pytest.approx(1.0), "价值系数应归一化到均值 1"
    assert v.argmax() > 12, "峰值应位于下半段（晚高峰）"
    s = supply_profile(24)
    assert s.sum() == pytest.approx(1.0)


def test_assign_slots_is_deterministic_and_covers_all_users():
    sup = supply_profile(24)
    a = assign_slots(1000, sup, seed=0)
    b = assign_slots(1000, sup, seed=0)
    assert np.array_equal(a, b)
    assert len(a) == 1000 and a.min() >= 0 and a.max() < 24


def test_pacing_respects_budget_and_supply():
    rng = np.random.default_rng(0)
    score = rng.random(500)
    value = rng.uniform(0.0, 1.0, size=500)
    for st in ["no_pacing", "uniform", "feedback"]:
        res = simulate_pacing(score, value, n_slots=12, budget_ratio=0.1,
                              cost_per_treatment=1.0, strategy=st, seed=0)
        tab = res["slots"]
        assert res["treated"] <= res["budget_units"], "不得超投"
        assert (tab["treated"] <= tab["supply"]).all(), "每时段不得超过供给"
        assert tab["treated"].sum() == res["treated"]


def test_uniform_spreads_and_no_pacing_frontloads():
    rng = np.random.default_rng(1)
    score = rng.random(2000)
    value = np.ones(2000)
    uni = simulate_pacing(score, value, 20, 0.1, 1.0, "uniform", 0.3, 1.8, seed=0)
    nop = simulate_pacing(score, value, 20, 0.1, 1.0, "no_pacing", 0.3, 1.8, seed=0)

    uni_tab, nop_tab = uni["slots"], nop["slots"]
    assert (uni_tab["treated"] > 0).sum() > 10, "均匀策略应铺开到多数时段"
    # no_pacing 应在最早的少数时段内把预算花光
    early = nop_tab["treated"].iloc[:3].sum()
    assert early == nop["treated"], "无节奏策略应在前几个时段耗尽预算"


def test_feedback_tracks_target_and_beats_no_pacing_on_tilted_curve():
    """在价值曲线倾斜的设定下，任何分散策略都应优于无节奏策略。"""
    rng = np.random.default_rng(2)
    score = rng.random(3000)
    value = rng.uniform(0.5, 1.5, size=3000)
    res = compare_pacing_strategies(score, value,
                                    ["no_pacing", "uniform", "feedback", "oracle_value_aware"],
                                    n_slots=24, budget_ratio=0.05, seed=0)
    gain = dict(zip(res["strategy"], res["gain"]))
    assert gain["uniform"] > gain["no_pacing"]
    assert gain["feedback"] > gain["no_pacing"]
    assert gain["oracle_value_aware"] >= gain["uniform"], (
        "已知时段价值的上界策略不应差于均匀投放")


def test_pacing_spend_never_exceeds_budget_even_with_low_supply():
    score = np.random.default_rng(3).random(120)
    value = np.ones(120)
    for st in ["no_pacing", "uniform", "feedback", "oracle_value_aware"]:
        res = simulate_pacing(score, value, 24, 0.9, 1.0, st, 0.3, 1.8, seed=0)
        assert res["spend"] <= res["budget"] + 1e-9
