"""数据模块测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest
from baua.data import ALL_ARMS, build_analysis_frame, make_synthetic, validate_hillstrom

HILLSTROM = os.path.join("data", "raw", "hillstrom_email_analytics.csv")


def test_synthetic_ate_matches_mean_true_cate():
    """半合成数据的核心性质：真实 CATE 的均值应接近经验上的组间差异。"""
    n = 60000
    df, true_cate = make_synthetic(n=n, seed=123, uplift_strength=2.0)
    t, y = df["treatment"].values, df["y"].values
    empirical_ate = y[t == 1].mean() - y[t == 0].mean()
    assert empirical_ate == pytest.approx(float(np.mean(true_cate)), abs=0.01), (
        "经验 ATE 与真实平均 CATE 应当一致（完全随机化 + 大样本）")


def test_synthetic_treatment_is_balanced():
    df, _ = make_synthetic(n=8000, seed=5)
    frac = df["treatment"].mean()
    assert 0.47 < frac < 0.53, f"处理组比例应接近 0.5，实际 {frac}"


def test_synthetic_reproducible():
    a, ca = make_synthetic(n=500, seed=42)
    b, cb = make_synthetic(n=500, seed=42)
    assert a.equals(b) and np.allclose(ca, cb)
    c, _ = make_synthetic(n=500, seed=43)
    assert not a.equals(c)


def test_validate_rejects_wrong_data():
    import pandas as pd
    bad = pd.DataFrame({"a": [1, 2, 3]})
    rep = validate_hillstrom(bad)
    assert rep["ok"] is False
    assert rep["columns_ok"] is False
    assert rep["missing_columns"], "应报告缺失的列"


@pytest.mark.skipif(not os.path.exists(HILLSTROM), reason="需要先运行 download 命令")
def test_hillstrom_real_file_structure():
    import pandas as pd
    from baua.data import validate_hillstrom as val
    df = pd.read_csv(HILLSTROM)
    rep = val(df)
    assert rep["ok"] is True
    assert rep["n_rows"] == 64000
    assert set(rep["arms"]) == set(ALL_ARMS)
    assert rep["conversion_without_visit"] == 0


@pytest.mark.skipif(not os.path.exists(HILLSTROM), reason="需要先运行 download 命令")
@pytest.mark.parametrize("definition,expected_control", [
    ("any_email", 21306), ("mens_only", 21306), ("womens_only", 21306)])
def test_build_analysis_frame_definitions(definition, expected_control):
    import pandas as pd
    df = pd.read_csv(HILLSTROM)
    out = build_analysis_frame(df, definition, "visit")
    assert int((out["treatment"] == 0).sum()) == expected_control
    assert set(out["treatment"].unique()) == {0, 1}
    assert out["y"].isin([0.0, 1.0]).all()


@pytest.mark.skipif(not os.path.exists(HILLSTROM), reason="需要先运行 download 命令")
def test_hillstrom_treatment_has_positive_effect_on_visit():
    """核验实验设计：处理组 visit 率应显著高于对照组（这是本项目的前提）。"""
    import pandas as pd
    df = pd.read_csv(HILLSTROM)
    r = df.groupby("segment")["visit"].mean()
    assert r["Mens E-Mail"] > r["No E-Mail"] + 0.03
    assert r["Womens E-Mail"] > r["No E-Mail"] + 0.02


def test_feature_whitelist_has_no_outcome_columns():
    """回归测试：结果列绝不能进入特征集合（曾因此出现标签泄漏导致增量恒为 0）。"""
    from baua.data import FORBIDDEN_FEATURES, assert_no_leakage, hillstrom_feature_columns
    cols = hillstrom_feature_columns()
    assert not (set(cols) & FORBIDDEN_FEATURES), "特征白名单混入结果列"
    for bad in ["visit", "conversion", "spend", "y", "treatment", "segment", "arm"]:
        with pytest.raises(ValueError):
            assert_no_leakage(cols + [bad])
