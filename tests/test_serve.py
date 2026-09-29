"""在线服务与模型持久化的测试（使用 FastAPI TestClient，进程内真实调用）。"""

from __future__ import annotations

import numpy as np
import pytest
from baua.config import ModelConfig
from baua.data import make_synthetic
from baua.models import build_model
from baua.persistence import ModelBundle, load_bundle, save_bundle
from baua.serve import create_app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def bundle():
    df, _ = make_synthetic(n=1500, seed=1)
    feats = [c for c in df.columns if c.startswith("x")]
    X, t, y = df[feats], df["treatment"].values, df["y"].values
    models = {}
    for name in ("s_learner", "t_learner"):
        m = build_model(name, ModelConfig(n_estimators=40, num_leaves=8), 1)
        m.fit(X, t, y)
        models[name] = m
    return ModelBundle(models=models, feature_cols=feats, cat_cols=[],
                       dataset="synthetic", seed=1, metrics={"s_learner": 0.12})


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(create_app(bundle))


@pytest.fixture(scope="module")
def rows(bundle):
    df, _ = make_synthetic(n=40, seed=2)
    return df[bundle.feature_cols].round(4).to_dict(orient="records")


def test_health_declares_not_production(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["is_production"] is False
    assert "研究原型" in body["disclaimer"]
    assert set(body["strategies"]) == {"s_learner", "t_learner"}


def test_strategies_lists_metrics(client):
    r = client.get("/strategies")
    assert r.status_code == 200
    assert r.json()["metrics"]["s_learner"] == pytest.approx(0.12)


def test_score_returns_all_strategies(client, rows, bundle):
    r = client.post("/score", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] == len(rows)
    assert set(body["strategies"]) == {"s_learner", "t_learner"}
    for name in body["strategies"]:
        assert len(body["scores"][name]) == len(rows)
        assert np.isfinite(body["scores"][name]).all()


def test_allocate_topk_selects_exactly_budget(client, rows):
    k = 7
    r = client.post("/allocate", json={"rows": rows, "strategy": "s_learner",
                                       "mode": "topk", "budget_units": k})
    assert r.status_code == 200
    body = r.json()
    assert body["n_selected"] == k
    assert len(body["selected_indices"]) == k
    # 选中的应当是分数最高的 k 个
    all_scores = client.post("/score", json={"rows": rows}).json()["scores"]["s_learner"]
    picked = sorted(body["selected_scores"])
    assert picked == sorted(sorted(all_scores)[-k:])


def test_allocate_threshold_only_positive(client, rows):
    r = client.post("/allocate", json={"rows": rows, "strategy": "s_learner",
                                       "mode": "threshold", "threshold": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert all(s > 0.0 for s in body["selected_scores"])
    assert body["n_selected"] == sum(1 for s in body["selected_scores"] if s > 0)


def test_allocate_topk_requires_budget(client, rows):
    r = client.post("/allocate", json={"rows": rows, "strategy": "s_learner", "mode": "topk"})
    assert r.status_code == 422


def test_unknown_strategy_returns_404(client, rows):
    r = client.post("/allocate", json={"rows": rows, "strategy": "does_not_exist",
                                       "mode": "topk", "budget_units": 3})
    assert r.status_code == 404
    assert "未知策略" in r.json()["detail"]


def test_missing_feature_returns_422(client, bundle):
    r = client.post("/score", json={"rows": [{bundle.feature_cols[0]: 0.1}]})
    assert r.status_code == 422
    assert "缺少特征列" in r.json()["detail"]


def test_invalid_mode_rejected(client, rows):
    r = client.post("/allocate", json={"rows": rows, "strategy": "s_learner",
                                       "mode": "nonsense"})
    assert r.status_code == 422


def test_save_and_load_roundtrip(bundle, tmp_path, rows, client):
    """持久化必须无损：重新加载后的打分与保存前完全一致。"""
    path = save_bundle(bundle, str(tmp_path / "m"))
    reloaded = load_bundle(path)
    assert reloaded.feature_cols == bundle.feature_cols
    assert reloaded.dataset == bundle.dataset
    assert set(reloaded.score_names()) == set(bundle.score_names())

    before = client.post("/score", json={"rows": rows}).json()["scores"]
    c2 = TestClient(create_app(reloaded))
    after = c2.post("/score", json={"rows": rows}).json()["scores"]
    for name in before:
        assert np.allclose(before[name], after[name]), f"{name} 的打分在重载后发生了变化"


def test_load_bundle_missing_dir_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        load_bundle(str(tmp_path / "nope"))
    assert "baua.cli fit" in str(e.value)
