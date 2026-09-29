"""在线推理服务（FastAPI）——研究原型，**不是生产系统**。

能力：
- `GET  /health`   健康检查 + 已加载的策略与元信息
- `POST /score`    批量打分：对每行特征给出响应概率与各策略的 uplift 分数
- `POST /allocate` 预算约束分配：按 top-k（固定预算）或阈值（增量>0）返回选中对象

明确的**非目标**（避免把它说成生产系统）：
- 无鉴权、无限流、无多租户；
- 无特征存储/在线特征拼接（特征必须由调用方提供，缺失即报错）；
- 无模型热更新与灰度（换模型需重启或重新加载）；
- 不做批量优化与 GPU 推理。

这些限制与"研究原型"的定位一致，也写进了 README 的已知限制。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import get_logger
from .metrics import policy_gain, select_top_k
from .persistence import ModelBundle

logger = get_logger()


# --------------------------------------------------------------------------
# 请求 / 响应模型
# --------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, description="特征行列表")

    @field_validator("rows")
    @classmethod
    def _limit(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(v) > 10000:
            raise ValueError("单次最多 10000 行（研究原型限制）")
        return v


class ScoreResponse(BaseModel):
    n_rows: int
    strategies: list[str]
    scores: dict[str, list[float]]


class AllocateRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    strategy: str = Field("s_learner", description="用哪个策略的分数分配")
    budget_units: int | None = Field(None, ge=0, description="固定预算下可触达人数")
    mode: str = Field("topk", description="topk（固定预算）或 threshold（增量>0 才投）")
    threshold: float = Field(0.0, description="threshold 模式下的分数门槛")

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in ("topk", "threshold"):
            raise ValueError("mode 必须是 topk 或 threshold")
        return v


class AllocateResponse(BaseModel):
    mode: str
    strategy: str
    n_selected: int
    selected_indices: list[int]
    selected_scores: list[float]
    total_score: float
    note: str


# --------------------------------------------------------------------------
# 打分与分配核心（与离线评估共用同一套函数，避免线上线下口径漂移）
# --------------------------------------------------------------------------
def build_feature_frame(bundle: ModelBundle, rows: list[dict[str, Any]]) -> pd.DataFrame:
    missing = [c for c in bundle.feature_cols if c not in rows[0]]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"缺少特征列: {missing}；期望全部: {bundle.feature_cols}")
    df = pd.DataFrame(rows)
    for c in bundle.cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df[bundle.feature_cols]


def score_rows(bundle: ModelBundle, X: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, model in bundle.models.items():
        out[name] = np.asarray(model.predict(X), dtype=float)
    return out


def allocate(scores: np.ndarray, mode: str, budget_units: int | None,
             threshold: float) -> np.ndarray:
    """返回选中掩码。topk = 固定预算；threshold = 增量超过门槛才投。"""
    if mode == "topk":
        if budget_units is None:
            raise HTTPException(status_code=422, detail="topk 模式必须提供 budget_units")
        return select_top_k(scores, int(budget_units))
    return np.asarray(scores, dtype=float) > float(threshold)


# --------------------------------------------------------------------------
# 应用工厂
# --------------------------------------------------------------------------
def create_app(bundle: ModelBundle) -> FastAPI:
    app = FastAPI(
        title="Budget-Aware Uplift Ads (research prototype)",
        version="0.1.0",
        description="研究原型：策略打分与预算约束分配。无鉴权、无在线特征，勿用于生产。",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset": bundle.dataset,
            "seed": bundle.seed,
            "strategies": bundle.score_names(),
            "n_features": len(bundle.feature_cols),
            "is_production": False,
            "disclaimer": "研究原型：成本/预算为模拟量；无鉴权与限流。",
        }

    @app.get("/strategies")
    def strategies() -> dict[str, Any]:
        return {"strategies": bundle.score_names(),
                "metrics": bundle.metrics or {},
                "feature_cols": bundle.feature_cols,
                "cat_cols": bundle.cat_cols}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        X = build_feature_frame(bundle, req.rows)
        sc = score_rows(bundle, X)
        return ScoreResponse(n_rows=len(X),
                             strategies=sorted(sc.keys()),
                             scores={k: v.tolist() for k, v in sc.items()})

    @app.post("/allocate", response_model=AllocateResponse)
    def allocate_endpoint(req: AllocateRequest) -> AllocateResponse:
        X = build_feature_frame(bundle, req.rows)
        sc_all = score_rows(bundle, X)
        if req.strategy not in sc_all:
            raise HTTPException(status_code=404,
                                detail=f"未知策略 {req.strategy}；可用: {sorted(sc_all)}")
        sc = sc_all[req.strategy]
        mask = allocate(sc, req.mode, req.budget_units, req.threshold)
        idx = np.flatnonzero(mask).tolist()
        note = ("topk：固定预算取前 k 个" if req.mode == "topk"
                else "threshold：只投预估增量 > 门槛的对象，可能花不完预算（这是正确行为）")
        return AllocateResponse(
            mode=req.mode, strategy=req.strategy, n_selected=int(mask.sum()),
            selected_indices=idx, selected_scores=sc[mask].tolist(),
            total_score=float(sc[mask].sum()), note=note)

    return app


# --------------------------------------------------------------------------
# 便捷工厂：给 TestClient / uvicorn 使用
# --------------------------------------------------------------------------
def app_from_dir(path: str) -> FastAPI:
    return create_app(load_bundle_from(path))


def load_bundle_from(path: str) -> ModelBundle:
    from .persistence import load_bundle
    return load_bundle(path)


def offline_policy_value(y, t, scores: np.ndarray, mode: str,
                         budget_units: int | None, threshold: float) -> dict[str, float]:
    """把线上分配规则在离线上做同样的评估（保证线上线下同一口径）。

    仅当调用方能提供结果标签 y 与处理标记 t 时可用；真实线上没有反事实标签，
    所以这里主要用于"分配逻辑与离线评估一致"的回归测试。
    """
    mask = allocate(scores, mode, budget_units, threshold)
    return policy_gain(np.asarray(y, dtype=float), np.asarray(t, dtype=float), mask)
