"""模型持久化：训练好的策略模型 + 元数据（特征列、类别列、指标）。

用途：
- `baua.cli fit` 训练并保存；
- `baua.cli serve` 加载后对外提供打分与预算分配接口。

设计上刻意保持简单：joblib 序列化 + 一份 JSON 元数据。
不引入 MLflow / 模型注册中心——本项目的目标是可复现的研究原型。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib

from .config import get_logger

logger = get_logger()

MODEL_FILE = "models.joblib"
META_FILE = "metadata.json"


@dataclass
class ModelBundle:
    """一次保存/加载的单位：多个策略模型 + 打分所需的元信息。"""

    models: dict[str, Any]
    feature_cols: list[str]
    cat_cols: list[str]
    dataset: str
    seed: int
    metrics: dict[str, float] | None = None

    def score_names(self) -> list[str]:
        return sorted(self.models.keys())


def save_bundle(bundle: ModelBundle, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(bundle.models, os.path.join(out_dir, MODEL_FILE))
    meta = {
        "feature_cols": list(bundle.feature_cols),
        "cat_cols": list(bundle.cat_cols),
        "dataset": bundle.dataset,
        "seed": int(bundle.seed),
        "strategies": bundle.score_names(),
        "metrics": bundle.metrics or {},
    }
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("模型已保存到 %s（策略: %s）", os.path.abspath(out_dir), meta["strategies"])
    return os.path.abspath(out_dir)


def load_bundle(path: str) -> ModelBundle:
    model_path = os.path.join(path, MODEL_FILE)
    meta_path = os.path.join(path, META_FILE)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"未找到模型文件。请先运行: python -m baua.cli fit --tag <名称>（当前: {path}）")
    models = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return ModelBundle(
        models=models,
        feature_cols=list(meta["feature_cols"]),
        cat_cols=list(meta.get("cat_cols", [])),
        dataset=str(meta.get("dataset", "")),
        seed=int(meta.get("seed", 0)),
        metrics=meta.get("metrics"),
    )
