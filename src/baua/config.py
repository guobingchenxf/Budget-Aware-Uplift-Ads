"""配置加载与日志。

设计原则：
- 单一配置入口（configs/*.yaml），命令行可覆盖关键字段；
- 所有随机性由 config.seed 控制；
- 不为"看起来工程化"而引入额外依赖。
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml

LOGGER_NAME = "baua"


def get_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    return logger


@dataclass
class DataConfig:
    name: str = "hillstrom"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    max_rows: int | None = None
    treatment_definition: str = "any_email"
    primary_outcome: str = "visit"
    test_size: float = 0.3
    eval_size: float = 0.5
    # 半合成数据的效应模式：aligned（响应与增量同向）/ conflicting（反向，睡狗场景）
    synthetic_mode: str = "aligned"


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    n_jobs: int = 2
    calibration: str = "none"


@dataclass
class BudgetConfig:
    cost_per_treatment: float = 1.0
    budget_ratio: float = 0.05
    strategies: list[str] = field(default_factory=lambda: [
        "random", "response", "s_learner", "t_learner", "x_learner", "class_transform"])


@dataclass
class PacingConfig:
    n_slots: int = 24
    prime_slot_boost: float = 1.8
    damping: float = 0.3
    strategies: list[str] = field(default_factory=lambda: ["no_pacing", "uniform", "feedback"])


@dataclass
class SensitivityConfig:
    budget_ratios: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.20])
    score_noise_levels: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0])
    calibration_methods: list[str] = field(default_factory=lambda: ["none", "isotonic"])
    sample_sizes: list[int] = field(default_factory=lambda: [5000, 20000, 64000])


@dataclass
class BootstrapConfig:
    """自助法区间估计参数（用于判断策略差异是否只是噪声）。"""
    n_boot: int = 200
    alpha: float = 0.05


@dataclass
class ExperimentConfig:
    """实验流程开关与校准分析参数。"""
    run_sensitivity: bool = True
    run_pacing: bool = True
    run_calibration: bool = True
    # 校准器必须在独立验证集上拟合；该比例从**训练集内部**再切分
    calibration_val_size: float = 0.2
    # 绝对阈值策略的业务阈值列表（"预测转化率低于 X 不投"）
    thresholds: list[float] = field(default_factory=lambda: [0.05, 0.08, 0.10, 0.15])
    n_seeds: int = 10


@dataclass
class OutputConfig:
    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"


@dataclass
class Config:
    seed: int = 20260929
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    pacing: PacingConfig = field(default_factory=PacingConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # 记录配置文件来源，便于复现
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": vars(self.data),
            "model": vars(self.model),
            "budget": vars(self.budget),
            "pacing": vars(self.pacing),
            "sensitivity": vars(self.sensitivity),
            "bootstrap": vars(self.bootstrap),
            "experiment": vars(self.experiment),
            "output": vars(self.output),
            "source_path": self.source_path,
        }

    def clone(self) -> Config:
        return copy.deepcopy(self)


def _build(cls, raw: dict[str, Any] | None):
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in (raw or {}).items() if k in known})


def load_config(path: str, overrides: dict[str, Any] | None = None) -> Config:
    """从 YAML 加载配置，overrides 为点号路径（如 data.max_rows）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        seed=raw.get("seed", 20260929),
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        budget=_build(BudgetConfig, raw.get("budget")),
        pacing=_build(PacingConfig, raw.get("pacing")),
        sensitivity=_build(SensitivityConfig, raw.get("sensitivity")),
        bootstrap=_build(BootstrapConfig, raw.get("bootstrap")),
        experiment=_build(ExperimentConfig, raw.get("experiment")),
        output=_build(OutputConfig, raw.get("output")),
        source_path=os.path.abspath(path),
    )

    for key, value in (overrides or {}).items():
        section, _, field_name = key.partition(".")
        if not field_name:
            setattr(cfg, section, value)
            continue
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, field_name):
            raise KeyError(f"未知配置项: {key}")
        setattr(target, field_name, value)
    return cfg
