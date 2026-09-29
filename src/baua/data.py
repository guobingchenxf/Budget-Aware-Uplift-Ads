"""数据加载：Hillstrom（主）/ 半合成（已知真实 CATE）/ Criteo（需手动获取）。

重要事实（已实测核验，见 docs/论文与仓库调研.md）：
- Hillstrom 数据集来自 MineThatData E-Mail Analytics And Data Mining Challenge (2008)，
  64,000 行、12 列，无缺失值，是三臂随机实验：
  Mens E-Mail 21307 / Womens E-Mail 21387 / No E-Mail 21306。
- 该数据集的干预是"营销邮件触达"，**不是广告曝光**。
  本项目把它作为"随机干预 + 二值结果"的公开基准，用于研究
  预算约束下的增量排序问题，不宣称它等价于真实广告投放数据。
- Criteo Uplift 数据集官方链接已核验存在但本环境不可达（详见文档），
  因此**不作为默认数据源**；代码只提供手动放置后的读取入口。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .config import get_logger

logger = get_logger()

HILLSTROM_URL = ("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_"
                 "E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
HILLSTROM_FILENAME = "hillstrom_email_analytics.csv"
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
HILLSTROM_ROWS = 64000

# 处理组/对照组定义（segment 取值已实测核验）
TREATMENT_ARM = "Mens E-Mail"
CONTROL_ARM = "No E-Mail"
ALL_ARMS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")

FEATURE_COLUMNS = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_COLUMNS = ["history_segment", "zip_code", "channel"]
OUTCOME_COLUMNS = ["visit", "conversion", "spend"]

# 绝不允许进入特征集合的列（结果列、处理列、标识列）。
# 这是本项目的硬性防线：一旦结果列混入特征，模型会"完美预测"，
# 选中集合的增量收益会恰好退化为 0（Y_t=N_t, Y_c=N_c ⇒ G≡0），
# 表面指标（Qini）却依然好看——属于最难发现的错误类型。
FORBIDDEN_FEATURES = set(OUTCOME_COLUMNS) | {"y", "treatment", "arm", "segment"}


def hillstrom_feature_columns() -> list:
    """Hillstrom 的特征列（显式白名单，不使用"排除法"以免漏掉结果列）。"""
    cols = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    leaked = set(cols) & FORBIDDEN_FEATURES
    if leaked:  # pragma: no cover - 白名单写错时立刻失败
        raise AssertionError(f"特征白名单中混入了禁用列: {sorted(leaked)}")
    return cols


def assert_no_leakage(feature_cols) -> None:
    leaked = set(feature_cols) & FORBIDDEN_FEATURES
    if leaked:
        raise ValueError(
            f"检测到标签泄漏！以下列不得作为特征: {sorted(leaked)}。"
            "结果列参与训练会让评估指标完全失真。"
        )

# 实测核验的原始列顺序（64,000 行文件的表头，逐字核对）：
#   recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend
# 注意：history_segment 在第 2 列、zip_code 在第 6 列、newbie 在第 7 列。
# 校验时按**集合**比较（列序不影响建模），并单独记录顺序是否与文档一致。
DOCUMENTED_COLUMN_ORDER = [
    "recency", "history_segment", "history", "mens", "womens",
    "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
]


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hillstrom_path(raw_dir: str) -> str:
    return os.path.join(raw_dir, HILLSTROM_FILENAME)


def load_hillstrom_raw(raw_dir: str, verify: bool = True) -> pd.DataFrame:
    """读取原始 CSV。**不会自动下载**（下载见 scripts/ 与 CLI download 命令）。"""
    path = hillstrom_path(raw_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}。请先运行:  python -m baua.cli download  （或 scripts/download_data.ps1）"
        )
    df = pd.read_csv(path)
    if verify:
        check = validate_hillstrom(df, path)
        if not check["ok"]:
            raise ValueError(f"Hillstrom 数据自检失败: {check}")
    return df


def validate_hillstrom(df: pd.DataFrame, path: str = "") -> dict[str, Any]:
    """数据结构自检：字段、行数、类别分布、缺失、标签一致性。

    columns_ok 按**集合**判定（列序不影响建模）；
    order_matches_documented 单独记录顺序是否与文档一致，
    以便在数据源变更时立刻发现（这是"不假定字段"的具体落实）。
    """
    required = set(DOCUMENTED_COLUMN_ORDER)
    actual = list(df.columns)
    report: dict[str, Any] = {
        "path": path,
        "n_rows": int(len(df)),
        "columns": actual,
        "columns_ok": set(actual) == required,
        "order_matches_documented": actual == DOCUMENTED_COLUMN_ORDER,
        "missing_columns": sorted(required - set(actual)),
        "unexpected_columns": sorted(set(actual) - required),
        "missing_values": int(df.isna().sum().sum()),
    }
    if not report["columns_ok"]:
        # 列不齐时直接返回，避免后续按列名取值抛出 KeyError
        report["ok"] = False
        report["error"] = "列集合与文档不一致，已跳过分布检查"
        return report

    report.update({
        "arms": {str(k): int(v) for k, v in df["segment"].value_counts().items()},
        "outcome_rate_by_arm": {
            arm: {c: float(df.loc[df["segment"] == arm, c].mean()) for c in OUTCOME_COLUMNS}
            for arm in ALL_ARMS if arm in set(df["segment"].unique())
        },
        "rows_ok": bool(len(df) == HILLSTROM_ROWS),
    })
    # conversion=1 必须伴随 visit=1（业务一致性检查）
    bad = int(((df["conversion"] == 1) & (df["visit"] == 0)).sum())
    report["conversion_without_visit"] = bad
    report["ok"] = bool(
        report["columns_ok"] and report["missing_values"] == 0
        and report["rows_ok"] and bad == 0
    )
    if not report["order_matches_documented"]:
        logger.warning("列顺序与文档不一致（不影响建模，但请更新文档）: %s", actual)
    return report


def build_analysis_frame(
    df: pd.DataFrame,
    treatment_definition: str = "any_email",
    primary_outcome: str = "visit",
) -> pd.DataFrame:
    """构造分析用表：二值 treatment + 特征 + outcome。

    treatment_definition:
      - any_email:   {Mens, Womens} -> 1, No E-Mail -> 0  （主设定，样本量最大）
      - mens_only:   Mens -> 1, No E-Mail -> 0
      - womens_only: Womens -> 1, No E-Mail -> 0
    """
    if primary_outcome not in OUTCOME_COLUMNS:
        raise ValueError(f"primary_outcome 必须是 {OUTCOME_COLUMNS} 之一")

    if treatment_definition == "any_email":
        treated = df["segment"].isin(["Mens E-Mail", "Womens E-Mail"])
        keep = pd.Series(True, index=df.index)
    elif treatment_definition == "mens_only":
        treated = df["segment"] == "Mens E-Mail"
        keep = df["segment"].isin(["Mens E-Mail", "No E-Mail"])
    elif treatment_definition == "womens_only":
        treated = df["segment"] == "Womens E-Mail"
        keep = df["segment"].isin(["Womens E-Mail", "No E-Mail"])
    else:
        raise ValueError(f"未知 treatment_definition: {treatment_definition}")

    out = df.loc[keep].copy()
    out["treatment"] = treated.loc[keep].astype(int)
    out["y"] = out[primary_outcome].astype(float)
    out["arm"] = out["segment"]
    return out.reset_index(drop=True)


def feature_matrix(f: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """返回特征矩阵与类别列名。对类别列做显式 dtype 标注，交给 LightGBM 处理。"""
    x = f[FEATURE_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for c in CATEGORICAL_COLUMNS:
        x[c] = x[c].astype("category")
    return x, list(CATEGORICAL_COLUMNS)


# --------------------------------------------------------------------------
# 半合成数据：真实 CATE 已知，用于验证指标实现与"最优策略"上界
# --------------------------------------------------------------------------
def make_synthetic(
    n: int = 20000,
    seed: int = 0,
    n_features: int = 6,
    base_rate: float = 0.10,
    uplift_strength: float = 2.0,
    mode: str = "aligned",
    confounding: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """生成带有异质处理效应的半合成数据。

    生成机制（明确写出，便于审查）：
      x ~ N(0, 1)^d
      propensity e(x) = 0.5                     （完全随机实验）
      baseline logit: b(x) = 0.4*x0 - 0.3*x1 + 0.2*x2
      mode="aligned"    : tau(x) = uplift_strength * ( 0.8*x0 - 0.6*x3)
      mode="conflicting": tau(x) = uplift_strength * (-0.8*x0 - 0.6*x3)
      P(y=1|x,t) = sigmoid(b(x) + base_logit + t * tau(x))

    mode 是本项目最关键的一组对照：
      aligned     —— 高基线人群同时也是高增量人群。此时"按响应概率排序"
                     恰好也能选对人，**会掩盖 uplift 建模的价值**。
      conflicting —— 高基线人群的增量低甚至为负（典型的"睡狗"客户：
                     本来就会买，投广告只是浪费预算）。此时只有 uplift 建模
                     才能选对人群，响应模型会系统性地选错。
    返回 (DataFrame 含 treatment/y/x_*, 真实 CATE（概率差）数组)
    """
    if mode not in ("aligned", "conflicting"):
        raise ValueError("mode 必须是 aligned 或 conflicting")

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features))

    # 处理分配：confounding=0 时为完全随机实验；>0 时倾向得分依赖 x0，
    # 制造**选择偏差**（观测数据的典型情形），用于检验 IPS/DR 能否纠正。
    e = 1.0 / (1.0 + np.exp(-float(confounding) * x[:, 0]))
    t = (rng.random(n) < e).astype(int)

    base_logit = float(np.log(base_rate / (1 - base_rate)))
    if mode == "conflicting":
        # 关键：让**基线异质性远大于效应大小**，且二者负相关。
        # 这样 P(y|t=1) 的排序几乎由基线决定（响应模型会挑中"本来就会转化"的人），
        # 而真实增量恰好与基线相反——响应模型才会真正选错人。
        b = base_logit + 1.2 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]
        tau_logit = uplift_strength * (-0.5 * x[:, 0] - 0.4 * x[:, 3 % n_features])
    else:
        b = base_logit + 0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]
        tau_logit = uplift_strength * (0.8 * x[:, 0] - 0.6 * x[:, 3 % n_features])

    p0 = 1.0 / (1.0 + np.exp(-b))
    p1 = 1.0 / (1.0 + np.exp(-(b + tau_logit)))
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(float)

    true_cate = p1 - p0  # 真实个体增量（概率差），仅半合成数据可知

    df = pd.DataFrame({f"x{i}": x[:, i] for i in range(n_features)})
    df["treatment"] = t
    df["y"] = y
    df["arm"] = np.where(t == 1, "Treated", "Control")
    # 真实倾向得分（仅半合成数据可知），供 IPS/DR 研究作为上界参照
    df["propensity_true"] = e
    return df, true_cate


def synthetic_feature_columns(n_features: int = 6) -> list:
    return [f"x{i}" for i in range(n_features)]


# --------------------------------------------------------------------------
# Criteo（手动放置后可读取；默认不可达，见文档）
# --------------------------------------------------------------------------
def load_criteo_manual(raw_dir: str) -> pd.DataFrame:
    """读取用户手动下载的 Criteo Uplift v2.1。

    官方链接（已核验存在，但本环境不可达）：
      http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
    许可：CC BY-NC-SA 4.0；引用：Diemert et al., "A Large Scale Benchmark
    for Uplift Modeling", AdKDD 2018。
    """
    path = os.path.join(raw_dir, "criteo-research-uplift-v2.1.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到 Criteo 数据。请手动下载到 data/raw/ 后重试（本项目默认不使用它）。"
        )
    nrows = None
    df = pd.read_csv(path, nrows=nrows)
    logger.info("Criteo 载入 %d 行，列: %s", len(df), list(df.columns)[:20])
    return df


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
