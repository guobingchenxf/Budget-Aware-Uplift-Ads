from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .budget import (
    bootstrap_gain_ci,
    budget_units_for_ratio,
    sensitivity_over_budget,
    sensitivity_over_noise,
    simulate_budget_allocation,
)
from .config import Config, get_logger
from .data import (
    CATEGORICAL_COLUMNS,
    assert_no_leakage,
    build_analysis_frame,
    hillstrom_feature_columns,
    load_hillstrom_raw,
    make_synthetic,
    validate_hillstrom,
)
from .metrics import auuc, calibration_report, cate_calibration, qini_coefficient, qini_curve
from .models import build_model
from .pacing import compare_pacing_strategies, simulate_pacing

logger = get_logger()

# matplotlib 必须在导入 pyplot 之前切换为无显示后端（E402 是这条约束的必然代价）
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    train: pd.DataFrame
    eval_: pd.DataFrame
    feature_cols: list[str]
    cat_cols: list[str]
    y_col: str
    true_cate_eval: np.ndarray | None = None
    report: dict[str, Any] = field(default_factory=dict)


def _strat_key(df: pd.DataFrame, y_col: str) -> pd.Series:
    """分层键：treatment × 结果分箱（连续结果先分箱）。"""
    y = df[y_col].astype(float)
    if y.nunique() > 2:
        ybin = pd.qcut(y.rank(method="first"), 5, labels=False, duplicates="drop")
    else:
        ybin = y.astype(int)
    return df["treatment"].astype(str) + "_" + pd.Series(ybin, index=df.index).astype(str)


def prepare_data(cfg: Config) -> PreparedData:
    rng = np.random.default_rng(cfg.seed)
    report: dict[str, Any] = {"dataset": cfg.data.name}

    if cfg.data.name == "hillstrom":
        raw = load_hillstrom_raw(cfg.data.raw_dir)
        report["raw_validation"] = validate_hillstrom(raw)
        frame = build_analysis_frame(raw, cfg.data.treatment_definition, cfg.data.primary_outcome)
        report["treatment_definition"] = cfg.data.treatment_definition
        true_cate = None
        if cfg.data.max_rows:
            frame = frame.sample(n=min(int(cfg.data.max_rows), len(frame)),
                                 random_state=cfg.seed).reset_index(drop=True)
    elif cfg.data.name == "synthetic":
        n = int(cfg.data.max_rows or 20000)
        frame, true_cate_all = make_synthetic(n=n, seed=cfg.seed,
                                              mode=cfg.data.synthetic_mode)
        frame = frame.rename(columns={})
        frame["y"] = frame["y"].astype(float)
        report["synthetic"] = {"generator": "make_synthetic", "n": n,
                               "mode": cfg.data.synthetic_mode,
                               "note": "半合成：真实 CATE 已知，用于验证指标与上界"}
        true_cate = true_cate_all
    else:
        raise ValueError(f"未知数据集: {cfg.data.name}（可选 hillstrom / synthetic）")

    if cfg.data.name == "synthetic":
        feature_cols = [c for c in frame.columns if c.startswith("x")]
        cat_cols: list[str] = []
    else:
        # 显式白名单 + 泄漏断言（不要用"排除法"，那正是本项目踩过的坑）
        feature_cols = hillstrom_feature_columns()
        cat_cols = list(CATEGORICAL_COLUMNS)
        assert_no_leakage(feature_cols)
        for c in cat_cols:
            frame[c] = frame[c].astype("category")

    # 分层划分 train / eval
    key = _strat_key(frame, "y")
    uniq = key.unique()
    eval_idx: list[int] = []
    for u in uniq:
        idx = np.flatnonzero(key.values == u)
        if len(idx) == 0:
            continue
        n_eval = max(1, int(round(len(idx) * cfg.data.test_size * cfg.data.eval_size)))
        n_eval = min(n_eval, len(idx))
        eval_idx.extend(rng.choice(idx, size=n_eval, replace=False).tolist())
    mask_eval = np.zeros(len(frame), dtype=bool)
    mask_eval[np.array(sorted(set(eval_idx)), dtype=int)] = True

    train = frame.loc[~mask_eval].reset_index(drop=True)
    eval_ = frame.loc[mask_eval].reset_index(drop=True)

    true_cate_eval = None
    if true_cate is not None:
        true_cate_eval = np.asarray(true_cate)[np.flatnonzero(mask_eval)]

    report.update({
        "n_total": int(len(frame)),
        "n_train": int(len(train)),
        "n_eval": int(len(eval_)),
        "treatment_rate_train": float(train["treatment"].mean()),
        "treatment_rate_eval": float(eval_["treatment"].mean()),
        "outcome_rate_train": float(train["y"].mean()),
        "outcome_rate_eval": float(eval_["y"].mean()),
        "outcome_rate_by_arm_eval": {
            str(k): float(v) for k, v in eval_.groupby("arm")["y"].mean().items()},
    })
    logger.info("数据准备完成: train=%d eval=%d 特征=%d", len(train), len(eval_), len(feature_cols))
    return PreparedData(train, eval_, feature_cols, cat_cols, "y", true_cate_eval, report)


# --------------------------------------------------------------------------
# 建模与打分
# --------------------------------------------------------------------------
def fit_all_models(
    data: PreparedData, cfg: Config, keep_models: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """训练全部策略模型并打分。

    返回 (scores, diagnostics, fitted_models)。
    `fitted_models` 只在 keep_models=True 时非空，用于持久化（供在线服务加载）。
    """
    train, ev = data.train, data.eval_
    Xtr = train[data.feature_cols]
    Xev = ev[data.feature_cols]
    t_tr, y_tr = train["treatment"].values, train["y"].values

    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    fitted: dict[str, Any] = {}

    for name in cfg.budget.strategies:
        logger.info("训练策略模型: %s", name)
        model = build_model(name, cfg.model, cfg.seed)
        model.fit(Xtr, t_tr, y_tr)
        scores[name] = np.asarray(model.predict(Xev), dtype=float)
        if keep_models:
            fitted[name] = model
        diagnostics[name] = {"score_kind": model.score_kind,
                            "score_mean": float(np.mean(scores[name])),
                            "score_std": float(np.std(scores[name]))}
        if name == "class_transform":
            diagnostics[name]["balanced_n"] = getattr(model, "balanced_n_", None)
        if name == "x_learner":
            diagnostics[name]["propensity"] = getattr(model, "propensity_", None)

    # 随机策略在同一个 eval 集上生成（可复现）
    if "random" not in scores:
        scores["random"] = np.random.default_rng(cfg.seed).random(len(ev))

    if data.true_cate_eval is not None:
        scores["oracle"] = np.asarray(data.true_cate_eval, dtype=float)
        diagnostics["oracle"] = {"score_kind": "true_cate",
                                 "note": "仅半合成数据可用，作为理论上界"}

    return scores, diagnostics, fitted


def _calibration_study(data: PreparedData, cfg: Config) -> dict[str, Any]:
    """概率校准研究：校准能否改善排序类决策与阈值类决策？

    关键设计：
    - 校准器**只在训练集内部切出的独立验证集上拟合**（且只用处理组样本，
      因为响应模型预测的是 P(y|t=1)），**绝不使用评估集**——否则就是另一种标签泄漏；
    - 先验证"不变量"：isotonic 是单调映射，top-k 选择与增量收益理论上不应改变；
      这个函数把"不该有差异"也实测一遍（可被证伪）；
    - 再验证"敏感项"：绝对阈值策略（"预测率低于 X 不投"）对校准高度敏感。
    """
    from .calibration import (
        IsotonicCalibrator,
        SigmoidCalibrator,
        compare_calibration_effect,
        threshold_policy_table,
    )
    from .metrics import calibration_report
    from .models import ResponseModel

    feats = data.feature_cols
    y_ev = data.eval_["y"].values.astype(float)
    t_ev = data.eval_["treatment"].values.astype(int)

    treated_train = data.train[data.train["treatment"] == 1].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(treated_train))
    val_idx: list[int] = []
    y_tr = treated_train["y"].values
    for label in np.unique(y_tr):
        sub = idx[y_tr == label]
        n_val = max(1, int(round(len(sub) * cfg.experiment.calibration_val_size)))
        val_idx.extend(rng.choice(sub, size=min(n_val, len(sub)), replace=False).tolist())
    val_mask = np.zeros(len(treated_train), dtype=bool)
    val_mask[sorted(set(val_idx))] = True
    fit_part = treated_train.loc[~val_mask].reset_index(drop=True)
    val_part = treated_train.loc[val_mask].reset_index(drop=True)

    model = ResponseModel(cfg.model, cfg.seed)
    model.fit(fit_part[feats], np.ones(len(fit_part)), fit_part["y"].values)
    raw_val = np.asarray(model.predict(val_part[feats]), dtype=float)
    y_val = val_part["y"].values

    calibrators: dict[str, Any] = {"isotonic": IsotonicCalibrator(),
                                   "platt": SigmoidCalibrator()}
    fitted = {name: cal.fit(raw_val, y_val) for name, cal in calibrators.items()}

    raw_ev = np.asarray(model.predict(data.eval_[feats]), dtype=float)
    m = t_ev == 1
    rep_raw = calibration_report(y_ev[m], raw_ev[m])

    score_dict: dict[str, np.ndarray] = {"response_raw": raw_ev}
    info: dict[str, Any] = {
        "n_fit": int(len(fit_part)),
        "n_val": int(len(val_part)),
        "ece_before": float(rep_raw["ece"]),
        "mean_abs_bin_gap_before": float(rep_raw["mean_abs_bin_gap"]),
        "note": ("校准器在训练集内部切出的独立验证集上拟合（仅处理组样本）；"
                 "isotonic 为分段常数映射会引入并列，Platt 为严格单调映射"),
        "methods": {},
    }

    invariance_frames = []
    for name, cal in fitted.items():
        sc = cal.transform(raw_ev)
        score_dict[f"response_{name}"] = sc
        rep = calibration_report(y_ev[m], sc[m])
        inv = compare_calibration_effect(raw_ev, sc, y_ev, t_ev)
        inv.insert(0, "calibrator", name)
        invariance_frames.append(inv)
        info["methods"][name] = {
            "fitted": bool(cal.fitted),
            "ece_after": float(rep["ece"]),
            "mean_abs_bin_gap_after": float(rep["mean_abs_bin_gap"]),
            "n_unique_eval": int(len(np.unique(sc))),
            "n_unique_raw_eval": int(len(np.unique(raw_ev))),
            "ties_introduced": int(len(np.unique(raw_ev)) - len(np.unique(sc))),
            "coef": float(getattr(cal, "coef_", float("nan"))),
        }

    invariance = pd.concat(invariance_frames, ignore_index=True)
    thresholds = threshold_policy_table(score_dict, y_ev, t_ev, cfg.experiment.thresholds)
    return {"info": info, "invariance": invariance, "thresholds": thresholds,
            "bins_before": rep_raw["bins"], "score_dict": score_dict}


# --------------------------------------------------------------------------
# 产物与绘图
# --------------------------------------------------------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def plot_qini(y, t, scores: dict[str, np.ndarray], path: str) -> None:
    plt.figure(figsize=(6.2, 4.2), dpi=140)
    for name, sc in scores.items():
        cur = qini_curve(y, t, sc)
        total = cur["qini"][-1]
        if abs(total) < 1e-12:
            continue
        plt.plot(cur["fractions"], cur["qini"] / total, label=name, linewidth=1.4)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="random")
    plt.xlabel("targeted fraction")
    plt.ylabel("normalized cumulative gain")
    plt.title("Qini curves (normalized)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_budget_sensitivity(df: pd.DataFrame, path: str, ylabel: str = "gain") -> None:
    plt.figure(figsize=(6.2, 4.2), dpi=140)
    for name, g in df.groupby("strategy"):
        g = g.sort_values("budget_ratio")
        plt.plot(g["budget_ratio"], g[ylabel], marker="o", label=name, linewidth=1.3)
    plt.xlabel("budget ratio (of full-population treatment cost)")
    plt.ylabel(ylabel)
    plt.title("Budget sensitivity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pacing_spend(res_by_strategy: dict[str, dict[str, Any]], path: str) -> None:
    plt.figure(figsize=(6.2, 4.2), dpi=140)
    for name, res in res_by_strategy.items():
        tab = res["slots"]
        plt.plot(tab["slot"], tab["cum_spend"], marker=".", label=name, linewidth=1.3)
    plt.xlabel("slot")
    plt.ylabel("cumulative spend (simulated)")
    plt.title("Pacing: cumulative spend by slot (simulated)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------------------------------
# 主实验
# --------------------------------------------------------------------------
def run_experiment(cfg: Config, tag: str = "default") -> dict[str, Any]:
    data = prepare_data(cfg)
    y_ev = data.eval_["y"].values.astype(float)
    t_ev = data.eval_["treatment"].values.astype(int)

    scores, diagnostics, _ = fit_all_models(data, cfg)

    # 1) 排序指标（真实观测结构上的 Qini/AUUC）
    rows = []
    for name, sc in scores.items():
        rows.append({"strategy": name,
                     "qini": qini_coefficient(y_ev, t_ev, sc),
                     "auuc": auuc(y_ev, t_ev, sc)})
    ranking = pd.DataFrame(rows).sort_values("qini", ascending=False)

    # 2) 校准（响应模型 vs 处理组真实结果）
    calib: dict[str, Any] = {}
    if "response" in scores:
        m = t_ev == 1
        calib["response_on_treated"] = calibration_report(y_ev[m], scores["response"][m])

    # 3) 半合成数据：CATE 校准 + 上界
    if data.true_cate_eval is not None:
        cate_rows = []
        for name, sc in scores.items():
            if name == "random":
                continue
            cc = cate_calibration(sc, data.true_cate_eval)
            cate_rows.append({"strategy": name, "cate_mae": cc["mae_binned"],
                              "cate_pearson": cc["pearson"], "cate_bias": cc["bias"]})
        calib["cate"] = cate_rows

    # 4) 预算约束分配（主场景）+ 自助法区间
    budget_units = budget_units_for_ratio(len(y_ev), cfg.budget.budget_ratio,
                                          cfg.budget.cost_per_treatment)
    strategies_no_oracle = {k: v for k, v in scores.items() if k != "oracle"}
    budget_table = simulate_budget_allocation(
        strategies_no_oracle, y_ev, t_ev, budget_units,
        cfg.budget.cost_per_treatment, 0.0, cfg.seed)
    budget_ci = bootstrap_gain_ci(strategies_no_oracle, y_ev, t_ev, budget_units,
                                  cfg.budget.cost_per_treatment,
                                  n_boot=cfg.bootstrap.n_boot,
                                  alpha=cfg.bootstrap.alpha,
                                  seed=cfg.seed)

    # 5) 敏感性分析（可由 experiment.run_sensitivity 关闭，用于多种子批量运行）
    if cfg.experiment.run_sensitivity:
        sb = sensitivity_over_budget(strategies_no_oracle,
                                     y_ev, t_ev, cfg.sensitivity.budget_ratios,
                                     cfg.budget.cost_per_treatment, cfg.seed)
        sn = sensitivity_over_noise(strategies_no_oracle,
                                    y_ev, t_ev, budget_units,
                                    cfg.sensitivity.score_noise_levels,
                                    cfg.budget.cost_per_treatment, cfg.seed)
        sens_sample = _sample_size_sensitivity(data, cfg, budget_units)
    else:
        sb = pd.DataFrame()
        sn = pd.DataFrame()
        sens_sample = pd.DataFrame()

    # 5b) 概率校准研究（与排序不变量验证 + 阈值策略）
    calib_study = _calibration_study(data, cfg) if cfg.experiment.run_calibration else None

    # 6) 节奏模拟（明确标注是否为"有真实 CATE 支撑"的模拟）
    paced = _run_pacing(data, scores, cfg) if cfg.experiment.run_pacing else {
        "table": pd.DataFrame(), "details": {}, "grounded": False,
        "note": "本次运行关闭了节奏模拟（experiment.run_pacing=false）", "scenarios": []}

    # 7) 落盘
    out_dir = _ensure_dir(os.path.join(cfg.output.artifacts_dir, f"{cfg.data.name}_{tag}"))
    ranking.to_csv(os.path.join(out_dir, "ranking_metrics.csv"), index=False)
    budget_table.to_csv(os.path.join(out_dir, "budget_allocation.csv"), index=False)
    budget_ci.to_csv(os.path.join(out_dir, "budget_bootstrap_ci.csv"), index=False)
    if not sb.empty:
        sb.to_csv(os.path.join(out_dir, "sensitivity_budget.csv"), index=False)
    if not sn.empty:
        sn.to_csv(os.path.join(out_dir, "sensitivity_noise.csv"), index=False)
    if not sens_sample.empty:
        sens_sample.to_csv(os.path.join(out_dir, "sensitivity_sample_size.csv"), index=False)
    if not paced["table"].empty:
        paced["table"].to_csv(os.path.join(out_dir, "pacing_strategies.csv"), index=False)
        for name, res in paced["details"].items():
            res["slots"].to_csv(os.path.join(out_dir, f"pacing_slots_{name}.csv"), index=False)

    if calib_study is not None:
        calib_study["invariance"].to_csv(
            os.path.join(out_dir, "calibration_invariance.csv"), index=False)
        calib_study["thresholds"].to_csv(
            os.path.join(out_dir, "calibration_thresholds.csv"), index=False)

    plot_qini(y_ev, t_ev, scores, os.path.join(out_dir, "qini_curves.png"))
    if not sb.empty:
        plot_budget_sensitivity(sb, os.path.join(out_dir, "budget_sensitivity.png"))
    if paced["details"]:
        relaxed = {k.replace("relaxed_supply__", ""): v for k, v in paced["details"].items()
                   if k.startswith("relaxed_supply__")}
        plot_pacing_spend(relaxed, os.path.join(out_dir, "pacing_cum_spend_relaxed.png"))
        constrained = {k.replace("constrained_supply__", ""): v for k, v in paced["details"].items()
                       if k.startswith("constrained_supply__")}
        plot_pacing_spend(constrained, os.path.join(out_dir, "pacing_cum_spend_constrained.png"))

    summary = {
        "project": "Budget-Aware Uplift Ads",
        "version": __version__,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": cfg.source_path,
        "config": cfg.to_dict(),
        "data_report": data.report,
        "model_diagnostics": diagnostics,
        "budget_units": int(budget_units),
        "pacing_grounded": paced["grounded"],
        "pacing_note": paced["note"],
        "artifacts_dir": os.path.abspath(out_dir),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    calib_out = dict(calib)
    if calib_study is not None:
        calib_out["calibration_study"] = calib_study["info"]
        calib_out["bins_before"] = calib_study["bins_before"]
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib_out, f, ensure_ascii=False, indent=2)

    logger.info("实验完成，产物目录: %s", os.path.abspath(out_dir))
    return {
        "summary": summary,
        "ranking": ranking,
        "budget": budget_table,
        "budget_ci": budget_ci,
        "sensitivity_budget": sb,
        "sensitivity_noise": sn,
        "sensitivity_sample": sens_sample,
        "pacing": paced["table"],
        "pacing_details": paced["details"],
        "pacing_grounded": paced["grounded"],
        "calibration": calib,
        "calibration_study": calib_study,
        "artifacts_dir": os.path.abspath(out_dir),
    }


def _sample_size_sensitivity(data: PreparedData, cfg: Config, budget_units: int) -> pd.DataFrame:
    """样本量敏感性：训练集规模变化对策略排序的影响。"""
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for n in cfg.sensitivity.sample_sizes:
        n = min(int(n), len(data.train))
        idx = rng.choice(len(data.train), size=n, replace=False)
        sub = data.train.iloc[np.sort(idx)].reset_index(drop=True)
        Xtr = sub[data.feature_cols]
        y_ev = data.eval_["y"].values.astype(float)
        t_ev = data.eval_["treatment"].values.astype(int)
        for name in cfg.budget.strategies:
            model = build_model(name, cfg.model, cfg.seed)
            model.fit(Xtr, sub["treatment"].values, sub["y"].values)
            sc = np.asarray(model.predict(data.eval_[data.feature_cols]), dtype=float)
            rows.append({"strategy": name, "train_n": n,
                         "qini": qini_coefficient(y_ev, t_ev, sc)})
    return pd.DataFrame(rows)


def _run_pacing(data: PreparedData, scores: dict[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    """节奏模拟。value 的来源决定结果是否"有真实依据"。

    跑两个供给场景（这是本项目 pacing 研究的关键对照）：
      relaxed_supply     : 流量在各时段比较均匀，任何策略都花得出去钱；
      constrained_supply : 流量高度集中在晚间，早时段"想花也花不出去"。
    预期：供给受限时，均匀策略会留下未花完的预算，而反馈控制器能把钱追回来。
    """
    if data.true_cate_eval is not None:
        value = np.asarray(data.true_cate_eval, dtype=float)
        grounded, note = True, "value = 半合成数据的真实 CATE；节奏结论有真值支撑（时段设定仍为模拟）"
        score = scores.get("oracle", scores["t_learner"])
    else:
        best = None
        for k in ("x_learner", "t_learner", "s_learner"):
            if k in scores:
                best = scores[k]
                break
        if best is None:
            raise RuntimeError("缺少 uplift 分数，无法做节奏模拟")
        value = np.asarray(best, dtype=float)
        score = np.asarray(best, dtype=float)
        grounded, note = False, ("value = 模型预测 uplift（非真实增量）；"
                                 "节奏结论为模型驱动的模拟，不能当作观测事实")

    scenarios = {"relaxed_supply": 0.0, "constrained_supply": 0.9}
    tables = []
    details: dict[str, dict[str, Any]] = {}
    for sc_name, conc in scenarios.items():
        tbl = compare_pacing_strategies(
            score, value, cfg.pacing.strategies,
            n_slots=cfg.pacing.n_slots,
            budget_ratio=cfg.budget.budget_ratio,
            cost_per_treatment=cfg.budget.cost_per_treatment,
            damping=cfg.pacing.damping,
            peak_boost=cfg.pacing.prime_slot_boost,
            supply_concentration=conc,
            seed=cfg.seed)
        tbl["scenario"] = sc_name
        tbl["supply_concentration"] = conc
        tables.append(tbl)
        for st in cfg.pacing.strategies:
            details[f"{sc_name}__{st}"] = simulate_pacing(
                score, value, n_slots=cfg.pacing.n_slots,
                budget_ratio=cfg.budget.budget_ratio,
                cost_per_treatment=cfg.budget.cost_per_treatment,
                strategy=st, damping=cfg.pacing.damping,
                peak_boost=cfg.pacing.prime_slot_boost,
                supply_concentration=conc, seed=cfg.seed)

    combined = pd.concat(tables, ignore_index=True)
    return {"table": combined, "details": details, "grounded": grounded,
            "note": note, "scenarios": list(scenarios)}
