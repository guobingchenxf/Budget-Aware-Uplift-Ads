"""多种子重训与配对方差分解。

为什么必须做（原项目的缺口）：
    自助法（bootstrap）只覆盖"评估集抽样"的不确定性，
    **不包含模型训练本身的方差**。而模型重训的随机性
    （数据划分、LightGBM 的 subsample/colsample、初始化）同样会让结果波动。

做法：
    1. 对 N 个种子各跑一次完整流程（含数据划分与模型重训）；
    2. 报告每个策略 gain 的 **均值 ± 标准差**（跨种子）；
    3. 做**配对比较**：同一 seed 内先算 d = gain_strategy − gain_random，
       再对 d 求均值与 t 区间。
       配对能显著降低方差，因为同一次实验的划分偏置被消掉了——
       这是评估策略差异的正确做法。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config, get_logger

logger = get_logger()


def aggregate_multiseed(
    per_seed: pd.DataFrame,
    baseline: str = "random",
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """把逐种子的结果聚合成"均值±标准差"与"配对差异"两张表。

    参数
    ----
    per_seed : 必须包含列 [seed, strategy, gain]（可选 qini）
    """
    need = {"seed", "strategy", "gain"}
    if not need.issubset(per_seed.columns):
        raise ValueError(f"per_seed 缺少列: {sorted(need - set(per_seed.columns))}")

    piv = per_seed.pivot_table(index="seed", columns="strategy", values="gain", aggfunc="mean")
    summary = pd.DataFrame({
        "n_seeds": piv.count(),
        "gain_mean": piv.mean(),
        "gain_std": piv.std(ddof=1),
    })
    summary["gain_cv"] = summary["gain_std"] / summary["gain_mean"].abs().replace(0, np.nan)

    if baseline in piv.columns:
        diff = piv.sub(piv[baseline], axis=0)
        n = diff.count()
        mean_d = diff.mean()
        std_d = diff.std(ddof=1)
        # 小样本用 t 分布；scipy 可用时取精确分位数
        try:
            from scipy.stats import t as _t
            tcrit = {c: float(_t.ppf(1 - alpha / 2, max(1, int(n[c]) - 1))) for c in diff.columns}
        except Exception:  # pragma: no cover
            tcrit = {c: 1.96 for c in diff.columns}
        sem = std_d / np.sqrt(n)
        summary[f"paired_diff_vs_{baseline}_mean"] = mean_d
        summary[f"paired_diff_vs_{baseline}_std"] = std_d
        summary[f"paired_diff_vs_{baseline}_lo"] = mean_d - pd.Series(tcrit) * sem
        summary[f"paired_diff_vs_{baseline}_hi"] = mean_d + pd.Series(tcrit) * sem
        summary[f"paired_diff_vs_{baseline}_significant"] = (
            (summary[f"paired_diff_vs_{baseline}_lo"] > 0)
            | (summary[f"paired_diff_vs_{baseline}_hi"] < 0)
        )

    summary = summary.reset_index().rename(columns={"index": "strategy"})
    return {"summary": summary, "per_seed_pivot": piv.reset_index(),
            "paired_diff": diff.reset_index() if baseline in piv.columns else pd.DataFrame()}


def run_multiseed(cfg: Config, n_seeds: int, tag: str = "multiseed",
                  strategies: Optional[Sequence[str]] = None,
                  seeds: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """对多个种子各跑一次完整流程并聚合。

    注意：为控制耗时，多种子运行会**关闭敏感性分析与节奏模拟**
    （它们与"策略差异是否显著"这一问题无关，且会让单次耗时翻倍）。
    """
    from .pipeline import run_experiment

    if seeds is None:
        seeds = [int(cfg.seed) + 1000 * i for i in range(int(n_seeds))]
    seeds = list(seeds)

    rows_per_seed: List[Dict[str, object]] = []
    for i, sd in enumerate(seeds):
        c = cfg.clone()
        c.seed = int(sd)
        c.experiment.run_sensitivity = False
        c.experiment.run_pacing = False
        logger.info("多种子进度 %d/%d (seed=%d)", i + 1, len(seeds), sd)
        res = run_experiment(c, tag=f"{tag}_seed{sd}")
        rank = res["ranking"].set_index("strategy")["qini"].to_dict()
        for _, r in res["budget"].iterrows():
            rows_per_seed.append({
                "seed": int(sd),
                "strategy": r["strategy"],
                "gain": float(r["gain"]),
                "qini": float(rank.get(r["strategy"], np.nan)),
            })

    per_seed = pd.DataFrame(rows_per_seed)
    agg = aggregate_multiseed(per_seed, baseline="random", alpha=cfg.bootstrap.alpha)

    # 额外做一次 qini 的跨种子汇总（排序指标是否同样不稳定）
    piv_q = per_seed.pivot_table(index="seed", columns="strategy", values="qini", aggfunc="mean")
    qini_summary = pd.DataFrame({
        "n_seeds": piv_q.count(),
        "qini_mean": piv_q.mean(),
        "qini_std": piv_q.std(ddof=1),
    }).reset_index().rename(columns={"index": "strategy"})

    out_dir = cfg.output.artifacts_dir
    import os
    os.makedirs(out_dir, exist_ok=True)
    per_seed.to_csv(os.path.join(out_dir, f"{tag}_per_seed.csv"), index=False)
    agg["summary"].to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False)
    qini_summary.to_csv(os.path.join(out_dir, f"{tag}_qini_summary.csv"), index=False)
    if not agg["paired_diff"].empty:
        agg["paired_diff"].to_csv(os.path.join(out_dir, f"{tag}_paired_diff.csv"), index=False)

    logger.info("多种子完成：%d 个种子，产物写入 %s", len(seeds), os.path.abspath(out_dir))
    return {"per_seed": per_seed, "summary": agg["summary"], "qini_summary": qini_summary,
            "paired_diff": agg["paired_diff"], "seeds": seeds,
            "out_dir": os.path.abspath(out_dir)}
