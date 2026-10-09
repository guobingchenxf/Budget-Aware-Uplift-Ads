from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__
from .config import get_logger, load_config

logger = get_logger()


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--set 需要 key=value 形式，收到: {p}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def cmd_download(args: argparse.Namespace) -> int:
    """下载数据（显式命令，绝不自动触发）。"""
    import urllib.request

    from .data import HILLSTROM_FILENAME, HILLSTROM_SHA256, HILLSTROM_URL, sha256_of

    cfg = load_config(args.config, _parse_overrides(args.set))
    os.makedirs(cfg.data.raw_dir, exist_ok=True)
    dst = os.path.join(cfg.data.raw_dir, HILLSTROM_FILENAME)

    if os.path.exists(dst):
        digest = sha256_of(dst)
        logger.info("文件已存在: %s (sha256=%s)", dst, digest)
        if digest == HILLSTROM_SHA256:
            logger.info("校验一致，无需重新下载。")
            return 0
        logger.warning("校验不一致，将重新下载。")

    logger.info("下载 Hillstrom 数据集: %s", HILLSTROM_URL)
    urllib.request.urlretrieve(HILLSTROM_URL, dst)  # noqa: S310 (公开只读数据集)
    digest = sha256_of(dst)
    logger.info("下载完成: %s  sha256=%s", dst, digest)
    if digest != HILLSTROM_SHA256:
        logger.error("校验失败！期望 %s，实际 %s。请检查网络或来源。", HILLSTROM_SHA256, digest)
        return 2
    logger.info("校验通过。")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .data import load_hillstrom_raw, validate_hillstrom, write_manifest

    cfg = load_config(args.config, _parse_overrides(args.set))
    raw = load_hillstrom_raw(cfg.data.raw_dir, verify=False)
    rep = validate_hillstrom(raw, path=os.path.join(cfg.data.raw_dir, "hillstrom_email_analytics.csv"))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    out = os.path.join(cfg.output.artifacts_dir, "data_report.json")
    write_manifest(out, rep)
    logger.info("数据概况已写入 %s", out)
    return 0 if rep.get("ok") else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    cfg = load_config("configs/smoke.yaml", _parse_overrides(args.set))
    cfg.output.artifacts_dir = args.artifacts or cfg.output.artifacts_dir
    res = run_experiment(cfg, tag="smoke")
    _print_headline(res)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.artifacts:
        cfg.output.artifacts_dir = args.artifacts
    res = run_experiment(cfg, tag=args.tag)
    _print_headline(res)
    return 0


def cmd_multiseed(args: argparse.Namespace) -> int:
    """多种子重训 + 配对方差分解（用于回答"策略差异是否稳定"）。"""
    from .multiseed import run_multiseed

    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.artifacts:
        cfg.output.artifacts_dir = args.artifacts
    res = run_multiseed(cfg, n_seeds=args.seeds, tag=args.tag)
    print("\n=== 多种子汇总（跨种子 gain 均值 ± 标准差）===")
    cols = [c for c in ["strategy", "n_seeds", "gain_mean", "gain_std",
                        "paired_diff_vs_random_mean", "paired_diff_vs_random_lo",
                        "paired_diff_vs_random_hi", "paired_diff_vs_random_significant"]
            if c in res["summary"].columns]
    print(res["summary"][cols].sort_values("gain_mean", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Qini 跨种子汇总 ===")
    print(res["qini_summary"].sort_values("qini_mean", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n产物目录:", res["out_dir"])
    print(f"（共 {len(res['seeds'])} 个种子: {res['seeds'][:3]} ... {res['seeds'][-1]}）")
    return 0


def cmd_confounding(args: argparse.Namespace) -> int:
    """选择偏差研究：naive / IPS / DR 三种估计量相对真值的偏差。"""
    from .causal import confounding_study

    cfg = load_config(args.config, _parse_overrides(args.set))
    os.makedirs(cfg.output.artifacts_dir, exist_ok=True)
    tab = confounding_study(n=args.n, seed=cfg.seed,
                            confounding_levels=tuple(args.levels),
                            budget_ratio=args.budget_ratio,
                            mode=args.mode)
    out = os.path.join(cfg.output.artifacts_dir, "confounding_study.csv")
    tab.to_csv(out, index=False)

    show = tab[["confounding", "policy", "truth", "naive_full", "ips", "dr",
                "naive_full_bias", "ips_bias", "dr_bias",
                "naive_qini_ratio_to_truth"]]
    print("\n=== 选择偏差强度 vs 估计量偏差（真值 = 真实 CATE 之和）===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n产物:", os.path.abspath(out))
    print("说明：confounding=0 为随机实验；>0 时倾向得分依赖 x0。")
    print("      naive_full = |S| * (mean_y_t - mean_y_c)，假设无混杂；IPS/DR 用于纠正；")
    print("      naive_qini_ratio_to_truth 显示 Qini 式估计量与真值的比例（口径不同，约为 n_t/|S|）。")
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    """阈值策略 vs 固定预算 top-k。"""
    from .threshold import threshold_study

    cfg = load_config(args.config, _parse_overrides(args.set))
    os.makedirs(cfg.output.artifacts_dir, exist_ok=True)
    frames = []
    for src in ("model", "oracle"):
        frames.append(threshold_study(n=args.n, seed=cfg.seed, mode=args.mode,
                                      score_source=src))
    tab = __import__("pandas").concat(frames, ignore_index=True)
    out = os.path.join(cfg.output.artifacts_dir, "threshold_vs_budget.csv")
    tab.to_csv(out, index=False)

    for src in ("model", "oracle"):
        sub = tab[tab.score_source == src]
        print(f"\n=== 阈值策略 vs top-k（score_source={src}, mode={args.mode}）===")
        cols = ["budget_ratio", "policy", "n_selected", "selected_fraction",
                "true_gain", "true_gain_per_cost", "neg_uplift_share"]
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n产物:", os.path.abspath(out))
    print("说明：neg_uplift_share = 选中集合中真实增量为负的比例。")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """训练全部策略模型并保存到磁盘（供在线服务加载）。"""
    from .metrics import qini_coefficient
    from .persistence import ModelBundle, save_bundle
    from .pipeline import fit_all_models, prepare_data

    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.rows:
        cfg.data.max_rows = int(args.rows)
    data = prepare_data(cfg)
    scores, _diagnostics, fitted = fit_all_models(data, cfg, keep_models=True)

    y_ev = data.eval_["y"].values.astype(float)
    t_ev = data.eval_["treatment"].values.astype(int)
    metrics = {name: float(qini_coefficient(y_ev, t_ev, sc)) for name, sc in scores.items()}

    bundle = ModelBundle(models=fitted, feature_cols=data.feature_cols,
                         cat_cols=data.cat_cols, dataset=cfg.data.name,
                         seed=cfg.seed, metrics=metrics)
    path = save_bundle(bundle, os.path.join(cfg.output.artifacts_dir, "models", args.tag))

    print("\n=== 已保存策略模型（Qini）===")
    for k, v in sorted(metrics.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:.4f}")
    print(f"\n模型目录: {path}")
    print(f"启动服务: .venv\\Scripts\\python.exe -m baua.cli serve --models {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """启动在线推理服务（研究原型，无鉴权）。"""
    from .serve import app_from_dir

    app = app_from_dir(args.models)
    print(f"\n启动服务: http://{args.host}:{args.port}  (模型: {args.models})")
    print("接口: GET /health, GET /strategies, POST /score, POST /allocate")
    print("提醒：这是研究原型，无鉴权/限流/在线特征，请勿用于生产。\n")
    import uvicorn
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


def _print_headline(res: dict[str, Any]) -> None:
    print("\n=== 排序指标（真实数据上的观测估计）===")
    print(res["ranking"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== 预算约束分配 ===")
    print(res["budget"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== 投放节奏模拟（grounded={}）===".format(res["pacing_grounded"]))
    print(res["pacing"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n产物目录:", res["artifacts_dir"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baua", description="Budget-Aware Uplift Ads CLI")
    p.add_argument("--version", action="version", version=f"baua {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("download", cmd_download),
                     ("inspect-data", cmd_inspect),
                     ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--set", nargs="*", help="覆盖配置，如 data.max_rows=20000")
        if name == "run":
            sp.add_argument("--tag", default="main")
            sp.add_argument("--artifacts", default=None)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("smoke", help="使用 configs/smoke.yaml 跑端到端小样本")
    sp.add_argument("--set", nargs="*")
    sp.add_argument("--artifacts", default="_smoke_out")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("multiseed", help="多种子重训 + 配对方差分解")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--set", nargs="*")
    sp.add_argument("--seeds", type=int, default=10, help="种子个数")
    sp.add_argument("--tag", default="multiseed")
    sp.add_argument("--artifacts", default=None)
    sp.set_defaults(func=cmd_multiseed)

    sp = sub.add_parser("confounding", help="选择偏差研究：naive vs IPS vs DR")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--set", nargs="*")
    sp.add_argument("--n", type=int, default=40000)
    sp.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    sp.add_argument("--budget-ratio", type=float, default=0.05)
    sp.add_argument("--mode", default="conflicting", choices=["aligned", "conflicting"])
    sp.set_defaults(func=cmd_confounding)

    sp = sub.add_parser("threshold", help="阈值策略 vs 固定预算 top-k")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--set", nargs="*")
    sp.add_argument("--n", type=int, default=40000)
    sp.add_argument("--mode", default="conflicting", choices=["aligned", "conflicting"])
    sp.set_defaults(func=cmd_threshold)

    sp = sub.add_parser("fit", help="训练并保存策略模型（供 serve 使用）")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--set", nargs="*")
    sp.add_argument("--rows", type=int, default=None, help="限制训练样本量")
    sp.add_argument("--tag", default="served")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("serve", help="启动在线推理服务（研究原型）")
    sp.add_argument("--models", default="artifacts/models/served",
                    help="由 fit 命令生成的模型目录")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
