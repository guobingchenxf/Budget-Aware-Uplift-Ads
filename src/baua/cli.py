"""命令行入口。

用法（全部在项目根目录执行，使用项目自带虚拟环境）：
  .venv\\Scripts\\python.exe -m baua.cli download
  .venv\\Scripts\\python.exe -m baua.cli inspect-data
  .venv\\Scripts\\python.exe -m baua.cli smoke
  .venv\\Scripts\\python.exe -m baua.cli run --config configs/default.yaml --tag main
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from . import __version__
from .config import get_logger, load_config

logger = get_logger()


def _parse_overrides(pairs: Optional[List[str]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
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


def _print_headline(res: Dict[str, object]) -> None:
    print("\n=== 排序指标（真实数据上的观测估计）===")
    print(res["ranking"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== 预算约束分配 ===")
    print(res["budget"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== 投放节奏模拟（grounded=%s）===" % res["pacing_grounded"])
    print(res["pacing"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n产物目录:", res["artifacts_dir"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baua", description="Budget-Aware Uplift Ads CLI")
    p.add_argument("--version", action="version", version=f"baua {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, needs_cfg in (("download", cmd_download, True),
                                ("inspect-data", cmd_inspect, True),
                                ("run", cmd_run, True)):
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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
