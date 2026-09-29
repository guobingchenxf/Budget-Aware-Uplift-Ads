#!/usr/bin/env bash
# 端到端运行（Linux / WSL / Git Bash）
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x ".venv/Scripts/python.exe" ] && PY=".venv/Scripts/python.exe"

echo "[1/5] 数据自检";        "$PY" -m baua.cli inspect-data
echo "[2/5] 单元测试";        "$PY" -m pytest tests -q
echo "[3/5] 冒烟测试";        "$PY" -m baua.cli smoke
echo "[4/5] 主实验 Hillstrom"; "$PY" -m baua.cli run --config configs/default.yaml --tag main
echo "[5/5] 对照实验"
"$PY" -m baua.cli run --config configs/default.yaml --tag synthetic --set data.name=synthetic data.max_rows=40000
"$PY" -m baua.cli run --config configs/default.yaml --tag synthetic_conflicting --set data.name=synthetic data.max_rows=40000 data.synthetic_mode=conflicting
echo "全部完成，产物见 artifacts/"
