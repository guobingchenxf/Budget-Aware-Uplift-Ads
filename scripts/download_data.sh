#!/usr/bin/env bash
# 下载 Hillstrom 数据集（Linux / WSL / Git Bash）
# 用法： bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"      # Windows(Git Bash)
fi

if [ ! -x "$PY" ]; then
  echo "未找到 .venv，正在创建..."
  python3 -m venv .venv
  PY=".venv/bin/python"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -e .
fi

echo "下载并校验数据（Hillstrom, 64,000 行, 约 4MB）..."
"$PY" -m baua.cli download
echo "完成。"
