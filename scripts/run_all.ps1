# 端到端运行（Windows PowerShell）
# 用法： powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "未找到 .venv，请先运行 scripts\download_data.ps1" }

Write-Host "`n[1/5] 数据自检" -ForegroundColor Cyan
& $py -m baua.cli inspect-data

Write-Host "`n[2/5] 单元测试" -ForegroundColor Cyan
& $py -m pytest tests -q

Write-Host "`n[3/5] 冒烟测试（半合成小样本，约 20 秒）" -ForegroundColor Cyan
& $py -m baua.cli smoke

Write-Host "`n[4/5] 主实验：Hillstrom（真实公开数据，约 1.5 分钟）" -ForegroundColor Cyan
& $py -m baua.cli run --config configs\default.yaml --tag main

Write-Host "`n[5/5] 对照实验：半合成 aligned / conflicting" -ForegroundColor Cyan
& $py -m baua.cli run --config configs\default.yaml --tag synthetic --set data.name=synthetic data.max_rows=40000
& $py -m baua.cli run --config configs\default.yaml --tag synthetic_conflicting --set data.name=synthetic data.max_rows=40000 data.synthetic_mode=conflicting

Write-Host "`n全部完成，产物见 artifacts\" -ForegroundColor Green
