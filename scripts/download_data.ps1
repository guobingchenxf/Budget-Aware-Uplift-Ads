# 下载 Hillstrom 数据集（Windows PowerShell）
# 用法： powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "未找到 .venv，正在创建虚拟环境..." -ForegroundColor Yellow
    py -3.10 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e .
}

Write-Host "下载并校验数据（Hillstrom, 64,000 行, 约 4MB）..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m baua.cli download
Write-Host "完成。" -ForegroundColor Green
