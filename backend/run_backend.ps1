# Convenience launcher for the FastAPI backend (Windows PowerShell).
# Usage:  ./run_backend.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path "data_cache/train.bin")) {
    Write-Host "No data found -> downloading + tokenizing dataset..." -ForegroundColor Yellow
    python prepare_data.py --max-rows 3000
}
if (-not (Test-Path "checkpoints/pretrain/model_config.json")) {
    Write-Host "No checkpoint found -> pretraining a tiny model..." -ForegroundColor Yellow
    python train.py --max-iters 2000 --eval-interval 500
}
Write-Host "Starting API on http://localhost:8000 ..." -ForegroundColor Green
python server.py
