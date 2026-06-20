#!/usr/bin/env bash
# Convenience launcher for the FastAPI backend (bash / WSL / Linux).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f data_cache/train.bin ]; then
  echo "No data found -> downloading + tokenizing dataset..."
  python prepare_data.py --max-rows 3000
fi
if [ ! -f checkpoints/pretrain/model_config.json ]; then
  echo "No checkpoint found -> pretraining a tiny model..."
  python train.py --max-iters 2000 --eval-interval 500
fi
echo "Starting API on http://localhost:8000 ..."
python -u server.py
