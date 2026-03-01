#!/bin/bash
# Fast derisk training script
# Usage: bash scripts/derisk.sh configs/derisk/<config>.json

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.json>}"
export HF_TOKEN=$(cat ~/secrets/hf-token 2>/dev/null || echo "dummy")
export CUDA_DEVICE_MAX_CONNECTIONS=1

RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG'))['logging']['run_name'])")

echo "=== Starting derisk run: $RUN_NAME ==="
echo "Config: $CONFIG"

torchrun --nproc_per_node 8 picotron/train.py --config "$CONFIG" 2>&1 | tee "log_${RUN_NAME}.out"
