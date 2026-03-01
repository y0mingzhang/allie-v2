#!/bin/bash
# Eval pipeline for 400M v3 model
# Run after checkpoint is saved at step 15000
# Usage: bash scripts/eval_400m.sh [step]

set -e
STEP=${1:-15000}
CONFIG="configs/main_runs_v2/qwen-3-400M-v3.json"
CKPT="models/main_runs_v2/qwen-3-400M-v3/${STEP}/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
EXPORT_DIR="/tmp/export_400M_v3_step${STEP}"

echo "=== Eval pipeline for 400M v3 step ${STEP} ==="

# 1. Export to HF format
echo "[1/3] Exporting checkpoint..."
CUDA_VISIBLE_DEVICES=4 MASTER_PORT=29503 .venv/bin/python src/tools/hf/export_checkpoint.py \
  --config "$CONFIG" --checkpoint "$CKPT" --output-dir "$EXPORT_DIR" --dtype bfloat16

# 2. Elo-stratified move accuracy
echo "[2/3] Running elo-stratified eval..."
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python src/tools/eval/elo_stratified_eval.py \
  --model-path "$EXPORT_DIR" --model-type hf \
  --output "results/elo_400M_v3_step${STEP}.json"

# 3. Play vs Stockfish at multiple levels
echo "[3/3] Playing vs Stockfish..."
for sf in 1 3 5; do
  gpu=$((4 + sf % 4))
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. .venv/bin/python src/tools/eval/vllm_play_stockfish.py \
    --model "$EXPORT_DIR" --sf-level $sf --sf-depth $sf --n-games 20 --elo 2000 \
    --output "results/vllm_400M_v3_step${STEP}_sf${sf}.json" &
done
wait

echo "=== Eval complete ==="
echo "Results in results/elo_400M_v3_step${STEP}.json and results/vllm_400M_v3_step${STEP}_sf*.json"
