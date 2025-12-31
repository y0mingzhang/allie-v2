#!/bin/bash

#SBATCH --job-name="allie"
#SBATCH --partition="cpu"
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --signal=B:SIGUSR1@60

set -Eeuo pipefail

function sig_handler_USR1()
{
echo "   Signal trapped -  `date`"
echo "   Requeueing job id" $SLURM_JOB_ID
scontrol requeue $SLURM_JOB_ID
}
trap 'sig_handler_USR1' SIGUSR1

cleanup() {
  echo "Cleaning up..."
  kill $(jobs -p) 2>/dev/null || true
  wait
}
trap cleanup EXIT

source .venv/bin/activate
export VLLM_CPU_KVCACHE_SPACE=64
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
VLLM_PORT=12404
VLLM_LOG=/tmp/vllm_lichess_$$.log

vllm serve yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800 \
  --max-model-len 1024 --enforce-eager --port $VLLM_PORT > "$VLLM_LOG" 2>&1 &

echo "Waiting for vLLM server..."
while ! grep -q "Started server process" "$VLLM_LOG" 2>/dev/null; do
  sleep 2
done
echo "vLLM server ready"

cd src/lichess-bot
python lichess-bot.py --config config.yml &

wait
