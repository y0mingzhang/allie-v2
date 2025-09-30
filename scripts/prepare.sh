#!/bin/bash

#SBATCH --job-name="prepare-npy"
#SBATCH --partition="array"
#SBATCH --qos="array_qos"
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --mail-user=yimingz3@cs.cmu.edu
#SBATCH --requeue
#SBATCH --output=logs/prepare/%A_%a.log
#SBATCH --array=0-99%100

source /home/yimingz3/src/chess-v2/.venv/bin/activate

export HF_HOME=/home/yimingz3/.cache/huggingface
export HF_HUB_CACHE=/scratch/yimingz3/hf_cache/hub
export HF_DATASETS_CACHE=/scratch/yimingz3/hf_cache/datasets
export HF_XET_CACHE=/scratch/yimingz3/hf_cache/xet

# Env vars with sensible defaults
REPO_ROOT=${REPO_ROOT:-/home/yimingz3/src/chess-v2}
LIST_FILE=${LIST_FILE:-$REPO_ROOT/data/filtered_parquets_shuffled.txt}
OUT_BASE=${OUT_BASE:-$REPO_ROOT/data/tokens_v2}

mkdir -p "$REPO_ROOT/logs/prepare" "$OUT_BASE"

echo "Preparing with 100-way sharding over: $LIST_FILE"

# Count non-empty, non-comment lines
TOTAL=$(grep -vE '^\s*(#|$)' "$LIST_FILE" | wc -l)

# Compute chunk size for 100 shards
TASKS=100
CHUNK=$(( (TOTAL + TASKS - 1) / TASKS ))

task_min=${SLURM_ARRAY_TASK_MIN:-0}
task_id=${SLURM_ARRAY_TASK_ID:-0}
zero_index=$(( task_id - task_min ))

start=$(( zero_index * CHUNK ))
if [ "$start" -ge "$TOTAL" ]; then
  echo "Task $SLURM_ARRAY_TASK_ID has no work (start=$start >= total=$TOTAL). Exiting."
  exit 0
fi
end=$(( start + CHUNK ))
if [ "$end" -gt "$TOTAL" ]; then end="$TOTAL"; fi

echo "Task $SLURM_ARRAY_TASK_ID: processing lines ${start}-${end} of $TOTAL (chunk=$CHUNK)"

# Unique output dir per slice to avoid collisions
OUT_DIR="$OUT_BASE/$(printf '%07d-%07d' "$start" "$end")"
mkdir -p "$OUT_DIR"

srun python -u "$REPO_ROOT/src/data/prepare_npy.py" \
  --list-file "$LIST_FILE" \
  --start "$start" \
  --end "$end" \
  --out-dir "$OUT_DIR" \
  --batch-size 1024 \
  --bos-token-id 2348 \
  --progress
