#!/bin/bash

#SBATCH --job-name="tokenize"
#SBATCH --partition="array"
#SBATCH --qos="array_qos"
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --mail-user=yimingz3@cs.cmu.edu
#SBATCH --requeue
#SBATCH --output=logs/process/%A_%a.log
#SBATCH --array=0-500%100

source /home/yimingz3/src/chess-v2/.venv/bin/activate

rm -rf /scratch/yimingz3/
export HF_HOME=/home/yimingz3/.cache/huggingface
export HF_HUB_CACHE=/scratch/yimingz3/hf_cache/hub
export HF_DATASETS_CACHE=/scratch/yimingz3/hf_cache/datasets
export HF_XET_CACHE=/scratch/yimingz3/hf_cache/xet

# Required env vars (provide defaults where sensible)
REPO_ROOT=${REPO_ROOT:-/home/yimingz3/src/chess-v2}
LIST_FILE=${LIST_FILE:-$REPO_ROOT/data/chess_parquets_shuffled.txt}
DATASET_ROOT_PATH=${DATASET_ROOT_PATH:-$REPO_ROOT/data/dataset_v2}

mkdir -p "$REPO_ROOT/logs" "$DATASET_ROOT_PATH"

# Count non-empty, non-comment lines
TOTAL=$(grep -vE '^\s*(#|$)' "$LIST_FILE" | wc -l)

# Determine array size and compute chunk size as ceil(TOTAL / ARRAY_SIZE)
TASKS="${SLURM_ARRAY_TASK_COUNT:-}"
if [ -z "$TASKS" ] || [ "$TASKS" -le 0 ] 2>/dev/null; then
  if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ -n "${SLURM_ARRAY_TASK_MIN:-}" ]; then
    TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
  else
    TASKS=1
  fi
fi

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

echo "Task $SLURM_ARRAY_TASK_ID: ${start}-${end} -> ${DATASET_ROOT_PATH} (total=$TOTAL, tasks=$TASKS, chunk=$CHUNK)"

srun python -u "$REPO_ROOT/src/data/process_hf.py" \
  --list-file "$LIST_FILE" \
  --start "$start" \
  --end "$end" \
  --dataset-root-path "$DATASET_ROOT_PATH"
