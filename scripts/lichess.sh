#!/bin/bash

#SBATCH --job-name="allie"
#SBATCH --partition="cpu"
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --signal=B:SIGUSR1@60

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BOT_DIR="$ROOT_DIR/src/lichess-bot"
MODEL="models/yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800"

source "$ROOT_DIR/.venv/bin/activate"
export VLLM_CPU_KVCACHE_SPACE=64
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
export VLLM_PORT=12404

VLLM_LOG=/tmp/vllm_lichess_$$.log
VLLM_PID=0

# --- signal handling ---

sig_handler_USR1() {
    echo "SIGUSR1 trapped at $(date), requeueing job $SLURM_JOB_ID"
    kill -TERM $(jobs -p) 2>/dev/null || true
    wait
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap 'sig_handler_USR1' SIGUSR1

cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null || true
    wait
}
trap cleanup EXIT

# --- vLLM server management ---

start_vllm() {
    vllm serve "$ROOT_DIR/$MODEL" \
        --max-model-len 1024 --enforce-eager --port "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "Starting vLLM (pid $VLLM_PID)..."

    local elapsed=0
    while ! grep -q "Started server process" "$VLLM_LOG" 2>/dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "FATAL: vLLM process died during startup. Log tail:"
            tail -20 "$VLLM_LOG"
            return 1
        fi
        if (( elapsed >= 600 )); then
            echo "FATAL: vLLM startup timed out after ${elapsed}s"
            kill "$VLLM_PID" 2>/dev/null || true
            return 1
        fi
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done
    echo "vLLM server ready (took ${elapsed}s)"
}

ensure_vllm() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        return 0
    fi
    echo "vLLM is down, restarting..."
    VLLM_LOG=/tmp/vllm_lichess_$$.log
    start_vllm
}

# --- lichess connectivity ---

LICHESS_TOKEN=$(grep '^token:' "$BOT_DIR/config.yml" | awk '{print $2}' | tr -d '"')

wait_for_lichess() {
    local delay=10
    while ! curl -sf -o /dev/null --max-time 5 \
        -H "Authorization: Bearer $LICHESS_TOKEN" \
        https://lichess.org/api/account 2>/dev/null; do
        echo "lichess.org unreachable, retrying in ${delay}s..."
        sleep "$delay"
        delay=$(( delay < 600 ? delay * 2 : 600 ))
    done
}

# --- main ---

start_vllm || exit 1
wait_for_lichess

backoff=60
rapid_failures=0
while true; do
    ensure_vllm || { sleep 60; continue; }

    start_time=$SECONDS
    (cd "$BOT_DIR" && python lichess-bot.py --config config.yml)
    code=$?
    runtime=$(( SECONDS - start_time ))
    echo "Bot exited with code $code after ${runtime}s"

    if (( runtime < 30 )); then
        rapid_failures=$(( rapid_failures + 1 ))
        if (( rapid_failures >= 5 )); then
            echo "Too many rapid failures ($rapid_failures), cooling down 10 minutes..."
            sleep 600
            rapid_failures=0
            backoff=60
        fi
    else
        rapid_failures=0
        backoff=60
    fi

    sleep "$backoff"

    if ! curl -sf -o /dev/null --max-time 5 \
        -H "Authorization: Bearer $LICHESS_TOKEN" \
        https://lichess.org/api/account 2>/dev/null; then
        wait_for_lichess
        backoff=60
    else
        backoff=$(( backoff < 600 ? backoff * 2 : 600 ))
    fi
done &

wait
