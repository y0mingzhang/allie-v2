#!/bin/bash

source .venv/bin/activate

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=-1
export LOG_LEVEL=INFO
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_AVOID_RECORD_STREAMS=1
export WANDB_API_KEY=$(cat ~/secrets/wandb-api-key)

RUN_NAME="llama3_1b_fp8"
CHECKPOINT_PATH=checkpoints/${RUN_NAME}
mkdir -p "$(dirname "$CHECKPOINT_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
TRAIN_SAMPLES=5380602

# Dataset prefix (no extension). Override by exporting DATA_PREFIX.
DATA_PREFIX=${DATA_PREFIX:-"data/bin/0923_processed_data"}
# Sanity check: ensure .idx and .bin exist
if [ ! -f "${DATA_PREFIX}.idx" ] || [ ! -f "${DATA_PREFIX}.bin" ]; then
    echo "Error: Dataset files not found for prefix: ${DATA_PREFIX}" >&2
    echo "Expected: ${DATA_PREFIX}.idx and ${DATA_PREFIX}.bin" >&2
    exit 1
fi

# Path to the pretrain_gpt.py script
PRETRAIN_SCRIPT_PATH="Megatron-LM/pretrain_gpt.py"

# Fixed model and training parameters (tunable via env overrides)
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
DTYPE="fp8"
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024
NUM_WORKERS=4

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="/scratch/yimingz3/chess-v2/benchmark_cache_${RUN_NAME}"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend auto
    --apply-layernorm-1p
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples ${TRAIN_SAMPLES}
    --lr-decay-samples 5380602
    --lr-warmup-samples 53806
    --lr 0.0003
    --min-lr 0.00002
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments for pre-tokenized chess data
DATA_ARGS_LIST=(
    "--data-path ${DATA_PREFIX}"
    "--tokenizer-type NullTokenizer"
    "--vocab-size 2350"
    "--data-cache-path ${DATA_CACHE_PATH}"
    "--split '100,0,0'"
    "--no-create-attention-mask-in-dataloader"
    "--num-workers ${NUM_WORKERS}"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --log-throughput
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --wandb-project "chess-v2"
    --wandb-exp-name "${RUN_NAME}"
    --wandb-save-dir "${CHECKPOINT_PATH}/wandb"
    --tensorboard-log-interval 10
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 2>&1 | tee logs/${RUN_NAME}.log
set +x
