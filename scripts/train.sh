#!/bin/bash

export HF_TOKEN=$(cat ~/secrets/hf-token)
torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs_v2/qwen-3-1.7b-57b.json 2>&1 | tee log_qwen-3-1.7b-57b.out
# torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs_v2/qwen-3-4b-58b.json 2>&1 | tee log_qwen-3-4b-58b.out
