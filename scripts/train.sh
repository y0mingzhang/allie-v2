#!/bin/bash

export HF_TOKEN=$(cat ~/secrets/hf-token)
torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs/qwen-3-4b-58b.json 2>&1 | tee log_qwen-3-4b-58b.out