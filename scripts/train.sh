#!/bin/bash

export HF_TOKEN=$(cat ~/secrets/hf-token)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 8 picotron/train.py --config configs/llama-3-1B-57B.json
# torchrun --nproc_per_node 8 picotron/train.py --config configs/llama-3-3B-57B.json
torchrun --nproc_per_node 4 picotron/train.py --config configs/tiny_runs/llama-3-1b-muon.json

export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --master_port 29493 --nproc_per_node 4 picotron/train.py --config configs/tiny_runs/llama-3-1b-adamw.json