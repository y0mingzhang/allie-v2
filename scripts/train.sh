#!/bin/bash

torchrun --nproc_per_node 8 picotron/train.py --config configs/llama-3-1B-57B.json
# torchrun --nproc_per_node 8 picotron/train.py --config configs/llama-3-3B-57B.json
