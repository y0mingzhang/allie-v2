#!/bin/bash

source .venv/bin/activate
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

vllm serve yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800 --max-model-len 1024 --enforce-eager
