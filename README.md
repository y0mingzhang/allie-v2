# chess-v2

### Quickstart
```bash
# Create and sync the virtual environment from pyproject
uv sync

uv pip install -e picotron --no-build-isolation

# Activate the virtual environment (ensures `which python` points to the venv)
source .venv/bin/activate
```

### Training
- Script: `scripts/train.sh`
- Requires 8 GPUs and an HF token stored at `~/secrets/hf-token`
- Command: `torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs/qwen-3-1.7b-57b.json`
- Logs stream to `log_qwen-3-1.7b-57b.out`

### Model Exporting
- Script: `picotron/picotron/export_to_hf.py`
- Example:
  ```bash
  uv run python picotron/picotron/export_to_hf.py \
    --config configs/tiny_runs/qwen-3-1.7b-muon.json \
    --checkpoint models/tiny-qwen-3-1.7b-muon-1b/3814/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth \
    --output-dir exports/tiny-qwen-3-1.7b-muon-1b \
    --dtype bfloat16
  ```
- Outputs `model.safetensors` and `config.json`; pass `--tokenizer-dir` to copy tokenizer files

### Tools
- `python src/tools/hf/download_tokens.py` — sync missing shards from `yimingzhang/lichess_tokens`
- `python src/tools/data/token_shard_stats.py` — report `.npy` token counts and 1024-length sequences
- `python src/tools/data/count_indexed_tokens.py PREFIX...` — sum tokens in Megatron indexed datasets
- `python src/tools/data/compute_train_samples.py --data-prefix ...` — compute `--train-samples`
- `python src/tools/hf/compare_checkpoints.py --picotron-ckpt ... --hf-dir ...` — verify HF export parity
- `python src/tools/hf/export_checkpoint.py --config ... --checkpoint ...` — convert Picotron checkpoint to HF
- `python src/tools/hf/sample_next_move.py --model-dir ...` — inspect next-move probabilities

### Token Statistics
- Latest `python src/tools/data/token_shard_stats.py`
- Current snapshot (`data/tokens`): 200 files, 58,205,514,736 tokens
- Complete sequences of length 1024: 56,841,217

### License
MIT
