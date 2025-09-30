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
- Command: `torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs/qwen-3-4b-58b.json`
- Logs stream to `log_qwen-3-4b-58b.out`

### Model Exporting
- Script: `src/tools/hf/export_checkpoint.py`
- Example:
  ```bash
  uv run python src/tools/hf/export_checkpoint.py \
    --config configs/tiny_runs/qwen-3-1.7b-muon.json \
    --checkpoint models/tiny-qwen-3-1.7b-muon-1b/3814/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth \
    --output-dir exports/tiny-qwen-3-1.7b-muon-1b \
    --dtype bfloat16
  ```
- Outputs `model.safetensors` and `config.json`; pass `--tokenizer-dir` to copy tokenizer files

### Tools
- `uv run python src/tools/hf/download_tokens.py` — sync missing shards from `yimingzhang/lichess_tokens_v2`
- `uv run python src/tools/hf/upload_tokens_to_hf.py` — upload `data/tokens/` to `yimingzhang/lichess_tokens`
- `uv run python src/tools/hf/upload_tokenized_dataset_to_hf.py` — upload `data/dataset_v2/` to `yimingzhang/lichess_tokenized`
- `uv run python src/tools/hf/export_checkpoint.py --config ... --checkpoint ... --output-dir ...` — convert Picotron checkpoint to HF format
- `uv run python src/tools/hf/compare_checkpoints.py --picotron-ckpt ... --hf-dir ...` — verify HF export parity
- `uv run python src/tools/hf/sample_next_move.py --model-dir ...` — inspect next-move probabilities
- `uv run python src/tools/data/token_shard_stats.py [PATH]` — report `.npy` token counts and complete sequences
- `uv run python src/tools/data/compute_cell_sampling.py --sparsity ...` — compute per-format sampling caps / heatmap
- `uv run python src/tools/data/estimate_sampling_ratios.py` — print baseline format/ELO sampling ratios
- `uv run python src/tools/data/process_tiny_dataset.py` — downsample + tokenize a tiny HF slice and push to the hub
- `uv run python src/tools/data/explore_data_distribution.py` — generate dataset diagnostics under `analysis/`

### Token Statistics
- `data/tokens_v2` sequences of length 1025: train 54,368,123 · val 5,371

### License
MIT
