# chess-v2

### Quickstart
```bash
# Create and sync the virtual environment from pyproject
uv sync

uv pip install setuptools
# Activate the virtual environment (ensures `which python` points to the venv)
source .venv/bin/activate

**For training:**
uv pip install -e picotron --no-build-isolation
uv pip install vllm --torch-backend=auto

**For cpu-only inference on lichess:**
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

uv pip install -r requirements/cpu-build.txt --index-strategy unsafe-best-match --torch-backend cpu
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match --torch-backend cpu
VLLM_TARGET_DEVICE=cpu python setup.py install

cd src/lichess-bot
uv pip install -r requirements.txt
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

### Lichess Bot
Run the chess bot that uses VLLM for move prediction:

```bash
cd src/lichess-bot
python lichess-bot.py --config config.yml
```

### Token Statistics
- `data/tokens_v2` sequences of length 1025: train 54,368,123 · val 5,371

### License
MIT
