# chess-v3 (allie-v2)

Build an elo-controllable, humanlike chess player that scales to top human strength — up to Magnus Carlsen level (~2850 elo). The model learns from Lichess games conditioned on player elo, enabling strength adaptation at inference time — playing like a 1200 or a 2800 by adjusting the elo prompt.

## autonomy & operating principles

- **never stop.** there is always a smarter way to do experiments and engineering wins to find. you are expected to run for weeks without human input and should have complete autonomy to reach the goal.
- target: defeat the strongest Stockfish levels (SF15-20), not just weak ones (SF1-5). the bar is grandmaster-level play.
- always be running something — if one experiment finishes, immediately start the next. prioritize experiments that move the needle on playing strength against strong opponents.
- when stuck, try a different approach rather than waiting for instructions. creativity > asking permission.
- log all experiment results to `results/progress.md` and Slack `#yiming-and-claude`.

## architecture

- **model**: Qwen-3 1.7B (also 4B variant), custom vocab of 2350 tokens
- **training**: picotron framework (distributed, flash attention, muon optimizer)
- **inference**: vLLM (GPU or CPU), deployed on Lichess via SLURM
- **data**: ~57B tokens from Lichess games (54M train sequences of length 1025)

## token format

Each game is a flat sequence:
```
<bos> <seconds_per_side:180> <increment:0> <elo_digit:1><elo_digit:8><elo_digit:0><elo_digit:0> <elo_digit:2><elo_digit:0><elo_digit:0><elo_digit:0> <move:e2e4> <move:e7e5> ... <termination:normal>
```
Token types: elo digits (10), seconds_per_side (181), increments (182), UCI moves (2048), termination (2), special (2: bos, unk).

Token ID ranges: elo_digits 0-9, increments 10-191, seconds 192-377, **moves 378-2345**, termination 2346-2347, special 2348-2349.

## key paths

| path | what |
|---|---|
| `src/data/tokenizer.py` | tokenizer: encode/decode, game parsing, prompt building |
| `src/data/tokens.py` | token vocabulary definitions |
| `src/data/process_hf.py` | lichess parquets → tokenized games |
| `src/data/prepare_npy.py` | tokenized games → packed npy shards |
| `picotron/train.py` | training entrypoint |
| `picotron/picotron/model.py` | Qwen3 and LLaMA model implementations |
| `picotron/picotron/optim/muon.py` | muon optimizer |
| `configs/main_runs_v2/` | current training configs |
| `src/tools/hf/export_checkpoint.py` | picotron checkpoint → HF safetensors |
| `src/tools/eval/` | vLLM-based eval and batch inference |
| `scripts/train.sh` | training launch script (8 GPU torchrun) |
| `scripts/lichess.sh` | SLURM job for lichess bot (vLLM server + bot) |
| `scripts/run_scaling_laws.py` | IsoFLOP scaling law experiments |
| `scripts/run_arch_ablations.py` | architecture ablation experiments |
| `scripts/run_lr_ablations.py` | LR schedule ablation experiments |
| `scripts/fast_pack_npy.py` | fast npy packing (pyarrow, 80x faster than HF datasets) |
| `results/progress.md` | **main progress log with all experiment results** |
| `results/*.json` | machine-readable experiment results |
| `nanogpt/` | nanogpt-speedrun clone (for arch comparison) |

## training

```bash
.venv/bin/torchrun --nproc_per_node 8 picotron/train.py --config configs/main_runs_v2/<config>.json
```
- 8 GPUs, muon optimizer, bfloat16, torch.compile
- FLASH_ATTEN=0 (external flash-attn has ABI mismatch, SDPA fallback works fine)
- `random_init: true` in config to train from scratch (skips HF download)
- lr schedule: warmup → stable → cosine decay
- best model: `yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800` (~62% move-match accuracy)
- checkpoints saved to `models/` dir, exportable to HF format
- wandb project: `chess-v2` (old), `chess-v3` (new)
- validation reports: val_loss, move_acc, bits_per_move

## scaling laws (CRITICAL — read before training)

Optimal tokens-per-parameter ratio for chess: **T/N ≈ 30-70** (NOT 20 like Chinchilla).

| Data budget | Optimal model size | Expected acc |
|---|---|---|
| 2.13B tokens | 30-70M params | ~51% |
| 14.6B tokens | 200-500M params | TBD |
| 57B tokens | 1.1-1.9B params | ~62% (existing) |

Architecture: Qwen3 baseline is already optimal. nanogpt-speedrun tricks (ReLU², embed shortcut, zero-init proj) don't help for chess. Higher LR (1.5-2x default) consistently wins.

Full results: `results/scaling_law_results.json`, `results/progress.md`

## data pipeline

1. `src/data/process_hf.py`: parse lichess parquets from HF, filter, tokenize → processed parquets
2. `scripts/fast_pack_npy.py`: processed parquets → packed npy (preferred, 80x faster)
3. `src/data/prepare_npy.py`: old packer using HF datasets streaming (slow, avoid)
4. data lives in `data/tokens_v2/**/train.npy` and `val.npy`

Datasets:
- `full_v1`: 2.13B tokens from 1028 parquets
- `full_v2`: 14.6B tokens from 6674 parquets
- 26K total parquets available, ~7600 processed so far

## inference / lichess bot

- vLLM serves the model, lichess-bot (submodule at `src/lichess-bot`) calls it via HTTP
- legal move enforcement via logit bias at inference time
- `scripts/lichess.sh`: SLURM job with watchdog, retry, connectivity monitoring
- CPU inference uses pinned vLLM submodule at `vllm_source/`

## submodules

- `src/lichess-bot` → `git@github.com:y0mingzhang/lichess-bot.git` (branch: allie)
- `vllm_source` → vLLM (pinned for CPU compat)

## environment

```bash
uv venv --python 3.12  # NOT 3.13 (vLLM compat)
uv pip install -e .
uv pip install -e picotron --no-build-isolation  # for training
```

## storage

- `~/user_data` → NFS mount (2TB quota, ~250GB free) — personal storage
- `~/group_data` → NFS mount (10TB quota, ~1TB free) — shared group storage
- large files (datasets, checkpoints, model weights) go here, not in the repo
- use symlinks from local paths (e.g., `data/`, `models/`) to storage buckets as needed
- `/tmp` is shared 1.3TB, fills up from other users — write logs to `~/`, clean `/tmp/torchinductor_yimingz3` periodically
- no HF token on this machine — use `random_init: true`

## conventions

- ruff for formatting/linting (line-length 100)
- `uv` for package management, not pip
- send periodic progress updates to Slack `#yiming-and-claude` (channel ID: `C0AGJCT1CH2`)
- use `results/progress.md` as the canonical experiment log
