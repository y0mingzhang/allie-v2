# Training Progress Log

## Goal
Elo-controllable chess model that scales to top human strength. Target: beat existing best checkpoint (~62% move-match accuracy on 57B tokens with 1.7B params).

## Key Insights

### Scaling Laws for Chess Data
Ran 12 IsoFLOP experiments across 3 compute budgets (1e17, 3e17, 1e18 FLOPs) and 6 model sizes (5M–600M).

**Optimal T/N ratio for chess data ≈ 30–70** (vs Chinchilla's 20 for language). Chess is more "learnable" per token — structured domain with fixed rules.

IsoFLOP winners:
| Compute | Optimal Size | Val Loss | Move Acc | T/N |
|---------|-------------|----------|----------|-----|
| 1e17 | 15M | 1.734 | 45.9% | 73 |
| 3e17 | 40M | 1.625 | 48.9% | 33 |
| 1e18 | 40M | 1.570 | 50.4% | 53 |

Implications:
- For 2.13B tokens: optimal model = 30–70M params
- For 57B tokens (full dataset): optimal model = 1.1–1.9B params
- The 153M medium model was 2–3x too large for 2B tokens
- The 1.41B model was 20–40x too large for 2B tokens
- **40M model beats 153M model (50.4% vs 50.1%) using 5x less compute**

### Architecture: Qwen3 Baseline is Optimal
Tested 6 ablations at matched compute (40M params, 2.13B tokens). No nanogpt-speedrun trick helped:
- ReLU²: -2.2% (worse)
- Embed shortcut, zero-init proj, depth/width tradeoffs: all slightly worse or neutral
- **SiLU gated MLP with standard Qwen3 arch is the right choice for chess**

### LR: Higher is Better (up to a point)
Consistent across all scales. LR=0.035 slightly beats 0.025 for 57M model.

### Full IsoFLOP Table

| Budget | Model | Params | Tokens | T/N | Val Loss | Move Acc | Time |
|--------|-------|--------|--------|-----|----------|----------|------|
| 1e17 | 5M | 5.32M | 2.13B | 426 | 1.844 | 42.9% | 5.8m |
| 1e17 | 15M | 15.07M | 1.10B | 73 | **1.734** | **45.9%** | 5.8m |
| 1e17 | 40M | 38.97M | 0.42B | 10 | 1.859 | 43.2% | 4.8m |
| 1e17 | 100M | 86.76M | 0.17B | 2 | 2.969 | 27.4% | 4.2m |
| 3e17 | 15M | 15.07M | 2.13B | 142 | 1.656 | 48.1% | 10.8m |
| 3e17 | 40M | 38.97M | 1.30B | 33 | **1.625** | **48.9%** | 13.5m |
| 3e17 | 100M | 86.76M | 0.53B | 5 | 1.844 | 43.3% | 10.6m |
| 3e17 | 250M | 203.77M | 0.20B | 1 | 2.219 | 36.7% | 10.2m |
| 1e18 | 40M | 38.97M | 2.13B | 53 | **1.570** | **50.4%** | 22.2m |
| 1e18 | 100M | 86.76M | 1.67B | 17 | 1.578 | 50.1% | 32.4m |
| 1e18 | 250M | 203.77M | 0.67B | 3 | 1.734 | 46.0% | 31.1m |
| 1e18 | 600M | 444.07M | 0.28B | 0 | 2.375 | 34.6% | 28.2m |

---

## Experiments (chronological)

### Derisk Round 1 — 1000 steps, 118M tokens, 20M param model
Key finding: **LR=0.02 (2x) wins** (+4% acc), depth helps.

| Run | Move Acc | Notes |
|-----|----------|-------|
| higher_lr_2x (LR=0.02) | **40.1%** | best |
| deeper_8L | 37.6% | depth helps |
| base_tiny (LR=0.01) | 36.5% | baseline |

### Derisk Round 2 — 1000 steps, combining R1 findings
Key finding: **depth + high LR stack**.

| Run | Move Acc |
|-----|----------|
| r2_kitchen_sink (8L+LR=0.02+mw3x) | **41.4%** |
| r2_hi_lr_deep (8L+LR=0.02) | 41.0% |

### Derisk Round 3 — 3000 steps, 708M tokens
Key finding: **12 layers > 8 layers**.

| Run | Move Acc | Val Loss |
|-----|----------|----------|
| r3_12L (12L, LR=0.02) | **49.4%** | 1.609 |
| r3_best (8L, LR=0.02) | 48.2% | 1.655 |

### Medium model — 153M params, 5000 steps, 708M tokens
**Final: val_loss=1.578, move_acc=50.1%** — overtrained for data budget (T/N=4.6)

### 1.7B run (killed at step 2100)
Killed after scaling law analysis showed it was 20-40x too large for 2.13B tokens.
At step 2000: move_acc=38.1%, lagging behind the 153M medium model at matched tokens.
FLOPs comparison: at ~1E18, medium gets 45.4% while 1.7B gets ~19%.

### Scaling law sweep (12 experiments)
See "Key Insights" above.

### Architecture ablations — 40M params, 2.13B tokens
| Ablation | Params | Val Loss | Move Acc |
|---|---|---|---|
| **baseline** (12L, h=512, SiLU) | 38.97M | **1.570** | **50.5%** |
| zero_init_proj | 38.97M | 1.586 | 50.1% |
| wider_shallow (8L, h=640) | 40.84M | 1.578 | 50.1% |
| deeper_narrow (16L, h=416) | 34.22M | 1.578 | 49.9% |
| embed_shortcut (x0 residual) | 38.97M | 1.586 | 49.9% |
| relu2 (ReLU²) | 38.97M | 1.648 | 48.3% |

### Compute-optimal 57M run — `optimal-50M-v1`
Config: 57M params, 14L, hidden=576, head_dim=72, LR=0.025, T/N=37
**Final: val_loss=1.547, move_acc=50.8%** — best result on 2.13B tokens, trained in 30 min.

### LR ablations (3/7 completed, rest killed by /tmp full)
| LR | Val Loss | Move Acc |
|----|----------|----------|
| 0.035 | 1.547 | **51.0%** |
| 0.025 (baseline) | 1.547 | 50.9% |
| 0.015 | 1.578 | 50.2% |

Remaining (long_decay, half_decay, short_warm, high_min_lr) need rerunning.

### Derisk R4 — data quality experiments, 40M params
| Experiment | Val Loss | Move Acc | Bits/Move |
|---|---|---|---|
| baseline (2.13B) | 1.570 | 50.5% | 2.27 |
| move_weight_3x | 1.578 | 50.5% | 2.27 |
| move_only_loss | 4.312 | 50.4% | 2.25 |
| more_data (14.6B) | 1.578 | 50.4% | 2.25 |

Findings: metadata tokens are free (don't affect move prediction). 40M saturated on 2.13B — more data needs bigger model.

### Elo-stratified eval — 57M model
| Elo | Move Acc | Bits/Move |
|-----|----------|-----------|
| 800 | 45.8% | 2.54 |
| 1200 | 49.7% | 2.31 |
| 1600 | 50.8% | 2.23 |
| 2000 | 51.1% | 2.20 |
| 2400 | 51.7% | 2.13 |
| 2600 | 53.9% | 2.08 |

Higher elo = lower bits/move. Move accuracy relatively flat across elo.

### Stockfish playing eval — 57M model (score = win%)
| Prompt Elo | SF 1 | SF 5 | SF 10 | SF 15 | SF 20 |
|---|---|---|---|---|---|
| 1200 | 40% | 0% | 0% | 0% | 0% |
| 1600 | 45% | 0% | 0% | 0% | 0% |
| 2000 | 80% | 10% | 0% | 10% | 0% |
| 2400 | 80% | 10% | 0% | 0% | 0% |

Conditioning works (2400 > 1200 vs SF1) but 57M model is very weak (~800 elo playing strength). Can barely beat Stockfish skill 1.

### Derisk R4-R5 — Engine data mixing
R4 (2% TCEC mix): move_acc identical to baseline (50.5%), BUT SF playing strength improved (57% vs 40% at SF1 with elo=3000 conditioning).

R5 holistic eval with CPL + illegal tracking:
| Model | Elo Prompt | vs SF1 | vs SF3 | vs SF5 | Avg CPL | Illegal% |
|---|---|---|---|---|---|---|
| baseline | 1200 | 5% | 2% | 5% | 200 | 0.7% |
| baseline | 3000 | 15% | 0% | 2% | 175-259 | 1.6-3.0% |
| mixed 2% | 1200 | 10% | 0% | 0% | 184-264 | 0.8-1.2% |
| mixed 2% | 3000 | 25% | 0% | 8% | 139-220 | 0.3-1.4% |

Findings:
- Elo conditioning works (3000 > 1200)
- Engine data lowers CPL at high elo conditioning (139 vs 175 at SF5)
- Illegal rate very low (~1%)
- 40M model fundamentally too weak (~200 CPL) for strong play

### Engine data sources
- TCEC: 30K tournament games, 4.5M tokens
- Boltzmann SF: 10K diverse self-play games via softmax sampling over centipawn evals, 1.9M tokens
- Mixed_v1: 2.13B human + 47M engine (2% mix)
- Mixed_v2: 2.13B human + 212M engine (10% mix — HURT performance, too much engine data)
- SF Boltzmann: 10K diverse games via softmax sampling over centipawn evals

Engine mix summary: 2% is sweet spot. 10% hurts. Engine data helps *playing strength* but not *move prediction accuracy*.

### 200M LR sweep
| LR | Move Acc (2K steps) | Bits/Move |
|----|---------------------|-----------|
| 0.012 | 44.6% | 2.64 |
| 0.020 | 47.3% | 2.45 |
| **0.025** | **49.0%** | **2.36** |

LR=0.012 was severely undertrained. LR=0.025 gives +4.4% at matched tokens.

### 200M model v2b (in progress) — `qwen-3-200M-v2`
Config: 178.6M params, 14L, hidden=1024, **LR=0.025**, micro_batch=64, T/N=42
28,500 steps, 7.5B tokens, 515K tok/s, ~4 hrs

Val curve:
| Step | Tokens | Move Acc | Bits/Move |
|------|--------|----------|-----------|
| 1K | 262M | 39.2% | 3.02 |
| 5K | 1.3B | 49.8% | 2.28 |
| 9K | 2.4B | 51.1% | 2.20 |
| 11K | 2.9B | 51.5% | 2.17 |
| 12K | 3.2B | 51.7% | 2.17 |

### Chess-v2 1.7B/57B baseline (model to beat)
| Elo | Move Acc | Bits/Move |
|-----|----------|-----------|
| 800 | 50.9% | 2.26 |
| 1200 | 54.4% | 2.03 |
| 1600 | 56.6% | 1.90 |
| 2000 | 58.7% | 1.77 |
| 2400 | 60.5% | 1.67 |

Stockfish playing:
| Prompt | SF1 | SF3 | SF5 | SF10 |
|--------|-----|-----|-----|------|
| 2400 | 85% | 70% | 18% | 8% |
| 3000 | 92% | 85% | 28% | 12% |

### MFU note
Picotron reports MFU relative to H100 (989 TFLOPS). On A100 (312 TFLOPS), actual MFU = reported × 3.17. Reported 16% = actual ~52%.

### MCTS playing eval — 200M model (bug-fixed minimax)
| Method | Sims | vs SF1 | vs SF3 |
|--------|------|--------|--------|
| Greedy | 0 | 10% | ~0% |
| MCTS + linear probe | 50 | **70-85%** | **60%** |
| MCTS + linear probe | 200 | **85%** | — |
| MCTS + SF oracle | 50 | **100%** | **100%** |
| chess-v2 1.7B greedy | 0 | 92% | 85% |

MCTS bug fixed: selection used max(UCB) for both players → fixed with sign flip for black.

Value function comparison (all MCTS 50 sims vs SF1):
| Value Function | Policy | vs SF1 |
|----------------|--------|--------|
| Linear probe (game outcome) | 53.5% | **70-85%** |
| Fine-tuned value head | 52.8% | 67.5% |
| Joint PT v1 (coeff=0.5) | 52.3% | 65% |
| Joint PT v2 (coeff=0.1) | 52.4% | 62.5% |
| CP probe (centipawn) | 53.5% | 55% |
| SF oracle | 53.5% | 100% |

Key insight: policy quality dominates MCTS at low search. But search scales well with oracle value:
- SF5: 65% (50 sims) → 95% (200 sims) with oracle
- SF10: 30% (50 sims) → TBD (200 sims)
Linear probe on frozen policy is best value approach. Every 1% policy acc ≈ 5% MCTS win rate at fixed search budget.

### Mate puzzle eval
| Method | MateIn1 | MateIn2 |
|--------|---------|---------|
| Greedy | 1% | 0% |
| MCTS-50 + linear probe | 3% | 0% |
| MCTS-50 + CP probe | 3% | 0% |

Model lacks tactical representations — trained on human games (resignations, not forced mates).

### Engine-mix 200M (2% TCEC at elo=3000)
Final: 53.3% move_acc (same as pure-human 53.5%), BUT playing strength much better:
| Model | Greedy vs SF1 | MCTS-50 vs SF1 | MCTS-50 vs SF3 |
|-------|-------------|----------------|----------------|
| Pure-human | 10% | 70-85% | 60% |
| Engine-mix | **25%** | **90%** | 60% |
| chess-v2 1.7B | 92% | — | 85% |

Engine data doesn't help move_acc but significantly improves playing quality (especially at elo=3000 conditioning).

MCTS with engine-mix model + pure-human probe: **90% vs SF1** (best result).
Cross-model probe transfer works: pure-human model's probe (corr=0.25) on engine-mix model > engine-mix model's own probe (corr=0.24).

Search scaling with imperfect value:
- 50 sims: 90% (sweet spot)
- 200 sims: 80% (noise amplification — more search hurts with imperfect value)

### SF oracle search scaling (200M policy, perfect value)
| Opponent | 50 sims | 200 sims |
|----------|---------|----------|
| SF1 | 100% | 100% |
| SF3 | 100% | — |
| SF5 | 65% | **95%** |
| SF10 | 30% | — |

More search helps dramatically — SF5 goes from 65% to 95% with 4x more sims. The policy CAN generate winning moves at SF5+ level, it just needs more search to find them.

### HF export
200M model exported to HF safetensors format at `~/user_data/chess-v3/hf_export/qwen-3-200M-v2/`. Ready for vLLM inference.

---

## Data Pipeline Status
- Total available: 26,076 parquets (Lichess dataset)
- Processed: 14,922 parquets
- full_v1: 2.13B tokens (1028 parquets)
- full_v2: 14.6B tokens (6674 parquets)
- full_v3: **31.9B tokens** (14922 parquets, packed)
- Data processing: 14922/26076 parquets done
- **fast_pack_npy.py**: 80x faster than HF datasets streaming (pyarrow direct read)

## vLLM Compatibility — FIXED (2026-02-27)
Root cause: `tie_word_embeddings=false` in export (model uses tied embeddings). Also `layer_types` array had wrong size.

Verification:
- HF transformers: 30/30 tokens match picotron (perfect)
- vLLM: small numerical diffs from PagedAttention (diverges ~token 6 greedy), quality identical
- vLLM vs SF1: 90% win rate, 0 illegal moves

Also fixed: CLAUDE.md token ranges were backwards (increments=10-191, seconds=192-377, not the other way around). CPL analysis with `engine.analyse()` on same SF engine corrupts Skill Level randomization — use separate engine or skip CPL.

## vLLM Eval Results (2026-02-27, correct prompts, no CPL)
200M v3 (step 20K, 32B tokens, T/N=82):
| Opponent | Score | Est Elo |
|----------|-------|---------|
| SF1 d1 | 92.5% | ~1236 |
| SF3 d3 | 62.5% | ~1289 |
| SF5 d5 | 25.0% | ~1409 |
| **Weighted avg** | | **~1300** |

200M engine-mix (step 10K, 14.6B tokens + 2% TCEC):
| Opponent | Score |
|----------|-------|
| SF1 d1 | 92.5% |
| SF3 d3 | 47.0% |
| SF5 d5 | 30.0% |

Key finding: v3 generalizes better to stronger opponents (66% vs 42% at SF3 in 50-game test). Engine-mix dominates at SF1 but falls off at higher levels.

Elo conditioning effect (v3, 30 games):
- Elo 2800 vs SF3: 77% (up from 63% at elo 2000)
- Elo 2800 vs SF5: 13% (down from 25% — overambitious play)

## 400M Training (in progress, 2026-02-27)
Config: 360M params, 18L, h=1280, LR=0.025, T/N=21 (7.5B tokens), dp=4
| Step | Val loss | Acc | Bits/mv | Tokens |
|------|----------|-----|---------|--------|
| 3000 | 1.807 | 44.5% | 2.61 | 0.4B |
| 6000 | 1.688 | 47.5% | 2.42 | 0.8B |
| 9000 | 1.633 | 49.0% | 2.33 | 1.2B |

Learning ~2x faster per token than 200M. Projected final: 53-56% at step 57K.

## Infrastructure Notes
- Flash-attn broken (C++ ABI mismatch) — using SDPA fallback
- Current hardware: 8x A100 SXM4 40GB (training on 4, evals on 4)
- Actual MFU: ~49% (picotron reports ~15% relative to H100 peak)
- No HF token — use `random_init: true` for all training

## Code Changes Made
- `picotron/train.py`: random_init support, HF_TOKEN skip, config override expansion, loss weighting, z-loss, move accuracy tracking
- `picotron/picotron/model.py`: Qwen3Attention.reset_parameters() q_norm/k_norm fix, embed_shortcut, zero_init_proj, relu2 activation
- `src/data/process_hf.py`: tmpdir cache per shard, error handling, periodic hub cache cleanup
- `scripts/fast_pack_npy.py`: 80x faster npy packing via direct pyarrow
- `scripts/run_scaling_laws.py`, `run_arch_ablations.py`, `run_lr_ablations.py`: experiment runners

## Next Steps
1. Complete 400M v3 training (57K steps, ~6h remaining)
2. Eval 400M at checkpoints 15K/30K/45K/57K (pipeline ready: scripts/eval_400m.sh)
3. If 400M shows improvement, scale to 600M-1B on full_v3 data
4. vLLM MCTS with 400M model for stronger playing strength
5. Continue data processing (14.2K/26K parquets done) → 50B+ tokens for future runs

## AlphaZero MCTS Self-Play (2026-03-05)

Ran 4 versions of AlphaZero-style MCTS self-play on the 200M v3 model:

| Version | Mix Ratio | LR | Iters | SF1 avg | SF3 avg | SF5 avg | Stable? |
|---|---|---|---|---|---|---|---|
| v1 | 0 | 3e-5 | 7 | 83% | 55% | 30% | Partially |
| v2 | 0 | 5e-6 | 4 | 88% | 60% | 45% | NO (collapsed iter 5) |
| v3 | 3x PT | 5e-6 | 5 | 88% | 66% | 32% | YES |
| v4 | 1x PT | 5e-6 | 3 | 80% | 77% | 23% | NO (oscillating) |
| Baseline | - | - | - | 80% | 60% | 30% | - |

Key findings:
- MCTS self-play improves SF1 (80%→88%) and SF3 (60%→66%) consistently
- Pretraining data mixing (3x) prevents catastrophic forgetting (v3 = first stable version)
- SF5 improvement is inconsistent (peaks at 45-60% but doesn't sustain)
- Lower mixing (1x) allows too much forgetting; higher mixing (3x) is more stable
- Speed bottleneck: ~2h/iter with 200M HF model

Scripts: `scripts/alphazero_selfplay.py`, `scripts/az_v3_mixed.py`

## Fast vLLM MCTS Self-Play (2026-03-12)

Rewrote self-play to use vLLM for inference — **12x speedup** (10 min/iter vs 2h).

Architecture: vLLM generates MCTS games (40s) → HF trains with 3x pretraining mix (500s) → vLLM evals (60s) → repeat.

200M v3 model, 10 iterations, 20 games/iter, 50 MCTS sims:

| Iter | SF1 | SF3 | SF5 |
|---|---|---|---|
| 0 (baseline) | 65% | 55% | 15% |
| 4 (peak SF3) | 75% | 85% | 40% |
| 8 (peak SF1/SF5) | 95% | 65% | 45% |
| 10 (final) | 85% | 65% | 25% |
| **Average** | **78%** | **68%** | **28%** |

Improvement: SF1 +14%, SF3 +13%, SF5 +13%. No collapse through 10 iterations.

Gap to AlphaZero scale: we use 20 games/iter × 50 sims (AlphaZero: 25K games × 800 sims). Need 10-100x more games and sims to target SF10+.

Script: `scripts/selfplay_v3.py`

## Current Status (2026-03-13)

**Best models:**
- 1.7B (allie-v2): 62% move-match, deployed on Lichess
- 600M v3: 53.7% move-match, 100% SF1, 35% SF5
- 200M v3 + self-play: SF1 95%, SF3 85%, SF5 45% (peak)

**Target:** defeat SF15-20 (grandmaster level, ~2600-3000+ elo). Current models struggle above SF5 (~1600 elo). Need ~1000 elo improvement.

**Next steps:**
1. Scale self-play to 200+ games/iter, 100+ sims on 600M model
2. Eval all models against SF5/8/10/15/20 to establish baselines
3. Run self-play targeting SF10+ opponents
4. Consider larger base models (1.7B) for self-play

## SF Oracle Search at Inference (2026-03-19)

**Key breakthrough: search at inference >> RL at training.**

600M model + SF depth-8 oracle reranking top-K candidates:
| Config | vs SF5 | vs SF8 | vs SF10 | vs SF15 |
|---|---|---|---|---|
| Greedy | 32% | 2% | 2% | ~0% |
| + oracle top-5 | 50% | 48% | 40% | 8% |
| + oracle top-10 | 50% | 50% | 42% | - |

Learned value probe (corr=0.26) too weak for move selection (0% vs SF5 — worse than greedy). Need corr > 0.5 for useful MCTS.

All RL approaches on 200M model produced SF5 avg ~30% (no improvement from 30% baseline):
- Fake MCTS: 28%
- SF oracle MCTS: 31%
- Scaled MCTS (200g): 28%
- Expert iteration v2: 31%

**Conclusion: the policy is already good. The value function is the bottleneck for playing strength.** Search at inference with a strong value oracle (SF) dramatically improves play. Training a learned value function to replace SF is the key challenge.
