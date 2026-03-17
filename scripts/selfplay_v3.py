#!/usr/bin/env python
"""Fast MCTS self-play with vLLM inference + HF training.

Architecture:
  Phase 1 (GENERATE): vLLM serves model, MCTS plays games
  Phase 2 (TRAIN): HF loads model, trains on MCTS data + pretraining mix
  Phase 3 (EVAL): vLLM evaluates vs Stockfish
  Repeat — vLLM reinit (~10s) between phases

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/selfplay_v3.py \
        --hf-model ~/user_data/chess-v3/hf_export/200M_v3_selfplay \
        --iterations 10 --games-per-iter 20 --mcts-sims 50 \
        --mix-ratio 3 --lr 5e-6
"""

import argparse
import json
import math
import os
import random
import sys
import time

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.data.tokens import CHESS_MOVES, SECONDS_PER_SIDE, INCREMENTS

MOVE_OFFSET = 378
tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}
STOCKFISH = "/home/yimingz3/bin/stockfish"
OLD_BOS = 2348


def make_prompt(elo_w, elo_b, tc=180, inc=0):
    p = [OLD_BOS]
    sec_str = str(tc)
    p.append(
        192 + SECONDS_PER_SIDE.index(sec_str)
        if sec_str in SECONDS_PER_SIDE
        else 192 + len(SECONDS_PER_SIDE) - 1
    )
    inc_str = str(inc)
    p.append(10 + INCREMENTS.index(inc_str) if inc_str in INCREMENTS else 10 + len(INCREMENTS) - 1)
    for d in f"{elo_w:04d}":
        p.append(int(d))
    for d in f"{elo_b:04d}":
        p.append(int(d))
    return p


def get_legal(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


# ====================== GENERATE PHASE (vLLM) ======================


def vllm_mcts_generate(model_path, n_games, mcts_sims, elo_high, elo_low):
    """Generate MCTS self-play games using vLLM policy + SF value oracle."""
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=True,
    )

    # SF engine as value oracle (shared across games)
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    sf_engine.configure({"Threads": 2})
    sf_eval_limit = chess.engine.Limit(depth=8)

    all_data = []
    for g in range(n_games):
        if random.random() < 0.5:
            elo_w = elo_b = random.choice([1800, 2000, 2200, 2400])
            train_both = True
        else:
            elo_w, elo_b = elo_high, elo_low
            if random.random() < 0.5:
                elo_w, elo_b = elo_b, elo_w
            train_both = False

        board = chess.Board()
        tokens = make_prompt(elo_w, elo_b)
        game_data = []

        for move_num in range(200):
            if board.is_game_over():
                break
            legal = get_legal(board)
            if not legal:
                break

            # Get policy prior from vLLM
            n_lp = min(len(legal), 20)
            params = SamplingParams(
                temperature=0, max_tokens=1, allowed_token_ids=legal, logprobs=n_lp
            )
            res = llm.generate(TokensPrompt(prompt_token_ids=tokens), params)

            logprobs = res[0].outputs[0].logprobs[0] if res[0].outputs[0].logprobs else {}
            policy = {}
            for tok in legal:
                if tok in logprobs:
                    policy[tok] = math.exp(logprobs[tok].logprob)
                else:
                    policy[tok] = math.exp(-10.0)
            total = sum(policy.values())
            policy = {k: v / total for k, v in policy.items()}

            # MCTS with SF value oracle: evaluate top-K candidate moves
            # Score each move = policy_prior * value_from_SF
            top_k = min(len(legal), mcts_sims)  # evaluate up to mcts_sims moves
            sorted_moves = sorted(policy.items(), key=lambda x: -x[1])[:top_k]

            visit_counts = {}
            for tok, prior in sorted_moves:
                uci = tok2uci.get(tok)
                if not uci:
                    continue
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    continue

                # SF evaluation after this move
                board.push(move)
                try:
                    info = sf_engine.analyse(board, sf_eval_limit)
                    cp = info["score"].white().score(mate_score=10000)
                    # Normalize to [0, 1] from white's perspective
                    value = 1.0 / (1.0 + math.exp(-cp / 200.0))  # sigmoid
                    if board.turn == chess.WHITE:
                        value = 1.0 - value  # flip: we just played as the side that moved
                except Exception:
                    value = 0.5
                board.pop()

                # Combined score: policy prior + SF value
                combined = prior * (0.5 + value)  # boost moves SF likes
                # Convert to visit counts proportional to combined score
                visit_counts[tok] = max(1, int(combined * mcts_sims))

            if not visit_counts:
                # Fallback to pure policy
                visit_counts = {tok: max(1, int(p * mcts_sims)) for tok, p in sorted_moves}

            total_visits = sum(visit_counts.values())
            mcts_policy = {tok: v / total_visits for tok, v in visit_counts.items() if v > 0}

            game_data.append(
                {
                    "prefix": list(tokens),
                    "mcts_policy": mcts_policy,
                    "turn_is_white": board.turn == chess.WHITE,
                }
            )

            # Sample move from visit distribution
            temp = 1.0 if move_num < 15 else 0.3
            toks = list(mcts_policy.keys())
            probs = [mcts_policy[t] ** (1.0 / temp) for t in toks]
            total_p = sum(probs)
            probs = [p / total_p for p in probs]
            chosen_tok = random.choices(toks, probs)[0]

            uci = tok2uci.get(chosen_tok)
            if not uci or chess.Move.from_uci(uci) not in board.legal_moves:
                break
            board.push(chess.Move.from_uci(uci))
            tokens.append(chosen_tok)

        # Game outcome
        result = board.result()
        value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
        for td in game_data:
            td["value"] = value if td["turn_is_white"] else -value

        # Filter for different-elo games
        if not train_both:
            high_is_white = elo_w == elo_high
            game_data = [d for d in game_data if d["turn_is_white"] == high_is_white]

        all_data.extend(game_data)

        if (g + 1) % 5 == 0:
            print(
                f"  Game {g + 1}/{n_games}: {result} ({board.fullmove_number}mv), "
                f"{len(all_data)} positions"
            )

    sf_engine.quit()
    del llm
    torch.cuda.empty_cache()
    return all_data


# ====================== TRAIN PHASE (HF) ======================


def train_mixed(model_path, mcts_data, lr, mix_ratio, pretrain_path, device="cuda:0"):
    """Train policy on MCTS data + pretraining mix. No value head (SF is oracle)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pretrain = np.memmap(pretrain_path, dtype=np.uint16, mode="r")
    pretrain_len = len(pretrain) // 1024

    model.train()
    total_mcts = total_pt = n_mcts = n_pt = 0

    random.shuffle(mcts_data)
    for td in mcts_data:
        # Policy distillation: match MCTS visit distribution
        input_ids = torch.tensor(td["prefix"], dtype=torch.long, device=device).unsqueeze(0)
        input_ids = input_ids.clamp(max=model.config.vocab_size - 1)
        logits = model(input_ids=input_ids).logits[0, -1]

        legal_toks = list(td["mcts_policy"].keys())
        target_probs = torch.tensor([td["mcts_policy"][t] for t in legal_toks], device=device)
        pred_logits = torch.stack([logits[t] for t in legal_toks])
        loss = F.kl_div(F.log_softmax(pred_logits, dim=-1), target_probs, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_mcts += loss.item()
        n_mcts += 1

        # Pretraining steps
        for _ in range(mix_ratio):
            idx = random.randint(1, pretrain_len - 2)
            seq = pretrain[idx * 1024 : (idx + 1) * 1024].astype(np.int64)
            seq = (
                torch.tensor(seq, device=device).unsqueeze(0).clamp(max=model.config.vocab_size - 1)
            )
            pt_out = model(input_ids=seq[:, :-1])
            pt_targets = seq[:, 1:]
            move_mask = (pt_targets >= 378) & (pt_targets <= 2345)
            if move_mask.any():
                pt_loss = F.cross_entropy(pt_out.logits[move_mask], pt_targets[move_mask])
                optimizer.zero_grad()
                pt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_pt += pt_loss.item()
                n_pt += 1

    # Save
    model.eval()
    model.save_pretrained(model_path)
    del model, optimizer
    torch.cuda.empty_cache()

    return total_mcts / max(n_mcts, 1), total_pt / max(n_pt, 1)


# ====================== EVAL PHASE (vLLM) ======================


def vllm_eval(model_path, n_games=10):
    """Evaluate vs Stockfish using vLLM."""
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=True,
    )

    results = {}
    for sf_level in [1, 3, 5]:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        engine.configure({"Skill Level": sf_level, "Threads": 1})
        w = d = l = 0

        for i in range(n_games):
            board = chess.Board()
            tokens = make_prompt(2000, 2000)
            mw = i % 2 == 0
            for _ in range(400):
                if board.is_game_over():
                    break
                if (board.turn == chess.WHITE) == mw:
                    legal = get_legal(board)
                    if not legal:
                        break
                    params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=legal)
                    res = llm.generate(TokensPrompt(prompt_token_ids=tokens), params)
                    tok = res[0].outputs[0].token_ids[0]
                    uci = tok2uci.get(tok)
                    if not uci or chess.Move.from_uci(uci) not in board.legal_moves:
                        break
                    board.push(chess.Move.from_uci(uci))
                    tokens.append(tok)
                else:
                    sf = engine.play(board, chess.engine.Limit(depth=sf_level))
                    board.push(sf.move)
                    uci = sf.move.uci()
                    if uci in uci2tok:
                        tokens.append(uci2tok[uci])
            r = board.result()
            if (r == "1-0" and mw) or (r == "0-1" and not mw):
                w += 1
            elif r in ("1/2-1/2", "*"):
                d += 1
            else:
                l += 1

        engine.quit()
        results[sf_level] = (w, d, l)
        print(f"  SF{sf_level}: {w}W {d}D {l}L")

    del llm
    torch.cuda.empty_cache()
    return results


# ====================== MAIN LOOP ======================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--elo-high", type=int, default=2500)
    parser.add_argument("--elo-low", type=int, default=1800)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--mix-ratio", type=int, default=3)
    parser.add_argument("--pretrain-data", default="data/tokens_v2/full_v3/train.npy")
    parser.add_argument("--output-dir", default="results/selfplay_v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log = []

    print(
        f"Fast MCTS self-play: {args.iterations} iters, {args.games_per_iter} games, "
        f"{args.mcts_sims} sims, mix={args.mix_ratio}x, lr={args.lr}"
    )

    # Baseline eval
    print("\n=== Baseline ===")
    eval0 = vllm_eval(args.hf_model, args.eval_games)
    log.append({"iter": 0, "eval": {k: list(v) for k, v in eval0.items()}})

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 50}")
        print(f"Iteration {it}/{args.iterations}")

        # Phase 1: Generate MCTS games with vLLM
        print(f"\n[GEN] {args.games_per_iter} MCTS games ({args.mcts_sims} sims)...")
        t0 = time.time()
        mcts_data = vllm_mcts_generate(
            args.hf_model, args.games_per_iter, args.mcts_sims, args.elo_high, args.elo_low
        )
        gen_time = time.time() - t0
        print(f"  Generated {len(mcts_data)} positions in {gen_time:.0f}s")

        # Phase 2: Train with HF + pretraining mix
        print(f"\n[TRAIN] MCTS + {args.mix_ratio}x pretraining mix...")
        t0 = time.time()
        mcts_loss, pt_loss = train_mixed(
            args.hf_model, mcts_data, args.lr, args.mix_ratio, args.pretrain_data
        )
        train_time = time.time() - t0
        print(f"  MCTS loss: {mcts_loss:.4f}, PT loss: {pt_loss:.4f} ({train_time:.0f}s)")

        # Phase 3: Eval with vLLM
        print(f"\n[EVAL]")
        eval_r = vllm_eval(args.hf_model, args.eval_games)

        log.append(
            {
                "iter": it,
                "eval": {k: list(v) for k, v in eval_r.items()},
                "n_positions": len(mcts_data),
                "mcts_loss": mcts_loss,
                "pt_loss": pt_loss,
                "gen_time": gen_time,
                "train_time": train_time,
            }
        )

        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
