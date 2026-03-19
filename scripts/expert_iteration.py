#!/usr/bin/env python
"""Expert iteration: play games, have SF find blunders, train to fix them.

For each game:
1. Model plays vs SF (with temperature for diverse games)
2. SF evaluates every position at depth 12
3. Find positions where model's move lost >100cp vs SF's best move
4. Train policy to prefer SF's move at those positions

This is like DAgger/DPO — learn from an expert's corrections on your mistakes.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/expert_iteration.py \
        --hf-model ~/user_data/chess-v3/hf_export/600M_v3_selfplay \
        --iterations 10 --games-per-iter 50
"""

import argparse
import json
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
from src.data.tokenizer import Tokenizer as T
from src.data.tokens import CHESS_MOVES

MOVE_OFFSET = 378
tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}
STOCKFISH = "/home/yimingz3/bin/stockfish"


def make_prompt(elo=2000):
    p = [T.token_to_idx["<bos>"]]
    p.append(T.token_to_idx["<seconds_per_side:180>"])
    p.append(T.token_to_idx["<increment:0>"])
    for d in f"{elo:04d}":
        p.append(T.token_to_idx[f"<elo_digit:{d}>"])
    for d in f"{elo:04d}":
        p.append(T.token_to_idx[f"<elo_digit:{d}>"])
    return p


def get_legal(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


def generate_and_annotate(
    hf_model_path, n_games, sf_level=5, sf_depth=5, analysis_depth=12, blunder_threshold=100
):
    """Play games and find blunder positions with SF corrections."""
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=hf_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        max_model_len=1024,
        enforce_eager=True,
    )

    # Separate engine for playing and analysis
    play_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    play_engine.configure({"Skill Level": sf_level, "Threads": 1})
    sf_limit = chess.engine.Limit(depth=sf_depth)

    analysis_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    analysis_engine.configure({"Threads": 1})

    corrections = []  # (token_prefix, model_move_tok, sf_best_tok, cp_loss)

    for game_idx in range(n_games):
        board = chess.Board()
        model_white = game_idx % 2 == 0
        tokens = make_prompt(2000)

        for move_num in range(200):
            if board.is_game_over():
                break

            if (board.turn == chess.WHITE) == model_white:
                legal = get_legal(board)
                if not legal:
                    break

                # Get SF eval BEFORE model's move
                try:
                    pre_info = analysis_engine.analyse(
                        board, chess.engine.Limit(depth=analysis_depth)
                    )
                    pre_score = pre_info["score"].white().score(mate_score=10000)
                    sf_best = pre_info.get("pv", [None])[0]
                except Exception:
                    pre_score = 0
                    sf_best = None

                # Model plays
                params = SamplingParams(temperature=0.5, max_tokens=1, allowed_token_ids=legal)
                res = llm.generate(TokensPrompt(prompt_token_ids=tokens), params)
                model_tok = res[0].outputs[0].token_ids[0]
                model_uci = tok2uci.get(model_tok)
                if not model_uci:
                    break
                model_move = chess.Move.from_uci(model_uci)
                if model_move not in board.legal_moves:
                    break

                # Get SF eval AFTER model's move
                board.push(model_move)
                try:
                    post_info = analysis_engine.analyse(
                        board, chess.engine.Limit(depth=analysis_depth)
                    )
                    post_score = post_info["score"].white().score(mate_score=10000)
                except Exception:
                    post_score = pre_score
                board.pop()

                # Calculate centipawn loss
                if model_white:
                    cp_loss = pre_score - post_score
                else:
                    cp_loss = post_score - pre_score

                # If blunder, record the correction
                if cp_loss > blunder_threshold and sf_best and sf_best.uci() in uci2tok:
                    sf_tok = uci2tok[sf_best.uci()]
                    corrections.append(
                        {
                            "prefix": list(tokens),
                            "model_tok": model_tok,
                            "sf_tok": sf_tok,
                            "cp_loss": cp_loss,
                        }
                    )

                board.push(model_move)
                tokens.append(model_tok)
            else:
                sf_move = play_engine.play(board, sf_limit)
                board.push(sf_move.move)
                uci = sf_move.move.uci()
                if uci in uci2tok:
                    tokens.append(uci2tok[uci])

        if (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}/{n_games}: {len(corrections)} corrections so far")

    play_engine.quit()
    analysis_engine.quit()
    del llm
    torch.cuda.empty_cache()
    return corrections


def train_on_corrections(
    model,
    corrections,
    optimizer,
    device,
    epochs=3,
    pretrain_path="data/tokens_v2/full_v3/train.npy",
    mix_ratio=3,
):
    """Train model to prefer SF's move, with pretraining data mixing."""
    pretrain = np.memmap(pretrain_path, dtype=np.uint16, mode="r")
    pretrain_len = len(pretrain) // 1024

    model.train()
    total_loss = 0
    n_updates = 0

    for epoch in range(epochs):
        random.shuffle(corrections)
        for corr in corrections:
            prefix = corr["prefix"]
            sf_tok = corr["sf_tok"]
            cp_loss = corr["cp_loss"]

            input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
            input_ids = input_ids.clamp(max=model.config.vocab_size - 1)

            logits = model(input_ids=input_ids).logits[0, -1]  # [vocab]

            # Cross-entropy to prefer SF's move
            target = torch.tensor(sf_tok, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

            # Weight by severity of blunder (cp_loss / 100 capped at 5)
            weight = min(cp_loss / 100.0, 5.0)
            loss = loss * weight

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

            # Pretraining mix steps to prevent forgetting
            for _ in range(mix_ratio):
                idx = random.randint(1, pretrain_len - 2)
                seq = pretrain[idx * 1024 : (idx + 1) * 1024].astype(np.int64)
                seq = (
                    torch.tensor(seq, device=device)
                    .unsqueeze(0)
                    .clamp(max=model.config.vocab_size - 1)
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

    model.eval()
    return total_loss / max(n_updates, 1)


def quick_eval(hf_model_path, n_games=10):
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=hf_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
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
            tokens = make_prompt(2000)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument(
        "--blunder-threshold", type=int, default=100, help="Min centipawn loss to count as blunder"
    )
    parser.add_argument("--sf-play-level", type=int, default=5)
    parser.add_argument("--analysis-depth", type=int, default=12)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--output-dir", default="results/expert_iter")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"
    log = []
    all_corrections = []

    print(f"Expert iteration: {args.iterations} iters, {args.games_per_iter} games/iter")
    print(f"  Blunder threshold: {args.blunder_threshold}cp, analysis depth: {args.analysis_depth}")

    print("\n=== Baseline eval ===")
    eval0 = quick_eval(args.hf_model, args.eval_games)
    log.append({"iter": 0, "eval": {k: list(v) for k, v in eval0.items()}})

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 50}")
        print(f"Iteration {it}/{args.iterations}")

        # Generate games and find blunders
        print(f"\n[GEN+ANALYZE] {args.games_per_iter} games vs SF{args.sf_play_level}...")
        t0 = time.time()
        corrections = generate_and_annotate(
            args.hf_model,
            args.games_per_iter,
            sf_level=args.sf_play_level,
            sf_depth=args.sf_play_level,
            analysis_depth=args.analysis_depth,
            blunder_threshold=args.blunder_threshold,
        )
        gen_time = time.time() - t0
        print(f"  {gen_time:.0f}s: found {len(corrections)} blunders")
        if corrections:
            avg_cp = sum(c["cp_loss"] for c in corrections) / len(corrections)
            print(f"  Avg blunder severity: {avg_cp:.0f}cp")

        all_corrections.extend(corrections)
        if len(all_corrections) > 2000:
            all_corrections = all_corrections[-2000:]

        if not all_corrections:
            print("  No corrections — skipping training")
            continue

        # Train
        print(f"\n[TRAIN] {args.train_epochs} epochs on {len(all_corrections)} corrections...")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        avg_loss = train_on_corrections(
            model, all_corrections, optimizer, device, args.train_epochs
        )
        print(f"  Avg loss: {avg_loss:.4f}")
        model.save_pretrained(args.hf_model)
        del model, optimizer
        torch.cuda.empty_cache()

        # Eval
        print(f"\n[EVAL]")
        eval_r = quick_eval(args.hf_model, args.eval_games)
        log.append(
            {
                "iter": it,
                "eval": {k: list(v) for k, v in eval_r.items()},
                "n_corrections": len(corrections),
                "total_corrections": len(all_corrections),
                "avg_loss": avg_loss,
                "gen_time": gen_time,
            }
        )
        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
