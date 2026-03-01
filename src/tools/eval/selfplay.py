#!/usr/bin/env python
"""Self-play game generation using vLLM.

Plays games between two copies of the model with different elo conditioning.
Outputs tokenized games for training.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python src/tools/eval/selfplay.py \
        --model /tmp/export_600M_v3 --n-games 100 --elo-high 2500 --elo-low 2100 \
        --output selfplay_games.npy
"""

import argparse
import json
import sys
import time

import chess
from vllm import LLM, SamplingParams, TokensPrompt

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T
from src.data.tokens import CHESS_MOVES

MOVE_OFFSET = 378

tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}


def get_legal_tokens(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


def make_prompt(elo_white, elo_black, time_ctrl=180, increment=0):
    prompt = [T.token_to_idx["<bos>"]]
    prompt.append(T.token_to_idx[f"<seconds_per_side:{time_ctrl}>"])
    prompt.append(T.token_to_idx[f"<increment:{increment}>"])
    for d in f"{elo_white:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    for d in f"{elo_black:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    return prompt


def play_game(llm, elo_white, elo_black, max_moves=200):
    """Play a full game, return (tokens, result, n_moves)."""
    board = chess.Board()
    game_tokens = make_prompt(elo_white, elo_black)

    for _ in range(max_moves * 2):
        if board.is_game_over():
            break

        legal = get_legal_tokens(board)
        if not legal:
            break

        # Both players use the same model, just different elo in the prompt
        params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=legal)
        results = llm.generate(TokensPrompt(prompt_token_ids=game_tokens), params)
        tok = results[0].outputs[0].token_ids[0]

        uci = tok2uci.get(tok)
        if uci is None:
            break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break

        board.push(move)
        game_tokens.append(tok)

    result = board.result()
    if result in ("1-0", "0-1"):
        game_tokens.append(T.token_to_idx["<termination:normal>"])
    else:
        game_tokens.append(T.token_to_idx["<termination:not_normal>"])

    return game_tokens, result, board.fullmove_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--elo-high", type=int, default=2500)
    parser.add_argument("--elo-low", type=int, default=2100)
    parser.add_argument("--output", type=str, default="selfplay_games.jsonl")
    parser.add_argument("--max-moves", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        max_model_len=1024,
        enforce_eager=True,
    )

    games = []
    wins_high = draws = wins_low = 0
    start = time.time()

    for i in range(args.n_games):
        # Alternate who plays white
        if i % 2 == 0:
            elo_w, elo_b = args.elo_high, args.elo_low
            high_is_white = True
        else:
            elo_w, elo_b = args.elo_low, args.elo_high
            high_is_white = False

        tokens, result, n_moves = play_game(llm, elo_w, elo_b, args.max_moves)

        # Determine winner from high-elo perspective
        if result == "1-0":
            if high_is_white:
                wins_high += 1
                outcome = "high_wins"
            else:
                wins_low += 1
                outcome = "low_wins"
        elif result == "0-1":
            if high_is_white:
                wins_low += 1
                outcome = "low_wins"
            else:
                wins_high += 1
                outcome = "high_wins"
        else:
            draws += 1
            outcome = "draw"

        games.append(
            {
                "tokens": tokens,
                "result": result,
                "outcome": outcome,
                "n_moves": n_moves,
                "elo_white": elo_w,
                "elo_black": elo_b,
            }
        )

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(
                f"  Game {i + 1}/{args.n_games}: {result} ({n_moves}mv) "
                f"[high:{wins_high}W {draws}D {wins_low}L] "
                f"{elapsed / (i + 1):.1f}s/game"
            )

    elapsed = time.time() - start
    total = args.n_games
    print(
        f"\nSelf-play complete: {wins_high}W {draws}D {wins_low}L "
        f"(high elo {args.elo_high} vs low {args.elo_low})"
    )
    print(f"High elo win rate: {wins_high / total:.1%}")
    print(f"Time: {elapsed:.0f}s ({elapsed / total:.1f}s/game)")

    # Save as JSONL
    with open(args.output, "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    print(f"Saved {len(games)} games to {args.output}")


if __name__ == "__main__":
    main()
