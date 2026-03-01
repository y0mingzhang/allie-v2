#!/usr/bin/env python
"""Play chess games against Stockfish using vLLM inference.

Usage:
    CUDA_VISIBLE_DEVICES=4 python src/tools/eval/vllm_play_stockfish.py \
        --model /tmp/export_200M_v3 \
        --sf-level 1 --n-games 20 --elo 2000
"""

import argparse
import json
import sys
import time

import chess
import chess.engine
from vllm import LLM, SamplingParams, TokensPrompt

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T
from src.data.tokens import CHESS_MOVES

MOVE_OFFSET = 378

tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}


def get_legal_tokens(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


def make_prompt(elo_white=2000, elo_black=2000, time_ctrl=180, increment=0):
    prompt = [T.token_to_idx["<bos>"]]
    prompt.append(T.token_to_idx[f"<seconds_per_side:{time_ctrl}>"])
    prompt.append(T.token_to_idx[f"<increment:{increment}>"])
    for d in f"{elo_white:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    for d in f"{elo_black:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    return prompt


def play_game(llm, engine, sf_limit, elo, model_is_white):
    board = chess.Board()
    game_tokens = make_prompt(elo_white=elo, elo_black=elo)

    for _ in range(400):
        if board.is_game_over():
            break

        model_turn = (board.turn == chess.WHITE) == model_is_white

        if model_turn:
            legal = get_legal_tokens(board)
            if not legal:
                break
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
        else:
            result = engine.play(board, sf_limit)
            board.push(result.move)
            uci = result.move.uci()
            if uci in uci2tok:
                game_tokens.append(uci2tok[uci])

    return board.result(), board.fullmove_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sf-level", type=int, default=1)
    parser.add_argument("--sf-depth", type=int, default=None)
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--elo", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=True,
    )

    engine = chess.engine.SimpleEngine.popen_uci("/home/yimingz3/bin/stockfish")
    engine.configure({"Skill Level": args.sf_level, "Threads": 1})
    sf_limit = (
        chess.engine.Limit(depth=args.sf_depth) if args.sf_depth else chess.engine.Limit(time=0.1)
    )

    wins = draws = losses = 0
    start = time.time()

    for i in range(args.n_games):
        model_white = i % 2 == 0
        result, nmoves = play_game(llm, engine, sf_limit, args.elo, model_white)

        if result == "1-0":
            if model_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if model_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

        color = "W" if model_white else "B"
        print(
            f"  Game {i + 1:3d}: {color} {result:5s} {nmoves:3d}moves  [{wins}W {draws}D {losses}L]"
        )

    engine.quit()
    elapsed = time.time() - start

    print(f"\nResults vs SF{args.sf_level}: {wins}W {draws}D {losses}L / {args.n_games} games")
    print(f"Win rate: {100 * wins / args.n_games:.0f}%")
    print(f"Time: {elapsed:.0f}s ({elapsed / args.n_games:.1f}s/game)")

    if args.output:
        data = {
            "model": args.model,
            "sf_level": args.sf_level,
            "elo_conditioning": args.elo,
            "n_games": args.n_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / args.n_games,
            "time_per_game": elapsed / args.n_games,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
