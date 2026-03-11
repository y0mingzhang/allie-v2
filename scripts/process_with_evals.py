#!/usr/bin/env python
"""Process Lichess parquets into tokenized games WITH SF eval tokens.

Each game becomes: <bos> <sec> <inc> <elo*8> <move1> <eval1> <move2> <eval2> ... <term>

The eval token after each move represents SF's evaluation of the position
AFTER that move was played. This teaches the model chain-of-thought reasoning.

Usage:
    PYTHONPATH=. python scripts/process_with_evals.py \
        --parquet-dir ~/user_data/chess-v3/data/processed_v3 \
        --output ~/user_data/chess-v3/data/tokens_v2/full_v3_eval/train.npy \
        --max-games 10000 --sf-depth 4 --workers 4
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T

STOCKFISH = "/home/yimingz3/bin/stockfish"
SEQ_LEN = 2049  # longer to accommodate eval tokens (was 1025)


def cp_to_eval_tok(cp):
    if cp < -300:
        return T.token_to_idx["<eval:losing>"]
    elif cp < -100:
        return T.token_to_idx["<eval:bad>"]
    elif cp < -25:
        return T.token_to_idx["<eval:slight_black>"]
    elif cp <= 25:
        return T.token_to_idx["<eval:equal>"]
    elif cp <= 100:
        return T.token_to_idx["<eval:slight_white>"]
    elif cp <= 300:
        return T.token_to_idx["<eval:good>"]
    else:
        return T.token_to_idx["<eval:winning>"]


def tokenize_game_with_eval(row, engine, sf_limit):
    """Tokenize a single game with SF eval tokens."""
    try:
        moves = row["moves"]
        if not moves or len(moves.split()) < 10:
            return None

        white_elo = row.get("white_elo") or row.get("WhiteElo", 1500)
        black_elo = row.get("black_elo") or row.get("BlackElo", 1500)
        tc = row.get("time_control", "180+0")
        parts = tc.split("+")
        seconds = int(parts[0]) if parts[0].isdigit() else 180
        increment = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        # Clamp values
        seconds = min(seconds, 10800)
        increment = min(increment, 180)

        # Build prompt
        tokens = [T.token_to_idx["<bos>"]]

        sec_tok = f"<seconds_per_side:{seconds}>"
        if sec_tok not in T.token_to_idx:
            sec_tok = "<seconds_per_side:*>"
        tokens.append(T.token_to_idx[sec_tok])

        inc_tok = f"<increment:{increment}>"
        if inc_tok not in T.token_to_idx:
            inc_tok = "<increment:*>"
        tokens.append(T.token_to_idx[inc_tok])

        for d in f"{int(white_elo):04d}":
            tokens.append(T.token_to_idx[f"<elo_digit:{d}>"])
        for d in f"{int(black_elo):04d}":
            tokens.append(T.token_to_idx[f"<elo_digit:{d}>"])

        # Parse and play moves with SF eval
        board = chess.Board()
        for move_str in moves.split():
            try:
                move = board.parse_san(move_str)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
                break

            uci = move.uci()
            move_tok = f"<move:{uci}>"
            if move_tok not in T.token_to_idx:
                break

            board.push(move)
            tokens.append(T.token_to_idx[move_tok])

            # SF eval of position after move
            try:
                info = engine.analyse(board, sf_limit)
                cp = info["score"].white().score(mate_score=10000)
                tokens.append(cp_to_eval_tok(cp))
            except Exception:
                tokens.append(T.token_to_idx["<eval:equal>"])

            if len(tokens) >= SEQ_LEN - 1:
                break

        # Termination
        result = row.get("result", row.get("Result", "*"))
        if result in ("1-0", "0-1"):
            tokens.append(T.token_to_idx["<termination:normal>"])
        else:
            tokens.append(
                T.token_to_idx.get(
                    "<termination:not_normal>", T.token_to_idx["<termination:normal>"]
                )
            )

        return tokens

    except Exception:
        return None


def process_parquets(parquet_files, max_games, sf_depth):
    """Process parquet files and return tokenized games with evals."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 2})
    sf_limit = chess.engine.Limit(depth=sf_depth)

    all_games = []
    total_processed = 0

    for pf in parquet_files:
        if total_processed >= max_games:
            break
        try:
            table = pq.read_table(pf)
            df = table.to_pandas()
        except Exception:
            continue

        for _, row in df.iterrows():
            if total_processed >= max_games:
                break
            tokens = tokenize_game_with_eval(row, engine, sf_limit)
            if tokens and len(tokens) >= 20:  # minimum game length
                all_games.append(tokens)
                total_processed += 1

            if total_processed % 100 == 0 and total_processed > 0:
                print(f"  Processed {total_processed}/{max_games} games")

    engine.quit()
    return all_games


def pack_games(games, seq_len=SEQ_LEN):
    """Pack games into fixed-length sequences."""
    pad_tok = T.token_to_idx["<bos>"]
    packed = []
    for game in games:
        if len(game) > seq_len:
            game = game[:seq_len]
        else:
            game = game + [pad_tok] * (seq_len - len(game))
        packed.append(game)
    return np.array(packed, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--sf-depth", type=int, default=4)
    args = parser.parse_args()

    # Find parquet files
    parquet_dir = Path(args.parquet_dir)
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")

    t0 = time.time()
    games = process_parquets(parquet_files, args.max_games, args.sf_depth)
    elapsed = time.time() - t0
    print(f"\nProcessed {len(games)} games in {elapsed:.0f}s ({len(games) / elapsed:.1f} games/s)")

    # Pack and save
    packed = pack_games(games)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    packed.tofile(args.output)
    print(f"Saved {packed.shape} to {args.output}")
    print(f"Total tokens: {packed.size:,} ({packed.size * 2 / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
