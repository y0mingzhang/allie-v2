#!/usr/bin/env python
"""Process chess games into natural language format for Qwen3 fine-tuning.

Each game becomes a text sequence with moves in algebraic notation,
interleaved with SF eval commentary in natural language.

Format:
  Game: 180+0, White 1800 vs Black 2000
  1. e4 [+0.3, equal] e5 [+0.1, equal]
  2. Nf3 [+0.4, slight white advantage] Nc6 [+0.3, equal]
  ...
  Result: 1-0

Usage:
    PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/process_nl_chess.py \
        --parquet-dir ~/user_data/chess-v3/data/processed_v3 \
        --output ~/user_data/chess-v3/data/nl_chess/train.jsonl \
        --max-games 10000 --sf-depth 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pyarrow.parquet as pq

STOCKFISH = "/home/yimingz3/bin/stockfish"


def cp_to_description(cp):
    """Convert centipawn score to natural language description."""
    if cp < -300:
        return f"{cp / 100:+.1f}, winning for black"
    elif cp < -100:
        return f"{cp / 100:+.1f}, black has advantage"
    elif cp < -25:
        return f"{cp / 100:+.1f}, slight black edge"
    elif cp <= 25:
        return f"{cp / 100:+.1f}, equal"
    elif cp <= 100:
        return f"{cp / 100:+.1f}, slight white edge"
    elif cp <= 300:
        return f"{cp / 100:+.1f}, white has advantage"
    else:
        return f"{cp / 100:+.1f}, winning for white"


def game_to_nl(old_tokens, engine, sf_limit, tokenizer_idx_to_token):
    """Convert a tokenized game to natural language with SF evals."""
    board = chess.Board()

    # Extract metadata from tokens
    # old vocab: elo=0-9, inc=10-191, sec=192-377, moves=378-2345
    tokens = [int(t) for t in old_tokens]
    if len(tokens) < 12:
        return None

    # Parse elo from tokens (positions 3-10: 4 white elo digits + 4 black elo digits)
    try:
        white_elo = int("".join(str(tokens[i]) for i in range(3, 7)))
        black_elo = int("".join(str(tokens[i]) for i in range(7, 11)))
    except (IndexError, ValueError):
        white_elo, black_elo = 1500, 1500

    # Parse time control
    sec_idx = tokens[1] if len(tokens) > 1 else 199
    inc_idx = tokens[2] if len(tokens) > 2 else 10
    # Approximate time control
    seconds = max(0, sec_idx - 192) * 60 if sec_idx >= 192 else 180
    increment = max(0, inc_idx - 10) if inc_idx >= 10 else 0

    lines = [f"Game: {seconds // 60}+{increment}, White {white_elo} vs Black {black_elo}"]

    move_num = 1
    moves_in_line = []

    for i in range(11, len(tokens)):
        tok = tokens[i]
        if 378 <= tok <= 2345:
            tok_name = tokenizer_idx_to_token.get(tok, "")
            if tok_name.startswith("<move:") and tok_name.endswith(">"):
                uci = tok_name[6:-1]
                try:
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break

                    # Get SAN notation (human-readable)
                    san = board.san(move)

                    # SF eval before move
                    try:
                        info = engine.analyse(board, sf_limit)
                        cp = info["score"].white().score(mate_score=10000)
                        eval_desc = cp_to_description(cp)
                    except Exception:
                        eval_desc = "0.0, equal"

                    board.push(move)

                    if board.turn == chess.BLACK:
                        # White just moved
                        moves_in_line = [f"{move_num}. {san} [{eval_desc}]"]
                    else:
                        # Black just moved
                        moves_in_line.append(f"{san} [{eval_desc}]")
                        lines.append(" ".join(moves_in_line))
                        moves_in_line = []
                        move_num += 1

                except Exception:
                    break

    # Flush remaining moves
    if moves_in_line:
        lines.append(" ".join(moves_in_line))

    # Result
    result = board.result()
    if result != "*":
        lines.append(f"Result: {result}")

    if move_num < 5:  # skip very short games
        return None

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--sf-depth", type=int, default=4)
    args = parser.parse_args()

    # Load old tokenizer for decoding
    sys.path.insert(0, ".")
    from src.data.tokenizer import Tokenizer as T

    idx_to_token = T.idx_to_token

    parquet_files = sorted(Path(args.parquet_dir).rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 4})
    sf_limit = chess.engine.Limit(depth=args.sf_depth)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    t0 = time.time()
    n_written = 0

    with open(args.output, "w") as f:
        for pf in parquet_files:
            if n_written >= args.max_games:
                break
            try:
                table = pq.read_table(pf, columns=["tokens"])
            except Exception:
                continue

            for row in table.to_pydict()["tokens"]:
                if n_written >= args.max_games:
                    break
                tokens = np.array(row, dtype=np.uint16)
                nl_text = game_to_nl(tokens, engine, sf_limit, idx_to_token)
                if nl_text:
                    f.write(json.dumps({"text": nl_text}) + "\n")
                    n_written += 1

                if n_written % 100 == 0 and n_written > 0:
                    elapsed = time.time() - t0
                    rate = n_written / elapsed
                    eta = (args.max_games - n_written) / rate / 60
                    print(f"  {n_written}/{args.max_games} ({rate:.1f}/s, ETA: {eta:.0f}min)")

    engine.quit()
    elapsed = time.time() - t0
    print(f"\nDone: {n_written} games in {elapsed:.0f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
