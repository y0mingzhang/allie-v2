#!/usr/bin/env python
"""Process pre-tokenized parquets to add SF eval tokens.

Reads processed parquets (which have 'tokens' column as uint16 arrays),
decodes moves, replays on a board, evaluates with SF, inserts eval tokens.

Usage:
    PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/process_eval_tokens.py \
        --parquet-dir ~/user_data/chess-v3/data/processed_v3 \
        --output ~/user_data/chess-v3/data/tokens_v2/eval_pilot/train.npy \
        --max-games 5000 --sf-depth 4
"""

import argparse
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
SEQ_LEN = 2049
OLD_BOS = 2348  # BOS in old 2350-token vocab
MOVE_START = 378
MOVE_END = 2345


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


def add_evals_to_game(old_tokens, engine, sf_limit):
    """Take old-vocab token IDs, add eval tokens after each move."""
    board = chess.Board()
    new_tokens = []

    # Copy prompt (first 11 tokens: bos + sec + inc + 8 elo digits)
    # Remap old BOS (2348) to new BOS (2355)
    prompt_len = 11
    for i in range(min(prompt_len, len(old_tokens))):
        tok = int(old_tokens[i])
        if tok == 2348 and i == 0:  # old BOS → new BOS
            tok = T.token_to_idx["<bos>"]
        new_tokens.append(tok)

    # Process moves
    for i in range(prompt_len, len(old_tokens)):
        tok = int(old_tokens[i])

        if MOVE_START <= tok <= MOVE_END:
            uci_tok = T.idx_to_token.get(tok, "")
            if uci_tok.startswith("<move:") and uci_tok.endswith(">"):
                uci = uci_tok[6:-1]
                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        new_tokens.append(cp_to_eval_tok(cp))
                        board.push(move)
                        new_tokens.append(tok)
                        # SF eval
                        try:
                            info = engine.analyse(board, sf_limit)
                            cp = info["score"].white().score(mate_score=10000)
                            new_tokens.append(cp_to_eval_tok(cp))
                        except Exception:
                            new_tokens.append(T.token_to_idx["<eval:equal>"])
                        continue
                except Exception:
                    pass

        # Remap old termination tokens
        if tok == 2346:  # old termination:normal
            tok = T.token_to_idx["<termination:normal>"]
        elif tok == 2347:  # old termination:not_normal
            tok = T.token_to_idx.get(
                "<termination:not_normal>", T.token_to_idx["<termination:normal>"]
            )
        elif tok == 2348:  # old BOS used as padding
            tok = T.token_to_idx["<bos>"]
        new_tokens.append(tok)

    return new_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--sf-depth", type=int, default=4)
    args = parser.parse_args()

    parquet_files = sorted(Path(args.parquet_dir).rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 4})
    sf_limit = chess.engine.Limit(depth=args.sf_depth)

    all_games = []
    t0 = time.time()

    for pf in parquet_files:
        if len(all_games) >= args.max_games:
            break
        try:
            table = pq.read_table(pf, columns=["tokens"])
        except Exception:
            continue

        for row in table.to_pydict()["tokens"]:
            if len(all_games) >= args.max_games:
                break

            tokens = np.array(row, dtype=np.uint16)
            if len(tokens) < 20:
                continue

            new_tokens = add_evals_to_game(tokens, engine, sf_limit)

            # Pad/truncate to SEQ_LEN
            if len(new_tokens) > SEQ_LEN:
                new_tokens = new_tokens[:SEQ_LEN]
            else:
                new_tokens += [T.token_to_idx["<bos>"]] * (SEQ_LEN - len(new_tokens))

            all_games.append(new_tokens)

            if len(all_games) % 100 == 0:
                elapsed = time.time() - t0
                rate = len(all_games) / elapsed
                eta = (args.max_games - len(all_games)) / rate / 60
                print(
                    f"  {len(all_games)}/{args.max_games} games ({rate:.1f}/s, ETA: {eta:.0f}min)"
                )

    engine.quit()

    # Save
    packed = np.array(all_games, dtype=np.uint16)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    packed.tofile(args.output)

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_games)} games in {elapsed:.0f}s")
    print(f"Shape: {packed.shape}, saved to {args.output}")


if __name__ == "__main__":
    main()
