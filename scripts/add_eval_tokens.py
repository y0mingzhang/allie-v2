#!/usr/bin/env python
"""Add SF eval tokens to existing tokenized games.

Reads packed npy data, replays each game, evaluates each position with SF,
and inserts eval bin tokens after each move.

Input: data/tokens_v2/full_v3/train.npy (packed sequences of length 1025)
Output: data/tokens_v2/full_v3_eval/train.npy (longer sequences with eval tokens)

Usage:
    PYTHONPATH=. python scripts/add_eval_tokens.py \
        --input data/tokens_v2/full_v3/train.npy \
        --output data/tokens_v2/full_v3_eval/train.npy \
        --max-games 10000 --sf-depth 8
"""

import argparse
import os
import sys
import time

import chess
import chess.engine
import numpy as np

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T

STOCKFISH = "/home/yimingz3/bin/stockfish"
MOVE_START = T.token_to_idx["<move:a1a2>"]  # first move token
MOVE_END = T.token_to_idx[
    f"<move:{list(T.token_to_idx.keys())[-1].split(':')[1][:-1]}>" if False else "<move:h8h7>"
]


def cp_to_eval_bin(cp):
    """Convert centipawn score to eval bin token ID."""
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


def is_move_token(tok_id):
    return 378 <= tok_id <= 2345


def process_game(tokens, engine, sf_limit):
    """Add eval tokens after each move in a game sequence.

    Input: [bos, seconds, increment, elo*8, move1, move2, ..., termination]
    Output: [bos, seconds, increment, elo*8, move1, eval1, move2, eval2, ..., termination]
    """
    board = chess.Board()
    new_tokens = []
    prompt_end = 11  # bos + seconds + increment + 4 white elo digits + 4 black elo digits

    # Copy prompt tokens as-is
    for i in range(min(prompt_end, len(tokens))):
        tok = int(tokens[i])
        if tok == T.token_to_idx.get("<bos>", 2348):
            new_tokens.append(tok)
        elif tok < 2346:  # not eval/termination/special from new vocab
            new_tokens.append(tok)
        else:
            break

    # Process moves
    for i in range(prompt_end, len(tokens)):
        tok = int(tokens[i])

        # Skip padding (old bos token used as padding)
        if tok >= 2353:  # termination or special in new vocab
            new_tokens.append(tok)
            continue

        if is_move_token(tok):
            # Get the UCI move
            uci = T.idx_to_token.get(tok, "")
            if uci.startswith("<move:") and uci.endswith(">"):
                move_str = uci[6:-1]
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        new_tokens.append(tok)

                        # Evaluate position after move
                        try:
                            info = engine.analyse(board, sf_limit)
                            score = info["score"].white()
                            cp = score.score(mate_score=10000)
                            new_tokens.append(cp_to_eval_bin(cp))
                        except Exception:
                            new_tokens.append(T.token_to_idx["<eval:equal>"])
                    else:
                        new_tokens.append(tok)
                except Exception:
                    new_tokens.append(tok)
            else:
                new_tokens.append(tok)
        else:
            new_tokens.append(tok)

    return new_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--sf-depth", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=1025)
    parser.add_argument(
        "--new-seq-length",
        type=int,
        default=1025,
        help="Output sequence length (longer to fit eval tokens)",
    )
    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    data = np.memmap(args.input, dtype=np.uint16, mode="r")
    n_sequences = len(data) // args.seq_length
    print(f"  {n_sequences:,} sequences of length {args.seq_length}")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 4})
    sf_limit = chess.engine.Limit(depth=args.sf_depth)

    n_games = min(args.max_games, n_sequences)
    all_tokens = []
    start = time.time()

    for i in range(n_games):
        seq = data[i * args.seq_length : (i + 1) * args.seq_length]
        new_seq = process_game(seq, engine, sf_limit)

        # Truncate or pad to new_seq_length
        if len(new_seq) > args.new_seq_length:
            new_seq = new_seq[: args.new_seq_length]
        else:
            pad_tok = T.token_to_idx["<bos>"]
            new_seq.extend([pad_tok] * (args.new_seq_length - len(new_seq)))

        all_tokens.append(new_seq)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_games - i - 1) / rate
            print(
                f"  Processed {i + 1}/{n_games} games ({rate:.1f} games/s, ETA: {eta / 60:.0f}min)"
            )

    engine.quit()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = np.array(all_tokens, dtype=np.uint16)
    output.tofile(args.output)
    print(f"\nSaved {len(all_tokens)} sequences to {args.output}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
