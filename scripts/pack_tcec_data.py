#!/usr/bin/env python3
"""Tokenize TCEC engine games and pack into npy format."""

import argparse
import logging
import os

import chess.pgn
import numpy as np

from data.tokenizer import Tokenizer, build_game_prompt_tokens
from data.tokens import TerminationTokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BOS = Tokenizer.token_to_idx[Tokenizer.bos_token]
PACK_SIZE = 1025
ASSIGNED_ELO = 3000  # treat engine games as elo 3000


def tokenize_engine_game(game):
    """Tokenize a PGN game into our token format."""
    moves = [m.uci() for m in game.mainline_moves()]
    if len(moves) < 5:
        return None

    # Parse time control to seconds_per_side + increment
    tc = game.headers.get("TimeControl", "")
    # TCEC TCs are complex (40/7200:20/3600:900+30), use the increment part
    if "+" in tc:
        parts = tc.split("+")
        increment = parts[-1].strip()
        # Base time: use the last segment before +
        base_part = parts[0].split(":")[-1].strip()
        try:
            seconds = str(int(base_part))
        except ValueError:
            seconds = "*"
    else:
        seconds = "*"
        increment = "0"

    result = game.headers.get("Result", "*")
    normal_termination = result in ("1-0", "0-1", "1/2-1/2")

    token_ids = build_game_prompt_tokens(seconds, increment, ASSIGNED_ELO, ASSIGNED_ELO, moves)
    # Add termination token
    if normal_termination:
        token_ids.append(Tokenizer.token_to_idx[TerminationTokens.NORMAL_TERMINATION.value])
    else:
        token_ids.append(Tokenizer.token_to_idx[TerminationTokens.NOT_NORMAL_TERMINATION.value])

    return token_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--elo", type=int, default=3000)
    args = parser.parse_args()

    global ASSIGNED_ELO
    ASSIGNED_ELO = args.elo

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    train_buf, val_buf = [], []
    train_packs, val_packs = [], []
    n_games = 0
    n_skipped = 0

    with open(args.pgn) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            tokens = tokenize_engine_game(game)
            if tokens is None:
                n_skipped += 1
                continue

            n_games += 1
            is_val = rng.random() < 0.001
            buf = val_buf if is_val else train_buf
            packs = val_packs if is_val else train_packs

            buf.extend(tokens)
            while len(buf) >= PACK_SIZE:
                packs.append(np.array(buf[:PACK_SIZE], dtype=np.uint16))
                del buf[:PACK_SIZE]

            if n_games % 5000 == 0:
                logger.info("Processed %d games, %d train packs", n_games, len(train_packs))

    train_arr = np.concatenate(train_packs) if train_packs else np.array([], dtype=np.uint16)
    val_arr = np.concatenate(val_packs) if val_packs else np.array([], dtype=np.uint16)

    np.save(os.path.join(args.out_dir, "train.npy"), train_arr)
    np.save(os.path.join(args.out_dir, "val.npy"), val_arr)
    logger.info(
        "Done: %d games (%d skipped), train=%d tokens, val=%d tokens",
        n_games,
        n_skipped,
        len(train_arr),
        len(val_arr),
    )


if __name__ == "__main__":
    main()
