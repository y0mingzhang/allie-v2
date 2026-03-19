#!/usr/bin/env python3
"""Fast npy packing with two-pass approach: count then write. Uses pyarrow directly."""

import argparse
import glob
import logging
import os

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BOS_TOKEN_ID = 2348
PACK_SIZE = 1025
VAL_PROB = 0.0001
RNG_SEED = 12345


RESULT_TOKEN_MAP = {"1-0": 2350, "0-1": 2352, "1/2-1/2": 2351}


def iter_games(files, include_result=False):
    columns = ["tokens", "result"] if include_result else ["tokens"]
    for f in files:
        table = pq.read_table(f, columns=columns)
        tok_col = table.column("tokens")
        res_col = (
            table.column("result") if include_result and "result" in table.column_names else None
        )
        for j in range(len(tok_col)):
            tokens = tok_col[j].as_py()
            if tokens and tokens[0] == BOS_TOKEN_ID:
                if include_result and res_col is not None:
                    result_str = res_col[j].as_py()
                    result_tok = RESULT_TOKEN_MAP.get(result_str, 2351)  # default draw
                    tokens = [tokens[0], result_tok] + tokens[1:]  # insert at position 1
                yield tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--include-result",
        action="store_true",
        help="Insert game result token at position 1 for value training",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob, recursive=True))
    logger.info("Found %d parquet files", len(files))
    os.makedirs(args.out_dir, exist_ok=True)

    # Pass 1: count
    logger.info("Pass 1: counting tokens...")
    rng = np.random.default_rng(RNG_SEED)
    train_count = val_count = 0
    train_buf_len = val_buf_len = 0
    n_games = 0

    for tokens in iter_games(files, include_result=args.include_result):
        n_games += 1
        is_val = rng.random() < VAL_PROB
        if is_val:
            val_buf_len += len(tokens)
            if val_buf_len >= PACK_SIZE:
                val_count += PACK_SIZE
                val_buf_len -= PACK_SIZE
        else:
            train_buf_len += len(tokens)
            if train_buf_len >= PACK_SIZE:
                train_count += PACK_SIZE
                train_buf_len -= PACK_SIZE
        if n_games % 5_000_000 == 0:
            logger.info(
                "  counted %dM games, %dB train tokens so far",
                n_games // 1_000_000,
                train_count // 1_000_000_000,
            )

    logger.info(
        "Pass 1 done: %d games, %d train tokens, %d val tokens", n_games, train_count, val_count
    )

    # Pass 2: write
    logger.info("Pass 2: writing...")
    train_out = np.empty(train_count, dtype=np.uint16)
    val_out = np.empty(val_count, dtype=np.uint16)
    rng = np.random.default_rng(RNG_SEED)
    train_pos = val_pos = 0
    train_buf, val_buf = [], []
    n_games = 0

    for tokens in iter_games(files, include_result=args.include_result):
        n_games += 1
        is_val = rng.random() < VAL_PROB
        buf = val_buf if is_val else train_buf
        arr = val_out if is_val else train_out
        pos_ref = [val_pos] if is_val else [train_pos]

        buf.extend(tokens)
        if len(buf) >= PACK_SIZE:
            arr[pos_ref[0] : pos_ref[0] + PACK_SIZE] = buf[:PACK_SIZE]
            pos_ref[0] += PACK_SIZE
            if is_val:
                val_pos = pos_ref[0]
            else:
                train_pos = pos_ref[0]
            del buf[:PACK_SIZE]
        if n_games % 5_000_000 == 0:
            logger.info("  wrote %dM games", n_games // 1_000_000)

    train_path = os.path.join(args.out_dir, "train.npy")
    val_path = os.path.join(args.out_dir, "val.npy")
    np.save(train_path, train_out)
    np.save(val_path, val_out)
    logger.info(
        "Saved: train=%s (%d tokens), val=%s (%d tokens)",
        train_path,
        len(train_out),
        val_path,
        len(val_out),
    )


if __name__ == "__main__":
    main()
