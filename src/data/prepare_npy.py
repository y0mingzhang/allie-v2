#!/usr/bin/env python3

import argparse
from collections.abc import Iterable
import logging
import os

from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _read_list_file(list_file: str) -> list[str]:
    paths: list[str] = []
    with open(list_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    if not paths:
        raise ValueError(f"No parquet paths found in {list_file}")
    return paths


def _iter_tokens(dataset) -> Iterable[list[int]]:
    for example in dataset:
        tokens = example.get("tokens")
        if tokens is None:
            continue
        yield tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate token arrays from parquet(s) and dump uint16 .npy"
    )
    parser.add_argument(
        "--list-file",
        type=str,
        required=True,
        help="Path to a text file containing newline-separated parquet file paths",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive) into the parquet list",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) into the parquet list",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory; will write train.npy and val.npy",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars while reading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Pack games into fixed-size batches (default: 1024)",
    )
    parser.add_argument(
        "--bos-token-id",
        type=int,
        default=2348,
        help="Token ID for beginning-of-sequence (default: 2348)",
    )
    return parser.parse_args()


def build_streaming_dataset(paths: list[str]):
    return load_dataset(
        "parquet",
        data_files=paths,
        split="train",
        streaming=True,
    ).shuffle(seed=913842)


def pack_games_into_batches(
    paths: list[str],
    out_dir: str,
    batch_size: int,
    bos_token_id: int,
    show_progress: bool = False,
    val_prob: float = 0.0001,
    rng_seed: int = 12345,
) -> None:
    pack_size = batch_size + 1
    logger.info("Packing games into batches of size %d...", pack_size)
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1: Count tokens
    logger.info("Pass 1: Counting tokens...")
    dataset = build_streaming_dataset(paths)
    rng = np.random.default_rng(rng_seed)

    train_count = 0
    val_count = 0
    train_buffer_len = 0
    val_buffer_len = 0

    iterator = _iter_tokens(dataset)
    if show_progress:
        iterator = tqdm(iterator, desc="Counting")

    for tokens in iterator:
        assert isinstance(tokens, list), str(tokens)
        assert tokens[0] == bos_token_id, str(tokens)

        is_val = rng.random() < val_prob
        if is_val:
            val_buffer_len += len(tokens)
            if val_buffer_len >= pack_size:
                val_count += pack_size
                val_buffer_len = 0
        else:
            train_buffer_len += len(tokens)
            if train_buffer_len >= pack_size:
                train_count += pack_size
                train_buffer_len = 0

    logger.info("Train: %d tokens, Val: %d tokens", train_count, val_count)

    # Pass 2: Write tokens
    logger.info("Pass 2: Writing tokens...")
    train_out = np.empty(train_count, dtype=np.uint16)
    val_out = np.empty(val_count, dtype=np.uint16)

    dataset = build_streaming_dataset(paths)
    rng = np.random.default_rng(rng_seed)

    train_pos = 0
    val_pos = 0
    train_buffer = []
    val_buffer = []

    iterator = _iter_tokens(dataset)
    if show_progress:
        iterator = tqdm(iterator, desc="Writing")

    for tokens in iterator:
        assert isinstance(tokens, list), str(tokens)
        assert tokens[0] == bos_token_id, str(tokens)

        is_val = rng.random() < val_prob
        buffer = val_buffer if is_val else train_buffer
        pos = val_pos if is_val else train_pos
        arr = val_out if is_val else train_out

        buffer.extend(tokens)
        if len(buffer) >= pack_size:
            arr[pos : pos + pack_size] = buffer[:pack_size]
            if is_val:
                val_pos += pack_size
            else:
                train_pos += pack_size
            buffer.clear()

    logger.info(
        "Created %d train batches (%d tokens), %d val batches (%d tokens)",
        len(train_out) // pack_size,
        len(train_out),
        len(val_out) // pack_size,
        len(val_out),
    )

    train_path = os.path.join(out_dir, "train.npy")
    val_path = os.path.join(out_dir, "val.npy")
    np.save(train_path, train_out)
    np.save(val_path, val_out)
    logger.info(
        "Done: train=%s (%s), val=%s (%s)", train_path, train_out.shape, val_path, val_out.shape
    )


def main() -> None:
    args = parse_args()
    parquet_paths = _read_list_file(args.list_file)
    start = max(0, int(args.start))
    end = int(args.end) if args.end is not None else len(parquet_paths)
    if start >= end:
        raise ValueError(f"Empty slice: start={start} >= end={end}")
    slice_paths = parquet_paths[start:end]
    logger.info("Using %d parquet(s)", len(slice_paths))
    pack_games_into_batches(
        slice_paths,
        args.out_dir,
        batch_size=args.batch_size,
        bos_token_id=args.bos_token_id,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
