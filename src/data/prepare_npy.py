#!/usr/bin/env python3

import argparse
import logging
import os
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _read_list_file(list_file: str) -> list[str]:
    paths: list[str] = []
    with open(list_file, "r", encoding="utf-8") as f:
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
    parser = argparse.ArgumentParser(description="Concatenate token arrays from parquet(s) and dump uint16 .npy")
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
    return parser.parse_args()


def build_streaming_dataset(paths: list[str]):
    return load_dataset(
        "parquet",
        data_files=paths,
        split="train",
        streaming=True,
    ).shuffle(seed=913842)


def concatenate_and_save_with_val_split(
    paths: list[str], out_dir: str, show_progress: bool = False, val_prob: float = 0.0001, rng_seed: int = 12345
) -> None:
    # First pass: count train/val tokens deterministically
    logger.info("Counting tokens for train/val (first pass)...")
    dataset = build_streaming_dataset(paths)
    rng = np.random.default_rng(rng_seed)
    train_total = 0
    val_total = 0
    iterator = _iter_tokens(dataset)
    if show_progress:
        iterator = tqdm(iterator, desc="Counting sequences")
    for tokens in iterator:
        is_val = rng.random() < val_prob
        if is_val:
            val_total += len(tokens)
        else:
            train_total += len(tokens)
    if train_total + val_total == 0:
        raise ValueError("No tokens found in provided parquets")
    logger.info("Total tokens -> train: %d, val: %d (%.6f%% val)", train_total, val_total, (val_total / max(1, train_total + val_total)) * 100.0)

    # Second pass: allocate and write
    logger.info("Concatenating into uint16 arrays (second pass)...")
    os.makedirs(out_dir, exist_ok=True)
    train_out = np.empty(train_total, dtype=np.uint16)
    val_out = np.empty(val_total, dtype=np.uint16)

    dataset = build_streaming_dataset(paths)
    rng = np.random.default_rng(rng_seed)
    train_pos = 0
    val_pos = 0
    iterator = _iter_tokens(dataset)
    if show_progress:
        iterator = tqdm(iterator, desc="Writing sequences")
    for tokens in iterator:
        is_val = rng.random() < val_prob
        arr = np.asarray(tokens, dtype=np.uint16)
        n = arr.size
        if is_val:
            val_out[val_pos : val_pos + n] = arr
            val_pos += n
        else:
            train_out[train_pos : train_pos + n] = arr
            train_pos += n

    assert train_pos == train_total, f"Wrote {train_pos} train tokens but expected {train_total}"
    assert val_pos == val_total, f"Wrote {val_pos} val tokens but expected {val_total}"

    train_path = os.path.join(out_dir, "train.npy")
    val_path = os.path.join(out_dir, "val.npy")
    logger.info("Saving train to %s", train_path)
    np.save(train_path, train_out)
    logger.info("Saving val to %s", val_path)
    np.save(val_path, val_out)
    logger.info(
        "Done: train=%s (shape=%s), val=%s (shape=%s), dtype=%s",
        train_path,
        train_out.shape,
        val_path,
        val_out.shape,
        train_out.dtype,
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
    concatenate_and_save_with_val_split(slice_paths, args.out_dir, show_progress=args.progress)


if __name__ == "__main__":
    main()
