#!/usr/bin/env python3

import argparse
import random

from megatron.core.datasets import indexed_dataset
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, IterableDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Megatron indexed dataset from parquet list")
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
        "--output-prefix",
        type=str,
        default="data/bin/0924",
        help="Prefix for output files; actual files include split and shard range",
    )
    return parser.parse_args()


def read_paths(list_file: str) -> list[str]:
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


def main():
    args = parse_args()

    parquet_paths = read_paths(args.list_file)
    start = max(0, int(args.start))
    end = int(args.end) if args.end is not None else len(parquet_paths)
    if start >= end:
        raise ValueError(f"Empty slice: start={start} >= end={end}")
    parquet_paths = parquet_paths[start:end]

    dataset: IterableDataset = load_dataset(
        "parquet",
        data_files=parquet_paths,
        split="train",
        streaming=True,
    )
    print("Loaded dataset")

    # Shuffle for split assignment reproducibility
    dataset = dataset.shuffle(seed=2093 + start)

    output_bin = f"{args.output_prefix}_{{split}}_{start}_{end}.bin"
    output_idx = f"{args.output_prefix}_{{split}}_{start}_{end}.idx"

    train_builder = indexed_dataset.IndexedDatasetBuilder(output_bin.format(split="train"), dtype=np.uint16)
    val_builder = indexed_dataset.IndexedDatasetBuilder(output_bin.format(split="val"), dtype=np.uint16)
    test_builder = indexed_dataset.IndexedDatasetBuilder(output_bin.format(split="test"), dtype=np.uint16)

    rng = random.Random(209 + start)
    for tokens in tqdm(dataset["tokens"]):
        rs = rng.random()
        if rs < 2e-4:
            val_builder.add_document(tokens, [len(tokens)])
        elif rs < 4e-4:
            test_builder.add_document(tokens, [len(tokens)])
        else:
            train_builder.add_document(tokens, [len(tokens)])

    train_builder.finalize(output_idx.format(split="train"))
    val_builder.finalize(output_idx.format(split="val"))
    test_builder.finalize(output_idx.format(split="test"))


if __name__ == "__main__":
    main()
