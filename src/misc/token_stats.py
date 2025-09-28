import argparse
import glob
import os
from typing import Iterable, List

import numpy as np


def find_npy_files(path: str) -> List[str]:
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.npy")
        paths = glob.glob(pattern, recursive=True)
    else:
        paths = [path] if path.endswith(".npy") and os.path.isfile(path) else []
    return sorted(paths)


def compute_stats(npy_paths: Iterable[str], seq_length: int) -> tuple[int, int]:
    total_tokens = 0
    total_sequences = 0

    for npy_path in npy_paths:
        arr = np.load(npy_path, mmap_mode="r")

        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array in {npy_path}, got shape {arr.shape}")

        num_tokens = len(arr)
        num_sequences = 0
        if num_tokens >= seq_length + 1:
            num_sequences = (num_tokens - 1) // seq_length

        total_tokens += num_tokens
        total_sequences += num_sequences

    return total_tokens, total_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute token and sequence statistics for .npy token shards.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/tokens",
        help="Directory or .npy file containing token data (default: data/tokens)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length used to count complete sequences (default: 1024)",
    )
    args = parser.parse_args()

    npy_paths = find_npy_files(args.path)
    if not npy_paths:
        raise ValueError(f"No .npy files found under {args.path}")

    total_tokens, total_sequences = compute_stats(npy_paths, args.seq_length)

    print(f"Scanned {len(npy_paths)} files under {os.path.abspath(args.path)}")
    print(f"Total tokens: {total_tokens:,}")
    print(
        f"Complete sequences of length {args.seq_length} (per picotron NpyTokenDataset logic): {total_sequences:,}"
    )


if __name__ == "__main__":
    main()

