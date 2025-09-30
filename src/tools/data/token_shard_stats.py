import argparse
from collections import defaultdict
from collections.abc import Iterable
import glob
import os

import numpy as np


def find_npy_files(path: str) -> list[str]:
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.npy")
        paths = glob.glob(pattern, recursive=True)
    else:
        paths = [path] if path.endswith(".npy") and os.path.isfile(path) else []
    return sorted(paths)


def _array_counts(arr: np.ndarray, seq_length: int, source: str) -> tuple[int, int]:
    if arr.ndim == 1:
        num_tokens = int(arr.shape[0])
        if num_tokens % seq_length != 0:
            raise ValueError(
                f"Array in {source} has {num_tokens} tokens which is not divisible by sequence length {seq_length}."
            )
        num_sequences = num_tokens // seq_length

    elif arr.ndim == 2:
        if arr.shape[1] != seq_length:
            raise ValueError(
                f"Expected second dimension {seq_length} in {source}, got shape {arr.shape}."
            )
        num_tokens = int(arr.shape[0] * arr.shape[1])
        num_sequences = int(arr.shape[0])
    else:
        raise ValueError(f"Unsupported array rank {arr.ndim} in {source}.")

    return num_tokens, num_sequences


def compute_stats(
    npy_paths: Iterable[str], seq_length: int
) -> tuple[tuple[int, int], dict[str, tuple[int, int]]]:
    total_tokens = 0
    total_sequences = 0
    per_suffix: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for npy_path in npy_paths:
        arr = np.load(npy_path, mmap_mode="r")
        tokens, sequences = _array_counts(arr, seq_length, npy_path)

        total_tokens += tokens
        total_sequences += sequences

        suffix = os.path.splitext(os.path.basename(npy_path))[0]
        bucket = per_suffix[suffix]
        bucket[0] += tokens
        bucket[1] += sequences

    per_suffix_final = {suffix: (vals[0], vals[1]) for suffix, vals in sorted(per_suffix.items())}

    return (total_tokens, total_sequences), per_suffix_final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute token and sequence statistics for .npy token shards."
    )
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

    (total_tokens, total_sequences), per_suffix = compute_stats(npy_paths, args.seq_length)

    print(f"Scanned {len(npy_paths)} files under {os.path.abspath(args.path)}")
    print(f"Total tokens: {total_tokens:,}")
    print(
        f"Complete sequences of length {args.seq_length} (per picotron NpyTokenDataset logic): {total_sequences:,}"
    )

    if len(per_suffix) > 1:
        print("\nPer-suffix breakdown:")
        for suffix, (tokens, sequences) in per_suffix.items():
            print(f"  {suffix}: {tokens:,} tokens · {sequences:,} complete sequences")


if __name__ == "__main__":
    main()
