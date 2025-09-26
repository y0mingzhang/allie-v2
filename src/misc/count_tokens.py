#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from typing import Iterable, List

from megatron.core.datasets.indexed_dataset import IndexedDataset


def iter_prefixes(inputs: List[str]) -> Iterable[str]:
    for inp in inputs:
        # If a list file is provided
        if os.path.isfile(inp) and inp.endswith(".txt"):
            with open(inp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
                for line in lines:
                    yield from iter_prefixes([line])
            continue

        # If .idx is provided, strip extension to get prefix
        if inp.endswith(".idx") and os.path.isfile(inp):
            yield inp[:-4]
            continue

        # If .bin is provided, strip extension
        if inp.endswith(".bin") and os.path.isfile(inp):
            yield inp[:-4]
            continue

        # If looks like a glob, expand once
        if any(ch in inp for ch in "*?[]"):
            for p in glob.glob(inp):
                # Re-run through checks for resolved path
                yield from iter_prefixes([p])
            continue

        # Otherwise assume it's a prefix; verify existence of either .idx or .bin
        if os.path.isfile(inp + ".idx") or os.path.isfile(inp + ".bin"):
            yield inp
            continue

        # Path did not resolve
        print(f"Warning: Skipping unresolved input: {inp}", file=sys.stderr)


def count_tokens(prefixes: Iterable[str]) -> int:
    total = 0
    seen = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        try:
            ds = IndexedDataset(prefix)
            total += int(ds.sequence_lengths.astype("int64").sum())
        except Exception as e:
            print(f"Warning: Failed to read {prefix}: {e}", file=sys.stderr)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Count tokens across Megatron indexed dataset prefixes")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Dataset prefixes (.idx/.bin), list files (.txt), or globs",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Optional seq length to also print number of samples (tokens // seq_length)",
    )
    args = parser.parse_args()

    prefixes = list(iter_prefixes(args.inputs))
    if not prefixes:
        print("No valid dataset prefixes found", file=sys.stderr)
        sys.exit(1)

    total_tokens = count_tokens(prefixes)

    print(f"prefixes={len(prefixes)} total_tokens={total_tokens}")
    if args.seq_length and args.seq_length > 0:
        print(f"samples={total_tokens // args.seq_length} (seq_length={args.seq_length})")


if __name__ == "__main__":
    main()


