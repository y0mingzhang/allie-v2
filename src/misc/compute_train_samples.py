import argparse
import os
import sys

import numpy as np

# Ensure local Megatron-LM checkout is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MEGATRON_PATH = os.path.join(_REPO_ROOT, "Megatron-LM")
if _MEGATRON_PATH not in sys.path:
    sys.path.insert(0, _MEGATRON_PATH)

try:
    # Import Megatron's IndexedDataset from the local checkout
    from megatron.core.datasets.indexed_dataset import IndexedDataset
except Exception as e:
    print(f"ERROR: failed to import megatron IndexedDataset: {e}", file=sys.stderr)
    sys.exit(2)


def _normalize_prefix(data_prefix: str) -> str:
    # Accept either prefix or explicit .idx/.bin path; convert to absolute prefix
    prefix = os.path.abspath(data_prefix)
    if prefix.endswith(".idx") or prefix.endswith(".bin"):
        prefix = os.path.splitext(prefix)[0]
    idx_path = prefix + ".idx"
    bin_path = prefix + ".bin"
    if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
        raise AssertionError(
            f"One or both of the .idx and .bin files cannot be found at the prefix: {prefix}.\n"
            f"Expected files:\n  - {idx_path}\n  - {bin_path}"
        )
    return prefix


def compute_train_samples(
    data_prefix: str,
    seq_length: int,
    global_batch_size: int,
    add_extra_token_to_sequence: bool = True,
) -> int:
    prefix = _normalize_prefix(data_prefix)
    ds = IndexedDataset(prefix, mmap=True)
    total_tokens = int(np.sum(ds.sequence_lengths))

    add_extra = 1 if add_extra_token_to_sequence else 0
    if total_tokens <= add_extra or seq_length <= 0 or global_batch_size <= 0:
        return 0

    samples_per_epoch = (total_tokens - add_extra) // seq_length
    if samples_per_epoch <= 0:
        return 0

    # Round down to a multiple of global batch size
    train_samples = (samples_per_epoch // global_batch_size) * global_batch_size
    return int(train_samples)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute --train-samples for a single pass over an IndexedDataset (.idx/.bin) "
            "given --seq-length and --global-batch-size."
        )
    )
    parser.add_argument(
        "--data-prefix",
        required=True,
        help="Path prefix of the dataset without extension (e.g., data/bin/foo)",
    )
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument(
        "--no-add-extra-token",
        action="store_true",
        help="Disable the +1 token per sample used by default in Megatron GPTDataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information to stderr while keeping stdout as the numeric result",
    )

    args = parser.parse_args()

    # Compute result
    train_samples = compute_train_samples(
        data_prefix=args.data_prefix,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        add_extra_token_to_sequence=not args.no_add_extra_token,
    )

    try:
        prefix = _normalize_prefix(args.data_prefix)
        ds = IndexedDataset(prefix, mmap=True)
        total_tokens = int(np.sum(ds.sequence_lengths))
        num_sequences = int(ds.sequence_lengths.shape[0])
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"
        add_extra = 0 if args.no_add_extra_token else 1
        samples_per_epoch = (
            (total_tokens - add_extra) // args.seq_length if total_tokens > add_extra else 0
        )
        rounded = (
            (samples_per_epoch // args.global_batch_size) * args.global_batch_size
            if samples_per_epoch > 0
            else 0
        )
        print(
            "\n".join(
                [
                    f"Data prefix     : {prefix}",
                    f"Index file      : {idx_path}",
                    f"Binary file     : {bin_path}",
                    f"Sequences       : {num_sequences}",
                    f"Total tokens    : {total_tokens}",
                    f"Seq length      : {args.seq_length}",
                    f"Add extra token : {bool(add_extra)}",
                    f"Samples/epoch   : {samples_per_epoch}",
                    f"Global batch    : {args.global_batch_size}",
                    f"Rounded samples : {rounded}",
                    f"Output (--train-samples): {train_samples}",
                ]
            ),
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[verbose] failed to collect details: {e}", file=sys.stderr)

    # Print only the numeric result to stdout
    print(train_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
