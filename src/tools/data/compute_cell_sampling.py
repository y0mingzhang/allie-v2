import argparse
from collections import Counter
import os
import random

from datasets import load_dataset
from huggingface_hub import HfFileSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-cell sampling rates for (format, mean_elo_bin)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Target fraction of games to keep, e.g., 0.2 for 20%",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shard sampling seed (default: 42)")
    parser.add_argument(
        "--num-shards", type=int, default=20, help="Number of shards to sample (default: 20)"
    )
    parser.add_argument("--bin", type=int, default=100, help="ELO bin width (default: 100)")
    return parser.parse_args()


def valid(game: dict):
    for key in ["Event", "WhiteElo", "BlackElo", "TimeControl", "movetext", "Termination"]:
        if game[key] is None:
            return False
    ev = game["Event"]
    format = " ".join(ev.split(" ")[:2])
    return format.startswith("Rated ")


def main() -> None:
    args = parse_args()
    assert 0.0 < args.sparsity <= 1.0, "--sparsity must be in (0, 1]"

    # Load shards
    fs = HfFileSystem()
    parquet_paths = fs.glob("datasets/Lichess/standard-chess-games/**/*.parquet")
    chosen = random.Random(args.seed).sample(parquet_paths, args.num_shards)
    data_files = [f"hf://{p}" for p in chosen]

    print(f"Loading {len(data_files)} shards…")
    _ds = load_dataset("parquet", data_files=data_files, split="train")
    total = len(_ds)

    ds = _ds.filter(valid, num_proc=8)

    print(f"Total games loaded: {total}")
    print(f"valid games: {len(ds)}")

    bin = args.bin

    def bin_low(x: float) -> int | None:
        lo = int(x) // bin * bin
        return lo

    def extract_fields(game: dict) -> dict:
        ev = game.get("Event", "")
        fmt = " ".join(ev.split(" ")[:2]) if isinstance(ev, str) and ev else None
        if fmt is None or not fmt.startswith("Rated "):
            print("unknown format", fmt)
            return {"format": None, "mean_elo_bin": None}
        w = game.get("WhiteElo")
        b = game.get("BlackElo")
        try:
            w = int(w)
        except Exception:
            w = None
        try:
            b = int(b)
        except Exception:
            b = None
        if w is None or b is None:
            return {"format": None, "mean_elo_bin": None}
        mean_elo = 0.5 * (w + b)
        return {"format": fmt, "mean_elo_bin": bin_low(mean_elo)}

    print("Mapping dataset to (format, mean_elo_bin)…")
    mapped = ds.map(extract_fields, num_proc=os.cpu_count() or 8)

    # Count per cell
    counts: Counter[tuple[str, int]] = Counter()
    for fmt, b_lo in zip(mapped["format"], mapped["mean_elo_bin"], strict=False):
        if fmt is not None and b_lo is not None:
            counts[(fmt, b_lo)] += 1

    if total == 0:
        raise RuntimeError("No valid (format, mean_elo_bin) rows found.")

    target_keep = round(args.sparsity * total)
    print(f"Total valid games: {total}; target keep: {target_keep} ({args.sparsity:.3f})")

    # Binary search for integer cap C such that sum(min(c_i, C)) ~= target_keep
    # with an additional per-cell cap of 25% for Bullet/HyperBullet formats.
    bullet_cap_fraction = 0.25
    assert target_keep <= total

    lo, hi = 0, max(counts.values())

    def kept(cap: int) -> int:
        total_kept = 0
        for (fmt, _), c in counts.items():
            cell_kept = min(c, cap * bullet_cap_fraction if "Bullet" in fmt else cap)
            total_kept += cell_kept
        return total_kept

    # lower_bound on C where kept(C) >= target_keep
    while lo < hi:
        mid = (lo + hi) // 2
        if kept(mid) >= target_keep:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    # Compute actual kept with this cap (for info)
    actual_keep = kept(cap)
    frac = actual_keep / total if total else 0.0
    print(f"Chosen cap: {cap}; expected kept: {actual_keep} ({frac:.4f})")

    # Convert to per-cell sampling rates r_cell = min(1, cap / c)

    rates_tuple: dict[tuple[str, int], float] = {}
    expected_counts: Counter[tuple[str, int]] = Counter()
    for (fmt, b_lo), c in counts.items():
        cell_kept = min(c, cap * bullet_cap_fraction if "Bullet" in fmt else cap)
        rates_tuple[(fmt, b_lo)] = cell_kept / c
        expected_counts[fmt, b_lo] = cell_kept

    # Plot expected kept counts as a heatmap
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except Exception as e:
        print(f"Skipping heatmap plotting because plotting deps are missing: {e}")
    else:
        formats = sorted({fmt for (fmt, _) in counts})
        bins = sorted({b_lo for (_, b_lo) in counts})
        z = np.full((len(formats), len(bins)), np.nan, dtype=float)
        for i, fmt in enumerate(formats):
            for j, b_lo in enumerate(bins):
                z[i, j] = expected_counts.get((fmt, b_lo), np.nan)

        os.makedirs("analysis", exist_ok=True)
        plt.figure(figsize=(1.0 * len(bins), max(1.0 * len(formats), 1.0)))
        ax = sns.heatmap(
            z,
            mask=np.isnan(z),
            cmap="viridis",
            annot=True,
            fmt=".0f",
            xticklabels=[str(b) for b in bins],
            yticklabels=formats,
            cbar_kws={"label": "Expected kept games"},
        )
        ax.set_xlabel("Mean ELO bin (low bound)")
        ax.set_ylabel("Format")
        ax.set_title(f"Expected kept per cell (sparsity={args.sparsity}, bin={bin})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        out_path = os.path.join("analysis", "format_mean_elo_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved heatmap to {out_path}")
        plt.close()

    items = ",\n".join(
        [f"  ({fmt!r}, {b_lo}): {rate:.8f}" for (fmt, b_lo), rate in sorted(rates_tuple.items())]
    )
    print("cell_sampling = {\n" + items + "\n}")


if __name__ == "__main__":
    main()
