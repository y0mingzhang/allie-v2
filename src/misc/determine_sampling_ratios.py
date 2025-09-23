from collections import Counter
import random

from datasets import load_dataset
from huggingface_hub import HfFileSystem

# Initialize the HfFileSystem
fs = HfFileSystem()

parquet_files = fs.glob("datasets/Lichess/standard-chess-games/**/*.parquet")

data_files = [f"hf://{p}" for p in random.Random(42).sample(parquet_files, 10)]
ds = load_dataset("parquet", data_files=data_files, split="train")

format_counts = Counter(" ".join(s.split(" ")[:2]) for s in ds["Event"] if s.startswith("Rated"))

format_cap = 50000
format_sampling = {}

for k, v in format_counts.items():
    fs = format_cap / v if v > format_cap else 1.0
    if "Bullet" in k:
        fs *= 0.1
    if "HyperBullet" in k:
        fs *= 0.1
    if "Blitz" in k:
        fs *= 0.75
    if fs < 1.0:
        format_sampling[k] = fs
print("***format_sampling***", format_sampling, sep="\n")


def filter_by_format(game: dict):
    format = " ".join(game["Event"].split(" ")[:2])
    return format.startswith("Rated ") and random.random() < format_sampling.get(format, 1.0)


ds_filtered = ds.filter(filter_by_format, num_proc=32)

format_counts = Counter(" ".join(s.split(" ")[:2]) for s in ds_filtered["Event"])


def get_elo_bucket(elo: int) -> int:
    """Get the ELO bucket range for a given ELO rating."""
    bucket = elo // 100 * 100
    return bucket


elo_cap = 200
bucket_counts = Counter(
    (get_elo_bucket(w), get_elo_bucket(b))
    for w, b in zip(ds_filtered["WhiteElo"], ds_filtered["BlackElo"], strict=False)
)
bucket_sampling = {}
for k, v in bucket_counts.items():
    if v > elo_cap:
        bucket_sampling[k] = elo_cap / v

print("***bucket_sampling***", bucket_sampling, sep="\n")


def filter_by_elo(game: dict):
    white_bucket = get_elo_bucket(game["WhiteElo"])
    black_bucket = get_elo_bucket(game["BlackElo"])

    weight = bucket_sampling.get((white_bucket, black_bucket), 1.0)
    return random.random() < weight


ds_filtered = ds_filtered.filter(filter_by_elo, num_proc=32)

bucket_counts = Counter(
    (get_elo_bucket(w), get_elo_bucket(b))
    for w, b in zip(ds_filtered["WhiteElo"], ds_filtered["BlackElo"], strict=False)
)

print("filtering ratio", f"{len(ds)}/{len(ds_filtered)} = {len(ds) / len(ds_filtered)}")
