import os

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.tokenizer import parse_game

# Download dataset

ds = load_dataset("yimingzhang/tmp", split="train")

print(f"Dataset size: {ds.num_rows} games")

os.makedirs("analysis", exist_ok=True)

# Plot distribution of event types
print("\nPlotting event type distribution...")


def extract_event_type(game):
    try:
        # Event key is expected in raw record
        event_raw = game.get("Event", "Unknown")
        event_type = " ".join(event_raw.split(" ")[:2]) if isinstance(event_raw, str) else "Unknown"
        return {"event_type": event_type}
    except Exception:
        return {"event_type": "Unknown"}


event_type_data = ds.map(extract_event_type, num_proc=32)

# Convert to pandas Series and count
event_types = [item["event_type"] for item in event_type_data]
event_counts = pd.Series(event_types).value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=event_counts.values, y=event_counts.index, orient="h", palette="viridis")
plt.title("Event Type Distribution")
plt.xlabel("Number of Games")
plt.ylabel("Event Type")
plt.tight_layout()
plt.savefig("analysis/event_type_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
print("Event type distribution saved to: analysis/event_type_distribution.png")


def extract_elos(game):
    try:
        parsed = parse_game(game)
        white_elo = int("".join(parsed.white_elo))
        black_elo = int("".join(parsed.black_elo))
        return {"white_elo": white_elo, "black_elo": black_elo}
    except Exception:
        return {"white_elo": None, "black_elo": None}


# Extract ELO data
print("Extracting ELO ratings...")
elo_data = ds.map(extract_elos, num_proc=32)

# Convert to pandas
elo_list = []
for item in elo_data:
    if item["white_elo"] is not None and item["black_elo"] is not None:
        elo_list.append(item)

df = pd.DataFrame(elo_list)

print(f"\nSuccessfully parsed {len(df)} games")
print(f"White ELO range: {df['white_elo'].min()} - {df['white_elo'].max()}")
print(f"Black ELO range: {df['black_elo'].min()} - {df['black_elo'].max()}")

print("\nWhite ELO statistics:")
print(f"  Mean: {df['white_elo'].mean():.0f}")
print(f"  Median: {df['white_elo'].median():.0f}")
print(f"  Std: {df['white_elo'].std():.0f}")

print("\nBlack ELO statistics:")
print(f"  Mean: {df['black_elo'].mean():.0f}")
print(f"  Median: {df['black_elo'].median():.0f}")
print(f"  Std: {df['black_elo'].std():.0f}")

# Define ELO bins (100x100 grids)
elo_min = 400
elo_max = 3000
bin_size = 100

# Create bins
elo_bins = np.arange(elo_min, elo_max + bin_size, bin_size).tolist()
white_elo_binned = pd.cut(df["white_elo"], bins=elo_bins, right=False)
black_elo_binned = pd.cut(df["black_elo"], bins=elo_bins, right=False)

# Create crosstab (heatmap data)
heatmap_data = pd.crosstab(white_elo_binned, black_elo_binned)

# Create labels for the bins
bin_labels = [f"{int(elo_bins[i])}-{int(elo_bins[i + 1])}" for i in range(len(elo_bins) - 1)]

# Show distribution by ranges
print("\nELO Distribution (White players):")
elo_ranges = [
    (0, 1000),
    (1000, 1200),
    (1200, 1400),
    (1400, 1600),
    (1600, 1800),
    (1800, 2000),
    (2000, 2200),
    (2200, 3000),
]
for low, high in elo_ranges:
    count = len(df[(df["white_elo"] >= low) & (df["white_elo"] < high)])
    pct = count / len(df) * 100
    print(f"  {low}-{high}: {count:,} games ({pct:.1f}%)")

print("\nELO Distribution (Black players):")
for low, high in elo_ranges:
    count = len(df[(df["black_elo"] >= low) & (df["black_elo"] < high)])
    pct = count / len(df) * 100
    print(f"  {low}-{high}: {count:,} games ({pct:.1f}%)")

# Create analysis directory (already ensured earlier)

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=bin_labels,
    yticklabels=bin_labels,
    cbar_kws={"label": "Number of Games"},
)

plt.title("ELO Distribution Heatmap\n(White vs Black Player Ratings)", fontsize=14)
plt.xlabel("Black ELO Rating", fontsize=12)
plt.ylabel("White ELO Rating", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("analysis/elo_distribution_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nHeatmap saved to: analysis/elo_distribution_heatmap.png")
print(f"Total games in heatmap: {heatmap_data.sum().sum():,}")

# Check for anomalous buckets in the 2000-2200 range
print("\nChecking 2000-2200 ELO range buckets:")
for i, _ in enumerate(heatmap_data.index):
    for j, _ in enumerate(heatmap_data.columns):
        count = heatmap_data.iloc[i, j]
        if count > 0:
            white_range = f"{int(elo_bins[i])}-{int(elo_bins[i + 1])}"
            black_range = f"{int(elo_bins[j])}-{int(elo_bins[j + 1])}"
            if ("2000-2100" in white_range or "2100-2200" in white_range) and count > 100:
                print(f"  White {white_range} vs Black {black_range}: {count:,} games")
            if ("2000-2100" in black_range or "2100-2200" in black_range) and count > 100:
                print(f"  White {white_range} vs Black {black_range}: {count:,} games")
