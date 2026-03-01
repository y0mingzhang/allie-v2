#!/bin/bash
# Pack processed parquets into npy for derisk training
set -euo pipefail

PROCESSED_DIR="data/processed_v2"
OUT_DIR="data/tokens_v2/shard_000"

# Create a list of all processed parquets
find "$PROCESSED_DIR" -name "*.parquet" | sort > /tmp/derisk_parquets.txt
COUNT=$(wc -l < /tmp/derisk_parquets.txt)
echo "Found $COUNT processed parquets"

if [ "$COUNT" -eq 0 ]; then
    echo "No processed parquets found!"
    exit 1
fi

source .venv/bin/activate
python src/data/prepare_npy.py \
    --list-file /tmp/derisk_parquets.txt \
    --start 0 \
    --end "$COUNT" \
    --out-dir "$OUT_DIR" \
    --batch-size 1024 \
    --progress
