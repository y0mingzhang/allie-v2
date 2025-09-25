import argparse
import logging
import os
import random

from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
import pyarrow as pa
import pyarrow.parquet as pq

from data.tokenizer import parse_game, tokenize_game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
disable_progress_bar()

keep_ratio = {
    ("Rated Blitz", 400): 1.00000000,
    ("Rated Blitz", 500): 1.00000000,
    ("Rated Blitz", 600): 1.00000000,
    ("Rated Blitz", 700): 0.72235112,
    ("Rated Blitz", 800): 0.30747959,
    ("Rated Blitz", 900): 0.16453221,
    ("Rated Blitz", 1000): 0.10575061,
    ("Rated Blitz", 1100): 0.08115952,
    ("Rated Blitz", 1200): 0.06360318,
    ("Rated Blitz", 1300): 0.05203546,
    ("Rated Blitz", 1400): 0.04565920,
    ("Rated Blitz", 1500): 0.04089603,
    ("Rated Blitz", 1600): 0.03869786,
    ("Rated Blitz", 1700): 0.03841250,
    ("Rated Blitz", 1800): 0.04017861,
    ("Rated Blitz", 1900): 0.04789621,
    ("Rated Blitz", 2000): 0.06477250,
    ("Rated Blitz", 2100): 0.10071058,
    ("Rated Blitz", 2200): 0.18222968,
    ("Rated Blitz", 2300): 0.35476887,
    ("Rated Blitz", 2400): 0.76532284,
    ("Rated Blitz", 2500): 1.00000000,
    ("Rated Blitz", 2600): 1.00000000,
    ("Rated Blitz", 2700): 1.00000000,
    ("Rated Blitz", 2800): 1.00000000,
    ("Rated Blitz", 2900): 1.00000000,
    ("Rated Blitz", 3000): 1.00000000,
    ("Rated Bullet", 400): 1.00000000,
    ("Rated Bullet", 500): 1.00000000,
    ("Rated Bullet", 600): 1.00000000,
    ("Rated Bullet", 700): 0.35023249,
    ("Rated Bullet", 800): 0.13953627,
    ("Rated Bullet", 900): 0.06992065,
    ("Rated Bullet", 1000): 0.04321353,
    ("Rated Bullet", 1100): 0.03173244,
    ("Rated Bullet", 1200): 0.02422802,
    ("Rated Bullet", 1300): 0.01970780,
    ("Rated Bullet", 1400): 0.01694768,
    ("Rated Bullet", 1500): 0.01484727,
    ("Rated Bullet", 1600): 0.01369574,
    ("Rated Bullet", 1700): 0.01344519,
    ("Rated Bullet", 1800): 0.01384097,
    ("Rated Bullet", 1900): 0.01534650,
    ("Rated Bullet", 2000): 0.01851822,
    ("Rated Bullet", 2100): 0.02520809,
    ("Rated Bullet", 2200): 0.03838693,
    ("Rated Bullet", 2300): 0.06063675,
    ("Rated Bullet", 2400): 0.10931648,
    ("Rated Bullet", 2500): 0.21127398,
    ("Rated Bullet", 2600): 0.39336253,
    ("Rated Bullet", 2700): 0.73892405,
    ("Rated Bullet", 2800): 1.00000000,
    ("Rated Bullet", 2900): 1.00000000,
    ("Rated Bullet", 3000): 1.00000000,
    ("Rated Bullet", 3100): 1.00000000,
    ("Rated Bullet", 3200): 1.00000000,
    ("Rated Classical", 600): 1.00000000,
    ("Rated Classical", 700): 1.00000000,
    ("Rated Classical", 800): 1.00000000,
    ("Rated Classical", 900): 1.00000000,
    ("Rated Classical", 1000): 1.00000000,
    ("Rated Classical", 1100): 1.00000000,
    ("Rated Classical", 1200): 1.00000000,
    ("Rated Classical", 1300): 1.00000000,
    ("Rated Classical", 1400): 0.99701110,
    ("Rated Classical", 1500): 0.86754598,
    ("Rated Classical", 1600): 1.00000000,
    ("Rated Classical", 1700): 1.00000000,
    ("Rated Classical", 1800): 1.00000000,
    ("Rated Classical", 1900): 1.00000000,
    ("Rated Classical", 2000): 1.00000000,
    ("Rated Classical", 2100): 1.00000000,
    ("Rated Classical", 2200): 1.00000000,
    ("Rated Classical", 2300): 1.00000000,
    ("Rated Classical", 2400): 1.00000000,
    ("Rated Classical", 2500): 1.00000000,
    ("Rated Classical", 2600): 1.00000000,
    ("Rated Classical", 2700): 1.00000000,
    ("Rated Classical", 2800): 1.00000000,
    ("Rated Correspondence", 800): 1.00000000,
    ("Rated Correspondence", 900): 1.00000000,
    ("Rated Correspondence", 1000): 1.00000000,
    ("Rated Correspondence", 1100): 1.00000000,
    ("Rated Correspondence", 1200): 1.00000000,
    ("Rated Correspondence", 1300): 1.00000000,
    ("Rated Correspondence", 1400): 1.00000000,
    ("Rated Correspondence", 1500): 1.00000000,
    ("Rated Correspondence", 1600): 1.00000000,
    ("Rated Correspondence", 1700): 1.00000000,
    ("Rated Correspondence", 1800): 1.00000000,
    ("Rated Correspondence", 1900): 1.00000000,
    ("Rated Correspondence", 2000): 1.00000000,
    ("Rated Correspondence", 2100): 1.00000000,
    ("Rated Correspondence", 2200): 1.00000000,
    ("Rated Correspondence", 2300): 1.00000000,
    ("Rated Correspondence", 2400): 1.00000000,
    ("Rated Correspondence", 2500): 1.00000000,
    ("Rated Correspondence", 2600): 1.00000000,
    ("Rated Rapid", 400): 1.00000000,
    ("Rated Rapid", 500): 1.00000000,
    ("Rated Rapid", 600): 1.00000000,
    ("Rated Rapid", 700): 1.00000000,
    ("Rated Rapid", 800): 0.80628453,
    ("Rated Rapid", 900): 0.44433873,
    ("Rated Rapid", 1000): 0.28256792,
    ("Rated Rapid", 1100): 0.21764459,
    ("Rated Rapid", 1200): 0.16804908,
    ("Rated Rapid", 1300): 0.13959675,
    ("Rated Rapid", 1400): 0.12149435,
    ("Rated Rapid", 1500): 0.11385523,
    ("Rated Rapid", 1600): 0.11718799,
    ("Rated Rapid", 1700): 0.13019962,
    ("Rated Rapid", 1800): 0.15212717,
    ("Rated Rapid", 1900): 0.21074483,
    ("Rated Rapid", 2000): 0.35586375,
    ("Rated Rapid", 2100): 0.69701493,
    ("Rated Rapid", 2200): 1.00000000,
    ("Rated Rapid", 2300): 1.00000000,
    ("Rated Rapid", 2400): 1.00000000,
    ("Rated Rapid", 2500): 1.00000000,
    ("Rated Rapid", 2600): 1.00000000,
    ("Rated Rapid", 2700): 1.00000000,
    ("Rated Rapid", 2800): 1.00000000,
    ("Rated Rapid", 2900): 1.00000000,
    ("Rated Rapid", 3000): 1.00000000,
    ("Rated UltraBullet", 500): 1.00000000,
    ("Rated UltraBullet", 600): 1.00000000,
    ("Rated UltraBullet", 700): 1.00000000,
    ("Rated UltraBullet", 800): 1.00000000,
    ("Rated UltraBullet", 900): 1.00000000,
    ("Rated UltraBullet", 1000): 1.00000000,
    ("Rated UltraBullet", 1100): 1.00000000,
    ("Rated UltraBullet", 1200): 0.73197492,
    ("Rated UltraBullet", 1300): 0.46403021,
    ("Rated UltraBullet", 1400): 0.33870032,
    ("Rated UltraBullet", 1500): 0.28031212,
    ("Rated UltraBullet", 1600): 0.28934325,
    ("Rated UltraBullet", 1700): 0.34767719,
    ("Rated UltraBullet", 1800): 0.48767753,
    ("Rated UltraBullet", 1900): 0.67446563,
    ("Rated UltraBullet", 2000): 1.00000000,
    ("Rated UltraBullet", 2100): 1.00000000,
    ("Rated UltraBullet", 2200): 1.00000000,
    ("Rated UltraBullet", 2300): 1.00000000,
    ("Rated UltraBullet", 2400): 1.00000000,
    ("Rated UltraBullet", 2500): 1.00000000,
}


def get_elo_bucket(elo: int) -> int:
    """Get the ELO bucket range for a given ELO rating."""
    bucket = elo // 100 * 100
    return bucket


def valid(game: dict):
    for key in ["Event", "WhiteElo", "BlackElo", "TimeControl", "movetext", "Termination"]:
        if game[key] is None:
            return False
    ev = game["Event"]
    format = " ".join(ev.split(" ")[:2])
    return format.startswith("Rated ")


def passes_filter(game: dict) -> bool:
    w = game["WhiteElo"]
    b = game["BlackElo"]
    ev = game["Event"]

    bucket = get_elo_bucket(int((w + b) / 2))
    format = " ".join(ev.split(" ")[:2])
    if not format.startswith("Rated "):
        return False
    return random.random() < keep_ratio.get((format, bucket), 1.0)


def should_keep_game(game: dict) -> bool:
    """Determine if a game should be kept based on bucket weights."""
    return valid(game) and passes_filter(game)


def process_game(game: dict) -> dict:
    """Process a game by parsing and tokenizing it."""
    parsed = parse_game(game)
    tokens = tokenize_game(parsed)
    return {"tokens": tokens}


def _read_parquet_list(list_file: str) -> list[str]:
    with open(list_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    return lines


def _resolve_input_and_relative_path(parquet_arg: str) -> tuple[list[str], str]:
    """Return data_files for load_dataset and relative path for output.

    - If input starts with "hf://" use as-is.
    - If input starts with "datasets/" prefix with "hf://" for reading.
    - Otherwise treat as local file path.

    The returned relative path preserves the path under "datasets/Lichess/" when
    available; otherwise it falls back to the basename of the provided path.
    """
    if parquet_arg.startswith("hf://"):
        hf_path = parquet_arg
        repo_path = parquet_arg[len("hf://") :]
    elif parquet_arg.startswith("datasets/"):
        repo_path = parquet_arg
        hf_path = f"hf://{repo_path}"
    else:
        if not os.path.exists(parquet_arg):
            raise FileNotFoundError(f"Parquet file not found: {parquet_arg}")
        hf_path = parquet_arg
        repo_path = parquet_arg

    prefix = "datasets/Lichess/standard-chess-games/"
    if repo_path.startswith(prefix):
        rel_path = repo_path[len(prefix) :]
    else:
        rel_path = os.path.basename(repo_path)

    return [hf_path], rel_path


def process_shard(
    path: str,
    dataset_root_path: str,
) -> Dataset:
    data_files, rel_path = _resolve_input_and_relative_path(path)

    # Determine output path and short-circuit if a valid processed parquet already exists
    out_path = f"{dataset_root_path}/{rel_path}"
    if not out_path.endswith(".parquet"):
        out_path = f"{out_path}.parquet"

    if _is_processed_parquet_ok(out_path):
        logger.info("Using existing processed shard: %s", out_path)
        return load_dataset("parquet", data_files=[out_path], split="train")

    logger.info("Processing shard: input=%s -> output=%s", path, out_path)
    shard = load_dataset("parquet", data_files=data_files, split="train")
    original_len = len(shard)

    # Filter and process
    num_proc = os.cpu_count()
    shard = shard.filter(should_keep_game, num_proc=num_proc)
    shard = shard.map(
        process_game,
        num_proc=num_proc,
        remove_columns=[c for c in shard.features if c != "Site"],
    )

    # Write as a single parquet using the underlying Arrow table
    table = shard.data.table
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)

    logger.info("Wrote processed shard to %s", out_path)
    logger.info("After filtering: %d games (from %d)", len(shard), original_len)

    return shard


def _is_processed_parquet_ok(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        table = pq.read_table(path)
    except Exception as e:
        logger.warning("Failed to read existing parquet %s: %s", path, e)
        return False
    if table.num_rows == 0:
        logger.warning("Existing parquet %s has 0 rows", path)
        return False
    if "tokens" not in table.column_names:
        logger.warning("Existing parquet %s missing 'tokens' column", path)
        return False
    try:
        tokens_field = table.schema.field("tokens").type
        # Accept list/large_list/fixed_size_list of uint16
        if (
            pa.types.is_list(tokens_field)
            or pa.types.is_large_list(tokens_field)
            or pa.types.is_fixed_size_list(tokens_field)
        ):
            value_type = tokens_field.value_type
        else:
            logger.warning("Parquet %s 'tokens' not a list type: %s", path, tokens_field)
            return False
        if value_type != pa.uint16():
            logger.warning(
                "Parquet %s 'tokens' value type is %s, expected uint16", path, value_type
            )
            return False
    except Exception as e:
        logger.warning("Could not validate schema of %s: %s", path, e)
        return False
    return True


def process_shards(
    paths: list[str],
    dataset_root_path: str,
) -> None:
    """Read a list of parquet paths, filter games, tokenize, and save concatenated tokens.

    Filtering uses the same logic as single-parquet processing in this module.
    """
    for path in paths:
        process_shard(path, dataset_root_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    # List-concat mode
    parser.add_argument(
        "--list-file",
        help="Text file with one parquet path per line",
        default="data/chess_parquets_shuffled.txt",
    )
    parser.add_argument(
        "--start", type=int, help="Start index in the parquet list (inclusive)", required=True
    )
    parser.add_argument(
        "--end", type=int, help="End index in the parquet list (exclusive)", required=True
    )
    parser.add_argument(
        "--dataset-root-path",
        help=("Path to write parquets locally."),
        required=True,
    )

    args = parser.parse_args()
    start, end = args.start, args.end
    all_paths = _read_parquet_list(args.list_file)
    if start < 0 or end < 0 or start >= end or end > len(all_paths):
        raise ValueError(f"Invalid range: start={start}, end={end}, total={len(all_paths)}")
    paths = all_paths[start:end]

    process_shards(
        paths,
        args.dataset_root_path,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
