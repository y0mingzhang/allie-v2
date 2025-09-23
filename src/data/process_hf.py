import argparse
import logging
import os
import random

from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data.tokenizer import parse_game, tokenize_game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
disable_progress_bar()


format_sampling = {
    "Rated Bullet": 0.005925258785677465,
    "Rated Rapid": 0.1667055646317474,
    "Rated UltraBullet": 0.1,
    "Rated Blitz": 0.03162848756789582,
}
elo_sampling = {
    (2000, 2000): 0.06818956699624958,
    (1400, 1500): 0.052534804307853955,
    (1800, 1700): 0.060808756460930376,
    (1400, 1400): 0.01954461057363432,
    (1300, 1400): 0.06325110689437065,
    (1100, 1100): 0.04265301770100235,
    (1200, 1200): 0.028429282160625444,
    (1700, 1600): 0.055509297807382736,
    (1500, 1500): 0.01508523155830442,
    (1600, 1600): 0.018195050946142648,
    (1900, 1800): 0.08247422680412371,
    (1400, 1300): 0.062421972534332085,
    (1300, 1300): 0.0221483942414175,
    (1500, 1600): 0.05023863350916855,
    (1300, 1100): 0.39215686274509803,
    (1700, 1900): 0.27548209366391185,
    (1900, 1900): 0.039401103230890466,
    (1800, 1800): 0.026263952724885097,
    (1700, 1700): 0.02008233758409479,
    (1800, 1900): 0.08123476848090982,
    (900, 900): 0.14781966001478197,
    (800, 800): 0.37453183520599254,
    (1700, 1800): 0.06207324643078833,
    (1100, 1200): 0.09328358208955224,
    (1200, 1000): 0.4819277108433735,
    (2000, 1900): 0.12048192771084337,
    (1400, 1200): 0.31446540880503143,
    (1200, 1300): 0.07385524372230429,
    (1000, 1000): 0.06997900629811056,
    (1100, 1000): 0.12338062924120913,
    (1100, 1300): 0.42283298097251587,
    (1800, 1600): 0.2484472049689441,
    (1200, 1600): 0.8333333333333334,
    (1700, 1500): 0.2,
    (1600, 1700): 0.05496015388843089,
    (900, 1000): 0.18066847335140018,
    (1200, 1100): 0.0974184120798831,
    (2000, 2100): 0.20161290322580644,
    (2200, 2200): 0.29027576197387517,
    (1000, 900): 0.19102196752626552,
    (1500, 1400): 0.05012531328320802,
    (2100, 2000): 0.20040080160320642,
    (800, 900): 0.4,
    (900, 1100): 0.6172839506172839,
    (1900, 2000): 0.12578616352201258,
    (1600, 1500): 0.05003752814610958,
    (2000, 1700): 0.6666666666666666,
    (1300, 1600): 0.3968253968253968,
    (2100, 2200): 0.37523452157598497,
    (2100, 2100): 0.1313197636244255,
    (1900, 1700): 0.2635046113306983,
    (1300, 1200): 0.07541478129713423,
    (2200, 2100): 0.37105751391465674,
    (1600, 1800): 0.24242424242424243,
    (2300, 2300): 0.6644518272425249,
    (1500, 1300): 0.19821605550049554,
    (1400, 1600): 0.23121387283236994,
    (1500, 1800): 0.398406374501992,
    (1300, 1500): 0.21367521367521367,
    (1400, 1800): 0.9009009009009009,
    (2000, 2200): 0.8264462809917356,
    (1000, 1100): 0.12586532410320955,
    (1900, 1600): 0.554016620498615,
    (1400, 1100): 0.8438818565400844,
    (2000, 1800): 0.37243947858473,
    (1800, 1500): 0.37243947858473,
    (1500, 1200): 0.4175365344467641,
    (900, 800): 0.4048582995951417,
    (1800, 2000): 0.352112676056338,
    (1500, 1100): 0.8733624454148472,
    (2100, 1900): 0.547945205479452,
    (1100, 1400): 0.8264462809917356,
    (1900, 2100): 0.49875311720698257,
    (1200, 1500): 0.4750593824228028,
    (1600, 1200): 0.8403361344537815,
    (2200, 2300): 0.8097165991902834,
    (1700, 2000): 0.7246376811594203,
    (1400, 1700): 0.4819277108433735,
    (1200, 1400): 0.3412969283276451,
    (1600, 1400): 0.22598870056497175,
    (1500, 1700): 0.19120458891013384,
    (2200, 2000): 0.9259259259259259,
    (1100, 1500): 0.9090909090909091,
    (1600, 1900): 0.5830903790087464,
    (1000, 800): 0.9259259259259259,
    (1500, 1900): 0.7936507936507936,
    (1300, 1000): 0.9852216748768473,
    (1900, 1500): 0.8771929824561403,
    (2300, 2200): 0.823045267489712,
    (1700, 1400): 0.45662100456621,
    (1100, 900): 0.6172839506172839,
    (1700, 1300): 0.7905138339920948,
    (1600, 1300): 0.35398230088495575,
    (1800, 1400): 0.8928571428571429,
    (1300, 1700): 0.8620689655172413,
    (1000, 1300): 0.91324200913242,
    (1000, 1200): 0.4878048780487805,
}


def get_elo_bucket(elo: int) -> int:
    """Get the ELO bucket range for a given ELO rating."""
    bucket = elo // 100 * 100
    return bucket


def valid(game: dict):
    for key in ["Event", "WhiteElo", "BlackElo", "TimeControl", "movetext", "Termination"]:
        if game[key] is None:
            return False
    return True


def filter_by_format(game: dict):
    format = " ".join(game["Event"].split(" ")[:2])
    return format.startswith("Rated ") and random.random() < format_sampling.get(format, 1.0)


def filter_by_elo(game: dict):
    white_bucket = get_elo_bucket(game["WhiteElo"])
    black_bucket = get_elo_bucket(game["BlackElo"])
    return random.random() < elo_sampling.get((white_bucket, black_bucket), 1.0)


def should_keep_game(game: dict) -> bool:
    """Determine if a game should be kept based on bucket weights."""
    return valid(game) and filter_by_format(game) and filter_by_elo(game)


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
        if pa.types.is_list(tokens_field) or pa.types.is_large_list(tokens_field) or pa.types.is_fixed_size_list(tokens_field):
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
    token_path: str,
    dataset_root_path: str,
) -> None:
    """Read a list of parquet paths, filter games, tokenize, and save concatenated tokens.

    Filtering uses the same logic as single-parquet processing in this module.
    """

    shards = concatenate_datasets(
        [process_shard(path, dataset_root_path) for path in paths]
    ).shuffle()
    # Flatten Arrow list column to a single Arrow array, then convert to NumPy
    arr = pc.list_flatten(shards.data.table.column("tokens"))
    tokens = np.asarray(arr.to_numpy(), dtype=np.uint16)

    np.save(token_path, tokens)
    logger.info("Got %d tokens from %d shards, saved to %s", len(tokens), len(shards), token_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    # List-concat mode
    parser.add_argument(
        "--list-file", help="Text file with one parquet path per line", required=True
    )
    parser.add_argument(
        "--start", type=int, help="Start index in the parquet list (inclusive)", required=True
    )
    parser.add_argument(
        "--end", type=int, help="End index in the parquet list (exclusive)", required=True
    )
    parser.add_argument(
        "--token-path",
        help=("Path to write concatenated token ids as NumPy .npy (list mode) locally."),
        required=True,
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
        args.token_path,
        args.dataset_root_path,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
