#!/usr/bin/env python3
"""Elo-stratified evaluation: measure move accuracy and bits/move at each elo bracket.

Reads processed parquets, groups by elo, and evaluates a model on each bracket.
Works with picotron checkpoints (loads model directly, no vLLM needed).
"""

import argparse
import glob
import json
import logging
import math
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MOVE_START, MOVE_END = 378, 2345
BOS = 2348


def decode_elo_from_tokens(tokens):
    if len(tokens) < 11:
        return None, None
    if not all(0 <= tokens[i] <= 9 for i in range(3, 11)):
        return None, None
    white_elo = int("".join(str(tokens[i]) for i in range(3, 7)))
    black_elo = int("".join(str(tokens[i]) for i in range(7, 11)))
    return white_elo, black_elo


def load_eval_games(parquet_glob, max_games=10000, seed=42):
    files = sorted(glob.glob(parquet_glob, recursive=True))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    games_by_elo = {}
    total = 0
    for f in files:
        if total >= max_games:
            break
        table = pq.read_table(f, columns=["tokens"])
        col = table.column("tokens")
        for i in range(len(col)):
            if total >= max_games:
                break
            tokens = col[i].as_py()
            if not tokens or tokens[0] != BOS or len(tokens) < 20:
                continue
            w_elo, b_elo = decode_elo_from_tokens(tokens)
            if w_elo is None:
                continue
            avg_elo = (w_elo + b_elo) // 2
            bucket = (avg_elo // 200) * 200
            games_by_elo.setdefault(bucket, []).append(tokens)
            total += 1
    return games_by_elo


@torch.no_grad()
def evaluate_games(model, games, device, max_seq_len=1024):
    model.eval()
    total_correct = total_moves = 0
    total_move_loss = 0.0
    for tokens in games:
        tokens = tokens[: max_seq_len + 1]
        if len(tokens) < 12:
            continue
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze(0)
        preds = logits.argmax(dim=-1)
        move_mask = (targets >= MOVE_START) & (targets <= MOVE_END)
        if not move_mask.any():
            continue
        n_moves = move_mask.sum().item()
        correct = (preds[move_mask] == targets[move_mask]).sum().item()
        loss = F.cross_entropy(logits[move_mask], targets[move_mask], reduction="sum").item()
        total_correct += correct
        total_moves += n_moves
        total_move_loss += loss
    if total_moves == 0:
        return {"move_acc": 0, "bits_per_move": 0, "n_moves": 0, "n_games": 0}
    return {
        "move_acc": total_correct / total_moves,
        "bits_per_move": (total_move_loss / total_moves) / math.log(2),
        "n_moves": total_moves,
        "n_games": len(games),
    }


def load_picotron_model(config_path, checkpoint_path, device):
    """Load a picotron model from config + checkpoint."""
    with open(config_path) as f:
        config = json.load(f)

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config["model"]["name"])
    for attr in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if attr in config["model"]:
            setattr(model_config, attr, config["model"][attr])
    model_config.vocab_size = config["model"].get("vocab_size", model_config.vocab_size)
    model_config.max_position_embeddings = config["training"]["seq_length"]

    os.environ["FLASH_ATTEN"] = "0"
    from picotron.model import Qwen3Model

    model = Qwen3Model(model_config)

    ckpt_file = os.path.join(
        checkpoint_path, "weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
    )
    if not os.path.exists(ckpt_file):
        ckpt_file = checkpoint_path
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_hf_model(model_dir, device):
    """Load a model from HF safetensors format."""
    os.environ["FLASH_ATTEN"] = "0"
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from picotron.model import Qwen3Model

    config = AutoConfig.from_pretrained(model_dir)
    config.vocab_size = 2350
    config.max_position_embeddings = 1024
    model = Qwen3Model(config)

    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    mapped = {}
    for k, v in state_dict.items():
        new_k = (
            k.replace("model.embed_tokens", "embedding")
            .replace("model.layers", "decoder_layers")
            .replace("self_attn", "attention")
            .replace("model.norm", "final_norm")
            .replace("lm_head", "final_proj")
            .replace("o_proj", "out_proj")
        )
        mapped[new_k] = v
    model.load_state_dict(mapped, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Config JSON (for picotron checkpoints)")
    parser.add_argument("--hf", action="store_true", help="Load from HF safetensors format")
    parser.add_argument(
        "--parquet-glob", default="~/user_data/chess-v3/data/processed_v2/data/**/*.parquet"
    )
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--max-games-per-bucket", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/elo_eval.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    if args.hf:
        model = load_hf_model(args.checkpoint, device)
    else:
        model = load_picotron_model(args.config, args.checkpoint, device)

    logger.info("Model loaded: %dM params", sum(p.numel() for p in model.parameters()) // 1_000_000)

    logger.info("Loading eval games from %s...", args.parquet_glob)
    games_by_elo = load_eval_games(os.path.expanduser(args.parquet_glob), max_games=args.max_games)

    for bucket in games_by_elo:
        if len(games_by_elo[bucket]) > args.max_games_per_bucket:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(games_by_elo[bucket]), args.max_games_per_bucket, replace=False
            )
            games_by_elo[bucket] = [games_by_elo[bucket][i] for i in indices]

    results = {}
    print(f"\n{'Elo':>8} {'Games':>6} {'Moves':>7} {'MoveAcc':>8} {'Bits/M':>8}")
    print("-" * 45)
    for bucket in sorted(games_by_elo.keys()):
        games = games_by_elo[bucket]
        if len(games) < 10:
            continue
        metrics = evaluate_games(model, games, device)
        results[str(bucket)] = metrics
        print(
            f"{bucket:>8} {metrics['n_games']:>6} {metrics['n_moves']:>7} {metrics['move_acc']:>7.1%} {metrics['bits_per_move']:>7.2f}"
        )

    all_games = [g for gs in games_by_elo.values() for g in gs]
    overall = evaluate_games(model, all_games, device)
    results["overall"] = overall
    print(
        f"\n{'OVERALL':>8} {overall['n_games']:>6} {overall['n_moves']:>7} {overall['move_acc']:>7.1%} {overall['bits_per_move']:>7.2f}"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
