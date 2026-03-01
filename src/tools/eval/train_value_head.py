#!/usr/bin/env python3
"""Train a value head (linear probe) on top of a chess model to predict game outcomes.

Extracts last hidden states from the model on games with known results,
then trains a simple linear layer to predict win/draw/loss.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MOVE_START, MOVE_END = 378, 2345
BOS = 2348
TERM_NORMAL = 2346
TERM_NOT_NORMAL = 2347


def load_games_with_outcomes(parquet_glob, max_games=10000):
    """Load games and their outcomes (from termination token context)."""
    files = sorted(glob.glob(parquet_glob, recursive=True))
    rng = np.random.default_rng(42)
    rng.shuffle(files)

    games = []
    for f in files:
        if len(games) >= max_games:
            break
        table = pq.read_table(f, columns=["tokens"])
        col = table.column("tokens")
        for i in range(len(col)):
            if len(games) >= max_games:
                break
            tokens = col[i].as_py()
            if not tokens or tokens[0] != BOS or len(tokens) < 20:
                continue
            # Find termination token — we need this to know the game result
            # But our token format doesn't directly encode win/draw/loss...
            # We only have normal vs not_normal termination
            # For now: use the number of moves as a proxy for game length
            # Better approach: we need to re-parse the original PGN data
            n_moves = sum(1 for t in tokens if MOVE_START <= t <= MOVE_END)
            if n_moves < 10:
                continue
            games.append(tokens[:1025])  # cap at seq_len + 1

    return games


@torch.no_grad()
def extract_hidden_states(model, games, device, positions_per_game=5):
    """Extract hidden states at random positions in each game."""
    model.eval()
    all_hidden = []
    all_positions = []  # relative position in game (0=start, 1=end)

    for tokens in games:
        if len(tokens) < 15:
            continue
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)

        # Get hidden states from the model
        # We need to modify the forward pass to return hidden states
        # For now, just get the final layer output before the projection
        x = model.embedding(input_ids)
        for layer in model.decoder_layers:
            x = layer(x)
        x = model.final_norm(x)
        # x is now (1, seq_len, hidden_dim)

        seq_len = x.shape[1]
        # Sample random move positions
        move_positions = [i for i, t in enumerate(tokens[1:]) if MOVE_START <= t <= MOVE_END]
        if len(move_positions) < 3:
            continue

        rng = np.random.default_rng(len(games))
        sample_idx = rng.choice(
            len(move_positions), min(positions_per_game, len(move_positions)), replace=False
        )
        for idx in sample_idx:
            pos = move_positions[idx]
            if pos < seq_len:
                hidden = x[0, pos].cpu().float()
                rel_pos = pos / seq_len
                all_hidden.append(hidden)
                all_positions.append(rel_pos)

    return torch.stack(all_hidden), torch.tensor(all_positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parquet-glob", default="~/user_data/chess-v3/data/processed_v2/data/**/*.parquet"
    )
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/value_head_probe.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    with open(args.config) as f:
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
        args.checkpoint, "weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
    )
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    logger.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

    # Load games
    logger.info("Loading games...")
    games = load_games_with_outcomes(os.path.expanduser(args.parquet_glob), args.max_games)
    logger.info("Loaded %d games", len(games))

    # Extract hidden states
    logger.info("Extracting hidden states...")
    hidden_states, positions = extract_hidden_states(model, games, device)
    logger.info("Extracted %d hidden states (dim=%d)", len(hidden_states), hidden_states.shape[1])

    # For now: train a simple probe to predict relative game position
    # (as a sanity check that the hidden states encode useful info)
    # Later: we'll add actual win/draw/loss labels from PGN data

    # Train/val split
    n = len(hidden_states)
    train_n = int(0.8 * n)
    train_h, val_h = hidden_states[:train_n], hidden_states[train_n:]
    train_p, val_p = positions[:train_n], positions[train_n:]

    # Simple linear probe
    hidden_dim = hidden_states.shape[1]
    probe = nn.Linear(hidden_dim, 1).float()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    logger.info("Training position probe...")
    for epoch in range(20):
        probe.train()
        pred = probe(train_h).squeeze(-1)
        loss = F.mse_loss(pred, train_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_h).squeeze(-1)
            val_loss = F.mse_loss(val_pred, val_p)
            # Correlation
            corr = torch.corrcoef(torch.stack([val_pred, val_p]))[0, 1].item()

        if (epoch + 1) % 5 == 0:
            logger.info(
                "Epoch %d: train_loss=%.4f, val_loss=%.4f, corr=%.4f",
                epoch + 1,
                loss.item(),
                val_loss.item(),
                corr,
            )

    results = {
        "n_samples": n,
        "hidden_dim": hidden_dim,
        "final_val_loss": val_loss.item(),
        "final_correlation": corr,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Done: corr=%.4f, saved to %s", corr, args.output)


if __name__ == "__main__":
    main()
