#!/usr/bin/env python3
"""Train a value probe on a chess model's residual stream to predict game outcomes.

Reads raw HF parquets (which have Result field), tokenizes, extracts hidden states,
and trains a linear probe to predict W/D/L at each position.
"""

import argparse
import json
import logging
import math
import os
import random

import chess
import chess.pgn
import io
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MOVE_START, MOVE_END = 378, 2345
BOS = 2348


def load_games_with_results(parquet_glob, max_games=5000):
    """Load tokenized games with game results. Uses 'result' column if available."""
    import glob

    RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

    files = sorted(glob.glob(parquet_glob, recursive=True))
    rng = random.Random(42)
    rng.shuffle(files)

    games = []

    for f in files:
        if len(games) >= max_games:
            break
        # Try to read with result column (v3 parquets)
        table = pq.read_table(f)
        has_result = "result" in table.column_names
        tokens_col = table.column("tokens")
        result_col = table.column("result") if has_result else None

        for i in range(len(tokens_col)):
            if len(games) >= max_games:
                break
            tokens = tokens_col[i].as_py()
            if not tokens or tokens[0] != BOS or len(tokens) < 20:
                continue

            if has_result:
                result_str = result_col[i].as_py()
                result = RESULT_MAP.get(result_str)
                if result is None:
                    continue
            else:
                # Fallback: infer from moves (old behavior)
                from data.tokenizer import Tokenizer

                board = chess.Board()
                valid = True
                for t in tokens:
                    if MOVE_START <= t <= MOVE_END:
                        uci = Tokenizer.idx_to_token[t][6:-1]
                        try:
                            board.push_uci(uci)
                        except Exception:
                            valid = False
                            break

                if not valid:
                    continue
                if board.is_checkmate():
                    result = -1.0 if board.turn == chess.WHITE else 1.0
                elif board.is_stalemate() or board.is_insufficient_material():
                    result = 0.0
                else:
                    result = 0.0

            games.append((tokens[:1025], result))

    return games


@torch.no_grad()
def extract_hidden_states_with_labels(model, games, device, positions_per_game=5):
    """Extract hidden states and game-outcome labels at move positions."""
    model.eval()
    all_hidden = []
    all_labels = []

    for tokens, result in games:
        if len(tokens) < 15:
            continue
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)

        # Forward pass to get hidden states (before final projection)
        x = model.embedding(input_ids)
        for layer in model.decoder_layers:
            x = layer(x)
        x = model.final_norm(x)

        seq_len = x.shape[1]
        move_positions = [i for i, t in enumerate(tokens[1:]) if MOVE_START <= t <= MOVE_END]
        if len(move_positions) < 3:
            continue

        rng = np.random.default_rng(hash(tuple(tokens[:10])) % 2**32)
        n_sample = min(positions_per_game, len(move_positions))
        sample_idx = rng.choice(len(move_positions), n_sample, replace=False)

        for idx in sample_idx:
            pos = move_positions[idx]
            if pos < seq_len:
                hidden = x[0, pos].cpu().float()
                # Label: game result from white's perspective
                # Adjust by whose turn it is at this position
                # If it's black's turn (odd move number), flip the sign
                is_white_turn = idx % 2 == 0
                label = result if is_white_turn else -result
                all_hidden.append(hidden)
                all_labels.append(label)

    return torch.stack(all_hidden), torch.tensor(all_labels, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parquet-glob", default="~/user_data/chess-v3/data/processed_v2/data/**/*.parquet"
    )
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/value_probe.json")
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

    logger.info("Loading games with results...")
    games = load_games_with_results(os.path.expanduser(args.parquet_glob), args.max_games)
    logger.info("Loaded %d games", len(games))

    # Stats on results
    results = [g[1] for g in games]
    logger.info(
        "Results: W=%.1f%%, D=%.1f%%, L=%.1f%%",
        sum(1 for r in results if r > 0) / len(results) * 100,
        sum(1 for r in results if r == 0) / len(results) * 100,
        sum(1 for r in results if r < 0) / len(results) * 100,
    )

    logger.info("Extracting hidden states...")
    hidden_states, labels = extract_hidden_states_with_labels(
        model, games, device, positions_per_game=8
    )
    logger.info("Extracted %d samples (dim=%d)", len(hidden_states), hidden_states.shape[1])

    # Train/val split
    n = len(hidden_states)
    perm = torch.randperm(n)
    hidden_states = hidden_states[perm]
    labels = labels[perm]
    train_n = int(0.8 * n)
    train_h, val_h = hidden_states[:train_n], hidden_states[train_n:]
    train_l, val_l = labels[:train_n], labels[train_n:]

    # Train linear probe: predict game result (regression: -1 to 1)
    hidden_dim = hidden_states.shape[1]
    probe = nn.Linear(hidden_dim, 1).float()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    logger.info("Training value probe (predict game outcome)...")
    for epoch in range(50):
        probe.train()
        pred = probe(train_h).squeeze(-1)
        loss = F.mse_loss(pred, train_l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_pred = probe(val_h).squeeze(-1)
                val_loss = F.mse_loss(val_pred, val_l)
                corr = torch.corrcoef(torch.stack([val_pred, val_l]))[0, 1].item()
                # Accuracy: sign match
                sign_acc = ((val_pred > 0) == (val_l > 0)).float().mean().item()
            logger.info(
                "Epoch %d: train=%.4f, val=%.4f, corr=%.4f, sign_acc=%.1f%%",
                epoch + 1,
                loss.item(),
                val_loss.item(),
                corr,
                sign_acc * 100,
            )

    # Save probe weights
    probe_path = os.path.splitext(args.output)[0] + "_weights.pt"
    torch.save(probe.state_dict(), probe_path)

    final_results = {
        "n_samples": n,
        "hidden_dim": hidden_dim,
        "final_val_loss": val_loss.item(),
        "final_correlation": corr,
        "final_sign_accuracy": sign_acc,
        "probe_weights_path": probe_path,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Done: corr=%.4f, sign_acc=%.1f%%, saved to %s", corr, sign_acc * 100, args.output)


if __name__ == "__main__":
    main()
