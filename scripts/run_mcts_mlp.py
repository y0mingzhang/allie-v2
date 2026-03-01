#!/usr/bin/env python3
"""Run MCTS with MLP value probe."""

import json, os, logging, torch, torch.nn as nn

os.environ["FLASH_ATTEN"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from transformers import AutoConfig
from picotron.model import Qwen3Model

with open("configs/main_runs_v2/qwen-3-200M-v2.json") as f:
    config = json.load(f)
mc = AutoConfig.from_pretrained(config["model"]["name"])
for a in (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
):
    if a in config["model"]:
        setattr(mc, a, config["model"][a])
mc.vocab_size = 2350
mc.max_position_embeddings = 1024
model = Qwen3Model(mc)
ckpt = torch.load(
    "models/main_runs_v2/qwen-3-200M-v2/25000/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(
    {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}, strict=False
)
model = model.to(device="cuda:0", dtype=torch.bfloat16)
model.eval()

probe = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1)).float().to("cuda:0")
probe.load_state_dict(torch.load("results/value_probe_mlp256_weights.pt", map_location="cuda:0"))

from src.tools.eval.mcts_player import MCTSPlayer, play_game_mcts

player = MCTSPlayer(model, probe, "cuda:0")

wins = draws = losses = 0
for g in range(10):
    game = play_game_mcts(player, 3000, "/home/yimingz3/bin/stockfish", 1, num_simulations=50)
    match game["outcome"]:
        case "win":
            wins += 1
        case "draw":
            draws += 1
        case "loss":
            losses += 1
    score = (wins + 0.5 * draws) / (g + 1)
    logger.info(
        "Game %d: %s (%d moves) | +%d =%d -%d (%.0f%%)",
        g + 1,
        game["outcome"],
        game["n_moves"],
        wins,
        draws,
        losses,
        score * 100,
    )
logger.info(
    "Final: +%d =%d -%d (%.1f%%)",
    wins,
    draws,
    losses,
    (wins + 0.5 * draws) / (wins + draws + losses) * 100,
)
