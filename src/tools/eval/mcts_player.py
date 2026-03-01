#!/usr/bin/env python3
"""MCTS chess player with vLLM policy + picotron value probe.

Uses vLLM for fast batched policy inference (KV-cached) and a separate
picotron model + linear probe for value estimation.
"""

import argparse
import json
import logging
import math
import os
import time

import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MOVE_START, MOVE_END = 378, 2345


class MCTSNode:
    __slots__ = ["move", "parent", "children", "visits", "value_sum", "prior"]

    def __init__(self, move=None, parent=None, prior=0.0):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class MCTSPlayer:
    def __init__(self, model, value_probe, device, c_puct=1.5, sf_value_engine=None):
        self.model = model
        self.value_probe = value_probe
        self.device = device
        self.c_puct = c_puct
        self.sf_value_engine = sf_value_engine

    @torch.no_grad()
    def get_policy_and_value(self, token_ids, board):
        from data.tokenizer import Tokenizer

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        outputs = self.model(input_ids=input_ids)

        if isinstance(outputs, tuple):
            logits_full, values_full = outputs
            model_value = values_full[0, -1].item()
        else:
            logits_full = outputs
            model_value = None

        if self.sf_value_engine is not None:
            info = self.sf_value_engine.analyse(board, chess.engine.Limit(depth=8))
            score = info["score"].white()
            cp = score.score() if not score.is_mate() else (10000 if score.mate() > 0 else -10000)
            value = max(-1.0, min(1.0, cp / 1000.0))
        elif model_value is not None:
            value = model_value
        elif self.value_probe is not None:
            x = self.model.embedding(input_ids.clamp(max=self.model.vocab_size - 1))
            for layer in self.model.decoder_layers:
                x = layer(x)
            x = self.model.final_norm(x)
            hidden = x[0, -1].float()
            value = torch.tanh(self.value_probe(hidden.unsqueeze(0))).item()
        else:
            value = 0.0

        logits = logits_full[0, -1]
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, value

        move_logits = {}
        for move in legal_moves:
            token_str = f"<move:{move.uci()}>"
            if token_str in Tokenizer.token_to_idx:
                move_logits[move] = logits[Tokenizer.token_to_idx[token_str]].item()

        if not move_logits:
            return {}, value

        max_logit = max(move_logits.values())
        exp_logits = {m: math.exp(l - max_logit) for m, l in move_logits.items()}
        total = sum(exp_logits.values())
        policy = {m: p / total for m, p in exp_logits.items()}
        return policy, value

    def search(self, board, token_ids, num_simulations=50):
        from data.tokenizer import Tokenizer

        root = MCTSNode()
        policy, root_value = self.get_policy_and_value(token_ids, board)
        for move, prior in policy.items():
            root.children.append(MCTSNode(move=move, parent=root, prior=prior))
        root.visits = 1
        root.value_sum = root_value

        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()
            sim_tokens = list(token_ids)

            while node.children and node.visits > 0:
                sign = 1 if sim_board.turn == chess.WHITE else -1
                best_child = max(
                    node.children,
                    key=lambda c: (
                        sign * c.q_value
                        + self.c_puct * c.prior * math.sqrt(c.parent.visits) / (1 + c.visits)
                    ),
                )
                node = best_child
                if node.move is not None:
                    sim_board.push(node.move)
                    token_str = f"<move:{node.move.uci()}>"
                    if token_str in Tokenizer.token_to_idx:
                        sim_tokens.append(Tokenizer.token_to_idx[token_str])

            if not sim_board.is_game_over() and len(sim_tokens) < 1020:
                policy, value = self.get_policy_and_value(sim_tokens, sim_board)
                for move, prior in policy.items():
                    node.children.append(MCTSNode(move=move, parent=node, prior=prior))
            else:
                if sim_board.is_game_over():
                    outcome = sim_board.outcome()
                    value = (
                        1.0
                        if outcome.winner == chess.WHITE
                        else (-1.0 if outcome.winner == chess.BLACK else 0.0)
                    )
                else:
                    value = 0.0

            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        if not root.children:
            return None, {}
        best_child = max(root.children, key=lambda c: c.visits)
        move_stats = {
            c.move.uci(): {"visits": c.visits, "q": c.q_value, "prior": c.prior}
            for c in root.children
            if c.visits > 0
        }
        return best_child.move, move_stats


def play_game_mcts(
    player, prompt_elo, stockfish_path, sf_skill, sf_time=0.01, num_simulations=50, max_moves=200
):
    from data.tokenizer import build_game_prompt_tokens

    board = chess.Board()
    moves_uci = []
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": sf_skill})
    result = {"moves": [], "outcome": None, "n_moves": 0, "search_stats": []}

    for _ in range(max_moves):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            token_ids = build_game_prompt_tokens("180", "0", prompt_elo, prompt_elo, moves_uci)
            if len(token_ids) > 1020:
                break
            move, stats = player.search(board, token_ids, num_simulations=num_simulations)
            if move is None:
                break
            result["search_stats"].append(stats)
        else:
            move = engine.play(board, chess.engine.Limit(time=sf_time)).move
        board.push(move)
        moves_uci.append(move.uci())
        result["moves"].append(move.uci())

    engine.quit()
    result["n_moves"] = len(result["moves"])
    if board.is_game_over():
        outcome = board.outcome()
        result["outcome"] = (
            "win"
            if outcome.winner == chess.WHITE
            else ("draw" if outcome.winner is None else "loss")
        )
    else:
        result["outcome"] = "draw"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--value-probe", default=None)
    parser.add_argument("--sf-value", action="store_true")
    parser.add_argument("--stockfish", default="/home/yimingz3/bin/stockfish")
    parser.add_argument("--prompt-elo", type=int, default=3000)
    parser.add_argument("--sf-skill", type=int, default=1)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/mcts_eval.json")
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
    if hasattr(config["model"], "value_head"):
        model_config.value_head = config["model"]["value_head"]

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

    sf_value_engine = None
    value_probe = None
    if args.sf_value:
        sf_value_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        logger.info("Using Stockfish oracle value")
    elif args.value_probe:
        hidden_dim = model_config.hidden_size
        value_probe = nn.Linear(hidden_dim, 1).float().to(device)
        value_probe.load_state_dict(torch.load(args.value_probe, map_location=device))
        logger.info("Loaded value probe")

    logger.info(
        "Loaded model (%dM params)", sum(p.numel() for p in model.parameters()) // 1_000_000
    )
    player = MCTSPlayer(model, value_probe, device, sf_value_engine=sf_value_engine)

    wins = draws = losses = 0
    for g in range(args.games):
        game = play_game_mcts(
            player,
            args.prompt_elo,
            args.stockfish,
            args.sf_skill,
            num_simulations=args.num_simulations,
        )
        match game["outcome"]:
            case "win":
                wins += 1
            case "draw":
                draws += 1
            case "loss":
                losses += 1
        score = (wins + 0.5 * draws) / (g + 1)
        logger.info(
            "Game %d: %s (%d moves) | Running: +%d =%d -%d (%.0f%%)",
            g + 1,
            game["outcome"],
            game["n_moves"],
            wins,
            draws,
            losses,
            score * 100,
        )

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total * 100
    logger.info(
        "Final: +%d =%d -%d (%.1f%%) with %d simulations/move",
        wins,
        draws,
        losses,
        score,
        args.num_simulations,
    )

    results = {
        "prompt_elo": args.prompt_elo,
        "sf_skill": args.sf_skill,
        "num_simulations": args.num_simulations,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if sf_value_engine:
        sf_value_engine.quit()


if __name__ == "__main__":
    main()
