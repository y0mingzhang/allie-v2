#!/usr/bin/env python3
"""vLLM-based MCTS: fast policy via vLLM + value via picotron probe on separate GPU.

This is the production-ready MCTS player that uses KV-cached inference for speed.
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


class VLLMMCTSPlayer:
    """MCTS player using vLLM for policy + picotron for value."""

    def __init__(self, llm, value_model, value_probe, value_device, c_puct=1.5):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.llm = llm
        self.value_model = value_model
        self.value_probe = value_probe
        self.value_device = value_device
        self.c_puct = c_puct
        self.SamplingParams = SamplingParams
        self.TokensPrompt = TokensPrompt

    def get_policy(self, token_ids, board):
        """Get move probabilities from vLLM (fast, KV-cached)."""
        from data.tokenizer import Tokenizer

        prompt = self.TokensPrompt(prompt_token_ids=token_ids)
        params = self.SamplingParams(max_tokens=1, temperature=0, logprobs=20)
        output = self.llm.generate([prompt], params, use_tqdm=False)
        logprobs = output[0].outputs[0].logprobs[0]

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        move_probs = {}
        for move in legal_moves:
            token_str = f"<move:{move.uci()}>"
            if token_str in Tokenizer.token_to_idx:
                tid = Tokenizer.token_to_idx[token_str]
                if tid in logprobs:
                    move_probs[move] = math.exp(logprobs[tid].logprob)
                else:
                    move_probs[move] = 1e-6  # small prior for moves not in top-50

        if not move_probs:
            return {}
        total = sum(move_probs.values())
        return {m: p / total for m, p in move_probs.items()}

    @torch.no_grad()
    def get_value(self, token_ids, board):
        """Get position value from picotron model + linear probe."""
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.value_device).unsqueeze(0)
        x = self.value_model.embedding(input_ids.clamp(max=self.value_model.vocab_size - 1))
        for layer in self.value_model.decoder_layers:
            x = layer(x)
        x = self.value_model.final_norm(x)
        hidden = x[0, -1].float()
        return torch.tanh(self.value_probe(hidden.unsqueeze(0))).item()

    def search(self, board, token_ids, num_simulations=50):
        from data.tokenizer import Tokenizer

        root = MCTSNode()
        policy = self.get_policy(token_ids, board)
        value = self.get_value(token_ids, board)

        for move, prior in policy.items():
            root.children.append(MCTSNode(move=move, parent=root, prior=prior))
        root.visits = 1
        root.value_sum = value

        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()
            sim_tokens = list(token_ids)

            # Selection
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

            # Expansion + evaluation
            if not sim_board.is_game_over() and len(sim_tokens) < 1020:
                policy = self.get_policy(sim_tokens, sim_board)
                value = self.get_value(sim_tokens, sim_board)
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

            # Backprop (white perspective)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        if not root.children:
            return None
        return max(root.children, key=lambda c: c.visits).move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="HF export dir for vLLM")
    parser.add_argument("--value-checkpoint", required=True, help="Picotron checkpoint for value")
    parser.add_argument("--value-config", required=True)
    parser.add_argument("--value-probe", required=True)
    parser.add_argument("--stockfish", default="/home/yimingz3/bin/stockfish")
    parser.add_argument("--prompt-elo", type=int, default=3000)
    parser.add_argument("--sf-skill", type=int, default=1)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--value-gpu", type=int, default=1)
    args = parser.parse_args()

    # Load picotron value model first (before vLLM grabs GPUs)
    with open(args.value_config) as f:
        config = json.load(f)
    from transformers import AutoConfig

    mc = AutoConfig.from_pretrained(config["model"]["name"])
    for attr in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if attr in config["model"]:
            setattr(mc, attr, config["model"][attr])
    mc.vocab_size = config["model"].get("vocab_size", mc.vocab_size)
    mc.max_position_embeddings = config["training"]["seq_length"]

    os.environ["FLASH_ATTEN"] = "0"
    from picotron.model import Qwen3Model

    value_model = Qwen3Model(mc)
    ckpt = torch.load(
        os.path.join(
            args.value_checkpoint, "weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
        ),
        map_location="cpu",
        weights_only=False,
    )
    value_model.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}, strict=False
    )
    value_device = torch.device(f"cuda:{args.value_gpu}")
    value_model = value_model.to(value_device, dtype=torch.bfloat16).eval()

    value_probe = nn.Linear(mc.hidden_size, 1).float().to(value_device)
    value_probe.load_state_dict(torch.load(args.value_probe, map_location=value_device))
    logger.info("Value model + probe loaded on GPU %d", args.value_gpu)

    # Load vLLM for policy (after value model to avoid GPU conflicts)
    from vllm import LLM

    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_num_seqs=1,
        max_model_len=1024,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    logger.info("vLLM loaded on GPU %d", args.policy_gpu)

    player = VLLMMCTSPlayer(llm, value_model, value_probe, value_device)

    # Play games
    from data.tokenizer import build_game_prompt_tokens

    wins = draws = losses = 0
    total_time = 0

    for g in range(args.games):
        board = chess.Board()
        moves_uci = []
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        engine.configure({"Skill Level": args.sf_skill})
        game_start = time.time()

        for _ in range(200):
            if board.is_game_over():
                break
            if board.turn == chess.WHITE:
                tids = build_game_prompt_tokens(
                    "180", "0", args.prompt_elo, args.prompt_elo, moves_uci
                )
                if len(tids) > 1020:
                    break
                move = player.search(board, tids, num_simulations=args.num_simulations)
                if move is None:
                    break
            else:
                move = engine.play(board, chess.engine.Limit(time=0.01)).move
            board.push(move)
            moves_uci.append(move.uci())

        engine.quit()
        game_time = time.time() - game_start
        total_time += game_time

        if board.is_game_over():
            outcome = board.outcome()
            result = (
                "win"
                if outcome.winner == chess.WHITE
                else ("draw" if outcome.winner is None else "loss")
            )
        else:
            result = "draw"

        match result:
            case "win":
                wins += 1
            case "draw":
                draws += 1
            case "loss":
                losses += 1

        score = (wins + 0.5 * draws) / (g + 1)
        logger.info(
            "Game %d: %s (%d moves, %.1fs) | %dW %dD %dL (%.0f%%)",
            g + 1,
            result,
            len(moves_uci),
            game_time,
            wins,
            draws,
            losses,
            score * 100,
        )

    total = wins + draws + losses
    logger.info(
        "Final: %dW %dD %dL (%.1f%%) | Avg %.1fs/game",
        wins,
        draws,
        losses,
        (wins + 0.5 * draws) / total * 100,
        total_time / total,
    )


if __name__ == "__main__":
    main()
