#!/usr/bin/env python
"""Sample the next-move distribution from a Hugging Face-exported chess model.

The script builds a token sequence matching the preprocessing pipeline used in
``src/data/process_hf.py`` (time control tokens, Elo digits, move tokens) and
computes the probability distribution over the next move after the supplied
partial movelist. By default it analyses the model's predictions for the first
reply to ``e2e4`` at blitz strength.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from importlib import import_module
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

tokenizer_module = import_module("src.data.tokenizer")
Tokenizer = tokenizer_module.Tokenizer
INCREMENTS = tokenizer_module.INCREMENTS
SECONDS_PER_SIDE = tokenizer_module.SECONDS_PER_SIDE
digitize_elo = tokenizer_module.digitize_elo


def _normalize_time_control(value: str, valid_values: Iterable[str]) -> str:
    if value in valid_values:
        return value
    if value == "*":
        return Tokenizer.unk_token
    raise ValueError(f"Value '{value}' not present in tokenizer vocabulary.")


def build_prompt_tokens(
    seconds_per_side: str,
    increment: str,
    white_elo: int,
    black_elo: int,
    moves: list[str],
) -> torch.Tensor:
    tokens: list[str] = [Tokenizer.bos_token]

    seconds_per_side = _normalize_time_control(seconds_per_side, SECONDS_PER_SIDE)
    increment = _normalize_time_control(increment, INCREMENTS)

    tokens.append(f"<seconds_per_side:{seconds_per_side}>")
    tokens.append(f"<increment:{increment}>")

    for digit in digitize_elo(white_elo):
        tokens.append(f"<elo_digit:{digit}>")
    for digit in digitize_elo(black_elo):
        tokens.append(f"<elo_digit:{digit}>")

    for move in moves:
        move_token = f"<move:{move}>"
        if move_token not in Tokenizer.token_to_idx:
            raise ValueError(f"Move '{move}' is not part of the tokenizer vocabulary.")
        tokens.append(move_token)

    token_ids = Tokenizer.encode(tokens)
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def load_model(model_dir: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    if torch_dtype != torch.float32:
        model = model.to(torch.float32)
    model.eval()
    return model


def compute_next_move_distribution(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids).logits
    next_logits = logits[0, -1]
    if temperature != 1.0:
        next_logits = next_logits / temperature
    move_indices = torch.tensor(
        [Tokenizer.token_to_idx[token] for token in Tokenizer.chess_move_tokens],
        dtype=torch.long,
    )
    move_logits = next_logits.index_select(0, move_indices)
    move_probs = F.softmax(move_logits, dim=-1)
    return move_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="exports/tiny-qwen-3-1.7b-muon-1b",
        help="Path to the Hugging Face-exported model directory",
    )
    parser.add_argument("--white-elo", type=int, default=1800)
    parser.add_argument("--black-elo", type=int, default=1800)
    parser.add_argument("--seconds-per-side", default="180")
    parser.add_argument("--increment", default="0")
    parser.add_argument(
        "--moves",
        nargs="*",
        default=["e2e4"],
        help="List of SAN moves in UCI format that have already been played",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Dtype to load the model before casting to float32 for CPU inference",
    )

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    input_ids = build_prompt_tokens(
        args.seconds_per_side,
        args.increment,
        args.white_elo,
        args.black_elo,
        args.moves,
    )

    model = load_model(args.model_dir, torch_dtype=dtype)
    move_probs = compute_next_move_distribution(model, input_ids, args.temperature)

    move_tokens = Tokenizer.chess_move_tokens
    topk = min(args.top_k, move_probs.numel())
    probs, indices = torch.topk(move_probs, k=topk)

    print(f"Next-move distribution after {' '.join(args.moves)}")
    for rank, (prob, idx) in enumerate(zip(probs.tolist(), indices.tolist(), strict=False), start=1):
        token = move_tokens[idx]
        move = token[len("<move:") : -1]
        print(f"{rank:2d}. {move:5s} -> {prob:.4%}")


if __name__ == "__main__":
    main()
