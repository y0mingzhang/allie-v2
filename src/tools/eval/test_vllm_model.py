#!/usr/bin/env python
"""Self-test for vLLM model loading and inference correctness.

Verifies that the model produces sensible chess moves for known positions.
"""

from __future__ import annotations

import argparse
import sys

import chess
from vllm import LLM, SamplingParams

from data.tokenizer import Tokenizer, build_game_prompt_tokens


TEST_CASES = [
    {
        "name": "Opening response to 1.e4",
        "moves": ["e2e4"],
        "elo": 1800,
        "expected_top5": {"e7e5", "c7c5", "e7e6", "c7c6", "d7d6", "d7d5", "g8f6"},
    },
    {
        "name": "Opening response to 1.d4",
        "moves": ["d2d4"],
        "elo": 1800,
        "expected_top5": {"d7d5", "g8f6", "e7e6", "d7d6", "c7c5", "f7f5"},
    },
    {
        "name": "Sicilian: White's 2nd move after 1.e4 c5",
        "moves": ["e2e4", "c7c5"],
        "elo": 1800,
        "expected_top5": {"g1f3", "b1c3", "c2c3", "d2d4", "f2f4"},
    },
    {
        "name": "Italian Game setup",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
        "elo": 1800,
        "expected_top5": {"f1b5", "f1c4", "d2d4", "b1c3", "f1e2"},
    },
    {
        "name": "Empty board - White's first move",
        "moves": [],
        "elo": 1500,
        "expected_top5": {"e2e4", "d2d4", "g1f3", "c2c4", "e2e3", "b1c3"},
    },
]


def get_legal_moves(moves: list[str]) -> set[str]:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return {m.uci() for m in board.legal_moves}


def run_test(llm: LLM, case: dict, sampling_params: SamplingParams) -> tuple[bool, str]:
    moves = case["moves"]
    elo = case["elo"]
    expected = case["expected_top5"]

    legal_moves = get_legal_moves(moves)
    logit_bias = {i: -100.0 for i in range(Tokenizer.vocab_size())}
    for move in legal_moves:
        token = f"<move:{move}>"
        if token in Tokenizer.token_to_idx:
            logit_bias[Tokenizer.token_to_idx[token]] = 0.0

    token_ids = build_game_prompt_tokens("180", "0", elo, elo, moves)
    outputs = llm.generate(
        [{"prompt_token_ids": token_ids}],
        SamplingParams(max_tokens=1, temperature=0.0, logit_bias=logit_bias),
    )

    output_token_ids = outputs[0].outputs[0].token_ids
    if not output_token_ids:
        return False, "No output token"

    output_token = Tokenizer.idx_to_token.get(output_token_ids[0], "")
    if not output_token.startswith("<move:"):
        return False, f"Invalid output token: {output_token}"

    predicted_move = output_token[6:-1]
    if predicted_move not in legal_moves:
        return False, f"Illegal move predicted: {predicted_move}"

    if predicted_move in expected:
        return True, f"OK: {predicted_move}"
    else:
        return False, f"Unexpected move: {predicted_move} (expected one of {expected})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Fail on unexpected moves")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=256,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    passed, failed = 0, 0
    for case in TEST_CASES:
        success, msg = run_test(llm, case, sampling_params)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {case['name']}: {msg}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed}/{len(TEST_CASES)} passed")
    if args.strict and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
