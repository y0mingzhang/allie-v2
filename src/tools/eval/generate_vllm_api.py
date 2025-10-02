#!/usr/bin/env python
"""Generate predictions using vLLM HTTP API - one prompt per game."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

import sys
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, project_root)

from data.tokenizer import Tokenizer, build_game_prompt_tokens


def process_game(game, api_url, move_token_ids):
    """Process a single game and return result."""
    time_control = game["time-control"]
    seconds_per_side, increment = (
        time_control.split("+", 1) if "+" in time_control else ("*", "*")
    )

    human_moves = game["human-move"]

    token_ids = build_game_prompt_tokens(
        seconds_per_side,
        increment,
        game["white-elo"],
        game["black-elo"],
        human_moves,
    )

    logit_bias = {i: -100.0 for i in range(Tokenizer.vocab_size())}
    for token_id in move_token_ids:
        logit_bias[token_id] = 0.0

    payload = {
        "prompt": token_ids,
        "max_tokens": 1,
        "echo": True,
        "logprobs": 5,
        "logit_bias": logit_bias,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        result = response.json()

        prompt_logprobs = result["choices"][0]["logprobs"]["top_logprobs"]

        model_moves = []
        for move_idx in range(len(human_moves)):
            logprob_position = 11 + move_idx

            if logprob_position >= len(prompt_logprobs) or not prompt_logprobs[logprob_position]:
                model_moves.append(None)
                continue

            best_token = max(
                (
                    (token, lp)
                    for token, lp in prompt_logprobs[logprob_position].items()
                    if token.startswith("<move:") and token.endswith(">")
                ),
                key=lambda x: x[1],
                default=(None, float("-inf")),
            )[0]

            if best_token is not None:
                model_moves.append(best_token[len("<move:") : -1])
            else:
                model_moves.append(None)
    except Exception as e:
        print(f"Error processing game {game['game-id']}: {e}")
        model_moves = [None] * len(human_moves)

    return {
        "game-id": game["game-id"],
        "time-control": game["time-control"],
        "white-elo": game["white-elo"],
        "black-elo": game["black-elo"],
        "human-moves": human_moves,
        "model-moves": model_moves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allie-file",
        default="data/allie/allie-policy.json",
        help="Path to allie-policy.json",
    )
    parser.add_argument(
        "--output-file",
        default="data/model-predictions.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--model-name",
        default="yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800",
        help="Model name for output metadata",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of games to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    api_url = f"http://localhost:{args.port}/v1/completions"
    move_token_ids = {Tokenizer.token_to_idx[token] for token in Tokenizer.chess_move_tokens}

    print(f"Loading Allie data from {args.allie_file}...")
    with open(args.allie_file) as f:
        allie_data = json.load(f)

    games = allie_data["games"]
    if args.limit:
        games = games[:args.limit]
    print(f"Processing {len(games)} games with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_game, game, api_url, move_token_ids) for game in games]
        results = [future.result() for future in tqdm(futures, desc="Generating predictions")]

    output_data = {
        "model": args.model_name,
        "num_games": len(results),
        "predictions": results,
    }

    print(f"Writing results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
