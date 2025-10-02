#!/usr/bin/env python
"""Generate predictions using vLLM with prompt_logprobs - one prompt per game."""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from data.tokenizer import Tokenizer, build_game_prompt_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800",
        help="Path or HF Hub ID for the model",
    )
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
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Max number of sequences to process in parallel",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_dir} with vLLM (TP={args.tensor_parallel_size})...")

    move_token_ids = {Tokenizer.token_to_idx[token] for token in Tokenizer.chess_move_tokens}

    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_model_len=1024,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )

    print(f"Loading Allie data from {args.allie_file}...")
    with open(args.allie_file) as f:
        allie_data = json.load(f)

    games = allie_data["games"]
    print(f"Processing {len(games)} games...")

    all_prompts = []
    for game in tqdm(games, desc="Preparing prompts"):
        time_control = game["time-control"]
        seconds_per_side, increment = (
            time_control.split("+", 1) if "+" in time_control else ("*", "*")
        )

        token_ids = build_game_prompt_tokens(
            seconds_per_side,
            increment,
            game["white-elo"],
            game["black-elo"],
            game["human-move"],
        )
        all_prompts.append(TokensPrompt(prompt_token_ids=token_ids))

    print(f"Total prompts: {len(all_prompts):,}")

    logit_bias = dict.fromkeys(range(Tokenizer.vocab_size()), -100.0)
    for token_id in move_token_ids:
        logit_bias[token_id] = 0.0

    sampling_params = SamplingParams(
        prompt_logprobs=5,
        max_tokens=1,
        logit_bias=logit_bias,
    )

    print("Generating with prompt_logprobs...")
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    print("Parsing prompt_logprobs...")
    results = []

    for game_idx, output in enumerate(tqdm(outputs, desc="Parsing")):
        game = games[game_idx]
        human_moves = game["human-move"]
        prompt_logprobs = output.prompt_logprobs

        if prompt_logprobs is None:
            print(f"Warning: No prompt_logprobs for game {game_idx}")
            model_moves = [None] * len(human_moves)
        else:
            model_moves = []
            # prompt_logprobs[i] predicts token at position i
            # Moves start at position 11: <bos> + <seconds> + <increment> + 8 elo digits
            for move_idx in range(len(human_moves)):
                logprob_position = 11 + move_idx

                if (
                    logprob_position >= len(prompt_logprobs)
                    or prompt_logprobs[logprob_position] is None
                ):
                    model_moves.append(None)
                    continue

                best_token_id = max(
                    (
                        (tid, lp.logprob if hasattr(lp, "logprob") else lp)
                        for tid, lp in prompt_logprobs[logprob_position].items()
                        if tid in move_token_ids
                    ),
                    key=lambda x: x[1],
                    default=(None, float("-inf")),
                )[0]

                if best_token_id is not None:
                    token_str = Tokenizer.idx_to_token[best_token_id]
                    model_moves.append(token_str[len("<move:") : -1])
                else:
                    model_moves.append(None)

        results.append(
            {
                "game-id": game["game-id"],
                "time-control": game["time-control"],
                "white-elo": game["white-elo"],
                "black-elo": game["black-elo"],
                "human-moves": human_moves,
                "model-moves": model_moves,
            }
        )

    output_data = {
        "model": args.model_dir,
        "num_games": len(results),
        "predictions": results,
    }

    print(f"Writing results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
