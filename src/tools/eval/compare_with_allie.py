#!/usr/bin/env python
"""Compare model predictions with Allie's predictions."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allie-file",
        default="data/allie/allie-policy.json",
        help="Path to allie-policy.json",
    )
    parser.add_argument(
        "--model-file",
        default="data/model-predictions.json",
        help="Path to model predictions",
    )
    args = parser.parse_args()

    with open(args.allie_file) as f:
        allie_data = json.load(f)
    with open(args.model_file) as f:
        model_data = json.load(f)

    allie_games = {g["game-id"]: g for g in allie_data["games"]}
    model_games = {g["game-id"]: g for g in model_data["predictions"]}

    assert len(allie_games) == len(model_games), "Game count mismatch"

    stats = {"total": 0, "allie": 0, "model": 0, "both": 0, "neither": 0}

    for game_id, allie_game in allie_games.items():
        model_game = model_games[game_id]
        human_moves = allie_game["human-move"]
        allie_moves = allie_game["model-move"]
        model_moves = model_game["model-moves"]

        assert len(human_moves) == len(allie_moves) == len(model_moves)

        for h, a, m in zip(human_moves, allie_moves, model_moves, strict=False):
            stats["total"] += 1
            a_correct = h == a
            m_correct = h == m

            if a_correct:
                stats["allie"] += 1
            if m_correct:
                stats["model"] += 1
            if a_correct and m_correct:
                stats["both"] += 1
            elif not a_correct and not m_correct:
                stats["neither"] += 1

    total = stats["total"]
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Model: {model_data['model']}")
    print(f"Total games: {len(allie_games)}")
    print(f"Total positions: {total}")
    print()
    print(f"Allie accuracy:      {stats['allie']}/{total} = {100 * stats['allie'] / total:.2f}%")
    print(f"Model accuracy:      {stats['model']}/{total} = {100 * stats['model'] / total:.2f}%")
    print()
    print(f"Both correct:        {stats['both']}/{total} = {100 * stats['both'] / total:.2f}%")
    print(
        f"Both incorrect:      {stats['neither']}/{total} = {100 * stats['neither'] / total:.2f}%"
    )
    print(
        f"Model correct only:  {stats['model'] - stats['both']}/{total} = {100 * (stats['model'] - stats['both']) / total:.2f}%"
    )
    print(
        f"Allie correct only:  {stats['allie'] - stats['both']}/{total} = {100 * (stats['allie'] - stats['both']) / total:.2f}%"
    )
    print()
    print(f"Accuracy difference: {100 * (stats['model'] - stats['allie']) / total:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
