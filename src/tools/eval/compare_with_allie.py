#!/usr/bin/env python
"""Compare model predictions with Allie's predictions."""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np


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
    parser.add_argument(
        "--no-default-figure",
        action="store_true",
        help="Disable saving a default PNG figure to data/analysis/{model}/move_matching.png",
    )
    parser.add_argument(
        "--png-scale",
        type=float,
        default=4.0,
        help="Scale factor for PNG export (via vl-convert). Higher = higher resolution.",
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

    def parse_time_control(tc: str) -> tuple[int, int]:
        try:
            total_time, increment = map(int, tc.split("+"))
            return total_time, increment
        except Exception:
            raise Exception(f"Failed to parse time control {tc}") from None

    def clock_before_move(time_control: str, moves_seconds: list[int]) -> list[int]:
        base_clock, increment = parse_time_control(time_control)
        clocks: list[int] = [base_clock, base_clock]
        for i, move_time in enumerate(moves_seconds[:-2]):
            clocks.append(clocks[i] + increment - move_time)
        return clocks

    def compute_trim_slice(allie_game: dict) -> tuple[int, int]:
        left_index = 10
        right_index = len(allie_game["human-move"])  # default: keep till end
        clocks = clock_before_move(allie_game["time-control"], allie_game["moves-seconds"])
        for idx, c in enumerate(clocks):
            if c < 30:
                right_index = idx
                break
        return left_index, right_index

    # For optional figure: collect per-rating accuracy for both systems
    per_method_buckets: dict[str, list[list[bool]]] = {}
    allie_label = "Allie-Policy"
    model_label = model_data.get("model", "Model")
    per_method_buckets[allie_label] = [[] for _ in range(35)]
    per_method_buckets[model_label] = [[] for _ in range(35)]

    for game_id, allie_game in allie_games.items():
        model_game = model_games[game_id]
        # Apply the same filtering used in move-matching-allie.ipynb
        left, right = compute_trim_slice(allie_game)
        human_moves = allie_game["human-move"][left:right]
        allie_moves = allie_game["model-move"][left:right]
        model_moves = model_game["model-moves"][left:right]

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

        # Figure data accumulation
        try:
            mean_elo_bucket = round((allie_game["black-elo"] + allie_game["white-elo"]) / 200)
            # Bound buckets to available range
            if 0 <= mean_elo_bucket < 35:
                allie_acc = [hm == am for hm, am in zip(human_moves, allie_moves, strict=False)]
                model_acc = [hm == mm for hm, mm in zip(human_moves, model_moves, strict=False)]
                per_method_buckets[allie_label][mean_elo_bucket].append(allie_acc)
                per_method_buckets[model_label][mean_elo_bucket].append(model_acc)
        except Exception:
            # If rating data is missing, skip figure accumulation for this game
            pass

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

    # Optionally render Altair figure (HTML) matching the notebook styling
    # Default PNG output to data/analysis/{model_name}/move_matching.png
    if not args.no_default_figure:
        try:
            import altair as alt
            import pandas as pd

            data_rows = []
            for method, buckets in per_method_buckets.items():
                for mean_elo, bucket in enumerate(buckets):
                    if not bucket:
                        continue
                    accuracies = np.concatenate([np.array(b, dtype=bool) for b in bucket])
                    data_rows.append(
                        {
                            "Method": method,
                            "Rating": mean_elo * 100,
                            "Accuracy": float(accuracies.mean()),
                            "Opacity": 1.0,
                        }
                    )

            if data_rows:
                palette = {
                    allie_label: "#f58518",
                    model_label: "#54a24b",
                }
                plot_df = pd.DataFrame(data_rows)
                plot = (
                    alt.Chart(plot_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Rating:O", title="Game rating"),
                        y=alt.Y(
                            "Accuracy:Q",
                            title="Move-matching accuracy",
                            axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0.33, 0.62]),
                        ),
                        color=alt.Color(
                            "Method",
                            title="System",
                            sort=[allie_label, model_label],
                            scale=alt.Scale(
                                domain=list(palette.keys()),
                                range=list(palette.values()),
                            ),
                        ),
                        opacity=alt.Opacity("Opacity", legend=None),
                    )
                    .properties(height=240, width=400)
                )

                def sanitize(name: str) -> str:
                    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
                    return safe or "model"

                model_dir_name = sanitize(model_label)
                out_dir = os.path.join("analysis", model_dir_name)
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "move_matching.png")
                try:
                    plot.save(png_path, scale_factor=args.png_scale)
                    print(f"Saved figure PNG to {png_path}")
                except Exception as save_err:
                    print(f"Failed to save PNG figure: {save_err}")
            else:
                print("No data available to render default figure.")
        except Exception as e:
            print(f"Failed to render/save default figure: {e}")


if __name__ == "__main__":
    main()
