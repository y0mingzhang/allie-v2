#!/usr/bin/env python3
"""Play games against Stockfish at various skill levels to evaluate strength conditioning.

Loads a model checkpoint, plays N games at each (prompt_elo, stockfish_level) combination.
Measures win/draw/loss rates to assess:
1. Does elo conditioning work? (1200-prompted should play worse than 2400-prompted)
2. What's the actual playing strength at high elo?
"""

import argparse
import json
import logging
import os
import time

import chess
import chess.engine
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MOVE_START, MOVE_END = 378, 2345
BOS = 2348


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
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
        checkpoint_path, "weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
    )
    if not os.path.exists(ckpt_file):
        ckpt_file = checkpoint_path
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


@torch.no_grad()
def get_model_move(model, token_ids, board, device):
    """Get the model's best legal move given current token sequence.

    Returns (move, n_illegal) where n_illegal counts how many of the model's
    top predictions were illegal moves (before finding a legal one).
    """
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids=input_ids).squeeze(0)[-1]  # last position logits

    from data.tokenizer import Tokenizer

    legal_moves = set(m.uci() for m in board.legal_moves)
    if not legal_moves:
        return None, 0

    # Get top predictions and count illegal ones
    move_logits = {}
    for tid in range(378, 2346):
        token_str = Tokenizer.idx_to_token.get(tid, "")
        if token_str.startswith("<move:"):
            uci = token_str[6:-1]
            move_logits[uci] = logits[tid].item()

    sorted_moves = sorted(move_logits.items(), key=lambda x: -x[1])
    n_illegal = 0
    for uci, _ in sorted_moves:
        if uci in legal_moves:
            return chess.Move.from_uci(uci), n_illegal
        n_illegal += 1

    return None, n_illegal


def play_game(
    model,
    device,
    prompt_elo,
    stockfish_path,
    sf_skill,
    sf_time=0.01,
    time_control="180+0",
    max_moves=200,
):
    """Play one game: model (white) vs Stockfish (black)."""
    from data.tokenizer import Tokenizer, build_game_prompt_tokens

    seconds, increment = time_control.split("+")
    board = chess.Board()
    moves_uci = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": sf_skill})

    result = {
        "moves": [],
        "outcome": None,
        "reason": None,
        "n_moves": 0,
        "total_illegal_attempts": 0,
        "model_cpl": [],
    }

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            # Model's turn
            token_ids = build_game_prompt_tokens(
                seconds, increment, prompt_elo, prompt_elo, moves_uci
            )
            if len(token_ids) > 1020:
                break

            # Get centipawn eval BEFORE model moves (for CPL tracking)
            try:
                pre_info = engine.analyse(board, chess.engine.Limit(time=0.01))
                pre_score = pre_info["score"].white()
                pre_cp = (
                    pre_score.score()
                    if not pre_score.is_mate()
                    else (10000 if pre_score.mate() > 0 else -10000)
                )
            except Exception:
                pre_cp = None

            move, n_illegal = get_model_move(model, token_ids, board, device)
            result["total_illegal_attempts"] += n_illegal
            if move is None:
                result["reason"] = "no_legal_move"
                break

            board.push(move)

            # Get eval AFTER model moves
            if pre_cp is not None:
                try:
                    post_info = engine.analyse(board, chess.engine.Limit(time=0.01))
                    post_score = post_info["score"].white()
                    post_cp = (
                        post_score.score()
                        if not post_score.is_mate()
                        else (10000 if post_score.mate() > 0 else -10000)
                    )
                    cpl = max(0, pre_cp - post_cp)  # centipawn loss
                    result["model_cpl"].append(cpl)
                except Exception:
                    pass
        else:
            sf_result = engine.play(board, chess.engine.Limit(time=sf_time))
            move = sf_result.move
            board.push(move)

        moves_uci.append(board.peek().uci())
        result["moves"].append(board.peek().uci())

    engine.quit()

    result["n_moves"] = len(result["moves"])
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            result["outcome"] = "draw"
        elif outcome.winner == chess.WHITE:
            result["outcome"] = "win"  # model wins
        else:
            result["outcome"] = "loss"
        result["reason"] = outcome.termination.name
    elif result["reason"] is None:
        result["reason"] = "max_moves"
        result["outcome"] = "draw"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--games-per-combo", type=int, default=20)
    parser.add_argument("--prompt-elos", type=str, default="800,1200,1600,2000,2400,2800")
    parser.add_argument("--sf-skills", type=str, default="1,5,10,15,20")
    parser.add_argument("--sf-time", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/stockfish_eval.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model = load_model(args.config, args.checkpoint, device)

    prompt_elos = [int(x) for x in args.prompt_elos.split(",")]
    sf_skills = [int(x) for x in args.sf_skills.split(",")]

    results = {}
    for prompt_elo in prompt_elos:
        for sf_skill in sf_skills:
            combo = f"elo{prompt_elo}_sf{sf_skill}"
            wins = draws = losses = 0
            games = []

            for g in range(args.games_per_combo):
                game = play_game(
                    model, device, prompt_elo, args.stockfish, sf_skill, sf_time=args.sf_time
                )
                games.append(game)
                match game["outcome"]:
                    case "win":
                        wins += 1
                    case "draw":
                        draws += 1
                    case "loss":
                        losses += 1

            total = wins + draws + losses
            score = (wins + 0.5 * draws) / total if total > 0 else 0
            all_cpl = [c for g in games for c in g.get("model_cpl", [])]
            avg_cpl = sum(all_cpl) / len(all_cpl) if all_cpl else 0
            total_illegal = sum(g.get("total_illegal_attempts", 0) for g in games)
            total_model_moves = sum(len(g.get("model_cpl", [])) for g in games)
            illegal_rate = total_illegal / max(1, total_model_moves)
            results[combo] = {
                "prompt_elo": prompt_elo,
                "sf_skill": sf_skill,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": score,
                "avg_cpl": avg_cpl,
                "illegal_rate": illegal_rate,
                "games": games,
            }
            logger.info(
                "elo=%d sf_skill=%d: +%d =%d -%d (score=%.1f%% cpl=%.0f illegal=%.1f%%)",
                prompt_elo,
                sf_skill,
                wins,
                draws,
                losses,
                score * 100,
                avg_cpl,
                illegal_rate * 100,
            )

    # Summary table
    print(f"\n{'':>12}", end="")
    for sf in sf_skills:
        print(f" SF{sf:>3}", end="")
    print()
    for elo in prompt_elos:
        print(f"Elo {elo:>5}:", end="")
        for sf in sf_skills:
            combo = f"elo{elo}_sf{sf}"
            r = results.get(combo, {})
            print(f" {r.get('score', 0) * 100:>4.0f}%", end="")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
