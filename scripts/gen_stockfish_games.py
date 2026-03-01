#!/usr/bin/env python3
"""Generate diverse Stockfish self-play games using Boltzmann sampling.

Each move: evaluate all legal moves, convert centipawn scores to probabilities
via softmax with temperature, then sample. This gives diverse high-quality games.
"""

import argparse
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine
import numpy as np

from data.tokenizer import Tokenizer, build_game_prompt_tokens
from data.tokens import TerminationTokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PACK_SIZE = 1025


def boltzmann_move(engine, board, temperature=0.5, depth=8):
    """Pick a move by Boltzmann sampling over centipawn evaluations."""
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 1:
        return legal_moves[0]

    scores = []
    for move in legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        # Score from opponent's perspective, negate for current player
        score = info["score"].relative
        if score.is_mate():
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()
        scores.append(-cp)  # negate because analyse gives score after move (opponent's view)
        board.pop()

    scores = np.array(scores, dtype=np.float64)
    # Softmax with temperature (higher temp = more random)
    scores = scores / (temperature * 100)  # normalize by 100cp
    scores -= scores.max()  # numerical stability
    probs = np.exp(scores)
    probs /= probs.sum()

    idx = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[idx]


def play_one_game(args):
    sf_path, depth, temperature, assigned_elo, game_idx = args
    np.random.seed(game_idx)

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 1})

    board = chess.Board()
    moves = []

    try:
        while not board.is_game_over() and board.fullmove_number < 200:
            move = boltzmann_move(engine, board, temperature=temperature, depth=depth)
            board.push(move)
            moves.append(move.uci())
    except Exception:
        pass
    finally:
        engine.quit()

    if len(moves) < 5:
        return None

    token_ids = build_game_prompt_tokens("10800", "0", assigned_elo, assigned_elo, moves)
    result_str = board.result() if board.is_game_over() else "*"
    normal = result_str in ("1-0", "0-1", "1/2-1/2")
    term = (
        TerminationTokens.NORMAL_TERMINATION if normal else TerminationTokens.NOT_NORMAL_TERMINATION
    )
    token_ids.append(Tokenizer.token_to_idx[term.value])
    return token_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default="/home/yimingz3/bin/stockfish")
    parser.add_argument("--n-games", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=8, help="search depth per move eval")
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Boltzmann temperature (lower=stronger)"
    )
    parser.add_argument("--elo", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [
        (args.stockfish, args.depth, args.temperature, args.elo, i) for i in range(args.n_games)
    ]

    train_buf = []
    train_packs = []
    n_done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for tokens in pool.map(play_one_game, tasks, chunksize=1):
            if tokens is None:
                continue
            n_done += 1
            train_buf.extend(tokens)
            while len(train_buf) >= PACK_SIZE:
                train_packs.append(np.array(train_buf[:PACK_SIZE], dtype=np.uint16))
                del train_buf[:PACK_SIZE]
            if n_done % 100 == 0:
                logger.info(
                    "%d games done, %d packs (%.1fM tokens)",
                    n_done,
                    len(train_packs),
                    len(train_packs) * PACK_SIZE / 1e6,
                )

    train_arr = np.concatenate(train_packs) if train_packs else np.array([], dtype=np.uint16)
    np.save(os.path.join(args.out_dir, "train.npy"), train_arr)
    np.save(os.path.join(args.out_dir, "val.npy"), np.array([], dtype=np.uint16))
    logger.info("Done: %d games, %d tokens saved to %s", n_done, len(train_arr), args.out_dir)


if __name__ == "__main__":
    main()
