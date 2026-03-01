#!/usr/bin/env python
"""Quick test: have vLLM play 10 games against SF1 to verify quality."""

import sys

import chess
import chess.engine
from vllm import LLM, SamplingParams, TokensPrompt

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T
from src.data.tokens import CHESS_MOVES

STOCKFISH = "/home/yimingz3/bin/stockfish"
MODEL_PATH = "/tmp/export_200M_v3"
MOVE_OFFSET = 378

tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}


def get_legal_tokens(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


def make_prompt(elo_white=2000, elo_black=2000, time_ctrl=180, increment=0):
    prompt = [T.token_to_idx["<bos>"]]
    prompt.append(T.token_to_idx[f"<seconds_per_side:{time_ctrl}>"])
    prompt.append(T.token_to_idx[f"<increment:{increment}>"])
    for d in f"{elo_white:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    for d in f"{elo_black:04d}":
        prompt.append(T.token_to_idx[f"<elo_digit:{d}>"])
    return prompt


print("Loading vLLM model...")
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=0.3,
    max_model_len=1024,
    enforce_eager=True,
)

print("Starting SF engine...")
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
engine.configure({"Skill Level": 1, "Threads": 1})

wins = draws = losses = illegal = 0
N_GAMES = 10

for game_idx in range(N_GAMES):
    board = chess.Board()
    prompt_tokens = make_prompt(elo_white=2000, elo_black=2000)
    game_tokens = list(prompt_tokens)
    model_is_white = game_idx % 2 == 0

    for move_num in range(200):
        if board.is_game_over():
            break

        model_turn = (board.turn == chess.WHITE) == model_is_white

        if model_turn:
            legal = get_legal_tokens(board)
            if not legal:
                break

            params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=legal)
            results = llm.generate(TokensPrompt(prompt_token_ids=game_tokens), params)
            tok = results[0].outputs[0].token_ids[0]

            uci = tok2uci.get(tok)
            if uci is None:
                illegal += 1
                break

            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                illegal += 1
                break
            board.push(move)
            game_tokens.append(tok)
        else:
            result = engine.play(board, chess.engine.Limit(depth=1))
            board.push(result.move)
            uci = result.move.uci()
            if uci in uci2tok:
                game_tokens.append(uci2tok[uci])

    # Score
    result = board.result()
    if result == "1-0":
        if model_is_white:
            wins += 1
        else:
            losses += 1
    elif result == "0-1":
        if model_is_white:
            losses += 1
        else:
            wins += 1
    else:
        draws += 1

    color = "W" if model_is_white else "B"
    print(f"  Game {game_idx + 1}: model={color}, result={result}, moves={board.fullmove_number}")

engine.quit()
print(f"\nResults: {wins}W {draws}D {losses}L / {N_GAMES} games ({illegal} illegal)")
print(f"Win rate: {100 * wins / N_GAMES:.0f}%")
