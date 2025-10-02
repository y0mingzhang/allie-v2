"""
VLLM Chess Engine for lichess-bot using HTTP API.
"""
import chess
from chess.engine import PlayResult, Limit
import sys
import os
import logging
import requests
import json

# Add the project root to sys.path so we can import from src
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, project_root)

import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine, MOVE, COMMANDS_TYPE, OPTIONS_TYPE
from typing import Any
import logging
from lib import model, lichess
from data.tokenizer import Tokenizer, build_game_prompt
from data.tokens import TerminationTokens
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
from lib.config import Configuration

logger = logging.getLogger(__name__)


class VLLMEngine(MinimalEngine):
    """Chess engine using VLLM HTTP API for move prediction."""
    
    def __init__(
        self,
        commands: COMMANDS_TYPE,
        options: OPTIONS_TYPE,
        stderr: int | None,
        draw_or_resign: Configuration,
        game: model.Game | None = None,
        name: str | None = None,
        **popen_args: str,
    ) -> None:
        super().__init__(
            commands, options, stderr, draw_or_resign, game, name, **popen_args
        )
        self.api_url = "http://localhost:8000/v1/completions"
        self.move_token_ids = {Tokenizer.token_to_idx[token] for token in Tokenizer.chess_move_tokens}
        if game is None:
            self.game_info = {
                "seconds_per_side": "300",
                "increment": "0",
                "white_elo": 1500,
                "black_elo": 1500,
            }
        else:
            opponent_elo = game.opponent.rating
            self.game_info = {
                "seconds_per_side": str(game.clock_initial.seconds),
                "increment": str(game.clock_increment.seconds),
                "white_elo": opponent_elo,
                "black_elo": opponent_elo,
            }
        
        print(self.game_info)

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if isinstance(root_moves, list):
            legal_moves = [move for move in legal_moves if move in root_moves]
        
        # Create logit bias for legal moves only
        logit_bias = {}
        for i in range(Tokenizer.vocab_size()):
            logit_bias[i] = -100.0
        for move in legal_moves:
            move_token = f"<move:{move.uci()}>"
            token_id = Tokenizer.token_to_idx[move_token]
            logit_bias[token_id] = 0.0
        logit_bias[Tokenizer.token_to_idx[TerminationTokens.NORMAL_TERMINATION.value]] = 0.0
        
        
        # Build prompt
        moves = [m.uci() for m in board.move_stack]
        seconds_per_side = self.game_info["seconds_per_side"]
        increment = self.game_info["increment"]
        white_elo = self.game_info["white_elo"]
        black_elo = self.game_info["black_elo"]
        prompt = build_game_prompt(seconds_per_side, increment, white_elo, black_elo, moves)
        
        # Make HTTP request to VLLM API
        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logit_bias": logit_bias,
        }
        
        response = requests.post(self.api_url, json=payload)
        result = response.json()

        print(prompt)
        print("result", result)
        token_str = result["choices"][0]["text"].strip()
        if token_str == TerminationTokens.NORMAL_TERMINATION.value:
            print("resigned")
            return PlayResult(None, None, resigned=True)
        
        
        predicted_move = Tokenizer.extract_move_from_move_token(token_str)
        logger.info(f"Selected move: {predicted_move.uci()}")
        return PlayResult(predicted_move, None)
