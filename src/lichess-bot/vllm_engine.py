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
        self.target_elo = 1500  # Default target ELO for the bot
        self.game = game
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
            self.target_elo = opponent_elo  # Start at opponent's level

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


        # Build prompt with correct ELO assignment
        moves = [m.uci() for m in board.move_stack]
        seconds_per_side = self.game_info["seconds_per_side"]
        increment = self.game_info["increment"]

        # Determine bot's color and assign ELOs correctly
        if self.game is not None:
            bot_is_white = self.game.is_white
            opponent_elo = self.game.opponent.rating
        else:
            bot_is_white = True  # Default assumption
            opponent_elo = 1500

        # Assign ELOs: bot gets target_elo, opponent gets their actual rating
        if bot_is_white:
            white_elo = self.target_elo
            black_elo = opponent_elo
        else:
            white_elo = opponent_elo
            black_elo = self.target_elo

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
        token_str = result["choices"][0]["text"].strip()
        if token_str == TerminationTokens.NORMAL_TERMINATION.value:
            return PlayResult(None, None, resigned=True)
        
        
        predicted_move = Tokenizer.extract_move_from_move_token(token_str)
        return PlayResult(predicted_move, None)

    def set_elo(self, elo: int) -> None:
        """Set the target ELO for the bot."""
        if 500 <= elo <= 3000:
            self.target_elo = elo
            logger.info(f"Target ELO set to {elo}")
        else:
            logger.warning(f"Invalid ELO {elo}. Must be between 500 and 3000.")

    def get_target_elo(self) -> int:
        """Get the current target ELO."""
        return self.target_elo
