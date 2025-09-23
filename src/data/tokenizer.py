from dataclasses import dataclass
import io
import logging
from typing import ClassVar

import chess
import chess.pgn
import numpy as np

from data.tokens import (
    CHESS_MOVES,
    ELO_DIGITS,
    INCREMENTS,
    SECONDS_PER_SIDE,
    TerminationTokens,
)

logger = logging.getLogger(__name__)


class Tokenizer:
    elo_digit_tokens: ClassVar[list[str]] = [f"<elo_digit:{digit}>" for digit in ELO_DIGITS]
    increment_tokens: ClassVar[list[str]] = [f"<increment:{increment}>" for increment in INCREMENTS]
    seconds_per_side_tokens: ClassVar[list[str]] = [
        f"<seconds_per_side:{seconds}>" for seconds in SECONDS_PER_SIDE
    ]
    chess_move_tokens: ClassVar[list[str]] = [f"<move:{move}>" for move in CHESS_MOVES]
    termination_tokens: ClassVar[list[str]] = [token.value for token in TerminationTokens]

    bos_token: ClassVar[str] = "<bos>"
    unk_token: ClassVar[str] = "<unk>"
    special_tokens: ClassVar[list[str]] = [bos_token, unk_token]

    vocab: ClassVar[list[str]] = (
        elo_digit_tokens
        + increment_tokens
        + seconds_per_side_tokens
        + chess_move_tokens
        + termination_tokens
        + special_tokens
    )
    token_to_idx: ClassVar[dict[str, int]] = {t: i for i, t in enumerate(vocab)}
    idx_to_token: ClassVar[dict[int, str]] = {i: t for t, i in token_to_idx.items()}

    @classmethod
    def encode(cls, tokens: list[str]) -> np.ndarray:
        """Convert a list of token strings to token IDs."""
        token_ids = [cls.token_to_idx[token] for token in tokens]
        return np.array(token_ids, dtype=np.uint16)

    @classmethod
    def decode(cls, token_ids: np.ndarray) -> list[str]:
        """Convert token IDs back to token strings."""
        return [cls.idx_to_token[int(token_id)] for token_id in token_ids]


def digitize_elo(elo: int) -> list[str]:
    assert 100 <= elo <= 9999
    elo_digits = list(str(elo))
    if elo < 1000:
        elo_digits = ["0", *elo_digits]
    return elo_digits


def san_movetext_to_uci(movetext: str) -> list[str]:
    # First try robust PGN parsing (handles comments, NAGs, ellipses, etc.)
    pgn_text = '[Event "?"]\n\n' + movetext.strip() + "\n"
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None
    return [m.uci() for m in game.mainline_moves()]


@dataclass(frozen=True)
class ParsedGame:
    seconds_per_side: str
    increment: str
    white_elo: list[str]
    black_elo: list[str]
    uci_moves: list[str]
    normal_termination: bool


def parse_game(game: dict) -> ParsedGame:
    tc = game["TimeControl"]
    if "+" in tc:
        seconds_per_side, increment = tc.split("+", 1)
    else:
        seconds_per_side, increment = "*", "*"  # correspondence game

    if seconds_per_side not in SECONDS_PER_SIDE:
        logger.debug("seconds_per_side %s not in vocabulary.", seconds_per_side)
        seconds_per_side = Tokenizer.unk_token
    if increment not in INCREMENTS:
        logger.debug("increment %s not in vocabulary.", increment)
        increment = Tokenizer.unk_token

    white_elo = digitize_elo(game["WhiteElo"])
    black_elo = digitize_elo(game["BlackElo"])
    uci_moves = san_movetext_to_uci(game["movetext"])

    normal_termination = game["Termination"] == "Normal"
    return ParsedGame(
        seconds_per_side, increment, white_elo, black_elo, uci_moves, normal_termination
    )


def tokenize_game(parsed_game: ParsedGame) -> np.ndarray:
    # 1) BOS token
    tokens = [Tokenizer.bos_token]

    # 2) Time control tokens (seconds per side + increment)
    tokens.append(f"<seconds_per_side:{parsed_game.seconds_per_side}>")
    tokens.append(f"<increment:{parsed_game.increment}>")

    # 3) White elo, black elo - need to format as tokens
    for digit in parsed_game.white_elo:
        tokens.append(f"<elo_digit:{digit}>")
    for digit in parsed_game.black_elo:
        tokens.append(f"<elo_digit:{digit}>")

    # 4) UCI moves - need to format as tokens
    for move in parsed_game.uci_moves:
        tokens.append(f"<move:{move}>")

    # 5) Termination token
    if parsed_game.normal_termination:
        tokens.append(TerminationTokens.NORMAL_TERMINATION.value)
    else:
        tokens.append(TerminationTokens.NOT_NORMAL_TERMINATION.value)

    # Convert string tokens to IDs
    return Tokenizer.encode(tokens)
