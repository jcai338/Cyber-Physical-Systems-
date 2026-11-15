<<<<<<< HEAD
"""Core winner-determination logic for a Rock-Paper-Scissors round."""

from typing import Literal

Move = Literal["rock", "paper", "scissors"]
Winner = Literal["player", "opponent", "tie"]

_VALID_MOVES = {"rock", "paper", "scissors"}
_BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def determine_winner(player_move: str, opponent_move: str) -> Winner:
    """Return the round winner; move names are case-insensitive.

    Raises:
        ValueError: If either move is not rock, paper, or scissors.
    """
    moves = (player_move, opponent_move)
    normalized_moves = []

    for move in moves:
        if not isinstance(move, str):
            raise ValueError("Moves must be rock, paper, or scissors.")

        normalized = move.strip().casefold()
        if normalized not in _VALID_MOVES:
            raise ValueError(
                f"Invalid move {move!r}; choose rock, paper, or scissors."
            )
        normalized_moves.append(normalized)

    player, opponent = normalized_moves
    if player == opponent:
        return "tie"
    if _BEATS[player] == opponent:
        return "player"
    return "opponent"
=======

>>>>>>> 739c31c99c940b0188cdb49ffee4a478bf96fcdf
