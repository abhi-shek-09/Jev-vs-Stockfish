"""
demo_players.py

Stand-ins for Jev and Stockfish so the UI can be developed and reviewed with
no TypeSafe key and no Stockfish installed.

The opening is scripted so the interesting animations actually happen, and
early: both sides castle (plies 7 and 10), White captures en passant (ply 15)
and then promotes by capturing a rook (ply 19). Once the script runs out, both
players pick a random legal move, preferring captures, checks and promotions so
the game keeps moving and more promotions turn up.

The demo Jev also produces fake `last_decision` data in exactly the shape the
real JevPlayer produces, so the decision panel can be exercised.
"""

from __future__ import annotations

import random
import time

import chess

# A scripted opening that reaches every special move we want to see.
#   - White castles kingside, Black castles kingside
#   - White plays an en-passant capture
#   - White promotes a pawn
SCRIPT = [
    "e4", "e5",
    "Nf3", "Nc6",
    "Bc4", "Bc5",
    "O-O", "Nf6",
    "d3", "O-O",
    "a4", "d6",
    "a5", "b5",
    "axb6", "Qe7",
    "b7", "Re8",
    "bxa8=Q", "h6",
    "Qxc8", "Rxc8",
    "Bxf7+", "Kh7",
    "Bd5", "Nxd5",
    "exd5", "Nd4",
    "Nxd4", "Bxd4",
    "c3", "Bb6",
]

CAPTURE_BONUS = 40
PROMOTION_BONUS = 60


class DemoPlayer:
    """Plays the script while it lasts, then plays lively random moves."""

    def __init__(self, name: str, seed: int = 0, think_seconds: float = 0.25):
        self.name = name
        self.rng = random.Random(seed)
        self.think_seconds = think_seconds
        self.last_decision = None

    # ------------------------------------------------------------- helpers

    def _scripted_move(self, board: chess.Board):
        """The next scripted move, if it is legal in this position."""
        index = len(board.move_stack)
        if index >= len(SCRIPT):
            return None
        try:
            return board.parse_san(SCRIPT[index])
        except ValueError:
            return None  # the script has drifted; fall back to random

    def _lively_move(self, board: chess.Board) -> chess.Move:
        """A random legal move, but captures and promotions come first."""
        moves = list(board.legal_moves)
        weights = []
        for move in moves:
            weight = 10
            if board.is_capture(move):
                weight += CAPTURE_BONUS
            if move.promotion:
                weight += PROMOTION_BONUS
            if board.gives_check(move):
                weight += 15
            weights.append(weight)
        return self.rng.choices(moves, weights=weights, k=1)[0]

    # --------------------------------------------------------------- moving

    def get_move(self, board: chess.Board) -> chess.Move:
        time.sleep(self.think_seconds)  # pretend to think, so the UI breathes
        move = self._scripted_move(board) or self._lively_move(board)
        print(f"{self.name} (demo) : {board.san(move)}")
        return move


class DemoJev(DemoPlayer):
    """Same as DemoPlayer, plus believable fake decision data."""

    def get_move(self, board: chess.Board) -> chess.Move:
        time.sleep(self.think_seconds)
        chosen = self._scripted_move(board) or self._lively_move(board)

        # Build a plausible set of "candidates" around the chosen move.
        others = [m for m in board.legal_moves if m != chosen]
        self.rng.shuffle(others)
        shown = [chosen] + others[:4]
        dropped = others[4:8]

        candidates = []
        for index, move in enumerate(shown):
            score = 40 - index * 17 - self.rng.randint(0, 9)
            candidates.append(
                {
                    "san": board.san(move),
                    "uci": move.uci(),
                    "score": score,
                    "score_text": f"{score:+d}cp",
                    "kept": True,
                    "reason": "ok",
                    "note": self._fake_note(board, move, score),
                    "chosen": move == chosen,
                }
            )

        rejected = []
        for move in dropped:
            score = -self.rng.randint(120, 900)
            rejected.append(
                {
                    "san": board.san(move),
                    "uci": move.uci(),
                    "score": score,
                    "score_text": f"{score:+d}cp",
                    "kept": False,
                    "reason": (
                        "allows forced checkmate"
                        if score < -700
                        else f"loses material/position vs best ({score:+d} vs +40)"
                    ),
                }
            )

        self.last_decision = {
            "side": "white" if board.turn == chess.WHITE else "black",
            "player": self.name,
            "move_number": board.fullmove_number,
            "fen": board.fen(),
            "chosen": board.san(chosen),
            "candidates": candidates,
            "rejected": rejected,
            "rejected_total": len(rejected),
            "depth": 4,
            "screened_out": max(0, len(list(board.legal_moves)) - 8),
            "demo": True,
        }

        print(f"{self.name} (demo) : {board.san(chosen)}")
        return chosen

    def _fake_note(self, board: chess.Board, move: chess.Move, score: int) -> str:
        piece = board.piece_at(move.from_square)
        bits = [f"score {score:+d}cp"]
        if piece is not None:
            bits.append(
                f"{chess.piece_name(piece.piece_type)} "
                f"{chess.square_name(move.from_square)}"
                f"-{chess.square_name(move.to_square)}"
            )
        if board.is_capture(move):
            bits.append("captures")
        if board.gives_check(move):
            bits.append("gives check")
        if move.promotion:
            bits.append("promotes to queen")
        return "; ".join(bits)


def build_demo_players(seed: int = 7, think_seconds: float = 0.25):
    """The pair of stubs used by `python main.py --ui --demo`."""
    return (
        DemoJev("Jev", seed=seed, think_seconds=think_seconds),
        DemoPlayer("Stockfish", seed=seed + 1, think_seconds=think_seconds),
    )
