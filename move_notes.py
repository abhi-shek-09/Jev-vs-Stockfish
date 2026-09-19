"""
move_notes.py

Turns a position and a candidate move into short, plain-English notes.

This exists because Jev used to be shown almost nothing: six SAN strings, a
FEN and an ASCII board. An LLM cannot reliably work out from a FEN that "Nf3
leaves the e4 pawn hanging" - but python-chess can tell it that in one line,
for free. Everything here is description, not decision: we never say which
move to play, only what each move does.

It lives in its own module (rather than inside JevPlayer) so it can be tested
without an API key.
"""

from __future__ import annotations

import chess

from tactical_filter import PIECE_VALUES, MoveAssessment

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


# ------------------------------------------------------------- position info

def material_balance(board: chess.Board) -> int:
    """Centipawns, positive means White is ahead."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def describe_material(board: chess.Board, color: chess.Color) -> str:
    """Material balance written from `color`'s point of view."""
    balance = material_balance(board)
    if color == chess.BLACK:
        balance = -balance
    if balance == 0:
        return "material is level"
    direction = "ahead by" if balance > 0 else "behind by"
    return f"you are {direction} about {abs(balance) / 100:.1f} pawns of material"


def recent_moves(board: chess.Board, plies: int = 8) -> list[str]:
    """The last few half-moves in SAN, oldest first."""
    replay = board.root()
    sans: list[str] = []
    for move in board.move_stack:
        sans.append(replay.san(move))
        replay.push(move)
    return sans[-plies:]


def position_summary(board: chess.Board) -> dict:
    """A dictionary of facts about the position, ready to hand to Jev."""
    color = board.turn
    castling = []
    if board.has_kingside_castling_rights(color):
        castling.append("kingside")
    if board.has_queenside_castling_rights(color):
        castling.append("queenside")

    return {
        "fen": board.fen(),
        "board": str(board),
        "side_to_move": "White" if color == chess.WHITE else "Black",
        "move_number": board.fullmove_number,
        "material": describe_material(board, color),
        "in_check": board.is_check(),
        "your_castling_rights": ", ".join(castling) if castling else "none left",
        "recent_moves": " ".join(recent_moves(board)) or "(none yet)",
        "undefended_pieces": ", ".join(
            f"{PIECE_NAMES[board.piece_at(sq).piece_type]} on {chess.square_name(sq)}"
            for sq in hanging_squares(board, color)
        )
        or "none",
    }


# ------------------------------------------------------------- move notes

def hanging_squares(board: chess.Board, color: chess.Color) -> list[int]:
    """
    Squares holding a piece of `color` that the opponent attacks and that
    nothing of `color` defends. This is a rough check (it ignores pins and
    the value of the attacker), which is fine: it is a hint for Jev, and the
    real tactical work is done by the search in TacticalFilter.
    """
    squares = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not color, square) and not board.is_attacked_by(
            color, square
        ):
            squares.append(square)
    return sorted(squares)


def describe_move(board: chess.Board, assessment: MoveAssessment) -> str:
    """
    One line describing what a candidate move does. The caller's board is
    never modified: we push and pop inside a try/finally.
    """
    move = assessment.move
    mover = board.piece_at(move.from_square)
    us = board.turn
    notes: list[str] = [f"score {assessment.describe_score()}"]

    if board.is_castling(move):
        notes.append("castles, tucking the king away")
    elif mover is not None:
        notes.append(
            f"{PIECE_NAMES[mover.piece_type]} "
            f"{chess.square_name(move.from_square)}"
            f"-{chess.square_name(move.to_square)}"
        )

    if board.is_capture(move):
        if board.is_en_passant(move):
            notes.append("captures a pawn en passant")
        else:
            victim = board.piece_at(move.to_square)
            if victim is not None:
                notes.append(
                    f"captures the {PIECE_NAMES[victim.piece_type]} on "
                    f"{chess.square_name(move.to_square)}"
                )

    if move.promotion:
        notes.append(f"promotes to {PIECE_NAMES[move.promotion]}")

    working = board.copy(stack=False)
    hanging_before = set(hanging_squares(board, us))
    working.push(move)
    try:
        if working.is_checkmate():
            notes.append("CHECKMATE")
        elif working.is_check():
            notes.append("gives check")

        if not board.is_castling(move) and mover is not None:
            landing = chess.square_name(move.to_square)
            attacked = working.is_attacked_by(not us, move.to_square)
            defended = working.is_attacked_by(us, move.to_square)
            if attacked and not defended:
                notes.append(f"the {PIECE_NAMES[mover.piece_type]} on {landing} "
                             "would be attacked and undefended")
            elif attacked:
                notes.append(f"the {PIECE_NAMES[mover.piece_type]} on {landing} "
                             "would be attacked but defended")

        # Anything else we leave loose after the move.
        hanging_after = set(hanging_squares(working, us))
        loose = sorted(sq for sq in hanging_after if sq != move.to_square)
        if loose:
            names = ", ".join(
                f"{PIECE_NAMES[working.piece_at(sq).piece_type]} on "
                f"{chess.square_name(sq)}"
                for sq in loose
            )
            notes.append(f"leaves undefended: {names}")

        # ...and anything it rescues. This is the note that was missing when
        # Jev was shown six moves that all abandoned a hanging pawn.
        rescued = sorted(
            sq
            for sq in hanging_before - hanging_after
            if sq != move.from_square and working.piece_at(sq) is not None
        )
        if rescued:
            names = ", ".join(
                f"{PIECE_NAMES[working.piece_at(sq).piece_type]} on "
                f"{chess.square_name(sq)}"
                for sq in rescued
            )
            notes.append(f"defends: {names}")

        if assessment.opponent_checks == 1:
            notes.append("1 checking reply available")
        elif assessment.opponent_checks:
            notes.append(
                f"{assessment.opponent_checks} checking replies available"
            )
    finally:
        working.pop()

    return "; ".join(notes)