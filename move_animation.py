"""
move_animation.py

Turns one chess move into an explicit list of piece movements, so the browser
never needs to know a single chess rule.

The UI just receives "this piece slides from e1 to g1, that piece slides from
h1 to f1, and the piece on d5 disappears" and plays it. Castling, en passant
and promotion are all handled here, in Python, where python-chess already
knows the answers.

Everything is computed BEFORE the move is played, from a copy of the board.
The caller's board is never touched.
"""

from __future__ import annotations

import chess

# Piece letters follow FEN: uppercase = White, lowercase = Black.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def piece_symbol(piece: chess.Piece) -> str:
    """'P' for a white pawn, 'p' for a black one."""
    return piece.symbol()


def describe_move(board: chess.Board, move: chess.Move) -> dict:
    """
    Describe `move` on `board` as raw piece movements.

    Returns a dictionary shaped like this:

        {
          "uci": "e1g1",
          "movements": [
             {"piece": "K", "from": "e1", "to": "g1"},
             {"piece": "R", "from": "h1", "to": "f1"},   # castling rook
          ],
          "captured": {"square": "d5", "piece": "p"},     # or None
          "promotion": "Q",                               # or None
        }

    The board is NOT modified.
    """
    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"No piece on {chess.square_name(move.from_square)}")

    movements = [
        {
            "piece": piece_symbol(mover),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
        }
    ]

    captured = None

    if board.is_en_passant(move):
        # The captured pawn is NOT on the destination square: it is on the
        # square the capturing pawn passes over.
        direction = -8 if mover.color == chess.WHITE else 8
        victim_square = move.to_square + direction
        victim = board.piece_at(victim_square)
        if victim is not None:
            captured = {
                "square": chess.square_name(victim_square),
                "piece": piece_symbol(victim),
            }
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim is not None:
            captured = {
                "square": chess.square_name(move.to_square),
                "piece": piece_symbol(victim),
            }

    if board.is_castling(move):
        # python-chess encodes castling as "king moves two squares" (or, in
        # Chess960, king takes own rook). Work out where the rook goes.
        king_file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.from_square)
        if king_file > chess.square_file(move.from_square):
            rook_from = chess.square(7, rank)   # h-file
            rook_to = chess.square(5, rank)     # f-file
        else:
            rook_from = chess.square(0, rank)   # a-file
            rook_to = chess.square(3, rank)     # d-file
        rook = board.piece_at(rook_from)
        if rook is not None:
            movements.append(
                {
                    "piece": piece_symbol(rook),
                    "from": chess.square_name(rook_from),
                    "to": chess.square_name(rook_to),
                }
            )

    promotion = None
    if move.promotion:
        promoted = chess.Piece(move.promotion, mover.color)
        promotion = piece_symbol(promoted)

    return {
        "uci": move.uci(),
        "movements": movements,
        "captured": captured,
        "promotion": promotion,
    }


def captured_lists(board: chess.Board) -> dict:
    """
    Which pieces each side has lost, worked out by comparing the position with
    a full starting set. Returned as FEN letters, best piece first.
    """
    start_counts = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    result = {"white": [], "black": []}
    for color, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for piece_type, expected in start_counts.items():
            missing = expected - len(board.pieces(piece_type, color))
            for _ in range(max(0, missing)):
                result[key].append(chess.Piece(piece_type, color).symbol())
        # Heaviest first reads better in a tray.
        result[key].sort(
            key=lambda symbol: PIECE_VALUES[
                chess.Piece.from_symbol(symbol).piece_type
            ],
            reverse=True,
        )
    return result


def move_list_san(board: chess.Board) -> list[str]:
    """The whole game so far in SAN, replayed from the starting position."""
    replay = board.root()
    sans = []
    for move in board.move_stack:
        sans.append(replay.san(move))
        replay.push(move)
    return sans
