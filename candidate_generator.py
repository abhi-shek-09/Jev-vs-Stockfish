"""
candidate_generator.py

Responsibility: from the legal moves, pick the "interesting" ones using cheap
heuristics. It does NOT check tactics or safety; that is TacticalFilter's job.

Important: this ordering no longer decides what Jev sees. TacticalFilter now
sorts the surviving moves by its own search score, so the job here is only to
(a) cap the pool when there are a lot of legal moves and (b) give the filter a
sensible order to break ties with.

In the game we analysed, the generator's order WAS what Jev saw, and it
produced things like "Nh3, Nf3, Nc3, Na3, e4, d4" as the six options on move
one, plus a knight bouncing e5-f3-e5-f3. The anti-shuffle and development
terms below attack that directly.
"""

import chess


class CandidateGenerator:
    def __init__(self, max_candidates: int = 12):
        # This is the size of the candidate POOL handed to the tactical
        # filter. Keep it larger than the number of moves you finally show
        # Jev, so the filter has room to throw away blunders. The default in
        # JevPlayer is "every legal move", because the cheap heuristics here
        # are not good enough to be trusted with removing a move outright.
        self.max_candidates = max_candidates

    def generate(self, board: chess.Board) -> list[chess.Move]:
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return []

        scored_moves = [
            (self._score_move(board, move), move) for move in legal_moves
        ]

        # sort() is stable, so equal scores keep python-chess's move order.
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        return [move for _, move in scored_moves[: self.max_candidates]]

    def _score_move(self, board: chess.Board, move: chess.Move) -> int:
        """Cheap "is this move interesting?" score. No lookahead here."""
        score = 0

        if board.gives_check(move):
            score += 100

        if board.is_capture(move):
            score += 50
            captured_piece = self._captured_piece(board, move)
            if captured_piece:
                score += self._piece_value(captured_piece.piece_type)

        if board.is_castling(move):
            # Castling is almost always good and almost never "interesting"
            # by the other rules, so it needs its own bonus to be seen.
            score += 120

        piece = board.piece_at(move.from_square)
        if piece:
            score += self._development_score(board, piece, move)

        if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
            score += 15

        score -= self._shuffle_penalty(board, move)

        return score

    # ------------------------------------------------------------- helpers

    def _development_score(
        self,
        board: chess.Board,
        piece: chess.Piece,
        move: chess.Move,
    ) -> int:
        """Encourage sensible opening play, discourage silly opening play."""
        score = 0
        back_rank = 0 if piece.color == chess.WHITE else 7
        from_rank = chess.square_rank(move.from_square)
        to_file = chess.square_file(move.to_square)

        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += 20
            # A minor piece leaving its starting rank is development.
            if from_rank == back_rank:
                score += 30
            # ...but not to the edge of the board (Na3, Nh3 and friends).
            if to_file in (0, 7):
                score -= 40

        # Dragging the queen out before the minor pieces just loses time.
        if piece.piece_type == chess.QUEEN and board.fullmove_number <= 8:
            score -= 40

        # Moving the king (other than castling, handled above) usually gives
        # up the right to castle.
        if piece.piece_type == chess.KING and not board.is_castling(move):
            score -= 30

        return score

    def _shuffle_penalty(self, board: chess.Board, move: chess.Move) -> int:
        """
        Penalise moving the same piece again, and penalise putting it straight
        back where it came from. `board.move_stack[-2]` is our own previous
        move (the last entry is the opponent's reply to it).
        """
        if len(board.move_stack) < 2:
            return 0

        our_previous = board.move_stack[-2]

        if move.from_square != our_previous.to_square:
            return 0  # a different piece, nothing to complain about

        # Same piece moving twice in a row.
        penalty = 35
        if move.to_square == our_previous.from_square:
            penalty += 70  # and going straight back home: pure time loss
        return penalty

    def _captured_piece(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> chess.Piece | None:
        if not board.is_en_passant(move):
            return board.piece_at(move.to_square)

        direction = -1 if board.turn == chess.WHITE else 1
        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.to_square) + direction,
        )
        return board.piece_at(captured_square)

    def _piece_value(self, piece_type: chess.PieceType) -> int:
        values = {
            chess.PAWN: 10,
            chess.KNIGHT: 30,
            chess.BISHOP: 30,
            chess.ROOK: 50,
            chess.QUEEN: 90,
            chess.KING: 900,
        }
        return values[piece_type]