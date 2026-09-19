"""
tactical_filter.py

A small, shallow tactical safety layer for ChessArena.

What it does
------------
For every candidate move it:
  1. plays the move on a COPY of the board,
  2. searches a few plies ahead (minimax / negamax with alpha-beta),
  3. scores the result (checkmate, material AND simple positional terms),
  4. rejects moves that are obvious blunders,
  5. returns the survivors sorted best-first, each with a score.

What it does NOT do
-------------------
* It does not choose Jev's move. It removes moves that are clearly bad and
  ranks the rest, so that Jev chooses among sensible options.
* It does not use Stockfish (or any engine). Only python-chess.
* It never returns an empty list if it was given candidates.

Scores
------
All scores are from the point of view of the side to move at that node
(negamax convention): positive = good for the side to move.
Units are "centipawns" (100 = one pawn). Checkmate is a huge number
(MATE_SCORE) minus the ply where it happens, so a faster mate scores higher
and a slower loss scores better than a faster loss.

Why the evaluation is not material-only any more
------------------------------------------------
With a material-only evaluation every quiet move scores exactly the same
(+0cp in the opening). That has two bad effects:
  * the filter cannot tell a developing move from a pointless one, so the
    order in which moves are offered ends up being decided by something
    else entirely, and
  * the "reject anything more than N centipawns worse than the best" rule
    has nothing to bite on, so it either lets everything through or (with a
    small margin) collapses to a single candidate the moment material moves.
Adding cheap positional terms (piece-square tables, mobility, pawn
structure, a king shield) gives the score some resolution between quiet
moves, which is what both the filter and Jev need.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import chess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants

MATE_SCORE = 100_000
# Any score with absolute value above this means "a checkmate was found".
MATE_BOUND = MATE_SCORE - 1_000
INFINITY = MATE_SCORE + 1

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # kings are never captured, so they don't count
}

# --------------------------------------------------------- piece-square tables
#
# Each table has 64 numbers written the way a chess board is normally drawn:
# the FIRST row is rank 8 (black's back rank) and the LAST row is rank 1.
# The numbers are bonuses in centipawns for a WHITE piece standing there.
#
# Looking a value up:
#   white piece on square sq  ->  table[chess.square_mirror(sq)]
#   black piece on square sq  ->  table[sq]
# (python-chess numbers a1 = 0, so mirroring flips the ranks for us.)

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

# In the middlegame the king wants to hide in a corner behind its pawns.
KING_MIDDLEGAME_TABLE = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

# In the endgame the king is a fighting piece and wants to be central.
KING_ENDGAME_TABLE = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}


def _for_white(table: list[int]) -> list[int]:
    """
    Rewrite a table so it can be indexed straight with a python-chess square
    number (a1 = 0) for a WHITE piece. Doing the flip once, here, keeps it out
    of the evaluation function, which runs tens of thousands of times a turn.
    """
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


# Black reads the tables exactly as they are written above; White reads the
# flipped copies.
WHITE_TABLES = {
    piece_type: _for_white(table)
    for piece_type, table in PIECE_SQUARE_TABLES.items()
}
BLACK_TABLES = PIECE_SQUARE_TABLES

WHITE_KING_MIDDLEGAME = _for_white(KING_MIDDLEGAME_TABLE)
WHITE_KING_ENDGAME = _for_white(KING_ENDGAME_TABLE)

# Small extra evaluation terms.
TEMPO_BONUS = 10            # having the move is worth a little
BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 15
ISOLATED_PAWN_PENALTY = 12
KING_SHIELD_BONUS = 12      # per file by the king that still has a pawn
# How much a piece is worth per square it can move to. Knights are already
# handled well by their table, so only the long-range pieces get this.
MOBILITY_WEIGHTS = {chess.BISHOP: 3, chess.ROOK: 2, chess.QUEEN: 1}

# Non-pawn material on the board at the start of a game, both sides added up.
# Used to decide how "endgame-y" the position is (0 = bare kings, 256 = start).
OPENING_NON_PAWN_MATERIAL = 2 * (2 * 320 + 2 * 330 + 2 * 500 + 900)


# ------------------------------------------------------------- data classes

@dataclass
class MoveAssessment:
    """The result of analysing one candidate move."""

    move: chess.Move
    san: str
    score: int                 # from the mover's point of view
    opponent_checks: int       # how many checking replies the opponent has
    depth: int = 0             # how many plies this score is based on
    reason: str = ""           # filled in by TacticalFilter.filter()

    @property
    def loses_to_mate(self) -> bool:
        return self.score <= -MATE_BOUND

    @property
    def delivers_mate(self) -> bool:
        return self.score >= MATE_BOUND

    def describe_score(self) -> str:
        """Human readable score for logs and for Jev's prompt."""
        if abs(self.score) >= MATE_BOUND:
            plies = MATE_SCORE - abs(self.score)
            moves = (plies + 1) // 2
            who = "we mate" if self.score > 0 else "we get mated"
            return f"{who} in {moves}"
        return f"{self.score:+d}cp"


@dataclass
class FilterResult:
    """What TacticalFilter.filter() returns."""

    # Surviving moves, BEST FIRST (this used to be "same order as the input").
    safe_moves: list[chess.Move]
    # The same surviving moves with their scores attached, same order.
    safe_assessments: list[MoveAssessment] = field(default_factory=list)
    rejected: list[MoveAssessment] = field(default_factory=list)
    assessments: list[MoveAssessment] = field(default_factory=list)
    # True when EVERY candidate loses to a forced mate inside the search
    # horizon. In that case safe_moves holds the "least bad" moves and the
    # caller may want to widen the search to all legal moves.
    all_lose_to_mate: bool = False
    # How deep the search actually got before the time budget ran out.
    depth_reached: int = 0
    # Moves dropped by the cheap pre-screen that were never searched deeply.
    screened_out: list[chess.Move] = field(default_factory=list)


# ------------------------------------------------------------ the filter

class TacticalFilter:
    def __init__(
        self,
        search_depth: int = 6,
        min_depth: int = 4,
        quiescence_depth: int = 3,
        material_margin: int = 60,
        screen_depth: int = 2,
        shortlist_size: int = 8,
        time_budget: float = 5.0,
    ):
        """
        search_depth     Maximum plies searched per candidate, INCLUDING the
                         candidate move itself. The search is iterative (2,
                         then 3, then 4 ...) and stops when the time budget
                         runs out, so this is a ceiling, not a promise.
        min_depth        Depth the search will always finish, clock or no
                         clock. 4 plies is the shallowest depth that reliably
                         sees a forced mate in two against us, which was the
                         original reason this filter exists, so it is not
                         something to give up when a position is complicated.
        quiescence_depth Extra plies used at the end of the search to finish
                         capture/check sequences (avoids fake "wins" like
                         taking a pawn defended by a queen).
        material_margin  A candidate is rejected when it is this many
                         centipawns worse than the best candidate, or worse.
                         60 = about two thirds of a pawn. Smaller = safer but
                         fewer choices for Jev; larger = more freedom for Jev
                         but more blunders get through.
        screen_depth     Depth of the cheap first pass that decides which
                         candidates deserve a deep search.
        shortlist_size   How many candidates survive that cheap first pass.
        time_budget      Soft limit, in seconds, for one call to filter().
        """
        self.search_depth = max(1, search_depth)
        self.min_depth = min(max(1, min_depth), self.search_depth)
        self.quiescence_depth = max(0, quiescence_depth)
        self.material_margin = material_margin
        self.screen_depth = max(1, screen_depth)
        self.shortlist_size = max(1, shortlist_size)
        self.time_budget = time_budget

        # Scratch state, reset at the start of every filter() call.
        self._killers: dict[int, list[chess.Move]] = {}
        self._nodes = 0

    # ------------------------------------------------------------ public API

    def filter(
        self,
        board: chess.Board,
        candidates: list[chess.Move],
    ) -> FilterResult:
        """
        Return the candidates that are not obvious blunders, best first.

        Steps:
          1. Cheap shallow pass over every candidate; keep the most promising
             `shortlist_size` of them (skipped when there are few candidates).
          2. Search the shortlist properly, getting deeper until the time
             budget runs out.
          3. If some candidates avoid forced mate, drop the ones that don't.
          4. Drop candidates that are `material_margin` or more worse than the
             best candidate.
          5. If everything loses to mate, keep the moves that survive longest
             (so we still return something legal).
        """
        if not candidates:
            return FilterResult(safe_moves=[])

        started = time.monotonic()
        deadline = started + self.time_budget
        self._killers = {}
        self._nodes = 0

        candidates = list(candidates)
        shortlist = self._shortlist(board, candidates)
        shortlisted = set(shortlist)
        screened_out = [m for m in candidates if m not in shortlisted]

        assessments, depth_reached = self._deepen(board, shortlist, deadline)

        # Safety net: if every shortlisted move loses to a forced mate, the
        # cheap screen may have thrown away the one move that saves us.
        # Searching the rest costs time, but this is rare and missing a
        # saving move is far worse than a slow turn.
        if assessments and screened_out and all(a.loses_to_mate for a in assessments):
            logger.info(
                "All shortlisted moves lose to mate; searching the other %d",
                len(screened_out),
            )
            extra, _ = self._deepen(
                board, screened_out, time.monotonic() + self.time_budget
            )
            assessments = assessments + extra
            screened_out = []

        best_score = max(a.score for a in assessments)

        result = FilterResult(
            safe_moves=[],
            assessments=assessments,
            depth_reached=depth_reached,
            screened_out=screened_out,
        )

        kept: list[MoveAssessment] = []
        if best_score <= -MATE_BOUND:
            # Everything loses by force. Keep the moves that last longest.
            result.all_lose_to_mate = True
            for a in assessments:
                if a.score == best_score:
                    a.reason = "least bad: delays the forced mate longest"
                    kept.append(a)
                else:
                    a.reason = "loses to a faster forced mate"
                    result.rejected.append(a)
        else:
            for a in assessments:
                if a.loses_to_mate:
                    a.reason = "allows forced checkmate"
                    result.rejected.append(a)
                elif not self.is_within_margin(a.score, best_score):
                    a.reason = (
                        f"loses material/position vs best "
                        f"({a.score:+d} vs {best_score:+d})"
                    )
                    result.rejected.append(a)
                else:
                    a.reason = "ok"
                    kept.append(a)

        # Safety net: never return an empty list.
        if not kept:
            best = max(assessments, key=lambda a: a.score)
            best.reason = "kept: nothing else survived the filter"
            kept.append(best)
            result.rejected = [a for a in result.rejected if a is not best]

        # Sort survivors best first. Python's sort is stable, so moves with
        # equal scores keep the order the caller gave us.
        kept.sort(key=lambda a: a.score, reverse=True)
        result.safe_assessments = kept
        result.safe_moves = [a.move for a in kept]

        for a in assessments:
            logger.info(
                "Tactical check | %-8s | %-18s | opp checks: %d | %s",
                a.san,
                a.describe_score(),
                a.opponent_checks,
                a.reason,
            )
        logger.info(
            "Tactical summary | depth %d | %d nodes | %.2fs | kept %d of %d"
            " | %d never deep-searched",
            depth_reached,
            self._nodes,
            time.monotonic() - started,
            len(kept),
            len(candidates),
            len(screened_out),
        )

        return result

    def is_within_margin(self, score: int, best_score: int) -> bool:
        """
        True when `score` is close enough to `best_score` to be offered.

        Note the strict < : a move that is EXACTLY `material_margin` worse
        than the best is rejected, not kept. The old code compared the other
        way round, so a move sitting exactly on the boundary slipped through.
        """
        return (best_score - score) < self.material_margin

    def assess(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int | None = None,
    ) -> MoveAssessment:
        """Analyse a single candidate move. The caller's board is untouched."""
        depth = self.search_depth if depth is None else max(1, depth)

        # Work on a copy so the real game board can never be corrupted.
        # We keep the move history (copy()'s default) because the search uses
        # it to notice repetitions.
        working = board.copy()

        if move not in working.legal_moves:
            raise ValueError(f"Illegal move passed to TacticalFilter: {move}")

        san = working.san(move)
        working.push(move)
        try:
            # Count the opponent's checking replies (information for logs and
            # for Jev's prompt; the search itself does not use it).
            opponent_checks = sum(
                1 for reply in working.legal_moves if working.gives_check(reply)
            )

            # Negamax returns the score for the opponent (side to move now),
            # so we flip the sign to get OUR score.
            score = -self._negamax(
                working,
                depth=depth - 1,
                alpha=-INFINITY,
                beta=INFINITY,
                ply=1,
            )
        finally:
            working.pop()

        return MoveAssessment(
            move=move,
            san=san,
            score=score,
            opponent_checks=opponent_checks,
            depth=depth,
        )

    # --------------------------------------------------- candidate selection

    def _shortlist(
        self,
        board: chess.Board,
        candidates: list[chess.Move],
    ) -> list[chess.Move]:
        """
        Cheap first pass: score every candidate with a very shallow search and
        keep the best `shortlist_size`. This is what buys the time to search
        the interesting moves more deeply.
        """
        if len(candidates) <= self.shortlist_size:
            return list(candidates)

        scored = [
            (self.assess(board, move, depth=self.screen_depth).score, index, move)
            for index, move in enumerate(candidates)
        ]
        # Best score first; ties keep the caller's order (that is what the
        # index is for).
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [move for _, _, move in scored[: self.shortlist_size]]

    def _deepen(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        deadline: float,
    ) -> tuple[list[MoveAssessment], int]:
        """
        Iterative deepening: search everything at depth 2, then 3, then 4 ...
        stopping when the clock runs out. Only fully finished passes count, so
        the scores we return are always comparable with each other.

        Searching the previous pass's best move first makes alpha-beta prune
        much harder, which is why this is usually FASTER than going straight
        to the final depth.
        """
        if not moves:
            return [], 0

        depths = list(range(2, self.search_depth + 1)) or [self.search_depth]
        order = list(moves)
        completed: dict[chess.Move, MoveAssessment] = {}
        depth_reached = 0

        last_pass_seconds = 0.0
        for depth in depths:
            # Do not start a pass we almost certainly cannot finish: each
            # extra ply costs roughly four times the previous one, and an
            # abandoned pass is wasted work. (Passes up to min_depth are
            # always attempted.)
            if depth > self.min_depth:
                if time.monotonic() + last_pass_seconds * 4 > deadline:
                    break

            pass_started = time.monotonic()
            results: dict[chess.Move, MoveAssessment] = {}
            ran_out = False
            for move in order:
                # Never abandon a pass at or below min_depth: those scores are
                # the safety guarantee, not a nice-to-have.
                if depth > self.min_depth and time.monotonic() >= deadline:
                    ran_out = True
                    break
                results[move] = self.assess(board, move, depth=depth)
            if ran_out:
                break
            last_pass_seconds = time.monotonic() - pass_started
            completed = results
            depth_reached = depth
            order.sort(key=lambda m: completed[m].score, reverse=True)
            if depth >= self.min_depth and time.monotonic() >= deadline:
                break

        if not completed:  # only possible when search_depth == 1
            completed = {m: self.assess(board, m, depth=1) for m in moves}
            depth_reached = 1

        return [completed[m] for m in moves], depth_reached

    # -------------------------------------------------------------- search

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        """
        Minimax in "negamax" form with alpha-beta pruning.
        Returns the score for the side to move on `board`.
        """
        self._nodes += 1

        # Draws. Treating a single repetition as a draw is a standard trick:
        # it stops us shuffling pieces when we are fine, and lets us aim for
        # a repetition when we are worse. is_repetition() is a bit expensive,
        # so we only ask right after our own candidate move (ply 1).
        if board.is_fifty_moves() or (ply == 1 and board.is_repetition(2)):
            return 0

        moves = list(board.legal_moves)

        if not moves:
            if board.is_check():
                return -MATE_SCORE + ply  # checkmated (sooner = worse)
            return 0                       # stalemate

        # Only worth asking when the board is nearly empty; the check is not
        # free and it can never be true with lots of pieces around.
        if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
            return 0

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, self.quiescence_depth)

        best = -INFINITY
        for move in self._order_moves(board, moves, ply):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                # The opponent would never allow this line. Remember quiet
                # moves that cause cut-offs: they often work elsewhere too.
                if not board.is_capture(move):
                    self._remember_killer(ply, move)
                break

        return best

    def _quiesce(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        qdepth: int,
    ) -> int:
        """
        "Quiet search": at the end of the normal search, keep playing only
        captures (or check evasions) until things calm down. This stops the
        search from stopping right in the middle of a capture exchange.
        """
        self._nodes += 1
        in_check = board.is_check()
        stand_pat = 0

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply  # checkmated
            if qdepth <= 0:
                return self._evaluate(board)
            best = -INFINITY  # can't "stand pat" while in check
        else:
            stand_pat = self._evaluate(board)
            if qdepth <= 0 or stand_pat >= beta:
                return stand_pat
            best = stand_pat
            alpha = max(alpha, stand_pat)
            moves = list(board.generate_legal_captures())
            if not moves:
                return stand_pat

        for move in self._order_moves(board, moves, ply):
            if not in_check:
                # "Delta pruning": if winning this piece outright still leaves
                # us far below alpha, the capture cannot rescue the position.
                gain = self._captured_value(board, move)
                if move.promotion:
                    gain += PIECE_VALUES[chess.QUEEN]
                if stand_pat + gain + 200 < alpha:
                    continue

            board.push(move)
            try:
                score = -self._quiesce(board, -beta, -alpha, ply + 1, qdepth - 1)
            finally:
                board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best

    # --------------------------------------------------------- evaluation

    def _evaluate(self, board: chess.Board) -> int:
        """
        Static score of a quiet position, from the side-to-move's point of
        view. Material first, then a handful of cheap positional terms.

        This function runs tens of thousands of times per turn, so it works
        on python-chess's raw bitboards (plain integers, one bit per square)
        instead of the friendlier Piece objects. `chess.scan_forward(mask)`
        walks the squares whose bit is set; `chess.popcount(mask)` counts them.
        """
        white_pieces = board.occupied_co[chess.WHITE]
        black_pieces = board.occupied_co[chess.BLACK]

        white = 0  # everything is computed from White's point of view first
        non_pawn_material = 0

        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN):
            value = PIECE_VALUES[piece_type]
            all_of_type = board.pieces_mask(piece_type, chess.WHITE) | \
                board.pieces_mask(piece_type, chess.BLACK)

            mine = all_of_type & white_pieces
            theirs = all_of_type & black_pieces

            white_count = chess.popcount(mine)
            black_count = chess.popcount(theirs)
            white += value * (white_count - black_count)

            if piece_type != chess.PAWN:
                non_pawn_material += value * (white_count + black_count)

            white_table = WHITE_TABLES[piece_type]
            for square in chess.scan_forward(mine):
                white += white_table[square]
            black_table = BLACK_TABLES[piece_type]
            for square in chess.scan_forward(theirs):
                white -= black_table[square]

        # 256 = opening, 0 = bare kings. The king's own table depends on this,
        # which is why kings are handled after the loop.
        phase = min(256, (non_pawn_material * 256) // OPENING_NON_PAWN_MATERIAL)

        white_king = board.king(chess.WHITE)
        if white_king is not None:
            white += (
                WHITE_KING_MIDDLEGAME[white_king] * phase
                + WHITE_KING_ENDGAME[white_king] * (256 - phase)
            ) // 256
        black_king = board.king(chess.BLACK)
        if black_king is not None:
            white -= (
                KING_MIDDLEGAME_TABLE[black_king] * phase
                + KING_ENDGAME_TABLE[black_king] * (256 - phase)
            ) // 256

        white += self._structure_and_activity(board, chess.WHITE, phase)
        white -= self._structure_and_activity(board, chess.BLACK, phase)

        # Having the move is worth a little; this also damps the "score jumps
        # around between odd and even depths" effect.
        white += TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS

        return white if board.turn == chess.WHITE else -white

    def _structure_and_activity(
        self,
        board: chess.Board,
        color: chess.Color,
        phase: int,
    ) -> int:
        """Pawn structure, piece mobility and king shelter for one side."""
        score = 0
        own_pieces = board.occupied_co[color]

        # Two bishops working together are worth more than two knights.
        if chess.popcount(board.bishops & own_pieces) >= 2:
            score += BISHOP_PAIR_BONUS

        # Doubled and isolated pawns, counted one file at a time.
        pawns = board.pawns & own_pieces
        if pawns:
            per_file = [chess.popcount(pawns & file_mask)
                        for file_mask in chess.BB_FILES]
            for file_index, count in enumerate(per_file):
                if count == 0:
                    continue
                if count > 1:
                    score -= DOUBLED_PAWN_PENALTY * (count - 1)
                left = per_file[file_index - 1] if file_index > 0 else 0
                right = per_file[file_index + 1] if file_index < 7 else 0
                if left == 0 and right == 0:
                    score -= ISOLATED_PAWN_PENALTY * count

        # Mobility of the long-range pieces: how many squares they cover that
        # are not blocked by their own pieces.
        free = ~own_pieces
        for piece_type, weight in MOBILITY_WEIGHTS.items():
            for square in chess.scan_forward(
                board.pieces_mask(piece_type, color)
            ):
                score += weight * chess.popcount(
                    board.attacks_mask(square) & free
                )

        # A king with pawns in front of it is much safer, but only while there
        # are still pieces on the board, hence the phase scaling.
        king_square = board.king(color)
        if king_square is not None and phase > 0:
            shield = self._king_shield(pawns, color, king_square)
            score += (shield * phase) // 256

        return score

    def _king_shield(
        self,
        own_pawns: int,
        color: chess.Color,
        king_square: int,
    ) -> int:
        """Bonus for each of the three files by the king that still has a pawn."""
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        forward = 1 if color == chess.WHITE else -1

        score = 0
        for file_index in (king_file - 1, king_file, king_file + 1):
            if not 0 <= file_index <= 7:
                continue
            for distance in (1, 2):
                rank = king_rank + forward * distance
                if not 0 <= rank <= 7:
                    break
                if own_pawns & chess.BB_SQUARES[chess.square(file_index, rank)]:
                    score += KING_SHIELD_BONUS
                    break
        return score

    # ------------------------------------------------------- move ordering

    def _remember_killer(self, ply: int, move: chess.Move) -> None:
        killers = self._killers.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]  # remember the two most recent ones

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        ply: int,
    ) -> list[chess.Move]:
        """Look at promising moves first so alpha-beta prunes more."""
        killers = self._killers.get(ply, ())
        return sorted(
            moves,
            key=lambda m: self._move_order_key(board, m, killers),
            reverse=True,
        )

    def _move_order_key(
        self,
        board: chess.Board,
        move: chess.Move,
        killers,
    ) -> int:
        # Note: we deliberately do NOT use board.gives_check() here. It is
        # accurate but slow (it plays the move internally), and calling it for
        # every move at every node roughly halves the search speed.
        key = 0
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
            key += 10_000 + 10 * self._captured_value(board, move) - attacker_value
        elif move in killers:
            key += 9_000
        if move.promotion:
            key += 8_000
        return key

    def _captured_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        victim = board.piece_at(move.to_square)
        return PIECE_VALUES[victim.piece_type] if victim else 0