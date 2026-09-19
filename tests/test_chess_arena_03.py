"""
test_chess_arena.py

Offline tests. No API key, no Stockfish, no network: only python-chess.

    python -m unittest test_chess_arena -v

They cover the behaviour that changed, plus the two rules the whole design
depends on:
  * the caller's board is never modified,
  * the candidate list is never empty when legal moves exist.
"""

import random
import time
import unittest

import chess

from candidate_generator import CandidateGenerator
from tactical_filter import MATE_BOUND, TacticalFilter
import move_notes


def fast_filter(**overrides) -> TacticalFilter:
    """A shallow, quick filter so the tests finish in seconds."""
    settings = dict(
        search_depth=3,
        quiescence_depth=3,
        material_margin=60,
        shortlist_size=10,
        time_budget=3.0,
    )
    settings.update(overrides)
    return TacticalFilter(**settings)


def snapshot(board: chess.Board):
    """Everything about a board that must not change."""
    return (board.fen(), len(board.move_stack), board.castling_rights)


class BoardIsNeverMutated(unittest.TestCase):
    """The single most important safety property."""

    POSITIONS = [
        chess.STARTING_FEN,
        "rnbqkb1r/pp1p1ppp/4pn2/2p1N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 2 4",
        "r3k2r/pp2bppp/4p3/1b1pN3/3q4/8/P1PB1PPP/R2QK2R w KQkq - 0 15",
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
    ]

    def test_filter_and_generate_leave_the_board_alone(self):
        generator = CandidateGenerator(max_candidates=256)
        tactical = fast_filter()
        for fen in self.POSITIONS:
            board = chess.Board(fen)
            before = snapshot(board)

            pool = generator.generate(board)
            self.assertEqual(snapshot(board), before, fen)

            result = tactical.filter(board, pool)
            self.assertEqual(snapshot(board), before, fen)

            for move in list(board.legal_moves)[:5]:
                tactical.assess(board, move)
                self.assertEqual(snapshot(board), before, fen)

            for assessment in result.safe_assessments:
                move_notes.describe_move(board, assessment)
                self.assertEqual(snapshot(board), before, fen)

            move_notes.position_summary(board)
            self.assertEqual(snapshot(board), before, fen)

    def test_move_history_survives(self):
        """assess() copies WITH history, so the stack must come back intact."""
        board = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6"]:
            board.push_san(san)
        before = snapshot(board)
        fast_filter().filter(board, list(board.legal_moves))
        self.assertEqual(snapshot(board), before)
        self.assertEqual(len(board.move_stack), 4)


class NeverEmpty(unittest.TestCase):

    def test_single_legal_move_is_returned(self):
        # White is in check and the only legal move is Kxg2.
        board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
        self.assertEqual(len(list(board.legal_moves)), 1)
        result = fast_filter().filter(board, list(board.legal_moves))
        self.assertEqual(len(result.safe_moves), 1)
        self.assertEqual(board.san(result.safe_moves[0]), "Kxg2")

    def test_hopeless_position_still_returns_a_move(self):
        # Black is about to mate whatever White does.
        board = chess.Board("6k1/5ppp/8/8/8/7q/6r1/6K1 w - - 0 1")
        result = fast_filter().filter(board, list(board.legal_moves))
        self.assertGreaterEqual(len(result.safe_moves), 1)
        for move in result.safe_moves:
            self.assertIn(move, board.legal_moves)

    def test_random_walk_never_produces_an_empty_list(self):
        """Twenty assorted positions from a random game."""
        rng = random.Random(12345)
        tactical = fast_filter(search_depth=2, shortlist_size=6, time_budget=1.0)
        generator = CandidateGenerator(max_candidates=256)
        board = chess.Board()
        checked = 0
        while checked < 20 and not board.is_game_over():
            before = snapshot(board)
            result = tactical.filter(board, generator.generate(board))
            self.assertGreaterEqual(len(result.safe_moves), 1, board.fen())
            self.assertEqual(snapshot(board), before)
            checked += 1
            board.push(rng.choice(list(board.legal_moves)))
        self.assertEqual(checked, 20)


class TacticalBehaviour(unittest.TestCase):

    def test_finds_mate_in_one_and_ranks_it_first(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
        result = fast_filter().filter(board, list(board.legal_moves))
        best = result.safe_assessments[0]
        self.assertTrue(best.delivers_mate, best.describe_score())
        self.assertEqual(board.san(best.move), "Ra8#")

    def test_mate_in_one_survives_the_cheap_shortlist(self):
        """The screen must not throw away a mate just because the pool is big."""
        board = chess.Board("6k1/5ppp/8/8/8/8/8/3QK2R w K - 0 1")
        self.assertGreater(len(list(board.legal_moves)), 10)
        tactical = fast_filter(shortlist_size=4)
        result = tactical.filter(board, list(board.legal_moves))
        self.assertTrue(result.safe_assessments[0].delivers_mate)

    def test_rejects_a_move_that_allows_mate_in_one(self):
        # White's rook on a1 is the only thing guarding the back rank.
        # Ra2 steps off it and allows Re1#.
        board = chess.Board("4r1k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        # A shortlist big enough to hold every move, so the rejection happens
        # in the deep search and we can check the reason it gives.
        result = fast_filter(shortlist_size=40).filter(
            board, list(board.legal_moves)
        )
        ra2 = chess.Move.from_uci("a1a2")
        self.assertNotIn(ra2, result.safe_moves)
        rejected = {a.move: a for a in result.rejected}
        self.assertIn(ra2, rejected)
        self.assertEqual(rejected[ra2].reason, "allows forced checkmate")

    def test_a_move_allowing_mate_is_dropped_by_the_shortlist_too(self):
        """Same position, small shortlist: the move must still never be offered."""
        board = chess.Board("4r1k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        result = fast_filter(shortlist_size=4).filter(
            board, list(board.legal_moves)
        )
        self.assertNotIn(chess.Move.from_uci("a1a2"), result.safe_moves)

    def test_rejects_hanging_the_queen(self):
        # Qxd5 wins a pawn but loses the queen to cxd5.
        board = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
        result = fast_filter().filter(board, list(board.legal_moves))
        self.assertNotIn(chess.Move.from_uci("d1d5"), result.safe_moves)

    def test_defending_a_hanging_pawn_outranks_losing_it(self):
        """Turn 4 of the analysed game: e4 is attacked and undefended."""
        board = chess.Board(
            "rnbqkb1r/pp1p1ppp/4pn2/2p1N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 2 4"
        )
        tactical = fast_filter(search_depth=4, shortlist_size=14)
        result = tactical.filter(board, list(board.legal_moves))

        # Nf3 retreats and simply drops the e4 pawn; Nc3 and Bd3 defend it.
        retreat = tactical.assess(board, chess.Move.from_uci("e5f3"), depth=4)
        defend = tactical.assess(board, chess.Move.from_uci("b1c3"), depth=4)
        self.assertGreater(defend.score, retreat.score)

        # The pawn-losing retreat must not be one of the moves Jev sees, and
        # the best-rated move must be one that keeps the pawn. In the game we
        # analysed, five of the six moves Jev was shown lost this pawn.
        self.assertNotIn(chess.Move.from_uci("e5f3"), result.safe_moves)
        self.assertIn(result.safe_assessments[0].san,
                      {"Nc3", "d3", "Bd3", "Qe2", "Qf3", "f3", "d4"})


class MarginRule(unittest.TestCase):

    def test_exact_boundary_is_rejected(self):
        tactical = TacticalFilter(material_margin=90)
        self.assertTrue(tactical.is_within_margin(-89, 0))
        self.assertFalse(tactical.is_within_margin(-90, 0))
        self.assertFalse(tactical.is_within_margin(-91, 0))
        self.assertTrue(tactical.is_within_margin(10, 0))


class RepetitionHandling(unittest.TestCase):

    def test_repeating_while_winning_scores_as_a_draw(self):
        # White is a rook up. Kh1 repeats a position that has already occurred.
        board = chess.Board("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
        for uci in ["g1h1", "h8g8", "h1g1", "g8h8"]:
            board.push(chess.Move.from_uci(uci))

        tactical = fast_filter()
        repeat = tactical.assess(board, chess.Move.from_uci("g1h1"))
        other = tactical.assess(board, chess.Move.from_uci("a1a8"))

        self.assertEqual(repeat.score, 0)
        self.assertGreater(other.score, 300)


class Evaluation(unittest.TestCase):

    POSITIONS = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/5k2/8/3P4/8/2K5/8/8 w - - 0 1",
    ]

    def test_evaluation_is_colour_symmetric(self):
        """A mirrored position must score exactly the same for the other side."""
        tactical = fast_filter()
        for fen in self.POSITIONS:
            board = chess.Board(fen)
            self.assertEqual(
                tactical._evaluate(board),
                tactical._evaluate(board.mirror()),
                fen,
            )

    def test_quiet_moves_are_no_longer_all_equal(self):
        """The old material-only evaluation scored every first move at +0."""
        board = chess.Board()
        tactical = fast_filter(search_depth=2)
        scores = {
            board.san(move): tactical.assess(board, move, depth=2).score
            for move in board.legal_moves
        }
        self.assertGreater(len(set(scores.values())), 5)
        # And the sensible moves beat the silly ones.
        self.assertGreater(scores["e4"], scores["a3"])
        self.assertGreater(scores["Nf3"], scores["Na3"])
        self.assertGreater(scores["Nf3"], scores["Nh3"])


class GeneratorHeuristics(unittest.TestCase):

    def test_pool_contains_every_legal_move_when_large_enough(self):
        board = chess.Board()
        pool = CandidateGenerator(max_candidates=256).generate(board)
        self.assertEqual(sorted(pool, key=str),
                         sorted(board.legal_moves, key=str))

    def test_generate_is_empty_only_when_the_game_is_over(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # checkmate
        self.assertEqual(CandidateGenerator().generate(board), [])

    def test_going_straight_back_is_penalised(self):
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3e5", "b8c6"]:
            board.push(chess.Move.from_uci(uci))
        generator = CandidateGenerator(max_candidates=256)
        back = generator._score_move(board, chess.Move.from_uci("e5f3"))
        forward = generator._score_move(board, chess.Move.from_uci("e5d3"))
        self.assertLess(back, forward)

    def test_edge_knight_moves_rank_below_central_ones(self):
        board = chess.Board()
        generator = CandidateGenerator(max_candidates=256)
        self.assertLess(
            generator._score_move(board, chess.Move.from_uci("b1a3")),
            generator._score_move(board, chess.Move.from_uci("b1c3")),
        )


class MoveNotes(unittest.TestCase):

    def test_notes_flag_an_undefended_landing_square(self):
        board = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
        assessment = fast_filter().assess(board, chess.Move.from_uci("d1d5"))
        note = move_notes.describe_move(board, assessment)
        self.assertIn("undefended", note)
        self.assertIn("queen", note)

    def test_notes_mention_defending_a_hanging_pawn(self):
        """Turn 4 again: Nc3 rescues the e4 pawn, and Jev should be told."""
        board = chess.Board(
            "rnbqkb1r/pp1p1ppp/4pn2/2p1N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 2 4"
        )
        assessment = fast_filter().assess(board, chess.Move.from_uci("b1c3"))
        note = move_notes.describe_move(board, assessment)
        self.assertIn("defends", note)
        self.assertIn("e4", note)

    def test_summary_reports_material_and_history(self):
        board = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Nxe5", "Nxe5"]:
            board.push_san(san)
        summary = move_notes.position_summary(board)
        self.assertEqual(summary["side_to_move"], "White")
        # White gave up a knight for a pawn in that sequence.
        self.assertIn("behind", summary["material"])
        self.assertIn("Nxe5", summary["recent_moves"])
        self.assertFalse(summary["in_check"])


class Latency(unittest.TestCase):

    SHARP = "r3k2r/pp2bppp/4p3/1b1pN3/3q4/8/P1PB1PPP/R2QK2R w KQkq - 0 15"

    def test_time_budget_caps_the_search(self):
        """Not a hard real-time guarantee, but it must not run for minutes."""
        board = chess.Board(self.SHARP)
        tactical = TacticalFilter(search_depth=8, min_depth=2, time_budget=1.5)
        started = time.monotonic()
        result = tactical.filter(board, list(board.legal_moves))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 10.0, f"took {elapsed:.1f}s")
        self.assertGreaterEqual(result.depth_reached, 2)
        self.assertGreaterEqual(len(result.safe_moves), 1)

    def test_minimum_depth_is_reached_even_in_a_sharp_position(self):
        """4 plies is what catches a forced mate in two against us."""
        board = chess.Board(self.SHARP)
        tactical = TacticalFilter(time_budget=0.01)  # budget deliberately tiny
        result = tactical.filter(board, list(board.legal_moves))
        self.assertGreaterEqual(result.depth_reached, tactical.min_depth)


if __name__ == "__main__":
    unittest.main(verbosity=2)