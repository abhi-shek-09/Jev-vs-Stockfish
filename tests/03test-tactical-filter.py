"""Run with:  python tests/test_tactical_filter.py  (no API key or Stockfish needed)"""
import logging
import sys
import time
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess
from candidate_generator import CandidateGenerator
from tactical_filter import TacticalFilter

logging.basicConfig(level=logging.INFO, format="%(message)s")
tf = TacticalFilter(search_depth=4)


def sans(board, moves):
    return [board.san(m) for m in moves]


# 1. The real failure: Kg1 allows Qe1#, c4 is the only non-immediately-losing move.
board = chess.Board("r1r3k1/1p6/p5p1/1b2qp2/3p4/8/PPP2PPP/5K2 w - - 2 27")
fen_before = board.fen()
start = time.time()
res = tf.filter(board, list(board.legal_moves))
print("time: %.2fs" % (time.time() - start))
kg1 = next(a for a in res.assessments if a.san == "Kg1")
assert kg1.loses_to_mate, "Kg1 must be flagged as losing to mate"
assert "Kg1" not in sans(board, res.safe_moves) or res.all_lose_to_mate
assert "c4" in sans(board, res.safe_moves)
assert board.fen() == fen_before and len(board.move_stack) == 0, "board mutated!"
print("PASS 1: Kg1 rejected / c4 kept, board untouched\n")

# 2. Mate in one for us is recognised.
board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
a = tf.assess(board, chess.Move.from_uci("a1a8"))
assert a.delivers_mate
print("PASS 2: Ra8# recognised as mate\n")

# 3. Hanging the queen is rejected in the opening-ish position.
board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
res = tf.filter(board, list(board.legal_moves))
assert res.safe_moves, "must never be empty"
print("PASS 3: safe moves:", sans(board, res.safe_moves), "\n")

# 4. Start position: nothing should be rejected, never empty.
board = chess.Board()
pool = CandidateGenerator(12).generate(board)
res = tf.filter(board, pool)
assert len(res.safe_moves) == len(pool)
print("PASS 4: start position keeps all", len(pool), "candidates\n")

# 5. Empty input never crashes.
assert tf.filter(chess.Board(), []).safe_moves == []
print("ALL TESTS PASSED")