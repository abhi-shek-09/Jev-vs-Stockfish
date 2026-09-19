"""
offline_selfplay.py

A sanity check you can run without an API key and without Stockfish.

It replaces Jev with a stub that just picks one of the candidate moves the
generator + filter offer, and plays a whole game. The point is not the result:
it is to see, cheaply, whether the moves being OFFERED look like chess, and how
long a turn takes.

    python offline_selfplay.py                 # top candidate vs random mover
    python offline_selfplay.py both            # both sides use the layer
    python offline_selfplay.py random-jev      # picks randomly among candidates

"random-jev" is the interesting one: it is the worst Jev could possibly be. If
the layer is doing its job, even a random choice among the offered moves should
not collapse in ten moves.
"""

import random
import sys
import time

import chess

from candidate_generator import CandidateGenerator
from tactical_filter import TacticalFilter


class LayerPlayer:
    """Uses the real candidate layer, then picks with a simple rule."""

    def __init__(self, pick="top", max_offered=6, seed=0, **filter_kwargs):
        self.generator = CandidateGenerator(max_candidates=256)
        self.filter = TacticalFilter(**filter_kwargs)
        self.pick = pick
        self.max_offered = max_offered
        self.rng = random.Random(seed)
        self.offered_counts = []
        self.turn_seconds = []

    def get_move(self, board):
        started = time.monotonic()
        result = self.filter.filter(board, self.generator.generate(board))
        offered = result.safe_assessments[: self.max_offered]
        self.offered_counts.append(len(offered))
        self.turn_seconds.append(time.monotonic() - started)
        if self.pick == "random":
            return self.rng.choice(offered).move
        return offered[0].move


class RandomPlayer:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def get_move(self, board):
        return self.rng.choice(list(board.legal_moves))


def play(white, black, max_moves=120):
    board = chess.Board()
    sans = []
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        player = white if board.turn == chess.WHITE else black
        move = player.get_move(board)
        assert move in board.legal_moves, f"illegal move {move}"
        sans.append(board.san(move))
        board.push(move)
    return board, sans


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "top"

    # A shallow, fast setting so a whole game finishes in a couple of minutes.
    settings = dict(search_depth=4, min_depth=3, time_budget=1.5, shortlist_size=8)

    if mode == "both":
        white = LayerPlayer(pick="top", **settings)
        black = LayerPlayer(pick="top", seed=2, **settings)
    elif mode == "random-jev":
        white = LayerPlayer(pick="random", **settings)
        black = RandomPlayer()
    else:
        white = LayerPlayer(pick="top", **settings)
        black = RandomPlayer()

    started = time.monotonic()
    board, sans = play(white, black)
    elapsed = time.monotonic() - started

    print(" ".join(
        f"{i // 2 + 1}. {san}" if i % 2 == 0 else san
        for i, san in enumerate(sans)
    ))
    print()
    print(f"result: {board.result(claim_draw=True)}  ({len(sans)} half-moves)")
    print(f"total time: {elapsed:.1f}s")
    for name, player in (("white", white), ("black", black)):
        if isinstance(player, LayerPlayer):
            turns = player.turn_seconds
            counts = player.offered_counts
            print(
                f"{name}: {len(turns)} turns, "
                f"mean {sum(turns) / len(turns):.2f}s, max {max(turns):.2f}s, "
                f"mean options offered {sum(counts) / len(counts):.1f}, "
                f"turns with only one option: {counts.count(1)}"
            )


if __name__ == "__main__":
    main()