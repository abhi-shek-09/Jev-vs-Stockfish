import os
import logging
import chess
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient
from candidate_generator import CandidateGenerator
from tactical_filter import TacticalFilter, MoveAssessment
from move_notes import describe_move, position_summary

load_dotenv()
logger = logging.getLogger(__name__)


class JevPlayer:
    """
    Jev makes the strategic decision. This class only makes sure Jev is shown
    a legal, tactically sensible, well-described set of candidate moves.

    After every turn, `last_decision` holds a plain dictionary describing what
    happened, so a UI can show it without reaching into any internals.
    """

    def __init__(
        self,
        max_offered: int = 6,
        search_depth: int = 6,
        time_budget: float = 4.0,
        material_margin: int = 60,
        fall_back_on_bad_answer: bool = True,
    ):
        self.client = TypeSafeClient(
            api_key=os.environ["TYPESAFE_API_KEY"]
        )

        # Step 1: cheap heuristics order the legal moves. The pool is every
        # legal move: these heuristics are not good enough to remove a move
        # outright, and in the game we analysed they hid the only good move
        # on several turns.
        self.candidate_generator = CandidateGenerator(max_candidates=256)

        # Step 2: shallow tactical search removes obvious blunders and ranks
        # what is left.
        self.tactical_filter = TacticalFilter(
            search_depth=search_depth,
            material_margin=material_margin,
            time_budget=time_budget,
        )

        # How many moves Jev is finally shown.
        self.max_offered = max_offered

        # If the SDK answers with something we did not offer, fall back to the
        # best-scoring candidate instead of crashing the whole game.
        self.fall_back_on_bad_answer = fall_back_on_bad_answer

        # Filled in by get_move(); read by the UI. Never required by the game.
        self.last_decision: dict | None = None

    # ------------------------------------------------------------ candidates

    def _select_candidates(self, board: chess.Board):
        """
        Generator -> TacticalFilter, with fallbacks. Returns
        (offered_assessments, filter_result). Never returns an empty list of
        offers unless the game is over.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return [], None

        pool = self.candidate_generator.generate(board=board)
        if not pool:
            pool = legal_moves

        result = self.tactical_filter.filter(board, pool)

        # If every pooled move loses to mate, the generator may have missed
        # a saving move. Re-check ALL legal moves before giving up.
        if result.all_lose_to_mate and len(pool) < len(legal_moves):
            logger.info(
                "All pooled moves lose to mate; widening to all legal moves"
            )
            result = self.tactical_filter.filter(board, legal_moves)

        offered = result.safe_assessments
        if not offered:
            # Should be impossible (the filter has its own safety net), but
            # the rule "never show Jev an empty list" is worth two belts.
            logger.warning("Filter returned nothing; offering legal moves raw")
            offered = [
                self.tactical_filter.assess(board, move, depth=1)
                for move in legal_moves[: self.max_offered]
            ]

        return offered[: self.max_offered], result

    # ----------------------------------------------------------- UI records

    def _build_decision(
        self,
        board: chess.Board,
        offered: list[MoveAssessment],
        result,
        notes: dict,
        chosen_san: str | None,
    ) -> dict:
        """A JSON-friendly record of this turn, for the UI panel."""
        candidates = [
            {
                "san": a.san,
                "uci": a.move.uci(),
                "score": a.score,
                "score_text": a.describe_score(),
                "kept": True,
                "reason": a.reason or "ok",
                "note": notes.get(a.san, ""),
                "chosen": a.san == chosen_san,
            }
            for a in offered
        ]

        rejected = []
        if result is not None:
            for a in result.rejected:
                rejected.append(
                    {
                        "san": a.san,
                        "uci": a.move.uci(),
                        "score": a.score,
                        "score_text": a.describe_score(),
                        "kept": False,
                        "reason": a.reason,
                    }
                )
            # Worst first reads badly; show the near-misses first.
            rejected.sort(key=lambda item: item["score"], reverse=True)

        return {
            "side": "white" if board.turn == chess.WHITE else "black",
            "player": "Jev",
            "move_number": board.fullmove_number,
            "fen": board.fen(),
            "chosen": chosen_san,
            "candidates": candidates,
            "rejected": rejected[:8],
            "rejected_total": len(rejected),
            "depth": getattr(result, "depth_reached", 0) if result else 0,
            "screened_out": len(getattr(result, "screened_out", []) or []),
        }

    # --------------------------------------------------------------- moving

    def get_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            raise RuntimeError("No legal moves available")

        offered, result = self._select_candidates(board)

        if not offered:
            raise RuntimeError("No legal moves available")

        candidates = {a.san: a.move for a in offered}
        # Plain-English note for each move: what it does, what it leaves
        # hanging, what the shallow search thinks of it.
        notes = {a.san: describe_move(board, a) for a in offered}

        # Record the decision BEFORE asking, so the UI has something to show
        # even if the call fails.
        self.last_decision = self._build_decision(
            board, offered, result, notes, chosen_san=None
        )

        logger.info("Jev evaluating position: %s", board.fen())
        logger.info("Jev candidates: %s", ", ".join(candidates))

        print("Candidate moves offered to Jev:")
        for san, note in notes.items():
            print(f"  {san:<8} {note}")
        print()

        # Build state for Jev. Everything here is description, not advice.
        state = position_summary(board)
        state["candidate_moves"] = list(candidates)
        state["candidate_notes"] = notes

        side = state["side_to_move"]

        question = Choice(
            instructions=(
                f"You are playing {side} in a chess game against a strong "
                "engine. Choose one move from the candidates.\n"
                "Every candidate has already survived a shallow tactical "
                "check, so none of them hangs material outright or allows a "
                "forced mate within a few moves. The score next to each move "
                "comes from that same shallow search, in hundredths of a pawn "
                "and from your point of view (positive is good for you).\n"
                "Treat the scores as a guide, not the truth: the search is "
                "only a few moves deep and knows nothing about plans. Use "
                "them to avoid drifting, and use your own judgement to pick "
                "between moves that score similarly.\n"
                "Play with a plan: develop your pieces towards the centre, "
                "castle early, do not move the same piece twice without a "
                "reason, do not bring the queen out early, keep your pieces "
                "defended, and improve your worst-placed piece. Prefer a "
                "clearly better score unless you have a concrete reason not "
                "to."
            ),
            criteria={san: notes[san] for san in candidates},
        )

        response = self.client.system_one(
            model="jev-latest",
            state=state,
            questions={
                "best_move": question,
            },
        )

        # Read Jev's typed Choice answer
        answer = response.answers["best_move"]
        chosen_san = answer.choice

        logger.info("Jev selected: %s", chosen_san)

        print("Jev's answer:")
        print(chosen_san)
        print()

        # Validate Jev's answer
        if chosen_san not in candidates:
            if not self.fall_back_on_bad_answer:
                raise RuntimeError(
                    f"Jev returned an unexpected move: {chosen_san}"
                )
            fallback = offered[0]
            logger.error(
                "Jev returned an unexpected move (%s); playing %s instead",
                chosen_san,
                fallback.san,
            )
            self.last_decision = self._build_decision(
                board, offered, result, notes, chosen_san=fallback.san
            )
            self.last_decision["fallback_from"] = chosen_san
            return fallback.move

        self.last_decision = self._build_decision(
            board, offered, result, notes, chosen_san=chosen_san
        )

        chosen_move = candidates[chosen_san]

        if chosen_move not in board.legal_moves:
            raise RuntimeError(
                f"Jev selected an illegal move: {chosen_san}"
            )

        return chosen_move