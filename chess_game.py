import logging
import threading
import time

import chess

from move_animation import captured_lists, describe_move, move_list_san

logger = logging.getLogger(__name__)


class GameControls:
    """
    Pause / resume / stop / speed, shared between the UI thread and the game
    thread. Everything here is thread-safe: the UI thread only ever sets
    flags, and the game thread only ever reads them between moves.

    A ChessGame with no controls behaves exactly like the original one.
    """

    def __init__(self, move_delay: float = 0.6):
        self._resume = threading.Event()
        self._resume.set()          # start running, not paused
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._move_delay = move_delay

    # --------------------------------------------------------------- state

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def move_delay(self) -> float:
        with self._lock:
            return self._move_delay

    def set_move_delay(self, seconds: float) -> None:
        with self._lock:
            self._move_delay = max(0.0, min(5.0, float(seconds)))

    # ------------------------------------------------------------ commands

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()  # so a paused game can actually exit

    # ----------------------------------------------------- game-thread side

    def wait_while_paused(self) -> None:
        """Block here while paused. Returns immediately when running."""
        while not self._resume.wait(timeout=0.2):
            if self._stop.is_set():
                return

    def sleep_between_moves(self) -> None:
        """Sleep the configured delay, but wake up quickly on stop."""
        remaining = self.move_delay
        while remaining > 0 and not self._stop.is_set():
            step = min(0.1, remaining)
            time.sleep(step)
            remaining -= step


class GameStopped(Exception):
    """Raised inside the game thread when someone asks the game to stop."""


class ChessGame:
    def __init__(
        self,
        white,
        black,
        max_moves: int = 200,
        on_event=None,
        controls: GameControls | None = None,
    ):
        """
        white, black   objects with a get_move(board) method
        max_moves      safety valve so a shuffling game cannot run forever
        on_event       optional callback taking a single dict. ChessGame calls
                       it whenever something worth watching happens. It knows
                       nothing about HTTP, threads or the UI.
        controls       optional GameControls for pause / stop / speed
        """
        self.board = chess.Board()
        self.white = white
        self.black = black
        self.max_moves = max_moves
        self.on_event = on_event
        self.controls = controls

    # --------------------------------------------------------------- events

    def _emit(self, event_type: str, **payload) -> None:
        """Publish one event. A broken listener must never break the game."""
        if self.on_event is None:
            return
        event = {"type": event_type}
        event.update(payload)
        try:
            self.on_event(event)
        except Exception:  # pragma: no cover - defensive only
            logger.exception("Event listener raised on %s", event_type)

    def _player_name(self, color: chess.Color) -> str:
        return "Jev" if color == chess.WHITE else "Stockfish"

    def _snapshot(self) -> dict:
        """The facts the UI needs to redraw everything from scratch."""
        return {
            "fen": self.board.fen(),
            "move_list": move_list_san(self.board),
            "captured": captured_lists(self.board),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "in_check": self.board.is_check(),
            "move_number": self.board.fullmove_number,
        }

    # ------------------------------------------------------------- the game

    def run(self):
        logger.info("========== NEW GAME ==========")
        logger.info("Game started")
        logger.info("Initial position: %s", self.board.fen())

        self._emit(
            "game_started",
            players={"white": "Jev", "black": "Stockfish"},
            **self._snapshot(),
        )

        move_number = 1
        try:
            while not self.board.is_game_over(claim_draw=True):
                if self.controls is not None:
                    self.controls.wait_while_paused()
                    if self.controls.stopped:
                        raise GameStopped()

                if move_number > self.max_moves:
                    logger.info("Move limit reached; stopping the game")
                    break

                if self.board.turn == chess.WHITE:
                    player = self.white
                    player_name = "Jev"
                else:
                    player = self.black
                    player_name = "Stockfish"

                logger.info(
                    "Turn %d | Player: %s | FEN: %s",
                    move_number,
                    player_name,
                    self.board.fen(),
                )
                self._emit(
                    "turn_started",
                    side="white" if self.board.turn == chess.WHITE else "black",
                    player=player_name,
                    move_number=move_number,
                    fen=self.board.fen(),
                )

                # The player sees a COPY: nothing outside ChessGame may ever
                # touch the real board.
                move = player.get_move(self.board.copy())

                if move not in self.board.legal_moves:
                    raise RuntimeError(
                        f"{player_name} returned an illegal move: {move}"
                    )

                san = self.board.san(move=move)
                # Work the animation out BEFORE the move is played.
                # Note the rename: "capture" is the piece taken by THIS move,
                # while "captured" (from the snapshot) is each side's tray.
                described = describe_move(self.board, move)
                animation = {
                    "uci": described["uci"],
                    "movements": described["movements"],
                    "capture": described["captured"],
                    "promotion": described["promotion"],
                }

                logger.info(
                    "Turn %d | %s played %s", move_number, player_name, san
                )
                print(f"{move_number}. {player_name} : {san}")

                side = "white" if self.board.turn == chess.WHITE else "black"
                fen_before = self.board.fen()
                self.board.push(move=move)

                decision = getattr(player, "last_decision", None)

                payload = self._snapshot()
                payload.update(animation)
                payload.update(
                    side=side,
                    player=player_name,
                    san=san,
                    move_number=move_number,
                    fen_before=fen_before,
                    decision=decision,
                )
                self._emit("move_played", **payload)

                if self.board.turn == chess.WHITE:
                    move_number += 1

                if self.controls is not None:
                    self.controls.sleep_between_moves()

        except GameStopped:
            logger.info("Game stopped by request")
            self._emit("game_stopped", **self._snapshot())
            return
        except Exception as error:
            logger.exception("Game thread failed")
            self._emit(
                "error",
                message=f"{type(error).__name__}: {error}",
                **self._snapshot(),
            )
            raise

        result = self.board.result(claim_draw=True)
        reason = self._result_reason()

        logger.info("Game over")
        logger.info("Result: %s", result)
        logger.info("Final position: %s", self.board.fen())
        logger.info("Moves: %s", " ".join(move_list_san(self.board)))

        print()
        print("Game Over")
        print(f"Result: {result}")

        self._emit("game_over", result=result, reason=reason, **self._snapshot())

    def _result_reason(self) -> str:
        """A short human explanation of how the game ended."""
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient material"
        if self.board.is_seventyfive_moves() or self.board.can_claim_fifty_moves():
            return "fifty-move rule"
        if self.board.is_fivefold_repetition() or self.board.can_claim_threefold_repetition():
            return "repetition"
        return "move limit reached"