"""
ui_server.py

A tiny spectator server for ChessArena.

How it fits together
--------------------
    main.py --ui
        -> GameHost            owns the game thread and the current state
        -> ThreadingHTTPServer serves one HTML page and an event stream
        -> browser             draws the board and animates moves

The game itself runs in a BACKGROUND THREAD, because both the TypeSafe call
and Stockfish block for seconds at a time. The HTTP server never calls the LLM
or the engine, and never touches the game's board: it only relays dictionaries
that ChessGame hands it.

Why Server-Sent Events rather than websockets: SSE is one long-lived GET, it
is in the standard library on the server side and one line of JavaScript on
the client, and the traffic here only ever goes server -> browser. No extra
dependency is needed for it.

Every event carries a "seq" number. A browser that reloads asks for the
snapshot plus everything since seq 0, so a second tab or a refresh shows the
full current game and then continues live.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess

from chess_game import ChessGame, GameControls

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "ui" / "index.html"

STARTING_FEN = chess.STARTING_FEN


class GameHost:
    """
    Owns the current game: its thread, its controls, its event history and a
    snapshot of where things stand. Also the fan-out point for browsers.
    """

    def __init__(self, build_players, move_delay: float = 0.6):
        """
        build_players  a zero-argument callable returning
                       (white_player, black_player, cleanup_callable).
                       It is called afresh for every new game, which is how
                       "New game" gets a clean Stockfish process.
        """
        self.build_players = build_players
        self.default_delay = move_delay

        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

        self._events: list[dict] = []
        self._seq = 0

        self._thread: threading.Thread | None = None
        self._controls: GameControls | None = None
        self._cleanup = None

        self.state = self._empty_state()

    # ----------------------------------------------------------- snapshots

    def _empty_state(self) -> dict:
        return {
            "fen": STARTING_FEN,
            "move_list": [],
            "captured": {"white": [], "black": []},
            "turn": "white",
            "in_check": False,
            "move_number": 1,
            "players": {"white": "Jev", "black": "Stockfish"},
            "status": "idle",          # idle | thinking | over | error | stopped
            "thinking": None,          # "white" / "black" / None
            "result": None,
            "reason": None,
            "error": None,
            "decision": None,
            "paused": False,
            "move_delay": self.default_delay,
        }

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self.state)
            state["seq"] = self._seq
            if self._controls is not None:
                state["paused"] = self._controls.paused
                state["move_delay"] = self._controls.move_delay
            return state

    def history(self, since: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self._events if e["seq"] > since]

    # -------------------------------------------------------------- events

    def publish(self, event: dict) -> None:
        """Called from the game thread. Updates state, then fans out."""
        with self._lock:
            self._seq += 1
            event = dict(event)
            event["seq"] = self._seq
            event["time"] = time.time()
            self._apply_to_state(event)
            self._events.append(event)
            # Keep memory bounded on very long games.
            if len(self._events) > 2000:
                del self._events[:500]
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:  # a browser that stopped reading
                pass

    def _apply_to_state(self, event: dict) -> None:
        """Fold one event into the snapshot. Caller holds the lock."""
        kind = event.get("type")

        for key in ("fen", "move_list", "captured", "turn", "in_check",
                    "move_number"):
            if key in event:
                self.state[key] = event[key]

        if kind == "game_started":
            self.state.update(
                players=event.get("players", self.state["players"]),
                status="thinking",
                result=None,
                reason=None,
                error=None,
                decision=None,
            )
        elif kind == "turn_started":
            self.state["thinking"] = event.get("side")
            self.state["status"] = "thinking"
        elif kind == "move_played":
            self.state["thinking"] = None
            if event.get("decision"):
                self.state["decision"] = event["decision"]
        elif kind == "game_over":
            self.state.update(
                status="over",
                thinking=None,
                result=event.get("result"),
                reason=event.get("reason"),
            )
        elif kind == "game_stopped":
            self.state.update(status="stopped", thinking=None)
        elif kind == "error":
            self.state.update(
                status="error", thinking=None, error=event.get("message")
            )

    # ---------------------------------------------------------- subscribers

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # ------------------------------------------------------- game lifecycle

    def start_new_game(self) -> None:
        """Stop whatever is running, then start a clean game."""
        self.stop_game()

        with self._lock:
            self._events.clear()
            self._seq = 0
            self.state = self._empty_state()
            self._controls = GameControls(move_delay=self.default_delay)
            controls = self._controls

        def run():
            cleanup = None
            try:
                white, black, cleanup = self.build_players()
                self._cleanup = cleanup
                game = ChessGame(
                    white=white,
                    black=black,
                    on_event=self.publish,
                    controls=controls,
                )
                game.run()
            except Exception as error:
                logger.exception("Game thread crashed")
                self.publish(
                    {
                        "type": "error",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:
                        logger.exception("Cleanup failed")
                self._cleanup = None

        thread = threading.Thread(target=run, name="chess-game", daemon=True)
        self._thread = thread
        thread.start()

    def stop_game(self, timeout: float = 20.0) -> None:
        """Ask the game thread to finish, and wait for it."""
        controls, thread = self._controls, self._thread
        if controls is not None:
            controls.stop()
        if thread is not None and thread.is_alive():
            # The thread can be inside a slow LLM or engine call; it will exit
            # at the next move boundary.
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Game thread did not stop within %.0fs", timeout)
        self._thread = None

    def pause(self) -> None:
        if self._controls:
            self._controls.pause()
            self.publish({"type": "paused"})

    def resume(self) -> None:
        if self._controls:
            self._controls.resume()
            self.publish({"type": "resumed"})

    def set_delay(self, seconds: float) -> None:
        if self._controls:
            self._controls.set_move_delay(seconds)
            self.publish({"type": "speed", "move_delay": self._controls.move_delay})


# --------------------------------------------------------------- HTTP layer

class Handler(BaseHTTPRequestHandler):
    host: GameHost = None          # set by serve()
    server_version = "ChessArena/1.0"

    # Quieter logs: one line per request at DEBUG, not INFO on stderr.
    def log_message(self, fmt, *args):
        logger.debug("http %s", fmt % args)

    # ------------------------------------------------------------ responses

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    # ----------------------------------------------------------------- GET

    def do_GET(self):
        url = urlparse(self.path)

        if url.path in ("/", "/index.html"):
            try:
                body = INDEX_PATH.read_bytes()
            except OSError:
                self._send(500, b"ui/index.html is missing", "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return

        if url.path == "/api/state":
            self._send_json(
                {
                    "state": self.host.snapshot(),
                    "history": self.host.history(0),
                }
            )
            return

        if url.path == "/events":
            self._stream_events(url)
            return

        self._send(404, b"not found", "text/plain")

    def _stream_events(self, url) -> None:
        """One long-lived Server-Sent Events connection."""
        params = parse_qs(url.query)
        try:
            since = int(params.get("since", ["0"])[0])
        except ValueError:
            since = 0

        subscriber = self.host.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # 1. the full picture, so a reload is indistinguishable from a
            #    fresh start
            self._write_event(
                {
                    "type": "snapshot",
                    "state": self.host.snapshot(),
                    "history": self.host.history(since),
                }
            )

            # 2. everything from here on, live
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    # A comment line keeps proxies and browsers from timing out.
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the tab was closed; entirely normal
        finally:
            self.host.unsubscribe(subscriber)

    def _write_event(self, payload: dict) -> None:
        data = json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ---------------------------------------------------------------- POST

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}

        if url.path == "/api/new":
            self.host.start_new_game()
            self._send_json({"ok": True})
            return

        if url.path == "/api/pause":
            self.host.pause()
            self._send_json({"ok": True})
            return

        if url.path == "/api/resume":
            self.host.resume()
            self._send_json({"ok": True})
            return

        if url.path == "/api/speed":
            try:
                self.host.set_delay(float(body.get("move_delay", 0.6)))
            except (TypeError, ValueError):
                self._send_json({"ok": False, "error": "bad value"}, code=400)
                return
            self._send_json({"ok": True})
            return

        self._send(404, b"not found", "text/plain")


def serve(host: GameHost, port: int = 8765, open_browser: bool = True):
    """Start the server and block until Ctrl+C."""
    Handler.host = host
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True

    url = f"http://127.0.0.1:{port}/"
    print(f"ChessArena UI on {url}   (Ctrl+C to stop)")

    if open_browser:
        # Open in a thread so a slow browser launch does not delay the server.
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    host.start_new_game()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        host.stop_game()
        httpd.shutdown()
        httpd.server_close()
        print("Stopped.")
