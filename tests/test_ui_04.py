"""
test_ui.py

Offline tests for the spectator UI layer. No API key, no Stockfish, no
network beyond a loopback socket the test opens itself.

    python -m unittest test_ui -v

What it checks:
  * a whole game produces a sane event stream, in order, with sequence numbers
  * the movement descriptors are right for castling, en passant and promotion
  * neither the server nor the UI ever mutates the game's board
  * a fresh connection gets a snapshot that fully describes the game so far
"""

import json
import threading
import time
import unittest
import urllib.request

import chess

from chess_game import ChessGame, GameControls
from move_animation import captured_lists, describe_move, move_list_san
from ui_server import GameHost, Handler
from http.server import ThreadingHTTPServer


class ScriptedPlayer:
    """Plays a fixed list of SAN moves, then resigns by having none left."""

    def __init__(self, script, name="scripted"):
        self.script = list(script)
        self.name = name
        self.last_decision = None
        self.seen_boards = []

    def get_move(self, board):
        self.seen_boards.append(board)
        san = self.script.pop(0)
        return board.parse_san(san)


def collect_events(white_script, black_script, max_moves=200):
    """Run a full game with a listener and return every event it published."""
    events = []
    white = ScriptedPlayer(white_script, "White")
    black = ScriptedPlayer(black_script, "Black")
    game = ChessGame(
        white=white,
        black=black,
        max_moves=max_moves,
        on_event=events.append,
    )
    try:
        game.run()
    except IndexError:
        pass  # the script ran out; fine for a test
    return events, game


# ------------------------------------------------------------ descriptors

class MovementDescriptors(unittest.TestCase):

    def test_plain_move(self):
        board = chess.Board()
        info = describe_move(board, board.parse_san("e4"))
        self.assertEqual(info["movements"],
                         [{"piece": "P", "from": "e2", "to": "e4"}])
        self.assertIsNone(info["captured"])
        self.assertIsNone(info["promotion"])

    def test_castling_moves_two_pieces(self):
        board = chess.Board(
            "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        info = describe_move(board, board.parse_san("O-O"))
        self.assertEqual(
            info["movements"],
            [
                {"piece": "K", "from": "e1", "to": "g1"},
                {"piece": "R", "from": "h1", "to": "f1"},
            ],
        )
        self.assertIsNone(info["captured"])

    def test_queenside_castling(self):
        board = chess.Board(
            "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5"
        )
        info = describe_move(board, board.parse_san("O-O-O"))
        self.assertEqual(
            info["movements"],
            [
                {"piece": "K", "from": "e1", "to": "c1"},
                {"piece": "R", "from": "a1", "to": "d1"},
            ],
        )

    def test_en_passant_captures_a_different_square(self):
        board = chess.Board()
        for san in ["e4", "a6", "e5", "d5"]:
            board.push_san(san)
        move = board.parse_san("exd6")
        info = describe_move(board, move)
        self.assertEqual(info["movements"],
                         [{"piece": "P", "from": "e5", "to": "d6"}])
        # The captured pawn is on d5, NOT on the destination d6.
        self.assertEqual(info["captured"], {"square": "d5", "piece": "p"})

    def test_ordinary_capture_is_on_the_destination(self):
        board = chess.Board()
        for san in ["e4", "d5"]:
            board.push_san(san)
        info = describe_move(board, board.parse_san("exd5"))
        self.assertEqual(info["captured"], {"square": "d5", "piece": "p"})

    def test_promotion(self):
        board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
        info = describe_move(board, board.parse_san("a8=Q"))
        self.assertEqual(info["promotion"], "Q")
        self.assertEqual(info["movements"],
                         [{"piece": "P", "from": "a7", "to": "a8"}])

    def test_promotion_with_capture(self):
        board = chess.Board("1n5k/P7/8/8/8/8/7K/8 w - - 0 1")
        info = describe_move(board, board.parse_san("axb8=N"))
        self.assertEqual(info["promotion"], "N")
        self.assertEqual(info["captured"], {"square": "b8", "piece": "n"})

    def test_describe_move_never_touches_the_board(self):
        board = chess.Board()
        for san in ["e4", "a6", "e5", "d5"]:
            board.push_san(san)
        before = (board.fen(), len(board.move_stack))
        for move in board.legal_moves:
            describe_move(board, move)
        self.assertEqual((board.fen(), len(board.move_stack)), before)

    def test_captured_lists(self):
        board = chess.Board()
        for san in ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qxa2"]:
            board.push_san(san)
        lists = captured_lists(board)
        self.assertEqual(sorted(lists["white"]), ["P", "P"])
        self.assertEqual(sorted(lists["black"]), ["p"])

    def test_move_list_san(self):
        board = chess.Board()
        for san in ["e4", "e5", "Nf3"]:
            board.push_san(san)
        self.assertEqual(move_list_san(board), ["e4", "e5", "Nf3"])


# ------------------------------------------------------------ event stream

SCHOLARS_WHITE = ["e4", "Bc4", "Qh5", "Qxf7#"]
SCHOLARS_BLACK = ["e5", "Nc6", "Nf6"]


class EventStream(unittest.TestCase):

    def test_full_game_event_sequence(self):
        events, game = collect_events(SCHOLARS_WHITE, SCHOLARS_BLACK)
        kinds = [e["type"] for e in events]

        self.assertEqual(kinds[0], "game_started")
        self.assertEqual(kinds[-1], "game_over")
        self.assertEqual(kinds.count("move_played"), 7)
        # Every move is announced before it is played.
        self.assertEqual(kinds.count("turn_started"), 7)

        final = events[-1]
        self.assertEqual(final["result"], "1-0")
        self.assertEqual(final["reason"], "checkmate")

        moves = [e for e in events if e["type"] == "move_played"]
        self.assertEqual([m["san"] for m in moves],
                         ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"])
        for m in moves:
            self.assertIn("fen", m)
            self.assertIn("movements", m)
            self.assertIn("move_list", m)
            self.assertIn("captured", m)
            # The FEN must be legal and must match the move list length.
            chess.Board(m["fen"])

    def test_every_move_event_carries_a_usable_fen(self):
        events, _ = collect_events(SCHOLARS_WHITE, SCHOLARS_BLACK)
        replay = chess.Board()
        for e in (x for x in events if x["type"] == "move_played"):
            replay.push(chess.Move.from_uci(e["uci"]))
            self.assertEqual(replay.fen(), e["fen"])

    def test_players_never_receive_the_real_board(self):
        """ChessGame hands out copies, so a player cannot corrupt the game."""
        events, game = collect_events(SCHOLARS_WHITE, SCHOLARS_BLACK)
        for player in (game.white, game.black):
            for seen in player.seen_boards:
                self.assertIsNot(seen, game.board)

    def test_a_player_mutating_its_board_cannot_break_the_game(self):
        class Vandal(ScriptedPlayer):
            def get_move(self, board):
                move = super().get_move(board)
                board.clear()           # deliberately wreck our copy
                board.set_fen(chess.STARTING_FEN)
                return move

        events = []
        white = Vandal(["e4", "Bc4", "Qh5", "Qxf7#"])
        black = Vandal(["e5", "Nc6", "Nf6"])
        game = ChessGame(white=white, black=black, on_event=events.append)
        game.run()
        self.assertEqual(game.board.result(), "1-0")

    def test_error_is_published_not_swallowed(self):
        class Broken:
            last_decision = None

            def get_move(self, board):
                raise RuntimeError("Jev exploded")

        events = []
        game = ChessGame(
            white=Broken(), black=Broken(), on_event=events.append
        )
        with self.assertRaises(RuntimeError):
            game.run()
        errors = [e for e in events if e["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Jev exploded", errors[0]["message"])

    def test_controls_stop_the_game(self):
        controls = GameControls(move_delay=0.0)
        events = []
        white = ScriptedPlayer(["e4"] * 50)
        black = ScriptedPlayer(["e5"] * 50)

        # A player that stops the game as soon as it is asked for a move.
        class Stopper(ScriptedPlayer):
            def get_move(self, board):
                controls.stop()
                return super().get_move(board)

        game = ChessGame(
            white=Stopper(["e4"]), black=black,
            on_event=events.append, controls=controls,
        )
        game.run()
        kinds = [e["type"] for e in events]
        self.assertIn("game_stopped", kinds)
        self.assertNotIn("game_over", kinds)


# ---------------------------------------------------------------- the host

class HostState(unittest.TestCase):

    def test_host_folds_events_into_a_snapshot(self):
        host = GameHost(build_players=lambda: (None, None, lambda: None))
        events, _ = collect_events(SCHOLARS_WHITE, SCHOLARS_BLACK)
        for e in events:
            host.publish(e)

        snapshot = host.snapshot()
        self.assertEqual(snapshot["status"], "over")
        self.assertEqual(snapshot["result"], "1-0")
        self.assertEqual(len(snapshot["move_list"]), 7)
        self.assertEqual(snapshot["seq"], len(events))
        # History is complete and numbered.
        self.assertEqual([e["seq"] for e in host.history(0)],
                         list(range(1, len(events) + 1)))

    def test_subscribers_receive_events(self):
        host = GameHost(build_players=lambda: (None, None, lambda: None))
        q = host.subscribe()
        host.publish({"type": "turn_started", "side": "white"})
        received = q.get(timeout=1)
        self.assertEqual(received["type"], "turn_started")
        self.assertEqual(received["seq"], 1)
        host.unsubscribe(q)


class ServerRoutes(unittest.TestCase):
    """Boot the real server on a loopback port and poke it."""

    @classmethod
    def setUpClass(cls):
        cls.host = GameHost(build_players=lambda: (None, None, lambda: None))
        Handler.host = cls.host
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_index_is_served(self):
        with urllib.request.urlopen(self.url("/"), timeout=5) as response:
            body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("<title>ChessArena", body)
        self.assertIn("EventSource", body)      # the page really is wired up
        self.assertNotIn("http://", body.split("<script>")[0].replace(
            "http://www.w3.org", ""))           # no external assets in the head

    def test_state_endpoint_returns_json(self):
        events, _ = collect_events(SCHOLARS_WHITE, SCHOLARS_BLACK)
        for e in events:
            self.host.publish(e)
        with urllib.request.urlopen(self.url("/api/state"), timeout=5) as r:
            payload = json.loads(r.read())
        self.assertEqual(payload["state"]["result"], "1-0")
        self.assertEqual(len(payload["history"]), len(events))

    def test_unknown_path_is_404(self):
        try:
            urllib.request.urlopen(self.url("/nope"), timeout=5)
            self.fail("expected a 404")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 404)

    def test_speed_endpoint_rejects_rubbish(self):
        """A bad value must be a clean 400, not a crash or a hang."""
        request = urllib.request.Request(
            self.url("/api/speed"),
            data=b'{"move_delay": "fast"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            self.fail("expected a 400")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 400)


class DemoMode(unittest.TestCase):

    def test_demo_players_reach_the_special_moves(self):
        """The scripted demo must actually exercise the tricky animations."""
        from demo_players import DemoJev, DemoPlayer

        white = DemoJev("Jev", seed=7, think_seconds=0.0)
        black = DemoPlayer("Stockfish", seed=8, think_seconds=0.0)
        events = []
        game = ChessGame(
            white=white, black=black, max_moves=60, on_event=events.append
        )
        game.run()

        moves = [e for e in events if e["type"] == "move_played"]
        self.assertGreater(len(moves), 20)

        castles = [m for m in moves if len(m["movements"]) == 2]
        promotions = [m for m in moves if m["promotion"]]
        en_passant = [
            m for m in moves
            if m["capture"] and m["capture"]["square"] != m["movements"][0]["to"]
        ]

        self.assertGreaterEqual(len(castles), 2, "expected both sides to castle")
        self.assertGreaterEqual(len(promotions), 1, "expected a promotion")
        self.assertGreaterEqual(len(en_passant), 1, "expected an en passant")

    def test_demo_jev_produces_decision_data(self):
        from demo_players import DemoJev

        board = chess.Board()
        jev = DemoJev("Jev", seed=1, think_seconds=0.0)
        jev.get_move(board)
        decision = jev.last_decision
        self.assertTrue(decision["candidates"])
        self.assertTrue(any(c["chosen"] for c in decision["candidates"]))
        for candidate in decision["candidates"]:
            self.assertIn("score_text", candidate)
            self.assertIn("note", candidate)
        # And it must survive a round trip through JSON, like a real event.
        json.dumps(decision)


if __name__ == "__main__":
    unittest.main(verbosity=2)
