"""
main.py

Two ways to run ChessArena:

    python main.py                 console, exactly as before
    python main.py --ui            console PLUS a spectator UI in the browser
    python main.py --ui --demo     the UI with stub players (no key, no engine)

The --demo flag only makes sense with --ui, but it works on the console too if
you want to watch the stubs play there.
"""

import argparse

from chess_game import ChessGame
from logging_config import setup_logging

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def build_real_players():
    """
    Create a fresh Jev and a fresh Stockfish process.

    Returns (white, black, cleanup). `cleanup` closes the engine, and the UI
    calls it whenever a game ends or is replaced, so no engine process leaks.
    Imports happen inside the function so that --demo needs neither the
    TypeSafe SDK nor a Stockfish binary.
    """
    from jev_player import JevPlayer
    from stockfish_player import StockfishPlayer

    jev = JevPlayer()
    stockfish = StockfishPlayer(engine_path=STOCKFISH_PATH)
    return jev, stockfish, stockfish.close


def build_demo_players_factory(think_seconds: float):
    """The same shape as build_real_players, but with stubs and no cleanup."""
    from demo_players import build_demo_players

    def factory():
        white, black = build_demo_players(think_seconds=think_seconds)
        return white, black, lambda: None

    return factory


def run_console(build_players):
    white, black, cleanup = build_players()
    game = ChessGame(white=white, black=black)
    try:
        game.run()
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="ChessArena: Jev vs Stockfish")
    parser.add_argument(
        "--ui", action="store_true", help="watch the game in a browser"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use stub players (no TypeSafe key, no Stockfish needed)",
    )
    parser.add_argument("--port", type=int, default=8765, help="UI port")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the UI server but do not open a browser",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="seconds to pause between moves in the UI (0-5)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.demo:
        build_players = build_demo_players_factory(think_seconds=0.25)
    else:
        build_players = build_real_players

    if not args.ui:
        run_console(build_players)
        return

    # Imported here so the console path does not pay for the server at all.
    from ui_server import GameHost, serve

    host = GameHost(build_players=build_players, move_delay=args.delay)
    serve(host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()