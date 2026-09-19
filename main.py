from jev_player import JevPlayer
from stockfish_player import StockfishPlayer
from chess_game import ChessGame
from logging_config import setup_logging


def main():
    setup_logging()
    jev = JevPlayer()

    stockfish = StockfishPlayer(
        engine_path="/opt/homebrew/bin/stockfish"
    )

    game = ChessGame(
        white=jev,
        black=stockfish,
    )

    try:
        game.run()
    finally:
        stockfish.close()

if __name__ == "__main__":
    main()