import logging
import chess

logger = logging.getLogger(__name__)


class ChessGame:
    def __init__(self, white, black, max_moves: int = 200):
        self.board = chess.Board()
        self.white = white
        self.black = black
        # Safety valve so a shuffling game cannot run forever.
        self.max_moves = max_moves

    def run(self):
        logger.info("========== NEW GAME ==========")
        logger.info("Game started")
        logger.info("Initial position: %s", self.board.fen())
        move_number = 1

        # claim_draw=True also ends the game on threefold repetition and the
        # fifty-move rule. Without it a drawn position just keeps going, and
        # against a strong opponent a draw is a result worth having.
        while not self.board.is_game_over(claim_draw=True):
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

            move = player.get_move(self.board)
            san = self.board.san(move=move)

            logger.info(
                "Turn %d | %s played %s",
                move_number,
                player_name,
                san,
            )

            print(f"{move_number}. {player_name} : {san}")

            self.board.push(move=move)

            if self.board.turn == chess.WHITE:
                move_number += 1

        result = self.board.result(claim_draw=True)

        logger.info("Game over")
        logger.info("Result: %s", result)
        logger.info("Final position: %s", self.board.fen())
        logger.info("Moves: %s", self._move_list())

        print()
        print("Game Over")
        print(f"Result: {result}")

    def _move_list(self) -> str:
        """The whole game in SAN, handy when reading the log afterwards."""
        replay = self.board.root()
        parts = []
        for index, move in enumerate(self.board.move_stack):
            if index % 2 == 0:
                parts.append(f"{index // 2 + 1}.")
            parts.append(replay.san(move))
            replay.push(move)
        return " ".join(parts)