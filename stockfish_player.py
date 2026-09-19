import chess
import chess.engine
import logging

logger = logging.getLogger(__name__)
class StockfishPlayer:
    def __init__(self, engine_path: str, depth: int = 12):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth

    def get_move(self, board):
        result = self.engine.play(
            board,
            chess.engine.Limit(
                depth=self.depth
            )
        ) # result = <PlayResult move=e2e4 draw_offered=False resigned=False statistical=False evaluation=None>

        logger.info(
            "Stockfish selected: %s",
            board.san(result.move),
        )
        
        return result.move

    def close(self):
        self.engine.quit()