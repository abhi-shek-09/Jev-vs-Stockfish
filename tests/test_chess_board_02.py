import os
import chess
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

def main():
    # 1. Set up the chess position
    board = chess.Board()

    # 1. e4 e5
    board.push_san("e4")
    board.push_san("e5")

    print("Position (White to move):")
    print(board)
    print()

    # 2. Build candidate moves from LEGAL moves only

    candidate_sans = ["Nf3", "Bc4", "d4", "Nc3"]
    candidates = {}

    for san in candidate_sans:
        move = board.parse_san(san)

        if move not in board.legal_moves:
            raise RuntimeError(f"{san} is not legal")

        candidates[san] = move

    print("Candidate moves offered to Jev:")
    for san in candidate_sans:
        print(f"  {san}")
    print()

    # 3. Ask Jev to choose

    client = TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"]
    )

    state = {
        "fen": board.fen(),
        "board": str(board),
        "side_to_move": "White",
        "candidate_moves": candidate_sans,
    }

    question = Choice(
        instructions=(
            "Choose the best chess move for White from the candidate moves. "
            "Consider king safety, piece development, and control of the center."
        ),
        criteria={
            "Nf3": "Develops the king knight and attacks the e5 pawn.",
            "Bc4": "Develops the bishop to an active diagonal toward the black king.",
            "d4": "Immediately challenges the black pawn in the center.",
            "Nc3": "Develops the queen knight and supports central control.",
        },
    )

    response = client.system_one(
        model="jev-latest",
        state=state,
        questions={
            "best_move": question,
        },
    )

    # 4. Read Jev's typed Choice answer

    answer = response.answers["best_move"]
    chosen_san = answer.choice

    print("Jev's answer:")
    print(chosen_san)
    print()

    # 5. Validate Jev's answer

    if chosen_san not in candidates:
        raise RuntimeError(
            f"Jev returned an unexpected move: {chosen_san}"
        )

    chosen_move = candidates[chosen_san]

    if chosen_move not in board.legal_moves:
        raise RuntimeError(
            f"Jev selected an illegal move: {chosen_san}"
        )

    # 6. Apply the move

    board.push(chosen_move)
    print(f"Jev chose: {chosen_san}")
    print()
    print("Position after Jev's move:")
    print(board)
    print()
    print("FEN:")
    print(board.fen())

if __name__ == "__main__":
    main()