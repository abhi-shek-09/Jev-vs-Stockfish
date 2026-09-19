from typesafe_sdk import Choice, TypeSafeClient


client = TypeSafeClient()

response = client.system_one(
    model="jev-latest",
    state="""
    A chess player has developed both knights and bishops.
    The king is still in the center.
    The opponent is beginning an attack against the king.
    """,
    questions={
        "priority": Choice(
            instructions="What should the player prioritize?",
            criteria={
                "attack": "Look for an immediate attack against the opponent.",
                "development": "Continue developing pieces.",
                "king_safety": "Improve the safety of the king.",
                "material_gain": "Look for an opportunity to win material.",
            },
        )
    },
)

print(response)