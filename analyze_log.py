"""Usage: python analyze_log.py   (reads logs/game.log, last game only)

Prints one line per Jev turn: the best score the filter found, the score of
the move Jev actually played, how deep the search got and how many moves Jev
was offered. The last column flags turns where Jev had no real choice, which
is just as interesting as turns where Jev chose badly.
"""
import re

lines = open("logs/game.log").read().splitlines()
start = max(i for i, l in enumerate(lines) if "NEW GAME" in l)
lines = lines[start:]


def parse_score(text):
    text = text.strip()
    if "get mated" in text:
        return -99999
    if "we mate" in text:
        return 99999
    return int(text.replace("cp", "").replace("+", ""))


turn = 0
assessed = {}
offered = []
fen = ""
depth = 0
deviations = 0
forced = 0

print(f"{'turn':>4} {'best':>7} {'picked':>7} {'d':>2} {'opts':>4}  move     position")
for l in lines:
    if "Tactical check |" in l:
        parts = [p.strip() for p in l.split("|")]
        # parts: date/level, "Tactical check", san, score, opp checks, reason
        assessed[parts[3]] = (parse_score(parts[4]), parts[-1])
    elif "Tactical summary |" in l:
        match = re.search(r"depth (\d+)", l)
        depth = int(match.group(1)) if match else 0
    elif "Jev candidates:" in l:
        offered = [m.strip() for m in l.split("candidates:")[1].split(",")]
    elif "Jev evaluating position:" in l:
        fen = l.split("position:")[1].strip()
    elif "Jev selected:" in l:
        turn += 1
        move = l.split("selected:")[1].strip()
        scores = {san: score for san, (score, _) in assessed.items()}
        best = max(scores.values()) if scores else 0
        picked = scores.get(move, 0)
        flags = []
        if picked < best:
            flags.append("<-- worse than best")
            deviations += 1
        if len(offered) <= 1:
            flags.append("<-- no choice offered")
            forced += 1
        print(
            f"{turn:>4} {best:>+7} {picked:>+7} {depth:>2} {len(offered):>4}"
            f"  {move:<8} {fen[:38]} {' '.join(flags)}"
        )
        assessed = {}
        offered = []

print()
print(f"turns: {turn} | Jev chose a lower-scoring move on {deviations} "
      f"| Jev had only one option on {forced}")
for l in lines:
    if "Result:" in l:
        print(l.split("|")[-1].strip())