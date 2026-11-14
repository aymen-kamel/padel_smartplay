"""Demo: replay a simulated padel match through the score engine.

    python scripts/demo_score.py [--mode advantage|golden_point|star_point] [--seed 7] [--p 0.55]

Team A wins each point with probability p. The point is only to show the engine at work:
score after each point, who serves, which team is on the NEAR side of the camera, and
the moment players change ends (FIP Rule 5).
"""

from __future__ import annotations

import argparse
import random

from smartplay.scoring.automaton import (
    DeuceMode,
    MatchRules,
    apply_point,
    format_score,
    initial_state,
    near_team,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="advantage", choices=[m.value for m in DeuceMode])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.55, help="probability that team A wins a point")
    ap.add_argument("--max-lines", type=int, default=60)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rules = MatchRules(deuce_mode=DeuceMode(args.mode))
    state = initial_state(rules)
    print(f"Rules: {args.mode} | best of 3 sets | tie-break to 7 | p(A wins a point) = {args.p}\n")
    print(f"{'#':>4}  {'point':<5} {'score':<24} {'server':<7} {'NEAR team':<9} note")

    shown = 0
    n = 0
    while not state.is_over:
        winner = "A" if rng.random() < args.p else "B"
        before = state
        state = apply_point(state, winner, rules)
        n += 1
        notes = []
        if state.ends_swapped != before.ends_swapped:
            notes.append("<< CHANGE OF ENDS")
        if state.completed_sets != before.completed_sets:
            notes.append("<< SET OVER")
        elif state.games != before.games and not state.is_over:
            notes.append("game")
        if state.tiebreak and not before.tiebreak:
            notes.append("<< TIE-BREAK STARTS")
        important = bool(notes) and any(x.startswith("<<") for x in notes)
        if shown < args.max_lines and (important or n <= 12):
            print(
                f"{n:>4}  {winner:<5} {format_score(state):<24} {state.server:<7} "
                f"{near_team(state):<9} {' '.join(notes)}"
            )
            shown += 1

    print(f"\nMatch over after {state.points_played} points.")
    print(f"Final: {format_score(state)}")


if __name__ == "__main__":
    main()
