"""Deterministic padel score engine.

Implements the FIP Rules of Padel (review applicable 01.01.2026):
  Rule 1  score in a game (Advantage, Star Point, Golden Point), sets, tie-break,
          alternative methods (mini set, match tie-break to 7, super tie-break to 10)
  Rule 5  changes of ends: after the 1st, 3rd and every subsequent odd game of a set,
          and after every 6 points in a tie-break

The engine is pure: ``apply_point`` takes a state and returns a new one. It knows nothing
about video. Teams are "A" and "B" for the whole match; which end each team occupies is
tracked by ``ends_swapped``.

Assumptions where the rulebook is silent (documented, covered by tests):
  * Service alternates between the pairs game by game across sets; a tie-break counts as one
    game served first by the pair whose turn it was (FIP tie-break rule 5 is consistent).
  * After a tie-break the pairs change ends relative to the ends at which the tie-break
    started (tennis convention: the tie-break counts as one, odd, game).
  * Only the serving *pair* is tracked, not the individual server.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

Team = Literal["A", "B"]


def other(team: Team) -> Team:
    return "B" if team == "A" else "A"


class DeuceMode(str, Enum):
    ADVANTAGE = "advantage"
    GOLDEN_POINT = "golden_point"
    STAR_POINT = "star_point"


class FinalSet(str, Enum):
    FULL = "full"  # a normal set
    MATCH_TIEBREAK = "match_tiebreak"  # tie-break to 7 replaces the deciding set
    SUPER_TIEBREAK = "super_tiebreak"  # tie-break to 10 replaces the deciding set


@dataclass(frozen=True)
class MatchRules:
    deuce_mode: DeuceMode = DeuceMode.ADVANTAGE
    games_per_set: int = 6  # 4 = mini set
    sets_to_win: int = 2  # best of three
    tiebreak_points: int = 7
    final_set: FinalSet = FinalSet.FULL
    final_set_tiebreak: bool = True  # False: deciding set is played out by two clear games
    first_server: Team = "A"

    def __post_init__(self) -> None:
        if self.games_per_set < 1 or self.sets_to_win < 1 or self.tiebreak_points < 1:
            raise ValueError("games_per_set, sets_to_win and tiebreak_points must be >= 1")


@dataclass(frozen=True)
class ScoreState:
    server: Team
    completed_sets: tuple[tuple[int, int], ...] = ()
    games: tuple[int, int] = (0, 0)
    points: tuple[int, int] = (0, 0)  # raw points won in the current game, (A, B)
    tiebreak: bool = False
    tiebreak_target: int = 0
    tiebreak_replaces_set: bool = False
    tiebreak_first_server: Team | None = None
    tiebreak_start_ends: bool = False
    ends_swapped: bool = False  # False = the ends the pairs had at the start of the match
    winner: Team | None = None
    points_played: int = 0

    @property
    def sets_won(self) -> tuple[int, int]:
        a = sum(1 for sa, sb in self.completed_sets if sa > sb)
        return a, len(self.completed_sets) - a

    @property
    def is_over(self) -> bool:
        return self.winner is not None


def initial_state(rules: MatchRules) -> ScoreState:
    state = ScoreState(server=rules.first_server)
    return _maybe_start_deciding_tiebreak(state, rules)


def apply_point(state: ScoreState, winner: Team, rules: MatchRules) -> ScoreState:
    """Return the state after ``winner`` wins the next point."""
    if winner not in ("A", "B"):
        raise ValueError(f"winner must be 'A' or 'B', got {winner!r}")
    if state.is_over:
        raise ValueError("the match is already over")

    i = 0 if winner == "A" else 1
    before_w, before_l = state.points[i], state.points[1 - i]
    pts = list(state.points)
    pts[i] += 1
    won, lost = pts[i], pts[1 - i]
    played = state.points_played + 1

    if state.tiebreak:
        if won >= state.tiebreak_target and won - lost >= 2:
            return _game_won(replace(state, points_played=played), winner, rules, (pts[0], pts[1]))
        total = pts[0] + pts[1]
        first = state.tiebreak_first_server
        assert first is not None
        server = first if ((total + 1) // 2) % 2 == 0 else other(first)
        ends = state.tiebreak_start_ends ^ ((total // 6) % 2 == 1)
        return replace(
            state, points=(pts[0], pts[1]), server=server, ends_swapped=ends, points_played=played
        )

    deciding_point = (
        before_w == before_l
        and before_w >= 3
        and (
            rules.deuce_mode is DeuceMode.GOLDEN_POINT
            or (rules.deuce_mode is DeuceMode.STAR_POINT and before_w >= 5)  # third deuce
        )
    )
    if won >= 4 and (won - lost >= 2 or deciding_point):
        return _game_won(replace(state, points_played=played), winner, rules, (pts[0], pts[1]))
    return replace(state, points=(pts[0], pts[1]), points_played=played)


def replay(winners: Iterable[Team], rules: MatchRules) -> list[ScoreState]:
    """Apply a sequence of point winners; returns the state after each point."""
    state = initial_state(rules)
    out: list[ScoreState] = []
    for w in winners:
        state = apply_point(state, w, rules)
        out.append(state)
    return out


def _game_won(
    state: ScoreState, winner: Team, rules: MatchRules, final_points: tuple[int, int]
) -> ScoreState:
    i = 0 if winner == "A" else 1
    was_tiebreak = state.tiebreak

    if was_tiebreak and state.tiebreak_replaces_set:
        set_score: tuple[int, int] = final_points  # e.g. (10, 8)
        set_over = True
        games = state.games
    else:
        g = list(state.games)
        g[i] += 1
        games = (g[0], g[1])
        set_score = games
        if was_tiebreak:
            set_over = True
        else:
            set_over = g[i] >= rules.games_per_set and g[i] - g[1 - i] >= 2

    # Service: the pair that did not start this game serves next.
    game_starter = state.tiebreak_first_server if was_tiebreak else state.server
    assert game_starter is not None
    next_server = other(game_starter)

    # Ends: change after every odd game of the set; a tie-break counts as one (odd) game.
    if was_tiebreak:
        ends = not state.tiebreak_start_ends
    else:
        ends = state.ends_swapped ^ ((games[0] + games[1]) % 2 == 1)

    base = replace(
        state,
        points=(0, 0),
        server=next_server,
        ends_swapped=ends,
        tiebreak=False,
        tiebreak_target=0,
        tiebreak_replaces_set=False,
        tiebreak_first_server=None,
    )

    if set_over:
        new = replace(base, completed_sets=state.completed_sets + (set_score,), games=(0, 0))
        if new.sets_won[i] >= rules.sets_to_win:
            return replace(new, winner=winner)
        return _maybe_start_deciding_tiebreak(new, rules)

    new = replace(base, games=games)
    n = rules.games_per_set
    if games == (n, n) and (rules.final_set_tiebreak or not _is_deciding_set(new, rules)):
        return replace(
            new,
            tiebreak=True,
            tiebreak_target=rules.tiebreak_points,
            tiebreak_first_server=new.server,
            tiebreak_start_ends=new.ends_swapped,
        )
    return new


def _is_deciding_set(state: ScoreState, rules: MatchRules) -> bool:
    return state.sets_won == (rules.sets_to_win - 1, rules.sets_to_win - 1)


def _maybe_start_deciding_tiebreak(state: ScoreState, rules: MatchRules) -> ScoreState:
    if rules.final_set is FinalSet.FULL or not _is_deciding_set(state, rules):
        return state
    target = 7 if rules.final_set is FinalSet.MATCH_TIEBREAK else 10
    return replace(
        state,
        tiebreak=True,
        tiebreak_target=target,
        tiebreak_replaces_set=True,
        tiebreak_first_server=state.server,
        tiebreak_start_ends=state.ends_swapped,
    )


# ---------------------------------------------------------------------------- display

_LABELS = ("0", "15", "30", "40")


def point_labels(state: ScoreState) -> tuple[str, str]:
    """Spoken score of the current game for (A, B), e.g. ("40", "AD")."""
    a, b = state.points
    if state.tiebreak:
        return str(a), str(b)
    if a >= 3 and b >= 3:
        if a == b:
            return "40", "40"
        return ("AD", "40") if a > b else ("40", "AD")
    return _LABELS[min(a, 3)], _LABELS[min(b, 3)]


def format_score(state: ScoreState) -> str:
    """Compact text, e.g. "6-4 3-2 30-15" (completed sets, current games, current points)."""
    parts = [f"{a}-{b}" for a, b in state.completed_sets]
    if state.is_over:
        return " ".join(parts) + f" (winner {state.winner})"
    if not state.tiebreak_replaces_set:
        parts.append(f"{state.games[0]}-{state.games[1]}")
    pa, pb = point_labels(state)
    parts.append(f"{pa}-{pb}")
    return " ".join(parts)


def near_team(state: ScoreState, initial_near: Team = "A") -> Team:
    """Team currently on the NEAR side of the camera."""
    return other(initial_near) if state.ends_swapped else initial_near
