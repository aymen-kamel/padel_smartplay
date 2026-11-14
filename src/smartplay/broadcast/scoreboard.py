"""Read the on-screen scoreboard of a broadcast and turn score changes into point labels.

The Premier Padel board shows, for each team, one cell per completed set, then the games of the
current set, then the points of the current game (0 / 15 / 30 / 40 / AD). New columns are
appended on the right. OCR runs on a 4x zoom of the number panel; tokens are assigned to columns
by their x position, and any frame whose tokens do not fit the grid is rejected (never guessed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from smartplay.scoring.automaton import MatchRules, ScoreState, apply_point, point_labels

# Number panel of the board in a 1920x1080 frame: rows y0:y1, columns x0:x1, zoomed 4x for OCR.
CROP = (40, 130, 380, 760)
ZOOM = 4
GRID_X0, GRID_DX, GRID_TOL = 205.0, 125.0, 32.0  # column centres in zoomed pixels
ROW_SPLIT_Y = 210.0
POINT_VALUES = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
LABEL_OF = {0: "0", 1: "15", 2: "30", 3: "40"}


@dataclass(frozen=True)
class Board:
    """Cells per row, left to right: completed sets..., games, points (as read)."""

    top: tuple[str, ...]
    bottom: tuple[str, ...]

    @property
    def n_sets_done(self) -> int:
        return len(self.top) - 2

    @property
    def sets(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(a), int(b)) for a, b in zip(self.top[:-2], self.bottom[:-2], strict=True))

    @property
    def games(self) -> tuple[int, int]:
        return int(self.top[-2]), int(self.bottom[-2])

    @property
    def points(self) -> tuple[str, str]:
        return self.top[-1], self.bottom[-1]

    def key(self) -> tuple:
        return (self.top, self.bottom)

    def text(self) -> str:
        done = " ".join(f"{a}-{b}" for a, b in self.sets)
        g, p = self.games, self.points
        return f"{done} | games {g[0]}-{g[1]} | points {p[0]}-{p[1]}".strip(" |")


def parse_tokens(tokens: list[tuple[float, float, str]]) -> Board | None:
    """``tokens`` = (x, y, text) in zoomed crop coordinates. Returns None if it does not fit."""
    rows: dict[int, dict[int, str]] = {0: {}, 1: {}}
    for x, y, text in tokens:
        if not re.fullmatch(r"[0-9]+|AD|A", text):
            continue
        col = round((x - GRID_X0) / GRID_DX)
        if col < 0 or abs(x - (GRID_X0 + col * GRID_DX)) > GRID_TOL:
            return None
        row = 0 if y < ROW_SPLIT_Y else 1
        if col in rows[row]:
            return None
        rows[row][col] = text
    n = len(rows[0])
    if n < 2 or n != len(rows[1]) or sorted(rows[0]) != list(range(n)) or sorted(rows[1]) != list(range(n)):
        return None
    top = tuple(rows[0][c] for c in range(n))
    bot = tuple(rows[1][c] for c in range(n))
    for cell in top[:-1] + bot[:-1]:  # sets and games are single digits
        if not re.fullmatch(r"[0-7]", cell):
            return None
    for cell in (top[-1], bot[-1]):  # points: 0/15/30/40/AD, or a small number in a tie-break
        if cell not in POINT_VALUES and not (cell.isdigit() and int(cell) <= 30):
            return None
    return Board(top, bot)


def to_state(board: Board, rules: MatchRules) -> ScoreState:
    """Engine state equivalent to the board (deuce count unknown: 40-40 is taken as deuce 1)."""
    pa, pb = board.points
    games = board.games
    tiebreak = games == (rules.games_per_set, rules.games_per_set) and not (
        pa in POINT_VALUES and pb in POINT_VALUES
    )
    if tiebreak or (pa.isdigit() and pa not in POINT_VALUES) or (pb.isdigit() and pb not in POINT_VALUES):
        pts = (int(pa), int(pb))
    else:
        pts = (POINT_VALUES[pa], POINT_VALUES[pb])
        if pa == "AD":
            pts = (4, 3)
        elif pb == "AD":
            pts = (3, 4)
    return ScoreState(
        server=rules.first_server,
        completed_sets=board.sets,
        games=games,
        points=pts,
        tiebreak=tiebreak,
        tiebreak_target=rules.tiebreak_points if tiebreak else 0,
        tiebreak_first_server=rules.first_server if tiebreak else None,
    )


def same_board(state: ScoreState, board: Board) -> bool:
    """Do the engine state and the board show the same score? (compared as displayed)."""
    labels = point_labels(state)
    shown = tuple("AD" if x == "A" else x for x in board.points)
    got = tuple(x for x in labels)
    got = tuple("AD" if x == "AD" else x for x in got)
    return state.completed_sets == board.sets and state.games == board.games and got == shown


def classify_transition(prev: Board, nxt: Board, rules: MatchRules) -> str:
    """'top' / 'bottom' if exactly one point by that team explains prev -> nxt, else
    'same', 'jump' (several points or other) or 'inconsistent' (neither team fits)."""
    if prev.key() == nxt.key():
        return "same"
    start = to_state(prev, rules)
    fits = []
    for winner, name in (("A", "top"), ("B", "bottom")):
        try:
            after = apply_point(start, winner, rules)  # type: ignore[arg-type]
        except ValueError:
            continue
        # a finished game/set resets the points; the board may show them after the reset
        if same_board(after, nxt):
            fits.append(name)
    if len(fits) == 1:
        return fits[0]
    return "jump"
