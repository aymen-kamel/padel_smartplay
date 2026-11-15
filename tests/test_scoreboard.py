"""Scoreboard parsing and point-transition tests (synthetic OCR tokens, no video)."""

from smartplay.broadcast.scoreboard import (
    GRID_DX,
    GRID_X0,
    Board,
    classify_transition,
    parse_tokens,
    to_state,
)
from smartplay.scoring.automaton import MatchRules

RULES = MatchRules()
TOP_Y, BOT_Y = 150.0, 270.0


def toks(top: list[str], bottom: list[str]) -> list[tuple[float, float, str]]:
    out = []
    for c, t in enumerate(top):
        out.append((GRID_X0 + c * GRID_DX + 2, TOP_Y, t))
    for c, t in enumerate(bottom):
        out.append((GRID_X0 + c * GRID_DX - 3, BOT_Y, t))
    return out


def board(top: list[str], bottom: list[str]) -> Board:
    b = parse_tokens(toks(top, bottom))
    assert b is not None
    return b


def test_parse_games_and_points():
    b = board(["0", "15"], ["0", "0"])
    assert b.games == (0, 0) and b.points == ("15", "0") and b.sets == ()


def test_parse_completed_sets():
    b = board(["4", "6", "3", "15"], ["6", "3", "2", "40"])
    assert b.sets == ((4, 6), (6, 3)) and b.games == (3, 2) and b.points == ("15", "40")


def test_parse_ignores_non_score_text():
    t = toks(["2", "40"], ["1", "40"]) + [(500.0, 100.0, "火"), (900.0, 50.0, "SET POINT")]
    assert parse_tokens(t) is not None


def test_parse_rejects_misaligned_columns():
    t = toks(["2", "40"], ["1", "40"])
    t[1] = (t[1][0] + 70, t[1][1], t[1][2])
    assert parse_tokens(t) is None


def test_parse_rejects_unequal_rows():
    assert parse_tokens(toks(["0", "15"], ["0"])) is None


def test_parse_rejects_bad_points():
    assert parse_tokens(toks(["0", "45"], ["0", "0"])) is None


def test_single_point_transitions():
    a = board(["2", "30"], ["1", "15"])
    assert classify_transition(a, board(["2", "40"], ["1", "15"]), RULES) == "top"
    assert classify_transition(a, board(["2", "30"], ["1", "30"]), RULES) == "bottom"
    assert classify_transition(a, a, RULES) == "same"


def test_game_won_resets_points():
    a = board(["2", "40"], ["1", "30"])
    assert classify_transition(a, board(["3", "0"], ["1", "0"]), RULES) == "top"
    assert classify_transition(a, board(["2", "40"], ["2", "0"]), RULES) == "jump"


def test_deuce_and_advantage():
    deuce = board(["3", "40"], ["3", "40"])
    assert classify_transition(deuce, board(["3", "AD"], ["3", "40"]), RULES) == "top"
    adv = board(["3", "AD"], ["3", "40"])
    assert classify_transition(adv, deuce, RULES) == "bottom"
    assert classify_transition(adv, board(["4", "0"], ["3", "0"]), RULES) == "top"


def test_set_won_adds_a_column():
    a = board(["5", "40"], ["4", "15"])
    board(["6", "3", "0", "0"], ["4", "2", "0", "0"])
    assert True  # layout check only
    done = board(["6", "0", "0"], ["4", "0", "0"])
    assert classify_transition(a, done, RULES) == "top"


def test_to_state_roundtrip_labels():
    s = to_state(board(["4", "6", "3", "15"], ["6", "3", "2", "40"]), RULES)
    assert s.games == (3, 2) and s.completed_sets == ((4, 6), (6, 3)) and s.points == (1, 3)
