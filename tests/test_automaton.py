"""Score engine tests. Each test replays a hand-written list of point winners."""

import pytest

from smartplay.scoring.automaton import (
    DeuceMode,
    FinalSet,
    MatchRules,
    apply_point,
    format_score,
    initial_state,
    near_team,
    point_labels,
    replay,
)

ADV = MatchRules()
GOLD = MatchRules(deuce_mode=DeuceMode.GOLDEN_POINT)
STAR = MatchRules(deuce_mode=DeuceMode.STAR_POINT)


def play(seq: str, rules: MatchRules = ADV):
    states = replay(list(seq), rules)
    return states[-1] if states else initial_state(rules)


def games(n_a: int, n_b: int) -> str:
    """Points giving A n_a love games and B n_b love games, alternating while possible."""
    out, a, b = [], n_a, n_b
    while a or b:
        if a:
            out.append("AAAA")
            a -= 1
        if b:
            out.append("BBBB")
            b -= 1
    return "".join(out)


# ---------------------------------------------------------------- game scoring

def test_initial_state():
    s = initial_state(ADV)
    assert (s.games, s.points, s.server, s.ends_swapped) == ((0, 0), (0, 0), "A", False)
    assert format_score(s) == "0-0 0-0"


@pytest.mark.parametrize(
    "seq,labels",
    [
        ("A", ("15", "0")),
        ("AA", ("30", "0")),
        ("AAA", ("40", "0")),
        ("AB", ("15", "15")),
        ("AAABBB", ("40", "40")),
        ("AAABBBA", ("AD", "40")),
        ("AAABBBB", ("40", "AD")),
        ("AAABBBAB", ("40", "40")),
    ],
)
def test_point_labels(seq, labels):
    assert point_labels(play(seq)) == labels


def test_love_game():
    s = play("AAAA")
    assert s.games == (1, 0) and s.points == (0, 0)


def test_advantage_needs_two_clear_points():
    assert play("AAABBB" + "A").games == (0, 0)
    assert play("AAABBB" + "AA").games == (1, 0)
    assert play("AAABBB" + "AB" * 5 + "BB").games == (0, 1)


def test_golden_point_decides_at_first_deuce():
    assert play("AAABBB" + "B", GOLD).games == (0, 1)
    assert play("AAABBB" + "A", GOLD).games == (1, 0)
    assert play("AAABB" + "B", GOLD).games == (0, 0)  # 40-30 is not a deciding point


def test_star_point_two_advantages_then_deciding_point():
    deuce1 = "AAABBB"
    assert play(deuce1 + "AA", STAR).games == (1, 0)  # advantage 1 converted
    deuce2 = deuce1 + "AB"
    assert play(deuce2 + "B", STAR).games == (0, 0)  # advantage 2, not yet game
    assert play(deuce2 + "BB", STAR).games == (0, 1)  # advantage 2 converted
    deuce3 = deuce2 + "AB"
    assert play(deuce3, STAR).games == (0, 0)
    assert play(deuce3 + "B", STAR).games == (0, 1)  # star point: one point decides
    assert play(deuce3 + "A", STAR).games == (1, 0)


# ---------------------------------------------------------------- sets

def test_set_six_four():
    s = play(games(6, 4))
    assert s.completed_sets == ((6, 4),) and s.games == (0, 0) and s.sets_won == (1, 0)


def test_set_needs_two_game_margin_seven_five():
    s = play(games(5, 5) + "AAAA")
    assert s.completed_sets == () and s.games == (6, 5)
    s = play(games(5, 5) + "AAAA" + "AAAA")
    assert s.completed_sets == ((7, 5),)


def test_tiebreak_starts_at_six_all():
    s = play(games(6, 6))
    assert s.tiebreak and s.games == (6, 6) and s.tiebreak_target == 7


def test_tiebreak_needs_seven_points_and_two_point_margin():
    tb = games(6, 6)
    assert play(tb + "A" * 7).completed_sets == ((7, 6),)
    s = play(tb + "AB" * 6 + "A")  # 7-6 in points: not finished
    assert s.tiebreak and s.points == (7, 6)
    assert play(tb + "AB" * 6 + "AA").completed_sets == ((7, 6),)
    assert play(tb + "AB" * 6 + "BB").completed_sets == ((6, 7),)


def test_tiebreak_point_labels_are_numbers():
    assert point_labels(play(games(6, 6) + "AAB")) == ("2", "1")


def test_mini_set():
    mini = MatchRules(games_per_set=4)
    assert play(games(4, 2), mini).completed_sets == ((4, 2),)
    assert play(games(4, 3), mini).completed_sets == ()
    assert play(games(4, 4), mini).tiebreak


# ---------------------------------------------------------------- match

def test_match_two_sets_to_love():
    s = play(games(6, 0) * 2)
    assert s.winner == "A" and s.completed_sets == ((6, 0), (6, 0)) and s.is_over
    assert format_score(s) == "6-0 6-0 (winner A)"


def test_no_point_after_match_end():
    with pytest.raises(ValueError):
        apply_point(play(games(6, 0) * 2), "A", ADV)


def test_invalid_winner_rejected():
    with pytest.raises(ValueError):
        apply_point(initial_state(ADV), "C", ADV)  # type: ignore[arg-type]


def test_three_set_match():
    s = play(games(6, 0) + games(0, 6) + games(6, 0))
    assert s.winner == "A" and s.sets_won == (2, 1)


def test_super_tiebreak_replaces_third_set():
    rules = MatchRules(final_set=FinalSet.SUPER_TIEBREAK)
    one_all = games(6, 0) + games(0, 6)
    s = play(one_all, rules)
    assert s.tiebreak and s.tiebreak_replaces_set and s.tiebreak_target == 10
    assert not play(one_all + "A" * 9, rules).is_over
    s = play(one_all + "A" * 10, rules)
    assert s.winner == "A" and s.completed_sets[-1] == (10, 0)


def test_match_tiebreak_to_seven():
    rules = MatchRules(final_set=FinalSet.MATCH_TIEBREAK)
    s = play(games(6, 0) + games(0, 6) + "B" * 7, rules)
    assert s.winner == "B" and s.completed_sets[-1] == (0, 7)


def test_deciding_set_without_tiebreak_is_played_out():
    rules = MatchRules(final_set_tiebreak=False)
    one_all = games(6, 0) + games(0, 6)
    s = play(one_all + games(6, 6), rules)
    assert not s.tiebreak and s.games == (6, 6)
    s = play(one_all + games(6, 6) + "AAAA" * 2, rules)
    assert s.winner == "A" and s.completed_sets[-1] == (8, 6)
    assert play(games(6, 6), rules).tiebreak  # the first set still uses a tie-break


# ---------------------------------------------------------------- service

def test_service_alternates_each_game():
    states = replay(list("AAAA" * 3), ADV)
    assert [states[i].server for i in (0, 3, 7, 11)] == ["A", "B", "A", "B"]


def test_tiebreak_service_one_then_two_each():
    base = list(games(6, 6))  # 12 games: A's turn to serve first in the tie-break
    start = replay(base, ADV)[-1]
    assert start.server == "A" and start.tiebreak_first_server == "A"
    servers = [start.server]
    seq = base[:]
    for w in "ABABABAB":
        seq.append(w)
        servers.append(replay(seq, ADV)[-1].server)
    assert servers == ["A", "B", "B", "A", "A", "B", "B", "A", "A"]


def test_next_set_started_by_pair_that_did_not_start_tiebreak():
    assert play(games(6, 6) + "A" * 7).server == "B"


# ---------------------------------------------------------------- changes of ends (Rule 5)

def test_ends_change_after_odd_games():
    states = replay(list("AAAA" * 4), ADV)
    assert [states[i].ends_swapped for i in (3, 7, 11, 15)] == [True, True, False, False]


def test_ends_at_set_end_follow_total_game_parity():
    # changes after games 1,3,5,7,9 of the set
    assert play(games(6, 4)).ends_swapped is True  # 10 games -> 5 changes
    assert play(games(6, 2)).ends_swapped is False  # 8 games -> 4 changes


def test_tiebreak_ends_change_every_six_points():
    base = games(6, 6)
    start = play(base).ends_swapped
    assert play(base + "ABABA").ends_swapped == start
    assert play(base + "ABABAB").ends_swapped != start
    assert play(base + "AB" * 6).ends_swapped == start
    assert play(base + "A" * 7).ends_swapped != start  # opposite ends after the tie-break


def test_near_team_follows_ends():
    assert near_team(initial_state(ADV), initial_near="A") == "A"
    assert near_team(play("AAAA"), initial_near="A") == "B"


# ---------------------------------------------------------------- bookkeeping

def test_points_played_counts_every_point():
    assert play(games(6, 4)).points_played == 40


def test_states_are_immutable():
    from dataclasses import FrozenInstanceError

    s = initial_state(ADV)
    with pytest.raises(FrozenInstanceError):
        s.games = (1, 0)  # type: ignore[misc]
