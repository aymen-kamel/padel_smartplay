import pytest
from pydantic import ValidationError

from smartplay.evaluation.metrics import (
    binomial_ci95,
    match_events,
    point_winner_accuracy,
    score_timeline_accuracy,
)
from smartplay.geometry.court import is_in_court, is_in_service_box, other_side, side_of
from smartplay.schemas.records import Event, PointOutcome, VideoInfo
from smartplay.scoring.automaton import MatchRules, replay


def test_side_of():
    assert side_of(-3.0) == "NEAR" and side_of(4.0) == "FAR" and other_side("NEAR") == "FAR"


def test_in_court_lines_are_in():
    assert is_in_court(5.0, 10.0) and is_in_court(-5.0, -10.0)
    assert not is_in_court(5.01, 0.0)
    assert is_in_court(5.1, 0.0, margin_m=0.2)


def test_service_box():
    assert is_in_service_box(-2.0, -3.0, "NEAR", "left")
    assert not is_in_service_box(2.0, -3.0, "NEAR", "left")
    assert not is_in_service_box(-2.0, -7.5, "NEAR", "left")  # behind the service line
    assert not is_in_service_box(-2.0, 3.0, "NEAR", "left")  # other half
    assert is_in_service_box(0.0, 6.95, "FAR", "right")  # lines count as good


def test_match_events_within_tolerance():
    s = match_events([10, 52, 99], [12, 50, 80], tolerance=3)
    assert (s.true_positives, s.false_positives, s.false_negatives) == (2, 1, 1)
    assert s.precision == pytest.approx(2 / 3) and s.recall == pytest.approx(2 / 3)


def test_match_events_is_one_to_one():
    s = match_events([10, 11, 12], [11], tolerance=3)
    assert (s.true_positives, s.false_positives, s.false_negatives) == (1, 2, 0)


def test_match_events_empty():
    s = match_events([], [5], tolerance=1)
    assert s.f1 == 0.0 and s.false_negatives == 1


def test_point_winner_accuracy():
    assert point_winner_accuracy(list("AABA"), list("AABB")) == 0.75
    with pytest.raises(ValueError):
        point_winner_accuracy(["A"], ["A", "B"])


def test_one_wrong_point_corrupts_the_displayed_score():
    rules = MatchRules()
    truth = replay(list("AAAABBBB"), rules)
    pred = replay(list("AAABBBBB"), rules)  # one wrong winner in the first game
    assert point_winner_accuracy(list("AAABBBBB"), list("AAAABBBB")) == 0.875
    assert score_timeline_accuracy(pred, truth) < 0.5


def test_binomial_ci_contains_estimate():
    lo, hi = binomial_ci95(360, 450)
    assert lo < 0.8 < hi and hi - lo < 0.08


def test_schemas_validate():
    info = VideoInfo(path="m.mp4", fps=25, n_frames=324, width=1920, height=1080, has_audio=True)
    assert info.duration_s == pytest.approx(12.96)
    assert PointOutcome(point_idx=0, p_team_a=0.2).likely_winner == "B"
    with pytest.raises(ValidationError):
        Event(t=1.0, frame=30, kind="hit", conf=1.5)
    with pytest.raises(ValidationError):
        PointOutcome(point_idx=0, p_team_a=0.5, extra_field=1)
