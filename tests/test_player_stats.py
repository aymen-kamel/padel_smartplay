import pytest

from smartplay.analytics.player_stats import PlayerStats
from smartplay.schemas.records import PlayerTrack


def track(frame, pid, x, y, observed=True):
    return PlayerTrack(frame=frame, player_id=pid, team="A" if pid <= 2 else "B",
                       side="NEAR" if y < 0 else "FAR", foot_xy_m=(x, y), conf=0.9, observed=observed)


def run(positions, fps=30.0, observed=None):
    st = PlayerStats(fps)
    for i, (x, y) in enumerate(positions):
        st.update([track(i, 1, x, y, True if observed is None else observed[i])])
    return st.snapshot()[1]


def test_distance_of_steady_run():
    snap = run([(0.05 * i, -5.0) for i in range(61)])  # 1.5 m/s for 2 s -> about 3 m
    assert snap["dist_m"] == pytest.approx(3.0, abs=0.25)
    assert snap["speed_kmh"] == pytest.approx(5.4, abs=0.8)
    assert snap["team"] == "A"


def test_standing_still_is_not_distance():
    assert run([(1.0, -5.0)] * 40)["dist_m"] == 0.0


def test_teleport_glitch_is_ignored():
    pos = [(0.0, -5.0)] * 10 + [(6.0, -5.0)] * 10  # a 6 m jump in one frame
    assert run(pos)["dist_m"] < 0.5


def test_predicted_positions_add_no_distance():
    pos = [(0.05 * i, -5.0) for i in range(30)]
    obs = [i < 10 for i in range(30)]
    assert run(pos, observed=obs)["dist_m"] < 0.6


def test_speed_never_exceeds_human_limit():
    pos = [(0.29 * i, -5.0) for i in range(30)]  # 8.7 m/s, just under the cap
    assert run(pos)["speed_kmh"] <= 9.0 * 3.6 + 0.01
