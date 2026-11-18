import numpy as np

from smartplay.ball.detector import Blob
from smartplay.ball.physics import refine_physics
from smartplay.schemas.records import BallState


def states_from(path, drop=()):
    out = []
    for f, p in enumerate(path):
        if p is None or f in drop:
            out.append(BallState(frame=f, visibility="lost", conf=0.0))
        else:
            out.append(BallState(frame=f, uv=(float(p[0]), float(p[1])), visibility="visible", conf=0.9))
    return out


def parabola(n, u0=100.0, du=12.0, v0=500.0, dv=-14.0, g=0.9, f0=0):
    return [(u0 + du * (f - f0), v0 + dv * (f - f0) + 0.5 * g * (f - f0) ** 2) for f in range(f0, f0 + n)]


def test_gap_inside_flight_is_filled_on_the_curve():
    path = parabola(30)
    out, kinks = refine_physics(states_from(path, drop={12, 13, 14}))
    assert kinks == []
    for f in (12, 13, 14):
        assert out[f].visibility == "inferred"
        assert np.hypot(out[f].uv[0] - path[f][0], out[f].uv[1] - path[f][1]) < 1.5


def test_measured_points_are_not_moved():
    path = parabola(30)
    st = states_from(path)
    out, _ = refine_physics(st)
    assert [o.uv for o in out] == [s.uv for s in st]


def test_bounce_is_a_kink_and_gaps_are_not_filled_across_it():
    a = parabola(15)
    b = [(a[-1][0] + 12 * k, a[-1][1] - 40 + (-26) * k + 0.45 * k * k) for k in range(1, 16)]
    out, kinks = refine_physics(states_from(a + b, drop={14, 15}))
    assert len(kinks) == 1 and abs(kinks[0].frame - 13) <= 1
    assert kinks[0].turn_deg > 20
    assert out[14].uv is None


def test_isolated_outlier_is_repaired():
    path = parabola(30)
    st = states_from(path)
    st[15] = BallState(frame=15, uv=(path[15][0] + 150, path[15][1] - 30), visibility="visible", conf=0.9)
    out, kinks = refine_physics(st)
    assert kinks == []
    assert np.hypot(out[15].uv[0] - path[15][0], out[15].uv[1] - path[15][1]) < 1.5


def test_raw_candidate_on_the_curve_is_recovered_as_visible():
    path = parabola(30)
    blobs = [[] for _ in path]
    blobs[12] = [Blob(path[12][0] + 2, path[12][1] - 1, 0.5)]
    out, _ = refine_physics(states_from(path, drop={12}), blobs)
    assert out[12].visibility == "visible"
