"""Tests for modules that need the vision stack (OpenCV, SciPy). Skipped when not installed."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from smartplay.events.audio import detect_hits
from smartplay.geometry.camera import WORLD_POINTS_12, CourtCamera
from smartplay.players.tracker import FourPlayerTracker


def _synthetic_points() -> list[tuple[float, float]]:
    """Project the 12 court points with a known perspective homography."""
    h = np.array([[60.0, 8.0, 960.0], [0.0, -25.0, 520.0], [0.0, 0.012, 1.0]])
    pts = []
    for x, y in WORLD_POINTS_12:
        p = h @ np.array([x, y, 1.0])
        pts.append((p[0] / p[2], p[1] / p[2]))
    return pts


def test_camera_roundtrip_and_error():
    cam = CourtCamera.from_points(_synthetic_points())
    assert cam.fit_error_cm < 0.5 and cam.loo_error_cm < 0.5
    u, v = cam.to_image(2.0, -3.0)
    x, y = cam.to_world(u, v)
    assert x == pytest.approx(2.0, abs=1e-6) and y == pytest.approx(-3.0, abs=1e-6)


def test_camera_needs_twelve_points():
    with pytest.raises(ValueError):
        CourtCamera.from_points(_synthetic_points()[:6])


def test_audio_hits_found_on_clicks():
    sr = 16000
    rng = np.random.default_rng(0)
    audio = 0.01 * rng.standard_normal(sr * 3).astype(np.float32)
    for t in (0.5, 1.4, 2.3):
        i = int(t * sr)
        audio[i:i + 80] += 0.8 * rng.standard_normal(80).astype(np.float32)
    hits = detect_hits(audio, sr)
    assert len(hits) == 3
    assert [round(h.t, 1) for h in hits] == [0.5, 1.4, 2.3]


def test_tracker_assigns_teams_by_side_and_keeps_ids():
    tr = FourPlayerTracker()
    first = tr.update(0, [(-2, -5), (2, -6), (-1, 5), (3, 6)], [0.9] * 4)
    by_id = {t.player_id: t for t in first}
    assert by_id[1].team == "A" and by_id[1].side == "NEAR" and by_id[1].foot_xy_m[0] < 0
    assert by_id[4].team == "B" and by_id[4].side == "FAR" and by_id[4].foot_xy_m[0] > 0
    nxt = tr.update(1, [(3.1, 6.1), (-2.1, -5.0), (2.0, -6.1), (-1.0, 5.1)], [0.9] * 4)
    pos = {t.player_id: t.foot_xy_m for t in nxt}
    assert pos[1] == pytest.approx((-2.1, -5.0)) and pos[4] == pytest.approx((3.1, 6.1))


def test_tracker_coasts_missing_player():
    tr = FourPlayerTracker()
    tr.update(0, [(-2, -5), (2, -6), (-1, 5), (3, 6)], [0.9] * 4)
    out = tr.update(1, [(-2, -5), (2, -6), (-1, 5)], [0.9] * 3)
    missing = [t for t in out if not t.observed]
    assert len(out) == 4 and len(missing) == 1 and missing[0].player_id == 4
