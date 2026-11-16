"""Ball post-processing tests (no model needed)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from smartplay.ball.detector import Blob, OnlineBallTracker, heatmap_blobs


def test_heatmap_blob_centre_is_scaled_to_image_pixels():
    hm = np.zeros((288, 512), dtype=np.float32)
    hm[100:103, 200:203] = 0.9
    blobs = heatmap_blobs(hm, 0.5, sx=1920 / 512, sy=1080 / 288)
    assert len(blobs) == 1
    assert blobs[0].u == pytest.approx(201.5 * 3.75) and blobs[0].v == pytest.approx(101.5 * 3.75)


def test_heatmap_below_threshold_gives_nothing():
    assert heatmap_blobs(np.full((10, 10), 0.4, dtype=np.float32), 0.5, 1, 1) == []


def test_tracker_prefers_blob_near_previous_position():
    tr = OnlineBallTracker(max_disp_px=100)
    tr.step(0, [Blob(500, 500, 0.9)])
    s = tr.step(1, [Blob(1500, 200, 0.95), Blob(520, 490, 0.6)])  # far blob is stronger
    assert s.uv == (520, 490)


def test_tracker_reports_lost_and_does_not_invent_positions():
    tr = OnlineBallTracker()
    tr.step(0, [Blob(500, 500, 0.9)])
    s = tr.step(1, [])
    assert s.visibility == "lost" and s.uv is None


# ---- gated tracker (reflection rejection) ----
from smartplay.ball.detector import track_gated


def _moving(n, x0=500.0, dx=10.0, y=400.0):
    return [[Blob(x0 + dx * i, y, 0.9)] for i in range(n)]


def test_gated_follows_a_straight_trajectory():
    states = track_gated(_moving(10))
    assert all(s.visibility == "visible" for s in states)
    assert states[9].uv == (590.0, 400.0)


def test_gated_ignores_a_stronger_mirror_reflection():
    frames = _moving(12)
    for i in range(5, 12):  # mirror image across x = 900, moving the other way, stronger
        frames[i].append(Blob(1800.0 - (500.0 + 10.0 * i), 400.0, 0.99))
    states = track_gated(frames)
    assert all(abs(s.uv[0] - (500 + 10 * i)) < 1 for i, s in enumerate(states))


def test_gated_reports_lost_and_never_fills_gaps():
    frames = _moving(10)
    frames[4] = []
    states = track_gated(frames)
    assert states[4].visibility == "lost" and states[4].uv is None
    assert states[5].visibility == "visible"


def test_gated_needs_confirmation_to_start_a_track():
    frames = [[] for _ in range(6)]
    frames[2] = [Blob(100.0, 100.0, 0.95)]  # isolated flash: not confirmed by the next frame
    states = track_gated(frames)
    assert all(s.visibility == "lost" for s in states)
