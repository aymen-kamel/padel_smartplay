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
