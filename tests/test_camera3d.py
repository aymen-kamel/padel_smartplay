import numpy as np
import pytest

from smartplay.geometry.camera import WORLD_POINTS_12
from smartplay.geometry.camera3d import HEIGHT_POINTS, Camera3D, dlt, project


def synthetic_camera():
    # camera 7 m high, 6 m behind the near baseline, looking at the court centre
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    c = np.array([0.0, -16.0, 7.0])
    f = -c / np.linalg.norm(c)  # look at the origin
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.vstack([r, d, f])
    return K @ np.hstack([R, (-R @ c)[:, None]]), c


def world_points():
    return [(x, y, 0.0) for x, y in WORLD_POINTS_12] + [p for _, p in HEIGHT_POINTS[:5]]


def test_dlt_recovers_projection_and_centre():
    P_true, c = synthetic_camera()
    W = np.array(world_points())
    I = project(P_true, W)
    cam = Camera3D.from_points(list(map(tuple, W)), list(map(tuple, I)))
    assert cam.reproj_px < 1e-6 and cam.loo_px < 1e-4
    assert np.allclose(cam.centre, c, atol=1e-6)
    assert cam.n_height_points == 5


def test_point_at_height_inverts_projection():
    P_true, _ = synthetic_camera()
    W = np.array(world_points())
    cam = Camera3D.from_points(list(map(tuple, W)), list(map(tuple, project(P_true, W))))
    ball = np.array([1.3, 4.2, 2.1])
    u, v = cam.to_image(ball)[0]
    assert np.allclose(cam.point_at_height(u, v, 2.1), ball, atol=1e-6)


def test_noise_gives_small_but_honest_errors():
    P_true, _ = synthetic_camera()
    W = np.array(world_points())
    rng = np.random.default_rng(0)
    I = project(P_true, W) + rng.normal(0, 1.5, (len(W), 2))
    cam = Camera3D.from_points(list(map(tuple, W)), list(map(tuple, I)))
    assert 0.5 < cam.reproj_px < 3.0 and cam.loo_px > cam.reproj_px


def test_floor_only_points_are_rejected():
    W = np.array([(x, y, 0.0) for x, y in WORLD_POINTS_12])
    with pytest.raises(ValueError, match="one plane"):
        dlt(W, W[:, :2])
