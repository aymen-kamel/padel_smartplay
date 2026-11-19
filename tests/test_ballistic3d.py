import numpy as np

from smartplay.ball.ballistic3d import G, fit_flights, states_with_3d
from smartplay.geometry.camera import WORLD_POINTS_12
from smartplay.geometry.camera3d import HEIGHT_POINTS, Camera3D, project
from smartplay.schemas.records import BallState


def synthetic_camera():
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    c = np.array([0.0, -16.0, 7.0])
    f = -c / np.linalg.norm(c)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    R = np.vstack([r, np.cross(f, r), f])
    return K @ np.hstack([R, (-R @ c)[:, None]]), c


def camera():
    P, _ = synthetic_camera()
    W = np.array([(x, y, 0.0) for x, y in WORLD_POINTS_12] + [p for _, p in HEIGHT_POINTS[:5]])
    return Camera3D.from_points(list(map(tuple, W)), list(map(tuple, project(P, W)))), P


def trajectory(p0, v0, n, fps):
    t = np.arange(n) / fps
    return np.array(p0) + np.outer(t, v0) + np.outer(0.5 * G * t * t, [0, 0, -1.0])


def test_recovers_a_lob_in_3d_from_one_camera():
    cam, P = camera()
    fps = 25.0
    path = trajectory([1.0, -7.0, 1.0], [-0.5, 12.0, 6.0], 25, fps)
    rng = np.random.default_rng(1)
    uv = project(P, path) + rng.normal(0, 1.0, (len(path), 2))
    states = [BallState(frame=i, uv=(float(u), float(v)), visibility="visible", conf=0.9) for i, (u, v) in enumerate(uv)]
    flights = fit_flights(states, cam, fps)
    assert len(flights) == 1
    fl = flights[0]
    err = np.linalg.norm(np.array([fl.at(i, fps) for i in range(25)]) - path, axis=1)
    assert err.max() < 0.35  # metres, with 1 px noise
    assert abs(fl.speed_kmh(0, fps) - np.linalg.norm([-0.5, 12.0, 6.0]) * 3.6) < 6.0
    filled = states_with_3d(states, flights, fps)
    assert filled[10].xyz_m is not None


def test_ball_going_under_the_floor_is_rejected():
    cam, P = camera()
    fps = 25.0
    path = trajectory([0.0, 0.0, 0.2], [0.0, 5.0, -8.0], 15, fps)  # dives 2 m under the floor: impossible
    uv = project(P, path)
    states = [BallState(frame=i, uv=(float(u), float(v)), visibility="visible", conf=0.9) for i, (u, v) in enumerate(uv)]
    assert fit_flights(states, cam, fps) == []
