"""Fixed-camera model: image <-> court floor, from hand-clicked court points.

The 12 clicked points follow the layout below (as seen from the camera, NEAR side at the bottom):

    k11 ----------- k12      far baseline       y = +10
    k8  ---- k9 --- k10      far service line   y = +6.95
    k6  ----------- k7       net (on the floor) y = 0
    k3  ---- k4 --- k5       near service line  y = -6.95
    k1  ----------- k2       near baseline      y = -10

Only valid for points on the floor (z = 0). A ball in the air needs the 3D model (later).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartplay.geometry.court import HALF_LENGTH_M, HALF_WIDTH_M, SERVICE_LINE_FROM_NET_M

S, X, Y = SERVICE_LINE_FROM_NET_M, HALF_WIDTH_M, HALF_LENGTH_M
WORLD_POINTS_12: list[tuple[float, float]] = [
    (-X, -Y), (X, -Y),
    (-X, -S), (0.0, -S), (X, -S),
    (-X, 0.0), (X, 0.0),
    (-X, S), (0.0, S), (X, S),
    (-X, Y), (X, Y),
]

COURT_LINES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((-X, -Y), (X, -Y)), ((X, -Y), (X, Y)), ((X, Y), (-X, Y)), ((-X, Y), (-X, -Y)),
    ((-X, 0.0), (X, 0.0)),
    ((-X, -S), (X, -S)), ((-X, S), (X, S)),
    ((0.0, -S), (0.0, 0.0)), ((0.0, S), (0.0, 0.0)),
]


def _world_err(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pts = np.hstack([src, np.ones((len(src), 1))]) @ h.T
    pts = pts[:, :2] / pts[:, 2:3]
    return np.linalg.norm(pts - dst, axis=1)


@dataclass(frozen=True)
class CourtCamera:
    h_img2world: np.ndarray
    h_world2img: np.ndarray
    fit_error_cm: float  # mean world error on the clicked points (consistency only)
    loo_error_cm: float  # leave-one-out mean world error (honest estimate of accuracy)

    @staticmethod
    def from_points(points_uv: list[tuple[float, float]]) -> CourtCamera:
        import cv2

        if len(points_uv) != len(WORLD_POINTS_12):
            raise ValueError(f"expected {len(WORLD_POINTS_12)} points, got {len(points_uv)}")
        src = np.asarray(points_uv, dtype=np.float64)
        dst = np.asarray(WORLD_POINTS_12, dtype=np.float64)
        h, _ = cv2.findHomography(src, dst, method=0)
        if h is None:
            raise ValueError("homography could not be estimated")
        fit = _world_err(h, src, dst)
        loo = []
        for i in range(len(src)):
            keep = [j for j in range(len(src)) if j != i]
            hi, _ = cv2.findHomography(src[keep], dst[keep], method=0)
            loo.append(
                _world_err(hi, src[i:i + 1], dst[i:i + 1])[0] if hi is not None else np.nan
            )
        return CourtCamera(h, np.linalg.inv(h), float(np.mean(fit)) * 100, float(np.nanmean(loo)) * 100)

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        p = self.h_img2world @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        p = self.h_world2img @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])
