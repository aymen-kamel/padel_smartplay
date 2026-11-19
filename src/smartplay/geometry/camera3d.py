"""Full 3D camera (3x4 projection matrix) from clicked court points with known heights, by DLT.

The floor homography (``camera.CourtCamera``) only maps the floor (z = 0). To place a ball in the air we need the full
projection P (11 unknowns): x_img ~ P [X, Y, Z, 1]. The Direct Linear Transform solves it by least squares from >= 6
points that are NOT all on one plane: the 12 floor points plus points of known height (net posts, glass tops).
No focal-length guess and no distortion model (assumed small; the reprojection error shows if it is not).

World axes (metres): x across the court (right positive as seen from the camera), y along it (negative = NEAR the
camera), z up; origin at the centre of the net on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartplay.geometry.court import HALF_LENGTH_M, HALF_WIDTH_M

X, Y = HALF_WIDTH_M, HALF_LENGTH_M

# Points of known height that are usually visible from a fixed camera behind one baseline. Skip any you cannot see.
# Side walls (FIP standard layout): from each back wall, 2 m of glass at 3 m height, then 2 m of glass at 2 m height.
HEIGHT_POINTS: list[tuple[str, tuple[float, float, float]]] = [
    ("net post top, LEFT", (-X, 0.0, 0.92)),
    ("net top at the CENTRE strap", (0.0, 0.0, 0.88)),
    ("net post top, RIGHT", (X, 0.0, 0.92)),
    ("FAR back glass, top LEFT corner", (-X, Y, 3.0)),
    ("FAR back glass, top RIGHT corner", (X, Y, 3.0)),
    ("FAR LEFT side glass: top where 3 m glass steps down to 2 m", (-X, Y - 2.0, 3.0)),
    ("FAR RIGHT side glass: top where 3 m glass steps down to 2 m", (X, Y - 2.0, 3.0)),
    ("FAR LEFT side glass: top END of the 2 m glass (towards the net)", (-X, Y - 4.0, 2.0)),
    ("FAR RIGHT side glass: top END of the 2 m glass (towards the net)", (X, Y - 4.0, 2.0)),
    ("NEAR LEFT side glass: top END of the 2 m glass (towards the net)", (-X, -Y + 4.0, 2.0)),
    ("NEAR RIGHT side glass: top END of the 2 m glass (towards the net)", (X, -Y + 4.0, 2.0)),
]


def _normaliser(pts: np.ndarray) -> np.ndarray:
    """Similarity transform: centroid to 0, mean distance to sqrt(dim) (Hartley normalisation)."""
    d = pts.shape[1]
    c = pts.mean(axis=0)
    s = np.sqrt(d) / (np.linalg.norm(pts - c, axis=1).mean() + 1e-12)
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = -s * c
    return T


def dlt(world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """3x4 projection matrix from >= 6 world (N,3) / image (N,2) correspondences, not all coplanar."""
    world, image = np.asarray(world, float), np.asarray(image, float)
    if len(world) < 6:
        raise ValueError("DLT needs at least 6 points")
    if np.linalg.matrix_rank(world - world.mean(axis=0)) < 3:
        raise ValueError("all points lie on one plane: add points of known height (net, glass tops)")
    Tw, Ti = _normaliser(world), _normaliser(image)
    w = (np.hstack([world, np.ones((len(world), 1))]) @ Tw.T)
    m = (np.hstack([image, np.ones((len(image), 1))]) @ Ti.T)
    rows = []
    for (Xw, Yw, Zw, Ww), (u, v, s) in zip(w, m, strict=True):
        P = [Xw, Yw, Zw, Ww]
        rows.append([0, 0, 0, 0] + [-s * p for p in P] + [v * p for p in P])
        rows.append([s * p for p in P] + [0, 0, 0, 0] + [-u * p for p in P])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Pn = vt[-1].reshape(3, 4)
    P = np.linalg.inv(Ti) @ Pn @ Tw
    return P / np.linalg.norm(P[2, :3])


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.atleast_2d(np.asarray(xyz, float))
    h = np.hstack([xyz, np.ones((len(xyz), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


@dataclass(frozen=True)
class Camera3D:
    P: np.ndarray
    reproj_px: float  # mean reprojection error on the clicked points (consistency)
    loo_px: float  # leave-one-out reprojection error (honest estimate)
    n_points: int
    n_height_points: int

    @staticmethod
    def from_points(world: list[tuple[float, float, float]], image: list[tuple[float, float]]) -> Camera3D:
        W, I = np.asarray(world, float), np.asarray(image, float)
        P = dlt(W, I)
        reproj = np.linalg.norm(project(P, W) - I, axis=1)
        loo = []
        for i in range(len(W)):
            keep = [j for j in range(len(W)) if j != i]
            try:
                Pi = dlt(W[keep], I[keep])
                loo.append(float(np.linalg.norm(project(Pi, W[i:i + 1])[0] - I[i])))
            except ValueError:
                loo.append(np.nan)
        return Camera3D(P, float(reproj.mean()), float(np.nanmean(loo)), len(W), int((W[:, 2] != 0).sum()))

    @property
    def centre(self) -> np.ndarray:
        """Camera position in world metres (null space of P)."""
        _, _, vt = np.linalg.svd(self.P)
        c = vt[-1]
        return c[:3] / c[3]

    def to_image(self, xyz) -> np.ndarray:
        return project(self.P, xyz)

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Camera centre and unit direction of the viewing ray through pixel (u, v)."""
        M, p4 = self.P[:, :3], self.P[:, 3]
        d = np.linalg.solve(M, np.array([u, v, 1.0]))
        c = -np.linalg.solve(M, p4)
        return c, d / np.linalg.norm(d)

    def point_at_height(self, u: float, v: float, z: float) -> np.ndarray:
        """3D point seen at pixel (u, v) if it is at height ``z`` (z = 0: the floor)."""
        c, d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            raise ValueError("ray parallel to the plane")
        return c + d * ((z - c[2]) / d[2])
