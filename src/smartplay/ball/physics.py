"""Physics post-processor for the ball track: piecewise-parabolic flights, kinks, gap filling.

Between two contacts (racket, floor, wall) the ball is a projectile, so its image path is smooth: roughly
linear in u and quadratic in v (gravity seen through the camera). This module takes the points the tracker
already chose and uses that physics to
  * cut the track into **flights**; each cut is a **kink** = a velocity change the projectile cannot
    explain, i.e. a hit or a bounce (candidate event for the hit detector);
  * fill short gaps inside one flight with the fitted curve (``inferred``), or with a raw detector
    candidate that lies on the curve (``visible``);
  * repair isolated outliers (a pick far off the curve while its neighbours agree, typically a glass
    reflection), replacing them by the curve value.
Measured points that fit the curve are never moved: a first attempt that smoothed everything with a Kalman
filter lowered the benchmark F1 (see docs/BENCHMARK_BASELINE.md), because detections are already precise.

Image-space model, valid over short arcs. A 3D model needs a camera pose (planned, see docs/NEXT_STEPS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartplay.ball.detector import Blob
from smartplay.schemas.records import BallState


@dataclass(frozen=True)
class Kink:
    frame: int  # last frame of the old flight
    speed_before: float  # px/frame
    speed_after: float
    turn_deg: float  # change of direction (0 = straight on)


@dataclass(frozen=True)
class PhysicsParams:
    base_tol: float = 7.0  # px: residual tolerated for a point right after the last one
    tol_per_gap: float = 2.0  # px per missing frame since the last point
    max_break_gap: int = 12  # frames: a longer silence ends the flight without a kink
    max_fill: int = 6  # fill gaps up to this many frames
    recover_px: float = 8.0  # a raw candidate this close to the curve is taken as the ball
    min_pts_quad: int = 5  # points needed before the vertical parabola is trusted
    extrapolate: int = 0  # frames predicted AFTER the last detection of a flight (0 = off); a contact may follow
    min_pts_extrapolate: int = 5  # a flight needs this many points before it is extrapolated


def _fit(t: np.ndarray, u: np.ndarray, v: np.ndarray, deg_v: int = 2) -> tuple[np.ndarray, np.ndarray]:
    t0 = t.mean()
    cu = np.polyfit(t - t0, u, 1 if len(t) < 7 else 2) if len(t) >= 3 else np.polyfit(t - t0, u, 1)
    cv = np.polyfit(t - t0, v, min(deg_v, len(t) - 1))
    return np.concatenate([[t0], cu]), np.concatenate([[t0], cv])


def _eval(c: np.ndarray, t: float) -> float:
    return float(np.polyval(c[1:], t - c[0]))


class _Flight:
    def __init__(self) -> None:
        self.pts: list[tuple[int, float, float]] = []
        self.fit: tuple[np.ndarray, np.ndarray] | None = None

    def refit(self, prm: PhysicsParams) -> None:
        if len(self.pts) >= 3:
            t, u, v = (np.array(a, float) for a in zip(*self.pts, strict=True))
            self.fit = _fit(t, u, v, 2 if len(t) >= prm.min_pts_quad else 1)

    def predict(self, frame: int) -> np.ndarray:
        if self.fit is not None:
            return np.array([_eval(self.fit[0], frame), _eval(self.fit[1], frame)])
        if len(self.pts) >= 2:
            (f0, u0, v0), (f1, u1, v1) = self.pts[-2], self.pts[-1]
            k = (frame - f1) / max(1, f1 - f0)
            return np.array([u1 + (u1 - u0) * k, v1 + (v1 - v0) * k])
        return np.array([self.pts[-1][1], self.pts[-1][2]])


def _velocity(pts: list[tuple[int, float, float]], head: bool) -> np.ndarray:
    seg = pts[:3] if head else pts[-3:]
    if len(seg) < 2:
        return np.zeros(2)
    dt = max(1, seg[-1][0] - seg[0][0])
    return np.array([seg[-1][1] - seg[0][1], seg[-1][2] - seg[0][2]]) / dt


def refine_physics(states: list[BallState], blob_lists: list[list[Blob]] | None = None,
                   prm: PhysicsParams | None = None) -> tuple[list[BallState], list[Kink]]:
    prm = prm or PhysicsParams()
    n = len(states)
    vis = [(s.frame, s.uv[0], s.uv[1]) for s in states if s.uv is not None and s.visibility == "visible"]
    if len(vis) < 3:
        return list(states), []

    flights: list[_Flight] = []
    outliers: set[int] = set()
    kinks: list[Kink] = []
    cur = _Flight()
    cur.pts.append(vis[0])
    k = 1
    while k < len(vis):
        f, u, v = vis[k]
        last_f = cur.pts[-1][0]
        gap = f - last_f - 1
        if gap > prm.max_break_gap:
            flights.append(cur)
            cur = _Flight()
            cur.pts.append(vis[k])
            k += 1
            continue
        tol = prm.base_tol + prm.tol_per_gap * gap
        res = float(np.hypot(*(np.array([u, v]) - cur.predict(f))))
        if res <= tol or len(cur.pts) < 3:
            cur.pts.append(vis[k])
            cur.refit(prm)
            k += 1
            continue
        # off the arc: isolated outlier (the next point is back on it) or a kink (it keeps going elsewhere)
        if k + 1 < len(vis):
            f2, u2, v2 = vis[k + 1]
            res2 = float(np.hypot(*(np.array([u2, v2]) - cur.predict(f2))))
            if res2 <= prm.base_tol + prm.tol_per_gap * (f2 - last_f - 1):
                outliers.add(f)
                k += 1
                continue
        else:
            outliers.add(f)  # a lone last point off the arc is not evidence of anything
            k += 1
            continue
        flights.append(cur)
        v_old, new = _velocity(cur.pts, head=False), _Flight()
        new.pts.append(vis[k])
        cur = new
        k += 1
        # velocity of the new flight from the next points (needs them to exist)
        nxt = [vis[j] for j in range(k - 1, min(len(vis), k + 2))]
        v_new = _velocity(nxt, head=True)
        na, nb = float(np.linalg.norm(v_old)), float(np.linalg.norm(v_new))
        turn = 0.0
        if na > 1e-6 and nb > 1e-6:
            turn = float(np.degrees(np.arccos(np.clip(v_old @ v_new / (na * nb), -1, 1))))
        kinks.append(Kink(frame=flights[-1].pts[-1][0], speed_before=na, speed_after=nb, turn_deg=turn))
    flights.append(cur)

    out = list(states)
    for fl in flights:
        fl.pts = [p for p in fl.pts if p[0] not in outliers] if len(fl.pts) > 3 else fl.pts
        fl.refit(prm)
    # outliers: replace by the curve of the flight whose time span contains them
    for f in outliers:
        host = next((fl for fl in flights if fl.fit is not None and fl.pts[0][0] < f < fl.pts[-1][0]), None)
        if host is not None:
            p = host.predict(f)
            out[f] = BallState(frame=f, uv=(float(p[0]), float(p[1])), visibility="inferred", conf=0.3)
    for fl in flights:
        if fl.fit is None:
            continue
        for (fa, _, _), (fb, _, _) in zip(fl.pts, fl.pts[1:], strict=False):
            if 1 < fb - fa <= prm.max_fill + 1:
                for f in range(fa + 1, fb):
                    if f >= n or out[f].uv is not None and out[f].visibility == "visible":
                        continue
                    p = fl.predict(f)
                    cand = None
                    if blob_lists is not None:
                        near = [b for b in blob_lists[f] if np.hypot(b.u - p[0], b.v - p[1]) <= prm.recover_px]
                        if near:
                            cand = min(near, key=lambda b: np.hypot(b.u - p[0], b.v - p[1]))
                    if cand is not None:
                        out[f] = BallState(frame=f, uv=(cand.u, cand.v), visibility="visible",
                                           conf=min(1.0, cand.score))
                    else:
                        out[f] = BallState(frame=f, uv=(float(p[0]), float(p[1])), visibility="inferred",
                                           conf=0.3)
    if prm.extrapolate > 0:
        for idx, fl in enumerate(flights):
            if fl.fit is None or len(fl.pts) < prm.min_pts_extrapolate:
                continue
            nxt = flights[idx + 1].pts[0][0] if idx + 1 < len(flights) and flights[idx + 1].pts else n
            for f in range(fl.pts[-1][0] + 1, min(n, nxt, fl.pts[-1][0] + 1 + prm.extrapolate)):
                if out[f].uv is not None:
                    break
                p = fl.predict(f)
                out[f] = BallState(frame=f, uv=(float(p[0]), float(p[1])), visibility="inferred", conf=0.2)
    return out, kinks
