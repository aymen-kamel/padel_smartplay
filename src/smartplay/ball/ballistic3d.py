"""Ball in 3D from one camera: fit a projectile (gravity known) to each 2D flight.

Between two contacts the ball follows p(t) = p0 + v0 t - 1/2 g t^2 z (air drag ignored: small over ~1 s flights, to be
measured). Six unknowns (p0, v0); each detection gives two equations through the 3x4 camera P, so a flight with enough
visible frames determines its 3D path, including depth: gravity fixes the scale that a single camera cannot see.
A flight grows while one projectile explains the detections; where it stops, there was a contact.

Every fitted flight is checked, and rejected (not guessed) when it is implausible: reprojection error, ball under the
floor, outside the court volume, faster than any padel shot. Contacts are typed from where a flight ends:
floor (z ~ 0), glass (|x| ~ 5 m or |y| ~ 10 m), otherwise a racket hit / unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartplay.geometry.camera3d import project
from smartplay.geometry.court import HALF_LENGTH_M, HALF_WIDTH_M
from smartplay.schemas.records import BallState

G = 9.81


@dataclass(frozen=True)
class Flight3D:
    f0: int
    f1: int
    p0: np.ndarray  # position at f0 (m)
    v0: np.ndarray  # velocity at f0 (m/s)
    rms_px: float
    n_obs: int
    end_contact: str  # floor / glass / other
    anchor: str = "free"  # free / player (starts at a player) / bounce (continues a floor bounce)

    def at(self, frame: float, fps: float) -> np.ndarray:
        t = (frame - self.f0) / fps
        return self.p0 + self.v0 * t + np.array([0.0, 0.0, -0.5 * G * t * t])

    def speed_kmh(self, frame: float, fps: float) -> float:
        t = (frame - self.f0) / fps
        return float(np.linalg.norm(self.v0 + np.array([0.0, 0.0, -G * t])) * 3.6)


def _fit_one(P: np.ndarray, frames: np.ndarray, uv: np.ndarray, fps: float, cam, x0: np.ndarray | None = None,
             prior=None) -> tuple[np.ndarray, float, float] | None:
    """Least-squares projectile fit; returns (params, rms px, residual of the last point px).

    ``prior(x)`` adds soft constraints (residuals in pixel-equivalent units) to resolve the one-camera depth ambiguity."""
    from scipy.optimize import least_squares

    t = (frames - frames[0]) / fps

    def model(x):
        return x[:3] + np.outer(t, x[3:]) + np.outer(0.5 * G * t * t, [0, 0, -1.0])

    def resid(x):
        return (project(P, model(x)) - uv).ravel()

    def full(x):
        r = resid(x)
        return r if prior is None else np.concatenate([r, prior(x)])

    starts = [x0] if x0 is not None else []
    for z0 in (0.5, 1.2, 2.5, 4.0):
        try:
            a = cam.point_at_height(*uv[0], z0)
            b = cam.point_at_height(*uv[-1], z0)
        except (ValueError, np.linalg.LinAlgError):
            continue
        v = (b - a) / max(t[-1], 1e-3)
        v[2] = 0.5 * G * t[-1]  # rises then falls back to the same height
        starts.append(np.concatenate([a, v]))
    best = None
    for s in starts:
        r = least_squares(full, s, loss="soft_l1", f_scale=3.0, max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    e = np.linalg.norm(resid(best.x).reshape(-1, 2), axis=1)
    return best.x, float(np.sqrt(np.mean(e**2))), float(e[-1])


def _plausible(x: np.ndarray, n_frames: int, fps: float, max_speed_kmh: float = 160.0) -> bool:
    t = np.arange(n_frames) / fps
    path = x[:3] + np.outer(t, x[3:]) + np.outer(0.5 * G * t * t, [0, 0, -1.0])
    speed = max(np.linalg.norm(x[3:]), np.linalg.norm(x[3:] + [0, 0, -G * t[-1]])) * 3.6
    return bool(path[:, 2].min() > -0.3 and path[:, 2].max() < 12.0
                and np.all(np.abs(path[:, 0]) < HALF_WIDTH_M + 1.5) and np.all(np.abs(path[:, 1]) < HALF_LENGTH_M + 1.5)
                and speed < max_speed_kmh)


def _contact(fl: Flight3D, fps: float) -> str:
    end = fl.at(fl.f1, fps)
    if end[2] < 0.35:
        return "floor"
    if abs(abs(end[0]) - HALF_WIDTH_M) < 0.6 or abs(abs(end[1]) - HALF_LENGTH_M) < 0.6:
        return "glass"
    return "other"


def fit_flights(states: list[BallState], cam, fps: float, min_obs: int = 8, max_gap: int = 6,
                max_rms_px: float = 6.0, max_speed_kmh: float = 160.0,
                players: dict[int, list[tuple[float, float]]] | None = None) -> list[Flight3D]:
    """Grow each flight while one projectile explains the detections; a contact is where it stops explaining them.

    Start with ``min_obs`` consecutive detections; if a plausible projectile fits them within ``max_rms_px``, add the next
    detection while the fit stays good (and the new point itself is within 2 x ``max_rms_px``); then start a new flight
    after it. Windows that never fit are skipped one detection at a time (no 3D is reported there).

    Depth ambiguity of one camera is reduced by two physical priors, tried as hypotheses:
      * bounce: right after a flight that ended on the floor, the next one starts where it landed, near the floor;
      * player: a flight that starts with a hit starts within reach of a player (players' floor positions in metres).
    A bounce continuation is preferred when it fits; between a free fit and a player-anchored one, the anchored one is
    kept when it reprojects (almost) as well, i.e. when the images cannot tell them apart.
    """
    obs = [(s.frame, s.uv) for s in states if s.uv is not None and s.visibility == "visible"]
    out: list[Flight3D] = []

    def ok(res, n_fr):
        return res is not None and res[1] <= max_rms_px and _plausible(res[0], n_fr, fps, max_speed_kmh)

    i = 0
    while i + min_obs <= len(obs):
        j = i + min_obs
        if any(obs[k + 1][0] - obs[k][0] > max_gap for k in range(i, j - 1)):
            i += 1
            continue
        f_start = obs[i][0]
        hyps = []
        prev = out[-1] if out else None
        if prev is not None and prev.end_contact == "floor" and f_start - prev.f1 <= max_gap + 2:
            land = prev.at(prev.f1, fps)
            w = 10.0 / (1.0 + (f_start - prev.f1))

            def bounce_prior(x, land=land, w=w):
                return w * np.array([x[0] - land[0], x[1] - land[1], x[2]])

            hyps.append(("bounce", bounce_prior))
        hyps.append(("free", None))
        if players and players.get(f_start):
            feet = np.array(players[f_start], float)

            def player_prior(x, feet=feet):
                d = float(np.min(np.hypot(feet[:, 0] - x[0], feet[:, 1] - x[1])))
                return 8.0 * np.array([max(0.0, d - 1.3), max(0.0, 0.2 - x[2]), max(0.0, x[2] - 3.0)])

            hyps.append(("player", player_prior))
        frames = np.array([f for f, _ in obs[i:j]], float)
        uv = np.array([u for _, u in obs[i:j]], float)
        n_fr = int(frames[-1] - frames[0]) + 1
        fits = {}
        for name, pr in hyps:
            r = _fit_one(cam.P, frames, uv, fps, cam, prior=pr)
            if ok(r, n_fr):
                fits[name] = (r, pr)
        if not fits:
            i += 1
            continue
        if "bounce" in fits:
            pick = "bounce"
        elif "player" in fits and ("free" not in fits or fits["player"][0][1] <= fits["free"][0][1] + 1.0):
            pick = "player"
        else:
            pick = "free"
        (x, rms, _), prior = fits[pick]
        while j < len(obs) and obs[j][0] - obs[j - 1][0] <= max_gap:
            fr2 = np.array([f for f, _ in obs[i:j + 1]], float)
            uv2 = np.array([u for _, u in obs[i:j + 1]], float)
            r2 = _fit_one(cam.P, fr2, uv2, fps, cam, x0=x, prior=prior)
            if not ok(r2, int(fr2[-1] - fr2[0]) + 1) or r2[2] > 2 * max_rms_px:
                break
            x, rms = r2[0], r2[1]
            j += 1
        fl = Flight3D(obs[i][0], obs[j - 1][0], x[:3], x[3:], rms, j - i, "", pick)
        out.append(Flight3D(fl.f0, fl.f1, fl.p0, fl.v0, rms, j - i, _contact(fl, fps), pick))
        i = j
    return out


def states_with_3d(states: list[BallState], flights: list[Flight3D], fps: float) -> list[BallState]:
    """Copy of the 2D states with ``xyz_m`` filled inside fitted flights."""
    out = list(states)
    for fl in flights:
        for f in range(fl.f0, fl.f1 + 1):
            if f < len(out):
                s = out[f]
                out[f] = BallState(frame=s.frame, uv=s.uv, xyz_m=tuple(float(v) for v in fl.at(f, fps)),
                                   visibility=s.visibility if s.uv is not None else "inferred", conf=s.conf)
    return out


@dataclass(frozen=True)
class Bounce:
    t_frame: float  # bounce instant, in (fractional) frames
    xy: tuple[float, float]  # landing point on the floor (m), z = 0 exactly
    restitution: float  # fitted vertical restitution |vz_out| / |vz_in| (FIP ball ~ 0.74: an independent check)
    horiz_ratio: float  # horizontal speed after / before (< 1 expected: friction)
    rms_px: float  # reprojection error of the joint fit over both flights
    rms_before_px: float  # mean of the two separate fits, for comparison


def _obs(states: list[BallState], f0: int, f1: int):
    pts = [(s.frame, s.uv) for s in states[f0:f1 + 1] if s.uv is not None and s.visibility == "visible"]
    return np.array([f for f, _ in pts], float), np.array([u for _, u in pts], float)


def join_bounces(states: list[BallState], flights: list[Flight3D], cam, fps: float, max_gap: int = 8,
                 max_rms_px: float = 6.0, max_speed_kmh: float = 160.0, tried: list | None = None
                 ) -> tuple[list[Flight3D], list[Bounce]]:
    """Refit pairs of consecutive flights that meet near the floor as ONE path through a floor bounce.

    Hard constraints: the ball is at z = 0 at the bounce instant tb (one shared point B for both arcs); before tb it
    arrives with v_in (vz < 0), after tb it leaves with v_out where vz_out = -e * vz_in. The horizontal velocity after the
    bounce and the restitution e (bounded 0.3-1.0) are free, so the fitted e is an independent physical check.
    The pair is replaced only if the joint fit reprojects within ``max_rms_px`` and stays plausible.
    """
    from scipy.optimize import least_squares

    out = list(flights)
    bounces: list[Bounce] = []
    k = 0
    while k + 1 < len(out):
        a, b = out[k], out[k + 1]
        # separate fits often disagree at a contact (wrong depth on each side), so their heights are not used to
        # decide; any close pair is tried and the images decide (joint reprojection error, plausibility)
        if b.f0 - a.f1 > max_gap:
            k += 1
            continue
        fa, ua = _obs(states, a.f0, a.f1)
        fb, ub = _obs(states, b.f0, b.f1)
        frames = np.concatenate([fa, fb])
        uv = np.vstack([ua, ub])
        t = (frames - a.f0) / fps
        n_a = len(fa)

        def path(x, t=t, n_a=n_a):
            bx, by, tb, vix, viy, viz, vox, voy, e = x
            B = np.array([bx, by, 0.0])
            d = t - tb
            v_in, v_out = np.array([vix, viy, viz]), np.array([vox, voy, -e * viz])
            v = np.where((np.arange(len(t)) < n_a)[:, None], v_in, v_out)
            return B + v * d[:, None] + np.outer(-0.5 * G * d * d, [0, 0, 1.0])

        def resid(x, uv=uv, path=path):
            return (project(cam.P, path(x)) - uv).ravel()

        tb0 = (0.5 * (a.f1 + b.f0) - a.f0) / fps
        land = a.at(a.f1, fps)
        vin = a.v0 + np.array([0, 0, -G * (a.f1 - a.f0) / fps])
        x0 = np.array([land[0], land[1], tb0, vin[0], vin[1], min(vin[2], -0.5), b.v0[0], b.v0[1], 0.74])
        lo = [-7, -12, (a.f1 - a.f0 - 2) / fps, -60, -60, -40, -60, -60, 0.3]
        hi = [7, 12, (b.f0 - a.f0 + 2) / fps, 60, 60, -0.1, 60, 60, 1.0]
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, max_nfev=600)
        err = np.linalg.norm(resid(r.x).reshape(-1, 2), axis=1)
        rms = float(np.sqrt(np.mean(err**2)))
        bx, by, tb, vix, viy, viz, vox, voy, e = r.x
        v_in, v_out = np.array([vix, viy, viz]), np.array([vox, voy, -e * viz])
        speeds = [np.linalg.norm(v_in + [0, 0, G * tb]) * 3.6, np.linalg.norm(v_in) * 3.6, np.linalg.norm(v_out) * 3.6]
        p = path(r.x)
        hz_in, hz_out = np.linalg.norm(v_in[:2]), np.linalg.norm(v_out[:2])
        # physical acceptance: a floor bounce keeps 45-95 % of the vertical speed (FIP ball ~74 %; values stuck on the
        # 0.3 / 1.0 bounds mean the model forced a bounce that is not there) and does not speed up horizontally
        # (a clear horizontal gain means a racket hit; the horizontal ratio is poorly determined with one camera,
        # 1.30 measured for a true 0.81 on a simulated bounce, so only a gain above 2x is used to reject)
        if tried is not None:  # every attempt, accepted or not, for reporting
            tried.append({"frames": (a.f0, b.f1), "t_frame": float(a.f0 + tb * fps), "e": float(e), "rms_px": rms,
                          "horiz_ratio": float(hz_out / max(hz_in, 1e-6)), "max_kmh": float(max(speeds)),
                          "z_min": float(p[:, 2].min())})
        if (rms > max_rms_px or max(speeds) > max_speed_kmh or p[:, 2].min() < -0.05
                or not (0.45 <= e <= 0.95) or hz_out > 2.0 * hz_in):
            k += 1
            continue
        B = np.array([bx, by, 0.0])
        tb_frame = a.f0 + tb * fps
        t0 = 0.0  # flight A rebuilt from the joint fit, drawn until the bounce
        p0_a = B + v_in * (t0 - tb) + np.array([0, 0, -0.5 * G * (t0 - tb) ** 2])
        v0_a = v_in + np.array([0, 0, -G * (t0 - tb)])
        f1_a = max(a.f1, int(np.floor(tb_frame)))
        f0_b = min(b.f0, int(np.ceil(tb_frame)))
        tB = (f0_b - a.f0) / fps - tb
        p0_b = B + v_out * tB + np.array([0, 0, -0.5 * G * tB * tB])
        v0_b = v_out + np.array([0, 0, -G * tB])
        out[k] = Flight3D(a.f0, f1_a, p0_a, v0_a, rms, a.n_obs, "floor", a.anchor)
        out[k + 1] = Flight3D(f0_b, b.f1, p0_b, v0_b, rms, b.n_obs, b.end_contact, "bounce")
        bounces.append(Bounce(float(tb_frame), (float(bx), float(by)), float(e), float(hz_out / max(hz_in, 1e-6)),
                              rms, 0.5 * (a.rms_px + b.rms_px)))
        k += 1
    return out, bounces
