"""Four-player tracker on the court floor.

Padel always has exactly four players: two per side. Identities are set on the first frame
where four players are seen on court: ids 1-2 = the two NEAR players (team A, left to right),
ids 3-4 = the two FAR players (team B). Each later frame assigns detections to the four
identities by court distance (Hungarian); a missing player coasts on its last velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from smartplay.geometry.court import side_of
from smartplay.schemas.records import PlayerTrack


@dataclass
class _Track:
    player_id: int
    team: str
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0
    conf: float = 1.0


class FourPlayerTracker:
    def __init__(self, gate_m: float = 2.5, max_gate_m: float = 6.0) -> None:
        self.gate_m, self.max_gate_m = gate_m, max_gate_m
        self.tracks: list[_Track] = []

    @property
    def initialized(self) -> bool:
        return bool(self.tracks)

    def update(
        self, frame: int, foot_xy_m: list[tuple[float, float]], confs: list[float]
    ) -> list[PlayerTrack]:
        """``foot_xy_m``: candidate floor positions already filtered to the court area."""
        if not self.tracks:
            if len(foot_xy_m) < 4:
                return []
            order = np.argsort(confs)[::-1][:4]
            pts = [np.array(foot_xy_m[i]) for i in order]
            cs = [confs[i] for i in order]
            near = sorted(range(4), key=lambda k: pts[k][1])[:2]
            far = [k for k in range(4) if k not in near]
            near.sort(key=lambda k: pts[k][0])
            far.sort(key=lambda k: pts[k][0])
            for pid, k in zip((1, 2, 3, 4), near + far, strict=True):
                self.tracks.append(_Track(pid, "A" if pid <= 2 else "B", pts[k], conf=cs[k]))
            return self._emit(frame, {t.player_id for t in self.tracks})

        matched: set[int] = set()
        if foot_xy_m:
            cand = np.array(foot_xy_m)
            pred = np.array([t.pos + t.vel for t in self.tracks])
            cost = np.linalg.norm(pred[:, None, :] - cand[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                t = self.tracks[r]
                if cost[r, c] <= min(self.gate_m + 0.15 * t.missed, self.max_gate_m):
                    new = cand[c]
                    t.vel = 0.7 * t.vel + 0.3 * (new - t.pos)
                    t.pos, t.missed, t.conf = new, 0, confs[c]
                    matched.add(t.player_id)
        for t in self.tracks:
            if t.player_id not in matched:
                t.missed += 1
                t.vel = t.vel * 0.9
                t.pos = t.pos + t.vel
                t.conf = max(0.0, t.conf - 0.05)
        return self._emit(frame, matched)

    def _emit(self, frame: int, observed: set[int]) -> list[PlayerTrack]:
        return [
            PlayerTrack(
                frame=frame,
                player_id=t.player_id,  # type: ignore[arg-type]
                team=t.team,  # type: ignore[arg-type]
                side=side_of(float(t.pos[1])),
                foot_xy_m=(float(t.pos[0]), float(t.pos[1])),
                conf=float(min(1.0, max(0.0, t.conf))),
                observed=t.player_id in observed,
            )
            for t in self.tracks
        ]
