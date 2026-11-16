"""Running player statistics: distance covered and current speed, from tracked floor positions.

Raw per-frame positions jitter (the old prototype reported 175 m in a 37 s clip because every
jitter step was summed). Here positions are smoothed over a short window, steps faster than a
human can run are ignored, and predicted (not observed) positions never add distance.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from smartplay.schemas.records import PlayerTrack

MAX_HUMAN_SPEED_MS = 9.0  # steps above this are tracking glitches, not running
SMOOTH_FRAMES = 5
SPEED_WINDOW_S = 0.6


class PlayerStats:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self._raw: dict[int, deque[np.ndarray]] = {}
        self._smooth: dict[int, deque[np.ndarray]] = {}
        self._dist: dict[int, float] = {}
        self._team: dict[int, str] = {}
        self._max_step = MAX_HUMAN_SPEED_MS / fps

    def update(self, tracks: list[PlayerTrack]) -> None:
        for t in tracks:
            pid = t.player_id
            self._team[pid] = t.team
            self._dist.setdefault(pid, 0.0)
            hist = self._smooth.setdefault(pid, deque(maxlen=max(2, round(self.fps * SPEED_WINDOW_S))))
            if not t.observed:
                continue
            raw = self._raw.setdefault(pid, deque(maxlen=SMOOTH_FRAMES))
            raw.append(np.array(t.foot_xy_m))
            smoothed = np.mean(raw, axis=0)
            if hist and 0 < np.linalg.norm(smoothed - hist[-1]) <= self._max_step:
                self._dist[pid] += float(np.linalg.norm(smoothed - hist[-1]))
            elif hist and np.linalg.norm(smoothed - hist[-1]) > self._max_step:
                raw.clear()  # glitch: restart the smoothing window from here
                raw.append(np.array(t.foot_xy_m))
                smoothed = raw[-1]
            hist.append(smoothed)

    def snapshot(self) -> dict[int, dict]:
        """Per player: team, distance covered so far (m), current speed (km/h)."""
        out = {}
        for pid in sorted(self._dist):
            hist = self._smooth.get(pid, deque())
            speed = 0.0
            if len(hist) >= 2:
                pts = np.array(hist)
                path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                speed = min(path / ((len(hist) - 1) / self.fps) * 3.6, MAX_HUMAN_SPEED_MS * 3.6)
            out[pid] = {"team": self._team[pid], "dist_m": self._dist[pid], "speed_kmh": speed}
        return out
