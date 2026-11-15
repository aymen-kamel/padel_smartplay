"""Label storage for the small benchmark: ball position per frame and hit frames per clip.

JSON format (one file per clip):
    {"video": "...", "fps": 30.0, "n_frames": 401, "stride": 3,
     "ball": {"12": [845.0, 301.5], "15": null},     # null = ball not visible in that frame
     "hits": [40, 98]}
Ball coordinates are pixels in the original video frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClipLabels:
    video: str
    fps: float
    n_frames: int
    stride: int = 3
    ball: dict[int, tuple[float, float] | None] = field(default_factory=dict)
    hits: set[int] = field(default_factory=set)

    def set_ball(self, frame: int, xy: tuple[float, float] | None) -> None:
        self._check(frame)
        self.ball[frame] = None if xy is None else (float(xy[0]), float(xy[1]))

    def clear_ball(self, frame: int) -> None:
        self.ball.pop(frame, None)

    def toggle_hit(self, frame: int) -> bool:
        """Add or remove a hit at ``frame``; returns True if the hit now exists."""
        self._check(frame)
        if frame in self.hits:
            self.hits.remove(frame)
            return False
        self.hits.add(frame)
        return True

    def stride_frames(self) -> list[int]:
        return list(range(0, self.n_frames, self.stride))

    def next_unlabelled(self, after: int) -> int | None:
        """First stride frame after ``after`` with no ball label (None when all are done)."""
        for f in self.stride_frames():
            if f > after and f not in self.ball:
                return f
        return None

    def progress(self) -> tuple[int, int]:
        sf = self.stride_frames()
        return sum(1 for f in sf if f in self.ball), len(sf)

    def _check(self, frame: int) -> None:
        if not 0 <= frame < self.n_frames:
            raise ValueError(f"frame {frame} outside 0..{self.n_frames - 1}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video": self.video, "fps": self.fps, "n_frames": self.n_frames, "stride": self.stride,
            "ball": {str(k): (list(v) if v is not None else None) for k, v in sorted(self.ball.items())},
            "hits": sorted(self.hits),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)  # atomic: a crash never leaves a half-written label file

    @staticmethod
    def load(path: str | Path) -> ClipLabels:
        d = json.loads(Path(path).read_text())
        return ClipLabels(
            video=d["video"], fps=d["fps"], n_frames=d["n_frames"], stride=d.get("stride", 3),
            ball={int(k): (tuple(v) if v is not None else None) for k, v in d["ball"].items()},
            hits=set(d["hits"]),
        )
