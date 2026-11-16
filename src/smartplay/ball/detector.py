"""Ball detection and tracking with the WASB HRNet model (MIT, BMVC 2023).

Model: 3 consecutive RGB frames, resized to 512x288 and ImageNet-normalised, stacked into 9
channels -> 3 heatmaps (one per input frame). We slide the window by one frame and average the
up-to-3 heatmaps each frame receives, then keep blobs above ``threshold`` (sigmoid score).

Tracking follows WASB's online tracker: among the blobs of a frame, prefer the strongest one
within ``max_disp_px`` of the last ball position; otherwise take the strongest blob.
Gaps are NOT filled: a frame without a confident blob is reported as ``lost`` (the audit showed
that unlimited interpolation invented positions).

The shipped weights were trained on broadcast **tennis**. On padel this is zero-shot use; its
accuracy is unmeasured until ball labels exist (PLAN.md, label L3).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartplay.schemas.records import BallState

WASB_CFG = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                       "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64],
                       "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                       "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}
class _AttrDict(dict):
    """Dict that also allows attribute access (the vendored HRNet code uses both styles)."""

    def __init__(self, data: dict) -> None:
        super().__init__({k: _AttrDict(v) if isinstance(v, dict) else v for k, v in data.items()})

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Blob:
    u: float
    v: float
    score: float


def heatmap_blobs(hm: np.ndarray, threshold: float, sx: float, sy: float) -> list[Blob]:
    """Connected components above ``threshold``; centre = score-weighted centroid (pixels)."""
    import cv2

    if hm.max() <= threshold:
        return []
    mask = (hm > threshold).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask)
    blobs = []
    for k in range(1, n):
        ys, xs = np.nonzero(labels == k)
        w = hm[ys, xs]
        blobs.append(Blob(float((xs * w).sum() / w.sum() + 0.5) * sx,
                          float((ys * w).sum() / w.sum() + 0.5) * sy, float(w.max())))
    return blobs


class OnlineBallTracker:
    """Pick one ball per frame (WASB-style online tracking, no gap filling)."""

    def __init__(self, max_disp_px: float = 300.0, max_gap: int = 5) -> None:
        self.max_disp, self.max_gap = max_disp_px, max_gap
        self.last: tuple[float, float] | None = None
        self.gap = 0

    def step(self, frame: int, blobs: list[Blob]) -> BallState:
        chosen: Blob | None = None
        if blobs:
            if self.last is not None and self.gap <= self.max_gap:
                near = [b for b in blobs if np.hypot(b.u - self.last[0], b.v - self.last[1])
                        <= self.max_disp * (1 + self.gap)]
                if near:
                    chosen = max(near, key=lambda b: b.score)
            if chosen is None:
                chosen = max(blobs, key=lambda b: b.score)
        if chosen is None:
            self.gap += 1
            return BallState(frame=frame, visibility="lost", conf=0.0)
        self.last, self.gap = (chosen.u, chosen.v), 0
        return BallState(frame=frame, uv=(chosen.u, chosen.v), visibility="visible",
                         conf=min(1.0, chosen.score))


class WasbBallDetector:
    def __init__(self, weights: str | Path, threshold: float = 0.5, device: str | None = None
                 ) -> None:
        import torch

        from smartplay.ball.wasb_hrnet import HRNet

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        model = HRNet(_AttrDict(WASB_CFG))
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        self.model = model.to(self.device).eval()
        self.threshold = threshold
        self.h, self.w = WASB_CFG["inp_height"], WASB_CFG["inp_width"]

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)  # (3, H, W)

    def blobs(self, frames: Iterable[tuple[int, np.ndarray]], width: int, height: int
              ) -> list[list[Blob]]:
        """Every ball candidate (heatmap blob) of every frame; the tracker decides later."""
        torch = self.torch
        sx, sy = width / self.w, height / self.h
        window: deque[np.ndarray] = deque(maxlen=3)
        acc: dict[int, list[np.ndarray]] = {}
        out: list[list[Blob]] = []
        next_out = 0

        def flush(upto: int) -> None:
            nonlocal next_out
            while next_out <= upto and next_out in acc:
                hm = np.mean(acc.pop(next_out), axis=0)
                out.append(heatmap_blobs(hm, self.threshold, sx, sy))
                next_out += 1

        last = -1
        for idx, frame in frames:
            window.append(self._prep(frame))
            last = idx
            if len(window) < 3:
                continue
            x = torch.from_numpy(np.concatenate(list(window), axis=0)[None]).to(self.device)
            with torch.no_grad():
                hms = self.model(x)[0].sigmoid()[0].cpu().numpy()  # (3, H, W)
            for k in range(3):
                acc.setdefault(idx - 2 + k, []).append(hms[k])
            flush(idx - 2)  # frame idx-2 has now received all its windows
        flush(last)
        while next_out <= last:  # clips shorter than 3 frames
            out.append([])
            next_out += 1
        return out

    def track(self, frames: Iterable[tuple[int, np.ndarray]], width: int, height: int
              ) -> list[BallState]:
        """One ``BallState`` per frame, using the gated tracker (rejects glass reflections)."""
        return track_gated(self.blobs(frames, width, height))


def track_gated(
    blob_lists: list[list[Blob]],
    gate_px: float = 70.0,
    vel_gain: float = 0.5,
    max_gap: int = 8,
    reacq_score: float = 0.6,
    confirm: int = 2,
) -> list[BallState]:
    """Choose one ball per frame from all candidates, following the trajectory.

    A reflection of the ball in the glass is a second blob far from where the real ball is going.
    Instead of taking the strongest blob near the last position (the baseline), this tracker
    predicts the next position from the last two (constant velocity), accepts only candidates
    inside a small gate around the prediction and takes the closest one. When the track is lost
    for more than ``max_gap`` frames, a new track starts only from a strong blob that is
    confirmed by ``confirm`` consecutive frames (the first of them is then reported too).
    Gaps are never filled: a frame with no accepted candidate is ``lost``.
    """
    n = len(blob_lists)
    states = [BallState(frame=i, visibility="lost", conf=0.0) for i in range(n)]
    last: np.ndarray | None = None
    vel = np.zeros(2)
    gap = 0
    i = 0
    while i < n:
        blobs = blob_lists[i]
        if last is not None and gap <= max_gap:
            pred = last + vel * (gap + 1)
            radius = gate_px + vel_gain * float(np.linalg.norm(vel)) * (gap + 1) + 20.0 * gap
            best, best_cost = None, 1e9
            for b in blobs:
                d = float(np.hypot(b.u - pred[0], b.v - pred[1]))
                if d <= radius and d - 20.0 * b.score < best_cost:
                    best, best_cost = b, d - 20.0 * b.score
            if best is not None:
                new = np.array([best.u, best.v])
                vel = 0.5 * vel + 0.5 * (new - last) / (gap + 1)
                last, gap = new, 0
                states[i] = BallState(frame=i, uv=(best.u, best.v), visibility="visible",
                                      conf=min(1.0, best.score))
            else:
                gap += 1
            i += 1
            continue
        # no active track: look for a confirmed start
        started = False
        for b in sorted(blobs, key=lambda x: -x.score):
            if b.score < reacq_score:
                break
            chain, pos = [b], np.array([b.u, b.v])
            for j in range(i + 1, min(n, i + confirm)):
                nxt = [c for c in blob_lists[j]
                       if np.hypot(c.u - pos[0], c.v - pos[1]) <= gate_px * 2]
                if not nxt:
                    break
                c = min(nxt, key=lambda x: np.hypot(x.u - pos[0], x.v - pos[1]))
                chain.append(c)
                pos = np.array([c.u, c.v])
            if len(chain) >= confirm:
                last, vel, gap = np.array([b.u, b.v]), np.zeros(2), 0
                states[i] = BallState(frame=i, uv=(b.u, b.v), visibility="visible",
                                      conf=min(1.0, b.score))
                started = True
                break
        if not started:
            gap = max_gap + 1  # stay in search mode
        i += 1
    return states
