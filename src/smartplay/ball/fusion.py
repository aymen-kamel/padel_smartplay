"""Merge ball candidates from two detectors (first = better placed, second = finds more balls)."""

from __future__ import annotations

import math

from smartplay.ball.detector import Blob


def merge_blobs(primary: list[list[Blob]], secondary: list[list[Blob]],
                radius_px: float = 12.0, secondary_weight: float = 0.9) -> list[list[Blob]]:
    """Keep every primary candidate; add a secondary one only if no primary lies within ``radius_px``.

    Added candidates get their score scaled by ``secondary_weight`` so that, when both are plausible,
    the tracker prefers the better-placed primary detector. Frames beyond the shorter list are dropped.
    """
    out = []
    for fa, fb in zip(primary, secondary, strict=False):
        merged = list(fa)
        for b in fb:
            if not any(math.hypot(b.u - a.u, b.v - a.v) < radius_px for a in fa):
                merged.append(Blob(b.u, b.v, b.score * secondary_weight))
        out.append(merged)
    return out
