"""Learned audio hit detector: low-threshold onset candidates + a classifier on their acoustic context.

The untrained detector (``audio.detect_hits``) keeps onsets above a fixed multiple of the background. It cannot tell a
racket hit from a shoe squeak, a voice or the crowd. Here every onset peak (low threshold, high recall) becomes a
candidate described by band energies before/after the onset, spectral shape, decay, and spacing; a gradient-boosted
classifier trained on labelled hits scores each candidate, then peaks closer than ``min_gap_s`` keep the best score.
"""

from __future__ import annotations

import numpy as np

from smartplay.events.audio import onset_strength

BANDS = ((300, 1000), (1000, 2000), (2000, 4000), (4000, 7000), (7000, 7900))
FEATURE_NAMES = (
    ["rel", "rel_nb_max", "dt_prev", "dt_next", "t_frac", "loud_ctx"]
    + [f"post_b{i}" for i in range(len(BANDS))] + [f"contrast_b{i}" for i in range(len(BANDS))]
    + ["centroid", "flatness", "decay"]
)


def candidates(audio: np.ndarray, sr: int, rel_min: float = 1.5, min_gap_s: float = 0.06):
    """Onset peaks above ``rel_min`` x the 95th percentile of the flux. Returns (times, rel, flux, dt)."""
    from scipy.signal import find_peaks

    flux, dt = onset_strength(audio, sr)
    if flux.size == 0:
        return np.zeros(0), np.zeros(0), flux, dt
    rel = flux / (float(np.percentile(flux, 95)) + 1e-9)
    peaks, _ = find_peaks(rel, height=rel_min, distance=max(1, int(min_gap_s / dt)))
    return peaks * dt, rel[peaks], rel, dt


def _band_energy(audio: np.ndarray, sr: int, a: int, b: int) -> np.ndarray:
    seg = audio[max(0, a):max(a + 64, b)]
    if seg.size < 64:
        return np.full(len(BANDS), -12.0), 0.0, 1.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2 / seg.size
    f = np.fft.rfftfreq(seg.size, 1.0 / sr)
    e = np.array([np.log(spec[(f >= lo) & (f < hi)].mean() + 1e-12) for lo, hi in BANDS])
    tot = spec[(f >= 300) & (f < 7900)] + 1e-12
    centroid = float((f[(f >= 300) & (f < 7900)] * tot).sum() / tot.sum())
    flatness = float(np.exp(np.log(tot).mean()) / tot.mean())
    return e, centroid, flatness


def features(audio: np.ndarray, sr: int, t_c: np.ndarray, rel_c: np.ndarray, rel_all: np.ndarray, dt: float
             ) -> np.ndarray:
    """One row per candidate (columns = FEATURE_NAMES)."""
    n = len(t_c)
    dur = len(audio) / sr
    rows = np.zeros((n, len(FEATURE_NAMES)))
    rms = np.sqrt(np.convolve(audio**2, np.ones(sr // 10) / (sr // 10), mode="same"))
    for i, (t, r) in enumerate(zip(t_c, rel_c, strict=True)):
        c = int(t * sr)
        post, centroid, flat = _band_energy(audio, sr, c, c + int(0.04 * sr))
        pre, _, _ = _band_energy(audio, sr, c - int(0.08 * sr), c - int(0.01 * sr))
        late, _, _ = _band_energy(audio, sr, c + int(0.08 * sr), c + int(0.2 * sr))
        lo, hi = max(0, int((t - 0.1) / dt)), min(len(rel_all), int((t + 0.1) / dt) + 1)
        ctx = rms[max(0, c - sr): c + sr]
        rows[i] = np.concatenate([
            [r, float(rel_all[lo:hi].max()), (t - t_c[i - 1]) if i else 5.0, (t_c[i + 1] - t) if i + 1 < n else 5.0,
             t / max(dur, 1e-6), float(np.median(ctx)) if ctx.size else 0.0],
            post, post - pre, [centroid, flat, float(post[2:4].mean() - late[2:4].mean())]])
    rows[:, 2] = np.clip(rows[:, 2], 0, 5)
    rows[:, 3] = np.clip(rows[:, 3], 0, 5)
    return rows


def suppress(times: np.ndarray, scores: np.ndarray, min_gap_s: float = 0.25) -> np.ndarray:
    """Greedy non-maximum suppression: keep the best score, drop neighbours closer than ``min_gap_s``."""
    keep: list[int] = []
    for i in np.argsort(-scores):
        if all(abs(times[i] - times[j]) >= min_gap_s for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), dtype=int)
