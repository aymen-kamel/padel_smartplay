"""Audio hit candidates: sharp broadband onsets (racket-on-ball sounds).

Baseline with no training: spectral flux in the 2-7 kHz band, robust z-score, peak picking.
It finds *candidate* sounds; it cannot tell a racket hit from a shoe squeak or a voice.
Accuracy is unmeasured until hit labels exist (PLAN.md, label L4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioHit:
    t: float
    strength: float  # robust z-score of the onset


def onset_strength(
    audio: np.ndarray,
    sr: int,
    hop: int = 128,
    win: int = 512,
    band: tuple[float, float] = (2000.0, 7000.0),
) -> tuple[np.ndarray, float]:
    """Spectral flux of the band energy; returns (flux per hop, hop duration in seconds)."""
    if len(audio) < win * 2:
        return np.zeros(0), hop / sr
    n = 1 + (len(audio) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = audio[idx] * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    energy = np.log1p(30.0 * spec[:, sel].sum(axis=1))
    flux = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    return flux, hop / sr


def detect_hits(
    audio: np.ndarray, sr: int, rel_min: float = 4.0, min_gap_s: float = 0.25
) -> list[AudioHit]:
    """Peaks of the onset strength at least ``rel_min`` times its 95th percentile.

    The flux is mostly zero (only rising energy counts), so a median/MAD scale collapses to
    ~0 on steady background noise and turns every small sound into a "hit". The 95th
    percentile is a scale that stays meaningful in both quiet and noisy recordings.
    """
    from scipy.signal import find_peaks

    flux, dt = onset_strength(audio, sr)
    if flux.size == 0:
        return []
    scale = float(np.percentile(flux, 95)) + 1e-9
    rel = flux / scale
    peaks, _ = find_peaks(rel, height=rel_min, distance=max(1, int(min_gap_s / dt)))
    return [AudioHit(t=float(p * dt), strength=float(rel[p])) for p in peaks]


def envelope_per_frame(audio: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """RMS loudness per video frame, scaled to [0, 1] (for drawing)."""
    out = np.zeros(n_frames)
    if audio.size == 0:
        return out
    per = sr / fps
    for i in range(n_frames):
        seg = audio[int(i * per): int((i + 1) * per)]
        out[i] = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
    peak = np.percentile(out, 99) + 1e-9
    return np.clip(out / peak, 0.0, 1.0)
