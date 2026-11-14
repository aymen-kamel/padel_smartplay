"""Video and audio input. OpenCV decodes frames; an FFmpeg binary extracts the audio track."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from smartplay.schemas.records import VideoInfo


def ffmpeg_exe() -> str:
    """An ffmpeg executable: the one on PATH, else the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc


def probe(path: str | Path) -> VideoInfo:
    """Read fps, size, frame count from the container; detect an audio stream."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False
    ).stderr
    has_audio = re.search(r"Stream #.*Audio:", out) is not None
    return VideoInfo(path=str(path), has_audio=has_audio, **info)


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR image) one frame at a time; memory stays bounded."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def extract_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 audio in [-1, 1] at ``sample_rate`` Hz (empty array if there is no audio)."""
    cmd = [
        ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
        "-ar", str(sample_rate), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
