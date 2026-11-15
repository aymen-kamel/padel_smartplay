"""Cut a broadcast video into shots and find the main (wide, fixed-camera) view.

One sequential pass over the video computes, per frame: a colour histogram (for cuts), the blue
court mask at low resolution (for main-view detection) and, every ``ocr_every`` frames, the OCR
tokens of the scoreboard number panel. Everything is cached in a ``.npz`` + ``.json`` pair so the
analysis can be re-run without decoding the video again.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartplay.broadcast.scoreboard import CROP, ZOOM, Board, parse_tokens

HSV_BLUE_LO, HSV_BLUE_HI = (95, 60, 40), (130, 255, 255)
MASK_SIZE = (40, 23)


def extract_features(video: Path, out_dir: Path, ocr_every: int = 15) -> dict:
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    out_dir.mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cut, blue, masks = [], [], []
    samples: list[dict] = []
    prev = None
    y0, y1, x0, x1 = CROP
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        hist /= hist.sum() + 1e-9
        cut.append(0.0 if prev is None else 0.5 * float(np.abs(hist - prev).sum()))
        prev = hist
        mask = cv2.inRange(hsv, HSV_BLUE_LO, HSV_BLUE_HI)
        blue.append(float(mask.mean() / 255.0))
        masks.append(cv2.resize(mask, MASK_SIZE, interpolation=cv2.INTER_AREA) > 127)
        if idx % ocr_every == 0:
            big = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                             interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(big)
            toks = [
                (float(np.mean([p[0] for p in b])), float(np.mean([p[1] for p in b])), str(t))
                for b, t, _c in (res or [])
            ]
            samples.append({"frame": idx, "tokens": toks})
        idx += 1
        if idx % 1000 == 0:
            print(f"  {idx}/{n_total} frames  ({idx / (time.time() - t0):.1f} fps)", flush=True)
    cap.release()
    np.savez_compressed(out_dir / "features.npz", cut=np.array(cut), blue=np.array(blue),
                        masks=np.array(masks), fps=fps)
    (out_dir / "ocr_samples.json").write_text(json.dumps(samples))
    return {"fps": fps, "n_frames": idx}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # inclusive
    kind: str  # "main" or "other"
    blue: float
    iou: float


def detect_cuts(cut: np.ndarray, threshold: float = 0.35, min_len: int = 8) -> list[int]:
    """Frames where a new shot starts (histogram jump), keeping shots at least ``min_len`` long."""
    cand = [i for i in range(1, len(cut)) if cut[i] > threshold]
    cuts: list[int] = []
    for c in cand:
        if not cuts or c - cuts[-1] >= min_len:
            cuts.append(c)
    return cuts


def reference_mask(blue: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Typical main-view court mask: median mask over the most common blue-fraction band."""
    sel = (blue > 0.18) & (blue < 0.55)
    return np.median(masks[sel], axis=0) > 0.5


def segments(cut: np.ndarray, blue: np.ndarray, masks: np.ndarray,
             cut_threshold: float = 0.35, iou_main: float = 0.80) -> list[Segment]:
    ref = reference_mask(blue, masks)
    bounds = [0, *detect_cuts(cut, cut_threshold), len(cut)]
    out = []
    for a, b in itertools.pairwise(bounds):
        m = masks[a:b]
        inter = (m & ref).sum(axis=(1, 2))
        union = (m | ref).sum(axis=(1, 2))
        iou = float(np.median(inter / np.maximum(union, 1)))
        out.append(Segment(a, b - 1, "main" if iou >= iou_main else "other",
                           float(np.mean(blue[a:b])), iou))
    return out


def board_timeline(samples: list[dict], min_repeat: int = 2) -> list[tuple[int, Board]]:
    """Stable scoreboard readings: a reading counts when it repeats in consecutive samples."""
    reads = [(s["frame"], parse_tokens([tuple(t) for t in s["tokens"]])) for s in samples]
    out: list[tuple[int, Board]] = []
    i = 0
    while i < len(reads):
        f, b = reads[i]
        if b is None:
            i += 1
            continue
        j = i
        while j + 1 < len(reads) and reads[j + 1][1] is not None and reads[j + 1][1].key() == b.key():
            j += 1
        if j - i + 1 >= min_repeat and (not out or out[-1][1].key() != b.key()):
            out.append((f, b))
        i = j + 1
    return out
