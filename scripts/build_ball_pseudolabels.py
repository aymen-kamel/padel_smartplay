"""Build a TrackNet fine-tuning set from CVSPORTS rallies with CONFIDENT pseudo-labels.

    python scripts/build_ball_pseudolabels.py --n-videos 30 --out data/train/tracknet

Labels come from our own fused ball track (runs/vision_hit_features.json: WASB + TrackNetV3 candidates, gated tracker,
physics post-processor). Only confident frames are kept as positives: detected (not inferred), with detected neighbours
on both sides, and lying on a smooth path (|p(f) - midpoint(p(f-1), p(f+1))| < 4 px, steps < 80 px). Every other frame is
UNSUPERVISED (mask 0): it is neither "ball" nor "no ball". Our hand-labelled clips (point_12, point_24) are never used.

Per video, writes <name>.npy (frames, 288x512 RGB uint8), <name>_bg.npy (median background, 3x288x512 uint8),
<name>_lab.npy (n x 3: x, y in 512x288 pixels, valid 0/1).
EXPERIMENT: CVSPORTS_Padel has no stated licence; a model trained on it is not for release until that is cleared.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

H, W = 288, 512
ap = argparse.ArgumentParser()
ap.add_argument("--n-videos", type=int, default=30)
ap.add_argument("--out", type=Path, default=Path("data/train/tracknet"))
ap.add_argument("--src", type=Path, default=Path("data/external/cvsports_padel/rallies"))
args = ap.parse_args()

vis = json.loads(Path("runs/vision_hit_features.json").read_text())
names = sorted(vis)
step = max(1, len(names) // args.n_videos)
chosen = names[::step][: args.n_videos]
args.out.mkdir(parents=True, exist_ok=True)
tot_frames = tot_pos = 0
for name in chosen:
    d = vis[name]
    ball = d["ball"]  # [frame, u, v, visibility]
    n = len(ball)
    lab = np.zeros((n, 3), np.float32)
    for i in range(1, n - 1):
        a, b, c = ball[i - 1], ball[i], ball[i + 1]
        if not all(x[3] == "visible" and x[1] is not None for x in (a, b, c)):
            continue
        pa, pb, pc = (np.array(x[1:3], float) for x in (a, b, c))
        if np.linalg.norm(pb - (pa + pc) / 2) < 4.0 and max(np.linalg.norm(pb - pa), np.linalg.norm(pc - pb)) < 80:
            lab[i] = [pb[0] * W / 1920.0, pb[1] * H / 1080.0, 1.0]
    cap = cv2.VideoCapture(str(args.src / name))
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    arr = np.stack(frames)
    lab = lab[: len(arr)]
    idx = np.linspace(0, len(arr) - 1, min(len(arr), 300)).astype(int)
    bg = np.moveaxis(np.median(arr[idx], axis=0), -1, 0).astype(np.uint8)
    stem = Path(name).stem
    np.save(args.out / f"{stem}.npy", arr)
    np.save(args.out / f"{stem}_bg.npy", bg)
    np.save(args.out / f"{stem}_lab.npy", lab)
    tot_frames += len(arr)
    tot_pos += int(lab[:, 2].sum())
    print(f"{stem}: {len(arr)} frames, {int(lab[:, 2].sum())} confident ball labels", flush=True)
print(f"TOTAL {len(chosen)} videos, {tot_frames} frames, {tot_pos} confident labels ({100 * tot_pos / max(1, tot_frames):.0f} %)")
