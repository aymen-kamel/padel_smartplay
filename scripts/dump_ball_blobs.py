"""Run the ball model once and save every candidate blob per frame (so trackers can be compared offline).

    python scripts/dump_ball_blobs.py --video <clip.mp4> --out runs/<clip>/ball_blobs.json
"""

import argparse
import json
from pathlib import Path

from smartplay.ball.detector import WasbBallDetector
from smartplay.ingestion.video import iter_frames, probe

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--weights", type=Path, default=Path("weights/wasb/wasb_tennis_best.pth.tar"))
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()

info = probe(args.video)
det = WasbBallDetector(args.weights, threshold=args.threshold)
blobs = det.blobs(iter_frames(args.video), info.width, info.height)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps([[[b.u, b.v, b.score] for b in fr] for fr in blobs]))
n_multi = sum(len(fr) > 1 for fr in blobs)
print(f"{args.out}: {len(blobs)} frames, {sum(bool(fr) for fr in blobs)} with a candidate, {n_multi} with several")
