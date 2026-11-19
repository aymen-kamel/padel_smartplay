"""Run TrackNetV3 (official checkpoint, own re-implementation in smartplay.ball.tracknet) and save candidate blobs per
frame, in the same format as dump_ball_blobs.py, so both detectors can be compared with the same tracker.

    python scripts/dump_tracknet_blobs.py --video <clip.mp4> --out runs/<clip>/ball_blobs_tracknet.json
"""

import argparse
import json
from pathlib import Path

from smartplay.ball.tracknet import TrackNetBallDetector
from smartplay.ingestion.video import iter_frames, probe
from smartplay.paths import TRACKNETV3

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--weights", type=Path, default=TRACKNETV3)
ap.add_argument("--threshold", type=float, default=0.3)
args = ap.parse_args()

info = probe(args.video)
blobs = TrackNetBallDetector(args.weights, threshold=args.threshold).blobs(iter_frames(args.video), info.width,
                                                                           info.height)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps([[[b.u, b.v, b.score] for b in fr] for fr in blobs]))
print(f"{args.out}: {len(blobs)} frames, {sum(bool(f) for f in blobs)} with a candidate, "
      f"{sum(len(f) > 1 for f in blobs)} with several")
