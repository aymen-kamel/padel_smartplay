"""Run the fused ball detector (WASB + TrackNetV3, gated tracker) on several videos and report how often the ball is
seen; optionally write an overlay video (ball only: no court calibration needed).

    python scripts/ball_on_videos.py --videos a.mp4 b.mp4 [--overlay-dir runs/ball_tests] [--predict 0]

Without labels this measures how often the ball is found (coverage), NOT accuracy.
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2

from smartplay.ball.detector import WasbBallDetector, track_gated
from smartplay.ball.fusion import merge_blobs
from smartplay.ball.tracknet import TrackNetBallDetector
from smartplay.ingestion.video import iter_frames, probe
from smartplay.viz.annotate import draw_ball

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", type=Path, required=True)
ap.add_argument("--overlay-dir", type=Path, default=None)
ap.add_argument("--max-frames", type=int, default=500)
ap.add_argument("--wasb", type=Path, default=Path("weights/wasb/wasb_tennis_best.pth.tar"))
ap.add_argument("--tracknet", type=Path, default=Path("weights/tracknetv3_official/ckpts/TrackNet_best.pt"))
ap.add_argument("--out", type=Path, default=Path("runs/ball_on_videos.json"))
args = ap.parse_args()

wasb = WasbBallDetector(args.wasb, threshold=0.3)
tn = TrackNetBallDetector(args.tracknet, threshold=0.3)
rows = {}
for v in args.videos:
    info = probe(v)
    n = min(info.n_frames, args.max_frames)
    a = wasb.blobs(itertools.islice(iter_frames(v), n), info.width, info.height)
    b = tn.blobs(itertools.islice(iter_frames(v), n), info.width, info.height)
    ball_w = track_gated(a)
    ball_m = track_gated(merge_blobs(a, b))
    seen_w = sum(s.visibility == "visible" for s in ball_w) / max(1, len(ball_w))
    seen_m = sum(s.visibility == "visible" for s in ball_m) / max(1, len(ball_m))
    rows[v.name] = {"frames": n, "seen_wasb": round(seen_w, 3), "seen_fused": round(seen_m, 3)}
    print(f"{v.name:28s} {n:4d} frames  seen: WASB {100 * seen_w:4.1f} %  fused {100 * seen_m:4.1f} %")
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(args.overlay_dir / f"{v.stem}_ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            info.fps, (info.width, info.height))
        for i, fr in itertools.islice(iter_frames(v), n):
            draw_ball(fr, ball_m, i)
            w.write(fr)
        w.release()
mean_w = sum(r["seen_wasb"] for r in rows.values()) / len(rows)
mean_m = sum(r["seen_fused"] for r in rows.values()) / len(rows)
print(f"mean seen: WASB {100 * mean_w:.1f} %, fused {100 * mean_m:.1f} % over {len(rows)} videos")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(rows, indent=1))
