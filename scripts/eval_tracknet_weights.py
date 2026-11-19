"""Compare TrackNet checkpoints on the hand-labelled clips, alone and merged with WASB (same tracker, same metric).

    python scripts/eval_tracknet_weights.py --weights official=<path> finetuned=<path>

Metric: TrackNet convention (as in eval_benchmark.py / compare_ball_trackers.py) at 5 and 10 px, after the gated tracker.
WASB candidates are read from runs/<clip>/ball_blobs_wasb03.json (threshold 0.3). The hand labels were never used in training.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from smartplay.ball.detector import Blob, track_gated
from smartplay.ball.fusion import merge_blobs
from smartplay.ball.tracknet import TrackNetBallDetector
from smartplay.ingestion.video import iter_frames, probe
from smartplay.labeling.store import ClipLabels

CLIPS = {"point_12": "runs/highlights/clips/point_12_t0280s_4-0_vs_5-40.mp4",
         "point_24": "runs/highlights/clips/point_24_t0558s_4-6-1-0_vs_6-3-0-15.mp4"}
ap = argparse.ArgumentParser()
ap.add_argument("--weights", nargs="+", required=True, help="name=path pairs")
args = ap.parse_args()


def counts(gt, pred, t):
    c = {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0}
    for f, g in gt.items():
        p = pred[f].uv if f < len(pred) else None
        if g is None:
            c["fp2"] += p is not None
        elif p is None:
            c["fn"] += 1
        elif np.hypot(p[0] - g[0], p[1] - g[1]) <= t:
            c["tp"] += 1
        else:
            c["fp1"] += 1
    return c


def f1(c):
    p = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fp2"])
    r = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fn"])
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


rows = {}
for spec in args.weights:
    name, path = spec.split("=", 1)
    det = TrackNetBallDetector(Path(path), threshold=0.3)
    tot = {mode: {t: {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0} for t in (5, 10)} for mode in ("alone", "merged")}
    for clip, video in CLIPS.items():
        info = probe(video)
        tn = det.blobs(iter_frames(video), info.width, info.height)
        wasb = [[Blob(*b) for b in fr] for fr in json.loads(Path(f"runs/{clip}/ball_blobs_wasb03.json").read_text())]
        lab = ClipLabels.load(f"data/labels/{clip}.json").ball
        for mode, bl in (("alone", tn), ("merged", merge_blobs(wasb, tn))):
            pred = track_gated(bl)
            for t in (5, 10):
                c = counts(lab, pred, t)
                for k in c:
                    tot[mode][t][k] += c[k]
    rows[name] = tot
    for mode in ("alone", "merged"):
        a5, a10 = f1(tot[mode][5]), f1(tot[mode][10])
        print(f"{name:10s} {mode:7s}  @5px P {a5[0]:.2f} R {a5[1]:.2f} F1 {a5[2]:.3f}   @10px P {a10[0]:.2f} R {a10[1]:.2f} "
              f"F1 {a10[2]:.3f}   (ball where label says none: {tot[mode][10]['fp2']})", flush=True)
Path("runs/eval_tracknet_weights.json").write_text(json.dumps(rows, indent=1))
