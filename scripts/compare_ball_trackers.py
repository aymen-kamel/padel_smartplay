"""Compare ball trackers offline on cached candidates against hand labels.

    python scripts/compare_ball_trackers.py --clips point_12 point_24
"""

import argparse
import json
from pathlib import Path

import numpy as np

from smartplay.ball.detector import Blob, OnlineBallTracker, track_gated
from smartplay.labeling.store import ClipLabels

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--blobs", default="ball_blobs", help="file stem under runs/<clip>/ (ball_blobs or ball_blobs_tracknet)")
ap.add_argument("--min-score", type=float, default=0.0, help="drop candidates below this score")
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


def prf(c):
    p = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fp2"])
    r = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fn"])
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def baseline(bl):
    tr = OnlineBallTracker()
    return [tr.step(i, b) for i, b in enumerate(bl)]


def oracle(bl, gt, t=10):
    """Upper bound for any tracker: is a candidate within t px of the label at all?"""
    hit = sum(1 for f, g in gt.items() if g is not None and f < len(bl)
              and any(np.hypot(b.u - g[0], b.v - g[1]) <= t for b in bl[f]))
    return hit / max(1, sum(1 for g in gt.values() if g is not None))


trackers = {"baseline (strongest near last)": baseline, "gated (trajectory)": track_gated}
total = {name: {t: {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0} for t in (5, 10)} for name in trackers}
for clip in args.clips:
    bl = [[Blob(*b) for b in fr if b[2] >= args.min_score] for fr in json.loads(Path(f"runs/{clip}/{args.blobs}.json").read_text())]
    lab = ClipLabels.load(f"data/labels/{clip}.json")
    print(f"{clip}: candidate recall upper bound @10px = {oracle(bl, lab.ball):.2f}")
    for name, fn in trackers.items():
        pred = fn(bl)
        row = []
        for t in (5, 10):
            c = counts(lab.ball, pred, t)
            for k in c:
                total[name][t][k] += c[k]
            row.append("P %.2f R %.2f F1 %.2f (wrong place %d)" % (*prf(c), c["fp1"]))
        print(f"  {name:32s} @5px {row[0]} | @10px {row[1]}")
print("ALL")
for name in trackers:
    print(f"  {name:32s} " + " | ".join("@%dpx P %.2f R %.2f F1 %.2f (wrong place %d)" % (t, *prf(total[name][t]), total[name][t]["fp1"]) for t in (5, 10)))
