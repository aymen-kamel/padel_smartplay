"""Compare pipeline output with hand labels on the benchmark clips.

    python scripts/eval_benchmark.py --clips point_12 point_24 [--out runs/benchmark_baseline.json]

Reads data/labels/<clip>.json (label tool) and runs/<clip>/{ball,audio_hits}.json (cli annotate).
Ball metric (TrackNet convention), per pixel threshold T:
    TP   predicted and within T px of a visible label
    FP1  predicted but farther than T from a visible label
    FP2  predicted while the label says the ball is not visible
    FN   label visible, nothing predicted
    precision = TP / (TP + FP1 + FP2)      recall = TP / (TP + FP1 + FN)
Hit metric: one-to-one matching of predicted and labelled frames within +-k frames.
"""

import argparse
from itertools import pairwise
import json
from pathlib import Path

import numpy as np

from smartplay.evaluation.metrics import match_events
from smartplay.labeling.store import ClipLabels

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--labels", type=Path, default=Path("data/labels"))
ap.add_argument("--runs", type=Path, default=Path("runs"))
ap.add_argument("--out", type=Path, default=Path("runs/benchmark_baseline.json"))
args = ap.parse_args()

THRESH = (5, 10, 20, 50)
TOLS = (1, 3, 5, 10)


def ball_counts(gt: dict, pred: dict[int, tuple[float, float] | None], t: float) -> dict:
    c = {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0, "tn": 0}
    for f, g in gt.items():
        p = pred.get(f)
        if g is None:
            c["fp2" if p is not None else "tn"] += 1
        elif p is None:
            c["fn"] += 1
        elif np.hypot(p[0] - g[0], p[1] - g[1]) <= t:
            c["tp"] += 1
        else:
            c["fp1"] += 1
    return c


def prf(c: dict) -> tuple[float, float, float]:
    p = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fp2"])
    r = c["tp"] / max(1, c["tp"] + c["fp1"] + c["fn"])
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


results: dict = {}
all_ball = {t: {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0, "tn": 0} for t in THRESH}
all_hits = {k: [0, 0, 0] for k in TOLS}
for clip in args.clips:
    lab = ClipLabels.load(args.labels / f"{clip}.json")
    ball = {b["frame"]: (tuple(b["uv"]) if b["uv"] else None)
            for b in json.loads((args.runs / clip / "ball.json").read_text())}
    hits = [h["frame"] for h in json.loads((args.runs / clip / "audio_hits.json").read_text())]
    res: dict = {"labelled_ball_frames": len(lab.ball), "labelled_hits": len(lab.hits), "predicted_hits": len(hits)}
    vis_gt = sum(1 for v in lab.ball.values() if v is not None)
    res["ball_visible_in_labels"] = vis_gt
    res["ball_predicted_on_labelled_frames"] = sum(1 for f in lab.ball if ball.get(f) is not None)
    errs = [float(np.hypot(ball[f][0] - g[0], ball[f][1] - g[1])) for f, g in lab.ball.items()
            if g is not None and ball.get(f) is not None]
    res["ball_error_px"] = {"n": len(errs), "median": float(np.median(errs)) if errs else None,
                            "p90": float(np.percentile(errs, 90)) if errs else None}
    res["ball"] = {}
    for t in THRESH:
        c = ball_counts(lab.ball, ball, t)
        for k in c:
            all_ball[t][k] += c[k]
        p, r, f1 = prf(c)
        res["ball"][str(t)] = {**c, "precision": p, "recall": r, "f1": f1}
    res["far_errors"] = sorted(
        (f, round(float(np.hypot(ball[f][0] - g[0], ball[f][1] - g[1])))) for f, g in lab.ball.items()
        if g is not None and ball.get(f) is not None and np.hypot(ball[f][0] - g[0], ball[f][1] - g[1]) > 50)
    res["hits"] = {}
    for k in TOLS:
        s = match_events(hits, sorted(lab.hits), k)
        all_hits[k][0] += s.true_positives
        all_hits[k][1] += s.false_positives
        all_hits[k][2] += s.false_negatives
        res["hits"][str(k)] = {"tp": s.true_positives, "fp": s.false_positives, "fn": s.false_negatives,
                               "precision": s.precision, "recall": s.recall, "f1": s.f1}
    offs = []  # signed offset predicted - labelled, nearest label within 10 frames
    for h in hits:
        near = min(lab.hits, key=lambda g: abs(g - h), default=None)
        if near is not None and abs(near - h) <= 10:
            offs.append(h - near)
    res["hit_offset_frames"] = {"n": len(offs), "median": float(np.median(offs)) if offs else None}
    close = sorted(lab.hits)
    res["labelled_hits_closer_than_5_frames"] = [(a, b) for a, b in pairwise(close) if b - a < 5]
    results[clip] = res

print(f"{'':10s} " + "  ".join(f"ball@{t}px P/R/F1" for t in THRESH))
for clip, r in results.items():
    print(f"{clip:10s} " + "  ".join(
        f"{r['ball'][str(t)]['precision']:.2f}/{r['ball'][str(t)]['recall']:.2f}/{r['ball'][str(t)]['f1']:.2f}" for t in THRESH))
tot = {t: prf(all_ball[t]) for t in THRESH}
print(f"{'ALL':10s} " + "  ".join(f"{tot[t][0]:.2f}/{tot[t][1]:.2f}/{tot[t][2]:.2f}" for t in THRESH))
print()
for clip, r in results.items():
    e = r["ball_error_px"]
    print(f"{clip}: ball visible in labels {r['ball_visible_in_labels']}/{r['labelled_ball_frames']}, predicted on "
          f"{r['ball_predicted_on_labelled_frames']} labelled frames; error px median {e['median']} p90 {e['p90']} (n={e['n']}); "
          f"predictions >50 px off: {len(r['far_errors'])}")
print()
print(f"{'':10s} " + "  ".join(f"hits+-{k}f P/R/F1" for k in TOLS))
for clip, r in results.items():
    print(f"{clip:10s} " + "  ".join(
        f"{r['hits'][str(k)]['precision']:.2f}/{r['hits'][str(k)]['recall']:.2f}/{r['hits'][str(k)]['f1']:.2f}" for k in TOLS))
agg = {}
for k in TOLS:
    tp, fp, fn = all_hits[k]
    p, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    agg[k] = (p, r_, 2 * p * r_ / (p + r_) if p + r_ else 0.0)
print(f"{'ALL':10s} " + "  ".join(f"{agg[k][0]:.2f}/{agg[k][1]:.2f}/{agg[k][2]:.2f}" for k in TOLS))
for clip, r in results.items():
    print(f"{clip}: predicted hits {r['predicted_hits']} vs labelled {r['labelled_hits']}; median offset "
          f"{r['hit_offset_frames']['median']} frames; labelled hits <5 frames apart: {r['labelled_hits_closer_than_5_frames']}")
args.out.write_text(json.dumps({"clips": results,
                                "ball_all": {str(t): {**all_ball[t], "p_r_f1": tot[t]} for t in THRESH},
                                "hits_all": {str(k): agg[k] for k in TOLS}}, indent=1))
print("\nwrote", args.out)
