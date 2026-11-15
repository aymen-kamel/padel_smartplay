"""Pass 2: analyse the cached features of a broadcast video (cuts, main view, scoreboard, points).

    python scripts/analyze_highlights.py --run runs/highlights
"""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from smartplay.broadcast.scoreboard import classify_transition, consistent_timeline
from smartplay.broadcast.segment import board_timeline, detect_cuts, segments
from smartplay.scoring.automaton import DeuceMode, MatchRules

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--cut-threshold", type=float, default=0.35)
args = ap.parse_args()

f = np.load(args.run / "features.npz")
cut, blue, masks, fps = f["cut"], f["blue"], f["masks"], float(f["fps"])
samples = json.loads((args.run / "ocr_samples.json").read_text())
print(f"frames {len(cut)}  fps {fps:g}  duration {len(cut) / fps:.1f}s  ocr samples {len(samples)}")

print("cut score percentiles 90/99/99.9:", np.percentile(cut, [90, 99, 99.9]).round(3))
segs = segments(cut, blue, masks, args.cut_threshold)
main = [s for s in segs if s.kind == "main"]
print(f"shots {len(segs)}  main-view shots {len(main)}  "
      f"main-view time {sum(s.end - s.start + 1 for s in main) / fps:.0f}s of {len(cut) / fps:.0f}s")
lens = np.array([(s.end - s.start + 1) / fps for s in main])
if len(lens):
    print(f"main-view shot length s: median {np.median(lens):.1f}  p10 {np.percentile(lens, 10):.1f}  "
          f"p90 {np.percentile(lens, 90):.1f}  max {lens.max():.1f}")

tl_raw = board_timeline(samples)
tl, dropped = consistent_timeline(tl_raw, MatchRules())
print(f"scoreboard readings: {len(tl_raw)} stable, {len(tl)} kept, {len(dropped)} dropped as unreachable (OCR error or replay)")
read_ok = sum(1 for s in samples if s["tokens"])
print(f"scoreboard: stable readings {len(tl)} distinct states (samples with any token: {read_ok}/{len(samples)})")

results = {}
for mode in (DeuceMode.ADVANTAGE, DeuceMode.GOLDEN_POINT, DeuceMode.STAR_POINT):
    rules = MatchRules(deuce_mode=mode)
    c = Counter(classify_transition(a[1], b[1], rules) for a, b in itertools.pairwise(tl))
    results[mode.value] = c
    print(f"  transitions under {mode.value:13s}: {dict(c)}")

rules = MatchRules()
points = []
for (fa, ba), (fb, bb) in itertools.pairwise(tl):
    who = classify_transition(ba, bb, rules)
    if who in ("top", "bottom"):
        points.append({"frame_before": fa, "frame_after": fb, "t_before": fa / fps, "t_after": fb / fps,
                       "winner_row": who, "from": ba.text(), "to": bb.text()})
(args.run / "points.json").write_text(json.dumps(points, indent=1))
(args.run / "segments.json").write_text(json.dumps(
    [{"start": s.start, "end": s.end, "t_start": s.start / fps, "t_end": s.end / fps, "kind": s.kind,
      "blue": s.blue, "iou": s.iou} for s in segs], indent=1))
(args.run / "scoreboard_timeline.json").write_text(json.dumps(
    [{"frame": fr, "t": fr / fps, "top": b.top, "bottom": b.bottom} for fr, b in tl], indent=1))
print(f"single-point transitions with a known winner: {len(points)}")
for p in points[:12]:
    print(f"  {p['t_before']:6.1f}s -> {p['t_after']:6.1f}s  {p['winner_row']:6s}  {p['from']}  =>  {p['to']}")

# ---- is every score jump explainable by playing more points under the FIP rules? ----
import itertools

from smartplay.broadcast.scoreboard import reachable

print("\nreachability of each consecutive scoreboard change (fewest points the engine needs):")
for mode in (DeuceMode.ADVANTAGE, DeuceMode.GOLDEN_POINT):
    rules = MatchRules(deuce_mode=mode)
    steps = [reachable(a[1], b[1], rules) for a, b in itertools.pairwise(tl)]
    ok = [s for s in steps if s is not None]
    print(f"  {mode.value:13s}: reachable {len(ok)}/{len(steps)}; points skipped between readings: "
          f"median {np.median(ok):.0f}, max {max(ok)}; unreachable {len(steps) - len(ok)}")
rules = MatchRules()
bad = [(a, b) for (a, b) in itertools.pairwise(tl) if reachable(a[1], b[1], rules) is None]
for (fa, ba), (fb, bb) in bad[:6]:
    print(f"  UNREACHABLE {fa / fps:6.1f}s {ba.text()}  ->  {fb / fps:6.1f}s {bb.text()}")

# ---- overview picture ----
import itertools

import cv2

W, H = 1800, 230
img = np.full((H, W, 3), 24, np.uint8)
x_of = lambda fr: int(20 + (W - 40) * fr / len(cut))
for s in segs:
    cv2.rectangle(img, (x_of(s.start), 50), (x_of(s.end + 1), 100),
                  (200, 120, 40) if s.kind == "main" else (90, 90, 90), -1)
for c in detect_cuts(cut, args.cut_threshold):
    cv2.line(img, (x_of(c), 50), (x_of(c), 100), (20, 20, 20), 1)
for (fa, ba), (fb, bb) in itertools.pairwise(tl):
    who = classify_transition(ba, bb, rules)
    col = {"top": (40, 140, 255), "bottom": (230, 200, 40)}.get(who, (160, 160, 160))
    cv2.line(img, (x_of(fb), 110), (x_of(fb), 180), col, 3)
for t in range(0, int(len(cut) / fps) + 1, 60):
    cv2.line(img, (x_of(t * fps), 195), (x_of(t * fps), 205), (255, 255, 255), 1)
    cv2.putText(img, f"{t // 60}m", (x_of(t * fps) - 8, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.putText(img, "shots: blue = main wide view, grey = close-up / crowd / replay", (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
cv2.putText(img, "score changes: orange = top team won the point, cyan = bottom team, grey = several points skipped",
            (20, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(str(args.run / "overview.png"), img)
print("wrote", args.run / "overview.png")
