"""Point-winner rules on every broadcast clip whose winner was marked by hand (data/labels/point_winners.json, by shirt colour).

    python scripts/point_winner_eval_all.py

Truth: the owner's red / dark answers ("unclear" skipped). They agree with the scoreboard on all 8 points where both exist
(red = top row TAPIA/COELLO, dark = bottom row AUGSBURGER/LEBRON). The near team's colour is read from the shirts (Lab a-channel:
red > 160), so no mapping has to be guessed. Inputs per clip: runs/<clip>_fused/{audio_hits,ball,tracks}.json from cli annotate.
Rules: last_hitter (team of the last hitter wins), ball_dies (team on whose half the ball was last seen loses), and the
majority baseline (always the team that won most points here).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.geometry.camera import CourtCamera
from smartplay.players.detector import PersonDetector

floor = CourtCamera.from_points([tuple(p) for p in json.loads(Path("data/calibration/highlights_main_view.json").read_text())])
det = PersonDetector(0.5)
truth = {k: v for k, v in json.loads(Path("data/labels/point_winners.json").read_text()).items() if v in ("red", "dark")}


def near_colour(video):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    near_a, far_a = [], []
    for f in np.linspace(n * 0.2, n * 0.6, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for d in det.detect(img):
            x1, y1, x2, y2 = (int(v) for v in d.box_uv)
            x, y = floor.to_world(*d.foot_uv)
            if abs(x) > 5.5 or abs(y) > 10.5:
                continue
            h, w = y2 - y1, x2 - x1
            torso = lab[y1 + int(0.22 * h): y1 + int(0.45 * h), x1 + int(0.3 * w): x1 + int(0.7 * w)]
            if torso.size:
                (near_a if y < 0 else far_a).append(float(np.median(torso[:, :, 1])))
    cap.release()
    na, fa = (np.median(near_a) if near_a else np.nan), (np.median(far_a) if far_a else np.nan)
    if np.isnan(na) or np.isnan(fa):
        return None, na, fa
    return ("red" if na > fa else "dark"), na, fa


def other(c):
    return "dark" if c == "red" else "red"


rows = []
for k, winner in sorted(truth.items()):
    run = Path(f"runs/{k}_fused")
    if not (run / "ball.json").exists():
        print(f"{k}: not analysed, skipped")
        continue
    video = next(Path("runs/highlights/clips").glob(f"{k}_*.mp4"))
    hits = [h["frame"] for h in json.loads((run / "audio_hits.json").read_text())]
    ball = json.loads((run / "ball.json").read_text())
    tracks: dict[int, list] = {}
    for r in json.loads((run / "tracks.json").read_text()):
        tracks.setdefault(r["frame"], []).append(r)
    seen = [b["frame"] for b in ball if b["uv"] is not None and b["visibility"] == "visible"]
    last_seen = seen[-1] if seen else None
    rally_hits = [h for h in hits if last_seen is not None and h <= last_seen + 10]
    hitter = None
    if rally_hits and rally_hits[-1] in tracks:
        h = rally_hits[-1]
        cand = [ball[f] for f in range(max(0, h - 3), min(len(ball), h + 4)) if ball[f]["uv"] is not None]
        if cand:
            u, v = cand[len(cand) // 2]["uv"]
            best = min(tracks[h], key=lambda r: abs(floor.to_image(*r["foot_xy_m"])[0] - u)
                       + 0.5 * abs(floor.to_image(*r["foot_xy_m"])[1] - v))
            hitter = "near" if best["foot_xy_m"][1] < 0 else "far"
    dies = None
    if last_seen is not None:
        dies = "near" if ball[last_seen]["uv"][1] > floor.to_image(0.0, 0.0)[1] else "far"
    nc, na, fa = near_colour(video)
    if nc is None:
        print(f"{k}: shirt colours not found, skipped")
        continue
    side_colour = {"near": nc, "far": other(nc)}
    pred_hitter = side_colour[hitter] if hitter else None
    pred_dies = side_colour[{"near": "far", "far": "near"}[dies]] if dies else None
    rows.append({"clip": k, "truth": winner, "near_colour": nc, "hitter_side": hitter, "dies_side": dies,
                 "last_hitter": pred_hitter, "ball_dies": pred_dies})
    print(f"{k}: winner {winner:4s} | near team {nc:4s} (a {na:.0f} vs {fa:.0f}) | last hitter {hitter} -> {pred_hitter} "
          f"| ball last seen {dies} -> {pred_dies}", flush=True)

n = len(rows)
major = max(("red", "dark"), key=lambda c: sum(r["truth"] == c for r in rows))
print(f"\n{n} points evaluated (truth: red {sum(r['truth'] == 'red' for r in rows)}, dark {sum(r['truth'] == 'dark' for r in rows)})")
for rule in ("last_hitter", "ball_dies"):
    have = [r for r in rows if r[rule] is not None]
    ok = sum(r[rule] == r["truth"] for r in have)
    print(f"  {rule:12s}: {ok}/{len(have)} correct ({100 * ok / max(1, len(have)):.0f} %)")
ok = sum(r["truth"] == major for r in rows)
print(f"  always {major:5s}: {ok}/{n} correct ({100 * ok / max(1, n):.0f} %)  <- baseline to beat")
print(f"  last hitter on the FAR side in {sum(r['hitter_side'] == 'far' for r in rows)}/{n} clips; "
      f"ball last seen on the FAR half in {sum(r['dies_side'] == 'far' for r in rows)}/{n}")
Path("runs/point_winner_eval_all.json").write_text(json.dumps(rows, indent=1))
