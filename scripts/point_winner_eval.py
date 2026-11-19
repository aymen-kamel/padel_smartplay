"""Who won the point? Simple, explainable rules evaluated on the broadcast points whose winner is known from the scoreboard.

    python scripts/point_winner_eval.py

Ground truth: runs/highlights/points.json (scoreboard read by OCR; only transitions explained by exactly ONE point).
Which team plays near / far changes during the match, so each clip's near/far teams are identified by SHIRT COLOUR
(median torso colour of the players detected on each half). The link "scoreboard row <-> shirt colour" is one unknown bit for
the whole match: both choices are reported, and the best one is marked as such (optimistic by that one bit).

Rules (inputs: audio hit candidates, 2D ball track, players on the floor in metres):
  last_hitter   the team of the player nearest to the ball at the last hit of the rally wins (winners are hit, errors are not
                always visible)
  ball_dies     the team on whose half the ball was last seen LOSES
  always_bottom baseline: always the scoreboard's bottom row (the majority class here)
"""

import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.geometry.camera import CourtCamera
from smartplay.players.detector import PersonDetector

CLIP_OF = {134.0: "point_06", 220.5: "point_09", 240.5: "point_10", 257.0: "point_11", 281.0: "point_12",
           415.5: "point_18", 429.0: "point_19", 469.0: "point_21", 559.0: "point_24"}
floor = CourtCamera.from_points([tuple(p) for p in json.loads(Path("data/calibration/highlights_main_view.json").read_text())])
det = PersonDetector(0.5)


def clip_video(k):
    return next(Path("runs/highlights/clips").glob(f"{k}_*.mp4"))


def shirt_colours(video):
    """Median torso colour (Lab) of players on the near half and on the far half, over 5 frames of the clip."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    near, far = [], []
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
                (near if y < 0 else far).append(np.median(torso.reshape(-1, 3), axis=0))
    cap.release()
    return np.median(near, axis=0), np.median(far, axis=0)


rows = []
for p in json.loads(Path("runs/highlights/points.json").read_text()):
    k = CLIP_OF[p["t_before"]]
    run = Path(f"runs/{k}_fused")
    hits = [h["frame"] for h in json.loads((run / "audio_hits.json").read_text())]
    ball = json.loads((run / "ball.json").read_text())
    tracks: dict[int, list] = {}
    for r in json.loads((run / "tracks.json").read_text()):
        tracks.setdefault(r["frame"], []).append(r)
    seen = [b["frame"] for b in ball if b["uv"] is not None and b["visibility"] == "visible"]
    last_seen = seen[-1] if seen else None
    # rally = up to the last ball detection (+10 frames): later sounds are celebration, applause, the next ball
    rally_hits = [h for h in hits if last_seen is not None and h <= last_seen + 10]
    hitter = None
    if rally_hits:
        h = rally_hits[-1]
        cand = [ball[f] for f in range(max(0, h - 3), min(len(ball), h + 4)) if ball[f]["uv"] is not None]
        if cand and h in tracks:
            u, v = cand[len(cand) // 2]["uv"]
            # nearest player in the image (feet projected; the ball is at hand height, so compare horizontally first)
            best = min(tracks[h], key=lambda r: abs(floor.to_image(*r["foot_xy_m"])[0] - u)
                       + 0.5 * abs(floor.to_image(*r["foot_xy_m"])[1] - v))
            hitter = "near" if best["foot_xy_m"][1] < 0 else "far"
    dies = None
    if last_seen is not None:
        u, v = ball[last_seen]["uv"]
        net_v = floor.to_image(0.0, 0.0)[1]
        dies = "near" if v > net_v else "far"  # below the net line in the image = near half (approximation)
    near_col, far_col = shirt_colours(clip_video(k))
    rows.append({"clip": k, "winner_row": p["winner_row"], "hitter": hitter, "dies": dies,
                 "near_col": near_col.tolist(), "far_col": far_col.tolist(), "n_hits": len(rally_hits)})
    print(f"{k}: winner row {p['winner_row']:6s} last hitter {hitter}  ball last seen on {dies} half  "
          f"near shirt Lab {np.round(near_col).astype(int)}  far shirt Lab {np.round(far_col).astype(int)}", flush=True)

# shirt colours -> two teams: the near team of each clip is team X if its colour is closer to the first clip's near colour
ref = np.array(rows[0]["near_col"])
for r in rows:
    near_is_x = np.linalg.norm(np.array(r["near_col"]) - ref) < np.linalg.norm(np.array(r["far_col"]) - ref)
    r["near_team"] = "X" if near_is_x else "Y"
print("near team by clip:", [(r["clip"], r["near_team"]) for r in rows])


def team(side, r):
    return r["near_team"] if side == "near" else ("Y" if r["near_team"] == "X" else "X")


def score(rule, x_is):
    ok = n = 0
    for r in rows:
        if rule == "always_bottom":
            pred_row = "bottom"
        else:
            side = r["hitter"] if rule == "last_hitter" else ({"near": "far", "far": "near"}.get(r["dies"]))
            if side is None:
                continue
            pred_row = x_is if team(side, r) == "X" else ("top" if x_is == "bottom" else "bottom")
        n += 1
        ok += pred_row == r["winner_row"]
    return ok, n


for rule in ("last_hitter", "ball_dies", "always_bottom"):
    res = {x: score(rule, x) for x in ("top", "bottom")}
    best = max(res, key=lambda x: res[x][0])
    print(f"{rule:14s} team X = top: {res['top'][0]}/{res['top'][1]}   team X = bottom: {res['bottom'][0]}/{res['bottom'][1]}"
          f"   (best mapping: X = {best})")
Path("runs/point_winner_eval.json").write_text(json.dumps(rows, indent=1))
