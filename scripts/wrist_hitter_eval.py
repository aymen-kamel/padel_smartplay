"""Identify the LAST HITTER with the wrist, not with the ball's image position, then re-test the winner rule.

    python scripts/wrist_hitter_eval.py

For each hand-labelled point (data/labels/point_winners.json, red/dark): take the last audio hit of the rally, read that frame,
run RTMPose-x (CPU), and take the wrist (left or right) closest to the ball's pixel at that moment. The skeleton owning that
wrist is matched to a tracked player by its ankles (floor metres). The hitter's side (near/far) gives the team through the
near colour read from the shirts. Rule: last hitter's team wins.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.geometry.camera import CourtCamera
from smartplay.paths import rtmpose_body

floor = CourtCamera.from_points([tuple(p) for p in json.loads(Path("data/calibration/highlights_main_view.json").read_text())])
truth = {k: v for k, v in json.loads(Path("data/labels/point_winners.json").read_text()).items() if v in ("red", "dark")}
colour = {}
for line in Path("runs/point_winner_eval_all.json").read_text().split("\n"):
    pass
prev = json.loads(Path("runs/point_winner_eval_all.json").read_text())
near_col = {r["clip"]: r["near_colour"] for r in prev}
body = rtmpose_body("performance", device="cpu")
WRISTS, ANKLES = (9, 10), (15, 16)
rows = []
for k, winner in sorted(truth.items()):
    if k not in near_col:
        continue
    run = Path(f"runs/{k}_fused")
    hits = [h["frame"] for h in json.loads((run / "audio_hits.json").read_text())]
    ball = json.loads((run / "ball.json").read_text())
    tracks: dict[int, list] = {}
    for r in json.loads((run / "tracks.json").read_text()):
        tracks.setdefault(r["frame"], []).append(r)
    seen = [b["frame"] for b in ball if b["uv"] is not None and b["visibility"] == "visible"]
    last_seen = seen[-1] if seen else None
    rally_hits = [h for h in hits if last_seen is not None and h <= last_seen + 10 and h in tracks]
    if not rally_hits:
        rows.append({"clip": k, "truth": winner, "hitter": None, "why": "no hit"})
        continue
    h = rally_hits[-1]
    cand = [ball[f] for f in range(max(0, h - 2), min(len(ball), h + 3)) if ball[f]["uv"] is not None]
    if not cand:
        rows.append({"clip": k, "truth": winner, "hitter": None, "why": "no ball at hit"})
        continue
    bu, bv = cand[len(cand) // 2]["uv"]
    video = next(Path("runs/highlights/clips").glob(f"{k}_*.mp4"))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, h)
    ok, img = cap.read()
    cap.release()
    if not ok:
        rows.append({"clip": k, "truth": winner, "hitter": None, "why": "frame unreadable"})
        continue
    kp, sc = body(img)
    kp, sc = np.asarray(kp, float).reshape(-1, 17, 2), np.asarray(sc, float).reshape(-1, 17)
    best = None  # (distance px, skeleton index)
    for s in range(len(kp)):
        for w in WRISTS:
            if sc[s, w] < 0.3:
                continue
            d = float(np.hypot(kp[s, w, 0] - bu, kp[s, w, 1] - bv))
            if best is None or d < best[0]:
                best = (d, s)
    if best is None:
        rows.append({"clip": k, "truth": winner, "hitter": None, "why": "no wrist seen"})
        continue
    s = best[1]
    ank = [kp[s, a] for a in ANKLES if sc[s, a] > 0.3]
    if not ank:
        rows.append({"clip": k, "truth": winner, "hitter": None, "why": "no ankle seen"})
        continue
    u, v = np.mean(ank, axis=0)
    wx, wy = floor.to_world(u, v)
    player = min(tracks[h], key=lambda r: (r["foot_xy_m"][0] - wx) ** 2 + (r["foot_xy_m"][1] - wy) ** 2)
    side = "near" if player["foot_xy_m"][1] < 0 else "far"
    nc = near_col[k]
    team = nc if side == "near" else ("dark" if nc == "red" else "red")
    rows.append({"clip": k, "truth": winner, "hitter": side, "team": team, "wrist_px": round(best[0]),
                 "ok": team == winner})
    print(f"{k}: truth {winner:4s} | hitter {side:4s} (wrist {round(best[0])} px from ball) -> team {team:4s} "
          f"{'OK' if team == winner else 'x'}", flush=True)

have = [r for r in rows if r.get("team")]
ok = sum(r["ok"] for r in have)
n_all = len(rows)
major = max(("red", "dark"), key=lambda c: sum(r["truth"] == c for r in rows))
print(f"\nwrist hitter rule: {ok}/{len(have)} correct ({100 * ok / max(1, len(have)):.0f} %), "
      f"{n_all - len(have)} points without a usable wrist")
print(f"always {major}: {sum(r['truth'] == major for r in rows)}/{n_all} ({100 * sum(r['truth'] == major for r in rows) / max(1, n_all):.0f} %)")
Path("runs/wrist_hitter_eval.json").write_text(json.dumps(rows, indent=1))
