"""Pose features for every audio hit candidate of the CVSPORTS rallies: is a wrist near the ball at that instant?

    python scripts/pose_hit_features.py   -> runs/pose_hit_features.npz (rows aligned with runs/hit_features.npz)

EXPERIMENT. Pose model: yolo26m-pose.pt from the user's earlier project (Ultralytics, AGPL-3.0: allowed for
experiments only, to be replaced, e.g. by RTMPose (Apache-2.0), before any release). Loaded from its original path.
One frame per candidate (the candidate's frame); the ball position comes from runs/vision_hit_features.json (nearest frame
within +-3 with a ball). Pixel distances are divided by the person's box height, so they do not depend on zoom.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DIR = Path("data/external/cvsports_padel")
POSE = Path(r"C:\Users\dell\OneDrive\Bureau\pfe\padel_analytics_repo\padel_analytics_repo\weights\yolo26m-pose.pt")
L_WRIST, R_WRIST, L_SH, R_SH = 9, 10, 5, 6
NAMES = ["n_people", "wrist_ball_min", "wrist_ball_2nd", "box_ball_min", "wrist_above_shoulder", "near_person_h", "ball_known"]

labels: dict[str, list[float]] = defaultdict(list)
with open(DIR / "metadata/hits.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        labels[r["filename"]].append(float(r["start"]))
names = [f for f in sorted(labels) if (DIR / "rallies" / f"{f}.ok").exists()]
z = np.load("runs/hit_features.npz", allow_pickle=True)
vid, times = z["vid"], z["times"]
vis = json.loads(Path("runs/vision_hit_features.json").read_text())

from ultralytics import YOLO

model = YOLO(str(POSE))
F = np.zeros((len(times), len(NAMES)))
for k, f in enumerate(names):
    idx = np.where(vid == k)[0]
    if len(idx) == 0:
        continue
    d = vis[f]
    fps, ball = d["fps"], d["ball"]
    want = {int(i): round(times[i] * fps) for i in idx}
    frames_needed = set(want.values())
    cap = cv2.VideoCapture(str(DIR / "rallies" / f))
    got: dict[int, np.ndarray] = {}
    fi = 0
    while frames_needed - got.keys():
        ok, img = cap.read()
        if not ok:
            break
        if fi in frames_needed:
            got[fi] = img
        fi += 1
    cap.release()
    order = sorted(got)
    results = {}
    for b0 in range(0, len(order), 16):
        batch = order[b0:b0 + 16]
        for fr, res in zip(batch, model.predict([got[x] for x in batch], verbose=False, conf=0.3, classes=[0]),
                           strict=True):
            kp = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((0, 17, 2))
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints is not None and res.keypoints.conf is not None \
                else np.zeros(kp.shape[:2])
            bx = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            results[fr] = (kp, kc, bx)
    for i, fr in want.items():
        if fr not in results:
            continue
        kp, kc, bx = results[fr]
        bpos = None
        for dfr in (0, -1, 1, -2, 2, -3, 3):
            j = fr + dfr
            if 0 <= j < len(ball) and ball[j][1] is not None:
                bpos = np.array(ball[j][1:3], float)
                break
        row = [len(bx), 9.0, 9.0, 9.0, 0.0, 0.0, float(bpos is not None)]
        if bpos is not None and len(bx):
            dists, boxd, above, hs = [], [], [], []
            for p in range(len(bx)):
                h = max(1.0, bx[p, 3] - bx[p, 1])
                ws = [kp[p, w] for w in (L_WRIST, R_WRIST) if kc[p, w] > 0.3]
                dw = min((np.linalg.norm(w - bpos) / h for w in ws), default=9.0)
                cx = np.clip(bpos[0], bx[p, 0], bx[p, 2]), np.clip(bpos[1], bx[p, 1], bx[p, 3])
                boxd.append(np.hypot(bpos[0] - cx[0], bpos[1] - cx[1]) / h)
                dists.append(dw)
                sh = [kp[p, s][1] for s in (L_SH, R_SH) if kc[p, s] > 0.3]
                above.append(float(any(kp[p, w][1] < min(sh) for w in (L_WRIST, R_WRIST) if kc[p, w] > 0.3))
                             if sh else 0.0)
                hs.append(h)
            o = np.argsort(dists)
            row = [len(bx), min(dists[o[0]], 9.0), min(dists[o[1]], 9.0) if len(o) > 1 else 9.0,
                   min(min(boxd), 9.0), above[o[0]], hs[o[0]] / 1080.0, 1.0]
        F[i] = row
    if k % 10 == 0:
        print(f"{k + 1}/{len(names)}", flush=True)
np.savez("runs/pose_hit_features.npz", F=F, names=np.array(NAMES))
print("done", F.shape)
