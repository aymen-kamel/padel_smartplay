"""Pose models on the same frames: how many on-court players get a skeleton (near / far), how complete, how fast.

    python scripts/pose_bakeoff.py --model yolo|rtm-balanced|rtm-performance --out runs/pose_bakeoff_<model>.json

No pose labels exist for these frames, so this measures COVERAGE and completeness, not keypoint accuracy.
A tracked player (runs/<clip>_fused/tracks.json) counts as covered when a skeleton's ankles fall within 1.5 m of it.
yolo = yolo26m-pose (Ultralytics AGPL-3.0, experiment only), full frame at 1280 px + a 2.5x pass on the far half.
rtm-* = RTMPose via rtmlib (Apache-2.0): balanced = YOLOX-m + RTMPose-m, performance = YOLOX-x + RTMPose-x (CPU here).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from smartplay.geometry.camera import CourtCamera
from smartplay.geometry.camera3d import project
from smartplay.paths import YOLO26_POSE, rtmpose_body

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["yolo", "rtm-balanced", "rtm-performance"])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--step", type=int, default=10)
args = ap.parse_args()

CLIPS = {"point_12": "runs/highlights/clips/point_12_t0280s_4-0_vs_5-40.mp4",
         "point_24": "runs/highlights/clips/point_24_t0558s_4-6-1-0_vs_6-3-0-15.mp4"}
floor_cam = CourtCamera.from_points([tuple(p) for p in json.loads(Path("data/calibration/highlights_main_view.json").read_text())])
P = np.array(json.loads(Path("data/calibration/highlights_main_view_3d.json").read_text())["P"])

if args.model == "yolo":
    from ultralytics import YOLO

    model = YOLO(str(YOLO26_POSE))
    far = project(P, np.array([[x, y, z] for x in (-6.0, 6.0) for y in (0.0, 10.5) for z in (0.0, 2.4)]))
    FB = (max(0, int(far[:, 0].min())), max(0, int(far[:, 1].min()) - 10), int(far[:, 0].max()), int(far[:, 1].max()) + 10)

    def run(img):
        def one(im):
            r = model.predict(im, verbose=False, conf=0.25, classes=[0], imgsz=1280)[0]
            if r.keypoints is None or len(r.keypoints.xy) == 0:
                return np.zeros((0, 17, 2)), np.zeros((0, 17))
            return r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        kp, kc = one(img)
        x0, y0, x1, y1 = FB
        kp2, kc2 = one(cv2.resize(img[y0:y1, x0:x1], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
        if len(kp2):
            kp2 = kp2 / 2.5 + np.array([x0, y0])
            keep = [i for i in range(len(kp2)) if not any(np.nanmedian(np.linalg.norm(kp2[i] - kp[j], axis=1)) < 15
                                                          for j in range(len(kp)))]
            kp, kc = np.concatenate([kp, kp2[keep]]), np.concatenate([kc, kc2[keep]])
        return kp, kc
else:
    body = rtmpose_body(args.model.split("-")[1], device="cpu")

    def run(img):
        kp, sc = body(img)
        kp, sc = np.asarray(kp, float), np.asarray(sc, float)
        if kp.size == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        return kp.reshape(-1, 17, 2), sc.reshape(-1, 17)
# score threshold: the same 0.3 cut for both libraries (rtmlib SimCC scores are not probabilities: the "complete"
# column is therefore not comparable across libraries, coverage is)
THR = 0.3
res = {"near": [0, 0], "far": [0, 0], "complete": [0, 0], "extra": 0, "frames": 0, "seconds": 0.0}
for clip, video in CLIPS.items():
    tracks: dict[int, list] = {}
    for r in json.loads(Path(f"runs/{clip}_fused/tracks.json").read_text()):
        tracks.setdefault(r["frame"], []).append(r)
    cap = cv2.VideoCapture(video)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % args.step == 0 and f in tracks:
            t0 = time.time()
            kp, kc = run(img)
            res["seconds"] += time.time() - t0
            res["frames"] += 1
            feet = []
            for k in range(len(kp)):
                a = [kp[k][i] for i in (15, 16) if kc[k][i] > THR]
                feet.append(floor_cam.to_world(*np.mean(a, axis=0)) if a else None)
            used = set()
            for r in tracks[f]:
                side = "near" if r["foot_xy_m"][1] < 0 else "far"
                res[side][1] += 1
                best = None
                for k, ft in enumerate(feet):
                    if ft is None or k in used:
                        continue
                    d = np.hypot(ft[0] - r["foot_xy_m"][0], ft[1] - r["foot_xy_m"][1])
                    if d < 1.5 and (best is None or d < best[1]):
                        best = (k, d)
                if best is not None:
                    used.add(best[0])
                    res[side][0] += 1
                    res["complete"][1] += 1
                    res["complete"][0] += int((kc[best[0]] > THR).sum() >= 13)
            res["extra"] += sum(1 for k, ft in enumerate(feet) if k not in used and ft is not None
                                and abs(ft[0]) < 5 and abs(ft[1]) < 10)
        f += 1
    cap.release()
summary = {"model": args.model, "frames": res["frames"],
           "near_covered": round(res["near"][0] / max(1, res["near"][1]), 3),
           "far_covered": round(res["far"][0] / max(1, res["far"][1]), 3),
           "complete_skeletons": round(res["complete"][0] / max(1, res["complete"][1]), 3),
           "extra_on_court": res["extra"], "sec_per_frame": round(res["seconds"] / max(1, res["frames"]), 3)}
print(summary)
args.out.write_text(json.dumps(summary, indent=1))
