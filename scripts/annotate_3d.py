"""Annotated video with the ball in 3D (full flight arcs, height, speed, shadow, bounces) and player skeletons.

    python scripts/annotate_3d.py --video <clip.mp4> --run runs/<clip>_fused \
        --calib data/calibration/highlights_main_view_3d.json --floor data/calibration/highlights_main_view.json \
        --out runs/<clip>_3d/annotated_3d.mp4 [--pose yolo|rtm]

Needs <run>/ball.json, <run>/ball3d.json (scripts/ball3d_clip.py) and <run>/tracks.json.
Drawing: magenta = ball detected in the image; cyan arc = the fitted 3D flight projected back (the whole flight, not
only the part already played); dark ellipse = the ball's shadow on the floor with a vertical line to the ball (height);
white ring = floor bounce (stays for 2 s); skeletons coloured by team (A near = orange, B far = cyan).
Pose: yolo = yolo26m-pose from the earlier project (Ultralytics AGPL-3.0, experiment only); rtm = RTMPose via rtmlib
(Apache-2.0) if installed.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.analytics.player_stats import PlayerStats
from smartplay.ball.ballistic3d import Flight3D
from smartplay.geometry.camera import CourtCamera
from smartplay.geometry.camera3d import project
from smartplay.ingestion.video import iter_frames, probe
from smartplay.paths import YOLO26_POSE, rtmpose_body
from smartplay.schemas.records import PlayerTrack
from smartplay.viz import annotate as viz

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--calib", type=Path, required=True)
ap.add_argument("--floor", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--pose", choices=["yolo", "rtm", "none"], default="yolo")
args = ap.parse_args()

info = probe(args.video)
fps = info.fps
P = np.array(json.loads(args.calib.read_text())["P"])
floor_cam = CourtCamera.from_points([tuple(p) for p in json.loads(args.floor.read_text())])
ball = json.loads((args.run / "ball.json").read_text())
b3 = json.loads((args.run / "ball3d.json").read_text())
flights = [Flight3D(f["f0"], f["f1"], np.array(f["p0"]), np.array(f["v0"]), f["rms_px"], f["n_obs"], f["end_contact"],
                    f.get("anchor", "free")) for f in b3["flights"]]
tracks: dict[int, list[dict]] = {}
for r in json.loads((args.run / "tracks.json").read_text()):
    tracks.setdefault(r["frame"], []).append(r)
verified = [(b["t_frame"], np.array([b["xy"][0], b["xy"][1], 0.0])) for b in b3.get("bounces", [])]
# floor contacts: verified bounces (joint fit, restitution checked) in white; other flights ending low in grey ("floor?")
bounces = [(int(round(tf)), xyz, True) for tf, xyz in verified]
bounces += [(fl.f1, fl.at(fl.f1, fps), False) for fl in flights if fl.end_contact == "floor"
            and not any(abs(fl.f1 - tf) <= 3 for tf, _ in verified)]

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)]
TEAM = {"A": (40, 140, 255), "B": (230, 200, 40)}

pose_fn = None
if args.pose == "yolo":
    from ultralytics import YOLO

    model = YOLO(str(YOLO26_POSE))

    # far half of the court, from the floor to 2.4 m high, as an image box (players there are ~50 px tall)
    _far = np.array([[x, y, z] for x in (-6.0, 6.0) for y in (0.0, 10.5) for z in (0.0, 2.4)])
    _fb = project(P, _far)
    FAR_BOX = (max(0, int(_fb[:, 0].min())), max(0, int(_fb[:, 1].min()) - 10),
               min(info.width, int(_fb[:, 0].max())), min(info.height, int(_fb[:, 1].max()) + 10))
    UP = 2.5

    def _run(img, imgsz):
        r = model.predict(img, verbose=False, conf=0.25, classes=[0], imgsz=imgsz)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        return r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()

    def pose_fn(img):
        kp, kc = _run(img, 1280)
        x0, y0, x1, y1 = FAR_BOX
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        kp2, kc2 = _run(crop, 1280)
        if len(kp2):
            kp2 = kp2 / UP + np.array([x0, y0])
            # keep far-crop skeletons that do not duplicate a full-frame one
            keep = [i for i in range(len(kp2)) if not any(
                np.nanmedian(np.linalg.norm(kp2[i] - kp[j], axis=1)) < 15 for j in range(len(kp)))]
            kp, kc = np.concatenate([kp, kp2[keep]]), np.concatenate([kc, kc2[keep]])
        return kp, kc
elif args.pose == "rtm":
    body = rtmpose_body("performance", device="cpu")  # GPU needs onnxruntime-gpu (CUDA 12): see docs/MIGRATION.md

    def pose_fn(img):
        kp, sc = body(img)
        return np.asarray(kp).reshape(-1, 17, 2), np.asarray(sc).reshape(-1, 17)


def team_of(kp, kc, frame):
    """Team of a skeleton: nearest tracked player (floor metres) to the point between its ankles."""
    feet = [kp[i] for i in (15, 16) if kc[i] > 0.3]
    if not feet or frame not in tracks:
        return None
    u, v = np.mean(feet, axis=0)
    x, y = floor_cam.to_world(u, v)
    best = min(tracks[frame], key=lambda r: (r["foot_xy_m"][0] - x) ** 2 + (r["foot_xy_m"][1] - y) ** 2)
    return best["team"] if (best["foot_xy_m"][0] - x) ** 2 + (best["foot_xy_m"][1] - y) ** 2 < 2.0 ** 2 else None


def pt(p):
    return int(round(p[0])), int(round(p[1]))  # noqa: RUF046 (numpy floats)


args.out.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (info.width, info.height + viz.BAR_H))
stats = PlayerStats(fps)
for idx, frame in iter_frames(args.video):
    if idx >= len(ball):
        break
    if pose_fn is not None:
        kps, kcs = pose_fn(frame)
        for kp, kc in zip(kps, kcs, strict=True):
            if (kc > 0.3).sum() < 6:
                continue
            team = team_of(kp, kc, idx)
            if team is None:  # not standing on a tracked player's position: spectator, staff, reflection
                continue
            col = TEAM[team]
            for a, b in SKELETON:
                if kc[a] > 0.3 and kc[b] > 0.3:
                    cv2.line(frame, pt(kp[a]), pt(kp[b]), col, 3, cv2.LINE_AA)
            for j in range(17):
                if kc[j] > 0.3:
                    cv2.circle(frame, pt(kp[j]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    # bounces: white ring on the floor for 2 s after landing
    for f_b, xyz, ok in bounces:
        if 0 <= idx - f_b <= 2 * fps:
            col, txt = ((255, 255, 255), "bounce") if ok else ((150, 150, 150), "floor?")
            ring = np.array([[xyz[0] + 0.25 * np.cos(a), xyz[1] + 0.25 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 24)])
            cv2.polylines(frame, [project(P, ring).astype(np.int32)], True, col, 2, cv2.LINE_AA)
            cv2.putText(frame, txt, pt(project(P, np.array([[xyz[0], xyz[1], 0.0]]))[0] + [12, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    ball_text = "ball: lost"
    fl = next((f for f in flights if f.f0 <= idx <= f.f1), None)
    if fl is not None:
        arc = np.array([fl.at(f, fps) for f in np.linspace(fl.f0, fl.f1, 40)])
        cv2.polylines(frame, [project(P, arc).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
        xyz = fl.at(idx, fps)
        p_ball, p_shadow = project(P, np.array([xyz, [xyz[0], xyz[1], 0.0]]))
        cv2.ellipse(frame, pt(p_shadow), (14, 5), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.line(frame, pt(p_shadow), pt(p_ball), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, pt(p_ball), 9, (255, 255, 0), 2, cv2.LINE_AA)
        sp = fl.speed_kmh(idx, fps)
        viz._put(frame, f"{sp:.0f} km/h  h {xyz[2]:.1f} m", (pt(p_ball)[0] + 14, pt(p_ball)[1] - 10), 0.6, (255, 255, 0))
        ball_text = f"ball 3D: {sp:.0f} km/h, height {xyz[2]:.1f} m"
    b = ball[idx]
    if b["uv"] is not None and b["visibility"] == "visible":
        cv2.circle(frame, pt(b["uv"]), 5, (255, 0, 255), -1, cv2.LINE_AA)
        if fl is None:
            ball_text = "ball: seen (2D only)"
    tr = [PlayerTrack(**r) for r in tracks.get(idx, [])]
    stats.update(tr)
    frame = viz.compose_with_top_bar(frame, stats.snapshot(), idx / fps, 0, ball_text)
    writer.write(frame)
writer.release()
print(f"wrote {args.out}: {len(flights)} flights in 3D, {len(verified)} verified bounces, "
      f"{len(bounces) - len(verified)} unverified floor contacts")
