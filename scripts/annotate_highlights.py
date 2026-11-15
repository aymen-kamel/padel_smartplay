"""Annotated copy of a broadcast video (uses the cached analysis in --run).

    python scripts/annotate_highlights.py --video "<video>" --run runs/highlights

Main wide shots get players (RF-DETR, kept only if their feet stand on the blue court) and the
ball (WASB, zero-shot). Every frame gets: time, shot type, last scoreboard reading, and a banner
when the scoreboard shows that a point was played. No court calibration is used here.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from smartplay.ball.detector import WasbBallDetector
from smartplay.ingestion.video import ffmpeg_exe
from smartplay.players.detector import PersonDetector
from smartplay.viz.annotate import _put, draw_ball

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--weights", type=Path, default=Path("weights/wasb/wasb_tennis_best.pth.tar"))
args = ap.parse_args()

segs = json.loads((args.run / "segments.json").read_text())
timeline = json.loads((args.run / "scoreboard_timeline.json").read_text())
points = json.loads((args.run / "points.json").read_text())

cap = cv2.VideoCapture(str(args.video))
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tmp = args.run / "annotated_highlights_noaudio.mp4"
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

ball_det = WasbBallDetector(args.weights)
person_det = PersonDetector(0.5)


def board_text(t: float) -> str:
    cur = None
    for row in timeline:
        if row["t"] <= t:
            cur = row
    if cur is None:
        return "scoreboard: not read yet"
    top, bot = cur["top"], cur["bottom"]
    return f"board (top/bottom): sets {list(zip(top[:-2], bot[:-2], strict=True))} games {top[-2]}-{bot[-2]} pts {top[-1]}-{bot[-1]}"


def on_court(frame: np.ndarray, foot: tuple[float, float]) -> bool:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (95, 60, 40), (130, 255, 255))
    x, y = int(foot[0] / 12), int(foot[1] / 12)
    patch = mask[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
    return patch.size > 0 and patch.mean() > 255 * 0.35


t0 = time.time()
done = 0
for seg in segs:
    a, b = seg["start"], seg["end"]
    ball = []
    if seg["kind"] == "main":
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        frames = enumerate(cap.read()[1] for _ in range(b - a + 1))
        ball = ball_det.track(frames, W, H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for idx in range(a, b + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if seg["kind"] == "main":
            for d in person_det.detect(frame):
                if 80 < d.box_uv[3] - d.box_uv[1] < 700 and on_court(frame, d.foot_uv):
                    x1, y1, x2, y2 = (int(v) for v in d.box_uv)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
            draw_ball(frame, ball, idx - a)
        _put(frame, f"t={t:6.1f}s   shot: {'MAIN VIEW' if seg['kind'] == 'main' else 'close-up / crowd / replay'}",
             (20, H - 70), 0.7)
        _put(frame, board_text(t), (20, H - 36), 0.6, (0, 255, 255))
        for p in points:
            if p["t_after"] <= t < p["t_after"] + 4.0:
                who = "TOP team" if p["winner_row"] == "top" else "BOTTOM team"
                _put(frame, f"POINT PLAYED -> {who} won it   ({p['from']}  =>  {p['to']})", (20, H - 110), 0.7,
                     (0, 140, 255))
        writer.write(frame)
        done += 1
    print(f"  {done}/{n_total} frames  ({done / (time.time() - t0):.1f} fps)", flush=True)
writer.release()
final = args.run / "annotated_highlights.mp4"
subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(tmp), "-i", str(args.video), "-map", "0:v", "-map",
                "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-shortest",
                str(final)], check=True)
tmp.unlink()
print("wrote", final)
