"""Side-by-side video: the ball as found with the OFFICIAL TrackNetV3 vs the FINE-TUNED one (both merged with WASB).

    python scripts/compare_tracknet_video.py --clip point_24 --out runs/compare_tracknet_point_24.mp4

Left panel: official TrackNetV3 + WASB. Right panel: fine-tuned TrackNetV3 + WASB. Same gated tracker, same drawing.
The hand labels of the clip are drawn as small green dots so the two can be compared by eye.
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.ball.detector import Blob, track_gated
from smartplay.ball.fusion import merge_blobs
from smartplay.ball.tracknet import TrackNetBallDetector
from smartplay.ingestion.video import iter_frames, probe
from smartplay.labeling.store import ClipLabels
from smartplay.viz.annotate import draw_ball

CLIPS = {"point_12": "runs/highlights/clips/point_12_t0280s_4-0_vs_5-40.mp4",
         "point_24": "runs/highlights/clips/point_24_t0558s_4-6-1-0_vs_6-3-0-15.mp4"}
ap = argparse.ArgumentParser()
ap.add_argument("--clip", choices=list(CLIPS), default="point_24")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--official", type=Path, default=Path("weights/tracknetv3_official/ckpts/TrackNet_best.pt"))
ap.add_argument("--finetuned", type=Path, default=Path("weights/tracknetv3_finetuned/trial.pt"))
args = ap.parse_args()

video = CLIPS[args.clip]
info = probe(video)
wasb = [[Blob(*b) for b in fr] for fr in json.loads(Path(f"runs/{args.clip}/ball_blobs_wasb03.json").read_text())]
n = len(wasb)
lab = ClipLabels.load(f"data/labels/{args.clip}.json").ball

tracks = {}
for name, weights in (("official", args.official), ("finetuned", args.finetuned)):
    tn = TrackNetBallDetector(weights, threshold=0.3).blobs(itertools.islice(iter_frames(video), n),
                                                            info.width, info.height)
    tracks[name] = track_gated(merge_blobs(wasb, tn))
    seen = sum(s.visibility == "visible" for s in tracks[name])
    print(f"{name}: ball seen in {seen}/{n} frames", flush=True)

scale = 960 / info.width
H = int(info.height * scale)
writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (960 * 2, H + 40))
for i, frame in iter_frames(video):
    if i >= n:
        break
    panels = []
    for name, title in (("official", "official TrackNetV3 + WASB"), ("finetuned", "fine-tuned TrackNetV3 + WASB")):
        img = frame.copy()
        draw_ball(img, tracks[name], i)
        if lab.get(i) is not None:
            cv2.circle(img, (int(lab[i][0]), int(lab[i][1])), 6, (0, 255, 0), 2, cv2.LINE_AA)
        img = cv2.resize(img, (960, H))
        bar = np.full((40, 960, 3), 28, np.uint8)
        cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([bar, img]))
    writer.write(np.hstack(panels))
writer.release()
print(f"wrote {args.out}")
