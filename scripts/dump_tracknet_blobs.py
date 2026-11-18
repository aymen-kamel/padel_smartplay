"""EXPERIMENT: run the old project's TrackNet (ball_detection.pt) and save candidate blobs per frame,
in the same format as dump_ball_blobs.py, so both detectors can be compared with the same tracker.

    python scripts/dump_tracknet_blobs.py --video <clip.mp4> --out runs/<clip>/ball_blobs_tracknet.json

Licence note: the model class comes from the user's earlier project (a fork of third-party code) and the
weights' licence is unverified. The class is loaded from its original path and NOT copied into this repo.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from smartplay.ball.detector import heatmap_blobs
from smartplay.ingestion.video import iter_frames, probe

OLD = Path(r"C:\Users\dell\OneDrive\Bureau\pfe\padel_analytics_repo\padel_analytics_repo")
H, W, SEQ = 288, 512, 8

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--weights", type=Path, default=OLD / "weights/ball_detection.pt")
ap.add_argument("--threshold", type=float, default=0.3)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("old_models", OLD / "trackers/ball_tracker/models.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ck = torch.load(args.weights, map_location="cpu", weights_only=False)
pd_ = ck["param_dict"]
print("checkpoint:", pd_)
assert pd_["seq_len"] == SEQ and pd_["bg_mode"] == "concat"
net = mod.TrackNet(in_dim=(SEQ + 1) * 3, out_dim=SEQ)
net.load_state_dict(ck["model"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
net.to(dev).eval()

info = probe(args.video)
frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)
          for _, f in iter_frames(args.video)]
n = len(frames)
arr = np.stack(frames)  # (n, H, W, 3)
# background = per-pixel median over up to 300 evenly spaced frames (fixed camera)
idx = np.linspace(0, n - 1, min(n, 300)).astype(int)
median = np.median(arr[idx], axis=0)  # (H, W, 3)
med_c = np.moveaxis(median, -1, 0)  # (3, H, W)
print(f"{n} frames, background from {len(idx)} frames, device {dev}")

acc = np.zeros((n, H, W), np.float32)
cnt = np.zeros(n, np.float32)
with torch.no_grad():
    for s in range(0, n - SEQ + 1, 1):
        win = np.moveaxis(arr[s:s + SEQ], -1, 1).reshape(SEQ * 3, H, W)  # (24, H, W)
        x = np.concatenate([med_c, win], axis=0)[None].astype(np.float32) / 255.0
        y = net(torch.from_numpy(x).to(dev))[0].cpu().numpy()  # (SEQ, H, W)
        for k in range(SEQ):
            acc[s + k] += y[k]
            cnt[s + k] += 1
acc /= np.maximum(cnt, 1)[:, None, None]
sx, sy = info.width / W, info.height / H
blobs = [heatmap_blobs(acc[i], args.threshold, sx, sy) for i in range(n)]
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps([[[b.u, b.v, b.score] for b in fr] for fr in blobs]))
print(f"{args.out}: {n} frames, {sum(bool(f) for f in blobs)} with a candidate, "
      f"{sum(len(f) > 1 for f in blobs)} with several")
