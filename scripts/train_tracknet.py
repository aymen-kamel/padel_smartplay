"""Fine-tune TrackNetV3 (official checkpoint) on confident pseudo-labels, for a fixed time budget.

    python scripts/train_tracknet.py --data data/train/tracknet --minutes 20 --out weights/tracknetv3_finetuned/trial.pt

Input = background + 8 frames (as at inference, smartplay.ball.tracknet). Target = a disk of radius 3 px (at 288x512)
around the ball for every confident frame. Unconfident frames are masked out of the loss: they teach neither "ball" nor
"no ball". Loss = the focal-weighted binary cross-entropy of TrackNetV3:  -(1-p)^2 y log p - p^2 (1-y) log(1-p).
Augmentation: horizontal flip, small brightness change. Mixed precision on the GPU.
The saved file keeps the official layout ({"model", "param_dict"}) so TrackNetBallDetector loads it unchanged.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from smartplay.ball.tracknet import SEQ, _build
from smartplay.paths import TRACKNETV3

H, W = 288, 512
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("data/train/tracknet"))
ap.add_argument("--minutes", type=float, default=20.0)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", type=Path, default=Path("weights/tracknetv3_finetuned/trial.pt"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

videos = []
for f in sorted(args.data.glob("*_lab.npy")):
    stem = f.name[: -len("_lab.npy")]
    frames = np.load(args.data / f"{stem}.npy", mmap_mode="r")
    lab = np.load(f)
    bg = np.load(args.data / f"{stem}_bg.npy")
    starts = [s for s in range(len(frames) - SEQ + 1) if lab[s:s + SEQ, 2].sum() >= 3]
    videos.append((frames, bg, lab, starts))
n_windows = sum(len(v[3]) for v in videos)
print(f"{len(videos)} videos, {n_windows} training windows (>= 3 confident frames of {SEQ})")

yy, xx = np.mgrid[0:H, 0:W]


def sample():
    frames, bg, lab, starts = videos[rng.integers(len(videos))]
    s = starts[rng.integers(len(starts))]
    win = np.asarray(frames[s:s + SEQ], np.float32)  # (8, H, W, 3)
    b = bg.astype(np.float32)  # (3, H, W)
    lb = lab[s:s + SEQ].copy()
    if rng.random() < 0.5:  # horizontal flip
        win, b = win[:, :, ::-1], b[:, :, ::-1]
        lb[:, 0] = np.where(lb[:, 2] > 0, W - 1 - lb[:, 0], lb[:, 0])
    gain = rng.uniform(0.85, 1.15)
    x = np.concatenate([b, np.moveaxis(win, -1, 1).reshape(SEQ * 3, H, W)], axis=0) * gain / 255.0
    y = np.zeros((SEQ, H, W), np.float32)
    for k in range(SEQ):
        if lb[k, 2] > 0:
            y[k] = ((xx - lb[k, 0]) ** 2 + (yy - lb[k, 1]) ** 2) <= 9.0
    return np.clip(x, 0, 1).astype(np.float32), y, lb[:, 2].astype(np.float32)


dev = "cuda"
net = _build()
ck = torch.load(TRACKNETV3, map_location="cpu", weights_only=True)
net.load_state_dict(ck["model"], strict=True)
net.to(dev).train()
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")
t_end = time.time() + args.minutes * 60
it, hist = 0, []
while time.time() < t_end:
    xs, ys, ms = zip(*(sample() for _ in range(args.batch)), strict=True)
    x = torch.from_numpy(np.stack(xs)).to(dev)
    y = torch.from_numpy(np.stack(ys)).to(dev)
    m = torch.from_numpy(np.stack(ms)).to(dev)[:, :, None, None]
    with torch.autocast("cuda", dtype=torch.float16):
        p = net(x)
    p = p.float().clamp(1e-6, 1 - 1e-6)
    loss_map = -((1 - p) ** 2 * y * torch.log(p) + p**2 * (1 - y) * torch.log(1 - p))
    loss = (loss_map * m).sum() / (m.sum() * H * W + 1e-6)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    hist.append(float(loss))
    it += 1
    if it % 100 == 0:
        print(f"iter {it}  loss {np.mean(hist[-100:]):.5f}  ({(t_end - time.time()) / 60:.1f} min left)", flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
param = dict(ck["param_dict"])
param.update({"finetuned_on": "CVSPORTS_Padel confident pseudo-labels (licence not stated: experiment only)",
              "iterations": it, "batch": args.batch, "lr": args.lr})
torch.save({"model": net.state_dict(), "param_dict": param}, args.out)
print(f"saved {args.out} after {it} iterations, last loss {np.mean(hist[-100:]):.5f}")
