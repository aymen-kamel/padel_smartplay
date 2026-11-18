"""TrackNetV3 ball detector (candidate blobs per frame), second source next to WASB.

Network layout re-implemented from the public TrackNetV3 project (https://github.com/qaz812345/TrackNetV3,
MIT, Copyright (c) 2024 qaz812345) so that its official checkpoint ``TrackNet_best.pt`` loads (see
THIRD_PARTY_NOTICES.md). Input: a background image (per-pixel median of the clip) plus 8 consecutive
frames, 288x512 RGB in [0, 1]; output: 8 heatmaps. Needs a FIXED camera for the background to be valid.

Memory: the resized clip is held in RAM (about 0.44 MB per frame): fine for rallies, split long videos.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from smartplay.ball.detector import Blob, heatmap_blobs

H, W, SEQ = 288, 512, 8


def _build():
    import torch
    from torch import nn

    class Conv(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(i, o, 3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Double(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv_1, self.conv_2 = Conv(i, o), Conv(o, o)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class Triple(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv_1, self.conv_2, self.conv_3 = Conv(i, o), Conv(o, o), Conv(o, o)

        def forward(self, x):
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.down_block_1, self.down_block_2 = Double(in_dim, 64), Double(64, 128)
            self.down_block_3, self.bottleneck = Triple(128, 256), Triple(256, 512)
            self.up_block_1, self.up_block_2, self.up_block_3 = Triple(768, 256), Double(384, 128), Double(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, (1, 1))
            self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.up = nn.Upsample(scale_factor=2)

        def forward(self, x):
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(self.pool(x1))
            x3 = self.down_block_3(self.pool(x2))
            x = self.bottleneck(self.pool(x3))
            x = self.up_block_1(torch.cat([self.up(x), x3], dim=1))
            x = self.up_block_2(torch.cat([self.up(x), x2], dim=1))
            x = self.up_block_3(torch.cat([self.up(x), x1], dim=1))
            return torch.sigmoid(self.predictor(x))

    return TrackNet((SEQ + 1) * 3, SEQ)


class TrackNetBallDetector:
    def __init__(self, weights: str | Path, threshold: float = 0.3, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        if ckpt["param_dict"]["seq_len"] != SEQ or ckpt["param_dict"]["bg_mode"] != "concat":
            raise ValueError("expected a TrackNetV3 checkpoint with seq_len 8 and bg_mode 'concat'")
        net = _build()
        net.load_state_dict(ckpt["model"], strict=True)
        self.net = net.to(self.device).eval()
        self.threshold = threshold

    def blobs(self, frames: Iterable[tuple[int, np.ndarray]], width: int, height: int,
              batch: int = 8, bg_samples: int = 300) -> list[list[Blob]]:
        import cv2

        small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)
                 for _, f in frames]
        n = len(small)
        if n < SEQ:
            return [[] for _ in range(n)]
        arr = np.stack(small)
        idx = np.linspace(0, n - 1, min(n, bg_samples)).astype(int)
        bg = np.moveaxis(np.median(arr[idx], axis=0), -1, 0).astype(np.float32)  # (3, H, W)
        acc = np.zeros((n, H, W), np.float32)
        cnt = np.zeros(n, np.float32)
        torch = self.torch
        starts = list(range(n - SEQ + 1))
        with torch.no_grad():
            for b0 in range(0, len(starts), batch):
                xs = []
                for s in starts[b0:b0 + batch]:
                    win = np.moveaxis(arr[s:s + SEQ], -1, 1).reshape(SEQ * 3, H, W).astype(np.float32)
                    xs.append(np.concatenate([bg, win], axis=0) / 255.0)
                y = self.net(torch.from_numpy(np.stack(xs)).to(self.device)).cpu().numpy()
                for s, ys in zip(starts[b0:b0 + batch], y, strict=True):
                    acc[s:s + SEQ] += ys
                    cnt[s:s + SEQ] += 1
        acc /= np.maximum(cnt, 1)[:, None, None]
        sx, sy = width / W, height / H
        return [heatmap_blobs(acc[i], self.threshold, sx, sy) for i in range(n)]
