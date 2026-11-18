# Third-party notices

SmartPlay AI includes or downloads the following third-party work. Each keeps its own licence.

## Code included in this repository

### WASB-SBDT — HRNet ball model (`src/smartplay/ball/wasb_hrnet.py`)
- Source: https://github.com/nttcom/WASB-SBDT (commit 923462cacdeb), `src/models/hrnet.py`
- Paper: Tarashima et al., *Widely Applicable Strong Baseline for Sports Ball Detection and Tracking*, BMVC 2023, https://arxiv.org/abs/2311.05237
- Licence: MIT — Copyright (c) 2023 NTT Communications Corporation
- Based on HRNet — Copyright (c) Microsoft, MIT License (Bin Xiao, modified by Bowen Cheng)

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions: The above copyright notice
> and this permission notice shall be included in all copies or substantial portions of the
> Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

## Model weights downloaded at run time (not stored in git)

| Weights | Source | Licence |
|---|---|---|
| WASB tennis (`wasb_tennis_best.pth.tar`, 6.1 MB) | WASB-SBDT model zoo (Google Drive link in `MODEL_ZOO.md`) | Released with the MIT-licensed repository; trained on the tennis dataset of Huang et al. (TrackNet) — dataset terms not verified |
| RF-DETR Small COCO (`rf-detr-small.pth`, 368 MB) | Roboflow, `rfdetr` package | Apache-2.0 (code and N/S/M/L weights) |

## Libraries used at run time
PyTorch (BSD-style), OpenCV (Apache-2.0), NumPy/SciPy (BSD), Pydantic (MIT), `rfdetr` (Apache-2.0),
`imageio-ffmpeg` (BSD-2) bundling FFmpeg (LGPL/GPL build — used as a separate executable).

## TrackNetV3 (ball detector, experiment)
- Source: https://github.com/qaz812345/TrackNetV3 - weights `TrackNet_best.pt`, `InpaintNet_best.pt` (badminton shuttlecock training). Licence: MIT (stated by the repository for code and checkpoints). The network class used for inference is still loaded from the user's earlier project, to be replaced by an own re-implementation or the official code before release.
