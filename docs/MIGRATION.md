# Migration: one folder, one environment (2026-11-19)

Decisions approved by the owner: copy every model into `weights/` (originals untouched); one Python environment in `.venv`
with the NVIDIA GPU; Docker postponed to deployment; RTMPose = main pose model, YOLO26-pose kept as AGPL experiment.
Rules: nothing in the old PFE project or the basketball project is modified or deleted; the old 7 GB environment
(`pfe\smartplay`) stays until everything is validated; every model is tested after migration; stop on any conflict.

## 1. Inventory (read-only, done)

| Model | Current location | Size | Licence | Target in `weights/` |
|---|---|---|---|---|
| WASB HRNet, tennis | `weights/wasb/wasb_tennis_best.pth.tar` (already here) | 6.1 MB | MIT | unchanged |
| TrackNetV3 + InpaintNet (official) | `weights/tracknetv3_official/ckpts/` (already here) | 136 + 6 MB | MIT | unchanged |
| RF-DETR Small (COCO) | `C:\Users\dell\.roboflow\models\rf-detr-small.pth` (library cache) | 386 MB | Apache-2.0 | `weights/rfdetr/` |
| RTMPose-m / RTMPose-x (body7) | `C:\Users\dell\.cache\rtmlib\hub\checkpoints\` (library cache) | 54 / 198 MB | Apache-2.0 | `weights/rtmpose/` |
| YOLOX-m / YOLOX-x (person boxes for RTMPose) | same cache | 101 / 396 MB | Apache-2.0 | `weights/rtmpose/` |
| YOLO26m-pose | old project `pfe\padel_analytics_repo\...\weights\yolo26m-pose.pt` | 49 MB | **AGPL-3.0 (experiment only)** | `weights/yolo26_pose/` |

Environments: old `pfe\smartplay` = Python 3.11.0, torch 2.7.1+cu118, torchvision 0.22.1+cu118, rfdetr 1.12.0,
ultralytics 8.4.13, scikit-learn 1.8.0, scipy 1.17.0, numpy 2.3.5, imageio-ffmpeg 0.6.0, opencv-python 4.11 **and**
opencv-contrib 4.13 (duplicate). New `.venv` = Python 3.11.0, numpy 2.4.6, onnxruntime 1.32.0 (CPU), rtmlib 0.0.16,
rapidocr-onnxruntime 1.4.4, opencv 5.x, pydantic, pytest, ruff, yt-dlp, gdown (no torch).
GPU: RTX 3050 6 GB Laptop, driver 610.60 (CUDA user-mode 13.3, so CUDA 11.8 and 12.x wheels both run). Disk: 58 GB free.
`.gitignore` already excludes `data/`, `weights/`, `runs/`, `.venv/`, `*.mp4`, `*.pt`, `*.pth`, `*.pkl` (to extend: `*.onnx`, `*.tar`, `*.zip`, `*.npz`).

## 2. Steps (one at a time, each checked)

1. Copy the 4 external models into `weights/` and record SHA-256 of every model file (source and copy must match).
2. Build the single environment in `.venv` (torch with CUDA, rfdetr, ultralytics, scikit-learn, scipy, imageio-ffmpeg, opencv,
   onnxruntime, rtmlib, rapidocr, pydantic, pytest, ruff). **Open choice, see section 3.**
3. Code: replace hard-coded paths into the old project and library caches by `weights/...`
   (RF-DETR `pretrain_weights`, rtmlib `det=` / `pose=`, YOLO26 path in scripts).
4. Tests: `torch.cuda.is_available()` + device name; load and run each model once on a real frame (WASB, TrackNetV3, RF-DETR,
   RTMPose-m/x with YOLOX, YOLO26-pose, RapidOCR); full `pytest`; re-run `cli annotate` on clip 24 and compare with the
   old-environment output (ball seen %, players observed %).
5. Docs and safety: `weights/MODELS.md` (source URL, licence, size, SHA-256, used by), `requirements.lock` (pip freeze of the
   validated `.venv`), extended `.gitignore`, README install/run section.
6. Only after the owner validates: the old 7 GB environment may be removed by the owner (never by me).

## 3. Conflict found: ONNX Runtime CPU vs GPU

- RTMPose (via rtmlib) runs on ONNX Runtime. On CPU it costs 0.6 s (m) to 1.9 s (x) per frame: too slow for whole videos.
- GPU needs the `onnxruntime-gpu` package, which uses **CUDA 12**. The validated torch build is **CUDA 11.8**.
  Both can coexist only if torch is also moved to a CUDA 12 build (e.g. torch 2.7.1+cu126), which changes the GPU stack
  under every model already measured (all must be re-tested anyway).
- `rapidocr-onnxruntime` depends on the CPU package `onnxruntime`; the CPU and GPU packages install the same module and must
  not be installed together (rapidocr would be installed without its onnxruntime dependency and use the GPU package).

Options:
- **A (safe, recommended first):** torch 2.7.1+cu118 as validated, `onnxruntime` CPU. Everything works as today; RTMPose stays
  slow (CPU) until B is done.
- **B (faster pose):** torch 2.7.1+cu126 + `onnxruntime-gpu` (+ rapidocr without its CPU dependency). RTMPose on GPU; more risk.
