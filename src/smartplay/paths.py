"""Single place for every model file: all weights live in <repo>/weights (see weights/MODELS.md).

Nothing here points to the earlier PFE project or to library caches, so the repository is self-contained.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"

WASB_TENNIS = WEIGHTS / "wasb" / "wasb_tennis_best.pth.tar"
TRACKNETV3 = WEIGHTS / "tracknetv3_official" / "ckpts" / "TrackNet_best.pt"
RFDETR_SMALL = WEIGHTS / "rfdetr" / "rf-detr-small.pth"
YOLO26_POSE = WEIGHTS / "yolo26_pose" / "yolo26m-pose.pt"  # Ultralytics AGPL-3.0: experiments only

_RTM = WEIGHTS / "rtmpose"
RTMPOSE = {  # mode -> (person detector, input size), (pose model, input size); same pairs as rtmlib's presets
    "balanced": ((_RTM / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx", (640, 640)),
                 (_RTM / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx", (192, 256))),
    "performance": ((_RTM / "yolox_x_8xb8-300e_humanart-a39d44ed.onnx", (640, 640)),
                    (_RTM / "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx", (288, 384))),
}


def rtmpose_body(mode: str = "performance", device: str = "cpu"):
    """rtmlib Body (YOLOX person boxes + RTMPose keypoints) loaded from weights/rtmpose, never downloaded."""
    from rtmlib import Body

    if device == "cuda":
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("RTMPose on GPU needs onnxruntime-gpu (CUDA 12); this environment has the CPU build only "
                               "(see docs/MIGRATION.md, option B). Use device='cpu' or install the GPU build.")
    (det, det_size), (pose, pose_size) = RTMPOSE[mode]
    return Body(det=str(det), det_input_size=det_size, pose=str(pose), pose_input_size=pose_size,
                backend="onnxruntime", device=device)
