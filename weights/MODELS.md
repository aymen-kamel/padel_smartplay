# Models used by SmartPlay

Every model file lives in this folder (paths in `src/smartplay/paths.py`). The files themselves are NOT in git (size,
licences); this list is, so anyone can fetch the same files and check them. Verify a file with
`python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>`.

| File | Role | Source | Licence | Size (bytes) | SHA-256 |
|---|---|---|---|---|---|
| `wasb/wasb_tennis_best.pth.tar` | ball candidates (3-frame HRNet) | WASB-SBDT model zoo, github.com/nttcom/WASB-SBDT | MIT (see `wasb/WASB_LICENSE.md`) | 6102633 | `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb` |
| `tracknetv3_official/ckpts/TrackNet_best.pt` | ball candidates (8 frames + background) | github.com/qaz812345/TrackNetV3 (Google Drive ckpts.zip) | MIT (code and checkpoints, stated by the repository); trained on badminton | 136190877 | `df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a` |
| `tracknetv3_official/ckpts/InpaintNet_best.pt` | trajectory inpainting (not used yet) | same | MIT | 6264451 | `5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c` |
| `rfdetr/rf-detr-small.pth` | person detection (players) | Roboflow RF-DETR Small, COCO; copied from the library cache `~/.roboflow/models` | Apache-2.0 | 386045550 | `d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33` |
| `rtmpose/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx` | **main pose model** (RTMPose-x) | download.openmmlab.com (rtmlib preset "performance"), copied from `~/.cache/rtmlib` | Apache-2.0 | 197660326 | `df0c0fa91e9870b1515dcaff741fd76cc753dcfb12786862f61b243bae81cd52` |
| `rtmpose/yolox_x_8xb8-300e_humanart-a39d44ed.onnx` | person boxes for RTMPose-x | same | Apache-2.0 | 396283264 | `8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da` |
| `rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` | lighter pose model (RTMPose-m) | same (preset "balanced") | Apache-2.0 | 54330655 | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` |
| `rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` | person boxes for RTMPose-m | same | Apache-2.0 | 101400344 | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` |
| `yolo26_pose/yolo26m-pose.pt` | pose, **experiments only** | Ultralytics YOLO26, copied from the earlier PFE project (original untouched) | **AGPL-3.0**: not for a closed product without an Ultralytics licence | 49036866 | `2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4` |

Notes
- The training data behind some weights (broadcast tennis/badminton, COCO, Body7/HumanArt) may carry their own terms; check
  before a commercial release (see `THIRD_PARTY_NOTICES.md`).
- `tracknetv3_official/ckpts.zip` is the original download (kept, not used directly).
- Sources and copies were compared by SHA-256 when they were copied on 2026-11-19 (all identical).
