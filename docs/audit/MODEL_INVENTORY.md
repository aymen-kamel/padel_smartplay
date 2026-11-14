# Model inventory

Date: 2026-11-13. Read-only: **no checkpoint was loaded and no binary was hashed**. Sizes are from the file system; "same size" is used as a duplicate hint, not proof. Architectures are inferred from the code that loads each file or from `BO/models/ckpt_summary.txt`. Licence entries are flags, not legal conclusions; each is **UNVERIFIED** until checked at the source in Phase 2.

Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.
Positioning decision (user, 2026-11-13): clean rewrite under the user's own name and licence. Hence a "licence flag" below means *cannot be shipped or relicensed freely*.

| ID | File (size in bytes) | Copies | Inferred role / architecture | Used by default? | Probable origin | Licence flag | Verdict |
|---|---|---|---|---|---|---|---|
| M01 | `MAIN/weights/ball_detection.pt` (136,190,877) | same size in `BO/models/`, and inside both root zips | Ball detector, TrackNet family, `seq_len 8`, `bg_mode concat` (`ball_tracker.py:367-380`) | **Yes** (`config.py:47`) | Upstream *Padel Analytics* Drive (`MAIN/README.md:37-38`); training data unknown | Upstream licence UNVERIFIED (secondary note: CC BY-NC-SA 4.0) → **restrictive** | Baseline only; replace |
| M02 | `MAIN/weights/InpaintNet_best.pt` (6,264,451) | same size in `BO/models/` | InpaintNet trajectory repair (`ball_tracker/models.py:135`) | No (`config.py:48` = None) | Upstream / TrackNetV3 reference | UNVERIFIED | Evaluate in Phase 2 |
| M03 | `MAIN/weights/model_best.pt` (45,647,717; 2026-05-09) | same size in `BO/models/` | Unknown TrackNet variant ("TrackNetUNet"/"Flat", compared in `BO/scratch/`) | No | **Unknown**; newer than the upstream files → possibly user-trained or downloaded | UNVERIFIED | Identify origin first |
| M04 | `BO/models/ball_detection_TrackNet.pt` (42,886,919; 03-31) | — | 18-conv TrackNet-style net, 9-channel input (3 frames × RGB), 256-channel output (`ckpt_summary.txt`; which file it describes is not stated) | No (`run_tracknet.py` only) | Padex release name (`padex/weights.py:31`) | **GPL-3.0 project's release** → treat as restrictive | Archive |
| M05 | `BO/models/yolo5_last.pt` (172,617,657) | in `ball_only.zip` | "last.pt" training checkpoint, YOLOv5-class size | **No — referenced by no script** | Unknown | YOLOv5 is Ultralytics **AGPL-3.0** (UNVERIFIED) | Archive |
| M06 | `BO/models/best (1).pt` (6,224,554; 2026-02-10) | — | Small YOLO ball detector "fine-tuned via Roboflow" (`BO/docs/RAPPORT_TRACKING_PADEL.md:13`) | In `BO` YOLO scripts | User, per report; **no training data, config or logs on disk** | Dataset licence unknown; Ultralytics format AGPL-3.0 | Archive; 30.0 % frames detected on `rally.mp4` |
| M07 | `MAIN/weights/player_keypoints_detection.pt` (53,352,505) | in `padel_analytics_repo.zip` | YOLO-Pose, 17 COCO keypoints, imgsz 1280 (`config.py:41`, `players_keypoints_tracker.py:65-83`) | **Yes** | Upstream | Upstream UNVERIFIED + Ultralytics AGPL-3.0 | Replace |
| M08 | `MAIN/weights/court_keypoints_detection.pt` (6,650,193) | in zip | YOLO court keypoints | **No** — `DYNAMIC_COURT_KEYPOINTS = False` (`config.py:26`) | Upstream | Same as M07 | Evaluate in Phase 2 |
| M09 | `MAIN/weights/yolo26m.pt` (44,255,705) | — | Ultralytics YOLO26-medium, COCO, person class 0 (`players_tracker.py:419`) | **Yes** (`config.py:33`) | Ultralytics (also in Padex release list) | **AGPL-3.0** (secondary: `iit_courtvison/MODEL_REGISTRY.md`) | Replace with a permissively licensed detector if possible |
| M10 | `MAIN/weights/players_detection.pt` and `weights/players_detection/yolov8m.pt` (52,136,884 each) | identical size | Generic YOLOv8m COCO | No (superseded by M09) | Ultralytics | AGPL-3.0 | Archive |
| M11 | `MAIN/weights/yolo26m-pose.pt` (49,036,866) | — | Ultralytics pose, COCO-17 | Only `analyze_full_match.py`, `detect_shoots_only.py` | Ultralytics | AGPL-3.0 | Candidate pose baseline |
| M12 | `MAIN/weights/sam2_s.pt` (184,309,650) | — | SAM 2 small via `ultralytics.SAM` (`sam2_tracker.py:38`) | **Yes** (`USE_SAM2 = True`) | Meta (SAM 2) | Apache-2.0 expected, UNVERIFIED | Drop (not needed for score) |
| M13 | `MAIN/weights/shot_classifier.pkl` (174,969; 05-04) + `shot_classifier_features.pkl` (230) | `weights/` at root has **different** sizes (190,713 + 230; 05-06) | scikit-learn RandomForest on `prox_*`, `rel_height`, `player_y_m` (`train_shot_classifier.py:35-41`) | **No** — never reached at inference (TD-02) | User | User's own; **pickle: load only trusted files** | Retrain on a real dataset |
| M14 | *absent*: `weights/lstm_shot_classifier.pth`, `lstm_scaler.pkl` | — | LSTM, 31 steps, 3 inputs, 5 classes (`train_lstm_classifier.py:18-31`) | — | — | — | Never produced |
| M15 | *absent, downloaded at run time*: `ball_detection_TrackNetV3.pt`, `yolo26m*.pt`, `ball_detection_TrackNet.pt` | `padex/assets/weights/` is empty | Padex weights from the author's GitHub release `v0.1.0` (`padex/src/padex/weights.py:25-32`) | — | Padex author | **GPL-3.0 project** | Do not use |
| — | `padex/rally_tracking_cache.pkl` (1,363,644) | — | Pickled tracking cache, not a model | — | Padex run | **Unpickling untrusted files can execute code** | Archive |

## Duplicates (by identical size; hashes not computed)
`ball_detection.pt`, `InpaintNet_best.pt`, `model_best.pt` exist twice (MAIN and BO) and again inside the two root zips. `players_detection.pt` = `players_detection/yolov8m.pt`. Roughly 450 MB of weights are duplicated. Deleting is **not** proposed here (rule: ask first).

## Observations
- Only three weights are used by the default pipeline: M01 (ball), M07 (pose), M09 (players) plus M12 (SAM2, optional in effect).
- No weight has a recorded training dataset, config, seed, epoch count or metric file. The user's own models (M06, possibly M03) cannot be reproduced.
- Every weight whose origin is known comes from an upstream project with a restrictive or copyleft licence, or from Ultralytics (AGPL-3.0). **A clean rewrite under the user's own licence cannot ship M01, M04, M07, M08, M09–M11 as-is.**

## UNVERIFIED
1. Licence of every file above; origin of M03, M05, M06.
2. Architecture of M03/M04 (not loaded).
3. Whether the two `shot_classifier.pkl` files differ in training data or only in library version.
4. Whether M01 was trained on the footage later used for evaluation (leakage risk).
