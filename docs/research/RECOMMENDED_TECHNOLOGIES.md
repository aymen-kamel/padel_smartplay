# Recommended technologies — one per component (Phase 2)

Date: 2026-11-14. Constraints: one fixed camera · after-match processing (assumption, to confirm) · v1 = points → full score · clean rewrite under the user's own licence (no GPL/NC/AGPL code or weights shipped) · 4 GB VRAM budget · Python 3.11+.
Every recommendation is a **candidate to be validated by measurement in Phase 3/4**, not a result. Sources and labels: `STATE_OF_THE_ART.md`, `MODEL_COMPARISON.md`.

| # | Component | Recommendation | Why (short) | Fallback |
|---|---|---|---|---|
| 1 | Video ingestion | **FFmpeg (via PyAV or subprocess) for decoding + audio extraction; OpenCV for image ops**; process in bounded windows, never the whole video in RAM | Audio becomes a first-class signal; fixes the 2.5 GB background-median design (audit C01). Note: ffmpeg is not on PATH today | OpenCV `VideoCapture` for video only |
| 2 | Court calibration | **One-time manual calibration per camera install** (click known court points) → RANSAC homography + `solvePnP` 3D camera model with FIP court geometry; report reprojection error in cm | Fixed camera = calibrate once; 3D model needed for floor-vs-wall and airborne ball | Homography only |
| 3 | Player detection | **RF-DETR Small** (Apache-2.0 code + weights) | Permissive, strong COCO AP, measured 256 MB inference on this GPU | RF-DETR Nano or RT-DETRv2-S |
| 4 | Player tracking | **OC-SORT** (MIT) via the Apache-2.0 Roboflow `trackers` library + a padel constraint layer (exactly 4 identities, 2 per side, teams swap only at changes of ends) | Proven tracker + domain prior; replaces the hand-made 4-slot manager | ByteTrack |
| 5 | Ball detection | **TrackNetV3-architecture model (MIT), fine-tuned on PadelTracker100 + own labelled frames**; zero-shot **WASB tennis** (MIT) as the comparison baseline | Same family as today but legally reusable and retrainable; WASB gives a strong reference | WASB fine-tuned |
| 6 | Ball tracking / gap filling | **Constant-acceleration Kalman with a coasting limit (~0.3 s) and visibility states** (visible / occluded / inferred), outlier gating in metres | Stops invented positions (audit TD-06) | Short-gap spline only |
| 7 | Hit detection | **Audio first**: onset/energy baseline → small CNN on log-mel trained on own labelled hits; fused with ball-direction change near a player | Padel evidence: audio hit F1 92 % (Decorte et al. 2024); independent of ball visibility | Trajectory-only |
| 8 | Bounce detection | **Trajectory-corner classifier** (gradient-boosted trees on trajectory features, method as in the MIT tennis bounce detector) + 3D-height and wall-proximity features from item 2 + bounce sound | Bounces are corners, not image-y peaks; walls are legal contacts that must not count | Rule-based corner detector |
| 9 | Rally / point segmentation | **Rule-based**: serve detection (server position + toss/bounce pattern) and end-of-point (silence + no ball + players walking), using FIP's ≤ 20 s between points | Simple, explainable, measurable; learnable later from labels | Pause-only segmentation |
| 10 | Point winner | **Event-based rules for the main FIP Rule 13 endings** (second floor bounce, direct-to-wall/out, net) producing a **probability** per point | Covers what the current FSM misses; uncertainty explicit | "Last hitter wins unless error" heuristic |
| 11 | Score engine | **Own typed, deterministic FIP score automaton** (Advantage / Star Point / Golden Point, tie-break, alternative formats, changes of ends) + **constrained decoding (Viterbi) of the point-winner probabilities**, anchored on observed side changes and server rotation | Rules verified in the FIP 2026 rulebook; turns noisy points into a consistent score | Greedy score update |
| 12 | Shot classification | **Deferred** (not needed for score v1 except "serve"); later RTMPose-s (Apache) + small temporal model trained on PadelTracker100 shot labels | Avoids spending effort before scoring works | — |
| 13 | Analytics | Own metrics computed on **smoothed** metric-space tracks with plausibility caps | Fixes 175.6 m / 6.7 km/h inconsistency (audit C13) | — |
| 14 | Visualization | OpenCV overlay + 2D court view from the calibrated model | Already familiar | — |
| 15 | API | **FastAPI + Pydantic v2** job API (reuse the existing endpoint shape) | Already in use; typed schemas | — |
| 16 | Evaluation | Own harness extending `evaluate_pipeline.py` ideas: per-stage metrics with tolerance windows, **match-level splits**, baselines, confidence intervals | Honest numbers (audit TD-21–26) | — |
| 17 | Data | **PadelTracker100** (CC BY 4.0) for training; **own club matches** as held-out test; highlights video for point-level labels | Only permissive padel dataset found; own matches are the true test | — |

## Not recommended (and why)
- **Ultralytics YOLO models and code** — AGPL-3.0 incompatible with a closed or differently licensed release.
- **Padex code and weights** — GPL-3.0.
- **Upstream Padel Analytics core and weights** — licence unclear, possibly NC-SA.
- **TennisCourtDetector code/weights** — no licence file.
- **SAM 2 segmentation** — not needed for scoring; costly.
- **E2E-Spot-style video spotting in v1** — strongest on paper, but needs many labels and probably more than 4 GB to train; revisit after the audio + trajectory baseline.

## Decisions to confirm before Phase 3
1. **YouTube source videos** (PadelTracker100 frames, your highlights video): using them for training/evaluation in a portfolio project needs a decision on download terms and copyright.
2. **Live vs after-match**: recommendations assume after-match; live would push towards Nano-size models and single-pass processing.
3. **Microphone/audio**: the fixed-camera recordings at the club must keep audio (audio hit detection depends on it).
