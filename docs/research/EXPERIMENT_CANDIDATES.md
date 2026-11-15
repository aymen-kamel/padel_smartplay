# Experiment candidates — performance first, licences reviewed before production

Date: 2026-11-15. Decision (user): during experimentation, technical performance outranks licensing; every component's licence is documented and reviewed before any production use.
Builds on `STATE_OF_THE_ART.md` and `MODEL_COMPARISON.md` (not repeated). Only the options that the earlier licence filter had excluded, or that were new, were researched again.

Labels: **[V]** read at a primary source in an earlier phase · **[S]** secondary source (search result, aggregator) · **UNVERIFIED** = could not be confirmed. Paper numbers come from different datasets and settings: **they cannot rank models for padel**. Only the bake-off below can.

## Candidates per component

| Component | Tier A (benchmark first) | Tier B (benchmark if A falls short) | Licence flags |
|---|---|---|---|
| Player detection | **RF-DETR S/M** (COCO AP 53.0 for S [V]) · **YOLO26-L/X** (paper: up to 57.5 mAP; a Roboflow table lists L at 53.6 — the two sources disagree [S]) · **D-FINE-L/X** (AP 54.0 / 55.8 [S]) | DEIM-D-FINE-L/X (54.7 / 56.5 [S]) · DEIMv2 (DINOv3 backbones, Sept 2025; AP not found) | RF-DETR Apache-2.0 [V] · YOLO26 **AGPL-3.0**, enterprise licence available [S] · D-FINE/DEIM licence UNVERIFIED |
| Tracking and re-ID | **Own 4-identity tracker + torso-colour re-ID** (your old method) · **OC-SORT / ByteTrack** | **Deep-EIoU** (SportsMOT HOTA 77.2, SoccerNet 85.4 [S]) · **SAM2 / SAMURAI** mask tracking with motion-aware memory [S] · McByte++ (2026 preprint, UNVERIFIED) | OC-SORT, ByteTrack MIT [V] · Deep-EIoU, SAMURAI licences UNVERIFIED (SAM 2 terms to check) |
| Pose | **RTMPose (m and x)** · **ViTPose-H** (largest model 80.9 AP test-dev, paper [S]) · **RF-DETR keypoint preview** (71.8 AP [V]) | **Sapiens** (Meta, claims +7.1 AP over DWPose-L on Humans-5K [S]) · YOLO26-pose | RTMPose Apache-2.0 [S] · ViTPose, Sapiens licences UNVERIFIED (Sapiens suspected non-commercial) · YOLO26 AGPL |
| Ball detection and tracking | **WASB fine-tuned on padel** (MIT [V]) · **TrackNetV3 fine-tuned** (MIT [V]) | **TrackNetV4** (code and weights via its project site, UNVERIFIED) · **TOTNet** (occlusion-aware; repo AugustRushG/TOTNet, UNVERIFIED) · **TrackNetV5** (no code found; funded by a sports company) · YOLO high-resolution tiling | MIT for WASB and TrackNetV3 · others UNVERIFIED · YOLO AGPL |
| Hit detection | **Own audio onset model + ball direction change + wrist proximity** | Learned audio CNN trained on labelled hits (padel evidence: F1 92 %, no public code [V]) | Own code |
| Shot classification | **Pose-sequence classifier on the six PadelTracker100 classes** (forehand, backhand, smash, serve, dropshot, other [V]) | Video models (not researched, UNVERIFIED) · wrist-sensor 1D-CNN reaches > 93 % on 13 strokes but uses a sensor, not video [S] | PadelTracker100 CC BY 4.0 [V] |
| Court and 3D | **Manual points + PnP 3D model** (from your `ball_only`) | Learned court keypoints if the camera ever moves | OpenCV Apache-2.0 |
| Commercial / proprietary APIs | None evaluated: no padel-specific API with documented accuracy was found; hosted APIs also cost per frame (~100,000 frames per match) | Possible only as a labelling aid | UNVERIFIED |

## What the research does *not* settle
- No head-to-head comparison of trackers under padel occlusion exists; sports benchmarks are football and basketball.
- No padel paper reports pose-based shot classification beyond six classes, or any for bandeja versus víbora.
- Whether TrackNetV4, TOTNet or TrackNetV5 are actually usable (code, weights) is unconfirmed.
- COCO AP differences of 1–3 points are small next to our real unknowns (far players, glass, the ball). Players are already found in about 100 % of frames; the open question is the far players' pose and ID stability, not detector AP.

## Bake-off protocol (needs the small benchmark first)
Benchmark: the two analysed clips (about 750 frames) with hand labels for ball position, hit frames, and 4 player boxes plus IDs on the occlusion stretches.

| Stage | Metric | Candidates compared at the same resolution |
|---|---|---|
| Players | Recall of the **far** players at IoU 0.5; AP | RF-DETR S/M, YOLO26-L, D-FINE-L |
| Tracking | ID switches, IDF1, share of frames with 4 correct IDs | own tracker (+ re-ID), OC-SORT, Deep-EIoU |
| Pose | Keypoint accuracy on PadelTracker100 keypoints (head excluded) | RTMPose-m/x, ViTPose-H, RF-DETR keypoint |
| Ball | Precision / recall / F1 at 5 px, longest gap, false tracks on glass | WASB (zero-shot), WASB fine-tuned, TrackNetV3 |
| Hits | F1 at ±3 frames | audio only, audio + ball, audio + ball + wrist |
Every row also records speed (frames/s), peak VRAM on the RTX 3050 6 GB, and the licence class. Nothing is declared "best" before these numbers exist.

## Licence matrix (to review before production)
Permissive: RF-DETR (S/M/L), OC-SORT, ByteTrack, RTMPose, WASB, TrackNetV3, OpenCV, PadelTracker100 data (CC BY 4.0).
Copyleft / restricted: Ultralytics YOLO (AGPL-3.0, enterprise licence available), Padex (GPL-3.0), upstream Padel Analytics (licence unclear).
To verify at source: D-FINE/DEIM, Deep-EIoU, SAM 2 / SAMURAI, ViTPose, Sapiens, TrackNetV4/V5, TOTNet.
Any model whose weights were trained on broadcast footage may also carry data-licence limits (see `THIRD_PARTY_NOTICES.md`).
