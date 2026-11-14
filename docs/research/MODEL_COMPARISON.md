# Model comparison (Phase 2)

Date: 2026-11-14. Labels: **[V]** primary source opened · **[S]** secondary · **MEASURED** = measured on this machine (by the audit or by the sibling basketball project's load tests in `iit_courtvison/environment/PHASE1_RECORD.md`, which ran in the same venv on the same GPU). Reported accuracy = authors' numbers on their datasets; **not comparable across rows**.

Scoring of each column: **Licence fit** for an own-licence release (✅ permissive · ⚠️ unclear · ❌ copyleft/NC/none) · **4 GB** (✅ fits by measurement or small size · ⚠️ to measure · ❌ unlikely) · **Effort** (L/M/H to integrate).

## 1. Ball detection / tracking

| Option | Licence fit | Weights | Reported | 4 GB | Effort | Vs current |
|---|---|---|---|---|---|---|
| Current: upstream TrackNet (`ball_detection.pt`) + unlimited interpolation | ❌ upstream (NC per secondary note) | upstream | 65.4 % frames with output on `rally.mp4` (MEASURED), accuracy unknown | ✅ runs | — | baseline |
| **TrackNetV3** (qaz812345) | ✅ MIT **[V]** | badminton **[V]** | F1 98.56 % shuttlecock **[V]** | ✅ likely (288×512) — measure | M (retrain on padel) | Same family, adds trajectory rectification; legal to reuse |
| **WASB** (nttcom) | ✅ MIT **[V]** | tennis among 5 sports **[S]** | SOTA on 5 datasets (paper) **[S]** | ✅ 436 MB reserved for basketball weights (MEASURED, sibling project) | M (Hydra configs, py3.8-era code) | HRNet, temporal consistency; zero-shot tennis test possible |
| TrackNetV4 add-on | ⚠️ | ⚠️ | gains on V2/V3 (authors); mixed reproduction **[S]** | ✅ | L–M | Optional upgrade after a baseline exists |
| TrackNetV5 | ⚠️ | ⚠️ | F1 0.9859 (authors) **[S]** | ⚠️ transformer head | H | Watch list |
| TOTNet | ⚠️ | ⚠️ | occlusion RMSE gains (authors) **[S]** | ⚠️ 3D conv | H | Watch list (glass/player occlusion) |
| YOLO ball (`best (1).pt`) | ❌ AGPL **[S]** | own, no data | 30.0 % frames (MEASURED) | ✅ | — | Drop |

## 2. Hit / bounce events

| Option | Licence fit | Reported | 4 GB | Effort | Vs current |
|---|---|---|---|---|---|
| Current: image-y peak + velocity/proximity rules | own/upstream | never measured; 0 points scored | ✅ | — | baseline |
| **Audio onset/energy baseline** | ✅ own code (librosa ISC/NumPy) — licence of librosa UNVERIFIED | — | ✅ CPU | L | New signal, independent of ball visibility |
| **Learned audio hit detector** (re-implementation of Decorte et al. idea) | ✅ own code | padel F1 92 % (authors) **[V abstract]** | ✅ small CNN on log-mel (estimate) | M | Needs ~hundreds of labelled hits |
| **Trajectory bounce classifier** (HistGradientBoosting on trajectory features; method as in kjfk/tennis-bounce-detector) | ✅ MIT **[V]** / own re-implementation | tennis F1 0.915 **[V]** | ✅ CPU | L–M | Replaces peak rule; add 3D height features |
| E2E-Spot (video) | ⚠️ UNVERIFIED | tennis mAP ≈ 96.9 **[S]** | ⚠️ training likely > 4 GB at paper settings (UNVERIFIED) → **possible cloud GPU** | H | Later |
| UMEG-Net (pose+ball graph) | ⚠️ | — | ⚠️ | H | Later |

## 3. Point segmentation and winner / score

| Option | Licence fit | Evidence | Effort | Vs current |
|---|---|---|---|---|
| Current single rule FSM | own | 0 points in 19 logs (MEASURED) | — | baseline |
| Pause/silence + serve detection segmentation | ✅ own | FIP: ≤ 20 s between points **[V]** | L–M | New |
| Rule-based point-winner from events (last valid hitter, floor-bounce count per side, ball out) | ✅ own | FIP Rule 13 **[V]** | M | Extends current rule to the main point-ending patterns |
| **Constrained score decoding (HMM/Viterbi over FIP score automaton, anchored by side changes and server rotation)** | ✅ own | Rules 1, 5, 6 **[V]**; method standard; padel use not found | M | Turns noisy per-point guesses into a consistent score |
| Scoreboard OCR (broadcast only) | depends on OCR lib | SmartTennisTV **[S]** | M | Only for labelling the highlights video, never in the product |

## 4. Court calibration

| Option | Licence fit | Reported | Effort | Vs current |
|---|---|---|---|---|
| Current: 12 clicks, least-squares homography, canvas-pixel metres | upstream/own | 0.90 px self-residual (MEASURED) | — | baseline |
| **Manual calibration + RANSAC + solvePnP 3D model + reprojection report** | ✅ OpenCV Apache-2.0 | — | L | Same effort for the user, adds height and an error figure |
| Learned court keypoints (TennisCourtDetector-style) | ❌ no licence on that repo **[V]** → own re-implementation only | tennis precision 0.963 **[V]** | H | Not needed for one fixed camera |

## 5. Players

| Option | Licence fit | Reported (COCO / MOT17) | 4 GB | Effort | Vs current |
|---|---|---|---|---|---|
| Current: YOLO26m + custom 4-slot manager | ❌ AGPL **[S]** | 7.6 % placeholder boxes (MEASURED) | ✅ | — | baseline |
| **RF-DETR Small** | ✅ Apache-2.0 code+weights **[V]** | AP 53.0 **[V]** | ✅ 256 MB reserved inference (MEASURED, sibling) | L–M | Permissive, accurate |
| RF-DETR Nano | ✅ **[V]** | AP 48.4 **[V]** | ✅ | L–M | Faster fallback |
| RT-DETRv2-S | ✅ Apache-2.0 **[V]** | AP 48.1 **[V]** | ✅ likely | M | Alternative |
| **OC-SORT** + 4-identity / 2-per-side constraint | ✅ MIT **[V]** | HOTA 63.2 **[V]** | ✅ CPU | L (via `trackers` lib, Apache **[S]**) | Replaces hand-made tracker |
| ByteTrack | ✅ MIT **[V]** | HOTA 63.1 **[V]** | ✅ | L | Equivalent baseline |
| RTMPose-s | ✅ Apache-2.0 **[S]** | AP 72.2 **[S]** | ✅ 94 MB for 6 crops (MEASURED, sibling) | M | Only needed for shots/serve |

## 6. Data

| Source | Licence | Use |
|---|---|---|
| **PadelTracker100** | CC BY 4.0 **[V]** (annotations; frames via YouTube — terms UNVERIFIED) | Ball/player/pose/shot training and a public benchmark |
| Decorte padel set | UNVERIFIED availability | Audio hits, rally boundaries |
| Highlights video (yours, from YouTube) | YouTube terms UNVERIFIED | Point labels from the scoreboard; broadcast domain |
| **Own club recordings** | yours | Held-out end-to-end score test (and fine-tuning on other matches) |

## Items that may need a cloud GPU (flagged)
- Training a video event-spotting network (E2E-Spot class) at published settings — UNVERIFIED, likely above 4 GB.
- Fine-tuning RF-DETR at full resolution with useful batch sizes — UNVERIFIED; try small batch + gradient accumulation first.
Everything else in the recommended path is expected to train or run locally; every VRAM figure must be **measured** in Phase 4 before it is relied on.
