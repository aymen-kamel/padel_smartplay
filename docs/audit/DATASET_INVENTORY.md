# Dataset inventory

Date: 2026-11-13. Read-only. Videos were inspected **by container header only** (duration, resolution, frame count, audio track); no frame was decoded. No dataset was downloaded. Tags: **MEASURED**, **CODE**, **UNVERIFIED**.
Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.

## 1. Headline

- **There is no labelled dataset** in the project: no ball positions, bounces, hits, rally/point boundaries, scores, court keypoints, player boxes or identities.
- The only labelled data is a shot-type table of 39–40 rows (two inconsistent copies).
- No footage recorded in the product's target setting (fixed camera) is identifiable. The clip used in all experiments looks like the upstream project's sample video (UNVERIFIED).
- User confirmed (2026-11-13): nothing exists outside this folder (no Roboflow, Colab or Drive datasets).

## 2. Source videos (133 `.mp4` in the project, 3.49 GB; 66 distinct file sizes)

| Video | Where | Duration | Resolution | fps | Frames | Audio | Notes |
|---|---|---|---|---|---|---|---|
| **`Video Project 1.mp4`** | `data_candidates/` | **664.8 s (11 min 05 s)** | 1920×1080 | 30.0 | 19,943 | yes (48 kHz) | 1,677,269,349 B. User-supplied highlights compilation (source https://youtu.be/EuKFW3H9qeU); on-screen score increments after points (user statement, not yet viewed). Created 2026-11-13 19:13. Not decoded in Phase 1. |
| `rally.mp4` | `BO/input`, `MAIN/input`, `MAIN/examples/videos`, `padex/`, 30+ job folders | 12.96 s | 1920×1080 | **25.0** | 324 | yes | 7,603,490 B; **47 identical-size copies** (≈357 MB). Used by 19 API jobs and all `BO` experiments. Probably upstream's sample clip (UNVERIFIED) |
| `rally_001.mp4` | `MAIN/input`, `the_best/rallie`, `the_best/rallies/match_1` | 37.2 s | 1920×1080 | 30.0 | 1,116 | yes | 3 copies. Config default input (`MAIN/config.py:11`) |
| `rally2.mp4` / `rally3.mp4` / `rally4.mp4` | `MAIN/input` | 12.9 s / 17.4 s / 30.2 s | 1920×1080 / 898×506 / 1920×1020 | 29.9 / 30 / 29.9 | 385 / 521 / 903 | yes / no / yes | Short clips |
| `highlights.mp4` | `BO/input`, `MAIN/input` | 41.6 s | **1162×640** | 30 | 1,244 | yes | Used by `BO/src/main.py:319` |
| `match3.mp4` | `BO/input`, `MAIN/input` | 14.2 s | 1306×666 | 29.6 | 421 | yes | |
| `padel.mp4` | `BO/input` | 17.3 s | 1880×930 | 29.9 | 518 | yes | |
| `vedio.mp4` / `vedio1.mp4` / `padel_manyball.mp4` / `Adobe Express - padel_match1.mp4` | `BO/input`, `MAIN/input` | 7.2 / 21.2 / 11.6 / 15.6 s | 1920×1080 / 1096×606 / 1230×688 / 1092×622 | ~30 | 214 / 630 / 343 / 465 | yes | Mixed resolutions, likely web/phone downloads |
| `goooooddddddd.mp4` | `BO/input` | 12.9 s | 1920×1080 | 25 | 323 | no | Re-encode of `rally.mp4` (same length), UNVERIFIED |
| `futuristic_rally.mp4` | `BO/input` | 10.8 s | 800×800 | 30 | 324 | no | Probably a render of the 3D simulation (same format as `simulated_rally_3d.mp4`), UNVERIFIED |
| `select_keypoints.mp4` | `MAIN/examples/videos` | 22.9 s | 2560×1526 | 29.9 | 684 | yes | Upstream demo of keypoint clicking |
| `vedio.mp4` (other file) | `the_best/` | 446.4 s | 1920×1080 | 30 | 13,390 | yes | 78 MB; broadcast-style footage ("Premier Padel" mentioned at `paddle_rally_extractor.py:56`, UNVERIFIED) |
| `full_matches_compilation.mp4` / `rally_only_compilation.mp4` | `the_best/` | 346.3 s / 397.1 s | 1920×1080 | 30 | 10,387 / 11,912 | yes | Outputs of the extractor |
| `rallie/rally_002…005.mp4` | `the_best/` | 82.2 / 108.9 / 118.0 / 50.9 s | 1920×1080 | 30 | 2,466 / 3,265 / 3,540 / 1,525 | yes | Not single rallies: a 118 s "rally" must contain several points (inference, UNVERIFIED by viewing). Copies in `rallies/match_1`, `match_2` |
| `rallies/rally_001…007.mp4` | `the_best/` | 4–15 s | 1920×1080 | 30 | 120–450 | yes | `rally_007.mp4` is 262 bytes (corrupt) |

Totals: only **one** video is longer than 8 minutes (`Video Project 1.mp4`; the `the_best` compilations are 5.8–7.4 min). Unique source footage is on the order of 30 minutes (the compilations overlap their sources). No full match.

## 3. Derived videos and job outputs (not datasets)

| Group | Count | Size | Notes |
|---|---|---|---|
| `MAIN/results_json/<uuid>/output.mp4` | 18 | ≈ 450 MB | Annotated renders of `rally.mp4`, 323 frames at 25 fps, no audio |
| `MAIN/results_json/*.json` (top level) | 13 | 1.2 MB each (identical size) | Per-job JSON |
| `MAIN/uploads/` | 42 files | 243 MB | Copies of `rally.mp4` per job |
| `OUTER/storage/` | 11 | 84 MB | 11 copies of `rally.mp4` |
| `MAIN/output/output_<uuid>.mp4` | 11 | 105 MB | Identical size (9,556,042 B) |
| `BO/output/*.mp4`, root and `output/` debug videos | ~20 | ~250 MB | Visual experiment outputs |
| `ball_only.zip` / `padel_analytics_repo.zip` | 2 | 419 / 377 MB | Older snapshots (newest entry 2026-03-28), 60 / 176 entries, contain weights, videos and a `.git` pack. Not extracted. |

## 4. Labelled and tabular data

| File | Rows | Content | Notes |
|---|---|---|---|
| `MAIN/shot_features_labeled.csv` (05-04) | 40 (39 labelled) | 27 columns: ±10-frame wrist-ball proximity series, `rel_height`, positions, `label` | other 23, backhand 7, forehand 5, smash 4, 1 blank |
| `shot_features_labeled.csv` (root, 05-06) | 40 | same schema | other 12, backhand 13, forehand 10, smash 5 — **different distribution** from the MAIN copy |
| `shot_features.csv` (root, MAIN) | – | unlabelled collection output | |
| `MAIN/data_new.csv` | 1,116 | per-frame telemetry (positions in metres, velocities, accelerations for 4 players + ball) for the 37.2 s clip | Positions jitter (C13) |
| `MAIN/results_json/<uuid>/data.csv` | 323 each × 18 | same schema for `rally.mp4` | 18 near-identical files |
| `MAIN/cache/*.json` | 5 | ball (1,116), players (1,116 × 4), pose, shot events, 12 court keypoints | The only per-frame dumps for a clip longer than 13 s |
| `BO/output/output_trajectory.csv` | 1,244 | ball track on `highlights.mp4` with zone/phase labels | Model output, not ground truth |
| `BO/config/calibration.json`, `calibration_3d.json`, `glass_zones.json` | – | 4-point court corners, 3D calibration, glass polygons for one camera | Hand-clicked |
| `the_best/rallies_output.json` | 20 | rally start/end times in mm:ss.mmm with "confidence: HIGH" | Source and producing script unknown; 8 of 20 durations are exactly 6.8 s and some overlap |

## 5. Quality and leakage risks

1. **Shot classifier validation is optimistic.** `evaluate_pipeline.py:100-105` uses *shuffled* stratified k-fold on ±10-frame windows from the same clip(s). Neighbouring hits share players, camera and ball track; there is no match-level or clip-level split. With n = 39 the 71.8 % has a wide uncertainty. (CODE/MEASURED)
2. **Two label files disagree** (see table), and two pickles of different size exist. Which data produced the shipped model: UNVERIFIED.
3. **Possible train/evaluate overlap with upstream weights.** If `rally.mp4` came from the upstream project's training matches, ball and pose models may have seen it. UNVERIFIED.
4. **`the_best` splits matches by an HSV threshold** (`paddle_rally_extractor.py:116-122`), so "match_1 / match_2" are not guaranteed to be distinct matches; duplicates of the same clip live in three folders.
5. **Resolution/camera mix.** Footage ranges from 898×506 to 2560×1526 and from fixed cameras to broadcast cuts; mixing them in one split would hide domain shift.
6. **The highlights video is biased** (see Phase 3 assessment): long spectacular rallies, cuts, replays and a broadcast camera instead of the product's fixed camera.

## 6. What is missing (needed in Phase 3)

| Need | Why |
|---|---|
| Ball position labels (per frame, ≥ several thousand frames) | Evaluate/train ball detection |
| Event labels: floor bounce / wall / racket hit with frame numbers | Evaluate event spotting |
| Point boundaries and point winners | Evaluate rally segmentation and scoring |
| Scoreboard ground truth with timestamps | End-to-end evaluation |
| Court keypoint labels on several cameras | Court model and reprojection error |
| Player boxes + identities across occlusion | Tracking IDF1/HOTA |
| Shot-type labels (hundreds per class, not 39) | Shot classifier |
| **Full-match footage from the target fixed-camera setup**, separated by match for train/test | End-to-end score evaluation without leakage |
Exact quantities and formats are set in Phase 3.

## 7. Space note (no deletion proposed)
Duplicated material totals roughly 1.5 GB (47 copies of `rally.mp4` ≈ 357 MB, job outputs ≈ 450 MB + 243 MB uploads + 84 MB storage + 105 MB outputs, duplicated weights ≈ 450 MB, two root zips 796 MB that largely mirror folders already on disk).

## UNVERIFIED
- Contents of `Video Project 1.mp4` (scoreboard layout, cuts, replays, number of points, camera): not decoded.
- Source and licence of every video, including the YouTube compilation (copyright/terms for a portfolio project).
- Whether any clip is user-recorded.
- Whether `rally.mp4` contains a finished point.
