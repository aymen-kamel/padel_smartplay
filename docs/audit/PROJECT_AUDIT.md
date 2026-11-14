# SmartPlayAI — Project audit (Phase 1)

Date: 2026-11-13. Method: read-only. Nothing was run except small throw-away scripts that parse files already on disk (video headers, JSON caches, CSV). No pipeline, test, model load or download.

**Evidence tags** — every claim carries one:
- **MEASURED**: a number read from an artefact on disk (file named).
- **CODE**: read in source (`file:line`).
- **UNVERIFIED**: needs a run, a look at a video, an answer from the user, or a source outside the folder.

**Path aliases**: `MAIN` = `padel_analytics_repo/padel_analytics_repo/` · `OUTER` = `padel_analytics_repo/` · `BO` = `ball_only/ball_only/`.
**Units**: folder sizes in the folder map are binary MB/GB as Windows reports them; byte counts are exact; sizes computed by script elsewhere in these documents are decimal MB.

---

## 1. Summary

1. The project is three code lines glued together: a fork of the public *Padel Analytics* repo (`MAIN`, 1 local commit, ~25 uncommitted files), a fork of the GPL-3.0 *Padex* library (`padex/`), and the user's own `ball_only/` experiments. Two of the three are third-party code. (CODE, git)
2. **No score has ever been produced.** All 18 completed API jobs and the CLI run end 0–0 or have no score fields. A paper replay of every saved hit/bounce log through the FSM rules also awards 0 points in all 19 logs (see `SCORE_DIAGNOSIS.md`). (MEASURED)
3. **No ground truth exists** for ball positions, bounces, hits, rallies, points or scores. `evaluate_pipeline.py` says so itself (`rapport_evaluation.json`: `"evaluable": false`). (MEASURED)
4. The scoring logic is a single rule (two same-side bounces without a hit in between) fed by a bounce detector that is "local maximum of image-space y". It has no serve, fault, out, wall-first, net, games, sets, side changes. (CODE)
5. Ball tracking is the weakest measured link: on the 37 s clip only **65.9 %** of frames hold a real detection; 380 of 1,116 positions are interpolated, 4 gaps exceed 15 frames, interpolation produces off-screen positions (y = −137). (MEASURED, `MAIN/cache/ball_detections.json`)
6. **Many numbers in the PFE report are unsupported or contradicted** by artefacts on disk (20 matches, 500+ h annotated, P/R/F1/mAP table, 92.1 % event agreement, 100 % score consistency, 2.1 % loss rate…). See section 6 — the user asked for this list explicitly.
7. Real, supported results are small: shot classifier 71.8 % CV on **39** samples (baseline 59.0 %), homography self-residual 0.90 px, job timings, one ball-tracker benchmark CSV. (MEASURED)
8. Everything ever processed is short: the longest clip analysed is 37.2 s. The pipeline loads up to 400 full frames in RAM for background subtraction (~2.5 GB at 1080p) and one job crashed on a 1.87 GiB allocation. (MEASURED/CODE)
9. Hardware found: RTX 3050 **6 GB** (not 4 GB). Design budget stays 4 GB as decided by the user. (MEASURED, `nvidia-smi`)
10. A new ground-truth candidate exists: `data_candidates/Video Project 1.mp4` — 664.8 s, 1920×1080, 30 fps, with audio (header only, not decoded). See `DATASET_INVENTORY.md`.

---

## 2. Folder map

| Path | Size / files | What it is | Origin | Verdict (clean rewrite) |
|---|---|---|---|---|
| `smartplay/` | 7.05 GB / 56,546 | Python 3.11 venv; created at another path (`pyvenv.cfg`). Shared with the unrelated `iit_courtvison` basketball project, which upgraded packages on 2026-11-11/12 (`supervision` 0.27→0.31.6, +`rfdetr`, `transformers`, `trackers`). | tooling | ARCHIVE; build a clean env |
| `MAIN` | 1.63 GB / 5,564 | Main padel pipeline. git origin `aymen-kamel/padel_analytics_repo`, single commit 2026-07-02, newest edits 2026-08-08 uncommitted | **upstream fork** + user edits | ARCHIVE (reference only) |
| `OUTER` | ~240 MB | Older copy: Streamlit `app.py`, 2 older FastAPI servers, React frontend, 11 copies of one clip in `storage/` | upstream fork + user | ARCHIVE |
| `BO` | 625 MB / 105 | Ball tracking, bounce, scoring experiments (Mar–May 2026) | user (AI-assisted per its README) + upstream utils | ARCHIVE; extract ideas |
| `padex/` | 48 MB / 186 | *Padex* by Xingnan Zhu, **GPL-3.0**; git origin `aymen-kamel/Play_Ai`; local edits to `pipeline.py`, `events/__init__.py`, new `events/shot_classifier.py` | **third party (GPL-3.0)** | ARCHIVE; do not copy |
| `the_best/` | 390 MB / 24 | Rally extractor script + compilations | user | REUSE idea only |
| `data_candidates/` | 1.68 GB / 1 | Highlights compilation supplied 2026-11-13 | user-supplied | KEEP (ground-truth candidate) |
| `iit_courtvison/` | 0.1 MB / 7 | Basketball project notes | out of scope | IGNORE |
| `web_site/` | 79 MB (4,315 files are `node_modules`) | React landing page | user | ARCHIVE |
| `pfe_latex/`, `rapport/` | 0.7 MB | PFE report sources (French) | user | REWRITE after claims cleanup |
| root files | ~900 MB | `ball_only.zip` 419 MB and `padel_analytics_repo.zip` 377 MB (older snapshots, 2026-03-28/29), helper scripts, 2 debug videos, shot CSVs, `weights/` (2 pickles), notes | mixed | ARCHIVE |

Large binaries across the project: 132 `.mp4` files (1.7 GB), 16 `.pt` files (~1 GB), 2 root zips (796 MB). No `.ipynb`, `data.yaml`, `results.csv`, `runs/` or label files outside the venv. (MEASURED)

---

## 3. Lineage and authorship

| Line | Evidence | Licence status |
|---|---|---|
| *Padel Analytics* (upstream, jsilvawasd) → `MAIN`/`OUTER` | README text is the upstream README (mentions "13 degrees of freedom", "models were trained", collaborator e-mail); files untouched since 2026-03-27 (`trackers/tracker.py`, `ball_tracker/{dataset,iterable,models*}.py`, `keypoints_tracker/`, `utils/`, `constants/court_dimensions.py`, `visualizations/`) are presumed upstream | **UNVERIFIED.** No LICENSE file locally. An earlier note in `iit_courtvison/RESEARCH_REPORT.md:58` says CC BY-NC-SA 4.0 (secondary). Confirm at source in Phase 2. If true: non-commercial + share-alike → cannot be relicensed under the user's own licence. |
| *Padex* (Xingnan Zhu / rondo-labs) → `padex/` | `padex/LICENSE` is GPL-3.0; `pyproject.toml:12` GPLv3 classifier; file headers carry the original author | **GPL-3.0 (CODE).** Any copied code makes the derivative GPL-3.0. |
| User's own → `BO`, `the_best`, edits in `MAIN` | `BO/README.md` ends "Created by Antigravity" (an AI coding assistant) | Owned by the user; **but see C09: `BO/src/bounce_classifier.py` mirrors Padex's GPL code** |

Authorship of individual files inside `MAIN` is inferred from modification times and `git status`; the single local commit contains the whole fork, so no upstream diff exists locally. (UNVERIFIED at function level)

---

## 4. Component map

Columns: Implementation · Model / data / training · Measured · Limits · Verdict. "Replace" means: pick a better method in Phase 2; the old one stays as a baseline for comparison only.

### C01 Video ingestion / memory model
- Impl: `supervision` frame generator (`MAIN/trackers/runner.py:129-134`). `TrackingRunner.run` runs **each tracker over the whole video**, then renders (`:345-398`). TrackNet background median stacks up to `BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM` = 400 full frames in memory (`trackers/ball_tracker/iterable.py:60-83`, `config.py:50`) ≈ 400 × 6.2 MB = 2.5 GB at 1080p.
- Measured: job `f1ee65ee` failed with `Unable to allocate 1.87 GiB for … (323, 1080, 1920, 3)` — exactly 323 frames of 1080p (`MAIN/results_json/f1ee65ee…/job_meta.json`). Longest clip processed: 1,116 frames / 37.2 s.
- Limits: cannot scale to a match; background from the first N frames only.
- Verdict: **REDESIGN** (streaming, bounded windows).

### C02 Court calibration / homography
- Impl: 12 manual clicks (`manual_keypoints_selection.py`) → `cv2.findHomography` default least-squares, no RANSAC (`MAIN/analytics/projected_court.py:387`); fixed once per video (`:553-556`). The YOLO court model (`weights/court_keypoints_detection.pt`) is bypassed because `DYNAMIC_COURT_KEYPOINTS = False` (`config.py:26`). Court canvas is `0.14 × video width` px wide, so metres go through integer canvas pixels, ~4.4 cm per px (`projected_court.py:221-230`, `:456`).
- Measured: self-residual 0.90 px mean / 1.71 px max on the 12 clicks = 3.95 / 7.49 cm **in canvas space** (`MAIN/rapport_evaluation.json`). This measures click consistency only (the script says so, `evaluate_pipeline.py:201-203`). Accuracy of real positions: UNVERIFIED.
- Limits: static camera only; ground-plane homography is applied to the ball pixel (`:441-458`), so any airborne ball is mis-projected (parallax); `BO/src/court_homography.py` has a 3D `solvePnP` model that was never wired into `MAIN`.
- Verdict: **REDESIGN** (automatic court keypoints + reprojection-error check + height-aware ball model).

### C03 Player detection and tracking
- Impl: YOLO person (`weights/yolo26m.pt`, `config.py:33`, conf 0.5, imgsz 1280) → custom 4-slot manager (`MAIN/trackers/players_tracker/players_tracker.py:445-601`): waits for a frame with exactly 4 detections, assigns IDs 1–4 sorted by box top (1–2 = far side), Hungarian matching on distance + mean torso colour (EMA 0.1), holds a missing player up to 90 frames.
- Measured (37 s clip cache): exactly 4 players in 1,116/1,116 frames, but **341 of 4,464 boxes (7.6 %)** are held placeholders with confidence 0.1 (`MAIN/cache/players_detections.json`). They are passed downstream as real players.
- Limits: IDs are tied to court *side at frame 0*, not to team identity → a change of ends (every odd game in padel; rule check in Phase 2) breaks any per-team score. If no frame ever has exactly 4 detections, no players are produced. (CODE)
- Verdict: **REDESIGN**.

### C04 Pose estimation
- Impl: `weights/player_keypoints_detection.pt` (upstream weights, 17 COCO keypoints, imgsz 1280). Pose→player assignment compares the **nose to the top-left corner of the player box** (`MAIN/trackers/runner.py:212-214`).
- Measured: 4 persons detected in only 595/1,116 frames (53 %); 0 persons in 50 frames (`cache/players_keypoints_detections.json`).
- Verdict: **REPLACE** (Phase 2 candidates: YOLO-Pose, RTMPose).

### C05 SAM2 segmentation
- Impl: `SAM2PlayerTracker` prompts `sam2_s.pt` with YOLO boxes every frame (`trackers/players_tracker/sam2_tracker.py`); **on by default** (`config.py:62`). Used only to refine "feet" for the homography (`Player.feet`, `players_tracker.py:76-95`).
- Limits: heavy (184 MB weights, per-frame prompt); contributes nothing to scoring. Comment says `sam2_hiera_small.pt` but the file is `sam2_s.pt`.
- Verdict: **ARCHIVE** (not needed for score).

### C06 Ball detection — TrackNet
- Impl: TrackNet, seq_len 8, 288×512, `bg_mode=concat`, `ensemble weight` (`MAIN/trackers/ball_tracker/ball_tracker.py:322-405, 571-754`); InpaintNet disabled (`config.py:48`). Structure (dataset, iterable, `predict_modified`, `generate_inpaint_mask`) resembles the public TrackNetV3 reference code — **UNVERIFIED** origin and licence.
- Model: `weights/ball_detection.pt` (136,190,877 B) — upstream weights (README:37-38 points to the author's Drive). Training data/method: not on disk.
- Measured: `BO/output/benchmark_results.csv` (clip `rally.mp4`, 324 frames): TrackNet 4.15 FPS, 65.4 % frames with an output; +Inpaint 3.95 FPS, 77.5 %. "Detection rate" counts frames with any output, not correctness (`BO/src/benchmark_all.py:55,80`).
- Verdict: **REPLACE** (baseline only).

### C07 Ball detection — YOLO
- Impl: `BO/src/yolomain.py`, `run_rally_score_yolo.py`, `simple_score.py` with `BO/models/best (1).pt` (6.2 MB, "fine-tuned via Roboflow" per `BO/docs/RAPPORT_TRACKING_PADEL.md:13`; no training artefacts on disk).
- Measured: 61.9 FPS, **30.0 %** frames detected on `rally.mp4` (CSV).
- Verdict: **ARCHIVE** (needs far more data; baseline only).

### C08 Ball post-processing
- Impl: `filter_reflections` (`ball_tracker.py:501-569`): fixed image-fraction "glass margins" (L 18 %, R 12 %, T 15 %); rolling-median gate 70 px; velocity gate 35 px/frame; **quadratic interpolation, `limit_direction='both'`, no gap limit**; median smoothing. Same idea re-implemented in `BO/src/{main,yolomain,run_rally_score_yolo,run_tracknet}.py` (four `robust_interpolate`).
- Defects (CODE): missed detections are stored as (0,0) (`predict.py:202`) and enter the rolling median (`ball_tracker.py:523-526`), so a window with ≥3 misses has median ≈ (0,0) and real detections can be rejected; thresholds are in pixels, not scaled to resolution or fps (35 px/frame at 25 or 30 fps, 1080p only); report says Δd > 100 px (`pfe_latex/chapitres/03_architecture.tex:127`) but code uses 35 and 70.
- Measured (37 s clip): 736 real / 380 interpolated frames; 49 gaps, longest 61 frames (2 s); 4 gaps > 15 frames cover 135 frames; 52 frames with NaN position (start and tail), 14 at (0,0); y range −137…937 (off-screen). (`cache/ball_detections.json`)
- Limits: unlimited interpolation invents positions across 2 s gaps, which then feed bounce and hit logic. `BO/src/kalman_tracker.py` (coasting limit 20 frames) is imported in `BO/src/run_rally_score.py:9` but never called.
- Verdict: **REDESIGN**.

### C09 Bounce detection
- Impl (4+ copies): `MAIN/trackers/ball_tracker/ball_tracker.py:17-65` (local max of image y, ±3 frames, prominence 4 px, min gap 0.45 s); `BO/src/bounce_detection.py:13` (±1 frame, prominence 15 px); `BO/src/simple_score.py:15`; `BO/src/bounce_classifier.py` (velocity reversal, disappearance, hybrid, MLP, court-geometry surface classifier).
- Defect (CODE): "image-y local maximum" fires on floor bounces, racket hits, wall rebounds and tracking jitter alike; no height, no audio, no player-proximity gate; frames in tracking gaps are filled with the last y (`:34-35`).
- **Licence flag**: `BO/src/bounce_classifier.py:274-334` `CourtGeometrySurfaceClassifier` has the same class and enum names, same coordinate system (x 0–10, y 0–20, net 10), same constants (0.8 / 1.0 / 0.5 / 1.0 m) and same branch order as `padex/src/padex/events/bounce.py:166-214` (GPL-3.0). Derivation is probable, **not proven** (UNVERIFIED). Do not carry this file into the rewrite.
- Measured: 19 bounce flags on the 37 s clip (17 after court filter); 5–7 on `rally.mp4` depending on the run (settings changed between runs).
- Verdict: **REPLACE**.

### C10 Hit (shot) detection
- Impl: `MAIN/analytics/shot_counter.py:70-141`: velocity direction/speed change over a 10-frame window, ball-to-player-centre or ball-to-wrist distance with hard-coded pixel thresholds (150 / 180 / 200 / 250, and `y < 450` at `:116`), per-player cooldown 25 frames, global 12.
- Defects (CODE): thresholds are 1080p pixels; the event is reported at the end of the window, ~5 frames after the mid-window frame it analysed; players come from bbox centres including held placeholders (C03); `confidence=1.0` and `ball_speed=0.0` are hard-coded in the report (`analytics/scoring.py:103-105`).
- Measured: 14 hits per 12.96 s on `rally.mp4`, 41 hits per 37.2 s on `rally_001.mp4`. Accuracy: **UNVERIFIED** (no ground truth).
- Verdict: **REPLACE**.

### C11 Shot-type classification
- Impl: rule engine (`analytics/shot_classifier/rules.py`) used in production; RandomForest (`train_shot_classifier.py`, `weights/shot_classifier.pkl`) and LSTM (`train_lstm_classifier.py`) exist but are not used at inference (see TECHNICAL_DEBT TD-02).
- Measured: RF, 4 classes, **n = 39**, 4-fold CV 71.8 % vs majority baseline 59.0 %; smash has 4 samples (`rapport_evaluation.json`). Shot types logged for rally_001: BANDEJA 17, SMASH 10, LOB 6, … (rule output, unverified). LSTM weights (`weights/lstm_shot_classifier.pth`, `lstm_scaler.pkl`) are **not on disk**.
- Verdict: **REDESIGN**, needs a labelled dataset first.

### C12 Scoring / rally logic (three versions)
- `MAIN/analytics/scoring.py:154-190` `PadelScorer`: one rule; `score_labels` saturate at "Game" (`:204-207`); no games/sets.
- `BO/src/score.py:84-209` `ScoreDetector` (3-state FSM, time/distance filters) + `PadelScore:9-70` (points, games, sets, deuce; no golden-point, tie-break or side-change logic); court coordinates normalised `[0,1]²` with `cy<0.5` = BOTTOM (`:123`) — **opposite sign and different units** from `MAIN` (metres, `y<0` = top).
- `padex/src/padex/events/point.py` (GPL-3.0): pause-based rally boundaries (gap ≥ 3 s), last shot's team = winner.
- Measured: see `SCORE_DIAGNOSIS.md`.
- Verdict: **REDESIGN** around explicit rules and uncertainty.

### C13 Analytics / reports
- `MAIN/analytics/match_report.py` (KPI cards, heatmaps, 35 unit tests), `data_analytics.py` (per-frame telemetry).
- Measured problem: `match_stats_table.tex` reports player 1 at 175.6 m and 6.7 km/h mean for a 37.2 s clip. The raw path length of the saved telemetry is exactly 175.6 m, i.e. **17.0 km/h** mean; 114 steps exceed 0.3 m/frame (> 9 m/s); a 5-frame moving average gives 107 m (10.4 km/h). Distance is inflated by position jitter and the distance and speed figures come from different computations. (MEASURED, `MAIN/data_new.csv`)
- Verdict: **REDESIGN** the metric layer (smoothing, plausibility caps); KPI formulas may be reused.

### C14 API and user interfaces
- 3 FastAPI servers (`OUTER/api_server.py`, `OUTER/api/main.py`, `MAIN/api_server.py` 428 lines), 2 Streamlit apps (`OUTER/app.py` imports a module `estimate_velocity` that does not exist; `MAIN/dashboard.py`), 3 React apps (`MAIN/padel-ui` 1,808 lines, `OUTER/frontend`, `web_site`), 2 `docker-compose.yml` with different build contexts. Not reviewed line by line (UNVERIFIED quality). `MAIN/api_server.py:42` allows ngrok origins by regex.
- Measured: 18 completed + 1 failed jobs, all on one clip.
- Verdict: keep the **job-API shape** (`/jobs`, `/jobs/{id}/results`) as a design reference; ARCHIVE the rest.

### C15 Rally extraction from broadcast video
- `the_best/paddle_rally_extractor.py`: "rally" = shots where blue pixels > 15 % and > 5 Hough lines (`:53-72`); new match when mean HSV jumps by > 40 (`:116-122`). `rallies_output.json` has another schema (8 of 20 entries are exactly 6.8 s, some overlap) and no script producing it exists. (CODE/MEASURED)
- Verdict: **REUSE the idea** (broadcast wide-shot filter) for pre-segmenting highlight videos.

### C16 Padex library
- 7,843 lines of source, 3,299 of tests. Schema-first design (Pydantic), pluggable strategies, TrackNetV3, SAHI-YOLO, Kalman ball tracker, MLP bounce detector, pose-based shot classifier. Local edits add an ML shot classifier. Weights are **not on disk** (`padex/assets/weights/` empty; `weights.py:25-32` downloads from the author's GitHub release).
- Measured: `padex/output/M_001_analytics.json`: one point (rally length 12); for player `P_011`, 6 of 7 shots are typed `unknown`.
- Verdict: **ARCHIVE**; study design ideas only (GPL-3.0).

### C17 Evaluation harness
- `MAIN/evaluate_pipeline.py` — computes CV accuracy against a majority baseline, per-class P/R, homography residual, and refuses to report hit precision/recall without ground truth. Honest and well structured.
- Verdict: **REUSE the approach**.

### C18 PFE report
- Two document trees (`main.tex` with template chapters, `main_smartplay.tex` with real chapters) and `_new` duplicates; contains the unsupported claims below.
- Verdict: **REWRITE** after the claims cleanup.

### C19 Environment and operations
- Baseline freeze (2026-11-11): torch 2.7.1+cu118, ultralytics 8.4.13, numpy 2.3.5, both `opencv-python` and `opencv-contrib-python` installed. Never imported by project code: `mediapipe`, `pyvista`, `vtk`, `trimesh`, `lap`, `pyarrow`, `kaleido`, `pims`, `sounddevice`, `shapely`, `seaborn`, `networkx`, `fire` (installed but unused). Only `padex` itself uses `sahi`, `polars`; only the 4 root scripts import `padex` — `MAIN` does not. (MEASURED, import scan)
- `MAIN/.gitignore` excludes `*.json`, `*.csv`, `*.mp4`, `weights/`, `models/`; one commit in total, so six months of work have no history. (CODE, git)

---

## 5. Supported results (what can be quoted)

| ID | Result | Source |
|---|---|---|
| S1 | 18 completed jobs on one 324-frame clip took 4.2–7.3 min (11 jobs) or 18–38 min (7 jobs) → 0.14–1.3 FPS end to end | `MAIN/results_json/*/job_meta.json` |
| S2 | Ball-tracker benchmark on `rally.mp4`: YOLO 30.0 % / 61.9 FPS; TrackNet 65.4 % / 4.15 FPS; +Inpaint 77.5 % / 3.95 FPS (frames with output, not accuracy) | `BO/output/benchmark_results.csv` |
| S3 | Shot classifier RF: n = 39, 4-fold CV 71.8 %, baseline 59.0 %, forehand recall 0.40, smash n = 4 | `MAIN/rapport_evaluation.json` |
| S4 | Homography click self-residual 0.90 px mean (3.95 cm in canvas) | same |
| S5 | 37 s clip: ball 65.9 % real detections; 4/4 players in every frame (7.6 % placeholders); pose 4 persons in 53 % of frames; 41 hits / 17 bounces | `MAIN/cache/*.json` |
| S6 | 83 unit-test functions exist for the FSM, shot counter, report, court bounds, conversions (an earlier record in `iit_courtvison/environment/PHASE1_RECORD.md` says 110 tests passed; **not re-run**). They test logic on synthetic events, not accuracy. | `MAIN/tests/` |

---

## 6. Unsupported claims

Every number below appears in the PFE sources with **no supporting artefact on disk**, or is contradicted by one. Verify with real data or remove before submission.

| ID | Claim (as written) | Where | What is on disk | Status |
|---|---|---|---|---|
| U01 | "20 matchs de padel complets (1–2 heures chacun)" | `pfe_latex/chapitres/04_implementation.tex:105` | No full match. Longest source clip ≈ 7.4 min (`the_best/vedio.mp4`); highlights video 11.1 min added 2026-11-13 | UNSUPPORTED |
| U02 | "500+ heures de vidéo annotée manuellement" | `:106` | No annotation files at all; one labelled CSV with 39–40 rows. Also arithmetically inconsistent with U01 (20 × 2 h = 40 h max) | UNSUPPORTED |
| U03 | Videos in "720p, 1080p, 4K" | `:107` | Max resolution on disk 2560×1526 (upstream example); no 4K | UNSUPPORTED |
| U04 | P/R/F1/mAP table: players 94.5/92.3/93.4/0.867, ball 88.2/85.6/86.9/0.801, hits 91.2/89.1/90.1/0.843, court 97.8/96.5/97.1/0.914 | `:122-133` | No validation set, no `results.csv`, no mAP output. "Court" has no trained model in use (manual clicks) | UNSUPPORTED |
| U05 | "validées manuellement sur 10 matchs": distance ±2–3 %, speed ±5 % | `:223-227` | No position ground truth; telemetry shows 114 impossible steps in 37 s (C13) | UNSUPPORTED, partly contradicted |
| U06 | "concordance de 92.1 % … avec les annotations manuelles" | `:228`; `06_conclusion_generale.tex:51` | `rapport_evaluation.json`: `"evaluable": false`; `evaluate_pipeline.py:248-250` prints "Pas de vérité terrain" | **CONTRADICTED** |
| U07 | "Consistance des scores : 100 %" | `04_implementation.tex:229` | Every run ends 0–0; no scoreboard ground truth | **CONTRADICTED / UNSUPPORTED** |
| U08 | "latence < 150 ms, précision > 88 %" | `06_conclusion_generale.tex:131` | Fastest end-to-end job: ≈ 0.78 s/frame (S1); TrackNet alone 0.24 s/frame (S2) | **CONTRADICTED** |
| U09 | CPU latency table (e.g. 2,461.9 ms/frame at 720p) "issus de benchmarks réels" on a machine without GPU | `04_implementation.tex:135-163, 219` | `benchmark_pipeline.py:186` writes `benchmark_results.json`; **no such file exists**. The machine has an RTX 3050. How 720p/4K rows were produced: UNVERIFIED | UNSUPPORTED |
| U10 | GPU speed: "~18 FPS" (RTX 3060 estimate) vs "108 ms ≈ 9.3 FPS" vs "28 fps" | `04_implementation.tex:212-217`; `03_architecture.tex:278`; `03_architecture_new.tex:222` | Three mutually inconsistent figures; measured ≤ 1.3 FPS end to end | MISMATCH |
| U11 | TrackNet raw PCK@5px 86.5 % / RMSE 4.82 px → with filters 96.2 % / 1.78 px ("+9.7 %") | `03_architecture.tex:139-142`; `rapport/Chapitre2_SmartPlayAI_Complet (1).tex:406-413` | No labelled ball positions exist | UNSUPPORTED |
| U12 | "MAE 4.8 px", "taux de déperdition 2.1 %", "28 fps", "excellent pour structures de 40 mm" | `03_architecture_new.tex:218-222` | Measured loss: 34.1 % of frames without a real detection (37 s clip), 34.6 % on `rally.mp4` with TrackNet alone (S2). 40 mm is not a padel-ball size (6.35–6.77 cm per the usual FIP spec — UNVERIFIED until Phase 2) | **CONTRADICTED** |
| U13 | TrackNet V1–V5 precision/recall/F1 table (99.7/97.3/98.5 … 99.2/98.0/98.6) | `rapport/Chapitre2_SmartPlayAI_Complet (1).tex:419-435` | Not produced by this project; no citation in the table | UNVERIFIED provenance — cite or remove |
| U14 | Accuracy 92/94/96 %, "Baseline / Proposé v1 / v2" chart | `pfe_latex/chapitres/04_resultats.tex` (whole file) | A template with placeholder text ("Décrivez vos résultats obtenus"). It is included by `main.tex`; `pfe_latex/main.pdf` (2026-05-21) was built from it | PLACEHOLDER |
| U15 | `ball_only` report: TrackNet 85.18 %, TrackNet+Inpaint 96.91 %, FPS 63.66/4.16/3.95 | `BO/docs/RAPPORT_TRACKING_PADEL.md:37-41` | Saved CSV from the same script: 65.43 % and 77.47 %, FPS 61.88/4.15/3.95. Only YOLO's 30.03 % matches. The metric is "frames with output" | **MISMATCH** |
| U16 | Frame skipping "−45 % GPU load"; velocity gate Δd > 100 px; "YOLOv8" detector; quantisation, pruning, MLOps | `03_architecture.tex:284, 127`; `04_implementation.tex:83-86` | No stride in code (`runner.py:63`); gate is 35 px (`ball_tracker.py:401`); detector is YOLO26m (`config.py:33`); no quantisation/pruning code | NOT IMPLEMENTED / MISMATCH |
| U17 | "at least 8 GB of VRAM" | `MAIN/README.md:80`; `04_implementation.tex:296` | 18 jobs completed on a 6 GB card (CUDA use per job: UNVERIFIED) | MISMATCH |
| U18 | Per-player table: 175.6 m, 6.7 km/h mean, 21.8 km/h max, 5.9 σ | `MAIN/match_stats_table.tex` | 175.6 m equals the raw jittery path (17.0 km/h mean); see C13 | **INTERNALLY INCONSISTENT** |
| U19 | Upstream claims inherited via README: pose with "13 degrees of freedom", pose classification, "ball hits prediction" trained models | `MAIN/README.md:12-16` | Upstream author's text; nothing in the user's folder shows these models being trained | NOT THE USER'S WORK — do not attribute |

---

## 7. UNVERIFIED register (consolidated)

1. Licence of upstream *Padel Analytics* (secondary note says CC BY-NC-SA 4.0) and licences of every weight file (Phase 2).
2. Whether `BO/src/bounce_classifier.py` was derived from Padex (similarity observed, derivation unproven).
3. Origin and training data of `ball_detection.pt`, `model_best.pt`, `yolo5_last.pt`, `best (1).pt`, `player_keypoints_detection.pt`, `court_keypoints_detection.pt`.
4. Whether the clips analysed contain a complete point (so whether 0–0 is wrong for them).
5. Accuracy of hits, bounces, positions, shot types — no ground truth.
6. How the CPU latency table (U09) was produced; whether the 18 jobs ran on CUDA.
7. Behaviour of pandas quadratic interpolation at clip ends (negative y suggests extrapolation).
8. Quality and security of the three FastAPI servers and React apps (not read line by line).
9. Contents of the two root zips beyond their file lists (older snapshots dated 2026-03-28/29).
10. The 110-test pass reported in `iit_courtvison/environment/PHASE1_RECORD.md` (not re-run).
