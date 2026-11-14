# Technical debt

Date: 2026-11-13. Read-only. Tags: **MEASURED**, **CODE** (`file:line`), **UNVERIFIED**.
Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `OUTER` = `padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.
Severity: **H** blocks correct scoring or reproducibility · **M** causes wrong or fragile behaviour · **L** hygiene. Since the plan is a clean rewrite, this list tells the new design what to avoid; it is not a fix list.

## A. Correctness bugs and broken paths

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| TD-01 | H | The trained shot classifier is never used at inference. `runner.py` reads `self.scorer.clf` and `self.scorer.feature_cols`, which `PadelScorer` never defines; the line is unreachable only because the second pose-to-player map looks up a `neck` keypoint that the 17-name COCO list does not contain, so features are `None` | `trackers/runner.py:303-306`, `:270`; `trackers/players_keypoints_tracker/players_keypoints_tracker.py:65-83`; `analytics/scoring.py:8` (unused `joblib`) |
| TD-02 | M | Pose is matched to players by the distance from the **nose to the top-left corner of the box**, not to the box centre or torso; a second map for features uses `neck` and is empty | `runner.py:212-214`, `:264-276` (map built twice with different rules) |
| TD-03 | M | Running with `collect_data=False` crashes: `persistent_bounces` is only created inside the `collect_data` branch but used unconditionally, and `self.data_analytics.frames[:-1]` runs even when it is `None` | `runner.py:93-100`, `:183`, `:329` |
| TD-04 | M | A frame with no ball position is turned into pixel (0,0) and then **projected to a court position**. Saved telemetry holds a constant ball position (−20.64 m, −48.75 m) — outside the court by a factor of five — on the last frames, consistent with this path (the causal link is inferred, not traced) | `ball_tracker.py:243-246` (`asint` returns (0,0) for NaN); `projected_court.py:451-456`; `MAIN/data_new.csv` rows 1114–1115 (MEASURED) |
| TD-05 | M | Missed detections are stored as (0,0) and enter the rolling median that gates real detections | `trackers/ball_tracker/predict.py:202`; `ball_tracker.py:523-526` |
| TD-06 | M | Unlimited quadratic interpolation (`limit_direction='both'`): 380 of 1,116 positions invented, gaps up to 61 frames, positions at y = −137 | `ball_tracker.py:545-558`; `MAIN/cache/ball_detections.json` (MEASURED) |
| TD-07 | M | Event timestamps assume 30 fps (`frame / 30`); `rally.mp4` is 25 fps, so times are 17 % short. `confidence` always 1.0, `ball_speed_px` always 0.0 | `analytics/shot_classifier/output.py:15,26`; `analytics/scoring.py:103-105` |
| TD-08 | M | Player slots: IDs set on the first frame with exactly 4 detections, by box top (far side = 1–2); if that frame never occurs no player exists; held "placeholder" boxes (conf 0.1, 7.6 % of boxes) are passed on as real players | `players_tracker.py:445-470`, `:562-579` |
| TD-09 | M | Team identity is tied to court side at frame 0 → breaks at changes of ends | `players_tracker.py:450-455`; `scoring.py:109-115` |
| TD-10 | M | `Ball.trajectory` is a **class-level** deque shared by every `Ball` in the process; the API runs jobs in one process, so trails leak between jobs | `ball_tracker.py:248` |
| TD-11 | L | Score labels saturate at "Game"; no deuce/golden point/games/sets in `MAIN` | `analytics/scoring.py:204-207` |
| TD-12 | L | Debug `print` inside loops (`print(seq_len)`, per-hit prints) | `ball_tracker.py:695`; `scoring.py:107,202` |
| TD-13 | L | Dead code: `_check_point_end`, unused `hit_cooldown`, `ball_history`, `RallyEvent.player_id` for bounces, `old_main.py`, `player.py`, `detect_shoots_only.py`, `analyze_full_match.py` (parallel entry points) | `scoring.py:57-58,192-194`; `MAIN/old_main.py`, `MAIN/player.py` |

## B. Duplicate code

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| TD-14 | M | Ball filtering/interpolation implemented five times | `BO/src/main.py:92`, `yolomain.py:43`, `run_rally_score_yolo.py:18`, `run_tracknet.py:90`; `MAIN/.../ball_tracker.py:501` (+ `BO/src/main.py:44` `suppress_reflection_pairs`) |
| TD-15 | M | Bounce detection implemented at least six times with different parameters (prominence 4, 10, 15 px; ±1 vs ±3 frames) | `ball_tracker.py:17`; `BO/src/bounce_detection.py:13`; `simple_score.py:15`; `test_scoring_only.py`; `bounce_classifier.py:71-260`; `padex/.../bounce.py:69` |
| TD-16 | M | Three scoring state machines with incompatible conventions | `MAIN/analytics/scoring.py`; `BO/src/score.py`; `padex/.../point.py` |
| TD-17 | L | Four copies of the 12-click court UI; four entry points | `MAIN/old_main.py:56`, `MAIN/player.py:23`, `OUTER/main.py:46`, `MAIN/run_analysis.py:48-79`, `MAIN/manual_keypoints_selection.py` |
| TD-18 | L | `BO/src/utils/*` are byte-for-byte-sized copies of `MAIN/utils/*` (518, 848, 1,284, 133 B) | file sizes (hash not computed) |
| TD-19 | L | Two label CSVs, two weight pickles, 47 copies of one video, 18 near-identical job folders | DATASET_INVENTORY |
| TD-20 | L | Three FastAPI servers, three React apps, two Streamlit apps, two `docker-compose.yml` with different build contexts; `OUTER/app.py` imports a non-existent `estimate_velocity` | PROJECT_AUDIT C14 |

## C. Coordinate systems (six, never unified)

| System | Where | Convention |
|---|---|---|
| Image pixels | everywhere | origin top-left, y down; thresholds tuned for 1080p |
| Canvas pixels (mini-court) | `MAIN/analytics/projected_court.py` | width = 0.14 × frame width; metres derived through integer canvas pixels (~4.4 cm per px) |
| Court metres, centre origin | `MAIN` (`court_bounds.py`, `scoring.py:130`) | x width ±5, y length ±10; **y < 0 = top (far)**; net at 0 |
| Court normalised [0,1]² | `BO/src/court_homography.py`, `score.py:123` | `cy < 0.5` = **BOTTOM**, opposite sign to `MAIN` |
| Court metres, corner origin | `padex` (`CLAUDE.md:89`) | x 0–10, y 0–20, net at y = 10 |
| 3D world | `BO/src/court_homography.py:405` (`CourtHomography3D`) | metres with height Z, glass planes |
Team labels are also inconsistent: players 1–2 = "top/team A", 3–4 = "bottom/team B" by *initial side* (`shot_counter.py:17`, `players_tracker.py:452-455`).

## D. Invalid or missing evaluation

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| TD-21 | H | No ground truth for any stage; hit detection `evaluable: false` | `rapport_evaluation.json`; `evaluate_pipeline.py:248-262` |
| TD-22 | H | "Detection rate" = frames with an output; used in the report as if it were accuracy; report table differs from the saved CSV | `BO/src/benchmark_all.py:55,80`; U15 |
| TD-23 | M | Shot-classifier CV is shuffled on correlated windows from the same clip(s), n = 39 | `evaluate_pipeline.py:100-105` |
| TD-24 | M | Homography "error" is the residual of the clicks in canvas space, not positional accuracy | `evaluate_pipeline.py:201-203` |
| TD-25 | M | Unit tests use synthetic events only; none covers trackers, homography, ball filtering or bounce detection | `MAIN/tests/` (83 test functions) |
| TD-26 | M | Many report numbers have no artefact (see PROJECT_AUDIT §6, U01–U19) | PROJECT_AUDIT |

## E. Dependencies and environment

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| TD-27 | H | The venv no longer matches `requirements.txt`: it pins `supervision==0.27.0.post1`, the venv now has 0.31.6 (upgraded for the basketball project on 2026-11-11/12, plus `rfdetr`, `transformers`, `trackers`); `supervision.ByteTrack` no longer exists | `MAIN/requirements.txt`; `iit_courtvison/environment/PHASE1_RECORD.md` |
| TD-28 | M | Both `opencv-python` 4.11 and `opencv-contrib-python` 4.13 are installed (they overwrite each other's `cv2`) | `smartplay_baseline_freeze.txt:66-67` |
| TD-29 | L | Installed but never imported by project code: `mediapipe`, `pyvista`, `vtk`, `trimesh`, `lap`, `pyarrow`, `kaleido`, `pims`, `sounddevice`, `shapely`, `seaborn`, `networkx`, `fire`; dev tools `black`, `flake8`, `isort`, `mypy`, `pre-commit` | import scan (MEASURED); `sahi`, `polars` used only by `padex` |
| TD-30 | M | `padex` (GPL-3.0) is installed **editable** in the venv from `aymen-kamel/Play_Ai@d1b9513`, although `MAIN` never imports it (only 4 root scripts do) | `smartplay_baseline_freeze.txt:71`; import scan |
| TD-31 | M | Hard-coded absolute paths `c:\Users\dell\OneDrive\Bureau\pfe\padel_analytics_repo\...` in six `BO` scripts and `find_images.py:4`; `BO` is not runnable without `MAIN` | `BO/src/{main:11,yolomain:13,benchmark_all:13,run_rally_score:12,run_tracknet:17,analyze_bounces:10}.py` |
| TD-32 | L | Broken references: `BO/README.md:81` → `src/plot_3d_interactive.py` (absent); benchmark scripts write `benchmark_results.json` that is absent | file lists |
| TD-33 | M | 1080p thresholds in pixels (35 px/frame, 70 px, 150–250 px, `y < 450`) and fps assumptions (30) | `ball_tracker.py:395-404`; `shot_counter.py:114-119`; `output.py:15` |
| TD-34 | M | Memory model: up to 400 full frames for the background median (~2.5 GB at 1080p); each tracker decodes the full video again; 1 job crashed | `iterable.py:60-83`; `config.py:50`; `results_json/f1ee65ee…/job_meta.json` |

## F. Repository, licence and operations

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| TD-35 | H | Six months of work has **one commit** (2026-07-02) in `MAIN`; ~25 files uncommitted; `.gitignore` excludes `*.json`, `*.csv`, `weights/`, `models/`, `train/`, `dataset/` so configs and results are invisible to git; the parent folder is not a repository | `git log`, `git status`, `MAIN/.gitignore` |
| TD-36 | M | The project sits in OneDrive with a 7 GB venv (56,546 files) and GBs of video syncing | path `...\OneDrive\Bureau\pfe`; `iit_courtvison/environment/PHASE0_RECORD.md` |
| TD-37 | H | **Licence exposure**: upstream *Padel Analytics* (no LICENSE locally; secondary note says CC BY-NC-SA 4.0), *Padex* GPL-3.0, Ultralytics AGPL-3.0 (code and weights), and `BO/src/bounce_classifier.py` closely mirroring Padex code | see MODEL_INVENTORY, REUSABLE_COMPONENTS |
| TD-38 | M | Pickles (`rally_tracking_cache.pkl`, `shot_classifier.pkl`) and an open CORS regex for ngrok origins; API/upload code not reviewed line by line | `MAIN/api_server.py:42`; UNVERIFIED |
| TD-39 | L | `MAIN/README.md` is the upstream's README (claims 13-DoF pose, trained models, 8 GB VRAM) and does not describe this pipeline; `MAIN/ai_assistant_prompt.md` describes a RAG assistant that has no code behind it | `MAIN/README.md:12-16,80`; `MAIN/ai_assistant_prompt.md` |
| TD-40 | L | Two LaTeX roots (`main.tex` includes template chapters, `main_smartplay.tex` includes real ones) plus `_new` duplicates; build artefacts committed | `pfe_latex/` |
| TD-41 | L | Typos in names used as paths: `vedio.mp4`, `goooooddddddd.mp4`, `rallie/` vs `rallies/`, `shoots` | file list |

## UNVERIFIED
- Behaviour of TD-03 (read from code, not run).
- API security (TD-38) and Docker build success (TD-20).
- Byte-level identity of "same-size" duplicates (TD-18, TD-19).
- Whether TD-01's `neck` lookup was always empty in the runs that produced the saved JSON (depends on the weights' keypoint order at run time).
