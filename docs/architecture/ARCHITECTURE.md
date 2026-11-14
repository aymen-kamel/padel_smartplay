# SmartPlay AI 2.0 — Architecture (Phase 3)

Date: 2026-11-14. Status: **proposal, nothing implemented**. Based on `docs/audit/` and `docs/research/`.
Fixed by the user: one fixed camera · v1 = points → full score · clean rewrite under own name and licence · 4 GB VRAM budget (machine: RTX 3050 6 GB) · Python 3.11+.
Assumption to confirm: after-match (offline) processing.

## 1. Design principles

1. **Score first.** Every stage exists to answer: when does a point start and end, and who won it?
2. **Typed contracts between stages.** Each stage reads and writes validated records (Pydantic v2). Stages can be run, cached and tested alone.
3. **Uncertainty is data.** Every detection and event carries a confidence; nothing downstream treats a guess as a fact (audit: errors compound).
4. **One coordinate convention** (section 3). Pixels never leave the perception stages.
5. **Bounded memory, streaming.** Video is processed in chunks; no stage loads the whole video (audit C01).
6. **Measured before trusted.** Each stage has a metric and a labelled test set before it is tuned (PLAN.md).
7. **Clean-room.** No code or weights from the upstream fork, Padex (GPL-3.0) or Ultralytics (AGPL-3.0). Ideas and public algorithms only.
8. **Only needed tools.** No SAM2, no Streamlit duplicate, no unused packages.

## 2. Data flow

```
 match.mp4 ──► [1 ingest] ──► frames (chunks) ─┬─► [3 players] ─► [4 tracking + teams] ─┐
     │                                         ├─► [5 ball detect] ─► [6 ball track 3D] ─┤
     │            calibration.json             │                                         │
     │        [2 court calibration] ───────────┴──────────(camera model)────────────────►│
     └─► audio.wav ──────────────────────────────► [7 events: hits (audio+ball),         │
                                                      bounces (trajectory+3D)] ◄─────────┘
                                                              │
                                              [8 rally segmentation: serve … point end]
                                                              │
                                              [9 point winner: P(team A wins) per point]
                                                              │
                              [10 score engine: FIP automaton + constrained decoding] ─► score timeline
                                                              │
                         [11 analytics] ─► [12 visualization] ─► [13 API / storage]
```

Stages 3–6 are the GPU stages and run **one model at a time** per chunk (fits 4 GB; simple). Stages 7–10 are CPU.
Every stage writes its output to `runs/<match_id>/<stage>.parquet|jsonl`, so later stages can be re-run without re-running perception.

## 3. Coordinate and naming convention (single source of truth)

| Item | Definition |
|---|---|
| World frame | Metres. Origin at the **centre of the net on the floor**. `x` across the court (−5 … +5, positive to the camera's right), `y` along the court (−10 … +10, **negative = near the camera, positive = far**), `z` up (0 = floor). |
| Court constants | 10 × 20 m, service lines at `y = ±6.95`, net 0.88 m centre / 0.92 m ends, back walls 3 m glass + 1 m mesh (FIP rules, verified in Phase 2). |
| Sides | `NEAR` (`y < 0`) and `FAR` (`y > 0`). Never "top/bottom". |
| Teams | `A` and `B` = fixed pair identities for the whole match. Which side a team is on **changes** after games 1, 3, 5 … (FIP Rule 5). |
| Time | `t` in seconds from video start (float), plus integer `frame`. fps read from the file — never assumed (audit TD-07). |
| Pixels | `(u, v)` in the original image; only stages 2–6 use them. |

## 4. Stage interfaces

Signatures are illustrative; exact fields are finalised in milestone M0. `Protocol` classes let implementations be swapped (e.g. two ball detectors benchmarked behind one interface).

```python
# schemas (pydantic.BaseModel, frozen)
class VideoInfo:        path: Path; fps: float; n_frames: int; width: int; height: int; has_audio: bool
class FrameChunk:       start_frame: int; frames: np.ndarray            # (N,H,W,3) uint8, N bounded
class CameraModel:      K: Mat3; R: Mat3; t: Vec3; dist: list[float]; reproj_err_cm: float
class PlayerDet:        frame: int; box_uv: Box; conf: float
class PlayerTrack:      frame: int; player_id: Literal[1,2,3,4]; team: Literal["A","B"]
                        foot_xy_m: Vec2; side: Literal["NEAR","FAR"]; conf: float; observed: bool
class BallObs:          frame: int; uv: Vec2 | None; conf: float
class BallState:        frame: int; uv: Vec2 | None; xyz_m: Vec3 | None
                        visibility: Literal["visible","occluded","inferred","lost"]; conf: float
class Event:            t: float; frame: int; kind: Literal["hit","floor_bounce","wall","net","serve"]
                        side: Literal["NEAR","FAR"] | None; player_id: int | None
                        xy_m: Vec2 | None; conf: float; sources: list[Literal["audio","trajectory","pose"]]
class Rally:            point_idx: int; t_start: float; t_end: float; server_team: Literal["A","B"] | None; conf: float
class PointOutcome:     point_idx: int; p_team_a: float; reason: str | None      # probability, not a verdict
class ScoreState:       point_idx: int; t: float; sets: tuple[int,int]; games: tuple[int,int]
                        points: tuple[str,str]; server_team: str; ends_swapped: bool; conf: float

# stage protocols
class Ingestor(Protocol):          def probe(self, path) -> VideoInfo; def chunks(self, size:int) -> Iterator[FrameChunk]; def audio(self) -> AudioClip
class Calibrator(Protocol):        def calibrate(self, clicks: list[CourtPoint]) -> CameraModel
class PlayerDetector(Protocol):    def detect(self, chunk: FrameChunk) -> list[PlayerDet]
class PlayerTracker(Protocol):     def update(self, dets: list[PlayerDet], cam: CameraModel) -> list[PlayerTrack]
class BallDetector(Protocol):      def detect(self, chunk: FrameChunk) -> list[BallObs]
class BallTracker(Protocol):       def track(self, obs: list[BallObs], cam: CameraModel) -> list[BallState]
class EventSpotter(Protocol):      def spot(self, ball, players, audio, cam) -> list[Event]
class RallySegmenter(Protocol):    def segment(self, events, ball, players, audio) -> list[Rally]
class PointJudge(Protocol):        def judge(self, rally: Rally, events, ball, players) -> PointOutcome
class ScoreEngine(Protocol):       def decode(self, outcomes: list[PointOutcome], anchors: list[Anchor], rules: MatchRules) -> list[ScoreState]
```

| # | Stage | v1 implementation (candidate, to validate) | Device | Output |
|---|---|---|---|---|
| 1 | Ingestion / preprocessing | FFmpeg-based decoding (PyAV) in chunks; audio to mono 16–22 kHz WAV; fps and size from the container | CPU | `VideoInfo`, chunks, audio |
| 2 | Court + camera | One-time manual clicks per camera install → RANSAC homography → `solvePnP` with the 3D court; saved `calibration.json`; reprojection error reported in cm | CPU | `CameraModel` |
| 3 | Player detection | RF-DETR Small (Apache-2.0), every frame or every 2nd frame | GPU | `PlayerDet` |
| 4 | Tracking + teams | OC-SORT (MIT) + padel constraints: 4 identities, 2 per side, appearance model per team, **side-swap detector** (all four players cross the net between points) | CPU | `PlayerTrack`, `Anchor(side_change)` |
| 5 | Ball detection | TrackNetV3-architecture heat-map model (MIT), fine-tuned on padel | GPU | `BallObs` |
| 6 | Ball tracking | Kalman filter with coasting limit and visibility states; 3D lift where possible (floor contact, known wall planes, ballistic fit) | CPU | `BallState` |
| 7 | Events | Hits: audio onset model fused with ball direction change and nearest player. Bounces: gradient-boosted classifier on trajectory-corner + height + wall-distance features | CPU (small CNN can use GPU) | `Event` |
| 8 | Rally segmentation | Serve detection (server behind service line, ball bounce then hit, diagonal flight) + end of point (no hits for ~3 s, ball lost, players relax); bounded by FIP's 20 s rule | CPU | `Rally` |
| 9 | Shot detection | v1: only `serve` (needed for stage 8). Shot types later | CPU | `Event(kind="serve")` |
| 10 | Point winner | Rules from FIP Rule 13 on the last events of a rally (second floor bounce on a side, ball direct to opponents' wall, net) → probability; calibrated on labelled points | CPU | `PointOutcome` |
| 11 | Scoring engine | Pure-Python FIP automaton (Advantage / Star Point / Golden Point, tie-break, alternatives, changes of ends, server rotation) + Viterbi decoding of `PointOutcome` probabilities constrained by anchors (side changes, server changes) | CPU | `ScoreState` timeline |
| 12 | Analytics | Distance, speed, positions on smoothed metric tracks; rally length, hits per rally | CPU | JSON |
| 13 | Visualization | Annotated video (OpenCV) + 2D court view + score overlay | CPU | MP4 |
| 14 | API | FastAPI + Pydantic: `POST /jobs`, `GET /jobs/{id}`, `/results`, `/score`, `/video`; one worker, job folder on disk | CPU | HTTP |

**Human-in-the-loop hooks (cheap, high value):** match rules (scoring option, who serves first, team names) are entered at upload; the score timeline can be corrected point by point, and corrections are stored as new labels.

## 5. Technology choices (only what is needed)

| Need | Choice | Licence (Phase 2 status) |
|---|---|---|
| Language / packaging | Python 3.11, `pyproject.toml`, `uv` or `pip` + lock file | — |
| Tensors | PyTorch (CUDA build matching the driver) | BSD-style |
| Video / audio I/O | PyAV (FFmpeg) ; OpenCV for image ops | LGPL/BSD — check build at install |
| Schemas / config | Pydantic v2 ; YAML configs | MIT |
| Tables | Parquet via PyArrow or Polars ; JSONL for events | Apache-2.0 / MIT |
| Detector | `rfdetr` (N/S weights) | Apache-2.0 [V] |
| Tracker | OC-SORT via `trackers` or own implementation | MIT [V] / Apache-2.0 [S] |
| Ball model | Own implementation of the TrackNetV3 design, own training | MIT reference [V] |
| Classical ML | scikit-learn (HistGradientBoosting) | BSD |
| Audio features | torchaudio or librosa | BSD / ISC (UNVERIFIED) |
| API | FastAPI + Uvicorn | MIT / BSD |
| Tests / quality | pytest, ruff, mypy | MIT |
| Experiment log | Plain run folders + `metrics.json` + git commit hash (no server needed) | — |
| Labelling | CVAT (used by PadelTracker100) or Label Studio | check at install |

Not used: Ultralytics, SAM2, Streamlit, mediapipe, pyvista/vtk, Docker (until the pipeline works).

## 6. Repository layout

Recommended location: **outside OneDrive** (e.g. `C:\dev\smartplay`), with `git init` on day one. Data and weights live outside git.

```
smartplay/
├── pyproject.toml  README.md  LICENSE  .gitignore
├── configs/                 # default.yaml, rules/{advantage,golden_point,star_point}.yaml, cameras/<club>.json
├── src/smartplay/
│   ├── schemas/             # pydantic models (section 4)
│   ├── geometry/            # court constants, coordinate transforms, camera model
│   ├── ingestion/           # video + audio readers
│   ├── calibration/         # click tool, solvePnP, reprojection report
│   ├── players/             # detector wrapper, tracker, team/side logic
│   ├── ball/                # model, dataset, training, inference, tracker
│   ├── events/              # audio hits, bounce classifier, fusion
│   ├── rally/               # serve detection, point segmentation
│   ├── scoring/             # FIP automaton, point judge, decoder
│   ├── analytics/           # metrics
│   ├── viz/                 # overlays, 2D court
│   ├── api/                 # FastAPI app
│   ├── evaluation/          # metrics, matching, splits, reports
│   └── pipeline.py  cli.py
├── scripts/                 # train_ball.py, label_import.py, run_match.py, eval_all.py
├── tests/                   # unit (scoring, geometry), golden-file tests per stage
├── experiments/             # dated notes + configs; results/ ignored
├── docs/                    # audit/, research/, architecture/, PROGRESS.md
├── data/                    # NOT in git: raw/, labels/, splits/
├── weights/                 # NOT in git
└── runs/                    # NOT in git: per-match outputs
```

## 7. Mapping of existing files

Because this is a clean rewrite, nothing from the forks is copied. "Re-implement as" names the new module that takes over the *role*. "Archive" = keep untouched in a read-only `legacy/` folder outside the new repo; **nothing is deleted without your decision**.
Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `OUTER` = `padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.

### Own code — port the logic (rewrite cleanly, keep the ideas and tests)
| Existing file | Action | New home |
|---|---|---|
| `MAIN/evaluate_pipeline.py` | Port approach | `src/smartplay/evaluation/` |
| `MAIN/analytics/court_bounds.py`, `MAIN/tests/test_court_bounds.py` | Port | `geometry/court.py`, `tests/` |
| `MAIN/analytics/match_report.py`, `MAIN/tests/test_match_report.py` | Port KPI formulas after smoothing fix | `analytics/` |
| `MAIN/tests/test_scoring_fsm.py`, `test_shot_counter.py`, `test_conversions.py`, `conftest.py`, `pytest.ini` | Port test *style* (synthetic event sequences) | `tests/scoring/`, `tests/events/` |
| `BO/src/score.py` (`PadelScore`) | Port bookkeeping, extend to full FIP rules | `scoring/automaton.py` |
| `BO/src/court_homography.py` (3D part), `calibrate_court_3d.py`, `calibrate_court.py` | Port after numerical validation | `calibration/`, `geometry/camera.py` |
| `BO/src/kalman_tracker.py` | Port idea (coasting limit) | `ball/tracker.py` |
| `MAIN/api_server.py`, `MAIN/run_analysis.py` | Port endpoint shape only | `api/` |
| `the_best/paddle_rally_extractor.py` | Port idea (wide-shot filter) for labelling broadcast footage | `scripts/segment_broadcast.py` |
| `MAIN/clean_artifacts.py`, `shot_dataset.py` | Merge ideas | `scripts/`, `cli.py` |
| `MAIN/padel-ui/src/*` | Rebuild later against new API | `web/` (after M7) |

### Own code — archive (superseded or unmeasured)
`MAIN/analytics/scoring.py`, `shot_counter.py`, `shot_feature_extractor.py`, `advanced_features.py`, `shot_classifier/*`, `MAIN/trackers/runner.py`, `trackers/players_tracker/*`, `MAIN/{collect_shot_data,annotate_shots,train_shot_classifier,train_lstm_classifier,analyze_full_match,detect_shoots_only,benchmark_pipeline,extract_latex_stats,extract_results,dashboard,config,main,manual_keypoints_selection}.py` ·
`BO/src/{main,yolomain,run_tracknet,run_rally_score,run_rally_score_yolo,simple_score,benchmark_all,analyze_bounces,bounce_detection,glass_zones,calibrate_glass,test_score_logic,test_scoring_only}.py`, `BO/scratch/*`, `BO/config/*`, `BO/docs/*`, `BO/{README,CLAUDE,analysis_results}.md` ·
root `benchmark_performance.py`, `run_padex*.py`, `test_padex_shots.py`, `find_images.py`, `print_*.py`, `read_docx.py`, `search_tex.py`, `docker-compose.yml`, `.dockerignore` ·
`OUTER/{app,main,api_server}.py`, `OUTER/api/*`, `OUTER/frontend/*`, `OUTER/analytics/*`, `OUTER/Dockerfile`, `OUTER/docker-compose.yml` · `web_site/*`.

### Third-party or licence-flagged — archive, reference only, never copy
| Existing | Reason |
|---|---|
| `MAIN/trackers/tracker.py`, `velocity_in_time.py`, `ball_tracker/*`, `keypoints_tracker/*`, `players_keypoints_tracker/*`, `MAIN/utils/*`, `visualizations/*`, `constants/*`, `MAIN/analytics/{data_analytics,projected_court}.py`, `MAIN/{old_main,player}.py`, `MAIN/README.md`, `MAIN/examples/*` | Upstream fork (licence unverified, possibly NC-SA) |
| `padex/**` (all) and `BO/src/bounce_classifier.py` | GPL-3.0 / mirrors GPL code |
| `BO/src/utils/*` | Copies of upstream utils |

### Data, weights, outputs
| Existing | Action | New home |
|---|---|---|
| `data_candidates/Video Project 1.mp4` | Keep | `data/raw/broadcast/highlights_01.mp4` (after the YouTube-terms decision) |
| One copy each of `rally.mp4`, `rally_001.mp4`, `rally2–4.mp4`, `highlights.mp4`, `match3.mp4`, `padel.mp4`, other `BO/input/*`, `the_best/vedio.mp4` | Keep one copy as smoke-test clips (not for metrics) | `data/raw/legacy_clips/` |
| The other 46 copies of `rally.mp4`; `MAIN/uploads/*`, `MAIN/results_json/*`, `MAIN/output/*`, `OUTER/storage/*`, `BO/output/*`, `output/*`, `the_best/rallie*/`, compilations, root debug videos, `MAIN/results_new.mp4` | Archive (regenerable or duplicates). Candidates for deletion **only on your approval** (~1.5 GB) | `legacy/` |
| `MAIN/cache/*.json`, `MAIN/data_new.csv`, `MAIN/summary.json`, `rapport_evaluation.json`, `match_stats_table.tex`, `BO/output/*.csv`, `the_best/rallies_output.json` | Archive as historical evidence (cited by the audit) | `legacy/` |
| `shot_features*.csv` (root and `MAIN`) | Archive; too small and inconsistent to train on | `legacy/` |
| All `.pt` / `.pkl` in `MAIN/weights`, `BO/models`, root `weights/` | Archive; **not shipped** (licence flags in `MODEL_INVENTORY.md`) | `legacy/weights/` |
| `ball_only.zip`, `padel_analytics_repo.zip` | Archive (old snapshots) | `legacy/` |
| `smartplay/` venv | Leave as is (still used by `iit_courtvison`); new project gets its own environment | — |

### Documents
| Existing | Action |
|---|---|
| `pfe_latex/*`, `rapport/*` | Keep; rewrite results sections once real measurements exist; remove or verify every item in audit §6 "Unsupported claims" |
| Root notes (`today_review_2026_04_14.md`, `player_tracker_documentation.md`, `diagrammes_architecture.md`, `pfe_report_plan.md`), `MAIN/{ai_assistant_prompt.md,player_tracking_explanation.txt}`, `OUTER/SmartPlayAI Sprints (2).pdf` | Archive under `legacy/notes/` |
| `docs/**` (this audit, research, architecture) | Move into the new repo's `docs/` |
| `iit_courtvison/**` | Out of scope; untouched |

## 8. Risks built into this design
- **Audio dependence**: hit detection assumes usable sound in club recordings. Mitigation: trajectory-only fallback; verify on the first recording.
- **Ball visibility through glass/mesh and under indoor lighting** is unknown for your camera. Mitigation: measure in M3 before investing in events.
- **Single-camera 3D** is under-determined in free flight. Mitigation: only trust 3D at contacts (floor, walls) and for short ballistic fits.
- **Licence of training data from YouTube** undecided. Mitigation: own recordings are sufficient for the test set; public data only if terms allow.
