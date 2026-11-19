# SmartPlay AI 2.0

Padel match analysis from **one fixed camera**: find the players and the ball, hear the hits,
cut the match into points, and derive the **full score** (points, games, sets) under the
official FIP rules.

> **Status (2026-11-14): early baseline, nothing is validated yet.**
> The score engine is finished and tested. The video pipeline already tracks four players and
> the ball and finds audio hit candidates, but **no stage has been measured against labelled
> data** (none exists yet). Numbers below are *observations on one 13-second clip*, not accuracies.

This is a clean rewrite of an earlier prototype whose score tracking never worked (every run
ended 0-0). The audit, research and plan that led here are in [`docs/`](docs/).

---

## Where we are

| Stage | State | What exists | Measured so far (rally.mp4, 324 frames) |
|---|---|---|---|
| Score engine (FIP rules) | **Done** | Advantage / Golden Point / Star Point, tie-break, mini set, match and super tie-break, service rotation, changes of ends | 45 unit tests |
| Court calibration | Baseline | Floor homography from 12 clicked points, leave-one-out error | 14.2 cm leave-one-out |
| Players | Baseline | RF-DETR Small + four-identity tracker (teams A/B, NEAR/FAR side) | 4 players observed in 99.7 % of player-frames |
| Ball | Baseline, zero-shot | WASB HRNet (tennis weights) + online tracker, no gap filling | seen in 65.9 % of frames; accuracy **unknown** |
| Hits | Baseline | Untrained audio onset detector | 10 candidates; 7 coincide (within 2 frames) with the old visual detector; accuracy **unknown** |
| Bounces | Not started | | |
| Point segmentation / winner | Not started | | |
| Score from video | Not started | | |
| API / UI | Not started | | |
| Evaluation on labelled data | Blocked | metric code exists; **needs recorded and labelled matches** | |

Speed on an RTX 3050 6 GB: about 15 frames/s for the whole pipeline (the old prototype: 0.14-1.3).

### Why the old prototype scored 0-0
Documented in [`docs/audit/SCORE_DIAGNOSIS.md`](docs/audit/SCORE_DIAGNOSIS.md): bounce = image-height
peak (cannot tell floor from wall from racket), one scoring rule, ball track lost 34 % of the time
and refilled by unlimited interpolation, and no ground truth anywhere.

## Quick start

Everything needed lives in this folder: code, one Python environment, all model files in `weights/`
(see [`weights/MODELS.md`](weights/MODELS.md) for sources, licences and SHA-256). Nothing is read from the
earlier PFE project or from library caches.

**Install (Windows, NVIDIA GPU; validated on an RTX 3050 6 GB, driver 610.60, CUDA 11.8 build of torch):**

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pytest
```

`requirements.lock` is the exact list of versions that passed the tests. RTMPose runs on the CPU in this setup
(ONNX Runtime CPU build); GPU for RTMPose needs `onnxruntime-gpu` with CUDA 12, see [`docs/MIGRATION.md`](docs/MIGRATION.md).

**Model files** must be in `weights/` (they are not in git): the WASB tennis model, the TrackNetV3 checkpoint,
RF-DETR Small, the RTMPose and YOLOX ONNX files, and optionally YOLO26-pose (AGPL, experiments only).
Each is listed in `weights/MODELS.md` with its SHA-256; check them after copying.

**Analyse a video** (fixed camera, 12 clicked court points in a JSON list):

```
.venv\Scripts\python -m smartplay.cli annotate --video dataawally.mp4 ^
    --keypoints data\calibrationally_court_keypoints.json --out runsally
```

Writes `annotated.mp4` (court lines, players with team and side, mini-map, ball with trail, stats bar,
audio hit candidates), `tracks.json`, `ball.json`, `audio_hits.json`. Options: `--predict-ball N` (physics estimates,
drawn as hollow rings). The 3D ball, skeletons and the replay page are produced by `scripts/calibrate_3d.py`,
`scripts/ball3d_clip.py`, `scripts/annotate_3d.py` and `smartplay.cli export-replay`.

Checked on 2026-11-19: the clip `point_24` gives the same counts in this environment as in the earlier one
(ball seen in 317/401 frames, players observed in 1604/1604 player-frames).

Data, weights and run outputs are git-ignored (`data/`, `weights/`, `runs/`, `.venv/`, `*.pt`, `*.onnx`, `*.mp4`, ...).
Only `weights/MODELS.md` is tracked in `weights/`.

### Test on a broadcast highlights video (11 min, Premier Padel)
`scripts/segment_broadcast.py` + `scripts/analyze_highlights.py` + `scripts/export_clips.py`
cut a broadcast into shots, find the wide main view, read the on-screen scoreboard with OCR and
turn score changes into point labels (the clips and JSON files are written to `runs/`, not stored in git).

| Measure (one video, 19,943 frames) | Result |
|---|---|
| Shots found | 84, of which 29 are the main wide view (424 s of 665 s) |
| Scoreboard states read | 30 distinct, 1 reading dropped as OCR error |
| Changes explained by exactly one point (winner known) | 9, including a game and a set ending |
| Consecutive score changes reachable with the FIP rules in this repo | **29 of 29** (up to 13 points skipped between readings) |

The last row is the useful one: real broadcast scores are consistent with the score engine
(Advantage and Golden Point give the same result here because no deuce sequence was observed).
Limits: highlights skip points, so a full score cannot be rebuilt from this video; the main-view
shots are not yet checked by eye one by one; this video has a different framing from `rally.mp4`,
so player/ball tracking on it still needs its own court calibration.

## Architecture

```
video + audio -> ingestion -> court model -> players + teams -> ball -> events (hits, bounces)
              -> point segmentation -> point winner (probability) -> score engine -> timeline
```
Every stage reads and writes typed, validated records ([`schemas/records.py`](src/smartplay/schemas/records.py)),
uses one coordinate convention (metres, origin at the net centre, NEAR/FAR sides, teams A/B) and
reports uncertainty instead of guesses. Full design: [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md).

| Package | Content |
|---|---|
| `smartplay.scoring` | FIP score automaton (pure logic, fully tested) |
| `smartplay.geometry` | Court constants, coordinate convention, camera model |
| `smartplay.schemas` | Typed records between stages |
| `smartplay.ingestion` | Frame reader, audio extraction |
| `smartplay.players` | Person detector, four-player tracker |
| `smartplay.ball` | WASB ball model (vendored, MIT) and online tracker |
| `smartplay.events` | Audio hit candidates |
| `smartplay.viz` | Annotated video |
| `smartplay.evaluation` | Event matching, point/score accuracy, confidence intervals |

## Roadmap (milestones ordered by impact on score accuracy)

| | Milestone | Status |
|---|---|---|
| M0 | Foundations: repo, schemas, FIP score engine, evaluation harness | done |
| M1 | Ground truth: record full matches with a fixed camera (audio on), label points | **next, needs you** |
| M2 | Point segmentation baseline | not started |
| M3 | Ball measured on own footage, then fine-tuned | baseline running |
| M4 | Players, teams, side-change detection | baseline running |
| M5 | Events: hits (audio + ball), bounces (trajectory + court height) | hits baseline |
| M6 | Point winner + constrained score decoding | not started |
| M7 | End-to-end evaluation on held-out matches | blocked by M1 |
| M8 | API, UI, analytics | not started |

Plan, evaluation protocol (match-level splits, oracle ladder) and labelling quantities:
[`docs/architecture/PLAN.md`](docs/architecture/PLAN.md).

## Known limits
- The ball model was trained on tennis; padel accuracy is unmeasured.
- Calibration is for floor points only; an airborne ball needs the 3D model (planned).
- The audio detector finds *sounds*, not verified racket hits.
- Only one short broadcast clip has been analysed; fixed-club-camera behaviour is unknown.

## Documentation
**Start with [`docs/FLOORPLAN.md`](docs/FLOORPLAN.md)**: map of every component and integration, with status, licence and measured numbers.
[`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md) (plain-language overview) ·
[`docs/audit/`](docs/audit/) (what the earlier prototype really did, with evidence) ·
[`docs/research/`](docs/research/) (state of the art, verified licences) ·
[`docs/architecture/`](docs/architecture/) (design and plan) · [`docs/PROGRESS.md`](docs/PROGRESS.md) (log).

## Licence and credits
Clean-room code: no code from earlier forks, Padex (GPL-3.0) or Ultralytics (AGPL-3.0).
Third-party work and its licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
(WASB-SBDT and HRNet, MIT; RF-DETR, Apache-2.0). The licence of this project itself is still to be chosen.
