# Floor plan: what is in SmartPlay and how it connects

Read this first. It is the map of the whole project: layers, every integration (with licence and status), data flow,
where things live. **Update it in the same commit as any new integration** (rule at the bottom).

Status: **BUILT** (works, tested), **MEASURED** (numbers in `BENCHMARK_BASELINE.md`), **PARTIAL**, **NOT BUILT**.

## 1. Data flow

```
video (fixed camera) ─┬─> players  (RF-DETR) ─> 4-identity tracker ─> floor positions (m) ─> player stats ─┐
                      ├─> ball     (WASB + TrackNetV3 candidates) ─> merge ─> gated tracker ─> 2D track ───┤
audio ───────────────> hit onsets (spectral flux) ──────────────────────────────────────────────────────┤──> annotated video
12 court clicks ─────> floor homography (metres) ───────────────────────────────────────────────────────┤──> JSON export ─> 3D replay page
broadcast scoreboard ─> OCR ─> reachable-state filter ─> score timeline (ground truth for broadcast clips)┘
score automaton (FIP rules) ... fed today only by synthetic/OCR points; NOT yet by the vision events
```

## 2. Integrations register

| Component | What it does | Source / licence | Status | Measured | Code |
|---|---|---|---|---|---|
| RF-DETR Small | person detection | Roboflow, Apache-2.0 | BUILT, MEASURED | players tracked in 99.7-100 % of frames | `players/detector.py` |
| Four-player tracker | keeps 4 identities, teams by side | own | BUILT | same | `players/tracker.py` |
| WASB HRNet (tennis weights) | ball candidates (3 frames) | NTT, MIT | BUILT, MEASURED | alone: F1 0.70 @5 px | `ball/detector.py`, `ball/wasb_hrnet.py` (vendored) |
| TrackNetV3 (official checkpoint) | ball candidates (8 frames + background) | qaz812345, MIT (weights trained on badminton) | BUILT, MEASURED | alone: F1 0.62 @5 px, 0.82 @10 px | `ball/tracknet.py` (class re-implemented) |
| Candidate merge | WASB first, add TrackNet where WASB has none | own | BUILT, MEASURED | fused: F1 **0.78 @5 px**, 0.86 @10 px | `ball/fusion.py` |
| Gated tracker | picks one ball per frame, rejects glass reflections | own | BUILT, MEASURED | wrong-place picks 24 -> 8 | `ball/detector.py::track_gated` |
| 2D physics post-processor | flights, kinks, gap fill, `--predict-ball` | own | PARTIAL | no position gain; predicted frames right in ~2 of 9 | `ball/physics.py` |
| Audio hit detector | impact onsets, 2-7 kHz band | own | BUILT, MEASURED | F1 0.64 (2 clips), **0.80 (99 CVSPORTS clips, offset-corrected)** | `events/audio.py` |
| Learned hit classifier (audio) | candidates + gradient boosting, leave-one-tournament-out | own, scikit-learn (BSD) | BUILT, MEASURED | F1 **0.85** vs 0.80 untrained (99 CVSPORTS clips) | `events/learned.py`, `scripts/train_hit_classifier.py` |
| Pose (hit features) | wrist-ball distance at each candidate | yolo26m-pose, Ultralytics **AGPL-3.0, experiment only** | MEASURED | pose alone F1 0.66; no gain when added (0.851) | `scripts/pose_hit_features.py` |
| Ball-kink hit candidates | trajectory breaks added to audio onsets | own | MEASURED | ceiling 0.92 -> 0.98; F1 +0.01-0.02 (same code path) | `scripts/ball_candidates_eval.py` |
| Floor homography | image -> court metres (floor only) | own | BUILT, MEASURED | fit 4.7 cm, leave-one-out 7.2 cm | `geometry/camera.py`, `geometry/court.py` |
| Score automaton | FIP scoring (advantage / golden point, tie-breaks, sides) | own, FIP Rules 2026 | BUILT (tests) | not fed by video yet | `scoring/automaton.py` |
| Scoreboard OCR | broadcast score timeline | RapidOCR, Apache-2.0 | BUILT, MEASURED | 29 of 29 transitions reachable | `broadcast/` |
| Player stats | smoothed distance and speed | own | BUILT (tests) | not validated against a reference | `analytics/player_stats.py` |
| Annotated video | court, players, ball, stats bar, timeline | own | BUILT | visual | `viz/annotate.py`, `cli annotate` |
| Replay exporter + 3D page | measured positions in the owner's HeroScore page | own + owner's page | BUILT | visual; published artifact | `export/replay.py`, `viewer/heroscore.html` |
| Label tool + benchmark | hand labels, P/R/F1 | own | BUILT | 2 clips, ~240 labelled ball frames | `labeling/`, `scripts/` |
| CVSPORTS_Padel | 99 rallies, 2 377 labelled hits (Ghent/imec) | licence **not stated** | DATA, MEASURED | see hits row | `data/external/cvsports_padel/` |
| PadelTracker100 labels | ball boxes, pose, shots (videos private) | CC BY 4.0 | DATA, unusable for training (no frames) | - | `data/external/padeltracker100/` |
| 3D camera (DLT) | full projection from floor + height clicks; ray / point at height | own | BUILT, MEASURED | highlights main view: reprojection 5.2 px, leave-one-out 6.6 px, camera 8.1 m high | `geometry/camera3d.py`, `scripts/calibrate_3d.py` |
| Ball in 3D | projectile fit per flight, plausibility filter | own | PARTIAL | with player + bounce priors: 32 % (clip 24) and 61 % (clip 12) of frames in 3D; not validated | `ball/ballistic3d.py`, `scripts/ball3d_clip.py` |
| Hit classifier + ball track | ball speed/turn/kinks added to the audio model | own | BUILT, MEASURED | F1 0.851 vs 0.847 audio only: no real gain | `scripts/train_hit_classifier_vision.py` |
| Pose, shot types, point winner | - | - | NOT BUILT | - | - |

Licence flags to clear before release: TrackNetV3 weights (MIT but trained on badminton data: check dataset terms),
CVSPORTS_Padel (no licence stated), any use of PadelTracker100 frames (videos private). Details: `THIRD_PARTY_NOTICES.md`.

## 3. Where things live

- `src/smartplay/`: `ingestion`, `players`, `ball`, `events`, `geometry`, `scoring`, `analytics`, `broadcast`, `viz`, `export`,
  `labeling`, `evaluation`, `schemas` (single data contract: `schemas/records.py`), `cli.py` (`annotate`, `export-replay`).
- `scripts/`: one-off experiments and evaluations (benchmarks, dataset tests). `tests/`: 61 tests. `viewer/`: 3D page.
- `docs/`: this file, `PROGRESS.md` (log), `BENCHMARK_BASELINE.md` (all measured numbers), `BRAINSTORM.md`, `NEXT_STEPS.md`,
  `architecture/`, `research/`, `audit/` (what the old project did wrong).
- Local only (git-ignored): `data/`, `weights/`, `runs/`.

## 4. The gap that matters
Vision events (ball, hits) are not yet turned into points. Missing links, in order: typed bounces (needs the 3D camera) ->
point segmentation -> point winner -> score automaton. Everything before that link is built and measured; everything
after it is the score engine, which is built but only fed by OCR.

## 5. Rule: how to keep this map true
Every new integration (model, dataset, script that produces a number) updates, **in the same commit**: the data-flow block
(if it changes), one row in the register (source, licence, status, measured number with its benchmark), and the licence
flags. Numbers come only from `BENCHMARK_BASELINE.md`; if there is no measurement the cell says "not validated".
