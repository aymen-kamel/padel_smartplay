# SmartPlay AI 2.0 — Progress log

Rules in force: only new files under `docs/`; nothing modified, moved or deleted; no downloads, installs, training or long compute; stop after each phase and wait for "continue".

## Decisions (user, 2026-11-13)
- No datasets or training runs exist outside this folder.
- Ground-truth candidate: `data_candidates/Video Project 1.mp4` (highlights compilation, source https://youtu.be/EuKFW3H9qeU, on-screen score increments after points). Phase 1: metadata only. Phase 3: assess what it can and cannot validate.
- Target machine: the measured RTX 3050 **6 GB** (`nvidia-smi`); design budget stays at 4 GB VRAM.
- Positioning: clean rewrite under the user's own name and licence; ideas and public algorithms only; flag GPL-3.0 (Padex), upstream-fork and restrictive-weight dependencies.
- `iit_courtvison/` (basketball) is out of scope; it shares the `smartplay/` venv and upgraded packages on 2026-11-11/12.

## Decisions from the clarification interview (user, 2026-11-14)
| Topic | Decision |
|---|---|
| `padel_analytics_repo` ownership | **Forked** from the public *Padel Analytics* repo, then modified. Only the user's own additions (scoring, shot counter, ball filters, API, UI, tests) are reusable as own code; upstream core stays reference-only. |
| Target camera | **One fixed camera** filming the whole court (mounted high behind one baseline). Broadcast footage = training data / demos only, never the final test. |
| Own footage | **Yes** — the user can record full matches at a club with a fixed camera and note the real score. These become the held-out test set. |
| v1 score scope | **Points → full score**: detect point start/end and point winner, then derive 15/30/40, golden point or advantage, games, sets, changes of ends. Serve faults / outs later. |
| Live vs after-match | **Not answered.** Working assumption: after-match (offline) processing, with stages kept streaming-friendly. To confirm. |

## Phase 1 — Audit: DONE (2026-11-13), awaiting "continue"

Files created (all under `docs/`):
`audit/PROJECT_AUDIT.md` (incl. §6 "Unsupported claims", U01–U19) · `audit/EXPERIMENT_INVENTORY.md` · `audit/MODEL_INVENTORY.md` · `audit/DATASET_INVENTORY.md` · `audit/TECHNICAL_DEBT.md` · `audit/REUSABLE_COMPONENTS.md` · `audit/SCORE_DIAGNOSIS.md` · `PROGRESS.md`.

What was done: read the code, notes, configs, caches and results of every component; header-only inspection of 133 videos; parsing of saved JSON/CSV; a paper replay of all saved hit/bounce logs through the FSM rules. Nothing was executed from the project; throw-away scripts live in the session scratchpad, outside the project.

Key findings:
1. No score has ever been produced; 18 jobs + 1 CLI run: 0–0 or no score. Replay of 19 saved event logs: max 1 consecutive same-side bounce without a hit, 0 points.
2. No ground truth at any stage; no labelled dataset; 39-sample shot table only.
3. Ball track: 65.9 % real detections on the 37 s clip; 380 interpolated frames, gaps up to 61 frames, off-screen positions.
4. Bounce = image-y local maximum; FSM covers one point-ending pattern; no serve/fault/out/side-change logic.
5. Report claims contradicted or unsupported: 92.1 % event agreement, 100 % score consistency, 2.1 % loss rate, P/R/F1/mAP table, 20 matches / 500+ h, `ball_only` 85.2 %/96.9 % vs CSV 65.4 %/77.5 % (full list: PROJECT_AUDIT §6).
6. Licence exposure: upstream fork (no LICENSE, secondary note says CC BY-NC-SA 4.0), Padex GPL-3.0, Ultralytics AGPL-3.0; `ball_only/src/bounce_classifier.py` mirrors Padex code.
7. Environment drift: venv no longer matches `requirements.txt` (`supervision` 0.27 → 0.31.6).
8. Highlights video (header only): 664.8 s, 1920×1080, 30 fps, 19,943 frames, audio present.

Open items carried to Phase 2: confirm upstream licence and every weight licence at the source; confirm padel point rules (FIP); decide how to treat the YouTube source (terms/copyright) for a portfolio project.

Verification: a manifest of 762 existing project files (outside the venv and `node_modules`) was recorded before writing; the re-check after writing is reported in the Phase 1 hand-off message.

## Phase 2 — Research: DONE (2026-11-14), awaiting "continue"

Files created: `research/STATE_OF_THE_ART.md`, `research/MODEL_COMPARISON.md`, `research/RECOMMENDED_TECHNOLOGIES.md`, plus `PROJECT_EXPLAINED.md` (plain-language overview, requested by the user). Web research only; nothing downloaded or installed.

Key findings:
1. FIP rules (official PDF, applicable 01.01.2026) read directly: three game-scoring options (Advantage / Star Point / Golden Point), changes of ends after odd games, ≤ 20 s between points, point-lost conditions (Rule 13).
2. Padel audio hit detection reaches avg F1 92 % (Decorte et al., CVPRW 2024; code/data availability unverified).
3. PadelTracker100 (CC BY 4.0): ~100k annotated frames from one fixed camera — annotations only; frames come from YouTube videos (terms to decide).
4. Permissive alternatives exist for every perception block: RF-DETR N/S (Apache-2.0), RT-DETR (Apache-2.0), ByteTrack / OC-SORT (MIT), TrackNetV3 (MIT), WASB (MIT), RTMPose (Apache-2.0, secondary). Ultralytics is AGPL-3.0.
5. No published padel system doing point-by-point scoring from a fixed camera was found; proposed: probabilistic point winners + constrained score decoding over the FIP score automaton.

Decisions needed before Phase 3: YouTube-sourced data (terms/copyright); live vs after-match (assumed after-match); keep audio in club recordings.

## Phase 3 — Architecture and plan: DONE (2026-11-14), awaiting approval

Files created: `architecture/ARCHITECTURE.md` (principles, data flow, coordinate convention, typed stage interfaces, technology choices, repository layout, mapping of existing files) and `architecture/PLAN.md` (milestones M0–M9 ordered by impact on score accuracy, evaluation protocol with match-level splits, training plan, labelling quantities and formats, highlights-video assessment, compute budget, risks, note on winner prediction).

Nothing implemented, moved, installed or downloaded. Next step needs explicit approval: create the new repository and environment, decide on YouTube-sourced data, confirm after-match processing, record the first match.

## M0 — Foundations: DONE (2026-11-14)

Approved by the user ("Approve, start M0"). New repository at `C:\dev\smartplay` (outside OneDrive), git initialised (nothing committed yet), own Python 3.11 venv with only `pydantic`, `numpy`, `pytest`, `ruff`.

Built (clean-room, no code copied from the forks, Padex or Ultralytics):
- `smartplay.geometry.court` — court constants and the single coordinate convention (metres, origin at net centre, NEAR/FAR).
- `smartplay.schemas.records` — typed, validated records between stages (VideoInfo, PlayerTrack, BallState, Event, Rally, PointOutcome, PointLabel).
- `smartplay.scoring.automaton` — FIP score engine: Advantage / Golden Point / Star Point, sets with 2-game margin, tie-break to 7 (service 1-2-2, ends every 6 points), mini set, match tie-break (7) and super tie-break (10), deciding set without tie-break, service rotation, changes of ends after odd games. Three documented assumptions where the rulebook is silent (in the module docstring).
- `smartplay.evaluation.metrics` — one-to-one event matching with tolerance, P/R/F1, point-winner accuracy, displayed-score accuracy, Wilson 95 % interval.
- `configs/rules/{advantage,golden_point,star_point}.yaml`; `docs/` copied into the repo.

Verification: **45 tests pass**; `ruff` clean. One test shows why constrained decoding matters: a single wrong point winner in the first game (87.5 % point accuracy) leaves the displayed score wrong after more than half of the points.

Not done in M0: the oracle check against a real match (needs recorded matches and point labels, milestone M1); config loading from YAML into `MatchRules` (trivial, added when the pipeline needs it). The original project folder was not modified apart from this log.

Next: **M1** — record the first match with a fixed camera and audio, calibrate, and label points (format L1 in `architecture/PLAN.md`).

## Video baseline and ball detection: DONE (2026-11-14)

New repo `C:\dev\smartplay` (GitHub: aymen-kamel/padel_smartplay). Added after M0: court camera from 12 clicked points, RF-DETR Small + four-player tracker, audio hit candidates, WASB ball detector (MIT, vendored and credited), annotated-video CLI. On `rally.mp4`: ~15 fps, players observed 99.7 %, ball seen 65.9 % (zero-shot tennis weights), 10 audio hit candidates, calibration leave-one-out 14.2 cm. **No stage is measured against labels yet.** Details: `README.md` in the new repo.

## Decision: licence relaxed for experimentation (user, 2026-11-15)
Technical performance outranks licensing during the experiment phase; every component's licence is documented and reviewed before production. Research for the newly allowed options: `docs/research/EXPERIMENT_CANDIDATES.md`. Next: a small labelled benchmark (two highlight clips) so candidates can be compared with real numbers.

## 2026-11-18 - 3D viewer on real data, brainstorm, physics post-processor
- `docs/BRAINSTORM.md`: physics/logic ideas for score from video, ball in 3D + bounces, hits and shot types, tactics (ideas, not results).
- `smartplay.cli export-replay` + `src/smartplay/export/replay.py`: one analysed clip -> compact JSON (players in metres, per-frame distance/speed, audio hits, scoreboard states). Ball in 3D stays null until the 3D calibration exists.
- `viewer/heroscore.html`: the owner's HeroScore 3D page, with the random rally generator replaced by the measured positions (walk-cycle only: no swings or ball invented). Published to the owner's artifact; the previous version is saved in `viewer/original_backup/index.html`.
- `ball/physics.py`: piecewise-parabola post-processor (tested, not wired: no benchmark gain; kinks = high-recall hit candidates).
- CVSPORTS_Padel (2 377 labelled hits, 99 rallies): labels downloaded; video download in progress (per-file, resumable).
