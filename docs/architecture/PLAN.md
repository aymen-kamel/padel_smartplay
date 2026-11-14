# SmartPlay AI 2.0 — Plan (Phase 3)

Date: 2026-11-14. Status: **proposal; nothing implemented**. Read with `ARCHITECTURE.md`.
All quantities below are **planning targets**, not measurements. Accuracy targets are goals to test, not promises. No expensive training before a baseline exists (rule 5).

## 1. Milestones, ordered by impact on score accuracy

| # | Milestone | Why it comes here | Deliverable | Exit check | Compute |
|---|---|---|---|---|---|
| M0 | **Foundations**: new repo outside OneDrive, clean environment, schemas, coordinate module, **FIP score automaton** with tests, evaluation harness skeleton | The score engine is pure logic: it can be finished and proven before any vision | `scoring/automaton.py` + tests for Advantage / Golden Point / Star Point, tie-break, changes of ends | Unit tests pass; replaying a hand-written point list gives the correct score | CPU, minutes |
| M1 | **Ground truth**: record matches (section 4), label points and scores, calibrate the camera | Nothing can be measured without it (audit cause #1) | 6 matches recorded; point labels for all; `calibration.json` per court | Labels validate against schema; oracle test: labelled point winners → automaton reproduces the real final score of every match | none |
| M2 | **Point segmentation baseline** (audio energy + motion, no deep model) | Splitting the match into points is the first thing scoring needs and the cheapest to try | `rally/` baseline + metric report | Rally boundary F1 measured on dev matches (target ≥ 0.85 at ±1 s) | CPU |
| M3 | **Ball baselines measured**: label ball frames; test WASB tennis weights zero-shot and a TrackNetV3-style model; then fine-tune the better one | Ball quality limits every event (audit: 65.9 % visible) | Ball benchmark table on own footage; fine-tuned model | Ball F1 measured (target ≥ 0.85 at 5 px on test frames); VRAM measured | GPU; fine-tune est. 2–6 h local (**ask before runs > 30 min**) |
| M4 | **Players, teams, side changes** | Needed to attribute a point to a *team* and to anchor game boundaries | `players/` with 4-identity constraint + side-swap detector | IDF1 measured on labelled clips; side-change detection recall measured (target ≥ 0.95) | GPU inference only |
| M5 | **Events**: audio hit detector, trajectory bounce classifier | Turns tracks into the facts the rules need | `events/` + event F1 report | Hit F1 and floor-bounce F1 measured (targets ≥ 0.85 at ±3 frames) | CPU / small GPU, < 30 min |
| M6 | **Point winner + constrained decoding** | Converts noisy events into a consistent score | `scoring/judge.py`, `scoring/decoder.py` | Point-winner accuracy and game-score accuracy on dev matches; ablation with/without anchors | CPU |
| M7 | **End-to-end on held-out matches** | The only number that proves the goal | Evaluation report on the 3 test matches (never used before) | Reported honestly, whatever the value | GPU inference; est. several hours per match (to measure) |
| M8 | API, web UI, analytics, visual output | Product layer; no effect on accuracy | `api/`, `viz/`, `analytics/` | Upload → score timeline → correction screen works | CPU |
| M9 | Later: shot types, serve faults/lets, live mode, automatic calibration | Out of v1 scope | — | — | — |

Decision gates: after **M2** and **M3**, compare measured numbers with the targets. If ball F1 on your camera is far below target, the plan changes (higher camera, 60 fps, or more labels) *before* building events.

## 2. Evaluation protocol

**Splits (no leakage).** Split by **match**, never by frame or clip. Proposed: 3 development matches (training + tuning) and 3 **test matches that are not opened until M7**. If two courts are available, at least one test match is on a court never seen in development. PadelTracker100 and broadcast footage are training-only and never count as test.

| Stage | Metric | Matching rule | Labelled test data |
|---|---|---|---|
| Player detection | mAP@0.5, mAP@0.5:0.95 | COCO | 300 frames |
| Tracking | IDF1, HOTA, ID switches, % frames with 4 correct identities | TrackEval-style | 4 clips × 30 s |
| Ball | Precision / recall / F1 at 5 px (1080p), position RMSE, % frames visible, longest gap | one ball per frame | 1,000 test frames |
| Court | Mean and max reprojection error (px and cm) on **held-out** court points | points not used for fitting | 6 extra points per camera |
| Events | Precision / recall / F1 per type (hit, floor bounce, wall) | ±3 frames (report ±1, ±5 too) | events of 40 test points |
| Rally segmentation | Boundary F1 at ±1 s; point count error | one-to-one | all test points |
| Point winner | Accuracy and Brier score, with 95 % confidence interval | per point | all test points |
| **Score (end to end)** | (a) % points with correct winner; (b) % games with correct winner; (c) **% of time the displayed score is exactly right**; (d) final set scores correct (yes/no per match) | against the labelled score timeline | 3 test matches |

Every table also reports a **trivial baseline** (e.g. "server wins every point", "alternate winners") and the **oracle ladder**: score accuracy with perfect segmentation, with perfect events, with perfect point winners. The ladder shows which stage costs the most accuracy.
With about 450 test points, a measured 80 % point accuracy carries roughly ±4 points of 95 % uncertainty (binomial estimate).

## 3. Training and fine-tuning

| Model | Train from | Data | Local on 4 GB? |
|---|---|---|---|
| Ball heat-map model | TrackNetV3 design; initial weights from own training or permissive public weights | PadelTracker100 ball labels (if frames may be obtained) + 3,000 own frames | Expected yes with small batches (**to measure**) |
| Player detector | RF-DETR Small COCO weights; **no fine-tuning unless measured mAP is poor** | — / 300 own frames | Fine-tuning VRAM unverified → possible **cloud GPU** |
| Audio hit detector | Small CNN on log-mel, from scratch | ~500 labelled hits + negatives | Yes (CPU possible) |
| Bounce classifier | Gradient-boosted trees | ~400 labelled floor bounces + wall contacts + negatives | CPU |
| Point judge calibration | Logistic calibration of rule outputs | dev-match point labels | CPU |
| Not trained in v1 | Pose, shot types, video event spotting (**cloud GPU likely**) | — | — |

## 4. Labelling you must do (exact quantities and formats)

Recording first: **6 full matches**, one fixed camera high behind a baseline (the PadelTracker100 camera was ~7.6 m high and 15.5 m behind the glass; get as high and central as the club allows), 1080p, **60 fps if possible (else 30), audio on**, camera never moved during a match. Write down for each match: scoring option used, who served first, final score.

| # | What | Quantity | Format | Est. effort |
|---|---|---|---|---|
| L1 | **Points** (all 6 matches) | every point (≈ 100–200 per match, estimate) | JSONL, one line per point: `{match, point_idx, t_start, t_end, server_team, winner_team, end_reason, score_after}`; `end_reason` ∈ {double_bounce, wall_or_out, net, fault, other} | ≈ 1.5× match duration |
| L2 | **Side changes and server per game** | every game | derived from L1 fields, checked by eye | included in L1 |
| L3 | **Ball position** | 3,000 dev frames (1,000 per dev match, in 20–30 short bursts of consecutive frames) + 1,000 test frames | CSV `frame,visibility,x,y` (TrackNet style; visibility 0 = not visible) | ≈ 15–20 h |
| L4 | **Events** | all hits, floor bounces and wall contacts in 60 dev points + 40 test points (≈ 500 hits, ≈ 400 bounces, estimate) | JSONL `{match, frame, kind, side, player_id?}` | ≈ 10–12 h |
| L5 | **Players** | 300 frames with 4 boxes (detection) + 4 clips × 30 s with identities (tracking) | COCO JSON / MOT format | ≈ 6 h with interpolation |
| L6 | **Court** | 12 fitting points + 6 held-out points per camera position | JSON pixel ↔ metre pairs | 15 min per court |
| L7 | **Highlights video** (optional, section 5) | every visible point | same schema as L1, `source: broadcast` | ≈ 1 h |

Order: L6 → L1 (enables M1–M2) → L3 (M3) → L5 (M4) → L4 (M5). Label a small pilot (one game) first to test the format.

## 5. What the highlights video can and cannot validate

File: `data_candidates/Video Project 1.mp4` — 664.8 s, 1920×1080, 30 fps, with audio (header only; **content not yet viewed**, so points per minute, cuts and scoreboard layout are UNVERIFIED).

**It can validate**
- Point-level labels: which team won each shown point.
- Rally start/end detection on broadcast footage.
- Possibly automatic labels by reading the scoreboard overlay (OCR at fixed screen position; a score change timestamps a point end) — a cheap way to produce L7.
- Audio hit detection and ball detection as *extra training/dev data*.

**It cannot validate**
- Full-match score tracking: points are selected and not consecutive, so games and sets cannot be reconstructed.
- Serve/fault/out handling across a game, change-of-ends logic, deuce / golden-point sequences.
- Fixed-camera performance: broadcast framing, zooms, cuts and replays differ from your product.

**Biases**: long spectacular rallies are over-represented (short points and errors are under-represented), replays may duplicate points, professional players and lighting only.
**Use**: development and demo material; never the reported test result. Copyright/terms of the YouTube source must be decided before it is used or redistributed.

**Extra footage needed for end-to-end evaluation**: the 6 own full matches above (3 dev + 3 test), ideally ≥ 2 courts, including at least one tie-break or deciding point and several changes of ends. That is roughly 6–9 hours of video. Three test matches give about 300–600 points, enough for ±4–5 point confidence intervals; fewer than 2 test matches would make the result anecdotal.

## 6. Compute and storage budget (estimates; measure in M3/M7)

| Item | Estimate | Note |
|---|---|---|
| Raw video | 6 matches × ~1.25 h × 1080p ≈ 30–60 GB | outside OneDrive; C: had 80 GB free on 2026-11-11 |
| Inference per match | several hours on the RTX 3050 | old pipeline measured 0.14–1.3 FPS; new target ≥ 5 FPS, **unproven** |
| Ball fine-tuning | 2–6 h local | ask before starting (> 30 min rule) |
| Cloud GPU | only if RF-DETR fine-tuning or video event spotting becomes necessary | flagged, not planned for v1 |
| Downloads | PadelTracker100 labels 70 MB; public weights < 500 MB each | each needs licence check; nothing > 2 GB planned |

## 7. Risks

| Risk | Likelihood | Effect | Response |
|---|---|---|---|
| Ball hard to see from the club camera | Medium–high | Events unreliable | Decide at M3 gate: camera position, 60 fps, more labels |
| Poor audio (echo, neighbouring courts) | Medium | Hit detection weaker | Trajectory-only fallback; test on first recording |
| Labelling takes longer than planned | High | Delay | Pilot first; L1 alone already enables M0–M2 and the oracle ladder |
| Point winner ambiguous from one camera | Medium | Score errors | Constrained decoding + human correction screen |
| Data licence (YouTube) | Open | Less training data | Own recordings suffice for testing |

## 8. Optional: winner prediction (honest note)

- **Pre-match prediction** needs a history of many matches with player identities and results (hundreds at least). The project has none, and 6 recorded matches cannot support it. **Not feasible with this data.**
- **Live win probability** is feasible *without machine learning*: once the score state is known, the probability of winning the game/set/match follows from a Markov model of padel scoring with one parameter (probability that the serving pair wins a point). With few matches that parameter is rough, so the output is an illustration, not a calibrated forecast. It depends entirely on score tracking working first.

## 9. Before implementation starts (needs your approval)
1. Create the new repository outside OneDrive and a new environment (installs).
2. Decide on YouTube-sourced data.
3. Confirm after-match processing.
4. Record the first match and run the labelling pilot.
Nothing in the existing folder is moved or deleted until you approve the mapping in `ARCHITECTURE.md` §7.
