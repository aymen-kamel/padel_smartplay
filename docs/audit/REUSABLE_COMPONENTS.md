# Reusable components (for a clean rewrite under the user's own licence)

Date: 2026-11-13. Read-only. Decision recorded 2026-11-13: **clean rewrite, own name, own licence. Do not continue on the forks. Reuse ideas and public algorithms only; flag every GPL-3.0 or upstream-fork dependency; flag restrictive weight licences.**
Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.
Licence statuses are **UNVERIFIED** until Phase 2 (they come from file headers, `pyproject.toml`, and a secondary note in `iit_courtvison/RESEARCH_REPORT.md:58`).

## How to read the status column
- **OWN** — written by the user (or with an AI assistant for the user); can be moved or re-used, after a quick check that no third-party code is embedded.
- **IDEA** — third-party or upstream-derived; take the *idea or public algorithm* and re-implement from a paper/spec. **Do not copy the code.**
- **NO** — do not carry over (licence, quality, or not needed).

## A. Own work worth carrying forward

| Component | Where | Status | What to take | Change needed |
|---|---|---|---|---|
| Evaluation approach | `MAIN/evaluate_pipeline.py` | OWN | Majority-class baseline next to every score; confusion matrix; refusing to report precision/recall without ground truth; tolerance-window event matching (`:263-290`) | Generalise to all stages; match-level splits; confidence intervals |
| Court bounds helper | `MAIN/analytics/court_bounds.py` (35 lines) | OWN | Margin-aware in-court test with a documented tolerance | Single coordinate convention |
| KPI formulas | `MAIN/analytics/match_report.py` + `tests/test_match_report.py` (35 tests) | OWN | Plausible-speed capping, net-presence %, performance score structure | Smooth positions first; fix distance/speed consistency (C13) |
| FSM test style | `MAIN/tests/test_scoring_fsm.py`, `test_shot_counter.py` | OWN | Specifications by synthetic event sequences ("hit resets counter", "events between points ignored") | Re-target at the new rule engine |
| Full point/game/set/deuce class | `BO/src/score.py:9-70` (`PadelScore`) | OWN | Points → games → sets bookkeeping and display | Add golden point, tie-break, change of ends, serve order; verify rules against FIP in Phase 2 |
| 3D court model | `BO/src/court_homography.py:405-` (`CourtHomography3D`), `calibrate_court_3d.py` | OWN (AI-assisted) | `solvePnP` camera pose, ray–plane intersection for floor vs glass | Validate numerically; use for height-aware ball position |
| Job API shape | `MAIN/api_server.py` (`/jobs`, `/jobs/{id}`, `/results`, `/analytics`, `/ratings`) | OWN | Resource design and status flow | Rebuild on typed schemas; add streaming and cancellation |
| Rally pre-segmentation idea | `the_best/paddle_rally_extractor.py:53-72` | OWN | Wide-shot filter (court-colour ratio + Hough lines) to cut broadcast footage; HSV jump to detect a change of court | Replace magic numbers; test on the highlights video |
| React dashboard layout | `MAIN/padel-ui/src/` (1,808 lines) | OWN (UNVERIFIED quality) | Screens/flows | Rebuild against the new API |
| Report structure | `pfe_latex/`, `rapport/` | OWN | Chapter skeleton, French prose | Remove or verify every unsupported claim (PROJECT_AUDIT §6) |
| Kalman coasting idea | `BO/src/kalman_tracker.py` (78 lines) | OWN | Constant-acceleration model with a **coasting limit** (20 frames) — the opposite of unlimited interpolation | Rewrite with proper gating; compare against modern trackers |

## B. Ideas to re-implement (do not copy code)

| Idea | Where it appears | Licence flag | Take |
|---|---|---|---|
| Schema-first typed data contracts per stage (Pydantic), hierarchical IDs (set/game/point/shot), ball visibility states (visible / occluded / inferred) | `padex/CLAUDE.md`, `padex/src/padex/schemas/` | **GPL-3.0** | The *design principles* only; write own schemas |
| Pluggable strategy interfaces (detector, tracker, event spotter) | `padex/src/padex/tracking/*.py`, `events/*.py` | **GPL-3.0** | Pattern only |
| Bounce features for a small MLP (speed, direction change, nearest-player distance) | `padex/.../events/bounce.py:344-450` | **GPL-3.0** | Feature ideas; derive own from physics |
| Surface taxonomy (ground, back/side wall, fence, corner, net) | `padex/.../bounce.py:166-214`; `BO/src/bounce_classifier.py:29-50, 274-334` | **GPL-3.0** — `BO` version mirrors it | Taxonomy only; **do not carry `bounce_classifier.py`** |
| Pause-based rally boundaries | `padex/.../events/point.py:60-112` | **GPL-3.0** | Weak baseline; replace with audio/vision cues |
| TrackNet heat-map ball tracker with frame stacks and background channel | `MAIN/trackers/ball_tracker/*` | **Upstream, UNVERIFIED** (looks derived from the public TrackNetV3 reference) | Re-implement or fine-tune from the paper's public code under its own licence (Phase 2) |
| 12-keypoint fixed court model and mini-court projection | `MAIN/analytics/projected_court.py` | Upstream | Concept only; own calibration and checks |
| Rule-based shot typing from pose and ball dynamics (overhead, vertical velocity) | `MAIN/analytics/shot_classifier/rules.py` | OWN-modified over upstream shot-counter | Features list only, after a labelled dataset exists |
| 17-keypoint wrist–ball proximity window as shot feature | `MAIN/analytics/shot_feature_extractor.py` | OWN-modified | Feature idea |
| Static 4-slot player manager with colour appearance | `MAIN/trackers/players_tracker/players_tracker.py` | OWN-modified over upstream | Idea only; modern trackers + re-ID are the replacement candidates (Phase 2) |

## C. Do not carry over

| Item | Reason |
|---|---|
| Any file untouched since 2026-03-27 in `MAIN` (`trackers/tracker.py`, `ball_tracker/{dataset,iterable,models}.py`, `keypoints_tracker/`, `utils/`, `visualizations/`, `constants/court_dimensions.py`) | Presumed upstream code; licence UNVERIFIED (possibly NC-SA) |
| All of `padex/` and any `padex` import | GPL-3.0; copying would make the new project GPL-3.0 |
| `BO/src/bounce_classifier.py` | Similarity to Padex's GPL module (PROJECT_AUDIT C09) |
| `BO/src/utils/*` | Copies of upstream utils |
| Weights M01, M04, M07–M11 | Upstream/NC or GPL-project or Ultralytics AGPL-3.0 weights; see MODEL_INVENTORY |
| SAM2 per-frame segmentation | Not needed for scoring; heavy |
| Both Streamlit apps, `OUTER/` APIs, `frontend/`, `web_site/` | Superseded duplicates |
| LSTM trainer, RF pickles | No data, no result; retrain after labelling |
| Docker files | Mismatched build contexts; rebuild |

## D. Third-party building blocks to evaluate in Phase 2 (nothing downloaded)
Detectors/trackers/pose models and their **licences** (Ultralytics AGPL-3.0 vs Apache/MIT alternatives), TrackNet-family and WASB-style ball trackers, event spotting incl. audio, court keypoint models, padel/tennis datasets. Each will be checked at its primary source before it appears in `RECOMMENDED_TECHNOLOGIES.md`.

## Summary
The reusable value is mostly **method and test discipline**, not code: honest evaluation, scoring bookkeeping, 3D camera geometry, job-API shape, a broadcast pre-segmentation idea, and a set of small own modules. Everything that does the actual perception (ball, players, pose, court) is upstream-derived, third-party, or unmeasured, and must be rebuilt or replaced.

## UNVERIFIED
- Exact upstream licence and exactly which files of `MAIN` are upstream (no upstream diff locally).
- Whether `CourtHomography3D` is numerically correct (never validated).
- Whether the "OWN" files embed third-party snippets.
