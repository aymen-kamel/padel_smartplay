# Score diagnosis — where score tracking breaks

Date: 2026-11-13. Read-only. Tags: **MEASURED** (artefact), **CODE** (`file:line`), **UNVERIFIED**.
Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.

## 1. The chain, as implemented

```
video ──► ball tracking ──► filtering/interpolation ──► bounce flags ┐
   │                                                                  ├─► PadelScorer FSM ─► score
   ├─► players (4 slots) ──► pose ──► hit detection ───────────────────┘
   └─► 12 clicked court points ──► homography ──► metres, side (top/bottom)
```

Entry: `MAIN/main.py` → `run_analysis.py:119` → `trackers/runner.py:337`. All trackers run over the full video first; the per-frame loop (`runner.py:113-317`) then feeds `PadelScorer.process_frame` (`analytics/scoring.py:74`).

## 2. Stage by stage: what goes in, what can go wrong, evidence

| # | Stage | What it does | Failure modes (evidence) |
|---|---|---|---|
| 1 | Ball detection | TrackNet heat-maps → one (x, y) per frame | Only **65.9 %** of frames hold a real detection on the 37.2 s clip; benchmark on `rally.mp4`: 65.4 % (TrackNet), 77.5 % (+Inpaint), 30.0 % (YOLO). *(MEASURED: `MAIN/cache/ball_detections.json`, `BO/output/benchmark_results.csv`)* |
| 2 | Filtering | median gate 70 px, speed gate 35 px/frame, quadratic interpolation without gap limit (`ball_tracker.py:501-569`) | Missing detections are (0,0) and enter the rolling median (`predict.py:202`, `ball_tracker.py:523-526`); 49 gaps, longest 61 frames (2 s), filled with invented positions; off-screen positions (y = −137); 52 NaN positions. *(MEASURED/CODE)* |
| 3 | Bounce | local maximum of image-y, prominence 4 px, gap ≥ 0.45 s (`ball_tracker.py:17-65`, call `:756-771`) | No notion of floor vs racket vs wall vs jitter. 19 flags on the 37 s clip, all on real-detection frames; 5–7 on `rally.mp4` depending on run. Truth unknown. *(CODE/MEASURED; accuracy UNVERIFIED)* |
| 4 | Projection | ground-plane homography on the ball pixel (`projected_court.py:441-458`) | Valid only if the ball is on the floor. For an airborne ball the projected point lies behind its true floor position (parallax). The court side is decided from this point (`scoring.py:130`). *(CODE; magnitude UNVERIFIED)* |
| 5 | Court filter | drops bounces outside court ± 0.2 m (`runner.py:226-240`, `court_bounds.py`) | 17 of the 19 flags on the 37 s clip reach the FSM (2 dropped by the filter or by a missing projection). Whether airborne wall rebounds survive the filter depends on camera geometry: **UNVERIFIED**. *(MEASURED count)* |
| 6 | Hit detection | direction/speed change over 10 frames + pixel distance to player/wrist (`shot_counter.py:70-141`) | 14 hits in 12.96 s and 41 in 37.2 s; thresholds are 1080p pixels; event lags ~5 frames; held "placeholder" players (7.6 % of boxes) count as hitters; pose is available in 53 % of frames. Accuracy UNVERIFIED. |
| 7 | Hitter side | sign of the player's projected y, else team label from ID (`scoring.py:109-115`) | IDs are fixed to the side seen at frame 0 (`players_tracker.py:450-455`); the fallback is wrong after a change of ends. *(CODE)* |
| 8 | FSM | hit → reset counter; same-side bounce twice → other side wins (`scoring.py:154-190`) | See section 3. |
| 9 | Score | integer counter; label saturates at "Game" (`scoring.py:204-207`) | No games, sets, deuce, golden point, tie-break, side change. |

## 3. The FSM: what it can and cannot score

Rule implemented: *after a hit, if the ball bounces twice in a row on the same side of the net with no hit in between, the other side wins the point.* (`scoring.py:166-190`)

**Valid in principle** — a second floor bounce on one side ends the point in padel, even if the ball touched a wall in between (rule wording to be confirmed from the FIP rules in Phase 2).

**Not handled** (CODE — none of these exist in the file):
- serve (service box, diagonal, fault, second serve, let);
- ball out of play / hits the fence or glass before touching the floor on the receiving side;
- ball into the net; ball hitting a player; double hit;
- point-ending events that are **not** a double bounce — in padel most points end with a smash/volley into the glass, an unreturned ball off a wall, or an error; these produce no "two same-side bounces";
- game, set and match scoring (`PadelScorer`), deuce / golden point, tie-break, change of ends;
- uncertainty: every event is treated as certain; a single false hit resets the bounce counter (`:166-170`).

`BO/src/score.py` adds a time filter (0.3 s, 50 px) and a points/games/sets class (`PadelScore:9-70`, deuce yes; golden point, tie-break and side change no), but it never saw hits or players, and its court coordinates have the opposite sign convention (`cy<0.5` = BOTTOM, `:123`).

## 4. Measured trace: replaying every saved event log

Method: for each saved `shot_events.json` (18 API jobs on `rally.mp4`, plus `MAIN/cache/shot_events.json` for the 37 s clip) the hit and bounce events were merged by frame (hits first on equal frames, as in `scoring.py:83-152`) and passed through a transcription of `_update_fsm`. This reads saved files; it does not run the pipeline. Script: scratchpad `replay_fsm.py` (not part of the project).

| Log set | Hits | Bounces | Max consecutive same-side bounces without a hit | Points awarded |
|---|---|---|---|---|
| 18 API jobs (same 324-frame clip, different code versions) | 14 each | 5–7 | **1** in every log | **0** in every log |
| 37.2 s clip (`cache`) | 41 | 17 | **1** | **0** |

Reading it:
1. There are 2.0–2.8 hits for each bounce; a hit always falls between two bounces, so the counter can never reach 2. The rule can fire only if the **last hit of a point is missed** *and* two same-side bounces are detected. Neither has ever happened.
2. This does **not** prove the clips contain a finished point. Whether `rally.mp4` (12.96 s, 25 fps) ends a point is **UNVERIFIED** (needs a look at the video). What is proven: the system has never been shown to award a single correct point.
3. Timestamps in the logs are wrong for 25 fps clips: `timestamp_sec = frame / 30` is hard-coded (`analytics/shot_classifier/output.py:15,26`); `rally.mp4` is 25 fps, so reported times are 17 % too small. `confidence` is always 1.0 and `ball_speed_px` always 0.0 (`scoring.py:103-105`).
4. The run history shows the bounce detector being retuned: the same clip gave 5, 6 or 7 bounces in different jobs.

## 5. How errors compound

Every stage consumes the previous stage's output as if it were true, and none returns a confidence:

`missing ball (34 %)` → `interpolated position (invented)` → `phantom or shifted bounce` → `wrong side` → `FSM counter wrong`
`+ hit missed or invented` → `counter reset or never reset`

Illustration (arithmetic, **not measured**): if a point needs 8 correct events in sequence and each is 90 % right, the chance all are right is 0.9⁸ ≈ 43 %; at 80 % per event it is ≈ 17 %. The real per-event accuracies are unknown — measuring them is the first job of the new evaluation protocol.

The errors start at stage 1–2 (ball visibility and interpolation) and are amplified at stage 3 (bounce semantics). Stage 8 (the FSM) is the *least* of the problems for the double-bounce rule itself, but it has no way to score the other point endings.

## 6. Ranked causes (hypotheses to test in Phase 3, not fixes)

| Rank | Cause | Evidence level |
|---|---|---|
| 1 | No ground truth, so nothing has ever been measured per stage | MEASURED (none on disk) |
| 2 | Bounce = image-y maximum; cannot separate floor bounce, racket hit, wall rebound | CODE |
| 3 | Ball track lost 34 % of the time and refilled by unbounded interpolation | MEASURED |
| 4 | FSM covers one point-ending pattern; no serve/fault/out; no rally segmentation | CODE |
| 5 | Hit detector: pixel thresholds, lag, placeholder players, pose gaps | CODE/MEASURED |
| 6 | IDs tied to side at frame 0; no team identity across a change of ends | CODE |
| 7 | Ground-plane homography applied to airborne ball; fixed camera assumed | CODE |
| 8 | Hard-coded fps (30) in timestamps; resolution-dependent thresholds | CODE |

## 7. UNVERIFIED

- Whether any analysed clip contains a finished point; what really happens at the clip ends.
- Accuracy of each stage (hits, bounces, hitter side, positions).
- Size of the parallax error for airborne ball positions.
- Whether wall rebounds or racket hits are the main false bounces. In job `66bfa3e6` (7 logged bounces) four lie at y = 6.75–8.42 m (near-side back zone, behind the service line), two within 1.6 m of the net (y = 0.66, 1.54) and one on the far side; six of seven are labelled "bottom". Plausibility cannot be judged without watching the clip.
- The padel point rules used above (checked against the FIP rulebook in Phase 2).
