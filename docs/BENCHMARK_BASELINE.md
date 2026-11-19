# Benchmark baseline — first measured numbers (2026-11-15)

Two broadcast clips, labelled by hand with `scripts/label_tool.py`; evaluated with `scripts/eval_benchmark.py`
(raw results: `runs/benchmark_baseline.json`, not in git). Pipeline under test: WASB tennis weights (zero-shot) for the ball,
untrained audio onset detector for hits. **Two clips and ~240 labelled ball frames: these numbers are indicative, not final.**

## Ball (TrackNet convention, 1080p pixels)

| Clip | Visible in labels | Predicted on labelled frames | Median error | P / R / F1 @ 5 px | @ 10 px |
|---|---|---|---|---|---|
| point_12 | 125 / 129 | 78 | 2.5 px | 0.90 / 0.56 / 0.69 | 0.99 / 0.62 / 0.76 |
| point_24 | 146 / 148 | 102 | 3.4 px | 0.65 / 0.45 / 0.53 | 0.77 / 0.54 / 0.64 |
| **All** | | | | **0.76 / 0.50 / 0.60** | **0.87 / 0.58 / 0.69** |

- **When the model finds the ball it is very precise** (median error 2.5-3.4 px).
- **It misses a lot:** the ball is visible in about 98 % of the labelled frames but predicted in only 60-70 %, so recall is the main limit.
- **Clip 24 has 23 predictions more than 50 px off, and they are glass reflections.** In frames 364-383 the predicted x moves right (1614 -> 1693) while the labelled ball moves left (1335 -> 1211) at the same height: the model follows the mirror image of the ball in the right glass wall. Frames 39-54 show the same pattern on the left side.
- Counting those wrong-location predictions as errors is what pulls clip 24 down (F1 0.53 against 0.69).

## Hits (audio onset detector, one-to-one matching)

| Clip | Predicted / labelled | F1 @ +-1 frame | @ +-3 | @ +-5 | @ +-10 |
|---|---|---|---|---|---|
| point_12 | 10 / 11 | 0.76 | **0.86** | 0.86 | 0.86 |
| point_24 | 13 / 13 | 0.23 | 0.46 | 0.62 | 0.69 |
| **All** | | 0.47 | **0.64** | 0.72 | 0.77 |

- Timing is not biased (median offset 0.0-0.5 frames); the misses are wrong or extra sounds, not a lag.
- Clip 24's labels contain three pairs of hits only 3 frames apart (18/21, 258/261, 297/300), which look like double marks. If they are merged (10 hits instead of 13), clip 24 reaches F1 0.52 at +-3 and 0.70 at +-5 frames. **To be confirmed by the labeller**; labels were not modified.

## What this changes
1. The largest measurable gain is on the ball: **reject glass reflections** and **raise recall** (fine-tune on padel footage).
2. Audio alone gives F1 of about 0.64-0.72 on hits; combining it with the ball's direction change and the nearest player's wrist is the next step.
3. Every later change is now compared with these numbers.

## Update 2026-11-16: glass-reflection rejection

Change: the ball tracker now follows the trajectory (constant-velocity prediction, small gate around the
prediction, closest candidate wins; a new track needs 2 consecutive confirming frames) instead of taking the
strongest blob near the last position. The detection threshold was lowered from 0.5 to 0.3 so the tracker sees
more candidates. Code: `track_gated` in `src/smartplay/ball/detector.py`; comparison: `scripts/compare_ball_trackers.py`.

| Ball, both clips | P / R / F1 @ 5 px | P / R / F1 @ 10 px | Predictions in the wrong place (> 10 px) |
|---|---|---|---|
| Baseline (threshold 0.5, strongest blob) | 0.76 / 0.50 / 0.60 | 0.87 / 0.58 / 0.69 | 24 |
| Gated tracker, threshold 0.5 | 0.86 / 0.52 / 0.65 | 0.96 / 0.58 / 0.72 | 6 |
| **Gated tracker, threshold 0.3 (now the default)** | **0.87 / 0.59 / 0.70** | **0.96 / 0.65 / 0.77** | 8 |

Measured end to end (re-run of `smartplay.cli annotate` on both clips, then `eval_benchmark.py`): same numbers.
Clip 24, the one with reflections: F1 @ 5 px 0.53 -> 0.69.

Limits, honestly:
- The tracker's few parameters and the 0.3 threshold were chosen on these same two clips, so the gain may be
  optimistic; it must be confirmed on new labelled clips.
- **Recall is now the hard limit:** even a perfect choice among the model's candidates would find the ball in only
  62-66 % of labelled frames at threshold 0.5. Raising it needs a better detector (fine-tuning on padel footage).

## Update 2026-11-18: WASB vs TrackNet (old project's `ball_detection.pt`), same tracker, same labels

Both detectors give candidate blobs at threshold 0.3; the same `track_gated` picks the ball; same two labelled clips
(342 + 401 frames). Scripts: `scripts/dump_tracknet_blobs.py`, `scripts/dump_ball_blobs.py`, `scripts/compare_ball_trackers.py --blobs ...`.

| Candidates from | P / R / F1 @ 5 px | P / R / F1 @ 10 px | Wrong place (> 10 px) | Candidate ceiling @ 10 px |
|---|---|---|---|---|
| WASB tennis weights (MIT) | 0.87 / 0.59 / 0.70 | 0.96 / 0.65 / 0.77 | 8 | 0.74 / 0.79 |
| TrackNet (old project weights) | 0.70 / 0.56 / 0.62 | 0.92 / 0.75 / 0.82 | 18 | 0.82 / 0.85 |
| **Both, merged (WASB first)** | **0.85 / 0.72 / 0.78** | **0.94 / 0.80 / 0.86** | 14 | 0.86 / 0.91 |

- TrackNet finds the ball more often (higher recall) but places it less precisely and makes more wrong-place picks; WASB is the reverse. Merging their candidates gives the best result (+0.08 F1 at 5 px, +0.09 at 10 px).
- Merge rule (one trial, not tuned): add a TrackNet candidate if no WASB candidate is within 12 px, with its score x 0.9.
- Caveats: two clips from one broadcast video; TrackNet's background is the median of the clip itself (fine for a fixed camera, to re-check on cuts); the TrackNet weights are **the official TrackNetV3 checkpoint** (qaz812345/TrackNetV3, MIT; verified identical to the official download by SHA-256 df867641a02712b0..., InpaintNet too; trained on badminton, not padel); the model class is still loaded from the old project path, not copied; TrackNet costs more GPU/time than WASB (not measured here).
- Not yet wired into `cli annotate`.

## Update 2026-11-18: physics-based tracking, first results (negative)

Candidates: fused WASB + TrackNetV3 (above). Two clips, 743 labelled frames; parameters not tuned except the kink sweep.

| Tracker | F1 @ 5 px | F1 @ 10 px |
|---|---|---|
| `track_gated` (default) | **0.78** | **0.86** |
| Image-space Kalman filter (projectile model, learned vertical acceleration), no smoothing | 0.75 | 0.81 |
| Same + RTS smoothing + gap filling | 0.56 | 0.80 |
| `track_gated` + piecewise-parabola post-processor (`ball/physics.py::refine_physics`) | 0.77 | 0.86 |

- Why the Kalman variant lost: measured points are already precise (median error 2.5-3.4 px), so smoothing only moves them;
  undetected contacts put several flights into one arc; the gate drops candidates after each contact. (Code deleted; numbers kept here.)
- The post-processor fills gaps on the fitted curve: on clip 24, 0 of the 4 labelled filled frames were within 10 px,
  and clip 12 gained 1 frame. **No position gain, so it is not wired into `cli annotate`.** The module stays (tested) for the 3D step.
- **Kinks (flight breaks) as hit candidates:** with the default tolerance there are 67-72 kinks per ~340-frame clip for 11-13
  labelled hits (precision 0.14-0.2) but recall is high: 83 % of labelled hits have a kink within +-3 frames, 92 % within +-5.
  Keeping only turns >= 60 degrees gives F1 0.45 (+-3) / 0.53 (+-5). Audio only: 0.64 / 0.72. Audio AND kink: 0.65 / 0.74;
  audio OR strong kink: 0.56 / 0.60. So hand-made rules do not beat audio; kinks are useful as **candidate generator
  with high recall for a learned classifier** (ball, player and wrist features, trained on the 2 377 labelled hits).
- What physics still needs: a **3D model** (gravity in metres, camera pose from heights), see the plan; the image-space
  parabola is only a local approximation under perspective.

## Update 2026-11-18: physics prediction mode (`cli annotate --predict-ball N`)
When the ball is not detected, the projectile path of the current flight fills short gaps and is extended up to N frames
after its last detection; these points are drawn as a hollow cyan ring labelled "pred" and the top bar says
"PREDICTED (physics)". They are estimates: measured on the two labelled clips, the predicted frames that have a label were
within 10 px of the true ball in only 2 of 9 cases (N = 4) and 2 of 12 (N = 6); the rest follow a flight that had
already ended (hit, bounce or glass). Off by default; use it to see what the physics expects, not as a measurement.

## Update 2026-11-18: audio hit detector on 81 other videos (CVSPORTS_Padel rallies)

Dataset: 99 rallies from 9 tournaments, labelled by its authors (`scripts/eval_cvsports_hits.py`; 81 videos downloaded so
far, 1 902 labelled hits, 1 643 predicted). Untrained detector, same default parameters as on our two clips.

| Match tolerance | P | R | F1 |
|---|---|---|---|
| +-0.06 s | 0.26 | 0.23 | 0.24 |
| +-0.12 s (3 frames) | 0.63 | 0.54 | 0.58 |
| +-0.20 s | 0.88 | 0.76 | 0.81 |
| +-0.12 s after removing a constant label offset of 0.09 s | 0.89 | 0.77 | **0.82** |

- The dataset's hit time is the **start of a ~0.5 s window around the hit**, not the impact: our audio onset falls a median
  +0.091 s after it (p10 +0.03, p90 +0.19), so the raw numbers under-state the detector. The offset is a property of these
  labels, not a bias to correct in the detector (our own labels mark the impact frame).
- Per video, F1 at +-0.12 s: median 0.78, worst 0.03, best 1.00 -> some videos (noisy audio, music, crowd) fail badly.
- On our two hand-labelled clips the F1 was 0.64 at +-3 frames; on 81 other videos it is 0.82 (offset-corrected): the first
  figure was pessimistic, but this is a different labelling convention, so the two are not directly comparable.

### Final numbers on all 99 CVSPORTS rallies (download complete, 2.1 GB; 2 377 labelled hits, 2 029 predicted)
| Tolerance | Raw F1 | F1 with the 0.09 s label offset removed (`--label-offset 0.09`) |
|---|---|---|
| +-0.06 s | 0.22 | 0.60 |
| +-0.12 s | 0.55 | **0.80** (P 0.87, R 0.75) |
| +-0.20 s | 0.79 | 0.85 |
Per video (offset-corrected, +-0.12 s): median 0.86, worst 0.23, best 1.00. The offset (0.09 s) was estimated on the first 81 videos and
applied to all 99, so the 0.80 is slightly optimistic. These numbers replace the 81-video ones above.

## Update 2026-11-18: ball coverage on 8 other videos (one rally per tournament, first 500 frames, no labels)
`scripts/ball_on_videos.py`. "Seen" = share of frames where the tracker reports a ball. It measures **coverage, not accuracy**:
more detections can also mean more wrong ones (no ball labels exist for these videos).

| Video | WASB alone | WASB + TrackNetV3 |
|---|---|---|
| VIGO_08 | 64.0 % | 77.6 % |
| BRUSSEL_10 | 60.0 % | 76.4 % |
| VALLADOLID_02 | 71.4 % | 79.8 % |
| VALENCIA_03 | 60.8 % | 81.0 % |
| MALAGA_04 | 73.6 % | 88.6 % |
| FINLAND_07 | 82.4 % | 83.8 % |
| MADRID_05 | 83.6 % | 82.4 % |
| DUITSLAND_02 | 68.0 % | 69.4 % |
| **Mean** | **70.5 %** | **79.9 %** |

The fusion raises coverage in 7 of 8 videos (+9.4 points on average); in MADRID_05 it is slightly lower (-1.2). Overlay videos:
`runs/ball_tests/`. To know whether the extra detections are right, some of these videos need hand-labelled ball positions.

## Update 2026-11-19: learned audio hit detector (`events/learned.py`, `scripts/train_hit_classifier.py`)
Candidates = onset peaks at a low threshold (5 914 over 99 rallies, covering 2 184 of 2 377 labelled hits = 92 % candidate
recall at +-0.12 s); a gradient-boosted classifier (scikit-learn `HistGradientBoostingClassifier`) scores each from acoustic
context (5 band energies after the onset and their contrast with before, centroid, flatness, decay, spacing to neighbours,
local loudness); non-maximum suppression at 0.25 s. **Leave-one-tournament-out** (11 groups): the scored rallies' tournament is
never in training. Same matching as before (+-0.12 s, label time + 0.09 s).

| Detector | P | R | F1 |
|---|---|---|---|
| Untrained (`audio.detect_hits`) | 0.87 | 0.75 | 0.80 |
| Learned, threshold 0.3 | 0.83 | 0.86 | **0.85** |
| Learned, threshold 0.4 | 0.85 | 0.84 | **0.85** |
| Learned, threshold 0.5 | 0.87 | 0.81 | 0.84 |

- Gain: +0.05 F1, mostly recall (0.75 -> 0.84-0.86). The threshold was chosen on the same held-out scores (optimistic by ~1 point).
- Most useful features: low-band (300-1000 Hz) contrast around the onset, spacing to the previous/next onset, neighbourhood strength.
- Ceiling: 8 % of labelled hits have no onset candidate within 0.12 s (soft hits masked by noise): audio alone cannot go above ~0.92 recall here.
- Not yet combined with the ball trajectory (kinks) or the players; that needs the vision features on these videos
  (about 90 min of compute for the 99 rallies: to be confirmed before running).

## Update 2026-11-19: adding the ball trajectory to the learned hit detector (`scripts/train_hit_classifier_vision.py`)
Ball track (WASB + TrackNetV3 fused, gated, kinks) computed on all 99 rallies (~3 h on the RTX 3050). For each audio candidate:
ball seen before/after, speed before/after, turn angle, vertical flip, distance to and turn of the nearest kink. Same
leave-one-tournament-out folds and matching as the audio model. Players were not used (no court calibration for these videos).

| Features | P | R | F1 (best threshold, 0.4) |
|---|---|---|---|
| Audio only | 0.85 | 0.84 | 0.847 |
| Ball only (on the audio candidates) | 0.69 | 0.74 | 0.716 |
| Audio + ball | 0.86 | 0.85 | **0.851** |

**The ball adds almost nothing (+0.004 F1)**, although it carries real signal on its own (0.72). Reasons, measured or likely:
the candidates are audio onsets, so the 8 % of hits without a sound peak are still unreachable; when there is a clear sound,
the audio already decides; the 2D pixel trajectory is noisy around contacts (the ball is often lost right at the hit).
The audio features stay the most useful ones. What could still help: ball-based candidates (kinks) for silent hits, the
hitter's position/wrist (needs players and pose), and a 3D trajectory.

## Update 2026-11-19: pose features for the hit detector (`scripts/pose_hit_features.py`)
Pose (yolo26m-pose from the earlier project, Ultralytics **AGPL-3.0: experiment only**, to replace before release) on the frame of
each of the 5 914 audio candidates: nearest wrist to the ball (divided by the person's height), second nearest, distance to the
nearest person box, wrist above shoulder, number of people. Same leave-one-tournament-out folds.

| Features | F1 (best threshold) |
|---|---|
| Audio | 0.847 |
| Pose only | 0.661 (P 0.56, R 0.80) |
| Audio + ball | 0.851 |
| Audio + ball + pose | 0.851 |

Pose carries signal on its own but nothing the audio does not already say: no gain. The limit is the candidate set (audio onsets
only), not the features; next test: candidates from the ball trajectory.

## Update 2026-11-19: hit candidates from the ball trajectory (`scripts/ball_candidates_eval.py`)
Candidates = audio onsets **plus** ball-trajectory kinks (kink frame + 0 or + 4 frames). Audio features are recomputed over all
candidates of a video; ball features without the kink-distance columns (they would leak: a kink candidate is always at distance 0).
Same leave-one-tournament-out folds; A and B are computed by the same code, so they compare fairly with each other.

| Candidates | Candidate ceiling (labels with a candidate within 0.12 s) | F1 (best threshold) |
|---|---|---|
| A: audio onsets only | 0.92 | 0.828 |
| B: audio + kinks (+0 frames) | **0.98** | 0.838 |
| B: audio + kinks (+4 frames) | 0.96 | **0.847** |

- Ball kinks do reach the silent hits: the ceiling rises from 92 % to 96-98 %. The classifier converts only part of it: +0.010 to
  +0.019 F1 (the +4 frame offset was chosen on these same results, so +0.019 is optimistic).
- Variant A (0.828) is lower than the earlier 0.851 because this script drops the kink-distance features; the best overall
  number is still ~0.85. The remaining errors need better evidence at the candidate (3D ball, hitter's wrist trajectory over
  several frames), not more candidates.

## Update 2026-11-19: 3D camera and first ball-in-3D fits (clip 24)
**Camera** (`geometry/camera3d.py`, DLT): 12 floor points + 11 height points clicked by the owner with `scripts/calibrate_3d.py`
(a first attempt had the height points clicked on the floor: camera 511 m under the floor, rejected; the tool now refuses
such clicks). Result: reprojection 5.2 px, leave-one-out 6.6 px; camera at x 0.08 m, y -26.0 m (16 m behind the near baseline),
height 8.1 m. Largest residuals: the two net-post tops (15 px). The 3D court drawn back on the frame sits on the lines,
the net top and the back glass (`data/calibration/highlights_main_view_3d_check.png`).

**Ball** (`ball/ballistic3d.py`): a flight grows while one projectile (g = 9.81 m/s^2, no drag) explains the 2D detections
(rms <= 6 px); fits that are implausible (under the floor, outside the court + 1.5 m, faster than 160 km/h) are dropped.
On clip 24 (401 frames, ball seen in 317): **5 flights, 59 frames in 3D (15 %)**, speeds 17-90 km/h, one flight ending on
the floor, one at the back glass. Without the speed cap, 3 more flights came out at 193-209 km/h: not credible, a sign of the
one-camera depth ambiguity on short flights (8 frames = 0.27 s: the ball drops only ~36 cm, a few pixels).
One kept flight ends 1.3 m behind the back glass: also suspect.

Not validated: no 3D ground truth exists. What should make it reliable: start each flight at the hitter (players are known in
metres), chain flights through floor bounces (z = 0 and restitution link two arcs), and longer flights.

### With physical priors (player at the start of a hit, bounce continuation)
Each window is fitted under three hypotheses: free; starting within reach of a player (players' floor positions in metres from
the tracker, soft penalty beyond 1.3 m, start height 0.2-3 m); continuing a floor bounce (starts where the previous flight
landed). A bounce fit is preferred; a player-anchored fit is kept when it reprojects within 1 px of the free fit.

| Clip | Flights | Frames in 3D | Anchors | Flight ends | Frames > 0.3 m outside the walls |
|---|---|---|---|---|---|
| point_24 | 14 | 129 / 401 (**32 %**, was 15 %) | player 13, free 1 | floor 1, glass 1, other 12 | 5 |
| point_12 | 13 | 210 / 342 (**61 %**) | player 8, free 4, bounce 1 | floor 4, glass 1, other 8 | 2 |

Speeds 6-153 km/h. Still suspect: flights at 144-153 km/h (close to the 160 km/h cap), a slow flight (7-11 km/h at 2.3 m)
at the end of clip 12 that is probably the ball held or bounced by a player after the point, and the few frames behind the walls.
Not validated against 3D ground truth (none exists); the same calibration was used for both clips (same broadcast camera).

## Update 2026-11-19: pose models on the same 76 frames (`scripts/pose_bakeoff.py`)
Clips 12 and 24, every 10th frame. A tracked player counts as covered when a skeleton's ankles fall within 1.5 m of it on
the floor. **Coverage only: no pose labels exist, so keypoint accuracy is not measured.**

| Model | Licence | Near players covered | Far players covered | Extra skeletons on court | Time per frame |
|---|---|---|---|---|---|
| YOLO26m-pose, 1280 px + 2.5x far-half pass | Ultralytics AGPL-3.0 (experiment only) | 98.7 % | 99.3 % | 2 | 0.12 s (GPU) |
| RTMPose-m + YOLOX-m (rtmlib "balanced") | Apache-2.0 | 99.3 % | 98.0 % | 5 | 0.63 s (CPU) |
| RTMPose-x + YOLOX-x (rtmlib "performance") | Apache-2.0 | 99.3 % | **99.3 %** | 2 | 1.85 s (CPU) |

All three find almost every player. RTMPose-x matches YOLO26 on coverage and false skeletons with a permissive licence; it was
run on CPU here (no onnxruntime-gpu installed), so its speed on the RTX 3050 is not measured. Without the far-half pass YOLO26 at
640 px found only the near players. Next: label wrists/elbows on ~30 frames to measure accuracy before choosing for good.

## Update 2026-11-19: checking the 3D ball against the hand labels (`scripts/check_ball3d_vs_labels.py`)
**Image consistency** (3D point projected back with the camera, compared with the hand-labelled pixel; the labels are
independent of the detector and of the 3D fit): 128 of the 271 labelled visible frames have a 3D position (47 %).
Median error 3.8 px, 96 % within 10 px, 99 % within 20 px, worst 70 px; the 2D tracker on the same frames: median 2.2-2.3 px.
No 3D position was placed on a frame labelled "ball not visible".

**This does not validate depth**: a point moved along the camera ray projects to the same pixel. Two depth checks:
- frames where the 2D tracker missed the ball but a 3D flight covers it: only **1** labelled frame (error 2 px): too few to
  say anything (flights are grown on detections and rarely bridge gaps);
- **floor contacts**: at a bounce the ball is at z = 0, so its pixel gives its true floor position through the floor
  homography, independently of the 3D fit. 6 flights end near the floor: horizontal gap between the 3D landing point and that
  floor point 0.36-2.19 m (median 0.93 m). Only one flight ends really at z ~ 0 (-0.03 m), with a 0.36 m gap; two end
  ~0.3 m UNDER the floor; for the others (end height 0.25-0.57 m) the floor projection itself is off by up to ~1-2 m at that
  distance, so their gap mixes both errors.

Conclusion: the 3D curves fit the images well, but the depth is uncertain by roughly 0.5-1 m on these clips; 3D speeds and
landing points must be shown as estimates, not measurements. The fix that targets exactly this: make the floor contact a hard
constraint (z = 0 at the landing frame, the next flight starting there), then re-run this check.

### Floor bounce as a hard constraint (`ball/ballistic3d.py::join_bounces`)
Two consecutive flights (gap <= 8 frames) are refitted as one path through a floor point B: z = 0 exactly at the bounce instant,
shared point, vz_out = -e * vz_in with the restitution e left free (0.3-1.0) as an independent physical check.
Separate fits are NOT used to choose candidates: they often disagree at a contact (clip 12, frame 30: one ends 0.28 m under the
floor, the next starts at 2.69 m two frames later), which is the depth error itself.

First version (any close pair, accepted if it reprojects within 6 px): 6 "bounces", but 4 of them were physically wrong
(e stuck on the 0.3 / 1.0 bounds, or horizontal speed x2.55 after the "bounce", i.e. a racket hit): with free depth the model
can push almost any contact to the floor. Acceptance now also requires 0.45 <= e <= 0.95 and no more than a 2x horizontal gain
(the horizontal ratio is poorly determined: 1.30 measured for a true 0.81 on a simulated bounce; a test covers the simulated case:
landing point within 0.3 m, e within 0.12).

| Clip | Bounce | e (FIP ball ~0.74) | 3D landing vs floor point of the nearest hand label | Before (separate fits) |
|---|---|---|---|---|
| point_12 | frame 32.1 | **0.71** | **0.17 m** (label 0.9 frame away) | flight ended 0.28 m under the floor, 1.13 m off |
| point_24 | frame 159.3 | 0.49 | **0.05 m** (label 0.3 frame away) | not detected as a bounce |

Image consistency unchanged (median 3.8 px, 95 % within 10 px, 131 labelled frames). Remaining: 5 frames of clip 12 still under
the floor (flights not joined), and only 2 bounces pass the physics check over 743 frames, so most flights still have free depth.
The landing-point agreement is partly built in (B is fitted to the pixels around the contact); the restitution 0.71 is the more
independent sign that the first bounce is real.
