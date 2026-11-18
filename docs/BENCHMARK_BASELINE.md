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
