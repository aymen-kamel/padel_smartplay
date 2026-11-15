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
