# Where we are and what comes next (hand-over file)

Written 2026-11-15 so that any session — yours or a new Claude session — can continue without the chat history.
Repo: `C:\dev\smartplay` (GitHub: aymen-kamel/padel_smartplay). Python env: `C:\dev\smartplay\.venv`.
Vision runs use the older env `C:\Users\dell\OneDrive\Bureau\pfe\smartplay\Scripts\python.exe` with `PYTHONPATH=C:\dev\smartplay\src`
(PyTorch + CUDA + RF-DETR are only there; the new `.venv` has OpenCV, RapidOCR, pytest).

## Decisions taken
- One fixed camera (a second camera is an open question, see below). Score tracking is still the main goal; match analytics come after.
- Licences relaxed for experiments, reviewed before production (`docs/research/EXPERIMENT_CANDIDATES.md`).
- Work only on the two highlight clips `point_24` and `point_12` (in `runs/highlights/clips/`) until a benchmark exists.

## Done
Score engine (FIP rules, 65 tests), court calibration for two views, players + 4-identity tracking, WASB ball tracking (zero-shot tennis weights), audio hit candidates, broadcast tools (cuts, scoreboard OCR), annotated videos, label store and label tool.

## Your task now: label the benchmark (about 2-3 hours, can be done in several sittings)
Run, one clip at a time, from a terminal:
```
cd C:\dev\smartplay
.venv\Scripts\python scripts\label_tool.py --video "runs\highlights\clips\point_24_t0558s_4-6-1-0_vs_6-3-0-15.mp4" --out data\labels\point_24.json
.venv\Scripts\python scripts\label_tool.py --video "runs\highlights\clips\point_12_t0280s_4-0_vs_5-40.mp4" --out data\labels\point_12.json
```
Keys are listed at the top of `scripts/label_tool.py` and in the window. In short: click the ball, `X` if it is not visible, `H` on the frame where a racket touches the ball, `N` for the next frame (every 3rd), `Q` to quit. Everything is saved at every action; run the same command again to continue.
Rules for consistent labels: click the centre of the ball; if only a blur streak is visible, click its middle; ball hidden behind a player or the net = `X`; mark a hit only on the frame closest to the racket-ball contact.
Estimated size: about 135 ball frames and 10-15 hits per clip.

## Claude's next steps (after the labels exist)
1. `scripts/eval_ball.py`: compare `runs/point_XX/ball.json` with `data/labels/point_XX.json` (F1 at 5 px, recall, false tracks on glass).
2. `scripts/eval_hits.py`: compare `runs/point_XX/audio_hits.json` with the labelled hits (F1 at +-3 frames).
3. Bake-off per `EXPERIMENT_CANDIDATES.md`: ball (WASB zero-shot vs fine-tuned vs TrackNetV3), players (RF-DETR vs YOLO26 vs D-FINE on far players), pose, tracking. Record speed and peak VRAM for every candidate.
4. Port from the old repo: glass-reflection filter, torso-colour re-identification, 3D court model (`ball_only/src/court_homography.py`), wrist-proximity hit assignment.
5. Then bounces, point segmentation, point winner, full score from video.

## Open questions for the user
- Allowed to download the two PadelTracker100 source videos from YouTube for fine-tuning the ball model?
- One camera or two? (A commercial product, Padmi, uses two cameras installed on the court.)
- Is cloud GPU acceptable for fine-tuning, or only the RTX 3050 6 GB?
