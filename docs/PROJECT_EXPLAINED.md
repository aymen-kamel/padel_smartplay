# SmartPlay AI — the project in plain words

## What you want
Film a padel match with **one fixed camera**, and get the **score automatically** (15/30/40, games, sets) after the match.

## What exists today (Feb–Jun 2026 work)
- `padel_analytics_repo/` — your main pipeline. You **forked** it from a public project and added your own parts: scoring, shot counter, ball filters, API, web UI, tests.
- `ball_only/` — your experiments on ball tracking, bounces and scoring.
- `padex/` — someone else's library (GPL-3.0). Used for experiments only.
- `smartplay/` — the Python environment (7 GB), not code.

## Why the score never worked (in one picture)
```
camera ─► find the ball ─► find bounces ─► decide who won the point ─► score
             ▲ loses it 34%     ▲ confuses floor,       ▲ only one rule        ▲ always 0–0
               of the time        wall and racket          (double bounce)
```
Each step trusts the previous one blindly, so small errors pile up. And nothing was ever compared with the real score, so nobody knew *where* it failed.

## What the research says (Phase 2)
- **Hear the hits**: in padel, racket hits can be found from the sound (a 2024 study reports 92 % F1).
- **Bounces are "corners"** in the ball path, not "lowest points" on screen.
- **Use the rules**: players change ends after games 1, 3, 5…; the server changes every game; the official FIP rules (2026) define Advantage, Star Point and Golden Point. These facts can correct mistakes.
- **A public padel dataset exists**: PadelTracker100 (about 100k frames, one fixed camera, free licence CC BY 4.0).
- **Free-licence models exist for every block** (RF-DETR, OC-SORT, TrackNetV3, WASB), so you can rebuild under your own name.

## The plan in one line
Rebuild cleanly: calibrate once → track the ball and players → hear hits, detect bounces → cut the video into points → guess each point's winner with a confidence → let the padel rules pick the most consistent score → **measure everything against your own recorded matches**.

## Where to read more
- What is broken and why: `docs/audit/SCORE_DIAGNOSIS.md`
- Everything found in the folder: `docs/audit/PROJECT_AUDIT.md` (incl. "Unsupported claims" to fix in your report)
- Best methods today: `docs/research/STATE_OF_THE_ART.md`
- The chosen tool for each step: `docs/research/RECOMMENDED_TECHNOLOGIES.md`
