# SmartPlay AI 2.0

Padel match analysis from **one fixed camera**. Goal of v1: detect each point and its winner,
then derive the full score (points, games, sets) under the official FIP rules.

Status: **M0 done + first video baseline.** The CLI analyses a fixed-camera video: court
calibration (with a leave-one-out error), four players tracked on the court floor (teams,
sides, mini-map) and audio hit candidates. **Not yet:** ball tracking, bounces, points, score
from video. No stage is measured against labels yet (needs milestone M1).

| Module | Content |
|---|---|
| `smartplay.geometry` | Court constants, coordinate convention, camera model from 12 clicked points |
| `smartplay.schemas` | Typed records exchanged between pipeline stages |
| `smartplay.scoring` | FIP score engine (Advantage / Golden Point / Star Point, tie-breaks, changes of ends) |
| `smartplay.ingestion` | Frame reader and audio extraction (ffmpeg) |
| `smartplay.players` | RF-DETR Small person detector + four-player court tracker |
| `smartplay.events` | Audio hit candidates (onset detector, untrained) |
| `smartplay.viz` | Annotated video: court lines, players, mini-map, audio timeline |
| `smartplay.evaluation` | Event matching and score metrics |

## Analyse a video
```
python -m smartplay.cli annotate --video data/raw/rally.mp4 --keypoints data/calibration/rally_court_keypoints.json --out runs/rally
```
Writes `runs/rally/annotated.mp4`, `tracks.json` and `audio_hits.json`. The vision stack
(OpenCV, SciPy, PyTorch with CUDA, `rfdetr`, `imageio-ffmpeg`) is not in the base install yet.
The first run downloads the RF-DETR Small COCO weights (Apache-2.0, 368 MB).

## Setup
```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pytest
```

Design documents: `docs/` (audit, research, architecture, plan).
This is a clean rewrite: it contains no code from earlier forks or GPL/AGPL projects.
