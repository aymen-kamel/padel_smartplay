# SmartPlay AI 2.0

Padel match analysis from **one fixed camera**. Goal of v1: detect each point and its winner,
then derive the full score (points, games, sets) under the official FIP rules.

Status: milestone **M0** (foundations). Only pure-logic modules exist; no vision model yet.

| Module | Content |
|---|---|
| `smartplay.geometry` | Court constants and the single coordinate convention |
| `smartplay.schemas` | Typed records exchanged between pipeline stages |
| `smartplay.scoring` | FIP score engine (Advantage / Golden Point / Star Point, tie-breaks, changes of ends) |
| `smartplay.evaluation` | Event matching and score metrics |

## Setup
```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pytest
```

Design documents: `docs/` (audit, research, architecture, plan).
This is a clean rewrite: it contains no code from earlier forks or GPL/AGPL projects.
