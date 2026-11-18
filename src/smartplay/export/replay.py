"""Export one analysed clip as a compact JSON for the 3D replay viewer (viewer/heroscore.html).

Axes: the viewer uses three.js axes, x across the court, y up, z along the court with team A (near the camera) at
z > 0. Our convention has y negative near the camera, so z = -y. The ball is NOT exported in 3D: a floor
homography cannot place a ball in the air (``ball3d`` stays null until the 3D calibration exists).
"""

from __future__ import annotations

import json
from pathlib import Path

from smartplay.analytics.player_stats import PlayerStats
from smartplay.schemas.records import PlayerTrack


def build_replay(tracks: list[dict], hit_frames: list[int], fps: float, n_frames: int, title: str = "",
                 score_before: str | None = None, score_after: str | None = None,
                 ball_seen_ratio: float | None = None) -> dict:
    by_frame: dict[int, dict[int, dict]] = {}
    for t in tracks:
        by_frame.setdefault(t["frame"], {})[t["player_id"]] = t
    ids = sorted({t["player_id"] for t in tracks})
    meta = {t["player_id"]: t for t in tracks}
    stats = PlayerStats(fps)
    players = {pid: {"id": pid, "team": meta[pid]["team"], "xs": [], "zs": [], "observed": [], "dist": [],
                     "speed": []} for pid in ids}
    last: dict[int, tuple[float, float]] = {}
    for f in range(n_frames):
        row = by_frame.get(f, {})
        stats.update([PlayerTrack(**r) for r in row.values()])
        snap_f = stats.snapshot()
        for pid in ids:
            r = row.get(pid)
            if r is not None:
                last[pid] = (r["foot_xy_m"][0], -r["foot_xy_m"][1])
            x, z = last.get(pid, (0.0, 0.0))
            players[pid]["xs"].append(round(x, 3))
            players[pid]["zs"].append(round(z, 3))
            players[pid]["observed"].append(bool(r and r["observed"]))
            players[pid]["dist"].append(round(snap_f[pid]["dist_m"], 1) if pid in snap_f else 0.0)
            players[pid]["speed"].append(round(snap_f[pid]["speed_kmh"], 1) if pid in snap_f else 0.0)
    snap = stats.snapshot()
    return {
        "title": title, "fps": fps, "n_frames": n_frames,
        "players": [players[p] for p in ids],
        "hits": [{"frame": int(f), "source": "audio"} for f in sorted(hit_frames) if f < n_frames],
        "stats": {str(p): {"dist_m": round(snap[p]["dist_m"], 1), "team": snap[p]["team"]} for p in snap},
        "score": {"before": score_before, "after": score_after},
        "ball_seen_ratio": ball_seen_ratio,
        "ball3d": None,
    }


def export_run(run_dir: Path, fps: float, out: Path, title: str = "", score_before: str | None = None,
               score_after: str | None = None) -> dict:
    tracks = json.loads((run_dir / "tracks.json").read_text())
    n = max(t["frame"] for t in tracks) + 1
    hits = [h["frame"] for h in json.loads((run_dir / "audio_hits.json").read_text())]
    ball = json.loads((run_dir / "ball.json").read_text())
    seen = sum(b["visibility"] == "visible" for b in ball) / max(1, len(ball))
    data = build_replay(tracks, hits, fps, n, title, score_before, score_after, round(seen, 3))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    return data
