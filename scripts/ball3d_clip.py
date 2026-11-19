"""Fit the ball in 3D on one analysed clip and print a plausibility report.

    python scripts/ball3d_clip.py --run runs/point_24_fused --calib data/calibration/highlights_main_view_3d.json --fps 30

Reads <run>/ball.json (2D track), writes <run>/ball3d.json (flights + per-frame xyz). No 3D ground truth exists: the
report checks physics only (heights, speeds, where flights end), not accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from smartplay.ball.ballistic3d import fit_flights, join_bounces, states_with_3d
from smartplay.ball.physics import refine_physics
from smartplay.geometry.camera3d import Camera3D
from smartplay.schemas.records import BallState

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--calib", type=Path, required=True)
ap.add_argument("--fps", type=float, required=True)
args = ap.parse_args()

cal = json.loads(args.calib.read_text())
P = np.array(cal["P"])
cam = Camera3D(P, cal["reproj_px"], cal["loo_px"], 0, cal["n_height_points"])
states = [BallState(**b) for b in json.loads((args.run / "ball.json").read_text())]
_, kinks = refine_physics(states)
players: dict[int, list[tuple[float, float]]] = {}
tracks_file = args.run / "tracks.json"
if tracks_file.exists():
    for r in json.loads(tracks_file.read_text()):
        players.setdefault(r["frame"], []).append(tuple(r["foot_xy_m"]))
flights = fit_flights(states, cam, args.fps, players=players or None)
flights, bounces = join_bounces(states, flights, cam, args.fps)
for bo in bounces:
    print(f"  BOUNCE at frame {bo.t_frame:6.1f}: floor point ({bo.xy[0]:+.2f}, {bo.xy[1]:+.2f}) m, restitution {bo.restitution:.2f} "
          f"(FIP ball ~0.74), horizontal speed kept {bo.horiz_ratio:.2f}, joint fit {bo.rms_px:.1f} px "
          f"(separate fits {bo.rms_before_px:.1f} px)")
seen = sum(s.visibility == "visible" for s in states)
covered = sum(fl.f1 - fl.f0 + 1 for fl in flights)
print(f"{len(states)} frames, ball seen in {seen}; {len(kinks)} kinks; {len(flights)} flights fitted in 3D "
      f"covering {covered} frames ({100 * covered / len(states):.0f} %)")
for fl in flights:
    path = np.array([fl.at(f, args.fps) for f in range(fl.f0, fl.f1 + 1)])
    print(f"  frames {fl.f0:3d}-{fl.f1:3d} ({fl.n_obs:2d} obs, {fl.rms_px:4.1f} px)  "
          f"start ({path[0][0]:+5.1f},{path[0][1]:+6.1f},{path[0][2]:4.1f}) m  end ({path[-1][0]:+5.1f},{path[-1][1]:+6.1f},"
          f"{path[-1][2]:4.1f}) m  apex {path[:, 2].max():4.1f} m  speed {fl.speed_kmh(fl.f0, args.fps):5.0f} -> "
          f"{fl.speed_kmh(fl.f1, args.fps):5.0f} km/h  ends: {fl.end_contact}  [{fl.anchor}]")
out = states_with_3d(states, flights, args.fps)
(args.run / "ball3d.json").write_text(json.dumps({
    "fps": args.fps,
    "flights": [{"f0": fl.f0, "f1": fl.f1, "p0": fl.p0.tolist(), "v0": fl.v0.tolist(), "rms_px": fl.rms_px,
                 "n_obs": fl.n_obs, "end_contact": fl.end_contact, "anchor": fl.anchor} for fl in flights],
    "bounces": [{"t_frame": bo.t_frame, "xy": bo.xy, "restitution": bo.restitution, "horiz_ratio": bo.horiz_ratio,
                 "rms_px": bo.rms_px} for bo in bounces],
    "xyz": [list(s.xyz_m) if s.xyz_m else None for s in out]}))
print(f"wrote {args.run / 'ball3d.json'}")
