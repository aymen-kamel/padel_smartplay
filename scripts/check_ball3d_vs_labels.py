"""Check the 3D ball against the hand labels (2D pixels) of the benchmark clips.

    python scripts/check_ball3d_vs_labels.py --clips point_12 point_24

For every labelled frame where the 3D fit gives a position, the 3D point is projected back with the camera P and compared
with the labelled ball pixel. The 2D tracker is compared on the same frames as a baseline. The labels were drawn by hand
on the image, independently of the detector and of the 3D fit, so this is a real (if small) accuracy test.
Limits: only frames inside fitted flights are scored; the camera P came from height clicks, not from the ball.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from smartplay.geometry.camera3d import project
from smartplay.labeling.store import ClipLabels

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=["point_12", "point_24"])
ap.add_argument("--calib", type=Path, default=Path("data/calibration/highlights_main_view_3d.json"))
args = ap.parse_args()

P = np.array(json.loads(args.calib.read_text())["P"])
tot = {"n3": 0, "err3": [], "err2": [], "lab_vis": 0, "fit_vis": 0, "fit_on_invisible": 0}
for clip in args.clips:
    lab = ClipLabels.load(f"data/labels/{clip}.json").ball
    run = Path(f"runs/{clip}_fused")
    b3 = json.loads((run / "ball3d.json").read_text())
    ball2 = json.loads((run / "ball.json").read_text())
    xyz = b3["xyz"]
    e3, e2 = [], []
    for f, g in lab.items():
        if g is None:
            if f < len(xyz) and xyz[f] is not None:
                tot["fit_on_invisible"] += 1
            continue
        tot["lab_vis"] += 1
        if f >= len(xyz) or xyz[f] is None:
            continue
        tot["fit_vis"] += 1
        p = project(P, np.array([xyz[f]]))[0]
        e3.append(float(np.hypot(*(p - np.array(g)))))
        b = ball2[f]
        if b["uv"] is not None:
            e2.append(float(np.hypot(b["uv"][0] - g[0], b["uv"][1] - g[1])))
    e3, e2 = np.array(e3), np.array(e2)
    print(f"{clip}: 3D scored on {len(e3)} labelled frames")
    if len(e3):
        print(f"  3D reprojection error  median {np.median(e3):5.1f} px   <=5 px {np.mean(e3 <= 5):.0%}   "
              f"<=10 px {np.mean(e3 <= 10):.0%}   <=20 px {np.mean(e3 <= 20):.0%}")
    if len(e2):
        print(f"  2D tracker, same frames: median {np.median(e2):5.1f} px   <=5 px {np.mean(e2 <= 5):.0%}   "
              f"<=10 px {np.mean(e2 <= 10):.0%}")
    tot["err3"] += list(e3)
    tot["err2"] += list(e2)
e3 = np.array(tot["err3"])
print(f"ALL: labelled & visible {tot['lab_vis']}, 3D available on {tot['fit_vis']} of them "
      f"({100 * tot['fit_vis'] / max(1, tot['lab_vis']):.0f} %), 3D placed on a frame labelled 'not visible': "
      f"{tot['fit_on_invisible']}")
if len(e3):
    print(f"ALL 3D: median {np.median(e3):.1f} px, <=10 px {np.mean(e3 <= 10):.0%}, <=20 px {np.mean(e3 <= 20):.0%}, "
          f"worst {e3.max():.0f} px")
