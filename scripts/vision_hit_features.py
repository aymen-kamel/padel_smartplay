"""Fused 2D ball track and trajectory kinks for every CVSPORTS rally (input of train_hit_classifier_vision.py).

    python scripts/vision_hit_features.py   -> runs/vision_hit_features.json (one entry per video, resumable)

Ball only: players are not computed (these videos have no court calibration). ~1.5-2.5 min per rally on an RTX 3050.
"""

import json
from pathlib import Path

from smartplay.ball.detector import WasbBallDetector, track_gated
from smartplay.ball.fusion import merge_blobs
from smartplay.ball.physics import refine_physics
from smartplay.ball.tracknet import TrackNetBallDetector
from smartplay.ingestion.video import iter_frames, probe

DIR = Path("data/external/cvsports_padel/rallies")
OUT = Path("runs/vision_hit_features.json")
wasb = WasbBallDetector(Path("weights/wasb/wasb_tennis_best.pth.tar"), threshold=0.3)
tn = TrackNetBallDetector(Path("weights/tracknetv3_official/ckpts/TrackNet_best.pt"), threshold=0.3)
res = {}
if OUT.exists():
    res = json.loads(OUT.read_text())
videos = sorted(p for p in DIR.glob("*.mp4") if (DIR / f"{p.name}.ok").exists())
for k, v in enumerate(videos):
    if v.name in res:
        continue
    info = probe(v)
    a = wasb.blobs(iter_frames(v), info.width, info.height)
    b = tn.blobs(iter_frames(v), info.width, info.height)
    st = track_gated(merge_blobs(a, b))
    st, kinks = refine_physics(st, merge_blobs(a, b))
    res[v.name] = {"fps": info.fps, "n": len(st),
                   "ball": [[s.frame, s.uv[0] if s.uv else None, s.uv[1] if s.uv else None,
                             s.visibility] for s in st],
                   "kinks": [[kk.frame, kk.turn_deg, kk.speed_before, kk.speed_after] for kk in kinks]}
    if k % 10 == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res))
        print(f"{k + 1}/{len(videos)}", flush=True)
OUT.write_text(json.dumps(res))
print("done", len(res))
