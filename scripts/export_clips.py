"""Export each main-view shot of a broadcast as its own clip, named with the scoreboard state.

    python scripts/export_clips.py --video "<video>" --run runs/highlights
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from smartplay.ingestion.video import ffmpeg_exe

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True)
args = ap.parse_args()

segs = [s for s in json.loads((args.run / "segments.json").read_text()) if s["kind"] == "main"]
timeline = json.loads((args.run / "scoreboard_timeline.json").read_text())
out = args.run / "clips"
out.mkdir(exist_ok=True)


def label(t: float) -> str:
    cur = None
    for row in timeline:
        if row["t"] <= t:
            cur = row
    if cur is None:
        return "score-unknown"
    top, bot = cur["top"], cur["bottom"]
    return re.sub(r"[^0-9A-Za-z-]", "", "-".join(top)) + "_vs_" + re.sub(r"[^0-9A-Za-z-]", "", "-".join(bot))


for i, s in enumerate(segs, 1):
    name = f"point_{i:02d}_t{int(s['t_start']):04d}s_{label(s['t_start'] + 1)}.mp4"
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{s['t_start']:.2f}", "-i", str(args.video),
           "-t", f"{s['t_end'] - s['t_start'] + 1 / 30:.2f}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "24", "-c:a", "aac", str(out / name)]
    subprocess.run(cmd, check=True)
    print(name)
