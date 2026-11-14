"""Pass 1 over a broadcast video: cuts, main-view detection, scoreboard OCR (cached to disk).

    python scripts/segment_broadcast.py --video "data/raw/highlights.mp4" --out runs/highlights
"""

import argparse
from pathlib import Path

from smartplay.broadcast.segment import extract_features

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--ocr-every", type=int, default=15)
args = ap.parse_args()
print(extract_features(args.video, args.out, args.ocr_every))
