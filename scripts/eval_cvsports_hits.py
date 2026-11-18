"""Evaluate the audio hit detector on the CVSPORTS_Padel rallies (hits labelled by the dataset authors).

    python scripts/eval_cvsports_hits.py [--dir data/external/cvsports_padel] [--out runs/cvsports_audio_hits.json]

Label time = ``start`` (onset, seconds from the rally start) in metadata/hits.csv. Prediction = audio onset time.
One-to-one matching within +-tol seconds (default 0.12 s = 3 frames at 25 fps). Only videos fully downloaded
(a ``<name>.ok`` marker next to the file) are used. Licence of the dataset: not stated (to verify before release).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from smartplay.evaluation.metrics import match_events
from smartplay.events.audio import detect_hits
from smartplay.ingestion.video import extract_audio

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path("data/external/cvsports_padel"))
ap.add_argument("--out", type=Path, default=Path("runs/cvsports_audio_hits.json"))
ap.add_argument("--rel-min", type=float, default=4.0)
ap.add_argument("--label-offset", type=float, default=0.0,
                help="seconds removed from predictions (the dataset marks the START of a ~0.5 s hit window)")
args = ap.parse_args()

labels: dict[str, list[float]] = defaultdict(list)
for r in csv.DictReader(open(args.dir / "metadata/hits.csv", encoding="utf-8")):
    labels[r["filename"]].append(float(r["start"]))

SR = 16000
TOLS = (0.06, 0.12, 0.2, 0.4)
tot = {t: [0, 0, 0] for t in TOLS}
per_video = {}
for f in sorted(labels):
    v = args.dir / "rallies" / f
    if not (v.exists() and (args.dir / "rallies" / f"{f}.ok").exists()):
        continue
    audio = extract_audio(v, SR)
    pred = [h.t - args.label_offset for h in detect_hits(audio, SR, rel_min=args.rel_min)]
    truth = sorted(labels[f])
    row = {"n_pred": len(pred), "n_true": len(truth)}
    for t in TOLS:
        s = match_events(pred, truth, t)
        row[f"f1@{t}"] = round(s.f1, 3)
        tot[t][0] += s.true_positives
        tot[t][1] += s.false_positives
        tot[t][2] += s.false_negatives
    per_video[f] = row

print(f"{len(per_video)} videos, {sum(r['n_true'] for r in per_video.values())} labelled hits, "
      f"{sum(r['n_pred'] for r in per_video.values())} predicted")
summary = {}
for t, (tp, fp, fn) in tot.items():
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    summary[str(t)] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
    print(f"  tol +-{t:4.2f} s: P {p:.2f}  R {r:.2f}  F1 {f1:.2f}")
f1s = sorted(r["f1@0.12"] for r in per_video.values())
print(f"  per-video F1@0.12 s: median {f1s[len(f1s) // 2]:.2f}, worst {f1s[0]:.2f}, best {f1s[-1]:.2f}")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"summary": summary, "per_video": per_video}, indent=1))
