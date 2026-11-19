"""Train and evaluate the learned audio hit detector on CVSPORTS_Padel with leave-one-tournament-out folds.

    python scripts/train_hit_classifier.py [--offset 0.09] [--tol 0.12]

Positive candidate = the nearest onset candidate within ``tol`` s of a labelled hit (label time + offset, because the
dataset marks the START of a hit window). Folds are tournaments (9), so no rally of a tournament is seen in training
when it is scored. Compared with the untrained detector (``audio.detect_hits``) under the same matching.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from smartplay.evaluation.metrics import match_events
from smartplay.events.audio import detect_hits
from smartplay.events.learned import FEATURE_NAMES, candidates, features, suppress
from smartplay.ingestion.video import extract_audio

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path("data/external/cvsports_padel"))
ap.add_argument("--offset", type=float, default=0.09)
ap.add_argument("--tol", type=float, default=0.12)
ap.add_argument("--cache", type=Path, default=Path("runs/hit_features.npz"))
ap.add_argument("--out", type=Path, default=Path("runs/hit_classifier.json"))
args = ap.parse_args()
SR = 16000

labels: dict[str, list[float]] = defaultdict(list)
with open(args.dir / "metadata/hits.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    labels[r["filename"]].append(float(r["start"]) + args.offset)
names = [f for f in sorted(labels) if (args.dir / "rallies" / f"{f}.ok").exists()]

if args.cache.exists():
    z = np.load(args.cache, allow_pickle=True)
    X, y, vid, times, base = z["X"], z["y"], z["vid"], z["times"], z["base"].item()
else:
    Xs, ys, vids, ts = [], [], [], []
    base = {}
    for k, f in enumerate(names):
        audio = extract_audio(args.dir / "rallies" / f, SR)
        t_c, rel_c, rel_all, dt = candidates(audio, SR)
        base[f] = [h.t for h in detect_hits(audio, SR)]
        if len(t_c) == 0:
            continue
        X_f = features(audio, SR, t_c, rel_c, rel_all, dt)
        y_f = np.zeros(len(t_c), int)
        used: set[int] = set()
        for h in sorted(labels[f]):  # one-to-one: each label takes its nearest unused candidate
            d = np.abs(t_c - h)
            for j in np.argsort(d):
                if d[j] > args.tol:
                    break
                if int(j) not in used:
                    y_f[j] = 1
                    used.add(int(j))
                    break
        Xs.append(X_f)
        ys.append(y_f)
        vids += [k] * len(t_c)
        ts.append(t_c)
        if k % 20 == 0:
            print(f"  features {k + 1}/{len(names)}")
    X, y, vid, times = np.vstack(Xs), np.concatenate(ys), np.array(vids), np.concatenate(ts)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, X=X, y=y, vid=vid, times=times, base=np.array(base, dtype=object))

group = np.array([names[v].split("_")[1] for v in vid])  # tournament = LOCATION
tournaments = sorted(set(group))
n_hits = sum(len(v) for v in labels.values())
print(f"{len(names)} videos, {len(y)} candidates, {int(y.sum())} positives of {n_hits} labelled hits "
      f"(candidate recall {y.sum() / n_hits:.2f}), {len(tournaments)} tournaments")

score = np.zeros(len(y))
for tn in tournaments:
    te = group == tn
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                         class_weight="balanced", random_state=0)
    clf.fit(X[~te], y[~te])
    score[te] = clf.predict_proba(X[te])[:, 1]


def evaluate(thr: float):
    tp = fp = fn = 0
    for k, f in enumerate(names):
        m = vid == k
        t_c, s_c = times[m], score[m]
        sel = np.where(s_c >= thr)[0]
        keep = sel[suppress(t_c[sel], s_c[sel])] if len(sel) else sel
        s = match_events(list(t_c[keep]), sorted(labels[f]), args.tol)
        tp += s.true_positives
        fp += s.false_positives
        fn += s.false_negatives
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


tp = fp = fn = 0
for f in names:
    s = match_events(base[f], sorted(labels[f]), args.tol)
    tp += s.true_positives
    fp += s.false_positives
    fn += s.false_negatives
p0, r0 = tp / max(1, tp + fp), tp / max(1, tp + fn)
print(f"untrained detector      : P {p0:.2f} R {r0:.2f} F1 {2 * p0 * r0 / (p0 + r0):.2f}")
res = {}
for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
    p, r, f1 = evaluate(thr)
    res[thr] = (round(p, 3), round(r, 3), round(f1, 3))
    print(f"learned, thr {thr:.1f}      : P {p:.2f} R {r:.2f} F1 {f1:.2f}")
best = max(res, key=lambda t: res[t][2])
print("note: the threshold is picked on the same held-out scores (optimistic by a point or two)")

clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                     class_weight="balanced", random_state=0).fit(X, y)
from sklearn.inspection import permutation_importance

pi = permutation_importance(clf, X[::3], y[::3], n_repeats=3, random_state=0, scoring="roc_auc")
top = sorted(zip(FEATURE_NAMES, pi.importances_mean, strict=True), key=lambda t: -t[1])[:8]
print("most useful features:", ", ".join(f"{n} ({v:.3f})" for n, v in top))
args.out.write_text(json.dumps({"untrained": [round(p0, 3), round(r0, 3)], "learned": {str(k): v for k, v in res.items()},
                                "best_thr": best, "n_candidates": len(y), "n_pos": int(y.sum()),
                                "n_hits": n_hits, "top_features": [(n, round(float(v), 4)) for n, v in top]}, indent=1))
