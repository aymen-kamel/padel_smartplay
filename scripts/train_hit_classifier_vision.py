"""Add ball-trajectory features to the learned audio hit detector and compare, leave-one-tournament-out.

    python scripts/train_hit_classifier_vision.py

Needs runs/hit_features.npz (train_hit_classifier.py) and runs/vision_hit_features.json (vision_hit_features.py).
Vision features per audio candidate, from the fused 2D ball track (pixels) around the candidate's frame:
ball seen before/after, speed before/after, turn angle between mean velocities, vertical-direction flip, distance (frames)
to the nearest trajectory kink and that kink's turn. No player features (not computed for these videos).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from smartplay.evaluation.metrics import match_events
from smartplay.events.learned import FEATURE_NAMES, suppress

DIR = Path("data/external/cvsports_padel")
OFFSET, TOL, W = 0.09, 0.12, 5
labels: dict[str, list[float]] = defaultdict(list)
with open(DIR / "metadata/hits.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        labels[r["filename"]].append(float(r["start"]) + OFFSET)
names = [f for f in sorted(labels) if (DIR / "rallies" / f"{f}.ok").exists()]
z = np.load("runs/hit_features.npz", allow_pickle=True)
X_a, y, vid, times = z["X"], z["y"], z["vid"], z["times"]
vis = json.loads(Path("runs/vision_hit_features.json").read_text())

VN = ["seen_before", "seen_after", "speed_before", "speed_after", "turn_deg", "vy_flip", "kink_dist", "kink_turn"]


def mean_vel(pts):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None
    (f0, u0, v0), (f1, u1, v1) = pts[0], pts[-1]
    return np.array([u1 - u0, v1 - v0]) / max(1, f1 - f0)


X_v = np.zeros((len(y), len(VN)))
for i in range(len(y)):
    d = vis[names[vid[i]]]
    fr = int(round(times[i] * d["fps"]))
    ball = d["ball"]
    def pt(k):
        if 0 <= k < len(ball) and ball[k][1] is not None:
            return (k, ball[k][1], ball[k][2])
        return None
    before = [pt(k) for k in range(fr - W, fr)]
    after = [pt(k) for k in range(fr + 1, fr + W + 1)]
    vb, va = mean_vel(before), mean_vel(after)
    sb, sa = (float(np.linalg.norm(vb)) if vb is not None else -1), (float(np.linalg.norm(va)) if va is not None else -1)
    turn = -1.0
    flip = -1.0
    if vb is not None and va is not None and sb > 0.5 and sa > 0.5:
        turn = float(np.degrees(np.arccos(np.clip(vb @ va / (sb * sa), -1, 1))))
        flip = float(np.sign(vb[1]) != np.sign(va[1]))
    kd, kt = 30.0, -1.0
    for kf, kturn, _, _ in d["kinks"]:
        if abs(kf - fr) < kd:
            kd, kt = abs(kf - fr), kturn
    X_v[i] = [sum(p is not None for p in before) / W, sum(p is not None for p in after) / W, sb, sa, turn, flip, kd, kt]

group = np.array([names[v].split("_")[1] for v in vid])


def run(X, label):
    score = np.zeros(len(y))
    for tn in sorted(set(group)):
        te = group == tn
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                             class_weight="balanced", random_state=0).fit(X[~te], y[~te])
        score[te] = clf.predict_proba(X[te])[:, 1]
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6):
        tp = fp = fn = 0
        for k, f in enumerate(names):
            m = vid == k
            t_c, s_c = times[m], score[m]
            sel = np.where(s_c >= thr)[0]
            keep = sel[suppress(t_c[sel], s_c[sel])] if len(sel) else sel
            s = match_events(list(t_c[keep]), sorted(labels[f]), TOL)
            tp, fp, fn = tp + s.true_positives, fp + s.false_positives, fn + s.false_negatives
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {label:18s} thr {thr:.1f}: P {p:.2f} R {r:.2f} F1 {f1:.3f}")
        if best is None or f1 > best[2]:
            best = (p, r, f1, thr)
    return best


print(f"{len(names)} videos, {len(y)} candidates, ball seen around {np.mean(X_v[:, 0] > 0):.0%} of candidates")
res = {"audio": run(X_a, "audio"), "ball": run(X_v, "ball only"), "audio+ball": run(np.hstack([X_a, X_v]), "audio + ball")}
for k, (p, r, f1, thr) in res.items():
    print(f"BEST {k:11s}: F1 {f1:.3f} (P {p:.2f} R {r:.2f}, thr {thr})")
clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                     class_weight="balanced", random_state=0).fit(np.hstack([X_a, X_v]), y)
from sklearn.inspection import permutation_importance

pi = permutation_importance(clf, np.hstack([X_a, X_v])[::3], y[::3], n_repeats=3, random_state=0, scoring="roc_auc")
top = sorted(zip(FEATURE_NAMES + VN, pi.importances_mean, strict=True), key=lambda t: -t[1])[:8]
print("most useful:", ", ".join(f"{n} ({v:.3f})" for n, v in top))
Path("runs/hit_classifier_vision.json").write_text(json.dumps(
    {k: [round(v[0], 3), round(v[1], 3), round(v[2], 3), v[3]] for k, v in res.items()}, indent=1))
