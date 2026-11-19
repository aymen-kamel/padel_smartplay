"""Hit candidates from the BALL track (trajectory kinks) added to the audio onset candidates; leave-one-tournament-out.

    python scripts/ball_candidates_eval.py [--offsets 0 4]

Variant A: audio onset candidates only (same features as train_hit_classifier_vision.py, recomputed on the same union code path).
Variant B: audio candidates + ball kinks (frame + offset). Features: audio context computed over ALL candidates of the
video (so neighbour spacing is consistent), plus ball context. Kink-distance features are dropped (a kink candidate
would always have distance 0: a leak). Positives: one-to-one nearest candidate within +-0.12 s of each label
(label time = start + 0.09 s, as before).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from smartplay.evaluation.metrics import match_events
from smartplay.events.learned import candidates, features, suppress
from smartplay.ingestion.video import extract_audio

ap = argparse.ArgumentParser()
ap.add_argument("--offsets", type=int, nargs="+", default=[0, 4], help="frames added to each kink frame")
args = ap.parse_args()

DIR = Path("data/external/cvsports_padel")
OFFSET, TOL, W, SR = 0.09, 0.12, 5, 16000
labels: dict[str, list[float]] = defaultdict(list)
with open(DIR / "metadata/hits.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        labels[r["filename"]].append(float(r["start"]) + OFFSET)
names = [f for f in sorted(labels) if (DIR / "rallies" / f"{f}.ok").exists()]
vis = json.loads(Path("runs/vision_hit_features.json").read_text())


def mean_vel(pts):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None
    (f0, u0, v0), (f1, u1, v1) = pts[0], pts[-1]
    return np.array([u1 - u0, v1 - v0]) / max(1, f1 - f0)


def vis_feats(d, t):
    fr = round(t * d["fps"])
    ball = d["ball"]

    def pt(k):
        if 0 <= k < len(ball) and ball[k][1] is not None:
            return (k, ball[k][1], ball[k][2])
        return None

    before = [pt(k) for k in range(fr - W, fr)]
    after = [pt(k) for k in range(fr + 1, fr + W + 1)]
    vb, va = mean_vel(before), mean_vel(after)
    sb = float(np.linalg.norm(vb)) if vb is not None else -1.0
    sa = float(np.linalg.norm(va)) if va is not None else -1.0
    turn, flip = -1.0, -1.0
    if vb is not None and va is not None and sb > 0.5 and sa > 0.5:
        turn = float(np.degrees(np.arccos(np.clip(vb @ va / (sb * sa), -1, 1))))
        flip = float(np.sign(vb[1]) != np.sign(va[1]))
    return [sum(p is not None for p in before) / W, sum(p is not None for p in after) / W, sb, sa, turn, flip]


# build the candidate table once per (video, offset)
rows = {off: {"t": [], "aud": [], "xa": [], "xv": [], "y": [], "vid": []} for off in args.offsets}
for k, f in enumerate(names):
    audio = extract_audio(DIR / "rallies" / f, SR)
    t_a, rel_a, rel_all, dt = candidates(audio, SR)
    kinks = [kk[0] for kk in vis[f]["kinks"]]
    fps = vis[f]["fps"]
    for off in args.offsets:
        t_k = (np.array(kinks, float) + off) / fps if kinks else np.zeros(0)
        t_k = t_k[t_k < len(audio) / SR]
        t_u = np.concatenate([t_a, t_k])
        is_aud = np.concatenate([np.ones(len(t_a)), np.zeros(len(t_k))]).astype(bool)
        order = np.argsort(t_u, kind="stable")
        t_u, is_aud = t_u[order], is_aud[order]
        if len(t_u) == 0:
            continue
        rel_u = rel_all[np.clip(np.round(t_u / dt).astype(int), 0, len(rel_all) - 1)]
        xa = features(audio, SR, t_u, rel_u, rel_all, dt)
        xv = np.array([vis_feats(vis[f], t) for t in t_u])
        y = np.zeros(len(t_u), int)
        used: set[int] = set()
        for h in sorted(labels[f]):
            for j in np.argsort(np.abs(t_u - h)):
                if abs(t_u[j] - h) > TOL:
                    break
                if int(j) not in used:
                    y[j] = 1
                    used.add(int(j))
                    break
        R = rows[off]
        R["t"].append(t_u)
        R["aud"].append(is_aud)
        R["xa"].append(xa)
        R["xv"].append(xv)
        R["y"].append(y)
        R["vid"].append(np.full(len(t_u), k))
    if k % 10 == 0:
        print(f"  {k + 1}/{len(names)}", flush=True)

group_of = {k: f.split("_")[1] for k, f in enumerate(names)}
n_hits = sum(len(v) for v in labels.values())
tournaments = sorted(set(group_of.values()))


def evaluate(off, use_kinks: bool):
    R = {key: np.concatenate(val) for key, val in rows[off].items() if key in ("t", "aud", "xa", "xv", "y", "vid")}
    keep = np.ones(len(R["y"]), bool) if use_kinks else R["aud"]
    t, vid, y = R["t"][keep], R["vid"][keep], R["y"][keep]
    X = np.hstack([R["xa"][keep], R["xv"][keep][:, :6]])
    grp = np.array([group_of[v] for v in vid])
    # candidate ceiling: labels with at least one candidate within TOL
    covered = 0
    for k, f in enumerate(names):
        m = vid == k
        covered += sum(bool(np.any(np.abs(t[m] - h) <= TOL)) for h in labels[f])
    score = np.zeros(len(y))
    for tn in tournaments:
        te = grp == tn
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                             class_weight="balanced", random_state=0).fit(X[~te], y[~te])
        score[te] = clf.predict_proba(X[te])[:, 1]
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6):
        tp = fp = fn = 0
        for k, f in enumerate(names):
            m = vid == k
            t_c, s_c = t[m], score[m]
            sel = np.where(s_c >= thr)[0]
            kept = sel[suppress(t_c[sel], s_c[sel])] if len(sel) else sel
            s = match_events(list(t_c[kept]), sorted(labels[f]), TOL)
            tp, fp, fn = tp + s.true_positives, fp + s.false_positives, fn + s.false_negatives
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if best is None or f1 > best[2]:
            best = (p, r, f1, thr)
    return best, covered / n_hits, len(y), int(y.sum())


print(f"{len(names)} videos, {n_hits} labels")
out = {}
(bp, br, bf, bt), cov, n, npos = evaluate(args.offsets[0], False)
print(f"A audio candidates only      : F1 {bf:.3f} (P {bp:.2f} R {br:.2f}, thr {bt}), candidate ceiling {cov:.2f}, rows {n}")
out["A_audio"] = [round(bf, 3), round(cov, 3)]
for off in args.offsets:
    (bp, br, bf, bt), cov, n, npos = evaluate(off, True)
    print(f"B audio + kinks (+{off} fr)   : F1 {bf:.3f} (P {bp:.2f} R {br:.2f}, thr {bt}), candidate ceiling {cov:.2f}, rows {n}, positives {npos}")
    out[f"B_kinks_plus{off}"] = [round(bf, 3), round(cov, 3)]
Path("runs/ball_candidates_eval.json").write_text(json.dumps(out, indent=1))
