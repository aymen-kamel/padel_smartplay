"""Command line: analyse a fixed-camera padel video and write an annotated copy.

    python -m smartplay.cli annotate --video rally.mp4 --keypoints court.json --out runs/rally

What it does today (M0-M5 baseline, see docs/architecture/PLAN.md):
  * court calibration from 12 clicked points (with a leave-one-out accuracy estimate)
  * four players detected (RF-DETR Small) and tracked on the court floor, teams A/B, sides
  * audio hit candidates (onset detector, unvalidated)
What it does NOT do yet: ball tracking, bounces, points, score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_BALL_WEIGHTS = Path(__file__).resolve().parents[2] / "weights/wasb/wasb_tennis_best.pth.tar"


def annotate(video: Path, keypoints: Path, out_dir: Path, threshold: float, max_frames: int | None,
             ball_weights: Path = DEFAULT_BALL_WEIGHTS) -> None:
    import itertools

    import cv2

    from smartplay.ball.detector import WasbBallDetector
    from smartplay.events.audio import detect_hits, envelope_per_frame
    from smartplay.geometry.camera import CourtCamera
    from smartplay.ingestion.video import extract_audio, iter_frames, probe
    from smartplay.players.detector import PersonDetector
    from smartplay.players.tracker import FourPlayerTracker
    from smartplay.viz import annotate as viz

    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    n = info.n_frames if max_frames is None else min(info.n_frames, max_frames)
    print(f"video: {info.width}x{info.height} @ {info.fps:g} fps, {info.n_frames} frames, "
          f"audio={info.has_audio}")

    cam = CourtCamera.from_points([tuple(p) for p in json.loads(keypoints.read_text())])
    print(f"court calibration: fit error {cam.fit_error_cm:.1f} cm, "
          f"leave-one-out error {cam.loo_error_cm:.1f} cm (floor points only)")

    sr = 16000
    audio = extract_audio(video, sr)
    hits = detect_hits(audio, sr)
    hit_frames = [round(h.t * info.fps) for h in hits]
    env = envelope_per_frame(audio, sr, info.fps, info.n_frames)
    print(f"audio: {len(audio) / sr:.1f} s, {len(hits)} hit candidates")

    tb = time.time()
    ball_det = WasbBallDetector(ball_weights, threshold=0.3)  # 0.3 + gated tracker: best on the benchmark
    ball = ball_det.track(itertools.islice(iter_frames(video), n), info.width, info.height)
    del ball_det
    seen = sum(b.visibility == "visible" for b in ball)
    print(f"ball (WASB tennis weights, zero-shot, gated tracker): seen in {seen}/{len(ball)} frames "
          f"({100 * seen / max(1, len(ball)):.1f} %), {len(ball) / (time.time() - tb):.1f} fps")

    detector = PersonDetector(threshold)
    tracker = FourPlayerTracker()
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps, (info.width, info.height))
    tracks_log: list[dict] = []
    t0 = time.time()
    for idx, frame in iter_frames(video):
        if idx >= n:
            break
        cands, boxes_for = [], []
        for d in detector.detect(frame):
            wx, wy = cam.to_world(*d.foot_uv)
            if abs(wx) <= 7.0 and abs(wy) <= 12.0:  # on court, 2 m margin
                cands.append((wx, wy, d))
        tracks = tracker.update(idx, [(c[0], c[1]) for c in cands], [c[2].conf for c in cands])
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for t in tracks:
            if t.observed and cands:
                j = min(range(len(cands)), key=lambda k: (cands[k][0] - t.foot_xy_m[0]) ** 2
                        + (cands[k][1] - t.foot_xy_m[1]) ** 2)
                boxes[t.player_id] = cands[j][2].box_uv
        del boxes_for
        viz.draw_court(frame, cam)
        viz.draw_players(frame, boxes, tracks)
        viz.draw_minimap(frame, tracks)
        viz.draw_ball(frame, ball, idx)
        near_hit = any(abs(idx - hf) <= 3 for hf in hit_frames)
        b = ball[idx] if idx < len(ball) else None
        ball_text = (f"ball: seen (score {b.conf:.2f})" if b is not None and b.uv is not None
                     else "ball: lost")
        viz.draw_hud(frame, idx, info.fps, near_hit, "not detected yet", ball_text)
        viz.draw_timeline(frame, env, hit_frames, idx, info.fps)
        writer.write(frame)
        tracks_log.extend(t.model_dump() for t in tracks)
        if idx % 50 == 0:
            print(f"  frame {idx}/{n}  ({(idx + 1) / (time.time() - t0):.1f} fps)")
    writer.release()

    (out_dir / "tracks.json").write_text(json.dumps(tracks_log))
    (out_dir / "ball.json").write_text(json.dumps([b.model_dump() for b in ball]))
    (out_dir / "audio_hits.json").write_text(
        json.dumps([{"t": h.t, "frame": f, "strength": h.strength}
                    for h, f in zip(hits, hit_frames, strict=True)], indent=1))
    observed = sum(1 for r in tracks_log if r["observed"])
    print(f"players observed in {observed}/{len(tracks_log)} player-frames "
          f"({100 * observed / max(1, len(tracks_log)):.1f} %)")
    print(f"wrote {out_dir / 'annotated.mp4'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="smartplay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate", help="analyse a video and write an annotated copy")
    a.add_argument("--video", type=Path, required=True)
    a.add_argument("--keypoints", type=Path, required=True, help="JSON list of 12 [u, v] points")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    annotate(args.video, args.keypoints, args.out, args.threshold, args.max_frames)


if __name__ == "__main__":
    main()
