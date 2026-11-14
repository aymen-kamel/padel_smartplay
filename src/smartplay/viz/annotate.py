"""Draw the analysis on top of a video frame: court lines, players, mini-map, audio timeline."""

from __future__ import annotations

import numpy as np

from smartplay.geometry.camera import COURT_LINES, CourtCamera
from smartplay.geometry.court import HALF_LENGTH_M, HALF_WIDTH_M
from smartplay.schemas.records import BallState, PlayerTrack

TEAM_COLOR = {"A": (40, 140, 255), "B": (230, 200, 40)}  # BGR: orange / cyan
WHITE, BLACK, GREEN, RED = (255, 255, 255), (0, 0, 0), (80, 220, 80), (60, 60, 255)


def _put(img, text, org, scale=0.6, color=WHITE, thick=1, bg=True):
    import cv2

    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base), BLACK, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_court(img: np.ndarray, cam: CourtCamera) -> None:
    import cv2

    for (x1, y1), (x2, y2) in COURT_LINES:
        p1 = tuple(int(v) for v in cam.to_image(x1, y1))
        p2 = tuple(int(v) for v in cam.to_image(x2, y2))
        cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)


def draw_players(
    img: np.ndarray, boxes: dict[int, tuple[float, float, float, float]],
    tracks: list[PlayerTrack],
) -> None:
    import cv2

    for t in tracks:
        col = TEAM_COLOR[t.team]
        label = f"{t.team}{t.player_id} {t.side}" + ("" if t.observed else " (predicted)")
        box = boxes.get(t.player_id)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            _put(img, label, (x1, max(18, y1 - 6)), 0.55, col)


def draw_minimap(img: np.ndarray, tracks: list[PlayerTrack], w: int = 170, h: int = 340) -> None:
    import cv2

    pad = 12
    x0, y0 = img.shape[1] - w - 20, 20
    panel = np.full((h, w, 3), (50, 90, 40), dtype=np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return (int((x + HALF_WIDTH_M) / (2 * HALF_WIDTH_M) * (w - 2 * pad) + pad),
                int((HALF_LENGTH_M - y) / (2 * HALF_LENGTH_M) * (h - 2 * pad) + pad))

    for (x1, y1), (x2, y2) in COURT_LINES:
        cv2.line(panel, px(x1, y1), px(x2, y2), WHITE, 1, cv2.LINE_AA)
    for t in tracks:
        cv2.circle(panel, px(*t.foot_xy_m), 7, TEAM_COLOR[t.team], -1, cv2.LINE_AA)
        cv2.putText(panel, str(t.player_id), (px(*t.foot_xy_m)[0] - 4, px(*t.foot_xy_m)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1, cv2.LINE_AA)
    img[y0:y0 + h, x0:x0 + w] = panel
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), WHITE, 1)
    _put(img, "FAR", (x0 + 6, y0 + 16), 0.45, WHITE, bg=False)
    _put(img, "NEAR (camera)", (x0 + 6, y0 + h - 6), 0.45, WHITE, bg=False)


def draw_timeline(
    img: np.ndarray, env: np.ndarray, hit_frames: list[int], frame: int, fps: float,
    height: int = 70,
) -> None:
    """Audio loudness over the whole clip, audio-hit markers, and a playhead."""
    import cv2

    n = len(env)
    x0, x1 = 20, img.shape[1] - 20
    y1 = img.shape[0] - 14
    y0 = y1 - height
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)
    xs = np.linspace(x0, x1, n)
    for x, v in zip(xs, env, strict=True):
        cv2.line(img, (int(x), y1), (int(x), int(y1 - v * (height - 6))), (150, 150, 150), 1)
    for hf in hit_frames:
        x = int(x0 + (x1 - x0) * hf / max(1, n - 1))
        cv2.line(img, (x, y0), (x, y1), GREEN, 2)
    xp = int(x0 + (x1 - x0) * frame / max(1, n - 1))
    cv2.line(img, (xp, y0 - 4), (xp, y1 + 4), RED, 2)
    _put(img, "audio loudness + hit candidates (green)", (x0, y0 - 8), 0.5, WHITE)


def draw_ball(img: np.ndarray, ball: list[BallState], frame: int, trail: int = 12) -> None:
    """Ball position (magenta) with a short trail of recent detections; nothing when lost."""
    import cv2

    pts = [b.uv for b in ball[max(0, frame - trail): frame + 1] if b.uv is not None]
    for p, q in zip(pts, pts[1:], strict=False):
        if np.hypot(p[0] - q[0], p[1] - q[1]) < 250:
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (255, 0, 255), 2,
                     cv2.LINE_AA)
    cur = ball[frame] if frame < len(ball) else None
    if cur is not None and cur.uv is not None:
        c = (int(cur.uv[0]), int(cur.uv[1]))
        cv2.circle(img, c, 10, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (255, 0, 255), -1, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray, frame: int, fps: float, hit_now: bool, score_text: str,
    ball_text: str = "ball: not tracked",
) -> None:
    """Text block at the bottom left, above the timeline (keeps broadcast overlays visible)."""
    y = img.shape[0] - 200
    _put(img, f"frame {frame}  t={frame / fps:5.2f}s", (20, y), 0.7)
    _put(img, f"score: {score_text}", (20, y + 32), 0.7, (0, 255, 255))
    _put(img, ball_text, (20, y + 62), 0.6, (180, 180, 180))
    if hit_now:
        _put(img, "HIT? (audio)", (20, y - 36), 0.9, GREEN)
