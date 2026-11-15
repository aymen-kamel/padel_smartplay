"""Label the ball and the hits of one clip (benchmark ground truth).

    python scripts/label_tool.py --video <clip.mp4> --out data/labels/<name>.json

Controls (click the video window first so it has the keyboard focus):
    left click     ball position at this frame        (a zoom inset follows the mouse)
    X              ball NOT visible in this frame
    H              mark / unmark a HIT at this frame (racket touches the ball)
    N / P          next / previous labelling frame (every 3rd frame)
    F              jump to the next frame that has no ball label yet
    D / A          +1 / -1 frame            W / S   +10 / -10 frames
    U              erase the ball label of this frame
    Q or Esc       save and quit             (every action is saved at once)
Green circle = saved ball label, orange text = hit frame, red "NOT VISIBLE" = labelled invisible.
You can quit at any time and run the same command again: your labels are kept.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from smartplay.labeling.store import ClipLabels

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--stride", type=int, default=3)
args = ap.parse_args()

cap = cv2.VideoCapture(str(args.video))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
labels = ClipLabels.load(args.out) if args.out.exists() else ClipLabels(str(args.video), fps, n, args.stride)
if labels.n_frames != n:
    raise SystemExit(f"{args.out} belongs to a clip with {labels.n_frames} frames, this one has {n}")

SCALE = min(1.0, 1500 / W)
DW, DH = int(W * SCALE), int(H * SCALE)
ZOOM_SRC, ZOOM_OUT = 60, 300  # source pixels shown around the mouse, inset size
state = {"frame": 0, "mouse": (0, 0), "cache": None}
LEFT, RIGHT, UP, DOWN = 2424832, 2555904, 2490368, 2621440


def get_frame(i: int) -> np.ndarray:
    if state["cache"] is None or state["cache"][0] != i:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        state["cache"] = (i, img if ok else np.zeros((H, W, 3), np.uint8))
    return state["cache"][1]


def render() -> np.ndarray:
    i = state["frame"]
    img = get_frame(i).copy()
    if i in labels.ball and labels.ball[i] is not None:
        x, y = (int(v) for v in labels.ball[i])
        cv2.circle(img, (x, y), 14, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    disp = cv2.resize(img, (DW, DH), interpolation=cv2.INTER_AREA)
    done, total = labels.progress()
    lines = [f"frame {i}/{n - 1}  t={i / fps:5.2f}s   labelled {done}/{total}   hits {len(labels.hits)}",
             "click=ball  X=not visible  H=hit  N/P=next/prev  F=next unlabelled  A/D=+-1  W/S=+-10  U=erase  Q=quit"]
    for k, t in enumerate(lines):
        cv2.putText(disp, t, (10, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, t, (10, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if i in labels.hits:
        cv2.putText(disp, "HIT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3, cv2.LINE_AA)
    if i in labels.ball and labels.ball[i] is None:
        cv2.putText(disp, "NOT VISIBLE", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255), 3, cv2.LINE_AA)
    mx, my = state["mouse"]
    ox, oy = int(mx / SCALE), int(my / SCALE)
    x0, y0 = max(0, min(W - ZOOM_SRC, ox - ZOOM_SRC // 2)), max(0, min(H - ZOOM_SRC, oy - ZOOM_SRC // 2))
    inset = cv2.resize(img[y0:y0 + ZOOM_SRC, x0:x0 + ZOOM_SRC], (ZOOM_OUT, ZOOM_OUT), interpolation=cv2.INTER_CUBIC)
    c = ZOOM_OUT // 2
    cv2.line(inset, (c, c - 12), (c, c + 12), (0, 255, 255), 1)
    cv2.line(inset, (c - 12, c), (c + 12, c), (0, 255, 255), 1)
    disp[DH - ZOOM_OUT - 10:DH - 10, DW - ZOOM_OUT - 10:DW - 10] = inset
    cv2.rectangle(disp, (DW - ZOOM_OUT - 10, DH - ZOOM_OUT - 10), (DW - 10, DH - 10), (255, 255, 255), 1)
    return disp


def on_mouse(event, x, y, flags, param):
    state["mouse"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        labels.set_ball(state["frame"], (x / SCALE, y / SCALE))
        labels.save(args.out)


def goto(i: int) -> None:
    state["frame"] = int(max(0, min(n - 1, i)))


cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("label", on_mouse)
first = labels.next_unlabelled(-1)
goto(first if first is not None else 0)
while True:
    cv2.imshow("label", render())
    k = cv2.waitKeyEx(30)
    if k == -1:
        continue
    key = chr(k).lower() if 0 <= k < 128 else ""
    i = state["frame"]
    if key in ("q", "\x1b"):
        break
    elif key == "x":
        labels.set_ball(i, None)
    elif key == "u":
        labels.clear_ball(i)
    elif key == "h":
        labels.toggle_hit(i)
    elif key in ("d",) or k == RIGHT:
        goto(i + 1)
    elif key in ("a",) or k == LEFT:
        goto(i - 1)
    elif key == "w" or k == UP:
        goto(i + 10)
    elif key == "s" or k == DOWN:
        goto(i - 10)
    elif key == "n":
        goto((i // labels.stride + 1) * labels.stride)
    elif key == "p":
        goto(((i - 1) // labels.stride) * labels.stride)
    elif key == "f":
        nxt = labels.next_unlabelled(i)
        if nxt is not None:
            goto(nxt)
    else:
        continue
    labels.save(args.out)
labels.save(args.out)
done, total = labels.progress()
print(f"saved {args.out}: ball labelled {done}/{total} frames, {len(labels.hits)} hits")
cv2.destroyAllWindows()
