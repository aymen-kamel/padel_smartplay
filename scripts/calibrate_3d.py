"""Click points of known HEIGHT (net, glass tops) to get the full 3D camera of a fixed view.

    python scripts/calibrate_3d.py --video <clip.mp4> --floor data/calibration/highlights_main_view.json \
        --out data/calibration/highlights_main_view_3d.json [--frame 0]

The 12 floor points are taken from --floor (already clicked). You click the height points one by one; the name of the
point to click is written at the top of the window. Controls (click the window first so it has the keyboard focus):
    left click   place the current point (a zoom inset follows the mouse)
    S            skip this point (not visible / not sure)       U   go back one point
    Q or Esc     save and quit (you can run the command again later: clicks are kept)
A green ring + arrow marks the FLOOR directly below the current point: click ABOVE it, at the top of the net / glass
(clicks on the green ring itself are refused). After 2 height points a yellow ring shows where the camera EXPECTS the next point: a big gap means a wrong click.
At the end: reprojection error (px), leave-one-out error, camera position, and a check image with the 3D court drawn.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from smartplay.geometry.camera import WORLD_POINTS_12, CourtCamera
from smartplay.geometry.camera3d import HEIGHT_POINTS, Camera3D, project
from smartplay.geometry.court import HALF_LENGTH_M, HALF_WIDTH_M

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--floor", type=Path, required=True, help="JSON list of the 12 floor points [u, v]")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--frame", type=int, default=0)
args = ap.parse_args()

cap = cv2.VideoCapture(str(args.video))
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, img = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read the frame")
H, W = img.shape[:2]
SCALE = min(1.0, 1500 / W)
floor = [tuple(p) for p in json.loads(args.floor.read_text())]
clicks: dict[str, list[float] | None] = {}
if args.out.exists():
    clicks = json.loads(args.out.read_text()).get("height_points", {})
X, Y = HALF_WIDTH_M, HALF_LENGTH_M
floor_cam = CourtCamera.from_points(floor)
warning = [""]


def floor_below(k):
    x, y, _ = HEIGHT_POINTS[k][1]
    return np.array(floor_cam.to_image(x, y))


def correspondences():
    w = [(x, y, 0.0) for x, y in WORLD_POINTS_12]
    i = list(floor)
    for name, xyz in HEIGHT_POINTS:
        if clicks.get(name):
            w.append(xyz)
            i.append(tuple(clicks[name]))
    return w, i


def camera():
    w, i = correspondences()
    if sum(1 for p in w if p[2] != 0) < 2:
        return None
    try:
        return Camera3D.from_points(w, i)
    except (ValueError, np.linalg.LinAlgError):
        return None


def save():
    cam = camera()
    data = {"video": str(args.video), "frame": args.frame, "floor_points": floor, "height_points": clicks}
    if cam is not None:
        data.update({"P": cam.P.tolist(), "reproj_px": cam.reproj_px, "loo_px": cam.loo_px,
                     "camera_centre_m": cam.centre.tolist(), "n_height_points": cam.n_height_points})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    return cam


WIRE = [  # 3D court wireframe for the check image
    ((-X, -Y, 0), (X, -Y, 0)), ((X, -Y, 0), (X, Y, 0)), ((X, Y, 0), (-X, Y, 0)), ((-X, Y, 0), (-X, -Y, 0)),
    ((-X, 0, 0), (X, 0, 0)), ((-X, 0, 0.92), (0, 0, 0.88)), ((0, 0, 0.88), (X, 0, 0.92)),
    ((-X, 0, 0), (-X, 0, 0.92)), ((X, 0, 0), (X, 0, 0.92)),
]
for s in (1, -1):
    WIRE += [((-X, s * Y, 3), (X, s * Y, 3))]
    for e in (1, -1):
        WIRE += [((e * X, s * Y, 0), (e * X, s * Y, 3)), ((e * X, s * Y, 3), (e * X, s * (Y - 2), 3)),
                 ((e * X, s * (Y - 2), 3), (e * X, s * (Y - 2), 2)), ((e * X, s * (Y - 2), 2), (e * X, s * (Y - 4), 2)),
                 ((e * X, s * (Y - 4), 2), (e * X, s * (Y - 4), 0))]


def draw_wire(canvas, cam, color=(0, 255, 255)):
    for a, b in WIRE:
        pa, pb = cam.to_image(np.array([a, b], float))
        if np.all(np.isfinite([pa, pb])):
            cv2.line(canvas, tuple(int(v) for v in pa), tuple(int(v) for v in pb), color, 2, cv2.LINE_AA)


mouse = [0, 0]
cur = 0
for k, (name, _) in enumerate(HEIGHT_POINTS):  # resume at the first point never answered
    if name not in clicks:
        cur = k
        break
else:
    cur = len(HEIGHT_POINTS)


def on_mouse(event, x, y, _flags, _param):
    global cur
    mouse[0], mouse[1] = x / SCALE, y / SCALE
    if event == cv2.EVENT_LBUTTONDOWN and cur < len(HEIGHT_POINTS):
        if np.hypot(x / SCALE - floor_below(cur)[0], y / SCALE - floor_below(cur)[1]) < 25:
            warning[0] = "That is the FLOOR point (green ring). Click HIGHER: the top of the net / glass above it."
            return
        warning[0] = ""
        clicks[HEIGHT_POINTS[cur][0]] = [x / SCALE, y / SCALE]
        save()
        cur += 1


win = "calibrate 3D - click the point named at the top"
cv2.namedWindow(win)
cv2.setMouseCallback(win, on_mouse)
while True:
    canvas = img.copy()
    for p in floor:
        cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (0, 200, 0), -1)
    for name, _ in HEIGHT_POINTS:
        if clicks.get(name):
            u, v = clicks[name]
            cv2.drawMarker(canvas, (int(u), int(v)), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
    if cur < len(HEIGHT_POINTS):  # where the floor is, directly below the point to click
        fb = floor_below(cur)
        cv2.circle(canvas, (int(fb[0]), int(fb[1])), 10, (0, 255, 0), 2)
        cv2.arrowedLine(canvas, (int(fb[0]), int(fb[1]) - 14), (int(fb[0]), int(fb[1]) - 70), (0, 255, 0), 2,
                        tipLength=0.3)
    cam = camera()
    if cam is not None and cur < len(HEIGHT_POINTS):
        u, v = project(cam.P, np.array([HEIGHT_POINTS[cur][1]]))[0]
        cv2.circle(canvas, (int(u), int(v)), 18, (0, 255, 255), 2)
    view = cv2.resize(canvas, None, fx=SCALE, fy=SCALE)
    # zoom inset around the mouse
    mx, my = int(mouse[0]), int(mouse[1])
    x0, y0 = max(0, mx - 40), max(0, my - 40)
    crop = canvas[y0:y0 + 80, x0:x0 + 80]
    if crop.size:
        z = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(z, ((mx - x0) * 3, (my - y0) * 3), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        view[10:250, view.shape[1] - 250:view.shape[1] - 10] = z
    if cur < len(HEIGHT_POINTS):
        name, xyz = HEIGHT_POINTS[cur]
        text = f"[{cur + 1}/{len(HEIGHT_POINTS)}] {name}  (height {xyz[2]:.2f} m)   S=skip U=back Q=save"
    else:
        text = "All points answered. Q = save and quit, U = go back."
    cv2.rectangle(view, (0, 0), (view.shape[1] - 260, 34), (0, 0, 0), -1)
    cv2.putText(view, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if warning[0]:
        cv2.putText(view, warning[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if cam is not None:
        cv2.putText(view, f"reprojection {cam.reproj_px:.1f} px, leave-one-out {cam.loo_px:.1f} px",
                    (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(win, view)
    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("s") and cur < len(HEIGHT_POINTS):
        clicks[HEIGHT_POINTS[cur][0]] = None
        save()
        cur += 1
    if key == ord("u") and cur > 0:
        cur -= 1
        clicks.pop(HEIGHT_POINTS[cur][0], None)
        save()
cv2.destroyAllWindows()

cam = save()
if cam is None:
    raise SystemExit("not enough height points yet (need at least 2). Run the same command again to continue.")
print(f"saved {args.out}")
print(f"points: 12 floor + {cam.n_height_points} with height")
print(f"reprojection error {cam.reproj_px:.2f} px (consistency), leave-one-out {cam.loo_px:.2f} px (honest)")
c = cam.centre
if c[2] <= 0.5:
    print("WARNING: the camera comes out BELOW the floor: some height points were clicked on the floor. "
          "Press U in the tool to redo them (click the TOP of the net / glass, not the green ring).")
print(f"camera position: x {c[0]:.2f} m, y {c[1]:.2f} m (negative = behind the near baseline at -10), height {c[2]:.2f} m")
check = img.copy()
draw_wire(check, cam)
out_png = args.out.with_name(args.out.stem + "_check.png")
cv2.imwrite(str(out_png), check)
print(f"check image (yellow = 3D court as the camera sees it; it must sit on the real lines and glass): {out_png}")
