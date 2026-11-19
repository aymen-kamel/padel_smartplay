"""Mark who won each broadcast point, by SHIRT COLOUR (no need to know the players' names).

    python scripts/label_winners.py

Plays every main-view clip of the highlights (runs/highlights/clips) in a window. Click the window first, then:
    R   the RED team won this point          D   the DARK team won this point
    U   unclear (no winner visible, replay, cut, not a full point)
    SPACE  replay the clip          B  back to the previous clip          Q / Esc  save and quit
Answers are saved at once in data/labels/point_winners.json; run again to continue where you stopped.
The 9 points whose winner is known from the scoreboard are included on purpose: your answers on them tell which shirt colour
is which scoreboard row, and check the answers.
"""

import json
from pathlib import Path

import cv2

CLIPS = sorted(Path("runs/highlights/clips").glob("point_*.mp4"))
OUT = Path("data/labels/point_winners.json")
answers: dict[str, str] = json.loads(OUT.read_text()) if OUT.exists() else {}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(answers, indent=1))


i = next((k for k, c in enumerate(CLIPS) if c.stem[:8] not in answers), len(CLIPS))
win = "who won? R = red, D = dark, U = unclear, SPACE = replay, B = back, Q = quit"
cv2.namedWindow(win)
while i < len(CLIPS):
    clip = CLIPS[i]
    key_id = clip.stem[:8]
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    decided = None
    last = None
    while decided is None:
        ok, img = cap.read()
        if not ok:  # end of clip: hold the last frame until a key
            img = last if last is not None else img
            wait = 0
        else:
            last = img
            wait = max(1, int(1000 / fps))
        view = cv2.resize(img, None, fx=0.6, fy=0.6)
        txt = f"[{i + 1}/{len(CLIPS)}] {key_id}   previous answer: {answers.get(key_id, '-')}   R red / D dark / U unclear"
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if not ok:
            cv2.putText(view, "END - press R, D or U (SPACE to replay)", (10, view.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(wait) & 0xFF
        if k in (ord("r"), ord("d"), ord("u")):
            decided = {"r": "red", "d": "dark", "u": "unclear"}[chr(k)]
        elif k == ord(" "):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif k == ord("b"):
            decided = "back"
        elif k in (ord("q"), 27):
            decided = "quit"
    cap.release()
    if decided == "quit":
        break
    if decided == "back":
        i = max(0, i - 1)
        continue
    answers[key_id] = decided
    save()
    i += 1
cv2.destroyAllWindows()
save()
print(f"saved {OUT}: {len(answers)} of {len(CLIPS)} clips answered "
      f"(red {sum(v == 'red' for v in answers.values())}, dark {sum(v == 'dark' for v in answers.values())}, "
      f"unclear {sum(v == 'unclear' for v in answers.values())})")
