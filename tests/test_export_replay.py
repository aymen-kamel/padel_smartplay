from smartplay.export.replay import build_replay


def tr(frame, pid, x, y, observed=True):
    return {"frame": frame, "player_id": pid, "team": "A" if pid <= 2 else "B", "side": "NEAR" if y < 0 else "FAR",
            "foot_xy_m": [x, y], "conf": 0.9, "observed": observed}


def test_axes_and_lengths():
    tracks = [tr(f, 1, 1.0, -7.0) for f in range(10)] + [tr(f, 3, -1.0, 7.0) for f in range(10)]
    d = build_replay(tracks, [3, 5, 99], 25.0, 10)
    p1 = next(p for p in d["players"] if p["id"] == 1)
    assert p1["zs"][0] == 7.0 and p1["xs"][0] == 1.0  # near the camera -> z > 0 in the viewer
    assert len(p1["xs"]) == 10 and d["ball3d"] is None
    assert [h["frame"] for h in d["hits"]] == [3, 5]  # frame 99 is outside the clip


def test_missing_frames_carry_last_position():
    tracks = [tr(f, 1, 0.0, -5.0) for f in range(3)]
    d = build_replay(tracks, [], 25.0, 6)
    p = d["players"][0]
    assert p["zs"][5] == 5.0 and p["observed"][5] is False
