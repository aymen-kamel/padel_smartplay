import pytest

from smartplay.labeling.store import ClipLabels


def make() -> ClipLabels:
    return ClipLabels(video="clip.mp4", fps=30.0, n_frames=10, stride=3)


def test_stride_frames_and_progress():
    c = make()
    assert c.stride_frames() == [0, 3, 6, 9]
    assert c.progress() == (0, 4)
    c.set_ball(0, (10, 20))
    c.set_ball(3, None)  # "not visible" still counts as labelled
    assert c.progress() == (2, 4)


def test_next_unlabelled_skips_done_frames():
    c = make()
    c.set_ball(3, (1, 1))
    assert c.next_unlabelled(-1) == 0
    assert c.next_unlabelled(0) == 6
    c.set_ball(0, None)
    c.set_ball(6, (2, 2))
    c.set_ball(9, (3, 3))
    assert c.next_unlabelled(-1) is None


def test_hit_toggle():
    c = make()
    assert c.toggle_hit(4) is True and 4 in c.hits
    assert c.toggle_hit(4) is False and 4 not in c.hits


def test_clear_ball_and_range_check():
    c = make()
    c.set_ball(3, (1, 2))
    c.clear_ball(3)
    assert 3 not in c.ball
    with pytest.raises(ValueError):
        c.set_ball(10, (0, 0))
    with pytest.raises(ValueError):
        c.toggle_hit(-1)


def test_save_load_roundtrip(tmp_path):
    c = make()
    c.set_ball(0, (10.5, 20.25))
    c.set_ball(3, None)
    c.toggle_hit(7)
    p = tmp_path / "sub" / "labels.json"
    c.save(p)
    d = ClipLabels.load(p)
    assert d.ball == {0: (10.5, 20.25), 3: None} and d.hits == {7}
    assert (d.video, d.fps, d.n_frames, d.stride) == ("clip.mp4", 30.0, 10, 3)
    assert not list(p.parent.glob("*.tmp"))
