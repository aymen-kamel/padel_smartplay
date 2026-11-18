from smartplay.ball.detector import Blob
from smartplay.ball.fusion import merge_blobs


def test_secondary_added_only_when_far_from_primary():
    a = [[Blob(100, 100, 0.8)]]
    b = [[Blob(105, 102, 0.9), Blob(400, 300, 0.7)]]
    out = merge_blobs(a, b)[0]
    assert len(out) == 2
    assert out[0].u == 100 and out[0].score == 0.8       # primary untouched
    assert (out[1].u, out[1].v) == (400, 300) and abs(out[1].score - 0.63) < 1e-9


def test_empty_primary_takes_secondary():
    assert len(merge_blobs([[]], [[Blob(1, 2, 0.5)]])[0]) == 1
