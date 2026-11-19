import numpy as np

from smartplay.events.learned import FEATURE_NAMES, candidates, features, suppress


def synthetic(sr=16000, hits=(1.0, 2.5, 4.0), dur=5.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.01, int(sr * dur))
    for h in hits:  # short broadband click
        i = int(h * sr)
        a[i:i + 400] += rng.normal(0, 0.5, 400) * np.exp(-np.arange(400) / 80)
    return a, sr


def test_candidates_find_the_clicks_and_features_have_the_right_shape():
    a, sr = synthetic()
    t, rel, rel_all, dt = candidates(a, sr)
    assert all(any(abs(x - h) < 0.05 for x in t) for h in (1.0, 2.5, 4.0))
    X = features(a, sr, t, rel, rel_all, dt)
    assert X.shape == (len(t), len(FEATURE_NAMES)) and np.isfinite(X).all()


def test_suppress_keeps_the_best_of_close_candidates():
    t = np.array([1.00, 1.10, 2.00])
    s = np.array([0.6, 0.9, 0.5])
    assert list(suppress(t, s, 0.25)) == [1, 2]
