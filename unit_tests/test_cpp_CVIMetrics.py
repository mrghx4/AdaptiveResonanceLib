import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from artlib.optimized.backends.cpp.cppCVIMetrics import EvaluateCVI


def _make_data():
    X, y = make_blobs(
        n_samples=120,
        centers=3,
        cluster_std=0.45,
        random_state=42,
        shuffle=True,
    )
    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.int32)


def test_calinski_harabasz_matches_sklearn():
    X, y = _make_data()
    cpp = EvaluateCVI(X, y, 1)
    py = calinski_harabasz_score(X, y)
    assert np.isclose(cpp, py, rtol=1e-9, atol=1e-9)


def test_davies_bouldin_matches_sklearn():
    X, y = _make_data()
    cpp = EvaluateCVI(X, y, 2)
    py = davies_bouldin_score(X, y)
    assert np.isclose(cpp, py, rtol=1e-9, atol=1e-9)


def test_silhouette_matches_sklearn():
    X, y = _make_data()
    cpp = EvaluateCVI(X, y, 3)
    py = silhouette_score(X, y)
    assert np.isclose(cpp, py, rtol=1e-9, atol=1e-9)
