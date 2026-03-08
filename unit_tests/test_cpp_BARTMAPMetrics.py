import numpy as np
import pytest
from scipy.stats import pearsonr

from artlib.optimized.backends.cpp.cppBARTMAPMetrics import (
    AveragePearsonCorr,
    AnyClusterMatch,
)


def _python_average_pearson_corr(X: np.ndarray, k: int, c_b: int, labels: np.ndarray) -> float:
    X_a = X[labels == c_b, :]
    if len(X_a) == 0:
        raise ValueError("X_a has length 0")
    b_components = labels == c_b
    X_k_cb = X[k, :][b_components]
    mean_r = np.mean([pearsonr(X_k_cb, x_a_l[b_components])[0] for x_a_l in X_a])
    return float(mean_r)


def test_average_pearson_corr_matches_python_reference():
    rng = np.random.default_rng(42)
    X = rng.random((16, 16), dtype=np.float64)
    labels = np.array([0, 1] * 8, dtype=np.int32)
    k = 3
    c_b = 1

    cpp = AveragePearsonCorr(X, k, c_b, labels)
    py = _python_average_pearson_corr(X, k, c_b, labels)
    assert np.isclose(cpp, py, rtol=1e-12, atol=1e-12)


def test_average_pearson_corr_raises_for_missing_cluster():
    X = np.eye(4, dtype=np.float64)
    labels = np.array([0, 0, 0, 0], dtype=np.int32)
    with pytest.raises(ValueError):
        AveragePearsonCorr(X, 0, 1, labels)


def test_any_cluster_match_matches_python_reference():
    rng = np.random.default_rng(123)
    X = rng.random((12, 12), dtype=np.float64)
    labels = np.array([0, 1, 2] * 4, dtype=np.int32)
    k = 5
    eta = -0.05
    n_clusters_b = 3

    cpp = AnyClusterMatch(X, k, n_clusters_b, eta, labels)
    py = any(
        _python_average_pearson_corr(X, k, c_b, labels) >= eta
        for c_b in range(n_clusters_b)
    )
    assert cpp == py


def test_any_cluster_match_raises_for_missing_cluster():
    X = np.eye(4, dtype=np.float64)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        AnyClusterMatch(X, k=0, n_clusters_b=3, eta=2.0, column_labels=labels)
