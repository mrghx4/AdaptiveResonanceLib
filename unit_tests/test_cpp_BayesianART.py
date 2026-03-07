import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.BayesianART import BayesianART as pyBayesianART
from artlib.optimized.backends.cpp.BayesianART import BayesianART as cppBayesianART
from artlib.optimized.backends.cpp.cppBayesianART import PredictBayesianART


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    cov_init = np.eye(2, dtype=np.float64)
    A = pyBayesianART(rho=0.7, cov_init=cov_init)
    B = cppBayesianART(rho=0.7, cov_init=cov_init)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=1e-7, atol=1e-9)


def test_consistency():
    data, _ = make_blobs(
        n_samples=600,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    cov_init = np.eye(2, dtype=np.float64)
    A = pyBayesianART(rho=0.7, cov_init=cov_init)
    B = cppBayesianART(rho=0.7, cov_init=cov_init)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert len(A.W) == len(B.W)
    for w_a, w_b in zip(A.W, B.W):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-5, atol=1e-7)

    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    cov_init = np.eye(2, dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 1.0], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictBayesianART(X, rho=0.7, cov_init=cov_init, weights=bad_weights)
