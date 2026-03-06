import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.GaussianART import GaussianART as pyGaussianART
from artlib.optimized.backends.cpp.GaussianART import GaussianART as cppGaussianART
from artlib.optimized.backends.cpp.cppGaussianART import PredictGaussianART


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.05, "alpha": 1e-10, "sigma_init": np.array([0.5, 0.5])}
    A = pyGaussianART(**params)
    B = cppGaussianART(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=0.0, atol=1e-12)


def test_consistency():
    data, _ = make_blobs(
        n_samples=1500,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    params = {"rho": 0.05, "alpha": 1e-10, "sigma_init": np.array([0.5, 0.5])}
    A = pyGaussianART(**params)
    B = cppGaussianART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert len(A.W) == len(B.W)
    for w_a, w_b in zip(A.W, B.W):
        assert np.allclose(w_a, w_b, rtol=0.0, atol=1e-12)

    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    sigma_init = np.array([0.5, 0.5], dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 0.5, 0.5, 4.0, 4.0, 0.25, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 0.5], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictGaussianART(
            X,
            rho=0.05,
            sigma_init=sigma_init,
            alpha=1e-10,
            weights=bad_weights,
        )
