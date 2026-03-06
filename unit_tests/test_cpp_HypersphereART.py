import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.HypersphereART import HypersphereART as pyHypersphereART
from artlib.optimized.backends.cpp.HypersphereART import HypersphereART as cppHypersphereART
from artlib.optimized.backends.cpp.cppHypersphereART import PredictHypersphereART


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "alpha": 1e-10, "beta": 1.0, "r_hat": 0.8}
    A = pyHypersphereART(**params)
    B = cppHypersphereART(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=1e-7, atol=1e-9)


def test_consistency():
    data, _ = make_blobs(
        n_samples=1500,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    params = {"rho": 0.7, "alpha": 1e-10, "beta": 1.0, "r_hat": 0.8}
    A = pyHypersphereART(**params)
    B = cppHypersphereART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    np.testing.assert_allclose(A.W, B.W, rtol=1e-7, atol=1e-9)
    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 0.0], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictHypersphereART(
            X,
            rho=0.7,
            alpha=1e-10,
            beta=1.0,
            r_hat=0.8,
            weights=bad_weights,
        )
