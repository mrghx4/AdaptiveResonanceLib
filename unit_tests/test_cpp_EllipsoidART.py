import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.EllipsoidART import EllipsoidART as pyEllipsoidART
from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as cppEllipsoidART
from artlib.optimized.backends.cpp.cppEllipsoidART import PredictEllipsoidART


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "alpha": 1e-5, "beta": 0.1, "mu": 0.5, "r_hat": 1.0}
    A = pyEllipsoidART(**params)
    B = cppEllipsoidART(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=1e-7, atol=1e-9)


def test_consistency():
    data, _ = make_blobs(
        n_samples=800,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    params = {"rho": 0.7, "alpha": 1e-5, "beta": 0.1, "mu": 0.5, "r_hat": 1.0}
    A = pyEllipsoidART(**params)
    B = cppEllipsoidART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert len(A.W) == len(B.W)
    for w_a, w_b in zip(A.W, B.W):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-6, atol=1e-8)

    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.1, 0.2, 0.0], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictEllipsoidART(
            X,
            rho=0.7,
            alpha=1e-5,
            beta=0.1,
            mu=0.5,
            r_hat=1.0,
            weights=bad_weights,
        )
