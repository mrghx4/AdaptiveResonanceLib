import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART as pyQuadraticNeuronART
from artlib.optimized.backends.cpp.QuadraticNeuronART import (
    QuadraticNeuronART as cppQuadraticNeuronART,
)
from artlib.optimized.backends.cpp.cppQuadraticNeuronART import PredictQuadraticNeuronART


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "s_init": 0.5, "lr_b": 0.1, "lr_w": 0.1, "lr_s": 0.05}
    A = pyQuadraticNeuronART(**params)
    B = cppQuadraticNeuronART(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=0.0, atol=1e-12)


def test_consistency():
    data, _ = make_blobs(
        n_samples=900,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    params = {"rho": 0.7, "s_init": 0.5, "lr_b": 0.1, "lr_w": 0.1, "lr_s": 0.05}
    A = pyQuadraticNeuronART(**params)
    B = cppQuadraticNeuronART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert len(A.W) == len(B.W)
    for w_a, w_b in zip(A.W, B.W):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-8, atol=1e-10)

    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    bad_weights = [
        np.array([1.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.5], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictQuadraticNeuronART(
            X,
            rho=0.7,
            s_init=0.5,
            lr_b=0.1,
            lr_w=0.1,
            lr_s=0.05,
            weights=bad_weights,
        )
