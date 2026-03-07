import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib.elementary.ART2 import ART2A as pyART2A
from artlib.optimized.backends.cpp.ART2 import ART2A as cppART2A
from artlib.optimized.backends.cpp.cppART2 import PredictART2


def test_prepare_data():
    data, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "alpha": 0.1, "beta": 0.5}
    A = pyART2A(**params)
    B = cppART2A(**params)

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

    params = {"rho": 0.7, "alpha": 0.1, "beta": 0.5}
    A = pyART2A(**params)
    B = cppART2A(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert len(A.W) == len(B.W)
    for w_a, w_b in zip(A.W, B.W):
        np.testing.assert_allclose(w_a, w_b, rtol=0.0, atol=1e-12)

    assert np.array_equal(A.labels_, B.labels_)


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
    ]

    with pytest.raises(ValueError):
        PredictART2(
            X,
            rho=0.7,
            alpha=0.1,
            beta=0.5,
            weights=bad_weights,
        )
