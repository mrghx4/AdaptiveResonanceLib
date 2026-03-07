import numpy as np
from sklearn.datasets import make_blobs

from artlib.elementary.BayesianART import BayesianART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.BayesianARTMAP import BayesianARTMAP


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "cov_init": np.eye(2)}
    A = SimpleARTMAP(BayesianART(**params))
    B = BayesianARTMAP(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=1e-7, atol=1e-9)


def test_consistency():
    data, target = make_blobs(
        n_samples=600,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    params = {"rho": 0.7, "cov_init": np.eye(2)}
    A = SimpleARTMAP(BayesianART(**params))
    B = BayesianARTMAP(**params)

    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    assert len(A.module_a.W) == len(B.module_a.W)
    for w_a, w_b in zip(A.module_a.W, B.module_a.W):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-5, atol=1e-7)

    assert np.array_equal(A.labels_, B.labels_)
