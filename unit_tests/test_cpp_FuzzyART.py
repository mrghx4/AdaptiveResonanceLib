import numpy as np
import pytest
from sklearn.datasets import make_blobs
from artlib.elementary.FuzzyART import FuzzyART as pyFuzzyART
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as cppFuzzyART
from artlib.optimized.backends.cpp.cppFuzzyART import PredictFuzzyART


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.8, "alpha": 1e-10, "beta": 1.0}
    A = pyFuzzyART(**params)
    B = cppFuzzyART(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    assert np.array_equal(X_A, X_B)



def test_consistency():
    data, target = make_blobs(
            n_samples=1500,
            centers=3,
            cluster_std=0.50,
            random_state=0,
            shuffle=False,
        )

    params = {"rho":0.8, "alpha":1e-10, "beta":1.0}
    A = pyFuzzyART(**params)
    B = cppFuzzyART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert np.array_equal(A.W, B.W)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)


def test_predict_rejects_inconsistent_weight_dimensions():
    # complement-coded sample with 4 columns (2 original dims)
    X = np.array([[0.2, 0.8, 0.8, 0.2]], dtype=np.float64)
    bad_weights = [np.array([0.1, 0.9, 0.9, 0.1]), np.array([0.2, 0.8])]

    with pytest.raises(ValueError):
        PredictFuzzyART(X, rho=0.8, alpha=1e-10, beta=1.0, weights=bad_weights)
