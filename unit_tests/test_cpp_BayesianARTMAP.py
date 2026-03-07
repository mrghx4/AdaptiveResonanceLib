import numpy as np
import pytest

from artlib.optimized.backends.cpp.cppBayesianARTMAP import PredictBayesianARTMAP


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    cov_init = np.eye(2, dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 1.0], dtype=np.float64),
    ]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictBayesianARTMAP(
            X,
            rho=0.7,
            cov_init=cov_init,
            MT="",
            epsilon=0.0,
            weights=bad_weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_weight_label_mismatch():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    cov_init = np.eye(2, dtype=np.float64)
    weights = [np.array([0.1, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictBayesianARTMAP(
            X,
            rho=0.7,
            cov_init=cov_init,
            MT="",
            epsilon=0.0,
            weights=weights,
            cluster_labels=cluster_labels,
        )
