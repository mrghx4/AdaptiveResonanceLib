import numpy as np
import pytest

from artlib.optimized.backends.cpp.cppGaussianARTMAP import PredictGaussianARTMAP


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    sigma_init = np.array([0.5, 0.5], dtype=np.float64)
    bad_weights = [
        np.array([0.1, 0.2, 0.5, 0.5, 4.0, 4.0, 0.25, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 0.5], dtype=np.float64),
    ]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictGaussianARTMAP(
            X,
            rho=0.05,
            alpha=1e-10,
            sigma_init=sigma_init,
            MT="",
            epsilon=0.0,
            weights=bad_weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_weight_label_mismatch():
    X = np.array([[0.1, 0.2]], dtype=np.float64)
    sigma_init = np.array([0.5, 0.5], dtype=np.float64)
    weights = [
        np.array([0.1, 0.2, 0.5, 0.5, 4.0, 4.0, 0.25, 1.0], dtype=np.float64)
    ]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictGaussianARTMAP(
            X,
            rho=0.05,
            alpha=1e-10,
            sigma_init=sigma_init,
            MT="",
            epsilon=0.0,
            weights=weights,
            cluster_labels=cluster_labels,
        )
