import numpy as np
import pytest

from artlib.optimized.backends.cpp.cppART1MAP import PredictART1MAP


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[1, 0, 1]], dtype=np.int16)
    bad_weights = [np.array([0.5, 0.5, 1.0, 0.0]), np.array([0.2, 0.2])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictART1MAP(
            X,
            rho=0.9,
            L=1.0,
            MT="MT+",
            epsilon=1e-10,
            weights=bad_weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_weight_label_count_mismatch():
    X = np.array([[1, 0, 1]], dtype=np.int16)
    weights = [np.array([0.5, 0.0, 0.5, 1.0, 0.0, 1.0])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictART1MAP(
            X,
            rho=0.9,
            L=1.0,
            MT="MT+",
            epsilon=1e-10,
            weights=weights,
            cluster_labels=cluster_labels,
        )
