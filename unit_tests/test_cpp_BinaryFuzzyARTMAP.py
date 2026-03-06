import numpy as np
import pytest

from artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP import PredictBinaryFuzzyARTMAP


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[1, 0, 1, 0]], dtype=np.int16)
    bad_weights = [np.array([1, 0, 1, 0]), np.array([1, 0])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictBinaryFuzzyARTMAP(
            X,
            rho=0.8,
            MT="MT+",
            epsilon=1,
            weights=bad_weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_weight_label_mismatch():
    X = np.array([[1, 0, 1, 0]], dtype=np.int16)
    weights = [np.array([1, 0, 1, 0])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictBinaryFuzzyARTMAP(
            X,
            rho=0.8,
            MT="MT+",
            epsilon=1,
            weights=weights,
            cluster_labels=cluster_labels,
        )
