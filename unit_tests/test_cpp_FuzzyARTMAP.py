import numpy as np
import pytest

from artlib.optimized.backends.cpp.cppFuzzyARTMAP import FitFuzzyARTMAP, PredictFuzzyARTMAP


def test_predict_rejects_inconsistent_weight_dimensions():
    X = np.array([[0.2, 0.8, 0.8, 0.2]], dtype=np.float64)
    bad_weights = [np.array([0.1, 0.9, 0.9, 0.1]), np.array([0.2, 0.8])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictFuzzyARTMAP(
            X,
            rho=0.8,
            alpha=1e-10,
            beta=1.0,
            MT="MT+",
            epsilon=1e-10,
            weights=bad_weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_weight_label_mismatch():
    X = np.array([[0.2, 0.8, 0.8, 0.2]], dtype=np.float64)
    weights = [np.array([0.1, 0.9, 0.9, 0.1])]
    cluster_labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        PredictFuzzyARTMAP(
            X,
            rho=0.8,
            alpha=1e-10,
            beta=1.0,
            MT="MT+",
            epsilon=1e-10,
            weights=weights,
            cluster_labels=cluster_labels,
        )


def test_predict_rejects_wrong_rank_and_fit_rejects_y_mismatch_without_crashing():
    weights = [np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float64)]
    cluster_labels = np.array([0], dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        PredictFuzzyARTMAP(
            np.array([0.2, 0.8, 0.8, 0.2], dtype=np.float64),
            rho=0.8,
            alpha=1e-10,
            beta=1.0,
            MT="MT+",
            epsilon=1e-10,
            weights=weights,
            cluster_labels=cluster_labels,
        )

    with pytest.raises((ValueError, RuntimeError)):
        FitFuzzyARTMAP(
            np.array([[0.2, 0.8, 0.8, 0.2]], dtype=np.float64),
            np.array([0, 1], dtype=np.int32),
            rho=0.8,
            alpha=1e-10,
            beta=1.0,
            MT="MT+",
            epsilon=1e-10,
        )
