import numpy as np
import pytest

cpp_fusion_utils = pytest.importorskip("artlib.optimized.backends.cpp.cppFusionUtils")


def test_argmax_weighted_activations_matches_numpy():
    activ = np.array(
        [
            [0.2, 0.6, 0.9],
            [0.8, 0.1, 0.3],
            [0.5, 0.4, 0.7],
        ],
        dtype=np.float64,
    )
    gamma = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    skip_mask = np.array([0, 1, 0], dtype=np.uint8)

    idx_cpp = cpp_fusion_utils.ArgmaxWeightedActivations(activ, gamma, skip_mask)
    scores = (activ * gamma[np.newaxis, :])[:, skip_mask == 0].sum(axis=1)
    idx_np = int(np.argmax(scores))
    assert int(idx_cpp) == idx_np


def test_argmax_weighted_activations_shape_mismatch_raises():
    activ = np.ones((4, 3), dtype=np.float64)
    gamma = np.ones((2,), dtype=np.float64)
    skip_mask = np.zeros((3,), dtype=np.uint8)
    with pytest.raises(Exception):
        cpp_fusion_utils.ArgmaxWeightedActivations(activ, gamma, skip_mask)


def test_argmax_weighted_channel_activations_matches_matrix_variant():
    rng = np.random.default_rng(7)
    channel_activ = rng.random((4, 9), dtype=np.float64)
    gamma = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    skip_mask = np.array([0, 1, 0, 0], dtype=np.uint8)

    idx_channel = cpp_fusion_utils.ArgmaxWeightedChannelActivations(
        channel_activ, gamma, skip_mask
    )
    activ = channel_activ.T.copy()  # (n_categories, n_channels)
    idx_matrix = cpp_fusion_utils.ArgmaxWeightedActivations(activ, gamma, skip_mask)
    assert int(idx_channel) == int(idx_matrix)
