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


def test_build_state_action_reward_query_matches_numpy():
    state = np.array([0.2, 0.8, 0.1, 0.9], dtype=np.float64)
    actions = np.array(
        [
            [0.0, 1.0, 0.3, 0.7],
            [1.0, 0.0, 0.4, 0.6],
        ],
        dtype=np.float64,
    )

    out = cpp_fusion_utils.BuildStateActionRewardQuery(
        state, actions, reward_dim=2, fill_value=0.5
    )
    expected = np.empty((2, 10), dtype=np.float64)
    expected[:, :4] = state
    expected[:, 4:8] = actions
    expected[:, 8:] = 0.5
    np.testing.assert_allclose(out, expected)


def test_build_state_action_reward_query_rejects_invalid_shape():
    state = np.ones((1, 4), dtype=np.float64)
    actions = np.ones((2, 4), dtype=np.float64)
    with pytest.raises(Exception):
        cpp_fusion_utils.BuildStateActionRewardQuery(state, actions, reward_dim=2)


def test_join_channels_with_fill_matches_numpy():
    channel_a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    channel_c = np.array([[0.9], [0.8]], dtype=np.float64)
    out = cpp_fusion_utils.JoinChannelsWithFill(
        [channel_a, channel_c],
        np.array([2, 3, 1], dtype=np.int64),
        np.array([1, 0, 1], dtype=np.uint8),
        fill_value=0.5,
    )
    expected = np.array(
        [
            [0.1, 0.2, 0.5, 0.5, 0.5, 0.9],
            [0.3, 0.4, 0.5, 0.5, 0.5, 0.8],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected)


def test_join_channels_with_fill_rejects_row_mismatch():
    channel_a = np.ones((2, 2), dtype=np.float64)
    channel_b = np.ones((3, 1), dtype=np.float64)
    with pytest.raises(Exception):
        cpp_fusion_utils.JoinChannelsWithFill(
            [channel_a, channel_b],
            np.array([2, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.uint8),
        )


def test_extract_present_channels_matches_expected_slices():
    joined = np.array(
        [
            [0.1, 0.2, 0.5, 0.5, 0.5, 0.9],
            [0.3, 0.4, 0.5, 0.5, 0.5, 0.8],
        ],
        dtype=np.float64,
    )
    out = cpp_fusion_utils.ExtractPresentChannels(
        joined,
        np.array([2, 3, 1], dtype=np.int64),
        np.array([1, 0, 1], dtype=np.uint8),
    )
    assert len(out) == 2
    np.testing.assert_allclose(out[0], joined[:, :2])
    np.testing.assert_allclose(out[1], joined[:, 5:6])


def test_extract_present_channels_rejects_bad_total_width():
    joined = np.ones((2, 5), dtype=np.float64)
    with pytest.raises(Exception):
        cpp_fusion_utils.ExtractPresentChannels(
            joined,
            np.array([2, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.uint8),
        )
