import pytest
import numpy as np
from artlib.common.BaseART import BaseART
from artlib.fusion.FusionART import FusionART
from artlib.elementary.FuzzyART import FuzzyART


@pytest.fixture
def fusionart_model():
    # Initialize FusionART with two FuzzyART modules
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.7, 0.01, 1.0)
    gamma_values = np.array([0.5, 0.5])
    channel_dims = [4, 4]
    return FusionART(
        modules=[module_a, module_b],
        gamma_values=gamma_values,
        channel_dims=channel_dims,
    )


def test_initialization(fusionart_model):
    # Test that the model initializes correctly
    assert isinstance(fusionart_model.modules[0], BaseART)
    assert isinstance(fusionart_model.modules[1], BaseART)
    assert np.all(
        fusionart_model.params["gamma_values"] == np.array([0.5, 0.5])
    )
    assert fusionart_model.channel_dims == [4, 4]


def test_validate_params():
    # Test the validate_params method
    valid_params = {"gamma_values": np.array([0.5, 0.5])}
    FusionART.validate_params(valid_params)

    invalid_params = {
        "gamma_values": np.array([0.6, 0.6])
    }  # sum of gamma_values must be 1.0
    with pytest.raises(AssertionError):
        FusionART.validate_params(invalid_params)


def test_get_cluster_centers(fusionart_model):
    # Test the get_cluster_centers method
    fusionart_model.modules[0].W = [np.array([0.1, 0.4, 0.5, 0.4])]
    fusionart_model.modules[1].W = [np.array([0.2, 0.2, 0.2, 0.2])]
    fusionart_model.modules[0].d_min_ = np.array([0.0, 0.0])
    fusionart_model.modules[1].d_min_ = np.array([0.0, 0.0])
    fusionart_model.modules[0].d_max_ = np.array([1.0, 1.0])
    fusionart_model.modules[1].d_max_ = np.array([1.0, 1.0])

    centers = fusionart_model.get_cluster_centers()

    assert len(centers) == 1
    assert np.allclose(centers[0], np.array([0.3, 0.5, 0.5, 0.5]))


def test_prepare_and_restore_data(fusionart_model):
    # Test prepare_data and restore_data methods
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]

    X_prep = fusionart_model.prepare_data(X)
    assert X_prep.shape == (10, 8)

    X_restored = fusionart_model.restore_data(X_prep)
    assert np.allclose(X_restored[0], X[0])
    assert np.allclose(X_restored[1], X[1])


def test_fit(fusionart_model):
    # Test the fit method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    assert fusionart_model.labels_.shape[0] == X_prep.shape[0]


def test_partial_fit(fusionart_model):
    # Test the partial_fit method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.partial_fit(X_prep)

    assert fusionart_model.labels_.shape[0] == X_prep.shape[0]


def test_predict(fusionart_model):
    # Test the predict method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    predictions = fusionart_model.predict(X_prep)
    assert predictions.shape[0] == X_prep.shape[0]


def test_step_fit(fusionart_model):
    # Test the step_fit method with base_module's internal methods
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    # Prepare data before fitting
    fusionart_model.modules[0].W = []
    fusionart_model.modules[1].W = []

    # Run step_fit for the first sample
    label = fusionart_model.step_fit(X_prep[0])
    assert isinstance(
        label, int
    )  # Ensure the result is an integer cluster label


def test_step_fit_avoids_deep_copy_when_first_category_matches(fusionart_model, monkeypatch):
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    deep_copy_calls = 0
    original_deep_copy = fusionart_model._deep_copy_params

    def _counting_deep_copy():
        nonlocal deep_copy_calls
        deep_copy_calls += 1
        return original_deep_copy()

    monkeypatch.setattr(fusionart_model, "_deep_copy_params", _counting_deep_copy)
    fusionart_model.step_fit(X_prep[0])
    assert deep_copy_calls == 0


def test_step_pred(fusionart_model):
    # Test the step_pred method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    label = fusionart_model.step_pred(X_prep[0])
    assert isinstance(label, int)  # Ensure the result is an integer


def test_step_pred_raises_when_unfit(fusionart_model):
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.modules[0].W = []
    fusionart_model.modules[1].W = []
    with pytest.raises(AssertionError):
        fusionart_model.step_pred(X_prep[0])


def test_predict_regression(fusionart_model):
    # Test the predict_regression method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    predicted_regression = fusionart_model.predict_regression(X_prep)
    assert predicted_regression.shape[0] == X_prep.shape[0]


def test_predict_regression_matches_channel_centers(fusionart_model):
    X = [np.random.rand(14, 2), np.random.rand(14, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    c = fusionart_model.predict(X_prep, skip_channels=[-1])
    centers = np.asarray(fusionart_model.get_channel_centers(1))
    expected = centers[c]
    pred = fusionart_model.predict_regression(X_prep, target_channels=[-1])
    np.testing.assert_allclose(pred, expected, rtol=1e-10, atol=1e-12)


def test_predict_regression_multiple_target_channels(fusionart_model):
    X = [np.random.rand(16, 2), np.random.rand(16, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    pred = fusionart_model.predict_regression(X_prep, target_channels=[0, 1])
    assert isinstance(pred, list)
    assert len(pred) == 2
    assert pred[0].shape[0] == X_prep.shape[0]
    assert pred[1].shape[0] == X_prep.shape[0]


def test_predict_regression_multi_target_matches_cached_centers(fusionart_model):
    X = [np.random.rand(12, 2), np.random.rand(12, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    labels = fusionart_model.predict(X_prep, skip_channels=[0, 1])
    centers_0 = np.asarray(fusionart_model.get_channel_centers(0))
    centers_1 = np.asarray(fusionart_model.get_channel_centers(1))
    pred = fusionart_model.predict_regression(X_prep, target_channels=[0, 1])
    np.testing.assert_allclose(pred[0], centers_0[labels], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pred[1], centers_1[labels], rtol=1e-10, atol=1e-12)


def test_predict_regression_rejects_duplicate_target_channels(fusionart_model):
    X = [np.random.rand(8, 2), np.random.rand(8, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    with pytest.raises(ValueError, match="duplicate target channel"):
        fusionart_model.predict_regression(X_prep, target_channels=[1, -1])


def test_predict_regression_rejects_out_of_range_target_channel(fusionart_model):
    X = [np.random.rand(8, 2), np.random.rand(8, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    with pytest.raises(ValueError, match="out of range"):
        fusionart_model.predict_regression(X_prep, target_channels=[2])


def test_predict_regression_channel_centers_cache_reuse_and_invalidate(fusionart_model):
    X = [np.random.rand(18, 2), np.random.rand(18, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    _ = fusionart_model.predict_regression(X_prep, target_channels=[-1])
    assert 1 in fusionart_model._channel_centers_cache
    cache_id_1 = id(fusionart_model._channel_centers_cache[1])

    _ = fusionart_model.predict_regression(X_prep, target_channels=[-1])
    cache_id_2 = id(fusionart_model._channel_centers_cache[1])
    assert cache_id_1 == cache_id_2

    # Any weight update path should invalidate the centers cache.
    w0 = fusionart_model.W[0]
    fusionart_model.set_weight(0, w0)
    assert fusionart_model._channel_centers_cache == {}
    assert fusionart_model._cluster_centers_cache is None


def test_get_cluster_centers_cache_reuse_and_invalidate(fusionart_model):
    X = [np.random.rand(18, 2), np.random.rand(18, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    centers1 = fusionart_model.get_cluster_centers()
    assert fusionart_model._cluster_centers_cache is not None
    cache_id_1 = id(fusionart_model._cluster_centers_cache)

    centers2 = fusionart_model.get_cluster_centers()
    cache_id_2 = id(fusionart_model._cluster_centers_cache)
    assert cache_id_1 == cache_id_2
    np.testing.assert_allclose(np.asarray(centers1), np.asarray(centers2))

    w0 = fusionart_model.W[0]
    fusionart_model.set_weight(0, w0)
    assert fusionart_model._cluster_centers_cache is None


def test_get_cluster_centers_uses_cached_channel_center_arrays(fusionart_model):
    X = [np.random.rand(18, 2), np.random.rand(18, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    _ = fusionart_model.predict_regression(X_prep, target_channels=[0, 1])
    assert 0 in fusionart_model._channel_centers_cache
    assert 1 in fusionart_model._channel_centers_cache

    def _unexpected_channel_center_call():
        raise AssertionError("module.get_cluster_centers should not be called")

    fusionart_model.modules[0].get_cluster_centers = _unexpected_channel_center_call
    fusionart_model.modules[1].get_cluster_centers = _unexpected_channel_center_call

    centers = fusionart_model.get_cluster_centers()
    assert len(centers) == fusionart_model.n_clusters


def test_join_channel_data(fusionart_model):
    # Test the join_channel_data method
    channel_1 = np.random.rand(10, 4)
    channel_2 = np.random.rand(10, 4)

    X = fusionart_model.join_channel_data([channel_1, channel_2])
    assert X.shape == (10, 8)


def test_join_channel_data_with_skip_fills_skipped_channel(fusionart_model):
    channel_1 = np.random.rand(8, 4)
    X = fusionart_model.join_channel_data([channel_1], skip_channels=[1])
    assert X.shape == (8, 8)
    np.testing.assert_allclose(X[:, :4], channel_1)
    np.testing.assert_allclose(X[:, 4:], 0.5)


def test_join_channel_data_rejects_wrong_number_of_present_channels(fusionart_model):
    channel_1 = np.random.rand(8, 4)
    with pytest.raises(ValueError, match="expected 1 present channels"):
        fusionart_model.join_channel_data([channel_1, channel_1], skip_channels=[1])


def test_join_channel_data_rejects_mismatched_channel_rows(fusionart_model):
    channel_1 = np.random.rand(8, 4)
    channel_2 = np.random.rand(7, 4)
    with pytest.raises(ValueError, match="same number of rows"):
        fusionart_model.join_channel_data([channel_1, channel_2])


def test_split_channel_data_rejects_wrong_joined_width(fusionart_model):
    bad_joined = np.random.rand(4, 7)
    with pytest.raises(ValueError, match="does not match expected"):
        fusionart_model.split_channel_data(bad_joined)


def test_split_channel_data_with_skip_returns_only_present_channels(fusionart_model):
    channel_1 = np.random.rand(6, 4)
    joined = fusionart_model.join_channel_data([channel_1], skip_channels=[1])
    split = fusionart_model.split_channel_data(joined, skip_channels=[1])
    assert len(split) == 1
    np.testing.assert_allclose(split[0], channel_1)


def test_category_choice_value_idx_matches_cached_activation(fusionart_model):
    X = [np.random.rand(12, 2), np.random.rand(12, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    x = X_prep[0]
    skip = fusionart_model._normalize_skip_channels(None)
    for c_idx in range(fusionart_model.n_clusters):
        act_cached, _ = fusionart_model._category_choice_idx(x, c_idx)
        act_value = fusionart_model._category_choice_value_idx(x, c_idx, skip)
        assert np.isclose(act_cached, act_value)


def test_step_pred_cpp_argmax_path_matches_python(fusionart_model):
    X = [np.random.rand(20, 2), np.random.rand(20, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    # Force C++ argmax path, if extension is available.
    fusionart_model._cpp_fusion_argmax_threshold = 0
    pred_cpp = fusionart_model.step_pred(X_prep[0])

    # Force pure Python path.
    fusionart_model._cpp_fusion_argmax_threshold = 10**9
    pred_py = fusionart_model.step_pred(X_prep[0])

    assert int(pred_cpp) == int(pred_py)


def test_predict_uses_single_active_channel_batch_predict(monkeypatch, fusionart_model):
    X = [np.random.rand(12, 2), np.random.rand(12, 2)]
    X_prep = fusionart_model.prepare_data(X)
    fusionart_model.fit(X_prep, max_iter=1)

    calls = {"n": 0}

    def _predict_batch(data, clip=False):
        calls["n"] += 1
        return np.arange(data.shape[0], dtype=int) % max(
            1, fusionart_model.modules[0].n_clusters
        )

    monkeypatch.setattr(fusionart_model.modules[0], "predict", _predict_batch)
    pred = fusionart_model.predict(X_prep, skip_channels=[1])
    assert pred.shape[0] == X_prep.shape[0]
    assert calls["n"] == 1


def test_normalize_skip_channels_rejects_out_of_range_indices(fusionart_model):
    with pytest.raises(ValueError, match="out of range"):
        fusionart_model._normalize_skip_channels([2])
    with pytest.raises(ValueError, match="out of range"):
        fusionart_model._normalize_skip_channels([-3])
