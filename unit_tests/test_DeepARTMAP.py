import pytest
import numpy as np
from artlib.hierarchical.DeepARTMAP import DeepARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.supervised.ARTMAP import ARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART


# Fixture to initialize a DeepARTMAP instance for testing
@pytest.fixture
def deep_artmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.7, 0.01, 1.0)
    return DeepARTMAP(modules=[module_a, module_b])


def test_initialization(deep_artmap_model):
    # Test that the model initializes correctly
    assert isinstance(deep_artmap_model.modules[0], BaseART)
    assert isinstance(deep_artmap_model.modules[1], BaseART)
    assert len(deep_artmap_model.modules) == 2


def test_get_params(deep_artmap_model):
    # Test the get_params method
    params = deep_artmap_model.get_params()
    assert "module_0" in params
    assert "module_1" in params


def test_set_params(deep_artmap_model):
    # Test the set_params method
    deep_artmap_model.set_params(module_0__rho=0.6)
    assert deep_artmap_model.modules[0].params["rho"] == 0.6


def test_validate_data(deep_artmap_model):
    # Test the validate_data method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)
    deep_artmap_model.validate_data(X, y)

    # Test invalid input data
    X_invalid = [np.random.rand(9, 5), np.random.rand(10, 5)]
    with pytest.raises(AssertionError):
        deep_artmap_model.validate_data(X_invalid, y)


def test_prepare_and_restore_data(deep_artmap_model):
    # Test prepare_data and restore_data methods
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    X_prep, _ = deep_artmap_model.prepare_data(X)

    X_restored, _ = deep_artmap_model.restore_data(X_prep)
    assert np.allclose(X_restored[0], X[0])
    assert np.allclose(X_restored[1], X[1])


def test_fit_supervised(deep_artmap_model):
    # Test the supervised fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    assert deep_artmap_model.layers[0].labels_.shape[0] == X[0].shape[0]


def test_fit_unsupervised(deep_artmap_model):
    # Test the unsupervised fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    assert deep_artmap_model.layers[0].labels_a.shape[0] == X[0].shape[0]


def test_partial_fit_supervised(deep_artmap_model):
    # Test the supervised partial_fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before partial fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.partial_fit(X_prep, y)

    assert deep_artmap_model.layers[0].labels_.shape[0] == X[0].shape[0]


def test_partial_fit_unsupervised(deep_artmap_model):
    # Test the unsupervised partial_fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before partial fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.partial_fit(X_prep)

    assert deep_artmap_model.layers[0].labels_a.shape[0] == X[0].shape[0]


def test_predict(deep_artmap_model):
    # Test the predict method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before fitting and predicting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    predictions = deep_artmap_model.predict(X_prep)
    assert predictions[-1].shape[0] == X[-1].shape[0]


def test_labels_deep(deep_artmap_model):
    # Test the labels_deep_ method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting and predicting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    labels_deep = deep_artmap_model.labels_deep_
    assert labels_deep.shape == (10, 3)


def test_map_deep(deep_artmap_model):
    # Test the map_deep method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    mapped_label = deep_artmap_model.map_deep(
        0, deep_artmap_model.layers[0].labels_a[0]
    )
    assert isinstance(mapped_label.tolist(), int)


def test_map_deep_negative_level(deep_artmap_model):
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    y_last = deep_artmap_model.layers[-1].labels_a[0]
    mapped = deep_artmap_model.map_deep(-1, y_last)
    assert isinstance(mapped.tolist(), int)


def test_map_deep_array_matches_layer_chain(deep_artmap_model):
    X = [np.random.rand(15, 5), np.random.rand(15, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    y_last = deep_artmap_model.layers[-1].labels_a
    mapped = deep_artmap_model.map_deep(-1, y_last)

    expected = y_last
    for layer in reversed(deep_artmap_model.layers):
        expected = layer.map_a2b(expected)
    assert np.array_equal(mapped, expected)


def test_predict_layer_map_cache_parity(deep_artmap_model):
    X = [np.random.rand(12, 5), np.random.rand(12, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    pred = deep_artmap_model.predict(X_prep)
    if len(deep_artmap_model.layers) == 1:
        assert len(deep_artmap_model._layer_map_cache) == 0
        return

    assert len(deep_artmap_model._layer_map_cache) >= 1
    for i, layer in enumerate(deep_artmap_model.layers[:-1]):
        mapped = deep_artmap_model._map_layer_labels(i, pred[i + 1])
        expected = layer.map_a2b(pred[i + 1])
        assert np.array_equal(mapped, expected)


def test_predict_uses_chain_levels_fast_path_when_available(monkeypatch, deep_artmap_model):
    X = [np.random.rand(12, 5), np.random.rand(12, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    calls = {"n": 0}
    orig = deep_artmap_model._map_layer_labels

    def _count_map(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(deep_artmap_model, "_map_layer_labels", _count_map)
    pred = deep_artmap_model.predict(X_prep)
    assert pred[-1].shape[0] == X_prep[-1].shape[0]
    assert calls["n"] == 0


def test_layer_map_cache_refreshes_on_manual_map_value_change(deep_artmap_model):
    X = [np.random.rand(12, 5), np.random.rand(12, 5)]
    y = np.random.randint(0, 2, size=12)
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y=y, max_iter=1)

    layer = deep_artmap_model.layers[0]
    if len(layer.map) == 0:
        pytest.skip("No layer map entries available for cache test.")
    key = int(next(iter(layer.map.keys())))

    arr_before = deep_artmap_model._get_layer_map_array(0)
    assert arr_before is not None
    old_val = int(layer.map[key])
    layer.map[key] = old_val + 1

    arr_after = deep_artmap_model._get_layer_map_array(0)
    assert arr_after is not None
    assert int(arr_after[key]) == old_val + 1


def test_map_chain_cache_reused_and_invalidated(deep_artmap_model):
    X = [np.random.rand(12, 5), np.random.rand(12, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    chain1 = deep_artmap_model._get_map_chain_arrays(0)
    chain2 = deep_artmap_model._get_map_chain_arrays(0)
    assert chain1 is not None
    assert chain1 is chain2

    deep_artmap_model._invalidate_layer_map_cache()
    chain3 = deep_artmap_model._get_map_chain_arrays(0)
    assert chain3 is not None
    assert chain3 is not chain1


def test_get_map_chain_arrays_reuses_cached_layer_signatures(monkeypatch, deep_artmap_model):
    X = [np.random.rand(12, 5), np.random.rand(12, 5)]
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    calls = {"n": 0}
    original_map_signature = deep_artmap_model._map_signature

    def _count_signature(layer):
        calls["n"] += 1
        return original_map_signature(layer)

    monkeypatch.setattr(deep_artmap_model, "_map_signature", _count_signature)
    deep_artmap_model._invalidate_layer_map_cache()
    chain = deep_artmap_model._get_map_chain_arrays(0)

    assert chain is not None
    assert calls["n"] == len(deep_artmap_model.layers[:1])
