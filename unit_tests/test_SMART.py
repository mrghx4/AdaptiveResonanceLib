import pytest
import numpy as np
from artlib.hierarchical.SMART import SMART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART
from matplotlib.axes import Axes


# Fixture to initialize a SMART instance for testing
@pytest.fixture
def smart_model():
    base_params = {"alpha": 0.01, "beta": 1.0}
    rho_values = [0.2, 0.5, 0.7]
    return SMART(FuzzyART, rho_values, base_params)


def test_initialization(smart_model):
    # Test that the model initializes correctly
    assert len(smart_model.rho_values) == 3
    assert isinstance(smart_model.modules[0], BaseART)
    assert isinstance(smart_model.modules[1], BaseART)
    assert isinstance(smart_model.modules[2], BaseART)


def test_prepare_and_restore_data(smart_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 5)

    X_prep = smart_model.prepare_data(X)

    X_restored = smart_model.restore_data(X_prep)
    assert np.allclose(X_restored, X)


def test_fit(smart_model):
    # Test the fit method
    X = np.random.rand(10, 5)

    # Prepare data before fitting
    X_prep = smart_model.prepare_data(X)
    smart_model.fit(X_prep, max_iter=1)

    assert smart_model.modules[0].labels_.shape[0] == X.shape[0]


def test_partial_fit(smart_model):
    # Test the partial_fit method
    X = np.random.rand(10, 5)

    # Prepare data before partial fitting
    X_prep = smart_model.prepare_data(X)
    print(smart_model.n_modules)
    smart_model.partial_fit(X_prep)

    assert smart_model.modules[0].labels_.shape[0] == X.shape[0]


def test_plot_cluster_bounds_vectorizes_map_deep(monkeypatch, smart_model):
    X = np.random.rand(12, 5)
    X_prep = smart_model.prepare_data(X)
    smart_model.fit(X_prep, max_iter=1)

    calls = {"n": 0}
    orig_map_deep = smart_model.map_deep

    def _count_map_deep(level, labels):
        calls["n"] += 1
        return orig_map_deep(level, labels)

    monkeypatch.setattr(smart_model, "map_deep", _count_map_deep)
    ax = type("DummyAxes", (), {})()

    captured = []

    def _capture_plot_cluster_bounds(self, _ax, layer_colors, linewidth):
        captured.append((len(layer_colors), linewidth))

    monkeypatch.setattr(FuzzyART, "plot_cluster_bounds", _capture_plot_cluster_bounds)
    smart_model.plot_cluster_bounds(ax, colors=list(range(64)), linewidth=2)

    assert len(captured) == smart_model.n_modules
    assert calls["n"] == max(0, smart_model.n_modules - 1)
