import pytest
import numpy as np
from typing import Optional
import artlib.biclustering.BARTMAP as bartmap_module
from artlib.biclustering.BARTMAP import BARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART


# Fixture to initialize a BARTMAP instance for testing
@pytest.fixture
def bartmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.5, 0.01, 1.0)
    return BARTMAP(module_a=module_a, module_b=module_b, eta=0.01)


def test_initialization(bartmap_model):
    # Test that the model initializes correctly
    assert bartmap_model.params["eta"] == 0.01
    assert isinstance(bartmap_model.module_a, BaseART)
    assert isinstance(bartmap_model.module_b, BaseART)


def test_validate_params():
    # Test the validate_params method
    valid_params = {"eta": 0.5}
    BARTMAP.validate_params(valid_params)

    invalid_params = {"eta": "invalid"}  # eta should be a float
    with pytest.raises(AssertionError):
        BARTMAP.validate_params(invalid_params)


def test_get_params(bartmap_model):
    # Test the get_params method
    params = bartmap_model.get_params()
    assert "eta" in params
    assert "module_a" in params
    assert "module_b" in params


def test_set_params(bartmap_model):
    # Test the set_params method
    bartmap_model.set_params(eta=0.7)
    assert bartmap_model.eta == 0.7


def test_step_fit(bartmap_model):
    # Test the step_fit method
    X = np.random.rand(10, 10)

    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)

    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    # init module A
    bartmap_model.module_a.W = []
    bartmap_model.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

    c_a = bartmap_model.step_fit(X_a, 0)
    assert isinstance(c_a, int)  # Ensure the result is an integer cluster label
    assert bartmap_model._match_reset_extra is None


def test_match_criterion_bin(bartmap_model):
    # Test the match_criterion_bin method
    X = np.random.rand(100, 100)

    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)

    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    # init module A
    bartmap_model.module_a.W = []
    bartmap_model.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
    c_a = bartmap_model.step_fit(X_a, 0)

    result = bartmap_model.match_criterion_bin(X, 9, 0, {"eta": 0.5})
    assert isinstance(result, bool)  # Ensure the result is a boolean


def test_average_pearson_corr_python_fallback(monkeypatch, bartmap_model):
    X = np.random.rand(12, 12)
    bartmap_model.X = X

    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    monkeypatch.setattr(bartmap_module, "AveragePearsonCorr", None)
    r = bartmap_model._average_pearson_corr(X, k=0, c_b=0)
    assert isinstance(r, float)


def test_match_reset_func_python_fallback(monkeypatch, bartmap_model):
    X = np.random.rand(10, 10)
    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    # init module A
    bartmap_model.module_a.W = []
    bartmap_model.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
    bartmap_model.step_fit(X_a, 0)

    monkeypatch.setattr(bartmap_module, "AnyClusterMatch", None)
    result = bartmap_model.match_reset_func(
        i=X_a[0],
        w=np.zeros_like(X_a[0]),
        cluster_a=0,
        params={},
        extra={"k": 0},
    )
    assert isinstance(result, bool)


def test_cpp_metric_cache_reused(bartmap_model):
    X = np.random.rand(10, 10)
    bartmap_model.X = X

    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    r1 = bartmap_model._average_pearson_corr(X, k=0, c_b=0)
    cache_id_1 = id(bartmap_model._cpp_metrics_X64)
    labels_id_1 = id(bartmap_model._cpp_metrics_labels)

    r2 = bartmap_model._average_pearson_corr(X, k=1, c_b=0)
    cache_id_2 = id(bartmap_model._cpp_metrics_X64)
    labels_id_2 = id(bartmap_model._cpp_metrics_labels)

    assert isinstance(r1, float)
    assert isinstance(r2, float)
    assert cache_id_1 == cache_id_2
    assert labels_id_1 == labels_id_2


def test_cpp_metric_cache_uses_original_when_already_compatible(bartmap_model):
    X = np.random.rand(8, 8).astype(np.float64, copy=False)
    bartmap_model.X = X

    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)
    X_cpp, labels_cpp = bartmap_model._ensure_cpp_metric_cache(X)
    assert X_cpp is X
    assert labels_cpp.dtype == np.int32
    assert labels_cpp.flags["C_CONTIGUOUS"]


def test_cpp_label_converter_no_copy_for_int32_contiguous():
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    out = BARTMAP._to_cpp_int32_c(labels)
    assert out is labels


def test_match_reset_func_memoizes_any_cluster_match(monkeypatch, bartmap_model):
    X = np.random.rand(10, 10)
    bartmap_model.X = X

    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    call_counter = {"n": 0}

    def _fake_any_cluster_match(Xv, k, n_clusters_b, eta, labels):
        call_counter["n"] += 1
        return True

    monkeypatch.setattr(bartmap_module, "AnyClusterMatch", _fake_any_cluster_match)
    state = {}

    r1 = bartmap_model.match_reset_func(
        i=X[0],
        w=np.zeros_like(X[0]),
        cluster_a=0,
        params={},
        extra={"k": 0, "match_state": state},
    )
    r2 = bartmap_model.match_reset_func(
        i=X[0],
        w=np.zeros_like(X[0]),
        cluster_a=1,
        params={},
        extra={"k": 0, "match_state": state},
    )

    assert r1 is True
    assert r2 is True
    assert call_counter["n"] == 1


def test_fit_sets_cached_module_b_cluster_count(bartmap_model):
    X = np.random.rand(20, 20)
    bartmap_model.fit(X, max_iter=1)
    assert bartmap_model._module_b_n_clusters_cached == len(bartmap_model.module_b.W)


def test_fit(bartmap_model):
    # Test the fit method
    X = np.random.rand(100, 100)

    bartmap_model.fit(X, max_iter=1)

    # Check that rows_ and columns_ are set
    assert hasattr(bartmap_model, "rows_")
    assert hasattr(bartmap_model, "columns_")

    # Check that the rows and columns shapes match the expected size
    assert (
        bartmap_model.rows_.shape[0]
        == bartmap_model.module_a.n_clusters * bartmap_model.module_b.n_clusters
    )
    assert (
        bartmap_model.columns_.shape[0]
        == bartmap_model.module_a.n_clusters * bartmap_model.module_b.n_clusters
    )


def test_fit_rows_columns_layout_matches_reference(bartmap_model):
    X = np.random.rand(40, 40)
    bartmap_model.fit(X, max_iter=1)

    rows_ref = np.vstack(
        [
            bartmap_model.row_labels_ == label
            for label in range(bartmap_model.module_a.n_clusters)
            for _ in range(bartmap_model.module_b.n_clusters)
        ]
    )
    cols_ref = np.vstack(
        [
            bartmap_model.column_labels_ == label
            for _ in range(bartmap_model.module_a.n_clusters)
            for label in range(bartmap_model.module_b.n_clusters)
        ]
    )
    assert np.array_equal(bartmap_model.rows_, rows_ref)
    assert np.array_equal(bartmap_model.columns_, cols_ref)


def test_step_fit_reuses_match_reset_state(bartmap_model):
    X = np.random.rand(12, 12)
    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)
    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)
    seen_state_ids = []
    seen_callback_ids = []

    def _capture_step_fit(x, match_reset_func=None, **kwargs):
        seen_state_ids.append(id(bartmap_model._match_reset_extra))
        seen_callback_ids.append(id(match_reset_func))
        return 0

    bartmap_model.module_a.step_fit = _capture_step_fit

    state_id_1 = id(bartmap_model._match_reset_state)
    callback_id_1 = id(bartmap_model._step_match_reset_func_cached)
    bartmap_model.step_fit(X_a, 0)
    state_id_2 = id(bartmap_model._match_reset_state)
    callback_id_2 = id(bartmap_model._step_match_reset_func_cached)
    bartmap_model.step_fit(X_a, 1)

    assert bartmap_model._match_reset_extra is None
    assert len(seen_state_ids) == 2
    assert len(seen_callback_ids) == 2
    assert len(set(seen_state_ids)) == 1
    assert len(set(seen_callback_ids)) == 1
    assert id(bartmap_model._match_reset_state) == state_id_1 == state_id_2
    assert id(bartmap_model._step_match_reset_func_cached) == callback_id_1 == callback_id_2
