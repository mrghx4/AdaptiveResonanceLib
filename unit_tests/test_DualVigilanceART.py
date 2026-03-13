import pytest
import numpy as np
from typing import Optional
from artlib.topological.DualVigilanceART import DualVigilanceART
from artlib.common.BaseART import BaseART


# Mock BaseART class for testing purposes
class MockBaseART(BaseART):
    def __init__(self):
        params = {"rho": 0.7}
        super().__init__(params)
        self.W = []
        self.labels_ = np.array([])
        self.dim_ = 2

    @staticmethod
    def validate_params(params: dict):
        pass

    def prepare_data(self, X: np.ndarray):
        return X

    def restore_data(self, X: np.ndarray):
        return X

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return i

    def add_weight(self, w: np.ndarray):
        self.W.append(w)
        self.weight_sample_counter_.append(1)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        return np.random.random(), {}

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
        op=None,
    ) -> tuple[bool, dict]:
        return True, {}

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        return w

    def get_cluster_centers(self) -> list:
        return [w for w in self.W]

    def check_dimensions(self, X: np.ndarray):
        assert X.shape[1] == self.dim_


@pytest.fixture
def art_model():
    base_module = MockBaseART()
    rho_lower_bound = 0.3
    return DualVigilanceART(
        base_module=base_module, rho_lower_bound=rho_lower_bound
    )


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho_lower_bound"] == 0.3
    assert isinstance(art_model.base_module, BaseART)


def test_prepare_data(art_model):
    # Test the prepare_data method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    prepared_X = art_model.prepare_data(X)
    assert np.array_equal(prepared_X, X)


def test_restore_data(art_model):
    # Test the restore_data method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    restored_X = art_model.restore_data(X)
    assert np.array_equal(restored_X, X)


def test_get_params(art_model):
    # Test the get_params method
    params = art_model.get_params(deep=True)
    assert "rho_lower_bound" in params
    assert "base_module" in params


def test_n_clusters(art_model):
    # Test the n_clusters property
    assert art_model.n_clusters == 0  # No clusters initially
    art_model.map = {0: 0, 1: 1}
    assert art_model.n_clusters == 2  # Two clusters


def test_check_dimensions(art_model):
    # Test the check_dimensions method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)  # Should pass without assertion errors


def test_validate_params(art_model):
    # Test the validate_params method
    valid_params = {"rho_lower_bound": 0.3}
    art_model.validate_params(valid_params)

    invalid_params = {"rho_lower_bound": -0.3}  # Invalid rho_lower_bound
    with pytest.raises(AssertionError):
        art_model.validate_params(invalid_params)


def test_step_fit(art_model):
    # Test the step_fit method
    x = np.array([0.1, 0.2])
    cluster_label = art_model.step_fit(x)
    assert cluster_label == 0  # First sample should create a new cluster


def test_step_pred(art_model):
    # Test the step_pred method
    x = np.array([0.1, 0.2])
    art_model.step_fit(x)  # Create the first cluster
    cluster_label = art_model.step_pred(x)
    assert cluster_label == 0  # Predict should return the correct cluster


def test_predict_matches_step_pred(art_model):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.15, 0.25]])
    art_model.step_fit(X[0])
    art_model.is_fitted_ = True

    pred = art_model.predict(X)
    expected = np.array([art_model.step_pred(x) for x in X])
    np.testing.assert_array_equal(pred, expected)


def test_get_cluster_centers(art_model):
    # Test the get_cluster_centers method
    art_model.step_fit(np.array([0.1, 0.2]))  # Create the first cluster
    centers = art_model.get_cluster_centers()
    assert len(centers) == 1
    assert np.array_equal(centers[0], np.array([0.1, 0.2]))


def test_predict_uses_base_module_batch_predict(monkeypatch, art_model):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.15, 0.25]])
    art_model.base_module.W = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    art_model.map = {0: 5, 1: 9}
    art_model.is_fitted_ = True

    calls = {"n": 0}

    def _predict_batch(data, clip=False):
        calls["n"] += 1
        return np.array([0, 1, 0], dtype=int)

    monkeypatch.setattr(art_model.base_module, "predict", _predict_batch)
    pred = art_model.predict(X)
    np.testing.assert_array_equal(pred, np.array([5, 9, 5]))
    assert calls["n"] == 1


def test_fit_resets_map_state(art_model):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.15, 0.25]])
    art_model.map = {99: 7}
    art_model._next_abstract_label = 42
    art_model.fit(X, max_iter=1)
    assert 99 not in art_model.map
    assert art_model.labels_.shape[0] == X.shape[0]
    assert art_model._next_abstract_label >= 1
