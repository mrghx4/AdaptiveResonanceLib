import pytest

from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.TopoARTFactory import TopoARTFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART
from artlib.topological.TopoART import TopoART


class DummyTopoARTCompatible(BaseART):
    def __init__(self):
        super().__init__({"rho": 0.6, "alpha": 1e-10, "beta": 0.7})

    @staticmethod
    def validate_params(params: dict):
        assert "beta" in params

    def category_choice(self, i, w, params):
        return 0.0, {}

    def match_criterion(self, i, w, params, cache=None):
        return 0.0, cache

    def update(self, i, w, params, cache=None):
        return w

    def new_weight(self, i, params):
        return i


def test_factory_cpp_accelerates_base_module_when_supported():
    model = TopoARTFactory(
        base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
        beta_lower=0.5,
        tau=10,
        phi=5,
        backend="c++",
    )
    assert isinstance(model, TopoART)
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_python_backend_returns_python_topoart_with_same_base_module():
    base = FuzzyART(rho=0.6, alpha=1e-10, beta=0.7)
    model = TopoARTFactory(
        base_module=base,
        beta_lower=0.5,
        tau=10,
        phi=5,
        backend="python",
    )
    assert isinstance(model, TopoART)
    assert model.base_module is base


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = TopoARTFactory(
            base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
            beta_lower=0.5,
            tau=10,
            phi=5,
            backend="torch",
        )
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_unsupported_base_module_falls_back_with_warning():
    base = DummyTopoARTCompatible()
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = TopoARTFactory(
            base_module=base,
            beta_lower=0.5,
            tau=10,
            phi=5,
            backend="c++",
        )
    assert isinstance(model, TopoART)
    assert model.base_module is base


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = TopoARTFactory(
            base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
            beta_lower=0.5,
            tau=10,
            phi=5,
            backend="invalid",
        )
    assert isinstance(model.base_module, CppFuzzyART)
