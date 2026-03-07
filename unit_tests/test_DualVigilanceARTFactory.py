import pytest

from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.DualVigilanceARTFactory import DualVigilanceARTFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART
from artlib.topological.DualVigilanceART import DualVigilanceART


class DummyDualVigilanceCompatible(BaseART):
    def __init__(self):
        super().__init__({"rho": 0.6})

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params

    def category_choice(self, i, w, params):
        return 0.0, {}

    def match_criterion(self, i, w, params, cache=None):
        return 0.0, cache

    def update(self, i, w, params, cache=None):
        return w

    def new_weight(self, i, params):
        return i


def test_factory_cpp_accelerates_base_module_when_supported():
    model = DualVigilanceARTFactory(
        base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
        rho_lower_bound=0.3,
        backend="c++",
    )
    assert isinstance(model, DualVigilanceART)
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_python_backend_returns_python_dualvigilance_with_same_base_module():
    base = FuzzyART(rho=0.6, alpha=1e-10, beta=0.7)
    model = DualVigilanceARTFactory(
        base_module=base,
        rho_lower_bound=0.3,
        backend="python",
    )
    assert isinstance(model, DualVigilanceART)
    assert model.base_module is base


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = DualVigilanceARTFactory(
            base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
            rho_lower_bound=0.3,
            backend="torch",
        )
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_unsupported_base_module_falls_back_with_warning():
    base = DummyDualVigilanceCompatible()
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = DualVigilanceARTFactory(
            base_module=base,
            rho_lower_bound=0.3,
            backend="c++",
        )
    assert isinstance(model, DualVigilanceART)
    assert model.base_module is base


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = DualVigilanceARTFactory(
            base_module=FuzzyART(rho=0.6, alpha=1e-10, beta=0.7),
            rho_lower_bound=0.3,
            backend="invalid",
        )
    assert isinstance(model.base_module, CppFuzzyART)
