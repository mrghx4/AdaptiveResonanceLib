import pytest

from artlib.biclustering.BARTMAP import BARTMAP
from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.BARTMAPFactory import BARTMAPFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART


class DummyBARTMAPCompatible(BaseART):
    def __init__(self):
        super().__init__({"rho": 0.5})

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


def test_factory_cpp_accelerates_supported_modules():
    model = BARTMAPFactory(
        module_a=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
        module_b=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
        eta=0.01,
        backend="c++",
    )
    assert isinstance(model, BARTMAP)
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, CppFuzzyART)


def test_factory_python_backend_returns_python_modules():
    a = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    b = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    model = BARTMAPFactory(module_a=a, module_b=b, eta=0.01, backend="python")
    assert isinstance(model, BARTMAP)
    assert model.module_a is a
    assert model.module_b is b


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = BARTMAPFactory(
            module_a=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            module_b=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            eta=0.01,
            backend="torch",
        )
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, CppFuzzyART)


def test_factory_unsupported_module_warns_and_keeps_python_module():
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = BARTMAPFactory(
            module_a=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            module_b=DummyBARTMAPCompatible(),
            eta=0.01,
            backend="c++",
        )
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, DummyBARTMAPCompatible)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = BARTMAPFactory(
            module_a=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            module_b=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            eta=0.01,
            backend="invalid",
        )
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, CppFuzzyART)
