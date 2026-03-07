import pytest

from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.hierarchical.DeepARTMAP import DeepARTMAP
from artlib.optimized.DeepARTMAPFactory import DeepARTMAPFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART


class DummyDeepARTMAPCompatible(BaseART):
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
    model = DeepARTMAPFactory(
        modules=[
            FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
        ],
        backend="c++",
    )
    assert isinstance(model, DeepARTMAP)
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)


def test_factory_python_backend_returns_python_modules():
    m1 = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    m2 = FuzzyART(rho=0.7, alpha=0.01, beta=1.0)
    model = DeepARTMAPFactory(modules=[m1, m2], backend="python")
    assert isinstance(model, DeepARTMAP)
    assert model.modules[0] is m1
    assert model.modules[1] is m2


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = DeepARTMAPFactory(
            modules=[
                FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
                FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            ],
            backend="torch",
        )
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)


def test_factory_unsupported_module_warns_and_keeps_python_layer():
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = DeepARTMAPFactory(
            modules=[FuzzyART(rho=0.5, alpha=0.01, beta=1.0), DummyDeepARTMAPCompatible()],
            backend="c++",
        )
    assert isinstance(model.modules[0], CppFuzzyART)
    assert isinstance(model.modules[1], DummyDeepARTMAPCompatible)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = DeepARTMAPFactory(
            modules=[
                FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
                FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            ],
            backend="invalid",
        )
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)
