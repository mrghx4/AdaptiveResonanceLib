import pytest

from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.ARTMAPFactory import ARTMAPFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART
from artlib.supervised.ARTMAP import ARTMAP


class DummyART(BaseART):
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


def test_factory_cpp_accelerates_module_a_when_supported():
    model = ARTMAPFactory(
        module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
        module_b=FuzzyART(rho=0.6, alpha=1e-10, beta=1.0),
        backend="c++",
    )
    assert isinstance(model, ARTMAP)
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, CppFuzzyART)


def test_factory_python_backend_returns_python_artmap():
    a = FuzzyART(rho=0.8, alpha=1e-10, beta=1.0)
    b = FuzzyART(rho=0.6, alpha=1e-10, beta=1.0)
    model = ARTMAPFactory(module_a=a, module_b=b, backend="python")
    assert isinstance(model, ARTMAP)
    assert model.module_a is a
    assert model.module_b is b


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = ARTMAPFactory(
            module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
            module_b=FuzzyART(rho=0.6, alpha=1e-10, beta=1.0),
            backend="torch",
        )
    assert isinstance(model.module_a, CppFuzzyART)
    assert isinstance(model.module_b, CppFuzzyART)


def test_factory_unsupported_module_a_keeps_python_impl_with_warning():
    d = DummyART(params={"rho": 0.5})
    model_b = FuzzyART(rho=0.6, alpha=1e-10, beta=1.0)
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping for module_a"):
        model = ARTMAPFactory(module_a=d, module_b=model_b, backend="c++")
    assert isinstance(model, ARTMAP)
    assert model.module_a is d
    assert isinstance(model.module_b, CppFuzzyART)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = ARTMAPFactory(
            module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
            module_b=FuzzyART(rho=0.6, alpha=1e-10, beta=1.0),
            backend="not-real",
        )
    assert isinstance(model.module_a, CppFuzzyART)
