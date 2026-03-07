import pytest

from artlib.elementary.FuzzyART import FuzzyART as PyFuzzyART
from artlib.optimized.FuzzyARTFactory import FuzzyARTFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART


def test_factory_cpp_backend_returns_cpp_model():
    model = FuzzyARTFactory(rho=0.8, alpha=1e-10, beta=1.0, backend="c++")
    assert isinstance(model, CppFuzzyART)


def test_factory_python_backend_returns_python_model():
    model = FuzzyARTFactory(rho=0.8, alpha=1e-10, beta=1.0, backend="python")
    assert isinstance(model, PyFuzzyART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = FuzzyARTFactory(rho=0.8, alpha=1e-10, beta=1.0, backend="torch")
    assert isinstance(model, CppFuzzyART)
