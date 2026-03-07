import pytest

from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART as PyBinaryFuzzyART
from artlib.optimized.BinaryFuzzyARTFactory import BinaryFuzzyARTFactory
from artlib.optimized.backends.cpp.BinaryFuzzyART import BinaryFuzzyART as CppBinaryFuzzyART


def test_factory_cpp_backend_returns_cpp_model():
    model = BinaryFuzzyARTFactory(rho=0.8, backend="c++")
    assert isinstance(model, CppBinaryFuzzyART)


def test_factory_python_backend_returns_python_model():
    model = BinaryFuzzyARTFactory(rho=0.8, backend="python")
    assert isinstance(model, PyBinaryFuzzyART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = BinaryFuzzyARTFactory(rho=0.8, backend="torch")
    assert isinstance(model, CppBinaryFuzzyART)
