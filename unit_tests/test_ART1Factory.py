import pytest

from artlib.elementary.ART1 import ART1 as PyART1
from artlib.optimized.ART1Factory import ART1Factory
from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1


def test_factory_cpp_backend_returns_cpp_model():
    model = ART1Factory(rho=0.8, L=1.0, backend="c++")
    assert isinstance(model, CppART1)


def test_factory_python_backend_returns_python_model():
    model = ART1Factory(rho=0.8, L=1.0, backend="python")
    assert isinstance(model, PyART1)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = ART1Factory(rho=0.8, L=1.0, backend="torch")
    assert isinstance(model, CppART1)
