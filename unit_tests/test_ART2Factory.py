import pytest

from artlib.elementary.ART2 import ART2A as PyART2A
from artlib.optimized.ART2Factory import ART2Factory
from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A


def test_factory_cpp_backend_returns_cpp_model():
    model = ART2Factory(rho=0.7, alpha=0.1, beta=0.5, backend="c++")
    assert isinstance(model, CppART2A)


def test_factory_python_backend_returns_python_model():
    model = ART2Factory(rho=0.7, alpha=0.1, beta=0.5, backend="python")
    assert isinstance(model, PyART2A)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = ART2Factory(rho=0.7, alpha=0.1, beta=0.5, backend="torch")
    assert isinstance(model, CppART2A)
