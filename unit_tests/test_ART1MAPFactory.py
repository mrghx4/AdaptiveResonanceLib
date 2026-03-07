import pytest

from artlib.optimized.ART1MAPFactory import ART1MAPFactory
from artlib.optimized.backends.cpp.ART1MAP import ART1MAP as CppART1MAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP


def test_factory_cpp_backend_returns_cpp_model():
    model = ART1MAPFactory(rho=0.8, L=1.0, backend="c++")
    assert isinstance(model, CppART1MAP)


def test_factory_python_backend_returns_python_model():
    model = ART1MAPFactory(rho=0.8, L=1.0, backend="python")
    assert isinstance(model, SimpleARTMAP)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = ART1MAPFactory(rho=0.8, L=1.0, backend="torch")
    assert isinstance(model, CppART1MAP)
