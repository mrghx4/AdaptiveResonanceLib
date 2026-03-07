import pytest

from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART as PyQuadraticNeuronART
from artlib.optimized.QuadraticNeuronARTFactory import QuadraticNeuronARTFactory
from artlib.optimized.backends.cpp.QuadraticNeuronART import (
    QuadraticNeuronART as CppQuadraticNeuronART,
)


def test_factory_cpp_backend_returns_cpp_model():
    model = QuadraticNeuronARTFactory(
        rho=0.7, s_init=0.5, lr_b=0.1, lr_w=0.1, lr_s=0.05, backend="c++"
    )
    assert isinstance(model, CppQuadraticNeuronART)


def test_factory_python_backend_returns_python_model():
    model = QuadraticNeuronARTFactory(
        rho=0.7, s_init=0.5, lr_b=0.1, lr_w=0.1, lr_s=0.05, backend="python"
    )
    assert isinstance(model, PyQuadraticNeuronART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = QuadraticNeuronARTFactory(
            rho=0.7, s_init=0.5, lr_b=0.1, lr_w=0.1, lr_s=0.05, backend="torch"
        )
    assert isinstance(model, CppQuadraticNeuronART)
