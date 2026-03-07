import pytest

from artlib.elementary.HypersphereART import HypersphereART as PyHypersphereART
from artlib.optimized.HypersphereARTFactory import HypersphereARTFactory
from artlib.optimized.backends.cpp.HypersphereART import HypersphereART as CppHypersphereART


def test_factory_cpp_backend_returns_cpp_model():
    model = HypersphereARTFactory(
        rho=0.7,
        alpha=1e-10,
        beta=1.0,
        r_hat=0.8,
        backend="c++",
    )
    assert isinstance(model, CppHypersphereART)


def test_factory_python_backend_returns_python_model():
    model = HypersphereARTFactory(
        rho=0.7,
        alpha=1e-10,
        beta=1.0,
        r_hat=0.8,
        backend="python",
    )
    assert isinstance(model, PyHypersphereART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = HypersphereARTFactory(
            rho=0.7,
            alpha=1e-10,
            beta=1.0,
            r_hat=0.8,
            backend="torch",
        )
    assert isinstance(model, CppHypersphereART)
