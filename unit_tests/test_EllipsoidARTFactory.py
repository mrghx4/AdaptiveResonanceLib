import pytest

from artlib.elementary.EllipsoidART import EllipsoidART as PyEllipsoidART
from artlib.optimized.EllipsoidARTFactory import EllipsoidARTFactory
from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEllipsoidART


def test_factory_cpp_backend_returns_cpp_model():
    model = EllipsoidARTFactory(
        rho=0.7,
        alpha=1e-5,
        beta=0.1,
        mu=0.5,
        r_hat=1.0,
        backend="c++",
    )
    assert isinstance(model, CppEllipsoidART)


def test_factory_python_backend_returns_python_model():
    model = EllipsoidARTFactory(
        rho=0.7,
        alpha=1e-5,
        beta=0.1,
        mu=0.5,
        r_hat=1.0,
        backend="python",
    )
    assert isinstance(model, PyEllipsoidART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = EllipsoidARTFactory(
            rho=0.7,
            alpha=1e-5,
            beta=0.1,
            mu=0.5,
            r_hat=1.0,
            backend="torch",
        )
    assert isinstance(model, CppEllipsoidART)
