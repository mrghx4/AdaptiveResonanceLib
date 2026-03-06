import numpy as np
import pytest

from artlib.elementary.GaussianART import GaussianART as PyGaussianART
from artlib.optimized.GaussianARTFactory import GaussianARTFactory
from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGaussianART


def test_factory_cpp_backend_returns_cpp_model():
    model = GaussianARTFactory(
        rho=0.5,
        sigma_init=np.array([0.5, 0.5], dtype=float),
        alpha=1e-10,
        backend="c++",
    )
    assert isinstance(model, CppGaussianART)


def test_factory_python_backend_returns_python_model():
    model = GaussianARTFactory(
        rho=0.5,
        sigma_init=np.array([0.5, 0.5], dtype=float),
        alpha=1e-10,
        backend="python",
    )
    assert isinstance(model, PyGaussianART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = GaussianARTFactory(
            rho=0.5,
            sigma_init=np.array([0.5, 0.5], dtype=float),
            alpha=1e-10,
            backend="torch",
        )
    assert isinstance(model, CppGaussianART)
