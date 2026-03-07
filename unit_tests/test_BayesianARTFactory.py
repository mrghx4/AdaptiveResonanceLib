import numpy as np
import pytest

from artlib.elementary.BayesianART import BayesianART as PyBayesianART
from artlib.optimized.BayesianARTFactory import BayesianARTFactory
from artlib.optimized.backends.cpp.BayesianART import BayesianART as CppBayesianART


def test_factory_cpp_backend_returns_cpp_model():
    model = BayesianARTFactory(rho=0.7, cov_init=np.eye(2), backend="c++")
    assert isinstance(model, CppBayesianART)


def test_factory_python_backend_returns_python_model():
    model = BayesianARTFactory(rho=0.7, cov_init=np.eye(2), backend="python")
    assert isinstance(model, PyBayesianART)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = BayesianARTFactory(rho=0.7, cov_init=np.eye(2), backend="torch")
    assert isinstance(model, CppBayesianART)
