import numpy as np
import pytest

from artlib.optimized.BayesianARTMAPFactory import BayesianARTMAPFactory
from artlib.optimized.backends.cpp.BayesianARTMAP import BayesianARTMAP as CppBayesianARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP


def test_factory_cpp_backend_returns_cpp_model():
    model = BayesianARTMAPFactory(rho=0.7, cov_init=np.eye(2), backend="c++")
    assert isinstance(model, CppBayesianARTMAP)


def test_factory_python_backend_returns_simple_artmap():
    model = BayesianARTMAPFactory(rho=0.7, cov_init=np.eye(2), backend="python")
    assert isinstance(model, SimpleARTMAP)


def test_factory_torch_falls_back_to_cpp_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = BayesianARTMAPFactory(rho=0.7, cov_init=np.eye(2), backend="torch")
    assert isinstance(model, CppBayesianARTMAP)
