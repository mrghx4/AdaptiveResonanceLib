import numpy as np
import pytest

from artlib.elementary.ART2 import ART2A
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.SimpleARTMAPFactory import SimpleARTMAPFactory
from artlib.optimized.backends.cpp.FuzzyARTMAP import FuzzyARTMAP as CppFuzzyARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP


def test_factory_cpp_dispatches_fuzzyart_to_cpp_artmap():
    model = SimpleARTMAPFactory(
        module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
        backend="c++",
    )
    assert isinstance(model, CppFuzzyARTMAP)


def test_factory_python_backend_returns_python_simple_artmap():
    model = SimpleARTMAPFactory(
        module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
        backend="python",
    )
    assert isinstance(model, SimpleARTMAP)


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = SimpleARTMAPFactory(
            module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
            backend="torch",
        )
    assert isinstance(model, CppFuzzyARTMAP)


def test_factory_unsupported_module_falls_back_to_python_simple_artmap():
    with pytest.warns(
        RuntimeWarning, match=r"No specialized c\+\+ SimpleARTMAP backend"
    ):
        model = SimpleARTMAPFactory(
            module_a=ART2A(rho=0.7, alpha=0.1, beta=0.5),
            backend="c++",
        )
    assert isinstance(model, SimpleARTMAP)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = SimpleARTMAPFactory(
            module_a=FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
            backend="not-real",
        )
    assert isinstance(model, CppFuzzyARTMAP)
