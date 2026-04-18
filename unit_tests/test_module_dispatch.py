import numpy as np
import pytest

from artlib.elementary.ART1 import ART1
from artlib.elementary.ART2 import ART2A
from artlib.elementary.BayesianART import BayesianART
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.elementary.EllipsoidART import EllipsoidART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.elementary.GaussianART import GaussianART
from artlib.elementary.HypersphereART import HypersphereART
from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

from artlib.optimized._module_dispatch import (
    accelerate_base_art_class,
    accelerate_base_art_module,
)


def test_accelerate_base_art_module_supported_types():
    modules = [
        ART1(rho=0.8, L=1.0),
        ART2A(rho=0.7, alpha=0.1, beta=0.5),
        FuzzyART(rho=0.8, alpha=1e-10, beta=1.0),
        BinaryFuzzyART(rho=0.8),
        GaussianART(rho=0.05, sigma_init=np.array([0.33, 0.33]), alpha=1e-10),
        HypersphereART(rho=0.8, alpha=1e-10, beta=1.0, r_hat=8.0),
        EllipsoidART(rho=0.7, alpha=1e-5, beta=0.1, mu=0.5, r_hat=1.0),
        BayesianART(rho=0.7, cov_init=np.array([[1.0, 0.0], [0.0, 1.0]])),
        QuadraticNeuronART(rho=0.7, s_init=0.5, lr_b=0.1, lr_w=0.1, lr_s=0.05),
    ]

    for m in modules:
        acc = accelerate_base_art_module(m)
        assert acc is not None
        assert ".cpp." in type(acc).__module__


def test_accelerate_base_art_class_supported_types():
    classes = [
        ART1,
        ART2A,
        FuzzyART,
        BinaryFuzzyART,
        GaussianART,
        HypersphereART,
        EllipsoidART,
        BayesianART,
        QuadraticNeuronART,
    ]

    for c in classes:
        acc_cls = accelerate_base_art_class(c)
        assert acc_cls is not None
        assert ".cpp." in acc_cls.__module__


def test_accelerate_base_art_module_refuses_prepared_module_state():
    module = FuzzyART(rho=0.8, alpha=1e-10, beta=1.0)
    module.prepare_data(np.array([[0.1, 0.9], [0.2, 0.8]]))

    acc, reason = accelerate_base_art_module(module, return_reason=True)

    assert acc is None
    assert "prepared data bounds" in reason


def test_accelerate_base_art_module_refuses_fitted_module_state():
    module = FuzzyART(rho=0.8, alpha=1e-10, beta=1.0)
    X = module.prepare_data(np.array([[0.1, 0.9], [0.2, 0.8]]))
    module.fit(X)

    acc, reason = accelerate_base_art_module(module, return_reason=True)

    assert acc is None
    assert "fitted state" in reason
