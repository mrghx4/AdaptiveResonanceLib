import pytest

from artlib.common.BaseART import BaseART
from artlib.cvi.CVIART import CVIART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.CVIARTFactory import CVIARTFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART


class DummyCVIARTCompatible(BaseART):
    def __init__(self):
        super().__init__({"rho": 0.5})

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params

    def category_choice(self, i, w, params):
        return 0.0, {}

    def match_criterion(self, i, w, params, cache=None):
        return 0.0, cache

    def update(self, i, w, params, cache=None):
        return w

    def new_weight(self, i, params):
        return i


def test_factory_cpp_accelerates_base_module_when_supported():
    model = CVIARTFactory(
        base_module=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
        validity=CVIART.CALINSKIHARABASZ,
        backend="c++",
    )
    assert isinstance(model, CVIART)
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_python_backend_returns_python_cviart_with_same_base_module():
    base = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    model = CVIARTFactory(
        base_module=base,
        validity=CVIART.CALINSKIHARABASZ,
        backend="python",
    )
    assert isinstance(model, CVIART)
    assert model.base_module is base


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = CVIARTFactory(
            base_module=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            validity=CVIART.CALINSKIHARABASZ,
            backend="torch",
        )
    assert isinstance(model.base_module, CppFuzzyART)


def test_factory_unsupported_base_module_falls_back_with_warning():
    base = DummyCVIARTCompatible()
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = CVIARTFactory(
            base_module=base,
            validity=CVIART.CALINSKIHARABASZ,
            backend="c++",
        )
    assert isinstance(model, CVIART)
    assert model.base_module is base


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = CVIARTFactory(
            base_module=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            validity=CVIART.CALINSKIHARABASZ,
            backend="invalid",
        )
    assert isinstance(model.base_module, CppFuzzyART)
