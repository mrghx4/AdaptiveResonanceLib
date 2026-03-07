import pytest

from artlib.elementary.FuzzyART import FuzzyART
from artlib.hierarchical.SMART import SMART
from artlib.optimized.SMARTFactory import SMARTFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART


def _rho_values():
    return [0.2, 0.5, 0.7]


def _base_params():
    return {"alpha": 0.01, "beta": 1.0}


def test_factory_cpp_accelerates_supported_base_class():
    model = SMARTFactory(
        base_ART_class=FuzzyART,
        rho_values=_rho_values(),
        base_params=_base_params(),
        backend="c++",
    )
    assert isinstance(model, SMART)
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)


def test_factory_python_backend_uses_python_base_class():
    model = SMARTFactory(
        base_ART_class=FuzzyART,
        rho_values=_rho_values(),
        base_params=_base_params(),
        backend="python",
    )
    assert isinstance(model, SMART)
    assert all(type(m) is FuzzyART for m in model.modules)


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = SMARTFactory(
            base_ART_class=FuzzyART,
            rho_values=_rho_values(),
            base_params=_base_params(),
            backend="torch",
        )
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = SMARTFactory(
            base_ART_class=FuzzyART,
            rho_values=_rho_values(),
            base_params=_base_params(),
            backend="invalid",
        )
    assert all(isinstance(m, CppFuzzyART) for m in model.modules)
