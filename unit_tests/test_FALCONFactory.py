import numpy as np
import pytest

from artlib.common.BaseART import BaseART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.FALCONFactory import FALCONFactory
from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART
from artlib.reinforcement.FALCON import FALCON


class DummyFalconCompatible(BaseART):
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


def _gamma():
    return np.array([0.33, 0.33, 0.34])


def _dims():
    return [4, 4, 2]


def test_factory_cpp_accelerates_supported_modules():
    model = FALCONFactory(
        state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
        action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
        reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
        gamma_values=_gamma(),
        channel_dims=_dims(),
        backend="c++",
    )
    assert isinstance(model, FALCON)
    assert all(isinstance(m, CppFuzzyART) for m in model.fusion_art.modules)


def test_factory_python_backend_returns_python_modules():
    s = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    a = FuzzyART(rho=0.7, alpha=0.01, beta=1.0)
    r = FuzzyART(rho=0.9, alpha=0.01, beta=1.0)
    model = FALCONFactory(
        state_art=s,
        action_art=a,
        reward_art=r,
        gamma_values=_gamma(),
        channel_dims=_dims(),
        backend="python",
    )
    assert isinstance(model, FALCON)
    assert model.fusion_art.modules[0] is s
    assert model.fusion_art.modules[1] is a
    assert model.fusion_art.modules[2] is r


def test_factory_torch_falls_back_to_cpp_dispatch_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = FALCONFactory(
            state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
            gamma_values=_gamma(),
            channel_dims=_dims(),
            backend="torch",
        )
    assert all(isinstance(m, CppFuzzyART) for m in model.fusion_art.modules)


def test_factory_unsupported_module_warns_and_keeps_python_channel():
    with pytest.warns(RuntimeWarning, match=r"No c\+\+ acceleration mapping"):
        model = FALCONFactory(
            state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            action_art=DummyFalconCompatible(),
            reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
            gamma_values=_gamma(),
            channel_dims=_dims(),
            backend="c++",
        )
    assert isinstance(model.fusion_art.modules[0], CppFuzzyART)
    assert isinstance(model.fusion_art.modules[1], DummyFalconCompatible)
    assert isinstance(model.fusion_art.modules[2], CppFuzzyART)


def test_factory_unknown_backend_warns_and_defaults_to_cpp_dispatch():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = FALCONFactory(
            state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
            gamma_values=_gamma(),
            channel_dims=_dims(),
            backend="invalid",
    )
    assert all(isinstance(m, CppFuzzyART) for m in model.fusion_art.modules)


def test_factory_requires_channel_dims():
    with pytest.raises(TypeError, match="channel_dims must be provided explicitly"):
        FALCONFactory(
            state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
            action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
        )


def test_factory_cpp_keeps_prepared_python_channel_with_warning():
    prepared = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
    prepared.prepare_data(np.array([[0.1, 0.2], [0.3, 0.4]]))

    with pytest.warns(RuntimeWarning, match="prepared data bounds"):
        model = FALCONFactory(
            state_art=prepared,
            action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
            reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
            gamma_values=_gamma(),
            channel_dims=_dims(),
            backend="c++",
        )

    assert model.fusion_art.modules[0] is prepared
    assert isinstance(model.fusion_art.modules[1], CppFuzzyART)
    assert isinstance(model.fusion_art.modules[2], CppFuzzyART)
