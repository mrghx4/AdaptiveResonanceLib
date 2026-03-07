import numpy as np

from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.TD_FALCONFactory import TD_FALCONFactory
from artlib.reinforcement.FALCON import TD_FALCON


def test_alias_factory_builds_td_falcon_instance():
    model = TD_FALCONFactory(
        state_art=FuzzyART(rho=0.5, alpha=0.01, beta=1.0),
        action_art=FuzzyART(rho=0.7, alpha=0.01, beta=1.0),
        reward_art=FuzzyART(rho=0.9, alpha=0.01, beta=1.0),
        gamma_values=np.array([0.33, 0.33, 0.34]),
        channel_dims=[4, 4, 2],
        backend="python",
    )
    assert isinstance(model, TD_FALCON)
