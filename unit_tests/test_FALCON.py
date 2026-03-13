import pytest
import numpy as np
from artlib.elementary.FuzzyART import FuzzyART
from artlib.reinforcement.FALCON import FALCON


@pytest.fixture
def falcon_model():
    # Initialize FALCON with three FuzzyART modules
    state_art = FuzzyART(0.5, 0.01, 1.0)
    action_art = FuzzyART(0.7, 0.01, 1.0)
    reward_art = FuzzyART(0.9, 0.01, 1.0)
    channel_dims = [4, 4, 2]
    return FALCON(
        state_art=state_art,
        action_art=action_art,
        reward_art=reward_art,
        channel_dims=channel_dims,
    )


def test_falcon_initialization(falcon_model):
    # Test that the model initializes correctly
    assert isinstance(falcon_model.fusion_art.modules[0], FuzzyART)
    assert isinstance(falcon_model.fusion_art.modules[1], FuzzyART)
    assert isinstance(falcon_model.fusion_art.modules[2], FuzzyART)
    assert falcon_model.fusion_art.channel_dims == [4, 4, 2]


def test_falcon_fit(falcon_model):
    # Test the fit method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    assert len(falcon_model.fusion_art.W) > 0
    assert falcon_model.fusion_art.labels_.shape[0] == states_prep.shape[0]


def test_falcon_partial_fit(falcon_model):
    # Test the partial_fit method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    assert len(falcon_model.fusion_art.W) > 0
    assert falcon_model.fusion_art.labels_.shape[0] == states_prep.shape[0]


def test_falcon_get_actions_and_rewards(falcon_model):
    # Test the get_actions_and_rewards method
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space, rewards = falcon_model.get_actions_and_rewards(
        states_prep[0, :]
    )

    assert action_space.shape[0] > 0
    assert rewards.shape[0] > 0


def test_falcon_get_action(falcon_model):
    # Test the get_action method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action = falcon_model.get_action(states_prep[0, :])

    assert action.shape[0] == actions.shape[1]


def test_falcon_get_probabilistic_action(falcon_model):
    # Test the get_probabilistic_action method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action = falcon_model.get_probabilistic_action(states_prep[0, :])

    assert isinstance(action.tolist(), float)


def test_falcon_get_rewards(falcon_model):
    # Test the get_rewards method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    predicted_rewards = falcon_model.get_rewards(states_prep, actions_prep)

    assert predicted_rewards.shape == rewards.shape


def test_get_actions_and_rewards_uses_single_predict_call(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    calls = {"n": 0}
    orig_predict = falcon_model.fusion_art.predict

    def _predict_once(*args, **kwargs):
        calls["n"] += 1
        return orig_predict(*args, **kwargs)

    monkeypatch.setattr(falcon_model.fusion_art, "predict", _predict_once)
    falcon_model.get_actions_and_rewards(states_prep[0, :])
    assert calls["n"] == 1


def test_get_probabilistic_action_handles_zero_reward_distribution(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space = actions[:4]
    zero_rewards = np.zeros((4, 1), dtype=float)

    def _zero_rewards(*args, **kwargs):
        return action_space, zero_rewards

    monkeypatch.setattr(falcon_model, "get_actions_and_rewards", _zero_rewards)
    action = falcon_model.get_probabilistic_action(states_prep[0, :])
    assert np.isscalar(action)


def test_default_action_space_cache_reuses_prepared_actions(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    calls = {"n": 0}
    orig_prepare = falcon_model.fusion_art.modules[1].prepare_data

    def _count_prepare(*args, **kwargs):
        calls["n"] += 1
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(
        falcon_model.fusion_art.modules[1], "prepare_data", _count_prepare
    )
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=None)
    falcon_model.get_actions_and_rewards(states_prep[1, :], action_space=None)
    assert calls["n"] == 1


def test_action_space_cache_invalidates_after_partial_fit(monkeypatch, falcon_model):
    states = np.random.rand(12, 2)
    actions = np.random.rand(12, 2)
    rewards = np.random.rand(12, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    # Build initial cache
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=None)

    calls = {"n": 0}
    orig_prepare = falcon_model.fusion_art.modules[1].prepare_data

    def _count_prepare(*args, **kwargs):
        calls["n"] += 1
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(
        falcon_model.fusion_art.modules[1], "prepare_data", _count_prepare
    )
    falcon_model.partial_fit(states_prep[:4], actions_prep[:4], rewards_prep[:4])
    falcon_model.get_actions_and_rewards(states_prep[1, :], action_space=None)
    assert calls["n"] >= 1


def test_default_action_queries_do_not_use_join_channel_data(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    def _fail_join(*args, **kwargs):
        raise AssertionError("join_channel_data should not be used for default action queries")

    monkeypatch.setattr(falcon_model.fusion_art, "join_channel_data", _fail_join)
    action_space, reward_values = falcon_model.get_actions_and_rewards(
        states_prep[0, :], action_space=None
    )
    assert action_space.shape[0] == reward_values.shape[0]


def test_external_action_queries_build_full_width_data(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space = np.random.rand(3, 2)
    prepared_actions = falcon_model.fusion_art.modules[1].prepare_data(action_space)
    captured = {}

    def _capture_predict(data, *args, **kwargs):
        captured["data"] = np.array(data, copy=True)
        return np.zeros((data.shape[0],), dtype=int)

    monkeypatch.setattr(falcon_model.fusion_art, "predict", _capture_predict)
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=action_space)

    assert captured["data"].shape == (
        prepared_actions.shape[0],
        sum(falcon_model.fusion_art.channel_dims),
    )
    np.testing.assert_allclose(
        captured["data"][:, : falcon_model.fusion_art.channel_dims[0]],
        np.repeat(states_prep[[0], :], prepared_actions.shape[0], axis=0),
    )
    state_dim = falcon_model.fusion_art.channel_dims[0]
    action_dim = falcon_model.fusion_art.channel_dims[1]
    np.testing.assert_allclose(
        captured["data"][:, state_dim : state_dim + action_dim], prepared_actions
    )
    np.testing.assert_allclose(captured["data"][:, state_dim + action_dim :], 0.5)


def test_external_action_space_cache_reuses_prepared_actions(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space = actions[:4]
    calls = {"n": 0}
    orig_prepare = falcon_model.fusion_art.modules[1].prepare_data

    def _count_prepare(*args, **kwargs):
        calls["n"] += 1
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(
        falcon_model.fusion_art.modules[1], "prepare_data", _count_prepare
    )
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=action_space)
    falcon_model.get_actions_and_rewards(states_prep[1, :], action_space=action_space)
    assert calls["n"] == 1


def test_external_action_queries_reuse_cached_template(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space = actions[:4]
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=action_space)

    def _fail_build_query(*args, **kwargs):
        raise AssertionError("_build_state_action_query should not be used once the external template is cached")

    monkeypatch.setattr(falcon_model, "_build_state_action_query", _fail_build_query)
    action_space_out, rewards_out = falcon_model.get_actions_and_rewards(
        states_prep[1, :], action_space=action_space
    )
    assert action_space_out.shape[0] == rewards_out.shape[0]


def test_external_action_space_cache_invalidates_after_partial_fit(monkeypatch, falcon_model):
    states = np.random.rand(12, 2)
    actions = np.random.rand(12, 2)
    rewards = np.random.rand(12, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    action_space = actions[:4]
    falcon_model.get_actions_and_rewards(states_prep[0, :], action_space=action_space)

    calls = {"n": 0}
    orig_prepare = falcon_model.fusion_art.modules[1].prepare_data

    def _count_prepare(*args, **kwargs):
        calls["n"] += 1
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(
        falcon_model.fusion_art.modules[1], "prepare_data", _count_prepare
    )
    falcon_model.partial_fit(states_prep[:4], actions_prep[:4], rewards_prep[:4])
    falcon_model.get_actions_and_rewards(states_prep[1, :], action_space=action_space)
    assert calls["n"] >= 1
    assert falcon_model._external_action_query_template_cache is not None


def test_get_rewards_does_not_use_join_channel_data(monkeypatch, falcon_model):
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(
        states, actions, rewards
    )
    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    def _fail_join(*args, **kwargs):
        raise AssertionError("join_channel_data should not be used for get_rewards")

    monkeypatch.setattr(falcon_model.fusion_art, "join_channel_data", _fail_join)
    predicted_rewards = falcon_model.get_rewards(states_prep, actions_prep)
    assert predicted_rewards.shape == rewards.shape


def test_get_rewards_rejects_mismatched_state_action_rows(falcon_model):
    states = np.random.rand(5, 4)
    actions = np.random.rand(4, 4)
    with pytest.raises(ValueError, match="same number of rows"):
        falcon_model.get_rewards(states, actions)


def test_get_rewards_rejects_invalid_action_width(falcon_model):
    states = np.random.rand(5, 4)
    actions = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="actions width"):
        falcon_model.get_rewards(states, actions)


def test_get_actions_and_rewards_rejects_invalid_state_width(falcon_model):
    bad_state = np.random.rand(3)
    with pytest.raises(ValueError, match="state width"):
        falcon_model.get_actions_and_rewards(bad_state)


def test_get_actions_and_rewards_rejects_empty_action_space(falcon_model):
    state = np.random.rand(4)
    action_space = np.empty((0, 2))
    with pytest.raises(ValueError, match="at least one row"):
        falcon_model.get_actions_and_rewards(state, action_space=action_space)
