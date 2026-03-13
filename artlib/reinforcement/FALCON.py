"""FALCON :cite:`tan2004falcon`, :cite:`tan2008integrating`."""
# Tan, A.-H. (2004).
# FALCON: a fusion architecture for learning, cognition, and navigation.
# In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
# (pp. 3297–3302). volume 4. doi:10.1109/ IJCNN.2004.1381208.

# Tan, A.-H., Lu, N., & Xiao, D. (2008).
# Integrating Temporal Difference Methods and Self-Organizing Neural Networks for
# Reinforcement Learning With Delayed Evaluative Feedback.
# IEEE Transactions on Neural Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839

import numpy as np
from typing import Optional, Literal, Tuple, Union, List
from artlib.common.BaseART import BaseART
from artlib.common.utils import complement_code, de_complement_code
from artlib.fusion.FusionART import FusionART

try:
    from artlib.optimized.backends.cpp.cppFusionUtils import (
        BuildStateActionRewardQuery,
        JoinChannelsWithFill,
    )
except ImportError:
    BuildStateActionRewardQuery = None
    JoinChannelsWithFill = None


class FALCON:
    """FALCON for Reinforcement Learning.

    This module implements the reactive FALCON as first described in:
    :cite:`tan2004falcon`.

    .. # Tan, A.-H. (2004).
    .. # FALCON: a fusion architecture for learning, cognition, and navigation.
    .. # In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
    .. # (pp. 3297–3302). volume 4. doi:10.1109/ IJCNN.2004.1381208.

    FALCON is based on a :class:`~artlib.fusion.FusionART.FusionART` backbone but only
    accepts 3 channels: State, Action, and Reward. Specific functions are implemented
    for getting optimal reward and action predictions.

    """

    def __init__(
        self,
        state_art: BaseART,
        action_art: BaseART,
        reward_art: BaseART,
        gamma_values: Union[List[float], np.ndarray] = np.array([0.33, 0.33, 0.34]),
        channel_dims: Union[List[int], np.ndarray] = list[int],
    ):
        """Initialize the FALCON model.

        Parameters
        ----------
        state_art : BaseART
            The instantiated ART module that will cluster the state-space.
        action_art : BaseART
            The instantiated ART module that will cluster the action-space.
        reward_art : BaseART
            The instantiated ART module that will cluster the reward-space.
        gamma_values : list of float or np.ndarray, optional
            The activation ratio for each channel, by default [0.33, 0.33, 0.34].
        channel_dims : list of int or np.ndarray
            The dimension of each channel.

        """
        self.fusion_art = FusionART(
            modules=[state_art, action_art, reward_art],
            gamma_values=gamma_values,
            channel_dims=channel_dims,
        )
        self._reward_skip_channel = [2]
        self._reward_centers_cache: Optional[np.ndarray] = None
        self._reward_centers_sig: Optional[int] = None
        self._action_space_cache: Optional[np.ndarray] = None
        self._action_space_prepared_cache: Optional[np.ndarray] = None
        self._action_query_template_cache: Optional[np.ndarray] = None
        self._action_space_sig: Optional[int] = None
        self._external_action_space_cache: Optional[np.ndarray] = None
        self._external_action_space_prepared_cache: Optional[np.ndarray] = None
        self._external_action_query_template_cache: Optional[np.ndarray] = None
        self._external_action_space_sig: Optional[tuple[int, tuple[int, ...]]] = None
        self._channel_dims_i64 = np.asarray(self.fusion_art.channel_dims, dtype=np.int64)
        self._state_action_present_mask = np.array([1, 1, 0], dtype=np.uint8)

    def _reward_centers_array(self) -> np.ndarray:
        sig = len(self.fusion_art.modules[2].W)
        if self._reward_centers_cache is None or self._reward_centers_sig != sig:
            self._reward_centers_cache = np.asarray(self.fusion_art.get_channel_centers(2))
            self._reward_centers_sig = sig
        return self._reward_centers_cache

    def _default_action_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sig = len(self.fusion_art.modules[1].W)
        if (
            self._action_space_cache is None
            or self._action_space_prepared_cache is None
            or self._action_query_template_cache is None
            or self._action_space_sig != sig
        ):
            self._action_space_cache = np.asarray(self.fusion_art.get_channel_centers(1))
            self._action_space_prepared_cache = self.fusion_art.modules[1].prepare_data(
                self._action_space_cache
            )
            state_dim = self.fusion_art.channel_dims[0]
            action_dim = self.fusion_art.channel_dims[1]
            reward_dim = self.fusion_art.channel_dims[2]
            self._action_query_template_cache = np.empty(
                (
                    self._action_space_prepared_cache.shape[0],
                    state_dim + action_dim + reward_dim,
                ),
                dtype=self._action_space_prepared_cache.dtype,
            )
            self._action_query_template_cache[:, state_dim : state_dim + action_dim] = (
                self._action_space_prepared_cache
            )
            self._action_query_template_cache[:, state_dim + action_dim :] = 0.5
            self._action_space_sig = sig
        return (
            self._action_space_cache,
            self._action_space_prepared_cache,
            self._action_query_template_cache,
        )

    def _build_state_action_query(
        self, state: np.ndarray, action_space_prepared: np.ndarray
    ) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64)
        action_space_arr = np.asarray(action_space_prepared, dtype=np.float64)
        if state_arr.ndim != 1:
            raise ValueError("state must be 1-D")
        if state_arr.shape[0] != self.fusion_art.channel_dims[0]:
            raise ValueError(
                f"state width {state_arr.shape[0]} does not match expected "
                f"{self.fusion_art.channel_dims[0]}"
            )
        if action_space_arr.ndim != 2:
            raise ValueError("action_space_prepared must be 2-D")
        if action_space_arr.shape[0] == 0:
            raise ValueError("action_space_prepared must have at least one row")
        if action_space_arr.shape[1] != self.fusion_art.channel_dims[1]:
            raise ValueError(
                f"action_space_prepared width {action_space_arr.shape[1]} does not "
                f"match expected {self.fusion_art.channel_dims[1]}"
            )
        if BuildStateActionRewardQuery is not None:
            return BuildStateActionRewardQuery(
                state_arr,
                action_space_arr,
                self.fusion_art.channel_dims[2],
                0.5,
            )
        state_dim = self.fusion_art.channel_dims[0]
        action_dim = self.fusion_art.channel_dims[1]
        reward_dim = self.fusion_art.channel_dims[2]
        data = np.empty(
            (action_space_arr.shape[0], state_dim + action_dim + reward_dim),
            dtype=action_space_arr.dtype,
        )
        data[:, :state_dim] = state_arr
        data[:, state_dim : state_dim + action_dim] = action_space_arr
        data[:, state_dim + action_dim :] = 0.5
        return data

    def _external_action_space_components(
        self, action_space: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action_space_arr = np.asarray(action_space, dtype=np.float64)
        if action_space_arr.ndim != 2:
            raise ValueError("action_space must be 2-D")
        if action_space_arr.shape[0] == 0:
            raise ValueError("action_space must have at least one row")
        sig = (id(action_space_arr), action_space_arr.shape)
        if (
            self._external_action_space_prepared_cache is not None
            and self._external_action_query_template_cache is not None
            and self._external_action_space_cache is action_space_arr
            and self._external_action_space_sig == sig
        ):
            return (
                self._external_action_space_cache,
                self._external_action_space_prepared_cache,
                self._external_action_query_template_cache,
            )
        prepared = self.fusion_art.modules[1].prepare_data(action_space_arr)
        state_dim = self.fusion_art.channel_dims[0]
        action_dim = self.fusion_art.channel_dims[1]
        reward_dim = self.fusion_art.channel_dims[2]
        template = np.empty(
            (prepared.shape[0], state_dim + action_dim + reward_dim),
            dtype=prepared.dtype,
        )
        template[:, state_dim : state_dim + action_dim] = prepared
        template[:, state_dim + action_dim :] = 0.5
        self._external_action_space_cache = action_space_arr
        self._external_action_space_prepared_cache = prepared
        self._external_action_query_template_cache = template
        self._external_action_space_sig = sig
        return action_space_arr, prepared, template

    def _build_default_state_action_query(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.ndim != 1:
            raise ValueError("state must be 1-D")
        if state_arr.shape[0] != self.fusion_art.channel_dims[0]:
            raise ValueError(
                f"state width {state_arr.shape[0]} does not match expected "
                f"{self.fusion_art.channel_dims[0]}"
            )
        action_space, _, template = self._default_action_space()
        data = np.array(template, copy=True)
        state_dim = self.fusion_art.channel_dims[0]
        data[:, :state_dim] = state_arr
        return action_space, data

    def _build_state_action_batch_query(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64)
        actions_arr = np.asarray(actions, dtype=np.float64)
        if states_arr.ndim != 2:
            raise ValueError("states must be 2-D")
        if actions_arr.ndim != 2:
            raise ValueError("actions must be 2-D")
        if states_arr.shape[0] != actions_arr.shape[0]:
            raise ValueError("states and actions must have the same number of rows")
        if states_arr.shape[1] != self.fusion_art.channel_dims[0]:
            raise ValueError(
                f"states width {states_arr.shape[1]} does not match expected "
                f"{self.fusion_art.channel_dims[0]}"
            )
        if actions_arr.shape[1] != self.fusion_art.channel_dims[1]:
            raise ValueError(
                f"actions width {actions_arr.shape[1]} does not match expected "
                f"{self.fusion_art.channel_dims[1]}"
            )
        if JoinChannelsWithFill is not None:
            return JoinChannelsWithFill(
                [states_arr, actions_arr],
                self._channel_dims_i64,
                self._state_action_present_mask,
                0.5,
            )
        return self.fusion_art.join_channel_data(
            [states_arr, actions_arr], skip_channels=self._reward_skip_channel
        )

    def _invalidate_center_caches(self):
        self._reward_centers_cache = None
        self._reward_centers_sig = None
        self._action_space_cache = None
        self._action_space_prepared_cache = None
        self._action_query_template_cache = None
        self._action_space_sig = None
        self._external_action_space_cache = None
        self._external_action_space_prepared_cache = None
        self._external_action_query_template_cache = None
        self._external_action_space_sig = None

    def prepare_data(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for clustering.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        tuple of np.ndarray
            Normalized state, action, and reward data.

        """
        return (
            self.fusion_art.modules[0].prepare_data(states),
            self.fusion_art.modules[1].prepare_data(actions),
            self.fusion_art.modules[2].prepare_data(rewards),
        )

    def restore_data(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restore data to its original form before preparation.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        tuple of np.ndarray
            Restored state, action, and reward data.

        """
        return (
            self.fusion_art.modules[0].restore_data(states),
            self.fusion_art.modules[1].restore_data(actions),
            self.fusion_art.modules[2].restore_data(rewards),
        )

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Fit the FALCON model to the data.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        FALCON
            The fitted FALCON model.

        """
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.fit(data)
        self._invalidate_center_caches()
        return self

    def partial_fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Partially fit the FALCON model to the data.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        FALCON
            The partially fitted FALCON model.

        """
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.partial_fit(data)
        self._invalidate_center_caches()
        return self

    def get_actions_and_rewards(
        self, state: np.ndarray, action_space: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get possible actions and their associated rewards for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.

        Returns
        -------
        tuple of np.ndarray
            The possible actions and their corresponding rewards.

        """
        if action_space is None:
            action_space, data = self._build_default_state_action_query(state)
        else:
            state_arr = np.asarray(state, dtype=np.float64)
            if state_arr.ndim != 1:
                raise ValueError("state must be 1-D")
            if state_arr.shape[0] != self.fusion_art.channel_dims[0]:
                raise ValueError(
                    f"state width {state_arr.shape[0]} does not match expected "
                    f"{self.fusion_art.channel_dims[0]}"
                )
            action_space, _, template = self._external_action_space_components(action_space)
            data = np.array(template, copy=True)
            data[:, : self.fusion_art.channel_dims[0]] = state_arr
        reward_centers_arr = self._reward_centers_array()
        viable_clusters = self.fusion_art.predict(
            data, skip_channels=self._reward_skip_channel
        )
        rewards = reward_centers_arr[viable_clusters]

        return action_space, rewards

    def get_action(
        self,
        state: np.ndarray,
        action_space: Optional[np.ndarray] = None,
        optimality: Literal["min", "max"] = "max",
    ) -> np.ndarray:
        """Get the best action for a given state based on optimality.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.
        optimality : {"min", "max"}, optional
            Whether to choose the action with the minimum or maximum reward,
            by default "max".

        Returns
        -------
        np.ndarray
            The optimal action.

        """
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        if optimality == "max":
            c_winner = np.argmax(rewards)
        else:
            c_winner = np.argmin(rewards)
        return action_space[c_winner]

    def get_probabilistic_action(
        self,
        state: np.ndarray,
        action_space: Optional[np.ndarray] = None,
        offset: float = 0.1,
        optimality: Literal["min", "max"] = "max",
    ) -> np.ndarray:
        """Get a probabilistic action for a given state based on reward distribution.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.
        offset : float, optional
            The reward offset to adjust probability distribution, by default 0.1.
        optimality : {"min", "max"}, optional
            Whether to prefer minimum or maximum rewards, by default "max".

        Returns
        -------
        np.ndarray
            The chosen action based on probability.

        """
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        action_indices = np.array(range(len(action_space)))

        reward_dist = np.array(rewards, dtype=float, copy=True)
        total = np.sum(reward_dist)
        if total <= 0:
            reward_dist = np.full_like(reward_dist, 1.0 / max(1, reward_dist.size))
        else:
            reward_dist /= total
        reward_dist = reward_dist.reshape((-1,))

        if optimality == "min":
            reward_dist = 1.0 - reward_dist

        reward_dist = np.maximum(np.minimum(reward_dist, offset), 0.0001)
        reward_dist /= np.sum(reward_dist)

        a_i = np.random.choice(action_indices, size=1, p=reward_dist)
        return action_space[a_i[0]][0]

    def get_rewards(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get the rewards for given states and actions.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.

        Returns
        -------
        np.ndarray
            The rewards corresponding to the given state-action pairs.

        """
        data = self._build_state_action_batch_query(states, actions)
        reward_centers = self._reward_centers_array()
        C = self.fusion_art.predict(data, skip_channels=self._reward_skip_channel)
        return reward_centers[C]


class TD_FALCON(FALCON):
    """TD-FALCON for Reinforcement Learning.

    This module implements TD-FALCON as first described in:
    :cite:`tan2008integrating`.

    .. # Tan, A.-H., Lu, N., & Xiao, D. (2008).
    .. # Integrating Temporal Difference Methods and Self-Organizing Neural Networks for
    .. # Reinforcement Learning With Delayed Evaluative Feedback.
    .. # IEEE Transactions on Neural Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839

    TD-FALCON is based on a :class:`FALCON` backbone but includes specific function for
    temporal-difference learning. Currently, only SARSA is implemented and only
    :class:`~artlib.elementary.FuzzyART.FuzzyART` base modules are supported.

    """

    def __init__(
        self,
        state_art: BaseART,
        action_art: BaseART,
        reward_art: BaseART,
        gamma_values: Union[List[float], np.ndarray] = np.array([0.33, 0.33, 0.34]),
        channel_dims: Union[List[int], np.ndarray] = list[int],
        td_alpha: float = 1.0,
        td_lambda: float = 1.0,
    ):
        """Initialize the TD-FALCON model.

        Parameters
        ----------
        state_art : BaseART
            The instantiated ART module that will cluster the state-space.
        action_art : BaseART
            The instantiated ART module that will cluster the action-space.
        reward_art : BaseART
            The instantiated ART module that will cluster the reward-space.
        gamma_values : list of float or np.ndarray, optional
            The activation ratio for each channel, by default [0.33, 0.33, 0.34].
        channel_dims : list of int or np.ndarray
            The dimension of each channel.
        td_alpha : float, optional
            The learning rate for the temporal difference estimator, by default 1.0.
        td_lambda : float, optional
            The future-cost factor for temporal difference learning, by default 1.0.

        """
        self.td_alpha = td_alpha
        self.td_lambda = td_lambda
        super(TD_FALCON, self).__init__(
            state_art, action_art, reward_art, gamma_values, channel_dims
        )

    def _decode_reward_values(self, rewards: np.ndarray) -> np.ndarray:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.ndim != 2:
            raise ValueError("rewards must be 2-D")
        if rewards_arr.shape[1] != self.fusion_art.channel_dims[2]:
            raise ValueError(
                f"rewards width {rewards_arr.shape[1]} does not match expected "
                f"{self.fusion_art.channel_dims[2]}"
            )
        if rewards_arr.shape[1] == 2:
            decoded = np.empty((rewards_arr.shape[0], 1), dtype=np.float64)
            decoded[:, 0] = (rewards_arr[:, 0] + (1.0 - rewards_arr[:, 1])) * 0.5
            return decoded
        return de_complement_code(rewards_arr)

    def _encode_reward_values(self, rewards: np.ndarray) -> np.ndarray:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.ndim != 2:
            raise ValueError("reward values must be 2-D")
        if rewards_arr.shape[1] == 1 and self.fusion_art.channel_dims[2] == 2:
            encoded = np.empty((rewards_arr.shape[0], 2), dtype=np.float64)
            encoded[:, 0] = rewards_arr[:, 0]
            encoded[:, 1] = 1.0 - rewards_arr[:, 0]
            return encoded
        return complement_code(rewards_arr)

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Fit the TD-FALCON model to the data.

        Raises
        ------
        NotImplementedError
            TD-FALCON can only be trained with partial fit.

        """
        raise NotImplementedError("TD-FALCON can only be trained with partial fit")

    def calculate_SARSA(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        single_sample_reward: Optional[float] = None,
    ):
        """Calculate the SARSA values for reinforcement learning.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.
        single_sample_reward : float, optional
            The reward for a single sample, if applicable, by default None.

        Returns
        -------
        tuple of np.ndarray
            The state, action, and SARSA-adjusted reward data to be used for fitting.

        """
        # calculate SARSA values
        rewards_dcc = self._decode_reward_values(rewards)
        if len(states) > 1:
            if hasattr(self.fusion_art.modules[0], "W"):
                # if FALCON has been trained get predicted rewards
                Q = self.get_rewards(states, actions)
            else:
                # otherwise set predicted rewards to 0
                Q = np.zeros_like(rewards_dcc)
            # SARSA equation
            sarsa_rewards = Q[:-1] + self.td_alpha * (
                rewards_dcc[:-1] + self.td_lambda * Q[1:] - Q[:-1]
            )
            # ensure SARSA values are between 0 and 1
            sarsa_rewards = np.clip(sarsa_rewards, 0.0, 1.0)
            # complement code rewards
            sarsa_rewards_fit = self._encode_reward_values(sarsa_rewards)
            # we cant train on the final state because no rewards are generated after it
            states_fit = states[:-1, :]
            actions_fit = actions[:-1, :]
        else:
            # if we only have a single sample, we cant learn from future samples
            if single_sample_reward is None:
                sarsa_rewards_fit = rewards
            else:
                if not 0.0 <= float(single_sample_reward) <= 1.0:
                    raise ValueError("single_sample_reward must be between 0.0 and 1.0")
                sarsa_rewards_fit = self._encode_reward_values(
                    np.array([[single_sample_reward]], dtype=np.float64)
                )
            states_fit = states
            actions_fit = actions

        return states_fit, actions_fit, sarsa_rewards_fit

    def partial_fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        single_sample_reward: Optional[float] = None,
    ):
        """Partially fit the TD-FALCON model using SARSA.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.
        single_sample_reward : float, optional
            The reward for a single sample, if applicable, by default None.

        Returns
        -------
        TD_FALCON
            The partially fitted TD-FALCON model.

        """
        states_fit, actions_fit, sarsa_rewards_fit = self.calculate_SARSA(
            states, actions, rewards, single_sample_reward
        )
        data = self.fusion_art.join_channel_data(
            [states_fit, actions_fit, sarsa_rewards_fit]
        )
        self.fusion_art = self.fusion_art.partial_fit(data)
        self._invalidate_center_caches()
        return self
