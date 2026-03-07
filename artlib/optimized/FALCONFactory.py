"""Factories for generating optimized FALCON models using various backends."""

import warnings
import numpy as np

from artlib.optimized._module_dispatch import accelerate_base_art_module


class FALCONFactory:
    """Factory for generating optimized FALCON models using various backends."""

    def __new__(
        cls,
        state_art,
        action_art,
        reward_art,
        gamma_values=np.array([0.33, 0.33, 0.34]),
        channel_dims=list[int],
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a FALCON model."""
        del device

        from artlib.common.BaseART import BaseART

        modules = [state_art, action_art, reward_art]
        if not all(isinstance(m, BaseART) for m in modules):
            raise TypeError("state_art, action_art, and reward_art must be BaseART instances")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for FALCON. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated = []
            unsupported = []
            for m in modules:
                m_acc = accelerate_base_art_module(m)
                if m_acc is None:
                    unsupported.append(type(m).__name__)
                    accelerated.append(m)
                else:
                    accelerated.append(m_acc)

            if unsupported:
                names = ", ".join(sorted(set(unsupported)))
                warnings.warn(
                    f"No c++ acceleration mapping for module types: {names}. "
                    "Using python module implementation(s) for those channels.",
                    RuntimeWarning,
                )

            from artlib.reinforcement.FALCON import FALCON

            return FALCON(
                state_art=accelerated[0],
                action_art=accelerated[1],
                reward_art=accelerated[2],
                gamma_values=gamma_values,
                channel_dims=channel_dims,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return FALCONFactory(
                state_art=state_art,
                action_art=action_art,
                reward_art=reward_art,
                gamma_values=gamma_values,
                channel_dims=channel_dims,
                backend="c++",
            )

        from artlib.reinforcement.FALCON import FALCON

        return FALCON(
            state_art=state_art,
            action_art=action_art,
            reward_art=reward_art,
            gamma_values=gamma_values,
            channel_dims=channel_dims,
        )


class TDFALCONFactory:
    """Factory for generating optimized TD_FALCON models using various backends."""

    def __new__(
        cls,
        state_art,
        action_art,
        reward_art,
        gamma_values=np.array([0.33, 0.33, 0.34]),
        channel_dims=list[int],
        td_alpha: float = 1.0,
        td_lambda: float = 1.0,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a TD_FALCON model."""
        del device

        from artlib.common.BaseART import BaseART

        modules = [state_art, action_art, reward_art]
        if not all(isinstance(m, BaseART) for m in modules):
            raise TypeError("state_art, action_art, and reward_art must be BaseART instances")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for TD_FALCON. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated = []
            unsupported = []
            for m in modules:
                m_acc = accelerate_base_art_module(m)
                if m_acc is None:
                    unsupported.append(type(m).__name__)
                    accelerated.append(m)
                else:
                    accelerated.append(m_acc)

            if unsupported:
                names = ", ".join(sorted(set(unsupported)))
                warnings.warn(
                    f"No c++ acceleration mapping for module types: {names}. "
                    "Using python module implementation(s) for those channels.",
                    RuntimeWarning,
                )

            from artlib.reinforcement.FALCON import TD_FALCON

            return TD_FALCON(
                state_art=accelerated[0],
                action_art=accelerated[1],
                reward_art=accelerated[2],
                gamma_values=gamma_values,
                channel_dims=channel_dims,
                td_alpha=td_alpha,
                td_lambda=td_lambda,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return TDFALCONFactory(
                state_art=state_art,
                action_art=action_art,
                reward_art=reward_art,
                gamma_values=gamma_values,
                channel_dims=channel_dims,
                td_alpha=td_alpha,
                td_lambda=td_lambda,
                backend="c++",
            )

        from artlib.reinforcement.FALCON import TD_FALCON

        return TD_FALCON(
            state_art=state_art,
            action_art=action_art,
            reward_art=reward_art,
            gamma_values=gamma_values,
            channel_dims=channel_dims,
            td_alpha=td_alpha,
            td_lambda=td_lambda,
        )
