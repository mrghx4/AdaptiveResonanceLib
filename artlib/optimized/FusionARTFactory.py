"""Factory for generating optimized FusionART models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class FusionARTFactory:
    """Factory for generating optimized FusionART models using various backends."""

    def __new__(
        cls,
        modules,
        gamma_values,
        channel_dims,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a FusionART model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(modules, list) or len(modules) == 0:
            raise TypeError("modules must be a non-empty list of BaseART instances")
        if not all(isinstance(m, BaseART) for m in modules):
            raise TypeError("all modules must be BaseART instances")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for FusionART. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated_modules = []
            unsupported = []
            for m in modules:
                m_acc = accelerate_base_art_module(m)
                if m_acc is None:
                    unsupported.append(type(m).__name__)
                    accelerated_modules.append(m)
                else:
                    accelerated_modules.append(m_acc)

            if unsupported:
                names = ", ".join(sorted(set(unsupported)))
                warnings.warn(
                    f"No c++ acceleration mapping for module types: {names}. "
                    "Using python module implementation(s) for those channels.",
                    RuntimeWarning,
                )

            from artlib.fusion.FusionART import FusionART

            return FusionART(
                modules=accelerated_modules,
                gamma_values=gamma_values,
                channel_dims=channel_dims,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return FusionARTFactory(
                modules=modules,
                gamma_values=gamma_values,
                channel_dims=channel_dims,
                backend="c++",
            )

        from artlib.fusion.FusionART import FusionART

        return FusionART(
            modules=modules,
            gamma_values=gamma_values,
            channel_dims=channel_dims,
        )
