"""Factory for generating optimized SMART models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_class


class SMARTFactory:
    """Factory for generating optimized SMART models using various backends."""

    def __new__(
        cls,
        base_ART_class,
        rho_values,
        base_params,
        *,
        backend: str = "c++",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize a SMART model."""
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for SMART. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated_cls = accelerate_base_art_class(base_ART_class)
            if accelerated_cls is None:
                warnings.warn(
                    f"No c++ acceleration mapping for base_ART_class '{base_ART_class.__name__}'. "
                    "Using python base class implementation.",
                    RuntimeWarning,
                )
                accelerated_cls = base_ART_class

            from artlib.hierarchical.SMART import SMART

            return SMART(
                base_ART_class=accelerated_cls,
                rho_values=rho_values,
                base_params=base_params,
                **kwargs,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return SMARTFactory(
                base_ART_class=base_ART_class,
                rho_values=rho_values,
                base_params=base_params,
                backend="c++",
                **kwargs,
            )

        from artlib.hierarchical.SMART import SMART

        return SMART(
            base_ART_class=base_ART_class,
            rho_values=rho_values,
            base_params=base_params,
            **kwargs,
        )
