"""Factory for generating optimized TopoART models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class TopoARTFactory:
    """Factory for generating optimized TopoART models using various backends."""

    def __new__(
        cls,
        base_module,
        beta_lower: float,
        tau: int,
        phi: int,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a TopoART model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(base_module, BaseART):
            raise TypeError("base_module must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for TopoART. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated, reason = accelerate_base_art_module(base_module, return_reason=True)
            if accelerated is None:
                warnings.warn(
                    reason,
                    RuntimeWarning,
                )
                accelerated = base_module

            from artlib.topological.TopoART import TopoART

            return TopoART(
                base_module=accelerated,
                beta_lower=beta_lower,
                tau=tau,
                phi=phi,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return TopoARTFactory(
                base_module=base_module,
                beta_lower=beta_lower,
                tau=tau,
                phi=phi,
                backend="c++",
            )

        from artlib.topological.TopoART import TopoART

        return TopoART(base_module=base_module, beta_lower=beta_lower, tau=tau, phi=phi)
