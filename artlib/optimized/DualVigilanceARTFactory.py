"""Factory for generating optimized DualVigilanceART models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class DualVigilanceARTFactory:
    """Factory for generating optimized DualVigilanceART models using various backends."""

    def __new__(
        cls,
        base_module,
        rho_lower_bound: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a DualVigilanceART model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(base_module, BaseART):
            raise TypeError("base_module must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for DualVigilanceART. "
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

            from artlib.topological.DualVigilanceART import DualVigilanceART

            return DualVigilanceART(
                base_module=accelerated,
                rho_lower_bound=rho_lower_bound,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return DualVigilanceARTFactory(
                base_module=base_module,
                rho_lower_bound=rho_lower_bound,
                backend="c++",
            )

        from artlib.topological.DualVigilanceART import DualVigilanceART

        return DualVigilanceART(base_module=base_module, rho_lower_bound=rho_lower_bound)
