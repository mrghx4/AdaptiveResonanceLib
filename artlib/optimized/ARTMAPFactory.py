"""Factory for generating optimized ARTMAP models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class ARTMAPFactory:
    """Factory for generating optimized ARTMAP models using various backends."""

    def __new__(
        cls,
        module_a,
        module_b,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an ARTMAP model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(module_a, BaseART):
            raise TypeError("module_a must be an instance of BaseART")
        if not isinstance(module_b, BaseART):
            raise TypeError("module_b must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for generic ARTMAPFactory. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated_a, reason_a = accelerate_base_art_module(module_a, return_reason=True)
            if accelerated_a is None:
                warnings.warn(
                    reason_a,
                    RuntimeWarning,
                )
                accelerated_a = module_a
            accelerated_b, reason_b = accelerate_base_art_module(module_b, return_reason=True)
            if accelerated_b is None:
                warnings.warn(
                    reason_b,
                    RuntimeWarning,
                )
                accelerated_b = module_b

            from artlib.supervised.ARTMAP import ARTMAP

            return ARTMAP(module_a=accelerated_a, module_b=accelerated_b)

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return ARTMAPFactory(module_a=module_a, module_b=module_b, backend="c++")

        from artlib.supervised.ARTMAP import ARTMAP

        return ARTMAP(module_a=module_a, module_b=module_b)
