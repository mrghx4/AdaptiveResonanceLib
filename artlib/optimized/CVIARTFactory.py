"""Factory for generating optimized CVIART models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class CVIARTFactory:
    """Factory for generating optimized CVIART models using various backends."""

    def __new__(
        cls,
        base_module,
        validity: int,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a CVIART model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(base_module, BaseART):
            raise TypeError("base_module must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for CVIART. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated = accelerate_base_art_module(base_module)
            if accelerated is None:
                warnings.warn(
                    f"No c++ acceleration mapping for base_module '{type(base_module).__name__}'. "
                    "Using python base module implementation.",
                    RuntimeWarning,
                )
                accelerated = base_module

            from artlib.cvi.CVIART import CVIART

            return CVIART(base_module=accelerated, validity=validity)

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return CVIARTFactory(
                base_module=base_module,
                validity=validity,
                backend="c++",
            )

        from artlib.cvi.CVIART import CVIART

        return CVIART(base_module=base_module, validity=validity)
