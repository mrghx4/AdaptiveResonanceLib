"""Factory for generating optimized BARTMAP models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class BARTMAPFactory:
    """Factory for generating optimized BARTMAP models using various backends."""

    def __new__(
        cls,
        module_a,
        module_b,
        eta: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a BARTMAP model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(module_a, BaseART) or not isinstance(module_b, BaseART):
            raise TypeError("module_a and module_b must be BaseART instances")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for BARTMAP. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            unsupported = []
            a_acc = accelerate_base_art_module(module_a)
            b_acc = accelerate_base_art_module(module_b)

            if a_acc is None:
                unsupported.append(type(module_a).__name__)
                a_acc = module_a
            if b_acc is None:
                unsupported.append(type(module_b).__name__)
                b_acc = module_b

            if unsupported:
                names = ", ".join(sorted(set(unsupported)))
                warnings.warn(
                    f"No c++ acceleration mapping for module types: {names}. "
                    "Using python module implementation(s) for those modules.",
                    RuntimeWarning,
                )

            from artlib.biclustering.BARTMAP import BARTMAP

            return BARTMAP(module_a=a_acc, module_b=b_acc, eta=eta)

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return BARTMAPFactory(
                module_a=module_a,
                module_b=module_b,
                eta=eta,
                backend="c++",
            )

        from artlib.biclustering.BARTMAP import BARTMAP

        return BARTMAP(module_a=module_a, module_b=module_b, eta=eta)
