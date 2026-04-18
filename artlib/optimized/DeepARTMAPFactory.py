"""Factory for generating optimized DeepARTMAP models using various backends."""

import warnings

from artlib.optimized._module_dispatch import accelerate_base_art_module


class DeepARTMAPFactory:
    """Factory for generating optimized DeepARTMAP models using various backends."""

    def __new__(
        cls,
        modules,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a DeepARTMAP model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(modules, list) or len(modules) == 0:
            raise TypeError("modules must be a non-empty list of BaseART instances")
        if not all(isinstance(m, BaseART) for m in modules):
            raise TypeError("all modules must be BaseART instances")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for DeepARTMAP. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated_modules = []
            reasons = []
            for m in modules:
                m_acc, reason = accelerate_base_art_module(m, return_reason=True)
                if m_acc is None:
                    reasons.append(reason)
                    accelerated_modules.append(m)
                else:
                    accelerated_modules.append(m_acc)

            if reasons:
                warnings.warn(
                    " ".join(sorted(set(reasons))),
                    RuntimeWarning,
                )

            from artlib.hierarchical.DeepARTMAP import DeepARTMAP

            return DeepARTMAP(modules=accelerated_modules)

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return DeepARTMAPFactory(modules=modules, backend="c++")

        from artlib.hierarchical.DeepARTMAP import DeepARTMAP

        return DeepARTMAP(modules=modules)
