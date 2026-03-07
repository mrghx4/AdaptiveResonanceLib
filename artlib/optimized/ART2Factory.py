"""Factory for generating optimized ART2A models using various backends."""

import warnings


class ART2Factory:
    """Factory for generating optimized ART2A models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an ART2A model."""
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for ART2A. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

            return CppART2A(rho=rho, alpha=alpha, beta=beta)

        if b == "python":
            from artlib.elementary.ART2 import ART2A as PyART2A

            return PyART2A(rho=rho, alpha=alpha, beta=beta)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

        return CppART2A(rho=rho, alpha=alpha, beta=beta)
