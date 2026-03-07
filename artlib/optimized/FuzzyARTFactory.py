"""Factory for generating optimized FuzzyART models using various backends."""

import warnings


class FuzzyARTFactory:
    """Factory for generating optimized FuzzyART models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a FuzzyART model."""
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for FuzzyART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

            return CppFuzzyART(rho=rho, alpha=alpha, beta=beta)

        if b == "python":
            from artlib.elementary.FuzzyART import FuzzyART as PyFuzzyART

            return PyFuzzyART(rho=rho, alpha=alpha, beta=beta)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

        return CppFuzzyART(rho=rho, alpha=alpha, beta=beta)
