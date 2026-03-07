"""Factory for generating optimized BinaryFuzzyART models using various backends."""

import warnings


class BinaryFuzzyARTFactory:
    """Factory for generating optimized BinaryFuzzyART models using various backends."""

    def __new__(
        cls,
        rho: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a BinaryFuzzyART model."""
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for BinaryFuzzyART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.BinaryFuzzyART import (
                BinaryFuzzyART as CppBinaryFuzzyART,
            )

            return CppBinaryFuzzyART(rho=rho)

        if b == "python":
            from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART as PyBinaryFuzzyART

            return PyBinaryFuzzyART(rho=rho)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.BinaryFuzzyART import (
            BinaryFuzzyART as CppBinaryFuzzyART,
        )

        return CppBinaryFuzzyART(rho=rho)
