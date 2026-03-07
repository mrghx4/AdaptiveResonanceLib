"""Factory for generating optimized ART1 models using various backends."""

import warnings


class ART1Factory:
    """Factory for generating optimized ART1 models using various backends."""

    def __new__(
        cls,
        rho: float,
        L: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an ART1 model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        L : float
            Uncommitted node bias parameter.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry with other factories; currently unused.

        """
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for ART1. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

            return CppART1(rho=rho, L=L)

        if b == "python":
            from artlib.elementary.ART1 import ART1 as PyART1

            return PyART1(rho=rho, L=L)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

        return CppART1(rho=rho, L=L)
