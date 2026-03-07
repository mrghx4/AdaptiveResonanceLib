"""Factory for generating optimized ART1MAP models using various backends."""

import warnings


class ART1MAPFactory:
    """Factory for generating optimized ART1MAP models using various backends."""

    def __new__(
        cls,
        rho: float,
        L: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an ART1MAP model.

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
                "Backend 'torch' is not implemented for ART1MAP. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.ART1MAP import ART1MAP as CppART1MAP

            return CppART1MAP(rho=rho, L=L)

        if b == "python":
            from artlib.elementary.ART1 import ART1
            from artlib.supervised.SimpleARTMAP import SimpleARTMAP

            return SimpleARTMAP(module_a=ART1(rho=rho, L=L))

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.ART1MAP import ART1MAP as CppART1MAP

        return CppART1MAP(rho=rho, L=L)
