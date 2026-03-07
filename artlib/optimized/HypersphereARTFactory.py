"""Factory for generating optimized HypersphereART models using various backends."""
import warnings


class HypersphereARTFactory:
    """Factory for generating optimized HypersphereART models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        r_hat: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a HypersphereART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning rate.
        r_hat : float
            Maximum possible category radius.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry with other factories; currently unused.

        """
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for HypersphereART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.HypersphereART import (
                HypersphereART as CppHA,
            )

            return CppHA(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)

        if b == "python":
            from artlib.elementary.HypersphereART import HypersphereART as PyHA

            return PyHA(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.HypersphereART import (
            HypersphereART as CppHA,
        )

        return CppHA(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)
