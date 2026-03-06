"""Factory for generating optimized GaussianART models using various backends."""
import warnings
import numpy as np


class GaussianARTFactory:
    """Factory for generating optimized GaussianART models using various backends."""

    def __new__(
        cls,
        rho: float,
        sigma_init: np.ndarray,
        alpha: float = 1e-10,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a GaussianART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        sigma_init : np.ndarray
            Initial diagonal standard deviations (length = n_features).
        alpha : float
            Small parameter to prevent division by zero errors.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry with other factories; currently unused.

        """
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for GaussianART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGA

            return CppGA(rho=rho, sigma_init=sigma_init, alpha=alpha)

        if b == "python":
            from artlib.elementary.GaussianART import GaussianART as PyGA

            return PyGA(rho=rho, sigma_init=sigma_init, alpha=alpha)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGA

        return CppGA(rho=rho, sigma_init=sigma_init, alpha=alpha)
