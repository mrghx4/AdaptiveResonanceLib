"""Factory for generating optimized BayesianARTMAP models using various backends."""
import warnings
import numpy as np


class BayesianARTMAPFactory:
    """Factory for generating optimized BayesianARTMAP models using various backends."""

    def __new__(
        cls,
        rho: float,
        cov_init: np.ndarray,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a BayesianARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        cov_init : np.ndarray
            Initial covariance matrix.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry with other factories; currently unused.

        """
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for BayesianARTMAP. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.BayesianARTMAP import (
                BayesianARTMAP as CppBAM,
            )

            return CppBAM(rho=rho, cov_init=cov_init)

        if b == "python":
            from artlib import SimpleARTMAP, BayesianART

            return SimpleARTMAP(BayesianART(rho=rho, cov_init=np.asarray(cov_init)))

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.BayesianARTMAP import (
            BayesianARTMAP as CppBAM,
        )

        return CppBAM(rho=rho, cov_init=cov_init)
