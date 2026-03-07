"""Factory for generating optimized EllipsoidART models using various backends."""
import warnings


class EllipsoidARTFactory:
    """Factory for generating optimized EllipsoidART models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        mu: float,
        r_hat: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an EllipsoidART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning parameter.
        mu : float
            Ratio between major and minor axes.
        r_hat : float
            Radius bias parameter.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry with other factories; currently unused.

        """
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for EllipsoidART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEA

            return CppEA(rho=rho, alpha=alpha, beta=beta, mu=mu, r_hat=r_hat)

        if b == "python":
            from artlib.elementary.EllipsoidART import EllipsoidART as PyEA

            return PyEA(rho=rho, alpha=alpha, beta=beta, mu=mu, r_hat=r_hat)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEA

        return CppEA(rho=rho, alpha=alpha, beta=beta, mu=mu, r_hat=r_hat)
