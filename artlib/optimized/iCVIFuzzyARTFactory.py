"""Factory for generating iCVIFuzzyART models across backends."""

import warnings


class iCVIFuzzyARTFactory:
    """Factory for generating iCVIFuzzyART models across backends.

    Notes
    -----
    iCVIFuzzyART has custom iCVI-driven fitting logic and currently only has a
    behavior-complete Python implementation.
    """

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        validity: int,
        offline: bool = True,
        *,
        backend: str = "python",
        device: str = "cpu",
    ):
        """Initialize an iCVIFuzzyART model."""
        del device

        b = backend.lower()

        if b in ("c++", "cpp", "torch"):
            warnings.warn(
                f"Backend '{backend}' is not implemented for iCVIFuzzyART. "
                "Falling back to 'python' backend.",
                RuntimeWarning,
            )
            b = "python"

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'python'.",
                RuntimeWarning,
            )

        from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART

        return iCVIFuzzyART(
            rho=rho,
            alpha=alpha,
            beta=beta,
            validity=validity,
            offline=offline,
        )
