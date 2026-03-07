"""Factory for generating optimized SimpleARTMAP models using various backends."""

import warnings

class SimpleARTMAPFactory:
    """Factory for generating optimized SimpleARTMAP models using various backends."""

    def __new__(
        cls,
        module_a,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a SimpleARTMAP model.

        Parameters
        ----------
        module_a : BaseART
            Instantiated A-side ART module.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry; currently unused.

        """
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(module_a, BaseART):
            raise TypeError("module_a must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for generic SimpleARTMAPFactory. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.elementary.ART1 import ART1
            from artlib.elementary.BayesianART import BayesianART
            from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
            from artlib.elementary.FuzzyART import FuzzyART
            from artlib.elementary.GaussianART import GaussianART
            from artlib.elementary.HypersphereART import HypersphereART

            if isinstance(module_a, FuzzyART):
                from artlib.optimized.backends.cpp.FuzzyARTMAP import FuzzyARTMAP

                return FuzzyARTMAP(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                )

            if isinstance(module_a, BinaryFuzzyART):
                from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import (
                    BinaryFuzzyARTMAP,
                )

                return BinaryFuzzyARTMAP(rho=module_a.params["rho"])

            if isinstance(module_a, ART1):
                from artlib.optimized.backends.cpp.ART1MAP import ART1MAP

                return ART1MAP(rho=module_a.params["rho"], L=module_a.params["L"])

            if isinstance(module_a, GaussianART):
                from artlib.optimized.backends.cpp.GaussianARTMAP import GaussianARTMAP

                return GaussianARTMAP(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    sigma_init=module_a.params["sigma_init"],
                )

            if isinstance(module_a, HypersphereART):
                from artlib.optimized.backends.cpp.HypersphereARTMAP import (
                    HypersphereARTMAP,
                )

                return HypersphereARTMAP(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                    r_hat=module_a.params["r_hat"],
                )

            if isinstance(module_a, BayesianART):
                from artlib.optimized.backends.cpp.BayesianARTMAP import BayesianARTMAP

                return BayesianARTMAP(
                    rho=module_a.params["rho"],
                    cov_init=module_a.params["cov_init"],
                )

            warnings.warn(
                f"No specialized c++ SimpleARTMAP backend for '{type(module_a).__name__}'. "
                "Falling back to python SimpleARTMAP.",
                RuntimeWarning,
            )

        elif b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return SimpleARTMAPFactory(module_a=module_a, backend="c++")

        from artlib.supervised.SimpleARTMAP import SimpleARTMAP

        return SimpleARTMAP(module_a)
