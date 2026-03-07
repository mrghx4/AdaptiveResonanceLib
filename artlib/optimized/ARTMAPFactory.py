"""Factory for generating optimized ARTMAP models using various backends."""

import warnings

class ARTMAPFactory:
    """Factory for generating optimized ARTMAP models using various backends."""

    def __new__(
        cls,
        module_a,
        module_b,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize an ARTMAP model.

        Parameters
        ----------
        module_a : BaseART
            Instantiated A-side ART module.
        module_b : BaseART
            Instantiated B-side ART module.
        backend : str
            c++, cpp, python, or torch (torch currently falls back to c++).
        device : str
            Included for API symmetry; currently unused.

        """
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(module_a, BaseART):
            raise TypeError("module_a must be an instance of BaseART")
        if not isinstance(module_b, BaseART):
            raise TypeError("module_b must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for generic ARTMAPFactory. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.elementary.ART1 import ART1
            from artlib.elementary.ART2 import ART2A
            from artlib.elementary.BayesianART import BayesianART
            from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
            from artlib.elementary.EllipsoidART import EllipsoidART
            from artlib.elementary.FuzzyART import FuzzyART
            from artlib.elementary.GaussianART import GaussianART
            from artlib.elementary.HypersphereART import HypersphereART
            from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

            accelerated_a = module_a

            if isinstance(module_a, ART1):
                from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

                accelerated_a = CppART1(
                    rho=module_a.params["rho"],
                    L=module_a.params["L"],
                )
            elif isinstance(module_a, ART2A):
                from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

                accelerated_a = CppART2A(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                )
            elif isinstance(module_a, FuzzyART):
                from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

                accelerated_a = CppFuzzyART(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                )
            elif isinstance(module_a, BinaryFuzzyART):
                from artlib.optimized.backends.cpp.BinaryFuzzyART import (
                    BinaryFuzzyART as CppBinaryFuzzyART,
                )

                accelerated_a = CppBinaryFuzzyART(rho=module_a.params["rho"])
            elif isinstance(module_a, GaussianART):
                from artlib.optimized.backends.cpp.GaussianART import (
                    GaussianART as CppGaussianART,
                )

                accelerated_a = CppGaussianART(
                    rho=module_a.params["rho"],
                    sigma_init=module_a.params["sigma_init"],
                    alpha=module_a.params["alpha"],
                )
            elif isinstance(module_a, HypersphereART):
                from artlib.optimized.backends.cpp.HypersphereART import (
                    HypersphereART as CppHypersphereART,
                )

                accelerated_a = CppHypersphereART(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                    r_hat=module_a.params["r_hat"],
                )
            elif isinstance(module_a, EllipsoidART):
                from artlib.optimized.backends.cpp.EllipsoidART import (
                    EllipsoidART as CppEllipsoidART,
                )

                accelerated_a = CppEllipsoidART(
                    rho=module_a.params["rho"],
                    alpha=module_a.params["alpha"],
                    beta=module_a.params["beta"],
                    mu=module_a.params["mu"],
                    r_hat=module_a.params["r_hat"],
                )
            elif isinstance(module_a, BayesianART):
                from artlib.optimized.backends.cpp.BayesianART import (
                    BayesianART as CppBayesianART,
                )

                accelerated_a = CppBayesianART(
                    rho=module_a.params["rho"],
                    cov_init=module_a.params["cov_init"],
                )
            elif isinstance(module_a, QuadraticNeuronART):
                from artlib.optimized.backends.cpp.QuadraticNeuronART import (
                    QuadraticNeuronART as CppQuadraticNeuronART,
                )

                accelerated_a = CppQuadraticNeuronART(
                    rho=module_a.params["rho"],
                    s_init=module_a.params["s_init"],
                    lr_b=module_a.params["lr_b"],
                    lr_w=module_a.params["lr_w"],
                    lr_s=module_a.params["lr_s"],
                )
            else:
                warnings.warn(
                    f"No c++ acceleration mapping for module_a '{type(module_a).__name__}'. "
                    "Using python module_a implementation.",
                    RuntimeWarning,
                )

            from artlib.supervised.ARTMAP import ARTMAP

            return ARTMAP(module_a=accelerated_a, module_b=module_b)

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return ARTMAPFactory(module_a=module_a, module_b=module_b, backend="c++")

        from artlib.supervised.ARTMAP import ARTMAP

        return ARTMAP(module_a=module_a, module_b=module_b)
