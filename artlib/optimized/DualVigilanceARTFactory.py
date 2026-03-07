"""Factory for generating optimized DualVigilanceART models using various backends."""

import warnings


class DualVigilanceARTFactory:
    """Factory for generating optimized DualVigilanceART models using various backends."""

    @staticmethod
    def _accelerate_base_module(base_module):
        from artlib.elementary.ART1 import ART1
        from artlib.elementary.ART2 import ART2A
        from artlib.elementary.BayesianART import BayesianART
        from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
        from artlib.elementary.EllipsoidART import EllipsoidART
        from artlib.elementary.FuzzyART import FuzzyART
        from artlib.elementary.GaussianART import GaussianART
        from artlib.elementary.HypersphereART import HypersphereART
        from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

        if isinstance(base_module, ART1):
            from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

            return CppART1(rho=base_module.params["rho"], L=base_module.params["L"])
        if isinstance(base_module, ART2A):
            from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

            return CppART2A(
                rho=base_module.params["rho"],
                alpha=base_module.params["alpha"],
                beta=base_module.params["beta"],
            )
        if isinstance(base_module, FuzzyART):
            from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

            return CppFuzzyART(
                rho=base_module.params["rho"],
                alpha=base_module.params["alpha"],
                beta=base_module.params["beta"],
            )
        if isinstance(base_module, BinaryFuzzyART):
            from artlib.optimized.backends.cpp.BinaryFuzzyART import (
                BinaryFuzzyART as CppBinaryFuzzyART,
            )

            return CppBinaryFuzzyART(rho=base_module.params["rho"])
        if isinstance(base_module, GaussianART):
            from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGaussianART

            return CppGaussianART(
                rho=base_module.params["rho"],
                sigma_init=base_module.params["sigma_init"],
                alpha=base_module.params["alpha"],
            )
        if isinstance(base_module, HypersphereART):
            from artlib.optimized.backends.cpp.HypersphereART import (
                HypersphereART as CppHypersphereART,
            )

            return CppHypersphereART(
                rho=base_module.params["rho"],
                alpha=base_module.params["alpha"],
                beta=base_module.params["beta"],
                r_hat=base_module.params["r_hat"],
            )
        if isinstance(base_module, EllipsoidART):
            from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEllipsoidART

            return CppEllipsoidART(
                rho=base_module.params["rho"],
                alpha=base_module.params["alpha"],
                beta=base_module.params["beta"],
                mu=base_module.params["mu"],
                r_hat=base_module.params["r_hat"],
            )
        if isinstance(base_module, BayesianART):
            from artlib.optimized.backends.cpp.BayesianART import BayesianART as CppBayesianART

            return CppBayesianART(
                rho=base_module.params["rho"],
                cov_init=base_module.params["cov_init"],
            )
        if isinstance(base_module, QuadraticNeuronART):
            from artlib.optimized.backends.cpp.QuadraticNeuronART import (
                QuadraticNeuronART as CppQuadraticNeuronART,
            )

            return CppQuadraticNeuronART(
                rho=base_module.params["rho"],
                s_init=base_module.params["s_init"],
                lr_b=base_module.params["lr_b"],
                lr_w=base_module.params["lr_w"],
                lr_s=base_module.params["lr_s"],
            )
        return None

    def __new__(
        cls,
        base_module,
        rho_lower_bound: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a DualVigilanceART model."""
        del device

        from artlib.common.BaseART import BaseART

        if not isinstance(base_module, BaseART):
            raise TypeError("base_module must be an instance of BaseART")

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for DualVigilanceART. "
                "Falling back to 'c++' dispatch.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            accelerated = cls._accelerate_base_module(base_module)
            if accelerated is None:
                warnings.warn(
                    f"No c++ acceleration mapping for base_module '{type(base_module).__name__}'. "
                    "Using python base module implementation.",
                    RuntimeWarning,
                )
                accelerated = base_module
            from artlib.topological.DualVigilanceART import DualVigilanceART

            return DualVigilanceART(
                base_module=accelerated,
                rho_lower_bound=rho_lower_bound,
            )

        if b != "python":
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to c++ dispatch.",
                RuntimeWarning,
            )
            return DualVigilanceARTFactory(
                base_module=base_module,
                rho_lower_bound=rho_lower_bound,
                backend="c++",
            )

        from artlib.topological.DualVigilanceART import DualVigilanceART

        return DualVigilanceART(base_module=base_module, rho_lower_bound=rho_lower_bound)
