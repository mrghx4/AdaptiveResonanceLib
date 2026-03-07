"""Shared dispatch helpers for mapping Python ART modules/classes to C++ backends."""


def accelerate_base_art_module(module):
    """Return a C++-accelerated module instance when available; otherwise None."""
    from artlib.elementary.ART1 import ART1
    from artlib.elementary.ART2 import ART2A
    from artlib.elementary.BayesianART import BayesianART
    from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
    from artlib.elementary.EllipsoidART import EllipsoidART
    from artlib.elementary.FuzzyART import FuzzyART
    from artlib.elementary.GaussianART import GaussianART
    from artlib.elementary.HypersphereART import HypersphereART
    from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

    if isinstance(module, ART1):
        from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

        return CppART1(rho=module.params["rho"], L=module.params["L"])
    if isinstance(module, ART2A):
        from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

        return CppART2A(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
        )
    if isinstance(module, FuzzyART):
        from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

        return CppFuzzyART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
        )
    if isinstance(module, BinaryFuzzyART):
        from artlib.optimized.backends.cpp.BinaryFuzzyART import (
            BinaryFuzzyART as CppBinaryFuzzyART,
        )

        return CppBinaryFuzzyART(rho=module.params["rho"])
    if isinstance(module, GaussianART):
        from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGaussianART

        return CppGaussianART(
            rho=module.params["rho"],
            sigma_init=module.params["sigma_init"],
            alpha=module.params["alpha"],
        )
    if isinstance(module, HypersphereART):
        from artlib.optimized.backends.cpp.HypersphereART import (
            HypersphereART as CppHypersphereART,
        )

        return CppHypersphereART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
            r_hat=module.params["r_hat"],
        )
    if isinstance(module, EllipsoidART):
        from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEllipsoidART

        return CppEllipsoidART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
            mu=module.params["mu"],
            r_hat=module.params["r_hat"],
        )
    if isinstance(module, BayesianART):
        from artlib.optimized.backends.cpp.BayesianART import BayesianART as CppBayesianART

        return CppBayesianART(
            rho=module.params["rho"],
            cov_init=module.params["cov_init"],
        )
    if isinstance(module, QuadraticNeuronART):
        from artlib.optimized.backends.cpp.QuadraticNeuronART import (
            QuadraticNeuronART as CppQuadraticNeuronART,
        )

        return CppQuadraticNeuronART(
            rho=module.params["rho"],
            s_init=module.params["s_init"],
            lr_b=module.params["lr_b"],
            lr_w=module.params["lr_w"],
            lr_s=module.params["lr_s"],
        )

    return None


def accelerate_base_art_class(base_art_class):
    """Return a C++-accelerated class for the provided ART class when available."""
    from artlib.elementary.ART1 import ART1
    from artlib.elementary.ART2 import ART2A
    from artlib.elementary.BayesianART import BayesianART
    from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
    from artlib.elementary.EllipsoidART import EllipsoidART
    from artlib.elementary.FuzzyART import FuzzyART
    from artlib.elementary.GaussianART import GaussianART
    from artlib.elementary.HypersphereART import HypersphereART
    from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

    if base_art_class is ART1:
        from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

        return CppART1
    if base_art_class is ART2A:
        from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

        return CppART2A
    if base_art_class is FuzzyART:
        from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

        return CppFuzzyART
    if base_art_class is BinaryFuzzyART:
        from artlib.optimized.backends.cpp.BinaryFuzzyART import (
            BinaryFuzzyART as CppBinaryFuzzyART,
        )

        return CppBinaryFuzzyART
    if base_art_class is GaussianART:
        from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGaussianART

        return CppGaussianART
    if base_art_class is HypersphereART:
        from artlib.optimized.backends.cpp.HypersphereART import (
            HypersphereART as CppHypersphereART,
        )

        return CppHypersphereART
    if base_art_class is EllipsoidART:
        from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEllipsoidART

        return CppEllipsoidART
    if base_art_class is BayesianART:
        from artlib.optimized.backends.cpp.BayesianART import BayesianART as CppBayesianART

        return CppBayesianART
    if base_art_class is QuadraticNeuronART:
        from artlib.optimized.backends.cpp.QuadraticNeuronART import (
            QuadraticNeuronART as CppQuadraticNeuronART,
        )

        return CppQuadraticNeuronART

    return None
