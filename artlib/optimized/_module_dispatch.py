"""Shared dispatch helpers for mapping Python ART modules/classes to C++ backends."""


def _module_runtime_state_block_reason(module):
    """Return a human-readable reason why a module should not be re-wrapped.

    Factory dispatch is intended for fresh module construction. Once a Python ART
    instance has preparation or fitted runtime state attached, rebuilding it as a
    new C++ instance would silently drop that state.
    """

    if getattr(module, "is_fitted_", False):
        return (
            f"Module type '{type(module).__name__}' already has fitted state; "
            "keeping the Python implementation."
        )
    if getattr(module, "d_min_", None) is not None or getattr(module, "d_max_", None) is not None:
        return (
            f"Module type '{type(module).__name__}' already has prepared data bounds; "
            "keeping the Python implementation."
        )
    if hasattr(module, "dim_") or hasattr(module, "dim_original"):
        return (
            f"Module type '{type(module).__name__}' already has prepared dimension state; "
            "keeping the Python implementation."
        )
    if getattr(module, "sample_counter_", 0) != 0:
        return (
            f"Module type '{type(module).__name__}' already has runtime sample state; "
            "keeping the Python implementation."
        )
    if getattr(module, "weight_sample_counter_", []):
        return (
            f"Module type '{type(module).__name__}' already has runtime cluster counters; "
            "keeping the Python implementation."
        )
    if hasattr(module, "W") and len(getattr(module, "W", [])) > 0:
        return (
            f"Module type '{type(module).__name__}' already has learned weights; "
            "keeping the Python implementation."
        )
    return None


def accelerate_base_art_module(module, return_reason: bool = False):
    """Return a C++-accelerated module instance when available.

    Parameters
    ----------
    module
        Python ART module instance to accelerate.
    return_reason : bool, default=False
        When True, return a tuple ``(accelerated_module_or_none, reason_or_none)``.
    """
    from artlib.elementary.ART1 import ART1
    from artlib.elementary.ART2 import ART2A
    from artlib.elementary.BayesianART import BayesianART
    from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
    from artlib.elementary.EllipsoidART import EllipsoidART
    from artlib.elementary.FuzzyART import FuzzyART
    from artlib.elementary.GaussianART import GaussianART
    from artlib.elementary.HypersphereART import HypersphereART
    from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

    reason = _module_runtime_state_block_reason(module)
    if reason is not None:
        return (None, reason) if return_reason else None

    if isinstance(module, ART1):
        from artlib.optimized.backends.cpp.ART1 import ART1 as CppART1

        result = CppART1(rho=module.params["rho"], L=module.params["L"])
        return (result, None) if return_reason else result
    if isinstance(module, ART2A):
        from artlib.optimized.backends.cpp.ART2 import ART2A as CppART2A

        result = CppART2A(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, FuzzyART):
        from artlib.optimized.backends.cpp.FuzzyART import FuzzyART as CppFuzzyART

        result = CppFuzzyART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, BinaryFuzzyART):
        from artlib.optimized.backends.cpp.BinaryFuzzyART import (
            BinaryFuzzyART as CppBinaryFuzzyART,
        )

        result = CppBinaryFuzzyART(rho=module.params["rho"])
        return (result, None) if return_reason else result
    if isinstance(module, GaussianART):
        from artlib.optimized.backends.cpp.GaussianART import GaussianART as CppGaussianART

        result = CppGaussianART(
            rho=module.params["rho"],
            sigma_init=module.params["sigma_init"],
            alpha=module.params["alpha"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, HypersphereART):
        from artlib.optimized.backends.cpp.HypersphereART import (
            HypersphereART as CppHypersphereART,
        )

        result = CppHypersphereART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
            r_hat=module.params["r_hat"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, EllipsoidART):
        from artlib.optimized.backends.cpp.EllipsoidART import EllipsoidART as CppEllipsoidART

        result = CppEllipsoidART(
            rho=module.params["rho"],
            alpha=module.params["alpha"],
            beta=module.params["beta"],
            mu=module.params["mu"],
            r_hat=module.params["r_hat"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, BayesianART):
        from artlib.optimized.backends.cpp.BayesianART import BayesianART as CppBayesianART

        result = CppBayesianART(
            rho=module.params["rho"],
            cov_init=module.params["cov_init"],
        )
        return (result, None) if return_reason else result
    if isinstance(module, QuadraticNeuronART):
        from artlib.optimized.backends.cpp.QuadraticNeuronART import (
            QuadraticNeuronART as CppQuadraticNeuronART,
        )

        result = CppQuadraticNeuronART(
            rho=module.params["rho"],
            s_init=module.params["s_init"],
            lr_b=module.params["lr_b"],
            lr_w=module.params["lr_w"],
            lr_s=module.params["lr_s"],
        )
        return (result, None) if return_reason else result

    reason = (
        f"No c++ acceleration mapping for module type '{type(module).__name__}'; "
        "keeping the Python implementation."
    )
    return (None, reason) if return_reason else None


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
