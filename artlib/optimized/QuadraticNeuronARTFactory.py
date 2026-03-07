"""Factory for generating optimized QuadraticNeuronART models using various backends."""

import warnings


class QuadraticNeuronARTFactory:
    """Factory for generating optimized QuadraticNeuronART models using various backends."""

    def __new__(
        cls,
        rho: float,
        s_init: float,
        lr_b: float,
        lr_w: float,
        lr_s: float,
        *,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize a QuadraticNeuronART model."""
        del device

        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not implemented for QuadraticNeuronART. "
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.QuadraticNeuronART import (
                QuadraticNeuronART as CppQN,
            )

            return CppQN(rho=rho, s_init=s_init, lr_b=lr_b, lr_w=lr_w, lr_s=lr_s)

        if b == "python":
            from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART as PyQN

            return PyQN(rho=rho, s_init=s_init, lr_b=lr_b, lr_w=lr_w, lr_s=lr_s)

        warnings.warn(
            f"Unknown backend '{backend}', defaulting to 'c++'.",
            RuntimeWarning,
        )
        from artlib.optimized.backends.cpp.QuadraticNeuronART import (
            QuadraticNeuronART as CppQN,
        )

        return CppQN(rho=rho, s_init=s_init, lr_b=lr_b, lr_w=lr_w, lr_s=lr_s)
