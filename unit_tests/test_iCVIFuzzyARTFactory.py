import pytest

from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART
from artlib.optimized.iCVIFuzzyARTFactory import iCVIFuzzyARTFactory


def test_factory_python_backend_returns_python_model():
    model = iCVIFuzzyARTFactory(
        rho=0.5,
        alpha=0.01,
        beta=1.0,
        validity=iCVIFuzzyART.CALINSKIHARABASZ,
        backend="python",
    )
    assert isinstance(model, iCVIFuzzyART)


def test_factory_cpp_falls_back_to_python_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = iCVIFuzzyARTFactory(
            rho=0.5,
            alpha=0.01,
            beta=1.0,
            validity=iCVIFuzzyART.CALINSKIHARABASZ,
            backend="c++",
        )
    assert isinstance(model, iCVIFuzzyART)


def test_factory_torch_falls_back_to_python_with_warning():
    with pytest.warns(RuntimeWarning, match="not implemented"):
        model = iCVIFuzzyARTFactory(
            rho=0.5,
            alpha=0.01,
            beta=1.0,
            validity=iCVIFuzzyART.CALINSKIHARABASZ,
            backend="torch",
        )
    assert isinstance(model, iCVIFuzzyART)


def test_factory_unknown_backend_warns_and_defaults_to_python():
    with pytest.warns(RuntimeWarning, match="Unknown backend"):
        model = iCVIFuzzyARTFactory(
            rho=0.5,
            alpha=0.01,
            beta=1.0,
            validity=iCVIFuzzyART.CALINSKIHARABASZ,
            backend="invalid",
        )
    assert isinstance(model, iCVIFuzzyART)
