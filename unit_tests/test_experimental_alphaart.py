import numpy as np
import pytest


pytest.importorskip("pyalphashape")

from artlib.experimental.AlphaART import AlphaART
from artlib.experimental.SphericalAlphaART import SphericalAlphaART


def test_alphaart_fit_predict_smoke():
    X = np.array(
        [
            [0.10, 0.10],
            [0.12, 0.11],
            [0.80, 0.82],
        ],
        dtype=float,
    )
    model = AlphaART(rho=0.2, alpha=1.0)
    model.fit(X, max_iter=1)

    pred = model.predict(X)
    assert pred.shape == (3,)
    assert pred[0] == pred[1]


def test_spherical_alphaart_validate_data_and_fit_smoke():
    X = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [-35.0, 120.0],
        ],
        dtype=float,
    )
    model = SphericalAlphaART(rho=0.2, alpha=1.0)
    model.validate_data(X)
    model.check_dimensions(X)
    model.fit(X, max_iter=1)

    pred = model.predict(X)
    assert pred.shape == (3,)
