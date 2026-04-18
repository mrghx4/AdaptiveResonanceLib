#!/usr/bin/env python3
"""Validate a portable ART bundle with the same checks used in the Kaggle notebook."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        default="dist/portable-artlib",
        help="Path to the portable ART bundle directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Portable ART bundle not found: {bundle_dir}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    sys.path.insert(0, str(bundle_dir))

    import artlib  # noqa: F401
    from artlib.elementary.FuzzyART import FuzzyART
    from artlib.fusion.FusionART import FusionART
    from artlib.supervised.ARTMAP import ARTMAP

    rng = np.random.default_rng(42)

    # FuzzyART smoke
    X = rng.random((24, 3))
    fuzzy = FuzzyART(0.5, 0.01, 1.0)
    X_prep = fuzzy.prepare_data(X)
    fuzzy.fit(X_prep, max_iter=1)
    y_fuzzy = fuzzy.predict(X_prep)
    assert y_fuzzy.shape == (24,)

    # FusionART smoke
    X0 = rng.random((18, 2))
    X1 = rng.random((18, 2))
    fusion = FusionART(
        modules=[FuzzyART(0.5, 0.01, 1.0), FuzzyART(0.6, 0.01, 1.0)],
        gamma_values=np.array([0.5, 0.5]),
        channel_dims=[4, 4],
    )
    fusion_data = fusion.prepare_data([X0, X1])
    fusion.fit(fusion_data, max_iter=1)
    y_fusion = fusion.predict(fusion_data)
    y_reg = fusion.predict_regression(fusion_data)
    assert y_fusion.shape == (18,)
    assert y_reg.shape[0] == 18

    # ARTMAP smoke
    Xa = rng.random((20, 2))
    yb = rng.random((20, 2))
    artmap = ARTMAP(FuzzyART(0.5, 0.01, 1.0), FuzzyART(0.5, 0.01, 1.0))
    Xa_prep, yb_prep = artmap.prepare_data(Xa, yb)
    artmap.fit(Xa_prep, yb_prep, max_iter=1)
    pred = artmap.predict(Xa_prep)
    pred_reg = artmap.predict_regression(Xa_prep)
    assert pred.shape == (20,)
    assert pred_reg.shape[0] == 20

    print(f"Portable ART validation passed for bundle: {bundle_dir}")
    print(f"Imported artlib from: {Path(sys.modules['artlib'].__file__).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
