#!/usr/bin/env python3
"""Smoke-test a portable ART bundle without compiled fracsort or numba."""

from __future__ import annotations

import argparse
import builtins
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_MPL_CLEANUP_DIR: Path | None = None
if "MPLCONFIGDIR" not in os.environ:
    _MPL_CLEANUP_DIR = Path(tempfile.mkdtemp(prefix="portable-artlib-mpl-"))
    os.environ["MPLCONFIGDIR"] = str(_MPL_CLEANUP_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional existing portable bundle directory. If omitted, a temporary bundle is created from the repo root.",
    )
    return parser.parse_args()


def build_temp_bundle(repo_root: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="portable-artlib-"))
    bundle_dir = temp_dir / "portable-artlib"
    shutil.copytree(
        repo_root / "artlib",
        bundle_dir / "artlib",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.so",
            "*.pyd",
            "*.dll",
            "*.dylib",
            "*.pyc",
        ),
    )
    shutil.copy2(repo_root / "requirements-portable.txt", bundle_dir / "requirements-portable.txt")
    return bundle_dir


def clear_artlib_modules() -> None:
    for name in list(sys.modules):
        if name == "artlib" or name.startswith("artlib."):
            sys.modules.pop(name, None)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cleanup_dir: Path | None = None

    if args.bundle_dir is None:
        bundle_dir = build_temp_bundle(repo_root)
        cleanup_dir = bundle_dir.parent
    else:
        bundle_dir = Path(args.bundle_dir).resolve()

    original_import = builtins.__import__
    sys.path.insert(0, str(bundle_dir))

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba":
            raise ImportError("portable smoke test: numba unavailable")
        if name == "artlib.optimized.backends.cpp.fracsort":
            raise ImportError("portable smoke test: fracsort unavailable")
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = fake_import
        clear_artlib_modules()

        from artlib.common.utils import fracsort, fracargmax
        from artlib.elementary.FuzzyART import FuzzyART
        from artlib.elementary.ART1 import ART1
        from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
        from artlib.fusion.FusionART import FusionART

        num = np.array([1, 2, 0, 2], dtype=np.uint32)
        den = np.array([2, 4, 1, 3], dtype=np.uint32)
        assert np.array_equal(fracsort(num, den), np.array([3, 0, 1, 2], dtype=np.intp))
        assert int(fracargmax(num, den)) == 3

        X = np.random.default_rng(0).random((12, 2))
        fuzzy = FuzzyART(0.5, 0.01, 1.0)
        X_prep = fuzzy.prepare_data(X)
        fuzzy.fit(X_prep, max_iter=1)
        assert fuzzy.predict(X_prep).shape == (12,)

        X_bin = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int16)
        art1 = ART1(0.5, 2.0)
        X_art1 = art1.prepare_data(X_bin)
        art1.fit(X_art1, max_iter=1)
        assert art1.predict(X_art1).shape == (4,)

        binary = BinaryFuzzyART(0.5)
        X_binary = binary.prepare_data(X_bin)
        binary.fit(X_binary, max_iter=1)
        assert binary.predict(X_binary).shape == (4,)

        modules = [FuzzyART(0.5, 0.01, 1.0), FuzzyART(0.6, 0.01, 1.0)]
        fusion = FusionART(modules, gamma_values=np.array([0.5, 0.5]), channel_dims=[4, 4])
        X0 = np.random.default_rng(1).random((10, 2))
        X1 = np.random.default_rng(2).random((10, 2))
        fusion_data = fusion.prepare_data([X0, X1])
        fusion.fit(fusion_data, max_iter=1)
        assert fusion.predict(fusion_data).shape == (10,)

        print(f"Portable ART smoke test passed for bundle: {bundle_dir}")
        return 0
    finally:
        builtins.__import__ = original_import
        if str(bundle_dir) in sys.path:
            sys.path.remove(str(bundle_dir))
        clear_artlib_modules()
        if cleanup_dir is not None and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir)
        if _MPL_CLEANUP_DIR is not None and _MPL_CLEANUP_DIR.exists():
            shutil.rmtree(_MPL_CLEANUP_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
