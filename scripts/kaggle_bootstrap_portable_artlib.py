"""Kaggle bootstrap helper for the portable ART bundle.

Usage inside a Kaggle notebook:

```python
%run /kaggle/input/portable-artlib/kaggle_bootstrap_portable_artlib.py
```

Or copy the function below into a notebook cell.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def bootstrap_portable_artlib(
    bundle_dir: str = "/kaggle/input/portable-artlib",
    install_requirements: bool = False,
) -> str:
    bundle_path = Path(bundle_dir).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Portable ART bundle not found: {bundle_path}")

    sys.path.insert(0, str(bundle_path))

    if install_requirements:
        req_path = bundle_path / "requirements-portable.txt"
        if req_path.exists():
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                check=True,
            )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    artlib = importlib.import_module("artlib")
    importlib.import_module("artlib.common.utils")
    importlib.import_module("artlib.elementary.FuzzyART")
    importlib.import_module("artlib.fusion.FusionART")

    print(f"Portable ART ready from: {bundle_path}")
    print(f"artlib module: {artlib.__file__}")
    return str(bundle_path)


if __name__ == "__main__":
    bootstrap_portable_artlib()
