# Kaggle Portable ARTLib

This repo now supports a Kaggle-friendly `artlib` bundle that does not require:

- local C++ extension compilation
- notebook-time import shims for `fracsort`
- `numba` to be installed

## What "portable" means

The portable bundle is intended for Kaggle notebooks and Kaggle Datasets where you want:

- predictable imports on fresh Linux runtimes
- pure-Python fallback behavior when optional compiled helpers are unavailable
- a simple attach-and-import workflow

It is not intended to preserve local C++ acceleration automatically.

## Build the portable bundle

From the repo root:

```bash
bash scripts/build_portable_bundle.sh
```

Default output:

```text
dist/portable-artlib/
```

This output is a generated artifact and should stay untracked in Git.

The bundle contains:

- `artlib/`
- `requirements-portable.txt`
- `KAGGLE_PORTABLE.md`
- `kaggle_bootstrap_portable_artlib.py`
- `kaggle_portable_validation.ipynb`

Compiled extension artifacts are excluded.

## Package the bundle for upload

Create a Kaggle-upload-friendly archive:

```bash
bash scripts/package_portable_bundle.sh
```

Default output:

```text
dist/portable-artlib.zip
```

This zip is also a generated artifact and should not be committed.

## Smoke-test the bundle locally

This simulates a portable runtime where:

- `numba` is unavailable
- the C++ `fracsort` extension is unavailable

Run:

```bash
.venv/bin/python scripts/test_portable_imports.py
```

Or test an existing bundle:

```bash
.venv/bin/python scripts/test_portable_imports.py --bundle-dir dist/portable-artlib
```

Recommended full local portable validation:

```bash
bash scripts/build_portable_bundle.sh
bash scripts/package_portable_bundle.sh
.venv/bin/python scripts/validate_portable_bundle.py --bundle-dir dist/portable-artlib
```

Canonical routine repo validation remains:

```bash
.venv/bin/python -m pytest -q
```

## Kaggle usage

### Option 1: Upload the portable bundle as a Kaggle Dataset

After uploading `dist/portable-artlib`, attach that dataset to the notebook.

Then in the notebook:

```python
import sys

PORTABLE_ARTLIB = "/kaggle/input/portable-artlib"
sys.path.insert(0, PORTABLE_ARTLIB)
```

If you also include the bootstrap helper in the uploaded dataset, you can use:

```python
from kaggle_bootstrap_portable_artlib import bootstrap_portable_artlib
bootstrap_portable_artlib("/kaggle/input/portable-artlib")
```

There is also a ready-to-run validation notebook in the bundle:

```text
/kaggle/input/portable-artlib/kaggle_portable_validation.ipynb
```

Use that notebook first to validate the bundle on Kaggle before moving into a
competition notebook.

Now import normally:

```python
from artlib.elementary.FuzzyART import FuzzyART
from artlib.fusion.FusionART import FusionART
```

### Option 2: Copy the bundle into the working directory

If you unpack it into `/kaggle/working/portable-artlib`:

```python
import sys
sys.path.insert(0, "/kaggle/working/portable-artlib")
```

## Recommended Kaggle dependency baseline

Portable baseline:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

Optional:

- `torch`

Install from the bundle file if needed:

```bash
pip install -r /kaggle/input/portable-artlib/requirements-portable.txt
```

## Important note about local acceleration

The portable bundle is designed to import safely without compiled C++ helpers.
If the runtime also has compatible compiled extensions available, those optional fast
paths may still be used where the code supports them.

For Kaggle contests, the safest assumption is:

- correctness first
- pure-Python / NumPy fallback by default
- C++ acceleration only when explicitly prebuilt and known compatible
