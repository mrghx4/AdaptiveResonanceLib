# Release Notes: 2026-04

This note captures the recent backend-alignment, portability, and workflow
cleanup work across the ART repo.

## Backend Alignment

- normalized factory/dispatch behavior around fresh-module acceleration
- made backend conversion behavior explicit: prepared/fitted Python ART modules
  remain Python unless explicit state transfer is implemented
- fixed `FALCON` / `TD_FALCON` factory argument handling for `channel_dims`
- expanded tests around backend dispatch safety and factory behavior

## C++ / Python Interop

- kept the existing C++ backend integration model:
  - dedicated C++ backends for core supported families
  - accelerated orchestration for higher-level families
- tightened the Python/C++ boundary so unsupported live-model conversion is not
  silently attempted

## Portability Hardening

- made `fracsort` import optional with pure-Python fallback behavior
- made shared `numba` usage optional via fallback decorator support
- removed remaining unconditional `numba` imports in elementary modules that
  matter for portable runtime use

## Kaggle Workflow

- added a portable bundle build/package workflow
- added a portable import smoke test and local bundle validator
- added a Kaggle validation notebook template and bootstrap helper

Primary files:

- `docs/KAGGLE_PORTABLE.md`
- `docs/BACKEND_STATUS.md`

## Validation Baseline

- full suite baseline: `488 passed, 1 skipped, 23 warnings`
- warning profile remains limited to the known `ART2` warnings

## Suggested Local Validation

Routine repo validation:

```bash
.venv/bin/python -m pytest -q
```

Portable bundle validation:

```bash
bash scripts/build_portable_bundle.sh
bash scripts/package_portable_bundle.sh
.venv/bin/python scripts/validate_portable_bundle.py --bundle-dir dist/portable-artlib
```
