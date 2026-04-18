#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/dist/portable-artlib}"

if [[ -d "$OUT_DIR" ]]; then
  rm -rf "$OUT_DIR"
fi

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" - "$ROOT_DIR" "$OUT_DIR" <<'PY'
import os
import shutil
import sys

root_dir = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])

ignore_suffixes = {".so", ".pyd", ".dll", ".dylib", ".pyc"}
ignore_names = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
}


def ignore_filter(_dir, names):
    ignored = []
    for name in names:
        if name in ignore_names:
            ignored.append(name)
            continue
        if any(name.endswith(suffix) for suffix in ignore_suffixes):
            ignored.append(name)
    return ignored


os.makedirs(out_dir, exist_ok=True)
shutil.copytree(
    os.path.join(root_dir, "artlib"),
    os.path.join(out_dir, "artlib"),
    ignore=ignore_filter,
    dirs_exist_ok=True,
)
cpp_backend_dir = os.path.join(out_dir, "artlib", "optimized", "backends", "cpp")
if os.path.isdir(cpp_backend_dir):
    shutil.rmtree(cpp_backend_dir)
shutil.copy2(
    os.path.join(root_dir, "requirements-portable.txt"),
    os.path.join(out_dir, "requirements-portable.txt"),
)
shutil.copy2(
    os.path.join(root_dir, "docs", "KAGGLE_PORTABLE.md"),
    os.path.join(out_dir, "KAGGLE_PORTABLE.md"),
)
shutil.copy2(
    os.path.join(root_dir, "scripts", "kaggle_bootstrap_portable_artlib.py"),
    os.path.join(out_dir, "kaggle_bootstrap_portable_artlib.py"),
)
shutil.copy2(
    os.path.join(root_dir, "templates", "kaggle_portable_validation.ipynb"),
    os.path.join(out_dir, "kaggle_portable_validation.ipynb"),
)
print(out_dir)
PY

echo "Portable ART bundle created at: $OUT_DIR"
