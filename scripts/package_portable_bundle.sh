#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${1:-$ROOT_DIR/dist/portable-artlib}"
ARCHIVE_PATH="${2:-$ROOT_DIR/dist/portable-artlib.zip}"

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "Portable bundle directory not found: $BUNDLE_DIR" >&2
  echo "Build it first with: bash scripts/build_portable_bundle.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"

python3 - "$BUNDLE_DIR" "$ARCHIVE_PATH" <<'PY'
import os
import sys
import zipfile

bundle_dir = os.path.abspath(sys.argv[1])
archive_path = os.path.abspath(sys.argv[2])

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(bundle_dir):
        if "__pycache__" in root.split(os.sep):
            continue
        for filename in files:
            if filename.endswith((".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib")):
                continue
            path = os.path.join(root, filename)
            arcname = os.path.relpath(path, bundle_dir)
            zf.write(path, arcname)

print(archive_path)
PY

echo "Portable ART archive created at: $ARCHIVE_PATH"
