#!/usr/bin/env bash
set -euo pipefail

# Runs Python-vs-C++ parity tests for currently ported backends.
# Extend this list as additional modules are ported.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_CMD=()
if [[ -x ".venv/bin/python" ]]; then
  PY_CMD=(".venv/bin/python")
elif command -v poetry >/dev/null 2>&1; then
  PY_CMD=("poetry" "run" "python")
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD=("python3")
elif command -v python >/dev/null 2>&1; then
  PY_CMD=("python")
else
  echo "No Python interpreter found (.venv/bin/python, poetry, python3, or python)." >&2
  exit 1
fi

export SCIKIT_LEARN_DATA="${ROOT_DIR}/.sklearn_data"

"${PY_CMD[@]}" -m pytest -q \
  unit_tests/test_ART1.py \
  unit_tests/test_ART1MAP.py \
  unit_tests/test_BinaryFuzzyARTMAP.py \
  unit_tests/test_GaussianART.py \
  unit_tests/test_GaussianARTMAP.py \
  unit_tests/test_HypersphereARTMAP.py \
  unit_tests/test_FuzzyARTMAP.py \
  unit_tests/test_cpp_ART1.py \
  unit_tests/test_cpp_ART1MAP.py \
  unit_tests/test_cpp_BinaryFuzzyART.py \
  unit_tests/test_cpp_BinaryFuzzyARTMAP.py \
  unit_tests/test_cpp_GaussianART.py \
  unit_tests/test_cpp_GaussianARTMAP.py \
  unit_tests/test_cpp_HypersphereARTMAP.py \
  unit_tests/test_GaussianARTFactory.py \
  unit_tests/test_fracsort.py \
  unit_tests/test_FuzzyART.py \
  unit_tests/test_cpp_FuzzyART.py \
  unit_tests/test_cpp_FuzzyARTMAP.py
