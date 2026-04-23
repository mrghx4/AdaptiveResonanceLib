#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ARTLIB_CPP_ANALYSIS_BUILD_DIR:-$ROOT_DIR/build/cpp-analysis}"
CMAKE_BIN="${CMAKE:-$ROOT_DIR/.venv/bin/cmake}"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CXX_BIN="${CXX:-c++}"
if [[ ! -x "$CMAKE_BIN" ]]; then
  CMAKE_BIN="${CMAKE:-cmake}"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi
RAN_ANALYZER=0

cd "$ROOT_DIR"

if command -v cppcheck >/dev/null 2>&1; then
  RAN_ANALYZER=1
  cppcheck \
    --std=c++17 \
    --enable=warning,performance,portability \
    --error-exitcode=1 \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    --suppress=unusedFunction:cpp/tests/* \
    -I cpp/include \
    cpp/include cpp/src
else
  echo "cppcheck not found; skipping cppcheck." >&2
fi

if command -v clang-tidy >/dev/null 2>&1; then
  if command -v "$CMAKE_BIN" >/dev/null 2>&1; then
    RAN_ANALYZER=1
    CLANG_TIDY_CHECKS="${CLANG_TIDY_CHECKS:-bugprone-use-after-move,bugprone-suspicious-missing-comma,bugprone-sizeof-expression,bugprone-incorrect-roundings,bugprone-integer-division,bugprone-misplaced-widening-cast,performance-noexcept-move-constructor,performance-unnecessary-copy-initialization,portability-simd-intrinsics,modernize-use-nullptr}"
    CLANG_TIDY_WARNINGS_AS_ERRORS="${CLANG_TIDY_WARNINGS_AS_ERRORS:-bugprone-use-after-move,bugprone-suspicious-missing-comma,bugprone-sizeof-expression,bugprone-incorrect-roundings,bugprone-integer-division,bugprone-misplaced-widening-cast}"
    "$CMAKE_BIN" -S "$ROOT_DIR/cpp" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DARTLIB_CPP_BUILD_TESTS=ON \
      -DARTLIB_CPP_BUILD_FUZZ=ON
    find cpp/src cpp/tests -name '*.cpp' -print0 | \
      xargs -0 clang-tidy -p "$BUILD_DIR" --quiet \
        --checks="$CLANG_TIDY_CHECKS" \
        --warnings-as-errors="$CLANG_TIDY_WARNINGS_AS_ERRORS"
  else
    echo "clang-tidy found but cmake is unavailable; skipping clang-tidy compile database analysis." >&2
  fi
else
  echo "clang-tidy not found; skipping clang-tidy." >&2
fi

if command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -m pybind11 --includes >/dev/null 2>&1; then
  if command -v "$CXX_BIN" >/dev/null 2>&1; then
    read -r -a PYBIND_INCLUDES <<< "$("$PYTHON_BIN" -m pybind11 --includes)"
    find artlib/optimized/backends/cpp -maxdepth 1 -name '*.cpp' -print0 | \
      xargs -0 -n 1 "$CXX_BIN" -std=c++17 -fsyntax-only -I "$ROOT_DIR/cpp/include" "${PYBIND_INCLUDES[@]}"
  else
    echo "C++ compiler not found; skipping pybind shim syntax checks." >&2
  fi
else
  echo "pybind11 includes unavailable; skipping pybind shim syntax checks." >&2
fi

if [[ "$RAN_ANALYZER" -eq 0 ]]; then
  echo "No C++ static-analysis tool was available. Install cppcheck and/or clang-tidy." >&2
  exit 2
fi
