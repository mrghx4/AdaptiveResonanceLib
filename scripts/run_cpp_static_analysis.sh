#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ARTLIB_CPP_ANALYSIS_BUILD_DIR:-$ROOT_DIR/build/cpp-analysis}"
CMAKE_BIN="${CMAKE:-$ROOT_DIR/.venv/bin/cmake}"
if [[ ! -x "$CMAKE_BIN" ]]; then
  CMAKE_BIN="${CMAKE:-cmake}"
fi
RAN_TOOL=0

cd "$ROOT_DIR"

if command -v cppcheck >/dev/null 2>&1; then
  RAN_TOOL=1
  cppcheck \
    --std=c++17 \
    --enable=warning,performance,portability \
    --error-exitcode=1 \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    -I cpp/include \
    cpp/include cpp/src artlib/optimized/backends/cpp
else
  echo "cppcheck not found; skipping cppcheck." >&2
fi

if command -v clang-tidy >/dev/null 2>&1; then
  if command -v "$CMAKE_BIN" >/dev/null 2>&1; then
    RAN_TOOL=1
    CLANG_TIDY_CHECKS="${CLANG_TIDY_CHECKS:-bugprone-*,performance-*,portability-*,readability-container-size-empty,modernize-use-nullptr}"
    CLANG_TIDY_WARNINGS_AS_ERRORS="${CLANG_TIDY_WARNINGS_AS_ERRORS:-bugprone-*,performance-*,portability-*}"
    "$CMAKE_BIN" -S "$ROOT_DIR/cpp" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DARTLIB_CPP_BUILD_TESTS=ON
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

if [[ "$RAN_TOOL" -eq 0 ]]; then
  echo "No C++ static-analysis tool was available. Install cppcheck and/or clang-tidy." >&2
  exit 2
fi
