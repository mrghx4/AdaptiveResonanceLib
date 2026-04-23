#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ARTLIB_CPP_SANITIZER_BUILD_DIR:-$ROOT_DIR/build/cpp-sanitizers}"
CMAKE_BIN="${CMAKE:-$ROOT_DIR/.venv/bin/cmake}"
CTEST_BIN="${CTEST:-$ROOT_DIR/.venv/bin/ctest}"
if [[ ! -x "$CMAKE_BIN" ]]; then
  CMAKE_BIN="${CMAKE:-cmake}"
fi
if [[ ! -x "$CTEST_BIN" ]]; then
  CTEST_BIN="${CTEST:-ctest}"
fi

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "cmake is required for sanitizer C++ tests. Install cmake or set CMAKE=/path/to/cmake." >&2
  exit 2
fi
if ! command -v "$CTEST_BIN" >/dev/null 2>&1; then
  echo "ctest is required for sanitizer C++ tests. Install ctest or set CTEST=/path/to/ctest." >&2
  exit 2
fi

"$CMAKE_BIN" -S "$ROOT_DIR/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DARTLIB_CPP_BUILD_TESTS=ON \
  -DARTLIB_CPP_BUILD_FUZZ=ON \
  -DARTLIB_CPP_ENABLE_ASAN=ON \
  -DARTLIB_CPP_ENABLE_UBSAN=ON

BUILD_ARGS=()
if [[ -n "${ARTLIB_CPP_BUILD_JOBS:-}" ]]; then
  BUILD_ARGS=(--parallel "$ARTLIB_CPP_BUILD_JOBS")
fi
if [[ ${#BUILD_ARGS[@]} -gt 0 ]]; then
  "$CMAKE_BIN" --build "$BUILD_DIR" "${BUILD_ARGS[@]}"
else
  "$CMAKE_BIN" --build "$BUILD_DIR"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  DEFAULT_ASAN_OPTIONS="detect_leaks=0:halt_on_error=1"
else
  DEFAULT_ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
fi

ASAN_OPTIONS="${ASAN_OPTIONS:-$DEFAULT_ASAN_OPTIONS}" \
UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}" \
  "$CTEST_BIN" --test-dir "$BUILD_DIR" --output-on-failure
