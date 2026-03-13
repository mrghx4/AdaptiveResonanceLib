#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTEST_CMD=()
if [[ -x ".venv/bin/pytest" ]]; then
  PYTEST_CMD=(".venv/bin/pytest")
elif command -v pytest >/dev/null 2>&1; then
  PYTEST_CMD=("pytest")
elif command -v poetry >/dev/null 2>&1; then
  PYTEST_CMD=("poetry" "run" "pytest")
else
  echo "No pytest executable found (.venv/bin/pytest, pytest, or poetry run pytest)." >&2
  exit 1
fi

TRAIN_SAMPLES="${ART_FACTORY_TRAIN_SAMPLES:-2000}"
TEST_SAMPLES="${ART_FACTORY_TEST_SAMPLES:-1000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-mnist)
      TRAIN_SAMPLES=10000
      TEST_SAMPLES=10000
      shift
      ;;
    --train-samples)
      TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --test-samples)
      TEST_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: scripts/run_factory_backend_vnv.sh [--full-mnist] [--train-samples N] [--test-samples N]" >&2
      exit 1
      ;;
  esac
done

export SCIKIT_LEARN_DATA="${ROOT_DIR}/.sklearn_data"
export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
export ART_FACTORY_TRAIN_SAMPLES="${TRAIN_SAMPLES}"
export ART_FACTORY_TEST_SAMPLES="${TEST_SAMPLES}"

"${PYTEST_CMD[@]}" -q \
  unit_tests/test_BinaryFuzzyARTMAP_factories.py \
  unit_tests/test_FuzzyARTMAP_factories.py \
  unit_tests/test_GaussianARTMAP_factories.py \
  unit_tests/test_HypersphereARTMAP_factories.py

echo "Factory backend V&V complete with train=${TRAIN_SAMPLES}, test=${TEST_SAMPLES}."
