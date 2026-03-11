#!/usr/bin/env bash
set -euo pipefail

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

OUT_CSV="benchmarks/backend_runtime_baseline_quick.csv"
FUSION_CSV="benchmarks/fusionart_benchmark_quick.csv"
HIER_CSV="benchmarks/hierarchical_mapping_benchmark_quick.csv"

bash scripts/run_cpp_parity_gate.sh
"${PY_CMD[@]}" scripts/benchmark_backends.py --quick --out "$OUT_CSV"
"${PY_CMD[@]}" scripts/summarize_backend_benchmarks.py --in "$OUT_CSV" --phase total_s --format plain
"${PY_CMD[@]}" scripts/benchmark_fusionart.py --quick --out "$FUSION_CSV"
"${PY_CMD[@]}" scripts/summarize_backend_benchmarks.py --in "$FUSION_CSV" --phase predict_s --format plain
"${PY_CMD[@]}" scripts/benchmark_hierarchical_mapping.py --out "$HIER_CSV" --repeats 5 --warmup 1
"${PY_CMD[@]}" scripts/summarize_backend_benchmarks.py --in "$HIER_CSV" --phase predict_s --format plain

echo "Local V&V complete. CSVs: ${OUT_CSV}, ${FUSION_CSV}, ${HIER_CSV}"
