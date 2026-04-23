# C++ Backend Hardening Matrix

This document tracks whether the C++ backend has comparable Python parity,
native C++ coverage, and hardening gates. It is intended as the PR-readiness
checklist for backend changes.

## Local Gates

Run these before submitting a C++ backend PR:

```bash
SCIKIT_LEARN_DATA="$PWD/.sklearn_data" .venv/bin/python -m pytest -q
bash scripts/run_cpp_parity_gate.sh
bash scripts/run_cpp_native_tests.sh
bash scripts/run_cpp_fuzz_smoke.sh
bash scripts/run_cpp_sanitizers.sh
bash scripts/run_cpp_static_analysis.sh
```

Notes:

- `run_cpp_parity_gate.sh` automatically includes every `unit_tests/test_cpp_*.py` file.
- Full-suite factory comparison tests use OpenML MNIST and require either network access or a warm `.sklearn_data` cache.
- Native, sanitizer, fuzz-smoke, and static-analysis gates require CMake/native tooling.
- Sanitizer and static-analysis tooling is dev/CI-only and does not add runtime dependencies.

## Backend Coverage Matrix

| Backend/component | Python reference coverage | Python/C++ parity or helper coverage | Native C++ coverage | Negative/security coverage | Hardening gates |
| --- | --- | --- | --- | --- | --- |
| `ART1` | `test_ART1.py` | `test_cpp_ART1.py` | `test_art1_core_smoke.cpp` | Dimension/input validation | parity, native, sanitizer, static-analysis |
| `ART2A` | `test_ART2.py` | `test_cpp_ART2.py` | `test_art2_core_smoke.cpp` | Dimension/input validation; known ART2 warning | parity, native, sanitizer, static-analysis |
| `BayesianART` | `test_BayesianART.py` | `test_cpp_BayesianART.py` | `test_bayesian_art_core_smoke.cpp` | Covariance/weight/dimension validation | parity, native, sanitizer, static-analysis |
| `BinaryFuzzyART` | `test_BinaryFuzzyARTFactory.py` | `test_cpp_BinaryFuzzyART.py` | `test_binary_fuzzy_art_core_smoke.cpp` | Binary/dimension validation | parity, native, sanitizer, static-analysis |
| `EllipsoidART` | `test_EllipsoidART.py` | `test_cpp_EllipsoidART.py` | `test_ellipsoid_art_core_smoke.cpp` | Param/weight/dimension validation | parity, native, sanitizer, static-analysis |
| `FuzzyART` | `test_FuzzyART.py` | `test_cpp_FuzzyART.py` | `test_fuzzy_art_core_smoke.cpp` | Rank/weight/dimension validation | parity, native, sanitizer, static-analysis |
| `GaussianART` | `test_GaussianART.py` | `test_cpp_GaussianART.py` | `test_gaussian_art_core_smoke.cpp` | Sigma/weight/dimension validation | parity, native, sanitizer, static-analysis |
| `HypersphereART` | `test_HypersphereART.py` | `test_cpp_HypersphereART.py` | `test_hypersphere_art_core_smoke.cpp` | Param/weight/dimension validation | parity, native, sanitizer, static-analysis |
| `QuadraticNeuronART` | `test_QuadraticNeuronART.py` | `test_cpp_QuadraticNeuronART.py` | `test_quadratic_neuron_art_core_smoke.cpp` | Weight/dimension validation | parity, native, sanitizer, static-analysis |
| `ART1MAP` | `test_ART1MAP.py` | `test_cpp_ART1MAP.py` | `test_art1map_core_smoke.cpp` | X/y, weight/label validation | parity, native, sanitizer, static-analysis |
| `BayesianARTMAP` | `test_BayesianARTMAP.py` | `test_cpp_BayesianARTMAP.py` | `test_bayesian_artmap_core_smoke.cpp` | X/y, covariance, weight/label validation | parity, native, sanitizer, static-analysis |
| `BinaryFuzzyARTMAP` | `test_BinaryFuzzyARTMAP.py` | `test_cpp_BinaryFuzzyARTMAP.py` | `test_binary_fuzzy_artmap_core_smoke.cpp` | X/y, match-tracking, weight/label validation | parity, native, sanitizer, static-analysis |
| `FuzzyARTMAP` | `test_FuzzyARTMAP.py` | `test_cpp_FuzzyARTMAP.py` | `test_fuzzy_artmap_core_smoke.cpp` | X/y, match-tracking, weight/label validation | parity, native, sanitizer, static-analysis |
| `GaussianARTMAP` | `test_GaussianARTMAP.py` | `test_cpp_GaussianARTMAP.py` | `test_gaussian_artmap_core_smoke.cpp` | X/y, sigma, weight/label validation | parity, native, sanitizer, static-analysis |
| `HypersphereARTMAP` | `test_HypersphereARTMAP.py` | `test_cpp_HypersphereARTMAP.py` | `test_hypersphere_artmap_core_smoke.cpp` | X/y, match-tracking, weight/label validation | parity, native, sanitizer, static-analysis |
| `SimpleARTMAP` helpers | `test_SimpleARTMAP.py` | `test_cpp_SimpleARTMAP.py` | `test_simple_artmap_core_smoke.cpp` | Out-of-range labels, malformed chains | parity, native, sanitizer, fuzz-smoke, static-analysis |
| `FusionART` helpers | `test_FusionART.py` | `test_cpp_FusionUtils.py` | `test_fusion_core_smoke.cpp` | Shape/null metadata validation | parity, native, sanitizer, fuzz-smoke, static-analysis |
| `CVI` metrics | `test_CVIART.py` | `test_cpp_CVIMetrics.py` | `test_cvi_metrics_core_smoke.cpp` | Label length/shape validation | parity, native, sanitizer, static-analysis |
| `BARTMAP` metrics | `test_BARTMAP.py` | `test_cpp_BARTMAPMetrics.py` | `test_bartmap_metrics_core_smoke.cpp` | Missing cluster/zero-cluster validation | parity, native, sanitizer, static-analysis |
| `fracsort` | `test_fracsort.py` | `test_fracsort.py` | `test_fraction_sort_core_smoke.cpp` | Dtype/shape/zero-denominator validation | parity, native, sanitizer, fuzz-smoke, static-analysis |

## PR Acceptance Checklist

A C++ backend PR should satisfy:

- full Python test suite passes
- C++ parity gate passes
- native CTest suite passes
- sanitizer CTest suite passes or has a documented platform-specific blocker
- static analysis passes or has a documented tool-availability blocker
- any new backend/helper has both Python parity tests and native C++ tests
- malformed input paths raise controlled Python/C++ exceptions instead of crashing
- no compiled artifacts or generated bundles are committed
