# Backend Status Matrix

This document tracks current backend coverage across the ART codebase.

Status meanings:

- `Dedicated C++ backend`: model has a specific implementation under
  `artlib/optimized/backends/cpp/` and matching core code under `cpp/`.
- `Accelerated orchestration`: outer algorithm remains Python, but it can use
  accelerated base modules and/or C++ helper kernels.
- `Python only`: no dedicated accelerated backend is currently implemented.
- `Torch`: dedicated torch backend exists.

## Backend Conversion Contract

Current factory/dispatch behavior is intentionally conservative:

- Factories accelerate fresh Python ART modules into supported C++ backends.
- Prepared or fitted Python modules are intentionally kept as Python instances.
- The current backend layer does **not** implement general Python-to-C++ state
  transfer for live models.

In practice, that means:

- use factories when constructing new accelerated models
- do not assume an existing prepared/fitted Python model can be upgraded in
  place to an equivalent C++ backend instance
- if full state transfer is needed later, it should be implemented explicitly
  and validated family-by-family

## Elementary Models

| Family | Python | C++ | Torch | Notes |
| --- | --- | --- | --- | --- |
| `ART1` | Yes | Dedicated | No | Factory + parity tests present |
| `ART2A` | Yes | Dedicated | No | Known warning: algorithm not recommended |
| `BayesianART` | Yes | Dedicated | No | Factory + parity tests present |
| `BinaryFuzzyART` | Yes | Dedicated | No | Factory + parity tests present |
| `EllipsoidART` | Yes | Dedicated | No | Factory + parity tests present |
| `FuzzyART` | Yes | Dedicated | No | Factory + parity tests present |
| `GaussianART` | Yes | Dedicated | No | Factory + parity tests present |
| `HypersphereART` | Yes | Dedicated | No | Factory + parity tests present |
| `QuadraticNeuronART` | Yes | Dedicated | No | Factory + parity tests present |

## ARTMAP / Supervised

| Family | Python | C++ | Torch | Notes |
| --- | --- | --- | --- | --- |
| `ART1MAP` | Yes | Dedicated | No | Specialized factory path |
| `BayesianARTMAP` | Yes | Dedicated | No | Specialized factory path |
| `BinaryFuzzyARTMAP` | Yes | Dedicated | Dedicated | Torch backend present |
| `FuzzyARTMAP` | Yes | Dedicated | Dedicated | Torch backend present |
| `GaussianARTMAP` | Yes | Dedicated | Dedicated | Torch backend present |
| `HypersphereARTMAP` | Yes | Dedicated | Dedicated | Torch backend present |
| `SimpleARTMAP` | Yes | Dedicated for selected bases | No | Generic factory dispatches to specialized C++ variants when supported |
| `ARTMAP` | Yes | Accelerated orchestration | No | Uses accelerated base modules via factory dispatch |

## Hierarchical / Fusion / Reinforcement / Topological

| Family | Python | C++ | Torch | Notes |
| --- | --- | --- | --- | --- |
| `FusionART` | Yes | Accelerated orchestration | No | Uses accelerated base modules + C++ helper kernels |
| `DeepARTMAP` | Yes | Accelerated orchestration | No | Uses accelerated layers via factory dispatch |
| `SMART` | Yes | Accelerated orchestration | No | Uses accelerated base class via factory dispatch |
| `FALCON` | Yes | Accelerated orchestration | No | Uses accelerated base modules via factory dispatch |
| `TD_FALCON` | Yes | Accelerated orchestration | No | Uses accelerated base modules via factory dispatch |
| `TopoART` | Yes | Accelerated orchestration | No | Uses accelerated base modules via factory dispatch |
| `DualVigilanceART` | Yes | Accelerated orchestration | No | Uses accelerated base modules via factory dispatch |

## CVI / Biclustering

| Family | Python | C++ | Torch | Notes |
| --- | --- | --- | --- | --- |
| `CVIART` | Yes | Accelerated orchestration | No | Uses accelerated base modules + C++ CVI helper |
| `iCVIFuzzyART` | Yes | No | No | Python implementation only |
| `BARTMAP` | Yes | Accelerated orchestration | No | Uses accelerated base modules + C++ metric helpers |

## Experimental

| Family | Python | C++ | Torch | Notes |
| --- | --- | --- | --- | --- |
| `AlphaART` | Yes | No | No | Optional `pyalphashape` dependency |
| `SeqART` | Yes | No | No | Experimental sequence alignment clustering |
| `SphericalAlphaART` | Yes | No | No | Optional `pyalphashape` dependency |
| `merging` utilities | Yes | No | No | Union-find helpers |

## Current Validation State

- Full Python suite baseline: `482 passed, 1 skipped, 23 warnings`
- Warning profile remains limited to the known `ART2` warnings
- Dedicated parity/smoke coverage exists for the current C++ backend families
- Benchmark scripts are present for backend runtime, fusion, and hierarchical mapping
- Canonical local validation command:
  `.venv/bin/python -m pytest -q`
- Factory MNIST backend-comparison tests now default to a smaller routine slice.
  Use `ART_FACTORY_TRAIN_SAMPLES` and `ART_FACTORY_TEST_SAMPLES` to opt into larger
  benchmark-style runs when you want full-MNIST comparisons explicitly.
- For a dedicated local factory-backend comparison run, use:
  `scripts/run_factory_backend_vnv.sh`
- For the larger MNIST slice used in extended local comparisons, use:
  `scripts/run_factory_backend_vnv.sh --full-mnist`

## Remaining Highest-Value Conversion Targets

1. Convert accelerated orchestration families into fuller standalone C++ flows:
   `FusionART`, `FALCON`, `TD_FALCON`, `TopoART`, `DualVigilanceART`, `BARTMAP`
2. Decide whether `iCVIFuzzyART` should remain Python-only or get a dedicated backend
3. Expand optional experimental coverage where dependencies are available:
   `AlphaART`, `SphericalAlphaART`
