# Backend Status Matrix

This document tracks current backend coverage across the ART codebase.

Status meanings:

- `Dedicated C++ backend`: model has a specific implementation under
  `artlib/optimized/backends/cpp/` and matching core code under `cpp/`.
- `Accelerated orchestration`: outer algorithm remains Python, but it can use
  accelerated base modules and/or C++ helper kernels.
- `Python only`: no dedicated accelerated backend is currently implemented.
- `Torch`: dedicated torch backend exists.

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

- Full Python suite baseline: `409 passed, 23 warnings`
- Warning profile remains limited to the known `ART2` warnings
- Dedicated parity/smoke coverage exists for the current C++ backend families
- Benchmark scripts are present for backend runtime, fusion, and hierarchical mapping

## Remaining Highest-Value Conversion Targets

1. Convert accelerated orchestration families into fuller standalone C++ flows:
   `FusionART`, `FALCON`, `TD_FALCON`, `TopoART`, `DualVigilanceART`, `BARTMAP`
2. Decide whether `iCVIFuzzyART` should remain Python-only or get a dedicated backend
3. Expand optional experimental coverage where dependencies are available:
   `AlphaART`, `SphericalAlphaART`
