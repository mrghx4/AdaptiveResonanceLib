# C++ Port Migration Plan

This repository already includes production C++ extension modules in
`artlib/optimized/backends/cpp`. The goal here is to scale from selective
accelerated modules to broad C++ coverage without breaking the existing Python API.

## Strategy

1. Keep one repository.
2. Use a dedicated long-lived branch for C++ migration work.
3. Keep existing pybind extension flow working while introducing a structured C++ workspace.
4. Port incrementally, one algorithm family at a time.
5. Require parity tests before switching defaults to C++.

## Branching Model

- Create and keep a branch like: `codex/cpp-port-foundation`.
- Use short-lived feature branches for each module port:
  - `codex/cpp-port-fuzzyart`
  - `codex/cpp-port-gaussianart`
  - etc.
- Merge only when:
  - Python/C++ parity tests pass
  - Existing unit tests stay green

## Directory Model

- `artlib/optimized/backends/cpp/`:
  - Existing pybind modules and current production C++ integration.
- `cpp/`:
  - New scalable C++ workspace (core library, tests, benchmarks, CMake entrypoint).

## Porting Order

1. Shared math + utility primitives.
2. Elementary unsupervised modules.
3. ARTMAP variants.
4. Higher-level/reinforcement modules.

Choose modules with highest runtime cost first for early performance returns.

## Parity Rules

- Compare behaviors, not byte-exact floating-point values.
- Use deterministic seeds.
- Validate:
  - cluster assignments
  - category counts
  - score metrics
  - prediction outputs
- Use tolerances (`np.allclose`) for floating point state.

## CI Guidance

Minimum jobs:

1. Python unit tests (existing).
2. C++ workspace build + smoke tests.
3. Python parity tests against C++ backends for newly ported modules.

## Acceptance Checklist Per Module

- C++ implementation added under `cpp/src`.
- pybind shim integrated in existing backend package.
- Parity tests added and passing.
- Performance sanity benchmark captured.
- Docs updated for backend availability and caveats.
