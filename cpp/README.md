# C++ Workspace

This folder is the scalable C++ workspace for broad ART porting.

Current production bindings remain under:

- `artlib/optimized/backends/cpp`

This workspace is for:

- shared C++ interfaces
- core algorithm implementations
- C++ tests/benchmarks

## Build

```bash
cmake -S cpp -B cpp/build -DARTLIB_CPP_BUILD_TESTS=ON
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

## Initial Port Workflow

1. Add/extend a model interface in `include/artlib_cpp`.
2. Implement algorithm code in `src/`.
3. Add C++ tests in `tests/`.
4. Add/extend pybind wrapper in `artlib/optimized/backends/cpp/`.
5. Add Python parity tests in `unit_tests/`.

## Python Parity Gate

Run the current parity gate before and after each new C++ backend integration:

```bash
./scripts/run_cpp_parity_gate.sh
```
