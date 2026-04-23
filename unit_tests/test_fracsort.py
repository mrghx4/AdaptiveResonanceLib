import importlib
import builtins
import numpy as np
import pytest

import artlib.common.utils as utils_module
from artlib.common.utils import fracsort, fracargmax


def test_fracsort_matches_numpy_division_argsort() -> None:
    """Test that fracsort matches a NumPy argsort on the division-based proxy.

    This test generates random denominators in [1, 10000] and random numerators
    satisfying num <= den, then verifies that the index order returned by the
    compiled division-free fracsort matches NumPy's stable argsort of num/den.

    Notes
    -----
    - We use dtype uint32 to exercise the intended fast path.
    - NumPy sorting uses a stable sort so ties are broken by lowest index.

    """
    rng = np.random.default_rng(0)
    n = 10000

    den = rng.integers(1, n+1, size=n, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=n, dtype=np.uint32)
    eps = 1e-10

    # Force exact ties in the ratio for a few indices:
    # 1/2 == 2/4; tie should pick larger denominator (4), then lower index among those.
    tie_idx = np.array([123, 456, 789], dtype=np.int64)
    den[tie_idx[0]] = np.uint32(2); num[tie_idx[0]] = np.uint32(1)
    den[tie_idx[1]] = np.uint32(4); num[tie_idx[1]] = np.uint32(2)
    den[tie_idx[2]] = np.uint32(4); num[tie_idx[2]] = np.uint32(2)

    # Ensure contiguous arrays (the backend requires C-contiguous input)
    den = np.ascontiguousarray(den)
    num = np.ascontiguousarray(num)

    idx_cpp = fracsort(num, den)

    ratio = num.astype(np.float64) / (den.astype(np.float64) + eps)
    idx_np = np.argsort(-ratio, kind="stable")

    # assertions
    zero = (num == 0)
    k = int(np.count_nonzero(~zero))  # number of non-zeros

    # sanity: zeros form the suffix in each ordering
    assert not np.any(zero[idx_cpp[:k]])
    assert np.all(zero[idx_cpp[k:]])
    assert not np.any(zero[idx_np[:k]])
    assert np.all(zero[idx_np[k:]])

    # Now compare only the non-zero prefix
    assert np.array_equal(idx_cpp[:k], idx_np[:k])


def test_fracargmax_matches_numpy_with_tiebreaks() -> None:
    """Test that fracargmax matches a NumPy argmax on the division-based proxy.

    We verify that fracargmax returns the index maximizing num/den, using a float
    proxy for the ratio and explicitly implementing the tie-breaks:
      1) larger denominator
      2) lower index

    Notes
    -----
    - Use dtype uint32 (fast path).
    - We keep num <= den so ratios are in [0, 1] and float comparisons are well-behaved.
    - We force ties to ensure tie-break logic is exercised.

    """
    rng = np.random.default_rng(1)
    n = 10000

    den = rng.integers(1, n + 1, size=n, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=n, dtype=np.uint32)
    eps = 1e-10

    # Force exact ties in the ratio for a few indices:
    # 1/2 == 2/4; tie should pick larger denominator (4), then lower index among those.
    tie_idx = np.array([123, 456, 789], dtype=np.int64)
    den[tie_idx[0]] = np.uint32(2); num[tie_idx[0]] = np.uint32(1)
    den[tie_idx[1]] = np.uint32(4); num[tie_idx[1]] = np.uint32(2)
    den[tie_idx[2]] = np.uint32(4); num[tie_idx[2]] = np.uint32(2)

    # Ensure contiguous arrays (the backend requires C-contiguous input)
    den = np.ascontiguousarray(den)
    num = np.ascontiguousarray(num)

    idx_cpp = fracargmax(num, den)

    ratio = num.astype(np.float64) / (den.astype(np.float64) + eps)
    max_ratio = ratio.max()

    # Candidates within exact max (float proxy).
    # Since we constrained num<=den and eps is tiny,
    # this is stable enough for the test; ties are resolved explicitly afterward.
    candidates = np.flatnonzero(ratio == max_ratio)
    # Apply tiebreaks: larger den, then lower index
    best = max(candidates, key=lambda i: (int(den[i]), -int(i)))  # max by den, then min i
    # The key above uses -i so larger key corresponds to lower index.

    assert int(idx_cpp) == int(best)


def test_fracsort_rejects_malformed_inputs_without_crashing() -> None:
    with pytest.raises((RuntimeError, ValueError)):
        fracsort(
            np.array([[1, 2]], dtype=np.uint32),
            np.array([[2, 3]], dtype=np.uint32),
        )

    with pytest.raises((RuntimeError, ValueError)):
        fracsort(
            np.array([1, 2], dtype=np.uint32),
            np.array([2], dtype=np.uint32),
        )

    with pytest.raises((RuntimeError, ValueError)):
        fracsort(
            np.array([1, 2], dtype=np.uint32),
            np.array([2, 0], dtype=np.uint32),
        )


def test_fracargmax_rejects_empty_and_zero_denominator_inputs() -> None:
    with pytest.raises((RuntimeError, ValueError)):
        fracargmax(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32))

    with pytest.raises((RuntimeError, ValueError)):
        fracargmax(
            np.array([1, 2], dtype=np.uint32),
            np.array([2, 0], dtype=np.uint32),
        )


def test_fracsort_python_fallback_matches_reference(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    den = rng.integers(1, 1000, size=256, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=256, dtype=np.uint32)

    monkeypatch.setattr(utils_module, "_fracsort", None)
    idx_fallback = utils_module.fracsort(num, den)

    ratio = num.astype(np.float64) / den.astype(np.float64)
    idx_np = np.argsort(-ratio, kind="stable")
    zero = num == 0
    k = int(np.count_nonzero(~zero))

    assert np.array_equal(idx_fallback[:k], idx_np[:k])
    assert np.all(zero[idx_fallback[k:]])


def test_fracargmax_python_fallback_matches_reference(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    den = rng.integers(1, 1000, size=256, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=256, dtype=np.uint32)

    tie_idx = np.array([5, 7, 11], dtype=np.int64)
    den[tie_idx[0]] = np.uint32(2)
    num[tie_idx[0]] = np.uint32(1)
    den[tie_idx[1]] = np.uint32(4)
    num[tie_idx[1]] = np.uint32(2)
    den[tie_idx[2]] = np.uint32(4)
    num[tie_idx[2]] = np.uint32(2)

    monkeypatch.setattr(utils_module, "_fracargmax", None)
    idx_fallback = utils_module.fracargmax(num, den)

    ratio = num.astype(np.float64) / den.astype(np.float64)
    max_ratio = ratio.max()
    candidates = np.flatnonzero(ratio == max_ratio)
    best = max(candidates, key=lambda i: (int(den[i]), -int(i)))

    assert int(idx_fallback) == int(best)


def test_utils_imports_without_cpp_fracsort_extension(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "artlib.optimized.backends.cpp.fracsort":
            raise ImportError("missing fracsort extension")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", _fake_import)
        reloaded = importlib.reload(utils_module)
        assert reloaded._fracsort is None
        assert reloaded._fracargmax is None
        assert np.array_equal(
            reloaded.fracsort(np.array([1, 2], dtype=np.uint32), np.array([2, 3], dtype=np.uint32)),
            np.array([1, 0], dtype=np.intp),
        )

    importlib.reload(utils_module)


def test_utils_imports_without_numba(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba":
            raise ImportError("missing numba")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", _fake_import)
        reloaded = importlib.reload(utils_module)
        assert callable(reloaded.njit)

        @reloaded.njit
        def _direct(x):
            return x + 1

        @reloaded.njit(cache=True)
        def _factory(x):
            return x + 2

        assert _direct(1) == 2
        assert _factory(1) == 3

    importlib.reload(utils_module)
