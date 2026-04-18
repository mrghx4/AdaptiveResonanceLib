import subprocess
import sys


def test_elementary_modules_import_without_numba():
    script = r"""
import builtins
import importlib

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numba":
        raise ImportError("missing numba")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

utils_module = importlib.import_module("artlib.common.utils")
fuzzy_module = importlib.import_module("artlib.elementary.FuzzyART")
art1_module = importlib.import_module("artlib.elementary.ART1")
binary_module = importlib.import_module("artlib.elementary.BinaryFuzzyART")

importlib.reload(utils_module)
importlib.reload(fuzzy_module)
importlib.reload(art1_module)
importlib.reload(binary_module)

fuzzy = fuzzy_module.FuzzyART(0.5, 0.01, 1.0)
art1 = art1_module.ART1(0.5, 2.0)
binary = binary_module.BinaryFuzzyART(0.5)

assert fuzzy.prepare_data(fuzzy_module.np.random.rand(4, 2)).shape == (4, 4)
assert art1.prepare_data(art1_module.np.array([[0, 1], [1, 0]])).shape == (2, 4)
assert binary.prepare_data(binary_module.np.array([[0, 1], [1, 0]])).shape == (2, 4)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
