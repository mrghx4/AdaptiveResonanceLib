# setup.py
import sys
import os
import pybind11
from setuptools import setup, Extension, find_packages

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

cpp_dir = os.path.join("artlib", "optimized", "backends", "cpp")
cpp_workspace_include = os.path.join("cpp", "include")
cpp_workspace_src = os.path.join("cpp", "src")

ext_modules = [
    Extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP",
        [
            os.path.join(cpp_dir, "cppBinaryFuzzyARTMAP.cpp"),
            os.path.join(cpp_workspace_src, "binary_fuzzy_artmap_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppFuzzyARTMAP",
        [
            os.path.join(cpp_dir, "cppFuzzyARTMAP.cpp"),
            os.path.join(cpp_workspace_src, "fuzzy_artmap_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppHypersphereARTMAP",
        [
            os.path.join(cpp_dir, "cppHypersphereARTMAP.cpp"),
            os.path.join(cpp_workspace_src, "hypersphere_artmap_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppHypersphereART",
        [
            os.path.join(cpp_dir, "cppHypersphereART.cpp"),
            os.path.join(cpp_workspace_src, "hypersphere_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppEllipsoidART",
        [
            os.path.join(cpp_dir, "cppEllipsoidART.cpp"),
            os.path.join(cpp_workspace_src, "ellipsoid_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppBayesianART",
        [
            os.path.join(cpp_dir, "cppBayesianART.cpp"),
            os.path.join(cpp_workspace_src, "bayesian_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppGaussianARTMAP",
        [
            os.path.join(cpp_dir, "cppGaussianARTMAP.cpp"),
            os.path.join(cpp_workspace_src, "gaussian_artmap_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppGaussianART",
        [
            os.path.join(cpp_dir, "cppGaussianART.cpp"),
            os.path.join(cpp_workspace_src, "gaussian_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppART1MAP",
        [
            os.path.join(cpp_dir, "cppART1MAP.cpp"),
            os.path.join(cpp_workspace_src, "art1map_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.fracsort",
        [os.path.join(cpp_dir, "fracsort.cpp")],
        include_dirs=[pybind11.get_include(), cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppFuzzyART",
        [
            os.path.join(cpp_dir, "cppFuzzyART.cpp"),
            os.path.join(cpp_workspace_src, "fuzzy_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_dir, cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyART",
        [
            os.path.join(cpp_dir, "cppBinaryFuzzyART.cpp"),
            os.path.join(cpp_workspace_src, "binary_fuzzy_art_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_dir, cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppART1",
        [
            os.path.join(cpp_dir, "cppART1.cpp"),
            os.path.join(cpp_workspace_src, "art1_core.cpp"),
        ],
        include_dirs=[pybind11.get_include(), cpp_dir, cpp_workspace_include],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="artlib",
    version="0.1.7",
    packages=find_packages(),  # This all are included
    ext_modules=ext_modules,
)
