"""This module provides factories for creating optimized versions of common ARTMAP
modules. Currently, the supported ARTMAP variants and their available backends are as
follows:

GaussianART:
    c++
    python
    torch (falls back to c++)

HypersphereART:
    c++
    python
    torch (falls back to c++)

EllipsoidART:
    c++
    python
    torch (falls back to c++)

BayesianART:
    c++
    python
    torch (falls back to c++)

FuzzyARTMAP:
    torch
    c++
    python

HypersphereARTMAP:
    c++
    python

GaussianARTMAP:
    c++
    python

BinaryFuzzyARTMAP:
    c++
    python


Additionally, a c++ implementation of a rational fraction sorting algorithm
"fracsort" is provided.
"""

from artlib.optimized.GaussianARTFactory import GaussianARTFactory
from artlib.optimized.HypersphereARTFactory import HypersphereARTFactory
from artlib.optimized.EllipsoidARTFactory import EllipsoidARTFactory
from artlib.optimized.BayesianARTFactory import BayesianARTFactory

__all__ = [
    "GaussianARTFactory",
    "HypersphereARTFactory",
    "EllipsoidARTFactory",
    "BayesianARTFactory",
]
