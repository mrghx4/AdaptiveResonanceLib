"""This module provides factories for creating optimized versions of common ARTMAP
modules. Currently, the supported ARTMAP variants and their available backends are as
follows:

GaussianART:
    c++
    python
    torch (falls back to c++)

ART2A:
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

QuadraticNeuronART:
    c++
    python
    torch (falls back to c++)

BayesianARTMAP:
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

ART1:
    c++
    python
    torch (falls back to c++)

ART1MAP:
    c++
    python (SimpleARTMAP + ART1)
    torch (falls back to c++)


Additionally, a c++ implementation of a rational fraction sorting algorithm
"fracsort" is provided.
"""

from artlib.optimized.ART1Factory import ART1Factory
from artlib.optimized.ART1MAPFactory import ART1MAPFactory
from artlib.optimized.SimpleARTMAPFactory import SimpleARTMAPFactory
from artlib.optimized.ARTMAPFactory import ARTMAPFactory
from artlib.optimized.TopoARTFactory import TopoARTFactory
from artlib.optimized.DualVigilanceARTFactory import DualVigilanceARTFactory
from artlib.optimized.FusionARTFactory import FusionARTFactory
from artlib.optimized.DeepARTMAPFactory import DeepARTMAPFactory
from artlib.optimized.SMARTFactory import SMARTFactory
from artlib.optimized.CVIARTFactory import CVIARTFactory
from artlib.optimized.FALCONFactory import FALCONFactory, TDFALCONFactory
from artlib.optimized.BARTMAPFactory import BARTMAPFactory
from artlib.optimized.iCVIFuzzyARTFactory import iCVIFuzzyARTFactory
from artlib.optimized.FuzzyARTFactory import FuzzyARTFactory
from artlib.optimized.BinaryFuzzyARTFactory import BinaryFuzzyARTFactory
from artlib.optimized.TD_FALCONFactory import TD_FALCONFactory
from artlib.optimized.GaussianARTFactory import GaussianARTFactory
from artlib.optimized.ART2Factory import ART2Factory
from artlib.optimized.HypersphereARTFactory import HypersphereARTFactory
from artlib.optimized.EllipsoidARTFactory import EllipsoidARTFactory
from artlib.optimized.BayesianARTFactory import BayesianARTFactory
from artlib.optimized.BayesianARTMAPFactory import BayesianARTMAPFactory
from artlib.optimized.QuadraticNeuronARTFactory import QuadraticNeuronARTFactory

__all__ = [
    "ART1Factory",
    "ART1MAPFactory",
    "SimpleARTMAPFactory",
    "ARTMAPFactory",
    "TopoARTFactory",
    "DualVigilanceARTFactory",
    "FusionARTFactory",
    "DeepARTMAPFactory",
    "SMARTFactory",
    "CVIARTFactory",
    "FALCONFactory",
    "TDFALCONFactory",
    "BARTMAPFactory",
    "iCVIFuzzyARTFactory",
    "FuzzyARTFactory",
    "BinaryFuzzyARTFactory",
    "TD_FALCONFactory",
    "GaussianARTFactory",
    "ART2Factory",
    "HypersphereARTFactory",
    "EllipsoidARTFactory",
    "BayesianARTFactory",
    "BayesianARTMAPFactory",
    "QuadraticNeuronARTFactory",
]
