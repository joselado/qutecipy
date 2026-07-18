"""qtcipy: Python port of TensorCrossInterpolation.jl.

Public API mirrors the Julia package's exports (crossinterpolate1,
crossinterpolate2, TensorTrain, ...), adapted to 0-based indexing -- see
CLAUDE.md for the full porting plan and design decisions.
"""
from qtcipy.contraction import Contraction, contract
from qtcipy.conversion import tci1_from_tci2, tci2_from_tci1
from qtcipy.gausskronrod import kronrod
from qtcipy.integration import integrate
from qtcipy.tci1 import TensorCI1, crossinterpolate1
from qtcipy.tci2 import TensorCI2, crossinterpolate2, optimize
from qtcipy.tensortrain.base import AbstractTensorTrain
from qtcipy.tensortrain.cache import TTCache
from qtcipy.tensortrain.cachedfunction import CachedFunction
from qtcipy.tensortrain.core import TensorTrain, add, subtract, tensortrain
from qtcipy.util import optfirstpivot

__all__ = [
    "AbstractTensorTrain",
    "TensorTrain",
    "tensortrain",
    "add",
    "subtract",
    "TTCache",
    "CachedFunction",
    "TensorCI1",
    "crossinterpolate1",
    "TensorCI2",
    "crossinterpolate2",
    "optimize",
    "optfirstpivot",
    "tci1_from_tci2",
    "tci2_from_tci1",
    "kronrod",
    "integrate",
    "Contraction",
    "contract",
]
