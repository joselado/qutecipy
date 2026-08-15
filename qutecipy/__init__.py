"""qutecipy: Python port of TensorCrossInterpolation.jl.

Public API mirrors the Julia package's exports (crossinterpolate1,
crossinterpolate2, TensorTrain, ...), adapted to 0-based indexing -- see
CLAUDE.md for the full porting plan and design decisions.
"""
from qutecipy.arrayvalued import (ArrayTensorTrain, ArrayValuedFunction,
                                 crossinterpolate2_array)
from qutecipy.contraction import Contraction, contract
from qutecipy.conversion import tci1_from_tci2, tci2_from_tci1
from qutecipy.gausskronrod import kronrod
from qutecipy.integration import integrate
from qutecipy.tci1 import TensorCI1, crossinterpolate1
from qutecipy.tci2 import TensorCI2, crossinterpolate2, optimize
from qutecipy.tensortrain.base import AbstractTensorTrain
from qutecipy.tensortrain.cache import TTCache
from qutecipy.tensortrain.cachedfunction import CachedFunction
from qutecipy.tensortrain.core import TensorTrain, add, subtract, tensortrain
from qutecipy.util import optfirstpivot

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
    "ArrayValuedFunction",
    "ArrayTensorTrain",
    "crossinterpolate2_array",
]
