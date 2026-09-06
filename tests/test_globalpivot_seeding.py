"""Regression test: TCI must not silently interpolate only one branch of a function
whose support straddles a high-order digit boundary.

A Gaussian centred on a quantics grid sits exactly on the most significant bit, so its
two halves differ in *every* digit. The global pivot search visits only points differing
from its start in one coordinate, and its starts were drawn uniformly at random -- which
on a 2**30 grid land in the numerically-zero region essentially always. The result was a
rank-7 train with a reported error of 7e-11 and a true error of 1.0, in 3 runs out of 6.
"""
import numpy as np

from qutecipy import CachedFunction, DiscretizedGrid, crossinterpolate2
from qutecipy.globalpivot import DefaultGlobalPivotFinder, GlobalPivotSearchInput
from qutecipy.tensortrain.core import TensorTrain

R = 30
XMIN, XMAX = -500.0, 500.0


def test_pivot_seeds_include_the_reflection():
    localdims = [2] * 6
    Iset = [[()], [(1,)], [(1, 0)], [(1, 0, 0)], [(1, 0, 0, 0)], [(1, 0, 0, 0, 0)]]
    Jset = [[(0, 0, 0, 0, 0)], [(0, 0, 0, 0)], [(0, 0, 0)], [(0, 0)], [(0,)], [()]]
    inp = GlobalPivotSearchInput(localdims, TensorTrain([np.zeros((1, 2, 1))] * 6), 1.0, Iset, Jset)
    seeds = DefaultGlobalPivotFinder()._pivot_seeds(inp)
    assert seeds, "no seeds derived from the pivot sets"
    for a, b in zip(seeds[::2], seeds[1::2]):
        assert all(x + y == d - 1 for x, y, d in zip(a, b, localdims)), "not a reflection"


def test_gaussian_on_the_bit_boundary_is_fully_interpolated():
    grid = DiscretizedGrid.from_resolutions(["x"], [R], lower_bound=(XMIN,),
                                            upper_bound=(XMAX,), includeendpoint=True)
    psi0 = lambda x: (1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)
    q = lambda b: complex(psi0(grid.quantics_to_origcoord(b)[0]))
    pivot = [0] * R
    pivot[0] = 1                               # index 2**(R-1): the centre, on the MSB

    dx = (XMAX - XMIN) / (2**R - 1)
    probe = 2 ** (R - 1) + np.round(np.linspace(-6 / dx, 6 / dx, 2001)).astype(np.int64)
    exact = psi0(XMIN + probe * dx)

    for run in range(4):
        ci, _, _ = crossinterpolate2(np.complex128, CachedFunction(np.complex128, q, [2] * R),
                                     [2] * R, [pivot], tolerance=1e-10)
        got = np.array([ci.evaluate([(i >> (R - 1 - n)) & 1 for n in range(R)]) for i in probe])
        true_err = np.max(np.abs(got - exact)) / np.max(np.abs(exact))
        # The point of the test: the *true* error, not the reported one. Before the fix
        # this was 1.0 on half the runs while the reported error was 7e-11.
        assert true_err < 1e-6, f"run {run}: true error {true_err:.2e}, rank {ci.rank()}"
