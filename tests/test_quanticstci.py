"""Tests for qutecipy.quanticstci (quantics TCI; an extension, no Julia counterpart).

There is no vendored QuanticsTCI.jl to cross-check against, so the gate here is:
the wrapper must be *exactly* the manual composition of the two ported halves (bit
for bit, not to a tolerance), and its reductions must match closed forms and
brute-force sums.
"""
import itertools

import numpy as np
import pytest

from qutecipy.quantics import DiscretizedGrid, InherentDiscreteGrid, quantics_function
from qutecipy.quanticstci import QuanticsTensorCI, quantics_crossinterpolate
from qutecipy.tci2 import crossinterpolate2
from qutecipy.tensortrain.core import TensorTrain


def _grid1d(R=10, lo=0.0, hi=1.0, **kw):
    return DiscretizedGrid.from_resolutions(["x"], [R], lower_bound=(lo,), upper_bound=(hi,), **kw)


def _f1(x):
    return np.exp(-x) * np.cos(20.0 * x) + 0.3


def _gridpoints(grid, n, d=0):
    """`n` coordinates spread over the grid, each landing exactly on a grid point.

    Evaluation snaps to the nearest grid point (see test_evaluation_snaps_to_the_grid),
    so accuracy assertions have to be made *on* the grid -- comparing qtt(x) to f(x) at
    an arbitrary x measures the grid spacing, not the interpolation.
    """
    npoints = grid.grid_bases()[d] ** grid.grid_Rs()[d]
    stride = max(1, npoints // n)
    return [grid.grididx_to_origcoord((i,))[d] for i in range(0, npoints, stride)]


# -- the wrapper is exactly the manual composition ---------------------------


def test_matches_manual_composition_bit_for_bit():
    """quantics_crossinterpolate must be doing precisely what a user would do by
    hand -- same pivots, same tensors, same values -- with no silent reinterpretation
    in between. Caching is off here so both runs make the same sequence of calls."""
    grid = _grid1d(R=10)
    qtt, ranks, errors = quantics_crossinterpolate(
        np.float64, _f1, grid, tolerance=1e-10, cache=False
    )

    qf = quantics_function(np.float64, grid, _f1)
    tci, ranks_ref, errors_ref = crossinterpolate2(
        np.float64, qf, grid.localdimensions(), tolerance=1e-10
    )

    assert ranks == ranks_ref
    assert errors == errors_ref
    for a, b in zip(qtt.tci.sitetensors(), tci.sitetensors()):
        assert np.array_equal(a, b)

    for x in _gridpoints(grid, 53):
        manual = tci.evaluate(list(grid.origcoord_to_quantics((x,))))
        assert qtt(x) == manual
        assert qtt.eval(x) == manual
        assert qtt([x]) == manual


def test_grid_and_quantics_evaluation_agree():
    grid = _grid1d(R=8)
    qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-10)
    for gi in range(0, 2 ** 8, 37):
        q = grid.grididx_to_quantics((gi,))
        assert qtt.grid_eval((gi,)) == qtt.quantics_eval(q)
        x = grid.grididx_to_origcoord((gi,))[0]
        assert qtt(x) == qtt.grid_eval((gi,))


# -- accuracy ---------------------------------------------------------------


def test_compresses_a_fine_grid():
    """The headline: 2**20 grid points at bond dimension 3."""
    grid = _grid1d(R=20)
    qtt, _, errors = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-9)
    assert len(qtt) == 20
    assert qtt.ndims() == 1
    assert qtt.rank() <= 5
    assert errors[-1] < 1e-9

    err = max(abs(qtt(x) - _f1(x)) for x in _gridpoints(grid, 401))
    assert err < 1e-8


def test_two_dimensional():
    """2-D exercises the tuple handling in origcoord_to_quantics and the product of
    step sizes in integral() -- both invisible to a 1-D-only test."""
    R = 6

    def f2(x, y):
        return np.exp(-x) * np.sin(2.0 * y) + 0.5

    grid = DiscretizedGrid.from_resolutions(
        ["x", "y"], [R, R], lower_bound=(0.0, 0.0), upper_bound=(1.0, 2.0)
    )
    qtt, _, _ = quantics_crossinterpolate(np.float64, f2, grid, tolerance=1e-10)
    assert qtt.ndims() == 2

    xs = grid.grid_origcoords(0)
    ys = grid.grid_origcoords(1)
    err = max(abs(qtt(x, y) - f2(x, y)) for x in xs[::7] for y in ys[::7])
    assert err < 1e-8

    # sum() and integral() against brute force over the whole (small) grid
    brute = sum(f2(x, y) for x in xs for y in ys)
    assert qtt.sum() == pytest.approx(brute, rel=1e-8)
    steps = grid.grid_step()
    assert qtt.integral() == pytest.approx(brute * steps[0] * steps[1], rel=1e-8)


def test_interleaved_unfolding():
    """A different unfolding scheme changes localdimensions(), which is what feeds
    TCI -- so it has to be carried through, not assumed."""
    R = 6

    def f2(x, y):
        return np.exp(-x - y) + 0.25

    grids = {
        scheme: DiscretizedGrid.from_resolutions(
            ["x", "y"], [R, R], lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0),
            unfoldingscheme=scheme,
        )
        for scheme in ("fused", "interleaved")
    }
    assert grids["fused"].localdimensions() != grids["interleaved"].localdimensions()

    for scheme, grid in grids.items():
        qtt, _, _ = quantics_crossinterpolate(np.float64, f2, grid, tolerance=1e-10)
        assert len(qtt) == len(grid.localdimensions())
        xs, ys = grid.grid_origcoords(0), grid.grid_origcoords(1)
        err = max(abs(qtt(x, y) - f2(x, y)) for x in xs[::5] for y in ys[::5])
        assert err < 1e-8, scheme


# -- reductions -------------------------------------------------------------


def test_sum_matches_brute_force():
    R = 8
    grid = _grid1d(R=R)
    qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-12)
    brute = sum(_f1(x) for x in grid.grid_origcoords(0))
    assert qtt.sum() == pytest.approx(brute, rel=1e-10)


def test_integral_matches_closed_form_at_first_order():
    """The rectangle rule converges at O(step), which is the documented caveat: the
    integral error must shrink like 2**-R even though the interpolation is far more
    accurate than that."""
    exact = (np.exp(-1.0) * (20.0 * np.sin(20.0) - np.cos(20.0)) + 1.0) / 401.0 + 0.3

    errs = {}
    for R in (10, 14, 18):
        qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, _grid1d(R=R), tolerance=1e-12)
        errs[R] = abs(qtt.integral() - exact)

    assert errs[18] < 1e-5
    # first order: four more bits of resolution should buy roughly 16x accuracy
    assert errs[10] / errs[14] > 8.0
    assert errs[14] / errs[18] > 8.0


def test_integral_refuses_endpoint_inclusive_grids():
    """A grid built with includeendpoint=True spans the closed interval, so the
    half-open rectangle rule would silently integrate over one step too many."""
    grid = _grid1d(R=8, includeendpoint=(True,))
    qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-10)
    with pytest.raises(ValueError, match="includeendpoint"):
        qtt.integral()
    qtt.sum()  # still fine


def test_integral_refuses_discrete_grids():
    grid = InherentDiscreteGrid.from_resolutions([6], variablenames=["n"])
    qtt, _, _ = quantics_crossinterpolate(np.float64, lambda n: 1.0 / (1.0 + n), grid,
                                          tolerance=1e-10)
    with pytest.raises(TypeError, match="DiscretizedGrid"):
        qtt.integral()
    brute = sum(1.0 / (1.0 + n) for n in range(2 ** 6))
    assert qtt.sum() == pytest.approx(brute, rel=1e-8)


# -- options ----------------------------------------------------------------


def test_cache_is_on_by_default_and_can_be_turned_off():
    grid = _grid1d(R=12)
    counts = {}
    for label, cache in (("cached", True), ("plain", False)):
        n = [0]

        def counting(x, n=n):
            n[0] += 1
            return _f1(x)

        qtt, _, _ = quantics_crossinterpolate(
            np.float64, counting, grid, tolerance=1e-9, cache=cache
        )
        counts[label] = n[0]
        x = _gridpoints(grid, 3)[1]
        assert abs(qtt(x) - _f1(x)) < 1e-8

    # the pivot search revisits points; caching is the whole reason this is the default
    assert counts["cached"] < counts["plain"]


def test_initialpivots_are_given_in_coordinates():
    """A sharp feature the default all-zero pivot would not find on its own."""
    grid = _grid1d(R=14)

    def peaked(x):
        return 1e-6 / (1e-6 + (x - 0.371) ** 2)

    # compare on the grid point the peak snaps to, not at 0.371 itself
    xpeak = grid.grididx_to_origcoord(grid.origcoord_to_grididx((0.371,)))[0]

    qtt, _, _ = quantics_crossinterpolate(
        np.float64, peaked, grid, initialpivots=[0.371], tolerance=1e-8
    )
    assert abs(qtt(xpeak) - peaked(xpeak)) / abs(peaked(xpeak)) < 1e-5

    # scalars and 1-tuples are both accepted in one dimension
    qtt2, _, _ = quantics_crossinterpolate(
        np.float64, peaked, grid, initialpivots=[(0.371,)], tolerance=1e-8
    )
    assert abs(qtt2(xpeak) - peaked(xpeak)) / abs(peaked(xpeak)) < 1e-5


def test_tensortrain_and_accessors():
    grid = _grid1d(R=8)
    qtt, ranks, errors = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-10)
    tt = qtt.tensortrain()
    assert isinstance(tt, TensorTrain)
    assert len(tt) == len(qtt)
    assert qtt.linkdims() == qtt.tci.linkdims()
    assert qtt.rank() == max(qtt.linkdims())
    assert qtt.ranks == ranks and qtt.errors == errors
    assert qtt.grid is grid
    assert qtt.func is not None
    x = 0.25
    assert tt.evaluate(list(grid.origcoord_to_quantics((x,)))) == qtt(x)


def test_maxbonddim_is_respected():
    grid = _grid1d(R=12)
    qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-14, maxbonddim=3)
    assert max(qtt.linkdims()) <= 3


def test_exported_at_top_level():
    import qutecipy

    for name in ("quantics_crossinterpolate", "QuanticsTensorCI", "DiscretizedGrid",
                 "InherentDiscreteGrid", "quantics_function"):
        assert hasattr(qutecipy, name), name
        assert name in qutecipy.__all__


def test_constructed_directly():
    """QuanticsTensorCI can wrap a TCI2 built elsewhere."""
    grid = _grid1d(R=8)
    qf = quantics_function(np.float64, grid, _f1)
    tci, _, _ = crossinterpolate2(np.float64, qf, grid.localdimensions(), tolerance=1e-10)
    qtt = QuanticsTensorCI(tci, grid, np.float64)
    assert qtt.ranks == [] and qtt.errors == []
    assert qtt(0.5) == tci.evaluate(list(grid.origcoord_to_quantics((0.5,))))


def test_complex_valued():
    grid = _grid1d(R=10)

    def fc(x):
        return np.exp(-x) * np.exp(2.0j * x)

    qtt, _, _ = quantics_crossinterpolate(np.complex128, fc, grid, tolerance=1e-10)
    for x in _gridpoints(grid, 29):
        assert qtt(x) == pytest.approx(fc(x), abs=1e-8)


def test_rook_pivotsearch():
    grid = _grid1d(R=12)
    qtt, _, _ = quantics_crossinterpolate(
        np.float64, _f1, grid, tolerance=1e-9, pivotsearch="rook"
    )
    err = max(abs(qtt(x) - _f1(x)) for x in _gridpoints(grid, 97))
    assert err < 1e-6


def test_evaluation_snaps_to_the_grid():
    """qtt(x) answers with the nearest *grid* point, so off-grid accuracy is bounded
    by the grid spacing rather than by `tolerance`. Worth pinning: it is the one thing
    about this layer that surprises."""
    R = 6
    grid = _grid1d(R=R)
    qtt, _, _ = quantics_crossinterpolate(np.float64, _f1, grid, tolerance=1e-12)

    step = grid.grid_step()[0]
    x0 = grid.grididx_to_origcoord((10,))[0]
    # anything within half a step of a grid point gives that grid point's value
    assert qtt(x0 + 0.25 * step) == qtt(x0)
    assert qtt(x0 - 0.25 * step) == qtt(x0)
    assert qtt(x0 + 0.75 * step) == qtt(grid.grididx_to_origcoord((11,))[0])
    # ...so at an off-grid coordinate the error is set by the spacing, not the tolerance
    assert abs(qtt(x0 + 0.5 * step) - _f1(x0 + 0.5 * step)) > 1e-10


def test_itertools_unused_guard():
    """Guard against the grid enumeration helpers drifting: grid_origcoords must line
    up with grididx_to_origcoord for every point."""
    grid = _grid1d(R=5)
    coords = grid.grid_origcoords(0)
    assert len(coords) == 2 ** 5
    for i, x in zip(itertools.count(), coords):
        assert grid.grididx_to_origcoord((i,))[0] == x
