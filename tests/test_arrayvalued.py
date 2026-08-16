"""Tests for qutecipy.arrayvalued (array-valued TCI; an extension, no Julia counterpart)."""
import itertools

import numpy as np
import pytest

from qutecipy.arrayvalued import (ArrayTensorTrain, ArrayValuedFunction,
                                  crossinterpolate2_array)
from qutecipy.tci2 import crossinterpolate2


def _bits_to_x(bits):
    """Map a bitstring to x in [0, 1)."""
    idx = 0
    for b in bits:
        idx = idx * 2 + b
    return idx / 2 ** len(bits)


R = 8
LOCALDIMS = [2] * R


def _vecfun(bits):
    """R^R -> R^3: three smooth, similarly-structured functions of the same x."""
    x = _bits_to_x(bits)
    return np.array([
        np.exp(-2.0 * x),
        np.cos(3.0 * x) + 0.5,
        1.0 / (1.0 + x),
    ])


def _allx(localdims):
    return list(itertools.product(*[range(d) for d in localdims]))


def _maxerr(att, f, localdims):
    err = 0.0
    for x in _allx(localdims):
        err = max(err, float(np.max(np.abs(att(list(x)) - f(list(x))))))
    return err


# -- basic accuracy ---------------------------------------------------------


def test_vector_valued_accuracy():
    att, ranks, errors = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10
    )
    assert att.valueshape == (3,)
    assert att(np.zeros(R, dtype=int)).shape == (3,)
    assert _maxerr(att, _vecfun, LOCALDIMS) < 1e-8
    assert errors[-1] < 1e-10
    assert len(ranks) == len(errors)


def test_matrix_valued_accuracy():
    def matfun(bits):
        x = _bits_to_x(bits)
        return np.array([
            [np.exp(-x), np.sin(2.0 * x)],
            [x ** 2 + 1.0, np.cos(x)],
            [1.0 / (2.0 + x), np.exp(-3.0 * x)],
        ])

    att, _, _ = crossinterpolate2_array(np.float64, matfun, LOCALDIMS, (3, 2), tolerance=1e-10)
    assert att.valueshape == (3, 2)
    assert att.K == 6
    assert att(np.zeros(R, dtype=int)).shape == (3, 2)
    assert _maxerr(att, matfun, LOCALDIMS) < 1e-8


def test_accuracy_matches_independent_scalar_runs():
    """Shared pivots must not cost accuracy relative to K separate TCI runs."""
    att, _, _ = crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10)

    for k in range(3):
        tci, _, _ = crossinterpolate2(
            np.float64, lambda b, k=k: _vecfun(b)[k], LOCALDIMS, [[0] * R], tolerance=1e-10
        )
        joint = max(
            abs(att(list(x))[k] - _vecfun(list(x))[k]) for x in _allx(LOCALDIMS)
        )
        alone = max(
            abs(tci.evaluate(list(x)) - _vecfun(list(x))[k]) for x in _allx(LOCALDIMS)
        )
        assert joint < max(10 * alone, 1e-9)


# -- the "one MPS per component, same pivots" deliverable --------------------


def test_component_tensortrains_share_pivots_and_reproduce_the_array():
    att, _, _ = crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10)
    comps = att.components()
    assert len(comps) == 3

    for tt in comps:
        assert len(tt) == R
        assert tt.linkdims() == att.tci.linkdims()[:-1]

    # every site tensor but the last is identical across components -- that is
    # what "the same pivots for all components" buys.
    for tt in comps[1:]:
        for b in range(R - 1):
            assert np.array_equal(tt.sitetensor(b), comps[0].sitetensor(b))
    assert not np.array_equal(comps[1].sitetensor(R - 1), comps[0].sitetensor(R - 1))

    for x in _allx(LOCALDIMS)[::37]:
        arr = att(list(x))
        for k in range(3):
            assert comps[k].evaluate(list(x)) == pytest.approx(arr[k], abs=1e-12)


def test_component_accepts_tuple_index_and_copies():
    def matfun(bits):
        x = _bits_to_x(bits)
        return np.array([[x, x ** 2], [np.exp(-x), np.cos(x)]])

    att, _, _ = crossinterpolate2_array(np.float64, matfun, LOCALDIMS, (2, 2), tolerance=1e-10)
    x = [1, 0, 1, 1, 0, 0, 1, 0]
    for i, j in itertools.product(range(2), range(2)):
        assert att.component((i, j)).evaluate(x) == pytest.approx(att(x)[i, j], abs=1e-12)
        assert att.component((i, j)).evaluate(x) == pytest.approx(
            att.component(2 * i + j).evaluate(x), abs=1e-14
        )

    # component() returns an independent copy: mutating it must not touch the parent
    tt = att.component(0)
    before = att.sitetensors()[0].copy()
    tt.sitetensors()[0][:] = 0.0
    assert np.array_equal(att.sitetensors()[0], before)

    with pytest.raises(IndexError):
        att.component(4)


# -- the adapter ------------------------------------------------------------


def test_one_f_call_per_distinct_x():
    calls = []

    def counting(bits):
        calls.append(tuple(bits))
        return _vecfun(bits)

    att, _, _ = crossinterpolate2_array(np.float64, counting, LOCALDIMS, (3,), tolerance=1e-8)
    assert len(calls) == len(set(calls))          # never evaluated twice at the same x
    assert len(calls) == att.func.ncalls
    assert att.func.ncalls == len(att.func.cache)
    # the whole point: TCI2 makes many more scalar (x, k) queries than there
    # are distinct x, and each distinct x costs exactly one array evaluation.
    assert att.func.nqueries > 3 * att.func.ncalls


def test_adapter_splitting_and_extended_dims():
    fa = ArrayValuedFunction(np.float64, _vecfun, [2, 3, 4], (5,))
    assert fa.extendedlocaldims == [2, 3, 4, 5]
    assert fa.split([1, 2, 3, 4]) == ((1, 2, 3), 4)
    assert fa.extend_pivot([1, 2, 3]) == [(1, 2, 3, k) for k in range(5)]

    fa = ArrayValuedFunction(np.float64, _vecfun, [2, 3, 4], (5,), componentposition="first")
    assert fa.extendedlocaldims == [5, 2, 3, 4]
    assert fa.split([4, 1, 2, 3]) == ((1, 2, 3), 4)
    assert fa.extend_pivot([1, 2, 3]) == [(k, 1, 2, 3) for k in range(5)]

    with pytest.raises(ValueError):
        fa.split([1, 2, 3])
    with pytest.raises(ValueError):
        ArrayValuedFunction(np.float64, _vecfun, [2, 2], (3,), componentposition="middle")


def test_adapter_rejects_wrong_output_shape():
    fa = ArrayValuedFunction(np.float64, lambda b: np.zeros((2, 2)), [2, 2], (3,))
    with pytest.raises(ValueError):
        fa([0, 0, 0])


def test_adapter_clear():
    fa = ArrayValuedFunction(np.float64, _vecfun, LOCALDIMS, (3,))
    fa([0] * R + [0])
    assert len(fa.cache) == 1
    fa.clear()
    assert len(fa.cache) == 0


# -- component position -----------------------------------------------------


def test_componentposition_first():
    att, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), componentposition="first", tolerance=1e-10
    )
    assert att.localdims() == LOCALDIMS
    assert _maxerr(att, _vecfun, LOCALDIMS) < 1e-8

    comps = att.components()
    for tt in comps:
        assert len(tt) == R
    for tt in comps[1:]:
        for b in range(1, R):
            assert np.array_equal(tt.sitetensor(b), comps[0].sitetensor(b))

    x = [0, 1, 1, 0, 1, 0, 0, 1]
    for k in range(3):
        assert comps[k].evaluate(x) == pytest.approx(att(x)[k], abs=1e-12)


# -- weights ----------------------------------------------------------------


def _scaledfun(bits):
    """Two components differing by 8 orders of magnitude in scale."""
    x = _bits_to_x(bits)
    return np.array([np.exp(-2.0 * x), 1e-8 * (np.cos(5.0 * x) + 1.5)])


def test_componentweights_fix_relative_error_of_small_components():
    tol = 1e-6

    # componentweights=None is the explicitly unweighted regime this test is about;
    # it is no longer the default (that is now "auto", covered just below).
    plain, _, _ = crossinterpolate2_array(
        np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=tol, componentweights=None
    )
    weighted, _, _ = crossinterpolate2_array(
        np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=tol, componentweights=[1.0, 1e-8]
    )

    def relerr(att, k):
        return max(
            abs(att(list(x))[k] - _scaledfun(list(x))[k]) / abs(_scaledfun(list(x))[k])
            for x in _allx(LOCALDIMS)
        )

    # the large component is fine either way; the small one is only accurate
    # once the weights put both components on the same scale
    assert relerr(plain, 0) < tol
    assert relerr(weighted, 0) < tol
    assert relerr(weighted, 1) < tol
    assert relerr(plain, 1) > 100 * relerr(weighted, 1)


def test_componentweights_roundtrip_through_components():
    att, _, _ = crossinterpolate2_array(
        np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=1e-8, componentweights=[1.0, 1e-8]
    )
    x = [1, 1, 0, 1, 0, 0, 0, 1]
    arr = att(x)
    for k in range(2):
        assert att.component(k).evaluate(x) == pytest.approx(arr[k], rel=1e-12)
    assert arr == pytest.approx(_scaledfun(x), rel=1e-6)

    with pytest.raises(ValueError):
        ArrayValuedFunction(np.float64, _scaledfun, LOCALDIMS, (2,), componentweights=[1.0, 0.0])
    with pytest.raises(ValueError):
        ArrayValuedFunction(np.float64, _scaledfun, LOCALDIMS, (2,), componentweights=[1.0])


# -- sum, initial pivots, misc ----------------------------------------------


def test_sum_over_x_per_component():
    att, _, _ = crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-12)
    brute = sum(_vecfun(list(x)) for x in _allx(LOCALDIMS))
    assert att.sum() == pytest.approx(brute, rel=1e-8)


def test_initialpivots_in_x_space_and_extended():
    xpivot = [1, 0, 1, 1, 0, 1, 0, 0]
    att, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), initialpivots=[xpivot], tolerance=1e-10
    )
    assert _maxerr(att, _vecfun, LOCALDIMS) < 1e-8

    att2, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), initialpivots=[xpivot + [2]], tolerance=1e-10
    )
    assert _maxerr(att2, _vecfun, LOCALDIMS) < 1e-8

    with pytest.raises(ValueError):
        crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), initialpivots=[[0, 1]])


def test_single_x_site():
    """N=1: an array-valued function of one variable is still a 2-site chain."""
    def f(x):
        return np.array([x[0], x[0] ** 2, 1.0])

    att, _, _ = crossinterpolate2_array(np.float64, f, [6], (3,), tolerance=1e-10)
    for i in range(6):
        assert att([i]) == pytest.approx(f([i]), abs=1e-10)
    assert len(att.component(0)) == 1


def test_rook_pivotsearch():
    att, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10, pivotsearch="rook"
    )
    assert _maxerr(att, _vecfun, LOCALDIMS) < 1e-8


def test_complex_valued():
    def f(bits):
        x = _bits_to_x(bits)
        return np.array([np.exp(1j * x), 1.0 / (1.0 + 1j * x)])

    att, _, _ = crossinterpolate2_array(np.complex128, f, LOCALDIMS, (2,), tolerance=1e-10)
    assert _maxerr(att, f, LOCALDIMS) < 1e-8


def test_maxbonddim_is_respected():
    att, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-12, maxbonddim=4
    )
    assert max(att.tci.linkdims()) <= 4


def test_arraytensortrain_validates_shape():
    att, _, _ = crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-8)
    with pytest.raises(ValueError):
        ArrayTensorTrain(att.tci, (4,))
    with pytest.raises(ValueError):
        ArrayTensorTrain(att.tci, (3,), componentposition="first")


# -- batch evaluation -------------------------------------------------------


def _random_array_f(rng, localdims, valueshape, dtype):
    """A deterministic-per-x random array-valued function (memoized, so repeated
    queries at the same x agree)."""
    table = {}

    def f(x):
        key = tuple(x)
        if key not in table:
            v = rng.standard_normal(valueshape)
            if np.dtype(dtype).kind == "c":
                v = v + 1j * rng.standard_normal(valueshape)
            table[key] = v.astype(dtype)
        return table[key]

    return f


@pytest.mark.parametrize("valueshape", [(4,), (2, 2)])
@pytest.mark.parametrize("componentposition", ["last", "first"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_batchevaluate_matches_generic_path_exactly(valueshape, componentposition, weighted, dtype):
    """ArrayValuedFunction.batchevaluate must agree cell for cell -- *bit* for bit,
    not merely to a tolerance -- with evaluating the adapter once per cell.

    It is exercised over every (nleft, ncenter, nright) split of the extended chain,
    which is what puts the component slot in the left block, a centre site, and the
    right block in turn -- the three cases the index bookkeeping has to distinguish.
    """
    from qutecipy.tensortrain.batcheval import _generic_batchevaluate

    rng = np.random.default_rng(20250816)
    localdims = [2, 3, 2]
    n = len(localdims)
    K = int(np.prod(valueshape))
    weights = np.linspace(0.5, 2.0, K) if weighted else None

    f = _random_array_f(rng, localdims, valueshape, dtype)
    fa = ArrayValuedFunction(
        dtype, f, localdims, valueshape,
        componentposition=componentposition, componentweights=weights,
    )
    ext = fa.extendedlocaldims
    assert len(ext) == n + 1

    seen_slots = set()
    for nl in range(n + 2):
        for ncent in range(n + 2 - nl):
            nr = n + 1 - nl - ncent
            if nr < 0:
                continue
            lefts = list(itertools.product(*[range(d) for d in ext[:nl]])) if nl else [()]
            rights = list(itertools.product(*[range(d) for d in ext[n + 1 - nr:]])) if nr else [()]
            lefts, rights = lefts[:3], rights[:4]

            expected = _generic_batchevaluate(dtype, fa.__call__, ext, lefts, rights, ncent)
            got = fa.batchevaluate(lefts, rights, ncent)
            assert got.shape == expected.shape
            assert got.dtype == np.dtype(dtype)
            assert np.array_equal(got, expected)
            seen_slots.add(fa._component_slot(nl, ncent)[0])

    assert seen_slots == {"left", "center", "right"}


def test_batchevaluate_empty_batch():
    fa = ArrayValuedFunction(np.float64, _vecfun, LOCALDIMS, (3,))
    assert fa.batchevaluate([], [(0,)], 1).shape == (0, 0, 0)
    assert fa.batchevaluate([(0,)], [], 1).shape == (0, 0, 0)


def test_batchevaluate_does_not_add_f_calls():
    """The batch path must not evaluate f more often than the per-cell path: values
    are memoized per x, so both cost exactly one call per distinct x."""
    calls = []

    def counting(bits):
        calls.append(tuple(bits))
        return _vecfun(bits)

    fa = ArrayValuedFunction(np.float64, counting, LOCALDIMS, (3,))
    lefts = [(0, 0), (1, 0), (0, 1)]
    rights = [(0,) * (R - 3) + (k,) for k in range(3)]
    out = fa.batchevaluate(lefts, rights, 1)
    assert out.shape == (3, 2, 3)
    # 3 lefts x 2 centre values x 3 right suffixes, but only one distinct x each
    # since the component index is what varies across the right set.
    assert len(calls) == len(set(calls)) == fa.ncalls
    assert fa.ncalls < out.size
    assert fa.nqueries == out.size


def test_arrayvaluedfunction_is_batch_evaluatable():
    from qutecipy.tensortrain.batcheval import isbatchevaluable

    fa = ArrayValuedFunction(np.float64, _vecfun, LOCALDIMS, (3,))
    assert isbatchevaluable(fa)
    # ...so crossinterpolate2's opt-in check accepts it rather than raising.
    att, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-8, checkbatchevaluatable=True
    )
    assert _maxerr(att, _vecfun, LOCALDIMS) < 1e-6


# -- auto-derived component weights -----------------------------------------


def _relerr(att, f, localdims, k):
    return max(
        abs(att(list(x))[k] - f(list(x))[k]) / abs(f(list(x))[k])
        for x in _allx(localdims)
    )


def test_auto_weights_are_the_default_and_rescue_small_components():
    """The headline: with the default settings, a component 1e-8 times smaller than
    its neighbour is resolved to the requested *relative* tolerance, instead of
    being silently dropped."""
    tol = 1e-6
    auto, _, _ = crossinterpolate2_array(np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=tol)
    plain, _, _ = crossinterpolate2_array(
        np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=tol, componentweights=None
    )

    assert auto.componentweights is not None
    assert _relerr(auto, _scaledfun, LOCALDIMS, 0) < tol
    assert _relerr(auto, _scaledfun, LOCALDIMS, 1) < tol
    # the unweighted run gets the small component badly wrong; auto does not
    assert _relerr(plain, _scaledfun, LOCALDIMS, 1) > 100 * _relerr(auto, _scaledfun, LOCALDIMS, 1)


def test_auto_weights_track_the_component_scales():
    w = crossinterpolate2_array(np.float64, _scaledfun, LOCALDIMS, (2,), tolerance=1e-8)[0].componentweights
    exact = np.zeros(2)
    for x in _allx(LOCALDIMS):
        exact = np.maximum(exact, np.abs(_scaledfun(list(x))))
    # sampled maxima, so <= the true maxima, but the same order of magnitude
    assert np.all(w <= exact * (1 + 1e-12))
    assert np.all(w > 0.1 * exact)


def test_auto_weights_are_harmless_when_components_share_a_scale():
    """Weighting must not cost accuracy (or much rank) when it is not needed."""
    auto, _, _ = crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10)
    plain, _, _ = crossinterpolate2_array(
        np.float64, _vecfun, LOCALDIMS, (3,), tolerance=1e-10, componentweights=None
    )
    assert _maxerr(auto, _vecfun, LOCALDIMS) < 1e-8
    assert auto.rank() <= plain.rank() + 2


def test_estimate_componentweights_guards():
    from qutecipy.arrayvalued import estimate_componentweights

    # a component that is exactly zero everywhere keeps the global scale rather
    # than being handed a near-zero weight
    def withzero(bits):
        return np.array([1.0 + _bits_to_x(bits), 0.0])

    fa = ArrayValuedFunction(np.float64, withzero, LOCALDIMS, (2,))
    w = estimate_componentweights(fa, nsample=20)
    assert w[1] == w.max()
    assert np.all(w > 0)

    # tiny-but-nonzero is floored relative to the largest, bounding amplification
    def tiny(bits):
        return np.array([1.0, 1e-300])

    fa2 = ArrayValuedFunction(np.float64, tiny, LOCALDIMS, (2,))
    w2 = estimate_componentweights(fa2, nsample=5, floor=1e-14)
    assert w2[1] == pytest.approx(1e-14 * w2[0])

    # all-zero f cannot be normalized at all
    fa3 = ArrayValuedFunction(np.float64, lambda bits: np.zeros(2), LOCALDIMS, (2,))
    with pytest.raises(ValueError):
        estimate_componentweights(fa3, nsample=5)


def test_estimate_componentweights_reuses_the_adapter_cache():
    """Sampling must go through the adapter, so TCI2 inherits the cached values."""
    from qutecipy.arrayvalued import estimate_componentweights

    calls = []

    def counting(bits):
        calls.append(tuple(bits))
        return _vecfun(bits)

    fa = ArrayValuedFunction(np.float64, counting, LOCALDIMS, (3,))
    estimate_componentweights(fa, nsample=17)
    assert len(calls) == fa.ncalls == len(fa.cache) == 17
    # re-estimating over the same points costs no further evaluations of f
    estimate_componentweights(fa, nsample=0, xsamples=list(fa.cache))
    assert len(calls) == 17


def test_componentweights_rejects_unknown_string():
    with pytest.raises(ValueError):
        crossinterpolate2_array(np.float64, _vecfun, LOCALDIMS, (3,), componentweights="scaled")


def test_set_componentweights_after_caching():
    fa = ArrayValuedFunction(np.float64, _vecfun, LOCALDIMS, (3,))
    x = [0, 1] * (R // 2)
    raw = fa.values(x).copy()
    fa.set_componentweights([1.0, 2.0, 4.0])
    for k in range(3):
        assert fa(list(x) + [k]) == pytest.approx(raw[k] / [1.0, 2.0, 4.0][k])
    # the cache still holds unweighted values, so weights can be changed again
    fa.set_componentweights(None)
    for k in range(3):
        assert fa(list(x) + [k]) == raw[k]
