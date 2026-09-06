"""Regression tests for contract_zipup's operand canonicalization.

Zip-up truncates at each cut as it sweeps. That truncation is only meaningful if the
not-yet-contracted right part is orthonormal; otherwise the singular values are weighted
by whatever norm that part carries and the cutoff keeps the wrong components. Before the
fix, contract_zipup did not canonicalize its operands, and a quantics operator applied to
a rank-14 state at maxbonddim=14 came out 2.8e-01 wrong instead of 4.8e-02.
"""
import numpy as np
import pytest

from qutecipy import CachedFunction, DiscretizedGrid, contract, crossinterpolate2
from qutecipy.contraction import _right_canonicalize, contract_zipup
from qutecipy.tensortrain.core import TensorTrain, subtract

R = 14
XMIN, XMAX = -20.0, 20.0


def _grid():
    return DiscretizedGrid.from_resolutions(["x"], [R], lower_bound=(XMIN,),
                                            upper_bound=(XMAX,), includeendpoint=True)


def _tci(fn, tol=1e-12):
    g = _grid()
    q = lambda b: complex(fn(g.quantics_to_origcoord(b)[0]))
    p = [0] * R
    p[0] = 1                                   # centre of the grid, where these live
    ci, _, _ = crossinterpolate2(np.complex128, CachedFunction(np.complex128, q, [2] * R),
                                 [2] * R, [p], tolerance=tol)
    return TensorTrain([np.asarray(ci.sitetensor(n)).reshape(
        np.asarray(ci.sitetensor(n)).shape[0], 2, -1) for n in range(R)])


def _as_mpo(tt):
    out = []
    for t in tt.sitetensors():
        l, d, r = t.shape
        M = np.zeros((l, d, d, r), dtype=np.complex128)
        for b in range(d):
            M[:, b, b, :] = t[:, b, :]
        out.append(M)
    return TensorTrain(out)


def _as_4leg(tt):
    return TensorTrain.reshaped(tt, [(*d, 1) for d in tt.sitedims()])


def _as_3leg(tt):
    return TensorTrain.reshaped(tt, [[int(np.prod(d))] for d in tt.sitedims()])


def _case():
    state = _tci(lambda x: np.exp(-(x**2) / 2))
    op = _as_mpo(_tci(lambda x: np.exp(-3j * np.sin(5 * x))))
    return op, state


def test_right_canonicalize_is_a_pure_gauge():
    """It must not change the represented train, only how the norm is distributed."""
    _, state = _case()
    canon = _right_canonicalize(state)
    idx = np.linspace(0, 2**R - 1, 257).astype(int)
    before = np.array([state.evaluate([[(i >> (R - 1 - n)) & 1] for n in range(R)]) for i in idx])
    after = np.array([canon.evaluate([[(i >> (R - 1 - n)) & 1] for n in range(R)]) for i in idx])
    assert np.allclose(before, after, rtol=0, atol=1e-12 * np.max(np.abs(before)))


def test_right_canonicalize_produces_right_orthogonal_cores():
    _, state = _case()
    cores = _right_canonicalize(state).sitetensors()
    for n, T in enumerate(cores[1:], start=1):
        m = T.reshape(T.shape[0], -1)
        assert np.allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-10), f"site {n}"


def test_zipup_does_not_mutate_its_operands():
    op, state = _case()
    before = [T.copy() for T in state.sitetensors()]
    contract_zipup(op, _as_4leg(state), tolerance=1e-12, maxbonddim=8)
    for a, b in zip(before, state.sitetensors()):
        assert np.array_equal(a, b)


def test_zipup_equals_naive_once_the_zip_bond_suffices():
    """With nothing discarded during the sweep, the final compression is the same optimal
    truncation contract_naive performs, so the two must agree.

    The condition is the zip bond reaching the *uncompressed* product bond, chi_A * chi_B
    -- not the compressed rank of the result, which is much smaller (here 275 vs 20).
    """
    op, state = _case()
    exact = contract(op, state, algorithm="naive", tolerance=1e-14)
    product_bond = max(a * b for a, b in zip(op.linkdims() + [1], state.linkdims() + [1]))
    assert product_bond > exact.rank(), "test case no longer distinguishes the two bounds"
    got = _as_3leg(contract_zipup(op, _as_4leg(state), tolerance=1e-14,
                                  maxbonddim=product_bond, oversample=1))
    assert subtract(got, exact).norm() / exact.norm() < 1e-10


def test_zipup_beats_the_ungauged_result_and_improves_with_oversample():
    """The regression itself.

    Without the canonicalization this case gives 9.9e-01, 9.5e-01, 8.9e-01 at
    oversample 1, 2, 4 -- an answer with no correct digits, and barely responsive to
    oversampling because the truncation is keeping the wrong components in the first
    place. With it: 2.7e-01, 1.4e-01, 4.1e-03.

    Zip-up is a greedy single pass and cannot match the global SVD at a binding bond
    dimension (naive reaches 4.8e-06 here); that is the algorithm, not a defect. What
    the fix restores is that the truncation is made on the right singular values, so
    the error is meaningful and oversampling actually buys something.
    """
    op, state = _case()
    exact = contract(op, state, algorithm="naive", tolerance=1e-14)
    maxdim = 8
    errs = []
    for oversample in (1, 2, 4):
        got = _as_3leg(contract_zipup(op, _as_4leg(state), tolerance=1e-12,
                                      maxbonddim=maxdim, oversample=oversample))
        assert got.rank() <= maxdim
        errs.append(subtract(got, exact).norm() / exact.norm())
    assert errs[0] < 0.5, f"ungauged behaviour (~0.99) appears to be back: {errs[0]:.2e}"
    assert errs[2] < errs[0] / 10, f"oversampling is not helping: {errs}"
