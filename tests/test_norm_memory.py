"""norm2 must not materialise a chi**4 transfer matrix.

It used to build ``tensordot(conj(t), t)`` per site, which is ``(chi, chi, chi, chi)``.
That is fine for a compressed train and fatal for one that is not: an uncompressed
operator-times-state product with rank 275 asks for 91 GB, and it OOM-killed a machine
twice before this was fixed. The environment formulation below is the same number in
``chi**2`` memory.
"""
import numpy as np
import pytest

from qutecipy.tensortrain.core import TensorTrain

rng = np.random.default_rng(0)


def _train(chi, n=6, d=2):
    cores = [rng.standard_normal((1, d, chi)) + 1j * rng.standard_normal((1, d, chi))]
    for _ in range(n - 2):
        cores.append(rng.standard_normal((chi, d, chi)) + 1j * rng.standard_normal((chi, d, chi)))
    cores.append(rng.standard_normal((chi, d, 1)) + 1j * rng.standard_normal((chi, d, 1)))
    return TensorTrain(cores)


def test_norm2_matches_the_dense_value():
    import itertools

    tt = _train(chi=3, n=5)
    vec = np.array([tt.evaluate(list(idx)) for idx in itertools.product(*[range(2)] * 5)])
    assert np.isclose(tt.norm2(), float(np.real(np.vdot(vec, vec))), rtol=1e-10)


def test_norm2_is_cheap_at_a_bond_dimension_that_would_not_fit_as_chi4():
    """chi=300 over 6 sites: chi**2 is 0.7 MB per core, chi**4 would be 129 GB."""
    tt = _train(chi=300, n=6)
    value = tt.norm2()
    assert np.isfinite(value) and value > 0
    assert np.isclose(tt.norm(), np.sqrt(value), rtol=1e-12)
