#!/usr/bin/env python3
"""Array-valued TCI demo: interpolate f(x) -> ndarray with pivots shared by all
components, and extract one tensor train per component.

Run with:  PYTHONPATH=. python3 examples/arrayvalued_demo.py

Also illustrates the cost model: at any cut, max_k chi_k <= chi_joint <= sum_k chi_k.
The "components share structure" assumption is exactly the statement that chi_joint is
near the lower bound -- so the demo runs one function family where that holds and one
where it doesn't, and prints both.

See CLAUDE.md's "Extensions beyond the Julia reference" section, and
qutecipy/arrayvalued.py's module docstring, for the design rationale.
"""
import time

import numpy as np

from qutecipy.arrayvalued import crossinterpolate2_array
from qutecipy.tci2 import crossinterpolate2

R = 20                     # quantics bits -> 2^20 grid points
LOCALDIMS = [2] * R
K = 4
TOL = 1e-8


def _x(bits):
    idx = 0
    for b in bits:
        idx = idx * 2 + b
    return idx / 2 ** R


def f_shared(bits):
    """All four components are one sharp peak times a smooth, low-rank envelope:
    the regime this design targets."""
    x = _x(bits)
    peak = 1.0 / (1.0 + (x - 0.5) ** 2 * 400.0)
    return peak * np.array([1.0, x, np.cos(x), np.exp(-x)])


def f_unrelated(bits):
    """Four peaks at different places: the components do *not* share structure."""
    x = _x(bits)
    return np.array([1.0 / (1.0 + (x - c) ** 2 * 400.0) for c in (0.2, 0.4, 0.6, 0.8)])


# x = 0.5. The all-zero default pivot sits at x = 0, where f_shared's second
# component vanishes -- fine for the joint run (it normalizes against the largest
# component, so a component vanishing at the initial pivot is harmless), but it makes
# the *independent* per-component run raise "maxsamplevalue is zero!".
INITIALPIVOT = [1] + [0] * (R - 1)


def run(name, f):
    t0 = time.process_time()
    att, _, errors = crossinterpolate2_array(
        np.float64, f, LOCALDIMS, (K,), initialpivots=[INITIALPIVOT], tolerance=TOL
    )
    t_joint = time.process_time() - t0

    ncalls, ranks_alone = 0, []
    t0 = time.process_time()
    for k in range(K):
        counter = [0]

        def fk(bits, k=k, counter=counter):
            counter[0] += 1
            return f(bits)[k]

        tci, _, _ = crossinterpolate2(np.float64, fk, LOCALDIMS, [INITIALPIVOT], tolerance=TOL)
        ranks_alone.append(tci.rank())
        ncalls += counter[0]
    t_indep = time.process_time() - t0

    rng = np.random.default_rng(0)
    err = max(
        float(np.max(np.abs(att(b) - f(b))))
        for b in ([list(rng.integers(0, 2, size=R)) for _ in range(200)])
    )

    print(f"\n=== {name} ===")
    print(f"  joint (shared pivots): rank={att.rank():3d}  f-calls={att.func.ncalls:6d}  "
          f"{t_joint:5.2f}s   (TCI made {att.func.nqueries} scalar queries)")
    print(f"  {K} independent runs:   ranks={ranks_alone} (max {max(ranks_alone)}, "
          f"sum {sum(ranks_alone)})  f-calls={ncalls:6d}  {t_indep:5.2f}s")
    print(f"  max |att(x) - f(x)| over 200 random x: {err:.2e}   (tolerance {TOL:.0e})")
    return att


def main():
    att = run("components share structure", f_shared)
    run("components unrelated", f_unrelated)

    # one tensor train per component, all sharing the same pivots
    comps = att.components()
    shared = all(
        np.array_equal(comps[k].sitetensor(b), comps[0].sitetensor(b))
        for k in range(K) for b in range(R - 1)
    )
    bits = [0, 1] * (R // 2)
    print(f"\nper-component tensor trains (first case): {len(comps)}, "
          f"ranks={[tt.rank() for tt in comps]}")
    print(f"  site tensors 0..{R - 2} identical across components: {shared}")
    print("  component TT minus array eval:",
          [f"{comps[k].evaluate(bits) - att(bits)[k]:.1e}" for k in range(K)])
    print("  linear-cost sum over all 2^R points, per component:",
          np.array2string(att.sum(), precision=4))


if __name__ == "__main__":
    main()
