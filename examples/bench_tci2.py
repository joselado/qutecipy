#!/usr/bin/env python3
"""crossinterpolate2 benchmark + a machine-parseable result dump, for cross-checking
against bench_tci2.jl (see examples/README.md and examples/compare.py).

Domain note: f uses (x+1) for each 0-based coordinate x, so this is the *same*
function over the *same* domain (values 1..10) as bench_tci2.jl's 1-based v -- results
(rank, sum) should match to near machine precision, even though the two
implementations' pivot searches use unrelated random-number streams and may not visit
the same pivots along the way.
"""
import time

import numpy as np

from qtcipy.tci2 import crossinterpolate2


def bench_and_report(name, f):
    tci, ranks, errors = f()  # warmup
    t0 = time.process_time()
    tci, ranks, errors = f()
    t1 = time.process_time()
    elapsed = t1 - t0
    s = tci.sum()
    print(f"[{name}] rank={tci.rank()} sum={s} time_s={elapsed}")


def f8(v):
    return 1.0 / (1.0 + sum((x + 1) ** 2 for x in v))


localdims8 = [10] * 8
bench_and_report("8D_full", lambda: crossinterpolate2(np.float64, f8, localdims8, tolerance=1e-8))


def f20(v):
    return 1.0 / (1.0 + sum((x + 1) ** 2 for x in v))


localdims20 = [2] * 20
bench_and_report(
    "20D_rook",
    lambda: crossinterpolate2(np.float64, f20, localdims20, tolerance=1e-8, pivotsearch="rook"),
)
