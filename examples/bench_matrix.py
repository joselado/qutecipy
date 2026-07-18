#!/usr/bin/env python3
"""Matrix-layer (rrLU) benchmark -- Python counterpart of bench_matrix.jl.
Run from the repo root with `python3 examples/bench_matrix.py` (needs qtcipy on
PYTHONPATH, e.g. `PYTHONPATH=. python3 examples/bench_matrix.py`).
"""
import time

import numpy as np

from qtcipy.matrix.rrlu import HAVE_NUMBA, rrlu


def bench(name, f, nrep=7):
    f()  # warmup (also triggers numba JIT compilation on first call, if available)
    times = []
    for _ in range(nrep):
        t0 = time.process_time()
        f()
        t1 = time.process_time()
        times.append(t1 - t0)
    print(f"{name}: {min(times) * 1000:.3f} ms CPU (min of {nrep})")


print(f"numba acceleration: {'ON' if HAVE_NUMBA else 'OFF (pip install qtcipy[fast])'}")

rng = np.random.default_rng(42)
A200 = rng.random((200, 200))
bench("rrLU 200x200 full rank", lambda: rrlu(A200.copy()))

A500 = rng.random((500, 100))
bench("rrLU 500x100 maxrank=50", lambda: rrlu(A500.copy(), maxrank=50))
