# Julia vs. Python comparison examples

Runnable scripts that exercise the Julia reference (`reference/TensorCrossInterpolation.jl`)
and the Python port (`qtcipy`) on the *same* problems, for both correctness
cross-checking and performance comparison.

## Quick start

```bash
# from the repo root
PYTHONPATH=. python3 examples/compare.py
```

This runs `bench_tci2.jl` (Julia) and `bench_tci2.py` (Python) on two problems --
an 8D Lorentzian with full pivot search, and a 20D binary (quantics-like) case with
rook pivot search -- and prints a side-by-side table of rank, `sum(tci)` (the
function's TT-summed value over the entire exponentially-large index grid, computed
in linear cost via the TT-sum trick), and timing.

Requires Julia with `TensorCrossInterpolation.jl` instantiated at
`reference/TensorCrossInterpolation.jl` (one-time setup, if not already done):

```bash
cd reference/TensorCrossInterpolation.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

If Julia isn't on `PATH`, `compare.py` still runs the Python side and says so.

## What's here

- `bench_matrix.jl` / `bench_matrix.py` -- raw matrix-layer (`rrLU`) benchmark, no
  correctness check (that's covered by the test suite); just timing. Run each
  directly (`julia --project=reference/TensorCrossInterpolation.jl examples/bench_matrix.jl`,
  `PYTHONPATH=. python3 examples/bench_matrix.py`).
- `bench_tci2.jl` / `bench_tci2.py` -- `crossinterpolate2` on the two example
  problems, printing a machine-parseable `[case] rank=... sum=... time_s=...` line
  per case. Domains are deliberately aligned (Python's 0-based coordinate `x` uses
  `x+1` inside `f` to match Julia's 1-based `v` directly) so the two `sum` values are
  a genuine correctness check, not just "both ran without crashing."
- `compare.py` -- orchestrates both languages and prints the comparison table.

## Interpreting the output

- **rank** and **sum** should match closely between languages (`sum` to roughly the
  requested tolerance, e.g. `1e-8`, sometimes much closer). They won't be bit-identical
  because the two pivot searches (rook pivoting, the randomized global-pivot finder)
  draw from unrelated random-number streams and can settle on different but equally
  valid pivot sets -- what should agree is that both converged to the *same underlying
  function* to the requested accuracy.
- **timing**: Python's `time.process_time()` (CPU time, immune to other processes on a
  shared machine) vs. Julia's `time_ns()` around a warmed-up (post-JIT-compilation)
  call. Both scripts do one untimed warmup call first. For a noise-free read, run
  `compare.py` a few times and look at the smallest ratio you see -- single-shot timing
  on a loaded machine can vary 2x+ run to run.
- As of this writing, typical ratios are ~5x (8D full pivot) and ~5-9x (20D rook pivot)
  Python-slower-than-Julia for `crossinterpolate2`, and near parity (~1.0x) for the raw
  `rrLU` matrix layer alone (numba-accelerated). See `CLAUDE.md`'s "Performance" section
  for the profiling story behind these numbers and what's been done about them
  (`qtcipy[fast]` numba acceleration, `CachedFunction` for expensive/redundant `f`).

## Using `CachedFunction` for a bigger win

The single biggest lever for an expensive or redundantly-evaluated `f` is
`CachedFunction` (TCI's pivot search re-visits the same points many times over a
sweep -- measured 5-15x redundant calls on these example problems). Try it:

```python
from qtcipy.tensortrain.cachedfunction import CachedFunction
from qtcipy.tci2 import crossinterpolate2

f_cached = CachedFunction(np.float64, f, localdims)
tci, ranks, errors = crossinterpolate2(np.float64, f_cached, localdims, tolerance=1e-8)
```

This isn't on by default (matching Julia's own documented default) since for a cheap
`f` a dict lookup can cost as much as just calling `f` again -- it pays off once `f`
is nontrivial.
