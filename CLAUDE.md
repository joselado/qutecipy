# qutecipy — Python port of TensorCrossInterpolation.jl

## Goal

Produce a **full, faithful Python port** of [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl)
(package `qutecipy`), implementing the tensor cross interpolation (TCI) algorithm for
efficient interpolation of multi-index tensors and multivariate functions, plus the
supporting tensor-train (TT/MPS) infrastructure it depends on.

The upstream Julia source is vendored for reference at `reference/TensorCrossInterpolation.jl`
(shallow clone, MIT license, commit `ae4329a`). Treat it as the spec: algorithm behavior,
default parameter values, and public API shape should match it unless there's a good
reason (documented below) to diverge. Do not vendor its `test/` fixtures verbatim into
the Python package, but do port the test *cases* (see Testing strategy).

`reference/QuadGK.jl` (shallow clone, MIT license) is also vendored, for the one real
transitive numerical dependency (`src/gausskronrod.jl`) that needs its own translation —
see "External dependencies" below.

`reference/QuanticsGrids.jl` (shallow clone, MIT license) is also vendored — a second,
independent library in scope for this port (`qutecipy.quantics`), see "QuanticsGrids"
under External dependencies below.

## Implementation status

The core port is **done and numerically cross-validated against the actual Julia
reference implementation** (both packages instantiated and run locally via
`julia --project=.` in `reference/`; not just read for reference — their live output
was diffed against the Python port's output during development).

**Implemented** (`qutecipy/`, ~4300 lines, `tests/`, 100 passing tests) — the full
scope of this porting plan, including everything originally listed as "lower
priority":
- `util.py`, `indexset.py` — full.
- `matrix/{base,aca,rrlu,luci}.py` — `MatrixCI`, `MatrixACA`, `rrLU`/`arrlu` (full +
  rook pivoting), `MatrixLUCI` — full, including the rook-pivoting adaptive search.
- `tensortrain/{base,core,cache,cachedfunction,batcheval}.py` — `AbstractTensorTrain`,
  `TensorTrain` (+ `compress`, `add`/`subtract`, `TensorTrainFit`), `TTCache`
  (including batched evaluation), `CachedFunction`, `BatchEvaluator` — full.
- `tci1.py` — `TensorCI1`, `crossinterpolate1`, including `add_global_pivot` — full.
- `tci2.py`, `globalpivot.py` — `TensorCI2`, `optimize`/`crossinterpolate2` (full and
  rook pivot search), `DefaultGlobalPivotFinder`/`AbstractGlobalPivotFinder`,
  `_floatingzone`/`estimate_true_error`/`search_global_pivots` — full.
- `gausskronrod.py` — the unit-weight/hollow-tridiagonal `kronrod()` subset — full,
  verified bit-for-bit against `QuadGK.kronrod` for several orders including the
  library's own hardcoded n=7 constants.
- `integration.py` — `integrate()` — full.
- `conversion.py` — `TensorCI1`⇄`TensorCI2` interconversion, plus
  `sweep1site_get_indices`/`tci2_from_tensortrain` (rebuilds a canonical
  pivot-based TCI2 from an arbitrary dense `TensorTrain`) — full.
- `contraction.py` — `Contraction` (lazy MPO×MPO `BatchEvaluator`, including batched
  evaluation with projectors), `contract_naive`, `contract_zipup`, `contract_TCI`,
  and the `contract()` dispatcher (3-leg⇄4-leg promotion included) — full.
- `quantics/{grid,discretized}.py` — `InherentDiscreteGrid`, `DiscretizedGrid`,
  `quantics_function` — full (all three unfolding schemes + custom index tables),
  verified against `QuanticsGrids.jl`'s live output including float edge cases.

**Remaining, low priority**:
- `TTCache`/`ThreadedBatchEvaluator` real threading (currently a documented sequential
  fallback, per the original plan below — revisit only if profiling shows it matters).
- A few Julia idioms were deliberately simplified rather than literally replicated
  (documented inline where it matters): `CachedFunction`'s cache key is
  `tuple(indexset)` rather than a mixed-radix integer encoding; `submatrixargmax`
  supports only the `(A, startindex)` form actually used internally, not the full
  Julia multiple-dispatch surface; `QuanticsGrids`'s base-2 bit-shift fast path and
  `kwargs`-based convenience API were dropped (see the "QuanticsGrids" section below).

**Bugs caught only by cross-checking against live Julia output** (worth knowing about
if extending this code — these would *not* have been caught by static review or
Python-only unit tests, since each produced internally-consistent-looking but wrong
results): a swap-semantics inversion in `gausskronrod.py`'s Laurie-algorithm port
(`normalize2!`'s swap was applied both inside the helper and at the call site,
cancelling out); two off-by-one slice bounds in the same file's eigensolver setup;
an index-offset asymmetry between `PiIset`/`PiJset` in `tci1.py`'s global-pivot
insertion; and, in `contraction.py`, a fused-index ordering convention
(`_fuse_idx`'s "i fast") that a plain C-order numpy reshape gets backwards relative
to Julia's native column-major reshape — hit twice, once in `batchevaluate`'s
center-site accumulation and once in `contract_TCI`'s final un-fuse step.

No test fixtures were copy-pasted from Julia's `test/*.jl` output — every ported test
either checks a closed-form/analytic result, or was validated once against a live
Julia run during development and then re-expressed as a self-contained Python
assertion (Julia is not a runtime dependency of the test suite).

## Performance

Benchmarked against the live Julia reference (same machine, JIT-warmed Julia, CPU-time
measurement for Python since this runs on a shared/contended single-core VM where
wall-clock is noisy). Two profiled cases, `crossinterpolate2` with `pivotsearch="full"`
(8D Lorentzian, localdims=10) and with `pivotsearch="rook"` (20D binary):

| | Julia | Python (baseline) | Python (current) |
|---|---|---|---|
| 8D full | 41 ms | 447 ms (10.9×) | ~245 ms (~6×) with `CachedFunction` |
| 20D rook | 10.3 ms | 112 ms (10.9×) | ~89 ms (~8.6×) with `CachedFunction` |

Profiling (cProfile) showed the gap is **not** primarily algorithmic — it's Python
interpreter overhead multiplied over millions of small operations, in two flavors:
1. **Library-side overhead in the generic batch-evaluation fallback**
   (`qutecipy/tensortrain/batcheval.py::_generic_batchevaluate`): the original
   `result[i,c,j] = f(...)` pattern's per-cell numpy `__setitem__` cost *more* than
   calling `f` itself. Fixed by building the whole batch as a flat list comprehension
   and casting once (`np.array(flat, dtype).reshape(...)`) — same iteration order, same
   values, ~4× less library overhead in isolation (confirmed via a controlled
   microbenchmark with a trivial `f`, isolated from system noise). Same fix applied to
   `CachedFunction._batcheval_default` (`qutecipy/tensortrain/cachedfunction.py`).
2. **Many small bookkeeping ops** in `sweep2site` (`copy.deepcopy(Iset)` → cheap
   per-site shallow copy, since the tuples inside are immutable and sites are always
   replaced wholesale, never mutated in place), `rrLU._optimize`'s pivot-search argmax
   (dropped a redundant `**2` — `argmax(|x|²) == argmax(|x|)`, and `self.error` re-reads
   the raw value separately), and `SubMatrix.__call__` (precompute row/col `list()`
   conversions once instead of per inner-loop iteration). All in `qutecipy/tci2.py` and
   `qutecipy/matrix/rrlu.py`.

**The single biggest lever, by far, is `CachedFunction`** (already implemented, see
"Algorithmic layers" below): TCI's pivot search — full and especially rook — re-visits
the same multi-index many times over a sweep (measured 5.7× redundant calls on the 8D
case, 14.7× on the 20D case). Julia doesn't cache by default either (its own docs say
so explicitly) and pays the same redundancy, just at JIT-compiled speed. In Python that
redundancy is worth eliminating whenever `f` costs more than a dict-of-tuple lookup —
wrap it: `crossinterpolate2(dtype, CachedFunction(dtype, f, localdims), localdims,
...)`. Both `crossinterpolate1` and `crossinterpolate2`'s docstrings point this out.

**`numba.njit` on `rrLU._optimize`'s dense pivoting kernel** (the argmax-scan + rank-1
update + row/col swaps) — implemented in `qutecipy/matrix/_numba_kernels.py`, a soft
dependency (`pip install qutecipy[fast]`; falls back to the pure-Python loop, which
remains the tested reference, if numba isn't installed or for dtypes/layouts the numba
path doesn't handle — currently float64/complex128, C-contiguous only). The kernel is a
direct, tie-break-faithful port (explicit row-major argmax scan with strict `>`,
matching `np.argmax`'s first-occurrence semantics exactly) — the full test suite passes
identically with numba installed or not, confirming bit-for-bit matching pivot
selection. Numba can only accelerate the *pure-numeric* part of rrLU — it cannot touch
anything that calls the user's arbitrary Python `f` (`_generic_batchevaluate`,
`arrlu`'s lazy submatrix evaluation), so this doesn't help those loops.

**Combined final numbers** (all fixes above + numba + `CachedFunction`, CPU-time
measurement on the same shared/contended single-core VM as the original benchmark):

| | Julia | Python, original | Python, now |
|---|---|---|---|
| `rrLU` 200×200 | 4.1 ms | 8.2 ms (2.0×) | 4.3 ms (**1.04×**) |
| 8D full pivot | 41 ms | 447 ms (10.9×) | 216 ms (**5.3×**) |
| 20D rook pivot | 10.3 ms | 112 ms (10.9×) | 51 ms (**5.0×**) |

The raw matrix layer is now at effective parity with Julia. `crossinterpolate2` is
roughly half the original gap in both profiled cases. See `examples/` for the runnable
benchmark/comparison scripts (both Julia and Python versions, plus a correctness
cross-check) this table was generated from.

Not pursued: vectorized-`BatchEvaluator`/numba-`f` as a *documented user pattern* (the
only way to close the remaining gap for a genuinely expensive black-box `f` — not
automatable, since the library can't compile a user's arbitrary Python callback for
them; see `crossinterpolate2`'s docstring for the `CachedFunction` guidance that already
covers the redundant-call-count part of this).

## Reference source map

`reference/TensorCrossInterpolation.jl/src/`, in dependency order (mirrors the
`include(...)` order in `src/TensorCrossInterpolation.jl`):

| Layer | Julia files | Purpose |
|---|---|---|
| Generic utils | `util.jl`, `sweepstrategies.jl` | pivot-init helper, misc array helpers, sweep-direction logic |
| Matrix CI/pivoting | `abstractmatrixci.jl`, `matrixci.jl`, `matrixaca.jl`, `matrixlu.jl`, `matrixluci.jl` | the numerical core: rank-revealing decompositions of a single matrix |
| Index bookkeeping | `indexset.jl` | bijective index↔position container used by TCI1 |
| TT data structure | `abstracttensortrain.jl`, `tensortrain.jl` | abstract TT interface + concrete dense `TensorTrain` |
| Caching/batching | `cachedtensortrain.jl`, `batcheval.jl`, `cachedfunction.jl` | memoized evaluation, vectorized batch evaluation over many pivots at once |
| Core algorithms | `tensorci1.jl`, `tensorci2.jl`, `globalpivotfinder.jl`, `globalsearch.jl` | the two TCI variants and global-pivot search |
| Glue | `conversion.jl`, `integration.jl`, `contraction.jl` | TCI1⇄TCI2 conversion, TT-based quadrature, MPO×MPO contraction |

`ext/TCIITensorConversion/` (ITensors.jl interop) is **out of scope** — no Python
equivalent of ITensors is being targeted; skip it.

## External dependencies — audit and translation decisions

`TensorCrossInterpolation.jl`'s own dependency footprint (`Project.toml`) plus what
its test suite pulls in, and what each one means for the Python port:

| Julia package | Where used | Decision |
|---|---|---|
| `LinearAlgebra` (stdlib) | throughout | → numpy/scipy; no separate package, already covered by the idiom-translation notes above. |
| `Random` (stdlib) | rook pivoting, global pivot search | → Python `random`/`numpy.random`. No translation needed. |
| `EllipsisNotation` | `..` splicing in `batcheval.jl`/`cachedfunction.jl`/`tensorci1.jl`/`tensorci2.jl` | Not a real library — it's a syntax macro. No package to port; already covered as an idiom (build explicit index tuples in Python). |
| `BitIntegers` | `CachedFunction`'s mixed-radix integer cache key (`cachedfunction.jl`), needed in Julia because fixed-width `UInt128`/`UInt256`/... can overflow for many-site problems | **Drop.** Already decided in this doc to key the Python cache by `tuple(indexset)` instead of an encoded integer. Bonus: even if we wanted to keep the integer-encoding scheme, Python's native `int` is arbitrary-precision, so the overflow-avoidance machinery `BitIntegers` exists for in Julia is simply moot in Python — no equivalent package needed either way. |
| `QuadGK` | **only** `QuadGK.kronrod(GKorder÷2, -1, 1)` in `integration.jl`, to get Gauss–Kronrod nodes/weights for the unit weight function on a symmetric interval | **Port the specific algorithm needed, not the whole package.** See below — this is the one real numerical dependency that needs actual translation work. |
| `ITensors`, `ITensorMPS` (weakdeps, only power the `TCIITensorConversion` extension) | not used by core `src/` at all | **Out of scope**, as already noted — no Python ITensors ecosystem target. |
| `Aqua`, `JET` (test extras) | ambiguity/type-stability static analysis, Julia-specific | No Python equivalent needed; `ruff`/`mypy` in CI cover the rough intent if desired, but this isn't a porting task. |
| `Optim` (test extra) | `test/test_tensortrain.jl` only — fits a `TensorTrainFit` via `LBFGS()` | The library itself is optimizer-agnostic (`TensorTrainFit`/`flatten`/`to_tensors` just expose a loss function; the user supplies the optimizer). Python equivalent for the **test only**: `scipy.optimize.minimize(method="L-BFGS-B")`. Not a runtime dependency of `qutecipy` itself, matching Julia's design. |
| `Zygote` (test extra) | same test, supplies the gradient via autodiff | No autodiff dependency planned for `qutecipy` (see "NumPy-only" decision below). For the ported test, use a finite-difference gradient (`scipy.optimize.minimize` can estimate this itself when `jac` is omitted) rather than pulling in `jax`/`autograd` — good enough to exercise `TensorTrainFit` correctness without adding a new dependency. |
| `QuanticsGrids` (test extra, separate `tensor4all/QuanticsGrids.jl` repo) | `test/test_tensorci2.jl`, `test/test_globalsearch.jl`, `test/test_cachedtensortrain.jl` — builds realistic quantics-representation test functions (`DiscretizedGrid`, `quantics_to_origcoord`, `origcoord_to_quantics`) | **Out of core scope, but flagged** — see below. |

### QuadGK — porting the Gauss–Kronrod node/weight algorithm

Cloned for reference at `reference/QuadGK.jl` (MIT license, same as upstream TCI).
`integration.jl` only calls the simplest case: `kronrod(n, -1, 1)` for the **unit
weight function on a symmetric interval**, i.e. the "hollow symmetric tridiagonal"
branch of `src/gausskronrod.jl`, not the fully general arbitrary-Jacobi-matrix
version. That branch is a self-contained ~150-line numerical algorithm (Laurie 1997
+ Golub–Welsch, per the file's own doc comments):

1. Build the Jacobi matrix coefficients `b[j] = j²/(4j²−1)` for the Legendre-type
   recurrence (`HollowSymTridiagonal`).
2. Run Laurie's `_kronrodjacobi` recursion to extend that Jacobi matrix to the
   `2n+1`-point Kronrod–Jacobi matrix (pure arithmetic recursion, no eigensolver
   needed for this step).
3. Find its eigenvalues via Newton iteration (`eignewt`/`eigpolyrat`), seeded from
   `scipy.linalg.eigh_tridiagonal` (equivalent of the Julia code's own
   `eigvals(SymTridiagonal{Float64}(H))` seed) — a numpy/scipy port is
   straightforward since `eigh_tridiagonal` exists exactly for this.
4. Get each eigenvalue's first eigenvector component (`eigvec1!`, closed-form
   3-term-recurrence back-substitution — no solver needed, just a loop) to obtain
   the quadrature weight.

**Decision: port this subset directly into `qutecipy/gausskronrod.py`** (numpy +
`scipy.linalg.eigh_tridiagonal`), rather than taking a dependency on `quadpy` (its
modern releases carry a non-free "Tidelift" license — avoid) or hand-maintaining a
table of hardcoded node/weight constants (works for the default `GKorder=15` but
breaks the general `GKorder` parameter `integrate()` exposes). Credit Laurie (1997)
and QuadGK.jl (Steven G. Johnson et al., MIT) in a header comment, matching the
Julia file's own citation. Scope is narrow — only the unit-weight/hollow-tridiagonal
path, not `gauss()`'s general-Jacobi-matrix API surface, not `BigFloat`/arbitrary
precision, not the `@generated`-function result-caching trick (a plain Python
`functools.lru_cache` on `(n,)` covers that).

### QuanticsGrids — in scope: `qutecipy.quantics`

`QuanticsGrids.jl` is a separate tensor4all package (grid ↔ quantics-bitstring
coordinate mapping) that `QuanticsTCI.jl` builds on top of `TensorCrossInterpolation.jl`
for exponentially efficient function interpolation with scale separation. It is
**not** a dependency of the core TCI algorithm itself, only of a few of its *tests* —
but per user decision, this port covers it too, as a second, independent top-level
subpackage (`qutecipy.quantics`), matching the `qutecipy` name's implication of covering
the full quantics-TCI stack, not just base cross-interpolation.

Cloned for reference at `reference/QuanticsGrids.jl` (MIT license, same authorship as
upstream TCI). Small and self-contained: **only 2 real source files**
(`src/grid.jl` 493 lines, `src/grid_discretized.jl` 549 lines), **zero real runtime
dependencies** (`Project.toml` lists only stdlib `LinearAlgebra`, and it's unused in
these two files), pure integer/float bookkeeping — no linear algebra needed. This
makes it one of the easiest, most independent pieces to port, and it can be done in
parallel with (or before) the TCI numerical core since neither depends on the other.

**Conceptual model**: a "quantics representation" encodes a grid index `1:base^R`
as `R` base-`b` digits (MSB-first, i.e. `bitnumber=1` is the most significant digit),
one digit — or several fused digits, depending on the *unfolding scheme* — per
tensor-train site. This is what lets a TT/TCI achieve exponential compression for
functions with scale separation: nearby length scales end up as nearby TT sites.

**Two concrete types** (both parametrized by dimensionality `D` in Julia; in Python,
`D = len(...)` derived at runtime, no separate type parameter needed):
- **`InherentDiscreteGrid`** (`grid.jl`): the foundational integer-grid ↔ quantics
  engine. Fields: `Rs` (quantics resolution per dim), `origin`/`step` (integer affine
  map grididx→coordinate), `variablenames`, `base` (radix per dim, can differ per
  dimension — "mixed bases" are supported and tested), and the derived bookkeeping
  that does the real work: `indextable` (`indextable[site] = [(variable, bitnumber),
  ...]`, the site "unfolding" structure — the central data structure), its inverse
  `lookup_table`, and per-site mixed-radix `site_radices`/`site_placevalues` (needed
  because one TT site can fuse digits from several variables, possibly with
  different bases, packed like a mixed-radix number).
- **`DiscretizedGrid`** (`grid_discretized.jl`): a `Float64` wrapper *composing* an
  internal `InherentDiscreteGrid` (always `origin=1, step=1`) plus `lower_bound`/
  `upper_bound`/`includeendpoint` per dimension for the float affine map. All
  quantics↔grididx logic delegates to the wrapped discrete grid; only
  `grididx_to_origcoord`/`origcoord_to_grididx` are reimplemented for floats.

**Unfolding schemes** (`unfoldingscheme` kwarg, default `:fused`) control how each
dimension's digit stream is laid into TT sites — `:interleaved` (one dimension's one
digit per site, cycling through dimensions per bit position), `:fused` (one site per
bit position, holding that bit-position's digit from *every* dimension at once, in
**reverse dimension order** — an exact ordering convention that must be replicated
bit-for-bit), `:grouped` (all of dimension 1's digits, then all of dimension 2's,
...), or a fully custom `indextable` passed directly. Preserve all four faithfully —
they're heavily cross-tested for exact equality against hand-computed tables.

**Public API to port** (`qutecipy/quantics/`): `quantics_to_grididx`,
`grididx_to_quantics`, `grididx_to_origcoord`, `origcoord_to_grididx`,
`origcoord_to_quantics`, `quantics_to_origcoord` (the last two are trivial
compositions of the first four), plus `quanticsfunction` (wraps a coordinate-space
function into quantics-space — the key adapter TCI actually calls),
`localdimensions`/`sitedim` (site physical dims, i.e. what TCI needs for
`localdims`), and the accessor family (`grid_Rs`, `grid_base(s)`, `grid_step`,
`grid_origin`, `grid_min`/`grid_max`, `lower_bound`/`upper_bound`, `grid_origcoords`).

**Numerical details to preserve exactly** (see the float edge-case test suite,
`test/origcoord_floating_point_tests.jl` — extreme ranges, tiny steps down to
`2^-58`, non-representable steps like `1/3`, catastrophic-cancellation regimes):
use `float64` throughout; keep the same arithmetic operation order as the Julia
source (`lower_bound + (idx-1)*step`, not an algebraically-equivalent reordering) so
round-trip (`grididx → origcoord → grididx`) exact-reproducibility tests still pass;
`origcoord_to_grididx` **rounds to nearest** (Python's `round()`/`np.round` banker's-
rounding matches Julia's default `round`, but verify against the ported tests rather
than assuming); bounds-check against the closed interval `[lower_bound, upper_bound]`
even though the grid itself is half-open; clamp the rounded index into `[1,
base^R]` so float roundoff at a boundary can't produce an out-of-range index; the
`includeendpoint=True` case rewrites the *stored* `upper_bound` at construction time
so the true last grid point lands exactly on the user's requested endpoint (a
`linspace(..., endpoint=True)`-style trick) — port this construction-time rewrite,
not just the final step-size formula. `Rs[d]=0` (a degenerate single-point
dimension) is a valid, tested edge case; `step[d]=0` on `InherentDiscreteGrid` is not
(raises).

**Julia idioms needing deliberate translation here**:
- Julia's overflow guard (`base^R <= typemax(Int)`, i.e. rejects `R=63` for base 2 on
  64-bit `Int`) is moot for Python's arbitrary-precision `int` — decide whether to
  drop it silently (recommended: drop; document the deviation) or preserve it for
  strict behavioral parity with the Julia test suite.
- Julia `Symbol` variable names (including auto-generated `Symbol(1)`, `Symbol(2)`,
  ...) back a `**kwargs`-style convenience API (`origcoord_to_grididx(g; x=.., y=..)`).
  Plain integers aren't valid Python kwarg names, so auto-generated names need a
  different Python convention (e.g. `"x0"`, `"x1"`, ...) — decide this up front since
  it's a real (not cosmetic) API mismatch, not just a naming idiom.
- Scalar-vs-tuple return polymorphism (`D=1` returns a bare `int`/`float`, `D>1`
  returns a tuple) is tested explicitly in Julia. Recommend the Python port
  **always returns a tuple** (more consistent/NumPy-friendly) as a deliberate,
  documented API deviation, rather than replicating the type-branching.
- Multiple-dispatch constructor overloads (by-resolution vs. by-indextable vs.
  positional-`D` forms) → a small number of Python classmethod factories
  (`InherentDiscreteGrid.from_resolutions(...)`, `.from_indextable(...)`) rather than
  one constructor with type-sniffing branches.
- The `all(base .== 2)` fast bit-shift path (`_quantics_to_grididx_base2` etc.) is a
  Julia performance idiom; a single general mixed-radix implementation (using
  `divmod`) is enough for the Python port — don't bother porting the base-2
  special-case unless profiling later shows it matters.

Port `qutecipy/quantics/` fully (both grid types, all four conversion directions, all
three unfolding schemes plus custom index tables) and translate its test suite
(`test/grid_tests.jl`, `discretizedgrid_misc_tests.jl`,
`inherentdiscretegrid_tests.jl`, `quantics_tests.jl`, `origcoord_tests.jl`,
`origcoord_floating_point_tests.jl`, `utilities_tests.jl`) into `tests/test_quantics_*.py`
— the float edge-case tests especially, since they're the ones most likely to expose
a subtle Python/NumPy floating-point translation bug. `QuanticsTCI.jl` itself
(the layer that actually *combines* `QuanticsGrids` + `TensorCrossInterpolation` into
one high-level "interpolate a scale-separated function" API) remains out of scope for
this task — `qutecipy.quantics` and the core `qutecipy` TCI port are both ported as
independent, complete libraries; wiring them together end-to-end (mirroring
`QuanticsTCI.jl`) can be a natural follow-up once both are solid.

## Target Python package layout (proposed)

```
qutecipy/
  __init__.py            # public API: crossinterpolate1, crossinterpolate2, TensorTrain, ...
  util.py                 # util.jl + sweepstrategies.jl
  indexset.py             # indexset.jl
  matrix/
    base.py               # abstractmatrixci.jl -> AbstractMatrixCI ABC
    aca.py                # matrixci.jl + matrixaca.jl (MatrixCI, MatrixACA)
    rrlu.py               # matrixlu.jl (rrLU, rrlu(), arrlu())
    luci.py               # matrixluci.jl (MatrixLUCI)
  tensortrain/
    base.py               # abstracttensortrain.jl -> AbstractTensorTrain ABC
    core.py               # tensortrain.jl -> TensorTrain, compress, add/subtract
    cache.py              # cachedtensortrain.jl -> TTCache, BatchEvaluator
    cachedfunction.py      # cachedfunction.jl -> CachedFunction
    batcheval.py           # batcheval.jl -> dispatch helpers, ThreadedBatchEvaluator
  tci1.py                 # tensorci1.jl -> TensorCI1, crossinterpolate1
  tci2.py                 # tensorci2.jl -> TensorCI2, optimize, crossinterpolate2
  globalpivot.py          # globalpivotfinder.jl + globalsearch.jl
  conversion.py           # conversion.jl (TCI1<->TCI2<->TensorTrain)
  gausskronrod.py         # QuadGK.jl's src/gausskronrod.jl (unit-weight/hollow-tridiagonal subset only)
  integration.py          # integration.jl (Gauss-Kronrod quadrature via TCI)
  contraction.py          # contraction.jl (MPO x MPO contract, 3 algorithms)
  quantics/
    grid.py                # QuanticsGrids.jl's grid.jl -> InherentDiscreteGrid
    discretized.py          # QuanticsGrids.jl's grid_discretized.jl -> DiscretizedGrid
tests/
  test_util.py, test_matrixaca.py, test_matrixlu.py, test_matrixluci.py,
  test_tensortrain.py, test_cachedtensortrain.py, test_cachedfunction.py,
  test_tensorci1.py, test_tensorci2.py, test_conversion.py, test_integration.py,
  test_contraction.py, test_globalsearch.py, test_gausskronrod.py, ...
  # one per Julia test/test_*.jl, adapted; test_gausskronrod.py checks nodes/weights
  # against reference/QuadGK.jl's own precomputed values (e.g. its hardcoded n=7 xd7/wd7/wgd7)
reference/TensorCrossInterpolation.jl/   # vendored Julia source, read-only, for lookup
reference/QuadGK.jl/                     # vendored Julia source, read-only, for the kronrod() port
reference/QuanticsGrids.jl/              # vendored Julia source, read-only, for qutecipy.quantics
pyproject.toml
```

This is a starting proposal, not gospel — adjust if a flatter or differently-split
layout turns out cleaner once porting starts, but keep the Julia-file → Python-file
mapping traceable (a comment at the top of each module noting which `.jl` file it
ports is useful, since this is the primary way to check parity against upstream).

## Dependencies

- **numpy** — all dense arrays (replaces Julia `Array`/`Matrix`/`Vector`).
- **scipy** — `scipy.linalg.solve_triangular` (replaces `LowerTriangular`/`UpperTriangular`
  dispatch), `scipy.linalg.qr` (replaces Julia's `qr` used in `AtimesBinv`/`AinvtimesB`),
  `scipy.linalg.eigh_tridiagonal` (for the `gausskronrod.py` port — see External
  dependencies below).
- No threading/multiprocessing dependency needed initially — `ThreadedBatchEvaluator`
  (Julia `Threads.@threads`) can be ported as a plain sequential fallback first; only
  add `concurrent.futures`/`joblib` parallelism later if profiling shows it's needed
  (GIL means a naive thread-per-callback port gives no benefit unless the callback
  releases the GIL).
- `pytest` for tests; `scipy.optimize` as a **test-only** dependency (replaces `Optim`
  in the ported `test_tensortrain.py`, see External dependencies below). No `jax`/
  `autograd`/`torch` dependency planned (replaces `Zygote` with finite-difference
  gradients in that same test).

## Core data model translation notes

- **Multi-index**: Julia `MultiIndex = Vector{Int}`. In Python, use `tuple[int, ...]`
  wherever it must be a dict key (pivot index sets, caches) and plain `list[int]`
  only for transient/mutable working copies. Do not use Python `list` as a dict key
  (unlike Julia `Vector`, unhashable).
- **1-based → 0-based indexing**: Julia is 1-indexed throughout (`localdims`, pivot
  values, `rowindices`/`colindices`, `Iset`/`Jset` entries are all 1-based integers).
  Decide up front whether the Python port keeps pivot *values* 1-based (to allow
  direct comparison against Julia doctest/example output) or switches everything to
  0-based (more idiomatic Python, less error-prone for internal array indexing).
  **Recommendation: fully 0-based internally**, since mixing conventions across a
  ~5000-line port is a major bug source; convert only at the public
  API boundary if exact value-for-value parity with Julia examples is ever needed.
- **Value type parametrization** (`TensorCI1{ValueType}`, `TensorTrain{ValueType,N}`):
  Julia's compile-time type parameter becomes an implicit numpy `dtype`. Track dtype
  explicitly (e.g. assert consistent dtype across site tensors) since Python won't
  catch mismatches statically. `N` (3 vs 4 legs, MPS vs MPO) becomes a runtime
  `ndim`/`shape` check instead of a type parameter.
- **Mutating (`!`) vs. non-mutating functions**: Julia convention `foo!(x)` mutates
  `x` in place and returns `nothing`; `foo(x)` copies first. Preserve this as a
  naming convention in Python too, e.g. `tci.add_pivot(...)` (mutates `self`, returns
  `None`) vs. a module-level `add(tt1, tt2)` (returns a new object) — don't silently
  make everything mutate-in-place or everything copy; match each Julia function's
  actual behavior one at a time.
- **Multiple dispatch → single dispatch / isinstance / ABCs**: Julia leans hard on
  multiple dispatch for both operator overloading (`getindex` on `Union{Int,Colon,
  AbstractVector}`) and strategy selection (`_batchevaluate_dispatch` picking a fast
  path when `f isa BatchEvaluator`). In Python: normalize index arguments through one
  helper up front rather than replicating every overload; use `isinstance` checks or
  an ABC (`AbstractMatrixCI`, `BatchEvaluator`, `AbstractGlobalPivotFinder`) with a
  required method/`__call__` for the strategy-pattern cases.
- **Callable structs**: `AbstractTensorTrain <: Function` (and `TTCache`,
  `CachedFunction`, `Contraction`, `SubMatrix`) are directly callable in Julia via a
  `(x::T)(...)` method. Port each as a Python class with `__call__`.
- **`EllipsisNotation.jl`'s `..`**: used to splice a variable-length index tuple into
  an array-indexing expression whose rank depends on `M` (number of "center" sites in
  batch evaluation). Port to explicit Python tuple construction, e.g.
  `arr[(slice(None), *i, slice(None))]`, not numpy's `Ellipsis` (different semantics —
  numpy's `...` fills *remaining* axes, it doesn't splice a fixed-length sub-tuple).
- **In-place permutation scatter gotcha**: `matrixluci.jl`'s `colstimespivotinv`/
  `pivotinvtimesrows` do `result[perm, :] = result`. This must **not** be translated
  literally to numpy (`arr[perm] = arr` risks aliasing/incorrect results with fancy
  indexing, unlike Julia's broadcast-assignment semantics). Use
  `result = result[np.argsort(perm)]` or an explicit copy before scatter-assigning.
- **Dict-keyed caches** (`cacheleft/cacheright: Dict{MultiIndex,...}`,
  `Contraction.leftcache: Dict{Vector{Tuple{Int,Int}},...}`): key by `tuple(...)` in
  Python (Julia `Vector` has value equality/hash for free; Python `list` does not).
- **`CachedFunction`'s mixed-radix integer cache key** (`BitIntegers`/`BigInt`-sized
  keys to flatten a multi-index into one integer, with overflow checking): consider
  dropping this in the Python port in favor of keying the cache directly by
  `tuple(indexset)` — simpler, no overflow-tracking machinery needed, and Python
  dict-of-tuple lookups are fast enough that the integer-encoding trick (a
  Julia-performance-motivated design) isn't worth reproducing faithfully.

## Algorithmic layers — what to preserve precisely

### 1. Matrix pivoting layer (`matrix/`)
This is the numerical foundation; get it right and well-tested before building
anything on top.

- **`MatrixCI`** (`matrixci.jl`): classic CI with lazy QR-based pivot-matrix inversion
  (`AtimesBinv`/`AinvtimesB` — stack `[A;B]`, QR, split blocks — avoids forming an
  ill-conditioned inverse directly). Greedy **global**-argmax pivoting over the whole
  error matrix in `crossinterpolate`.
- **`MatrixACA`** (`matrixaca.jl`): incremental Adaptive Cross Approximation —
  maintains rank-1 deflation factors `u`, `v`, and reciprocal pivot values `alpha`
  (no explicit inverse, O(rank) update per new pivot). Pivoting heuristic is
  **local** (argmax of the *last* computed residual row/column only), a genuine
  algorithmic difference from `MatrixCI`'s global search — preserve both variants,
  don't conflate them.
- **`rrLU`/`arrlu`** (`matrixlu.jl`, the most important file in this layer): rank-
  revealing LU with full pivoting (`_optimizerrlu!`, global argmax of remaining
  trailing submatrix each step, dual reltol/abstol stopping criterion, tracks
  `maxerror` as the largest pivot magnitude seen) and **rook pivoting** (`arrlu`,
  alternates fixing rows vs. columns with randomized candidate-set growth, evaluates
  the target function `f` lazily rather than materializing the whole matrix — this
  is what makes it usable against expensive black-box functions). The `leftorthogonal`
  flag (which factor carries the unit diagonal) must be threaded through precisely —
  downstream TT-orthogonality assumptions in TCI2 depend on it. Manual rank-1 (`ger`)
  updates via explicit loops in Julia should become vectorized `np.outer` updates in
  Python (faster in numpy, no reason to keep the manual loop).
- **`MatrixLUCI`** (`matrixluci.jl`): thin `AbstractMatrixCI`-conforming wrapper
  around `rrLU`, reconstructing CI-style left/right factors via triangular solves
  (`scipy.linalg.solve_triangular`) rather than a general solve.

### 2. Tensor-train infrastructure (`tensortrain/`)
- **`AbstractTensorTrain`** (`abstracttensortrain.jl`): the shared contract — an ABC
  requiring `sitetensors()`/`sitetensor(i)`, with default mixin implementations of
  `linkdims`, `sitedims`, `rank`, `__len__`/`__iter__`/`__getitem__`, `evaluate`
  (naive O(L·D²) chain contraction), `__call__`, `sum` (linear-cost full-index-space
  sum by contracting `sum(T, axis=1)` site by site — the key TT trick to avoid
  exponential enumeration), `add`/`subtract` (direct-sum bond construction +
  recompression), `norm`/`norm2` (transfer-matrix contraction).
- **`TensorTrain`** (`tensortrain.jl`): concrete dense TT. `compress()` does the
  standard two-pass sweep (left-to-right orthogonalize with no truncation, then
  right-to-left truncate) — preserve this two-pass structure exactly, it's what
  makes the truncation numerically meaningful. Also: `_factorize` (LU/CI/SVD
  dispatch for single-core compression), `multiply`/`divide` (scale only the last
  core), `reverse`, `flatten`/`to_tensors` (for external-optimizer fitting via
  `TensorTrainFit`), `fulltensor` (exponential-cost dense materialization, debug-only).
- **`TTCache`** (`cachedtensortrain.jl`): the performance-critical piece. Memoized
  left/right "environment" vectors keyed by index-prefix/suffix (`evaluateleft`/
  `evaluateright`), so repeated queries sharing a prefix reuse work — exactly the
  access pattern TCI2's pivot search produces. `batchevaluate` is the single most
  important routine to port carefully: it builds stacked left/right environment
  matrices for *all* candidate left/right index sets at once and contracts the
  "center" sites via a handful of big matmuls instead of one contraction per
  candidate pivot pair — this is what makes TCI2's rook/full pivot search tractable.
- **`CachedFunction`** (`cachedfunction.jl`): memoizes an arbitrary user function;
  its batch-eval path specifically avoids re-evaluating already-cached entries and
  batches only the *missing* ones into as few calls as possible when wrapping a
  `BatchEvaluator`. Port the three-phase strategy (fill from cache → batch-call
  only missing entries → fill remaining) faithfully; simplify the cache key to
  `tuple(indexset)` (see above) rather than the mixed-radix integer encoding.
- **`batcheval.jl`**: `_batchevaluate_dispatch` picks between the generic
  nested-loop evaluator and delegating to an object's own `batch_evaluate` — port as
  an `isinstance(f, BatchEvaluator)` check.
- **`Contraction`** (`contraction.jl`): lazy MPO×MPO product exposing the same
  `BatchEvaluator` interface (own left/right environment caches, keyed by pairs of
  local indices) so it can itself be fed into `crossinterpolate2` (the `:TCI`
  contraction algorithm). Three algorithms: `contract_naive` (dense contract then
  SVD-recompress, largest transient bond dim), `contract_zipup` (sweep + factorize
  as you go, bounded intermediate bond dim), `contract_TCI` (treat the contraction
  as a black-box function and cross-interpolate it directly — can beat both when the
  *result* is low rank even if intermediates aren't). `_contract` (generic
  reshape/permute/matmul contraction primitive) maps directly to `np.tensordot`.

### 3. Core algorithms (`tci1.py`, `tci2.py`, `globalpivot.py`)
- **`TensorCI1`/`crossinterpolate1`** (`tensorci1.jl`): the original algorithm.
  Maintains a persistent 4-leg Π matrix per bond plus an incremental `MatrixACA`
  per bond; `addpivot!` does one ACA-greedy update per bond per sweep, propagating
  the change to the neighboring bond's Π (`updatePirows!`/`updatePicols!`) since `T`
  tensors are shared across adjacent bonds. Global pivot insertion
  (`addglobalpivot!`) is a bespoke fixed-point walk of the new point's
  prefixes/suffixes through the ACA machinery bond-by-bond — this is intricate
  stateful logic, port and test it carefully rather than approximating it.
  Convergence: single most recent sweep's max bond-error vs. tolerance.
- **`TensorCI2`/`optimize!`/`crossinterpolate2`** (`tensorci2.jl`, ~1000 lines, the
  most important file in the whole port): per-sweep-rebuilt rank-revealing LU
  (`rrLU`/`MatrixLUCI`, full or rook pivoting) over a genuine 2-site combined Π
  domain (`Iset[b]⊗localdim[b]` × `localdim[b+1]⊗Jset[b+1]`), with optional
  `strictlynested=False` behavior that carries over the *previous* sweep's index
  sets as extra candidate pivots (`Iset_history`/`Jset_history`) to avoid getting
  stuck. `maxbonddim` directly caps rank; rank can *shrink* between sweeps (unlike
  TCI1) via `sweep0site!`/`sweep1site!` compression passes. Built-in automatic
  **global pivot search** every outer iteration via a pluggable
  `AbstractGlobalPivotFinder` (default: randomized-restart greedy coordinate search
  against current-TT-vs-`f` disagreement) — this is TCI2's headline improvement over
  TCI1 (finds isolated peaks invisible to purely local bond sweeps). Convergence is
  history-windowed (`ncheckhistory`) requiring stable rank + low error + no new
  global pivots over several iterations, not just the latest sweep.
  Keep the **pluggable finder** as a first-class extension point (ABC + `__call__`,
  matching the Julia `AbstractGlobalPivotFinder` pattern) — the Julia test suite
  exercises user-defined custom finders, and the Python port's tests should too.
- **`globalpivotfinder.jl`/`globalsearch.jl`**: `DefaultGlobalPivotFinder` (random
  restarts, single-pass greedy coordinate scan, keep-best-if-above-margin,
  truncate to `maxnglobalpivot`) vs. `_floatingzone`/`estimatetrueerror`
  (unconditional coordinate-ascent to a true local error maximum, used for
  diagnostics and by the older `searchglobalpivots`). These are two genuinely
  different greedy search variants — don't merge them into one function.

Read `test/test_tensorci1.jl` and `test/test_tensorci2.jl` for concrete example
usage (function signatures, typical `localdims`/`tolerance`/`maxiter` choices,
multi-pivot initialization patterns) — port these as the first integration tests
for `crossinterpolate1`/`crossinterpolate2`.

### 4. Glue (`conversion.py`, `integration.py`)
- **`conversion.jl`**: TCI1⇄TCI2 conversion, and `sweep1sitegetindices!`/
  `TensorCI2(tt::TensorTrain)` which re-derives a canonical pivot-based TCI2
  representation from an arbitrary dense `TensorTrain` (needed after
  `contract_naive`/`contract_zipup` produce raw cores). Lower priority than the
  core algorithms but needed for full parity.
- **`integration.jl`**: Gauss-Kronrod quadrature built entirely on top of TCI2 +
  the linear-cost `sum(tt)` trick — good end-to-end test of the whole stack once
  it's ported. Needs a Gauss-Kronrod node/weight source (see Dependencies).

## Testing strategy

- Port each `test/test_*.jl` file to a corresponding `tests/test_*.py`, translating
  Julia `@test`/`@testset` blocks to `pytest` functions/classes, preserving the
  original test's *intent* and numerical tolerances (don't loosen tolerances just to
  make a test pass — if a ported test fails, the port has a bug).
  `test/test_with_aqua.jl`/`test_with_jet.jl` are Julia-specific static-analysis
  tests (Aqua.jl ambiguity checks, JET.jl type-stability checks) — no Python
  equivalent needed, skip.
- Where numerically meaningful, cross-check Python output directly against Julia
  output for the same inputs (e.g. run the Julia reference once, save expected
  arrays, compare in a Python test) rather than just re-deriving the same asserted
  values independently — this catches subtle porting bugs (off-by-one in 0- vs
  1-based indexing, wrong pivoting order, sign/conjugation differences in complex
  dot products) that a from-scratch reimplementation of the same test logic would
  not catch.
- Test the matrix layer (`MatrixCI`, `MatrixACA`, `rrLU`, `MatrixLUCI`) most
  thoroughly first and in isolation — every higher layer depends on it being
  numerically correct.

## Open questions to resolve early (ask the user, don't guess silently)

- 0-based vs. 1-based public API for pivot values (recommendation above: 0-based
  internally; decide whether the public API should offer a 1-based compatibility
  mode for users porting Julia code).
- Complex-number support priority: Julia's `_dot` dispatches to `BLAS.dotu`
  (unconjugated) for `ComplexF64` specifically to match tensor-network convention —
  decide if complex dtype support is in scope for v1 or real-only first.
- Whether to target NumPy-only or allow optional JAX/PyTorch backends for
  autodiff/GPU (Julia version has no such backend abstraction — recommend NumPy-only
  for the initial faithful port, revisit later).
- **Scope of the quantics-TCI ecosystem**: this project is named `qutecipy`, which
  reads as "quantics TCI python" — but `QuanticsGrids.jl`/`QuanticsTCI.jl` (the
  packages that actually add the quantics representation on top of
  `TensorCrossInterpolation.jl`) are currently out of scope per the "External
  dependencies" audit above, since only `TensorCrossInterpolation.jl` itself was
  requested for translation. Worth confirming with the user whether `qutecipy` is
  meant to eventually cover the full quantics-TCI stack (in which case
  `QuanticsGrids.jl` deserves its own porting plan alongside this one) or is scoped
  to the base cross-interpolation library only, with the name chosen for other
  reasons.
