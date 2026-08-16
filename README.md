# qutecipy

Quantics tensor cross interpolation in Python.

A NumPy/SciPy port of [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl)
(the TCI algorithm and the tensor-train machinery under it) and
[`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl) (coordinate ↔
quantics-bitstring mapping), plus two extensions with no Julia counterpart:
array-valued TCI and the layer that joins the two ports together.

TCI learns a low-rank tensor-train representation of a function of many indices by
sampling it at adaptively chosen points — it never enumerates the full index space, so
it works where the tensor itself would not fit in memory.

## Install

```bash
pip install qutecipy          # numpy + scipy
pip install qutecipy[fast]    # adds numba, which accelerates the rank-revealing LU kernel
```

Python 3.10+.

## Quantics: a million-point function at bond dimension 3

Quantics encodes a grid coordinate as a string of bits, one tensor-train site per bit,
so nearby length scales land on nearby sites. Functions with scale separation compress
exponentially:

```python
import numpy as np
from qutecipy import DiscretizedGrid, quantics_crossinterpolate

grid = DiscretizedGrid.from_resolutions(["x"], [20], lower_bound=(0.0,), upper_bound=(1.0,))
qtt, ranks, errors = quantics_crossinterpolate(
    np.float64, lambda x: np.exp(-x) * np.cos(20 * x) + 0.3, grid, tolerance=1e-9
)

qtt.rank()      # 3        -- 2**20 = 1,048,576 grid points, bond dimension 3
qtt(0.375)      # evaluate at a coordinate
qtt.integral()  # integrate over the grid, in time linear in the number of sites
```

That run takes ~0.3 s and reproduces the function to `1.1e-9` at every grid point.

Two things worth knowing up front:

- **Evaluation snaps to the grid.** `qtt(x)` answers with the nearest grid point, so
  off-grid accuracy is bounded by the grid spacing, not by `tolerance`.
- **`integral()` is the rectangle rule**, so its error is `O(2**-R)` per dimension and
  is unrelated to `tolerance` — in the run above the interpolation is good to `1.1e-9`
  while the integral is good to `1.3e-6`. Raise `R` to improve it, not `tolerance`. For
  high-accuracy quadrature of a smooth low-dimensional function use
  `qutecipy.integrate`, which is Gauss–Kronrod based.

Multiple variables, mixed radices, and all four unfolding schemes (`fused`,
`interleaved`, `grouped`, or a custom index table) are supported.

## Plain TCI on a discrete index space

If you have a function of several discrete indices rather than a continuous
coordinate, use TCI directly:

```python
import numpy as np
from qutecipy import crossinterpolate2

localdims = [10] * 8
f = lambda v: 1.0 / (1.0 + sum((x + 1) ** 2 for x in v))
tci, ranks, errors = crossinterpolate2(np.float64, f, localdims, tolerance=1e-8)

tci.rank()   # 12    -- a 10**8-entry tensor, learned from ~4e5 samples
tci.sum()    # sum over all 10**8 entries, in linear time
```

`crossinterpolate2` does not memoize `f` (matching the Julia original). If `f` is
expensive, wrap it — the pivot search revisits points, 5–15× on typical problems:

```python
from qutecipy import CachedFunction
tci, _, _ = crossinterpolate2(
    np.float64, CachedFunction(np.float64, f, localdims), localdims, tolerance=1e-8
)
```

`crossinterpolate1` (the original TCI algorithm), TT arithmetic and compression,
MPO × MPO contraction (naive / zip-up / TCI), TCI1 ⇄ TCI2 conversion and Gauss–Kronrod
quadrature are all available too — see `qutecipy.__all__`.

## Array-valued TCI

Interpolate `f(x) -> np.ndarray` with **all components sharing one set of pivots**, by
carrying the component index as an extra tensor-train leg:

```python
import numpy as np
from qutecipy import crossinterpolate2_array

def f(bits):                       # -> shape (3,)
    x = sum(b << i for i, b in enumerate(reversed(bits))) / 2 ** len(bits)
    return np.array([np.exp(-2 * x), np.cos(3 * x) + 0.5, 1.0 / (1 + x)])

att, ranks, errors = crossinterpolate2_array(np.float64, f, [2] * 8, (3,), tolerance=1e-10)

att([0, 1, 1, 0, 1, 0, 0, 1])   # the interpolated array
att.component(0)                 # component 0 as an ordinary TensorTrain
```

Every component gets the same pivots by construction, and one call to your `f` serves
all `K` components at a given `x`.

Component weights are derived automatically (`componentweights="auto"`), which matters
whenever components differ in scale: without them, `tolerance` is relative to the
largest component and a smaller one is dropped *silently*. On a 3-component function
whose middle component is `1e-8` times the others (R=10, `tolerance=1e-6`):

| | component 0 | component 1 (tiny) | component 2 | rank |
|---|---|---|---|---|
| `componentweights=None` | 1.4e-15 | **9.2e-01** | 8.1e-07 | 4 |
| `"auto"` (default) | 1.7e-15 | 4.5e-07 | 8.9e-08 | 6 |

Pass `None` when the small components genuinely do not matter — resolving them costs
bond dimension.

## Notes

Indexing is **0-based throughout**, unlike the 1-based Julia originals: local indices
run `0 .. localdim-1`, and grid indices, quantics digits and bit numbers likewise.
Convert at the boundary if you are comparing against Julia output.

The port was cross-validated against the Julia packages run locally during
development — several bugs survived static review and Python-only unit tests and were
caught only by diffing against live Julia output. The two extensions
(`crossinterpolate2_array`, `quantics_crossinterpolate`) have no Julia counterpart and
are validated differently: against direct TCI runs they must reproduce exactly, and
against closed forms.

`CLAUDE.md` has the full porting map (which Python module ports which `.jl` file),
every deliberate deviation from the originals and why, the performance work and its
measurements, and the design rationale for both extensions.

## Tests

```bash
pip install -e .[test]
pytest
```

## License

MIT, matching the upstream Julia packages. `TensorCrossInterpolation.jl`,
`QuanticsGrids.jl` and `QuadGK.jl` are MIT-licensed works of their respective authors;
this port credits them in the header of each module that translates their code.
