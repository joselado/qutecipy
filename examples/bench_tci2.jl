#!/usr/bin/env julia
# crossinterpolate2 benchmark + a machine-parseable result dump, for cross-checking
# against bench_tci2.py (see examples/README.md and examples/compare.py).
#
# Domain note: f uses v (1-based Julia indices, values 1..10) directly, so this is the
# *same* function over the *same* domain as bench_tci2.py, which uses (index+1) to
# compensate for qutecipy's 0-based indexing -- results (rank, sum) should match to near
# machine precision.
import TensorCrossInterpolation as TCI

function bench_and_report(name, f, localdims; kwargs...)
    tci, ranks, errors = f()  # warmup / JIT compile
    t0 = time_ns()
    tci, ranks, errors = f()
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9
    s = sum(tci)
    println("[$name] rank=$(TCI.rank(tci)) sum=$s time_s=$elapsed")
end

f8(v) = 1.0 / (1.0 + sum(v .^ 2))
localdims8 = fill(10, 8)
bench_and_report("8D_full", () -> TCI.crossinterpolate2(Float64, f8, localdims8; tolerance=1e-8), localdims8)

f20(v) = 1.0 / (1.0 + sum(v .^ 2))
localdims20 = fill(2, 20)
bench_and_report(
    "20D_rook",
    () -> TCI.crossinterpolate2(Float64, f20, localdims20; tolerance=1e-8, pivotsearch=:rook),
    localdims20,
)
