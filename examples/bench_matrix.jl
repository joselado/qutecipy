#!/usr/bin/env julia
# Matrix-layer (rrLU) benchmark -- run from reference/TensorCrossInterpolation.jl with
# `julia --project=. ../../examples/bench_matrix.jl`, or see examples/README.md.
import TensorCrossInterpolation as TCI
using Random

function bench(name, f, nrep=7)
    f()  # warmup / JIT compile
    times = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    println("$name: $(round(minimum(times) * 1000, digits=3)) ms (min of $nrep)")
end

Random.seed!(42)
A200 = rand(200, 200)
bench("rrLU 200x200 full rank", () -> TCI.rrlu(copy(A200)))

A500 = rand(500, 100)
bench("rrLU 500x100 maxrank=50", () -> TCI.rrlu(copy(A500); maxrank=50))
