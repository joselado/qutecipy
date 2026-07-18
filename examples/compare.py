#!/usr/bin/env python3
"""Runs the Julia and Python benchmark/example scripts back to back and prints a
side-by-side correctness + performance comparison.

Usage (from the repo root):
    PYTHONPATH=. python3 examples/compare.py

Requires: Julia with TensorCrossInterpolation.jl instantiated at
reference/TensorCrossInterpolation.jl (see the repo's reference/ clone; run
`julia --project=. -e 'import Pkg; Pkg.instantiate()'` there once if needed).
If Julia isn't available, this still runs the Python side and says so.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JULIA_PROJECT = REPO_ROOT / "reference" / "TensorCrossInterpolation.jl"

LINE_RE = re.compile(r"\[(?P<name>\w+)\] rank=(?P<rank>\d+) sum=(?P<sum>[-\d.eE+]+) time_s=(?P<time>[-\d.eE+]+)")


def parse(output: str) -> dict:
    results = {}
    for line in output.splitlines():
        m = LINE_RE.search(line)
        if m:
            results[m["name"]] = {
                "rank": int(m["rank"]),
                "sum": float(m["sum"]),
                "time_s": float(m["time"]),
            }
    return results


def run_julia() -> dict:
    if not shutil.which("julia"):
        print("Julia not found on PATH -- skipping Julia side.")
        return {}
    script = REPO_ROOT / "examples" / "bench_tci2.jl"
    proc = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", str(script)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        print("Julia run failed:\n" + proc.stderr[-2000:])
        return {}
    print("--- Julia output ---")
    print(proc.stdout)
    return parse(proc.stdout)


def run_python() -> dict:
    script = REPO_ROOT / "examples" / "bench_tci2.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print("Python run failed:\n" + proc.stderr[-2000:])
        return {}
    print("--- Python output ---")
    print(proc.stdout)
    return parse(proc.stdout)


def main():
    julia_results = run_julia()
    python_results = run_python()

    print("\n=== Comparison ===")
    header = f"{'case':<10} {'Julia rank':>11} {'Python rank':>12} {'Julia sum':>16} {'Python sum':>16} {'rel. sum diff':>14} {'Julia ms':>10} {'Python ms':>11} {'ratio':>7}"
    print(header)
    for name in python_results:
        p = python_results[name]
        j = julia_results.get(name)
        if j is None:
            print(f"{name:<10} {'--':>11} {p['rank']:>12} {'--':>16} {p['sum']:>16.6g} {'--':>14} {'--':>10} {p['time_s'] * 1000:>11.2f} {'--':>7}")
            continue
        rel_diff = abs(p["sum"] - j["sum"]) / abs(j["sum"])
        ratio = p["time_s"] / j["time_s"]
        print(
            f"{name:<10} {j['rank']:>11} {p['rank']:>12} {j['sum']:>16.6g} {p['sum']:>16.6g} "
            f"{rel_diff:>14.2e} {j['time_s'] * 1000:>10.2f} {p['time_s'] * 1000:>11.2f} {ratio:>6.2f}x"
        )

    print(
        "\nNote: 'sum' is sum(tci) over the entire (exponentially large) index grid, computed in\n"
        "linear cost via the TT-sum trick -- an end-to-end check that both implementations\n"
        "converged to the same function, not just that they run. Small relative differences\n"
        "(~1e-6 to 1e-8, matching the requested tolerance) are expected since the two pivot\n"
        "searches use unrelated random-number streams and may settle on different (but equally\n"
        "valid) pivot sets. Timing is CPU time (Python) vs. wall time post-JIT-warmup (Julia);\n"
        "on a loaded/shared machine, run this a few times and look at the minimum."
    )


if __name__ == "__main__":
    main()
