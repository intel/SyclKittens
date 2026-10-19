#!/usr/bin/env python3
"""Tests for perf_gate.py's verdict logic.

    python3 benchmarks/tools/test_perf_gate.py

No GPU needed: each test synthesises the two logs perf_check.sh would have
written. The cases here are the ones that decide whether the gate is trusted --
a false REGRESSION gets the gate switched off, and a missed one defeats the
point -- so they are pinned with real measured magnitudes from Battlemage.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path(__file__).resolve().parent / "perf_gate.py"

HEADER = """SK_BENCHMARK_SCHEMA=1
SK_BENCHMARK_SUITE=perf_check
SK_BENCHMARK_TRIALS={trials}
SK_GPU_TARGET=intel_gpu_bmg_g31
SK_PERF_SIDE={side}
SK_PERF_COMMIT={commit}
"""


def write_log(path, side, commit, cases):
    """cases: {label: [(ms, value, status, rel_l2), ...]} one tuple per trial."""
    trials = max(len(v) for v in cases.values())
    out = [HEADER.format(trials=trials, side=side, commit=commit)]
    for t in range(trials):
        for label, samples in cases.items():
            if t >= len(samples):
                continue
            ms, value, status, err = samples[t]
            out.append(f"SK_TRIAL label={label} trial={t + 1}")
            out.append(f"{label} case: {ms} ms  {value} TFLOP/s  "
                       f"(iters=50 warmup=5)  rel_l2={err}  {status}")
    Path(path).write_text("\n".join(out) + "\n")


def run_gate(base_cases, head_cases, *extra):
    with tempfile.TemporaryDirectory() as d:
        b, h = Path(d) / "base.log", Path(d) / "head.log"
        write_log(b, "base", "aaaaaaa", base_cases)
        write_log(h, "head", "bbbbbbb", head_cases)
        proc = subprocess.run(
            [sys.executable, str(GATE), "--base", str(b), "--head", str(h), *extra],
            capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr


def flat(label, ms_list, status="PASS", err="1.70e-03", value=150.0):
    return {label: [(ms, value, status, err) for ms in ms_list]}


FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")


def main():
    print("perf_gate verdict logic")

    # A real regression, larger than the spread on either side.
    rc, out = run_gate(flat("gqa_forward", [1.397, 1.420, 1.428, 1.412, 1.423]),
                       flat("gqa_forward", [1.580, 1.593, 1.597, 1.587, 1.594]))
    check("clean regression is caught", "REGRESSION" in out and rc == 1, out)

    # The same measurement reversed must read as an improvement, not a pass.
    rc, out = run_gate(flat("gqa_forward", [1.580, 1.593, 1.597, 1.587, 1.594]),
                       flat("gqa_forward", [1.397, 1.420, 1.428, 1.412, 1.423]))
    check("improvement is not a failure", "IMPROVEMENT" in out and rc == 0, out)

    # Identical distributions must not be called either way.
    rc, out = run_gate(flat("gqa_forward", [1.397, 1.420, 1.428]),
                       flat("gqa_forward", [1.400, 1.419, 1.425]))
    check("no change reads NEUTRAL", "NEUTRAL" in out and rc == 0, out)

    # The measured bf16_gemm spread (0.874-0.994, 13.7%) with no real change.
    # A fixed 3% threshold would call this a regression; the noise floor must not.
    rc, out = run_gate(flat("bf16_gemm", [0.874, 0.895, 0.916, 0.924, 0.991]),
                       flat("bf16_gemm", [0.916, 0.924, 0.874, 0.994, 0.895]))
    check("GEMM noise does not fake a regression",
          "REGRESSION" not in out and rc == 0, out)

    # ... but it must say so rather than quietly passing.
    check("that case is reported NOISY", "NOISY" in out, out)

    # One lucky fast base trial: minimum moves, median does not. The sign-
    # agreement rule should refuse to call it.
    rc, out = run_gate(flat("bf16_gemm", [0.800, 0.950, 0.955, 0.960, 0.958]),
                       flat("bf16_gemm", [0.950, 0.952, 0.955, 0.953, 0.956]))
    check("outlier in base does not fake a regression",
          "REGRESSION" not in out and rc == 0, out)

    # Correctness outranks perf: twice as fast but wrong is still a failure.
    rc, out = run_gate(flat("bf16_gemm", [0.920, 0.925, 0.930]),
                       flat("bf16_gemm", [0.460, 0.462, 0.465], status="FAIL"))
    check("faster but FAIL is blocked", "CORRECTNESS" in out and rc == 1, out)

    # A residual that grows by more than the factor, with perf flat.
    rc, out = run_gate(flat("bf16_gemm", [0.920, 0.925, 0.930], err="1.70e-03"),
                       flat("bf16_gemm", [0.921, 0.924, 0.931], err="9.00e-03"))
    check("grown residual is flagged NUMERICS", "NUMERICS" in out, out)
    check("NUMERICS does not block by default", rc == 0, out)
    rc, _ = run_gate(flat("bf16_gemm", [0.920, 0.925, 0.930], err="1.70e-03"),
                     flat("bf16_gemm", [0.921, 0.924, 0.931], err="9.00e-03"),
                     "--fail-on-numerics")
    check("--fail-on-numerics blocks it", rc == 1)

    # Launch-latency probes: 0.0024 -> 0.0031 ms is 29% and means nothing.
    rc, out = run_gate(flat("silu_mul", [0.0024, 0.0027, 0.0031]),
                       flat("silu_mul", [0.0031, 0.0029, 0.0027]))
    check("latency probe is not judged", "BELOW-RES" in out and rc == 0, out)

    # Losing a benchmark must not read as a pass -- that is how coverage rots.
    rc, out = run_gate({**flat("bf16_gemm", [0.92, 0.93, 0.94]),
                        **flat("gqa_forward", [1.40, 1.41, 1.42])},
                       flat("bf16_gemm", [0.92, 0.93, 0.94]))
    check("a dropped case blocks", "MISSING" in out and rc == 1, out)

    # A case only the head has is new coverage, not a failure.
    rc, out = run_gate(flat("bf16_gemm", [0.92, 0.93, 0.94]),
                       {**flat("bf16_gemm", [0.92, 0.93, 0.94]),
                        **flat("gqa_forward", [1.40, 1.41, 1.42])})
    check("a new case does not block", "NEW" in out and rc == 0, out)

    # Empty logs are a harness error (2), not a pass (0) or a regression (1).
    with tempfile.TemporaryDirectory() as d:
        for n in ("a.log", "b.log"):
            (Path(d) / n).write_text("")
        proc = subprocess.run(
            [sys.executable, str(GATE), "--base", f"{d}/a.log", "--head", f"{d}/b.log"],
            capture_output=True, text=True)
    check("empty logs exit 2", proc.returncode == 2, proc.stderr)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
