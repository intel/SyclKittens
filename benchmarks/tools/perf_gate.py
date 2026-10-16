#!/usr/bin/env python3
"""Decide whether a change moved SyclKittens kernel performance.

    python3 perf_gate.py --base BASE.log --head HEAD.log

Reads two logs written by benchmarks/suites/perf_check.sh -- the same cases
measured against the merge-base build and against the working-tree build,
interleaved trial by trial -- and prints a per-case verdict. Exits non-zero on a
regression or a correctness break, so it can gate a pull request.

Why a paired A/B instead of a diff against a stored baseline: on Battlemage the
same binary varies run to run by more than the effects worth catching.
`bf16_gemm` 4096^3 spans 0.874-0.994 ms (13.7%) across nine trials of a single
build, while `gqa_forward` spans 1.3%. One fixed threshold is therefore both too
loose for attention and too tight for GEMM, and a stored number additionally
folds in driver, compiler and ambient-temperature drift. Measuring both sides
back to back on one machine cancels all of that and lets each case derive its
own threshold from the spread just observed.

Three rules keep the verdict honest:

  * peak@N per side. Interference is one-sided -- it only ever makes a kernel
    slower -- so the fastest trial is the best estimator of what the code can
    do, and it matches the benchmarks/README convention.
  * A case may not be called either way by less than the noise it just showed.
    `threshold = max(--threshold, observed spread)`.
  * The median must agree in sign with the minimum. That stops one lucky fast
    base trial from reading as a regression.

Cases whose best time is under --min-ms are launch-latency probes (`T=1` runs at
0.003 ms and swings 30% on scheduling alone); they are reported as
below-resolution rather than being allowed to pass or fail.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_baseline import parse_samples  # noqa: E402

# Verdicts that make the gate fail.
BLOCKING = {"REGRESSION", "CORRECTNESS", "MISSING"}
# Longest-first so the summary reads worst news first.
ORDER = ["CORRECTNESS", "MISSING", "REGRESSION", "NOISY", "NUMERICS",
         "NEUTRAL", "IMPROVEMENT", "BELOW-RES", "NEW"]


def spread(values):
    """Run-to-run range as a fraction of the best trial."""
    lo = min(values)
    return (max(values) - lo) / lo if lo > 0 else 0.0


def peak(samples):
    """peak@N: the fastest correct trial, mirroring parse_baseline's reduction."""
    passing = [s for s in samples if s["status"] == "PASS"]
    pool = passing or samples
    best = min(pool, key=lambda s: s["ms"])
    return best, [s["ms"] for s in pool]


def judge(base, head, args):
    """-> (verdict, detail dict) for one case present on both sides."""
    b_best, b_ms = peak(base["samples"])
    h_best, h_ms = peak(head["samples"])

    b_pass = any(s["status"] == "PASS" for s in base["samples"])
    h_pass = any(s["status"] == "PASS" for s in head["samples"])

    noise = max(spread(b_ms), spread(h_ms))
    threshold = max(args.threshold, noise)
    d_min = h_best["ms"] / b_best["ms"] - 1.0
    d_med = statistics.median(h_ms) / statistics.median(b_ms) - 1.0

    detail = {
        "base_ms": b_best["ms"], "head_ms": h_best["ms"],
        "base_med": statistics.median(b_ms), "head_med": statistics.median(h_ms),
        "base_value": b_best["value"], "head_value": h_best["value"],
        "unit": head["unit"],
        "delta": d_min, "delta_median": d_med,
        "noise": noise, "threshold": threshold,
        "base_error": b_best["error"], "head_error": h_best["error"],
        "base_status": "PASS" if b_pass else b_best["status"],
        "head_status": "PASS" if h_pass else h_best["status"],
        "trials": [len(b_ms), len(h_ms)],
    }

    # Correctness outranks perf: a change that "wins" by breaking numerics must
    # not read as an improvement.
    if b_pass and not h_pass:
        return "CORRECTNESS", detail
    if not b_pass and not h_pass:
        # Already broken before the change; perf here is not meaningful.
        detail["note"] = "fails on both sides"
        return "CORRECTNESS", detail

    # A silently-worsening residual is the other way a tuning change cheats.
    be, he = detail["base_error"], detail["head_error"]
    if be is not None and he is not None and be > 0 and he > be * args.error_factor:
        detail["note"] = f"error {be:.2e} -> {he:.2e}"
        return "NUMERICS", detail

    if b_best["ms"] < args.min_ms:
        detail["note"] = f"best < {args.min_ms} ms"
        return "BELOW-RES", detail

    # Sign agreement between min and median: one lucky base trial is not a
    # regression.
    confident = abs(d_min) > threshold and d_min * d_med > 0
    if confident:
        return ("REGRESSION" if d_min > 0 else "IMPROVEMENT"), detail
    if noise > args.threshold:
        detail["note"] = f"spread {noise * 100:.1f}% hides effects below it"
        return "NOISY", detail
    return "NEUTRAL", detail


def collect(base_log, head_log, args):
    b_meta, b_cases, _ = parse_samples(base_log)
    h_meta, h_cases, _ = parse_samples(head_log)

    rows = []
    for key in sorted(set(b_cases) | set(h_cases)):
        name = f"{key[0]}|{key[1]}"
        if key not in h_cases:
            rows.append((name, "MISSING",
                         {"note": "measured on base, absent from head"}))
        elif key not in b_cases:
            rows.append((name, "NEW", {"note": "absent from base"}))
        else:
            verdict, detail = judge(b_cases[key], h_cases[key], args)
            rows.append((name, verdict, detail))
    return b_meta, h_meta, rows


def fmt_pct(x):
    return f"{x * 100:+.1f}%"


def render(b_meta, h_meta, rows, args):
    base_c = b_meta.get("SK_PERF_COMMIT", "?")
    head_c = h_meta.get("SK_PERF_COMMIT", "?")
    print()
    print(f"== perf gate   base {base_c}  ->  head {head_c}")
    print(f"   target={h_meta.get('SK_GPU_TARGET', '?')} "
          f"iters={h_meta.get('SK_ITERS', '?')} "
          f"trials={h_meta.get('SK_BENCHMARK_TRIALS', '?')} "
          f"floor={args.threshold * 100:.0f}%")
    print()

    width = max((len(r[0]) for r in rows), default=10)
    head = (f"   {'case':<{width}}  {'base ms':>9}  {'head ms':>9}  "
            f"{'delta':>7}  {'noise':>6}  verdict")
    print(head)
    print("   " + "-" * (len(head) - 3))
    for name, verdict, d in sorted(rows, key=lambda r: (ORDER.index(r[1]), r[0])):
        if "delta" not in d:
            print(f"   {name:<{width}}  {'':>9}  {'':>9}  {'':>7}  {'':>6}  "
                  f"{verdict}  ({d.get('note', '')})")
            continue
        note = f"  ({d['note']})" if "note" in d else ""
        print(f"   {name:<{width}}  {d['base_ms']:9.4f}  {d['head_ms']:9.4f}  "
              f"{fmt_pct(d['delta']):>7}  {d['noise'] * 100:5.1f}%  "
              f"{verdict}{note}")

    counts = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    print()
    print("   " + "  ".join(f"{v}={counts[v]}" for v in ORDER if v in counts))

    blocking = [r for r in rows if r[1] in BLOCKING]
    if args.fail_on_noisy:
        blocking += [r for r in rows if r[1] == "NOISY"]
    if args.fail_on_numerics:
        blocking += [r for r in rows if r[1] == "NUMERICS"]

    print()
    if blocking:
        print(f"   FAIL: {len(blocking)} blocking case(s)")
        for name, verdict, d in blocking:
            extra = f" {fmt_pct(d['delta'])}" if "delta" in d else ""
            print(f"      {verdict}{extra}  {name}")
    else:
        print("   PASS: no regression above the per-case noise floor")

    noisy = [r for r in rows if r[1] == "NOISY"]
    if noisy and not args.fail_on_noisy:
        worst = max(noisy, key=lambda r: r[2]["noise"])
        print(f"   note: {len(noisy)} case(s) inconclusive; worst spread "
              f"{worst[2]['noise'] * 100:.1f}% on {worst[0]}. "
              f"Raise --trials to tighten.")
    return blocking


def write_step_summary(rows, blocking, path):
    """GitHub Actions step summary, so wiring this into CI needs no code change."""
    with open(path, "a") as fh:
        fh.write("## SyclKittens perf gate\n\n")
        fh.write(f"**{'❌ FAIL' if blocking else '✅ PASS'}**\n\n")
        fh.write("| Case | Base ms | Head ms | Delta | Noise | Verdict |\n")
        fh.write("|---|---:|---:|---:|---:|---|\n")
        for name, verdict, d in sorted(rows, key=lambda r: (ORDER.index(r[1]), r[0])):
            if "delta" not in d:
                fh.write(f"| `{name}` | - | - | - | - | {verdict} |\n")
                continue
            fh.write(f"| `{name}` | {d['base_ms']:.4f} | {d['head_ms']:.4f} | "
                     f"{fmt_pct(d['delta'])} | {d['noise'] * 100:.1f}% | "
                     f"{verdict} |\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, metavar="LOG",
                    help="log from the merge-base build")
    ap.add_argument("--head", required=True, metavar="LOG",
                    help="log from the working-tree build")
    ap.add_argument("--threshold", type=float, default=0.03, metavar="FRAC",
                    help="minimum effect size to report, as a fraction; the "
                         "per-case noise floor overrides it upward (default 0.03)")
    ap.add_argument("--min-ms", type=float, default=0.05, metavar="MS",
                    help="cases faster than this are launch-latency probes and "
                         "are not judged (default 0.05)")
    ap.add_argument("--error-factor", type=float, default=2.0, metavar="X",
                    help="flag NUMERICS when the residual grows by more than "
                         "this factor (default 2.0)")
    ap.add_argument("--fail-on-noisy", action="store_true",
                    help="treat inconclusive cases as failures")
    ap.add_argument("--fail-on-numerics", action="store_true",
                    help="treat a grown residual as a failure")
    ap.add_argument("--json", metavar="PATH", help="write the full verdict as JSON")
    args = ap.parse_args()

    for p in (args.base, args.head):
        if not Path(p).exists():
            print(f"perf_gate: no such log: {p}", file=sys.stderr)
            return 2

    b_meta, h_meta, rows = collect(args.base, args.head, args)
    if not rows:
        print("perf_gate: no measurements parsed from either log", file=sys.stderr)
        return 2

    blocking = render(b_meta, h_meta, rows, args)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schema_version": 1,
            "base_meta": b_meta, "head_meta": h_meta,
            "thresholds": {"floor": args.threshold, "min_ms": args.min_ms,
                           "error_factor": args.error_factor},
            "blocking": [r[0] for r in blocking],
            "cases": {name: {"verdict": v, **d} for name, v, d in rows},
        }, indent=2) + "\n")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        write_step_summary(rows, blocking, step_summary)

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
