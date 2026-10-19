#!/usr/bin/env python3
"""Turn a SyclKittens benchmark log into a CSV/JSON baseline, or diff against one.

    python3 parse_baseline.py RUN.log --csv out.csv --json out.json
    python3 parse_baseline.py NEW.log --compare baseline.json

Per the benchmarks/README convention a case keeps its fastest correct trial
(peak@N), so repeated trials reduce to one number per case.

`parse_samples` keeps every trial instead of reducing; `perf_gate.py` needs the
full set to measure run-to-run spread.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# "<label>: <ms> ms[/token]  [<value> TFLOP/s|GB/s]"
PERF_RE = re.compile(
    r"^(?P<label>[^:]+?):\s+"
    r"(?P<ms>\d+(?:\.\d+)?)\s*ms(?P<ms_suffix>/token|/layer)?"
    r"(?:\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>TFLOP/s|GB/s))?"
)
TRIAL_RE = re.compile(r"^SK_TRIAL label=(?P<label>\S+) trial=(?P<trial>\d+)")
SKIP_RE = re.compile(r"^SK_CASE_SKIPPED label=(?P<label>\S+) reason=(?P<reason>\S+)")
META_RE = re.compile(r"^(SK_[A-Z_]+)=(.*)$")
ERR_RE = re.compile(r"\b(rel_l2|max_err|L_err|delta|dQ|dK|dV)=([-+0-9.eE]+)")
# Harnesses print several error metrics; rel_l2 is the one most of them share.
ERR_PREFERENCE = ("rel_l2", "max_err", "L_err")


def pick_error(line):
    """One number per case, so a trend line is comparable across runs."""
    found = {}
    for k, v in ERR_RE.findall(line):
        try:
            found[k] = float(v)
        except ValueError:
            pass
    for key in ERR_PREFERENCE:
        if key in found:
            return found[key]
    # gqa_backward prints "rel_l2 delta=... dQ=... dK=... dV=..." with no bare
    # rel_l2=; track the worst of its per-tensor errors.
    return max(found.values()) if found else None

# Keys whose value is run metadata rather than a measurement.
META_KEYS = {
    "SK_BENCHMARK_SCHEMA", "SK_BENCHMARK_SUITE", "SK_BENCHMARK_RUN_ID",
    "SK_BENCHMARK_TRIALS", "SK_GPU_TARGET", "SK_ITERS", "SK_WARMUP",
    "SK_ICPX_VERSION", "SK_PERF_SIDE", "SK_PERF_COMMIT",
}


def parse_samples(path):
    """-> (meta dict, {(binary, case): record}, skipped dict), every trial kept.

    Each record carries a `samples` list in log order, one entry per
    measurement line. Use this when the spread between trials matters; use
    `parse_log` when one number per case is enough.
    """
    meta, cases, skipped = {}, {}, {}
    binary = None

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue

        m = META_RE.match(line)
        if m and m.group(1) in META_KEYS:
            meta[m.group(1)] = m.group(2).strip()
            continue

        m = TRIAL_RE.match(line)
        if m:
            binary = m.group("label")
            continue

        m = SKIP_RE.match(line)
        if m:
            skipped[m.group("label")] = m.group("reason")
            binary = None
            continue

        m = PERF_RE.match(line)
        if not m or binary is None:
            continue

        case = " ".join(m.group("label").split())
        ms = float(m.group("ms"))
        status = "FAIL" if "FAIL" in line else ("PASS" if "PASS" in line else "UNKNOWN")

        unit = m.group("unit")
        if unit is None:
            # decode reports latency only; ms/token is its figure of merit.
            unit = "ms" + (m.group("ms_suffix") or "")
            value = ms
        else:
            value = float(m.group("value"))

        rec = cases.setdefault((binary, case), {
            "binary": binary, "case": case, "unit": unit, "samples": [],
        })
        rec["samples"].append({
            "ms": ms, "value": value, "status": status, "error": pick_error(line),
        })

    return meta, cases, skipped


def parse_log(path):
    """-> (meta dict, {(binary, case): peak@N record}, skipped dict)"""
    meta, raw_cases, skipped = parse_samples(path)
    cases = {}

    for key, src in raw_cases.items():
        rec = {
            "binary": src["binary"], "case": src["case"], "unit": src["unit"],
            "ms": None, "value": None, "error": None,
            "status": "UNKNOWN", "trials": 0,
        }
        for s in src["samples"]:
            rec["trials"] += 1
            # peak@N: keep the fastest *correct* trial. A PASS always displaces
            # a non-PASS record, so a single bad trial cannot hide a good
            # measurement.
            first = rec["ms"] is None
            better_status = s["status"] == "PASS" and rec["status"] != "PASS"
            faster = first or s["ms"] < rec["ms"]
            if first or better_status or (s["status"] == rec["status"] and faster):
                rec.update(ms=s["ms"], value=s["value"],
                           status=s["status"], error=s["error"])
        cases[key] = rec

    return meta, cases, skipped


def write_csv(path, cases):
    cols = ["binary", "case", "ms", "value", "unit", "error", "status", "trials"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in sorted(cases):
            w.writerow({c: cases[key][c] for c in cols})


def write_json(path, meta, cases, skipped):
    payload = {
        "schema_version": 1,
        "suite": meta.get("SK_BENCHMARK_SUITE", "baseline"),
        "run_id": meta.get("SK_BENCHMARK_RUN_ID"),
        "meta": meta,
        "skipped": skipped,
        "cases": {f"{b}|{c}": cases[(b, c)] for (b, c) in sorted(cases)},
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def report(meta, cases, skipped):
    print()
    print(f"== baseline  run_id={meta.get('SK_BENCHMARK_RUN_ID', '?')} "
          f"target={meta.get('SK_GPU_TARGET', '?')} "
          f"iters={meta.get('SK_ITERS', '?')} trials={meta.get('SK_BENCHMARK_TRIALS', '?')}")
    if not cases:
        print("   no measurements parsed")
    width = max((len(c["case"]) for c in cases.values()), default=10)
    last = None
    for key in sorted(cases):
        rec = cases[key]
        if rec["binary"] != last:
            print(f"\n-- {rec['binary']}")
            last = rec["binary"]
        # Latency-only kernels (decode) have no throughput figure to show.
        value = "-" if rec["unit"].startswith("ms") else f"{rec['value']:.3f} {rec['unit']}"
        err = "" if rec["error"] is None else f"  err={rec['error']:.2e}"
        print(f"   {rec['case']:<{width}}  {rec['ms']:9.4f} ms  {value:>16}"
              f"  [{rec['status']}]{err}")

    n_pass = sum(1 for c in cases.values() if c["status"] == "PASS")
    print(f"\n== {len(cases)} cases: {n_pass} PASS, {len(cases) - n_pass} not PASS"
          + (f"; {len(skipped)} binaries skipped ({', '.join(sorted(skipped))})" if skipped else ""))


def compare(old_path, cases):
    old = json.loads(Path(old_path).read_text())
    old_cases = old.get("cases", {})
    print(f"\n== vs {old.get('run_id', old_path)}")
    rows, missing = [], []
    for (b, c) in sorted(cases):
        new = cases[(b, c)]
        ref = old_cases.get(f"{b}|{c}")
        if ref is None or not ref.get("ms"):
            missing.append(f"{b}|{c}")
            continue
        # Latency ratio: >1 means the new run is faster.
        rows.append((f"{b}|{c}", ref["ms"], new["ms"], ref["ms"] / new["ms"], new["status"]))

    if rows:
        width = max(len(r[0]) for r in rows)
        for name, old_ms, new_ms, speedup, status in rows:
            flag = "" if status == "PASS" else f"  <-- {status}"
            print(f"   {name:<{width}}  {old_ms:9.4f} -> {new_ms:9.4f} ms  "
                  f"{speedup:5.2f}x{flag}")
        best = max(rows, key=lambda r: r[3])
        worst = min(rows, key=lambda r: r[3])
        print(f"\n   best  {best[3]:.2f}x  {best[0]}")
        print(f"   worst {worst[3]:.2f}x  {worst[0]}")
    if missing:
        print(f"\n   {len(missing)} case(s) absent from the reference: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--compare", metavar="BASELINE_JSON")
    args = ap.parse_args()

    meta, cases, skipped = parse_log(args.log)
    report(meta, cases, skipped)

    if args.csv:
        write_csv(args.csv, cases)
    if args.json:
        write_json(args.json, meta, cases, skipped)
    if args.compare:
        compare(args.compare, cases)

    # Non-zero when nothing was measured or any case did not pass.
    if not cases:
        return 1
    return 0 if all(c["status"] == "PASS" for c in cases.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
