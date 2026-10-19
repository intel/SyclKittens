#!/usr/bin/env python3
"""Pick the kernels that a given machine can actually run.

    select_kernels.py --tier fast --devices 1
    select_kernels.py --tier multi --devices 2 --what dirs

Reads benchmarks/suites/kernels.tsv, keeps the rows whose tier was asked for and
whose device requirement the machine meets, and prints them one per line. Skips
go to stderr with a reason, so a runner can surface "needs 12 GPUs, have 2"
instead of silently measuring less than the operator thinks.

Splitting single-GPU from multi-GPU matters because they are different machines:
a single B70 workstation can run the whole `fast`/`full` set but none of the
collectives, while a 12-tile host is the only place `all_to_all` means anything.
Hardcoding one list per runner is what gets a benchmark quietly dropped.
"""

import argparse
import sys
from pathlib import Path

TIERS = ("fast", "full", "multi")
MANIFEST = Path(__file__).resolve().parent.parent / "suites" / "kernels.tsv"


def load(path):
    """-> list of row dicts, in manifest order."""
    rows = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(
                f"{path}:{lineno}: expected 5 columns, got {len(parts)}: {raw!r}")
        rel, min_dev, max_dev, tier, target = parts
        if tier not in TIERS:
            raise SystemExit(f"{path}:{lineno}: unknown tier {tier!r}")
        if "/" not in rel:
            raise SystemExit(f"{path}:{lineno}: path must be <dir>/<binary>: {rel!r}")
        try:
            lo, hi = int(min_dev), int(max_dev)
        except ValueError:
            raise SystemExit(f"{path}:{lineno}: device counts must be integers")
        if lo < 1 or (hi and hi < lo):
            raise SystemExit(
                f"{path}:{lineno}: need 1 <= min_dev <= max_dev (or max_dev 0)")
        rows.append({"path": rel, "dir": rel.split("/", 1)[0],
                     "binary": rel.split("/", 1)[1], "min": lo, "max": hi,
                     "tier": tier, "target": target})
    if not rows:
        raise SystemExit(f"{path}: no kernels declared")
    return rows


def fits(row, devices):
    """-> (bool, reason).

    A kernel needs at least min_dev GPUs. An exact-count kernel -- one whose
    max_dev equals its min_dev -- additionally refuses any other count, because
    its NUM_DEVICES is a compile-time constant with no environment override.
    max_dev == 0 means "no upper bound": a single-GPU kernel runs on a 12-tile
    host perfectly well, it just uses one device.
    """
    if devices < row["min"]:
        return False, f"needs {row['min']} GPU(s), have {devices}"
    if row["max"] and row["min"] == row["max"] and devices != row["min"]:
        return False, f"needs exactly {row['min']} GPU(s), have {devices}"
    return True, ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", action="append", choices=TIERS, metavar="TIER",
                    help="tier to include; repeatable (default: fast)")
    ap.add_argument("--devices", type=int, required=True, metavar="N",
                    help="GPUs visible to the run")
    ap.add_argument("--what", choices=("paths", "dirs", "targets", "table"),
                    default="paths",
                    help="paths: <dir>/<binary> lines (default); dirs: unique "
                         "kernel dirs; targets: unique '<dir> <make-target>' "
                         "pairs; table: human-readable")
    ap.add_argument("--manifest", default=str(MANIFEST), metavar="PATH")
    ap.add_argument("--quiet", action="store_true",
                    help="do not report skipped kernels on stderr")
    args = ap.parse_args()

    tiers = args.tier or ["fast"]
    if args.devices < 1:
        raise SystemExit("--devices must be at least 1")

    rows = [r for r in load(args.manifest) if r["tier"] in tiers]
    kept, skipped = [], []
    for row in rows:
        ok, why = fits(row, args.devices)
        (kept if ok else skipped).append((row, why))

    if skipped and not args.quiet:
        for row, why in skipped:
            print(f"select_kernels: skip {row['path']}: {why}", file=sys.stderr)

    if args.what == "paths":
        for row, _ in kept:
            print(row["path"])
    elif args.what == "dirs":
        seen = []
        for row, _ in kept:
            if row["dir"] not in seen:
                seen.append(row["dir"])
        print("\n".join(seen))
    elif args.what == "targets":
        # A dir can hold binaries from different make targets; `all` supersedes
        # `standalone`, so take the widest target any selected binary needs.
        widest = {}
        for row, _ in kept:
            if widest.get(row["dir"]) != "all":
                widest[row["dir"]] = row["target"]
        for d, t in widest.items():
            print(f"{d} {t}")
    else:
        width = max((len(r["path"]) for r, _ in kept), default=10)
        for row, _ in kept:
            if not row["max"]:
                rng = f"{row['min']}+"
            elif row["min"] == row["max"]:
                rng = str(row["min"])
            else:
                rng = f"{row['min']}-{row['max']}"
            print(f"{row['path']:<{width}}  {rng:>5} GPU  {row['tier']:<5}  "
                  f"{row['target']}")

    # An empty selection is a configuration error, not an empty success: a
    # runner that measures nothing must not report a pass.
    return 0 if kept else 3


if __name__ == "__main__":
    sys.exit(main())
