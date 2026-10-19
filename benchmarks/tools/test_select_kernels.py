#!/usr/bin/env python3
"""Tests for select_kernels.py and for the committed kernels.tsv.

    python3 benchmarks/tools/test_select_kernels.py

No GPU needed. Selection decides what a run measures, so a mistake here is
silent under-measurement: the operator sees a pass and believes it covered the
multi-GPU set. The cases pinned below are the ones that would hide that.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECT = HERE / "select_kernels.py"
MANIFEST = HERE.parent / "suites" / "kernels.tsv"

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")


def run(*args, manifest=None):
    cmd = [sys.executable, str(SELECT), "--manifest", str(manifest or MANIFEST), *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def lines(out):
    return [l for l in out.splitlines() if l.strip()]


def main():
    print("select_kernels")

    # One scratch dir for the whole run: a per-call TemporaryDirectory would be
    # removed before the subprocess could open the file, and every "is
    # rejected" check would then pass on a FileNotFoundError instead of on the
    # validation it names.
    scratch = tempfile.TemporaryDirectory()
    counter = [0]

    def with_manifest(body):
        counter[0] += 1
        p = Path(scratch.name) / f"k{counter[0]}.tsv"
        p.write_text(body)
        assert p.is_file()
        return p

    # --- the committed manifest -------------------------------------------
    rc, out, err = run("--tier", "fast", "--devices", "1")
    check("fast tier parses and selects on 1 GPU", rc == 0 and len(lines(out)) == 5,
          f"rc={rc} out={out!r}")

    # A single-GPU kernel must not be dropped just because the host has more
    # GPUs than it uses -- that would silently shrink the gate on a big box.
    rc1, out1, _ = run("--tier", "fast", "--tier", "full", "--devices", "1")
    rc12, out12, _ = run("--tier", "fast", "--tier", "full", "--devices", "12")
    check("single-GPU set is identical on 1 and 12 GPUs",
          rc1 == rc12 == 0 and lines(out1) == lines(out12) and len(lines(out1)) == 11,
          f"1gpu={len(lines(out1))} 12gpu={len(lines(out12))}")

    # Multi-GPU kernels must not be offered on a single-GPU host.
    rc, out, err = run("--tier", "multi", "--devices", "1")
    check("multi tier is empty on 1 GPU", rc == 3 and not lines(out), f"rc={rc}")
    check("and says why", "needs 2 GPU(s), have 1" in err, err)

    # 2 GPUs: the scalable six, none of the fixed-topology three.
    rc, out, _ = run("--tier", "multi", "--devices", "2")
    got = lines(out)
    check("2 GPUs selects the six scalable kernels", rc == 0 and len(got) == 6,
          f"{got}")
    check("and excludes the fixed-topology ones",
          not any(b in " ".join(got) for b in ("all_to_all", "gemm_rs", "ag_gemm")),
          f"{got}")

    # Exactly-12 kernels appear at 12 and nowhere else.
    _, out6, _ = run("--tier", "multi", "--devices", "6")
    _, out12, _ = run("--tier", "multi", "--devices", "12")
    check("gemm_rs appears at 6 GPUs", any("gemm_rs" in l for l in lines(out6)))
    check("gemm_rs is gone at 12 GPUs", not any("gemm_rs" in l for l in lines(out12)))
    check("all_to_all appears only at 12",
          not any("all_to_all" in l for l in lines(out6))
          and any("all_to_all" in l for l in lines(out12)))

    # --- make targets ------------------------------------------------------
    # `all` must win over `standalone` when one dir contributes both, or the
    # build would not produce every selected binary.
    _, out, _ = run("--tier", "multi", "--devices", "12", "--what", "targets", "--quiet")
    targets = dict(l.split() for l in lines(out))
    check("collective at 12 GPUs builds the wider 'all' target",
          targets.get("collective") == "all", f"{targets}")
    _, out, _ = run("--tier", "multi", "--devices", "2", "--what", "targets", "--quiet")
    targets = dict(l.split() for l in lines(out))
    check("collective at 2 GPUs only needs 'standalone'",
          targets.get("collective") == "standalone", f"{targets}")

    # Every declared dir must exist, or a runner builds a phantom family.
    kernels = HERE.parent.parent / "kernels"
    _, out, _ = run("--tier", "fast", "--tier", "full", "--tier", "multi",
                    "--devices", "12", "--quiet")
    missing = [d for d in {l.split("/")[0] for l in lines(out)}
               if not (kernels / d / "Makefile").is_file()]
    check("every declared kernel dir has a Makefile", not missing, f"{missing}")

    # --- malformed manifests ----------------------------------------------
    p = with_manifest("gemm/x 1 0 fast\n")
    rc, _, err = run("--tier", "fast", "--devices", "1", manifest=p)
    check("a short row is rejected", rc != 0 and "5 columns" in err, err)

    p = with_manifest("gemm/x 1 0 turbo all\n")
    rc, _, err = run("--tier", "fast", "--devices", "1", manifest=p)
    check("an unknown tier is rejected", rc != 0, err)

    p = with_manifest("bare_binary 1 0 fast all\n")
    rc, _, err = run("--tier", "fast", "--devices", "1", manifest=p)
    check("a path without a dir is rejected", rc != 0 and "dir" in err, err)

    p = with_manifest("gemm/x 4 2 fast all\n")
    rc, _, err = run("--tier", "fast", "--devices", "4", manifest=p)
    check("max_dev below min_dev is rejected", rc != 0, err)

    p = with_manifest("# only comments\n\n")
    rc, _, err = run("--tier", "fast", "--devices", "1", manifest=p)
    check("an empty manifest is rejected", rc != 0, err)

    # An empty selection must not look like success to a shell caller.
    p = with_manifest("collective/x 8 8 multi all\n")
    rc, out, _ = run("--tier", "multi", "--devices", "2", manifest=p)
    check("empty selection exits 3, not 0", rc == 3 and not lines(out), f"rc={rc}")

    scratch.cleanup()

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
