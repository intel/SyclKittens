# SyclKittens Benchmarks

This directory contains scheduler-independent benchmark contracts for the
public kernel release. Platform allocation, host selection, scheduler options,
and raw campaign evidence are deliberately kept outside this repository.

## Performance Check for a Change

`benchmarks/suites/perf_check.sh` answers one question: did this branch make a
kernel slower? It builds the working tree *and* its merge-base, runs each
binary base-then-head back to back for several rounds, and exits non-zero on a
regression. Run it before opening a pull request.

### Prerequisites

- A local Intel GPU. The script detects the AOT target itself.
- `icpx` on `PATH`. The script sources `env.src` for you; nothing to do if that
  already works for a normal kernel build.
- **Git history deep enough to reach the base commit.** A `--depth 1` clone has
  no merge-base and the script will refuse to guess. `git fetch --unshallow`, or
  pass `--base <ref>` naming something you do have.
- oneMKL only for `--all`, which includes `decode`. The fast subset skips it.

### Running it

```bash
benchmarks/suites/perf_check.sh                # fast subset, 5 trials per side
benchmarks/suites/perf_check.sh --all          # every single-GPU case
benchmarks/suites/perf_check.sh --multi        # the multi-GPU cases only
benchmarks/suites/perf_check.sh --all --multi  # everything this host can run
benchmarks/suites/perf_check.sh --list         # what would run here, no build
benchmarks/suites/perf_check.sh --trials 9     # tighter noise floor
benchmarks/suites/perf_check.sh --base v0.9.0  # against a tag instead of main
```

What it does, in order: resolve `git merge-base HEAD origin/main`; check that
commit out into a throwaway `git worktree`; build both trees, one family per
background job; run every binary base-then-head back to back, repeating for
`--trials` rounds; hand the two logs to `tools/perf_gate.py`; remove the
worktree.

The head side is the **working tree, uncommitted changes included**, so it is
usable while still iterating. The base side is always a clean checkout.

Measured on an Arc Pro B70, 2026.1 toolchain:

| Scope | Binaries | Families | Cases | Trials | Wall clock |
|:--|--:|--:|--:|--:|--:|
| default | 5 | 4 | 16 | 5 | ~1m 25s (26s build, 57s run) |
| `--all` | 11 | 6 | 35 | 3 | ~1m 20s |

Both builds dominate the fixed cost, so raising `--trials` is cheap: each extra
round adds only its own run time.

The default subset is `bf16_gemm`, `qkv_gemm`, `gqa_forward`, `rmsnorm` and
`silu_mul` — the cases that are cheap, quiet, and cover the matrix, reduction
and elementwise paths. `--all` adds `gqa_backward`, `layernorm`, the three
`rotary` binaries and `decode`.

### Single-GPU and multi-GPU scopes

Single-GPU and multi-GPU benchmarks belong to different machines, so they are
selected separately and the same command is correct on both. A one-card
workstation runs the whole `fast`/`full` set and none of the collectives; a
12-tile host is the only place `all_to_all` means anything.

| Scope | Flags | Tiers | Needs |
|:--|:--|:--|:--|
| PR gate default | *(none)* | `fast` | 1 GPU |
| whole single-GPU set | `--all` | `fast` + `full` | 1 GPU, oneMKL for `decode` |
| multi-GPU only | `--multi` | `multi` | 2+ GPUs, `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |
| everything runnable | `--all --multi` | all three | as above |

`--multi` alone **narrows** the run to the collectives; combined with `--all` it
widens instead. Nothing is hardcoded in the script: the kernel list, each
binary's device requirement and its make target are declared once in
[`suites/kernels.tsv`](suites/kernels.tsv), and `tools/select_kernels.py`
applies the filtering. Adding a kernel is a line in that file, not an edit to
three runners.

The visible device count is detected from `ZE_AFFINITY_MASK` when a platform
adapter set one — that is the count the kernels will actually see — and
otherwise from `sycl-ls`. Override it with `--devices N` or `SK_NUM_DEVICES`.

Check what a host will do before spending a build on it. `--list` prints the
selection and exits, with every skip and its reason on stderr:

```
$ benchmarks/suites/perf_check.sh --multi --list
select_kernels: skip collective/gemm_rs: needs 6 GPU(s), have 2
select_kernels: skip collective/all_to_all: needs 12 GPU(s), have 2
select_kernels: skip collective/ag_gemm: needs 12 GPU(s), have 2
scope    multi on 2 GPU(s), target intel_gpu_bmg_g31
collective/all_reduce                      2-12 GPU  multi  standalone
collective/all_gather                      2-12 GPU  multi  standalone
collective/reduce_scatter                  2-12 GPU  multi  standalone
context_attention/ring_attention            2-6 GPU  multi  all
context_attention/ring_attention_overlap   2-12 GPU  multi  all
context_attention/all_gather_attention      2-6 GPU  multi  all
```

Two kinds of device requirement show up in that `GPU` column. `2-12` scales:
the binary reads the topology (or `SK_NUM_DEVICES`) and adapts, so any count in
range works. A bare `6` or `12` is fixed: `NUM_DEVICES` is a compile-time
constant with no environment override, so the binary is skipped on every other
count rather than run wrong. Single-GPU rows read `1+` — they are *not* dropped
on a big host, they simply use one device.

Skips are printed, never hidden, and a selection that comes out empty exits
non-zero. A run that measured nothing must not read as a pass.

> **On a 2-card host the multi-GPU kernels do not currently run.** Verified on
> this box: `all_reduce` across two discrete B70s fails with
> `UR_RESULT_ERROR_DEVICE_LOST`; with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` it instead
> busy-waits on its barrier forever; `ring_attention` reports `GPUs: 4` and then
> throws `Can't add devices across platforms to a single context`. These
> harnesses assume tiles of one multi-tile device, not separate cards. `--multi`
> therefore selects and builds correctly here but every trial times out, and the
> gate exits 2 (harness error) rather than 0. Treat the multi-GPU scope as
> valid only on a true multi-tile host for now.

### Reading the output

Real output, abridged to one row per verdict:

```
== perf check
   base     9432113 For each kernel, add a standalone path to build the cpp source. (#1)
   head     27b206a Add a local perf regression gate + uncommitted changes
   target   intel_gpu_bmg_g31
   devices  2 visible
   scope    fast (5 binaries, 4 families)
   trials   5 per side, iters=50 warmup=5

== build base (9432113...)
   base/activation: OK
   base/attention: OK
   base/gemm: OK
   base/norm: OK
== build head
   head/activation: OK
   ...
   note: base/gemm has 3 spilling kernel(s)

== run
   trial 1/5: bf16_gemm_4096x4096x4096 qkv_gemm gqa_forward rmsnorm silu_mul
   ...

== perf gate   base 9432113  ->  head 27b206a-dirty
   target=intel_gpu_bmg_g31 iters=50 trials=5 floor=3%

   case                                                              base ms    head ms    delta   noise  verdict
   --------------------------------------------------------------------------------------------------------------
   silu_mul|silu_mul T=2048 inter=14336                               0.3288     0.3388    +3.0%    1.3%  REGRESSION
   rmsnorm|rmsnorm T=2048 D=4096                                      0.0545     0.0560    +2.8%    1.2%  NEUTRAL
   bf16_gemm_4096x4096x4096|GEMM 4096x4096x4096 (bf16 in, bf16 out)  18.0910     0.8700   -95.2%    6.6%  IMPROVEMENT
   rmsnorm|rmsnorm T=1 D=4096                                         0.0021     0.0028   +33.3%  461.9%  BELOW-RES  (best < 0.05 ms)

   REGRESSION=1  NEUTRAL=5  IMPROVEMENT=7  BELOW-RES=3

   FAIL: 1 blocking case(s)
      REGRESSION +3.0%  silu_mul|silu_mul T=2048 inter=14336

   logs: benchmarks/results/perf_check_20261014T074556Z.base.log
         benchmarks/results/perf_check_20261014T074556Z.head.log
```

- `case` is `binary|case-label`, so the same shape measured by two binaries stays
  distinguishable.
- `base ms` / `head ms` are **peak@N** — the fastest correct trial on each side.
- `delta` is the head/base latency change. Positive is slower.
- `noise` is the widest run-to-run spread seen on either side, as a fraction of
  that side's best trial. It is the resolution limit of this particular run.
- Rows are sorted worst-news-first, not alphabetically.
- `-dirty` on the head commit means uncommitted changes were included.

Compare the two middle rows to see the point: `+3.0%` against `1.3%` noise is a
call, while `+2.8%` against `1.2%` noise is not, because the floor is 3%.

### What to do about each verdict

| Verdict | Meaning | Blocks | What to do |
|:--|:--|:--|:--|
| `REGRESSION` | slower by more than the per-case threshold | yes | Confirm it, then fix or justify. See below. |
| `CORRECTNESS` | stopped passing its own numeric check | yes | Fix. Perf is not meaningful until it passes. |
| `MISSING` | measured on base, absent from head | yes | Either restore the case or say in the PR why it went. |
| `NUMERICS` | residual grew by more than `--error-factor` | `--fail-on-numerics` | Check you did not trade accuracy for speed by accident. |
| `NOISY` | spread exceeded the effect; inconclusive | `--fail-on-noisy` | Re-run with more `--trials`, or quiet the machine. |
| `NEUTRAL` | within noise, and the noise was small | no | Nothing. |
| `IMPROVEMENT` | faster by more than the threshold | no | Quote the number in the PR. |
| `BELOW-RES` | best trial under `--min-ms`; a launch-latency probe | no | Ignore. `T=1` runs at 0.003 ms and swings 30% on scheduling alone. |
| `NEW` | absent from base | no | Nothing. |

`CORRECTNESS` outranks perf on purpose: a change that "wins" by breaking
numerics must not read as an improvement. `MISSING` blocks because silently
losing a benchmark is how coverage rots.

Exit status is 0 for no regression, 1 for a blocking verdict, 2 for a harness
error — including a build failure on either side, since half an A/B would read
as a passing gate.

#### Confirming a reported regression

A verdict is a reason to look, not a conclusion. Re-run with more trials first;
if it holds, isolate it with a knob rather than a rebuild of the whole branch.
The `silu_mul T=2048` row above is a worked example — it survived `--trials 9`,
and one `LARGE_GRF` A/B on the head tree alone explained it:

```bash
make -B -C kernels/activation all CONFIG=standalone LARGE_GRF=0   # 0.3294 ms
make -B -C kernels/activation all CONFIG=standalone LARGE_GRF=1   # 0.3426 ms (+4.0%)
```

`LARGE_GRF ?= auto` turns on 256-GRF mode for every Battlemage kernel. The
matrix kernels need it — that is where those `-95%` rows come from — but a
bandwidth-bound kernel only loses occupancy, and the cache-resident `T=2048`
shape pays 4% for it while DRAM-bound `T=8192` pays 0.4%.

### Options

| Flag | Env | Default | Effect |
|:--|:--|:--|:--|
| `--base REF` | `SK_PERF_BASE` | `origin/main` | What to take the merge-base against |
| `--trials N` | `SK_PERF_TRIALS` | 5 | Rounds per side. Below 3 the spread cannot be estimated and the gate warns. |
| `--all` | — | off | The whole single-GPU set instead of the fast subset |
| `--multi` | — | off | Multi-GPU cases only; with `--all`, added to the single-GPU set |
| `--devices N` | `SK_NUM_DEVICES` | detected | GPUs to select against. Detection prefers `ZE_AFFINITY_MASK`, then `sycl-ls`. |
| `--list` | — | off | Print the selection with skip reasons and exit; builds nothing |
| `--iters N` | `SK_ITERS` | 50 | Timed iterations inside each harness invocation |
| — | `SK_WARMUP` | 5 | Warmup iterations before timing starts |
| `--threshold F` | — | 0.03 | Smallest effect reportable, as a fraction. The per-case noise floor overrides it upward. |
| `--min-ms MS` | — | 0.05 | Below this a case is a latency probe and is not judged |
| `--error-factor X` | — | 2.0 | Residual growth that earns a `NUMERICS` flag |
| `--fail-on-noisy` | — | off | Treat inconclusive cases as failures |
| `--fail-on-numerics` | — | off | Treat a grown residual as a failure |
| `--json PATH` | — | — | Write the full verdict, including every threshold, as JSON |
| `--keep` | — | off | Leave the base worktree in place for inspection |
| — | `SK_TIMEOUT` | `300s` | Per-binary wall clock. A collective that cannot reach its peers busy-waits forever; this records it as a failure instead of hanging the gate. |
| — | `GPU_TARGET` | detected | AOT target, pinned identically on both sides |
| — | `SK_BENCH_RESULTS` | `benchmarks/results` | Where the two logs land |

### Why `--trials` matters more than `--threshold`

The same `silu_mul` effect reads differently depending only on how many rounds
were run:

| Trials | delta | noise | Verdict |
|--:|--:|--:|:--|
| 3 | +3.1% | 4.4% | `NOISY` — does not block |
| 5 | +3.0% | 1.3% | `REGRESSION` — blocks |

Nothing about the code changed between those two runs; the 3-trial run simply
happened to catch a slow round, and the gate then refused to call an effect
smaller than the spread it had just observed. That is the intended behaviour:
lowering `--threshold` would not have helped, because the noise floor, not the
fixed floor, was binding. More rounds is the only real fix.

### Why it measures both sides instead of storing a baseline

The same binary varies run to run by more than the effects worth catching. On
an Arc Pro B70, `bf16_gemm` 4096³ spans 0.874–0.994 ms — 13.7% — across nine
trials of one build, while `gqa_forward` spans 1.3%. A single committed
threshold is therefore simultaneously too loose for attention and too tight for
GEMM, and a stored number also drifts with the driver, the compiler and the
ambient temperature. Measuring both sides on one machine minutes apart cancels
all of that. The cost is building twice.

Three rules keep the verdict honest:

- **peak@N per side.** Interference is one-sided — it only ever makes a kernel
  slower — so the fastest trial is the best estimator, matching the convention
  used everywhere else in this directory.
- **No case is called by less than the noise it just showed.** The threshold is
  `max(--threshold, observed spread)`.
- **The median must agree in sign with the minimum.** One lucky fast base trial
  is not a regression.

### Troubleshooting

| Symptom | Cause and fix |
|:--|:--|
| `cannot resolve a merge-base` | Shallow clone, or the base branch was never fetched. `git fetch --unshallow`, or `--base <ref>`. |
| `working tree is identical to the merge-base` | Nothing to compare; exits 0. Commit or edit something first. |
| `BUILD FAILED` on the base side only | The base predates something the build now needs. Pick a newer `--base`. |
| `BUILD FAILED` on `decode` | oneMKL headers missing. Use the default scope, which excludes it. |
| `icpx not found` | `env.src` did not find a oneAPI install. Source your own environment first. |
| Everything is `NOISY` | Machine is busy, or `--trials` is too low. Close other GPU work and raise it. |
| Every case shows a huge delta | The two sides built for different architectures. Pin `GPU_TARGET` explicitly. |
| A worktree is left in `/tmp` | The run was killed. `git worktree prune` cleans the bookkeeping. |
| `no kernel in scope ... runs on N GPU(s)` | The scope needs more devices than are visible; exits 2. `--list` shows which and why. Drop `--multi`, or raise `--devices`. |
| `!` printed after a kernel name | That trial hit `SK_TIMEOUT`. Grep the log for `SK_TRIAL_TIMEOUT`. On a multi-card host this is the expected collective hang described above. |
| A multi-GPU case shows `MISSING` | Every trial timed out or exited non-zero, so there is nothing to compare. The log holds the harness's own stderr. |
| `skip ...: needs exactly 12 GPU(s)` | Fixed-topology binary; `NUM_DEVICES` is compiled in. Run it from `distributed.sh` on a host with that count. |

The AOT target is pinned explicitly for both sides because `common.mk` only
learned to detect the local GPU recently; an older base checkout would otherwise
default to `intel_gpu_pvc` and the comparison would be across two
architectures.

IGC spilling the mainloop is the single biggest perf factor on this hardware and
is only visible at compile time, so the script greps both build logs and prints
e.g. `note: base/gemm has 3 spilling kernel(s)`. A `-95%` row with that note
next to it is a spill being fixed, not an algorithmic win.

### Tracking performance over time

`perf_check.sh` compares two commits. To track absolute numbers across weeks,
capture a run and diff later ones against it:

```bash
python3 benchmarks/tools/parse_baseline.py RUN.log --csv out.csv --json out.json
python3 benchmarks/tools/parse_baseline.py NEW.log --compare out.json
```

Those numbers are only comparable on the same host, driver and toolchain — which
is exactly why the PR gate does not use them.

### Pieces

| Path | Role |
|:--|:--|
| `suites/perf_check.sh` | build both sides, interleave the runs |
| `suites/kernels.tsv` | the kernel list: path, device range, tier, make target |
| `tools/select_kernels.py` | manifest → the subset this host can run |
| `tools/perf_gate.py` | paired statistics, verdict, exit code |
| `tools/parse_baseline.py` | log → CSV/JSON; also `--compare` against a stored run |
| `tools/test_perf_gate.py` | verdict-logic tests; no GPU required |
| `tools/test_select_kernels.py` | selection and manifest tests; no GPU required |

`select_kernels.py` is usable on its own, which is how another runner or a
scheduler asks what a host supports without parsing the manifest itself:

```bash
python3 benchmarks/tools/select_kernels.py --tier multi --devices 12
python3 benchmarks/tools/select_kernels.py --tier fast --tier full --devices 1 \
    --what targets --quiet          # '<dir> <make-target>' pairs, for a build loop
```

It exits 0 with a selection, 3 when the selection is empty, and 1 on a
malformed manifest.

`perf_gate.py` can also be re-run on its own against two logs you already have,
which is how you re-judge an old run at a different threshold without
remeasuring:

```bash
python3 benchmarks/tools/perf_gate.py \
    --base results/perf_check_*.base.log \
    --head results/perf_check_*.head.log \
    --threshold 0.05
```

### CI

Wiring this into CI needs a self-hosted runner with an Intel GPU; the script
already writes a `GITHUB_STEP_SUMMARY` table when that variable is set, so the
job body is a checkout with `fetch-depth: 0` plus one invocation. A single-GPU
runner uses the default scope; a multi-tile runner adds `--multi`.

The no-GPU parts run in the existing lint job on every pull request:
`test_perf_gate.py`, which synthesises its own logs, and
`test_select_kernels.py`, which also checks the committed manifest — including
that every kernel directory it names still has a `Makefile`, so a renamed
kernel family fails the PR instead of silently shrinking the gate.

## Distributed Suites

Run a suite after an external platform adapter has selected the devices:

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9,10,11
benchmarks/suites/distributed.sh collectives-12tile
```

Supported suites:

| Suite | Required visible devices | Targets |
|:--|--:|:--|
| `collectives-12tile` | 12 | All-reduce, all-gather, reduce-scatter, all-to-all |
| `ag-gemm-12tile` | 12 | Topology-aware all-gather plus GEMM |
| `gemm-rs-6gpu` | 6 | GEMM plus reduce-scatter |
| `context-attention-6gpu` | 6 | Ring, overlap ring, and all-gather attention |

The context-attention suite requires copy-engine routing so the overlap result
measures the intended algorithm:

```bash
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1
export UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1
```

Useful controls:

```bash
export SK_BENCH_TRIALS=3
export SK_BENCH_RESULTS=/writable/result/directory
export SK_FORCE_REBUILD=1
```

Every run writes a raw log and a JSON environment manifest. A successful log
ends with `SK_BENCHMARK_COMPLETED`. Three-trial performance studies use the
fastest correct result, or peak@3.

Generated output defaults to `benchmarks/results/`, which is ignored by Git and
must not be committed to the public repository.
