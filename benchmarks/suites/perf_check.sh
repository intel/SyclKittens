#!/usr/bin/env bash
# Local performance check for a pull request: A/B the working tree against its
# merge-base on this machine, and fail if a kernel got slower.
#
#   benchmarks/suites/perf_check.sh                 # fast subset, 5 trials/side
#   benchmarks/suites/perf_check.sh --all           # every single-GPU case
#   benchmarks/suites/perf_check.sh --multi         # the multi-GPU cases only
#   benchmarks/suites/perf_check.sh --all --multi   # everything this host can run
#   benchmarks/suites/perf_check.sh --list          # what would run here, no build
#   benchmarks/suites/perf_check.sh --trials 9      # tighter noise floor
#   benchmarks/suites/perf_check.sh --base v0.9.0   # against a tag, not main
#
# Which kernels exist and how many GPUs each needs is declared once, in
# benchmarks/suites/kernels.tsv. The visible device count is detected and
# anything that does not fit is skipped with a reason, so the same command is
# correct on a single-GPU workstation and on a 12-tile host.
#
# It checks out the merge-base into a throwaway git worktree, builds both trees,
# then runs each binary base-then-head back to back, repeating for --trials
# rounds. benchmarks/tools/perf_gate.py turns the two logs into a verdict.
#
# Measuring both sides here and now is what makes the result trustworthy: the
# same binary varies run to run by more than the effects worth catching (see the
# rationale in perf_gate.py), so a number stored in the repo cannot be compared
# against. The cost is building twice.
#
# The head side is the *working tree*, uncommitted changes included, which is
# what you want while iterating. The base side is always a clean checkout.
#
# Controls:
#   SK_PERF_BASE=origin/main   base ref (--base wins)
#   SK_PERF_TRIALS=5           rounds per side (--trials wins)
#   SK_BENCH_RESULTS=DIR       output directory (default benchmarks/results)
#   SK_ITERS=50 SK_WARMUP=5    harness timing arguments
#   GPU_TARGET=...             AOT target; detected when unset
#   SK_NUM_DEVICES=N           visible GPUs (--devices wins); detected when unset
#   SK_TIMEOUT=300s            per-binary wall clock; a stuck collective is
#                              recorded as a failure instead of hanging the run
#
# Exit status: 0 no regression, 1 regression or correctness break, 2 harness error.
set -uo pipefail

SK_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null) || {
    echo "perf_check: not inside a git checkout" >&2; exit 2
}

BASE_REF=${SK_PERF_BASE:-origin/main}
TRIALS=${SK_PERF_TRIALS:-5}
ITERS=${SK_ITERS:-50}
WARMUP=${SK_WARMUP:-5}
TIMEOUT=${SK_TIMEOUT:-300s}
RESULTS_DIR=${SK_BENCH_RESULTS:-$SK_ROOT/benchmarks/results}
KEEP=0
GATE_ARGS=()
DEVICES=${SK_NUM_DEVICES:-}
# A gate that nobody trusts gets ignored, so the default tier is the cases that
# are both cheap and quiet: no latency probes, no decode (which needs oneMKL and
# is the usual build failure). --all adds the rest of the single-GPU set,
# --multi the ones that need more than one device.
WANT_SINGLE_FULL=0
WANT_MULTI=0
ONLY_MULTI=0
LIST_ONLY=0

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)      BASE_REF=$2; shift 2 ;;
        --trials)    TRIALS=$2; shift 2 ;;
        --all)       WANT_SINGLE_FULL=1; shift ;;
        # --multi on its own means "only the multi-GPU cases"; combined with
        # --all it widens the run instead of narrowing it.
        --multi)     WANT_MULTI=1; ONLY_MULTI=1; shift ;;
        --devices)   DEVICES=$2; shift 2 ;;
        --list)      LIST_ONLY=1; shift ;;
        --keep)      KEEP=1; shift ;;
        --iters)     ITERS=$2; shift 2 ;;
        # Everything perf_gate.py understands: --threshold, --min-ms,
        # --fail-on-noisy, --fail-on-numerics, --json.
        --threshold|--min-ms|--error-factor|--json)
                     GATE_ARGS+=("$1" "$2"); shift 2 ;;
        --fail-on-noisy|--fail-on-numerics)
                     GATE_ARGS+=("$1"); shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "perf_check: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || { echo "perf_check: --trials must be a positive integer" >&2; exit 2; }
if [[ $TRIALS -lt 3 ]]; then
    echo "perf_check: warning: --trials $TRIALS cannot estimate run-to-run spread;" \
         "the gate will fall back to its fixed threshold" >&2
fi

[[ $WANT_SINGLE_FULL == 1 && $WANT_MULTI == 1 ]] && ONLY_MULTI=0

TIER_ARGS=()
if [[ $ONLY_MULTI == 1 ]]; then
    TIER_ARGS=(--tier multi)
    SCOPE=multi
else
    TIER_ARGS=(--tier fast)
    SCOPE=fast
    if [[ $WANT_SINGLE_FULL == 1 ]]; then
        TIER_ARGS+=(--tier full); SCOPE=single-all
    fi
    if [[ $WANT_MULTI == 1 ]]; then
        TIER_ARGS+=(--tier multi); SCOPE="$SCOPE+multi"
    fi
fi

# --- resolve the base commit ----------------------------------------------
if ! BASE_SHA=$(git -C "$SK_ROOT" merge-base HEAD "$BASE_REF" 2>/dev/null); then
    # A fresh clone has no origin/main; fall back to a local branch of that name.
    FALLBACK=${BASE_REF##*/}
    if ! BASE_SHA=$(git -C "$SK_ROOT" merge-base HEAD "$FALLBACK" 2>/dev/null); then
        echo "perf_check: cannot resolve a merge-base with '$BASE_REF' or '$FALLBACK'." >&2
        echo "            Fetch the base branch, or pass --base <ref>." >&2
        exit 2
    fi
    echo "perf_check: '$BASE_REF' not found, using '$FALLBACK'" >&2
    BASE_REF=$FALLBACK
fi

HEAD_SHA=$(git -C "$SK_ROOT" rev-parse HEAD)
DIRTY=$(git -C "$SK_ROOT" status --porcelain --untracked-files=no)
if [[ "$BASE_SHA" == "$HEAD_SHA" && -z "$DIRTY" ]]; then
    echo "perf_check: working tree is identical to the merge-base ($BASE_REF); nothing to check."
    exit 0
fi

# --- toolchain ------------------------------------------------------------
# Source the *current* tree's env.src for both sides. The base checkout may
# predate a change to it, and in any case the A/B is only meaningful if both
# sides see the same compiler.
# shellcheck disable=SC1091
source "$SK_ROOT/env.src"
command -v icpx >/dev/null 2>&1 || {
    echo "perf_check: icpx not found; source an oneAPI environment first" >&2; exit 2
}
# decode includes <oneapi/mkl/blas.hpp>; setvars does not put MKL on CPATH.
[[ -n "${MKLROOT:-}" ]] && export CPATH="$MKLROOT/include${CPATH:+:$CPATH}"

# The AOT target must be pinned explicitly. common.mk only learned to detect the
# local GPU recently, so an older base checkout would silently default to
# intel_gpu_pvc and the comparison would be between two architectures.
if [[ -z "${GPU_TARGET:-}" ]]; then
    if [[ -x "$SK_ROOT/kernels/detect_gpu_target.sh" ]]; then
        GPU_TARGET=$(sh "$SK_ROOT/kernels/detect_gpu_target.sh" icpx)
    fi
    if [[ -z "${GPU_TARGET:-}" ]]; then
        probe=$(mktemp -d)
        printf '#include <sycl/sycl.hpp>\nint main(){return 0;}\n' > "$probe/p.cpp"
        for c in intel_gpu_bmg_g31 intel_gpu_bmg_g21 intel_gpu_pvc spir64; do
            if icpx -fsycl -fsycl-targets="$c" -std=c++20 -o "$probe/p" "$probe/p.cpp" >/dev/null 2>&1; then
                GPU_TARGET=$c; break
            fi
        done
        rm -rf "$probe"
    fi
    GPU_TARGET=${GPU_TARGET:-spir64}
fi
export GPU_TARGET

# --- device count and kernel selection ------------------------------------
# ZE_AFFINITY_MASK wins when a platform adapter set it: that is the count the
# kernels will actually see, regardless of what the machine holds.
if [[ -z "$DEVICES" ]]; then
    if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
        DEVICES=$(awk -F, '{print NF}' <<< "$ZE_AFFINITY_MASK")
    else
        DEVICES=$(sycl-ls 2>/dev/null | grep -c "level_zero.*gpu") || DEVICES=0
    fi
    [[ ${DEVICES:-0} -lt 1 ]] && DEVICES=1
fi
[[ "$DEVICES" =~ ^[1-9][0-9]*$ ]] || {
    echo "perf_check: --devices must be a positive integer, got '$DEVICES'" >&2; exit 2
}

SELECT="$SK_ROOT/benchmarks/tools/select_kernels.py"
# Skips are reported, not hidden: a run that silently measured less than the
# operator believes is worse than one that refused.
BINS=$(python3 "$SELECT" "${TIER_ARGS[@]}" --devices "$DEVICES")
select_rc=$?
if [[ $select_rc -eq 3 ]]; then
    echo "perf_check: no kernel in scope '$SCOPE' runs on $DEVICES GPU(s)." >&2
    echo "            Try --all for the single-GPU set, or run on more devices." >&2
    exit 2
elif [[ $select_rc -ne 0 ]]; then
    echo "perf_check: kernel selection failed" >&2; exit 2
fi

# dir -> make target, widest wins when a dir contributes several binaries.
TARGETS=$(python3 "$SELECT" "${TIER_ARGS[@]}" --devices "$DEVICES" --what targets --quiet)
FAMILIES=$(cut -d' ' -f1 <<< "$TARGETS")
# Needed at run time to decide whether a binary takes timing arguments.
MULTI_BINS=$(python3 "$SELECT" --tier multi --devices "$DEVICES" --quiet | tr '\n' ' ')

# The scalable collectives read this; the fixed-topology ones ignore it.
export SK_NUM_DEVICES=$DEVICES

if [[ $LIST_ONLY == 1 ]]; then
    # Skips were already reported on stderr by the selection above.
    echo "scope    $SCOPE on $DEVICES GPU(s), target $GPU_TARGET"
    python3 "$SELECT" "${TIER_ARGS[@]}" --devices "$DEVICES" --what table --quiet
    exit 0
fi

mkdir -p "$RESULTS_DIR"
RUN_ID="perf_check_$(date -u +%Y%m%dT%H%M%SZ)"
BASE_LOG="$RESULTS_DIR/${RUN_ID}.base.log"
HEAD_LOG="$RESULTS_DIR/${RUN_ID}.head.log"
BASE_TREE=$(mktemp -d "${TMPDIR:-/tmp}/sk-perf-base.XXXXXX")

cleanup() {
    if [[ $KEEP == 1 ]]; then
        echo "perf_check: keeping base worktree at $BASE_TREE" >&2
        return
    fi
    git -C "$SK_ROOT" worktree remove --force "$BASE_TREE" >/dev/null 2>&1 \
        || rm -rf "$BASE_TREE"
}
trap cleanup EXIT

echo "== perf check"
echo "   base     $(git -C "$SK_ROOT" log --oneline -1 "$BASE_SHA")"
echo "   head     $(git -C "$SK_ROOT" log --oneline -1 HEAD)${DIRTY:+ + uncommitted changes}"
echo "   target   $GPU_TARGET"
echo "   devices  $DEVICES visible$([[ -n "${ZE_AFFINITY_MASK:-}" ]] && echo " (ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK)")"
echo "   scope    $SCOPE ($(echo "$BINS" | wc -w) binaries, $(echo "$FAMILIES" | wc -w) families)"
echo "   trials   $TRIALS per side, iters=$ITERS warmup=$WARMUP"
echo

rmdir "$BASE_TREE"   # git worktree add wants to create the directory itself
# git narrates the checkout on stderr; the base commit is already in the header.
git -C "$SK_ROOT" worktree add --detach "$BASE_TREE" "$BASE_SHA" >/dev/null 2>&1 || {
    echo "perf_check: could not create a worktree for $BASE_SHA" >&2; exit 2
}

# --- build ----------------------------------------------------------------
# One family per background job, as baseline.sh does. A build failure on either
# side is fatal: half an A/B is worse than none, because the missing side reads
# as a passing gate.
build_tree() {
    local tree=$1 side=$2 rc=0
    declare -A pid
    while read -r fam tgt; do
        [[ -z "$fam" ]] && continue
        mkdir -p "$tree/kernels/$fam/build"
        make -B -C "$tree/kernels/$fam" "$tgt" CONFIG=standalone GPU_TARGET="$GPU_TARGET" \
            > "$tree/kernels/$fam/build/perf_check_build.log" 2>&1 &
        pid[$fam]=$!
    done <<< "$TARGETS"
    for fam in $FAMILIES; do
        if wait "${pid[$fam]}"; then
            echo "   $side/$fam: OK"
        else
            echo "   $side/$fam: BUILD FAILED"
            tail -n 15 "$tree/kernels/$fam/build/perf_check_build.log" | sed 's/^/     | /'
            rc=1
        fi
    done
    return $rc
}

echo "== build base ($BASE_SHA)"
build_tree "$BASE_TREE" base || exit 2
echo "== build head"
build_tree "$SK_ROOT" head || exit 2

# IGC spilling the mainloop is the single biggest perf factor on this hardware
# and it is only visible at compile time, so surface it rather than leaving it
# in a log nobody opens.
for fam in $FAMILIES; do
    for side in base head; do
        tree=$BASE_TREE; [[ $side == head ]] && tree=$SK_ROOT
        n=$(grep -c "spilled around" "$tree/kernels/$fam/build/perf_check_build.log" 2>/dev/null)
        [[ ${n:-0} -gt 0 ]] && echo "   note: $side/$fam has $n spilling kernel(s)"
    done
done

# --- run ------------------------------------------------------------------
write_header() {
    local path=$1 side=$2 commit=$3
    cat > "$path" <<EOF
SK_BENCHMARK_SCHEMA=1
SK_BENCHMARK_SUITE=perf_check
SK_BENCHMARK_RUN_ID=$RUN_ID
SK_BENCHMARK_TRIALS=$TRIALS
SK_GPU_TARGET=$GPU_TARGET
SK_ITERS=$ITERS
SK_WARMUP=$WARMUP
SK_ICPX_VERSION=$(icpx --version 2>/dev/null | head -1)
SK_PERF_SIDE=$side
SK_PERF_COMMIT=$commit
EOF
}

write_header "$BASE_LOG" base "$(git -C "$SK_ROOT" rev-parse --short "$BASE_SHA")"
write_header "$HEAD_LOG" head "$(git -C "$SK_ROOT" rev-parse --short HEAD)${DIRTY:+-dirty}"

echo
echo "== run"
for ((t = 1; t <= TRIALS; t++)); do
    printf '   trial %d/%d:' "$t" "$TRIALS"
    for b in $BINS; do
        label=${b##*/}
        printf ' %s' "$label"
        # base then head back to back, so thermal and ambient drift lands on
        # both sides of each comparison equally.
        for side in base head; do
            if [[ $side == base ]]; then tree=$BASE_TREE; log=$BASE_LOG
            else tree=$SK_ROOT; log=$HEAD_LOG; fi
            exe="$tree/kernels/${b%%/*}/build/${b#*/}"
            echo "SK_TRIAL label=$label trial=$t" >> "$log"
            if [[ ! -x "$exe" ]]; then
                echo "SK_CASE_SKIPPED label=$label reason=not_built" >> "$log"
                continue
            fi
            # The distributed harnesses size themselves from the topology and
            # take no timing arguments; the single-GPU ones want iters/warmup.
            run_args=("$ITERS" "$WARMUP")
            [[ " $MULTI_BINS " == *" $b "* ]] && run_args=()
            # A collective that cannot reach its peers busy-waits forever, so
            # every binary is bounded. Without this one wedged kernel hangs the
            # whole gate and no verdict is ever produced.
            # Launched as a job and reaped with wait: bash prints "Aborted
            # (core dumped)" for a foreground child killed by a signal, which
            # would land in the middle of the progress line. A crashing kernel
            # is expected input here, not an emergency -- its stderr is already
            # captured in the log.
            timeout --kill-after=10s "$TIMEOUT" "$exe" \
                "${run_args[@]+"${run_args[@]}"}" >> "$log" 2>&1 &
            wait $! 2>/dev/null
            rc=$?
            if [[ $rc -eq 124 || $rc -eq 137 ]]; then
                echo "SK_TRIAL_TIMEOUT label=$label trial=$t after=$TIMEOUT" >> "$log"
                printf '!'
            elif [[ $rc -ne 0 ]]; then
                echo "SK_TRIAL_NONZERO_EXIT label=$label trial=$t rc=$rc" >> "$log"
            fi
        done
    done
    printf '\n'
done

# --- verdict --------------------------------------------------------------
python3 "$SK_ROOT/benchmarks/tools/perf_gate.py" \
    --base "$BASE_LOG" --head "$HEAD_LOG" "${GATE_ARGS[@]+"${GATE_ARGS[@]}"}"
rc=$?

echo
echo "   logs: $BASE_LOG"
echo "         $HEAD_LOG"
exit $rc
