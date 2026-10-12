#!/usr/bin/env bash
# Build every kernel in standalone mode and run its harness; prints a PASS/FAIL summary.
#
#   source ~/oneapi.sh                   # or env.src: icpx (and oneMKL for decode) on PATH
#   ./kernels/run_standalone.sh [iters=50] [warmup=5]
#
# Pick the GPU with ZE_AFFINITY_MASK. Extra make variables go in MAKE_ARGS,
# e.g. MAKE_ARGS="GPU_TARGET=intel_gpu_pvc". Exit code is non-zero if anything fails.
set -u
cd "$(dirname "$0")"
ITERS=${1:-50}
WARMUP=${2:-5}

command -v icpx > /dev/null || { echo "icpx not found: source your oneAPI environment first" >&2; exit 1; }
# decode includes <oneapi/mkl/blas.hpp>; oneAPI setvars does not put it on CPATH.
[ -n "${MKLROOT:-}" ] && export CPATH="$MKLROOT/include${CPATH:+:$CPATH}"

DIRS="gemm attention norm rotary activation decode"
# dir/binary for every harness, built by `make -C <dir> all CONFIG=standalone`
BINS="gemm/bf16_gemm_4096x4096x4096 gemm/qkv_gemm attention/gqa_forward attention/gqa_backward
      norm/rmsnorm norm/layernorm rotary/rotary rotary/rotary_qk rotary/rope_kvwrite
      activation/silu_mul decode/decode"

echo "== build (CONFIG=standalone), logs in <dir>/build/standalone.log"
declare -A pid
for d in $DIRS; do
    mkdir -p "$d/build"
    make -C "$d" all CONFIG=standalone ${MAKE_ARGS:-} > "$d/build/standalone.log" 2>&1 &
    pid[$d]=$!
done
build_fail=0; broken=" "
for d in $DIRS; do
    if ! wait "${pid[$d]}"; then
        echo "   $d: BUILD FAILED (see $d/build/standalone.log)"
        build_fail=1; broken="$broken$d "
    fi
done

echo "== run (iters=$ITERS warmup=$WARMUP)"
pass=0; fail=0; failed=""
for b in $BINS; do
    d="${b%%/*}"; exe="$d/build/${b#*/}"
    # Never run an executable left over from an earlier build of a failed directory.
    if [[ "$broken" == *" $d "* ]] || [ ! -x "$exe" ]; then
        echo "-- $b: not built"; fail=$((fail + 1)); failed="$failed $b"; continue
    fi
    out=$("./$exe" "$ITERS" "$WARMUP" 2>&1); rc=$?
    echo "-- $b"
    echo "$out" | grep -E "PASS|FAIL" | sed 's/^/   /'
    if [ $rc -eq 0 ] && ! echo "$out" | grep -q FAIL; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); failed="$failed $b"
        [ $rc -ne 0 ] && echo "$out" | tail -n 3 | sed 's/^/   ! /'
    fi
done

echo "== summary: $pass passed, $fail failed${failed:+ ($failed )}"
[ $fail -eq 0 ] && [ $build_fail -eq 0 ]
