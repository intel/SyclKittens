#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SUITE=${1:-}
TRIALS=${SK_BENCH_TRIALS:-3}
RESULTS_DIR=${SK_BENCH_RESULTS:-$SK_ROOT/benchmarks/results}
FORCE_REBUILD=${SK_FORCE_REBUILD:-1}

usage() {
    cat <<'EOF'
usage: distributed.sh SUITE

SUITE is one of:
  collectives-12tile
  ag-gemm-12tile
  gemm-rs-6gpu
  context-attention-6gpu
EOF
}

case "$SUITE" in
    collectives-12tile|ag-gemm-12tile)
        EXPECTED_DEVICES=12
        ;;
    gemm-rs-6gpu|context-attention-6gpu)
        EXPECTED_DEVICES=6
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if ! [[ "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SK_BENCH_TRIALS must be a positive integer" >&2
    exit 2
fi

MASK=${ZE_AFFINITY_MASK:-}
if [[ -z "$MASK" ]]; then
    echo "ZE_AFFINITY_MASK must be set by the platform adapter" >&2
    exit 2
fi
IFS=',' read -r -a MASK_DEVICES <<< "$MASK"
if [[ ${#MASK_DEVICES[@]} -ne $EXPECTED_DEVICES ]]; then
    echo "suite $SUITE requires $EXPECTED_DEVICES visible devices; mask '$MASK' selects ${#MASK_DEVICES[@]}" >&2
    exit 2
fi

if [[ "$SUITE" == "context-attention-6gpu" ]]; then
    if [[ ${SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY:-0} != 1 ||
          ${UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY:-0} != 1 ]]; then
        echo "context-attention-6gpu requires both Level Zero D2D copy-engine variables to equal 1" >&2
        exit 2
    fi
fi

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="${SUITE}_${TIMESTAMP}"
LOG_PATH="$RESULTS_DIR/${RUN_ID}.log"
MANIFEST_PATH="$RESULTS_DIR/${RUN_ID}.manifest.json"

source "$SK_ROOT/env.src"
export ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}

python3 "$SK_ROOT/benchmarks/tools/capture_manifest.py" \
    --suite "$SUITE" --output "$MANIFEST_PATH"

exec > >(tee "$LOG_PATH") 2>&1

echo "SK_BENCHMARK_SCHEMA=1"
echo "SK_BENCHMARK_SUITE=$SUITE"
echo "SK_BENCHMARK_RUN_ID=$RUN_ID"
echo "SK_BENCHMARK_TRIALS=$TRIALS"
echo "ZE_FLAT_DEVICE_HIERARCHY=$ZE_FLAT_DEVICE_HIERARCHY"
echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo "MANIFEST=$MANIFEST_PATH"

do_make() {
    local directory=$1
    shift
    if [[ "$FORCE_REBUILD" == 1 ]]; then
        make -B -C "$directory" "$@"
    else
        make -C "$directory" "$@"
    fi
}

run_trials() {
    local label=$1
    local binary=$2
    local trial
    for ((trial = 1; trial <= TRIALS; trial++)); do
        echo "${label}_TRIAL=$trial"
        "$binary"
    done
}

case "$SUITE" in
    collectives-12tile)
        do_make "$SK_ROOT/kernels/collective" standalone
        run_trials ALL_REDUCE "$SK_ROOT/kernels/collective/build/all_reduce"
        run_trials ALL_GATHER "$SK_ROOT/kernels/collective/build/all_gather"
        run_trials REDUCE_SCATTER "$SK_ROOT/kernels/collective/build/reduce_scatter"
        run_trials ALL_TO_ALL "$SK_ROOT/kernels/collective/build/all_to_all"
        ;;
    ag-gemm-12tile)
        do_make "$SK_ROOT/kernels/collective" ag_gemm
        run_trials AG_GEMM "$SK_ROOT/kernels/collective/build/ag_gemm"
        ;;
    gemm-rs-6gpu)
        do_make "$SK_ROOT/kernels/collective" gemm_rs
        run_trials GEMM_RS "$SK_ROOT/kernels/collective/build/gemm_rs"
        ;;
    context-attention-6gpu)
        do_make "$SK_ROOT/kernels/context_attention" all
        run_trials RING "$SK_ROOT/kernels/context_attention/build/ring_attention"
        run_trials OVERLAP_RING "$SK_ROOT/kernels/context_attention/build/ring_attention_overlap"
        run_trials ALL_GATHER_ATTENTION "$SK_ROOT/kernels/context_attention/build/all_gather_attention"
        ;;
esac

echo "SK_BENCHMARK_COMPLETED suite=$SUITE run_id=$RUN_ID"
