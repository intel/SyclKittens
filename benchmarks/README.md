# SyclKittens Benchmarks

This directory contains scheduler-independent benchmark contracts for the
public kernel release. Platform allocation, host selection, scheduler options,
and raw campaign evidence are deliberately kept outside this repository.

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
