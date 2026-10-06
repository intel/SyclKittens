# SyclKittens

SyclKittens is a header-only tile programming library for writing fast Intel GPU
kernels in SYCL. It follows the
[ThunderKittens 2.0](https://github.com/HazyResearch/ThunderKittens) repository
model ([announcement](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)):
reusable primitives live under `include/`, kernels carry their own build files,
and the API test suite lives under `tests/`.

Paper: [SyclKittens: A Tile Programming Model for Programmers and Coding Agents on Intel GPUs](https://arxiv.org/abs/2610.04277) (arXiv:2610.04277).

## Requirements

- Intel oneAPI DPC++ compiler with `icpx` and SYCL 2020 support
- Intel Data Center GPU Max (PVC) for the supplied AOT target
- C++20
- XPU-enabled PyTorch and pybind11 for Python extension kernels
- oneMKL for the decode GEMV fallback
- Level Zero development headers for the direct-copy all-gather and distributed
  GEMM targets; set
  `LEVEL_ZERO_INCLUDE=-I/path/to/include` when they are outside the compiler path

To configure the build environment, run:

```bash
source env.src
```

## Layout

```text
SyclKittens/
├── include/                  # Header-only tile library
├── kernels/
│   ├── attention/            # Adaptive GQA forward and fused backward
│   ├── gemm/                 # BF16 GEMM and fused QKV projection
│   ├── norm/                 # RMSNorm and LayerNorm
│   ├── rotary/               # RoPE, fused Q/K, and fused RoPE-K/cache write
│   ├── activation/           # Fused SiLU multiply
│   ├── decode/               # Registered TP/CP decode with automatic GEMV fallback
│   ├── collective/           # Production *_kernel.hpp plus executable harnesses
│   └── context_attention/    # Production .dp.cpp plus *_bench.dp.cpp harnesses
├── tests/                    # Reusable API unit tests
├── benchmarks/               # Portable suites and result schemas
├── demos/                    # Minimal usage guidance
└── docs/                     # API documentation
```

The distributed targets keep production launch and dispatch code physically
separate from allocation, initialization, timing, correctness checks, reporting,
and `main()`. Collective production code lives in `*_kernel.hpp`; context
attention production code lives in the non-`_bench` `.dp.cpp` files. Each local
benchmark harness includes its production implementation into one translation
unit, preserving the validated SYCL device-code compilation boundary.

## Build

List the supported targets:

```bash
make help
```

Build the header/API test suite:

```bash
make tests
```

Build one operation family. Each family writes only to its local `build/`
directory.

```bash
make -C kernels/attention forward
make -C kernels/gemm bf16 M=4096 N=4096 K=4096
make -C kernels/decode decode
make -C kernels/collective all_reduce
make -C kernels/context_attention ring_overlap
```

Python extension targets use `PYTHON` to select an XPU-enabled environment:

```bash
make -C kernels/norm all \
  PYTHON=/path/to/xpu/python3 RMSNORM_D=4096 LN_D=4096
```

## API

Include the master header and keep kernels in the shared `kittens` namespace:

```cpp
#include "kittens.dp.hpp"

using namespace kittens;
```

The collective construction API is documented in
[docs/collective_api.md](docs/collective_api.md).
