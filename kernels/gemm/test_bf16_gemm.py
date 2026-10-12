#!/usr/bin/env python3
"""Correctness + performance check for the SyclKittens bf16 GEMM extension.

Build first (the shape is baked in at compile time):
    make -C kernels/gemm bf16 M=4096 N=4096 K=4096 PYTHON=$(which python3)
Then run:
    python3 kernels/gemm/test_bf16_gemm.py [--iters 50] [--warmup 10]
"""
import argparse
import pathlib
import sys

import torch

BUILD_DIR = pathlib.Path(__file__).resolve().parent / "build"
sys.path.insert(0, str(BUILD_DIR))
import sk_bf16_gemm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fp32-out", action="store_true",
                    help="use when built with -DGEMM_FP32_OUT")
    args = ap.parse_args()

    M, N, K = sk_bf16_gemm.M, sk_bf16_gemm.N, sk_bf16_gemm.K
    dev = torch.device("xpu")
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    out_dtype = torch.float32 if args.fp32_out else torch.bfloat16
    C = torch.empty(M, N, dtype=out_dtype, device=dev)

    # Correctness vs. fp32 reference.
    sk_bf16_gemm.dispatch_gemm(A, B, C)
    torch.xpu.synchronize()
    ref = A.float() @ B.float()
    err = (C.float() - ref).abs()
    rel = err.max().item() / ref.abs().max().item()
    ok = rel < 2e-2
    print(f"shape M={M} N={N} K={K}  max_abs_err={err.max().item():.4f} "
          f"mean_abs_err={err.mean().item():.4f} rel_err={rel:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # Performance.
    flops = 2.0 * M * N * K

    def bench(fn):
        for _ in range(args.warmup):
            fn()
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.xpu.synchronize()
        return start.elapsed_time(end) / args.iters  # ms

    sk_ms = bench(lambda: sk_bf16_gemm.dispatch_gemm(A, B, C))
    th_ms = bench(lambda: torch.matmul(A, B))
    print(f"SyclKittens: {sk_ms:.3f} ms  {flops / sk_ms / 1e9:.1f} TFLOPS")
    print(f"torch.matmul: {th_ms:.3f} ms  {flops / th_ms / 1e9:.1f} TFLOPS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
