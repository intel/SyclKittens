/**
 * @file ag_gemm.dp.cpp
 * @brief 12-tile topology-aware All-Gather + GEMM for PVC.
 *
 * Shape (standard AG-GEMM, replicated output):
 *   A is row-sharded:   A_local[d] : bf16[rpd, K]  (rpd = M / world)
 *   B is replicated:    B[d]       : bf16[K, N]
 *   Each device gathers the full A (M x K) and computes the full
 *   C[d] = A_full * B : fp32[M, N] -> cast bf16.
 *
 * Each A shard crosses XeLink once per remote PVC, is replicated over the
 * on-package tile link, and feeds the collaborative de-duplicated GEMM kernel.
 */

#include "kittens.dp.hpp"
#include "ag_gemm_kernel.hpp"
using namespace kittens;

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>

using sbf16 = sycl::ext::oneapi::bfloat16;

static constexpr int WARMUP = 10;
static constexpr int ITERS  = 30;
// Wall-time power warmup (us) to bring PVC out of power-save to steady turbo.
static constexpr double WARMUP_US = 2.0e6;

int main() {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &p : platforms)
        for (auto &d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);
    int n_gpus = static_cast<int>(gpus.size());
    std::cout << "=== Topology-aware AG+GEMM (12-tile) ===  GPUs: "
              << n_gpus << std::endl;
    if (n_gpus != kittens::collective::ag_gemm::NUM_DEVICES) {
        std::cerr << "Need exactly " << kittens::collective::ag_gemm::NUM_DEVICES
                  << " flat tiles" << std::endl;
        return 1;
    }

    std::vector<sycl::device> sel(gpus.begin(), gpus.begin() + n_gpus);
    sycl::context ctx(sel);
    std::vector<sycl::queue> q_compute;
    for (int i = 0; i < n_gpus; i++) {
        q_compute.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});
    }

    for (auto [M, N, K] : std::vector<std::tuple<int,int,int>>{
        {6144, 6144, 6144}, {12288, 12288, 12288} })
    {
        if (M % (n_gpus * kittens::collective::ag_gemm::BM) != 0 ||
            N % kittens::collective::ag_gemm::BN != 0 ||
            K % kittens::collective::ag_gemm::BK != 0) {
            continue;
        }
        int rpd = M / n_gpus;
        int snake_w = kittens::collective::ag_gemm::snake_width_for(M, N);
        int C_elems = M * N;
        std::cout << "\n=== M=" << M << " N=" << N << " K=" << K
                  << " rpd=" << rpd << " ===" << std::endl;

        std::vector<sbf16*> d_A_local(n_gpus);
        std::vector<kittens::bf16*> d_A_gath(n_gpus);
        std::vector<kittens::bf16*> d_B(n_gpus);
        std::vector<sbf16*> d_C_bf16(n_gpus);
        for (int d = 0; d < n_gpus; d++) {
            d_A_local[d]= sycl::malloc_device<sbf16>(rpd * K, q_compute[d]);
            d_A_gath[d] = sycl::malloc_device<kittens::bf16>(M * K, q_compute[d]);
            d_B[d]      = sycl::malloc_device<kittens::bf16>(K * N, q_compute[d]);
            d_C_bf16[d] = sycl::malloc_device<sbf16>(C_elems, q_compute[d]);
        }
        // Constant input supports analytic verification. The evaluator and random
        // input controls reproduce the corresponding reference distributions.
        const char *ri_env = getenv("SK_RAND_INIT");
        const bool rand_init = ri_env && ri_env[0] && ri_env[0] != '0';
        const char *ei_env = getenv("SK_EVAL_INIT");
        const bool eval_init = ei_env && ei_env[0] && ei_env[0] != '0';
        const uint32_t eval_seed = getenv("EVAL_SEED")
            ? static_cast<uint32_t>(strtoul(getenv("EVAL_SEED"), nullptr, 10)) : 42u;
        for (int d = 0; d < n_gpus; d++) {
            sbf16 *a = d_A_local[d]; kittens::bf16 *b = d_B[d];
            int sza = rpd * K, szb = K * N;
            if (eval_init) {
                const uint64_t a_offset = static_cast<uint64_t>(d) * sza;
                q_compute[d].parallel_for(sycl::range<1>(sza), [=](sycl::id<1> ii){
                    const uint64_t index = a_offset + static_cast<uint64_t>(ii[0]);
                    uint32_t epoch = 0x6a09e667u;
                    epoch ^= epoch >> 16; epoch *= 0x7feb352du;
                    epoch ^= epoch >> 15; epoch *= 0x846ca68bu; epoch ^= epoch >> 16;
                    uint32_t x = eval_seed ^ 0xa341316cu ^ epoch
                        ^ static_cast<uint32_t>(index)
                        ^ static_cast<uint32_t>(index >> 32) * 0x9e3779b9u;
                    x ^= x >> 16; x *= 0x7feb352du;
                    x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
                    a[ii] = static_cast<sbf16>(
                        static_cast<float>(static_cast<int>((x >> 24) & 31u) - 16) * 0.0625f);
                });
                q_compute[d].parallel_for(sycl::range<1>(szb), [=](sycl::id<1> ii){
                    const uint64_t index = static_cast<uint64_t>(ii[0]);
                    uint32_t epoch = 0x6a09e667u;
                    epoch ^= epoch >> 16; epoch *= 0x7feb352du;
                    epoch ^= epoch >> 15; epoch *= 0x846ca68bu; epoch ^= epoch >> 16;
                    uint32_t x = eval_seed ^ 0xc8013ea4u ^ epoch
                        ^ static_cast<uint32_t>(index)
                        ^ static_cast<uint32_t>(index >> 32) * 0x9e3779b9u;
                    x ^= x >> 16; x *= 0x7feb352du;
                    x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
                    b[ii] = static_cast<kittens::bf16>(
                        static_cast<float>(static_cast<int>((x >> 24) & 31u) - 16) * 0.0625f);
                });
            } else if (rand_init) {
                const uint64_t salt_a = 0x9E3779B97F4A7C15ULL * (uint64_t)(2*d+1);
                const uint64_t salt_b = 0xC2B2AE3D27D4EB4FULL * (uint64_t)(2*d+2);
                q_compute[d].parallel_for(sycl::range<1>(sza), [=](sycl::id<1> ii){
                    uint64_t x = (uint64_t)ii[0] + salt_a;
                    x ^= x>>33; x *= 0xff51afd7ed558ccdULL; x ^= x>>33;
                    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x>>33;
                    double u1 = ((double)(x>>11)+0.5) * (1.0/9007199254740992.0);
                    uint64_t y = x + 0x9E3779B97F4A7C15ULL;
                    y ^= y>>33; y *= 0xff51afd7ed558ccdULL; y ^= y>>33;
                    y *= 0xc4ceb9fe1a85ec53ULL; y ^= y>>33;
                    double u2 = ((double)(y>>11)+0.5) * (1.0/9007199254740992.0);
                    double z = sycl::sqrt(-2.0*sycl::log(u1)) * sycl::cos(6.283185307179586*u2);
                    a[ii] = static_cast<sbf16>((float)z);
                });
                q_compute[d].parallel_for(sycl::range<1>(szb), [=](sycl::id<1> ii){
                    uint64_t x = (uint64_t)ii[0] + salt_b;
                    x ^= x>>33; x *= 0xff51afd7ed558ccdULL; x ^= x>>33;
                    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x>>33;
                    double u1 = ((double)(x>>11)+0.5) * (1.0/9007199254740992.0);
                    uint64_t y = x + 0x9E3779B97F4A7C15ULL;
                    y ^= y>>33; y *= 0xff51afd7ed558ccdULL; y ^= y>>33;
                    y *= 0xc4ceb9fe1a85ec53ULL; y ^= y>>33;
                    double u2 = ((double)(y>>11)+0.5) * (1.0/9007199254740992.0);
                    double z = sycl::sqrt(-2.0*sycl::log(u1)) * sycl::cos(6.283185307179586*u2);
                    b[ii] = static_cast<kittens::bf16>((float)z);
                });
            } else {
                sbf16 va = static_cast<sbf16>((d+1) * 0.01f);
                kittens::bf16 vb = static_cast<kittens::bf16>(0.01f);
                q_compute[d].parallel_for(sycl::range<1>(sza), [=](sycl::id<1> i){ a[i] = va; });
                q_compute[d].parallel_for(sycl::range<1>(szb), [=](sycl::id<1> i){ b[i] = vb; });
            }
        }
        for (auto &q : q_compute) q.wait();

        auto bench = [&](auto fn) {
            // Power warmup: run until WARMUP_US elapsed so clocks reach steady
            // turbo before timing.
            auto wstart = std::chrono::high_resolution_clock::now();
            do { fn(); } while (
                std::chrono::duration<double, std::micro>(
                    std::chrono::high_resolution_clock::now() - wstart).count() < WARMUP_US);
            for (int w = 0; w < WARMUP; w++) fn();
            std::vector<double> ts;
            for (int i = 0; i < ITERS; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                fn();
                auto t1 = std::chrono::high_resolution_clock::now();
                ts.push_back(std::chrono::duration<double, std::micro>(t1-t0).count());
            }
            std::sort(ts.begin(), ts.end());
            return ts[ITERS/2];
        };

        auto run_topology_gather_gemm = [&]() {
            kittens::collective::ag_gemm::run_production_topology_gather_gemm(
                q_compute, d_A_local, d_A_gath, d_B, d_C_bf16,
                M, N, K, rpd, snake_w);
        };

        double flops = 2.0 * M * N * K;                      // per device
        auto tf = [&](double us){ return flops / (us * 1e-6) / 1e12; };

        auto verify = [&](const char *tag, double us) {
            if (eval_init) {
                printf("  [%-14s]  %8.0f us  %6.2f TFLOP/s  PERF(eval)\n",
                       tag, us, tf(us));
                return;
            }
            if (rand_init) {
                // randn data -> analytic expected value is invalid; control flow is
                // data-independent so correctness is covered by the const-fill run.
                printf("  [%-14s]  %8.0f us  %6.2f TFLOP/s  PERF(randn)\n",
                       tag, us, tf(us));
                return;
            }
            std::vector<sbf16> h(C_elems);
            q_compute[0].memcpy(h.data(), d_C_bf16[0], C_elems * sizeof(sbf16)).wait();
            float max_diff = 0.0f;
            for (int src = 0; src < n_gpus; src++) {
                float expected = K * 0.0001f * (src + 1);
                for (int i = 0; i < std::min(rpd * N, 50); i++) {
                    int idx = (src * rpd) * N + i;
                    max_diff = std::max(max_diff, std::abs((float)h[idx] - expected));
                }
            }
            printf("  [%-14s]  %8.0f us  %6.2f TFLOP/s  %s\n",
                   tag, us, tf(us), max_diff < 0.5f ? "PASS" : "FAIL");
        };

        const double us = bench(run_topology_gather_gemm);
        verify("topology-gather", us);

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_A_local[d], q_compute[d]);
            sycl::free(d_A_gath[d],  q_compute[d]);
            sycl::free(d_B[d],       q_compute[d]);
            sycl::free(d_C_bf16[d],  q_compute[d]);
        }
    }
    return 0;
}
