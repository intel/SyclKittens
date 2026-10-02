/**
 * @file
 * @brief 12-tile All-Reduce for PVC (FLAT hierarchy).
 *
 * 12-tile All-Reduce benchmark: contiguous tile selection (torch set_device
 * 0..11) and the standard algorithm-bandwidth convention:
 *     algo_bw = 2*(world-1)/world * N*sizeof(bf16) / time.
 *
 * Semantics: every tile has an N-element input; every tile ends with the
 * elementwise sum across all `world` tiles.
 *
 * Dispatch is payload-adaptive: direct full reduction for latency-bound small
 * messages and fused balanced reduce-scatter/all-gather for larger messages.
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "all_reduce_kernel.hpp"
#include "dispatch.hpp"
#include "topology.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int NUM_DEVICES  = 12;
static constexpr int WARMUP       = 5;
static constexpr int ITERS        = 15;

int main() {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &p : platforms) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto &d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);
    }

    int requested_gpus = NUM_DEVICES;
    if (const char *value = std::getenv("SK_NUM_DEVICES"))
        requested_gpus = std::max(2, std::min(NUM_DEVICES, std::atoi(value)));
    int n_gpus = std::min((int)gpus.size(), requested_gpus);
    std::cout << "All-Reduce 12: " << n_gpus << " GPUs" << std::endl;
    if (n_gpus < 2) { std::cout << "Need >= 2 GPUs" << std::endl; return 1; }

    std::vector<sycl::device> sel(gpus.begin(), gpus.begin() + n_gpus);
    std::vector<std::string> uuid_strs(n_gpus);
    for (int i = 0; i < n_gpus; i++) {
        std::string us;
        try {
            auto uu = sel[i].get_info<sycl::ext::intel::info::device::uuid>();
            char b[4];
            for (auto by : uu) { snprintf(b, sizeof(b), "%02x", (unsigned)by); us += b; }
        } catch (...) { us = "?"; }
        uuid_strs[i] = us;
    }
    auto pvc_of = kittens::topo::group_pvc(uuid_strs);
    int npvc = 0; for (int p : pvc_of) npvc = std::max(npvc, p + 1);
    std::vector<std::vector<int>> tiles(npvc);
    for (int t = 0; t < n_gpus; t++) tiles[pvc_of[t]].push_back(t);
    std::cout << "  npvc=" << npvc << "\n";

    sycl::context ctx(sel);
    std::vector<sycl::queue> queues;
    for (int i = 0; i < n_gpus; i++)
        queues.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});

    // XCCL All-Reduce sizes (total elems == per-tile N; all divisible by 12).
    for (int N : {12288, 98304, 786432, 3145728, 6291456, 12582912,
                  50331648}) {
        if (std::getenv("SK_ONLY_96") && N != 50331648) continue;
        if (std::getenv("SK_ONLY_12") && N != 6291456) continue;
        int chunk = N / n_gpus;             // reduce-scatter sub-vector size

        std::vector<bf16*> d_in(n_gpus), d_res(n_gpus), d_par(n_gpus);
        for (int d = 0; d < n_gpus; d++) {
            d_in[d]  = sycl::malloc_device<bf16>(N, queues[d]);
            d_res[d] = sycl::malloc_device<bf16>(N, queues[d]);
            d_par[d] = sycl::malloc_device<bf16>(N, queues[d]);
            if (!d_in[d] || !d_res[d] || !d_par[d]) {
                std::cerr << "  FATAL: malloc_device failed d=" << d << std::endl;
                return 1;
            }
        }
        for (int d = 0; d < n_gpus; d++) {
            bf16 *ptr = d_in[d]; int n = N, src = d;
            queues[d].parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                ptr[i] = static_cast<bf16>((src * 7 + static_cast<int>(i[0])) % 9 - 4);
            });
        }
        for (auto &q : queues) q.wait();

        auto run_N_fn = [&]() {
            kittens::collective::all_reduce::run_full_reduction(
                queues, d_in, d_res, n_gpus, N);
        };

        auto run_B_fn = [&]() {
            kittens::collective::all_reduce::run_balanced_rs_ag(
                queues, d_in, d_res, d_par, npvc, tiles, n_gpus, chunk);
        };

        auto measure = [&](const char *label, auto &&run_fn) {
            for (int w = 0; w < WARMUP; w++) run_fn();
            std::vector<double> times;
            for (int i = 0; i < ITERS; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                run_fn();
                auto t1 = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(times.begin(), times.end());
            auto percentile = [&](double q) {
                return times[static_cast<size_t>(q * (times.size() - 1) + 0.5)];
            };
            double p10_us = percentile(0.10);
            double med_us = percentile(0.50);
            double p90_us = percentile(0.90);

            std::vector<bf16> h_out(N);
            float max_diff = 0.0f;
            size_t mismatches = 0;
            for (int d = 0; d < n_gpus; d++) {
                queues[d].memcpy(h_out.data(), d_res[d], N * sizeof(bf16)).wait();
                for (int i = 0; i < N; i++) {
                    float expected = 0.0f;
                    for (int s = 0; s < n_gpus; s++)
                        expected += static_cast<float>((s * 7 + i) % 9 - 4);
                    float diff = std::abs(static_cast<float>(h_out[i]) - expected);
                    max_diff = std::max(max_diff, diff);
                    mismatches += diff >= 0.02f;
                }
            }

            double data_bytes = (double)N * sizeof(bf16);
            double algo_bw = 2.0 * (n_gpus - 1) / n_gpus * data_bytes / (med_us * 1e-6) / 1e9;
            double size_mb = data_bytes / (1024 * 1024);
            std::cout << "  [" << label << "]"
                      << "  N=" << N << "  size=" << size_mb << "MB"
                      << "  p10/p50/p90=" << (int)p10_us << "/" << (int)med_us
                      << "/" << (int)p90_us << " us"
                      << "  BW=" << algo_bw << " GB/s"
                      << "  " << (mismatches == 0 ? "PASS" : "FAIL")
                      << " mismatches=" << mismatches
                      << " max_diff=" << max_diff
                      << std::endl;
        };

        size_t nb = (size_t)N * sizeof(bf16);
        const auto mode = kittens::collective::all_reduce::select_mode(n_gpus, nb);
        if (mode == kittens::collective::all_reduce::Mode::DirectFullReduction)
            measure("AR-adaptive(N)", run_N_fn);
        else
            measure("AR-adaptive(B)", run_B_fn);

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_in[d],  queues[d]);
            sycl::free(d_res[d], queues[d]);
            sycl::free(d_par[d], queues[d]);
        }
    }
    return 0;
}
