/**
 * @file
 * @brief 12-tile All-to-All for PVC (FLAT hierarchy).
 *
 * Contiguous tile selection (torch set_device 0..11) and the bandwidth
 * convention:
 *     BW = world*(world-1)*chunk*sizeof(bf16) / time.
 *
 * Semantics (transpose scatter): each tile owns `world` chunks of `chunk`
 * elements. Tile s's chunk d is delivered to tile d's output slot s:
 *     out[d][s*chunk .. ] = in[s][d*chunk .. ].
 *
 * A2A has no reduction and no replication (every chunk has a unique
 * destination+slot), so the topology dedup lever does not apply. The optimization
 * is the fused single-kernel-per-tile EU-copy path that kills the launch/latency
 * floor plus by-value dispatch tables.
 *
 * Each work-group writes one destination so independent remote endpoints can
 * make progress concurrently.
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "all_to_all_kernel.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int NUM_DEVICES = 12;
static constexpr int WARMUP      = 5;
static constexpr int ITERS       = 15;

int main() {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &p : platforms)
        for (auto &d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);

    int n_gpus = std::min((int)gpus.size(), NUM_DEVICES);
    std::cout << "All-to-All 12: " << n_gpus << " GPUs" << std::endl;
    if (n_gpus < 2) { std::cout << "Need >= 2 GPUs" << std::endl; return 1; }

    std::vector<sycl::device> sel(gpus.begin(), gpus.begin() + n_gpus);
    sycl::context ctx(sel);
    std::vector<sycl::queue> queues;
    for (int i = 0; i < n_gpus; i++)
        queues.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});

    for (int tpd : {12288, 98304, 786432, 3145728, 6291456, 12582912,
                    50331648}) {
        if (std::getenv("SK_ONLY_12") && tpd != 6291456) continue;
        if (std::getenv("SK_ONLY_96") && tpd != 50331648) continue;
        int chunk = tpd / n_gpus;

        std::vector<bf16*> d_in(n_gpus), d_out(n_gpus);
        for (int d = 0; d < n_gpus; d++) {
            d_in[d]  = sycl::malloc_device<bf16>(tpd, queues[d]);
            d_out[d] = sycl::malloc_device<bf16>(tpd, queues[d]);
            if (!d_in[d] || !d_out[d]) {
                std::cerr << "  FATAL: malloc_device failed d=" << d << std::endl;
                return 1;
            }
        }
        for (int s = 0; s < n_gpus; s++) {
            bf16 *ptr = d_in[s]; int cs = chunk, sid = s, ng = n_gpus;
            queues[s].parallel_for(sycl::range<1>(tpd), [=](sycl::id<1> idx) {
                int offset = static_cast<int>(idx[0]) % cs;
                int d = static_cast<int>(idx[0]) / cs;
                ptr[idx] = static_cast<bf16>((sid * 17 + d * 5 + offset) % 97 - 48);
            });
        }
        for (auto &q : queues) q.wait();

        auto run_all_to_all = [&]() {
            kittens::collective::all_to_all::run(
                queues, d_in, d_out, chunk);
        };

        auto measure = [&](const char *label, auto &&run_fn) {
            for (auto &q : queues) q.wait();
            for (int d = 0; d < n_gpus; d++)
                queues[d].memset(d_out[d], 0, tpd * sizeof(bf16));
            for (auto &q : queues) q.wait();

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

            std::vector<bf16> h_out(tpd);
            float max_diff = 0.0f;
            size_t mismatches = 0;
            for (int d = 0; d < n_gpus; d++) {
                queues[d].memcpy(h_out.data(), d_out[d], tpd * sizeof(bf16)).wait();
                for (int s = 0; s < n_gpus; s++) {
                    for (int i = 0; i < chunk; i++) {
                        float expected = static_cast<float>((s * 17 + d * 5 + i) % 97 - 48);
                        float diff = std::abs(static_cast<float>(h_out[(size_t)s * chunk + i]) - expected);
                        max_diff = std::max(max_diff, diff);
                        mismatches += diff != 0.0f;
                    }
                }
            }

            double p2p_bytes = (double)n_gpus * (n_gpus - 1) * chunk * sizeof(bf16);
            double bw = p2p_bytes / (med_us * 1e-6) / 1e9;
            double size_mb = (double)tpd * 2.0 / (1024 * 1024);
            std::cout << "  [" << label << "]"
                      << "  tpd=" << tpd << "  size=" << size_mb << "MB"
                      << "  p10/p50/p90=" << (int)p10_us << "/" << (int)med_us
                      << "/" << (int)p90_us << " us"
                      << "  BW=" << bw << " GB/s"
                      << "  " << (mismatches == 0 ? "PASS" : "FAIL")
                      << " mismatches=" << mismatches
                      << " max_diff=" << max_diff
                      << std::endl;
        };

        measure("A2A-S split-dst   ", run_all_to_all);

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_in[d],  queues[d]);
            sycl::free(d_out[d], queues[d]);
        }
    }
    return 0;
}
