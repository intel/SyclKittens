/**
 * @file
 * @brief Payload-adaptive topology-aware All-Gather for paired PVC tiles.
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "all_gather_kernel.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int NUM_DEVICES = 12;
static constexpr int WG_SIZE     = 256;
static constexpr int EPT         = 8;
static constexpr int VEC_BF16    = 8;
static constexpr int WARMUP      = 5;
static constexpr int ITERS       = 15;

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
    std::cout << "All-Gather: " << n_gpus << " GPUs" << std::endl;
    if (n_gpus < 2) { std::cout << "Need >= 2 GPUs" << std::endl; return 1; }

    std::vector<sycl::device> sel(gpus.begin(), gpus.begin() + n_gpus);
    std::cout << "  device-select: CONTIGUOUS first-" << n_gpus << " tiles\n";
    sycl::context ctx(sel);
    std::vector<sycl::queue> queues;
    for (int i = 0; i < n_gpus; i++)
        queues.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});

    for (int total_target : {12288, 98304, 786432, 3145728, 6291456, 12582912,
                             50331648}) {
        if (std::getenv("SK_ONLY_96") && total_target != 50331648) continue;
        if (std::getenv("SK_ONLY_12") && total_target != 6291456) continue;
        int epd = ((total_target / n_gpus + VEC_BF16 - 1) / VEC_BF16) * VEC_BF16;
        int total = epd * n_gpus;
        int vpd = epd / VEC_BF16;
        int n_wgs = std::max(1, (vpd + WG_SIZE * EPT - 1) / (WG_SIZE * EPT));

        std::vector<bf16*> d_in(n_gpus), d_out(n_gpus);
        for (int d = 0; d < n_gpus; d++) {
            d_in[d]  = sycl::malloc_device<bf16>(epd, queues[d]);
            d_out[d] = sycl::malloc_device<bf16>(total, queues[d]);
        }

        for (int d = 0; d < n_gpus; d++) {
            bf16 *ptr = d_in[d]; int ep = epd, src = d;
            queues[d].parallel_for(sycl::range<1>(ep), [=](sycl::id<1> i) {
                ptr[i] = static_cast<bf16>((src * 17 + static_cast<int>(i[0])) % 97 - 48);
            });
        }
        for (auto &q : queues) q.wait();

        const size_t bytes = (size_t)total * sizeof(bf16);
        const auto mode = kittens::collective::all_gather::select_mode(n_gpus, bytes);
        const char *label = kittens::collective::all_gather::mode_label(mode);
        auto run_selected = [&]() {
            kittens::collective::all_gather::run_mode(
                mode, queues, d_in, d_out, n_gpus, epd, vpd, n_wgs);
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

            for (int d = 0; d < n_gpus; d++)
                queues[d].memset(d_out[d], 0xa5, total * sizeof(bf16));
            for (auto &queue : queues) queue.wait();
            run_fn();

            std::vector<bf16> h_out(total);
            float max_diff = 0.0f;
            size_t mismatches = 0;
            for (int d = 0; d < n_gpus; d++) {
                queues[d].memcpy(h_out.data(), d_out[d], total * sizeof(bf16)).wait();
                for (int s = 0; s < n_gpus; s++) {
                    for (int i = 0; i < epd; i++) {
                        float expected = static_cast<float>((s * 17 + i) % 97 - 48);
                        float diff = std::abs(static_cast<float>(h_out[(size_t)s * epd + i]) - expected);
                        max_diff = std::max(max_diff, diff);
                        mismatches += diff != 0.0f;
                    }
                }
            }
            double p2p_bytes = (double)n_gpus * (n_gpus - 1) * epd * sizeof(bf16);
            double bw = p2p_bytes / (med_us * 1e-6) / 1e9;
            std::cout << "  [" << label << "]"
                      << " N=" << total
                      << " size=" << (double)total * 2 / (1024 * 1024) << "MB"
                      << " p10/p50/p90=" << (int)p10_us << "/" << (int)med_us
                      << "/" << (int)p90_us << " us"
                      << " BW=" << bw << " GB/s"
                      << " " << (mismatches == 0 ? "PASS" : "FAIL")
                      << " mismatches=" << mismatches
                      << " max_diff=" << max_diff
                      << std::endl;
        };

        measure(label, run_selected);

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_in[d],  queues[d]);
            sycl::free(d_out[d], queues[d]);
        }
    }
    return 0;
}
