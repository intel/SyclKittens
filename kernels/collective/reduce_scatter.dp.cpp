/**
 * @file
 * @brief 12-tile Reduce-Scatter for PVC (FLAT hierarchy).
 *
 * Selects the contiguous first-N tiles (torch set_device rank 0..N-1) and
 * reports bandwidth with the convention:
 *     BW = world*(world-1)*epd*sizeof(bf16) / time.
 *
 * Semantics: each tile owns an input buffer of world*epd bf16 (world chunks of
 * `epd`). Output on tile t = sum over all `world` tiles of their chunk t.
 *
 * Dispatch is payload-adaptive: fused reduction for small messages, hierarchical
 * package pre-reduction for the middle regime, and destination-split pair
 * reduction for large messages.
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "../../include/types/device/pgl.dp.hpp"
#include "reduce_scatter_kernel.hpp"
#include "topology.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

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

    int requested_gpus = kittens::collective::reduce_scatter::NUM_DEVICES;
    if (const char *value = std::getenv("SK_NUM_DEVICES"))
        requested_gpus = std::max(2, std::min(
            kittens::collective::reduce_scatter::NUM_DEVICES,
            std::atoi(value)));
    int n_gpus = std::min((int)gpus.size(), requested_gpus);
    std::cout << "Reduce-Scatter 12: " << n_gpus << " GPUs" << std::endl;
    if (n_gpus < 2) { std::cout << "Need >= 2 GPUs" << std::endl; return 1; }

    // Contiguous first-N tiles to match the XCCL baseline hardware set.
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
    std::cout << "  npvc=" << npvc << "  (2 tiles/PVC expected)\n";

    sycl::context ctx(sel);
    std::vector<sycl::queue> queues;
    for (int i = 0; i < n_gpus; i++)
        queues.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});

    for (int total_target : {12288, 98304, 786432, 3145728, 6291456, 12582912,
                             50331648}) {
        if (std::getenv("SK_ONLY_96") && total_target != 50331648) continue;
        if (std::getenv("SK_ONLY_12") && total_target != 6291456) continue;
        int chunk = total_target / n_gpus;  // per-tile output chunk = E/world
        int total = chunk * n_gpus;         // per-tile input size (== total_target)

        std::vector<bf16*> d_in(n_gpus), d_out(n_gpus), d_par(n_gpus);
        for (int d = 0; d < n_gpus; d++) {
            d_in[d]  = sycl::malloc_device<bf16>(total, queues[d]);
            d_out[d] = sycl::malloc_device<bf16>(chunk, queues[d]);
            d_par[d] = sycl::malloc_device<bf16>(total, queues[d]);  // PVC partials
            if (!d_in[d] || !d_out[d] || !d_par[d]) {
                std::cerr << "  FATAL: malloc_device failed d=" << d << std::endl;
                return 1;
            }
        }
        for (int d = 0; d < n_gpus; d++) {
            bf16 *ptr = d_in[d]; int t = total, src = d;
            queues[d].parallel_for(sycl::range<1>(t), [=](sycl::id<1> i) {
                ptr[i] = static_cast<bf16>((src * 7 + static_cast<int>(i[0])) % 9 - 4);
            });
        }
        for (auto &q : queues) q.wait();

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

            std::vector<bf16> h_out(chunk);
            float max_diff = 0.0f;
            size_t mismatches = 0;
            for (int d = 0; d < n_gpus; d++) {
                queues[d].memcpy(h_out.data(), d_out[d], chunk * sizeof(bf16)).wait();
                for (int i = 0; i < chunk; i++) {
                    int global_index = d * chunk + i;
                    float expected = 0.0f;
                    for (int s = 0; s < n_gpus; s++)
                        expected += static_cast<float>((s * 7 + global_index) % 9 - 4);
                    float diff = std::abs(static_cast<float>(h_out[i]) - expected);
                    max_diff = std::max(max_diff, diff);
                    mismatches += diff >= 0.02f;
                }
            }

            double p2p_bytes = (double)n_gpus * (n_gpus - 1) * chunk * sizeof(bf16);
            double bw = p2p_bytes / (med_us * 1e-6) / 1e9;
            double size_mb = (double)total * 2.0 / (1024 * 1024);
            std::cout << "  [" << label << "]"
                      << "  N=" << total << "  size=" << size_mb << "MB"
                      << "  p10/p50/p90=" << (int)p10_us << "/" << (int)med_us
                      << "/" << (int)p90_us << " us"
                      << "  BW=" << bw << " GB/s"
                      << "  " << (mismatches == 0 ? "PASS" : "FAIL")
                      << " mismatches=" << mismatches
                      << " max_diff=" << max_diff
                      << std::endl;
        };

        const auto mode = kittens::collective::reduce_scatter::select_mode(
            n_gpus, (size_t)total * sizeof(bf16));
        if (mode == kittens::collective::reduce_scatter::Mode::Fused)
            measure("RS-fused", [&] {
                kittens::collective::reduce_scatter::run_fused(
                    queues, d_in, d_out, n_gpus, chunk);
            });
        else if (mode == kittens::collective::reduce_scatter::Mode::DestinationSplit)
            measure("RS-destination-split", [&] {
                kittens::collective::reduce_scatter::run_destination_split(
                    queues, d_in, d_out, d_par, npvc, tiles, n_gpus, chunk);
            });
        else
            measure("RS-hierarchical", [&] {
                kittens::collective::reduce_scatter::run_hierarchical(
                    queues, d_in, d_out, d_par, npvc, tiles, n_gpus, total,
                    chunk);
            });

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_in[d],  queues[d]);
            sycl::free(d_out[d], queues[d]);
            sycl::free(d_par[d], queues[d]);
        }
    }
    return 0;
}
