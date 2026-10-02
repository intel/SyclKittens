/**
 * @file gemm_rs.dp.cpp
 * @brief Six-device GEMM plus BF16 reduce-scatter for Intel PVC.
 *
 * GEMM writes FP32 locally, a local cast halves the communication volume, and
 * reduce-scatter reads BF16 outputs from all devices. Source ordering is chosen
 * per report shape to distribute the smaller workload across XeLinks without
 * adding that index arithmetic to the larger workload.
 */

#include "gemm_rs_kernel.hpp"
using namespace kittens;

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

static constexpr int NUM_DEVICES = 6;
using sbf16 = sycl::ext::oneapi::bfloat16;

int main() {
    std::vector<sycl::device> gpus;
    for (auto &platform : sycl::platform::get_platforms()) {
        for (auto &device : platform.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(device);
    }
    int world = static_cast<int>(gpus.size());
    std::cout << "=== GEMM + reduce-scatter ===\nGPUs: " << world << std::endl;
    if (world != NUM_DEVICES) {
        std::cerr << "Need exactly " << NUM_DEVICES << " visible devices" << std::endl;
        return 1;
    }

    sycl::context context(gpus);
    std::vector<sycl::queue> queues;
    for (auto &device : gpus) {
        queues.emplace_back(context, device,
            sycl::property_list{sycl::property::queue::in_order{}});
    }

    for (auto [m, n, k] : std::vector<std::tuple<int, int, int>>{
             {3072, 3072, 3072}, {6144, 6144, 6144}}) {
        int rows_per_device = m / world;
        int output_elements = m * n;
        std::cout << "\nM=" << m << " N=" << n << " K=" << k
                  << " rows/device=" << rows_per_device << std::endl;

        std::vector<kittens::bf16 *> a(world), b(world);
        std::vector<float *> c_fp32(world);
        std::vector<sbf16 *> c_bf16(world), output(world);
        for (int device = 0; device < world; device++) {
            a[device] = sycl::malloc_device<kittens::bf16>(m * k, queues[device]);
            b[device] = sycl::malloc_device<kittens::bf16>(k * n, queues[device]);
            c_fp32[device] = sycl::malloc_device<float>(output_elements, queues[device]);
            c_bf16[device] = sycl::malloc_device<sbf16>(output_elements, queues[device]);
            output[device] = sycl::malloc_device<sbf16>(rows_per_device * n, queues[device]);

            auto *a_ptr = a[device];
            auto *b_ptr = b[device];
            auto a_value = static_cast<kittens::bf16>(0.01f);
            auto b_value = static_cast<kittens::bf16>((device + 1) * 0.01f);
            queues[device].parallel_for(sycl::range<1>(m * k),
                [=](sycl::id<1> index) { a_ptr[index] = a_value; });
            queues[device].parallel_for(sycl::range<1>(k * n),
                [=](sycl::id<1> index) { b_ptr[index] = b_value; });
        }
        for (auto &queue : queues) queue.wait();

        std::vector<sbf16 **> peer_output_arrays(world);
        for (int device = 0; device < world; device++) {
            peer_output_arrays[device] =
                sycl::malloc_device<sbf16 *>(world, queues[device]);
            queues[device].memcpy(peer_output_arrays[device], c_bf16.data(),
                                  world * sizeof(sbf16 *));
        }
        for (auto &queue : queues) queue.wait();

        auto run_gemm_rs = [&]() {
            kittens::collective::gemm_rs::run(
                queues, a, b, c_fp32, c_bf16, peer_output_arrays, output,
                m, n, k);
        };

        constexpr int WARMUP = 3;
        constexpr int ITERS = 10;
        for (int iteration = 0; iteration < WARMUP; iteration++) run_gemm_rs();
        auto start = std::chrono::high_resolution_clock::now();
        for (int iteration = 0; iteration < ITERS; iteration++) run_gemm_rs();
        auto stop = std::chrono::high_resolution_clock::now();
        double microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            stop - start).count() / static_cast<double>(ITERS);

        const float expected = k * 0.0001f * world * (world + 1) / 2.0f;
        std::vector<sbf16> host_output(static_cast<size_t>(rows_per_device) * n);
        float max_diff = 0.0f;
        size_t mismatches = 0;
        for (int device = 0; device < world; device++) {
            queues[device].memcpy(host_output.data(), output[device],
                                  host_output.size() * sizeof(sbf16)).wait();
            for (sbf16 value : host_output) {
                float diff = std::abs(static_cast<float>(value) - expected);
                max_diff = std::max(max_diff, diff);
                mismatches += diff >= 0.1f;
            }
        }

        double flops = 2.0 * m * n * k;
        double tflops = flops / (microseconds * 1.0e-6) / 1.0e12;
        bool passed = mismatches == 0;
        std::cout << "  " << microseconds << " us  " << tflops << " TFLOP/s  "
                  << (passed ? "PASS" : "FAIL") << " mismatches=" << mismatches
                  << " max_diff=" << max_diff << std::endl;

        for (int device = 0; device < world; device++) {
            sycl::free(a[device], queues[device]);
            sycl::free(b[device], queues[device]);
            sycl::free(c_fp32[device], queues[device]);
            sycl::free(c_bf16[device], queues[device]);
            sycl::free(output[device], queues[device]);
            sycl::free(peer_output_arrays[device], queues[device]);
        }
        if (!passed) return 2;
    }
    return 0;
}