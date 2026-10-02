#pragma once

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../include/ops/collective/collective.dp.hpp"

namespace kittens::collective::all_to_all {

using bf16 = sycl::ext::oneapi::bfloat16;
using vec128 = sycl::vec<uint32_t, 4>;

inline constexpr int MAX_DEVICES = 12;
inline constexpr int WG_SIZE = 256;
inline constexpr int EPT = 8;
inline constexpr int VEC_BF16 = 8;

inline void run(std::vector<sycl::queue> &queues,
                const std::vector<bf16 *> &inputs,
                const std::vector<bf16 *> &outputs,
                int chunk) {
    const int n_gpus = static_cast<int>(queues.size());
    const int vector_count = chunk / VEC_BF16;
    const int workgroups =
        (vector_count + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);

    for (int source = 0; source < n_gpus; source++) {
        bf16 *input = inputs[source];
        endpoint_array<bf16 *, MAX_DEVICES> destinations;
        for (int destination = 0; destination < n_gpus; destination++)
            destinations.set(
                destination,
                outputs[destination] + static_cast<size_t>(source) * chunk);
        const int endpoint_count = n_gpus;
        queues[source].parallel_for(
            sycl::nd_range<1>(
                split_workgroup_count(workgroups, endpoint_count) * WG_SIZE,
                WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const int lane = item.get_local_linear_id();
                const int group = item.get_group_linear_id();
                const auto split =
                    split_workgroup_id(group, endpoint_count);
                const int destination = split.endpoint;
                const int workgroup_base =
                    split.data_group * WG_SIZE * EPT;
                const auto *source_ptr = reinterpret_cast<const vec128 *>(
                    input + static_cast<size_t>(destination) * chunk);
                auto *destination_ptr =
                    reinterpret_cast<vec128 *>(destinations[destination]);
#pragma unroll
                for (int element = 0; element < EPT; element++) {
                    const int vector_index =
                        workgroup_base + element * WG_SIZE + lane;
                    if (vector_index < vector_count)
                        destination_ptr[vector_index] = source_ptr[vector_index];
                }
            });
    }
    for (auto &queue : queues)
        queue.wait();
}

} // namespace kittens::collective::all_to_all