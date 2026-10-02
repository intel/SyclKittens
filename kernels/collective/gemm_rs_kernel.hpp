#pragma once

#include "kittens.dp.hpp"

#include <stdexcept>
#include <vector>

namespace kittens::collective::gemm_rs {

inline constexpr int BM = 32;
inline constexpr int BN = 64;
inline constexpr int BK = 32;

using global_a = gl<bf16, -1, -1, -1, -1>;
using global_b = gl<bf16, -1, -1, -1, -1>;
using global_c = gl<float, -1, -1, -1, -1>;
using sycl_bf16 = sycl::ext::oneapi::bfloat16;

template <int GK>
void gemm_kernel_fp32(global_a a, global_b b, global_c c, int rows, int columns) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int workgroup_row = item.get_global_id(1);
    int workgroup_column = item.get_global_id(2) / 16;
    if (workgroup_row >= rows / BM || workgroup_column >= columns / BN)
        return;

    uint32_t subgroup_id = item.get_sub_group().get_group_id();
    rt_bf<BM, BK, ducks::rt_layout::row, ducks::rt_shape::rt_32x32> a_register;
    rt_bf<BK, BN, ducks::rt_layout::col, ducks::rt_shape::rt_32x32> b_register;
    rt_fl<BM, BN, ducks::rt_layout::row, ducks::rt_shape::rt_32x32> c_register;
    zero(c_register);

    constexpr int k_tiles = GK / BK;
#pragma unroll
    for (int index = 0; index < 1; index++) {
        if (subgroup_id % 5)
            prefetch_load(a_register, a, {0, 0, workgroup_row, index});
        if (subgroup_id & 8)
            prefetch_load(b_register, b, {0, 0, index, workgroup_column});
    }
    int prefetch = 1;
#pragma unroll 32
    for (int k_tile = 0; k_tile < k_tiles; k_tile++, prefetch++) {
        load(a_register, a, {0, 0, workgroup_row, k_tile});
        load(b_register, b, {0, 0, k_tile, workgroup_column});
        if (prefetch < k_tiles) {
            if (subgroup_id % 5)
                prefetch_load(a_register, a, {0, 0, workgroup_row, prefetch});
            if (subgroup_id & 8)
                prefetch_load(b_register, b, {0, 0, prefetch, workgroup_column});
        }
        mma_AB(c_register, a_register, b_register, c_register);
    }
    store(c, c_register, {0, 0, workgroup_row, workgroup_column});
}

template <int GK>
inline sycl::event submit_gemm_ct(sycl::queue &queue, bf16 *a, bf16 *b,
                                  float *c, int rows, int columns) {
    global_a a_argument{a, 0u, 0u, static_cast<size_t>(rows), static_cast<size_t>(GK)};
    global_b b_argument{b, 0u, 0u, static_cast<size_t>(GK), static_cast<size_t>(columns)};
    global_c c_argument{c, 0u, 0u, static_cast<size_t>(rows), static_cast<size_t>(columns)};

    constexpr int MAX_SUBGROUPS = 32;
    constexpr int WORKGROUP_COLUMNS =
        (256 / BN < MAX_SUBGROUPS) ? (256 / BN) : MAX_SUBGROUPS;
    constexpr int WORKGROUP_ROWS = MAX_SUBGROUPS / WORKGROUP_COLUMNS;
    uint32_t grid_rows =
        ((rows / BM + WORKGROUP_ROWS - 1) / WORKGROUP_ROWS) * WORKGROUP_ROWS;
    uint32_t grid_columns =
        ((columns / BN + WORKGROUP_COLUMNS - 1) / WORKGROUP_COLUMNS) *
        WORKGROUP_COLUMNS;
    auto properties = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync,
        sycl::ext::oneapi::experimental::sub_group_size<16>};

    return queue.submit([&](sycl::handler &handler) {
        handler.parallel_for(
            sycl::nd_range<3>(
                {1, grid_rows, grid_columns * 16},
                {1, static_cast<size_t>(WORKGROUP_ROWS),
                 static_cast<size_t>(WORKGROUP_COLUMNS) * 16}),
            properties, [=](sycl::nd_item<3>) {
                gemm_kernel_fp32<GK>(
                    a_argument, b_argument, c_argument, rows, columns);
            });
    });
}

inline sycl::event submit_gemm(sycl::queue &queue, bf16 *a, bf16 *b,
                               float *c, int rows, int columns, int inner) {
    switch (inner) {
        case 3072:
            return submit_gemm_ct<3072>(queue, a, b, c, rows, columns);
        case 6144:
            return submit_gemm_ct<6144>(queue, a, b, c, rows, columns);
        default:
            throw std::invalid_argument("GEMM-RS supports K=3072 or K=6144");
    }
}

template <bool INTERLEAVED>
inline void submit_reduce_scatter(sycl::queue &queue, sycl_bf16 **peer_outputs,
                                  sycl_bf16 *output, int rows_per_device,
                                  int columns, int world, int device) {
    queue.parallel_for(sycl::range<1>(rows_per_device * columns),
        [=](sycl::id<1> index) {
            int output_index = index[0];
            int row = output_index / columns;
            int column = output_index % columns;
            int global_index =
                (device * rows_per_device + row) * columns + column;
            int first_source = INTERLEAVED ? output_index % world : 0;
            float sum = 0.0f;
            for (int step = 0; step < world; step++) {
                int source = INTERLEAVED ? (first_source + step) % world : step;
                sum += static_cast<float>(peer_outputs[source][global_index]);
            }
            output[output_index] = static_cast<sycl_bf16>(sum);
        });
}

inline void run(std::vector<sycl::queue> &queues,
                const std::vector<bf16 *> &a,
                const std::vector<bf16 *> &b,
                const std::vector<float *> &c_fp32,
                const std::vector<sycl_bf16 *> &c_bf16,
                const std::vector<sycl_bf16 **> &peer_output_arrays,
                const std::vector<sycl_bf16 *> &output,
                int rows, int columns, int inner) {
    const int world = static_cast<int>(queues.size());
    const int rows_per_device = rows / world;
    const int output_elements = rows * columns;

    for (int device = 0; device < world; device++)
        submit_gemm(queues[device], a[device], b[device], c_fp32[device],
                    rows, columns, inner);

    for (int device = 0; device < world; device++) {
        float *source = c_fp32[device];
        sycl_bf16 *destination = c_bf16[device];
        queues[device].parallel_for(sycl::range<1>(output_elements),
            [=](sycl::id<1> index) {
                destination[index] = static_cast<sycl_bf16>(source[index]);
            });
    }

    for (int device = 0; device < world; device++) {
        if (rows == 3072) {
            submit_reduce_scatter<true>(
                queues[device], peer_output_arrays[device], output[device],
                rows_per_device, columns, world, device);
        } else {
            submit_reduce_scatter<false>(
                queues[device], peer_output_arrays[device], output[device],
                rows_per_device, columns, world, device);
        }
    }
    for (auto &queue : queues)
        queue.wait();
}

} // namespace kittens::collective::gemm_rs