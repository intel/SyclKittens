#pragma once

#include "kittens.dp.hpp"

#include <stdexcept>
#include <vector>

namespace kittens::collective::ag_gemm {

inline constexpr int NUM_DEVICES = 12;
inline constexpr int BM = 32;
inline constexpr int BN = 64;
inline constexpr int BK = 32;
inline constexpr int WG_SIZE = 256;
inline constexpr int EPT = 8;
inline constexpr int VEC_BF16 = 8;

using sycl_bf16 = sycl::ext::oneapi::bfloat16;
using vec128 = sycl::vec<uint32_t, 4>;
using global_a = gl<bf16, -1, -1, -1, -1>;
using global_b = gl<bf16, -1, -1, -1, -1>;
using global_cb = gl<bf16, -1, -1, -1, -1>;

inline uint32_t group_to_linear(const sycl::nd_item<3> &item) {
    return item.get_group(2) + item.get_group(1) * item.get_group_range(2);
}

inline int snake_tile_m(const sycl::nd_item<3> &item, uint32_t snake_w) {
    constexpr uint32_t MAX_WG = 64;
    uint32_t wg_num_m = MAX_WG / snake_w;
    uint32_t group_range_n = item.get_group_range(2);
    uint32_t wg_repeat_n = group_range_n / snake_w;
    uint32_t linear = group_to_linear(item);
    uint32_t repeat_id = linear / MAX_WG;
    uint32_t repeat_id_m = repeat_id / wg_repeat_n;
    uint32_t repeat_start_m = repeat_id_m * wg_num_m;
    uint32_t wg_inner_id = linear % MAX_WG;
    uint32_t wg_coord_m = wg_inner_id / snake_w;
    return repeat_start_m + wg_coord_m;
}

inline int snake_tile_n(const sycl::nd_item<3> &item, uint32_t snake_w) {
    constexpr uint32_t MAX_WG = 64;
    uint32_t group_range_n = item.get_group_range(2);
    uint32_t wg_repeat_n = group_range_n / snake_w;
    uint32_t linear = group_to_linear(item);
    uint32_t repeat_id = linear / MAX_WG;
    uint32_t repeat_id_n = repeat_id % wg_repeat_n;
    uint32_t repeat_id_m = repeat_id / wg_repeat_n;
    uint32_t repeat_start_n_0 = repeat_id_n * snake_w;
    uint32_t repeat_start_n_1 = (wg_repeat_n - repeat_id_n - 1) * snake_w;
    uint32_t repeat_start_n = (repeat_id_m & 1) == 0 ? repeat_start_n_0 : repeat_start_n_1;
    uint32_t wg_inner_id = linear % MAX_WG;
    uint32_t wg_coord_n = wg_inner_id % snake_w;
    return repeat_start_n + wg_coord_n;
}

inline int snake_width_for(int rows, int columns) {
    if (rows == 6144 && columns == 6144)
        return 8;
    if (rows == 12288 && columns == 12288)
        return 4;
    throw std::invalid_argument("unsupported AG-GEMM shape");
}

template <int GK_CT>
inline void gemm_kernel_bf16_ct(global_a a, global_b b, global_cb c,
                                int rows, int columns, int snake_w) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int wg_m = snake_tile_m(item, snake_w) * item.get_local_range(1) +
        item.get_local_id(1);
    int wg_n = (snake_tile_n(item, snake_w) * item.get_local_range(2) +
        item.get_local_id(2)) / 16;
    if (wg_m >= rows / BM || wg_n >= columns / BN)
        return;

    uint32_t subgroup_id = item.get_sub_group().get_group_id();
    rt_bf<BM, BK, ducks::rt_layout::row, ducks::rt_shape::rt_32x32> a_register;
    rt_bf<BK, BN, ducks::rt_layout::col, ducks::rt_shape::rt_32x32> b_register;
    rt_fl<BM, BN, ducks::rt_layout::row, ducks::rt_shape::rt_32x32> c_register;
    zero(c_register);

    constexpr int k_tiles = GK_CT / BK;
    constexpr int PREFETCH = 2;
    const int coop_h_index = subgroup_id % (256 / BN);
    const int coop_v_index = subgroup_id / (256 / BN);
    const int pref_a_row_offset = BM / (256 / BN);
    const int pref_b_subgroup_rows = (256 / (BM * 2));
    const int pref_b_row_offset = BK / (256 / BN);
    const int coop_b_row_index = coop_v_index % pref_b_subgroup_rows;
    const int coop_b_col_index = coop_v_index / pref_b_subgroup_rows;
    const int coop_a_row_offset = coop_h_index * pref_a_row_offset;
    const int coop_b_row_offset = coop_b_row_index * pref_b_row_offset;
    const int coop_b_col_offset = coop_b_col_index * 32;

    int prefetch = PREFETCH;
#pragma unroll
    for (int index = 0; index < PREFETCH; index++) {
        prefetch_load_coop(a_register, a, {0, 0, wg_m, index}, {coop_a_row_offset, 0});
        prefetch_load_coop(b_register, b, {0, 0, index, wg_n},
            {coop_b_row_offset, coop_b_col_offset});
    }

#pragma unroll 32
    for (int k_tile = 0; k_tile < k_tiles; k_tile++, prefetch++) {
        load(a_register, a, {0, 0, wg_m, k_tile});
        load(b_register, b, {0, 0, k_tile, wg_n});
        if (prefetch < k_tiles) {
            prefetch_load_coop(a_register, a, {0, 0, wg_m, prefetch},
                {coop_a_row_offset, 0});
            prefetch_load_coop(b_register, b, {0, 0, prefetch, wg_n},
                {coop_b_row_offset, coop_b_col_offset});
        }
        mma_AB(c_register, a_register, b_register, c_register);
    }

    store(c, c_register, {0, 0, wg_m, wg_n});
}

template <int GK_CT>
inline sycl::event submit_gemm_ct(sycl::queue &queue, global_a a_argument,
                                  global_b b_argument, global_cb c_argument,
                                  int rows, int columns, int snake_w,
                                  uint32_t grid_rows, uint32_t grid_columns) {
    constexpr int MAX_SUBGROUPS = 32;
    constexpr int WORKGROUP_COLUMNS =
        (256 / BN < MAX_SUBGROUPS) ? (256 / BN) : MAX_SUBGROUPS;
    constexpr int WORKGROUP_ROWS = MAX_SUBGROUPS / WORKGROUP_COLUMNS;
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
                gemm_kernel_bf16_ct<GK_CT>(
                    a_argument, b_argument, c_argument, rows, columns, snake_w);
            });
    });
}

inline sycl::event dispatch_gemm_bf16(sycl::queue &queue,
                                      bf16 *a, bf16 *b, bf16 *c,
                                      int rows, int columns, int inner,
                                      int snake_w) {
    global_a a_argument{a, 0u, 0u, static_cast<size_t>(rows), static_cast<size_t>(inner)};
    global_b b_argument{b, 0u, 0u, static_cast<size_t>(inner), static_cast<size_t>(columns)};
    global_cb c_argument{c, 0u, 0u, static_cast<size_t>(rows), static_cast<size_t>(columns)};

    constexpr int MAX_SUBGROUPS = 32;
    constexpr int WORKGROUP_COLUMNS =
        (256 / BN < MAX_SUBGROUPS) ? (256 / BN) : MAX_SUBGROUPS;
    constexpr int WORKGROUP_ROWS = MAX_SUBGROUPS / WORKGROUP_COLUMNS;

    uint32_t grid_rows;
    uint32_t grid_columns;
    if (snake_w > 0) {
        uint32_t pad_rows = static_cast<uint32_t>(WORKGROUP_ROWS) *
            (64u / static_cast<uint32_t>(snake_w));
        uint32_t pad_columns = static_cast<uint32_t>(WORKGROUP_COLUMNS) *
            static_cast<uint32_t>(snake_w);
        grid_rows = ((rows / BM + pad_rows - 1) / pad_rows) * pad_rows;
        grid_columns = ((columns / BN + pad_columns - 1) / pad_columns) * pad_columns;
    } else {
        grid_rows = ((rows / BM + WORKGROUP_ROWS - 1) / WORKGROUP_ROWS) * WORKGROUP_ROWS;
        grid_columns = ((columns / BN + WORKGROUP_COLUMNS - 1) / WORKGROUP_COLUMNS) *
            WORKGROUP_COLUMNS;
    }

    switch (inner) {
        case 6144:
            return submit_gemm_ct<6144>(queue, a_argument, b_argument, c_argument,
                rows, columns, snake_w, grid_rows, grid_columns);
        case 12288:
            return submit_gemm_ct<12288>(queue, a_argument, b_argument, c_argument,
                rows, columns, snake_w, grid_rows, grid_columns);
        default:
            throw std::invalid_argument("AG-GEMM supports K=6144 or K=12288");
    }
}

inline void run_production_topology_gather_gemm(
    std::vector<sycl::queue> &queues,
    const std::vector<sycl_bf16 *> &a_local,
    const std::vector<bf16 *> &a_gathered,
    const std::vector<bf16 *> &b,
    const std::vector<sycl_bf16 *> &c_bf16,
    int rows, int columns, int inner, int rows_per_device, int snake_w) {
    const int world = static_cast<int>(queues.size());
    const int pvc_count = world / 2;
    const int elements_per_device = rows_per_device * inner;
    const int vectors_per_device = elements_per_device / VEC_BF16;
    const int workgroups =
        (vectors_per_device + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);

    for (int tile = 0; tile < world; tile++) {
        sycl_bf16 *input = a_local[tile];
        bf16 *destinations[8];
        int destination_count = 0;
        int partner = tile ^ 1;
        int pvc = tile / 2;
        destinations[destination_count++] =
            a_gathered[tile] + static_cast<size_t>(tile) * elements_per_device;
        destinations[destination_count++] =
            a_gathered[partner] + static_cast<size_t>(tile) * elements_per_device;
        for (int remote_pvc = 0; remote_pvc < pvc_count; remote_pvc++) {
            if (remote_pvc == pvc)
                continue;
            int landing = 2 * remote_pvc + (pvc & 1);
            destinations[destination_count++] =
                a_gathered[landing] + static_cast<size_t>(tile) * elements_per_device;
        }

        int local_vectors = vectors_per_device;
        int local_workgroups = workgroups;
        queues[tile].parallel_for(
            sycl::nd_range<1>(local_workgroups * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int lane = item.get_local_linear_id();
                int base = item.get_group_linear_id() * WG_SIZE * EPT;
                auto *source = reinterpret_cast<const vec128 *>(input);
                for (int destination = 0; destination < destination_count; destination++) {
                    auto *target = reinterpret_cast<vec128 *>(destinations[destination]);
#pragma unroll
                    for (int e = 0; e < EPT; e++) {
                        int vector_index = base + e * WG_SIZE + lane;
                        if (vector_index < local_vectors)
                            target[vector_index] = source[vector_index];
                    }
                }
            });
    }
    for (auto &queue : queues)
        queue.wait();

    for (int tile = 0; tile < world; tile++) {
        int partner = tile ^ 1;
        bf16 *local_gather = a_gathered[tile];
        bf16 *partner_gather = a_gathered[partner];
        int offsets[8];
        int offset_count = 0;
        int pvc = tile / 2;
        int parity = tile & 1;
        for (int remote_pvc = 0; remote_pvc < pvc_count; remote_pvc++) {
            if (remote_pvc == pvc || (remote_pvc & 1) != parity)
                continue;
            offsets[offset_count++] = 2 * remote_pvc;
        }

        int local_vectors = vectors_per_device;
        int local_workgroups = workgroups;
        int local_elements = elements_per_device;
        queues[tile].parallel_for(
            sycl::nd_range<1>(local_workgroups * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int lane = item.get_local_linear_id();
                int base = item.get_group_linear_id() * WG_SIZE * EPT;
                for (int block = 0; block < offset_count; block++) {
                    int slot = offsets[block];
#pragma unroll
                    for (int half = 0; half < 2; half++) {
                        auto *source = reinterpret_cast<const vec128 *>(
                            local_gather + static_cast<size_t>(slot + half) * local_elements);
                        auto *target = reinterpret_cast<vec128 *>(
                            partner_gather + static_cast<size_t>(slot + half) * local_elements);
#pragma unroll
                        for (int e = 0; e < EPT; e++) {
                            int vector_index = base + e * WG_SIZE + lane;
                            if (vector_index < local_vectors)
                                target[vector_index] = source[vector_index];
                        }
                    }
                }
            });
    }
    for (auto &queue : queues)
        queue.wait();

    for (int device = 0; device < world; device++)
        dispatch_gemm_bf16(queues[device], a_gathered[device], b[device], c_bf16[device],
            rows, columns, inner, snake_w);
    for (auto &queue : queues)
        queue.wait();
}

} // namespace kittens::collective::ag_gemm