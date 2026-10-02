/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

namespace kittens {

namespace detail {

static inline int row_from_indices_dim2(int laneid, int inner_dim, int x_or_y) {
    return 8 * inner_dim + (laneid % 4) * 2 + x_or_y;
}
static inline int row_from_indices_dim1(int laneid, int x_or_y) {
    return 8 * x_or_y + (laneid / 4);
}
static inline int canonical_src_lane_dim2(int row) {
    return (row / 2) % 4 + 4 * (row % 2);
}
static inline int canonical_src_lane_dim1(int row) {
    return (row * 4) % 32;
}

}

// // Specialized fast path: naive bf16 -> naive float, using explicit bf16-to-float
// // conversion to avoid backend intrinsic issues on PVC.
// template<int L>
// static inline void copy(rv<float, L, naive_l> &dst, const rv<bf16, L, naive_l> &src) {
//     constexpr int stride = rv<float, L, naive_l>::stride_elems;
//     constexpr int repeats = rv<float, L, naive_l>::repeats;
//     #pragma unroll
//     for(int i = 0; i < rv<float, L, naive_l>::outer_dim; ++i) {
//         #pragma unroll
//         for(int r = 0; r < repeats; ++r) {
//             dst[i][r] = sycl::ext::intel::math::bfloat162float(src[i][r]);
//         }
//     }
// }

/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
template<ducks::rv::all RV1, ducks::rv::all RV2>
static inline void copy(RV1 &dst, const RV2 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;
    const int laneid = kittens::laneid();

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) {
        if constexpr (std::is_same_v<typename RV1::layout, align_l>) {
            // align -> align: copy all packed elements per lane (inner_dim may be > 1 on Intel)
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for (int j = 0; j < RV1::inner_dim; j++) {
                    dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
                }
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, naive_l>) {
            // naive -> naive: copy with stride-aware bounds to skip padding
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for (int r = 0; r < RV1::repeats; r++) {
                    dst[i][r] = base_types::convertor<D1, D2>::convert(src[i][r]);
                }
            }
        } else {
            // ortho -> ortho: simple per-element copy
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for (int j = 0; j < RV1::inner_dim; j++) {
                    dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
                }
            }
        }
    } else {
        if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, align_l>) {
            // align -> ortho: shuffle packed elements into ortho lanes
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][0].x() : src[i][0].y(),
                    detail::canonical_src_lane_dim2(detail::row_from_indices_dim1(laneid, 0))
                );
                dst[i][0].y() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][1].x() : src[i][1].y(),
                    detail::canonical_src_lane_dim2(detail::row_from_indices_dim1(laneid, 1))
                );
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, ortho_l>) {
            // ortho -> align: shuffle ortho lanes into packed align form
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x(),
                    detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 0, 0))
                );
                dst[i][0].y() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x(),
                    detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 0, 1))
                );
                dst[i][1].x() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y(),
                    detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 1, 0))
                );
                dst[i][1].y() = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y(),
                    detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 1, 1))
                );
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, naive_l>) {
            // naive -> ortho: gather naive rows into ortho vector pairs
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x() = packed_shfl_sync(
                    kittens::MASK_ALL, src[i / 2][0],
                    16 * (i % 2) + 0 + (laneid / 4)
                );
                dst[i][0].y() = packed_shfl_sync(
                    kittens::MASK_ALL, src[i / 2][0],
                    16 * (i % 2) + 8 + (laneid / 4)
                );
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, ortho_l>) {
            // ortho -> naive: scatter ortho components back into naive rows
            int lane_replication = laneid % 4;
            #pragma unroll
            for (int i = 0; i < RV1::outer_dim; i++) {
                D1 tmp = 0;
                if (RV1::length % 32 == 0 || i < RV1::outer_dim - 1 || lane_replication < 2) {
                    tmp = lane_replication % 2 ? src[2 * i + (lane_replication >= 2)][0].y() : src[2 * i + (lane_replication >= 2)][0].x();
                }
                dst[i][0] = packed_shfl_sync(
                    kittens::MASK_ALL, tmp,
                    (laneid % 8) * 4 + (laneid / 8)
                );
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, naive_l>) {
            // naive -> align: broadcast each naive scalar to all lanes, pack into align pairs.
            // Naive: lane L, data[0][r] = element L + r*WARP_THREADS (distributed)
            // Align: all lanes, data[tile][j] = {element tile*tile_len+2j, element tile*tile_len+2j+1} (replicated)
            using V = typename base_types::packing<D2>::unpacked_type;
            using C = typename base_types::packing<D1>::unpacked_type;
            constexpr int tile_len = RV1::reductions;
            auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            #pragma unroll
            for (int tile = 0; tile < RV1::outer_dim; tile++) {
                #pragma unroll
                for (int j = 0; j < RV1::inner_dim; j++) {
                    const int g0 = tile * tile_len + j * 2;
                    const int g1 = g0 + 1;
                    const int r0 = g0 / WARP_THREADS;
                    const int lane0 = g0 % WARP_THREADS;
                    const int r1 = g1 / WARP_THREADS;
                    const int lane1 = g1 % WARP_THREADS;
                    V v0 = sycl::group_broadcast(sg, src[0][r0], lane0);
                    V v1 = (g1 < RV1::length) ? (V)sycl::group_broadcast(sg, src[0][r1], lane1) : V(0);
                    dst[tile][j].x() = base_types::convertor<C, V>::convert(v0);
                    dst[tile][j].y() = base_types::convertor<C, V>::convert(v1);
                }
            }
        } else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, align_l>) {
            // align -> naive: extract scalar from replicated align data.
            // Align: all lanes, data[tile][j] = {element tile*tile_len+2j, element tile*tile_len+2j+1} (replicated)
            // Naive: lane L, data[0][r] = element L + r*WARP_THREADS (distributed)
            using C = typename base_types::packing<typename RV2::dtype>::unpacked_type;
            constexpr int tile_len = RV2::reductions;
            #pragma unroll
            for (int r = 0; r < RV1::repeats; r++) {
                const int g = laneid + r * WARP_THREADS;
                if (g < RV1::length) {
                    const int tile = g / tile_len;
                    const int within = g % tile_len;
                    const int j = within / 2;
                    const int comp = within % 2;
                    C val = (comp == 0) ? src[tile][j].x() : src[tile][j].y();
                    dst[0][r] = base_types::convertor<D1, C>::convert(val);
                }
            }
        }
    }
}

} // namespace kittens

