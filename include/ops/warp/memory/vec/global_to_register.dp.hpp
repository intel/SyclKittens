/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        const U2 *src_pack = reinterpret_cast<const U2 *>(src_ptr);
        constexpr int pack_elems = base_types::packing<U2>::num();
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            int base = w * dst.reductions;
            #pragma unroll
            for(int i = 0; i < dst.inner_dim; i++) {
                if((base + i * pack_elems) < dst.length) {
                    dst[w][i] = base_types::convertor<T2, U2>::convert(
                        *(const U2*)&src_ptr[base + i * pack_elems]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // coalesced load mirroring the store path
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*WARP_THREADS + (laneid%4)*(WARP_THREADS/4) + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            if(idx < dst.length) {
                T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                if(laneid%2==0) dst[o_dim][0].x() = tmp;
                else dst[o_dim][0].y() = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        constexpr int stride = RV::stride_elems; // keep in lockstep with rv naive layout
        constexpr int repeats = RV::repeats;
        if constexpr (std::is_same_v<U, bf16> && (stride % 2 == 0) && repeats == 1 && (RV::outer_dim % 2 == 0)) {
            const auto *pack_ptr = reinterpret_cast<const std::uint32_t *>(src_ptr);
            #pragma unroll
            for (auto w = 0; w < dst.outer_dim; w += 2) {
                const int base = (w>>1) * stride + laneid; // two consecutive bf16 for w and w+1
                const std::uint32_t raw = pack_ptr[base];
                const std::uint16_t lo_bits = static_cast<std::uint16_t>(raw & 0xFFFF);
                const std::uint16_t hi_bits = static_cast<std::uint16_t>(raw >> 16);
                dst[w][0] = base_types::convertor<T, bf16>::convert(sycl::bit_cast<bf16>(lo_bits));
                dst[w + 1][0] = base_types::convertor<T, bf16>::convert(sycl::bit_cast<bf16>(hi_bits));
            }
        } else {
            // Fallback scalar path for other types.
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                #pragma unroll
                for(int r = 0; r < repeats; ++r) {
                    int base = w * stride + laneid + r * WARP_THREADS;
                    dst[w][r] = base_types::convertor<T, U>::convert(src_ptr[base]);
                }
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        U2 *dst_pack = reinterpret_cast<U2 *>(dst_ptr);
        constexpr int pack_elems = base_types::packing<U2>::num();
        // Distribute writes across lanes for coalescing
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            int base = w * src.reductions;
            #pragma unroll
            for(int i = laneid; i < src.inner_dim; i += WARP_THREADS) {
                if((base + i * pack_elems) < src.length) {
                    *(U2*)&dst_ptr[base + i * pack_elems] =
                        base_types::convertor<U2, T2>::convert(src[w][i]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*WARP_THREADS + (laneid%4)*(WARP_THREADS/4) + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x());
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y());
                dst_ptr[idx] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        constexpr int stride = RV::stride_elems; // keep in lockstep with rv naive layout
        constexpr int repeats = RV::repeats;
        if constexpr (std::is_same_v<U, bf16> && (stride % 2 == 0) && repeats == 1 && (RV::outer_dim % 2 == 0)) {
            auto *dst_pack = reinterpret_cast<std::uint32_t *>(dst_ptr);
            #pragma unroll
            for (auto w = 0; w < src.outer_dim; w += 2) {
                const int base = (w>>1)* stride + laneid; // two consecutive bf16 for w and w+1
                const bf16 lo = base_types::convertor<bf16, T>::convert(src[w][0]);
                const bf16 hi = base_types::convertor<bf16, T>::convert(src[w + 1][0]);
                const std::uint32_t packed = (static_cast<std::uint32_t>(sycl::bit_cast<std::uint16_t>(hi)) << 16)
                    | static_cast<std::uint32_t>(sycl::bit_cast<std::uint16_t>(lo));
                dst_pack[base] = packed;
            }
        } else {
            // Fallback scalar path for other types.
            #pragma unroll
            for(auto w = 0; w < src.outer_dim; w++) {
                #pragma unroll
                for(int r = 0; r < repeats; ++r) {
                    int base = w * stride + laneid + r * WARP_THREADS;
                    dst_ptr[base] = base_types::convertor<U, T>::convert(src[w][r]);
                }
            }
        }
    }
}

// ============================================================
// Wide (128-bit) coalesced load/store for naive register vectors.
// ============================================================
namespace detail {
/// 16-byte aligned quad-word used to force a single 128-bit memory transaction.
struct alignas(16) wide128_t { std::uint32_t v[4]; };
} // namespace detail

/**
 * @brief Load a naive register vector from bf16 global memory using 128-bit
 *        coalesced transactions.
 *
 * The default naive `load()` lowers to 16-bit (scalar) or 32-bit strided
 * transactions because each lane owns interleaved single/paired elements. For
 * bandwidth-bound reductions (RMSNorm / LayerNorm) that leaves ~2-4x of HBM on
 * the table. `load_wide` re-tiles the row into 128-byte blocks: at vector step
 * `g`, lane `l` owns the 8 contiguous bf16 at uint4 index `g*WARP_THREADS + l`,
 * so the 16 lanes of a step cover 16 consecutive 16-byte words == 256 contiguous
 * bytes, i.e. a fully coalesced 128-bit-per-lane load.
 *
 * The element -> (lane, slot) permutation differs from the default naive layout,
 * so a `load_wide` result must be paired with `store_wide` (and any companion
 * vectors, e.g. gamma, must also be loaded with `load_wide`). Full-vector
 * reductions (`sum`, `max`, ...) and element-wise ops (`mul`, `add`, ...) are
 * permutation-invariant and therefore remain correct.
 *
 * Requires: naive layout, bf16 global dtype, length % (8*WARP_THREADS) == 0,
 * and a 16-byte aligned base address (satisfied by contiguous [.,.,D] tensors
 * with D % 8 == 0).
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
inline static void load_wide(RV &dst, const GL &src, const COORD &idx) {
    static_assert(std::is_same_v<typename RV::layout, ducks::rv_layout::naive>,
                  "load_wide requires a naive-layout register vector");
    static_assert(std::is_same_v<typename GL::dtype, bf16>,
                  "load_wide requires a bf16 global source");
    static_assert(RV::length % (8 * WARP_THREADS) == 0,
                  "load_wide requires length divisible by 8*WARP_THREADS (128)");
    using T = typename RV::dtype; // float for rv_fl

    const bf16 *src_ptr = (const bf16 *)&src[(idx.template unit_coord<-1, 3>())];
    const auto *pack = reinterpret_cast<const detail::wide128_t *>(src_ptr);
    const int laneid = ::kittens::laneid();
    constexpr int U4_PER_LANE = RV::length / (8 * WARP_THREADS);

    #pragma unroll
    for (int g = 0; g < U4_PER_LANE; ++g) {
        const detail::wide128_t raw = pack[g * WARP_THREADS + laneid];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const std::uint32_t a = raw.v[k];
            dst[0][g * 8 + 2 * k]     = base_types::convertor<T, bf16>::convert(
                sycl::bit_cast<bf16>(static_cast<std::uint16_t>(a & 0xFFFF)));
            dst[0][g * 8 + 2 * k + 1] = base_types::convertor<T, bf16>::convert(
                sycl::bit_cast<bf16>(static_cast<std::uint16_t>(a >> 16)));
        }
    }
}

/**
 * @brief Store a naive register vector to bf16 global memory using 128-bit
 *        coalesced transactions. Inverse permutation of `load_wide`; must be
 *        paired with vectors produced by `load_wide`.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
inline static void store_wide(const GL &dst, const RV &src, const COORD &idx) {
    static_assert(std::is_same_v<typename RV::layout, ducks::rv_layout::naive>,
                  "store_wide requires a naive-layout register vector");
    static_assert(std::is_same_v<typename GL::dtype, bf16>,
                  "store_wide requires a bf16 global destination");
    static_assert(RV::length % (8 * WARP_THREADS) == 0,
                  "store_wide requires length divisible by 8*WARP_THREADS (128)");
    using T = typename RV::dtype;

    bf16 *dst_ptr = (bf16 *)&dst[(idx.template unit_coord<-1, 3>())];
    auto *pack = reinterpret_cast<detail::wide128_t *>(dst_ptr);
    const int laneid = ::kittens::laneid();
    constexpr int U4_PER_LANE = RV::length / (8 * WARP_THREADS);

    #pragma unroll
    for (int g = 0; g < U4_PER_LANE; ++g) {
        detail::wide128_t raw;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const bf16 lo = base_types::convertor<bf16, T>::convert(src[0][g * 8 + 2 * k]);
            const bf16 hi = base_types::convertor<bf16, T>::convert(src[0][g * 8 + 2 * k + 1]);
            raw.v[k] = (static_cast<std::uint32_t>(sycl::bit_cast<std::uint16_t>(hi)) << 16)
                     |  static_cast<std::uint32_t>(sycl::bit_cast<std::uint16_t>(lo));
        }
        pack[g * WARP_THREADS + laneid] = raw;
    }
}

/**
 * @brief L1/L3 prefetch of the exact contiguous bf16 footprint that `load_wide`
 *        will read — the natural pair to `load_wide` (mirrors the tile path's
 *        `load()` / `prefetch_load()`).
 *
 * `load_wide` streams a contiguous `RV::length`-bf16 run (one 128-bit/lane
 * coalesced burst per vector step). This warms that same run into cache ahead
 * of the load, reshaped as a `(RV::length/16) x 16` contiguous 2D block (pitch
 * == row width, so no gaps) via the Xe 2D-block prefetch. Discards the data
 * (prefetch only). Issue it `D` vector-steps AHEAD of the matching `load_wide`
 * to add outstanding loads (deeper memory-level parallelism) without spending
 * registers — the lever for latency-bound streaming reductions (decode GEMV).
 *
 * Requires the same contract as `load_wide`: bf16 source, length % 128 == 0,
 * 16 B-aligned base. RV is explicit (no dst): `prefetch_load_wide<rv_fl<C>>(gl, idx)`.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
inline static void prefetch_load_wide(const GL &src, const COORD &idx) {
    static_assert(std::is_same_v<typename GL::dtype, bf16>,
                  "prefetch_load_wide requires a bf16 global source");
    static_assert(RV::length % (8 * WARP_THREADS) == 0,
                  "prefetch_load_wide requires length divisible by 8*WARP_THREADS (128)");
#ifdef __SYCL_DEVICE_ONLY__
    const bf16 *src_ptr = (const bf16 *)&src[(idx.template unit_coord<-1, 3>())];
    constexpr int BW    = 16;                          // block width  (elements)
    constexpr int BH    = RV::length / BW;             // block height (rows)
    constexpr int pitch = BW * (int)sizeof(bf16);      // contiguous (no row gap)
    __spirv_Subgroup2DBlockPrefetchINTEL(
        (int)sizeof(bf16), BW, BH, 1, (const void *)src_ptr,
        pitch, BH, pitch, intel::coord_t{0, 0});
#else
    (void)src; (void)idx;
#endif
}

} // namespace kittens
