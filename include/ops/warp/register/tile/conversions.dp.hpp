/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Perform a matrix transpose on a block of 8 bf16_2 elements using inline assembly.
 *
 * This low-level operation is utilized by higher-level layout swap functions to transpose
 * the layout of bf16_2 elements within a register tile. The function leverages inline PTX
 * assembly to efficiently swap the layout of the given block.
 *
 * @param[out] dst A reference to the destination bf16_2 element where the transposed result is stored.
 * @param[in] src A reference to the source bf16_2 element to be transposed.
 */
inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    /*
    DPCT1053:4: Migration of device assembly code is not supported.
    */
    //asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
    //             : "+r"(*(uint32_t *)(&dst))
    //             : "r"(*(uint32_t *)(&src)));
}

#if defined(KITTENS_INTEL)
// ---- Intel register transpose (warp size 16 or 32) ----
struct rt_scalar_loc {
    int lane;
    int word;
    int comp; // 0 -> x, 1 -> y
};

template<int warp_threads, bool row_layout>
static inline void lane_word_to_coord(int lane, int word, int comp, int &row, int &col) {
    if constexpr (warp_threads == 16) {
        // Linear 16-scalar-per-lane mapping: each lane owns 16 contiguous scalars.
        int scalar_idx = lane * 16 + word * 2 + comp; // 0..255
        if constexpr (row_layout) {
            row = scalar_idx / 16;
            col = scalar_idx % 16;
        } else {
            col = scalar_idx / 16;
            row = scalar_idx % 16;
        }
    } else if constexpr (warp_threads == 32) {
        // Preserve existing NVIDIA mapping when compiled on non-Intel targets.
        if constexpr (row_layout) {
            row = (word % 2) * 8 + lane / 4;
            col = (word / 2) * 8 + (lane % 4) * 2 + comp;
        } else {
            row = (word / 2) * 8 + (lane % 4) * 2 + comp;
            col = (word % 2) * 8 + lane / 4;
        }
    } else {
        static_assert(warp_threads == 16 || warp_threads == 32, "Unsupported warp size");
    }
}

template<int warp_threads, bool row_layout>
static inline rt_scalar_loc coord_to_lane_word(int row, int col) {
    rt_scalar_loc out{};
    if constexpr (warp_threads == 16) {
        int scalar_idx;
        if constexpr (row_layout) {
            scalar_idx = row * 16 + col;
        } else {
            scalar_idx = col * 16 + row;
        }
        out.lane = scalar_idx / 16;
        int local = scalar_idx % 16; // 0..15
        out.word = local / 2;        // 0..7
        out.comp = local & 1;        // 0 or 1
    } else if constexpr (warp_threads == 32) {
        if constexpr (row_layout) {
            int k_row = row / 8;
            int k_col = col / 8;
            out.word = k_row + 2 * k_col;
            out.lane = (row % 8) * 4 + (col % 8) / 2;
            out.comp = col & 1;
        } else {
            int k_col = col / 8;
            int k_row = row / 8;
            out.word = 2 * k_row + k_col;
            out.lane = (col % 8) * 4 + (row % 8) / 2;
            out.comp = row & 1;
        }
    } else {
        static_assert(warp_threads == 16 || warp_threads == 32, "Unsupported warp size");
    }
    return out;
}

template<typename T, ducks::rt_layout::all layout>
inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    constexpr bool is_row = std::is_same_v<layout, ducks::rt_layout::row>;
    constexpr int warp_threads = WARP_THREADS;
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int lane = laneid();

    if constexpr (warp_threads == 16) {
        using comp_t = typename rt_base<T, layout>::T;
        sycl::vec<comp_t, 16> lane_scalars;

        #pragma unroll
        for(int w = 0; w < src.packed_per_thread; ++w) {
            auto pack = src.data[w];
            lane_scalars[2 * w]     = pack.x();
            lane_scalars[2 * w + 1] = pack.y();
        }

        #pragma unroll
        for(int w = 0; w < dst.packed_per_thread; w++) {
            typename rt_base<T, layout>::T2 out{};
            #pragma unroll
            for(int comp = 0; comp < 2; comp++) {
                // Flat-index: destination is transposed layout, so its logical coords are swapped.
                int dst_scalar_idx = lane * 16 + w * 2 + comp;
                int dst_row, dst_col;
                if constexpr (is_row) {
                    dst_col = dst_scalar_idx / 16;
                    dst_row = dst_scalar_idx % 16;
                } else {
                    dst_row = dst_scalar_idx / 16;
                    dst_col = dst_scalar_idx % 16;
                }

                int src_row = dst_col;
                int src_col = dst_row;

                int src_scalar_idx = is_row ? (src_row * 16 + src_col) : (src_col * 16 + src_row);
                int src_lane = src_scalar_idx / 16;
                int src_offset = src_scalar_idx % 16;

                auto src_lane_scalars = sycl::select_from_group(sg, lane_scalars, src_lane);
                const auto v = src_lane_scalars[src_offset];
                if(comp == 0) out.x() = v; else out.y() = v;
            }
            dst.data[w] = out;
        }
    } else {
        #pragma unroll
        for(int w = 0; w < dst.packed_per_thread; w++) {
            typename rt_base<T, layout>::T2 out{};
            #pragma unroll
            for(int comp = 0; comp < 2; comp++) {
                int dst_row, dst_col;
                lane_word_to_coord<warp_threads, !is_row>(lane, w, comp, dst_row, dst_col);

                int src_row = dst_col;
                int src_col = dst_row;
                auto src_loc = coord_to_lane_word<warp_threads, is_row>(src_row, src_col);

                auto packed_src = sycl::select_from_group(sg, src.data[src_loc.word], src_loc.lane);
                const auto v = (src_loc.comp == 0) ? packed_src.x() : packed_src.y();
                if(comp == 0) out.x() = v; else out.y() = v;
            }
            dst.data[w] = out;
        }
    }
}

template<typename T, ducks::rt_layout::all layout>
inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    constexpr bool is_row = std::is_same_v<layout, ducks::rt_layout::row>;
    constexpr int warp_threads = WARP_THREADS;
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int lane = laneid();

    if constexpr (warp_threads == 16) {
        using comp_t = typename rt_base<T, layout>::T;
        sycl::vec<comp_t, 16> lane_scalars;

        #pragma unroll
        for(int w = 0; w < src.packed_per_thread; ++w) {
            auto pack = src.data[w];
            lane_scalars[2 * w]     = pack.x();
            lane_scalars[2 * w + 1] = pack.y();
        }

        #pragma unroll
        for(int w = 0; w < dst.packed_per_thread; w++) {
            typename rt_base<T, layout>::T2 out{};
            #pragma unroll
            for(int comp = 0; comp < 2; comp++) {
                int dst_row, dst_col;
                lane_word_to_coord<warp_threads, is_row>(lane, w, comp, dst_row, dst_col);
                int src_row = dst_col;
                int src_col = dst_row;

                int src_scalar_idx = is_row ? (src_row * 16 + src_col) : (src_col * 16 + src_row);
                int src_lane = src_scalar_idx / 16;
                int src_offset = src_scalar_idx % 16;

                auto src_lane_scalars = sycl::select_from_group(sg, lane_scalars, src_lane);
                const auto v = src_lane_scalars[src_offset];
                if(comp == 0) out.x() = v; else out.y() = v;
            }
            dst.data[w] = out;
        }
    } else {
        #pragma unroll
        for(int w = 0; w < dst.packed_per_thread; w++) {
            typename rt_base<T, layout>::T2 out{};
            #pragma unroll
            for(int comp = 0; comp < 2; comp++) {
                int dst_row, dst_col;
                lane_word_to_coord<warp_threads, is_row>(lane, w, comp, dst_row, dst_col);

                int src_row = dst_col;
                int src_col = dst_row;

                auto src_loc = coord_to_lane_word<warp_threads, is_row>(src_row, src_col);
                auto packed_src = sycl::select_from_group(sg, src.data[src_loc.word], src_loc.lane);
                const auto v = (src_loc.comp == 0) ? packed_src.x() : packed_src.y();
                if(comp == 0) out.x() = v; else out.y() = v;
            }
            dst.data[w] = out;
        }
    }
}
#else
/**
 * @brief Swaps the layout of a register base tile.
 *
 * This function swaps the layout of a register base tile by performing a series of layout swaps
 * on its constituent bf16_2 elements. It is used to change the data layout within a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T, ducks::rt_layout::all layout>
inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it.
    typename rt_base<T, layout>::T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T, ducks::rt_layout::all layout>
inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    typename rt_base<T, layout>::T2 data1_cache = src.data[1];
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
#endif // KITTENS_INTEL

/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * This function swaps the layout of a register base tile in place by casting it to the
 * transposed layout type and then performing the layout swap.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}
/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type, shape>& swap_layout_inplace(rt<T2, _rows, _cols, layout, shape> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */
/**
 * @brief Transposes a register tile.
 *
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the src register tile, and the width of the dst tile.
 * @tparam _width The width of the src register tile, and the height of the dst tile.
 * @tparam layout The layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<ducks::rt::all RT>
static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout>
static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}
#ifdef KITTENS_HOPPER
/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template <typename T2, typename U2, int _height, int _width,
          ducks::rt_layout::all layout>
/*
DPCT1110:417: The total declared local variable size in device function copy
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static inline void copy(rt<T2, _height, _width, layout> &dst,
                        const rt<U2, _height, _width, layout> &src) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if constexpr ((std::is_same_v<U2, float> && std::is_same_v<T2, fp8e4m3>) ||
                  (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e5m2>) ||
                  (std::is_same_v<U2, kittens::bf16> &&
                   std::is_same_v<T2, fp8e4m3>) ||
                  (std::is_same_v<U2, kittens::bf16> &&
                   std::is_same_v<T2, fp8e5m2>) ||
                  (std::is_same_v<U2, sycl::half> &&
                   std::is_same_v<T2, fp8e4m3>) ||
                  (std::is_same_v<U2, sycl::half> &&
                   std::is_same_v<T2, fp8e5m2>)) {
        // FLOAT (SRC -- 1H x 2W) to FP8 (DST -- 1H x 1W)
        int laneid = item_ct1.get_local_id(2) % 32;

#pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for(int k = 0; k < dst.tiles[0][0].packed_per_thread; k++) {

                    // check for half, float, bf16
                    using src_t = std::conditional_t<
                        std::is_same_v<U2, float>, sycl::float2,
                        std::conditional_t<std::is_same_v<U2, kittens::bf16>,
                                           bf16_2, sycl::half2>>;
                    src_t val1, val2;

                    // Put something up for adoption
                    if (laneid % 2 == 0) {
                        // put up src left core matrix first as 0, 2
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                    } else {
                        // put up src right core matrix first as 1, 3
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                    }

                    // Shuffle first 4 floats
                    int row_mask = 4 * ( laneid / 4 );
                    int row_offset = row_mask + ( (laneid-row_mask) / 2 ) + ( laneid % 2 );
                    int src_offset = (laneid % 2 == 0 ) ? row_offset + 0 : ( row_offset + 1 );
                    src_t val01 = packed_shfl_sync(MASK_ALL, val1, src_offset);  // Get from even thread

                    int src_offset2 = (laneid % 4 < 2 ) ? src_offset + 1 : (src_offset - 1);
                    src_t val23 = packed_shfl_sync(MASK_ALL, val2, src_offset2);  // Get from odd thread

                    // Convert to fp8e4m3_4
                    sycl::float4 f4;
                    using fp8_4_t = std::conditional_t<std::is_same_v<T2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t f4_fp8;
                    if ( laneid % 4 < 2 ) {
                        f4.x() = val01.x(); // Thread 2N's first value
                        f4.y() = val01.y(); // Thread 2N's second value
                        f4.z() = val23.x(); // Thread 2N+1's first value
                        f4.w() = val23.y(); // Thread 2N+1's second value
                        f4_fp8 =
                            base_types::convertor<fp8_4_t,
                                                  sycl::float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    } else {
                        f4.x() = val23.x(); // Thread 2N+1's first value
                        f4.y() = val23.y(); // Thread 2N+1's second value
                        f4.z() = val01.x(); // Thread 2N's first value
                        f4.w() = val01.y(); // Thread 2N's second value
                        f4_fp8 =
                            base_types::convertor<fp8_4_t,
                                                  sycl::float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    }
                }
            }
        }
    } else if constexpr ((std::is_same_v<U2, fp8e4m3> &&
                          std::is_same_v<T2, float>) ||
                         (std::is_same_v<U2, fp8e5m2> &&
                          std::is_same_v<T2, float>) ||
                         (std::is_same_v<U2, fp8e4m3> &&
                          std::is_same_v<T2, kittens::bf16>) ||
                         (std::is_same_v<U2, fp8e5m2> &&
                          std::is_same_v<T2, kittens::bf16>) ||
                         (std::is_same_v<U2, fp8e4m3> &&
                          std::is_same_v<T2, sycl::half>) ||
                         (std::is_same_v<U2, fp8e5m2> &&
                          std::is_same_v<T2, sycl::half>)) {
        // FP8 (SRC -- 1H x 1W) to FLOAT (DST -- 1H x 2W)
        int laneid = item_ct1.get_local_id(2) % 32;

#pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                #pragma unroll
                for(int k = 0; k < src.tiles[0][0].packed_per_thread; k++) {
                    int dst_j = 2*j + k/2;

                    // Put something up for adoption
                    using fp8_4_t = std::conditional_t<std::is_same_v<U2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t val = src.tiles[i][j].data[k];
                    sycl::float4 f4 =
                        base_types::convertor<sycl::float4, fp8_4_t>::convert(
                            val);
                    sycl::float2 f2_0, f2_1;
                    if ( laneid % 4 < 2 ) { // src 0 and 1 should put up .x and .y first
                        f2_0 = sycl::float2(f4.x(), f4.y());
                        f2_1 = sycl::float2(f4.z(), f4.w());
                    }
                    else { // src 2 and 3 should put up .z and .w first
                        f2_0 = sycl::float2(f4.z(), f4.w());
                        f2_1 = sycl::float2(f4.x(), f4.y());
                    }

                    int row_offset = 4 * (laneid/4) + (laneid%2) * 2 + (laneid%4) / 2;
                    sycl::float2 f2_0_shfl =
                        packed_shfl_sync(MASK_ALL, f2_0, row_offset);
                    sycl::float2 f2_1_shfl =
                        packed_shfl_sync(MASK_ALL, f2_1, row_offset ^ 2);

                    // convert to dst type if needed
                    using dst_t = std::conditional_t<
                        std::is_same_v<T2, float>, sycl::float2,
                        std::conditional_t<std::is_same_v<T2, kittens::bf16>,
                                           bf16_2, sycl::half2>>;
                    if constexpr (!(std::is_same_v<T2, float>)) {
                        dst_t f2_0_shfl_t =
                            base_types::convertor<dst_t, sycl::float2>::convert(
                                f2_0_shfl);
                        dst_t f2_1_shfl_t =
                            base_types::convertor<dst_t, sycl::float2>::convert(
                                f2_1_shfl);
                        if (laneid % 2 == 0) {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl_t;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl_t;
                        }
                    } else {
                        if (laneid % 2 == 0) {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl;
                        }
                    }
                }
            }
        }
    }
    // default case where the layouts map 1:1 in thread ownership logic
    else {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                copy(dst.tiles[i][j], src.tiles[i][j]);
            }
        }
    }
}
#else
/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}
#endif

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0, int part = 0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
      for (int j = 0; j < dst.width; j++) {
        if (j < i) { // below the diagonal, copy
#pragma unroll
          for (int k = 0; k < dst.packed_per_base_tile; k++) {
            dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
          }
        } else if (j > i) { // above the diagonal, zero
#pragma unroll
          for (int k = 0; k < dst.packed_per_base_tile; k++) {
            dst.tiles[i][j].data[k] = packed_val;
          }
        } else { // on the diagonal, interesting!
          const int lane = laneid();

          if constexpr (std::is_same_v<typename RT::shape,
                                       typename ducks::rt_shape::rt_16x16>) {
            constexpr uint16_t MASKS[16] = {
                0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
                0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};

#pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
              if ((MASKS[k * 2] >> lane) & 1) {
                dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x();
              } else {
                dst.tiles[i][j].data[k].x() = val;
              }
              if ((MASKS[k * 2 + 1] >> lane) & 1) {
                dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y();
              } else {
                dst.tiles[i][j].data[k].y() = val;
              }
            }

          } else if constexpr (std::is_same_v<
                                   typename RT::shape,
                                   typename ducks::rt_shape::rt_16x32>) {
            constexpr uint64_t MASKS_1[32] = {
                0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
                0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};
            constexpr uint64_t MASKS_2[32] = {
                0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
                0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};
            auto MASKS = (part == 0) ? MASKS_1 : MASKS_2;
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
              if ((MASKS[k * 2] >> lane) & 1) {
                dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x();
              } else {
                dst.tiles[i][j].data[k].x() = val;
              }
              if ((MASKS[k * 2 + 1] >> lane) & 1) {
                dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y();
              } else {
                dst.tiles[i][j].data[k].y() = val;
              }
            }
          } else if constexpr (std::is_same_v<
                                   typename RT::shape,
                                   typename ducks::rt_shape::rt_8x32>) {
            // 8-row Q-tile (FWD_SEQ_Q=8). A 32-col KV tile spans ratio=4
            // diagonal sub-blocks of 8 cols each; `part` in {0,1,2,3} selects
            // which 8-col band carries the causal diagonal. Layout (validated
            // against rt_16x32 above): lane = column within a 16-col DPAS
            // sub-tile; data[0..3] = cols 0-15, data[4..7] = cols 16-31;
            // data[k].x/.y = rows 2*(k%4), 2*(k%4)+1. Keep element (row r,
            // col c) iff c <= 8*part + r.
            constexpr uint64_t MASKS_P[4][16] = {
                // part=0: diagonal in cols 0-7
                {0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
                 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                // part=1: diagonal in cols 8-15
                {0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
                 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                // part=2: cols 0-15 kept, diagonal in cols 16-23
                {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF},
                // part=3: cols 0-23 kept, diagonal in cols 24-31
                {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF},
            };
            const uint64_t *MASKS = MASKS_P[part];
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
              if ((MASKS[k * 2] >> lane) & 1) {
                dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x();
              } else {
                dst.tiles[i][j].data[k].x() = val;
              }
              if ((MASKS[k * 2 + 1] >> lane) & 1) {
                dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y();
              } else {
                dst.tiles[i][j].data[k].y() = val;
              }
            }
          } else {
            static_assert(false, "Unsupported rt shape");
          }
          sycl::group_barrier(
              sycl::ext::oneapi::this_work_item::get_sub_group());
        }
      }
    }
}

    /**
     * @brief Makes a square register tile anti-causal by zeroing elements below
     * the main diagonal.
     *
     * This function modifies a square register tile in-place to make it
     * anti-causal. All elements below the main diagonal are set to zero, while
     * elements on or above the main diagonal are left unchanged.
     *
     * @tparam T The data type of the register tile elements.
     * @tparam _size The size (height and width) of the square register tile.
     * @tparam layout The current layout of the register tile.
     * @param tile[in,out] Reference to the register tile to be made causal.
     */
    template <ducks::rt::row_layout RT>
    static inline void make_causal_t(
        RT & dst, const RT &src,
        const typename base_types::packing<typename RT::dtype>::unpacked_type
            &val = 0) {
      const typename RT::dtype packed_val =
          base_types::packing<typename RT::dtype>::pack(val);
#ifdef KITTENS_HOPPER
      static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> &&
                        !std::is_same_v<typename RT::dtype, fp8e5m2_4>,
                    "Unsupported type for make_causal");
#endif
#pragma unroll
      for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // above the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // below the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0x88CCEEF;
                constexpr uint32_t MASK_Y = 0x88CCEEFF;

                dst.tiles[i][j].data[1] = packed_val;              // below diagonal, zero
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2]; // above diagonal, copy

                // on the diagonal or above
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x() = src.tiles[i][j].data[0].x();
                    dst.tiles[i][j].data[3].x() = src.tiles[i][j].data[3].x();
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].x() = val;
                    dst.tiles[i][j].data[3].x() = val;
                }

                // on the diagonal or above
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y() = src.tiles[i][j].data[0].y();
                    dst.tiles[i][j].data[3].y() = src.tiles[i][j].data[3].y();
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].y() = val;
                    dst.tiles[i][j].data[3].y() = val;
                }

            }
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
        }
    }
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + ((k % 2) * 8) + (laneid() / 4);
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                if (global_col_idx_x <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                else                                               { dst.tiles[i][j].data[k].x() = val; }

                if (global_col_idx_y <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
                else                                               { dst.tiles[i][j].data[k].y() = val; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}
template<ducks::rt::col_layout RT>
static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4);

                if (global_col_idx <= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                else                                               { dst.tiles[i][j].data[k].x() = val; }
                if (global_col_idx <= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
                else                                               { dst.tiles[i][j].data[k].y() = val; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}

/**
 * @brief Makes a register tile triangular by zeroing elements below the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + ((k % 2) * 8) + (laneid() / 4);
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                if (global_col_idx_x >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                else                                               { dst.tiles[i][j].data[k].x() = val; }

                if (global_col_idx_y >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
                else                                               { dst.tiles[i][j].data[k].y() = val; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}
template<ducks::rt::col_layout RT>
static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4);

                if (global_col_idx >= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                else                                               { dst.tiles[i][j].data[k].x() = val; }

                if (global_col_idx >= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
                else                                               { dst.tiles[i][j].data[k].y() = val; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile right filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= dst.cols) return;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x() = val; }                         // NYI
                else                       { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); } // NYI
                if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y() = val; }                         // NYI
                else                       { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); } // NYI
            }
        }
    }
}
template<ducks::rt::col_layout RT>
static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int t_col_idx = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4);
                if (t_col_idx >= col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}

/**
 * @brief Makes a register tile left filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (col_idx_x < col_idx)  { dst.tiles[i][j].data[k].x() = val; }
                else                      { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                if (col_idx_y < col_idx)  { dst.tiles[i][j].data[k].y() = val; }
                else                      { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_col = (j * dst.tile_size_col) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_col < col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}

/**
 * @brief Makes a register tile upper filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_row < row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (row_idx_x < row_idx)  { dst.tiles[i][j].data[k].x() = val; }
                else                      { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                if (row_idx_y < row_idx)  { dst.tiles[i][j].data[k].y() = val; }
                else                      { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
            }
        }
    }
}

/**
 * @brief Makes a register tile lower filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom of the tile (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= dst.rows) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_row >= row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                        { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (row_idx_x >= row_idx)  { dst.tiles[i][j].data[k].x() = val; }
                else                       { dst.tiles[i][j].data[k].x() = src.tiles[i][j].data[k].x(); }
                if (row_idx_y >= row_idx)  { dst.tiles[i][j].data[k].y() = val; }
                else                       { dst.tiles[i][j].data[k].y() = src.tiles[i][j].data[k].y(); }
            }
        }
    }
}


/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}

}