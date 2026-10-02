/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <type_traits>

#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"
#include "../util/util.dp.hpp"

namespace kittens {

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
inline static void load(RT &dst, const ST &src) {

    // sycl::ext::oneapi::experimental::printf("ENTERED THE LOAD\n");

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = kittens::laneid();

    // // convert to shared state space
    uintptr_t shared_addr = reinterpret_cast<uintptr_t>(&src.data[0]);

    // #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
#ifdef KITTENS_INTEL
            // Intel SIMD16 register layout:
            // row_layout: lane L holds column L, data[k] = {M[2k][L], M[2k+1][L]}
            // col_layout: lane L holds row L,    data[k] = {M[L][2k], M[L][2k+1]}
            // NOTE: use move_local<> (opencl_local pointer qualifier) to avoid
            // IGC's generic-address-space runtime dispatch (was adding 144+ extra
            // control-flow branches per kernel for v40-style SLM staging).
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    int row = i * dst.base_tile_rows + 2*k;
                    int col = j * dst.base_tile_cols + laneid;
                    U val0, val1;
                    move_local<U>::lds(val0, src.idx(shared_addr, {row, col}));
                    move_local<U>::lds(val1, src.idx(shared_addr, {row+1, col}));
                    U2 packed;
                    packed.x() = val0;
                    packed.y() = val1;
                    dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(packed);
                }
            }
            else {
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    int row = i * dst.base_tile_rows + laneid;
                    int col = j * dst.base_tile_cols + 2*k;
                    U val0, val1;
                    move_local<U>::lds(val0, src.idx(shared_addr, {row, col}));
                    move_local<U>::lds(val1, src.idx(shared_addr, {row, col+1}));
                    U2 packed;
                    packed.x() = val0;
                    packed.y() = val1;
                    dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(packed);
                }
            }
#else
            if constexpr (sizeof(typename ST::dtype) == 2) { // half and bfloat16
                // handle 16-bit types

                U2 tmp[4];
                int row = i*dst.base_tile_rows + (laneid % 16);
                int col = j*dst.base_tile_cols + (laneid / 16) * 8;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1 ) {
                U2 tmp[4];
                int row = i*dst.base_tile_rows + (laneid % 16);
                int col = j*dst.base_tile_cols + (laneid / 16) * 16;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                int row, col;
                if constexpr (ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols) {
                    row = i*dst.base_tile_rows + (laneid / 4);
                    col = j*dst.base_tile_cols + 2*(laneid % 4);
                }
                else {
                    row = i*dst.base_tile_rows + (laneid / 4)   + src.row_offset;
                    col = j*dst.base_tile_cols + 2*(laneid % 4) + src.col_offset;
                }
                int blit = sizeof(typename ST::dtype)*((laneid%4)/2);
                U2 tmp[4];
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uintptr_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uintptr_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);

                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;

                move<U>::lds(tmp[0].x(), (addr_1+ 0)^swizzle_1);
                move<U>::lds(tmp[0].y(), (addr_1+ 4)^swizzle_1);
                move<U>::lds(tmp[2].x(), (addr_1+32)^swizzle_1);
                move<U>::lds(tmp[2].y(), (addr_1+36)^swizzle_1);
                move<U>::lds(tmp[1].x(), (addr_2+ 0)^swizzle_2);
                move<U>::lds(tmp[1].y(), (addr_2+ 4)^swizzle_2);
                move<U>::lds(tmp[3].x(), (addr_2+32)^swizzle_2);
                move<U>::lds(tmp[3].y(), (addr_2+36)^swizzle_2);

                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);

                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        dst.tiles[i][j].data[k] = T2{dst.tiles[i][j].data[k].y(), dst.tiles[i][j].data[k].x()};
                    }
                }
            }
            else if constexpr (sizeof(typename ST::dtype) != 1) {
                // handle the column-major layout
                U2 tmp[4];
                int row = i*dst.base_tile_rows + 2*(laneid % 4);
                int col = j*dst.base_tile_cols + (laneid / 4);
                move<U>::lds(tmp[0].x(), src.idx(shared_addr, {row+0, col+0}));
                move<U>::lds(tmp[0].y(), src.idx(shared_addr, {row+1, col+0}));
                move<U>::lds(tmp[1].x(), src.idx(shared_addr, {row+0, col+8}));
                move<U>::lds(tmp[1].y(), src.idx(shared_addr, {row+1, col+8}));
                move<U>::lds(tmp[2].x(), src.idx(shared_addr, {row+8, col+0}));
                move<U>::lds(tmp[2].y(), src.idx(shared_addr, {row+9, col+0}));
                move<U>::lds(tmp[3].x(), src.idx(shared_addr, {row+8, col+8}));
                move<U>::lds(tmp[3].y(), src.idx(shared_addr, {row+9, col+8}));
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
#endif
        }
    }
}


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    // convert to shared state space
    uintptr_t shared_addr =reinterpret_cast<uintptr_t>(&dst.data[0]);

    int laneid = kittens::laneid();
#pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
#ifdef KITTENS_INTEL
            // Use move_local<> for SLM stores — see comment on load() above.
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                #pragma unroll
                for(int k = 0; k < src.packed_per_base_tile; k++) {
                    int row = i * src.base_tile_rows + 2*k;
                    int col = j * src.base_tile_cols + laneid;
                    U2 packed = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                    move_local<U>::sts(dst.idx(shared_addr, {row, col}), packed.x());
                    move_local<U>::sts(dst.idx(shared_addr, {row+1, col}), packed.y());
                }
            }
            else {
                #pragma unroll
                for(int k = 0; k < src.packed_per_base_tile; k++) {
                    int row = i * src.base_tile_rows + laneid;
                    int col = j * src.base_tile_cols + 2*k;
                    U2 packed = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                    move_local<U>::sts(dst.idx(shared_addr, {row, col}), packed.x());
                    move_local<U>::sts(dst.idx(shared_addr, {row, col+1}), packed.y());
                }
            }
#else
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // handle the 16-bit types
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    int row = i*src.base_tile_rows + (laneid / 4);
                    int col = j*src.base_tile_cols + 2*(laneid % 4);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[1]);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[2]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3]);
                }
                else {
                    int row = i*src.base_tile_rows + 2*(laneid % 4);
                    int col = j*src.base_tile_cols + (laneid / 4);
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x());
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y());
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x());
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y());
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x());
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y());
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x());
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y());
                }
            } else if constexpr (sizeof(typename ST::dtype) == 1) {
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                int row = i*src.base_tile_rows + (laneid % 16);
                int col = j*src.base_tile_cols + (laneid / 16) * 16;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::stsm4(dst.idx(shared_addr, {row, col}), tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                else {
                    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
                }
            } else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                int row, col;
                if constexpr (ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols) {
                    row = i*src.base_tile_rows + (laneid / 4);
                    col = j*src.base_tile_cols + 2*(laneid % 4);
                }
                else {
                    row = i*src.base_tile_rows + (laneid / 4)   + dst.row_offset;
                    col = j*src.base_tile_cols + 2*(laneid % 4) + dst.col_offset;
                }
                int blit = sizeof(typename ST::dtype)*((laneid%4) / 2);
                T2 reg_tmp[4];
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = T2{src.tiles[i][j].data[k].y(), src.tiles[i][j].data[k].x()};
                    }
                }
                else {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = src.tiles[i][j].data[k];
                    }
                }
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(reg_tmp[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(reg_tmp[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(reg_tmp[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(reg_tmp[3]);
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uintptr_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uintptr_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;
                move<U>::sts((addr_1+ 0)^swizzle_1, tmp[0].x());
                move<U>::sts((addr_1+ 4)^swizzle_1, tmp[0].y());
                move<U>::sts((addr_1+32)^swizzle_1, tmp[2].x());
                move<U>::sts((addr_1+36)^swizzle_1, tmp[2].y());
                move<U>::sts((addr_2+ 0)^swizzle_2, tmp[1].x());
                move<U>::sts((addr_2+ 4)^swizzle_2, tmp[1].y());
                move<U>::sts((addr_2+32)^swizzle_2, tmp[3].x());
                move<U>::sts((addr_2+36)^swizzle_2, tmp[3].y());
            }
            else if constexpr (sizeof(typename ST::dtype) != 1) {
                // handle the column-major layout
                int row = i*src.base_tile_rows + 2*(laneid % 4);
                int col = j*src.base_tile_cols + (laneid / 4);
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x());
                move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y());
                move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x());
                move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y());
                move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x());
                move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y());
                move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x());
                move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y());
            }
#endif
        }
    }
}

/**
 * @brief Store a register tile into a shared tile with a logical transpose.
 *
 * Produces `dst[r, c] = src[c, r]` (i.e. the shared tile holds the logical
 * transpose of the register tile). Intended for cooperative-SLM patterns
 * (flash-attn bwd: P, dS written transposed, then read by other SGs for
 * the subsequent gemm without any register shuffle).
 *
 * Requirements (compile-time):
 *   RT::height == ST::width   (tile-grid transpose)
 *   RT::width  == ST::height
 *
 * Current impl targets Intel PVC (SIMD16 DPAS row_l / col_l). Leverages the
 * fact that a "transposed" store is exactly the store pattern used for the
 * OPPOSITE layout within each 16×16 base tile, plus a swap of (i,j) tile
 * position.
 *
 * @tparam RT register tile type (row_l or col_l)
 * @tparam ST shared tile type (the transposed shape of RT)
 */
template<ducks::rt::all RT, ducks::st::all ST>
inline static void store_transpose(ST &dst, const RT &src) {
#ifdef KITTENS_INTEL
    static_assert(RT::height == ST::width,
                  "store_transpose: RT::height must equal ST::width");
    static_assert(RT::width  == ST::height,
                  "store_transpose: RT::width must equal ST::height");

    using T2 = RT::dtype;
    using U  = ST::dtype;
    using U2 = base_types::packing<U>::packed_type;

    uintptr_t shared_addr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    int laneid = kittens::laneid();

#pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            // In the transposed dst tile, src.tiles[i][j] lives at dst position (j, i).
            // Each 16×16 base tile has the *opposite* access pattern vs. non-transposed
            // store, because transposing a row_l tile yields a col_l-shaped memory write
            // (and vice-versa).
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                // row_l: lane L owns column L; data[k] = {M[2k][L], M[2k+1][L]}.
                // Transposed: shared_T[L][2k] = M[2k][L], shared_T[L][2k+1] = M[2k+1][L].
                #pragma unroll
                for(int k = 0; k < src.packed_per_base_tile; k++) {
                    int row_T = j * src.base_tile_cols + laneid;
                    int col_T = i * src.base_tile_rows + 2*k;
                    U2 packed = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                    move<U>::sts(dst.idx(shared_addr, {row_T, col_T}),   packed.x());
                    move<U>::sts(dst.idx(shared_addr, {row_T, col_T+1}), packed.y());
                }
            }
            else {
                // col_l: lane L owns row L; data[k] = {M[L][2k], M[L][2k+1]}.
                // Transposed: shared_T[2k][L] = M[L][2k], shared_T[2k+1][L] = M[L][2k+1].
                #pragma unroll
                for(int k = 0; k < src.packed_per_base_tile; k++) {
                    int row_T = j * src.base_tile_cols + 2*k;
                    int col_T = i * src.base_tile_rows + laneid;
                    U2 packed = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                    move<U>::sts(dst.idx(shared_addr, {row_T,   col_T}), packed.x());
                    move<U>::sts(dst.idx(shared_addr, {row_T+1, col_T}), packed.y());
                }
            }
        }
    }
#else
    static_assert(std::is_same_v<RT, void>,
                  "kittens::store_transpose is only implemented for KITTENS_INTEL");
#endif
}

}