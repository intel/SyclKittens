/**
 * @file
 * @brief Functions for transferring data directly between global memory and
 * registers and back.
 */

#pragma once

#include "../../../../common/common.dp.hpp"
#include "../../../../types/sycl_type.hpp"
#include "../../../../types/types.dp.hpp"
#include "types/register/rt.dp.hpp"
#include "types/register/rt_shape.hpp"
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/math.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace kittens {

template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<RT>>
inline static void load_part(RT &dst, const GL &src, const COORD &idx, const std::tuple<int, int> tile_posistion) {
  using T2 = RT::dtype;
  using U = typename GL::dtype;
  using U2 = base_types::packing<U>::packed_type;

  // Fold the contiguous (column) element offset into the 2D-block-load surface
  // coordinate instead of baking it into the base pointer. Baking it into
  // src_ptr misaligns the base for odd rt_16x16 tiles (col offset = 16 bf16 =
  // 32B) but Subgroup2DBlockLoadINTEL requires a 64B-aligned base. The column
  // coordinate is surface-addressed by the hardware with no such constraint.
  auto uc = idx.template unit_coord<axis, 3>();
  const int col_offset = uc.c;
  uc.c = 0;
  U *src_ptr = (U *)&src[uc];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height =  src.template shape<axis>();

  const int base_row = std::get<0>(tile_posistion) * dst.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * dst.base_tile_cols + col_offset;
#pragma unroll
  for (int i = 0; i < dst.height; i++) {
    const int row = base_row + i * dst.base_tile_rows;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
      const int col = base_col + j * dst.base_tile_cols;
      U2 data[RT::packed_per_base_tile];
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_32x32>) {
        if constexpr (sizeof(U) <= 2) {
          // bf16 / half: single (16-wide, 32-tall, 2-block) load fits (2*16*32*2=2KB, 128B/lane).
          void *load_destination = std::is_same_v<T2, U2>
              ? (void *)&dst.tiles[i][j].data[0] : (void *)&data[0];
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 32, 2, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, load_destination);
          if constexpr (std::is_same_v<T2, U2>) continue;
        } else {
          // fp32: driver lacks the (32b, 32r, 16c x 2 blocks) block-read intrinsic
          // (__internal_intel_sub_group_2d_block_read_32b_32r16x2c_cache_controls).
          // Split into 8x (16-wide, 8-tall, 1-block) loads matching the store_part
          // layout so load/store roundtrips preserve element ordering.
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, (void *)&data[0]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row + 8}, (void *)&data[4]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row + 16}, (void *)&data[8]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row + 24}, (void *)&data[12]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row}, (void *)&data[16]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row + 8}, (void *)&data[20]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row + 16}, (void *)&data[24]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row + 24}, (void *)&data[28]);
        }
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        if constexpr (sizeof(U) <= 2) {
          // bf16/half: BlockCount=2 is within 64B limit (2*16*2=64)
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 16, 2, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, (void *)&data[0]);
        } else {
          // float: BlockCount=2 would be 4*16*2=128B, exceeds 64B limit.
          // Split into two BlockCount=1 loads.
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 16, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, (void *)&data[0]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 16, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row}, (void *)&data[8]);
        }
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_8x32>) {
        if constexpr (sizeof(U) <= 2) {
          // bf16/half: 8-tall x 16-wide x 2 col-blocks (2*16*2=64B, within limit).
          // Fills data[0-3]=rows0-7/cols0-15, data[4-7]=rows0-7/cols16-31
          // (top-8-rows of the rt_16x32 layout).
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 2, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, (void *)&data[0]);
        } else {
          // float: BlockCount=2 would exceed 64B; split into two BlockCount=1.
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row}, (void *)&data[0]);
          __spirv_Subgroup2DBlockLoadINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col + 16, row}, (void *)&data[4]);
        }
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_32x16>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 32, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&data[0]);
      } else {
        static_assert(!std::is_same_v<RT, RT>, "Unsupported row-layout load shape");
      }
#pragma unroll
      for (int k = 0; k < RT::packed_per_base_tile; k++) {
        dst.tiles[i][j].data[k] =
            base_types::convertor<T2, U2>::convert(*(U2 *)&data[k]);
      }
    }
  }
}


template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load(RT &dst, const GL &src, const COORD &idx) {
  load_part<axis, RT, GL>(dst, src, idx, {0,0});
}

template <int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load_part(RT &dst, const GL &src, const COORD &idx,
                             const std::tuple<int, int> tile_posistion) {
  using T2 = RT::dtype;
  using T = base_types::packing<typename RT::dtype>::unpacked_type;
  using U = typename GL::dtype;

  using U2 = base_types::packing<U>::packed_type;
  constexpr int word_size = std::is_same_v<U, float> ? 2 : 1;
  // Fold the contiguous (column) element offset into the surface coordinate
  // instead of the base pointer (see row-layout load_part above) so odd
  // rt_16x16 N-tiles keep a 64B-aligned base for Subgroup2DBlockLoadTransform.
  auto uc = idx.template unit_coord<axis, 3>();
  const int col_offset = uc.c;
  uc.c = 0;
  U *src_ptr = (U *)&src[uc];
  const int row_stride = src.template stride<axis>();

  const int base_row = std::get<0>(tile_posistion) * dst.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * dst.base_tile_cols + col_offset;

#pragma unroll
  for (int i = 0; i < dst.height; i++) {
      const int row = base_row + i * dst.base_tile_rows;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
      const int col = base_col + j * dst.base_tile_cols;
      U2 data[RT::packed_per_base_tile];
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_32x32>) {
        __spirv_Subgroup2DBlockLoadTransformINTEL(
            sizeof(U), 16, 32, 2, (void *)src_ptr,
            src.template stride<axis>() * sizeof(U), src.template shape<axis>(),
            row_stride * sizeof(U), intel::coord_t{col, row}, (void *)&data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockLoadTransformINTEL(
            sizeof(U), 16, 16, 1, (void *)src_ptr,
            src.template stride<axis>() * sizeof(U), src.template shape<axis>(),
            row_stride * sizeof(U), intel::coord_t{col, row}, (void *)&data[0]);

      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        __spirv_Subgroup2DBlockLoadTransformINTEL(
            sizeof(U), 16, 16, 2, (void *)src_ptr,
            src.template stride<axis>() * sizeof(U), src.template shape<axis>(),
            row_stride * sizeof(U), intel::coord_t{col, row}, (void *)&data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_32x16>) {
        __spirv_Subgroup2DBlockLoadTransformINTEL(
            sizeof(U), 16, 32, 1, (void *)src_ptr,
            src.template stride<axis>() * sizeof(U), src.template shape<axis>(),
            row_stride * sizeof(U), intel::coord_t{col, row}, (void *)&data[0]);
      } else {
        static_assert(!std::is_same_v<RT, RT>, "Unsupported column-layout load shape");
      }
#pragma unroll
      for (int k = 0; k < RT::packed_per_base_tile; k++) {
        dst.tiles[i][j].data[k] =
            base_types::convertor<T2, U2>::convert(*(U2 *)&data[k]);
      }
    }
  }
}

template <int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load(RT &dst, const GL &src, const COORD &idx) {
  load_part<axis, RT, GL>(dst, src, idx, {0,0});
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>,
          typename std::enable_if<!std::is_same<typename GL::T, float>::value,
                                  int>::type = 0>
inline static void load_transpose_part(RT &dst, const GL &src, const COORD &idx, std::tuple<int, int> tile_posistion) {
  using T2 = RT::dtype;
  using U = typename GL::dtype;
  using U2 = base_types::packing<U>::packed_type;

  const int base_row = std::get<0>(tile_posistion) * dst.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * dst.base_tile_cols;

  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height =  src.template shape<axis>();
  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];

#pragma unroll
  for (int i = 0; i < dst.height; i++) {
    int row = base_row +i * dst.base_tile_rows;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
      int col = base_col + j * dst.base_tile_cols;
      U2 data[RT::packed_per_base_tile];
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2, row}, (void *)&data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_32x32>) {
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2, row}, (void *)&data[0]);
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2 + 8, row}, (void *)&data[8]);
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2, row + 16}, (void *)&data[16]);
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2 + 8, row + 16}, (void *)&data[24]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2, row}, (void *)&data[0]);
        __spirv_Subgroup2DBlockLoadTransposeINTEL(
            sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{(col) / 2 + 8, row}, (void *)&data[8]);
        } else {
          static_assert(!std::is_same_v<RT, RT>, "Unsupported transpose load shape");
      }
#pragma unroll
      for (int k = 0; k < RT::packed_per_base_tile; k++) {
        dst.tiles[i][j].data[k] =
            base_types::convertor<T2, U2>::convert(*(U2 *)&data[k]);
      }
    }
  }
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>,
          typename std::enable_if<!std::is_same<typename GL::T, float>::value,
                                  int>::type = 0>
inline static void load_transpose(RT &dst, const GL &src, const COORD &idx) {
  load_transpose_part<axis, RT, GL>(dst, src, idx, {0,0});
}

// ============================================================================
// load-into-packed (packed-tile design, increment 2) — deposit a single 16x16
// bf16 base tile DIRECTLY into the dpas operand registers, skipping the
// rt_base.data[] round-trip AND the per-dpas operand marshalling that
// hmma81616 would otherwise emit every call.
//
// These reuse the EXACT same 2D-block-load intrinsics as load_part /
// load_transpose_part; only the destination pointer changes (packed operand
// register instead of a local data[] array + convert loop). The block-load
// bytes are bit-identical to what premarshal_A / premarshal_B reconstruct, so
// this is valid ONLY when the global dtype is bf16/half (no numeric convert).
//
// Motivation: narrow single-use operands (e.g. attention SdP) reload fresh
// every iteration and cannot amortise marshalling via CSE the way GEMM's wide
// reused tiles do. Fusing the pack INTO the load makes even fresh operands
// marshalling-free and is register-neutral (no extra persistent state).
// See docs/dsl_packed_tile_design.md. ADDITIVE — does not touch mma paths.
// ============================================================================

// A operand (row layout, mma<N,*>): 16x16 bf16 -> short8[2] (dst points at
// dpas_A_bf16_16x16::h). Mirrors load_part<rt_16x16> surface addressing.
// tile_position {row,col} indexes the base tile inside the surface (same
// convention as load_part); the contiguous column offset is folded into the
// 2D-block surface coordinate (NOT the base pointer) to keep the 64B base
// alignment the non-transpose block load requires.
template <int axis, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<rt_bf<16, 16>>>
inline static void load_packed_A_16x16(intel::short8 *dst /*[2]*/, const GL &src, const COORD &idx,
                                       const std::tuple<int, int> tile_position = {0, 0}) {
  using U = typename GL::dtype;
  static_assert(sizeof(U) <= 2, "load_packed_A_16x16 requires a bf16/half global (no numeric convert).");
  auto uc = idx.template unit_coord<axis, 3>();
  const int col_offset = uc.c;
  uc.c = 0;
  U *src_ptr = (U *)&src[uc];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height = src.template shape<axis>();
  const int base_row = std::get<0>(tile_position) * 16;
  const int base_col = std::get<1>(tile_position) * 16 + col_offset;
  __spirv_Subgroup2DBlockLoadINTEL(
      sizeof(U), 16, 16, 1, (void *)src_ptr, row_stride_width,
      memory_height, row_stride_width,
      intel::coord_t{base_col, base_row}, (void *)dst);
}
template <ducks::gl::all GL, ducks::coord::tile COORD = coord<rt_bf<16, 16>>>
inline static void load_packed_A_16x16(intel::short8 *dst, const GL &src, const COORD &idx,
                                       const std::tuple<int, int> tile_position = {0, 0}) {
  load_packed_A_16x16<2, GL, COORD>(dst, src, idx, tile_position);
}

// B operand (mma<N,T>, transpose load): 16x16 bf16 -> int8 (dst points at
// dpas_B_bf16_16x16::m). Mirrors load_transpose_part<rt_16x16> addressing:
// the (row,col) tile_position feeds the transpose block coordinate; any coord
// offset baked in idx (e.g. the query-row / head offset) is carried by the
// base pointer exactly as load_transpose_part does.
template <int axis, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<rt_bf<16, 16>>>
inline static void load_packed_Bt_16x16(intel::int8 *dst, const GL &src, const COORD &idx,
                                        const std::tuple<int, int> tile_position = {0, 0}) {
  using U = typename GL::dtype;
  using U2 = typename base_types::packing<U>::packed_type;
  static_assert(sizeof(U) <= 2, "load_packed_Bt_16x16 requires a bf16/half global (no numeric convert).");
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height = src.template shape<axis>();
  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
  const int base_row = std::get<0>(tile_position) * 16;
  const int base_col = std::get<1>(tile_position) * 16;
  __spirv_Subgroup2DBlockLoadTransposeINTEL(
      sizeof(U2), 8, 16, 1, (void *)src_ptr, row_stride_width,
      memory_height, row_stride_width,
      intel::coord_t{base_col / 2, base_row}, (void *)dst);
}
template <ducks::gl::all GL, ducks::coord::tile COORD = coord<rt_bf<16, 16>>>
inline static void load_packed_Bt_16x16(intel::int8 *dst, const GL &src, const COORD &idx,
                                        const std::tuple<int, int> tile_position = {0, 0}) {
  load_packed_Bt_16x16<2, GL, COORD>(dst, src, idx, tile_position);
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_part(RT &dst, const GL &src, const COORD &idx, const std::tuple<int, int> tile_posistion) {
  using U = typename GL::dtype;

  const int base_row = std::get<0>(tile_posistion) * dst.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * dst.base_tile_cols;

  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height =  src.template shape<axis>();
#pragma unroll
  for (int i = 0; i < dst.height; i++) {
    const int row = base_row + i * dst.base_tile_rows;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
      int col = base_col + j * dst.base_tile_cols;
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_32x32>) {
        if constexpr (sizeof(U) <= 2) {
          // bf16 / half: single (16-wide, 32-tall, 2-block) prefetch matches the load fast path.
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 32, 2, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width,
              intel::coord_t{col, row});
        } else {
          // fp32: driver lacks the (32b, 32r, 16c x 2 blocks) prefetch intrinsic
          // (same restriction as the read intrinsic — see load_part). Emit 8x
          // (16-wide, 8-tall, 1-block) prefetches covering the same footprint
          // the load will pull.
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col,      row     });
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col,      row + 8 });
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col,      row + 16});
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col,      row + 24});
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col + 16, row     });
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col + 16, row + 8 });
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col + 16, row + 16});
          __spirv_Subgroup2DBlockPrefetchINTEL(
              sizeof(U), 16, 8, 1, (void *)src_ptr, row_stride_width,
              memory_height, row_stride_width, intel::coord_t{col + 16, row + 24});
        }
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockPrefetchINTEL(
            sizeof(U), 16, 16, 1, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{col, row});
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        __spirv_Subgroup2DBlockPrefetchINTEL(
            sizeof(U), 16, 16, 2, (void *)src_ptr, row_stride_width,
            memory_height, row_stride_width,
            intel::coord_t{col, row});
      }
    }
  }
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load(RT &dst, const GL &src, const COORD &idx) {
  prefetch_load_part<axis, RT, GL>(dst, src, idx, {0, 0});
}

// Collaborative (de-duplicated) prefetch: warm a SINGLE 16-wide x 8-tall x
// 2-block region at the RAW {row, col} given by `tile_posistion` (no scaling by
// base_tile_rows/cols, no per-tile loop -- unlike prefetch_load_part).  Callers
// gate this by sub-group id so only selected sub-groups warm each shared A/B
// tile, cutting redundant cache-warming across the work-group.  `dst` supplies
// only the shape/coord type; no registers are written.
template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>, int pref_shape = 1>
inline static void prefetch_load_part_coop(RT &dst, const GL &src, const COORD &idx,
  const std::tuple<int, int> tile_posistion) {
  using U = typename GL::dtype;

  const int offset_row = std::get<0>(tile_posistion) ;
  const int offset_col = std::get<1>(tile_posistion) ;

  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height =  src.template shape<axis>();

  {
    int row = offset_row;
    int col = offset_col ;

      __spirv_Subgroup2DBlockPrefetchINTEL(
          sizeof(U), 16, 8, 2, (void *)src_ptr, row_stride_width,
          memory_height, row_stride_width,
          intel::coord_t{col, row});

  }
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_coop(RT &dst, const GL &src, const COORD &idx,
     const std::tuple<int, int> tile_posistion) {
  prefetch_load_part_coop<axis, RT, GL>(dst, src, idx, tile_posistion);
}

// Coarse prefetch: warm one region `count` base-blocks wide (count*16 elements)
// x 32 rows in a SINGLE __spirv 2D-block prefetch, instead of `width` separate
// calls.  Used by the GEMM_COOP_COARSE experiment to cut prefetch instruction
// count (e.g. a 32x64 B col-tile in one count=4 call instead of two count=2).
// `dst` supplies only the shape/coord type; no registers are written.
template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_wide(RT &dst, const GL &src, const COORD &idx, int count) {
  (void)dst;
  using U = typename GL::dtype;
  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height = src.template shape<axis>();
  __spirv_Subgroup2DBlockPrefetchINTEL(
      sizeof(U), 16, 32, count, (void *)src_ptr, row_stride_width,
      memory_height, row_stride_width, intel::coord_t{0, 0});
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load(RT &dst, const GL &src, const COORD &idx) {
  load<2, RT, GL>(dst, src, idx);
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load(RT &dst, const GL &src, const COORD &idx) {
  prefetch_load<2, RT, GL>(dst, src, idx);
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_coop(RT &dst, const GL &src, const COORD &idx,
     const std::tuple<int, int> tile_posistion) {
  prefetch_load_coop<2, RT, GL>(dst, src, idx, tile_posistion);
}

// ---------------------------------------------------------------------------
// L2-TARGETED prefetch for long-latency (XeLink / remote-peer) tiles.  Emits a
// SINGLE 2D-block prefetch per base tile with LSC cache-control L1UC_L3C: the
// remote tile lands in the large shared L2 without thrashing the small L1.
// Implemented for the bf16 rt_32x32 fast path (the GEMM A-operand tile); any
// other shape/dtype falls back to the default L1 prefetch.
// ---------------------------------------------------------------------------
template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_l2(RT &dst, const GL &src, const COORD &idx) {
  using U = typename GL::dtype;
  if constexpr (std::is_same_v<typename RT::shape, ducks::rt_shape::rt_32x32> &&
                sizeof(U) == 2) {
    U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride_width = src.template stride<axis>() * sizeof(U);
    const int memory_height    = src.template shape<axis>();
#pragma unroll
    for (int i = 0; i < dst.height; i++) {
      const int row = i * dst.base_tile_rows;
#pragma unroll
      for (int j = 0; j < dst.width; j++) {
        const int col = j * dst.base_tile_cols;
        // 16-wide x 32-tall x 2-block bf16 == exactly one rt_32x32 base tile:
        // ONE XeLink transaction, cached in the shared L2 only (L1 bypassed).
        __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
            (intptr_t)src_ptr, row_stride_width - 1, memory_height - 1,
            row_stride_width - 1, intel::coord_t{col, row},
            LscCacheControl::kL1UC_L3C);
      }
    }
  } else {
    prefetch_load<axis, RT, GL>(dst, src, idx);
  }
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_l2(RT &dst, const GL &src, const COORD &idx) {
  prefetch_load_l2<2, RT, GL>(dst, src, idx);
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void prefetch_load_wide(RT &dst, const GL &src, const COORD &idx, int count) {
  prefetch_load_wide<2, RT, GL>(dst, src, idx, count);
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load_transpose(RT &dst, const GL &src, const COORD &idx) {
  load_transpose<2, RT, GL>(dst, src, idx);
}

// Note: col_layout store uses the same block store as row_layout (generic store_part below)
// because col_layout load now also uses regular (non-VNNI) LoadINTEL, giving both layouts
// the same register data format.

// Load for DPAS matrix B consumption.
// For bf16 (VNNI factor=2), a regular 2D block load already produces data in
// VNNI-compatible format because it naturally pairs consecutive rows:
//   data[k] = {B[2k][L], B[2k+1][L]} per lane L
// which is the same byte layout as int32 VNNI = pack(B[2k][L], B[2k+1][L]).
template <int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load_transform_part(RT &dst, const GL &src, const COORD &idx,
                             const std::tuple<int, int> tile_posistion) {
  using T2 = RT::dtype;
  using T = base_types::packing<typename RT::dtype>::unpacked_type;
  using U = typename GL::dtype;
  using U2 = base_types::packing<U>::packed_type;

  U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
  const int row_stride_width = src.template stride<axis>() * sizeof(U);
  const int memory_height = src.template shape<axis>();

  const int base_row = std::get<0>(tile_posistion) * dst.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * dst.base_tile_cols;

#pragma unroll
  for (int i = 0; i < dst.height; i++) {
    const int row = base_row + i * dst.base_tile_rows;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
      const int col = base_col + j * dst.base_tile_cols;
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_32x32>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 32, 2, (void *)src_ptr,
            row_stride_width, memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&dst.tiles[i][j].data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 16, 1, (void *)src_ptr,
            row_stride_width, memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&dst.tiles[i][j].data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 16, 2, (void *)src_ptr,
            row_stride_width, memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&dst.tiles[i][j].data[0]);
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_32x16>) {
        __spirv_Subgroup2DBlockLoadINTEL(
            sizeof(U), 16, 32, 1, (void *)src_ptr,
            row_stride_width, memory_height, row_stride_width,
            intel::coord_t{col, row}, (void *)&dst.tiles[i][j].data[0]);
      }
    }
  }
}

template <int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load_transform(RT &dst, const GL &src, const COORD &idx) {
  load_transform_part<axis, RT, GL>(dst, src, idx, {0,0});
}

template <ducks::rt::col_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load_transform(RT &dst, const GL &src, const COORD &idx) {
  load_transform<2, RT, GL>(dst, src, idx);
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store_part(const GL &dst, const RT &src, const COORD &idx, const std::tuple<int, int> tile_posistion) {
  using T2 = base_types::packing<typename RT::dtype>::packed_type;
  using U = typename GL::dtype;
  using U2 = base_types::packing<U>::packed_type;

  const int base_row = std::get<0>(tile_posistion) * src.base_tile_rows;
  const int base_col = std::get<1>(tile_posistion) * src.base_tile_cols;

  U *dst_ptr = (U *)&dst[(idx.template unit_coord<axis, 3>())];
  const int row_stride_width = dst.template stride<axis>() * sizeof(U);
  const int memory_height =  dst.template shape<axis>();
#pragma unroll
  for (int i = 0; i < src.height; i++) {
    int row = base_row + i * src.base_tile_rows;
#pragma unroll
    for (int j = 0; j < src.width; j++) {
      int col = base_col + j * src.base_tile_cols;
      U2 data[RT::packed_per_base_tile];
#pragma unroll
      for (int i_d = 0; i_d < RT::packed_per_base_tile; i_d++) {
        data[i_d] =
            base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[i_d]);
      }

      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_32x32>) {
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[0], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[4], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 8});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[8], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 16});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[12], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 24});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[16], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[20], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row + 8});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[24], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row + 16});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[28], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row + 24});
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x32>) {
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[0], (void *)dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[8], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[4], (void *)dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 8});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[12], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row + 8});
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_8x32>) {
        // O tile (8x32 float): 8 rows only. data[0-3]=rows0-7/cols0-15,
        // data[4-7]=rows0-7/cols16-31 (top-8-rows of the rt_16x32 layout).
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[0], (void *)dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[4], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col + 16, row});
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_32x16>) {
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[0], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[4], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 8});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[8], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 16});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[12], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 24});
      } else if constexpr (std::is_same_v<typename RT::shape,
                                          ducks::rt_shape::rt_16x16>) {
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[0], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row});
        __spirv_Subgroup2DBlockStoreINTEL(
            sizeof(U), 16, 8, 1, (void *)&data[4], dst_ptr,
            row_stride_width, memory_height,
            row_stride_width, intel::coord_t{col, row + 8});
      }
    }
  }
}

template <int axis, ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store(const GL &dst, const RT &src, const COORD &idx) {
  store_part<axis, RT, GL, COORD>(dst, src, idx, {0,0});
}

template <ducks::rt::all RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store(const GL &dst, const RT &src, const COORD &idx) {
  store<2, RT, GL, COORD>(dst, src, idx);
}

// ============================================================
// atomic_add_part / atomic_add — fp32 row_l register tile → HBM scatter.
// Uses per-lane sycl::atomic_ref<float> fetch_add.
// Use for FA2-bwd dQ accumulation where multiple WGs must sum into the
// same dst rows.
//
// Constraints (narrower than store_part — only what FA2-bwd needs):
//   - RT::dtype must be float (fp32 atomics only; bf16 CAS not implemented)
//   - RT::layout must be row_layout
//   - GL::dtype must be float (dst buffer is fp32)
//
// Layout (row_l, rt_16x16 base): lane `l` owns 2 floats per data[k] (k=0..7):
//   data[k].x() = element at (row = 2k,     col = l)
//   data[k].y() = element at (row = 2k + 1, col = l)
// Wider tiles (W > 16) pack multiple base tiles along width.
// ============================================================
template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void atomic_add_part(const GL &dst, const RT &src,
                                   const COORD &idx,
                                   const std::tuple<int, int> tile_position) {
  using T = typename RT::T;
  using U = typename GL::dtype;
  static_assert(std::is_same_v<T, float>,
                "atomic_add_part: only fp32 register tiles supported");
  static_assert(std::is_same_v<U, float>,
                "atomic_add_part: dst global memory must be fp32");
  static_assert(std::is_same_v<typename RT::shape, ducks::rt_shape::rt_16x16> ||
                    std::is_same_v<typename RT::shape, ducks::rt_shape::rt_32x32>,
                "atomic_add_part: only rt_16x16 / rt_32x32 base shapes supported");

  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  const int lane = sg.get_local_linear_id();

  const int base_row = std::get<0>(tile_position) * src.base_tile_rows;
  const int base_col = std::get<1>(tile_position) * src.base_tile_cols;

  U *dst_ptr = (U *)&dst[(idx.template unit_coord<axis, 3>())];
  const int row_stride = dst.template stride<axis>();

  auto aadd = [&](int row, int col, float v) {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ar(dst_ptr[(size_t)row * row_stride + col]);
    ar.fetch_add(v);
  };

#pragma unroll
  for (int i = 0; i < src.height; i++) {
    const int row_block = base_row + i * src.base_tile_rows;
#pragma unroll
    for (int j = 0; j < src.width; j++) {
      const int col_block = base_col + j * src.base_tile_cols;
      if constexpr (std::is_same_v<typename RT::shape,
                                   ducks::rt_shape::rt_16x16>) {
        // rt_16x16 (base 16x16): lane owns col `col_block+lane`; data[k].x/.y ->
        // rows 2k / 2k+1.  Mirrors store_part<rt_16x16>.
#pragma unroll
        for (int k = 0; k < RT::packed_per_base_tile; k++) {
          const int col = col_block + lane;
          aadd(row_block + 2 * k,     col, src.tiles[i][j].data[k].x());
          aadd(row_block + 2 * k + 1, col, src.tiles[i][j].data[k].y());
        }
      } else {
        // rt_32x32 (base 32x32): store_part writes it as 8 sub-blocks of
        // 16 cols x 8 rows.  Mirror that EXACTLY (col_off, row_off, data_base):
        //   {0,0,0} {0,8,4} {0,16,8} {0,24,12}   (cols 0-15)
        //   {16,0,16} {16,8,20} {16,16,24} {16,24,28} (cols 16-31)
        // Within a sub-block, data[base+r2].x/.y -> rows row_off+2*r2 / +1 at
        // col col_off+lane.  Explicit (no loop-bound dependence) so BOTH column
        // halves are always emitted.
        constexpr int SB[8][3] = {{0, 0, 0},  {0, 8, 4},  {0, 16, 8},  {0, 24, 12},
                                  {16, 0, 16}, {16, 8, 20}, {16, 16, 24}, {16, 24, 28}};
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const int col_off = SB[s][0];
          const int row_off = SB[s][1];
          const int base_d  = SB[s][2];
#pragma unroll
          for (int r2 = 0; r2 < 4; r2++) {
            const int dd  = base_d + r2;
            const int col = col_block + col_off + lane;
            const int rx  = row_block + row_off + 2 * r2;
            aadd(rx,     col, src.tiles[i][j].data[dd].x());
            aadd(rx + 1, col, src.tiles[i][j].data[dd].y());
          }
        }
      }
    }
  }
}

template <ducks::rt::row_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void atomic_add_part(const GL &dst, const RT &src,
                                   const COORD &idx,
                                   const std::tuple<int, int> tile_position) {
  atomic_add_part<2, RT, GL, COORD>(dst, src, idx, tile_position);
}

template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void atomic_add(const GL &dst, const RT &src, const COORD &idx) {
  atomic_add_part<axis, RT, GL, COORD>(dst, src, idx, {0, 0});
}

template <ducks::rt::row_layout RT, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<RT>>
inline static void atomic_add(const GL &dst, const RT &src, const COORD &idx) {
  atomic_add<2, RT, GL, COORD>(dst, src, idx);
}

} // namespace kittens