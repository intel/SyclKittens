/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

#pragma once

#include <type_traits>
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

namespace kittens {
template <typename op>
inline float reduce_16x16(const sycl::float2 *first_half,
                          const sycl::float2 *second_half) {
  auto first_vector = std::bit_cast<vector_t<float, 8>>(
      *reinterpret_cast<const std::array<float, 8> *>(first_half));
  auto second_vector = std::bit_cast<vector_t<float, 8>>(
      *reinterpret_cast<const std::array<float, 8> *>(second_half));

  return op::template reduce<float>(first_vector, second_vector);
}
/**
 * @brief Perform a row-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template <typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
  // I actually like these static asserts because they give more verbose errors
  // when things go wrong.
  static_assert(
      std::is_same_v<typename V::layout,
                     typename rt_base<typename T::T, typename T::layout>::
                         col_vec_layout>); // compatible layout
  static_assert(
      std::is_same_v<typename V::T2, typename T::dtype>); // compatible type
  static_assert(V::outer_dim == T::height);               // compatible size
  static_assert(std::is_same_v<typename T::shape, ducks::rt_shape::rt_16x32> ||
                std::is_same_v<typename T::shape,
                               ducks::rt_shape::rt_16x16> ||
                std::is_same_v<typename T::shape,
                               ducks::rt_shape::rt_8x32>); // supported shape
  using dtype = T::dtype;
  using type = V::T;
  // On Intel SIMD16 row_layout: lane L holds column L of the tile.
  // data[k] = {M[2k][L], M[2k+1][L]}, so row R's data is spread across
  // all 16 lanes at data[R/2].{x() if R even, y() if R odd}.
  // Row reduction requires cross-lane XOR butterfly shuffle to sum across columns.
  const int lane = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                       .get_sub_group()
                       .get_local_linear_id();
#pragma unroll
  for (int i = 0; i < src.height; i++) {
#pragma unroll
    for (int j = 0; j < src.width; j++) {
        type tile_result{};
      if constexpr (std::is_same_v<typename T::shape,
                                   ducks::rt_shape::rt_8x32>) {
        // 8x32: 8 rows, data[0-3]=cols0-15, data[4-7]=cols16-31 (offset +4,
        // not +8). Only 4 packed pairs -> fills lanes 0-7 (rows 0-7).
#pragma unroll
        for (int k = 0; k < 4; k++) {
          type val_x = op::template op<type>(src.tiles[i][j].data[k].x(),
                                             src.tiles[i][j].data[k + 4].x());
          type val_y = op::template op<type>(src.tiles[i][j].data[k].y(),
                                             src.tiles[i][j].data[k + 4].y());
          // XOR butterfly reduction across 16 SIMD lanes (32 cols -> scalar)
#pragma unroll
          for (int stride = 8; stride >= 1; stride >>= 1) {
            val_x = op::template op<type>(
                val_x, packed_shfl_sync(MASK_ALL, val_x, lane ^ stride));
            val_y = op::template op<type>(
                val_y, packed_shfl_sync(MASK_ALL, val_y, lane ^ stride));
          }
          if (lane == 2 * k)     tile_result = val_x;
          if (lane == 2 * k + 1) tile_result = val_y;
        }
      } else {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        type val_x, val_y;
        if constexpr (std::is_same_v<typename T::shape,
                                     ducks::rt_shape::rt_16x32>) {
          // 16x32: combine two 16-column halves within-lane first
          val_x = op::template op<type>(src.tiles[i][j].data[k].x(),
                                        src.tiles[i][j].data[k + 8].x());
          val_y = op::template op<type>(src.tiles[i][j].data[k].y(),
                                        src.tiles[i][j].data[k + 8].y());
        } else {
          val_x = src.tiles[i][j].data[k].x();
          val_y = src.tiles[i][j].data[k].y();
        }
        // XOR butterfly reduction across 16 SIMD lanes
#pragma unroll
        for (int stride = 8; stride >= 1; stride >>= 1) {
          val_x = op::template op<type>(
              val_x, packed_shfl_sync(MASK_ALL, val_x, lane ^ stride));
          val_y = op::template op<type>(
              val_y, packed_shfl_sync(MASK_ALL, val_y, lane ^ stride));
        }
        // All lanes now have reduce(row 2k) in val_x, reduce(row 2k+1) in val_y.
        // Scatter: lane 2k stores reduce(row 2k), lane 2k+1 stores reduce(row 2k+1).
        if (lane == 2 * k)     tile_result = val_x;
        if (lane == 2 * k + 1) tile_result = val_y;
      }
      }
      if (j == 0) {
        if constexpr (reset) {
          row_accum[i][0] = tile_result;
        } else {
          row_accum[i][0] = op::template op<type>(src_accum[i][0], tile_result);
        }
      } else {
        row_accum[i][0] = op::template op<type>(row_accum[i][0], tile_result);
      }
    }
  }
}

/**
 * @brief Perform a row-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = V::dtype;

    const int leader =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        0x3; // 00011 in binary
#pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile/2; k++) {
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k+0]);
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k+2]);
            }
        }

        // Now we need to do a lil shuffle to make everyone happy.

        if constexpr (WARP_THREADS > 16) {
            accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 16));
        }
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 4));

        if constexpr (WARP_THREADS > 16) {
            accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 16));
        }
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 4));

        accum_top_rows    = packed_shfl_sync(MASK_ALL, accum_top_rows,    leader);
        accum_bottom_rows = packed_shfl_sync(MASK_ALL, accum_bottom_rows, leader);

        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);

            //   sycl::ext::oneapi::experimental::printf("vvv idx %d   value %f %f \n", i, vec.outer_dim, vec[i]->x(), vec[i]->y());
        }
    }
}

// Col reduction.
/**
 * @brief Perform a column-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the columns of a
 * matrix using a specified operation. It leverages warp shuffle functions for
 * efficient intra-warp communication and is optimized for row-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator
 * (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is
 * stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is
 * false.
 */
template <typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
  static_assert(
      std::is_same_v<typename V::layout,
                     typename rt_base<typename T::T, typename T::layout>::
                         row_vec_layout>); // compatible layout
  static_assert(
      std::is_same_v<typename V::T2, typename T::dtype>); // compatible type
  static_assert(V::outer_dim == T::width);                   // compatible size

  using dtype = V::dtype;
  using type = V::T;
  constexpr int packed_per_tile = T::packed_per_base_tile;

  // On Intel SIMD16: each lane L holds a complete column L of the matrix.
  // data[k] = {M[2k][L], M[2k+1][L]}. Column reduction reduces all rows
  // within each lane — no cross-lane shuffle needed.
#pragma unroll
  for (int j = 0; j < src.width; j++) {
    // Reduce all row elements within each lane to a single scalar
    type scalar_accum = op::template op<type>(
        src.tiles[0][j].data[0].x(), src.tiles[0][j].data[0].y());
#pragma unroll
    for (int k = 1; k < packed_per_tile; k++) {
      scalar_accum = op::template op<type>(scalar_accum, src.tiles[0][j].data[k].x());
      scalar_accum = op::template op<type>(scalar_accum, src.tiles[0][j].data[k].y());
    }
    // Reduce across height subtiles
#pragma unroll
    for (int i = 1; i < src.height; i++) {
#pragma unroll
      for (int k = 0; k < packed_per_tile; k++) {
        scalar_accum = op::template op<type>(scalar_accum, src.tiles[i][j].data[k].x());
        scalar_accum = op::template op<type>(scalar_accum, src.tiles[i][j].data[k].y());
      }
    }
    // col_map reads col_accum[j][0] with .x() and .y() applied to all data elements.
    // Both components should be the same column reduction scalar.
    const dtype result_packed{scalar_accum, scalar_accum};

    if constexpr (reset) {
      col_accum[j][0] = result_packed;
    } else {
      col_accum[j][0] = op::template op<dtype>(src_accum[j][0], result_packed);
    }
  }
}
/**
 * @brief Perform a column-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the columns of a
 * matrix using a specified operation. It leverages warp shuffle functions for
 * efficient intra-warp communication and is optimized for column-major
 * matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator
 * (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is
 * stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is
 * false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = V::dtype;
    const int leader =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        0x1C; // 11100 in binary
#pragma unroll
    for(int j = 0; j < src.width; j++) { // note now width is the outer loop
        dtype accum_left_col  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[2]);
        dtype accum_right_col = op::template op<dtype>(src.tiles[0][j].data[1], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < src.height; i++) { // and height is the inner loop
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k+=2) {
                accum_left_col  = op::template op<dtype>(accum_left_col,  src.tiles[i][j].data[k+0]);
                accum_right_col = op::template op<dtype>(accum_right_col, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed{};
        accum_packed.x() = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_left_col.x(),  accum_left_col.y());
        accum_packed.y() = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_right_col.x(), accum_right_col.y());

        // Now we need to do a lil shuffle to make everyone happy.

        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            col_accum[j][0] = accum_packed;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
/**
 * @brief Store the maximum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)

/**
 * @brief Store the maximum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

// templated versions of each

template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void max(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline auto max(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void max(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}
template<int ax, ducks::rt::all T>
static inline auto max(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void min(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline auto min(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void min(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::rt::all T>
static inline auto min(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void sum(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline auto sum(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void sum(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::rt::all T>
static inline auto sum(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void prod(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline auto prod(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
static inline void prod(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::rt::all T>
static inline auto prod(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}

} // namespace kittens