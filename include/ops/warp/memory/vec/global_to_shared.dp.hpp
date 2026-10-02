/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void load(SV &dst, const GL &src, const COORD &idx) {
    // Use scalar copies to avoid type-punning issues across bf16/half/float.
    constexpr int elem_per_transfer = 1;
    constexpr int total_calls = (SV::length + WARP_THREADS - 1) / WARP_THREADS;
    typename GL::dtype *src_ptr =
        (typename GL::dtype *)&src[(idx.template unit_coord<-1, 3>())];
#pragma unroll
    for (int iter = 0, i = ::kittens::laneid(); iter < total_calls;
         iter++, i += WARP_THREADS) {
            if (i < dst.length) {
                dst.data[i] = src_ptr[i];
            }
    }
}

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void store(const GL &dst, const SV &src, const COORD &idx) {
  // Use scalar copies to avoid type-punning issues across bf16/half/float.
  constexpr int elem_per_transfer = 1;
  constexpr int total_calls = (SV::length + WARP_THREADS - 1) / WARP_THREADS;
  typename GL::dtype *dst_ptr =
      (typename GL::dtype *)&dst[(idx.template unit_coord<-1, 3>())];
#pragma unroll
  for (int iter = 0, i = ::kittens::laneid(); iter < total_calls;
       iter++, i += WARP_THREADS) {
        if (i < SV::length) {
            dst_ptr[i] = src.data[i];
        }
  }
}

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam SV The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void load_async(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer =
        sizeof(sycl::float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = (/*dst.length +*/ WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up//NYI
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    auto dst_ptr = &dst.data[0];
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            /*
            DPCT1053:103: Migration of device assembly code is not supported.
            */
            // asm volatile(
            //     "cp.async.cg.shared::cta.global [%0], [%1], 16;\n" ::"r"(
            //         dst_ptr + uint32_t(sizeof(typename SV::dtype)) * i *
            //                       elem_per_transfer),
            //     "l"((uint64_t)&src_ptr[i * elem_per_transfer])
            //     : "memory");
        }
    }
    /*
    DPCT1026:102: The call to "cp.async.commit_group;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
    */
}

}