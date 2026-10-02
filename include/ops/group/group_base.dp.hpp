/**
 * @file
 * @brief Minimal group struct definition for kittens::group<N> sync primitives.
 *        Does NOT include group memory/shared/register ops (which need complex types).
 *        Use group.dp.hpp for the full set of group operations.
 */

#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../../common/common.dp.hpp"

namespace kittens {

template<int N_WARPS>
struct group {
static constexpr int GROUP_WARPS = N_WARPS;
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS;
static inline int laneid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
               GROUP_THREADS;
}
static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
static inline int groupid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) /
               GROUP_THREADS;
}

static inline void sync(int id) {
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
}
static inline void arrive(int id) {
    // No-op on SYCL; bar.arrive not supported.
}

// Register rebalancing stubs (no Intel equivalent for setmaxnreg)
template<int n_reg> static inline void increase_registers() {}
template<int n_reg> static inline void decrease_registers() {}
static inline void producer_registers() {}
template<int NCWG> static inline void consumer_registers() {}

};

using warpgroup = group<4>;

}
