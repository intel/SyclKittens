/**
 * @file
 * @brief A collection of all of the operations that ThunderKittens defines.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "warp/warp.dp.hpp"
#include "collective/collective.dp.hpp"
#include "collective/topology_sycl.dp.hpp"
// Note: group/group.dp.hpp needs complex type support to compile fully.
// Include group_base.dp.hpp for the kittens::group<N> struct only (sync, barrier).
// Kernels using prototype framework will get this through prototype.dp.hpp.
#include "group/group_base.dp.hpp"
