/**
 * @file
 * @brief The master header file of SyclKittens.
 *        This file includes everything you need!
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// Global-scope SPIRV intrinsic declarations must come BEFORE namespace kittens opens.
// split_barrier_decl.dp.hpp declares __spirv_ControlBarrierArriveINTEL / WaitINTEL
// so that kittens::split_barrier_arrive() can call them with :: prefix.
#include "ops/warp/sync/split_barrier_decl.dp.hpp"
#include "common/common.dp.hpp"
#include "types/types.dp.hpp"
#include "ops/ops.dp.hpp"
#include "pyutils/util.dp.hpp"
