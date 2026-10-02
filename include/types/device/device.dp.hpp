/**
 * @file
 * @brief An aggregate header file for all the device types defined by ThunderKittens.
 */

 #pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifdef KITTENS_MULTI_GPU
#include "pgl.dp.hpp"
#endif
