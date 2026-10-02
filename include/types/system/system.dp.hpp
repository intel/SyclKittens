/**
 * @file
 * @brief System-wide types for multi-GPU support on Intel PVC.
 *
 * Provides:
 *   - PGL (Parallel Global Layout) — multi-device gl arrays
 *   - Barrier/sync primitives — cross-device atomic synchronization
 *
 * Ported from ThunderKittens CUDA types/system/ headers.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// Parallel Global Layout — use device/pgl.dp.hpp (clean SYCL port) instead
// #include "pgl.dp.hpp" // Disabled: superseded by types/device/pgl.dp.hpp

// Cross-device synchronization (barrier_all, signal, wait)
// Only include when multi-GPU code is actually needed
#ifdef KITTENS_MULTI_GPU
#include "sync.dp.hpp"
#endif
