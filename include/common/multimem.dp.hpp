/**
 * @file
 * @brief Stubs for multi-memory operations (SYCL port).
 *
 * These operations require multi-GPU distributed shared memory which is
 * architecture-specific. On Intel Xe, equivalent functionality would use
 * Level Zero peer-to-peer or SYCL USM with explicit multi-device allocation.
 *
 * TODO: Implement using Level Zero IPC / SYCL 2020 USM multi-device features.
 */

#pragma once

#include "base_types.dp.hpp"

namespace kittens {

enum class reduce_op {
    ADD = 0,
    MIN = 1,
    MAX = 2
};

enum class memory_model {
    WEAK = 0,
    STRONG = 1
};

// Stub: multimem operations are NVIDIA-specific (multimem.ld_reduce, multimem.st, multimem.red).
// For Intel GPUs, multi-GPU reductions should use Level Zero IPC or SYCL USM.
template <typename T>
struct multimem {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    static inline void ld_reduce(T &dst, const T *src) {
        // Fallback: simple load (no reduction across GPUs)
        dst = *src;
    }
    template <memory_model M = memory_model::WEAK>
    static inline void st(T *dst, const T &src) {
        *dst = src;
    }
    template <reduce_op Op>
    static inline void red(T *dst, const T &src) {
        // Fallback: simple store (no multi-GPU reduction)
        *dst = src;
    }
};

} // namespace kittens
