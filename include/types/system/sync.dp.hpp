/**
 * @file
 * @brief Multi-GPU synchronization primitives for Intel PVC.
 *
 * Ported from ThunderKittens CUDA sync operations.
 * Uses sycl::atomic_ref with system scope instead of CUDA inline asm.
 *
 * On CUDA (H100), TK uses:
 *   - multimem.red.release.sys.global.add.s32 (NVLS multicast atomic)
 *   - red.release.sys.global.add.s32 (single-device atomic)
 *   - ld.relaxed.sys.global.s32 (polling load)
 *
 * On Intel PVC, we use:
 *   - sycl::atomic_ref<int, release, system> for cross-device signal
 *   - sycl::atomic_ref<int, relaxed, system> for cross-device wait
 *   - No multicast — signal_all loops over all device barrier slots
 */

#pragma once

#include <sycl/sycl.hpp>
#include "../device/pgl.dp.hpp"

namespace kittens {

/**
 * @brief Signal a single device's barrier slot.
 * Equivalent to CUDA: red.release.sys.global.add.s32
 */
template <int NUM_DEVICES>
inline void signal(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    int dst_dev_idx,
    int val
) {
    sycl::atomic_ref<int,
        sycl::memory_order::acq_rel,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
    ref(*const_cast<int*>(&barrier[dst_dev_idx][idx]));
    ref.fetch_add(val, sycl::memory_order::release);
}

/**
 * @brief Signal ALL devices' barrier slots.
 * On CUDA, this uses a single NVLS multicast atomic.
 * On Intel PVC, we loop over all devices (no multicast).
 */
template <int NUM_DEVICES>
inline void signal_all(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    int val
) {
    #pragma unroll
    for (int d = 0; d < NUM_DEVICES; d++) {
        sycl::atomic_ref<int,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>
        ref(*const_cast<int*>(&barrier[d][idx]));
        ref.fetch_add(val, sycl::memory_order::release);
    }
}

/**
 * @brief Wait until a device's barrier slot reaches the expected value.
 * Equivalent to CUDA: ld.relaxed.sys.global.s32 polling loop.
 */
template <int NUM_DEVICES>
inline void wait(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    int dev_idx,
    int expected
) {
    sycl::atomic_ref<int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
    ref(*const_cast<int*>(&barrier[dev_idx][idx]));
    while (ref.load() != expected) { /* spin */ }
}

/**
 * @brief Full barrier across all devices.
 * 1. signal_all(+1) — each device signals all others
 * 2. wait(NUM_DEVICES) — wait for all signals to arrive
 * 3. signal(-NUM_DEVICES) — reset own slot for next barrier
 */
template <int NUM_DEVICES>
inline void barrier_all(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    int dev_idx
) {
    signal_all(barrier, idx, 1);
    wait(barrier, idx, dev_idx, NUM_DEVICES);
    // Reset own barrier slot
    sycl::atomic_ref<int,
        sycl::memory_order::acq_rel,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
    ref(*const_cast<int*>(&barrier[dev_idx][idx]));
    ref.fetch_add(-NUM_DEVICES, sycl::memory_order::release);
}

} // namespace kittens
