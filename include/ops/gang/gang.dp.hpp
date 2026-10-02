/**
 * @file
 * @brief Multi-GPU synchronization operations for Intel PVC.
 *
 * Ported from ThunderKittens CUDA gang operations.
 *
 * Key differences from CUDA version:
 *   - Uses SYCL system-scope atomic_ref instead of PTX multimem instructions
 *   - signal_all() loops over per-device P2P pointers instead of multicast
 *   - Uses sycl::ext::oneapi::this_work_item::get_nd_item<3>() for work-item access
 *   - barrier_all uses dev_ptr_at() instead of mc_ptr_at()
 *
 * All synchronization relies on P2P-accessible USM device memory
 * within a shared SYCL context.
 */

#pragma once

#include <sycl/sycl.hpp>
#include "../../types/device/pgl.dp.hpp"
#include "../../types/device/sync_manager.dp.hpp"

namespace kittens {

/* ----------   Multi-GPU barrier primitives (replace CUDA PTX)  ---------- */

/**
 * @brief Signal a specific device's barrier counter via P2P atomic add.
 */
template <int NUM_DEVICES>
inline void signal(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    const int dst_dev_idx,
    const int val)
{
    int *ptr = const_cast<int*>(&barrier[dst_dev_idx][idx]);
    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    ref.fetch_add(val);
}

/**
 * @brief Signal ALL devices' barrier counters via P2P atomic adds.
 *
 * This is the Intel PVC equivalent of CUDA's multimem.red.release.sys.global.add.
 * Instead of one multicast write, we loop over per-device pointers.
 */
template <int NUM_DEVICES>
inline void signal_all(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    const int val)
{
    // Access each device's counter via P2P and atomically increment it
    #pragma unroll
    for (int d = 0; d < NUM_DEVICES; d++) {
        int *ptr = barrier.dev_ptr_at(idx, d);
        sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                         sycl::memory_scope::system,
                         sycl::access::address_space::global_space> ref(*ptr);
        ref.fetch_add(val);
    }
}

/**
 * @brief Wait until this device's barrier counter reaches expected value.
 * Polls via P2P-readable device memory.
 */
template <int NUM_DEVICES>
inline void wait(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    const int dev_idx,
    const int expected)
{
    int *ptr = const_cast<int*>(&barrier[dev_idx][idx]);
    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    while (ref.load() != expected) { /* spin */ }
}

/**
 * @brief Full barrier across all devices.
 *
 * 1. signal_all: atomically add 1 to all devices' counters
 * 2. wait: poll local counter until == NUM_DEVICES
 * 3. reset: atomically subtract NUM_DEVICES from own counter
 */
template <int NUM_DEVICES>
inline void barrier_all(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    const int dev_idx)
{
    signal_all(barrier, idx, 1);
    wait(barrier, idx, dev_idx, NUM_DEVICES);
    // Reset local counter
    int *ptr = const_cast<int*>(&barrier[dev_idx][idx]);
    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    ref.fetch_add(-NUM_DEVICES);
}

/* ----------   Gang template: multi-GPU collective sync  ---------- */

/**
 * @brief Gang template represents a collection of GPUs working together.
 */
template <int NUM_DEVICES>
struct gang {

/**
 * @brief Synchronizes all threads in all blocks in all GPUs.
 *
 * Protocol (adapted for SYCL from CUDA):
 * 1. Memory fence + work-group barrier
 * 2. Thread 0 of non-leader blocks: increment device-local counter
 * 3. Thread 0 of block 0 (leader):
 *    a. Wait for all local blocks to check in
 *    b. Signal all devices via P2P atomic adds
 *    c. Poll local cross-device counter until == NUM_DEVICES
 *    d. Reset counters
 * 4. Non-leader blocks: wait for device counter reset, then proceed
 * 5. Final work-group barrier
 */
struct everyone {
    template <ducks::sync_manager::all SyncManager>
    static inline void sync(const SyncManager &sm, const int dev_idx) {
        if (dev_idx >= NUM_DEVICES) return;

        auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

        // Fence + barrier
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        item.barrier();

        if (item.get_local_id(2) == 0 &&
            item.get_local_id(1) == 0 &&
            item.get_local_id(0) == 0) {

            auto sp = sm.get_all_sync_point(dev_idx);

            // Cross-device counter: p2p_ptrs[dev_idx][0]  (all devices can P2P read/write)
            // Device-local counter: p2p_ptrs[dev_idx][1]  (for local block counting)
            int *cross_ptr = sp.p2p_ptrs[dev_idx];     // offset 0
            int *local_ctr = sp.p2p_ptrs[dev_idx] + 1; // offset 1

            sycl::atomic_ref<int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space> cross_ref(*cross_ptr);
            sycl::atomic_ref<int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> local_ref(*local_ctr);

            size_t nblocks = item.get_group_range(0) *
                             item.get_group_range(1) *
                             item.get_group_range(2);

            int linear_block = item.get_group(2) +
                               item.get_group(1) * item.get_group_range(2) +
                               item.get_group(0) * item.get_group_range(2) *
                                   item.get_group_range(1);

            if (linear_block == 0) {
                // Leader block: wait for all other blocks on this device
                while (local_ref.load() < static_cast<int>(nblocks) - 1) { /* spin */ }

                // Signal all devices: P2P atomic add to each device's cross counter
                #pragma unroll
                for (int d = 0; d < NUM_DEVICES; d++) {
                    int *target = sp.p2p_ptrs[d]; // offset 0
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::system,
                                     sycl::access::address_space::global_space> t_ref(*target);
                    t_ref.fetch_add(1);
                }

                // Wait for all devices to arrive
                while (cross_ref.load() < NUM_DEVICES) { /* spin */ }

                // Reset both counters
                cross_ref.store(0);
                local_ref.store(0);
            } else {
                // Non-leader block: check in and wait for reset
                local_ref.fetch_add(1);
                while (local_ref.load() > 0) { /* spin */ }
            }
        }

        item.barrier();
    }
};

/**
 * @brief Synchronizes blocks with the same blockIdx across all GPUs.
 */
struct blockwise {
    template <ducks::sync_manager::all SyncManager>
    static inline void sync(const SyncManager &sm, const int dev_idx) {
        static_assert(SyncManager::max_blocks > 0,
            "gang::blockwise::sync() requires MAX_BLOCKS > 0");

        if (dev_idx >= NUM_DEVICES) return;

        auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

        int block_idx = item.get_group(2) +
                        item.get_group(1) * item.get_group_range(2) +
                        item.get_group(0) * item.get_group_range(2) *
                            item.get_group_range(1);

        if (block_idx >= SyncManager::max_blocks) return;

        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        item.barrier();

        if (item.get_local_id(2) == 0 &&
            item.get_local_id(1) == 0 &&
            item.get_local_id(0) == 0) {

            auto sp = sm.get_blockwise_sync_point(dev_idx, block_idx);

            // Signal all devices via P2P atomic add
            #pragma unroll
            for (int d = 0; d < NUM_DEVICES; d++) {
                sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space> t_ref(*sp.p2p_ptrs[d]);
                t_ref.fetch_add(1);
            }

            // Wait for all devices
            sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space> local_ref(*sp.local_ptr);
            while (local_ref.load() < NUM_DEVICES) { /* spin */ }

            // Reset
            local_ref.store(0);
        }

        item.barrier();
    }
};

/**
 * @brief Synchronizes an arbitrary set of blocks across GPUs using a shared sync ID.
 */
struct blockgroup {
    template <ducks::sync_manager::all SyncManager>
    static inline void sync(const SyncManager &sm, const int dev_idx,
                            const int sync_id, const int expected_arrivals) {
        static_assert(SyncManager::max_sync_points > 0,
            "gang::blockgroup::sync() requires MAX_SYNC_POINTS > 0");

        if (dev_idx >= NUM_DEVICES) return;

        auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        item.barrier();

        if (item.get_local_id(2) == 0 &&
            item.get_local_id(1) == 0 &&
            item.get_local_id(0) == 0) {

            auto sp = sm.get_blockgroup_sync_point(dev_idx, sync_id);

            sycl::atomic_ref<int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space> local_ref(*sp.local_ptr);

            int expected = 0;
            if (local_ref.compare_exchange_strong(expected, 1)) {
                // Leader: collect arrivals from all devices via P2P reads
                int total = 0;
                do {
                    total = 0;
                    #pragma unroll
                    for (int d = 0; d < NUM_DEVICES; d++) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                         sycl::memory_scope::system,
                                         sycl::access::address_space::global_space> ref(*sp.p2p_ptrs[d]);
                        total += ref.load();
                    }
                } while (total < expected_arrivals);

                // Reset and release
                local_ref.store(0);
            } else {
                // Non-leader: increment and wait for leader to finish
                local_ref.fetch_add(1);
                while (local_ref.load() != 0) { /* spin */ }
            }
        }

        item.barrier();
    }
};

}; // struct gang

} // namespace kittens
