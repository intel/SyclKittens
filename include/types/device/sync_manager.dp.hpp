/**
 * @file
 * @brief Sync manager for Intel PVC multi-GPU synchronization.
 *
 * Ported from ThunderKittens CUDA sync_manager.cuh.
 *
 * Key differences from CUDA version:
 *   - Uses SYCL system-scope atomics instead of CUDA multimem instructions
 *   - Uses P2P-accessible int buffers (via shared SYCL context) instead of multicast
 *   - sync_point stores P2P pointer + device-local pointer (no multicast VA)
 *
 * Sync space layout (same as CUDA):
 *   [0..1]:                              gang::everyone counters
 *   [2..2+MAX_BLOCKS-1]:                 gang::blockwise per-block counters
 *   [2+MAX_BLOCKS..2+MAX_BLOCKS+MAX_SYNC_POINTS-1]: gang::blockgroup counters
 */

#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include "../../common/common.dp.hpp"

namespace kittens {

namespace ducks {
namespace sync_manager {

struct identifier {};

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace sync_manager
} // namespace ducks

/**
 * @brief Sync point for cross-device synchronization.
 *
 * On CUDA, sys_mc points to multicast memory and sys_uc to device memory.
 * On Intel PVC, both point to device memory accessible via P2P.
 *   - p2p_ptrs[i] = sync counter on device i (P2P readable/writable from any device)
 *   - local_ptr    = pointer to this device's own counter (for fast local access)
 */
template <typename SYNC_SPACE_DTYPE>
struct sync_point {
    SYNC_SPACE_DTYPE *p2p_ptrs[6];  // P2P counters on each device (max 6 PVC GPUs)
    SYNC_SPACE_DTYPE *local_ptr;    // Device-local counter (== p2p_ptrs[dev_idx])
    int num_devices;
};

/**
 * @brief Manages synchronized memory spaces across multiple Intel PVC GPUs.
 *
 * Allocates a small int buffer on each device for synchronization counters.
 * All buffers are P2P-accessible via shared SYCL context.
 *
 * @tparam NUM_DEVICES Number of GPU devices.
 * @tparam MAX_BLOCKS Maximum number of distinct blockIdx values for blockwise sync.
 * @tparam MAX_SYNC_POINTS Maximum number of sync point IDs for blockgroup sync.
 */
template <int NUM_DEVICES, int MAX_BLOCKS = 256, int MAX_SYNC_POINTS = 16>
struct sync_manager {
    static_assert(NUM_DEVICES > 0, "NUM_DEVICES must be greater than 0");
    static_assert(MAX_BLOCKS >= 0, "MAX_BLOCKS must be >= 0");
    static_assert(MAX_SYNC_POINTS >= 0, "MAX_SYNC_POINTS must be >= 0");

    using identifier = ducks::sync_manager::identifier;
    static constexpr int num_devices = NUM_DEVICES;
    static constexpr int all_sync_point_size = 2;
    static constexpr int max_blocks = MAX_BLOCKS;
    static constexpr int max_sync_points = MAX_SYNC_POINTS;
    static constexpr int sync_space_size = all_sync_point_size + MAX_BLOCKS + MAX_SYNC_POINTS;

    using SYNC_SPACE_DTYPE = int;

    // Per-device sync space pointers (P2P accessible)
    SYNC_SPACE_DTYPE *sync_spaces[NUM_DEVICES];

    /**
     * @brief Create a sync_manager by allocating sync space on each device.
     *
     * @param queues SYCL queues for each device (must share a context for P2P)
     */
    static inline sync_manager create(std::vector<sycl::queue> &queues) {
        sync_manager sm;
        for (int i = 0; i < NUM_DEVICES; i++) {
            sm.sync_spaces[i] = sycl::malloc_device<SYNC_SPACE_DTYPE>(
                sync_space_size, queues[i]);
            // Zero-initialize
            queues[i].memset(sm.sync_spaces[i], 0,
                sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }
        for (auto &q : queues) q.wait();
        return sm;
    }

    /**
     * @brief Free all sync space allocations.
     */
    inline void free(std::vector<sycl::queue> &queues) {
        for (int i = 0; i < NUM_DEVICES; i++) {
            if (sync_spaces[i]) {
                sycl::free(sync_spaces[i], queues[i]);
                sync_spaces[i] = nullptr;
            }
        }
    }

    /**
     * @brief Get sync point for gang::everyone::sync().
     * Uses offsets 0 and 1 in the sync space.
     */
    inline sync_point<SYNC_SPACE_DTYPE> get_all_sync_point(int dev_idx) const {
        sync_point<SYNC_SPACE_DTYPE> sp;
        sp.num_devices = NUM_DEVICES;
        sp.local_ptr = sync_spaces[dev_idx];  // offset 0
        for (int i = 0; i < NUM_DEVICES; i++)
            sp.p2p_ptrs[i] = sync_spaces[i];  // offset 0
        return sp;
    }

    /**
     * @brief Get sync point for gang::blockwise::sync().
     * Uses offsets [2 .. 2+MAX_BLOCKS-1].
     */
    inline sync_point<SYNC_SPACE_DTYPE> get_blockwise_sync_point(int dev_idx, int block_idx) const {
        int offset = all_sync_point_size + block_idx;
        sync_point<SYNC_SPACE_DTYPE> sp;
        sp.num_devices = NUM_DEVICES;
        sp.local_ptr = sync_spaces[dev_idx] + offset;
        for (int i = 0; i < NUM_DEVICES; i++)
            sp.p2p_ptrs[i] = sync_spaces[i] + offset;
        return sp;
    }

    /**
     * @brief Get sync point for gang::blockgroup::sync().
     * Uses offsets [2+MAX_BLOCKS .. end].
     */
    inline sync_point<SYNC_SPACE_DTYPE> get_blockgroup_sync_point(int dev_idx, int sync_id) const {
        int offset = all_sync_point_size + MAX_BLOCKS + sync_id;
        sync_point<SYNC_SPACE_DTYPE> sp;
        sp.num_devices = NUM_DEVICES;
        sp.local_ptr = sync_spaces[dev_idx] + offset;
        for (int i = 0; i < NUM_DEVICES; i++)
            sp.p2p_ptrs[i] = sync_spaces[i] + offset;
        return sp;
    }

private:
    sync_manager() = default;
};

} // namespace kittens
