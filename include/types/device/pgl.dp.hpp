/**
 * @file
 * @brief Parallel Global Layout for Intel PVC multi-GPU.
 *
 * Ported from ThunderKittens CUDA pgl.cuh.
 *
 * Key differences from CUDA version:
 *   - No NVLS multicast (Intel PVC doesn't have NVSwitch multicast)
 *   - No TMA descriptors (Intel PVC uses explicit loads)
 *   - Uses P2P device pointers via shared SYCL context
 *   - dev_ptrs[] replaces mc_ptr for cross-device access
 *
 * The PGL stores per-device GL objects (same shape, different data pointers)
 * and an array of raw device pointers for direct P2P access.
 * All devices must share a SYCL context for P2P to work.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdint>
#include <cstring>
#include "../../common/common.dp.hpp"
#include "../shared/shared.dp.hpp"
#include "../global/global.dp.hpp"

namespace kittens {

/* ----------  Parallel global layout descriptor  ---------- */

namespace ducks {
namespace pgl {

struct identifier {};

/**
 * @brief Concept for all parallel global layouts.
 */
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace pgl
} // namespace ducks

/**
 * @brief Parallel global layout for Intel PVC.
 *
 * Represents the same tensor replicated across multiple GPUs.
 * Each device has its own GL with a local data pointer.
 * dev_ptrs[] stores all device pointers for P2P access.
 *
 * @tparam GL The underlying global layout on each device.
 * @tparam NUM_DEVICES The number of GPU devices.
 * @tparam P2P_ACCESS Whether cross-device P2P access is enabled.
 *         When true, dev_ptrs[] is populated for cross-device reads/writes.
 *         When false, only local GL access is available.
 */
template<kittens::ducks::gl::all _GL, int NUM_DEVICES = 6, bool P2P_ACCESS = true>
struct pgl {
    using identifier = ducks::pgl::identifier;
    using GL = _GL;
    using T = GL::dtype;
    using dtype = T;

    static constexpr int num_devices = NUM_DEVICES;
    static constexpr bool p2p_access = P2P_ACCESS;

    GL gls[NUM_DEVICES];

    // Per-device raw pointers for P2P access.
    // All pointers must be USM device allocations within a shared SYCL context.
    T *dev_ptrs[NUM_DEVICES];

    // Access the GL for a specific device
    const GL &operator[](int idx) const { return gls[idx]; }

    /**
     * @brief Get a P2P-accessible pointer at a specific coordinate on a specific device.
     *
     * This is the Intel PVC equivalent of CUDA's mc_ptr_at().
     * Instead of going through multicast, it directly accesses the target device's memory
     * via P2P (requires shared SYCL context).
     */
    inline T* dev_ptr_at(const coord<ducks::default_type> &idx, int dev_idx) const {
        static_assert(P2P_ACCESS, "P2P access is not enabled for this PGL.");
        const GL &gl = gls[0]; // all gls have the same shape
        return &dev_ptrs[dev_idx][((idx.b * static_cast<uint64_t>(gl.depth()) + idx.d)
                                    * gl.rows() + idx.r) * gl.cols() + idx.c];
    }

    pgl() = delete;

    // Constructor: no P2P (local-only access)
    inline pgl(T **_data,
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _data, _batch, _depth, _rows, _cols) {}

    // Constructor: with P2P access
    inline pgl(T **_dev_ptrs,  // P2P-accessible pointers for all devices
               T **_data,      // per-device local data pointers (may be same as _dev_ptrs)
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _dev_ptrs, _data, _batch, _depth, _rows, _cols) {}

    // Internal: no P2P
    template<size_t... I>
    inline pgl(std::index_sequence<I...>,
               T **_data,
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : gls{GL(_data[I], _batch, _depth, _rows, _cols)...}, dev_ptrs{} {
        static_assert(!P2P_ACCESS, "P2P pointers not passed to P2P-enabled PGL.");
        static_assert(NUM_DEVICES > 1,
            "No point in using PGL with a single device.");
    }

    // Internal: with P2P
    template<size_t... I>
    inline pgl(std::index_sequence<I...>,
               T **_dev_ptrs,
               T **_data,
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : gls{GL(_data[I], _batch, _depth, _rows, _cols)...},
          dev_ptrs{_dev_ptrs[I]...} {
        static_assert(P2P_ACCESS, "P2P pointers passed to P2P-disabled PGL.");
    }

    // Dimension accessors (forwarded to gls[0], all devices have same shape)
    inline auto batch() const { return gls[0].batch(); }
    inline auto depth() const { return gls[0].depth(); }
    inline auto rows()  const { return gls[0].rows(); }
    inline auto cols()  const { return gls[0].cols(); }
    inline auto numel() const { return gls[0].numel(); }

    template<int axis> inline size_t shape()  const { return gls[0].template shape<axis>(); }
    template<int axis> inline size_t stride() const { return gls[0].template stride<axis>(); }

    // Total size in bytes of one device's data
    inline size_t gl_size() const {
        return static_cast<size_t>(this->batch()) * static_cast<size_t>(this->depth())
             * static_cast<size_t>(this->rows()) * static_cast<size_t>(this->cols()) * sizeof(T);
    }
};

#ifndef KITTENS_DEVICE_ONLY

/**
 * @brief Create a PGL from raw pointer arrays (no P2P).
 */
template<ducks::pgl::all PGL, bool safe=true>
inline PGL make_pgl(
    uint64_t *data, int b, int d, int r, int c
) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__)
            throw std::runtime_error("Batch dimension mismatch.");
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__)
            throw std::runtime_error("Depth dimension mismatch.");
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__)
            throw std::runtime_error("Row dimension mismatch.");
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__)
            throw std::runtime_error("Column dimension mismatch.");
    }
    // Convert uint64_t[] to typed T*[] via memcpy to avoid strict-aliasing UB.
    // Reading through reinterpret_cast<T**>(uint64_t*) is UB (icpx at -O3
    // will elide the reads and yield zero pointers).
    using T = typename PGL::dtype;
    T *data_typed[PGL::num_devices];
    for (int i = 0; i < PGL::num_devices; i++) {
        std::memcpy(&data_typed[i], &data[i], sizeof(T*));
    }
    return PGL(
        data_typed,
        make_unsafe_gl_arg<PGL::GL::__b__>(b),
        make_unsafe_gl_arg<PGL::GL::__d__>(d),
        make_unsafe_gl_arg<PGL::GL::__r__>(r),
        make_unsafe_gl_arg<PGL::GL::__c__>(c)
    );
}

/**
 * @brief Create a PGL from raw pointer arrays (with P2P).
 */
template<ducks::pgl::all PGL, bool safe=true>
inline PGL make_pgl(
    uint64_t *dev_ptrs, uint64_t *data, int b, int d, int r, int c
) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__)
            throw std::runtime_error("Batch dimension mismatch.");
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__)
            throw std::runtime_error("Depth dimension mismatch.");
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__)
            throw std::runtime_error("Row dimension mismatch.");
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__)
            throw std::runtime_error("Column dimension mismatch.");
    }
    // Convert uint64_t[] to typed T*[] via memcpy to avoid strict-aliasing UB.
    using T = typename PGL::dtype;
    T *dev_typed[PGL::num_devices];
    T *data_typed[PGL::num_devices];
    for (int i = 0; i < PGL::num_devices; i++) {
        std::memcpy(&dev_typed[i],  &dev_ptrs[i], sizeof(T*));
        std::memcpy(&data_typed[i], &data[i],     sizeof(T*));
    }
    return PGL(
        dev_typed,
        data_typed,
        make_unsafe_gl_arg<PGL::GL::__b__>(b),
        make_unsafe_gl_arg<PGL::GL::__d__>(d),
        make_unsafe_gl_arg<PGL::GL::__r__>(r),
        make_unsafe_gl_arg<PGL::GL::__c__>(c)
    );
}

#endif // KITTENS_DEVICE_ONLY

/**
 * @brief Convenience type alias for inter-device barriers.
 *
 * Uses P2P-accessible int buffer for cross-GPU synchronization
 * via SYCL atomic operations (replaces CUDA multicast barriers).
 */
template <int NUM_DEVICES>
using barrier_t = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

/**
 * @brief Allocate USM device memory with P2P access for PGL use.
 *
 * Allocates memory on a specific device within a shared SYCL context.
 * The returned pointer is accessible from all devices in the context.
 *
 * @param queue SYCL queue for the target device
 * @param num_elements Number of elements to allocate
 * @return Device pointer accessible via P2P
 */
template <typename T>
inline T* pgl_malloc_device(sycl::queue &queue, size_t num_elements) {
    return sycl::malloc_device<T>(num_elements, queue);
}

/**
 * @brief Free USM device memory allocated for PGL use.
 */
template <typename T>
inline void pgl_free(T *ptr, sycl::queue &queue) {
    sycl::free(ptr, queue);
}

} // namespace kittens

// Trivially-copyable trait specializations so gl / pgl objects can be captured
// by value into SYCL kernels (both types hold only pointers and integer dims).
namespace sycl {
template <typename _T, int b, int d, int r, int c, typename... TMA>
struct is_device_copyable<kittens::gl<_T, b, d, r, c, TMA...>> : std::true_type {};

template <kittens::ducks::gl::all GL, int N, bool P2P>
struct is_device_copyable<kittens::pgl<GL, N, P2P>> : std::true_type {};
} // namespace sycl
