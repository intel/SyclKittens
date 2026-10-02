/**
 * @file
 * @brief Parallel Global Layout (PGL) for Intel PVC multi-GPU.
 *
 * Ported from ThunderKittens CUDA (pgl.cuh).
 * Intel PVC has no NVLS multicast, so PGL is simplified:
 *   - Array of per-device gl objects
 *   - Array of raw device pointers for P2P cross-device access
 *   - No multicast pointer (mc_ptr)
 *   - No TMA descriptors for multicast
 *
 * The MULTICAST template parameter is kept for API compatibility
 * but is effectively a no-op on Intel PVC.
 */

#pragma once

#include <sycl/sycl.hpp>
#include "../../common/common.dp.hpp"
#include "../global/global.dp.hpp"

namespace kittens {

namespace ducks {
namespace pgl {

struct identifier {};

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace pgl
} // namespace ducks

/**
 * @brief Parallel global layout for multi-GPU data.
 * @tparam GL      The per-device global layout type.
 * @tparam NUM_DEVICES Number of GPU devices.
 * @tparam MULTICAST   Kept for API compatibility; no-op on Intel PVC.
 */
template<kittens::ducks::gl::all _GL, int NUM_DEVICES = 6, bool MULTICAST = false>
struct pgl {
    using identifier = ducks::pgl::identifier;
    using GL = _GL;
    using T = GL::dtype;
    using dtype = T;

    static constexpr int num_devices = NUM_DEVICES;
    static constexpr bool multicast = MULTICAST;

    GL gls[NUM_DEVICES];

    const GL &operator[](int idx) const { return gls[idx]; }

    // No multicast on Intel PVC — provide a P2P-accessible pointer helper
    // Returns pointer into the specified device's buffer at the given coordinate
    inline T* ptr_at(int dev_idx, const coord<ducks::default_type> &idx) const {
        const GL &gl = gls[dev_idx];
        return &gls[dev_idx].raw_ptr[
            ((idx.b * static_cast<uint64_t>(gl.depth()) + idx.d) * gl.rows() + idx.r) * gl.cols() + idx.c
        ];
    }

    pgl() = default;

    // Constructor: array of per-device data pointers
    inline pgl(T **_data,
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _data, _batch, _depth, _rows, _cols) {}

    template<size_t... I>
    inline pgl(std::index_sequence<I...>,
               T **_data,
               ducks::gl::make_arg_t<GL::__b__> _batch,
               ducks::gl::make_arg_t<GL::__d__> _depth,
               ducks::gl::make_arg_t<GL::__r__> _rows,
               ducks::gl::make_arg_t<GL::__c__> _cols)
        : gls{GL(_data[I], _batch, _depth, _rows, _cols)...} {}

    inline auto batch() const { return gls[0].batch(); }
    inline auto depth() const { return gls[0].depth(); }
    inline auto rows()  const { return gls[0].rows(); }
    inline auto cols()  const { return gls[0].cols(); }
    inline auto numel() const { return gls[0].numel(); }

    template<int axis> inline size_t shape() const { return gls[0].template shape<axis>(); }
    template<int axis> inline size_t stride() const { return gls[0].template stride<axis>(); }
};

// Factory: create PGL from array of uint64_t pointers
template<ducks::pgl::all PGL, bool safe=true>
inline PGL make_pgl(
    uint64_t *data, int b, int d, int r, int c
) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__)
            throw std::runtime_error("Batch dimension mismatch");
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__)
            throw std::runtime_error("Depth dimension mismatch");
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__)
            throw std::runtime_error("Row dimension mismatch");
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__)
            throw std::runtime_error("Column dimension mismatch");
    }
    return PGL(
        reinterpret_cast<typename PGL::dtype**>(data),
        make_unsafe_gl_arg<PGL::GL::__b__>(b),
        make_unsafe_gl_arg<PGL::GL::__d__>(d),
        make_unsafe_gl_arg<PGL::GL::__r__>(r),
        make_unsafe_gl_arg<PGL::GL::__c__>(c)
    );
}

// Convenience barrier type — uses int pgl for cross-device synchronization
template <int NUM_DEVICES>
using barrier_t = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

} // namespace kittens
