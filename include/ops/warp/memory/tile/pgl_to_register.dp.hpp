/**
 * @file
 * @brief PGL <-> register-tile operations for Intel PVC.
 *
 * Primary path is register<->global; each operation composes on top of the
 * tested `kittens::load` / `kittens::store` GL primitives that use
 * `__spirv_Subgroup2DBlockLoadINTEL` / `__spirv_Subgroup2DBlockStoreINTEL`.
 *
 * Cross-device access uses `pgl[dev_idx]`, which returns a GL whose data
 * pointer is peer-visible when all devices share a single SYCL context
 * (the convention already used by `pgl_malloc_device` + `make_pgl`).
 *
 * NOTE: `store_add` (RMW at tile granularity) is NOT concurrent-safe. Callers
 * must guarantee that at most one work-item targets a given (dev_idx, coord)
 * within a single kernel launch. This suffices for the reduce-scatter fused
 * kernel pattern where each producer tile is uniquely assigned.
 */
#pragma once
#include <sycl/sycl.hpp>
#include "global_to_register.dp.hpp"

namespace kittens {

/* ============================================================
 *  register-tile -> PGL[dev_idx]  (non-atomic peer store)
 * ============================================================ */

template <int axis, ducks::rt::all RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store(const PGL &dst, const RT &src, int dev_idx,
                         const COORD &idx) {
    ::kittens::store<axis, RT, typename PGL::GL, COORD>(dst[dev_idx], src, idx);
}

template <ducks::rt::all RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store(const PGL &dst, const RT &src, int dev_idx,
                         const COORD &idx) {
    store<2, RT, PGL, COORD>(dst, src, dev_idx, idx);
}

/* ============================================================
 *  PGL[dev_idx] -> register-tile  (peer load)
 * ============================================================ */

template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load(RT &dst, const PGL &src, int dev_idx,
                        const COORD &idx) {
    ::kittens::load<axis, RT, typename PGL::GL, COORD>(dst, src[dev_idx], idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void load(RT &dst, const PGL &src, int dev_idx,
                        const COORD &idx) {
    load<2, RT, PGL, COORD>(dst, src, dev_idx, idx);
}

/* ============================================================
 *  register-tile +=> PGL[dev_idx]  (read-modify-write, non-atomic)
 *
 *  Semantics: dst[dev_idx][coord] += src
 *  Implementation: tile-granularity load -> add -> store.
 *  Constraint: caller MUST ensure no two work-items target the same
 *  (dev_idx, coord) within one kernel launch (otherwise: race).
 * ============================================================ */

template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store_add(const PGL &dst, const RT &src, int dev_idx,
                             const COORD &idx) {
    RT tmp;
    ::kittens::load<axis, RT, typename PGL::GL, COORD>(tmp, dst[dev_idx], idx);
    add(tmp, tmp, src);
    ::kittens::store<axis, RT, typename PGL::GL, COORD>(dst[dev_idx], tmp, idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void store_add(const PGL &dst, const RT &src, int dev_idx,
                             const COORD &idx) {
    store_add<2, RT, PGL, COORD>(dst, src, dev_idx, idx);
}

/* ============================================================
 *  Cross-device reductions:  dst_reg = OP over pgl[0..N-1] at coord
 *
 *  Implementation: sequential peer loads accumulated into dst.
 *  Bandwidth-bound; latency = N * peer-load latency (no NVLS multicast).
 * ============================================================ */

namespace detail {

template <typename T> struct pgl_reduce_add {
    inline static void apply(T &acc, const T &v) { add(acc, acc, v); }
};

} // namespace detail

template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void all_reduce_add(RT &dst, const PGL &src, int /*dev_idx*/,
                                  const COORD &idx) {
    ::kittens::load<axis, RT, typename PGL::GL, COORD>(dst, src[0], idx);
    RT tmp;
    #pragma unroll
    for (int d = 1; d < PGL::num_devices; d++) {
        ::kittens::load<axis, RT, typename PGL::GL, COORD>(tmp, src[d], idx);
        add(dst, dst, tmp);
    }
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void all_reduce_add(RT &dst, const PGL &src, int dev_idx,
                                  const COORD &idx) {
    all_reduce_add<2, RT, PGL, COORD>(dst, src, dev_idx, idx);
}

/* ============================================================
 *  broadcast: write src register tile to ALL devices in pgl at coord.
 * ============================================================ */

template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void broadcast(const PGL &dst, const RT &src, int /*dev_idx*/,
                             const COORD &idx) {
    #pragma unroll
    for (int d = 0; d < PGL::num_devices; d++) {
        ::kittens::store<axis, RT, typename PGL::GL, COORD>(dst[d], src, idx);
    }
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL,
          ducks::coord::tile COORD = coord<RT>>
inline static void broadcast(const PGL &dst, const RT &src, int dev_idx,
                             const COORD &idx) {
    broadcast<2, RT, PGL, COORD>(dst, src, dev_idx, idx);
}

} // namespace kittens
