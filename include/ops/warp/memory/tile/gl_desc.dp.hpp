/**
 * @file
 * @brief Hoisted 2D-block descriptor ("Option A"): build the block2d address
 *        payload ONCE and mutate only the surface X/Y coordinate per load.
 *
 * ============================================================================
 * WHY THIS EXISTS
 * ----------------------------------------------------------------------------
 * The default load path (load_part / load / prefetch_load in
 * global_to_register.dp.hpp) lowers through the FLAT intrinsic
 * `__spirv_Subgroup2DBlockLoadINTEL(ElemSize, W, H, Count, base, mem_w, mem_h,
 * pitch, coord, dst)` — every descriptor field is passed on EVERY call, so IGC
 * re-materializes the full block2d payload GRF each load (the redundant
 * width/pitch/height/shape `mov`s that dominate small-GEMM / issue-bound
 * kernels).
 *
 * `gl_desc` builds the payload once via `createBlock2DAddressPayload` (base +
 * width/height/pitch/shape) and, inside the K-loop, only calls
 * `setBlock2DAddressPayloadBlockX/Y`. This is exactly the model used by
 * sycl-tla / CUTLASS-Xe (Xe2DTraitsBase::device_init + update_payload). The
 * block read reuses sycl-tla's proven `lsc_load_block2d ... flat[payload]`
 * inline asm, so the bytes deposited are bit-identical to the flat path.
 *
 * ============================================================================
 * SMOOTH-TRANSITION CONTRACT (read before adopting)
 * ----------------------------------------------------------------------------
 *  - PURELY ADDITIVE. This header introduces only new symbols in `kittens`.
 *    It is NOT included by tile.dp.hpp / the aggregate headers, so existing
 *    translation units are byte-for-byte unchanged until you `#include` it.
 *  - OPT-IN per call site. `load()/load_part()/prefetch_load()/mma_*` are
 *    untouched. Migrate ONE operand at a time behind a compile flag and gate
 *    each step on a bit-exact vs. the flat path check.
 *  - The descriptor is a LOOP-INVARIANT object: construct it BEFORE the K-loop
 *    (so `payload` stays resident in one GRF); inside the loop touch only X/Y.
 *    If it is constructed inside the loop, IGC re-materializes it and the win
 *    is lost (this is the likely cause of an earlier "only small speedup").
 *  - Register cost: each live `gl_desc` is ~1 GRF. GEMM runs near the 256-GRF
 *    ceiling — share one descriptor per operand-stream and advance X/Y rather
 *    than holding many; re-run spillcheck after adopting.
 *  - The `.load()` inline asm is unvalidated in THIS tree until you build+run
 *    it on PVC. Confirm bit-exact against the flat load before trusting it.
 * ============================================================================
 */

#pragma once

#include "../../../../types/sycl_type.hpp"
#include "../../../../types/types.dp.hpp"
#include <type_traits>

namespace kittens {

/**
 * @brief A hoisted 2D-block-load descriptor.
 *
 * @tparam Bits        bits per memory element (16 for bf16/half, 32 for float).
 * @tparam BlockHeight rows per block (matches the flat intrinsic BlockHeight).
 * @tparam BlockWidth  contiguous elements per block (flat intrinsic BlockWidth).
 * @tparam NumBlocks   number of blocks along the contiguous dim (flat BlockCount).
 * @tparam VNNI        true selects the transform/VNNI read (`nt`, matches
 *                     Subgroup2DBlockLoadTransformINTEL — the B operand);
 *                     false selects the plain read (`nn` — the A operand).
 *
 * Coordinates are ELEMENT coordinates (identical to the {col,row} the flat
 * intrinsic takes): X = contiguous/column element index, Y = strided/row
 * element index.
 */
template <int Bits, int BlockHeight, int BlockWidth, int NumBlocks,
          bool VNNI = false>
struct gl_desc {
  int *payload = nullptr; ///< the block2d address payload (one GRF, built once)

  static constexpr int sg_size = 16;
  static constexpr int atom_width = BlockWidth * NumBlocks; ///< total contiguous elems
  // Per-lane storage vector matching the flat load's deposited bytes.
  using elem_t = std::conditional_t<(Bits <= 16), short, int>;
  static constexpr int lane_count =
      (atom_width * BlockHeight * Bits / sg_size) /
      static_cast<int>(sizeof(elem_t) * 8);
  using store_vec_t = intel::vector_t<elem_t, lane_count>;

  inline gl_desc() = default;

  /**
   * @brief Build the payload once from a 2D global layout (rows=axis 2,
   *        contiguous cols=axis 3). Base points at the tensor origin; per-tile
   *        selection is done later with set_block/set_x/set_y.
   */
  template <ducks::gl::all GL> inline explicit gl_desc(const GL &g) {
    init(g);
  }

  template <ducks::gl::all GL> inline void init(const GL &g) {
    using U = typename GL::dtype;
    const long base = reinterpret_cast<long>(g.raw_ptr);
    const int width_bytes = static_cast<int>(g.template shape<3>() * sizeof(U));
    const int height = static_cast<int>(g.template shape<2>());
    const int pitch_bytes =
        static_cast<int>(g.template stride<2>() * sizeof(U));
    payload = __builtin_IB_subgroup_createBlock2DAddressPayload(
        base, width_bytes - 1, height - 1, pitch_bytes - 1,
        /*blockX=*/0, /*blockY=*/0, BlockWidth, BlockHeight, NumBlocks);
  }

  /// Set both surface coordinates (element units).
  inline void set_block(int x, int y) const {
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payload, x);
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payload, y);
  }
  /// Set only the contiguous/column coordinate (element units).
  inline void set_x(int x) const {
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payload, x);
  }
  /// Set only the strided/row coordinate (element units).
  inline void set_y(int y) const {
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payload, y);
  }
  /// Advance the contiguous/column coordinate by dx (element units).
  inline void advance_x(int dx) const {
    __builtin_IB_subgroup_addBlock2DAddressPayloadBlockX(payload, dx);
  }
  /// Advance the strided/row coordinate by dy (element units).
  inline void advance_y(int dy) const {
    __builtin_IB_subgroup_addBlock2DAddressPayloadBlockY(payload, dy);
  }

  /**
   * @brief Load the block(s) into register-tile storage. `dst` is the same
   *        pointer the flat path writes (e.g. &reg.tiles[i][j].data[0]); the
   *        deposited bytes are bit-identical. Mirrors sycl-tla XE_LOAD_2D /
   *        XE_LOAD_2D_VNNI.
   */
  template <typename T> inline void load(T *dst) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto &dv = *reinterpret_cast<store_vec_t *>(dst);
    if constexpr (VNNI) {
      asm("lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
          : "=rw"(dv)
          : "rw.u"(payload), "P"(Bits), "P"(NumBlocks), "P"(BlockWidth),
            "P"(BlockHeight));
    } else {
      asm("lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
          : "=rw"(dv)
          : "rw.u"(payload), "P"(Bits), "P"(NumBlocks), "P"(BlockWidth),
            "P"(BlockHeight));
    }
#else
    (void)dst;
    INVALID_CONTROL_PATH("gl_desc::load is a device-only path.");
#endif
  }

  /**
   * @brief Prefetch (cache-warm) the block region at the CURRENT X/Y into
   *        L1/L3 (`.ca.ca`), discarding the data (`%null`). Mirrors sycl-tla
   *        XE_PREFETCH_2D — layout-agnostic (no VNNI/transpose), so the same
   *        method serves the A (plain) and B (VNNI) streams.
   *
   * For software-pipelined prefetch (a K-step AHEAD of the load), give the
   * prefetch its OWN descriptor per operand, built once and advanced
   * independently — do NOT reuse the load descriptor's X/Y, or the pending
   * load coordinate is clobbered. See the migration recipe. Each extra
   * descriptor is ~1 GRF; re-run spillcheck.
   */
  inline void prefetch() const {
#ifdef __SYCL_DEVICE_ONLY__
    asm("lsc_load_block2d.ugm.ca.ca (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(0,0)]"
        :: "rw.u"(payload), "P"(Bits), "P"(atom_width), "P"(BlockHeight));
#endif
  }
};

// ---------------------------------------------------------------------------
// Typed factory helpers for the locked bf16 GEMM operands (rt_32x32).
//   A: row-major, plain (nn)  — 16x32 x2 blocks, matches load_part<row>.
//   B: col-major, VNNI  (nt)  — 16x32 x2 blocks, matches load_part<col>
//                               (__spirv_Subgroup2DBlockLoadTransformINTEL).
// Build once (outside the K-loop); see the migration recipe in the docs.
// ---------------------------------------------------------------------------
template <ducks::gl::all GL>
inline gl_desc<16, 32, 16, 2, /*VNNI=*/false> make_desc_A_bf16_32x32(const GL &g) {
  return gl_desc<16, 32, 16, 2, false>(g);
}
template <ducks::gl::all GL>
inline gl_desc<16, 32, 16, 2, /*VNNI=*/true> make_desc_B_bf16_32x32(const GL &g) {
  return gl_desc<16, 32, 16, 2, true>(g);
}

} // namespace kittens
