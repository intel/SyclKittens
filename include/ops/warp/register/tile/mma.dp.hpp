/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"
#include "../../../../types/sycl_type.hpp"

namespace kittens {



/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
static inline void hmma16816(sycl::float2 &d0, sycl::float2 &d1,
                             const bf16_2 &a0, const bf16_2 &a1,
                             const bf16_2 &a2, const bf16_2 &a3,
                             const bf16_2 &b0, const bf16_2 &b1,
                             const sycl::float2 &c0, const sycl::float2 &c1) {

//    asm volatile(
//        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
//        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
//        "{%0, %1, %2, %3}, "
//        "{%4, %5, %6, %7}, "
//        "{%8, %9}, "
//        "{%10, %11, %12, %13};"

//        // D matrix
//        : "+f"(d0.x()), "+f"(d0.y()), "+f"(d1.x()), "+f"(d1.y())

//        // A matrix
//        : "r"(*(uint32_t *)(&a0)), "r"(*(uint32_t *)(&a1)),
//          "r"(*(uint32_t *)(&a2)), "r"(*(uint32_t *)(&a3)),

//          // B matrix
//          "r"(*(uint32_t *)(&b0)), "r"(*(uint32_t *)(&b1)),

//          // C matrix
//          "f"(c0.x()), "f"(c0.y()), "f"(c1.x()), "f"(c1.y()));
    volatile void *d_mat_frag_ct1[4] = { &d0.x(), &d0.y(), &d1.x(), &d1.y() };

    sycl::vec<uint32_t, 4> a_mat_frag_ct1(*(uint32_t *)(&a0), *(uint32_t *)(&a1), *(uint32_t *)(&a2), *(uint32_t *)(&a3));

    sycl::vec<uint32_t, 2> b_mat_frag_ct1(*(uint32_t *)(&b0), *(uint32_t *)(&b1));

    sycl::vec<float, 4> c_mat_frag_ct1(c0.x(), c0.y(), c1.x(), c1.y());

    dpct::experimental::matrix::mma<16, 8, 16, sycl::ext::oneapi::bfloat16, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);

}

//
static inline void
hmma81616(sycl::float2 &d0, sycl::float2 &d1, sycl::float2 &d2,
          sycl::float2 &d3, const bf16_2 &a0, const bf16_2 &a1,
          const bf16_2 &a2, const bf16_2 &a3, const bf16_2 &b0,
          const bf16_2 &b1, const bf16_2 &b2, const bf16_2 &b3,
          const bf16_2 &b4, const bf16_2 &b5, const bf16_2 &b6,
          const bf16_2 &b7, const sycl::float2 &c0, const sycl::float2 &c1,
          const sycl::float2 &c2, const sycl::float2 &c3) {
  const intel::short8 matrix_a{
      sycl::bit_cast<short>(a0.x()), sycl::bit_cast<short>(a0.y()),
      sycl::bit_cast<short>(a1.x()), sycl::bit_cast<short>(a1.y()),
      sycl::bit_cast<short>(a2.x()), sycl::bit_cast<short>(a2.y()),
      sycl::bit_cast<short>(a3.x()), sycl::bit_cast<short>(a3.y())};

  const intel::int8 matrix_b{sycl::bit_cast<int>(b0), sycl::bit_cast<int>(b1),
                             sycl::bit_cast<int>(b2), sycl::bit_cast<int>(b3),
                             sycl::bit_cast<int>(b4), sycl::bit_cast<int>(b5),
                             sycl::bit_cast<int>(b6), sycl::bit_cast<int>(b7)};

  const intel::float8 matrix_c{c0.x(), c0.y(), c1.x(), c1.y(),
                               c2.x(), c2.y(), c3.x(), c3.y()};

  const intel::float8 result = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
      16, matrix_a, matrix_b, matrix_c,
      intel::SPIRV_MMAOperands::SPIRV_MatrixABf16 |
          intel::SPIRV_MMAOperands::SPIRV_MatrixBBf16);
  d0.x() = result[0];
  d0.y() = result[1];
  d1.x() = result[2];
  d1.y() = result[3];
  d2.x() = result[4];
  d2.y() = result[5];
  d3.x() = result[6];
  d3.y() = result[7];
}

// ============================================================================
// dpas-native packed tiles — explicit pack / bare-dpas mma (increment 1).
//
// DESIGN: docs/dsl_packed_tile_design.md. A high-perf mma() should compile to a
// bare dpas with zero operand marshalling. The standard mma_*_base -> hmma81616
// path instead REBUILDS the SPIRV operand vectors (short8 A / int8 B / float8 C)
// from rt_base.data[] on EVERY dpas (~16-24 register moves/dpas; measured in
// tests/unit/mma_marshal_probe.dp.cpp). The compiler cannot hoist this because it
// is buried inside each inlined mma call (probe MODE3: a loop-invariant operand is
// still re-marshalled 8x/dpas).
//
// These types expose the dpas-native format (packed-bf16, VNNI, linear) as a
// first-class tile with EXPLICIT pack (premarshal_*) / unpack (store_C), so the
// kernel author places the conversion optimally: pack loop-invariant / reused
// operands ONCE, keep the accumulator resident as float8 across the K reduction,
// and unpack only at the mma->elementwise boundary. Probe MODE4/5 measure
// ~0.5 marshal moves/dpas with this path vs 16-24 for mma_*_base.
//
// SAFETY (affects other kernels): this block is ADDITIVE and OPT-IN. Do NOT change
// hmma81616 / mma_*_base / mma<>() — attention, gemm, layernorm, rotary, ring all
// depend on their exact behaviour. Validate any use: zebin spill-free + bit-exact
// + interleaved perf (see the design doc).
//
// Layout mirrors the rt_16x16 mma_AB_base branch exactly:
//   half0: a.data[0..3] * b.data[0..7] -> c.data[0..3]
//   half1: a.data[4..7] * b.data[0..7] -> c.data[4..7]   (B shared)
// ============================================================================

// A 16x16 bf16 row tile -> two dpas A operands (top/bottom 8-row halves).
struct dpas_A_bf16_16x16 { intel::short8 h[2]; };
// A 16x16 bf16 col tile -> one dpas B operand (shared across both A halves).
struct dpas_B_bf16_16x16 { intel::int8 m; };
// A 16x16 f32 row accumulator -> two dpas C operands (top/bottom 8-row halves).
struct dpas_C_f32_16x16 { intel::float8 h[2]; };

static inline void premarshal_A(
    dpas_A_bf16_16x16 &o,
    const rt_base<bf16, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &a) {
#pragma unroll
  for (int t = 0; t < 2; t++) {
    const int b = t * 4;
    o.h[t] = intel::short8{
        sycl::bit_cast<short>(a.data[b + 0].x()), sycl::bit_cast<short>(a.data[b + 0].y()),
        sycl::bit_cast<short>(a.data[b + 1].x()), sycl::bit_cast<short>(a.data[b + 1].y()),
        sycl::bit_cast<short>(a.data[b + 2].x()), sycl::bit_cast<short>(a.data[b + 2].y()),
        sycl::bit_cast<short>(a.data[b + 3].x()), sycl::bit_cast<short>(a.data[b + 3].y())};
  }
}

static inline void premarshal_B(
    dpas_B_bf16_16x16 &o,
    const rt_base<bf16, ducks::rt_layout::col, ducks::rt_shape::rt_16x16> &b) {
  o.m = intel::int8{
      sycl::bit_cast<int>(b.data[0]), sycl::bit_cast<int>(b.data[1]),
      sycl::bit_cast<int>(b.data[2]), sycl::bit_cast<int>(b.data[3]),
      sycl::bit_cast<int>(b.data[4]), sycl::bit_cast<int>(b.data[5]),
      sycl::bit_cast<int>(b.data[6]), sycl::bit_cast<int>(b.data[7])};
}

static inline void zero_C(dpas_C_f32_16x16 &c) {
#pragma unroll
  for (int t = 0; t < 2; t++)
    c.h[t] = intel::float8{0, 0, 0, 0, 0, 0, 0, 0};
}

// C += A * B  (raw dpas, no per-call operand reconstruction).
static inline void mma_raw_AB(dpas_C_f32_16x16 &c,
                              const dpas_A_bf16_16x16 &a,
                              const dpas_B_bf16_16x16 &b) {
#pragma unroll
  for (int t = 0; t < 2; t++)
    c.h[t] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
        16, a.h[t], b.m, c.h[t],
        intel::SPIRV_MMAOperands::SPIRV_MatrixABf16 |
            intel::SPIRV_MMAOperands::SPIRV_MatrixBBf16);
}

// dpas-native accumulator -> rt_base f32 row tile.
static inline void store_C(
    rt_base<float, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &d,
    const dpas_C_f32_16x16 &c) {
#pragma unroll
  for (int t = 0; t < 2; t++) {
    const int b = t * 4;
    d.data[b + 0] = sycl::float2{c.h[t][0], c.h[t][1]};
    d.data[b + 1] = sycl::float2{c.h[t][2], c.h[t][3]};
    d.data[b + 2] = sycl::float2{c.h[t][4], c.h[t][5]};
    d.data[b + 3] = sycl::float2{c.h[t][6], c.h[t][7]};
  }
}

// ----------------------------------------------------------------------------
// ABt 16x32 pre-marshalled operands (S = K[16x32] * Q[16x32]^T -> 16x16).
// A 16x32 register tile (qk_chunk_t) is [1][2] of rt_16x16 base tiles: col-tile
// 0 = K-block 0 (cols 0..15), col-tile 1 = K-block 1 (cols 16..31). This is
// bit-identical to a genuine rt_16x32 base tile's data[0..7]/data[8..15] split,
// so the contraction mirrors the mma_ABt_base rt_16x16/rt_16x16 x2 accumulation:
//   acc_h0 += A_top(kb0)*B(kb0) + A_top(kb1)*B(kb1)
//   acc_h1 += A_bot(kb0)*B(kb0) + A_bot(kb1)*B(kb1)
// A operand (row) is layout-identical for AB and ABt; B (row, "row-major mode").
// ----------------------------------------------------------------------------
struct dpas_A_bf16_16x32 { intel::short8 h[2][2]; };  // [out-half][K-block]
struct dpas_B_bf16_16x32 { intel::int8 k[2]; };        // [K-block]

// pack 4 consecutive bf16_2 (data[base..base+3]) into a dpas A short8.
static inline intel::short8 pack_A4(const bf16_2 *d, int base) {
  return intel::short8{
      sycl::bit_cast<short>(d[base + 0].x()), sycl::bit_cast<short>(d[base + 0].y()),
      sycl::bit_cast<short>(d[base + 1].x()), sycl::bit_cast<short>(d[base + 1].y()),
      sycl::bit_cast<short>(d[base + 2].x()), sycl::bit_cast<short>(d[base + 2].y()),
      sycl::bit_cast<short>(d[base + 3].x()), sycl::bit_cast<short>(d[base + 3].y())};
}
// pack 8 consecutive bf16_2 (data[base..base+7]) into a dpas B int8.
static inline intel::int8 pack_B8(const bf16_2 *d, int base) {
  return intel::int8{
      sycl::bit_cast<int>(d[base + 0]), sycl::bit_cast<int>(d[base + 1]),
      sycl::bit_cast<int>(d[base + 2]), sycl::bit_cast<int>(d[base + 3]),
      sycl::bit_cast<int>(d[base + 4]), sycl::bit_cast<int>(d[base + 5]),
      sycl::bit_cast<int>(d[base + 6]), sycl::bit_cast<int>(d[base + 7])};
}

static inline void premarshal_A_16x32(
    dpas_A_bf16_16x32 &o,
    const rt_base<bf16, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &kb0,
    const rt_base<bf16, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &kb1) {
  // qk_chunk_t is [1][2] of rt_16x16 base tiles: kb0 = K-cols 0..15, kb1 = 16..31.
  // Per 16x16 tile the A operand is data[0..3] (top 8 rows) + data[4..7] (bottom).
  o.h[0][0] = pack_A4(kb0.data, 0);  // top-half, K-block 0
  o.h[1][0] = pack_A4(kb0.data, 4);  // bottom-half, K-block 0
  o.h[0][1] = pack_A4(kb1.data, 0);  // top-half, K-block 1
  o.h[1][1] = pack_A4(kb1.data, 4);  // bottom-half, K-block 1
}

static inline void premarshal_B_16x32(
    dpas_B_bf16_16x32 &o,
    const rt_base<bf16, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &kb0,
    const rt_base<bf16, ducks::rt_layout::row, ducks::rt_shape::rt_16x16> &kb1) {
  o.k[0] = pack_B8(kb0.data, 0);
  o.k[1] = pack_B8(kb1.data, 0);
}

// S(16x16) += A(16x32) * B(16x32)^T  (raw dpas, pre-marshalled operands).
static inline void mma_raw_ABt_16x32(dpas_C_f32_16x16 &c,
                                     const dpas_A_bf16_16x32 &a,
                                     const dpas_B_bf16_16x32 &b) {
  const auto op = intel::SPIRV_MMAOperands::SPIRV_MatrixABf16 |
                  intel::SPIRV_MMAOperands::SPIRV_MatrixBBf16;
  c.h[0] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a.h[0][0], b.k[0], c.h[0], op);
  c.h[0] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a.h[0][1], b.k[1], c.h[0], op);
  c.h[1] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a.h[1][0], b.k[0], c.h[1], op);
  c.h[1] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a.h[1][1], b.k[1], c.h[1], op);
}

/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` instruction.
 *
 * @param[out] d0 The first half of the output half_2 accumulator.
 * @param[out] d1 The second half of the output half_2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the half_2 accumulator matrix C.
 * @param[in] c1 The second half of the half_2 accumulator matrix C.
 */
static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
    /*
    DPCT1053:6: Migration of device assembly code is not supported.
    */
//    asm volatile(
//        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
//        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
//        "{%0, %1}, "
//        "{%2, %3, %4, %5}, "
//        "{%6, %7}, "
//        "{%8, %9};"
//
//        // D matrix
//        : "=r"(*(uint32_t *)(&d0)), "=r"(*(uint32_t *)(&d1))
//
//        // A matrix
//        : "r"(*(uint32_t *)(&a0)), "r"(*(uint32_t *)(&a1)),
//          "r"(*(uint32_t *)(&a2)), "r"(*(uint32_t *)(&a3)),
//
//          // B matrix
//          "r"(*(uint32_t *)(&b0)), "r"(*(uint32_t *)(&b1)),
//
//          // C matrix
//          "r"(*(uint32_t *)(&c0)), "r"(*(uint32_t *)(&c1)));
}


/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template <ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row, D_shape> &d,
                                    const rt_base<bf16,  ducks::rt_layout::row, A_shape> &a,
                                    const rt_base<bf16,  ducks::rt_layout::col, B_shape> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row, C_shape> &c) {

if constexpr(std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_32x32> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_32x32> ) {
    //        Matrix A (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |  A[2,0]  | A[2,1]   |
    // |  A[3,0]  | A[3,1]   |
    // |----------|----------|

    //        Matrix B (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // | B[1,0]   | B[1,1]   |
    // |----------|----------|

    // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

   hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[16], a.data[17], a.data[18], a.data[19],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
// d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[20], a.data[21], a.data[22], a.data[23],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

// d[2,0]

    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[24], a.data[25], a.data[26], a.data[27],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
// d[3,0]
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[28], a.data[29], a.data[30], a.data[31],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );

// d[0,1]

    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );
    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[16], a.data[17], a.data[18], a.data[19],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );

// d[1,1]
  hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );
    hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[20], a.data[21], a.data[22], a.data[23],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );

// d[2,1]
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[24], a.data[25], a.data[26], a.data[27],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );

// d[2,2]
    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );
    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[28], a.data[29], a.data[30], a.data[31],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );

} else if constexpr(std::is_same_v< A_shape,
                                     ducks::rt_shape::rt_16x16> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_16x16>)  {
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        // a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        // a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_16x32> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_32x16> ) {
    //        Matrix A (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |----------|----------|

    //        Matrix B (32x16)
    // |----------|
    // | Col1     |
    // |----------|
    // | B[0,0]   |
    // | B[1,0]   |
    // |----------|
 // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

     // d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_32x16> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_16x32> ) {
       //        Matrix A (32x16)
    // |----------|
    // | Col1     |
    // |----------|
    // |  A[0,0]  |
    // |  A[1,0]  |
    // |  A[2,0]  |
    // |  A[3,0]  |
    // |----------|

    //        Matrix B (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // |----------|----------|
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );

    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );

    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );
    hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );

    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );

    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_16x16> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_16x32> ) {
       //        Matrix A (16x16)
    // |----------|
    // | Col1     |
    // |----------|
    // |  A[0,0]  |
    // |  A[1,0]  |
    // |----------|

    //        Matrix B (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // |----------|----------|
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );

    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );

    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_16x32> && std::is_same_v<B_shape,
                                     ducks::rt_shape::rt_32x32>) {
       //        Matrix A (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |----------|----------|

    //        Matrix B (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // | B[1,0]   | B[1,1]   |
    // |----------|----------|

     // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

   hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
// d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

// d[0,1]
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );

// d[1,1]
  hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
    // sycl::ext::oneapi::experimental::printf("exited mma_ABaaaa\n");
    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_8x32> && std::is_same_v<B_shape,
                                     ducks::rt_shape::rt_32x32>) {
    //  A (8x32): 8 rows, K=32 (two 16-col K-blocks packed consecutively).
    //    a.data[0-3] = rows0-7 / K0-15 ,  a.data[4-7] = rows0-7 / K16-31
    //  B (32x32) col-major, same packing as the 16x32xAB branch.
    //  D (8x32): d.data[0-3] = rows0-7 / N0-15 ,  d.data[4-7] = rows0-7 / N16-31
    //  This is exactly the top-8-rows (d[0,0] and d[0,1]) of the 16x32xAB branch,
    //  with A's second K-block at data[4-7] (not data[8-11], only 8 packed exist).

    // d[0,0]: rows0-7, cols0-15
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    // d[0,1]: rows0-7, cols16-31
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    }
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3, row_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<half_2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<half_2, row_layout> accumulator matrix.
 */
static inline void mma_AB_base(
    rt_base<sycl::half, ducks::rt_layout::row> &d,
    const rt_base<sycl::half, ducks::rt_layout::row> &a,
    const rt_base<sycl::half, ducks::rt_layout::col> &b, // in col-major mode
    const rt_base<sycl::half, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template <ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row, D_shape> &d,
                                     const rt_base<bf16,  ducks::rt_layout::row, A_shape> &a,
                                     const rt_base<bf16,  ducks::rt_layout::row, B_shape> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row, C_shape> &c) {
// sycl::ext::oneapi::experimental::printf("enteredxxxxxxxx mma_ABt\n");
if constexpr(std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_32x32> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_32x32> ) {
    //        Matrix A (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |  A[2,0]  | A[2,1]   |
    // |  A[3,0]  | A[3,1]   |
    // |----------|----------|

    //        Matrix B (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // | B[1,0]   | B[1,1]   |
    // |----------|----------|

    // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

   hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[16], a.data[17], a.data[18], a.data[19],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
// d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[20], a.data[21], a.data[22], a.data[23],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

// d[2,0]

    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[24], a.data[25], a.data[26], a.data[27],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
// d[3,0]
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[28], a.data[29], a.data[30], a.data[31],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );

// d[0,1]

    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );
    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[16], a.data[17], a.data[18], a.data[19],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );

// d[1,1]
  hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );
    hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[20], a.data[21], a.data[22], a.data[23],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );

// d[2,1]
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[24], a.data[25], a.data[26], a.data[27],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );

// d[2,2]
    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );
    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[28], a.data[29], a.data[30], a.data[31],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );

} else if constexpr(std::is_same_v< A_shape,
                                     ducks::rt_shape::rt_16x16> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_16x16>)  {
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base \n");
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        // a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        // a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_16x32> && std::is_same_v< B_shape,
                                     ducks::rt_shape::rt_16x32> ) {
    //        Matrix A (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |----------|----------|

    //        Matrix B (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // |----------|----------|
 // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

     // d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base bbbbbbb  %f %f %f %f \n", sycl::ext::intel::math::bfloat162float(b.data[1].x()), sycl::ext::intel::math::bfloat162float(b.data[2].x()), sycl::ext::intel::math::bfloat162float(b.data[3].x()), sycl::ext::intel::math::bfloat162float((b.data[4].x())));
    //     sycl::ext::oneapi::experimental::printf("entered mma_ABt_base aaaaaa  %f %f %f %f %f %f %f %f \n", sycl::ext::intel::math::bfloat162float(a.data[1].x()), sycl::ext::intel::math::bfloat162float(a.data[2].x()), sycl::ext::intel::math::bfloat162float(a.data[3].x()), sycl::ext::intel::math::bfloat162float((a.data[4].x())),
    //     sycl::ext::intel::math::bfloat162float(a.data[5].x()), sycl::ext::intel::math::bfloat162float(a.data[6].x()), sycl::ext::intel::math::bfloat162float(a.data[7].x()), sycl::ext::intel::math::bfloat162float((a.data[8].x())));
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base ccccc  %f %f %f %f \n", c.data[16].x(), c.data[17].x(), c.data[18].x(), c.data[19].x());

    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa1111 %f %f %f %f \n", d.data[0].x(), d.data[1].x(), d.data[2].x(), d.data[3].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa2222 %f %f %f %f \n", d.data[4].x(), d.data[5].x(), d.data[6].x(), d.data[7].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa3333 %f %f %f %f \n", d.data[8].x(), d.data[9].x(), d.data[10].x(), d.data[11].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa4444 %f %f %f %f \n", d.data[12].x(), d.data[13].x(), d.data[14].x(), d.data[15].x());
    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_32x16> && std::is_same_v<B_shape,
                                     ducks::rt_shape::rt_32x16>) {
    //        Matrix A (32x16)
    // |----------|
    // | Col1     |
    // |----------|
    // |  A[0,0]  |
    // |  A[1,0]  |
    // |  A[2,0]  |
    // |  A[3,0]  |
    // |----------|

    //        Matrix B (32x16)
    // |----------|
    // | Col1     |
    // |----------|
    // | B[0,0]   |
    // | B[1,0]   |
    // |----------|
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );

    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );

    hmma81616(
        d.data[16], d.data[17], d.data[18], d.data[19],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[16], c.data[17], c.data[18], c.data[19]
    );
    hmma81616(
        d.data[20], d.data[21], d.data[22], d.data[23],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[20], c.data[21], c.data[22], c.data[23]
    );
    hmma81616(
        d.data[24], d.data[25], d.data[26], d.data[27],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[24], c.data[25], c.data[26], c.data[27]
    );

    hmma81616(
        d.data[28], d.data[29], d.data[30], d.data[31],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[28], c.data[29], c.data[30], c.data[31]
    );
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base bbbbbbb  %f %f %f %f \n", sycl::ext::intel::math::bfloat162float(b.data[1].x()), sycl::ext::intel::math::bfloat162float(b.data[2].x()), sycl::ext::intel::math::bfloat162float(b.data[3].x()), sycl::ext::intel::math::bfloat162float((b.data[4].x())));
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base ccccc  %f %f %f %f \n", c.data[16].x(), c.data[17].x(), c.data[18].x(), c.data[19].x());

    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa1111 %f %f %f %f \n", d.data[0].x(), d.data[1].x(), d.data[2].x(), d.data[3].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa2222 %f %f %f %f \n", d.data[4].x(), d.data[5].x(), d.data[6].x(), d.data[7].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa3333 %f %f %f %f \n", d.data[8].x(), d.data[9].x(), d.data[10].x(), d.data[11].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa4444 %f %f %f %f \n", d.data[12].x(), d.data[13].x(), d.data[14].x(), d.data[15].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyaaaa5555 %f %f %f %f \n", d.data[16].x(), d.data[17].x(), d.data[18].x(), d.data[19].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy %f %f %f %f \n", d.data[20].x(), d.data[21].x(), d.data[22].x(), d.data[23].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyyxxx %f %f %f %f \n", d.data[24].x(), d.data[25].x(), d.data[26].x(), d.data[27].x());
        // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy %f %f %f %f \n", d.data[20], d.data[13], d.data[16], d.data[17]);

    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_16x32> && std::is_same_v<B_shape,
                                     ducks::rt_shape::rt_32x32>) {
    //        Matrix A (16x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // |  A[0,0]  | A[0,1]   |
    // |  A[1,0]  | A[1,1]   |
    // |----------|----------|

    //        Matrix B (32x32)
    // |----------|----------|
    // | Col1     | Col2     |
    // |----------|----------|
    // | B[0,0]   | B[0,1]   |
    // | B[1,0]   | B[1,1]   |
    // |----------|----------|

     // d[0,0]
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

   hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        // b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
// d[1,0]
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        // b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

// d[0,1]
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        // b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );
    hmma81616(
        d.data[8], d.data[9], d.data[10], d.data[11],
        a.data[8], a.data[9], a.data[10], a.data[11],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[8], c.data[9], c.data[10], c.data[11]
    );

// d[1,1]
  hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        // b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
    hmma81616(
        d.data[12], d.data[13], d.data[14], d.data[15],
        a.data[12], a.data[13], a.data[14], a.data[15],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[12], c.data[13], c.data[14], c.data[15]
    );
        // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base ccccbbbbbbb  %f %f %f %f  %f %f %f  %f %f %f \n", sycl::ext::intel::math::bfloat162float(b.data[1].x()), sycl::ext::intel::math::bfloat162float(b.data[2].x()), sycl::ext::intel::math::bfloat162float(b.data[3].x()), sycl::ext::intel::math::bfloat162float((b.data[4].x())),

        // sycl::ext::intel::math::bfloat162float(b.data[5].x()), sycl::ext::intel::math::bfloat162float(b.data[7].x()), sycl::ext::intel::math::bfloat162float(b.data[9].x()), sycl::ext::intel::math::bfloat162float(b.data[5].y()), sycl::ext::intel::math::bfloat162float(b.data[7].y()), sycl::ext::intel::math::bfloat162float(b.data[9].y()));

        // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base ccccaaaaaa %f %f %f %f %f %f %f %f %f %f %f\n", sycl::ext::intel::math::bfloat162float(a.data[1].x()), sycl::ext::intel::math::bfloat162float(a.data[1].y()), sycl::ext::intel::math::bfloat162float(a.data[2].x()), sycl::ext::intel::math::bfloat162float(a.data[3].x()), sycl::ext::intel::math::bfloat162float((a.data[4].x())),
        // sycl::ext::intel::math::bfloat162float(a.data[5].x()), sycl::ext::intel::math::bfloat162float(a.data[6].x()), sycl::ext::intel::math::bfloat162float(a.data[7].x()), sycl::ext::intel::math::bfloat162float((a.data[8].x())), sycl::ext::intel::math::bfloat162float((a.data[8].y())), sycl::ext::intel::math::bfloat162float((a.data[9].x())));

    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base ccccc  %f %f %f %f \n", c.data[16].x(), c.data[17].x(), c.data[18].x(), c.data[19].x());

    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy1111 %f %f %f %f \n", d.data[0].x(), d.data[1].x(), d.data[2].x(), d.data[3].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy2222 %f %f %f %f \n", d.data[4].x(), d.data[5].x(), d.data[6].x(), d.data[7].x());
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy3333  %f %f %f %f \n", d.data[8].x(), d.data[9].x(), d.data[10].x(), d.data[11].x());
    //     sycl::ext::oneapi::experimental::printf("entered mma_ABt_base yyyy4444 %f %f %f %f \n", d.data[12].x(), d.data[13].x(), d.data[14].x(), d.data[15].x());

    } else if constexpr (std::is_same_v<A_shape,
                                     ducks::rt_shape::rt_8x32> && std::is_same_v<B_shape,
                                     ducks::rt_shape::rt_32x32>) {
    //  A (8x32) row-major, B (32x32) row-major used transposed (A @ B^T).
    //  8 rows only -> the top-8-rows (d[0,0], d[0,1]) of the 16x32xABt branch,
    //  with A's second K-block at data[4-7] (not data[8-11]) and d[0,1] output
    //  at d.data[4-7] (not d.data[8-11]) since only 8 packed elements exist.
    //  D (8x32): d.data[0-3] = rows0-7 / N0-15 , d.data[4-7] = rows0-7 / N16-31.

    // d[0,0]: rows0-7, cols0-15
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[8], b.data[9], b.data[10], b.data[11], b.data[12], b.data[13], b.data[14], b.data[15],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );
    // d[0,1]: rows0-7, cols16-31
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[16], b.data[17], b.data[18], b.data[19], b.data[20], b.data[21], b.data[22], b.data[23],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[24], b.data[25], b.data[26], b.data[27], b.data[28], b.data[29], b.data[30], b.data[31],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );

    }
}



/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::col> &a,
                                     const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    // hmma16816(
    //     d.data[0], d.data[1],
    //     a.data[0], a.data[1], a.data[2], a.data[3],
    //     b.data[0], b.data[2],
    //     c.data[0], c.data[1]
    // );
    // hmma16816(
    //     d.data[2], d.data[3],
    //     a.data[0], a.data[1], a.data[2], a.data[3],
    //     b.data[1], b.data[3],
    //     c.data[2], c.data[3]
    // );
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]
    );

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]
    );
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                 const rt_base<bf16, ducks::rt_layout::col> &a,
                                 const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                 const rt_base<float, ducks::rt_layout::row> &c)
{
    hmma81616(
        d.data[0], d.data[1], d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[0], c.data[1], c.data[2], c.data[3]);

    hmma81616(
        d.data[4], d.data[5], d.data[6], d.data[7],
        a.data[4], a.data[5], a.data[6], a.data[7],
        b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6], b.data[7],
        c.data[4], c.data[5], c.data[6], c.data[7]);
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    // sycl::ext::oneapi::experimental::printf("entered mma_ABt\n");
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert((std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, bf16> &&
                   std::is_same_v<typename B::T, bf16> &&
                   std::is_same_v<typename C::T, float>) ||
                  (std::is_same_v<typename D::T, sycl::half> &&
                   std::is_same_v<typename A::T, sycl::half> &&
                   std::is_same_v<typename B::T, sycl::half> &&
                   std::is_same_v<typename C::T, sycl::half>) ||
                  (std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, fp8e4m3> &&
                   std::is_same_v<typename B::T, fp8e4m3> &&
                   std::is_same_v<typename C::T, float>));
#else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    // sycl::ext::oneapi::experimental::printf("xxxxxxxxzzzzz %d %d %d  \n", D::height, D::width, A::width);
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert((std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, bf16> &&
                   std::is_same_v<typename B::T, bf16> &&
                   std::is_same_v<typename C::T, float>) ||
                  (std::is_same_v<typename D::T, sycl::half> &&
                   std::is_same_v<typename A::T, sycl::half> &&
                   std::is_same_v<typename B::T, sycl::half> &&
                   std::is_same_v<typename C::T, sycl::half>) ||
                  (std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, fp8e4m3> &&
                   std::is_same_v<typename B::T, fp8e4m3> &&
                   std::is_same_v<typename C::T, float>));
#else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert((std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, bf16> &&
                   std::is_same_v<typename B::T, bf16> &&
                   std::is_same_v<typename C::T, float>) ||
                  (std::is_same_v<typename D::T, sycl::half> &&
                   std::is_same_v<typename A::T, sycl::half> &&
                   std::is_same_v<typename B::T, sycl::half> &&
                   std::is_same_v<typename C::T, sycl::half>) ||
                  (std::is_same_v<typename D::T, float> &&
                   std::is_same_v<typename A::T, fp8e4m3> &&
                   std::is_same_v<typename B::T, fp8e4m3> &&
                   std::is_same_v<typename C::T, float>));
#else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<int trans_A, int trans_B, ducks::rt::all D, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
static inline void mma(D &d,
                                  const A &a,
                                  const B &b,
                                  const C &c) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
}
template<int trans_A, int trans_B, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
static inline C mma(const A &a,
                               const B &b,
                               const C &c) {
    C d;
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
    return d;
}

}