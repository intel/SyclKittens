/**
 * @file gqa_forward.dp.cpp
 * @brief SyclKittens GQA/MHA Forward Attention: the single "best-of-best"
 *        adaptive kernel (SYCL / Intel PVC).
 *
 * This is the consolidated production forward attention. It replaces the zoo of
 * per-config experimental modules (v2/v16 × wg64/128/256 × seq_q8/16 × causal/
 * non-causal) with ONE .so whose `dispatch_fwd` selects the empirically-best
 * kernel instantiation per problem shape at runtime.
 *
 * Compute core is v16 (== v7 Q-hoist pipeline + LPT launch ordering):
 *   - Q preloaded ONCE into registers before the KV loop (v2 reloaded 4×/KV tile).
 *   - Grid (Z,H,B) with in-kernel reversal so the heaviest causal Q-blocks
 *     dispatch first (longest-processing-time-first => short tail wave).
 *   - Online softmax on S[seq_q×32], SPLIT_D output accumulation for D>64.
 *
 * Two template axes are exposed so a SINGLE translation unit can hold every
 * winning variant (co-loading separate extension .so files is impossible — their SYCL kernel
 * names collide, an ODR clash that silently corrupts results / segfaults):
 *   - SEQ_Q ∈ {8,16}: seq_q=8 halves the rows/tile => DOUBLES the number of
 *     query-tiles (occupancy) at the SAME HBM traffic. This is the fix for the
 *     small-N / low-batch "occupancy wall". seq_q=16 stays byte-
 *     identical to v16 for the large-N MMA-bound regime.
 *   - WG ∈ {128,256}: workgroup size (subgroups = WG/16).
 *
 * Selection policy (measured). GQA is now a full 2-D (N, B, causal) table;
 * MHA stays (N, causal).
 * seq_q=8 (occupancy) wins small (N, B); the crossover to seq_q=16 (MMA-bound)
 * moves to smaller N as batch grows, and the winning WG grows with B. See the
 * annotated table above `choose_fwd()`.
 *
 * NOTE: causal seq_q=8 IS dispatched — make_causal() handles the rt_8x32
 * register shape (part∈{0,1,2,3}) correctly (max_abs ≤ 0.014). Non-causal
 * seq_q=8 is fully correct (rt_8x32 mma/load/store/reduce validated).
 */

#include "kittens.dp.hpp"
#include "ops/warp/register/tile/conversions.dp.hpp"
#include "pyutils/torch_helpers.dp.hpp"

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

using namespace kittens;

// ============================================================
// Configuration — compile-time constants (overridable via -D)
// ============================================================
#ifndef ATTN_H_VALUE
constexpr int ATTN_H = 32;      // Q heads   (GQA default: Llama-3-8B)
#else
constexpr int ATTN_H = ATTN_H_VALUE;
#endif

#ifndef ATTN_H_KV_VALUE
constexpr int ATTN_H_KV = 8;    // KV heads
#else
constexpr int ATTN_H_KV = ATTN_H_KV_VALUE;
#endif

#ifndef ATTN_D_VALUE
constexpr int ATTN_D = 128;     // head dim
#else
constexpr int ATTN_D = ATTN_D_VALUE;
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
static_assert(ATTN_H % ATTN_H_KV == 0, "Q heads must be divisible by KV heads");

template <int D> using global_layout      = gl<bf16,  -1, -1, -1, D>;
template <int D> using global_layout_out  = gl<float, -1, -1, -1, D>;   // legacy f32 out (training / bwd recompute)
template <int D> using global_layout_outh = gl<bf16,  -1, -1, -1, D>;   // E2E inference: bf16 out (store converts in-register, no post-cast)

// ============================================================
// Per-(WG, seq_q) tile configuration & register-tile types
// ============================================================
template <int D, int WG, int SEQ_Q>
struct fwd_cfg {
  static constexpr int sg_size          = 16;
  static constexpr int num_sg           = WG / sg_size;
  static constexpr int kv_stride        = 32;
  static constexpr int PF               = 1;
  static constexpr int inner_loop       = D / kv_stride;          // 4 for D=128
  static constexpr int second_index_start = D / 2 / kv_stride;    // 2
  static constexpr bool SPLIT_D         = (D > 64);
  static constexpr int rows_per_wg      = num_sg * SEQ_Q;

  static_assert(SEQ_Q == 16 || SEQ_Q == 8, "SEQ_Q must be 16 or 8");

  // 8-row query tiles (rt_8x32) double the number of concurrent query-tiles
  // (occupancy); seq_q=16 (rt_16x32) is byte-identical to the original v16.
  using q_rt_shape = std::conditional_t<(SEQ_Q == 8),
                                        kittens::ducks::rt_shape::rt_8x32,
                                        kittens::ducks::rt_shape::rt_16x32>;

  using q_chunk_t = rt<bf16,  SEQ_Q,     kv_stride, row_l, q_rt_shape>;
  using k_chunk_t = rt<bf16,  kv_stride, kv_stride, row_l, kittens::ducks::rt_shape::rt_32x32>;
  using attn_t    = rt<float, SEQ_Q,     kv_stride, row_l, q_rt_shape>;
  using v_half_t  = rt<bf16,  kv_stride, D / (SPLIT_D ? 2 : 1), col_l,
                       kittens::ducks::rt_shape::rt_32x32>;
  using o_half_t  = rt<float, SEQ_Q,     D / (SPLIT_D ? 2 : 1), row_l, q_rt_shape>;
  using p_mma_t   = rt<bf16,  SEQ_Q,     kv_stride, row_l, q_rt_shape>;
};

// ============================================================
// NOTE on software fences (deliberately absent):
// The original CUDA-derived kernel sprinkled fence_sw() (an Xe software-
// scoreboard DRAIN) after every load->DPAS and around the softmax reductions.
// On PVC these are unnecessary: IGC's SWSB inserts precise per-dependency SBID
// tokens for load->DPAS and DPAS->consumer hazards automatically (the GEMM
// kernel, kernels/gemm/bf16_gemm.dp.cpp, runs its load->DPAS inner loop
// entirely fence-free). An ablation (4 seeds x 20 repeats, N=512..4096,
// causal+non-causal) showed removing all 7 fences is
// bit-identical in output (max_abs unchanged to 5 d.p.) and FASTER: the
// load->MMA fences alone cost up to ~15% at large-N causal. Do NOT re-add them.
// ============================================================

// ============================================================
// Kernel (v16 compute core, parametrised on WG / SEQ_Q / CAUSAL)
// ============================================================
template <int D, int WG, int SEQ_Q, bool CAUSAL_T, typename OutGL>
SYCL_EXTERNAL void attend_ker_gqa_fwd(
        global_layout<D> Qg, global_layout<D> Kg,
        global_layout<D> Vg, OutGL Og,
        float* __restrict__ L_out, int H_total, int N_seq) {
  using cfg = fwd_cfg<D, WG, SEQ_Q>;
  constexpr int num_sg    = cfg::num_sg;
  constexpr int kv_stride = cfg::kv_stride;
  constexpr int PF        = cfg::PF;
  constexpr int inner_loop = cfg::inner_loop;
  constexpr int second_index_start = cfg::second_index_start;
  constexpr bool SPLIT_D  = cfg::SPLIT_D;
  using q_chunk_t = typename cfg::q_chunk_t;
  using k_chunk_t = typename cfg::k_chunk_t;
  using attn_t    = typename cfg::attn_t;
  using v_half_t  = typename cfg::v_half_t;
  using o_half_t  = typename cfg::o_half_t;
  using p_mma_t   = typename cfg::p_mma_t;

  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int sg_id       = item_ct1.get_local_id(2) / 16;
  // v16/LPT: heaviest Q-blocks (lowest linear WG id) dispatch first.
  const int Z           = item_ct1.get_group_range(0);
  const int z_rank      = item_ct1.get_group(0);
  const int head_id     = item_ct1.get_group(1);
  const int batch_id    = item_ct1.get_group(2);
  const int start_q_seq = (Z - 1 - z_rank) * num_sg + sg_id;

  if (start_q_seq * SEQ_Q >= N_seq) return;

  const int kv_head_id = head_id / GROUP_SIZE;

  // ── Accumulator and softmax state ──────────────────────────
  o_half_t o_reg_1, o_reg_2;
  zero(o_reg_1);
  if constexpr (SPLIT_D) zero(o_reg_2);

  typename attn_t::col_vec max_vec_last, max_vec, norm_vec;
  zero(norm_vec);
  zero(max_vec);

  const float scale = (1.0f / __builtin_sqrtf(static_cast<float>(D))) * 1.44269504089f;

  // ── Pre-load Q into registers (invariant across all KV tiles) ──
  q_chunk_t q_preloaded[inner_loop];
  #pragma unroll
  for (int idx = 0; idx < inner_loop; idx++) {
    kittens::load_part<1>(q_preloaded[idx], Qg,
                          {batch_id, start_q_seq, head_id, 0}, {0, idx});
  }

  // ── KV tile count (causal: only tiles up to current Q block) ──
  const int num_kv_tiles = Kg.depth() / kv_stride;
  int loop_kv_tiles;
  if constexpr (CAUSAL_T) {
    constexpr int ratio = kv_stride / SEQ_Q;   // 2 (seq_q=16) or 4 (seq_q=8)
    const int max_kv = start_q_seq / ratio + 1;
    loop_kv_tiles = (max_kv < num_kv_tiles) ? max_kv : num_kv_tiles;
  } else {
    loop_kv_tiles = num_kv_tiles;
  }

  // ── Prefetch first K tile ──────────────────────────────────
  k_chunk_t k_pf;
  #pragma unroll
  for (int idx = 0; idx < PF && idx < loop_kv_tiles; idx++) {
    #pragma unroll
    for (int c = 0; c < inner_loop; c++) {
      kittens::prefetch_load_part<1>(k_pf, Kg,
                                     {batch_id, idx, kv_head_id, 0}, {0, c});
    }
  }

  int prefetch = PF;

  // ══════════════════════════════════════════════════════════
  // Main KV loop
  // ══════════════════════════════════════════════════════════
  #pragma unroll
  for (int i = 0; i < loop_kv_tiles; i++, prefetch++) {

    attn_t att_block;
    zero(att_block);
    max_vec_last = max_vec;

    // ── Step 1: S[seq_q×32] = Q[seq_q×D] × K^T[D×32] ──────
    k_chunk_t k_reg;
    #pragma unroll
    for (int c = 0; c < inner_loop; c++) {
      kittens::load_transpose_part<1>(k_reg, Kg,
                                      {batch_id, i, kv_head_id, 0}, {0, c});
      if (prefetch < loop_kv_tiles)
        kittens::prefetch_load_part<1>(k_reg, Kg,
                                       {batch_id, prefetch, kv_head_id, 0}, {0, c});
      mma<transpose::N, transpose::T>(att_block, q_preloaded[c], k_reg, att_block);
    }

    // ── Step 2: Causal mask ────────────────────────────────
    if constexpr (CAUSAL_T) {
      constexpr int ratio = kv_stride / SEQ_Q;
      const int q_blk = start_q_seq;
      const int k_blk = i;
      if (k_blk > q_blk / ratio) {
        neg_infty(att_block);
      } else if (k_blk == q_blk / ratio) {
        // ratio diagonal sub-blocks per KV tile (2 for seq_q=16, 4 for seq_q=8).
        int part = start_q_seq % ratio;
        make_causal(att_block, att_block,
                    kittens::base_types::constants<float>::neg_infty(), part);
      }
    }

    // ── Step 3: Online softmax ────────────────────────────
    max_vec = max<axis::COL>(att_block, max_vec);
    auto max_val = max_vec * scale;
    max_vec_last = exp2(max_vec_last * scale - max_val);

    o_reg_1 *= max_vec_last;
    if constexpr (SPLIT_D) o_reg_2 *= max_vec_last;
    norm_vec *= max_vec_last;

    att_block = exp2(att_block * scale - max_val);
    norm_vec = sum<axis::COL>(att_block, norm_vec);

    // ── Step 4: O += P × V ────────────────────────────────
    p_mma_t p_mma;
    p_mma = att_block;  // float→bf16 conversion

    v_half_t v_reg;
    if constexpr (SPLIT_D) {
      kittens::load_part<1>(v_reg, Vg, {batch_id, i, kv_head_id, 0}, {0, 0});
      mma<transpose::N, transpose::N>(o_reg_1, p_mma, v_reg, o_reg_1);
      if (prefetch < loop_kv_tiles)
        kittens::prefetch_load_part<1>(v_reg, Vg,
                                       {batch_id, prefetch, kv_head_id, 0}, {0, 0});
      kittens::load_part<1>(v_reg, Vg,
                            {batch_id, i, kv_head_id, 0}, {0, second_index_start});
      mma<transpose::N, transpose::N>(o_reg_2, p_mma, v_reg, o_reg_2);
      if (prefetch < loop_kv_tiles)
        kittens::prefetch_load_part<1>(v_reg, Vg,
                                       {batch_id, prefetch, kv_head_id, 0},
                                       {0, second_index_start});
    } else {
      kittens::load<1>(v_reg, Vg, {batch_id, i, kv_head_id, 0});
      mma<transpose::N, transpose::N>(o_reg_1, p_mma, v_reg, o_reg_1);
      if (prefetch < loop_kv_tiles)
        kittens::prefetch_load<1>(v_reg, Vg, {batch_id, prefetch, kv_head_id, 0});
    }
  }

  // ── Final normalisation and store ──────────────────────────
  o_reg_1 *= inv(norm_vec);
  if constexpr (SPLIT_D) {
    o_reg_2 *= inv(norm_vec);
    kittens::store_part<1>(Og, o_reg_1, {batch_id, start_q_seq, head_id, 0}, {0, 0});
    kittens::store_part<1>(Og, o_reg_2, {batch_id, start_q_seq, head_id, 0},
                           {0, second_index_start});
  } else {
    kittens::store<1>(Og, o_reg_1, {batch_id, start_q_seq, head_id, 0});
  }

  // ── L_vec = logsumexp for backward ────────────────────────
  {
    auto sg  = item_ct1.get_sub_group();
    int lane = sg.get_local_linear_id();
    const float inv_sqrt_D_val = 1.0f / __builtin_sqrtf(static_cast<float>(D));
    float L_val = max_vec.data[0][0] * inv_sqrt_D_val
                + sycl::log(norm_vec.data[0][0]);
    long L_idx = (long)batch_id * H_total * N_seq
               + (long)head_id  * N_seq
               + (long)start_q_seq * SEQ_Q + lane;
    if (lane < SEQ_Q)   // only the first seq_q lanes map to valid query rows
      L_out[L_idx] = L_val;
  }
}

// ============================================================
// Launch helper — one instantiation per (WG, SEQ_Q, CAUSAL)
// ============================================================
// Unique SYCL kernel-name tag per variant (WG, SEQ_Q, CAUSAL) so all winning
// instantiations coexist in this single .so without an ODR symbol clash.
template <int WG, int SEQ_Q, bool CAUSAL_T, bool OUT_BF16> class gqa_fwd_kernel;

// ============================================================
// PyTorch dispatch — adaptive best-config selection
// ============================================================
struct fwd_choice { int wg; int seq_q; };

// Grid-derived best (WG, seq_q). MHA is keyed on (N, causal); GQA is keyed on
// (N, B, causal) — a full 2-D table (below). B matters because seq_q=8
// (occupancy) only pays while the launch is thread-starved: as batch grows the
// grid already fills the machine, so the seq_q=8->16 crossover moves to smaller
// N and the winning WG grows with B.
//
// MHA (ATTN_H==ATTN_H_KV) owns its OWN champions — a dedicated full grid search
// (B=4 H=16 D=128, fence-free, wg{64,128,256} x seq_q{8,16} x N{512..8192},
// single tile). B is ignored on the MHA path
// (its table was taken at the throughput batch B=4); kept explicit so a GQA
// re-tune cannot silently regress MHA.
//
// GQA 2-D table — Mistral-7B dims H=32 HKV=8 D=128,
// B{1,2,4,8} x N{256,512,1024,2048} x wg{64,128,256} x seq_q{8,16},
// 24 reps/cell, CoV<=7%. Winner (wg/sq)
// per (N,B), mapped onto the instantiated variants (causal sq8 always routes
// wg128; non-causal sq16 always routes wg256):
//   causal   N=256 : sq8            all B
//            N=512 : sq8 (B<=2)   / sq16 wg128 (B>=4)
//            N=1024: sq8 (B=1)    / sq16 wg64 (B=2, -16% vs wg256)
//                                 / wg128 (B=4) / wg256 (B>=8)
//            N=2048: sq16 wg128 (B=1, -6% vs wg256) / wg256 (B>=2)
//   non-caus N=256 : sq8 wg256 (B<=4)  / sq16 wg256 (B>=8)
//            N=512 : sq8 wg128 (B<=2)  / sq16 wg256 (B>=4)
//            N=1024: sq8 wg256 (B=1)   / sq16 wg256 (B>=2)
//            N=2048: sq16 wg256        all B
//   N>2048 : sq16 wg256 (long-context champion).
static inline fwd_choice choose_fwd(int N, int B, bool causal, bool bf16_out) {
  if constexpr (ATTN_H == ATTN_H_KV) {
    // ── MHA champions — best-of-best per OUTPUT DTYPE (B-agnostic, B=4) ─────
    // From a WG{64,128,256}×seq_q{8,16}×pf{1,2,3,4} grid search
    // (PF=1 optimal everywhere, so pf is fixed to the kernel default). Only
    // causal N=2048 diverges by dtype: the bf16 store epilogue spills the
    // seq_q=16 accumulator at wg128/wg256, so wg64 wins; f32 has no such
    // spill and wg256 wins. All other (N,causal) cells agree across dtypes.
    (void)B;
    if (causal) {
      if (N <= 512)  return fwd_choice{128, 8};
      if (N <= 1024) return fwd_choice{128, 16};
      if (N <= 2048) return bf16_out ? fwd_choice{64,  16}
                                     : fwd_choice{256, 16};
      return fwd_choice{256, 16};                 // N>=4096: {256,16} both dtypes
    }
    if (N <= 512) return fwd_choice{256, 8};      // non-causal: {256,x} both dtypes
    return fwd_choice{256, 16};                    // N>=1024: {256,16} both dtypes
  } else {
    // ── GQA champions — 2-D (N, B) policy, per OUTPUT DTYPE ────────────────
    // From a WG{64,128,256} × seq_q{8,16} × pf{1,2,3,4} × out{bf16,f32}
    // grid search (12 reps/shape). PF=1 is optimal for EVERY shape (pf>1 never
    // beats pf1 by more than the 1-3% CoV noise), so pf is fixed to the kernel
    // default. bf16 and f32 winners diverge on 22 shapes (the bf16 store
    // epilogue shifts the register-pressure balance), so we keep two tables.
    // N>2048 is uniformly wg256/sq16 (both dtypes, both causal, all B).
    if (N > 2048) return fwd_choice{256, 16};
    if (bf16_out) {
      if (causal) {
        if (N <= 256)  return (B <= 1) ? fwd_choice{256, 8}
                     : (B <= 2)        ? fwd_choice{128, 8}
                                       : fwd_choice{64,  8};   // B>=4
        if (N <= 512)  return (B <= 1) ? fwd_choice{64,  8}
                     : (B <= 2)        ? fwd_choice{128, 8}
                     : (B <= 4)        ? fwd_choice{256, 8}
                                       : fwd_choice{128, 16};  // B>=8
        if (N <= 1024) return (B <= 1) ? fwd_choice{64,  8}
                                       : fwd_choice{128, 16};  // B>=2
        /* N<=2048 */  return (B <= 1) ? fwd_choice{64,  16}
                     : (B <= 2)        ? fwd_choice{128, 16}
                                       : fwd_choice{256, 16};  // B>=4
      }
      // bf16 non-causal
      if (N <= 256)  return fwd_choice{256, 8};                 // all B
      if (N <= 512)  return (B <= 2) ? fwd_choice{256, 8}
                                     : fwd_choice{256, 16};     // B>=4
      if (N <= 1024) return (B <= 1) ? fwd_choice{64,  16}
                                     : fwd_choice{256, 16};     // B>=2
      return fwd_choice{256, 16};                               // N<=2048, all B
    }
    // f32 output
    if (causal) {
      if (N <= 256)  return (B <= 1) ? fwd_choice{256, 8}
                   : (B <= 2)        ? fwd_choice{128, 8}
                   : (B <= 4)        ? fwd_choice{256, 8}
                                     : fwd_choice{64,  16};     // B>=8
      if (N <= 512)  return (B <= 1) ? fwd_choice{64,  8}
                   : (B <= 2)        ? fwd_choice{128, 8}
                   : (B <= 4)        ? fwd_choice{64,  16}
                                     : fwd_choice{128, 16};     // B>=8
      if (N <= 1024) return (B <= 1) ? fwd_choice{64,  8}
                   : (B <= 2)        ? fwd_choice{128, 16}
                   : (B <= 4)        ? fwd_choice{128, 16}
                                     : fwd_choice{256, 16};     // B>=8
      /* N<=2048 */  return (B <= 1) ? fwd_choice{128, 16}
                                     : fwd_choice{256, 16};     // B>=2
    }
    // f32 non-causal
    if (N <= 256)  return (B <= 2) ? ((B <= 1) ? fwd_choice{256, 8}
                                               : fwd_choice{128, 8})
                                   : fwd_choice{256, 8};        // B>=4
    if (N <= 512)  return (B <= 1) ? fwd_choice{128, 8}
                 : (B <= 2)        ? fwd_choice{64,  8}
                                   : fwd_choice{256, 16};       // B>=4
    if (N <= 1024) return (B <= 1) ? fwd_choice{256, 8}
                                   : fwd_choice{256, 16};       // B>=2
    return fwd_choice{256, 16};                                 // N<=2048, all B
  }
}

template <int WG, int SEQ_Q, bool CAUSAL_T, typename OutGL>
static void submit_fwd(sycl::queue &queue,
                       global_layout<ATTN_D> Qg, global_layout<ATTN_D> Kg,
                       global_layout<ATTN_D> Vg, OutGL Og,
                       float *d_L, int B, int H, int N) {
  constexpr int rows_per_wg = fwd_cfg<ATTN_D, WG, SEQ_Q>::rows_per_wg;
  constexpr int sg_size     = fwd_cfg<ATTN_D, WG, SEQ_Q>::sg_size;
  sycl::range<3> grid((N + rows_per_wg - 1) / rows_per_wg, H, B);
  sycl::range<3> block(1, 1, (WG + sg_size - 1) / sg_size * 16);
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::sub_group_size<16>,
      sycl::ext::oneapi::experimental::work_group_scratch_size(
          kittens::MAX_SHARED_MEMORY)};
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<gqa_fwd_kernel<WG, SEQ_Q, CAUSAL_T,
             std::is_same_v<typename OutGL::dtype, bf16>>>(
        sycl::nd_range<3>(grid * block, block), exp_props,
        [=](sycl::nd_item<3> item) {
          attend_ker_gqa_fwd<ATTN_D, WG, SEQ_Q, CAUSAL_T, OutGL>(Qg, Kg, Vg, Og, d_L, H, N);
        });
  });
}

// Shared routing: choose_fwd(N,B,causal) -> winning (WG, seq_q) instantiation.
// Templated on the output global-layout type so the SAME policy serves both the
// legacy f32-out path (training / bwd) and the E2E bf16-out fast path (the store
// epilogue converts float accumulators -> bf16 in-register, eliminating the
// per-layer f32->bf16 elementwise cast + HBM round-trip on the composed path).
template <typename OutGL>
static inline void route_fwd(sycl::queue &queue,
                             global_layout<ATTN_D> Qg, global_layout<ATTN_D> Kg,
                             global_layout<ATTN_D> Vg, OutGL Og, float *d_L,
                             int B, int H, int N, bool is_causal) {
  constexpr bool bf16_out = std::is_same_v<typename OutGL::dtype, bf16>;
  const fwd_choice c = choose_fwd(N, B, is_causal, bf16_out);
  // Dispatch the chosen (wg, seq_q) exactly — every combo the two dtype tables
  // can return is instantiated here (pf fixed to the kernel default = 1).
#define FWD_ROUTE(CA)                                                          \
  do {                                                                         \
    if (c.seq_q == 8) {                                                        \
      if (c.wg == 64)       submit_fwd<64,  8, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
      else if (c.wg == 128) submit_fwd<128, 8, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
      else                  submit_fwd<256, 8, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
    } else {                                                                   \
      if (c.wg == 64)       submit_fwd<64,  16, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
      else if (c.wg == 128) submit_fwd<128, 16, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
      else                  submit_fwd<256, 16, CA>(queue, Qg, Kg, Vg, Og, d_L, B, H, N); \
    }                                                                          \
  } while (0)
  if (is_causal) FWD_ROUTE(true);
  else           FWD_ROUTE(false);
#undef FWD_ROUTE
}

void dispatch_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                  torch::Tensor O, torch::Tensor L, bool is_causal) {
  CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
  CHECK_INPUT(O); CHECK_INPUT(L);

  const int B    = Q.size(0);
  const int N    = Q.size(1);
  const int H    = Q.size(2);
  const int D    = Q.size(3);
  const int H_KV = K.size(2);

  TORCH_CHECK(D    == ATTN_D,    "Head dim must be ", ATTN_D);
  TORCH_CHECK(H    == ATTN_H,    "Q heads must be ", ATTN_H, " (compiled)");
  TORCH_CHECK(H_KV == ATTN_H_KV, "KV heads must be ", ATTN_H_KV, " (compiled)");
  TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
  TORCH_CHECK(O.dtype() == torch::kFloat32 || O.dtype() == torch::kBFloat16,
              "O must be float32 (legacy) or bfloat16 (fast path)");
  TORCH_CHECK(L.dtype() == torch::kFloat32,  "L must be float32");

  auto stream = c10::xpu::getCurrentXPUStream(Q.device().index());
  auto &queue = stream.queue();
  queue.wait();

  bf16  *d_Q = reinterpret_cast<bf16*>(Q.data_ptr<c10::BFloat16>());
  bf16  *d_K = reinterpret_cast<bf16*>(K.data_ptr<c10::BFloat16>());
  bf16  *d_V = reinterpret_cast<bf16*>(V.data_ptr<c10::BFloat16>());
  float *d_L = L.data_ptr<float>();

  global_layout<ATTN_D> Qg(d_Q, B, N, H,    nullptr);
  global_layout<ATTN_D> Kg(d_K, B, N, H_KV, nullptr);
  global_layout<ATTN_D> Vg(d_V, B, N, H_KV, nullptr);

  // bf16 output (E2E inference): the store epilogue converts the float softmax
  // accumulators -> bf16 in-register, so the composed path no longer needs a
  // separate f32->bf16 elementwise cast (+ HBM round-trip) per decoder layer.
  // f32 output (training / bwd recompute) keeps the exact legacy behaviour.
  if (O.dtype() == torch::kBFloat16) {
    global_layout_outh<ATTN_D> Og(
        reinterpret_cast<bf16*>(O.data_ptr<c10::BFloat16>()), B, N, H, nullptr);
    route_fwd(queue, Qg, Kg, Vg, Og, d_L, B, H, N, is_causal);
  } else {
    global_layout_out<ATTN_D> Og(O.data_ptr<float>(), B, N, H, nullptr);
    route_fwd(queue, Qg, Kg, Vg, Og, d_L, B, H, N, is_causal);
  }
}

// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "SyclKittens GQA/MHA forward: single adaptive best-of-best kernel";
  m.def("dispatch_fwd", &dispatch_fwd,
        "Forward attention: Q,K,V (bf16 BNHD) -> O (f32 BNHD), L (f32 [B,H,N]). "
        "Adaptively selects the best (WG, seq_q) per shape.",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("L"),
        py::arg("is_causal") = true);
}
