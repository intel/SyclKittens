/**
 * @file all_gather.dp.cpp
 * @brief All-Gather Attention — the in-HBM / inference-prefill counterpart to
 *        ring attention (ring.dp.cpp), for Intel PVC multi-GPU.
 *
 * AG-attn shards the queries across the 6 GPUs (like ring) but, instead of
 * rotating the KV shard one hop per step and merging with an online softmax,
 * it does a SINGLE concurrent all-gather of every KV shard into a full
 * [B, N_total, H_KV, D] buffer on each device, then runs ONE pass of the
 * fence-free flash core over the full context. No online-softmax merge, no
 * P-step serial rotation — one gather + one flash.
 *
 *   * Collective: fan-out PUSH all-gather (each device reads its LOCAL shard once
 *     and writes it into the full buffer of all N-1 peers at its own slot).
 *     PUSH > PULL on PVC and fan-out leaves HBM headroom so it does not collapse
 *     at large payload. A strict ring cannot do this: it uses one link per hop.
 *   * Compute: the SAME fence-free wg256/seq_q16 flash core as ring attention
 *     (from kernels/attention/gqa_forward.dp.cpp), called once
 *     with N_q = N_per (local queries) attending N_kv = N_total (full context).
 *
 * Trade-off vs ring: O(N_total) KV memory per GPU (does NOT scale context past
 * a single tile's HBM) in exchange for one push all-gather instead of P serial
 * rotations. Wins the in-HBM regime; ring owns the memory-scaling regime.
 *
 * FLOP accounting is identical to ring (4*B*H*N_per*N_total*D per GPU), so the
 * TFLOP/s/GPU numbers are directly comparable to ring attention.
 *
 * Compile:
 *   icpx -fsycl -fsycl-targets=intel_gpu_pvc -std=c++20 -O3 \
 *     -DKITTENS_INTEL -DKITTENS_XE -DKITTENS_MULTI_GPU -DCAUSAL=0 \
 *     -fsycl-unnamed-lambda -I<SyclKittens>/include \
 *     -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
 *     all_gather_bench.dp.cpp -o all_gather_attention
 *
 * Collective design notes (shared with ring.dp.cpp):
 *
 *   1. COLLECTIVE (K/V rotation): double-buffered *bulk memcpy*
 *      rotation over the shared SYCL context (XeLink). A "zero-copy P2P read"
 *      variant is a TRAP on PVC:
 *      reading K/V over XeLink per-DPAS stalls the compute engine (no async DMA)
 *      and runs 5-50x SLOWER. Bulk memcpy → local-HBM compute is the right shape.
 *
 *   2. COMPUTE (attention): the fence-free core from
 *      kernels/attention/gqa_forward.dp.cpp:
 *        - Q pre-loaded ONCE into registers before the KV loop.
 *        - NO software fences (IGC SWSB handles load→DPAS hazards; +~15% large-N).
 *        - LPT launch ordering (grid Z,H,B with in-kernel Z reversal).
 *        - SPLIT_D output accumulation for D>64.
 *      For ring attention (non-causal, N_per ≥ 1024) the empirically-best config
 *      is wg256 / seq_q16 (the non-causal MHA champion), instantiated here.
 */

#include "kittens.dp.hpp"
#include "ops/warp/register/tile/conversions.dp.hpp"
using namespace kittens;

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

// ============================================================
// Attention problem config (MHA: H == H_KV)
// ============================================================
constexpr int ATTN_D    = 128;
constexpr int ATTN_H    = 16;
constexpr int ATTN_H_KV = 16;
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
static_assert(ATTN_H % ATTN_H_KV == 0, "Q heads must be divisible by KV heads");

// Ring attention is full (non-causal) attention over the rotated KV shards.
constexpr bool RING_CAUSAL = false;
// Non-causal MHA config for N_per >= 1024 (from gqa_forward.dp.cpp choose_fwd).
constexpr int RING_WG    = 256;
constexpr int RING_SEQ_Q = 16;

template <int D> using global_layout     = gl<bf16,  -1, -1, -1, D>;
template <int D> using global_layout_out = gl<float, -1, -1, -1, D>;

// ============================================================
// Per-(WG, seq_q) tile configuration & register-tile types
// (same as kernels/attention/gqa_forward.dp.cpp)
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
// Kernel (fence-free compute core, same as gqa_forward.dp.cpp)
// ============================================================
template <int D, int WG, int SEQ_Q, bool CAUSAL_T>
SYCL_EXTERNAL void attend_ker_gqa_fwd(
        global_layout<D> Qg, global_layout<D> Kg,
        global_layout<D> Vg, global_layout_out<D> Og,
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
  const int Z           = item_ct1.get_group_range(0);
  const int z_rank      = item_ct1.get_group(0);
  const int head_id     = item_ct1.get_group(1);
  const int batch_id    = item_ct1.get_group(2);
  const int start_q_seq = (Z - 1 - z_rank) * num_sg + sg_id;

  if (start_q_seq * SEQ_Q >= N_seq) return;

  const int kv_head_id = head_id / GROUP_SIZE;

  o_half_t o_reg_1, o_reg_2;
  zero(o_reg_1);
  if constexpr (SPLIT_D) zero(o_reg_2);

  typename attn_t::col_vec max_vec_last, max_vec, norm_vec;
  zero(norm_vec);
  zero(max_vec);

  const float scale = (1.0f / __builtin_sqrtf(static_cast<float>(D))) * 1.44269504089f;

  q_chunk_t q_preloaded[inner_loop];
  #pragma unroll
  for (int idx = 0; idx < inner_loop; idx++) {
    kittens::load_part<1>(q_preloaded[idx], Qg,
                          {batch_id, start_q_seq, head_id, 0}, {0, idx});
  }

  const int num_kv_tiles = Kg.depth() / kv_stride;
  int loop_kv_tiles;
  if constexpr (CAUSAL_T) {
    constexpr int ratio = kv_stride / SEQ_Q;
    const int max_kv = start_q_seq / ratio + 1;
    loop_kv_tiles = (max_kv < num_kv_tiles) ? max_kv : num_kv_tiles;
  } else {
    loop_kv_tiles = num_kv_tiles;
  }

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

  #pragma unroll
  for (int i = 0; i < loop_kv_tiles; i++, prefetch++) {

    attn_t att_block;
    zero(att_block);
    max_vec_last = max_vec;

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

    if constexpr (CAUSAL_T) {
      constexpr int ratio = kv_stride / SEQ_Q;
      const int q_blk = start_q_seq;
      const int k_blk = i;
      if (k_blk > q_blk / ratio) {
        neg_infty(att_block);
      } else if (k_blk == q_blk / ratio) {
        int part = start_q_seq % ratio;
        make_causal(att_block, att_block,
                    kittens::base_types::constants<float>::neg_infty(), part);
      }
    }

    max_vec = max<axis::COL>(att_block, max_vec);
    auto max_val = max_vec * scale;
    max_vec_last = exp2(max_vec_last * scale - max_val);

    o_reg_1 *= max_vec_last;
    if constexpr (SPLIT_D) o_reg_2 *= max_vec_last;
    norm_vec *= max_vec_last;

    att_block = exp2(att_block * scale - max_val);
    norm_vec = sum<axis::COL>(att_block, norm_vec);

    p_mma_t p_mma;
    p_mma = att_block;

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

  o_reg_1 *= inv(norm_vec);
  if constexpr (SPLIT_D) {
    o_reg_2 *= inv(norm_vec);
    kittens::store_part<1>(Og, o_reg_1, {batch_id, start_q_seq, head_id, 0}, {0, 0});
    kittens::store_part<1>(Og, o_reg_2, {batch_id, start_q_seq, head_id, 0},
                           {0, second_index_start});
  } else {
    kittens::store<1>(Og, o_reg_1, {batch_id, start_q_seq, head_id, 0});
  }

  {
    auto sg  = item_ct1.get_sub_group();
    int lane = sg.get_local_linear_id();
    const float inv_sqrt_D_val = 1.0f / __builtin_sqrtf(static_cast<float>(D));
    float L_val = max_vec.data[0][0] * inv_sqrt_D_val
                + sycl::log(norm_vec.data[0][0]);
    long L_idx = (long)batch_id * H_total * N_seq
               + (long)head_id  * N_seq
               + (long)start_q_seq * SEQ_Q + lane;
    if (lane < SEQ_Q)
      L_out[L_idx] = L_val;
  }
}

// Unique SYCL kernel-name tag per variant so instantiations never ODR-clash.
template <int WG, int SEQ_Q, bool CAUSAL_T> class gqa_fwd_kernel;

template <int WG, int SEQ_Q, bool CAUSAL_T>
static void submit_fwd(sycl::queue &queue,
                       global_layout<ATTN_D> Qg, global_layout<ATTN_D> Kg,
                       global_layout<ATTN_D> Vg, global_layout_out<ATTN_D> Og,
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
    cgh.parallel_for<gqa_fwd_kernel<WG, SEQ_Q, CAUSAL_T>>(
        sycl::nd_range<3>(grid * block, block), exp_props,
        [=](sycl::nd_item<3> item) {
          attend_ker_gqa_fwd<ATTN_D, WG, SEQ_Q, CAUSAL_T>(Qg, Kg, Vg, Og, d_L, H, N);
        });
  });
}

using sbf16 = sycl::ext::oneapi::bfloat16;

// AG dispatch: N_q local queries attend N_kv (= full gathered context) keys in a
// SINGLE pass. Same validated flash core; the kernel reads Kg.depth() for the KV
// tile count and uses N_q for the query grid/bound, so no online merge is needed.
void dispatch_attn_fwd_ag(sycl::queue &queue,
                          sbf16 *d_Q, sbf16 *d_Kfull, sbf16 *d_Vfull,
                          float *d_O, float *d_L,
                          int B, int N_q, int N_kv, int H, int H_KV) {
  global_layout<ATTN_D>     Qg(reinterpret_cast<bf16*>(d_Q),     B, N_q,  H,    nullptr);
  global_layout<ATTN_D>     Kg(reinterpret_cast<bf16*>(d_Kfull), B, N_kv, H_KV, nullptr);
  global_layout<ATTN_D>     Vg(reinterpret_cast<bf16*>(d_Vfull), B, N_kv, H_KV, nullptr);
  global_layout_out<ATTN_D> Og(d_O, B, N_q, H, nullptr);
  submit_fwd<RING_WG, RING_SEQ_Q, false>(queue, Qg, Kg, Vg, Og, d_L, B, H, N_q);
}

// ============================================================
// Fan-out PUSH KV scatter (best-of-best all-gather, mirrors all_gather_v9)
// ------------------------------------------------------------
// PUSH > PULL on PVC (18.8 vs 15.7 GB/s/link): each device reads its LOCAL
// shard ONCE and WRITES it into all (n_gpus-1) remote full buffers at its own
// slot `dev`, so a device's HBM only spends epd on local reads (not (N-1)*epd)
// leaving headroom for the incoming remote writes -> no HBM collapse at large
// payload. Self slot is a local memcpy. Measured 277-455 GB/s aggregate vs the
// old fan-in pull gather's ~108-120 GB/s.
//   local shard element (b, off)  ->  peer_full[dest][(b*N_total+dev*N_per)*HD + off]
// where HD = H_KV*D and off spans one batch-shard = N_per*H_KV*D elements.
// ============================================================
using vec128_ag = sycl::vec<uint32_t, 4>;   // 16 bytes = 8 bf16

template <int UNUSED> class ag_scatter_kernel;

static void scatter_kv_push(sycl::queue &queue, sbf16 *full_local,
                            sbf16 **peer_full, sbf16 *self_shard,
                            int dev, int n_gpus, int B, int N_per,
                            int N_total, int H_KV) {
  const long HD   = (long)H_KV * ATTN_D;              // elems per (b,n)
  const long Svec = (long)N_per * HD / 8;             // vec128 per batch-shard
  const long total_local_vec = (long)B * Svec;        // whole local shard
  const int WG = 256, EPT = 8;
  long n_wgs = (total_local_vec + (long)WG * EPT - 1) / ((long)WG * EPT);
  if (n_wgs < 1) n_wgs = 1;
  const int Np = N_per;

  // self shard: local contiguous memcpy into own full buffer at slot `dev`
  for (int b = 0; b < B; b++) {
    long dst_off = ((long)b * N_total + (long)dev * Np) * HD;
    long src_off = (long)b * Np * HD;
    queue.memcpy(full_local + dst_off, self_shard + src_off,
                 (size_t)Np * HD * sizeof(sbf16));
  }

  queue.parallel_for<ag_scatter_kernel<0>>(
      sycl::nd_range<1>(n_wgs * WG, WG), [=](sycl::nd_item<1> item) {
        const long g = item.get_group_linear_id();
        const int  lid = item.get_local_linear_id();
        auto *sp = reinterpret_cast<const vec128_ag*>(self_shard);
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
          long vi = g * (long)WG * EPT + (long)e * WG + lid;
          if (vi >= total_local_vec) return;
          long b_blk = vi / Svec;
          long offv  = vi - b_blk * Svec;
          long dst_vec = ((b_blk * N_total + (long)dev * Np) * HD) / 8 + offv;
          vec128_ag val = sp[vi];
          for (int j = 1; j < n_gpus; j++) {
            int dest = (dev + j) % n_gpus;
            reinterpret_cast<vec128_ag*>(peer_full[dest])[dst_vec] = val;
          }
        }
      });
}

    void run_all_gather_kv_push(
      std::vector<sycl::queue> &queues,
      const std::vector<sbf16*> &d_Kf,
      const std::vector<sbf16**> &d_Kptrs,
      const std::vector<sbf16*> &d_K,
      const std::vector<sbf16*> &d_Vf,
      const std::vector<sbf16**> &d_Vptrs,
      const std::vector<sbf16*> &d_V,
      int n_gpus,
      int B,
      int N_per,
      int N_total,
      int H_KV) {
      for (int d = 0; d < n_gpus; d++) {
      scatter_kv_push(queues[d], d_Kf[d], d_Kptrs[d], d_K[d],
              d, n_gpus, B, N_per, N_total, H_KV);
      scatter_kv_push(queues[d], d_Vf[d], d_Vptrs[d], d_V[d],
              d, n_gpus, B, N_per, N_total, H_KV);
      }
      for (auto &q : queues) q.wait();
    }

    void run_all_gather_attention_flash(
      std::vector<sycl::queue> &queues,
      const std::vector<sbf16*> &d_Q,
      const std::vector<sbf16*> &d_Kf,
      const std::vector<sbf16*> &d_Vf,
      const std::vector<float*> &d_O,
      const std::vector<float*> &d_L,
      int n_gpus,
      int B,
      int N_per,
      int N_total,
      int H,
      int H_KV) {
      for (int d = 0; d < n_gpus; d++) {
      dispatch_attn_fwd_ag(queues[d], d_Q[d], d_Kf[d], d_Vf[d], d_O[d], d_L[d],
                 B, N_per, N_total, H, H_KV);
      }
      for (auto &q : queues) q.wait();
    }

    void run_all_gather_attention(
      std::vector<sycl::queue> &queues,
      const std::vector<sbf16*> &d_Q,
      const std::vector<sbf16*> &d_Kf,
      const std::vector<sbf16**> &d_Kptrs,
      const std::vector<sbf16*> &d_K,
      const std::vector<sbf16*> &d_Vf,
      const std::vector<sbf16**> &d_Vptrs,
      const std::vector<sbf16*> &d_V,
      const std::vector<float*> &d_O,
      const std::vector<float*> &d_L,
      int n_gpus,
      int B,
      int N_per,
      int N_total,
      int H,
      int H_KV) {
      run_all_gather_kv_push(queues, d_Kf, d_Kptrs, d_K, d_Vf, d_Vptrs, d_V,
                 n_gpus, B, N_per, N_total, H_KV);
      run_all_gather_attention_flash(queues, d_Q, d_Kf, d_Vf, d_O, d_L,
                     n_gpus, B, N_per, N_total, H, H_KV);
    }
