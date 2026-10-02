/**
 * @file ring_overlap.dp.cpp
 * @brief Ring Attention with compute/communication OVERLAP for Intel PVC.
 *
 * Same compute kernel (fence-free wg256/sq16) and fused online-softmax reduce
 * as ring.dp.cpp, but the ring loop is restructured so the K/V rotation
 * DMA runs CONCURRENTLY with the attention compute instead of after it.
 *
 * Why ring.dp.cpp does NOT overlap (measured 28-40% of time is exposed P2P):
 *   - the rotation memcpy is submitted on the SAME in-order queue as compute,
 *     right after it  -> the copy engine cannot start until compute drains;
 *   - a full `for(q) q.wait()` host barrier sits between every ring step
 *     -> step s's transfer can never overlap step s+1's compute.
 *
 * This version removes both:
 *   1. TWO in-order queues per device: a COMPUTE queue (CCS / XMX) and a
 *      dedicated COPY queue (BCS copy engine). On PVC the copy engines are
 *      separate hardware from the vector engines, so a bulk memcpy on the copy
 *      queue genuinely overlaps an attention kernel on the compute queue.
 *   2. NO per-step host barrier. Step ordering is expressed with SYCL events
 *      (cross-device events are legal inside the shared context). A single
 *      wait() closes the whole ring at the end.
 *
 * Pipeline (per device d, block m(d,s) = (d - s) mod P):
 *   for s in 0..P-1:
 *     copy_s :  nxt <- neighbor(d-1).cur      (prefetch step s+1 data, COPY q)
 *     cmp_s  :  attn(Q, cur) -> O_blk         (COMPUTE q, overlaps copy_s)
 *     red_s  :  merge O_blk into O            (COMPUTE q, s>0)
 *   Dependencies (no barrier):
 *     cmp_s   depends on copy_{s-1}[d]                 (cur just arrived)
 *     copy_s  depends on cmp_{s-1}[d]                  (WAR: nxt was read by cmp_{s-1})
 *     copy_s  depends on copy_{s-1}[neighbor]          (RAW: neighbor.cur was produced there)
 *   Double-buffered K0/K1,V0/V1 make the WAR slack exactly one step.
 *
 * Critical path per step collapses from (compute+comm+reduce) to
 * max(comm, compute+reduce) -> for compute-bound N the P2P term is hidden.
 *
 * Bring-up target: 2 tiles first (ZE_AFFINITY_MASK=0,2). The code is general in
 * P; at P=2 there is exactly one overlapping copy (step 0), the cleanest case.
 *
 * Compile:
 *   icpx -fsycl -fsycl-targets=intel_gpu_pvc -std=c++20 -O3 \
 *     -I<SyclKittens>/include ring_overlap.dp.cpp -o ring_attention_overlap
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
template <int WG, int SEQ_Q, bool CAUSAL_T> class gqa_fwd_kernel_v4;

template <int WG, int SEQ_Q, bool CAUSAL_T>
static sycl::event submit_fwd(sycl::queue &queue,
                       global_layout<ATTN_D> Qg, global_layout<ATTN_D> Kg,
                       global_layout<ATTN_D> Vg, global_layout_out<ATTN_D> Og,
                       float *d_L, int B, int H, int N,
                       const std::vector<sycl::event> &deps) {
  constexpr int rows_per_wg = fwd_cfg<ATTN_D, WG, SEQ_Q>::rows_per_wg;
  constexpr int sg_size     = fwd_cfg<ATTN_D, WG, SEQ_Q>::sg_size;
  sycl::range<3> grid((N + rows_per_wg - 1) / rows_per_wg, H, B);
  sycl::range<3> block(1, 1, (WG + sg_size - 1) / sg_size * 16);
  auto exp_props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::sub_group_size<16>,
      sycl::ext::oneapi::experimental::work_group_scratch_size(
          kittens::MAX_SHARED_MEMORY)};
  return queue.submit([&](sycl::handler &cgh) {
    if (!deps.empty()) cgh.depends_on(deps);
    cgh.parallel_for<gqa_fwd_kernel_v4<WG, SEQ_Q, CAUSAL_T>>(
        sycl::nd_range<3>(grid * block, block), exp_props,
        [=](sycl::nd_item<3> item) {
          attend_ker_gqa_fwd<ATTN_D, WG, SEQ_Q, CAUSAL_T>(Qg, Kg, Vg, Og, d_L, H, N);
        });
  });
}

using sbf16 = sycl::ext::oneapi::bfloat16;

// Dispatch the ring champion (wg256 / seq_q16 / non-causal). Returns the compute
// event and honours a dependency list (cross-queue arrival of the KV block).
sycl::event dispatch_attn_fwd(sycl::queue &queue,
                       sbf16 *d_Q, sbf16 *d_K, sbf16 *d_V,
                       float *d_O, float *d_L,
                       int B, int N, int H, int H_KV,
                       const std::vector<sycl::event> &deps = {}) {
  global_layout<ATTN_D>     Qg(reinterpret_cast<bf16*>(d_Q), B, N, H,    nullptr);
  global_layout<ATTN_D>     Kg(reinterpret_cast<bf16*>(d_K), B, N, H_KV, nullptr);
  global_layout<ATTN_D>     Vg(reinterpret_cast<bf16*>(d_V), B, N, H_KV, nullptr);
  global_layout_out<ATTN_D> Og(d_O, B, N, H, nullptr);
  return submit_fwd<RING_WG, RING_SEQ_Q, RING_CAUSAL>(queue, Qg, Kg, Vg, Og, d_L, B, H, N, deps);
}

// ============================================================
// Fused online softmax reduction kernel (verbatim from v2/v3)
//   O_new = exp(L - L_new) * O + exp(L_block - L_new) * O_block
//   L_new = L + log(1 + exp(L_block - L))
// ============================================================
sycl::event online_softmax_reduce_kernel(
    sycl::queue &q,
    float *O, float *L, const float *O_block, const float *L_block,
    int B, int N, int H, int D,
    const std::vector<sycl::event> &deps = {}) {
  int total_items = B * N * H;
  return q.submit([&](sycl::handler &cgh) {
    if (!deps.empty()) cgh.depends_on(deps);
    cgh.parallel_for(sycl::range<1>(total_items), [=](sycl::id<1> idx) {
      int i = idx[0];
      int b = i / (N * H);
      int rem = i % (N * H);
      int n = rem / H;
      int h = rem % H;
      int l_idx = b * H * N + h * N + n;
      int o_base = (b * N * H + n * H + h) * D;
      float l_old = L[l_idx];
      float l_blk = L_block[l_idx];
      float l_new = l_old + sycl::log(1.0f + sycl::exp(l_blk - l_old));
      float w_old = sycl::exp(l_old - l_new);
      float w_blk = sycl::exp(l_blk - l_new);
      for (int d = 0; d < D; d++) {
        float o_val = O[o_base + d] * w_old + O_block[o_base + d] * w_blk;
        O[o_base + d] = o_val;
      }
      L[l_idx] = l_new;
    });
  });
}

void run_ring_attention_overlap(
    std::vector<sycl::queue> &compute_queues,
    std::vector<sycl::queue> &copy_queues,
    const std::vector<sbf16 *> &queries,
    std::vector<sbf16 *> &keys_0, std::vector<sbf16 *> &keys_1,
    std::vector<sbf16 *> &values_0, std::vector<sbf16 *> &values_1,
    const std::vector<float *> &outputs,
    const std::vector<float *> &block_outputs,
    const std::vector<float *> &logsumexp,
    const std::vector<float *> &block_logsumexp,
    int batch, int sequence, int heads, int kv_heads, size_t kv_elements) {
  const int device_count = static_cast<int>(compute_queues.size());
  for (int step = 0; step < device_count; step++) {
    const bool even = step % 2 == 0;
    auto &current_keys = even ? keys_0 : keys_1;
    auto &current_values = even ? values_0 : values_1;
    auto &next_keys = even ? keys_1 : keys_0;
    auto &next_values = even ? values_1 : values_0;

    if (step < device_count - 1) {
      for (int device = 0; device < device_count; device++) {
        int source = (device - 1 + device_count) % device_count;
        copy_queues[device].memcpy(next_keys[device], current_keys[source],
                                   kv_elements * sizeof(sbf16));
        copy_queues[device].memcpy(next_values[device], current_values[source],
                                   kv_elements * sizeof(sbf16));
      }
    }

    for (int device = 0; device < device_count; device++) {
      float *output = step == 0 ? outputs[device] : block_outputs[device];
      float *lse = step == 0 ? logsumexp[device] : block_logsumexp[device];
      dispatch_attn_fwd(compute_queues[device], queries[device],
                        current_keys[device], current_values[device], output,
                        lse, batch, sequence, heads, kv_heads);
    }
    if (step > 0) {
      for (int device = 0; device < device_count; device++)
        online_softmax_reduce_kernel(
            compute_queues[device], outputs[device], logsumexp[device],
            block_outputs[device], block_logsumexp[device], batch, sequence,
            heads, ATTN_D);
    }
    for (auto &queue : compute_queues)
      queue.wait();
    for (auto &queue : copy_queues)
      queue.wait();
  }
}
