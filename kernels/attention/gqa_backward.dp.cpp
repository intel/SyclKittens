/**
 * @file gqa_backward.dp.cpp
 * @brief SyclKittens fused causal attention backward pass.
 *
 * Each 32-subgroup workgroup owns one 128-row KV block. The fused kernel computes
 * P and dS once, stores their transposed BF16 tiles in global reorder scratch,
 * accumulates dV and dK in 32-column head-dimension slices, and atomically adds
 * each KV block's contribution to dQ. Packed 2D loads feed the score and dP DPAS
 * contractions directly.
 */

#include "kittens.dp.hpp"
#include "ops/warp/register/tile/conversions.dp.hpp"
#include "pyutils/torch_helpers.dp.hpp"

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
using namespace kittens;

// ============================================================
// Configuration
// ============================================================

#ifndef ATTN_H_VALUE
constexpr int ATTN_H = 16;
#else
constexpr int ATTN_H = ATTN_H_VALUE;
#endif

#ifndef ATTN_H_KV_VALUE
constexpr int ATTN_H_KV = 16;
#else
constexpr int ATTN_H_KV = ATTN_H_KV_VALUE;
#endif

#ifndef ATTN_D_VALUE
constexpr int ATTN_D = 128;
#else
constexpr int ATTN_D = ATTN_D_VALUE;
#endif

// ============================================================
// Tile sizes
// ============================================================

constexpr int BLOCK_M     = 128;  // rows of Q tile
constexpr int BLOCK_N     = 128;  // rows of KV tile
constexpr int SG_SIZE     = 16;
constexpr int NUM_SG      = BLOCK_N / SG_SIZE;   // 8 SGs (prep + dq, v35-native)
constexpr int WG_SIZE     = NUM_SG * SG_SIZE;    // 128 threads (prep + dq)
constexpr int D_CHUNK     = 32;
constexpr int D_INNER     = ATTN_D / D_CHUNK;    // 4
constexpr int D_SUBTILES  = D_CHUNK / 16;        // 2

// --- v47 cooperative dvdk geometry ---
// 32 SGs / WG. Partition = 8 KV-subgroups x 4 (q-groups in SdP / d-groups in dKV).
constexpr int COOP_KV_GRP = BLOCK_N / SG_SIZE;   // 8  (KV rows per WG / 16)
constexpr int COOP_SPLIT  = 4;                   // AtomLayout4 factor
constexpr int COOP_NUM_SG = COOP_KV_GRP * COOP_SPLIT;  // 32
constexpr int COOP_WG     = COOP_NUM_SG * SG_SIZE;     // 512 threads
constexpr int COOP_DSLICE = ATTN_D / COOP_SPLIT; // 32  (headdim per SG for dKV)
constexpr int COOP_QSLICE = BLOCK_M / COOP_SPLIT;// 32  (q cols per SG for SdP)
constexpr int COOP_DSUB   = COOP_DSLICE / 16;    // 2   (16-col subtiles per d-slice)
constexpr int COOP_QSUB   = COOP_QSLICE / 16;    // 2   (16-col subtiles per q-slice)
constexpr int Q_SUBTILES  = BLOCK_M / 16;        // 8   (16-row q-subtiles per Q block)

constexpr int Q_HEADS  = ATTN_H;
constexpr int KV_HEADS = ATTN_H_KV;
constexpr int GROUP_SIZE = Q_HEADS / KV_HEADS;
static_assert(Q_HEADS % KV_HEADS == 0);
static_assert(ATTN_D % D_CHUNK == 0);

// ============================================================
// Type aliases
// ============================================================

template <int D> using bwd_gl     = gl<bf16,  -1, -1, -1, D>;
template <int D> using bwd_gl_out = gl<float, -1, -1, -1, D>;

using qk_chunk_t = rt<bf16,  16, D_CHUNK, row_l>;   // 16×32 bf16

using small_f    = rt<float, 16, 16, row_l>;        // 16×16 float
using small_bf   = rt<bf16,  16, 16, row_l>;        // 16×16 bf16

// v47 cooperative dvdk tiles
using acc_dslice = rt<float, 16, COOP_DSLICE, row_l>; // 16×32 float — per-SG dV/dK accumulator
using col_dslice = rt<bf16,  16, COOP_DSLICE, col_l>; // 16×32 bf16 col — dO/Q d-slice (dKV B operand)

static constexpr float inv_sqrt_D = (ATTN_D == 128) ? 0.08838834764f : 0.125f;
static constexpr float LOG2E      = 1.44269504089f;
static constexpr float P_SCALE    = inv_sqrt_D * LOG2E;
static constexpr float L_SCALE    = LOG2E;

// ============================================================
// Kernel 1 — Prep (v35, WG=128)
// ============================================================

template <int D>
SYCL_EXTERNAL void attend_prep_ker(
        bwd_gl<D> dOg, bwd_gl<D> Og,
        float* __restrict__ delta_out,
        int B, int H, int N)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg   = item.get_sub_group();
    const int sg_id    = item.get_local_id(2) / SG_SIZE;
    const int batch_id = item.get_group(0);
    const int head_id  = item.get_group(1);
    const int q_start  = item.get_group(2) * NUM_SG + sg_id;
    const int lane     = sg.get_local_linear_id();

    if (q_start * 16 >= N) return;

    typename small_f::col_vec delta_vec;
    zero(delta_vec);

    qk_chunk_t dO_chunk, O_chunk;
    rt<float, 16, D_CHUNK, row_l> dO_f, O_f, prod;

    #pragma unroll
    for (int c = 0; c < D_INNER; c++) {
        load_part<1>(dO_chunk, dOg, {batch_id, q_start, head_id, 0}, {0, c * D_SUBTILES});
        load_part<1>(O_chunk,  Og,  {batch_id, q_start, head_id, 0}, {0, c * D_SUBTILES});
        copy(dO_f, dO_chunk);
        copy(O_f,  O_chunk);
        mul(prod, dO_f, O_f);
        row_sum(delta_vec, prod, delta_vec);
    }

    float* base = delta_out + ((long)batch_id * H + head_id) * N + q_start * 16;
    base[lane] = delta_vec.data[0][0];
}

// ============================================================
// Kernel 2 — v50 FUSED dV + dK + dQ (32 SGs, headdim-split, global reorder)
//
// WG owns one 128-row KV block. 32 SGs. Single kernel does ALL THREE gradients,
// computing S/P/dP/dS ONCE (vs twice in the v48 split dvdk+dq design):
//   SdP phase  : SG(kv_sub, q_grp) computes S^T/P^T/dP^T/dS^T, writes P^T/dS^T
//                (bf16) to global reorder scratch.
//   dKV phase  : SG(kv_grp, d_grp) reads scratch, dV += P^T·dO, dK += dS^T·Q.
//   dQ phase   : SG(q_sub, d_grp) reads the SAME dS^T scratch, dQ += dS·K via
//                mma_AtB, then atomic_add_part to global dQ (this KV block's
//                contribution; other KV blocks sum in via fp32 atomics @ ~86%
//                store-BW on PVC L2). Eliminates dq's redundant S/P/dP/dS.
// ============================================================

template <int D>
SYCL_EXTERNAL void attend_bwd_fused_ker(
        bwd_gl<D> Qg, bwd_gl<D> Kg, bwd_gl<D> Vg, bwd_gl<D> dOg,
        bwd_gl<D> Ptg, bwd_gl<D> dStg,           // reorder scratch (bf16, [kv x q])
        bwd_gl<D> dSqg,                          // dQ scratch (bf16, dS[q x kv], q-major)
        const float* __restrict__ L_in,
        const float* __restrict__ delta_in,
        float* __restrict__ dV_out,
        float* __restrict__ dK_out,
        float* __restrict__ dQ_out,              // FUSED: dQ via atomic_add (KV-centric)
        int B, int H, int H_KV, int N)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg   = item.get_sub_group();
    const int sg_id      = item.get_local_id(2) / SG_SIZE;
    const int batch_id   = item.get_group(0);
    const int kv_head_id = item.get_group(1);
    const int lane       = sg.get_local_linear_id();

    const int kv_block = item.get_group(2);
    const int kv_base  = kv_block * BLOCK_N;

    // Shared kv-row ownership across both phases.
    const int kv_sub    = sg_id / COOP_SPLIT;    // 0..7   (which 16-row KV strip)
    const int split_idx = sg_id % COOP_SPLIT;    // 0..3   (q-group in SdP / d-group in dKV)
    const int kv_row    = kv_base + kv_sub * 16;

    // Global scratch row-tile base for this SG's KV strip: scratch laid out with
    // "N" dim = kv_blocks*128, so kv row r maps to row-tile r/16.
    const int pt_rowtile = (kv_base + kv_sub * 16) / 16;

    const int num_q_blocks = N / BLOCK_M;

    acc_dslice dV_acc, dK_acc;
    zero(dV_acc); zero(dK_acc);

    auto grp = item.get_group();

    for (int g = 0; g < GROUP_SIZE; g++) {
        const int head_id  = kv_head_id * GROUP_SIZE + g;
        const int m_start  = kv_block;

        for (int m_block = m_start; m_block < num_q_blocks; m_block++) {
            const int q_base = m_block * BLOCK_M;

            // ---------- SdP phase: produce P^T / dS^T for my (kv_sub, q_grp) ----------
            #pragma unroll
            for (int sub = 0; sub < COOP_QSUB; sub++) {
                const int q_col_base = q_base + split_idx * COOP_QSLICE + sub * 16;

                // S^T[16kv x 16q] = K[16kv x D] · Q[16q x D]^T
                small_f S;
                zero(S);
                // Load-into-packed SdP: K row-loaded straight into the dpas A
                // operand, Q transpose-loaded straight into B, raw dpas per 16x16
                // block; S resident as float8. No persistent packed state.
                dpas_C_f32_16x16 S_acc; zero_C(S_acc);
                #pragma unroll
                for (int c = 0; c < D_INNER; c++) {
                    #pragma unroll
                    for (int kb = 0; kb < D_SUBTILES; kb++) {
                        dpas_A_bf16_16x16 k_pk;
                        load_packed_A_16x16<1>(&k_pk.h[0], Kg, {batch_id, kv_row / 16, kv_head_id, 0}, {0, c * D_SUBTILES + kb});
                        dpas_B_bf16_16x16 q_pk;
                        load_packed_Bt_16x16<1>(&q_pk.m, Qg, {batch_id, q_col_base / 16, head_id, 0}, {0, c * D_SUBTILES + kb});
                        mma_raw_AB(S_acc, k_pk, q_pk);
                    }
                }
                store_C(S.tiles[0][0], S_acc);

                // Causal mask (lane = q, data[k] = kv rows 2k,2k+1).
                if (kv_row > q_col_base + 15) {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        S.tiles[0][0].data[k] = {
                            kittens::base_types::constants<float>::neg_infty(),
                            kittens::base_types::constants<float>::neg_infty()
                        };
                    }
                } else if (kv_row + 15 <= q_col_base) {
                    // no masking
                } else {
                    int offset = q_col_base - kv_row;
                    for (int k = 0; k < 8; k++) {
                        int r0 = 2 * k, r1 = 2 * k + 1;
                        int c = lane;
                        if (r0 - c > offset)
                            S.tiles[0][0].data[k].x() = kittens::base_types::constants<float>::neg_infty();
                        if (r1 - c > offset)
                            S.tiles[0][0].data[k].y() = kittens::base_types::constants<float>::neg_infty();
                    }
                }

                // P^T = exp2(S * P_SCALE - L[q] * L_SCALE)
                float L_val;
                {
                    const float* lp = L_in + ((long)batch_id * H + head_id) * N + q_col_base;
                    L_val = lp[lane] * L_SCALE;
                }
                mul(S, S, P_SCALE);
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    S.tiles[0][0].data[k].x() -= L_val;
                    S.tiles[0][0].data[k].y() -= L_val;
                }
                small_f P;
                exp2(P, S);

                // Load-into-packed dP^T: V row-loaded straight into A, dO
                // transpose-loaded straight into B, raw dpas per 16x16 block;
                // dP resident as float8. Mirrors the S-loop, register-neutral.
                small_f dP;
                dpas_C_f32_16x16 dP_acc; zero_C(dP_acc);
                #pragma unroll
                for (int c = 0; c < D_INNER; c++) {
                    #pragma unroll
                    for (int kb = 0; kb < D_SUBTILES; kb++) {
                        dpas_A_bf16_16x16 v_pk;
                        load_packed_A_16x16<1>(&v_pk.h[0], Vg, {batch_id, kv_row / 16, kv_head_id, 0}, {0, c * D_SUBTILES + kb});
                        dpas_B_bf16_16x16 dO_pk;
                        load_packed_Bt_16x16<1>(&dO_pk.m, dOg, {batch_id, q_col_base / 16, head_id, 0}, {0, c * D_SUBTILES + kb});
                        mma_raw_AB(dP_acc, v_pk, dO_pk);
                    }
                }
                store_C(dP.tiles[0][0], dP_acc);

                // dS^T = P * (dP - δ[q]) * inv_sqrt_D
                float delta_val;
                {
                    const float* dp = delta_in + ((long)batch_id * H + head_id) * N + q_col_base;
                    delta_val = dp[lane];
                }
                small_f dS;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float dpx = dP.tiles[0][0].data[k].x() - delta_val;
                    float dpy = dP.tiles[0][0].data[k].y() - delta_val;
                    dS.tiles[0][0].data[k].x() = P.tiles[0][0].data[k].x() * dpx * inv_sqrt_D;
                    dS.tiles[0][0].data[k].y() = P.tiles[0][0].data[k].y() * dpy * inv_sqrt_D;
                }

                // Store P^T / dS^T (bf16) to global reorder scratch. The scratch q-dim is
                // only 128 (one Q-block, reused every m_block) so the column index is
                // RELATIVE to this Q-block: split_idx*QSUB + sub  (0..7).
                const int q_rel_tile = split_idx * COOP_QSUB + sub;
                small_bf P_bf, dS_bf;
                copy(P_bf, P);
                copy(dS_bf, dS);
                store_part<1>(Ptg,  P_bf,  {batch_id, pt_rowtile, kv_head_id, 0}, {0, q_rel_tile});
                store_part<1>(dStg, dS_bf, {batch_id, pt_rowtile, kv_head_id, 0}, {0, q_rel_tile});
                // Transpose dS^T[16kv x 16q] -> dS[16q x 16kv] ONCE here (register-native,
                // 64 total vs 256 redundant in the dQ loop) and store to dSqg so the dQ
                // phase reads it shuffle-free. Row = kv_block*8 + q_rel_tile so concurrent
                // KV-block WGs occupy disjoint row ranges; col = kv_sub (relative kv strip).
                small_bf dSq_bf;
                transpose_sep(dSq_bf, dS_bf);
                store_part<1>(dSqg, dSq_bf,
                              {batch_id, kv_block * Q_SUBTILES + q_rel_tile, kv_head_id, 0},
                              {0, kv_sub});
            }

            sycl::group_barrier(grp);

            // ---------- dKV phase: dV += P^T·dO, dK += dS^T·Q, headdim slice = split_idx ----------
            #pragma unroll
            for (int qs = 0; qs < Q_SUBTILES; qs++) {
                const int q_row16 = q_base + qs * 16;

                small_bf P_slice;
                load_part<1>(P_slice, Ptg, {batch_id, pt_rowtile, kv_head_id, 0}, {0, qs});
                col_dslice dO_col;
                load_part<1>(dO_col, dOg, {batch_id, q_row16 / 16, head_id, 0}, {0, split_idx * COOP_DSUB});
                mma<transpose::N, transpose::N>(dV_acc, P_slice, dO_col, dV_acc);

                small_bf dS_slice;
                load_part<1>(dS_slice, dStg, {batch_id, pt_rowtile, kv_head_id, 0}, {0, qs});
                col_dslice Q_col;
                load_part<1>(Q_col, Qg, {batch_id, q_row16 / 16, head_id, 0}, {0, split_idx * COOP_DSUB});
                mma<transpose::N, transpose::N>(dK_acc, dS_slice, Q_col, dK_acc);
            }

            // ---------- dQ phase (FUSED): dQ += dS·K, q-centric ownership ----------
            // Reuses the dS^T[128kv x 128q] block already in scratch (dStg) — no
            // recompute of S/P/dP/dS. SG owns (q_sub, d_grp); sums over all 128 kv
            // via mma_AtB(dS^T_col, K_col) then atomic-adds to global dQ (this KV
            // block's contribution; other KV blocks sum in via atomic).
            {
                const int q_sub2 = sg_id / COOP_SPLIT;   // 0..7  which 16-q-row subtile
                const int d_grp2 = sg_id % COOP_SPLIT;   // 0..3  which 32-headdim slice
                acc_dslice dQ_acc;
                zero(dQ_acc);
                #pragma unroll
                for (int kk = 0; kk < COOP_KV_GRP; kk++) {   // all 8 kv-subtiles (128 kv)
                    // dS[16q x 16kv] pre-transposed in SdP -> load shuffle-free (q-major
                    // dSqg). Row = this KV-block's region (kv_block*8) + q subtile; col =
                    // relative kv strip kk. Then standard mma_AB(dS, K) = dS·K = dQ.
                    small_bf dS_qkv;
                    load_part<1>(dS_qkv, dSqg,
                                 {batch_id, kv_block * Q_SUBTILES + q_sub2, kv_head_id, 0}, {0, kk});
                    col_dslice K_col;                        // K[16kv x 32d] col
                    load_part<1>(K_col, Kg,
                                 {batch_id, kv_base / 16 + kk, kv_head_id, 0}, {0, d_grp2 * COOP_DSUB});
                    mma<transpose::N, transpose::N>(dQ_acc, dS_qkv, K_col, dQ_acc);  // dS·K
                }
                bwd_gl_out<D> dQg(dQ_out, (unsigned long)B, (unsigned long)N,
                                  (unsigned long)H, nullptr);
                atomic_add_part<1>(dQg, dQ_acc,
                                   {batch_id, m_block * Q_SUBTILES + q_sub2, head_id, 0},
                                   {0, d_grp2 * COOP_DSUB});
            }

            sycl::group_barrier(grp);  // ensure all reads done before next m_block overwrites scratch
        }
    }

    // Store dV / dK: my 16 KV rows x 32 headdim slice [split_idx*32 .. +32].
    bwd_gl_out<D> dVg(dV_out, (unsigned long)B, (unsigned long)N,
                     (unsigned long)H_KV, nullptr);
    bwd_gl_out<D> dKg(dK_out, (unsigned long)B, (unsigned long)N,
                     (unsigned long)H_KV, nullptr);
    store_part<1>(dVg, dV_acc, {batch_id, kv_row / 16, kv_head_id, 0}, {0, split_idx * COOP_DSUB});
    store_part<1>(dKg, dK_acc, {batch_id, kv_row / 16, kv_head_id, 0}, {0, split_idx * COOP_DSUB});
}

// ============================================================
// Dispatch
// ============================================================

void dispatch_prep(torch::Tensor O, torch::Tensor dO, torch::Tensor delta) {
    CHECK_INPUT(O); CHECK_INPUT(dO); CHECK_INPUT(delta);
    const int B = O.size(0), N = O.size(1), H = O.size(2);

    auto stream = c10::xpu::getCurrentXPUStream(O.device().index());
    auto &queue = stream.queue();
    queue.wait();

    bf16  *d_O_bf  = reinterpret_cast<bf16*>(O.data_ptr<c10::BFloat16>());
    bf16  *d_dO_bf = reinterpret_cast<bf16*>(dO.data_ptr<c10::BFloat16>());
    float *d_delta = delta.data_ptr<float>();

    bwd_gl<ATTN_D> Og(d_O_bf,  B, N, H, nullptr);
    bwd_gl<ATTN_D> dOg(d_dO_bf, B, N, H, nullptr);

    int q_tiles_total = (N + 16 * NUM_SG - 1) / (16 * NUM_SG);
    sycl::range<3> grid(B, H, q_tiles_total);
    sycl::range<3> block(1, 1, WG_SIZE);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>,
        sycl::ext::oneapi::experimental::work_group_scratch_size(
            kittens::MAX_SHARED_MEMORY)};

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block), exp_props,
            [=](sycl::nd_item<3> item) {
                attend_prep_ker<ATTN_D>(dOg, Og, d_delta, B, H, N);
            });
    });
    queue.wait();
}

// Allocate the P^T / dS^T reorder scratch: [B, kv_blocks*128, H_KV, 128] bf16.
static torch::Tensor make_reorder_scratch(int B, int N, int H_KV, torch::Tensor like) {
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(like.device());
    return torch::empty({(long)B, (long)N, (long)H_KV, (long)ATTN_D}, opts);
}

void dispatch_bwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                  torch::Tensor dO, torch::Tensor L, torch::Tensor delta,
                  torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V); CHECK_INPUT(dO);
    CHECK_INPUT(L); CHECK_INPUT(delta);
    CHECK_INPUT(dQ); CHECK_INPUT(dK); CHECK_INPUT(dV);

    const int B = Q.size(0), N = Q.size(1), H = Q.size(2), H_KV = K.size(2);

    auto stream = c10::xpu::getCurrentXPUStream(Q.device().index());
    auto &queue = stream.queue();
    queue.wait();

    bf16  *d_Q  = reinterpret_cast<bf16*>(Q.data_ptr<c10::BFloat16>());
    bf16  *d_K  = reinterpret_cast<bf16*>(K.data_ptr<c10::BFloat16>());
    bf16  *d_V  = reinterpret_cast<bf16*>(V.data_ptr<c10::BFloat16>());
    bf16  *d_dO = reinterpret_cast<bf16*>(dO.data_ptr<c10::BFloat16>());
    float *d_L     = L.data_ptr<float>();
    float *d_delta = delta.data_ptr<float>();
    float *d_dQ    = dQ.data_ptr<float>();
    float *d_dK    = dK.data_ptr<float>();
    float *d_dV    = dV.data_ptr<float>();

    // Reorder scratch (bf16 P^T and dS^T), shaped [B, N(kv), H_KV, D(q=128)].
    torch::Tensor Pt  = make_reorder_scratch(B, N, H_KV, Q);
    torch::Tensor dSt = make_reorder_scratch(B, N, H_KV, Q);
    torch::Tensor dSq = make_reorder_scratch(B, N, H_KV, Q);  // dQ q-major scratch
    bf16 *d_Pt  = reinterpret_cast<bf16*>(Pt.data_ptr<c10::BFloat16>());
    bf16 *d_dSt = reinterpret_cast<bf16*>(dSt.data_ptr<c10::BFloat16>());
    bf16 *d_dSq = reinterpret_cast<bf16*>(dSq.data_ptr<c10::BFloat16>());

    bwd_gl<ATTN_D> Qg(d_Q, B, N, H, nullptr);
    bwd_gl<ATTN_D> Kg(d_K, B, N, H_KV, nullptr);
    bwd_gl<ATTN_D> Vg(d_V, B, N, H_KV, nullptr);
    bwd_gl<ATTN_D> dOg(d_dO, B, N, H, nullptr);
    bwd_gl<ATTN_D> Ptg(d_Pt, B, N, H_KV, nullptr);
    bwd_gl<ATTN_D> dStg(d_dSt, B, N, H_KV, nullptr);
    bwd_gl<ATTN_D> dSqg(d_dSq, B, N, H_KV, nullptr);

    // FUSED: dQ is accumulated via atomic_add across KV blocks — must be zeroed first.
    queue.memset(d_dQ, 0, (size_t)B * N * H * ATTN_D * sizeof(float)).wait();

    sycl::range<3> block_coop(1, 1, COOP_WG);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>,
        sycl::ext::oneapi::experimental::work_group_scratch_size(1)};

    {
        int kv_blocks = N / BLOCK_N;
        sycl::range<3> grid_kv(B, H_KV, kv_blocks);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid_kv * block_coop, block_coop), exp_props,
                [=](sycl::nd_item<3> item) {
                    attend_bwd_fused_ker<ATTN_D>(Qg, Kg, Vg, dOg, Ptg, dStg, dSqg,
                        d_L, d_delta, d_dV, d_dK, d_dQ, B, H, H_KV, N);
                });
        });
        queue.wait();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_prep", &dispatch_prep);
    m.def("dispatch_bwd", &dispatch_bwd);
}
