/**
 * @file qkv_gemm.dp.cpp
 * @brief SyclKittens fused-QKV projection GEMM (PVC, bf16) PyTorch extension.
 *
 * Computes the Llama-3.1-8B fused Q/K/V projection
 *
 *     out(M, N_out) = hidden(M, K) @ W_kn(K, N_out)       (bf16 in, fp32 accum, bf16 out)
 *
 * where W_kn is the PRE-TRANSPOSED weight, i.e. torch `Linear.weight` (shape
 * (N_out, K)) transposed once to (K, N_out) row-major.  The weight is constant
 * per layer, so the transpose is a one-time prepack — there is NO per-call
 * transpose.
 *
 * Compile-time constants (dims are fixed for the model; M is RUNTIME):
 *   -DQKV_K=4096          hidden size
 *   -DQKV_NQ=4096         q width   (32 heads x 128)
 *   -DQKV_NK=1024         k width   (8  KV heads x 128)
 *   -DQKV_NV=1024         v width   (8  KV heads x 128)
 *   (N_out = NQ+NK+NV = 6144)
 *
 * Two output forms:
 *   (A) dispatch_qkv_gemm (hidden, W_kn, out)              -> out (M, 6144) bf16
 *   (B) dispatch_qkv_gemm3(hidden, W_kn, q_out,k_out,v_out) writes three
 *       CONTIGUOUS buffers q(M,4096) k(M,1024) v(M,1024) directly, so the
 *       attention side consumes q/k/v with no layout copy.  The q/k/v split
 *       boundaries (4096, 5120, 6144) are all multiples of BN=64, so every
 *       output column tile belongs to exactly one of q/k/v.
 *
 * The compute core is the shipping masked-self-prefetch GEMM inner loop
 * (bf16_gemm.dp.cpp), register-resident rt_32x32 DPAS, bf16-output store.
 * M is a runtime dimension and the work-group shape (WG_M x WG_N sub-groups) is
 * chosen at runtime by M for occupancy — small tall-skinny M needs many small
 * work-groups to fill the machine; large M uses the wide 32-subgroup tile.
 */

#include "kittens.dp.hpp"
#include <cstdlib>
using namespace kittens;

// ============================================================
// Compile-time problem constants (M is runtime)
// ============================================================
#ifndef QKV_K
#define QKV_K 4096
#endif
#ifndef QKV_NQ
#define QKV_NQ 4096
#endif
#ifndef QKV_NK
#define QKV_NK 1024
#endif
#ifndef QKV_NV
#define QKV_NV 1024
#endif
constexpr int GK  = QKV_K;
constexpr int NQ  = QKV_NQ;
constexpr int NK  = QKV_NK;
constexpr int NV  = QKV_NV;
constexpr int GN  = NQ + NK + NV;   // fused output width = 6144

// Tile sizes — proven PVC register sweet spot (rt_32x32).
#ifndef QKV_BM
#define QKV_BM 32
#endif
#ifndef QKV_BN
#define QKV_BN 64
#endif
#ifndef QKV_BK
#define QKV_BK 32
#endif
#define BM QKV_BM
#define BN QKV_BN
#define BK QKV_BK

static_assert(NQ % BN == 0 && NK % BN == 0 && NV % BN == 0,
              "q/k/v widths must be multiples of BN so each col-tile routes to one buffer");

#if (BM % 32 == 0) && (BN % 32 == 0) && (BK % 32 == 0)
#define RT_SHAPE ducks::rt_shape::rt_32x32
#else
#define RT_SHAPE ducks::rt_shape::rt_16x16
#endif

// Prefetch: K=4096 -> PF=2 (deeper pipeline, K fits L2 comfortably).
#ifndef QKV_PF
#define QKV_PF 2
#endif
#define PF QKV_PF
#ifndef QKV_PF_MODM
#define QKV_PF_MODM 5
#endif
#ifndef QKV_PF_MODN
#define QKV_PF_MODN 8
#endif
#define PF_MODM QKV_PF_MODM
#define PF_MODN QKV_PF_MODN

#ifndef QKV_UNROLL
#define QKV_UNROLL 32
#endif

#define NUM_THREADS (kittens::WARPGROUP_THREADS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;   // hidden  (M, K)
using _gl_B = gl<bf16, -1, -1, -1, -1>;   // W_kn    (K, N_out)
using _gl_C = gl<bf16, -1, -1, -1, -1>;   // out / q / k / v  (M, *)

// ============================================================
// Fused-QKV micro kernel.  SPLIT=false -> single (M,N_out) out;
// SPLIT=true -> route each col-tile to q/k/v.
// ============================================================
template <bool SPLIT>
void micro_qkv(_gl_A g_a, _gl_B g_b,
               _gl_C g_c, _gl_C g_q, _gl_C g_k, _gl_C g_v,
               int m_tiles) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t sg_id = item.get_sub_group().get_group_id();

    // Linear tile indexing (runtime M, tall-skinny grid).
    int wg_m = item.get_global_id(1);
    int wg_n = item.get_global_id(2) / 16;
    if (wg_m >= m_tiles || wg_n >= GN / BN) return;

    rt_bf<BM, BK, ducks::rt_layout::row, RT_SHAPE> a_reg;
    rt_bf<BK, BN, ducks::rt_layout::col, RT_SHAPE> b_reg;
    rt_fl<BM, BN, ducks::rt_layout::row, RT_SHAPE> c_reg;
    zero(c_reg);

    int prefetch = PF;
#pragma unroll
    for (int idx = 0; idx < PF; idx++) {
        if (sg_id % PF_MODM)
            prefetch_load(a_reg, g_a, {0, 0, wg_m, idx});
        if (sg_id & PF_MODN)
            prefetch_load(b_reg, g_b, {0, 0, idx, wg_n});
    }

#pragma unroll QKV_UNROLL
    for (int bkIdx = 0; bkIdx < GK / BK; bkIdx++, prefetch++) {
        load(a_reg, g_a, {0, 0, wg_m, bkIdx});
        load(b_reg, g_b, {0, 0, bkIdx, wg_n});
        if (prefetch < GK / BK) {
            if (sg_id % PF_MODM)
                prefetch_load(a_reg, g_a, {0, 0, wg_m, prefetch});
            if (sg_id & PF_MODN)
                prefetch_load(b_reg, g_b, {0, 0, prefetch, wg_n});
        }
        mma_AB(c_reg, a_reg, b_reg, c_reg);
    }

    if constexpr (!SPLIT) {
        store(g_c, c_reg, {0, 0, wg_m, wg_n});
    } else {
        constexpr int NQT = NQ / BN;         // q col-tiles
        constexpr int NKT = NK / BN;         // k col-tiles
        if (wg_n < NQT)
            store(g_q, c_reg, {0, 0, wg_m, wg_n});
        else if (wg_n < NQT + NKT)
            store(g_k, c_reg, {0, 0, wg_m, wg_n - NQT});
        else
            store(g_v, c_reg, {0, 0, wg_m, wg_n - NQT - NKT});
    }
}

// ============================================================
// Runtime work-group shape selection by M.
//   WG_N sub-groups cover the (always-large) N; WG_M sub-groups cover M.
//   A work-group computes a (WG_M*BM) x (WG_N*BN) output block.
//   A 32-subgroup work-group (WG_M=8, WG_N=4) is the sweet spot at EVERY M on
//   PVC: it maximizes per-WG data reuse while staying under the 1024-work-item
//   (64-subgroup) hardware limit. Shrinking WG_M at small M (the old
//   m_tiles/8 heuristic) starved reuse.
// ============================================================
static void pick_wg(int m_tiles, int &wg_m, int &wg_n) {
    wg_m = std::min(8, m_tiles);
    wg_n = 256 / BN;
}

template <bool SPLIT>
static void launch_qkv(bf16 *d_A, bf16 *d_B,
                       bf16 *d_C, bf16 *d_Q, bf16 *d_K, bf16 *d_V,
                       int M, sycl::queue &queue) {
    const int m_tiles = M / BM;
    const int n_tiles = GN / BN;

    int WG_M, WG_N;
    pick_wg(m_tiles, WG_M, WG_N);

    // Pad grid up to a whole number of work-groups; the bounds check in the
    // kernel drops padding tiles.
    int grid_m = ((m_tiles + WG_M - 1) / WG_M) * WG_M;
    int grid_n = ((n_tiles + WG_N - 1) / WG_N) * WG_N;

    _gl_A a_arg{d_A, 0, 0, (unsigned long)M,  (unsigned long)GK};
    _gl_B b_arg{d_B, 0, 0, (unsigned long)GK, (unsigned long)GN};
    _gl_C c_arg{d_C, 0, 0, (unsigned long)M,  (unsigned long)GN};
    _gl_C q_arg{d_Q, 0, 0, (unsigned long)M,  (unsigned long)NQ};
    _gl_C k_arg{d_K, 0, 0, (unsigned long)M,  (unsigned long)NK};
    _gl_C v_arg{d_V, 0, 0, (unsigned long)M,  (unsigned long)NV};

    sycl::range<3> grid(1, grid_m, grid_n * 16);
    sycl::range<3> block(1, WG_M, WG_N * 16);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>};

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid, block), exp_props,
                         [=](sycl::nd_item<3> item_ct1) {
                             micro_qkv<SPLIT>(a_arg, b_arg, c_arg,
                                              q_arg, k_arg, v_arg, m_tiles);
                         });
    });
}

// ============================================================
// PyTorch dispatch
// ============================================================
#include "pyutils/torch_helpers.dp.hpp"
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

static void check_2d(const torch::Tensor &t, int64_t rows, int64_t cols,
                     const char *name) {
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.is_xpu(), name, " must be on XPU");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    const int64_t n = t.numel();
    TORCH_CHECK(n == rows * cols, name, " numel ", n, " != ", rows, "x", cols);
}

// (A) single fused output: out (M, 6144)
void dispatch_qkv_gemm(torch::Tensor hidden, torch::Tensor w_kn, torch::Tensor out) {
    const int64_t M = hidden.size(0);
    TORCH_CHECK(hidden.size(1) == GK, "hidden K must be ", GK);
    TORCH_CHECK(M % BM == 0, "M (", M, ") must be a multiple of ", BM);
    check_2d(hidden, M, GK, "hidden");
    check_2d(w_kn, GK, GN, "w_kn (must be pre-transposed to [K, N_out])");
    check_2d(out, M, GN, "out");

    auto stream = c10::xpu::getCurrentXPUStream(hidden.device().index());
    auto &queue = stream.queue();

    bf16 *d_A = reinterpret_cast<bf16 *>(hidden.data_ptr<c10::BFloat16>());
    bf16 *d_B = reinterpret_cast<bf16 *>(w_kn.data_ptr<c10::BFloat16>());
    bf16 *d_C = reinterpret_cast<bf16 *>(out.data_ptr<c10::BFloat16>());

    launch_qkv<false>(d_A, d_B, d_C, d_C, d_C, d_C, static_cast<int>(M), queue);
}

// (B) split output: q (M,4096) k (M,1024) v (M,1024), each contiguous.
void dispatch_qkv_gemm3(torch::Tensor hidden, torch::Tensor w_kn,
                        torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out) {
    const int64_t M = hidden.size(0);
    TORCH_CHECK(hidden.size(1) == GK, "hidden K must be ", GK);
    TORCH_CHECK(M % BM == 0, "M (", M, ") must be a multiple of ", BM);
    check_2d(hidden, M, GK, "hidden");
    check_2d(w_kn, GK, GN, "w_kn (must be pre-transposed to [K, N_out])");
    check_2d(q_out, M, NQ, "q_out");
    check_2d(k_out, M, NK, "k_out");
    check_2d(v_out, M, NV, "v_out");

    auto stream = c10::xpu::getCurrentXPUStream(hidden.device().index());
    auto &queue = stream.queue();

    bf16 *d_A = reinterpret_cast<bf16 *>(hidden.data_ptr<c10::BFloat16>());
    bf16 *d_B = reinterpret_cast<bf16 *>(w_kn.data_ptr<c10::BFloat16>());
    bf16 *d_Q = reinterpret_cast<bf16 *>(q_out.data_ptr<c10::BFloat16>());
    bf16 *d_K = reinterpret_cast<bf16 *>(k_out.data_ptr<c10::BFloat16>());
    bf16 *d_V = reinterpret_cast<bf16 *>(v_out.data_ptr<c10::BFloat16>());

    launch_qkv<true>(d_A, d_B, nullptr, d_Q, d_K, d_V, static_cast<int>(M), queue);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens fused-QKV GEMM (SYCL, bf16): K=" +
              std::to_string(GK) + " N_out=" + std::to_string(GN) +
              " (q " + std::to_string(NQ) + " k " + std::to_string(NK) +
              " v " + std::to_string(NV) + ")";
    m.def("dispatch_qkv_gemm", &dispatch_qkv_gemm,
          "out = hidden @ W_kn  (bf16). hidden:[M,K] W_kn:[K,N_out] out:[M,N_out]",
          py::arg("hidden"), py::arg("w_kn"), py::arg("out"));
    m.def("dispatch_qkv_gemm3", &dispatch_qkv_gemm3,
          "q,k,v = split(hidden @ W_kn). Writes three contiguous buffers.",
          py::arg("hidden"), py::arg("w_kn"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"));
    m.attr("K") = GK;
    m.attr("N_out") = GN;
    m.attr("NQ") = NQ;
    m.attr("NK") = NK;
    m.attr("NV") = NV;
    m.attr("BM") = BM;
    m.attr("BN") = BN;
}
