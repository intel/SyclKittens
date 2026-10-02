/**
 * @file kernels.dp.hpp
 * @brief SyclKittens kernels (rmsnorm, silu_mul, rotary_qk) made callable
 *        device-side from the C++ decode orchestrator.
 *
 * The kernel + launch bodies are copied VERBATIM from the standalone modules
 *   kernels/norm/rmsnorm.dp.cpp, kernels/activation/silu_mul.dp.cpp,
 *   kernels/rotary/rotary_qk.dp.cpp
 * (the only change: launch_rotary_qk drops its trailing queue.wait() so the
 * orchestrator keeps everything async on the torch stream). This lets the
 * orchestrator compose the SAME tile-DSL / wide-vector kernels as the .so
 * modules, without going through their Python dispatch wrappers.
 */
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#include "kittens.dp.hpp"
#pragma clang diagnostic pop
#include <cstdint>

using namespace kittens;
namespace decode_ops {
using kbf16 = kittens::bf16;

// ============================================================
// RMSNorm v6 — from kernels/norm/rmsnorm.dp.cpp (tile DSL: load_wide/store_wide)
// ============================================================
struct rmsnorm_globals_v6 {
    kbf16 *x; kbf16 *residual; kbf16 *o; kbf16 *o_resid; kbf16 *norm_w;
    float *inv_rms_out; int N; int B; float eps;
};
}  // namespace decode_ops
template <>
struct sycl::is_device_copyable<decode_ops::rmsnorm_globals_v6> : std::true_type {};
namespace decode_ops {

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
inline void rmsnorm_v6_kernel(const rmsnorm_globals_v6 g,
                              sycl::nd_item<3> item_ct1, float *slm) {
    constexpr int DC = D / WPR;
    using vec_c    = rv_fl<DC, kittens::ducks::rv_layout::naive>;
    using gl_chunk = gl<kbf16, 1, 1, 1, DC>;

    const int warp  = kittens::warpid();
    const int batch = item_ct1.get_group(1);
    const int seq   = item_ct1.get_group(2);
    const size_t row = static_cast<size_t>(batch) * g.N + seq;
    const size_t off = row * D + static_cast<size_t>(warp) * DC;
    const auto c0 = coord<vec_c>(0, 0, 0, 0);

    vec_c accum;
    gl_chunk x_gl(g.x + off, nullptr, nullptr, nullptr, nullptr);
    if constexpr (ADD_RESIDUAL) {
        vec_c x_vec, residual_vec;
        gl_chunk r_gl(g.residual + off, nullptr, nullptr, nullptr, nullptr);
        load_wide(x_vec, x_gl, c0);
        load_wide(residual_vec, r_gl, c0);
        add(accum, x_vec, residual_vec);
    } else {
        load_wide(accum, x_gl, c0);
    }
    if constexpr (WRITE_RESIDUAL) {
        gl_chunk or_gl(g.o_resid + off, nullptr, nullptr, nullptr, nullptr);
        store_wide(or_gl, accum, c0);
    }
    vec_c sq;
    mul(sq, accum, accum);
    const float partial = sum(sq);
    auto sg = item_ct1.get_sub_group();
    if (sg.get_local_linear_id() == 0) slm[warp] = partial;
    sycl::group_barrier(item_ct1.get_group());
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < WPR; ++i) total += slm[i];
    const float inv_rms = sycl::rsqrt(total / static_cast<float>(D) + g.eps);
    if (g.inv_rms_out != nullptr && warp == 0 && sg.get_local_linear_id() == 0)
        g.inv_rms_out[row] = inv_rms;
    gl_chunk w_gl(g.norm_w + static_cast<size_t>(warp) * DC,
                  nullptr, nullptr, nullptr, nullptr);
    vec_c gamma;
    load_wide(gamma, w_gl, c0);
    mul(accum, accum, inv_rms);
    mul(accum, accum, gamma);
    gl_chunk o_gl(g.o + off, nullptr, nullptr, nullptr, nullptr);
    store_wide(o_gl, accum, c0);
}

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
inline void launch_rmsnorm_v6(
    sycl::queue &queue,
    kbf16 *d_x, kbf16 *d_residual, kbf16 *d_norm_weight,
    kbf16 *d_o, kbf16 *d_o_resid, float *d_inv_rms,
    int B, int N, float eps)
{
    constexpr int NTHREADS = WPR * kittens::WARP_THREADS;
    sycl::range<3> grid(1, B, N);
    sycl::range<3> block(1, 1, NTHREADS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};
    rmsnorm_globals_v6 g{d_x, d_residual, d_o, d_o_resid, d_norm_weight,
                         d_inv_rms, N, B, eps};
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(WPR), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block), exp_props,
            [=](sycl::nd_item<3> item_ct1) [[intel::kernel_args_restrict]] {
                rmsnorm_v6_kernel<D, WPR, ADD_RESIDUAL, WRITE_RESIDUAL>(
                    g, item_ct1, &slm_acc[0]);
            });
    });
}

// ============================================================
// SiLU*mul — from kernels/activation/silu_mul.dp.cpp (wide bf16 vector kernel)
// ============================================================
struct silu_mul_globals {
    const uint16_t *gate; const uint16_t *up; uint16_t *out; size_t num_vec;
};
}  // namespace decode_ops
template <>
struct sycl::is_device_copyable<decode_ops::silu_mul_globals> : std::true_type {};
namespace decode_ops {

template<int SILU_VEC>
inline void silu_mul_kernel(const silu_mul_globals g, sycl::nd_item<1> it) {
    using bf16_t = sycl::ext::oneapi::bfloat16;
    using vu = sycl::vec<uint16_t, SILU_VEC>;
    const size_t v = it.get_global_linear_id();
    if (v >= g.num_vec) return;
    const vu *gate_v = reinterpret_cast<const vu *>(g.gate);
    const vu *up_v   = reinterpret_cast<const vu *>(g.up);
    vu       *out_v  = reinterpret_cast<vu *>(g.out);
    const vu gv = gate_v[v];
    const vu uv = up_v[v];
    vu ov;
    #pragma unroll
    for (int j = 0; j < SILU_VEC; ++j) {
        const float gf = static_cast<float>(sycl::bit_cast<bf16_t>(gv[j]));
        const float uf = static_cast<float>(sycl::bit_cast<bf16_t>(uv[j]));
        const float s = gf / (1.0f + sycl::exp(-gf));
        const float r = s * uf;
        ov[j] = sycl::bit_cast<uint16_t>(static_cast<bf16_t>(r));
    }
    out_v[v] = ov;
}

inline void launch_silu_mul(sycl::queue &queue, const uint16_t *gate,
                            const uint16_t *up, uint16_t *out, size_t total) {
    constexpr int SILU_VEC = 8;
    constexpr int SILU_WG = 256;
    const size_t num_vec = total / SILU_VEC;
    const size_t wg = SILU_WG;
    const size_t global = ((num_vec + wg - 1) / wg) * wg;
    silu_mul_globals g{gate, up, out, num_vec};
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                silu_mul_kernel<SILU_VEC>(g, it);
            });
    });
}

// ============================================================
// Rotary Q&K — from kernels/rotary/rotary_qk.dp.cpp (128-bit coalesced RoPE)
// ============================================================
namespace rotdetail {
struct alignas(16) wide8_t { std::uint32_t v[4]; };
inline void unpack8(const wide8_t &r, float (&o)[8]) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t a = r.v[k];
        o[2 * k]     = static_cast<float>(sycl::bit_cast<kbf16>(
                          static_cast<std::uint16_t>(a & 0xFFFF)));
        o[2 * k + 1] = static_cast<float>(sycl::bit_cast<kbf16>(
                          static_cast<std::uint16_t>(a >> 16)));
    }
}
inline wide8_t pack8(const float (&in)[8]) {
    wide8_t r;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint16_t lo = sycl::bit_cast<std::uint16_t>(static_cast<kbf16>(in[2 * k]));
        const std::uint16_t hi = sycl::bit_cast<std::uint16_t>(static_cast<kbf16>(in[2 * k + 1]));
        r.v[k] = (static_cast<std::uint32_t>(hi) << 16) | static_cast<std::uint32_t>(lo);
    }
    return r;
}
template<int D, bool NEGATE_SIN>
inline void rotary_row_chunk(const kbf16 *__restrict__ x, kbf16 *__restrict__ out,
                             const kbf16 *__restrict__ cos_ptr, const kbf16 *__restrict__ sin_ptr,
                             int heads, int N, long local_tid) {
    constexpr int HALF = D / 2;
    constexpr int CHUNKS = HALF / 8;
    const long row = local_tid / CHUNKS;
    const int chunk = static_cast<int>(local_tid % CHUNKS);
    const int coff = chunk * 8;
    const long row_off = row * D;
    const int n = static_cast<int>((row / heads) % N);
    const long cs_off = static_cast<long>(n) * HALF + coff;
    const wide8_t *xw = reinterpret_cast<const wide8_t *>(x);
    const wide8_t *cw = reinterpret_cast<const wide8_t *>(cos_ptr);
    const wide8_t *sw = reinterpret_cast<const wide8_t *>(sin_ptr);
    wide8_t *ow = reinterpret_cast<wide8_t *>(out);
    const wide8_t ra = xw[(row_off + coff) / 8];
    const wide8_t rb = xw[(row_off + HALF + coff) / 8];
    const wide8_t rc = cw[cs_off / 8];
    const wide8_t rs = sw[cs_off / 8];
    float a[8], b[8], c[8], s[8];
    unpack8(ra, a); unpack8(rb, b); unpack8(rc, c); unpack8(rs, s);
    float o1[8], o2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sv = NEGATE_SIN ? -s[i] : s[i];
        o1[i] = a[i] * c[i] - b[i] * sv;
        o2[i] = b[i] * c[i] + a[i] * sv;
    }
    ow[(row_off + coff) / 8] = pack8(o1);
    ow[(row_off + HALF + coff) / 8] = pack8(o2);
}
}  // namespace rotdetail

template<int D, bool NEGATE_SIN>
inline void rotary_qk_kernel(const kbf16 *__restrict__ q, kbf16 *__restrict__ q_out,
                             const kbf16 *__restrict__ k, kbf16 *__restrict__ k_out,
                             const kbf16 *__restrict__ cos_ptr, const kbf16 *__restrict__ sin_ptr,
                             int H, int HKV, int N, long q_threads, long total_threads,
                             sycl::nd_item<3> item) {
    const long tid = static_cast<long>(item.get_global_linear_id());
    if (tid >= total_threads) return;
    if (tid < q_threads)
        rotdetail::rotary_row_chunk<D, NEGATE_SIN>(q, q_out, cos_ptr, sin_ptr, H, N, tid);
    else
        rotdetail::rotary_row_chunk<D, NEGATE_SIN>(k, k_out, cos_ptr, sin_ptr, HKV, N, tid - q_threads);
}

template<int D, bool NEGATE_SIN>
inline void launch_rotary_qk(sycl::queue &queue,
                             const kbf16 *d_q, kbf16 *d_qout,
                             const kbf16 *d_k, kbf16 *d_kout,
                             const kbf16 *d_cos, const kbf16 *d_sin,
                             int B, int N, int H, int HKV) {
    constexpr int CHUNKS = (D / 2) / 8;
    constexpr int NTHREADS = 4 * kittens::WARP_THREADS;
    const long q_rows = static_cast<long>(B) * N * H;
    const long k_rows = static_cast<long>(B) * N * HKV;
    const long q_threads = q_rows * CHUNKS;
    const long total_threads = q_threads + k_rows * CHUNKS;
    const int block = NTHREADS;
    const long n_wgs = (total_threads + block - 1) / block;
    sycl::range<3> grid(1, 1, static_cast<size_t>(n_wgs));
    sycl::range<3> blk(1, 1, block);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * blk, blk), exp_props,
            [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {
                rotary_qk_kernel<D, NEGATE_SIN>(d_q, d_qout, d_k, d_kout, d_cos, d_sin,
                                                H, HKV, N, q_threads, total_threads, item);
            });
    });
    // NB: no queue.wait() (orchestrator stays async on the torch stream)
}

// ============================================================
// GEMV (M=1 decode) — bandwidth-optimal streaming reduction.
//   out[N] = W[N,K] @ in[K]      (bf16 in/out, fp32 accumulate, no cast)
//
// A single-token GEMV is a per-row dot product, structurally identical to the
// RMSNorm v6 per-row reduction above — so it reuses the same wide-load path
// (DPAS is the wrong tool: M=1 wastes the whole systolic array; the game is
// purely reading W at HBM peak). Layout:
//   * one work-group per output row n                 (grid = N work-groups)
//   * WPR subgroups split the K reduction (split-K).  Launched work = N*WPR*16,
//     independent of N, so the small-N shapes (Wo/Wdown) stay fully occupied —
//     that is where the plain oneMKL / 1-WG-per-row path is latency-bound.
//   * each subgroup streams its K-slice in CHUNK-wide pieces with 128-bit/lane
//     `load_wide`, software-prefetching the next chunk, fp32 accumulate.
//   * subgroup `sum` + tiny SLM combine over WPR partials -> one bf16 store.
// Requires K % (WPR*CHUNK) == 0 (caller guards; else falls back to oneMKL).
// ============================================================
struct gemv_globals { kbf16 *in; kbf16 *W; kbf16 *out; int K; };
}  // namespace decode_ops
template <>
struct sycl::is_device_copyable<decode_ops::gemv_globals> : std::true_type {};
namespace decode_ops {

template<int WPR, int CHUNK>
inline void wide_gemv_kernel(const gemv_globals g, sycl::nd_item<3> item,
                           float *slm) {
    using vec_c    = rv_fl<CHUNK, kittens::ducks::rv_layout::naive>;
    using gl_chunk = gl<kbf16, 1, 1, 1, CHUNK>;
    const int warp = kittens::warpid();               // subgroup id in [0,WPR)
    const int n    = item.get_group(2);               // output row
    const int K    = g.K;
    const int DC   = K / WPR;                          // K-slice per subgroup
    const size_t wbase = static_cast<size_t>(n) * K + static_cast<size_t>(warp) * DC;
    const size_t xbase = static_cast<size_t>(warp) * DC;
    const auto c0 = coord<vec_c>(0, 0, 0, 0);

    vec_c acc; zero(acc);
    // software-pipelined stream over the K-slice: prefetch chunk c+1 while the
    // fp32 FMA of chunk c is in flight (latency coverage, like the GEMM PF path)
    vec_c wcur, xcur;
    {
        gl_chunk wg(g.W  + wbase, nullptr, nullptr, nullptr, nullptr);
        gl_chunk xg(g.in + xbase, nullptr, nullptr, nullptr, nullptr);
        load_wide(wcur, wg, c0);
        load_wide(xcur, xg, c0);
    }
    #pragma unroll 2
    for (int c = 0; c < DC; c += CHUNK) {
        vec_c w = wcur, x = xcur;
        const int cn = c + CHUNK;
        if (cn < DC) {
            gl_chunk wg(g.W  + wbase + cn, nullptr, nullptr, nullptr, nullptr);
            gl_chunk xg(g.in + xbase + cn, nullptr, nullptr, nullptr, nullptr);
            load_wide(wcur, wg, c0);
            load_wide(xcur, xg, c0);
        }
        mul(w, w, x);            // w = W_chunk * x_chunk
        add(acc, acc, w);        // acc += w   (fp32, per-lane partials)
    }
    const float partial = sum(acc);                   // subgroup reduce -> scalar
    auto sg = item.get_sub_group();
    if (sg.get_local_linear_id() == 0) slm[warp] = partial;
    sycl::group_barrier(item.get_group());
    if (warp == 0 && sg.get_local_linear_id() == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < WPR; ++i) total += slm[i];
        g.out[n] = static_cast<kbf16>(total);         // single bf16 round
    }
}

template<int WPR, int CHUNK>
inline void launch_wide_gemv(sycl::queue &queue, kbf16 *in, kbf16 *W,
                           kbf16 *out, int K, int N) {
    constexpr int NTHREADS = WPR * kittens::WARP_THREADS;
    sycl::range<3> grid(1, 1, static_cast<size_t>(N));
    sycl::range<3> block(1, 1, NTHREADS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};
    gemv_globals g{in, W, out, K};
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(WPR), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block), exp_props,
            [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {
                wide_gemv_kernel<WPR, CHUNK>(g, item, &slm_acc[0]);
            });
    });
}

}  // namespace decode_ops
