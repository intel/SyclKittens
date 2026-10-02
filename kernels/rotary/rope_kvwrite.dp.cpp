/**
 * @file rope_kvwrite.dp.cpp
 * @brief SyclKittens fused RoPE-K + KV-cache write pybind11 wrapper.
 *
 * Fuses three per-layer decode/prefill steps into ONE SYCL kernel:
 *   1. RoPE applied to K            (gather-on-read of the (B,N,H,D) K tensor)
 *   2. transposed store of rot-K    into the (B,H,S,D) K cache at cache_pos
 *   3. transposed store of V        into the (B,H,S,D) V cache at cache_pos
 *
 * The unfused path is:
 *     k = rotary(k, cos, sin)                    # (B,N,H,D) rotary
 *     k_bhnd = k.transpose(1, 2)                 # (B,H,N,D) copy
 *     v_bhnd = v.transpose(1, 2)                 # (B,H,N,D) copy
 *     past_kv.update(k_bhnd, v_bhnd, ...)        # scatter into cache
 * i.e. one rotary launch + two full (B,N,H,D)->(B,H,N,D) transpose+copies per
 * layer. This kernel does all of it in a single pass with zero intermediate
 * tensors: RoPE math is fused with the transposed strided store.
 *
 * The read side is identical to rotary.dp.cpp (128-bit coalesced chunks so
 * every HBM transaction is line-rate). The write side targets the cache's
 * transposed (B,H,S,D) layout: each destination row is still contiguous in D,
 * so every store remains a single 16-byte coalesced transaction — only the
 * row base is scattered, which the transpose fundamentally requires.
 *
 * Exposed:
 *   dispatch_rope_kvwrite(k, v, cos, sin, k_cache, v_cache, cache_pos) -> ()
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#include "kittens.dp.hpp"
#pragma clang diagnostic pop

#include "pyutils/torch_helpers.dp.hpp"
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

#ifndef ROTARY_D
#define ROTARY_D 128
#endif

#ifndef ROTARY_NUM_WORKERS
#define ROTARY_NUM_WORKERS 4
#endif

#define ROTARY_NUM_THREADS (ROTARY_NUM_WORKERS * kittens::WARP_THREADS)

using namespace kittens;
using bf16 = kittens::bf16;

// ============================================================
// 128-bit coalesced packing (mirrors rotary.dp.cpp)
// ============================================================
// Each work-item owns one contiguous 8-element (16-byte) bf16 chunk of the
// first half and its paired chunk of the second half, so every K/V/cos/sin
// load and every cache store is a single 128-bit transaction.

namespace {
/// 16-byte aligned quad-word == 8 contiguous bf16, one 128-bit transaction.
struct alignas(16) wide8_t { std::uint32_t v[4]; };

inline void unpack8(const wide8_t &r, float (&o)[8]) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t a = r.v[k];
        o[2 * k]     = static_cast<float>(sycl::bit_cast<bf16>(
                          static_cast<std::uint16_t>(a & 0xFFFF)));
        o[2 * k + 1] = static_cast<float>(sycl::bit_cast<bf16>(
                          static_cast<std::uint16_t>(a >> 16)));
    }
}

inline wide8_t pack8(const float (&in)[8]) {
    wide8_t r;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint16_t lo = sycl::bit_cast<std::uint16_t>(
            static_cast<bf16>(in[2 * k]));
        const std::uint16_t hi = sycl::bit_cast<std::uint16_t>(
            static_cast<bf16>(in[2 * k + 1]));
        r.v[k] = (static_cast<std::uint32_t>(hi) << 16)
               |  static_cast<std::uint32_t>(lo);
    }
    return r;
}
} // namespace

// ============================================================
// Kernel: fused RoPE-K + transposed KV-cache write (128-bit coalesced)
// ============================================================
// k, v:      (B, N, H, D) bf16 contiguous          (input, H == H_KV)
// cos, sin:  (N, D/2) bf16
// k_cache:   (B, H, Sk, D) bf16 contiguous          (output, transposed layout)
// v_cache:   (B, H, Sv, D) bf16 contiguous          (output, transposed layout)
// cache_pos: starting seq offset in the cache
//
// One work-item == one (row, 8-element chunk), row == (b,n,h) in the K/V input
// layout. Reads x1/x2 chunks (input, contiguous), applies RoPE to K, and stores
// rotated K and copied V into the cache at seq index (cache_pos + n). D must be
// a multiple of 16.

template<int D>
void rope_kvwrite_kernel(
    const bf16 * __restrict__ k,
    const bf16 * __restrict__ v,
    const bf16 * __restrict__ cos_ptr,
    const bf16 * __restrict__ sin_ptr,
    bf16 * __restrict__ k_cache,
    bf16 * __restrict__ v_cache,
    int H, int N, long Sk, long Sv, long cache_pos, long total_threads,
    sycl::nd_item<3> item
) {
    constexpr int HALF   = D / 2;
    constexpr int CHUNKS = HALF / 8;          // 8-element chunks per half

    const long tid = static_cast<long>(item.get_global_linear_id());
    if (tid >= total_threads) return;

    const long row   = tid / CHUNKS;
    const int  chunk = static_cast<int>(tid % CHUNKS);
    const int  coff  = chunk * 8;             // bf16 offset within a half

    // Decompose the input row into (b, n, h): row = ((b*N)+n)*H + h.
    const long NH  = static_cast<long>(N) * H;
    const long b   = row / NH;
    const long rem = row - b * NH;
    const int  n   = static_cast<int>(rem / H);
    const int  h   = static_cast<int>(rem - static_cast<long>(n) * H);

    // Input row base (contiguous (B,N,H,D)).
    const long row_off = row * D;
    // cos/sin row base (contiguous (N,D/2)).
    const long cs_off  = static_cast<long>(n) * HALF + coff;
    // Transposed cache row bases: seq index == cache_pos + n.
    const long s        = cache_pos + n;
    const long kc_off   = ((b * H + h) * Sk + s) * D;
    const long vc_off   = ((b * H + h) * Sv + s) * D;

    const wide8_t *kw  = reinterpret_cast<const wide8_t *>(k);
    const wide8_t *vw  = reinterpret_cast<const wide8_t *>(v);
    const wide8_t *cw  = reinterpret_cast<const wide8_t *>(cos_ptr);
    const wide8_t *sw  = reinterpret_cast<const wide8_t *>(sin_ptr);
    wide8_t       *kcw = reinterpret_cast<wide8_t *>(k_cache);
    wide8_t       *vcw = reinterpret_cast<wide8_t *>(v_cache);

    // ---- K: gather-on-read RoPE ----
    // 16-byte coalesced loads: x1 chunk, x2 chunk, cos chunk, sin chunk.
    const wide8_t ka = kw[(row_off + coff) / 8];
    const wide8_t kb = kw[(row_off + HALF + coff) / 8];
    const wide8_t rc = cw[cs_off / 8];
    const wide8_t rs = sw[cs_off / 8];

    float a[8], bb[8], c[8], sn[8];
    unpack8(ka, a); unpack8(kb, bb); unpack8(rc, c); unpack8(rs, sn);

    float o1[8], o2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        o1[i] = a[i] * c[i] - bb[i] * sn[i];
        o2[i] = bb[i] * c[i] + a[i] * sn[i];
    }

    // ---- K: strided store into transposed cache ----
    kcw[(kc_off + coff) / 8]        = pack8(o1);
    kcw[(kc_off + HALF + coff) / 8] = pack8(o2);

    // ---- V: pure layout-transposing copy (no rotary, raw chunk move) ----
    const wide8_t va = vw[(row_off + coff) / 8];
    const wide8_t vb = vw[(row_off + HALF + coff) / 8];
    vcw[(vc_off + coff) / 8]        = va;
    vcw[(vc_off + HALF + coff) / 8] = vb;
}

// ============================================================
// Launch helper
// ============================================================

template<int D>
void launch_rope_kvwrite(
    sycl::queue &queue,
    const bf16 *d_k, const bf16 *d_v,
    const bf16 *d_cos, const bf16 *d_sin,
    bf16 *d_kc, bf16 *d_vc,
    int B, int N, int H, long Sk, long Sv, long cache_pos
) {
    constexpr int CHUNKS = (D / 2) / 8;
    const long total_rows    = static_cast<long>(B) * N * H;
    const long total_threads = total_rows * CHUNKS;
    const int  block = ROTARY_NUM_THREADS;
    const long n_wgs = (total_threads + block - 1) / block;

    sycl::range<3> grid(1, 1, static_cast<size_t>(n_wgs));
    sycl::range<3> blk(1, 1, block);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * blk, blk),
            exp_props,
            [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {
                rope_kvwrite_kernel<D>(
                    d_k, d_v, d_cos, d_sin, d_kc, d_vc,
                    H, N, Sk, Sv, cache_pos, total_threads, item);
            });
    });
    queue.wait();
}

// ============================================================
// PyTorch dispatch
// ============================================================

void dispatch_rope_kvwrite(
    torch::Tensor k,          // (B, N, H, D) bf16
    torch::Tensor v,          // (B, N, H, D) bf16
    torch::Tensor cos,        // (N, D/2) bf16
    torch::Tensor sin,        // (N, D/2) bf16
    torch::Tensor k_cache,    // (B, H, Sk, D) bf16   (written in place)
    torch::Tensor v_cache,    // (B, H, Sv, D) bf16   (written in place)
    int64_t cache_pos         // starting seq offset in the cache
) {
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);

    const int B = k.size(0);
    const int N = k.size(1);
    const int H = k.size(2);
    const int D = k.size(3);
    TORCH_CHECK(D == ROTARY_D, "Expected D=", ROTARY_D, " but got D=", D);
    TORCH_CHECK(v.size(0) == B && v.size(1) == N && v.size(2) == H &&
                v.size(3) == D, "v must match k shape (B,N,H,D)");
    TORCH_CHECK(cos.size(0) >= N, "cos seq dim too small");
    TORCH_CHECK(sin.size(0) >= N, "sin seq dim too small");
    TORCH_CHECK(cos.size(1) == D / 2, "cos last dim must be D/2");
    TORCH_CHECK(sin.size(1) == D / 2, "sin last dim must be D/2");

    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(1) == H &&
                k_cache.size(3) == D, "k_cache must be (B,H,Sk,D)");
    TORCH_CHECK(v_cache.size(0) == B && v_cache.size(1) == H &&
                v_cache.size(3) == D, "v_cache must be (B,H,Sv,D)");

    const long Sk = k_cache.size(2);
    const long Sv = v_cache.size(2);
    TORCH_CHECK(cache_pos >= 0, "cache_pos must be >= 0");
    TORCH_CHECK(cache_pos + N <= Sk, "cache_pos + N exceeds K cache capacity");
    TORCH_CHECK(cache_pos + N <= Sv, "cache_pos + N exceeds V cache capacity");

    auto stream = c10::xpu::getCurrentXPUStream(k.device().index());
    auto &queue = stream.queue();

    const bf16 *d_k   = reinterpret_cast<const bf16*>(k.data_ptr<c10::BFloat16>());
    const bf16 *d_v   = reinterpret_cast<const bf16*>(v.data_ptr<c10::BFloat16>());
    const bf16 *d_cos = reinterpret_cast<const bf16*>(cos.data_ptr<c10::BFloat16>());
    const bf16 *d_sin = reinterpret_cast<const bf16*>(sin.data_ptr<c10::BFloat16>());
    bf16       *d_kc  = reinterpret_cast<bf16*>(k_cache.data_ptr<c10::BFloat16>());
    bf16       *d_vc  = reinterpret_cast<bf16*>(v_cache.data_ptr<c10::BFloat16>());

    launch_rope_kvwrite<ROTARY_D>(
        queue, d_k, d_v, d_cos, d_sin, d_kc, d_vc,
        B, N, H, Sk, Sv, static_cast<long>(cache_pos));
}

// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens fused RoPE-K + KV-cache write (SYCL)";
    m.def("dispatch_rope_kvwrite", &dispatch_rope_kvwrite,
          "Fused RoPE-K + transposed KV-cache write. "
          "k,v (bf16 [B,N,H,D]); cos,sin (bf16 [N,D/2]); "
          "k_cache,v_cache (bf16 [B,H,S,D], written in place at cache_pos).",
          py::arg("k"), py::arg("v"), py::arg("cos"), py::arg("sin"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("cache_pos"));
}
