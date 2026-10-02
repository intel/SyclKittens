/**
 * @file rotary_qk.dp.cpp
 * @brief SyclKittens fused Q+K Rotary Positional Embedding pybind11 wrapper.
 *
 * Applies RoPE to BOTH Q (B,N,H,D) and K (B,N,HKV,D) with a SHARED cos/sin
 * (N, D/2) in a *single* SYCL launch, replacing two separate dispatch_rotary
 * calls (2 launches -> 1). This halves launch/queue overhead and lets Q and K
 * head-rows share the same in-flight cos/sin cache lines.
 *
 * Design: one nd_range whose linear id space is the concatenation of Q's and
 * K's (row, 8-elem chunk) work items. Work items with tid < q_threads rotate a
 * Q row; the rest rotate a K row (with tid rebased). Both branches call the
 * exact same per-row RoPE worker (rotary_row_chunk) — identical math to
 * rotary.dp.cpp, ROTARY_D=128, 128-bit coalesced loads/stores.
 *
 * Exposed:
 *   dispatch_rotary_qk(q, k, cos, sin) -> (q_out, k_out)   (forward)
 *
 * The rotary transform is its own inverse when sin is negated, so a backward
 * variant is provided too (dispatch_rotary_qk_backward) for symmetry/reuse.
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
#include <tuple>

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

// ------------------------------------------------------------
// Per-row RoPE worker (shared by Q and K sub-grids)
// ------------------------------------------------------------
// Handles ONE (row, 8-element chunk) of a (.,N,heads,D) tensor. `heads` is H
// for the Q sub-grid and HKV for the K sub-grid; `local_tid` is rebased to the
// tensor so the row/chunk/seq-index arithmetic is identical to rotary.dp.cpp. cos
// and sin are shared (N, D/2) and indexed only by the sequence position n, so
// the Q and K sub-grids read the same cos/sin lines for a given (b, n).
template<int D, bool NEGATE_SIN>
inline void rotary_row_chunk(
    const bf16 * __restrict__ x,
    bf16 * __restrict__ out,
    const bf16 * __restrict__ cos_ptr,
    const bf16 * __restrict__ sin_ptr,
    int heads, int N, long local_tid
) {
    constexpr int HALF   = D / 2;
    constexpr int CHUNKS = HALF / 8;          // 8-element chunks per half

    const long row   = local_tid / CHUNKS;
    const int  chunk = static_cast<int>(local_tid % CHUNKS);
    const int  coff  = chunk * 8;             // bf16 offset within a half
    const long row_off = row * D;
    const int  n = static_cast<int>((row / heads) % N);
    const long cs_off = static_cast<long>(n) * HALF + coff;

    const wide8_t *xw = reinterpret_cast<const wide8_t *>(x);
    const wide8_t *cw = reinterpret_cast<const wide8_t *>(cos_ptr);
    const wide8_t *sw = reinterpret_cast<const wide8_t *>(sin_ptr);
    wide8_t       *ow = reinterpret_cast<wide8_t *>(out);

    // 16-byte coalesced loads: x1 chunk, x2 chunk, cos chunk, sin chunk
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

    // 16-byte coalesced stores
    ow[(row_off + coff) / 8]        = pack8(o1);
    ow[(row_off + HALF + coff) / 8] = pack8(o2);
}
} // namespace

// ============================================================
// Fused kernel: rotary over the concatenation of Q and K work items
// ============================================================
// q:   (B, N, H,   D) bf16     q_out: (B, N, H,   D) bf16
// k:   (B, N, HKV, D) bf16     k_out: (B, N, HKV, D) bf16
// cos: (N, D/2) bf16           sin:   (N, D/2) bf16   (shared)
// tid in [0, q_threads)                  -> a Q (row, chunk)
// tid in [q_threads, total_threads)      -> a K (row, chunk), rebased
template<int D, bool NEGATE_SIN>
void rotary_qk_kernel(
    const bf16 * __restrict__ q,
    bf16 * __restrict__ q_out,
    const bf16 * __restrict__ k,
    bf16 * __restrict__ k_out,
    const bf16 * __restrict__ cos_ptr,
    const bf16 * __restrict__ sin_ptr,
    int H, int HKV, int N,
    long q_threads, long total_threads,
    sycl::nd_item<3> item
) {
    const long tid = static_cast<long>(item.get_global_linear_id());
    if (tid >= total_threads) return;

    if (tid < q_threads) {
        rotary_row_chunk<D, NEGATE_SIN>(q, q_out, cos_ptr, sin_ptr, H, N, tid);
    } else {
        rotary_row_chunk<D, NEGATE_SIN>(
            k, k_out, cos_ptr, sin_ptr, HKV, N, tid - q_threads);
    }
}

// ============================================================
// Launch helper
// ============================================================

template<int D, bool NEGATE_SIN>
void launch_rotary_qk(
    sycl::queue &queue,
    const bf16 *d_q, bf16 *d_qout,
    const bf16 *d_k, bf16 *d_kout,
    const bf16 *d_cos, const bf16 *d_sin,
    int B, int N, int H, int HKV
) {
    constexpr int CHUNKS = (D / 2) / 8;
    const long q_rows       = static_cast<long>(B) * N * H;
    const long k_rows       = static_cast<long>(B) * N * HKV;
    const long q_threads    = q_rows * CHUNKS;
    const long k_threads    = k_rows * CHUNKS;
    const long total_threads = q_threads + k_threads;
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
                rotary_qk_kernel<D, NEGATE_SIN>(
                    d_q, d_qout, d_k, d_kout, d_cos, d_sin,
                    H, HKV, N, q_threads, total_threads, item);
            });
    });
    queue.wait();
}

// ============================================================
// PyTorch dispatch
// ============================================================

std::tuple<torch::Tensor, torch::Tensor> dispatch_rotary_qk(
    torch::Tensor q,    // (B, N, H,   D) bf16
    torch::Tensor k,    // (B, N, HKV, D) bf16
    torch::Tensor cos,  // (N, D/2) bf16
    torch::Tensor sin   // (N, D/2) bf16
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B   = q.size(0);
    const int N   = q.size(1);
    const int H   = q.size(2);
    const int D   = q.size(3);
    const int HKV = k.size(2);

    TORCH_CHECK(D == ROTARY_D, "Expected D=", ROTARY_D, " but got D=", D);
    TORCH_CHECK(k.size(0) == B, "q/k batch mismatch");
    TORCH_CHECK(k.size(1) == N, "q/k seq mismatch");
    TORCH_CHECK(k.size(3) == D, "q/k head_dim mismatch");
    TORCH_CHECK(cos.size(0) >= N, "cos seq dim too small");
    TORCH_CHECK(sin.size(0) >= N, "sin seq dim too small");
    TORCH_CHECK(cos.size(1) == D / 2, "cos last dim must be D/2");
    TORCH_CHECK(sin.size(1) == D / 2, "sin last dim must be D/2");

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);

    auto stream = c10::xpu::getCurrentXPUStream(q.device().index());
    auto &queue = stream.queue();

    const bf16 *d_q    = reinterpret_cast<const bf16*>(q.data_ptr<c10::BFloat16>());
    bf16       *d_qout = reinterpret_cast<bf16*>(q_out.data_ptr<c10::BFloat16>());
    const bf16 *d_k    = reinterpret_cast<const bf16*>(k.data_ptr<c10::BFloat16>());
    bf16       *d_kout = reinterpret_cast<bf16*>(k_out.data_ptr<c10::BFloat16>());
    const bf16 *d_cos  = reinterpret_cast<const bf16*>(cos.data_ptr<c10::BFloat16>());
    const bf16 *d_sin  = reinterpret_cast<const bf16*>(sin.data_ptr<c10::BFloat16>());

    launch_rotary_qk<ROTARY_D, false>(
        queue, d_q, d_qout, d_k, d_kout, d_cos, d_sin, B, N, H, HKV);

    return std::make_tuple(q_out, k_out);
}

std::tuple<torch::Tensor, torch::Tensor> dispatch_rotary_qk_backward(
    torch::Tensor dq,   // (B, N, H,   D) bf16
    torch::Tensor dk,   // (B, N, HKV, D) bf16
    torch::Tensor cos,  // (N, D/2) bf16
    torch::Tensor sin   // (N, D/2) bf16
) {
    CHECK_INPUT(dq);
    CHECK_INPUT(dk);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B   = dq.size(0);
    const int N   = dq.size(1);
    const int H   = dq.size(2);
    const int D   = dq.size(3);
    const int HKV = dk.size(2);

    TORCH_CHECK(D == ROTARY_D, "Expected D=", ROTARY_D, " but got D=", D);
    TORCH_CHECK(dk.size(0) == B && dk.size(1) == N && dk.size(3) == D,
                "dq/dk shape mismatch");

    auto dq_out = torch::empty_like(dq);
    auto dk_out = torch::empty_like(dk);

    auto stream = c10::xpu::getCurrentXPUStream(dq.device().index());
    auto &queue = stream.queue();

    const bf16 *d_dq    = reinterpret_cast<const bf16*>(dq.data_ptr<c10::BFloat16>());
    bf16       *d_dqout = reinterpret_cast<bf16*>(dq_out.data_ptr<c10::BFloat16>());
    const bf16 *d_dk    = reinterpret_cast<const bf16*>(dk.data_ptr<c10::BFloat16>());
    bf16       *d_dkout = reinterpret_cast<bf16*>(dk_out.data_ptr<c10::BFloat16>());
    const bf16 *d_cos   = reinterpret_cast<const bf16*>(cos.data_ptr<c10::BFloat16>());
    const bf16 *d_sin   = reinterpret_cast<const bf16*>(sin.data_ptr<c10::BFloat16>());

    // Backward = rotate with -sin
    launch_rotary_qk<ROTARY_D, true>(
        queue, d_dq, d_dqout, d_dk, d_dkout, d_cos, d_sin, B, N, H, HKV);

    return std::make_tuple(dq_out, dk_out);
}

// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens Fused Q+K Rotary Positional Embedding (SYCL)";
    m.def("dispatch_rotary_qk", &dispatch_rotary_qk,
          "Fused RoPE forward: q (bf16 [B,N,H,D]), k (bf16 [B,N,HKV,D]), "
          "cos/sin (bf16 [N,D/2]) -> (q_out, k_out)",
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"));
    m.def("dispatch_rotary_qk_backward", &dispatch_rotary_qk_backward,
          "Fused RoPE backward: dq (bf16 [B,N,H,D]), dk (bf16 [B,N,HKV,D]), "
          "cos/sin (bf16 [N,D/2]) -> (dq_out, dk_out)",
          py::arg("dq"), py::arg("dk"), py::arg("cos"), py::arg("sin"));
}
