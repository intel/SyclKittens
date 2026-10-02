/**
 * @file rotary.dp.cpp
 * @brief SyclKittens Rotary Positional Embedding pybind11 wrapper.
 *
 * Simple SYCL kernel for RoPE. Each sub-group processes one row (b,n,h).
 * With D=128 and SG=16, each lane handles 4 element pairs.
 *
 * The rotary transform is its own inverse when sin is negated, so the
 * backward pass reuses the same kernel with -sin.
 *
 * Exposed:
 *   dispatch_rotary(x, cos, sin) -> out        (forward)
 *   dispatch_rotary_backward(dy, cos, sin) -> dx  (backward = rotary with -sin)
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
// 128-bit coalesced packing (mirrors kittens::load_wide / store_wide)
// ============================================================
// The original kernel issued scalar 16-bit bf16 loads in a strided lane loop,
// which caps HBM utilisation at ~460 GB/s (~half roofline) on PVC. The fix is
// the same one that took RMSNorm/LayerNorm v6 to line-rate: issue a single
// 128-bit (8xbf16 = 16-byte) coalesced transaction per lane. RoPE pairs element
// d with d+half_d across the two halves, so each work-item owns one contiguous
// 8-element chunk of x1 and its paired 8-element chunk of x2; both chunks (and
// their cos/sin) are 16-byte wide loads, the rotary math is local, and the two
// results are 16-byte wide stores. Every x/out transaction is fully coalesced.

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
// Kernel: element-wise rotary embedding (128-bit coalesced)
// ============================================================
// x:   (B, N, H, D) bf16 contiguous
// cos: (N, D/2) bf16
// sin: (N, D/2) bf16
// out: (B, N, H, D) bf16
// One work-item == one (row, 8-element chunk). D must be a multiple of 16.

template<int D, bool NEGATE_SIN>
void rotary_kernel(
    const bf16 * __restrict__ x,
    bf16 * __restrict__ out,
    const bf16 * __restrict__ cos_ptr,
    const bf16 * __restrict__ sin_ptr,
    int H, int N, long total_threads,
    sycl::nd_item<3> item
) {
    constexpr int HALF   = D / 2;
    constexpr int CHUNKS = HALF / 8;          // 8-element chunks per half

    const long tid = static_cast<long>(item.get_global_linear_id());
    if (tid >= total_threads) return;

    const long row   = tid / CHUNKS;
    const int  chunk = static_cast<int>(tid % CHUNKS);
    const int  coff  = chunk * 8;             // bf16 offset within a half
    const long row_off = row * D;
    const int  n = static_cast<int>((row / H) % N);
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

// ============================================================
// Launch helper
// ============================================================

template<int D, bool NEGATE_SIN>
void launch_rotary(
    sycl::queue &queue,
    const bf16 *d_x, bf16 *d_out,
    const bf16 *d_cos, const bf16 *d_sin,
    int B, int N, int H
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
                rotary_kernel<D, NEGATE_SIN>(
                    d_x, d_out, d_cos, d_sin, H, N, total_threads, item);
            });
    });
    queue.wait();
}

// ============================================================
// PyTorch dispatch
// ============================================================

torch::Tensor dispatch_rotary(
    torch::Tensor x,    // (B, N, H, D) bf16
    torch::Tensor cos,  // (N, D/2) bf16
    torch::Tensor sin   // (N, D/2) bf16
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B = x.size(0);
    const int N = x.size(1);
    const int H = x.size(2);
    const int D = x.size(3);
    TORCH_CHECK(D == ROTARY_D, "Expected D=", ROTARY_D, " but got D=", D);
    TORCH_CHECK(cos.size(0) >= N, "cos seq dim too small");
    TORCH_CHECK(sin.size(0) >= N, "sin seq dim too small");
    TORCH_CHECK(cos.size(1) == D / 2, "cos last dim must be D/2");

    auto out = torch::empty_like(x);

    auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
    auto &queue = stream.queue();

    const bf16 *d_x   = reinterpret_cast<const bf16*>(x.data_ptr<c10::BFloat16>());
    bf16       *d_out = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    const bf16 *d_cos = reinterpret_cast<const bf16*>(cos.data_ptr<c10::BFloat16>());
    const bf16 *d_sin = reinterpret_cast<const bf16*>(sin.data_ptr<c10::BFloat16>());

    launch_rotary<ROTARY_D, false>(queue, d_x, d_out, d_cos, d_sin, B, N, H);

    return out;
}

torch::Tensor dispatch_rotary_backward(
    torch::Tensor dy,   // (B, N, H, D) bf16
    torch::Tensor cos,  // (N, D/2) bf16
    torch::Tensor sin   // (N, D/2) bf16
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B = dy.size(0);
    const int N = dy.size(1);
    const int H = dy.size(2);
    const int D = dy.size(3);
    TORCH_CHECK(D == ROTARY_D, "Expected D=", ROTARY_D, " but got D=", D);

    auto dx = torch::empty_like(dy);

    auto stream = c10::xpu::getCurrentXPUStream(dy.device().index());
    auto &queue = stream.queue();

    const bf16 *d_dy  = reinterpret_cast<const bf16*>(dy.data_ptr<c10::BFloat16>());
    bf16       *d_dx  = reinterpret_cast<bf16*>(dx.data_ptr<c10::BFloat16>());
    const bf16 *d_cos = reinterpret_cast<const bf16*>(cos.data_ptr<c10::BFloat16>());
    const bf16 *d_sin = reinterpret_cast<const bf16*>(sin.data_ptr<c10::BFloat16>());

    // Backward = rotate with -sin
    launch_rotary<ROTARY_D, true>(queue, d_dy, d_dx, d_cos, d_sin, B, N, H);

    return dx;
}

// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens Rotary Positional Embedding (SYCL)";
    m.def("dispatch_rotary", &dispatch_rotary,
          "RoPE forward: x (bf16 [B,N,H,D]), cos/sin (bf16 [N,D/2]) -> out",
          py::arg("x"), py::arg("cos"), py::arg("sin"));
    m.def("dispatch_rotary_backward", &dispatch_rotary_backward,
          "RoPE backward: dy (bf16 [B,N,H,D]), cos/sin (bf16 [N,D/2]) -> dx",
          py::arg("dy"), py::arg("cos"), py::arg("sin"));
}
