/**
 * @file layernorm.dp.cpp
 * @brief SyclKittens LayerNorm v6 with cooperative multi-subgroup tiling expressed
 *        purely with the SyclKittens DSL, using the 128-bit `load_wide`/`store_wide` ops.
 *
 * This is the LayerNorm analog of rmsnorm.dp.cpp. LayerNorm v2 used ONE subgroup
 * per row with D/16 fp32 accumulators per lane (256 floats/lane at D=4096) which
 * spills the register file and caps bandwidth at ~570 GB/s. v6 borrows the v4/v6
 * RMSNorm control flow instead:
 *   - WPR subgroups COOPERATE on one row; each owns a DC = D/WPR contiguous chunk
 *     (DC/16 fp32/lane => no spill), loaded/stored with 128-bit coalesced ops.
 *   - LayerNorm needs BOTH the mean and the variance, so two partials (sum_x and
 *     sum_x2) are reduced across subgroups via a single 2*WPR-float SLM scratch.
 *   - out = (x - mean) * inv_std * gamma + beta.
 *
 * Exposes dispatch_layernorm / dispatch_layernorm_fused. eps is fixed at 1e-5f.
 *
 * Compile with:
 *   -DLN_D=4096                 (hidden dim; DC = D/WPR must be a multiple of 128)
 *   -DLN_WARPS_PER_ROW=8        (subgroups cooperating per row)
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

#ifndef LN_D
#define LN_D 4096
#endif

#ifndef LN_WARPS_PER_ROW
#define LN_WARPS_PER_ROW 8
#endif

#define LN_NUM_THREADS (LN_WARPS_PER_ROW * kittens::WARP_THREADS)
#define LN_EPS 1e-5f

using namespace kittens;
using bf16 = kittens::bf16;

// ============================================================
// Globals struct (device-copyable)
// ============================================================

struct layernorm_globals_v6 {
    bf16 *x;
    bf16 *residual;
    bf16 *o;
    bf16 *o_resid;
    bf16 *norm_w;      // gamma (D,)
    bf16 *norm_b;      // beta  (D,)
    int N;
    int B;
    float eps;
};
template <>
struct sycl::is_device_copyable<layernorm_globals_v6> : std::true_type {};

// ============================================================
// Kernel — v6 cooperative flow, DSL 128-bit load_wide/store_wide
// ============================================================

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
inline void layernorm_v6_kernel(const layernorm_globals_v6 g,
                                sycl::nd_item<3> item_ct1, float *slm) {
    constexpr int DC = D / WPR;                       // elements per subgroup
    using vec_c    = rv_fl<DC, kittens::ducks::rv_layout::naive>;
    using gl_chunk = gl<bf16, 1, 1, 1, DC>;

    const int warp  = kittens::warpid();              // 0 .. WPR-1
    const int batch = item_ct1.get_group(1);
    const int seq   = item_ct1.get_group(2);

    const size_t row = static_cast<size_t>(batch) * g.N + seq;
    const size_t off = row * D + static_cast<size_t>(warp) * DC;

    const auto c0 = coord<vec_c>(0, 0, 0, 0);

    // ---- load this subgroup's chunk via 128-bit coalesced loads ----
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

    // ---- partial sum and sum-of-squares over this chunk (subgroup reduce) ----
    vec_c sq;
    mul(sq, accum, accum);
    const float partial_sum = sum(accum);
    const float partial_sq  = sum(sq);

    // ---- combine partials across subgroups via SLM (2*WPR scratch) ----
    auto sg = item_ct1.get_sub_group();
    if (sg.get_local_linear_id() == 0) {
        slm[warp]       = partial_sum;
        slm[WPR + warp] = partial_sq;
    }
    sycl::group_barrier(item_ct1.get_group());

    float total_sum = 0.0f, total_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < WPR; ++i) {
        total_sum += slm[i];
        total_sq  += slm[WPR + i];
    }

    constexpr float inv_D = 1.0f / static_cast<float>(D);
    const float mean    = total_sum * inv_D;
    const float var     = total_sq * inv_D - mean * mean;
    const float inv_std = sycl::rsqrt(var + g.eps);

    // ---- out = (x - mean) * inv_std * gamma + beta ; store (128-bit) ----
    gl_chunk w_gl(g.norm_w + static_cast<size_t>(warp) * DC,
                  nullptr, nullptr, nullptr, nullptr);
    gl_chunk b_gl(g.norm_b + static_cast<size_t>(warp) * DC,
                  nullptr, nullptr, nullptr, nullptr);
    vec_c gamma, beta;
    load_wide(gamma, w_gl, c0);
    load_wide(beta,  b_gl, c0);

    sub(accum, accum, mean);
    mul(accum, accum, inv_std);
    mul(accum, accum, gamma);
    add(accum, accum, beta);

    gl_chunk o_gl(g.o + off, nullptr, nullptr, nullptr, nullptr);
    store_wide(o_gl, accum, c0);
}

// ============================================================
// Launch
// ============================================================

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
void launch_layernorm_v6(
    sycl::queue &queue,
    bf16 *d_x, bf16 *d_residual, bf16 *d_gamma, bf16 *d_beta,
    bf16 *d_o, bf16 *d_o_resid,
    int B, int N, float eps
) {
    sycl::range<3> grid(1, B, N);
    sycl::range<3> block(1, 1, LN_NUM_THREADS);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};

    layernorm_globals_v6 g{d_x, d_residual, d_o, d_o_resid, d_gamma, d_beta,
                           N, B, eps};

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(2 * WPR), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[intel::kernel_args_restrict]] {
                layernorm_v6_kernel<D, WPR, ADD_RESIDUAL, WRITE_RESIDUAL>(
                    g, item_ct1, &slm_acc[0]);
            });
    });
}

// ============================================================
// PyTorch dispatch functions
// ============================================================

torch::Tensor dispatch_layernorm(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(D == LN_D, "Expected D=", LN_D, " but got D=", D);
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");

    auto out = torch::empty_like(x);

    auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
    auto &queue = stream.queue();

    bf16 *d_x = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16 *d_w = reinterpret_cast<bf16*>(weight.data_ptr<c10::BFloat16>());
    bf16 *d_b = reinterpret_cast<bf16*>(bias.data_ptr<c10::BFloat16>());
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

    launch_layernorm_v6<LN_D, LN_WARPS_PER_ROW, false, false>(
        queue, d_x, nullptr, d_w, d_b, d_o, nullptr, B, N, LN_EPS);

    return out;
}

std::vector<torch::Tensor> dispatch_layernorm_fused(
    torch::Tensor x, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(D == LN_D, "Expected D=", LN_D, " but got D=", D);

    auto out = torch::empty_like(x);
    auto resid_out = torch::empty_like(x);

    auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
    auto &queue = stream.queue();

    bf16 *d_x     = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16 *d_res   = reinterpret_cast<bf16*>(residual.data_ptr<c10::BFloat16>());
    bf16 *d_w     = reinterpret_cast<bf16*>(weight.data_ptr<c10::BFloat16>());
    bf16 *d_b     = reinterpret_cast<bf16*>(bias.data_ptr<c10::BFloat16>());
    bf16 *d_o     = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    bf16 *d_o_res = reinterpret_cast<bf16*>(resid_out.data_ptr<c10::BFloat16>());

    launch_layernorm_v6<LN_D, LN_WARPS_PER_ROW, true, true>(
        queue, d_x, d_res, d_w, d_b, d_o, d_o_res, B, N, LN_EPS);

    return {out, resid_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens LayerNorm v6 (DSL load_wide/store_wide, 128-bit): D="
              + std::to_string(LN_D);
    m.def("dispatch_layernorm", &dispatch_layernorm,
          "Plain LayerNorm: x [B,N,D] bf16 -> out [B,N,D] bf16",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
    m.def("dispatch_layernorm_fused", &dispatch_layernorm_fused,
          "Fused Add+LayerNorm: (x + residual) -> LN, residual_out",
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("bias"));
}
