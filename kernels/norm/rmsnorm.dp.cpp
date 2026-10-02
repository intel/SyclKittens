/**
 * @file rmsnorm.dp.cpp
 * @brief SyclKittens RMSNorm v6 with multi-subgroup tiling expressed with
 *        the SyclKittens DSL, using the new 128-bit `load_wide`/`store_wide` ops.
 *
 * This is the "productized" form of the v5 hand-tuned kernel: v5 proved that
 * 128-bit coalesced loads are needed for full bandwidth, but did the
 * packing with raw `u128` pointer math inside the kernel. v6 keeps the exact v4
 * control flow (WPR subgroups per row, SLM cross-subgroup reduction) and simply
 * swaps `load`/`store` for `kittens::load_wide`/`kittens::store_wide`, so the
 * wide-load optimization now lives in the DSL and is reusable by any bandwidth-
 * bound vector kernel (see include/ops/warp/memory/vec/global_to_register.dp.hpp).
 *
 * Compile with:
 *   -DRMSNORM_D=2048            (hidden dim; DC = D/WPR must be a multiple of 128)
 *   -DRMSNORM_WARPS_PER_ROW=8   (subgroups cooperating per row)
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

#ifndef RMSNORM_D
#define RMSNORM_D 2048
#endif

#ifndef RMSNORM_WARPS_PER_ROW
#define RMSNORM_WARPS_PER_ROW 8
#endif

#define RMSNORM_NUM_THREADS (RMSNORM_WARPS_PER_ROW * kittens::WARP_THREADS)

using namespace kittens;
using bf16 = kittens::bf16;

// ============================================================
// Globals struct (device-copyable)
// ============================================================

struct rmsnorm_globals_v6 {
    bf16 *x;
    bf16 *residual;
    bf16 *o;
    bf16 *o_resid;
    bf16 *norm_w;
    float *inv_rms_out;   // (B*N,) for backward pass; nullptr to skip
    int N;
    int B;
    float eps;
};
template <>
struct sycl::is_device_copyable<rmsnorm_globals_v6> : std::true_type {};

// ============================================================
// Kernel — v4 flow, DSL 128-bit load_wide/store_wide
// ============================================================

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
inline void rmsnorm_v6_kernel(const rmsnorm_globals_v6 g,
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

    // ---- partial sum of squares over this chunk (subgroup all-reduce) ----
    vec_c sq;
    mul(sq, accum, accum);
    const float partial = sum(sq);

    // ---- combine partials across subgroups via SLM ----
    auto sg = item_ct1.get_sub_group();
    if (sg.get_local_linear_id() == 0) slm[warp] = partial;
    sycl::group_barrier(item_ct1.get_group());

    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < WPR; ++i) total += slm[i];
    const float inv_rms = sycl::rsqrt(total / static_cast<float>(D) + g.eps);

    if (g.inv_rms_out != nullptr && warp == 0 && sg.get_local_linear_id() == 0)
        g.inv_rms_out[row] = inv_rms;

    // ---- scale by inv_rms * gamma_chunk and store (128-bit) ----
    gl_chunk w_gl(g.norm_w + static_cast<size_t>(warp) * DC,
                  nullptr, nullptr, nullptr, nullptr);
    vec_c gamma;
    load_wide(gamma, w_gl, c0);

    mul(accum, accum, inv_rms);
    mul(accum, accum, gamma);

    gl_chunk o_gl(g.o + off, nullptr, nullptr, nullptr, nullptr);
    store_wide(o_gl, accum, c0);
}

// ============================================================
// Launch
// ============================================================

template<int D, int WPR, bool ADD_RESIDUAL, bool WRITE_RESIDUAL>
void launch_rmsnorm_v6(
    sycl::queue &queue,
    bf16 *d_x, bf16 *d_residual, bf16 *d_norm_weight,
    bf16 *d_o, bf16 *d_o_resid, float *d_inv_rms,
    int B, int N, float eps
) {
    sycl::range<3> grid(1, B, N);
    sycl::range<3> block(1, 1, RMSNORM_NUM_THREADS);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>};

    rmsnorm_globals_v6 g{d_x, d_residual, d_o, d_o_resid, d_norm_weight,
                         d_inv_rms, N, B, eps};

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(WPR), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[intel::kernel_args_restrict]] {
                rmsnorm_v6_kernel<D, WPR, ADD_RESIDUAL, WRITE_RESIDUAL>(
                    g, item_ct1, &slm_acc[0]);
            });
    });
}

// ============================================================
// PyTorch dispatch functions
// ============================================================

std::vector<torch::Tensor> dispatch_rmsnorm(
    torch::Tensor x, torch::Tensor weight, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(D == RMSNORM_D, "Expected D=", RMSNORM_D, " but got D=", D);

    auto out = torch::empty_like(x);
    auto inv_rms = torch::empty({B, N}, torch::TensorOptions()
                                .dtype(torch::kFloat32).device(x.device()));

    auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
    auto &queue = stream.queue();

    bf16 *d_x = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16 *d_w = reinterpret_cast<bf16*>(weight.data_ptr<c10::BFloat16>());
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    float *d_inv_rms = inv_rms.data_ptr<float>();

    launch_rmsnorm_v6<RMSNORM_D, RMSNORM_WARPS_PER_ROW, false, false>(
        queue, d_x, nullptr, d_w, d_o, nullptr, d_inv_rms,
        B, N, static_cast<float>(eps));

    return {out, inv_rms};
}

std::vector<torch::Tensor> dispatch_rmsnorm_residual(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(D == RMSNORM_D, "Expected D=", RMSNORM_D, " but got D=", D);

    auto out = torch::empty_like(x);
    auto resid_out = torch::empty_like(x);
    auto inv_rms = torch::empty({B, N}, torch::TensorOptions()
                                .dtype(torch::kFloat32).device(x.device()));

    auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
    auto &queue = stream.queue();

    bf16 *d_x     = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16 *d_res   = reinterpret_cast<bf16*>(residual.data_ptr<c10::BFloat16>());
    bf16 *d_w     = reinterpret_cast<bf16*>(weight.data_ptr<c10::BFloat16>());
    bf16 *d_o     = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    bf16 *d_o_res = reinterpret_cast<bf16*>(resid_out.data_ptr<c10::BFloat16>());
    float *d_inv_rms = inv_rms.data_ptr<float>();

    launch_rmsnorm_v6<RMSNORM_D, RMSNORM_WARPS_PER_ROW, true, true>(
        queue, d_x, d_res, d_w, d_o, d_o_res, d_inv_rms,
        B, N, static_cast<float>(eps));

    return {out, resid_out, inv_rms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens RMSNorm v6 (DSL load_wide/store_wide, 128-bit)";
    m.def("dispatch_rmsnorm", &dispatch_rmsnorm,
          "RMSNorm: x (bf16 [B,N,D]) -> (out [B,N,D], inv_rms [B,N])",
          py::arg("x"), py::arg("weight"), py::arg("eps"));
    m.def("dispatch_rmsnorm_residual", &dispatch_rmsnorm_residual,
          "Fused add + RMSNorm: x,residual (bf16 [B,N,D]) -> (out, residual_out, inv_rms)",
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps"));
}
