/**
 * @file silu_mul.dp.cpp
 * @brief SyclKittens fused SwiGLU activation: out = SiLU(gate) * up.
 *
 * Fuses the two elementwise ops that sit between up_proj and down_proj in a
 * SwiGLU MLP:
 *
 *     down_in = SiLU(gate) * up            (SiLU(x) = x * sigmoid(x))
 *
 * The unfused path materialises a (B, N, inter) temporary between the SiLU and
 * the multiply, costing a full HBM write + read of that (14336-wide) tensor.
 * This kernel streams gate + up straight to out in one pass, removing that
 * round-trip: 5 tensor-passes (silu: r gate / w tmp ; mul: r tmp / r up / w out)
 * collapse to 3 (r gate / r up / w out).
 *
 * Design: pure memory-bound elementwise. Flatten (B, N, inter) to a 1-D stream,
 * vectorised bf16 loads (SILU_VEC contiguous elements per work-item), fp32 SiLU
 * compute, bf16 store. Launch one work-item per vector with many small
 * work-groups so the driver can maximise EU occupancy.
 *
 * Exposed function:
 *   dispatch_silu_mul(gate, up) -> out      all (B, N, inter) bf16
 *
 * Compile-time knobs:
 *   -DSILU_VEC=8   (bf16 elements per work-item; must divide numel)
 *   -DSILU_WG=256  (work-group size in work-items)
 */

#include <sycl/sycl.hpp>

#include "pyutils/torch_helpers.dp.hpp"
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

#ifndef SILU_VEC
#define SILU_VEC 8
#endif

#ifndef SILU_WG
#define SILU_WG 256
#endif

using bf16_t = sycl::ext::oneapi::bfloat16;

// ============================================================
// Globals struct (device-copyable)
// ============================================================

struct silu_mul_globals {
    const uint16_t *gate;   // bf16 bits, [B*N*inter]
    const uint16_t *up;     // bf16 bits, [B*N*inter]
    uint16_t       *out;    // bf16 bits, [B*N*inter]
    size_t          num_vec; // numel / SILU_VEC
};
template <>
struct sycl::is_device_copyable<silu_mul_globals> : std::true_type {};

// ============================================================
// Kernel — one work-item per SILU_VEC-wide vector
// ============================================================

inline void silu_mul_kernel(const silu_mul_globals g, sycl::nd_item<1> it) {
    using vu = sycl::vec<uint16_t, SILU_VEC>;

    const size_t v = it.get_global_linear_id();
    if (v >= g.num_vec) return;

    const vu *gate_v = reinterpret_cast<const vu *>(g.gate);
    const vu *up_v   = reinterpret_cast<const vu *>(g.up);
    vu       *out_v  = reinterpret_cast<vu *>(g.out);

    const vu gv = gate_v[v];   // wide bf16 load (gate)
    const vu uv = up_v[v];     // wide bf16 load (up)
    vu ov;

    #pragma unroll
    for (int j = 0; j < SILU_VEC; ++j) {
        const float gf = static_cast<float>(sycl::bit_cast<bf16_t>(gv[j]));
        const float uf = static_cast<float>(sycl::bit_cast<bf16_t>(uv[j]));
        // SiLU(x) * up = (x * sigmoid(x)) * up, computed in fp32.
        const float s = gf / (1.0f + sycl::exp(-gf));
        const float r = s * uf;
        ov[j] = sycl::bit_cast<uint16_t>(static_cast<bf16_t>(r));
    }

    out_v[v] = ov;             // wide bf16 store (out)
}

// ============================================================
// Launch
// ============================================================

void launch_silu_mul(
    sycl::queue &queue,
    const uint16_t *gate, const uint16_t *up, uint16_t *out,
    size_t total
) {
    const size_t num_vec = total / SILU_VEC;
    const size_t wg      = SILU_WG;
    const size_t global  = ((num_vec + wg - 1) / wg) * wg;

    silu_mul_globals g{gate, up, out, num_vec};

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                silu_mul_kernel(g, it);
            });
    });
    // NO queue.wait() — caller synchronizes via torch.xpu.synchronize()
}

// ============================================================
// PyTorch dispatch
// ============================================================

torch::Tensor dispatch_silu_mul(
    torch::Tensor gate,   // (B, N, inter), bf16
    torch::Tensor up      // (B, N, inter), bf16
) {
    CHECK_INPUT(gate);
    CHECK_INPUT(up);
    TORCH_CHECK(gate.scalar_type() == torch::kBFloat16, "gate must be bf16");
    TORCH_CHECK(up.scalar_type() == torch::kBFloat16, "up must be bf16");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");

    auto out = torch::empty_like(gate);

    const size_t total = static_cast<size_t>(gate.numel());
    TORCH_CHECK(total % SILU_VEC == 0,
                "numel (", total, ") must be divisible by SILU_VEC=", SILU_VEC);

    auto stream = c10::xpu::getCurrentXPUStream(gate.device().index());
    auto &queue = stream.queue();

    const uint16_t *d_gate = reinterpret_cast<const uint16_t *>(gate.data_ptr<c10::BFloat16>());
    const uint16_t *d_up   = reinterpret_cast<const uint16_t *>(up.data_ptr<c10::BFloat16>());
    uint16_t       *d_out  = reinterpret_cast<uint16_t *>(out.data_ptr<c10::BFloat16>());

    launch_silu_mul(queue, d_gate, d_up, d_out, total);

    return out;
}

// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens fused SwiGLU activation: out = SiLU(gate) * up (bf16)";
    m.def("dispatch_silu_mul", &dispatch_silu_mul,
          "Fused SiLU*mul: gate,up (bf16 [B,N,inter]) -> out (bf16 [B,N,inter])",
          py::arg("gate"), py::arg("up"));
}
