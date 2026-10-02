/**
 * @file bf16_gemm.dp.cpp
 * @brief SyclKittens GEMM PyTorch extension wrapper.
 *
 * Wraps the SyclKittens bf16 GEMM kernel as a pybind11 module so it can
 * accept torch::Tensor inputs directly.
 *
 * Compile-time parameters:
 *   -DGEMM_M=<val>  -DGEMM_N=<val>  -DGEMM_K=<val>
 *   -DTORCH_EXTENSION_NAME=sk_gemm_<size>
 *
 * Example build (4096x4096):
 *   icpx -shared -fPIC -fsycl -fsycl-targets=intel_gpu_pvc \
 *       -DKITTENS_INTEL -DKITTENS_XE -DGEMM_M=4096 -DGEMM_N=4096 -DGEMM_K=4096 \
 *       -DTORCH_EXTENSION_NAME=sk_gemm_4096 ... bf16_gemm.dp.cpp -o sk_gemm_4096.so
 */

#include "kittens.dp.hpp"
using namespace kittens;

// ============================================================
// Compile-time matrix dimensions
// ============================================================

#ifndef GEMM_M
#error "Define -DGEMM_M=<value>"
#endif
#ifndef GEMM_N
#error "Define -DGEMM_N=<value>"
#endif
#ifndef GEMM_K
#error "Define -DGEMM_K=<value>"
#endif

// NOTE: We use GEMM_M/GEMM_N/GEMM_K directly instead of #define M/K/N
// to avoid macro collisions with PyTorch header template parameters.
constexpr int GM = GEMM_M;
constexpr int GN = GEMM_N;
constexpr int GK = GEMM_K;

#define BM 32
#define BN 64
#define BK 32
#define RT_SHAPE ducks::rt_shape::rt_32x32

// Golden autotuned collaborative-prefetch depth (six-node co-measurement).
#if (GEMM_M == GEMM_N) && (GEMM_M == GEMM_K)
  #define GEMM_PF 2
#elif (GEMM_M==2048) && (GEMM_K==4096) && (GEMM_N==11008)
  #define GEMM_PF 1
#elif (GEMM_M==2048) && (GEMM_K==1024) && (GEMM_N==4096)
  #define GEMM_PF 1
#elif (GEMM_M==8192) && (GEMM_K==4096) && (GEMM_N==4096)
  #define GEMM_PF 2
#elif (GEMM_M==4096) && (GEMM_K==4096) && (GEMM_N==11008)
  #define GEMM_PF 2
#elif (GEMM_M==2048) && (GEMM_K==11008) && (GEMM_N==4096)
  #define GEMM_PF 2
#else
  #define GEMM_PF 3
#endif
#define PF GEMM_PF

#define GEMM_PF_A PF
#define GEMM_PF_B PF

#define GEMM_PF_MODM 5
#define GEMM_PF_MODN 8
#define PF_MODM GEMM_PF_MODM
#define PF_MODN GEMM_PF_MODN

#define GEMM_MAX_SG 32

// Compile split-K only for the measured occupancy-starved regime. Keeping its
// unused instantiations in dense modules perturbs register allocation.
#if (((GEMM_M / BM) * (GEMM_N / BN)) >= 256 || (GEMM_K / BK) < 8) \
    && !defined(GEMM_NO_SPLITK_CODE)
#define GEMM_NO_SPLITK_CODE
#endif

// Shape-adaptive root-sync policy from the six-node square and rectangular A/B.
#if ((GEMM_M == GEMM_N) && (GEMM_M == GEMM_K) && (GEMM_M >= 4096)) \
      || ((GEMM_M == 2048) && (GEMM_K == 1024)  && (GEMM_N == 4096)) \
      || ((GEMM_M == 2048) && (GEMM_K == 11008) && (GEMM_N == 4096)) \
      || ((GEMM_M == 2048) && (GEMM_K == 4096)  && (GEMM_N == 1024)) \
      || ((GEMM_M == 2048) && (GEMM_K == 4096)  && (GEMM_N == 11008)) \
      || ((GEMM_M == 2048) && (GEMM_K == 4096)  && (GEMM_N == 4096)) \
      || ((GEMM_M == 256)  && (GEMM_K == 4096)  && (GEMM_N == 4096)) \
      || ((GEMM_M == 4096) && (GEMM_K == 4096)  && (GEMM_N == 11008)) \
      || ((GEMM_M == 4096) && (GEMM_K == 4096)  && (GEMM_N == 12288)) \
      || ((GEMM_M == 512)  && (GEMM_K == 4096)  && (GEMM_N == 4096)) \
      || ((GEMM_M == 8192) && (GEMM_K == 4096)  && (GEMM_N == 12288)) \
      || ((GEMM_M == 8192) && (GEMM_K == 4096)  && (GEMM_N == 4096))
  #define GEMM_USE_ROOT_SYNC 0
#else
  #define GEMM_USE_ROOT_SYNC 1
#endif

#define NUM_THREADS (kittens::WARPGROUP_THREADS)

// ============================================================
// Global layout types
// ============================================================

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
// Output type is compile-time selectable:
//   default            -> bf16 out (fp32 accumulator down-converted IN-REGISTER
//                         by store(): packed <bf16_2,float2> RNE convert, 2-byte
//                         block store; half the store traffic, no fp32 round-trip).
//   -DGEMM_FP32_OUT    -> fp32 out.  store() writes the rt_fl accumulator
//                         DIRECTLY as fp32 (identity convert) — no in-register
//                         down-convert, no separate cast kernel, no sync.  This
//                         is the natural output of a bf16xbf16 DPAS GEMM.
#ifdef GEMM_FP32_OUT
using _gl_C = gl<float, -1, -1, -1, -1>;
#else
using _gl_C = gl<bf16, -1, -1, -1, -1>;
#endif

// ============================================================
// Snake-curve group swizzle (from original kernel)
// ============================================================

uint32_t get_2d_group_linear_id(sycl::nd_item<3> &item) {
    return item.get_group(2) + item.get_group(1) * item.get_group_range(2);
}

template <int wg_num_n_>
struct group_swizzle_snake {
public:
    template <int>
    int get_tile_idx(sycl::nd_item<3> &item);
    template <>
    int get_tile_idx<0>(sycl::nd_item<3> &item) {
        return item.get_group(0);
    }
    template <>
    int get_tile_idx<1>(sycl::nd_item<3> &item) {
        uint32_t group_range_n = item.get_group_range(2);
        uint32_t wg_repeat_n = group_range_n / wg_num_n;
        uint32_t repeat_id = get_2d_group_linear_id(item) / max_wg_num;
        uint32_t repeat_id_m = repeat_id / wg_repeat_n;
        uint32_t repeat_start_m = repeat_id_m * wg_num_m;
        uint32_t wg_inner_id = get_2d_group_linear_id(item) % max_wg_num;
        uint32_t wg_coord_m = wg_inner_id / wg_num_n;
        int start_m_id = repeat_start_m + wg_coord_m;
        return start_m_id;
    }
    template <>
    int get_tile_idx<2>(sycl::nd_item<3> &item) {
        uint32_t group_range_n = item.get_group_range(2);
        uint32_t wg_repeat_n = group_range_n / wg_num_n;
        uint32_t repeat_id = get_2d_group_linear_id(item) / max_wg_num;
        uint32_t repeat_id_n = repeat_id % wg_repeat_n;
        uint32_t repeat_id_m = repeat_id / wg_repeat_n;
        uint32_t repeat_start_n_0 = repeat_id_n * wg_num_n;
        uint32_t repeat_start_n_1 = (wg_repeat_n - repeat_id_n - 1) * wg_num_n;
        uint32_t repeat_start_n
                = (repeat_id_m & 1) == 0 ? repeat_start_n_0 : repeat_start_n_1;
        uint32_t wg_inner_id = get_2d_group_linear_id(item) % max_wg_num;
        uint32_t wg_coord_n = wg_inner_id % wg_num_n;
        int start_n_id = repeat_start_n + wg_coord_n;
        return start_n_id;
    }
    static void update_group_range(
            uint32_t &group_range_m, uint32_t &group_range_n) {
        group_range_m = (group_range_m + wg_num_m - 1) / wg_num_m * wg_num_m;
        group_range_n = (group_range_n + wg_num_n - 1) / wg_num_n * wg_num_n;
    }
private:
    static constexpr uint32_t max_wg_num = 64;
    static constexpr uint32_t wg_num_n = wg_num_n_;
    static_assert(!(max_wg_num % wg_num_n),
            "max_wg_num cannot be divisible by given wg_num_n!");
    static constexpr uint32_t wg_num_m = max_wg_num / wg_num_n;
};

// ============================================================
// GEMM kernel (from original kernel.dp.cpp)
// ============================================================

// Use snake traversal only for measured aligned shapes. The 7168 cube is the
// sole padded exception and remains a measured win.
#if (GEMM_M == 7168) && (GEMM_N == 7168) && (GEMM_K == 7168)
#define SNAKE_ALIGNED 1
#else
#define SNAKE_ALIGNED (GEMM_M % 2048 == 0 && GEMM_N % 2048 == 0)
#endif

#if (GEMM_M == 2048) && (GEMM_K == 4096) && (GEMM_N == 4096)
  #define GEMM_SNAKE_WIDTH 16
#elif (GEMM_M == GEMM_N) && ((GEMM_M == 2048) || (GEMM_M == 3072) \
      || (GEMM_M == 7168) || (GEMM_M == 8192))
  #define GEMM_SNAKE_WIDTH 16
#elif (GEMM_M == GEMM_N) && ((GEMM_M == 12288) || (GEMM_M == 14336) \
      || (GEMM_M == 16384))
  #define GEMM_SNAKE_WIDTH 4
#else
  #define GEMM_SNAKE_WIDTH 8
#endif

void micro_tk(_gl_A g_a, _gl_B g_b, _gl_C g_c, float alpha, float beta) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t sg_id = item.get_sub_group().get_group_id();
#if SNAKE_ALIGNED
    group_swizzle_snake<GEMM_SNAKE_WIDTH> g;
    int wg_m =  g.get_tile_idx<1>(item) * item.get_local_range(1) + item.get_local_id(1);
    int wg_n = (g.get_tile_idx<2>(item) * item.get_local_range(2) + item.get_local_id(2)) / 16;
    // Snake grid may be padded beyond actual matrix dims — skip padding tiles.
    if (wg_m >= GM / BM || wg_n >= GN / BN) return;
#else
    // Linear indexing fallback for non-aligned grid sizes
    int wg_m = item.get_global_id(1);
    int wg_n = item.get_global_id(2) / 16;
#endif

    rt_bf<BM, BK, ducks::rt_layout::row, RT_SHAPE> a_reg;
    rt_bf<BK, BN, ducks::rt_layout::col, RT_SHAPE> b_reg;
    rt_fl<BM, BN, ducks::rt_layout::row, RT_SHAPE> c_reg;
    zero(c_reg);
#define GEMM_UNROLL 32
    // Collaborative de-duplicated prefetch. Each subgroup warms only its
    // assigned slice of the shared A/B work-group panel.
    const int coop_h_idx         = sg_id % (256 / BN);
    const int coop_v_idx         = sg_id / (256 / BN);
    const int pref_a_row_offset  = BM / (256 / BN);
    const int pref_b_sg_row_size = (256 / (BM * 2));
    const int pref_b_row_offset  = BK / (256 / BN);
    const int coop_b_row_idx     = coop_v_idx % pref_b_sg_row_size;
    const int coop_b_col_idx     = coop_v_idx / pref_b_sg_row_size;
    const int coop_a_row_offset  = coop_h_idx * pref_a_row_offset;
    const int coop_b_row_offset  = coop_b_row_idx * pref_b_row_offset;
    const int coop_b_col_offset  = coop_b_col_idx * 32;

    int prefetch_a = GEMM_PF_A;
    int prefetch_b = GEMM_PF_B;
#pragma unroll
    for (int idx = 0; idx < prefetch_a; idx++)
      prefetch_load_coop(a_reg, g_a, {0, 0, wg_m, idx}, {coop_a_row_offset, 0});
#pragma unroll
    for (int idx = 0; idx < prefetch_b; idx++)
      prefetch_load_coop(b_reg, g_b, {0, 0, idx, wg_n}, {coop_b_row_offset, coop_b_col_offset});
#pragma unroll GEMM_UNROLL
    for (int bkIdx = 0; bkIdx < GK / BK; bkIdx++, prefetch_a++, prefetch_b++) {
      load(a_reg, g_a, {0, 0, wg_m, bkIdx});
      load(b_reg, g_b, {0, 0, bkIdx, wg_n});
      if (prefetch_a < GK / BK)
        prefetch_load_coop(a_reg, g_a, {0, 0, wg_m, prefetch_a}, {coop_a_row_offset, 0});
      if (prefetch_b < GK / BK)
        prefetch_load_coop(b_reg, g_b, {0, 0, prefetch_b, wg_n}, {coop_b_row_offset, coop_b_col_offset});
      mma_AB(c_reg, a_reg, b_reg, c_reg);
    }

    store(g_c, c_reg, {0, 0, wg_m, wg_n});
}

// ============================================================
// Raw-pointer dispatch (for internal use)
// ============================================================

static void dispatch_micro(kittens::bf16 *d_A, kittens::bf16 *d_B,
                           typename _gl_C::dtype *d_C, sycl::queue &queue) {
    _gl_A a_arg{d_A, 0, 0, GM, GK};
    _gl_B b_arg{d_B, 0, 0, GK, GN};
    _gl_C c_arg{d_C, 0, 0, GM, GN};

    uint32_t grid_m = GM / BM;
    uint32_t grid_n = GN / BN;

    constexpr int MAX_SG = GEMM_MAX_SG;
    constexpr int WG_N = (256 / BN < MAX_SG) ? (256 / BN) : MAX_SG;
    constexpr int WG_M = MAX_SG / WG_N;

#if SNAKE_ALIGNED
    // Pad tile counts so WG counts are multiples of snake super-tile dims.
    // Snake operates on WGs: wg_count = tile_count / WG_DIM.
    // Requires: wg_count_m % wg_num_m == 0 and wg_count_n % wg_num_n == 0.
    // wg_num_m = 64/GEMM_SNAKE_WIDTH, wg_num_n = GEMM_SNAKE_WIDTH.
    // Thus tile_count must be a multiple of WG_DIM * wg_num.
    constexpr uint32_t SNAKE_PAD_M = static_cast<uint32_t>(WG_M) * (64 / GEMM_SNAKE_WIDTH);
    constexpr uint32_t SNAKE_PAD_N = static_cast<uint32_t>(WG_N) * GEMM_SNAKE_WIDTH;
    grid_m = ((grid_m + SNAKE_PAD_M - 1) / SNAKE_PAD_M) * SNAKE_PAD_M;
    grid_n = ((grid_n + SNAKE_PAD_N - 1) / SNAKE_PAD_N) * SNAKE_PAD_N;
#else
    // Linear fallback: just ensure divisibility by WG block dimensions.
    grid_m = ((grid_m + WG_M - 1) / WG_M) * WG_M;
    grid_n = ((grid_n + WG_N - 1) / WG_N) * WG_N;
#endif

    sycl::range<3> grid(1, grid_m, grid_n * 16);
    sycl::range<3> block(1, WG_M, WG_N * 16);

#if GEMM_USE_ROOT_SYNC
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync,
        sycl::ext::oneapi::experimental::sub_group_size<16>};
#else
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>};
#endif

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid, block),
                         exp_props, [=](sycl::nd_item<3> item_ct1) {
                             micro_tk(a_arg, b_arg, c_arg, 1.0f, 0.0f);
                         });
    });
}

// ============================================================
// Split-K path for occupancy-starved small-output, tall-K shapes.
//
// Motivation: tile-starved shapes (small M,N -> few output WGs) leave the
// machine under-filled, so the mainloop cannot hide DPAS/load latency.  Split-K
// partitions the K reduction across S work-group planes (grid dim 0),
// multiplying the resident WG count so the mainloop runs at high occupancy.
// Each plane computes a partial C over its K-slice; the S partials are combined
// with fp32 device-scope atomics.  The atomic epilogue emits the L2-roofline-
// optimal 64B SIMD16 message and is effectively free while the M*N output stays
// L2-resident — i.e. the win regime is small M,N + large K (decode/QKV/MoE).
// No cooperative launch (use_root_sync): the combine is via atomics, not a grid
// barrier, so the S*WG planes schedule in normal occupancy-filling waves.
// ============================================================
#ifndef GEMM_NO_SPLITK_CODE
static_assert(GK % BK == 0, "K must be divisible by BK");

using _gl_Cf = gl<float, -1, -1, -1, -1>;

#define SK_SHAPE RT_SHAPE

constexpr int split_k_planes() {
  constexpr int tiles = (GM / BM) * (GN / BN);
  constexpr int max_planes = (GK / BK) / 4;
  constexpr int target_planes = (1024 + tiles - 1) / tiles;
  return target_planes < max_planes ? target_planes : max_planes;
}

constexpr int SPLIT_K_PLANES = split_k_planes();
static_assert(SPLIT_K_PLANES > 1, "split-K requires at least two planes");

void micro_tk_splitk(_gl_A g_a, _gl_B g_b, _gl_Cf g_cf) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t sg_id = item.get_sub_group().get_group_id();
    const int split = item.get_group(0);

#if SNAKE_ALIGNED
    // IDENTICAL tile mapping to the tuned default micro_tk (snake swizzle on the
    // (m,n) grid; the split index lives on grid dim 0 and does not perturb it).
    group_swizzle_snake<GEMM_SNAKE_WIDTH> g;
    int wg_m =  g.get_tile_idx<1>(item) * item.get_local_range(1) + item.get_local_id(1);
    int wg_n = (g.get_tile_idx<2>(item) * item.get_local_range(2) + item.get_local_id(2)) / 16;
    if (wg_m >= GM / BM || wg_n >= GN / BN) return;
#else
  int wg_m = item.get_global_id(1);
    int wg_n = item.get_global_id(2) / 16;
    if (wg_m >= GM / BM || wg_n >= GN / BN) return;
#endif

    constexpr int KT = GK / BK;
    constexpr int KPS = KT / SPLIT_K_PLANES;
    constexpr int REM = KT - KPS * SPLIT_K_PLANES;
    const int k0 = split * KPS + (split < REM ? split : REM);
    const int k1 = k0 + KPS + (split < REM ? 1 : 0);

    rt_bf<BM, BK, ducks::rt_layout::row, SK_SHAPE> a_reg;
    rt_bf<BK, BN, ducks::rt_layout::col, SK_SHAPE> b_reg;
    rt_fl<BM, BN, ducks::rt_layout::row, SK_SHAPE> c_reg;
    zero(c_reg);

    // SAME masked self-prefetch pipeline as the default, but bounded to this
    // plane's K-slice [k0, k1).
    int prefetch = k0 + PF;
#pragma unroll
    for (int idx = 0; idx < PF; idx++) {
      if (sg_id % PF_MODM)
        prefetch_load(a_reg, g_a, {0, 0, wg_m, k0 + idx});
      if (sg_id & PF_MODN)
        prefetch_load(b_reg, g_b, {0, 0, k0 + idx, wg_n});
    }

#pragma unroll GEMM_UNROLL
    for (int bkIdx = k0; bkIdx < k1; bkIdx++, prefetch++) {
      load(a_reg, g_a, {0, 0, wg_m, bkIdx});
      load(b_reg, g_b, {0, 0, bkIdx, wg_n});
      if (prefetch < k1) {
        if (sg_id % PF_MODM)
          prefetch_load(a_reg, g_a, {0, 0, wg_m, prefetch});
        if (sg_id & PF_MODN)
          prefetch_load(b_reg, g_b, {0, 0, prefetch, wg_n});
      }
      mma_AB(c_reg, a_reg, b_reg, c_reg);
    }
    // fp32 device-scope atomic combine of the S partials (rt_32x32 epilogue).
    atomic_add(g_cf, c_reg, {0, 0, wg_m, wg_n});
}

void cast_f32_to_bf16(const float *src, bf16 *dst, size_t count) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const size_t i = item.get_global_id(0);
    if (i < count)
        dst[i] = static_cast<bf16>(src[i]);
}

// fp32 scratch -> bf16 output for the atomic split-K path.
static sycl::event launch_cast_strip(const float *scratch, bf16 *dst,
                   size_t nelem,
                                     sycl::queue &queue, sycl::event dep) {
    constexpr size_t CAST_WG = 256;
    const size_t grid = ((nelem + CAST_WG - 1) / CAST_WG) * CAST_WG;
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
        cgh.parallel_for(sycl::nd_range<1>(grid, CAST_WG),
             [=](sycl::nd_item<1>) { cast_f32_to_bf16(scratch, dst, nelem); });
    });
}

static void launch_splitk(kittens::bf16 *d_A, kittens::bf16 *d_B,
              typename _gl_C::dtype *d_C, sycl::queue &queue) {
    _gl_A a_arg{d_A, 0, 0, GM, GK};
    _gl_B b_arg{d_B, 0, 0, GK, GN};

    uint32_t grid_m = GM / BM;
    uint32_t grid_n = GN / BN;
    constexpr int MAX_SG = GEMM_MAX_SG;
    constexpr int WG_N = (256 / BN < MAX_SG) ? (256 / BN) : MAX_SG;
    constexpr int WG_M = MAX_SG / WG_N;
#if SNAKE_ALIGNED
    // Same snake super-tile padding as the tuned default (dispatch_micro).
    constexpr uint32_t SNAKE_PAD_M = static_cast<uint32_t>(WG_M) * (64 / GEMM_SNAKE_WIDTH);
    constexpr uint32_t SNAKE_PAD_N = static_cast<uint32_t>(WG_N) * GEMM_SNAKE_WIDTH;
    grid_m = ((grid_m + SNAKE_PAD_M - 1) / SNAKE_PAD_M) * SNAKE_PAD_M;
    grid_n = ((grid_n + SNAKE_PAD_N - 1) / SNAKE_PAD_N) * SNAKE_PAD_N;
#else
    grid_m = ((grid_m + WG_M - 1) / WG_M) * WG_M;
    grid_n = ((grid_n + WG_N - 1) / WG_N) * WG_N;
#endif
    // grid dim 0 = S split planes.  NO use_root_sync (atomic combine, not a
    // grid barrier) so the S*WG planes schedule in normal occupancy-filling waves.
    sycl::range<3> grid(SPLIT_K_PLANES, grid_m, grid_n * 16);
    sycl::range<3> block(1, WG_M, WG_N * 16);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>};
#ifdef GEMM_FP32_OUT
    // ---- fp32-output split-K (the real target) ----------------------------
    // The atomic combine accumulates DIRECTLY into the caller's fp32 C tensor.
    // No scratch buffer, no cast kernel — just two stages: zero C, then atomic-
    // add the S partial planes.  d_C is float* (GEMM_FP32_OUT routes it here).
    _gl_Cf cf_arg{d_C, 0, 0, GM, GN};
    constexpr size_t nelem = static_cast<size_t>(GM) * GN;
    // Cross-call ordering: this call's memset must wait for the previous call's
    // atomic mainloop to finish (torch XPU stream not guaranteed in-order for
    // raw USM).  prev_done tracks the previous atomic-GEMM completion.
    static sycl::event prev_done{};
    sycl::event e_zero = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(prev_done);
      cgh.memset(d_C, 0, nelem * sizeof(float));
    });
    prev_done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e_zero);
        cgh.parallel_for(sycl::nd_range<3>(grid, block), exp_props,
                         [=](sycl::nd_item<3> item_ct1) {
                   micro_tk_splitk(a_arg, b_arg, cf_arg);
                         });
    });
#else
    // ---- bf16-output split-K (parity path) --------------------------------
    // Accumulate in a persistent fp32 scratch, then cast fp32 -> bf16.
    static float *d_scratch = nullptr;
    if (!d_scratch)
        d_scratch = sycl::malloc_device<float>((size_t)GM * GN, queue);
    // Local copy: a SYCL kernel lambda may not capture a static data variable.
    float *scratch = d_scratch;
    _gl_Cf cf_arg{scratch, 0, 0, GM, GN};
    constexpr size_t nelem = static_cast<size_t>(GM) * GN;

    // Cross-call ordering: the persistent scratch is REUSED every dispatch, so
    // this call's memset must wait for the PREVIOUS call's cast to finish
    // reading it (a torch XPU stream is not guaranteed in-order for raw USM;
    // without this, a benchmark's back-to-back un-synced iterations race the
    // scratch — memset(N+1) vs cast(N) — and corrupt results).
    static sycl::event prev_done{};
    sycl::event e_zero = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(prev_done);
      cgh.memset(scratch, 0, nelem * sizeof(float));
    });

    sycl::event e_mma = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e_zero);
        cgh.parallel_for(sycl::nd_range<3>(grid, block), exp_props,
                         [=](sycl::nd_item<3> item_ct1) {
                   micro_tk_splitk(a_arg, b_arg, cf_arg);
                         });
    });

    // Cast fp32 accumulator -> bf16 output (parity with the default bf16 path).
      prev_done = launch_cast_strip(scratch, d_C, nelem, queue, e_mma);
#endif  // GEMM_FP32_OUT
}
#endif  // GEMM_NO_SPLITK_CODE

// ============================================================
// PyTorch dispatch — wall-clock timing done by caller
// ============================================================

#include "pyutils/torch_helpers.dp.hpp"
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

void dispatch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    TORCH_CHECK(A.size(0) == GM && A.size(1) == GK,
                "A must be [", GM, ", ", GK, "], got [", A.size(0), ", ", A.size(1), "]");
    TORCH_CHECK(B.size(0) == GK && B.size(1) == GN,
                "B must be [", GK, ", ", GN, "], got [", B.size(0), ", ", B.size(1), "]");
    TORCH_CHECK(C.size(0) == GM && C.size(1) == GN,
                "C must be [", GM, ", ", GN, "], got [", C.size(0), ", ", C.size(1), "]");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");
#ifdef GEMM_FP32_OUT
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be fp32 (GEMM_FP32_OUT)");
#else
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16");
#endif

    auto stream = c10::xpu::getCurrentXPUStream(A.device().index());
    auto &queue = stream.queue();
    // No queue.wait() — PyTorch stream ordering handles dependencies.
    // The previous queue.wait() added ~7% overhead per dispatch.

    bf16  *d_A = reinterpret_cast<bf16*>(A.data_ptr<c10::BFloat16>());
    bf16  *d_B = reinterpret_cast<bf16*>(B.data_ptr<c10::BFloat16>());
#ifdef GEMM_FP32_OUT
    float *d_C = C.data_ptr<float>();
#else
    bf16  *d_C = reinterpret_cast<bf16*>(C.data_ptr<c10::BFloat16>());
#endif

#ifdef GEMM_NO_SPLITK_CODE
    dispatch_micro(d_A, d_B, d_C, queue);
#else
  launch_splitk(d_A, d_B, d_C, queue);
#endif
    // No queue.wait() — caller synchronizes via torch.xpu.synchronize()
}

// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens bf16 GEMM (SYCL): " +
              std::to_string(GM) + "x" + std::to_string(GN) + "x" + std::to_string(GK);
    m.def("dispatch_gemm", &dispatch_gemm,
          "C = A @ B (bf16 -> bf16/fp32).  A:[M,K] B:[K,N] C:[M,N].",
          py::arg("A"), py::arg("B"), py::arg("C"));
    m.attr("M") = GM;
    m.attr("N") = GN;
    m.attr("K") = GK;
}
