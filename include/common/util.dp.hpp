/**
 * @file
 * @brief General utilities for SyclKittens.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

// Error checking macros for SYCL
#define CUCHECK(cmd) do {                                                      \
        int err = cmd;                                                         \
        if (err != 0) {                                                        \
            const char *errStr;                                                \
            errStr = dpct::get_error_string_dummy(err);                        \
            fprintf(stderr, "Failed: SYCL error %s:%d '%s'\n", __FILE__,       \
                    __LINE__, errStr);                                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDACHECK(cmd) do {                                                    \
        dpct::err0 err = cmd;                                                  \
    } while (0)

/**
 * @namespace kittens
 * @brief The main namespace of SyclKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

constexpr int BASE_TILE_DIM = 16;
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
template<typename T> constexpr int TILE_ROW_DIM = BASE_TILE_DIM;
template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T>*TILE_ROW_DIM<T>};

#if defined(KITTENS_INTEL)
constexpr int WARP_THREADS{16};
#else
constexpr int WARP_THREADS{32};
#endif

constexpr int WARPGROUP_THREADS{128};
constexpr int WARPGROUP_WARPS{4};

#if defined(KITTENS_INTEL)
__dpct_inline__ int warpid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) >> 4;
}
#else
__dpct_inline__ int warpid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) >> 5;
}
#endif

__dpct_inline__ int warpgroupid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) >> 7;
}

#if defined(KITTENS_INTEL)
__dpct_inline__ int laneid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) & 0xf;
}
#else
__dpct_inline__ int laneid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) & 0x1f;
}
#endif

// smid() not available via SYCL standard - stub for Intel
__dpct_inline__ int smid() {
    // Intel Xe does not directly expose EU/slice ID in SYCL.
    // Return 0 as placeholder; use sycl queue profiling for placement info.
    return 0;
}

#if defined(KITTENS_HOPPER)
constexpr int MAX_SHARED_MEMORY = 227000;
#elif defined(KITTENS_A100)
constexpr int MAX_SHARED_MEMORY = 164000;
#elif defined(KITTENS_4090)
constexpr int MAX_SHARED_MEMORY = 100000;
#elif defined(KITTENS_INTEL)
constexpr int MAX_SHARED_MEMORY = 100000;
#endif

struct transpose {
    static constexpr int N = 0;
    static constexpr int T = 1;
};
struct axis {
    static constexpr int ROW = 0;
    static constexpr int COL = 1;
};

/* ----------  TYPE HELPERS  ---------- */

namespace ducks {
struct default_type {};
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type
}

/* ----------  SHUFFLE UTILS  ---------- */

static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

template<typename T>
static inline T packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    return dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f, delta);
}
template <>
inline sycl::float2 packed_shfl_down_sync<sycl::float2>(uint32_t mask,
                                                        const sycl::float2 &f,
                                                        int delta) {
    sycl::float2 r;
    r.x() = dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.x(), delta);
    r.y() = dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.y(), delta);
    return r;
}

template<typename T>
static inline T packed_shfl_sync(uint32_t mask, const T &f, int src) {
    return dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f, src);
}
template <>
inline sycl::float2
packed_shfl_sync<sycl::float2>(uint32_t mask, const sycl::float2 &f, int src) {
    sycl::float2 r;
    r.x() = dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.x(), src);
    r.y() = dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.y(), src);
    return r;
}

/* ----------  SHARED MEMORY UTILS  ---------- */

#if defined(SYCL_LANGUAGE_VERSION)
#define KITTENS_ALIGN_AS(n) __dpct_align__(n)
#else
#define KITTENS_ALIGN_AS(n) alignas(n)
#endif

#if defined(KITTENS_HOPPER) || defined(KITTENS_INTEL)
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(128)
#else
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)
#endif

template <typename T>
constexpr T cdiv(T a, T b) {
    return (a + b - 1) / b;
}

struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };

#if defined(KITTENS_HOPPER) || defined(KITTENS_INTEL)
template<int default_alignment=1024>
#else
template<int default_alignment=16>
#endif
struct shared_allocator {
    int *ptr;

private:
    template<typename A, size_t... dims>
    struct variadic_array;
    template<typename A, size_t first_dim, size_t... rest_dims>
    struct variadic_array<A, first_dim, rest_dims...> {
        using type = typename variadic_array<A, rest_dims...>::type[first_dim];
    };
    template<typename A>
    struct variadic_array<A> {
        using type = A;
    };
    template<typename A, size_t... dims>
    using variadic_array_t = typename variadic_array<A, dims...>::type;

    template<int alignment>
    inline void align_ptr() {
        if constexpr (alignment > 0) {
            uint64_t p = reinterpret_cast<uint64_t>(ptr);
            if(p % alignment != 0) {
                ptr = (int*)(p + (alignment-(p%alignment)));
            }
        }
    }

public:
    shared_allocator(int *_ptr): ptr(_ptr) {}
    template<typename A, size_t... dims>
    inline variadic_array_t<A, dims...>& allocate() {
        align_ptr<default_alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
    template<int alignment, typename A, size_t... dims>
    inline variadic_array_t<A, dims...>& allocate() {
        align_ptr<alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

#if defined(KITTENS_HOPPER) || defined(KITTENS_INTEL)
using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator;
#endif

/* ----------  PIPELINE / SWIZZLE UTILS (from TK v2.0)  ---------- */

/**
 * @brief Map a linear task index to a 2D swizzled (row, col) index using supergroup snake ordering.
 */
template <int SUPERGROUP_SIZE, bool ROW_MAJOR = true>
static inline sycl::int2 get_swizzled_2d_idx(const int num_rows, const int num_cols, const int linear_idx) {
    static_assert(SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    if constexpr (ROW_MAJOR) {
        const int supergroup_numel = num_rows*SUPERGROUP_SIZE;
        const int supergroup_idx = linear_idx/supergroup_numel;
        const int supersection_cols = (num_cols/SUPERGROUP_SIZE)*SUPERGROUP_SIZE;
        const int supersection_numel = num_rows*supersection_cols;
        const int finalsection_cols = num_cols-supersection_cols;
        int row_idx, col_idx;
        if (linear_idx < supersection_numel) {
            row_idx = (linear_idx%supergroup_numel)/SUPERGROUP_SIZE;
            col_idx = supergroup_idx*SUPERGROUP_SIZE + linear_idx%SUPERGROUP_SIZE;
        } else {
            const int remainder_task_id = linear_idx - supersection_numel;
            row_idx = remainder_task_id/finalsection_cols;
            col_idx = supersection_cols + remainder_task_id%finalsection_cols;
        }
        return { (supergroup_idx&1) ? num_rows-row_idx-1 : row_idx, col_idx };
    } else {
        const int supergroup_numel = num_cols*SUPERGROUP_SIZE;
        const int supergroup_idx = linear_idx/supergroup_numel;
        const int supersection_rows = (num_rows/SUPERGROUP_SIZE)*SUPERGROUP_SIZE;
        const int supersection_numel = num_cols*supersection_rows;
        const int finalsection_rows = num_rows-supersection_rows;
        int row_idx, col_idx;
        if (linear_idx < supersection_numel) {
            row_idx = supergroup_idx*SUPERGROUP_SIZE + linear_idx%SUPERGROUP_SIZE;
            col_idx = (linear_idx%supergroup_numel)/SUPERGROUP_SIZE;
        } else {
            const int remainder_task_id = linear_idx - supersection_numel;
            row_idx = supersection_rows + remainder_task_id%finalsection_rows;
            col_idx = remainder_task_id/finalsection_rows;
        }
        return { row_idx, (supergroup_idx&1) ? num_cols-col_idx-1 : col_idx };
    }
}

template<int half>
static inline int get_phasebit(uint32_t bitfield, int ring_id) {
    if constexpr (half == 0)
        return (bitfield >> (ring_id)) & 0b1;
    else if constexpr (half == 1)
        return (bitfield >> (ring_id + 16)) & 0b1;
    return -1;
}

template<int half>
static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    if constexpr (half == 0)
        bitfield ^= (1 << (ring_id));
    else if constexpr (half == 1)
        bitfield ^= (1 << (ring_id + 16));
}

template<int N> static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }

/* ----------  HOST-SIDE UTILS  ---------- */

#ifndef KITTENS_NO_HOST

#define CHECK_SYCL_ERROR(val) check_sycl((val), #val, __FILE__, __LINE__)

inline int num_sms(int device_id = -1) {
    // Intel Xe: return number of execution units (compute units) as SM analog
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (device_id < 0) device_id = 0;
    if (device_id < static_cast<int>(devices.size())) {
        return devices[device_id].get_info<sycl::info::device::max_compute_units>();
    }
    return 1;
}

/**
 * @brief SYCL launch configuration wrapper (analogous to CUDA LaunchConfig).
 */
template <bool CLUSTER = false, bool PDL = false>
struct LaunchConfig {
    sycl::nd_range<3> nd_range;
    size_t dynamic_shared_memory;
    sycl::queue *queue;

    inline LaunchConfig(sycl::range<3> grid, sycl::range<3> block,
                        size_t dynamic_smem, sycl::queue *q) noexcept
        : nd_range(grid * block, block), dynamic_shared_memory(dynamic_smem), queue(q) {}
};

#endif // KITTENS_NO_HOST

} // namespace kittens
