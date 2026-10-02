/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

/* ----------   To prevent generic addressing, PTX  ---------- */

template<typename T> struct move {
    static inline void lds(T& dst, uintptr_t src);
    static inline void sts(uintptr_t dst, const T& src);
    static inline void ldg(T& dst, T* src);
    static inline void stg(T* dst, const T& src);
};

// ----------------------------------------------------------------
// move_local<T>: SLM-pointer-qualified lds/sts.
//
// The generic move<T>::lds/sts above do `*(T*)uintptr` which produces
// an unqualified (generic) address-space pointer. IGC then inserts
// runtime dispatch to determine whether the deref is local/global/private
// — adding hundreds of control-flow branches per kernel and killing perf.
//
// move_local<T> annotates the pointer with __attribute__((opencl_local))
// so the compiler knows it targets SLM, eliminating the generic-AS path.
// Use it for shared-tile <-> register-tile transfers only.
// ----------------------------------------------------------------
template<typename T> struct move_local {
    static inline void lds(T& dst, uintptr_t src) {
        dst = *((__attribute__((opencl_local)) T*)src);
    }
    static inline void sts(uintptr_t dst, const T& src) {
        *((__attribute__((opencl_local)) T*)dst) = src;
    }
};
template<> struct move_local<bf16> {
    static inline void lds(bf16& dst, uintptr_t src) {
        *(uint16_t*)&dst = *((__attribute__((opencl_local)) uint16_t*)src);
    }
    static inline void sts(uintptr_t dst, const bf16& src) {
        *((__attribute__((opencl_local)) uint16_t*)dst) = *(uint16_t*)&src;
    }
};
template<> struct move_local<sycl::half> {
    static inline void lds(sycl::half& dst, uintptr_t src) {
        *(uint16_t*)&dst = *((__attribute__((opencl_local)) uint16_t*)src);
    }
    static inline void sts(uintptr_t dst, const sycl::half& src) {
        *((__attribute__((opencl_local)) uint16_t*)dst) = *(uint16_t*)&src;
    }
};
template<> struct move_local<bf16_2> {
    static inline void lds(bf16_2& dst, uintptr_t src) {
        *(uint32_t*)&dst = *((__attribute__((opencl_local)) uint32_t*)src);
    }
    static inline void sts(uintptr_t dst, const bf16_2& src) {
        *((__attribute__((opencl_local)) uint32_t*)dst) = *(uint32_t*)&src;
    }
};
template<> struct move_local<half_2> {
    static inline void lds(half_2& dst, uintptr_t src) {
        *(uint32_t*)&dst = *((__attribute__((opencl_local)) uint32_t*)src);
    }
    static inline void sts(uintptr_t dst, const half_2& src) {
        *((__attribute__((opencl_local)) uint32_t*)dst) = *(uint32_t*)&src;
    }
};
// unpacked types
template<> struct move<bf16> {
    static inline void lds(bf16& dst, uintptr_t src) {
        *(uint16_t *)&dst = *((uint16_t *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const bf16& src) {
        *((uint16_t *)(uintptr_t)dst) = *(uint16_t *)&src;
    }
    static inline void ldg(bf16& dst, bf16* src) {
        *(uint16_t *)&dst = *((uint16_t *)(uintptr_t)src);
    }
    static inline void stg(bf16* dst, const bf16& src) {
        *((uint16_t *)(uintptr_t)dst) = *(uint16_t *)&src;
    }
};
template <> struct move<sycl::half> {
    static inline void lds(sycl::half &dst, uintptr_t src) {
        *(uint16_t *)&dst = *((uint16_t *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const sycl::half &src) {
        *((uint16_t *)(uintptr_t)dst) = *(uint16_t *)&src;
    }
    static inline void ldg(sycl::half &dst, sycl::half *src) {
        *(uint16_t *)&dst = *((uint16_t *)(uintptr_t)src);
    }
    static inline void stg(sycl::half *dst, const sycl::half &src) {
        *((uint16_t *)(uintptr_t)dst) = *(uint16_t *)&src;
    }
};
template<> struct move<float> {
    static inline void lds(float& dst, uintptr_t src) {
        dst = *((float *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const float& src) {
        *((float *)(uintptr_t)dst) = src;
    }
    static inline void ldg(float& dst, float* src) {
        dst = *src;
    }
    static inline void stg(float* dst, const float& src) {
        *dst = src;
    }
};
// packed types
template<> struct move<bf16_2> {
    static inline void lds(bf16_2& dst, uintptr_t src) {
        *(uint32_t *)&dst = *((uint32_t *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const bf16_2& src) {
        *((uint32_t *)(uintptr_t)dst) = (*(uint32_t *)&src);
    }
    static inline void ldg(bf16_2& dst, bf16_2* src) {
        *(uint32_t *)&dst = *((uint32_t *)(uintptr_t)src);
    }
    static inline void stg(bf16_2* dst, const bf16_2& src) {
        *((uint32_t *)(uintptr_t)dst) = (*(uint32_t *)&src);
    }
    static inline void ldsm4(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uintptr_t src) {
        dpct::experimental::matrix::ldmatrix((uintptr_t)src, &dst1, &dst2, &dst3, &dst4, false);
    }
    static inline void ldsm4t(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uintptr_t src) {
        dpct::experimental::matrix::ldmatrix((uintptr_t)src, &dst1, &dst2, &dst3, &dst4, true);

    }
    static inline void stsm4(uintptr_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        dpct::experimental::matrix::stmatrix((uintptr_t)dst, src1, src2, src3, src4, false);
    }
    static inline void stsm4t(uintptr_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        dpct::experimental::matrix::stmatrix((uintptr_t)dst, src1, src2, src3, src4, true);

    }
};
template<> struct move<half_2> {
    static inline void lds(half_2& dst, uintptr_t src) {
        *(uint32_t *)&dst = *((uint32_t *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const half_2& src) {
        *((uint32_t *)(uintptr_t)dst) = (*(uint32_t *)&src);
    }
    static inline void ldg(half_2& dst, half_2* src) {
        *(uint32_t *)&dst = *((uint32_t *)(uintptr_t)src);
    }
    static inline void stg(half_2* dst, const half_2& src) {
        *((uint32_t *)(uintptr_t)dst) = (*(uint32_t *)&src);
    }
    static inline void ldsm4(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uintptr_t src) {
        dpct::experimental::matrix::ldmatrix((uintptr_t)src, &dst1, &dst2, &dst3, &dst4, false);

    }
    static inline void ldsm4t(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uintptr_t src) {
        dpct::experimental::matrix::ldmatrix((uintptr_t)src, &dst1, &dst2, &dst3, &dst4, true);

    }
    static inline void stsm4(uintptr_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
          dpct::experimental::matrix::stmatrix((uintptr_t)dst, src1, src2, src3, src4, false);
    }
    static inline void stsm4t(uintptr_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
          dpct::experimental::matrix::stmatrix((uintptr_t)dst, src1, src2, src3, src4, true);
    }
};
template <> struct move<sycl::float2> {
    static inline void lds(sycl::float2 &dst, uintptr_t src) {
        float *p = (float *)(uintptr_t)src;
        dst.x() = *p;
        dst.y() = *(p + 1);
        //{dst.x(), dst.y()} = *((float *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const sycl::float2 &src) {
        float *p = (float *)(uintptr_t)dst;
        *p = src.x();
        *(p + 1) = src.y();
        //*((float *)(uintptr_t)dst) = {src.x(), src.y()};
    }
    static inline void ldg(sycl::float2 &dst, sycl::float2 *src) {
        float *p = (float *)(uintptr_t)src;
        dst.x() = *p;
        dst.y() = *(p + 1);
        //{dst.x(), dst.y()} = *((float *)(uintptr_t)src);
    }
    static inline void stg(sycl::float2 *dst, const sycl::float2 &src) {
        float *p = (float *)(uintptr_t)dst;
        *p = src.x();
        *(p + 1) = src.y();
        //*((float *)(uintptr_t)dst) = {src.x(), src.y()};
    }
};
template <> struct move<sycl::float4> {
    static inline void lds(sycl::float4 &dst, uintptr_t src) {
        float *p = (float *)(uintptr_t)src;
        dst.x() = *p;
        dst.y() = *(p + 1);
        dst.z() = *(p + 2);
        dst.w() = *(p + 3);
        //{dst.x(), dst.y(), dst.z(), dst.w()} = *((float *)(uintptr_t)src);
    }
    static inline void sts(uintptr_t dst, const sycl::float4 &src) {
        float *p = (float *)(uintptr_t)dst;
        *p = src.x();
        *(p + 1) = src.y();
        *(p + 2) = src.z();
        *(p + 3) = src.w();
        //*((float *)(uintptr_t)dst) = {src.x(), src.y(), src.z(), src.w()};
    }
    static inline void ldg(sycl::float4 &dst, sycl::float4 *src) {
        float *p = (float *)(uintptr_t)src;
        dst.x() = *p;
        dst.y() = *(p + 1);
        dst.z() = *(p + 2);
        dst.w() = *(p + 3);
        //{dst.x(), dst.y(), dst.z(), dst.w()} = *((float *)(uintptr_t)src);
    }
    static inline void stg(sycl::float4 *dst, const sycl::float4 &src) {
        float *p = (float *)(uintptr_t)dst;
        *p = src.x();
        *(p + 1) = src.y();
        *(p + 2) = src.z();
        *(p + 3) = src.w();
        //*((float *)(uintptr_t)dst) = {src.x(), src.y(), src.z(), src.w()};
    }
};
#ifdef KITTENS_HOPPER
template<> struct move<fp8e4m3_4> {
    static inline void ldsm4(fp8e4m3_4& dst1, fp8e4m3_4& dst2, fp8e4m3_4& dst3, fp8e4m3_4& dst4, uintptr_t src) {
        /*
        DPCT1053:17: Migration of device assembly code is not supported.
        */
        // asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, "
        //              "%2, %3}, [%4];\n"
        //              : "=r"(*(uintptr_t *)&dst1), "=r"(*(uintptr_t *)&dst2),
        //                "=r"(*(uintptr_t *)&dst3), "=r"(*(uintptr_t *)&dst4)
        //              : "r"(src));
    }
    static inline void stsm4(uintptr_t dst, fp8e4m3_4& src1, fp8e4m3_4& src2, fp8e4m3_4& src3, fp8e4m3_4& src4) {
        /*
        DPCT1053:18: Migration of device assembly code is not supported.
        */
        // asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, "
        //              "%1, %2, %3};\n" ::"r"(*(uintptr_t *)&src1),
        //              "r"(*(uintptr_t *)&src2), "r"(*(uintptr_t *)&src3),
        //              "r"(*(uintptr_t *)&src4), "r"(dst));
    }

};
template<> struct move<fp8e5m2_4> {
    static inline void ldsm4(fp8e5m2_4& dst1, fp8e5m2_4& dst2, fp8e5m2_4& dst3, fp8e5m2_4& dst4, uintptr_t src) {
        /*
        DPCT1053:19: Migration of device assembly code is not supported.
        */
        // asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, "
        //              "%2, %3}, [%4];\n"
        //              : "=r"(*(uintptr_t *)&dst1), "=r"(*(uintptr_t *)&dst2),
        //                "=r"(*(uintptr_t *)&dst3), "=r"(*(uintptr_t *)&dst4)
        //              : "r"(src));
    }
    static inline void stsm4(uintptr_t dst, fp8e5m2_4& src1, fp8e5m2_4& src2, fp8e5m2_4& src3, fp8e5m2_4& src4) {
        /*
        DPCT1053:20: Migration of device assembly code is not supported.
        */
        // asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, "
        //              "%1, %2, %3};\n" ::"r"(*(uintptr_t *)&src1),
        //              "r"(*(uintptr_t *)&src2), "r"(*(uintptr_t *)&src3),
        //              "r"(*(uintptr_t *)&src4), "r"(dst));
    }
};
#endif

/* ----------   Constants for Cache policies  ---------- */

enum cache_policy {
    NORMAL = 0,
    EVICT_FIRST = 1,
    EVICT_LAST = 2
};
template<cache_policy policy> inline uint64_t make_cache_policy() {
    uint64_t cache_policy_val;
    constexpr float fraction = 1.0f;
    static_assert(policy == cache_policy::EVICT_FIRST || policy == cache_policy::EVICT_LAST, "Unexpected cache policy");
    if constexpr (policy == cache_policy::EVICT_FIRST) {
        /*
        DPCT1053:21: Migration of device assembly code is not supported.
        */
        // asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n"
        //              : "=l"(cache_policy_val)
        //              : "f"(fraction));
    }
    else {
        /*
        DPCT1053:22: Migration of device assembly code is not supported.
        */
        // asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n"
        //              : "=l"(cache_policy_val)
        //              : "f"(fraction));
    }
    return cache_policy_val;
}
/* ----------   Generic (non-Hopper specific) semaphore functions  ---------- */

struct semaphore {
private:
    uint64_t value;
}; // note that this is an opaque type, so the value should not be accessed directly.
template<int num_warps> struct barrier {
    int barrier_id;
    __dpct_inline__ barrier(int _id) : barrier_id(_id) {}
    __dpct_inline__ barrier operator[](int i) {
        return barrier(barrier_id + i);
    }
};

/**
 * @brief Initializes a synchronization semaphore with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a semaphore that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the semaphore with a thread count semaphore.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        auto bar_ptr = ptr;

        /*
        DPCT1053:23: Migration of device assembly code is not supported.
        */
        //asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
        //             "r"(thread_count + transaction_count));
    }
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
static inline void invalidate_semaphore(semaphore& bar) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        auto bar_ptr = ptr;
        /*
        DPCT1053:24: Migration of device assembly code is not supported.
        */
        // asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(bar_ptr));
    }
}

/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
static inline void arrive(semaphore& sem) {
    // CUDA stub — not used on Intel. Use split_barrier_arrive() instead.
}
template<int num_warps> static inline void arrive(barrier<num_warps> bar) {
    // CUDA stub — not used on Intel.
}

// ═══════════════════════════════════════════════════════════════════════════
// Intel PVC split-barrier (SPV_INTEL_split_barrier extension)
//
// Splits a workgroup barrier into an Arrive phase (non-blocking: signal that
// this subgroup has issued its memory operations) and a Wait phase (blocking:
// stall until all participants have arrived).
//
// Usage pattern for PF=3 prefetch pipeline:
//
//   // Preamble: prefetch tiles 0..PF-1, arrive after each
//   for tile in [0..PF-1]:
//       prefetch(K[tile])
//       split_barrier_arrive()
//
//   // Main loop: wait for current tile, compute, prefetch next, arrive
//   for i in range(N_tiles):
//       split_barrier_wait()           // K[i] is in L1/L2
//       MMA(Q, K[i], att)
//       if i+PF < N_tiles:
//           prefetch(K[i+PF])
//           split_barrier_arrive()     // K[i+PF] prefetch issued
//
// SPIRV scope/semantics constants:
//   Subgroup=3, WorkgroupMemory=256, CrossWorkgroupMemory=512, AcqRel=8
// ═══════════════════════════════════════════════════════════════════════════

#ifdef KITTENS_INTEL
// ═══════════════════════════════════════════════════════════════════════════
// Intel PVC split-barrier (SPV_INTEL_split_barrier extension)
// Declared at global scope in split_barrier_decl.dp.hpp (SYCL_EXTERNAL for
// device, empty inline stub for host linker). Use :: so calls don't resolve
// inside the kittens namespace.
// ═══════════════════════════════════════════════════════════════════════════

// Subgroup-level — correct for single-SG-per-WG kernels (v7, v12)
// Scope::Subgroup=3, AcquireRelease|CrossWorkgroupMemory=8|512=520
static inline void split_barrier_arrive() {
    ::__spirv_ControlBarrierArriveINTEL(3u, 3u, 520u);
}
static inline void split_barrier_wait() {
    ::__spirv_ControlBarrierWaitINTEL(3u, 3u, 520u);
}

// Workgroup-level — use when multiple SGs in a WG share a barrier
// Scope::Workgroup=2, AcquireRelease|WorkgroupMemory=8|256=264
static inline void wg_split_barrier_arrive() {
    ::__spirv_ControlBarrierArriveINTEL(2u, 2u, 264u);
}
static inline void wg_split_barrier_wait() {
    ::__spirv_ControlBarrierWaitINTEL(2u, 2u, 264u);
}
#endif // KITTENS_INTEL

#ifdef KITTENS_HOPPER
/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
static inline void arrive(semaphore& sem, uintptr_t count) {
    auto mbar_ptr = &sem;
    /*
    DPCT1053:27: Migration of device assembly code is not supported.
    */
    // asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
    //              :
    //              : "r"(mbar_ptr), "r"(count)
    //              : "memory");
}
#endif

/**
* @brief Waits for the requested semaphore phase.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    auto mbar_ptr = ptr;

#ifdef KITTENS_HOPPER
    /*
    DPCT1053:28: Migration of device assembly code is not supported.
    */
    // asm volatile("{\n"
    //              ".reg .pred                P1;\n"
    //              "LAB_WAIT:\n"
    //              "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    //              "@P1                       bra.uni DONE;\n"
    //              "bra.uni                   LAB_WAIT;\n"
    //              "DONE:\n"
    //              "}\n" ::"r"(mbar_ptr),
    //              "r"(kPhaseBit));
#else
    // asm volatile (
    //     "{\n"
    //     ".reg .pred                P1;\n"
    //     "LAB_WAIT:\n"
    //     "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    //     "@P1                       bra.uni DONE;\n"
    //     "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
    //     "bra.uni                   LAB_WAIT;\n"
    //     "DONE:\n"
    //     "}\n"
    //     :: "r"(mbar_ptr),
    //     "r"(kPhaseBit)
    // );
#endif
}

/**
* @brief Checks if the requested semaphore phase is ready.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
static inline int test_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    auto mbar_ptr = ptr;
    int result;
    /*
    DPCT1053:29: Migration of device assembly code is not supported.
    */
    // asm volatile("{\n"
    //              ".reg .pred P1;\n"
    //              "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
    //              "selp.u32 %0,1,0,P1;"
    //              "}\n"
    //              : "=r"(result)
    //              : "r"(mbar_ptr), "r"(kPhaseBit));
    return result;
}

static inline void arrive_and_wait(semaphore& sem, int kPhaseBit) {
    arrive(sem);
    wait(sem, kPhaseBit);
}
template<int num_warps> static inline void arrive_and_wait(barrier<num_warps> bar) {
    /*
    DPCT1053:30: Migration of device assembly code is not supported.
    */
    // asm volatile("bar.sync %0, %1;\n" ::"r"(bar.barrier_id),
    //              "n"(num_warps * WARP_THREADS)
    //              : "memory");
}

template<int N=0> static inline void load_async_wait() { // for completing (non-TMA) async loads
    if constexpr (N == 0) {
        /*
        DPCT1026:31: The call to "cp.async.wait_all;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
        */

    } else {
        /*
        DPCT1026:32: The call to "cp.async.wait_group %0;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
        */
    }
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
}

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uintptr_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uintptr_t elements = ST::num_elements;
    static constexpr uintptr_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uintptr_t elements = SV::length;
    static constexpr uintptr_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>                       inline constexpr uintptr_t size_bytes             = 0; // base case
template<typename T, typename... Args>           inline constexpr uintptr_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens

#ifdef KITTENS_HOPPER
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tma.dp.hpp"
#endif
