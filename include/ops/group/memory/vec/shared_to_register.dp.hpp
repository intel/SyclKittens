// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly
 * between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared vector into register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
SYCL_EXTERNAL inline static void load(RV &dst, const SV &_src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length*N_WARPS);// confirm size correct//NYI
    auto &src = subvec_inplace<RV::length>(_src, warpid()); // pretend it's smaller and do warp-level load//NYI

    ::kittens::load(dst, src); // warp-level
}

/**
 * @brief Collaboratively store data into a shared vector from register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
SYCL_EXTERNAL inline static void store(SV &_dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length*N_WARPS);// confirm size correct//NYI

    auto &dst = subvec_inplace<RV::length>(_dst, warpid()); // pretend it's smaller and do warp-level load//NYI

    ::kittens::store(dst, src); // warp-level
}