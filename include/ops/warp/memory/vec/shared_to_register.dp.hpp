/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <type_traits>

#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"
#include "../util/util.dp.hpp"

namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    //static_assert(src.length == dst.length);//NYI

    int laneid = ::kittens::laneid();
    uintptr_t src_ptr =  reinterpret_cast<uintptr_t>(&src.data[0]);
    // sycl::ext::oneapi::experimental::printf("xxxxxxxxxxxxxxxxsrc_ptr11111 %p\n", src_ptr);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        constexpr int pack_elems = base_types::packing<T2>::num();
        // Align layout is replicated — all lanes load all elements.
        #pragma unroll
        for (int w = 0; w < dst.outer_dim; w++) {
            int base = w * dst.reductions;
            #pragma unroll
            for (int i = 0; i < dst.inner_dim; i++) {
                if ((base + i * pack_elems) < dst.length) {
                    U2 tmp;
                    move<U2>::lds(tmp, src_ptr + sizeof(U) * (base + i * pack_elems));
                    dst[w][i] = base_types::convertor<T2, U2>::convert(tmp);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
    // sycl::ext::oneapi::experimental::printf("xxxxxxxxxxxxxxxxsrc_ptr33333 %p\n", src_ptr);

        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16) {
                U tmp;
                move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*idx);
                if(laneid%2==0) dst[o_dim][0].x() =  base_types::convertor<T, U>::convert(tmp);
                else dst[o_dim][0].y() = base_types::convertor<T, U>::convert(tmp);
            }
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
            //YHJ DPCT1108:2: '__shfl_sync' was migrated with the experimental feature masked
            // dst[w][0].x() = __shfl_sync(MASK_ALL, dst[w][0].x(), leader);
            // dst[w][0].y() = __shfl_sync(MASK_ALL, dst[w][0].y(), leader+1);

          dst[w][0].x()=  dpct::experimental::select_from_sub_group(
        MASK_ALL, sycl::ext::oneapi::this_work_item::get_sub_group(),  dst[w][0].x(), leader);
            dst[w][0].y() =  dpct::experimental::select_from_sub_group(
        MASK_ALL, sycl::ext::oneapi::this_work_item::get_sub_group(),  dst[w][0].y(), leader+1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
    // sycl::ext::oneapi::experimental::printf("xxxxxxxxxxxxxxxxsrc_ptr %p\n", src_ptr);

        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
                U tmp;
                move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*(w*32 + laneid));
                dst[w][0] = base_types::convertor<T, U>::convert(tmp);
            }
        }
    }
}

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    //static_assert(dst.length == src.length);//NYI

    int laneid = ::kittens::laneid();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]); /*= static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]))*/;//NYI

    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        constexpr int pack_elems = base_types::packing<T2>::num();
        // Distribute writes across lanes to avoid conflicts.
        #pragma unroll
        for (int w = 0; w < src.outer_dim; w++) {
            int base = w * src.reductions;
            #pragma unroll
            for (int i = laneid; i < src.inner_dim; i += WARP_THREADS) {
                if ((base + i * pack_elems) < src.length) {
                    U2 tmp = base_types::convertor<U2, T2>::convert(src[w][i]);
                    move<U2>::sts(dst_ptr + sizeof(U) * (base + i * pack_elems), tmp);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.outer_dim*16) {
                U tmp;
                if (laneid % 2 == 0) tmp =
                    base_types::convertor<U, T>::convert(src[o_dim][0].x());
                else tmp =
                    base_types::convertor<U, T>::convert(src[o_dim][0].y());
                move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*idx, tmp);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                U tmp = base_types::convertor<U, T>::convert(src[w][0]);
                move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*(w*32 + laneid), tmp);
            }
        }
    }
}

}