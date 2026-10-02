#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"

#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS

template<typename T>
struct shared_vec_convert {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_convert_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_convert_gmem=half" :
                                                                                         "shared_vec_convert_gmem=float";
    template<int S, int NW, gl_t GL>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL>
    static void device_func(const GL &input, const GL &output) {
        /*
        DPCT1115:56: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        auto &vec1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::sv<dtype, 16 * S>>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        /*
        DPCT1115:57: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        auto &vec2 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::sv<dtype, 16 * S>>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        kittens::load(vec1, input, {});
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        kittens::copy(vec2, vec1);
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        kittens::store(output, vec2, {});
    }
};

void warp::shared::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_1d_warp<shared_vec_convert, SIZE>::run(results);
}

#endif