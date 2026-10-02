#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared_to_register.dp.hpp"

#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct group_shared_reg_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<
        ( H%NW==0 && W*H<=64 )
        #ifdef KITTENS_HOPPER
        && ( (!std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2>) || W%2 == 0 )
        #endif
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_reg_loadstore_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                        std::is_same_v<T, kittens::fp8e4m3> ? "group_shared_reg_loadstore_gmem=fp8e4m3" :
                                                        std::is_same_v<T, kittens::fp8e5m2> ? "group_shared_reg_loadstore_gmem=fp8e5m2" :
                                                        #endif
                                                                                         "group_shared_reg_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    /*
    DPCT1110:30: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL>
    static void device_func(const GL &input, const GL &output,
                            uint8_t *dpct_local = nullptr) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory(); // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        G::load(shared_tile, input, {});
        /*
        DPCT1065:82: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        kittens::rt<dtype, 16*H/NW, 16*W, RL> reg_tile;
        G::load(reg_tile, shared_tile);
    //     /*
    //     DPCT1065:83: Consider replacing sycl::nd_item::barrier() with
    //     sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
    //     better performance if there is no access to global memory.
    //     */
        item_ct1.barrier();
        G::store(shared_tile, reg_tile);
    //     /*
    //     DPCT1065:84: Consider replacing sycl::nd_item::barrier() with
    //     sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
    //     better performance if there is no access to global memory.
    //     */
        item_ct1.barrier();
        G::store(output, shared_tile, {});
    }
};

void group::memory::tile::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);

    #ifdef KITTENS_HOPPER
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e4m3>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e4m3>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e4m3>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e5m2>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e5m2>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::fp8e5m2>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    #endif
}

#endif