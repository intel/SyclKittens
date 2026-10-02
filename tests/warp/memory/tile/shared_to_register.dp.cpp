#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared_to_register.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
// #define SYCL_EXT_ONEAPI_WORK_GROUP_STATIC

template<typename T>
struct sharedreg_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<
      ( NW == 1 && W*H<=64 )
      #ifdef KITTENS_HOPPER
      && ( (!std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2> ) || W%2==0 ) &&
      ( (!std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2> ) || std::is_same_v<RL, kittens::ducks::rt_layout::row> )
      #endif
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_reg_loadstore_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_reg_loadstore_gmem=fp8e4m3" :
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "shared_reg_loadstore_gmem=fp8e5m2" :
                                                      #endif
                                                                                         "shared_reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    /*
    DPCT1110:14: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all RL>
    static void device_func(const GL input, const GL output) {
        // sycl::ext::oneapi::experimental::printf("this is s2r device func\n");
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

        // auto __shm = sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        // kittens::shared_allocator<1024> al((int*)&__shm);
        // using ST = kittens::st<T, 16*H, 16*W>;
        // ST& shared_tile = al.allocate<ST>();
        auto __shm = sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator<1024> al((int*)__shm);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST& shared_tile = al.allocate<ST>();

        // sycl::ext::oneapi::experimental::printf("shared mem %x | hello? %d %d\n", __shm, sizeof(kittens::alignment_dummy), shared_tile.rows*shared_tile.cols*sizeof(T));

        kittens::load(shared_tile, input, {0, 0, 0, 0});
        /*
        DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        kittens::rt<T, 16*H, 16*W, RL> reg_tile;
        kittens::load(reg_tile, shared_tile);
        /*
        DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        kittens::store(shared_tile, reg_tile);
        /*
        DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        kittens::store(output, shared_tile, {0, 0, 0, 0});
    }
};

void warp::memory::tile::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    // SIZE = 2;

    sweep_gmem_type_2d_warp<sharedreg_load_store, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_gmem_type_2d_warp<sharedreg_load_store, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif