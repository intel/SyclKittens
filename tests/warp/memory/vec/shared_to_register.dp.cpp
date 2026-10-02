#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared_to_register.dp.hpp"

#ifdef TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<int S, int NW, kittens::ducks::rv_layout::all L> using valid = std::bool_constant<
        (NW == 1 && S<=64)
        #ifdef KITTENS_HOPPER
        && ( !std::is_same_v<kittens::fp8e4m3, T> && !std::is_same_v<kittens::fp8e5m2, T>)
        #endif
    >; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "shared_reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "shared_reg_vec_loadstore_gmem=half" :
                                                                                             "shared_reg_vec_loadstore_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f;
        // o_ref = i_ref;
    }
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> static void device_func(const GL &input, const GL &output) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        auto __shm = sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)__shm);
        kittens::sv<dtype, 16*S> &shared_vec = al.allocate<kittens::sv<dtype, 16*S>>();
        kittens::rv<dtype, 16*S, kittens::TILE_ROW_DIM<dtype>, kittens::ducks::rt_shape::rt_16x16, L> reg_vec;
        // Step 0: Print input values from global device memory
        // if (kittens::laneid() == 0) {
        //     sycl::ext::oneapi::experimental::printf("=== Step 0: Input values in global device memory (warp %d) ===\n", kittens::warpid());
        //     sycl::ext::oneapi::experimental::printf("Input dimensions: batch=%d, depth=%d, rows=%d, cols=%d\n",
        //         input.batch(), input.depth(), input.rows(), input.cols());
        //
        //     // Method 1: Using raw pointer access
        //     // sycl::ext::oneapi::experimental::printf("--- Using raw pointer access ---\n");
        //     // for(int i = 0; i < 16*S && i < 32; i++) {
        //     //     float val = static_cast<float>(input.raw_ptr[i]);
        //     //     sycl::ext::oneapi::experimental::printf("input.raw_ptr[%d] = %f\n", i, val);
        //     // }
        //
        //     // Method 2: Using coordinate-based access (assuming 1D layout for vector)
        //     sycl::ext::oneapi::experimental::printf("--- Using coordinate access ---\n");
        //     for(int i = 0; i < 16*S && i < 32; i++) {
        //         kittens::coord<kittens::ducks::default_type> coord_idx{0, 0, 0, i}; // b=0, d=0, r=0, c=i
        //         float val = static_cast<float>(input[coord_idx]);
        //         sycl::ext::oneapi::experimental::printf("input[{0,0,0,%d}] = %f\n", i, val);
        //     }
        // }
        item_ct1.barrier();

        kittens::load(shared_vec, input, {});
        /*
        DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // sycl::ext::oneapi::experimental::printf("after first load\n");
        item_ct1.barrier();
        // Debug: Print shared vector after loading from global memory
        // if (kittens::laneid() == 0) {
        //     sycl::ext::oneapi::experimental::printf("=== After loading from global to shared (warp %d) ===\n", kittens::warpid());
        //     for(int i = 0; i < 16*S && i < 32; i++) { // Limit output to avoid too much text
        //         // float val = static_cast<float>(shared_vec[i]);
        //         sycl::ext::oneapi::experimental::printf("shared_vec[%d] = %f\n", i, shared_vec[i]);
        //     }
        // }
        item_ct1.barrier();
        kittens::load(reg_vec, shared_vec);
        // Debug: Print register vector after loading from shared memory
        // if (kittens::laneid() == 0) {
        //     sycl::ext::oneapi::experimental::printf("=== After loading from shared to register (warp %d) ===\n", kittens::warpid());
        //     for(int i = 0; i < 16*S && i < 32; i++) { // Limit output
        //         // float val = static_cast<float>(reg_vec[i]);
        //         sycl::ext::oneapi::experimental::printf("reg_vec[%d] = %f\n", i, reg_vec[i]);
        //     }
        // }
        item_ct1.barrier();
        kittens::add(reg_vec, reg_vec, float(1.));
        // sycl::ext::oneapi::experimental::printf("reg_vec address %p val %f\n", &reg_vec, reg_vec[0]);
        // Debug: Print register vector after loading from shared memory
        item_ct1.barrier();
        // if (kittens::laneid() == 0) {
        //     sycl::ext::oneapi::experimental::printf("=== After adding 1 to register (warp %d) ===\n", kittens::warpid());
        //     for(int i = 0; i < 16*S && i < 32; i++) { // Limit output
        //         // float val = static_cast<float>(reg_vec[i]);
        //         sycl::ext::oneapi::experimental::printf("reg_vec[%d] = %f\n", i, reg_vec[i]);
        //     }
        // }
        item_ct1.barrier();
        kittens::store(shared_vec, reg_vec);
        item_ct1.barrier();
        // Debug: Print register vector after loading from shared memory
        // Debug: Print shared vector after loading back from register
        // if (kittens::laneid() == 0) {
        //     sycl::ext::oneapi::experimental::printf("=== After loading back from register to shared (warp %d) ===\n", kittens::warpid());
        //     for(int i = 0; i < 16*S && i < 32; i++) { // Limit output to avoid too much text
        //         // float val = static_cast<float>(shared_vec[i]);
        //         sycl::ext::oneapi::experimental::printf("shared_vec[%d] = %f\n", i, shared_vec[i]);
        //     }
        // }
        item_ct1.barrier();
        /*
        DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        kittens::store(output, shared_vec, {});
    }
};

void warp::memory::vec::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    // sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    // sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::align>::run(results);
}

#endif