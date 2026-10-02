#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "global_to_shared.dp.hpp"

#ifdef KITTENS_INTEL
#undef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
#endif

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename test, int H, int W, int NUM_WORKERS, typename axis, typename... args>
struct g2s_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        // std::cout << "\n ----- running in wrapper -----\n" << std::endl;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, axis, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, axis, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int SIZE = H*W*B*D*R*C*256;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, -1, -1, 16*C*W>;
            static_assert(axis::value==0 || axis::value==1 || axis::value==2, "Axis must be 0, 1, or 2.");
            GL input  (d_i, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            GL output (d_o, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            // run kernel

            // std::cout << i_ref[0] << " " << &i_ref[0] << " " << i_ref[1] << " " << &i_ref[1] << " " << i_ref[2] << " " << &i_ref[2] << std::endl;

            constexpr int INFO_SIZE = 8;
            // float *info_buffer_host = new float[INFO_SIZE]{};
            // float *info_buffer_device = nullptr;
            // info_buffer_device = sycl::malloc_device<float>(INFO_SIZE * sizeof(float), dpct::get_default_queue());

            /*
            DPCT1026:58: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
            /*
            DPCT1049:12: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  {
                        std::cout << "[HOST] About to launch kernel for test: " << this_result.label << std::endl;
                        std::cout << "[HOST] NUM_WORKERS=" << NUM_WORKERS << ", testing WITHOUT properties" << std::endl;
                        std::cout.flush();

                        try {
                        dpct::get_in_order_queue().submit([&](sycl::handler
                                                                  &cgh) {
                              std::cout << "[HOST] Inside submit lambda..." << std::endl;
                              std::cout.flush();
                              cgh.depends_on(
                                  dpct::get_current_device()
                                      .get_in_order_queues_last_events());

                              std::cout << "[HOST] Set dependencies, about to call parallel_for..." << std::endl;
                              std::cout.flush();

                              cgh.parallel_for(
                                  sycl::nd_range<3>(
                                      sycl::range<3>(1,1,NUM_WORKERS * kittens::WARP_THREADS),
                                      sycl::range<3>(1,1,NUM_WORKERS * kittens::WARP_THREADS)),
                                  [=](sycl::nd_item<3> item_ct1) {
                                        global_wrapper_2d<test, dtype, H, W,
                                                          NUM_WORKERS, GL, ;klid,
                                                          args...>(input,
                                                                   output);
                                  });
                        });
                        std::cout << "[HOST] Kernel submitted, waiting for completion..." << std::endl;
                        std::cout.flush();
                        dpct::get_current_device().queues_wait_and_throw();
                        std::cout << "[HOST] Kernel completed successfully" << std::endl;
                        } catch (const std::exception& e) {
                            std::cout << "[HOST] Exception caught: " << e.what() << std::endl;
                            throw;
                        }
                  }
            // fill in correct results on cpu
            // dpct::dpct_memcpy(info_buffer_host, info_buffer_device, INFO_SIZE * sizeof(float), dpct::device_to_host);

            test::template host_func<H, W, NUM_WORKERS, GL, axis, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
            // delete[] info_buffer_host;
            // sycl::free(info_buffer_device, dpct::get_default_queue());
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2s_sweep_size_2d = loop_h<g2s_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using g2s_sweep_size_2d_warp = g2s_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args>
struct g2s_sweep_gmem_type_2d_warp {
    static void run(test_data &results) {
        g2s_sweep_size_2d_warp<test<float>, MAX_H, MAX_W, args...>::run(results);
        g2s_sweep_size_2d_warp<test<kittens::bf16>, MAX_H, MAX_W, args...>::run(results);
        g2s_sweep_size_2d_warp<test<kittens::half>, MAX_H, MAX_W, args...>::run(results);
        // #ifdef KITTENS_HOPPER
        // g2s_sweep_size_2d_warp<test<kittens::fp8e4m3>, MAX_H, MAX_W, args...>::run(results);
        // g2s_sweep_size_2d_warp<test<kittens::fp8e5m2>, MAX_H, MAX_W, args...>::run(results);
        // #endif
    }
};


template<typename T>
struct st_load_store {
    using dtype = T;
    template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64)
    >;

    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "shared_loadstore_gmem=fp8e5m2":
                                                      #endif
                                                                                         "shared_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref, float *info_buffer=nullptr) {
        // std::cout << "\nenter st_load_store (host_func)\n" << std::endl;
        o_ref = i_ref; // overwrite the whole thing

        // Print info_buffer contents if present
        // if(info_buffer) {
        //     std::cout << "Info buffer from device: ";
        //     for(int i=0; i<8; ++i) // adjust size as needed
        //         std::cout << info_buffer[i] << " ";
        //     std::cout << std::endl;
        // }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void device_func(const GL &input, const GL &output) {
        // uint8_t *dpct_local = nullptr;
        // auto __shm = (kittens::alignment_dummy *)
            // dpct_local; // this is the CUDA shared memory

        // auto&& __shm = local_shm[0];
        auto __shm = sycl::ext::oneapi::experimental::get_work_group_scratch_memory();

        kittens::shared_allocator<1024> al((int*)__shm);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        // sycl::ext::oneapi::experimental::printf("shared mem %x | hello? %d %d\n", __shm, sizeof(kittens::alignment_dummy), shared_tile.rows*shared_tile.cols*sizeof(T));


        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        // out << "st.shape " << (int)shared_tile.rows << " " << (int)shared_tile.cols << sycl::endl;
        // out << "num_rows " << num_rows << " num_depths " << num_depths << " num_batches " << num_batches << sycl::endl;
        // out << "shared tile address " << &shared_tile << " thread id " << sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) << sycl::endl;
        // out << "localid " << sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_id(0) << sycl::endl;

        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            kittens::load <axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile,  input, {i, j, k, l});
            kittens::store <axis::value, false, ST, GL, kittens::coord<ST>>(output, shared_tile, {i, j, k, l});
        }
    }
};

// template<typename T>
// struct st_load_store_async {
//     using dtype = T;
//     template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
//         (NW == 1 && W*H<=64)
//         #ifdef KITTENS_HOPPER
//         && ( ( !std::is_same_v<kittens::fp8e4m3, T> && !std::is_same_v<kittens::fp8e5m2, T> ) || W%2 == 0 )
//         #endif
//     >;
//     static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_async_gmem=bf16" :
//                                                       std::is_same_v<T, kittens::half> ? "shared_loadstore_async_gmem=half" :
//                                                       #ifdef KITTENS_HOPPER
//                                                       std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_async_gmem=fp8e4m3":
//                                                       std::is_same_v<T, kittens::fp8e5m2> ? "shared_loadstore_async_gmem=fp8e5m2":
//                                                       #endif
//                                                                                          "shared_loadstore_async_gmem=float";
//     template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

//         o_ref = i_ref; // overwrite the whole thing
//     }
//     template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void device_func(const GL input, const GL output, uint8_t *dpct_local = nullptr) {
//         auto __shm = (kittens::alignment_dummy *)
//             dpct_local; // this is the CUDA shared memory
//         kittens::shared_allocator<16> al((int*)&__shm[0]);
//         using ST = kittens::st<T, 16*H, 16*W>;
//         ST &shared_tile = al.allocate<ST>();
//         int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
//         int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
//         int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
//         for(int i = 0; i < num_batches; i++)
//             for(int j = 0; j < num_depths; j++)
//                 for(int k = 0; k < num_rows; k++)
//                     for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
//             kittens::load<axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile, input, {i, j, k, l});
//             // kittens::load_async<axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile, input, {i, j, k, l});
//             // kittens::load_async_wait();
//             kittens::store<axis::value, false, ST, GL, kittens::coord<ST>>(output, shared_tile, {i, j, k, l});
//         }
//     }
// };

using I0_t = std::integral_constant<int, 0>;
using I1_t = std::integral_constant<int, 1>;
using I2_t = std::integral_constant<int, 2>;
void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    // constexpr int SIZE = 1;
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I0_t>::run(results);
    // g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I2_t>::run(results);
    // g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I1_t>::run(results);
    // g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I0_t>::run(results);
}
#endif
