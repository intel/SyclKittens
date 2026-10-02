#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include "global_to_register.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static void g2r_global_wrapper_2d(const GL &input, const GL &output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, typename RT_SHAPE, int H, int W, int NUM_WORKERS, typename... args>
struct g2r_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {

            // constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int B = 1, D = 1, R = 1, C = 1;
            constexpr int SIZE = H*W*RT_SHAPE::rows *RT_SHAPE::cols * B * D * R * C;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, D, -1, RT_SHAPE::cols*C*W>;

            GL input(d_i, B, nullptr, RT_SHAPE::rows*R*H, nullptr);
            GL output(d_o, B, nullptr, RT_SHAPE::rows*R*H, nullptr);
            // run kernel
        {
            auto exp_props =
                sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync,
                    sycl::ext::oneapi::experimental::sub_group_size<kittens::WARP_THREADS>,
                    sycl::ext::oneapi::experimental::work_group_scratch_size(kittens::MAX_SHARED_MEMORY)};

            dpct::get_in_order_queue().submit([&](sycl::handler
                                                        &cgh) {
                    cgh.depends_on(
                        dpct::get_current_device()
                            .get_in_order_queues_last_events());

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, NUM_WORKERS * kittens::WARP_THREADS),
                            sycl::range<3>(1, 1, NUM_WORKERS * kittens::WARP_THREADS)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            global_wrapper_2d<test, dtype, H, W,
                                                NUM_WORKERS, GL,
                                                args...>(input,
                                                        output);
                        });
            });
        }
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*RT_SHAPE::cols);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(std::move(this_result));
    }
};
template<typename test, typename RT_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2r_sweep_size_2d = loop_h<g2r_wrapper_2d, test, RT_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE, int MAX_H=8, int MAX_W=8, typename... args> using g2r_sweep_size_2d_warp = g2r_sweep_size_2d<test, RT_SHAPE, MAX_H, MAX_W, 1, args...>;

template<typename T>
struct load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                    //   std::is_same_v<T, kittens::fp8e4m3> ? "reg_loadstore_gmem=fp8e4m3" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        // sycl::ext::oneapi::experimental::printf("HHH %d, WWW %d %d, %d, %d, %d tile row, %d %d\n", H, W, input.batch(), input.depth(), input.rows(), input.cols(),reg_tile.rows, reg_tile.cols);
        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        kittens::load(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};


template<typename T>
struct load_store_16x32 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                    //   std::is_same_v<T, kittens::fp8e4m3> ? "reg_loadstore_gmem=fp8e4m3" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<16*H, 32*W, L, kittens::ducks::rt_shape::rt_16x32> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        kittens::load(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};

template<typename T>
struct load_store_32x16 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<32*H, 16*W, L, kittens::ducks::rt_shape::rt_32x16> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        // sycl::ext::oneapi::experimental::printf("ssssss %d %d \n", k, l);
                        kittens::load(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};


template<typename T>
struct load_store_32x64 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<32*H, 64*W, L, kittens::ducks::rt_shape::rt_32x64> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        // sycl::ext::oneapi::experimental::printf("ssssss %d %d \n", k, l);
                        kittens::load(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};


template<typename T>
struct load_store_transpose_16x16 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_transpose_gmem16x16=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_transpose_gmem16x16=half" :
                                                                                         "reg_loadstore_transpose_gmem16x16=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // o_ref = i_ref; // overwrite the whole thing/
        std::cout << "H is " << H << " ,W is " << W << std::endl;
        for(int i = 0; i < H * 16; i++) {
            for(int j = 0; j < W * 16; j++) {
                o_ref[j * H*16 + i] = i_ref[i * W*16 + j];
        }
    }
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<16*H, 16*W, L, kittens::ducks::rt_shape::rt_16x16> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        kittens::load_transpose(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};

template<typename T>
struct load_store_transpose_32x32 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_transpose_gmem32x32=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_transpose_gmem32x32=half" :
                                                                                         "reg_loadstore_transpose_gmem32x32=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // o_ref = i_ref; // overwrite the whole thing/
        std::cout << "H is " << H << " ,W is " << W << std::endl;
        for(int i = 0; i < H * 32; i++) {
            for(int j = 0; j < W * 32; j++) {
                o_ref[j * H*32 + i] = i_ref[i * W*32 + j];
        }
    }
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<32*H, 32*W, L, kittens::ducks::rt_shape::rt_32x32> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        kittens::load_transpose(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};

template<typename T>
struct load_store_32x32 {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {

        kittens::rt_bf<32*H, 32*W, L, kittens::ducks::rt_shape::rt_32x32> reg_tile;

        for(int i = 0; i < input.batch(); i++)
            for(int j = 0; j < input.depth(); j++)
                for(int k = 0; k < input.rows()/reg_tile.rows; k++) // 4
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) { // 5
                        // sycl::ext::oneapi::experimental::printf("ssssss %d %d \n", k, l);
                        kittens::load(reg_tile, input, {i, j, k, l});
                        kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};

template<typename Source>
static void widening_load_32x32_contract(test_data &results, const std::string &label) {
    auto &queue = dpct::get_in_order_queue();
    auto release = [&](Source *pointer) { sycl::free(pointer, queue); };
    std::unique_ptr<Source, decltype(release)> input(
        sycl::aligned_alloc_shared<Source>(64, 1024, queue), release);
    if (!input) throw std::bad_alloc();
    std::fill_n(input.get(), 1024, Source(1.25f));
    std::array<float, 1024> output{};
    {
        sycl::buffer<float, 1> buffer(output.data(), sycl::range<1>(output.size()));
        Source *source = input.get();
        queue.submit([&](sycl::handler &handler) {
            auto values = buffer.get_access<sycl::access::mode::write>(handler);
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                    kittens::gl<Source, 1, 1, 32, 32> global(source, nullptr, nullptr, nullptr, nullptr);
                    kittens::rt_fl<32, 32, kittens::ducks::rt_layout::row,
                                   kittens::ducks::rt_shape::rt_32x32> tile;
                    kittens::load(tile, global, {});
                    const auto offset = item.get_local_linear_id() * 64;
                    for (int element = 0; element < tile.packed_per_base_tile; ++element) {
                        values[offset + element * 2] = tile.tiles[0][0].data[element].x();
                        values[offset + element * 2 + 1] = tile.tiles[0][0].data[element].y();
                    }
                });
        }).wait_and_throw();
    }
    const bool passed = std::all_of(output.begin(), output.end(),
        [](float value) { return test_values_match(1.25f, value, 0.0f); });
    results.push_back({label, passed ? test_result::PASSED : test_result::FAILED});
}

void warp::memory::tile::global_to_register::tests(test_data &results) {
    widening_load_32x32_contract<kittens::bf16>(results, "load_32x32_bf16_to_float");
    widening_load_32x32_contract<kittens::half>(results, "load_32x32_half_to_float");
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    //g2r_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);   // no __internal_intel_sub_group_2d_block_read_32b_32r16x2c_cache_controls when tile size is 32.
    // g2r_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);  // intel read transform intrincs only has type 8b & 16b.
    g2r_sweep_size_2d_warp<load_store<kittens::bf16>, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::bf16>, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::half>, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::half>, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store_16x32<kittens::bf16>, kittens::ducks::rt_shape::rt_16x32, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store_16x32<kittens::bf16>, kittens::ducks::rt_shape::rt_16x32, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store_32x16<kittens::bf16>, kittens::ducks::rt_shape::rt_32x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store_transpose_16x16<kittens::bf16>, kittens::ducks::rt_shape::rt_16x16, 1, 1, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store_transpose_32x32<kittens::bf16>, kittens::ducks::rt_shape::rt_32x32, 1, 1, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store_32x16<kittens::bf16>, kittens::ducks::rt_shape::rt_32x16, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store_32x32<kittens::bf16>, kittens::ducks::rt_shape::rt_32x32, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store_32x32<kittens::bf16>, kittens::ducks::rt_shape::rt_32x32, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif