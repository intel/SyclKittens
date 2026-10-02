#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "mma.dp.hpp"

#ifdef TEST_WARP_REGISTER_TILE_MMA

struct test_mma_AB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AB";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K_DIM*K + k]*i_ref[(RT_ACCUM_SHAPE::rows * K_DIM*H*K) + k*RT_ACCUM_SHAPE::cols*W + j];
                }
                o_ref[i*RT_ACCUM_SHAPE::cols*W + j] = sum;
            }
        }
    }

    template <typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = std::conditional_t<std::is_same_v<RT_ACCUM_SHAPE, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_32x16, kittens::ducks::rt_shape::rt_16x32>;
        using B_SHAPE = std::conditional_t<std::is_same_v<RT_ACCUM_SHAPE, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_16x32, kittens::ducks::rt_shape::rt_32x16>;
        kittens::rt_bf<RT_ACCUM_SHAPE::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<K_DIM*K, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::col, B_SHAPE > b;
        kittens::rt_fl<RT_ACCUM_SHAPE::rows*H, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_ACCUM_SHAPE> c;
        kittens::load(a, a_input, {});
        kittens::load_transform(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value,  RT_SHAPE::cols*W>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, RT_SHAPE::cols*W>;
};

struct test_mma_AB_16x32 {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "test_mma_AB_16x32";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K_DIM*K + k]*i_ref[(RT_ACCUM_SHAPE::rows * K_DIM*H*K) + k*RT_ACCUM_SHAPE::cols*W + j];
                }
                o_ref[i*RT_ACCUM_SHAPE::cols*W + j] = sum;
            }
        }
    }

    template <typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = kittens::ducks::rt_shape::rt_16x32;
        using B_SHAPE = kittens::ducks::rt_shape::rt_32x32;
        kittens::rt_bf<RT_ACCUM_SHAPE::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<K_DIM*K, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::col, B_SHAPE > b;
        kittens::rt_fl<RT_ACCUM_SHAPE::rows*H, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_ACCUM_SHAPE> c;
        kittens::load(a, a_input, {});
        kittens::load_transform(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value,  RT_SHAPE::cols*W>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, RT_SHAPE::cols*W>;
};

struct test_mma_AB_16x16 {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "test_mma_AB_16x16";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K_DIM*K + k]*i_ref[(RT_ACCUM_SHAPE::rows * K_DIM*H*K) + k*RT_ACCUM_SHAPE::cols*W + j];
                }
                o_ref[i*RT_ACCUM_SHAPE::cols*W + j] = sum;
            }
        }
    }

    template <typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = kittens::ducks::rt_shape::rt_16x32;
        using B_SHAPE = kittens::ducks::rt_shape::rt_32x16;
        kittens::rt_bf<RT_ACCUM_SHAPE::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<K_DIM*K, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::col, B_SHAPE > b;
        kittens::rt_fl<RT_ACCUM_SHAPE::rows*H, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_ACCUM_SHAPE> c;
        kittens::load(a, a_input, {});
        kittens::load_transform(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value,  RT_SHAPE::cols*W>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE::rows*H, RT_SHAPE::cols*W>;
};

struct test_mma_ABt_16x16_T {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "test_mma_ABt_16x16_T";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        std::cout << "rows is " << RT_ACCUM_SHAPE::rows << " cols is " << RT_ACCUM_SHAPE::cols <<  " K_DIM is " << K_DIM << " k is " << K << std::endl;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K*K_DIM+k]*i_ref[RT_ACCUM_SHAPE::rows*K_DIM*K*H + j*K*K_DIM + k];
                }
                // std::cout << "sum is  "<< "i " << i << " j " << j << " " << sum << "\n";
                o_ref[i*W*RT_ACCUM_SHAPE::cols+j] = sum;
            }
        }
    }

template <typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = kittens::ducks::rt_shape::rt_16x32;
        using B_SHAPE =  kittens::ducks::rt_shape::rt_16x32;
        kittens::rt_bf<RT_ACCUM_SHAPE::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<RT_ACCUM_SHAPE::cols*W, K_DIM*K, kittens::ducks::rt_layout::row, B_SHAPE> b;
        kittens::rt_fl<RT_ACCUM_SHAPE::rows*H, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_ACCUM_SHAPE> c;
        kittens::load(a, a_input, {});
        kittens::load_transpose(b, b_input, {});
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::rows*H, RT_SHAPE::cols*W>;
};

struct test_mma_ABt_16x32_T {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        std::cout << "rows is " << RT_ACCUM_SHAPE::rows << " cols is " << RT_ACCUM_SHAPE::cols <<  " K_DIM is " << K_DIM << " k is " << K << std::endl;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K*K_DIM+k]*i_ref[RT_ACCUM_SHAPE::rows*K_DIM*K*H + j*K*K_DIM + k];
                }
                o_ref[i*W*RT_ACCUM_SHAPE::cols+j] = sum;
            }
        }
    }

template <typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = kittens::ducks::rt_shape::rt_16x32;
        using B_SHAPE =  kittens::ducks::rt_shape::rt_32x32;
        kittens::rt_bf<RT_ACCUM_SHAPE::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<RT_ACCUM_SHAPE::cols*W, K_DIM*K, kittens::ducks::rt_layout::row, B_SHAPE> b;
        kittens::rt_fl<RT_ACCUM_SHAPE::rows*H, RT_ACCUM_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_ACCUM_SHAPE> c;
        kittens::load(a, a_input, {});
        kittens::load_transpose(b, b_input, {});
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1,  1, RT_SHAPE::rows*H, RT_SHAPE::cols*W>;
};
struct test_mma_AtB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtB";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i + k*16*H]*i_ref[(256*H*K) + k*16*W + j];
                }
                o_ref[i*16*W + j] = sum;
            }
        }
    }

    template <int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;

        kittens::rt_bf<16*K, 16*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<16*K, 16*W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<16*H, 16*W> c;
        kittens::load_transpose(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*H>;
    template<typename RT_SHAPE, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>;
    template<typename RT_SHAPE, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};
struct test_mma_AtBt {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtBt";
    template<typename RT_ACCUM_SHAPE, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_ACCUM_SHAPE::rows; i++) {
            for(int j = 0; j < W*RT_ACCUM_SHAPE::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i+k*H*RT_ACCUM_SHAPE::rows]*i_ref[256*K*H + j*K*16+k];
                }
                o_ref[i*W*RT_ACCUM_SHAPE::cols+j] = sum;
            }
        }
    }
    /*
    DPCT1110:18: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C,
              typename _K>
    static void device_func(const GTL_A &a_input, const GTL_B &b_input,
                            const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<16*K, 16*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<16*W, 16*K> b;
        kittens::rt_fl<16*H, 16*W> c;
        kittens::load_transpose(a, a_input, {});
        kittens::load_transpose(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtBt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};


// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, typename RT_SHAPE, int K_DIM,  int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<RT_SHAPE, K_DIM, H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}
template<typename test, typename RT_SHAPE, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        constexpr int K_DIM = std::is_same_v<RT_SHAPE, ducks::rt_shape::rt_32x32> ? 16 : 32;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(H* K*RT_SHAPE::rows * K_DIM +  W * K*RT_SHAPE::cols * K_DIM); //(H+W)*K*RT_SHAPE::rows * K_DIM);
            std::vector<float> o_ref(H*W*RT_SHAPE::rows * RT_SHAPE::cols);
            initialize(&d_i, &d_o, i_ref, o_ref);
            std::cout << "XXXXX   " << H << "   " << W << "    " << K << std::endl;
            // make descriptors
            using GTL_A = test::template make_a_layout<RT_SHAPE, K_DIM, H, W, _K>;
            using GTL_B = test::template make_b_layout<RT_SHAPE, K_DIM, H, W, _K>;
            using GTL_C = test::template make_c_layout<RT_SHAPE, K_DIM, H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*RT_SHAPE::rows * K_DIM, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
            // run kernel
                  {
                        auto exp_props =
                             sycl::ext::oneapi::experimental::properties{
                                sycl::ext::oneapi::experimental::use_root_sync,
                                sycl::ext::oneapi::experimental::sub_group_size<16>,
                                sycl::ext::oneapi::experimental::work_group_scratch_size(kittens::MAX_SHARED_MEMORY)};  // NYI

                        dpct::get_in_order_queue().submit([&](sycl::handler
                                                                  &cgh) {
                              cgh.depends_on(
                                  dpct::get_current_device()
                                      .get_in_order_queues_last_events());

                              cgh.parallel_for(
                                  sycl::nd_range<3>(
                                      sycl::range<3>(1, 1, NUM_WORKERS * 16 ),
                                      sycl::range<3>(1, 1, NUM_WORKERS * 16 )),
                                  exp_props, [=](sycl::nd_item<3> item_ct1) {
                                        mma_global_wrapper_2d<
                                            test, kittens::bf16, RT_SHAPE, K_DIM, H, W,
                                            NUM_WORKERS, GTL_A, GTL_B, GTL_C,
                                            _K, args...>(a_input, b_input,
                                                         c_output);
                                  });
                        });
                  }
            // fill in correct results on cpu
            test::template host_func<RT_SHAPE, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*RT_SHAPE::cols, 0.02); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(std::move(this_result));
    }
};

template<typename test, typename RT_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using mma_sweep_size = loop_h<mma_wrapper_2d, test, RT_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE, int MAX_H=4, int MAX_W=4, typename... args> using mma_sweep_size_warp = mma_sweep_size<test, RT_SHAPE, MAX_H, MAX_W, 1, args...>;
template <kittens::ducks::rt_shape::all RT_SHAPE>
void test_generator(test_data &results) {
  constexpr int SIZE = INTENSITY_1 ? 1 :
                         INTENSITY_2 ? 2 :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

  mma_sweep_size_warp<test_mma_AB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
  mma_sweep_size_warp<test_mma_AB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
  mma_sweep_size_warp<test_mma_AB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
  mma_sweep_size_warp<test_mma_AB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
  mma_sweep_size_warp<test_mma_AB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
  mma_sweep_size_warp<test_mma_AB_16x32, kittens::ducks::rt_shape::rt_16x32, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
  mma_sweep_size_warp<test_mma_ABt_16x32_T, kittens::ducks::rt_shape::rt_16x32, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);

  mma_sweep_size_warp<test_mma_AB_16x16, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
  mma_sweep_size_warp<test_mma_ABt_16x16_T, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);

  //to do: to support more type shape.
//   mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
//   mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
//   mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
//   mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
//   mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
//   mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
//   mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
//   mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
//   mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
//   mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
//   mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}

void warp::reg::tile::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);

}

#endif