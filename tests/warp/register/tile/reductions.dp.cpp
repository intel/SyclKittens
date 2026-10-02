#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <array>
#include <type_traits>
#include "reductions.dp.hpp"

// Skip ortho-dependent reductions on this toolchain.
#ifndef KITTENS_SKIP_ORTHO_LAYOUT_TESTS
#define KITTENS_SKIP_ORTHO_LAYOUT_TESTS 1
#endif

#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS

struct normalize_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            // printf("Value of row_sum %f \n", row_sum);
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
    }
    /*
    DPCT1110:20: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

struct reduce_row_max {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_max = std::max(row_max, i_ref[i*W*16+j]);
            // printf("Value of row_max %f %f \n", row_max, i_ref[i*W*16+j]);
            }
            for(int j = 0; j < W*16; j++) {
                // printf("Before dividing %f %f \n", o_ref[i*W*16+j], row_max);
                o_ref[i*W*16+j] /= row_max;
            }
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_max(accum, reg_tile);
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

struct normalize_row_16x32 {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "normalize_row_16x32";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*32; j++) {
                o_ref[i*W*32+j]  = i_ref[i*W*32+j];
                row_sum += i_ref[i*W*32+j];
            }
            for(int j = 0; j < W*32; j++) {
                o_ref[i*W*32+j] /= row_sum;
            }
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 32*W, L, kittens::ducks::rt_shape::rt_16x32> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 32*W, L, kittens::ducks::rt_shape::rt_16x32>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

struct reduce_row_max_16x32 {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reduce_row_max_16x32";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*32; j++) {
                o_ref[i*W*32+j]  = i_ref[i*W*32+j];
                row_max = std::max(row_max, i_ref[i*W*32+j]);
            }
            for(int j = 0; j < W*32; j++) {
                o_ref[i*W*32+j] /= row_max;
            }
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 32*W, L, kittens::ducks::rt_shape::rt_16x32> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 32*W, L, kittens::ducks::rt_shape::rt_16x32>::col_vec accum;
        zero(accum);
        kittens::row_max(accum, reg_tile);
        // sycl::ext::oneapi::experimental::printf("rrrrrrr max value %f \n", accum[0][0]);
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};


struct reduce_col_max {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reduce_col_max";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            float col_max = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[j*W*16+i]  = i_ref[j*W*16+i];
                col_max = std::max(col_max, i_ref[j*W*16+i]);
            }
            // printf("xxxxxxxxValue of col_max  %f \n", col_max);

            for(int j = 0; j < H*16; j++) {
                o_ref[j*W*16+i] /= col_max;
            }
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::col_max(accum, reg_tile);
        kittens::div_col(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

// Test the rt - rv sub.
struct reduce_sub {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reduce_sub";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_max = std::max(row_max, i_ref[i*W*16+j]);
            // printf("Value of row_max %f %f \n", row_max, i_ref[i*W*16+j]);
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] -= row_max;
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_max(accum, reg_tile);
        // kittens::div_row(reg_tile, reg_tile, accum);
        // accum -= accum;
        kittens::sub_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

// test the rt x rv,
struct reduce_mul {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reduce_mul";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_max = std::max(row_max, i_ref[i*W*16+j]);
            // printf("Value of row_max %f %f \n", row_max, i_ref[i*W*16+j]);
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] *= row_max;
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_max(accum, reg_tile);
        // kittens::div_row(reg_tile, reg_tile, accum);
        // accum -= accum;
        kittens::mul_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};


// test the operator *(rv, rv)
struct reduce_mul_rv {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reduce_mul";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            float row_max = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_max = std::max(row_max, i_ref[i*W*16+j]);
            // printf("Value of row_max %f %f \n", row_max, i_ref[i*W*16+j]);
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] *= (row_max*row_max);
        }
    }

    template <int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_max(accum, reg_tile);
        // kittens::div_row(reg_tile, reg_tile, accum);
        accum *= accum;
        kittens::mul_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};


struct normalize_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        kittens::div_col(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        kittens::broadcast_row(reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        kittens::broadcast_col(reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

static void row_sum_8x32_contract(test_data &results) {
    std::array<float, 32> output{};
    {
        sycl::buffer<float, 1> buffer(output.data(), sycl::range<1>(output.size()));
        dpct::get_in_order_queue().submit([&](sycl::handler &handler) {
            auto values = buffer.get_access<sycl::access::mode::write>(handler);
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                    using tile_type = kittens::rt_fl<8, 64, kittens::ducks::rt_layout::row,
                                                     kittens::ducks::rt_shape::rt_8x32>;
                    tile_type source;
                    for (int column = 0; column < tile_type::width; ++column) {
                        for (int element = 0; element < tile_type::packed_per_base_tile; ++element) {
                            const float value = float(column + 1);
                            source.tiles[0][column].data[element] = sycl::float2{value, value};
                        }
                    }
                    tile_type::col_vec accumulator;
                    kittens::row_sum(accumulator, source);
                    const auto lane = item.get_local_linear_id();
                    values[lane] = accumulator[0][0];
                    accumulator[0][0] = 5.0f;
                    kittens::row_sum(accumulator, source, accumulator);
                    values[16 + lane] = accumulator[0][0];
                });
        }).wait_and_throw();
    }
    bool reset_passed = true;
    bool accumulate_passed = true;
    for (int lane = 0; lane < 16; ++lane) {
        const float expected = lane < 8 ? 96.0f : 0.0f;
        reset_passed &= test_values_match(expected, output[lane], 0.0f);
        accumulate_passed &= test_values_match(expected + 5.0f, output[16 + lane], 0.0f);
    }
    results.push_back({"row_sum_8x32_reset", reset_passed ? test_result::PASSED : test_result::FAILED});
    results.push_back({"row_sum_8x32_accumulate", accumulate_passed ? test_result::PASSED : test_result::FAILED});
}

void warp::reg::tile::reductions::tests(test_data &results) {
    row_sum_8x32_contract(results);
    std::cout << "\n ----- Starting ops/warp/register/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<normalize_row, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_row_max, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_row_max_16x32, kittens::ducks::rt_shape::rt_16x32, 2, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row_16x32, kittens::ducks::rt_shape::rt_16x32, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_col_max, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_sub, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_mul, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<reduce_mul_rv, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif