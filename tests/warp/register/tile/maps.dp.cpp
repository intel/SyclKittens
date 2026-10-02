#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "maps.dp.hpp"
#include <cmath>

#ifdef TEST_WARP_REGISTER_TILE_MAPS

struct test_exp2 {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_exp";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for(int i = 0; i < i_ref.size(); i++){
            o_ref[i] = ::exp2f(i_ref[i]);
            // std::cout << "Value " << i_ref[i] << ",  Reuslt " <<  o_ref[i]  << std::endl;
        }
    }
    /*
    DPCT1110:11: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::exp2(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};


struct test_exp {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_exp";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for(int i = 0; i < i_ref.size(); i++){
            o_ref[i] = ::expf(i_ref[i]);
            // std::cout << "Value " << i_ref[i] << ",  Reuslt " <<  o_ref[i]  << std::endl;
        }
    }
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::exp(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};

struct test_sub_rt_rv {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "test_exp_rt_rv";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for(int i = 0; i < i_ref.size(); i++){
            o_ref[i] = (i_ref[i] + 1.0f);
            // std::cout << "Value " << i_ref[i] << ",  Reuslt " <<  o_ref[i]  << std::endl;
        }
    }
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec col_v;
        one(col_v);
        kittens::load(reg_tile, input, {});
        reg_tile += col_v;
        kittens::store(output, reg_tile, {});
    }
};

struct test_inv {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_inv";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for(int i = 0; i < i_ref.size(); i++){
            o_ref[i] =  sycl::ext::intel::math::inv(i_ref[i]);
            // std::cout << "Value " << i_ref[i] << ",  Reuslt " <<  o_ref[i]  << std::endl;
        }
    }
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::inv(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};

struct test_exp_32x32 {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "test_exp_32x32";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for(int i = 0; i < i_ref.size(); i++){
            o_ref[i] = ::expf(i_ref[i]);
            // std::cout << "Value " << i_ref[i] << ",  Reuslt " <<  o_ref[i]  << std::endl;
        }
    }
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    SYCL_EXTERNAL static void device_func(const GL input, const GL output) {
        kittens::rt_bf<32*H, 32*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::exp(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};

void warp::reg::tile::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests!aaa -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<test_exp, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_sub_rt_rv, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_inv, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_exp2, kittens::ducks::rt_shape::rt_16x16 , SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_exp, kittens::ducks::rt_shape::rt_16x16, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_exp_32x32, kittens::ducks::rt_shape::rt_32x32, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
     sweep_size_2d_warp<test_exp_32x32, kittens::ducks::rt_shape::rt_32x32, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif