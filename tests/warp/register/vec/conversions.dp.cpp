#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"

// Disable ortho layout sweeps on Intel toolchain to avoid invalid instantiations during debug.
#ifndef KITTENS_INTEL
#define KITTENS_INTEL 1
#endif

#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS

struct vec_copy_convert {
    template<int S, int NW, kittens::ducks::rv_layout::all L1, kittens::ducks::rv_layout::all L2>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_convert";
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1, kittens::ducks::rv_layout::all L2>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
        if constexpr (std::is_same_v<L1, kittens::naive_l> && std::is_same_v<L2, kittens::align_l>) {
            std::cout << "host naive->align S=" << S
                      << " ref0=" << (i_ref.empty() ? 0.0f : i_ref[0])
                      << " ref1=" << (i_ref.size() > 1 ? i_ref[1] : 0.0f)
                      << " out0=" << (o_ref.empty() ? 0.0f : o_ref[0])
                      << " out1=" << (o_ref.size() > 1 ? o_ref[1] : 0.0f)
                      << std::endl;
        }
    }
    template <int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1,
              kittens::ducks::rv_layout::all L2>
    SYCL_EXTERNAL static void device_func(const GL &input, const GL &output) {
        kittens::rv_bf<16*S, L1> vec1;
        kittens::rv_bf<16*S, L2> vec2;
        kittens::load(vec1, input, {});
        kittens::copy(vec2, vec1);
        if constexpr (std::is_same_v<L1, kittens::naive_l> && std::is_same_v<L2, kittens::align_l>) {
            // Dump a couple of packs to help debug the four failing naive->align cases.
            if (0) { //kittens::laneid() == 0 || kittens::laneid() == 1) {
                const int max_packs = 4; // keep output small
                #pragma unroll
                for (int pack = 0; pack < max_packs && pack < vec2.outer_dim; ++pack) {
                    auto p = vec2[pack][0];
                    float dst0 = kittens::base_types::convertor<float, decltype(p.x())>::convert(p.x());
                    float dst1 = kittens::base_types::convertor<float, decltype(p.y())>::convert(p.y());

                    // Sample the naive source scalars we expect to feed this pack.
                    float s0 = kittens::base_types::convertor<float, decltype(vec1[pack][0])>::convert(vec1[pack][0]);
                    float s1 = 0.0f;
                    if (pack < vec1.outer_dim) {
                        // vec1[pack][1] exists when the naive tile has the second repeat for this row.
                        s1 = kittens::base_types::convertor<float, decltype(vec1[pack][0])>::convert(
                            vec1[pack][(vec1.inner_dim > 1) ? 1 : 0]);
                    }

                }
            }
        }
        kittens::store(output, vec2, {});
    }
};

// Also exercise bf16 -> float -> bf16 conversions so the specialized
// convertor paths are covered, not just bf16->bf16 copies.
struct vec_copy_convert_float {
    template<int S, int NW, kittens::ducks::rv_layout::all L1, kittens::ducks::rv_layout::all L2>
    using valid = std::bool_constant<NW == 1 && S<=64>; // warp-level
    static inline const std::string test_identifier = "reg_vec_convert_float";
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1, kittens::ducks::rv_layout::all L2>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref;
    }
    template <int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1,
              kittens::ducks::rv_layout::all L2>
    SYCL_EXTERNAL static void device_func(const GL &input, const GL &output) {
        kittens::rv_fl<16*S, L1> vec1;
        kittens::rv_fl<16*S, L2> vec2;
        kittens::load(vec1, input, {});
        kittens::copy(vec2, vec1); // triggers bf16->float on load and float->bf16 on store
        kittens::store(output, vec2, {});
    }
};

void warp::reg::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::align_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::align_l, kittens::naive_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::naive_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::naive_l, kittens::naive_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::align_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::align_l, kittens::naive_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::naive_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::naive_l, kittens::naive_l>::run(results);
    #ifndef KITTENS_INTEL
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::align_l, kittens::ortho_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ortho_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ortho_l, kittens::ortho_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ortho_l, kittens::naive_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::naive_l, kittens::ortho_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::align_l, kittens::ortho_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::ortho_l, kittens::align_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::ortho_l, kittens::ortho_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::ortho_l, kittens::naive_l>::run(results);
    sweep_size_1d_warp<vec_copy_convert_float, SIZE, kittens::naive_l, kittens::ortho_l>::run(results);
    #endif
}

#endif