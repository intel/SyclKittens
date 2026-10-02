#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <dpct/dpct.hpp>
#include "reductions.dp.hpp"
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#ifdef TEST_WARP_SHARED_TILE_REDUCTIONS

template<typename T>
struct normalize_row {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_norm_row_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_norm_row_gmem=half" :
                                                                                         "shared_norm_row_gmem=float";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++)
                i_ref[i] = sycl::ext::intel::math::float2bfloat16(i_ref_f[i]);
            for(int i = 0; i < H*16; i++) {
                kittens::bf16 row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = sycl::ext::oneapi::experimental::fabs(row_sum) +
                          sycl::ext::intel::math::float2bfloat16(1.f);
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i];
            for(int i = 0; i < H*16; i++) {
                float row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = abs(row_sum)+1.f;
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++)
                i_ref[i] = sycl::ext::intel::math::float2half_rn(i_ref_f[i]);
            for(int i = 0; i < H*16; i++) {
                kittens::half row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = sycl::ext::intel::math::habs(row_sum) +
                          sycl::ext::intel::math::float2half_rn(1.f);
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::half2float(o_ref[i]);
        }
    }
    //YHJ: Add default to uint8_t *dpct_local = nullptr
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output) {
                auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        /*
        DPCT1115:59: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        // auto &accum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        //     kittens::col_vec<typeof(shared_tile)>>(
        //     sycl::ext::oneapi::this_work_item::get_work_group<3>());
        sycl::ext::oneapi::experimental::work_group_static<kittens::col_vec<typeof(shared_tile)>> test;
        kittens::col_vec<typeof(shared_tile)> &accum = *(kittens::col_vec<typeof(shared_tile)> *)&test;

        kittens::load(shared_tile, input, {});
        kittens::row_sum(accum, shared_tile);
        kittens::abs(accum, accum);
        kittens::add(accum, accum, kittens::base_types::constants<dtype>::one());
        kittens::div_row(shared_tile, shared_tile, accum);
        kittens::store(output, shared_tile, {});
    }
};
template<typename T>
struct normalize_col {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_norm_col_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_norm_col_gmem=half" :
                                                                                         "shared_norm_col_gmem=float";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++)
                i_ref[i] = sycl::ext::intel::math::float2bfloat16(i_ref_f[i]);
            for(int i = 0; i < W*16; i++) {
                kittens::bf16 col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = sycl::ext::oneapi::experimental::fabs(col_sum) +
                          sycl::ext::intel::math::float2bfloat16(1.f);
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i];
            for(int i = 0; i < W*16; i++) {
                float col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = abs(col_sum)+1.f;
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++)
                i_ref[i] = sycl::ext::intel::math::float2half_rn(i_ref_f[i]);
            for(int i = 0; i < W*16; i++) {
                kittens::half col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = sycl::ext::intel::math::habs(col_sum) +
                          sycl::ext::intel::math::float2half_rn(1.f);
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::half2float(o_ref[i]);
        }
    }
    //YHJ: Add default to uint8_t *dpct_local = nullptr
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output) {
        auto __shm = (kittens::alignment_dummy *)
        sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        /*
        DPCT1115:60: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        // auto &accum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        //     kittens::row_vec<typeof(shared_tile)>>(
        //     sycl::ext::oneapi::this_work_item::get_work_group<3>());

        sycl::ext::oneapi::experimental::work_group_static<kittens::row_vec<typeof(shared_tile)>> test;
        kittens::row_vec<typeof(shared_tile)> &accum = *(kittens::row_vec<typeof(shared_tile)> *)&test;

        kittens::load(shared_tile, input, {});
        kittens::col_sum(accum, shared_tile);
        kittens::abs(accum, accum);
        kittens::add(accum, accum, kittens::base_types::constants<dtype>::one());
        kittens::div_col(shared_tile, shared_tile, accum);
        kittens::store(output, shared_tile, {});
    }
};
template<typename T>
struct broadcast_row {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_broadcast_row_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_broadcast_row_gmem=half" :
                                                                                         "shared_broadcast_row_gmem=float";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++) i_ref[i] =
                sycl::ext::intel::math::float2bfloat16(i_ref_f[i] / 8.f);
            for(int i = 0; i < H*16; i++) {
                kittens::bf16 row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i]/8.f;
            for(int i = 0; i < H*16; i++) {
                float row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++) i_ref[i] =
                sycl::ext::intel::math::float2half_rn(i_ref_f[i] / 8.f);
            for(int i = 0; i < H*16; i++) {
                kittens::half row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::half2float(o_ref[i]);
        }
    }
    //YHJ: Add default to uint8_t *dpct_local = nullptr
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output) {
        auto __shm = (kittens::alignment_dummy *)sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        /*
        DPCT1115:61: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */

        // auto &accum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        //     kittens::col_vec<typeof(shared_tile)>>(
        //     sycl::ext::oneapi::this_work_item::get_work_group<3>());
        sycl::ext::oneapi::experimental::work_group_static<kittens::col_vec<typeof(shared_tile)>> test;
        kittens::col_vec<typeof(shared_tile)> &accum = *(kittens::col_vec<typeof(shared_tile)> *)&test;


        kittens::load(shared_tile, input, {});
        kittens::mul(shared_tile, shared_tile, kittens::base_types::convertor<dtype, float>::convert(0.125f));
        kittens::row_sum(accum, shared_tile);
        kittens::broadcast_row(shared_tile, accum);
        kittens::store(output, shared_tile, {});
    }
};

template<typename T>
struct broadcast_col {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_broadcast_col_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_broadcast_col_gmem=half" :
                                                                                         "shared_broadcast_col_gmem=float";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++) i_ref[i] =
                sycl::ext::intel::math::float2bfloat16(i_ref_f[i] / 8.f);
            for(int i = 0; i < W*16; i++) {
                kittens::bf16 col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i]/8.f;
            for(int i = 0; i < W*16; i++) {
                float col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for (int i = 0; i < i_ref.size(); i++) i_ref[i] =
                sycl::ext::intel::math::float2half_rn(i_ref_f[i] / 8.f);
            for(int i = 0; i < W*16; i++) {
                kittens::half col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
            }
            for (int i = 0; i < o_ref.size(); i++)
                o_ref_f[i] = sycl::ext::intel::math::half2float(o_ref[i]);
        }
    }
    //YHJ: Add default to uint8_t *dpct_local = nullptr
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output) {
        auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        /*
        DPCT1115:62: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        // auto &accum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        //     kittens::row_vec<typeof(shared_tile)>>(
        //     sycl::ext::oneapi::this_work_item::get_work_group<3>());
        sycl::ext::oneapi::experimental::work_group_static<kittens::row_vec<typeof(shared_tile)>> test;
        kittens::row_vec<typeof(shared_tile)> &accum = *(kittens::row_vec<typeof(shared_tile)> *)&test;

        kittens::load(shared_tile, input, {});
        kittens::mul(shared_tile, shared_tile, kittens::base_types::convertor<dtype, float>::convert(0.125f));
        kittens::col_sum(accum, shared_tile);
        kittens::broadcast_col(shared_tile, accum);
        kittens::store(output, shared_tile, {});
    }
};

void warp::shared::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_2d_warp<normalize_row, SIZE, SIZE>::run(results);
    // sweep_gmem_type_2d_warp<normalize_col, SIZE, SIZE>::run(results);
    // sweep_gmem_type_2d_warp<broadcast_row, SIZE, SIZE>::run(results);
    // sweep_gmem_type_2d_warp<broadcast_col, SIZE, SIZE>::run(results);
}

#endif