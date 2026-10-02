#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <dpct/dpct.hpp>
#include "reductions.dp.hpp"
#include <sycl/ext/intel/math.hpp>

#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS

struct group_normalize_row {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "group_shared_norm_row";
    template<int H, int W, int N, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
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
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
        for (int i = 0; i < o_ref.size(); i++)
            o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
    }
    template <int H, int W, int N, gl_t GL>
    SYCL_EXTERNAL static void device_func(const GL &input, const GL &output,
                                          uint8_t *dpct_local = nullptr) {
        using G = kittens::group<N>;
        auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st_bf<16*H, 16*W> &shared_tile = al.allocate<kittens::st_bf<16*H, 16*W>>();

        sycl::ext::oneapi::experimental::work_group_static<kittens::col_vec<typeof(shared_tile)>> test1;
        kittens::col_vec<typeof(shared_tile)> &accum  = *(kittens::col_vec<typeof(shared_tile)> *)&test1;
        G::load(shared_tile, input, {});
        G::sync(0);
        G::row_sum(accum, shared_tile);
        G::sync(0);
        G::div_row(shared_tile, shared_tile, accum);
        G::sync(0);
        G::store(output, shared_tile, {});
    }
};
struct group_normalize_col {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "group_shared_norm_col";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
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
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
        for (int i = 0; i < o_ref.size(); i++)
            o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output, uint8_t *dpct_local = nullptr) {
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st_bf<16*H, 16*W> &shared_tile = al.allocate<kittens::st_bf<16*H, 16*W>>();
        /*
        DPCT1115:51: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        sycl::ext::oneapi::experimental::work_group_static<kittens::row_vec<typeof(shared_tile)>> test1;
        kittens::row_vec<typeof(shared_tile)> &accum  = *(kittens::row_vec<typeof(shared_tile)> *)&test1;
        G::load(shared_tile, input, {});
        G::sync(0);
        G::col_sum(accum, shared_tile);
        G::sync(0);
        G::div_col(shared_tile, shared_tile, accum);
        G::sync(0);
        G::store(output, shared_tile, {});
    }
};
struct group_broadcast_row {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "group_shared_broadcast_row";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
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
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
        for (int i = 0; i < o_ref.size(); i++)
            o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
    }
    template<int H, int W, int N, gl_t GL> static void device_func(const GL &input, const GL &output, uint8_t *dpct_local = nullptr) {
        using G = kittens::group<N>;
        auto __shm = (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st_bf<16*H, 16*W> &shared_tile = al.allocate<kittens::st_bf<16*H, 16*W>>();
        /*
        DPCT1115:52: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        sycl::ext::oneapi::experimental::work_group_static<kittens::col_vec<typeof(shared_tile)>> test1;
        kittens::col_vec<typeof(shared_tile)> &accum  = *(kittens::col_vec<typeof(shared_tile)> *)&test1;
        G::load(shared_tile, input, {});
        G::sync(0);
        G::row_sum(accum, shared_tile);
        G::sync(0);
        G::broadcast_row(shared_tile, accum);
        G::sync(0);
        G::store(output, shared_tile, {});
    }
};
struct group_broadcast_col {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "group_shared_broadcast_col";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
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
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
        for (int i = 0; i < o_ref.size(); i++)
            o_ref_f[i] = sycl::ext::intel::math::bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output, uint8_t *dpct_local = nullptr) {
        using G = kittens::group<NW>;
        auto __shm =  (kittens::alignment_dummy *)
            sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::st_bf<16*H, 16*W> &shared_tile = al.allocate<kittens::st_bf<16*H, 16*W>>();
        /*
        DPCT1115:53: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        sycl::ext::oneapi::experimental::work_group_static<kittens::row_vec<typeof(shared_tile)>> test1;
        kittens::row_vec<typeof(shared_tile)> &accum  = *(kittens::row_vec<typeof(shared_tile)> *)&test1;
        G::load(shared_tile, input, {});
        G::sync(0);
        G::col_sum(accum, shared_tile);
        G::sync(0);
        G::broadcast_col(shared_tile, accum);
        G::sync(0);
        G::store(output, shared_tile, {});
    }
};

void group::shared::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<group_normalize_row, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_normalize_col, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_broadcast_row, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_broadcast_col, SIZE, SIZE, 2>::run(results);


    if constexpr (TEST_INTENSITY > 1) {

        sweep_size_2d<group_normalize_row, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<group_normalize_col, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<group_broadcast_row, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<group_broadcast_col, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_size_2d<group_normalize_row, 12, 5, 12>::run(results);
            sweep_size_2d<group_normalize_col, 12, 5, 12>::run(results);
            sweep_size_2d<group_broadcast_row, 12, 5, 12>::run(results);
            sweep_size_2d<group_broadcast_col, 12, 5, 12>::run(results);

        }
    }
}

#endif