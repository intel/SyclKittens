#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "pgl_to_shared.dp.hpp"
#include <cmath>

#ifdef TEST_GROUP_MEMORY_VEC_PGL_TO_SHARED

template<typename Ker, int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
static void group_vec_p2s_global_wrapper_1d(const PGL input, const PGL output, const int dev_idx,
                                            uint8_t *dpct_local) {
    /*
    DPCT1084:34: The function call "group_vec_p2s_all_reduce_test::device_func"
    has multiple migration results in different template instantiations that
    could not be unified. You may need to adjust the code.
    */
    Ker::template device_func<S, NW, PGL, args...>(input, output, dev_idx,
                                                   dpct_local);
}

template<typename test, int NUM_DEVICES, int S, int NUM_WORKERS, typename... args>
struct group_vec_p2s_test_wrapper_1d {
    constexpr static int SIZE = S * 16 * 3;

    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using shared_layout = shared_layouts<dtype, NUM_DEVICES>;

    static void run(test_data& results) {
        int device_count;
        device_count = dpct::device_count();
        if (device_count < NUM_DEVICES) {
            std::cerr << "Warning: Not enough GPU devices to run test " << test::test_identifier << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = generate_test_name<S, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS>::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            int device_ids[NUM_DEVICES];
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
            initialize<NUM_DEVICES>(device_ids, d_i_arr, d_o_arr, i_ref, o_ref);

            // set parralel global layouts
            shared_pgl_replace_gl<dtype, NUM_DEVICES>(d_i_arr, d_o_arr, 1, 1, 1, 16*S*3);
            shared_layout::input_pgl->multicast_bind();
            shared_layout::output_pgl->multicast_bind();

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                /*
                DPCT1093:89: The "dev_idx" device may be not the one intended
                for use. Adjust the selected device if needed.
                */
                dpct::select_device(dev_idx);
                /*
                DPCT1026:90: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1049:33: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                        {
                              auto exp_props =
                                  sycl::ext::oneapi::experimental::properties{
                                      sycl::ext::oneapi::experimental::
                                          use_root_sync};

                              dpct::get_in_order_queue().submit([&](sycl::handler
                                                                        &cgh) {
                                    sycl::local_accessor<uint8_t, 1>
                                        dpct_local_acc_ct1(
                                            sycl::range<1>(
                                                kittens::MAX_SHARED_MEMORY),
                                            cgh);

                                    auto shared_layout_input_pgl_ct0 =
                                        *shared_layout::input_pgl;
                                    auto shared_layout_output_pgl_ct1 =
                                        *shared_layout::output_pgl;

                                    cgh.depends_on(
                                        dpct::get_current_device()
                                            .get_in_order_queues_last_events());

                                    cgh.parallel_for(
                                        sycl::nd_range<3>(
                                            sycl::range<3>(1, 1,
                                                           NUM_WORKERS * 32),
                                            sycl::range<3>(1, 1,
                                                           NUM_WORKERS * 32)),
                                        exp_props,
                                        [=](sycl::nd_item<3> item_ct1) {
                                              group_vec_p2s_global_wrapper_1d<
                                                  test, S, NUM_WORKERS,
                                                  typename shared_layout::PGL,
                                                  args...>(
                                                  shared_layout_input_pgl_ct0,
                                                  shared_layout_output_pgl_ct1,
                                                  dev_idx,
                                                  dpct_local_acc_ct1
                                                      .get_multi_ptr<
                                                          sycl::access::
                                                              decorated::no>()
                                                      .get());
                                        });
                              });
                        }
            }

            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, typename shared_layout::PGL>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, typename shared_layout::PGL, dtype>(*shared_layout::input_pgl, *shared_layout::output_pgl, i_ref, o_ref, this_result.label, S * 16);

            shared_layout::input_pgl->multicast_unbind();
            shared_layout::output_pgl->multicast_unbind();
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp OP>
struct group_vec_p2s_all_reduce_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<S%NW==0 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2s_all_reduce=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2s_all_reduce=half" :
                                                                                         "p2s_all_reduce=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
    static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
                for (int other_dev_idx = 1; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    if constexpr (OP == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if constexpr (OP == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if constexpr (OP == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    static void device_func(const PGL &input, const PGL &output, const int dev_idx,
                            uint8_t *dpct_local) {
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)dpct_local; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using SV = kittens::sv<dtype, 16*S>;
        SV &s_vec = al.allocate<SV>();
        int num_iters = (input.gl_size() + s_vec.length - 1) / s_vec.length;
        for (int i = 0; i < num_iters; i++) {
            if constexpr (OP == kittens::ReduceOp::ADD) {
                G::template all_reduce_add(s_vec, input, dev_idx, {i});
            } else if constexpr (OP == kittens::ReduceOp::MIN) {
                G::template all_reduce_min(s_vec, input, dev_idx, {i});
            } else if constexpr (OP == kittens::ReduceOp::MAX) {
                G::template all_reduce_max(s_vec, input, dev_idx, {i});
            }
            G::template store(output[dev_idx], s_vec, {i});
        }
    }
};

template<typename T>
struct group_vec_p2s_atomic_add_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<S%NW==0 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2s_atomic_add=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2s_atomic_add=half" :
                                                                                         "p2s_atomic_add=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
    static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = 0;
            }
        }
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                    o_ref[other_dev_idx][i] += i_ref[dev_idx][i];
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    static void device_func(const PGL &input, const PGL &output, const int dev_idx,
                            uint8_t *dpct_local) {
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)dpct_local; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using SV = kittens::sv<dtype, 16*S>;
        SV &s_vec = al.allocate<SV>();
        int num_iters = (input.gl_size() + s_vec.length - 1) / s_vec.length;
        for (int i = 0; i < num_iters; i++) {
            G::template load(s_vec, input[dev_idx], {i});
            G::template atomic_add(output, s_vec, dev_idx, {i});
        }
    }
};

template<typename T>
struct group_vec_p2s_broadcast_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<S%NW==0 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = true;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2s_broadcast=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2s_broadcast=half" :
                                                                                         "p2s_broadcast=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
    static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    static void device_func(const PGL &input, const PGL &output, const int dev_idx,
                            uint8_t *dpct_local) {
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)dpct_local; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using SV = kittens::sv<dtype, 16*S>;
        SV &s_vec = al.allocate<SV>();
        int num_iters = (input.gl_size() + s_vec.length - 1) / s_vec.length;
        for (int i = 0; i < num_iters; i++) {
            G::template load(s_vec, input[dev_idx], {i});
            G::template broadcast(output, s_vec, dev_idx, {i});
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_S, int NUM_WORKERS, typename... args>
using group_p2s_sweep_size_1d = mg_loop_s<group_vec_p2s_test_wrapper_1d, test, NUM_DEVICES, MAX_S, NUM_WORKERS, MAX_S, args...>;

template<typename test, int NUM_DEVICES, int MAX_S, typename... args>
struct group_p2s_sweep_size_1d_warp_nw {
    static void run(test_data &results) {
        group_p2s_sweep_size_1d<test, NUM_DEVICES, MAX_S, 2>::run(results);
        group_p2s_sweep_size_1d<test, NUM_DEVICES, MAX_S, 4>::run(results);
        group_p2s_sweep_size_1d<test, NUM_DEVICES, MAX_S, 12>::run(results);
    }
};

// This might seem like an overkill, but is needed to minimize the number of PGL instantiations
template<typename T, int NUM_DEVICES, int MAX_S, typename... args>
struct group_vec_p2s_sweep_size_1d_warp_axes_ops {
    static void run(test_data &results) {
        using shared_layout = shared_layouts<T, NUM_DEVICES>;

        // Initialize shared PGLs
        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
        T *d_arr[NUM_DEVICES] = {}; // nullptrs (create dummy GLs)
        shared_layout::input_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16*MAX_S*3); // dummy dimensions, only used for mc allocation
        shared_layout::output_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16*MAX_S*3);
        shared_layout::input_pgl->multicast_init(); // can't to bind, but init is fine with just the dimensions
        shared_layout::output_pgl->multicast_init();

        group_p2s_sweep_size_1d_warp_nw<group_vec_p2s_all_reduce_test<T, kittens::ReduceOp::ADD>, NUM_DEVICES, MAX_S>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            group_p2s_sweep_size_1d_warp_nw<group_vec_p2s_all_reduce_test<T, kittens::ReduceOp::MIN>, NUM_DEVICES, MAX_S>::run(results);
            group_p2s_sweep_size_1d_warp_nw<group_vec_p2s_all_reduce_test<T, kittens::ReduceOp::MAX>, NUM_DEVICES, MAX_S>::run(results);
        }

        group_p2s_sweep_size_1d_warp_nw<group_vec_p2s_atomic_add_test<T>, NUM_DEVICES, MAX_S>::run(results);
        group_p2s_sweep_size_1d_warp_nw<group_vec_p2s_broadcast_test<T>, NUM_DEVICES, MAX_S>::run(results);

        // Delete shared PGLs
        shared_layout::input_pgl->multicast_destroy();
        shared_layout::output_pgl->multicast_destroy();
        delete shared_layout::input_pgl;
        delete shared_layout::output_pgl;
    }
};

void group::memory::vec::pgl_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/pgl_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    if (check_multi_gpus()) {
        group_vec_p2s_sweep_size_1d_warp_axes_ops<float, NUM_GPUS, SIZE>::run(results);
        group_vec_p2s_sweep_size_1d_warp_axes_ops<kittens::bf16, NUM_GPUS, SIZE>::run(results);
        group_vec_p2s_sweep_size_1d_warp_axes_ops<kittens::half, NUM_GPUS, SIZE>::run(results);
    }
}

#endif
