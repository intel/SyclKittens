#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <type_traits>
#include "kittens.dp.hpp"

/* --------------------  CUDA ERROR UTILS  -------------------- */

#include <stdio.h>
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char *file, const int line) try {
    /*
    DPCT1010:38: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    dpct::err0 err = 0;

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/* --------------------  TEST STRUCTS  -------------------- */

/*
General Unit Test Struct
struct test {
    template<typename... args> using valid = true; // Set this invalid if you don't want the test to be compiled and run.
    static inline const std::string test_identifier; ("block::load" as an example.)
    template<typename... args> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref);
    template<typename... args> __global__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output);
};
*/
enum test_result {
    PASSED = 0,
    FAILED = 1,
    INVALID = 2 // This is a useful one for tests that are only defined for certain template specializations, but we still want to sweep.
};
struct test_info {
    std::string label;
    test_result result = test_result::INVALID;
};
using test_data = std::vector<test_info>;


/* --------------------  TEMPLATE METAPROGRAMMING UTILS  -------------------- */

// 1D wrapper
template<template<typename,int,int,typename...> typename base, typename test, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct loop_s {
    static void run(test_data& results) {
        if constexpr (S > 1) {
            loop_s<base, test, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
        base<test, S, NUM_WORKERS, args...>::run(results);
    }
};

// 2D wrappers
template<template<typename, typename, int, int,int,typename...> typename base, typename test, typename RT_SHAPE, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct loop_w {
    static void run(test_data& results) {
        if constexpr (W > 1) {
            loop_w<base, test, RT_SHAPE, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
        base<test, RT_SHAPE, H, W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename, typename, int,int,int, typename...> typename base, typename test, typename RT_SHAPE, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct loop_h {
    static void run(test_data& results) {
        if constexpr (H > 1) {
            loop_h<base, test, RT_SHAPE, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
        loop_w<base, test, RT_SHAPE, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
    }
};

// 1D wrapper for multi-gpu tests
template<template<typename, typename, int,int,int,typename...> typename base, typename test, typename RT_SHAPE, int NUM_DEVICES, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct mg_loop_s {
    static void run(test_data& results) {
        if constexpr (S > 1) {
            mg_loop_s<base, test, RT_SHAPE, NUM_DEVICES, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
        base<test, RT_SHAPE, NUM_DEVICES, S, NUM_WORKERS, args...>::run(results);
    }
};

// 2D wrappers for multi-gpu tests
template<template<typename, typename, int, int,int,int,typename...> typename base, typename test, typename RT_SHAPE, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct mg_loop_w {
    static void run(test_data& results) {
        if constexpr (W > 1) {
            mg_loop_w<base, test, RT_SHAPE,  NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
        base<test, RT_SHAPE, NUM_DEVICES, H, W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename, typename,int, int,int,int,typename...> typename base, typename test, typename RT_SHAPE, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct mg_loop_h {
    static void run(test_data& results) {
        if constexpr (H > 1) {
            mg_loop_h<base, test, RT_SHAPE, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
        mg_loop_w<base, test, RT_SHAPE, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
    }
};

/* --------------------  TEST INITIALIZE+VALIDATE FUNCS  -------------------- */

enum initializers {
    RANDOM = 0, // uniform random init. useful for confirming correctness.
    ARANGE = 1, // write an increasing sequence into i_ref and d_i arrays. useful for debugging memory movement.
    NONE   = 2  // use whatever values were already in i_ref. useful for detailed debugging.
};
// template<typename T, initializers initializer=initializers::ARANGE, int SEED=42>
template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(T **d_i, T **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {
    using namespace kittens;

    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();

    // Initialize matrices
    std::vector<T> i_t(input_size);

    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int idx = 0; idx < input_size; idx++) {
        float f;
        if constexpr (initializer == initializers::RANDOM) {
            f = dis(gen);
            // f = float(1.0);
        }
        else if constexpr (initializer == initializers::ARANGE) {  //3k
            f = float(idx);
        }
        else {
            f = i_ref[idx];
        }
        if constexpr (std::is_same_v<T, bf16>) {
            i_t[idx] =static_cast<T>(f); // fill in for transfer to device
            i_ref[idx] =
                static_cast<float>(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
            // printf("i_ref[%d] cccc= %f %x %f\n", idx, i_ref[idx], idx, f);
        }
        else if constexpr (std::is_same_v<T, float>) {
            i_t[idx] = f;
            i_ref[idx] = f;
            // printf("i_ref[%d] =aaaa %f %x %f, %d\n", idx, i_t[idx], idx, f, sizeof(T));

        } else if constexpr (std::is_same_v<T, sycl::half>) {
            i_t[idx] = sycl::ext::intel::math::float2half_rn(f);

            // printf("i_ref[%d] = %f %x %f, %d\n", idx, sycl::ext::intel::math::half2float(i_t[idx]), idx, f, sizeof(T));
            i_ref[idx] = sycl::ext::intel::math::half2float(i_t[idx]);
        }
        #ifdef KITTENS_HOPPER
        else if constexpr (std::is_same_v<T, fp8e4m3>) {
            i_t[idx] = __nv_fp8_e4m3(f);
            i_ref[idx] = float(i_t[idx]);
        }
        else if constexpr (std::is_same_v<T, fp8e5m2>) {
            i_t[idx] = __nv_fp8_e5m2(f);
            i_ref[idx] = float(i_t[idx]);
        }
        #endif
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported test element type");
        }
    }

    *d_i = sycl::malloc_device<T>(input_size, dpct::get_in_order_queue());
    *d_o = sycl::malloc_device<T>(output_size, dpct::get_in_order_queue());
    CudaCheckError();

    dpct::get_in_order_queue()
        .memcpy(*d_i, i_t.data(), input_size * sizeof(T))
        .wait();
    CudaCheckError();
}

#if defined(TEST_GANG) || defined(TEST_WARP_MEMORY_TILE_PGL_TO_REGISTER) || \
    defined(TEST_WARP_MEMORY_TILE_PGL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_TILE_TMA_PGL) || \
    defined(TEST_WARP_MEMORY_VEC_PGL_TO_REGISTER) || \
    defined(TEST_WARP_MEMORY_VEC_PGL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_VEC_TMA_PGL) || \
    defined(TEST_WARP_MEMORY_UTIL_REDUCE) || \
    defined(TEST_GROUP_MEMORY_TILE_PGL_TO_REGISTER) || \
    defined(TEST_GROUP_MEMORY_TILE_PGL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_VEC_PGL_TO_REGISTER) || \
    defined(TEST_GROUP_MEMORY_VEC_PGL_TO_SHARED)

// Initializer for legacy multi-GPU PGL tests.
template<int NUM_DEVICES, typename T, initializers initializer=initializers::RANDOM, int SEED=42>
static void initialize(int *device_ids, T **d_i_arr, T **d_o_arr, std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {

    const int input_size  = i_ref[0].size();
    const int output_size = o_ref[0].size();

    // Initialize matrices
    std::vector<T> i_t(input_size);

    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        for(int idx = 0; idx < input_size; idx++) {
            float f;
            if constexpr (initializer == initializers::RANDOM) {
                f = dis(gen);
            }
            else if constexpr (initializer == initializers::ARANGE) {
                f = float(idx);
            }
            else {
                f = i_ref[dev_idx][idx];
            }
            if constexpr (std::is_same_v<T, kittens::bf16>) {
                i_t[idx] = sycl::ext::intel::math::float2bfloat16(
                    f); // fill in for transfer to device
                i_ref[dev_idx][idx] = sycl::ext::intel::math::bfloat162float(
                    i_t[idx]); // ensure lossiness of fp16 is captured on cpu
            }
            else if constexpr (std::is_same_v<T, float>) {
                i_t[idx] = f;
                i_ref[dev_idx][idx] = f;
            }
            else if constexpr (std::is_same_v<T, kittens::half>) {
                i_t[idx] = sycl::ext::intel::math::float2half_rn(f);
                i_ref[dev_idx][idx] =
                    sycl::ext::intel::math::half2float(i_t[idx]);
            }
            else {
                assert(false && "Unsupported data type");
            }
        }

        /*
        DPCT1093:41: The "dev_idx" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev_idx);
        kittens::pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_i_arr[dev_idx], input_size * sizeof(T));
        kittens::pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_o_arr[dev_idx], output_size * sizeof(T));
        dpct::get_in_order_queue()
            .memcpy(d_i_arr[dev_idx], i_t.data(), input_size * sizeof(T))
            .wait();
        CudaCheckError();
    }
}

#endif

extern int should_write_outputs;

inline bool test_values_match(float expected, float actual, float tolerance) {
    return std::isfinite(expected) && std::isfinite(actual) &&
           std::abs(expected - actual) <= tolerance;
}

template<typename T>
test_result validate(T *d_i, T *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps=0.1) { // default eps has to be fairly high due to lots of different types
    using namespace kittens;
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    // copy back
    T* o_t = new T[output_size]{};
    float *o = new float[output_size]{};
    dpct::get_current_device().queues_wait_and_throw();
    CudaCheckError();
    dpct::get_in_order_queue().memcpy(o_t, d_o, output_size * sizeof(T)).wait();
    CudaCheckError();
    for(int idx = 0; idx < output_size; idx++) {
        if constexpr (std::is_same_v<T, bf16>) {
            o[idx] = sycl::ext::intel::math::bfloat162float(o_t[idx]);
            o_ref[idx] = sycl::ext::intel::math::bfloat162float(
                sycl::ext::intel::math::float2bfloat16(o_ref[idx]));
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            o[idx] = sycl::ext::intel::math::half2float(o_t[idx]);
            o_ref[idx] = sycl::ext::intel::math::half2float(
                sycl::ext::intel::math::float2half_rn(o_ref[idx]));
        }
        else if constexpr(std::is_same_v<T, float>) {
            o[idx] = o_t[idx];
        }
        #ifdef KITTENS_HOPPER
        else if constexpr(std::is_same_v<T, fp8e4m3>) {
            o[idx] = float(o_t[idx]);
            o_ref[idx] = float(__nv_fp8_e4m3(o_ref[idx]));
        }
        else if constexpr(std::is_same_v<T, fp8e5m2>) {
            o[idx] = float(o_t[idx]);
            o_ref[idx] = float(__nv_fp8_e5m2(o_ref[idx]));
        }
        #endif
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported test element type");
        }
    }
    // check
    std::cout << "test `" << test_name << "`";
    bool good = true;
    for(int i = 0; i < output_size; i++) {
        // std::cout << "output Ref : I["<< i << "]  " << o_ref[i] << "\n";
        // std::cout << "output : I["<< i << "]  " << o[i] << "\n";

        if(!test_values_match(o_ref[i], o[i], eps)) {
            std::cout << i << " fail ref: " << o_ref[i] <<  " o: " << o[i] << "\n";

            for(int j = 0; j < output_size; j++) {
                std::cout << "output Ref vs Output : I["<< j << "]  " << o_ref[j] << " " << o[j] << "\n";
            }

            good = false;
            break;
        }

    }
    if(good) std::cout << " -- PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;
    if(should_write_outputs && !good) {
        std::ofstream reffile("outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("outputs/"+test_name+"_out.txt");
        for(int i = 0; i < output_size; i++) {
            reffile << o_ref[i] << ' ';
            outfile << o[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
                outfile << '\n';
            }
        }
        reffile << "\n\n\nINPUTS:\n\n";
        for(int i = 0; i < input_size; i++) {
            reffile << i_ref[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
            }
        }
        reffile.close();
        outfile.close();
    }
    dpct::dpct_free(d_i, dpct::get_in_order_queue());
    dpct::dpct_free(d_o, dpct::get_in_order_queue());
    // delete[] o_t, o;
    delete[] o_t;
    delete[] o;
    CudaCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}

#if defined(TEST_GANG) || defined(TEST_WARP_MEMORY_TILE_PGL_TO_REGISTER) || \
    defined(TEST_WARP_MEMORY_TILE_PGL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_TILE_TMA_PGL) || \
    defined(TEST_WARP_MEMORY_VEC_PGL_TO_REGISTER) || \
    defined(TEST_WARP_MEMORY_VEC_PGL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_VEC_TMA_PGL) || \
    defined(TEST_WARP_MEMORY_UTIL_REDUCE) || \
    defined(TEST_GROUP_MEMORY_TILE_PGL_TO_REGISTER) || \
    defined(TEST_GROUP_MEMORY_TILE_PGL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_VEC_PGL_TO_REGISTER) || \
    defined(TEST_GROUP_MEMORY_VEC_PGL_TO_SHARED)

// Validation for multi-gpu tests
template<int NUM_DEVICES, kittens::ducks::pgl::all PGL, typename T>
test_result validate(PGL &input, PGL &output, const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref, std::string test_name, int cols=16, float eps=1e-1) { // default eps even higher due to multiple GPUs parallelizing
    const int input_size  = i_ref[0].size();
    const int output_size = o_ref[0].size();

    // copy back
    T* o_t = new T[NUM_DEVICES * output_size]{};
    float *o = new float[NUM_DEVICES * output_size]{};

    std::cout << "test `" << test_name << "`";
    bool good = true;

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        /*
        DPCT1093:42: The "dev_idx" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev_idx);
        dpct::get_current_device().queues_wait_and_throw();
        CudaCheckError();
        dpct::get_in_order_queue()
            .memcpy(&o_t[dev_idx * output_size], output[dev_idx].raw_ptr,
                    output_size * sizeof(T))
            .wait();
        CudaCheckError();

        for(int idx = 0; idx < output_size; idx++) {
            int unit_idx = dev_idx * output_size + idx;
            if constexpr (std::is_same_v<T, kittens::bf16>) {
                o[unit_idx] =
                    sycl::ext::intel::math::bfloat162float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = sycl::ext::intel::math::bfloat162float(
                    sycl::ext::intel::math::float2bfloat16(
                        o_ref[dev_idx][idx]));
            }
            else if constexpr (std::is_same_v<T, kittens::half>) {
                o[unit_idx] = sycl::ext::intel::math::half2float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = sycl::ext::intel::math::half2float(
                    sycl::ext::intel::math::float2half_rn(o_ref[dev_idx][idx]));
            }
            else if constexpr(std::is_same_v<T, float>) {
                o[unit_idx] = o_t[unit_idx];
            }
            #ifdef KITTENS_HOPPER
            else if constexpr(std::is_same_v<T, kittens::fp8e4m3>) {
                o[unit_idx] = float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = float(__nv_fp8_e4m3(o_ref[dev_idx][idx]));
            }
            else if constexpr(std::is_same_v<T, kittens::fp8e5m2>) {
                o[unit_idx] = float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = float(__nv_fp8_e5m2(o_ref[dev_idx][idx]));
            }
            #endif
            else {
                assert(false && "Unsupported data type");
            }
        }

        // check
        for(int i = 0; i < output_size; i++) {
            if(!test_values_match(o_ref[dev_idx][i], o[dev_idx * output_size + i], eps)) {
                good = false;
                break;
            }
        }

        // Even if we failed, continue so we can print all the results in the output file
    }

    if(good) std::cout << " -- PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;

    if(should_write_outputs && !good) {
        std::ofstream reffile("outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("outputs/"+test_name+"_out.txt");
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            outfile << "Device " << dev_idx << ":\n\n";
            reffile << "Device " << dev_idx << ":\n\n";
            for(int i = 0; i < output_size; i++) {
                reffile << o_ref[dev_idx][i] << ' ';
                outfile << o[dev_idx * output_size + i] << ' ';
                if(i%cols == cols-1) {
                    reffile << '\n';
                    outfile << '\n';
                }
            }
            reffile << "\n\n\nINPUTS:\n\n";
            for(int i = 0; i < input_size; i++) {
                reffile << i_ref[dev_idx][i] << ' ';
                if(i%cols == cols-1) {
                    reffile << '\n';
                }
            }
            outfile << "\n\n\n\n";
            reffile << "\n\n\n\n";
        }
        reffile.close();
        outfile.close();
    }

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        /*
        DPCT1093:43: The "dev_idx" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev_idx);
        kittens::pglCudaFree(dev_idx, input[dev_idx].raw_ptr, input_size * sizeof(T));
        kittens::pglCudaFree(dev_idx, output[dev_idx].raw_ptr, output_size * sizeof(T));
    }
    // delete[] o_t, o;
    delete[] o_t;
    delete[] o;
    CudaCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}

#endif
