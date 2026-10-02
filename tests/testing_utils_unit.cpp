#include "testing_commons/testing_commons.dp.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_WARP
#ifndef TEST_INTENSITY
#define TEST_INTENSITY 1
#endif
#define main test_runner_main
#include "unit_tests.dp.cpp"
#undef main
#undef TEST_WARP

enum class runner_mode { passing, failing, standard_exception, unknown_exception };
runner_mode active_runner_mode = runner_mode::passing;
int should_write_outputs = 0;

void warp::tests(test_data &results) {
    if (active_runner_mode == runner_mode::standard_exception) {
        throw std::runtime_error("expected test-runner regression exception");
    }
    if (active_runner_mode == runner_mode::unknown_exception) throw 7;
    results.push_back({"injected result", active_runner_mode == runner_mode::failing
                                           ? test_result::FAILED : test_result::PASSED});
}

bool runner_exit_contract() {
    char program_name[] = "test_runner";
    char *arguments[] = {program_name, nullptr};
    for (const auto mode : {runner_mode::passing, runner_mode::failing,
                            runner_mode::standard_exception, runner_mode::unknown_exception}) {
        active_runner_mode = mode;
        const int status = test_runner_main(1, arguments);
        if ((status == 0) != (mode == runner_mode::passing)) return false;
    }
    return true;
}

#if defined(TEST_UNSUPPORTED_ROW_LOAD)
using unsupported_tile = kittens::rt_bf<32, 64, kittens::ducks::rt_layout::row,
                                      kittens::ducks::rt_shape::rt_32x64>;
#elif defined(TEST_UNSUPPORTED_COLUMN_LOAD)
using unsupported_tile = kittens::rt_bf<8, 32, kittens::ducks::rt_layout::col,
                                      kittens::ducks::rt_shape::rt_8x32>;
#elif defined(TEST_UNSUPPORTED_TRANSPOSE_LOAD)
using unsupported_tile = kittens::rt_bf<32, 16, kittens::ducks::rt_layout::row,
                                      kittens::ducks::rt_shape::rt_32x16>;
#endif

#if defined(TEST_UNSUPPORTED_ROW_LOAD) || defined(TEST_UNSUPPORTED_COLUMN_LOAD) || \
    defined(TEST_UNSUPPORTED_TRANSPOSE_LOAD)
void unsupported_load_contract(unsupported_tile &tile,
                               const kittens::gl<kittens::bf16, 1, 1, 64, 64> &source) {
#ifdef TEST_UNSUPPORTED_TRANSPOSE_LOAD
    kittens::load_transpose_part<2>(tile, source, kittens::coord<unsupported_tile>{}, {0, 0});
#else
    kittens::load_part<2>(tile, source, kittens::coord<unsupported_tile>{}, {0, 0});
#endif
}
#endif

#if defined(TEST_INITIALIZE_TYPE) || defined(TEST_VALIDATE_TYPE)
template<typename Element>
void testing_type_contract() {
    Element *device_input = nullptr;
    Element *device_output = nullptr;
    std::vector<float> reference_input(1, 1.0f);
    std::vector<float> reference_output(1, 1.0f);
#ifdef TEST_INITIALIZE_TYPE
    initialize<Element>(&device_input, &device_output, reference_input, reference_output);
#else
    validate<Element>(device_input, device_output, reference_input, reference_output,
                      "type_contract", 1);
#endif
}

void instantiate_testing_type_contracts() {
#ifdef TEST_UNSUPPORTED_TEST_TYPE
    testing_type_contract<int>();
#else
    testing_type_contract<float>();
    testing_type_contract<kittens::bf16>();
    testing_type_contract<sycl::half>();
#endif
}
#endif

struct skipped_test {
    template<int... Dimensions> using valid = std::false_type;
    static inline const std::string test_identifier = "skipped";
};

bool test_name_contract() {
    using row_layout = kittens::ducks::rt_layout::row;
    using col_layout = kittens::ducks::rt_layout::col;
    using extent_two = std::integral_constant<int, 2>;
    using extent_four = std::integral_constant<int, 4>;
    const std::string identifier(96, 'x');
    return generate_test_name<1,1>(identifier) == identifier + "_[1]" &&
           generate_test_name<2,4>(identifier) == identifier + "_[2]_[4warps]" &&
           generate_test_name<2,1,row_layout>(identifier) == identifier + "_[2]_[rt_row_layout]" &&
           generate_test_name<2,1,col_layout>(identifier) == identifier + "_[2]_[rt_col_layout]" &&
           generate_test_name<2,1,row_layout,col_layout>(identifier) == identifier + "_[2]_[rt_row_layout]_[rt_col_layout]" &&
           generate_test_name<2,1,kittens::naive_l>(identifier) == identifier + "_[2]_[rv_naive_layout]" &&
           generate_test_name<2,1,kittens::ortho_l>(identifier) == identifier + "_[2]_[rv_ortho_layout]" &&
           generate_test_name<2,1,kittens::align_l,kittens::naive_l>(identifier) == identifier + "_[2]_[rv_align_layout]_[rv_naive_layout]" &&
           generate_test_name<2,3,1>(identifier) == identifier + "_[2x3]" &&
           generate_test_name<2,3,4,extent_four>(identifier) == identifier + "_[2x3]x4_[4warps]" &&
           generate_test_name<2,3,1,row_layout>(identifier) == identifier + "_[2x3]_[rt_row_layout]" &&
           generate_test_name<2,3,1,extent_two,extent_four>(identifier) == identifier + "_[2x3_2x4]" &&
           generate_test_name<2,3,1,float,kittens::bf16>(identifier) == identifier + "_[2x3]_[bf16->float]" &&
           generate_test_name<2,3,1,kittens::bf16,kittens::half>(identifier) == identifier + "_[2x3]_[half->bf16]" &&
           generate_test_name<2,3,1,kittens::half,float>(identifier) == identifier + "_[2x3]_[float->half]" &&
           identifier == std::string(96, 'x');
}

bool column_reduction_contract() {
    using tile_type = kittens::rt_fl<32, 32>;
    tile_type source;
    float expected_sum[tile_type::width]{};
    for (int column = 0; column < tile_type::width; ++column) {
        for (int row = 0; row < tile_type::height; ++row) {
            for (int element = 0; element < tile_type::packed_per_base_tile; ++element) {
                const float value = float(row * 16 + element * 2 + column - 20);
                source.tiles[row][column].data[element] = sycl::float2{value, value + 1};
                expected_sum[column] += value + value + 1;
            }
        }
    }
    tile_type::row_vec result;
    kittens::col_sum(result, source);
    for (int column = 0; column < tile_type::width; ++column) {
        if (result[column][0].x() != expected_sum[column] ||
            result[column][0].y() != expected_sum[column]) return false;
        result[column][0] = sycl::float2{5, 7};
    }
    kittens::col_sum(result, source, result);
    for (int column = 0; column < tile_type::width; ++column) {
        if (result[column][0].x() != expected_sum[column] + 5 ||
            result[column][0].y() != expected_sum[column] + 7) return false;
    }
    return true;
}

template<typename Operation>
bool aborts_host_intrinsic(Operation operation) {
    const pid_t child = fork();
    if (child < 0) return false;
    if (child == 0) {
        operation();
        std::_Exit(EXIT_SUCCESS);
    }
    int status = 0;
    return waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
           WTERMSIG(status) == SIGABRT;
}

bool host_intrinsic_contract() {
    float source = 1.0f;
    float destination = 73.0f;
    return aborts_host_intrinsic([&] {
               __spirv_Subgroup2DBlockLoadINTEL(2, 16, 16, 1, &source, 32, 16, 32, {}, &destination);
           }) &&
           aborts_host_intrinsic([&] {
               __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 16, 1, &source, 32, 16, 32, {}, &destination);
           }) &&
           aborts_host_intrinsic([&] {
               __spirv_Subgroup2DBlockLoadTransposeINTEL(4, 8, 16, 1, &source, 32, 16, 32, {}, &destination);
           }) &&
           aborts_host_intrinsic([&] {
               __spirv_Subgroup2DBlockStoreINTEL(2, 16, 16, 1, &source, &destination, 32, 16, 32, {});
           }) &&
           aborts_host_intrinsic([&] {
               __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 1, &source, 32, 16, 32, {});
           }) &&
           aborts_host_intrinsic([] {
               (void)__spirv_SubgroupMatrixMultiplyAccumulateINTEL(8, {}, {}, {}, 0);
           }) &&
           aborts_host_intrinsic([] {
               __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(0, 0, 0, 0, {}, {});
           }) && destination == 73.0f;
}

int main() try {
    test_info default_record;
    if (default_record.result != test_result::INVALID) return 1;

    test_info passed_record{"passed", test_result::PASSED};
    test_info failed_record{"failed", test_result::FAILED};
    if (passed_record.result != test_result::PASSED) return 2;
    if (failed_record.result != test_result::FAILED) return 3;
    if (!column_reduction_contract()) return 4;

    test_data results;
    wrapper_1d<skipped_test, 1, 1>::run(results);
    wrapper_2d<skipped_test, kittens::ducks::rt_shape::rt_16x16, 1, 1, 1>::run(results);
    if (results.size() != 2) return 5;
    if (results[0].label != "skipped_[1]" || results[0].result != test_result::INVALID) return 6;
    if (results[1].label != "skipped_[1x1]" || results[1].result != test_result::INVALID) return 7;
    if (!runner_exit_contract()) return 8;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    if (!test_values_match(1.0f, 1.125f, 0.125f)) return 9;
    if (test_values_match(1.0f, 1.25f, 0.125f)) return 10;
    if (test_values_match(1.0f, nan, 0.125f) || test_values_match(nan, 1.0f, 0.125f)) return 11;
    if (test_values_match(infinity, infinity, 0.125f) ||
        test_values_match(1.0f, infinity, 0.125f)) return 12;
    if (!test_name_contract()) return 13;
    if (!host_intrinsic_contract()) return 14;
} catch (const std::exception &error) {
    std::fprintf(stderr, "Testing utilities failed: %s\n", error.what());
    return 1;
} catch (...) {
    std::fprintf(stderr, "Testing utilities failed: unknown exception\n");
    return 1;
}