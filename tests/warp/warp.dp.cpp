#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <exception>
#include "warp.dp.hpp"

#ifdef TEST_WARP

using namespace warp;

void warp::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
    try {
#ifdef TEST_WARP_MEMORY
        memory::tests(results);
#endif
#ifdef TEST_WARP_REGISTER
        reg::tests(results); // register is a reserved word, hence reg
#endif
#ifdef TEST_WARP_SHARED
        shared::tests(results);
#endif
    } catch (const std::exception &error) {
        results.push_back({"warp test suite exception: " + std::string(error.what()),
                           test_result::FAILED});
    } catch (...) {
        results.push_back({"warp test suite exception: unknown exception",
                           test_result::FAILED});
    }
}

#endif