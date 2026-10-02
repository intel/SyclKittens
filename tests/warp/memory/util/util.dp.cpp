#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "util.dp.hpp"

#ifdef TEST_WARP_MEMORY_UTIL

void warp::memory::util::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/util tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_UTIL_REDUCE
    warp::memory::util::reduce::tests(results);
#endif
}

#endif
