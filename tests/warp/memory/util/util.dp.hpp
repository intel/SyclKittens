#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_UTIL

#include "testing_commons.cuh"

#include "reduce.cuh"

namespace warp {
namespace memory {
namespace util {

void tests(test_data &results);

}
}
}

#endif
