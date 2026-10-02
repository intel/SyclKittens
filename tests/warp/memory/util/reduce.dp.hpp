#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_UTIL_REDUCE

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace util {
namespace reduce {

void tests(test_data &results);

}
}
}
}

#endif
