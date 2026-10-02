#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_VEC_TMA_MULTICAST

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace vec {
namespace tma_multicast {

void tests(test_data &results);

}
}
}
}

#endif