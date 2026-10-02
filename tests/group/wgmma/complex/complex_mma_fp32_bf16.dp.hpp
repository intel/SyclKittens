#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_COMPLEX_WGMMA_MMA_FP32_BF16

#include "testing_commons.cuh"

namespace group {
namespace wgmma {
namespace complex {
namespace complex_mma_fp32_bf16 {
void tests(test_data &results);
}
}
}
}

#endif