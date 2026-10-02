#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_WGMMA

#include "testing_commons.cuh"

#include "complex_mma_fp32_bf16.cuh"
#include "complex_mma_fp32_fp16.cuh"
#include "complex_mma_fp16_fp16.cuh"

namespace group {
namespace wgmma {
namespace complex {
void tests(test_data &results);
}
}
}

#endif