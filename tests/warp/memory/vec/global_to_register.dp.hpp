#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER

#include "testing_commons.dp.hpp"

namespace warp {
namespace memory {
namespace vec {
namespace global_to_register {

void tests(test_data &results);

}
}
}
}

#endif