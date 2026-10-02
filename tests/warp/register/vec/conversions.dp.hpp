#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS

#include "testing_commons.dp.hpp"

namespace warp {
namespace reg {
namespace vec {
namespace conversions {

void tests(test_data &results);

}
}
}
}

#endif