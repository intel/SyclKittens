#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_REGISTER_VEC

#include "testing_commons.dp.hpp"

#include "maps.dp.hpp"
#include "reductions.dp.hpp"
#include "conversions.dp.hpp"

namespace warp {
namespace reg {
namespace vec {

void tests(test_data &results);

}
}
}

#endif