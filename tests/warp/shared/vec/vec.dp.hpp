#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_SHARED_VEC

#include "testing_commons.dp.hpp"

#include "maps.dp.hpp"
#include "reductions.dp.hpp"
#include "conversions.dp.hpp"

namespace warp {
namespace shared {
namespace vec {

void tests(test_data &results);

}
}
}

#endif