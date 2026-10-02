#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_SHARED_TILE

#include "testing_commons.dp.hpp"

#include "conversions.dp.hpp"
#include "maps.dp.hpp"
#include "reductions.dp.hpp"

namespace warp {
namespace shared {
namespace tile {

void tests(test_data &results);

}
}
}

#endif