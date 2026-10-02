#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_SHARED_TILE_MAPS

#include "testing_commons.dp.hpp"

namespace warp {
namespace shared {
namespace tile {
namespace maps {

void tests(test_data &results);

}
}
}
}

#endif