#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY

#include "testing_commons.dp.hpp"

#include "tile/tile.dp.hpp"
#include "vec/vec.dp.hpp"
#include "util/util.dp.hpp"

namespace warp {
namespace memory {

void tests(test_data &results);

}
}

#endif
