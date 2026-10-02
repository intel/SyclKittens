#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS

#include "testing_commons.dp.hpp"

namespace group {
namespace shared {
namespace tile {
namespace reductions {

void tests(test_data &results);

}
}
}
}

#endif