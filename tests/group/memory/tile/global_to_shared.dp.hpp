#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED

#include "testing_commons.dp.hpp"

namespace group {
namespace memory {
namespace tile {
namespace global_to_shared {

void tests(test_data &results);

}
}
}
}

#endif