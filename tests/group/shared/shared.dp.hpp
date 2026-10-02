#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_SHARED

#include "testing_commons.dp.hpp"

#include "tile/tile.dp.hpp"
#include "vec/vec.dp.hpp"

namespace group {
namespace shared {

void tests(test_data &results);

}
}

#endif