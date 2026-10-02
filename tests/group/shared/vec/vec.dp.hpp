#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_SHARED_VEC

#include "testing_commons.dp.hpp"

#include "maps.dp.hpp"
#include "conversions.dp.hpp"

namespace group {
namespace shared {
namespace vec {

void tests(test_data &results);

}
}
}

#endif