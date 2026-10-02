#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_MEMORY_VEC

#include "testing_commons.dp.hpp"

#include "pgl_to_register.dp.hpp"
#include "pgl_to_shared.dp.hpp"
#include "global_to_register.dp.hpp"
#include "global_to_shared.dp.hpp"
#include "shared_to_register.dp.hpp"

namespace group {
namespace memory {
namespace vec {

void tests(test_data &results);

}
}
}

#endif