#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP

#include "testing_commons.dp.hpp"

#include "memory/memory.dp.hpp"
#include "register/register.dp.hpp"
#include "shared/shared.dp.hpp"

namespace warp {

void tests(test_data &results);

}

#endif