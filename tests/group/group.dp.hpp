#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP

#include "testing_commons.dp.hpp"

#include "memory/memory.dp.hpp"
#include "shared/shared.dp.hpp"
#include "wgmma/wgmma.dp.hpp"

namespace group {

void tests(test_data &results);

}

#endif