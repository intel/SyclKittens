#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "group.dp.hpp"

#ifdef TEST_GROUP

void group::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/group tests!     ------------------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY
    group::memory::tests(results);
#endif
#ifdef TEST_GROUP_SHARED
    group::shared::tests(results);
#endif
#ifdef TEST_GROUP_WGMMA
    group::wgmma::tests(results);
#endif
}

#endif