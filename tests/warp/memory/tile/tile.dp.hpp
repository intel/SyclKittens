#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE

// Skip the ATTENTION variant on this toolchain to avoid missing headers.
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER_ATTN
#undef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER_ATTN
#endif

#include "testing_commons.dp.hpp"

#include "global_to_register.dp.hpp"
#include "global_to_shared.dp.hpp"
#include "pgl_to_register.dp.hpp"
#include "pgl_to_shared.dp.hpp"
#include "shared_to_register.dp.hpp"
#include "tma.dp.hpp"
#include "tma_multicast.dp.hpp"
#include "tma_pgl.dp.hpp"
#include "dsmem.dp.hpp"

namespace warp {
namespace memory {
namespace tile {

void tests(test_data &results);

}
}
}

#endif