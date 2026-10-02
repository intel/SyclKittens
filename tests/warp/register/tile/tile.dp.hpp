#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_REGISTER_TILE

#include "testing_commons.dp.hpp"

#include "maps.dp.hpp"
#include "reductions.dp.hpp"
#include "mma.dp.hpp"
#include "conversions.dp.hpp"

#ifdef TEST_WARP_REGISTER_TILE_COMPLEX

#include "complex/complex_maps.cuh"
#include "complex/complex_mul.cuh"
#include "complex/complex_mma.cuh"
#include "complex/complex_conversions.cuh"
#endif


namespace warp {
namespace reg {
namespace tile {

void tests(test_data &results);

}
}
}

#endif