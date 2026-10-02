/**
 * @file
 * @brief Stubs for tensor types (SYCL port).
 *
 * Tensor types (tt, tensor_allocator) are specific to NVIDIA Blackwell's
 * dedicated tensor memory. Intel Xe GPUs use different memory hierarchies.
 *
 * TODO: Map to Intel Xe Matrix Extensions (XMX) or joint_matrix when applicable.
 */

#pragma once

#include <sycl/sycl.hpp>
#include "common/util.dp.hpp"

namespace kittens {

// Tensor memory is a Blackwell-specific feature.
// On Intel, shared local memory (SLM) or register files serve similar roles.
// This stub allows code to compile but operations are no-ops.

namespace ducks {
namespace tt {
struct identifier {};
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
} // namespace tt

namespace tensor_allocator {
struct identifier {};
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
} // namespace tensor_allocator
} // namespace ducks

// Stub tensor_allocator - Blackwell tensor memory not available on Intel
template<int _nblocks_per_sm, int _ncta, bool _managed = true>
struct tensor_allocator {
    using identifier = ducks::tensor_allocator::identifier;
    static constexpr int nblocks_per_sm = _nblocks_per_sm;
    static constexpr int ncta = _ncta;
    static constexpr bool managed = _managed;
    uint32_t addr;
};

} // namespace kittens
