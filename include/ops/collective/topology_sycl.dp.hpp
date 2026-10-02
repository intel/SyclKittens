/**
 * @file
 * @brief SYCL device adapters for collective topology discovery.
 */

#pragma once

#include <sycl/sycl.hpp>

#include "topology.dp.hpp"

namespace kittens::collective {

inline std::string device_uuid_string(const sycl::device &device) {
    static constexpr char hex[] = "0123456789abcdef";
    const auto uuid = device.get_info<sycl::ext::intel::info::device::uuid>();
    std::string value;
    value.reserve(uuid.size() * 2);
    for (const auto byte : uuid) {
        value.push_back(hex[(static_cast<unsigned>(byte) >> 4) & 0xf]);
        value.push_back(hex[static_cast<unsigned>(byte) & 0xf]);
    }
    return value;
}

inline discovered_topology discover_pvc_topology(const std::vector<sycl::device> &devices) {
    std::vector<std::string> uuids;
    uuids.reserve(devices.size());
    for (const auto &device : devices) uuids.push_back(device_uuid_string(device));
    return discover_pvc_topology(uuids);
}

} // namespace kittens::collective