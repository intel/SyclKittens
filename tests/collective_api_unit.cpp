#include "../include/ops/collective/collective.dp.hpp"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <type_traits>

using namespace kittens::collective;

constexpr bool endpoint_table_contract() {
    fixed_endpoints<int, 4> endpoints;
    if (endpoints.size() != 0) return false;
    for (int index = 0; index < 4; ++index) {
        if (endpoints[index] != 0) return false;
    }
    endpoints.push(3);
    endpoints.push(7);
    const auto copied = endpoints;
    return copied.size() == 2 && copied[0] == 3 && copied[1] == 7 &&
           copied[2] == 0 && copied[3] == 0;
}

static_assert(endpoint_table_contract());
static_assert(std::is_trivially_copyable_v<endpoint_array<int, 4>>);
static_assert(std::is_trivially_copyable_v<fixed_endpoints<int, 4>>);
static_assert(std::is_trivially_copyable_v<push_pull_endpoints<float, 4>>);
static_assert(std::is_trivially_copyable_v<pull_replicate_endpoint<float>>);
static_assert(split_workgroup_count(5, 3) == 15);
static_assert(split_workgroup_id(11, 3).data_group == 3);
static_assert(split_workgroup_id(11, 3).endpoint == 2);
static_assert(partitioned_workgroup_count(5, 3) == 15);
static_assert(partition_workgroup(11, 3).data_group == 3);
static_assert(partition_workgroup(11, 3).begin == 2);
static_assert(partition_workgroup(11, 3).end == 3);
static_assert(partitioned_workgroup_count(5, 3, false) == 5);
static_assert(partition_workgroup(4, 3, false).data_group == 4);
static_assert(partition_workgroup(4, 3, false).begin == 0);
static_assert(partition_workgroup(4, 3, false).end == 3);

constexpr paired_topology topology{12};
static_assert(topology.groups() == 6);
static_assert(topology.group(7) == 3);
static_assert(topology.rank(7) == 1);
static_assert(topology.tile(3, 1) == 7);
static_assert(topology.partner(7) == 6);
static_assert(topology.exchange_owner(0, 1) == 1);
static_assert(topology.exchange_owner(1, 2) == 2);
static_assert(topology.exchange_owner(2, 5) == 5);
static_assert(topology.destination(1, 4) == 9);

int main() try {
    const std::vector<std::string> crossed_uuids = {
        "pvc0-01", "pvc1-02", "pvc2-01", "pvc3-02", "pvc4-01", "pvc5-02",
        "pvc0-02", "pvc1-01", "pvc2-02", "pvc3-01", "pvc4-02", "pvc5-01",
    };
    const auto discovered = discover_pvc_topology(crossed_uuids);
    const auto rings = plan_lane_rings(discovered);
    const int ring_size = rings.ring_size();

    if (discovered.groups() != 6 || discovered.participants() != 12 ||
        rings.lane_count() != 2 || ring_size != 6) return 1;

    const std::vector<int> expected_lane0 = {0, 7, 2, 9, 4, 11};
    const std::vector<int> expected_lane1 = {6, 1, 8, 3, 10, 5};
    if (rings.lanes[0] != expected_lane0 || rings.lanes[1] != expected_lane1) return 2;

    for (int lane = 0; lane < rings.lane_count(); ++lane) {
        for (int rank = 0; rank < ring_size; ++rank) {
            const int participant = rings.participant(lane, rank);
            if (rings.lane(participant) != lane || rings.rank(participant) != rank) return 3;
            if (rings.next(participant) !=
                rings.participant(lane, (rank + 1) % ring_size)) return 4;
            if (rings.previous(participant) !=
                rings.participant(lane, (rank + ring_size - 1) % ring_size)) return 5;
        }
    }

    bool rejected_uneven = false;
    try {
        const auto uneven = discover_pvc_topology(
            std::vector<std::string>{"pvc0-01", "pvc0-02", "pvc1-01"});
        (void)plan_lane_rings(uneven);
    } catch (const std::invalid_argument &) {
        rejected_uneven = true;
    }
    return rejected_uneven ? 0 : 6;
} catch (const std::exception &error) {
    std::fprintf(stderr, "Collective API tests failed: %s\n", error.what());
    return 1;
} catch (...) {
    std::fprintf(stderr, "Collective API tests failed: unknown exception\n");
    return 1;
}