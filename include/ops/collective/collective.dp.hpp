/**
 * @file
 * @brief Zero-cost building blocks for topology-aware multi-device collectives.
 */

#pragma once

#include "topology.dp.hpp"

namespace kittens::collective {

/** Fixed-capacity endpoint storage for callers that track cardinality externally. */
template <typename Endpoint, int Capacity>
struct endpoint_array {
    Endpoint endpoints[Capacity];

    constexpr void set(int index, Endpoint endpoint) {
        endpoints[index] = endpoint;
    }

    constexpr Endpoint operator[](int index) const {
        return endpoints[index];
    }
};

/** Fixed-capacity endpoint table captured by value in a device kernel. */
template <typename Endpoint, int Capacity>
struct fixed_endpoints {
    Endpoint endpoints[Capacity]{};
    int count = 0;

    constexpr void push(Endpoint endpoint) {
        endpoints[count++] = endpoint;
    }

    constexpr int size() const {
        return count;
    }

    constexpr Endpoint operator[](int index) const {
        return endpoints[index];
    }
};

/** Endpoints for one fused push/pull exchange captured by value. */
template <typename T, int Capacity>
struct push_pull_endpoints {
    const T *pull_sources[Capacity];
    T *pull_self[Capacity];
    T *pull_partner[Capacity];
    T *push_destinations[Capacity];
    int count = 0;

    constexpr void push(const T *pull_source, T *self_destination,
                        T *partner_destination, T *push_destination) {
        pull_sources[count] = pull_source;
        pull_self[count] = self_destination;
        pull_partner[count] = partner_destination;
        push_destinations[count] = push_destination;
        count++;
    }

    constexpr int size() const {
        return count;
    }
};

/** One source routed to a local output and an optional partner output. */
template <typename T>
struct pull_replicate_endpoint {
    const T *source;
    T *self_destination;
    T *partner_destination;
};

/** Logical coordinates of a workgroup in a data-group by endpoint launch. */
struct split_workgroup {
    int data_group;
    int endpoint;
};

constexpr int split_workgroup_count(int data_groups, int endpoints) {
    return data_groups * endpoints;
}

constexpr split_workgroup split_workgroup_id(int group, int endpoints) {
    return {group / endpoints, group % endpoints};
}

/** Endpoint interval assigned to one workgroup, with optional splitting. */
struct endpoint_partition {
    int data_group;
    int begin;
    int end;
};

constexpr int partitioned_workgroup_count(int data_groups, int endpoints,
                                          bool split = true) {
    return split ? split_workgroup_count(data_groups, endpoints) : data_groups;
}

constexpr endpoint_partition partition_workgroup(int group, int endpoints,
                                                  bool split = true) {
    if (!split) return {group, 0, endpoints};
    const auto split_group = split_workgroup_id(group, endpoints);
    return {split_group.data_group, split_group.endpoint, split_group.endpoint + 1};
}

/** Helpers for fabrics whose fast local group contains a fixed number of tiles. */
template <int TilesPerGroup = 2>
struct grouped_topology {
    static_assert(TilesPerGroup > 0);

    int tiles;

    constexpr int groups() const {
        return (tiles + TilesPerGroup - 1) / TilesPerGroup;
    }

    static constexpr int group(int tile) {
        return tile / TilesPerGroup;
    }

    static constexpr int rank(int tile) {
        return tile % TilesPerGroup;
    }

    static constexpr int tile(int group_index, int rank_index) {
        return group_index * TilesPerGroup + rank_index;
    }

    static constexpr int partner(int tile_index) {
        static_assert(TilesPerGroup == 2,
                      "partner() is defined only for paired topologies");
        return tile_index ^ 1;
    }

    static constexpr int exchange_owner(int first_group, int second_group) {
        const int left = first_group < second_group ? first_group : second_group;
        const int right = first_group < second_group ? second_group : first_group;
        return ((left + right) & 1) ? right : left;
    }

    static constexpr int destination(int rank_index, int destination_index) {
        return rank_index + TilesPerGroup * destination_index;
    }
};

using paired_topology = grouped_topology<2>;

} // namespace kittens::collective