/**
 * @file
 * @brief Host-side discovery and ring planning for grouped device topologies.
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace kittens::collective {

/** Runtime topology discovered from physical group and local-rank keys. */
struct discovered_topology {
    std::vector<int> group_of;
    std::vector<int> rank_in_group;
    std::vector<std::vector<int>> members;

    int participants() const {
        return static_cast<int>(group_of.size());
    }

    int groups() const {
        return static_cast<int>(members.size());
    }

    int group(int participant_index) const {
        return group_of.at(participant_index);
    }

    int rank(int participant_index) const {
        return rank_in_group.at(participant_index);
    }

    int group_size(int group_index) const {
        return static_cast<int>(members.at(group_index).size());
    }

    int participant(int group_index, int rank_index) const {
        return members.at(group_index).at(rank_index);
    }
};

/**
 * Discover groups and assign stable local ranks by sorting each group's
 * participants by its physical local-rank key.
 */
template <typename GroupKey, typename LocalRankKey>
discovered_topology discover_topology(const std::vector<GroupKey> &group_keys,
                                      const std::vector<LocalRankKey> &local_rank_keys) {
    if (group_keys.size() != local_rank_keys.size()) {
        throw std::invalid_argument("group and local-rank key counts differ");
    }

    discovered_topology topology;
    topology.group_of.resize(group_keys.size(), -1);
    topology.rank_in_group.resize(group_keys.size(), -1);
    std::vector<GroupKey> unique_groups;

    for (int participant = 0; participant < static_cast<int>(group_keys.size()); ++participant) {
        auto found = std::find(unique_groups.begin(), unique_groups.end(), group_keys[participant]);
        int group_index = static_cast<int>(found - unique_groups.begin());
        if (found == unique_groups.end()) {
            unique_groups.push_back(group_keys[participant]);
            topology.members.emplace_back();
            group_index = static_cast<int>(unique_groups.size()) - 1;
        }
        topology.group_of[participant] = group_index;
        topology.members[group_index].push_back(participant);
    }

    for (auto &group_members : topology.members) {
        std::stable_sort(group_members.begin(), group_members.end(),
            [&](int left, int right) {
                return local_rank_keys[left] < local_rank_keys[right];
            });
        for (int rank = 0; rank < static_cast<int>(group_members.size()); ++rank) {
            if (rank > 0 &&
                local_rank_keys[group_members[rank]] == local_rank_keys[group_members[rank - 1]]) {
                throw std::invalid_argument("duplicate local-rank key within a group");
            }
            topology.rank_in_group[group_members[rank]] = rank;
        }
    }
    return topology;
}

/** Discover PVC groups and tile ranks from Intel device UUID strings. */
inline discovered_topology discover_pvc_topology(const std::vector<std::string> &uuids) {
    std::vector<std::string> group_keys;
    std::vector<std::string> local_rank_keys;
    group_keys.reserve(uuids.size());
    local_rank_keys.reserve(uuids.size());
    for (const auto &uuid : uuids) {
        if (uuid.size() < 2) {
            throw std::invalid_argument("PVC UUID is missing its tile suffix");
        }
        group_keys.push_back(uuid.substr(0, uuid.size() - 2));
        local_rank_keys.push_back(uuid.substr(uuid.size() - 2));
    }
    return discover_topology(group_keys, local_rank_keys);
}

/** Disjoint rings formed by taking one stable local rank from every group. */
struct lane_ring_plan {
    std::vector<std::vector<int>> lanes;
    std::vector<int> lane_of;
    std::vector<int> rank_in_lane;
    std::vector<int> next_participant;
    std::vector<int> previous_participant;

    int lane_count() const {
        return static_cast<int>(lanes.size());
    }

    int ring_size() const {
        return lanes.empty() ? 0 : static_cast<int>(lanes.front().size());
    }

    int participant(int lane_index, int rank_index) const {
        return lanes.at(lane_index).at(rank_index);
    }

    int lane(int participant_index) const {
        return lane_of.at(participant_index);
    }

    int rank(int participant_index) const {
        return rank_in_lane.at(participant_index);
    }

    int next(int participant_index) const {
        return next_participant.at(participant_index);
    }

    int previous(int participant_index) const {
        return previous_participant.at(participant_index);
    }
};

/**
 * Build one ring per local rank. Every group must contain the same number of
 * participants, so each ring visits exactly one participant from every group.
 */
inline lane_ring_plan plan_lane_rings(const discovered_topology &topology) {
    lane_ring_plan plan;
    const int participants = topology.participants();
    plan.lane_of.resize(participants, -1);
    plan.rank_in_lane.resize(participants, -1);
    plan.next_participant.resize(participants, -1);
    plan.previous_participant.resize(participants, -1);
    if (topology.groups() == 0) return plan;

    const int lane_count = topology.group_size(0);
    if (lane_count == 0) {
        throw std::invalid_argument("topology contains an empty group");
    }
    for (int group = 1; group < topology.groups(); ++group) {
        if (topology.group_size(group) != lane_count) {
            throw std::invalid_argument("lane rings require uniform group sizes");
        }
    }

    plan.lanes.resize(lane_count);
    for (int lane = 0; lane < lane_count; ++lane) {
        auto &ring = plan.lanes[lane];
        ring.reserve(topology.groups());
        for (int group = 0; group < topology.groups(); ++group) {
            ring.push_back(topology.participant(group, lane));
        }
        for (int rank = 0; rank < static_cast<int>(ring.size()); ++rank) {
            const int participant = ring[rank];
            plan.lane_of[participant] = lane;
            plan.rank_in_lane[participant] = rank;
            plan.next_participant[participant] = ring[(rank + 1) % ring.size()];
            plan.previous_participant[participant] =
                ring[(rank + static_cast<int>(ring.size()) - 1) % ring.size()];
        }
    }
    return plan;
}

} // namespace kittens::collective