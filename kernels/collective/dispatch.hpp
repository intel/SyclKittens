#pragma once

#include <cstddef>
#include <limits>

namespace kittens::collective {

enum class Kind { AllGather, ReduceScatter, AllReduce };

struct DispatchThreshold {
    Kind kind;
    int tiles;
    size_t latency_max_bytes;
    size_t large_min_bytes;
};

inline DispatchThreshold thresholds(Kind kind, int tiles) {
    static constexpr DispatchThreshold table[] = {
        {Kind::AllGather, 12, 4ull * 1024 * 1024, 12ull * 1024 * 1024},
        {Kind::ReduceScatter, 12, 2ull * 1024 * 1024, 12ull * 1024 * 1024},
        {Kind::AllReduce, 12, 512ull * 1024, 512ull * 1024},
    };
    for (const auto &entry : table)
        if (entry.kind == kind && entry.tiles == tiles) return entry;
    return {kind, tiles, 0, std::numeric_limits<size_t>::max()};
}

} // namespace kittens::collective