#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../include/ops/collective/collective.dp.hpp"
#include "dispatch.hpp"
#include "topology.hpp"

namespace kittens::collective::reduce_scatter {

using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int NUM_DEVICES = 12;
inline constexpr int WG_SIZE = 256;
inline constexpr int EPT = 8;
inline constexpr int RS_B_WG_SIZE = 256;
inline constexpr int RS_B_VPT = 1;
inline constexpr bool RS_B_SPLIT_DEST = true;
inline constexpr int RS_B_FINAL_WG_SIZE = 256;
inline constexpr int RS_B_FINAL_VPT = 1;

enum class Mode {
    Fused,
    Hierarchical,
    DestinationSplit,
};

inline Mode select_mode(int n_gpus, size_t total_bytes) {
    const auto dispatch = kittens::collective::thresholds(
        kittens::collective::Kind::ReduceScatter, n_gpus);
    if (total_bytes <= dispatch.latency_max_bytes)
        return Mode::Fused;
    if (n_gpus == 12 && total_bytes >= dispatch.large_min_bytes)
        return Mode::DestinationSplit;
    return Mode::Hierarchical;
}

inline void run_fused(std::vector<sycl::queue> &queues,
                      const std::vector<bf16 *> &d_in,
                      const std::vector<bf16 *> &d_out,
                      int n_gpus,
                      int chunk) {
    bf16 *in_ptr[NUM_DEVICES] = {nullptr};
    for (int d = 0; d < n_gpus; d++)
        in_ptr[d] = d_in[d];

    for (int d = 0; d < n_gpus; d++) {
        bf16 *out = d_out[d];
        int dev = d, cs = chunk, ng = n_gpus;
        bf16 *srcs[NUM_DEVICES];
        for (int s = 0; s < NUM_DEVICES; s++)
            srcs[s] = in_ptr[s];
        int nwg = (cs + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);
        queues[d].parallel_for(
            sycl::nd_range<1>(nwg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int gid = it.get_global_linear_id();
                int base = dev * cs;
#pragma unroll
                for (int e = 0; e < EPT; e++) {
                    int idx = gid * EPT + e;
                    if (idx >= cs)
                        break;
                    float acc = 0.0f;
#pragma unroll
                    for (int s = 0; s < NUM_DEVICES; s++)
                        if (s < ng)
                            acc += static_cast<float>(srcs[s][base + idx]);
                    out[idx] = static_cast<bf16>(acc);
                }
            });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_hierarchical(std::vector<sycl::queue> &queues,
                             const std::vector<bf16 *> &d_in,
                             const std::vector<bf16 *> &d_out,
                             const std::vector<bf16 *> &d_par,
                             int npvc,
                             const std::vector<std::vector<int>> &tiles,
                             int n_gpus,
                             int total,
                             int chunk) {
    bf16 *in_ptr[NUM_DEVICES] = {nullptr};
    bf16 *par_ptr[NUM_DEVICES] = {nullptr};
    for (int d = 0; d < n_gpus; d++) {
        in_ptr[d] = d_in[d];
        par_ptr[d] = d_par[d];
    }

    // Phase 0: intra-PVC pre-reduce on each leader over the fast link.
    for (int p = 0; p < npvc; p++) {
        int a = tiles[p][0];
        int b = (tiles[p].size() > 1) ? tiles[p][1] : tiles[p][0];
        bf16 *pa = d_par[a];
        bf16 *ia = in_ptr[a], *ib = in_ptr[b];
        int tot = total;
        int nwg = (tot + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);
        queues[a].parallel_for(
            sycl::nd_range<1>(nwg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int gid = it.get_global_linear_id();
#pragma unroll
                for (int e = 0; e < EPT; e++) {
                    int idx = gid * EPT + e;
                    if (idx >= tot)
                        break;
                    pa[idx] = static_cast<bf16>(
                        static_cast<float>(ia[idx]) + static_cast<float>(ib[idx]));
                }
            });
    }
    for (auto &q : queues)
        q.wait();

    // Phase 1: each tile reduces the npvc leader partials for its chunk.
    // Leader ptr per PVC captured by value.
    kittens::collective::endpoint_array<bf16 *, NUM_DEVICES> leadp;
    int leaders = npvc;
    for (int p = 0; p < npvc; p++)
        leadp.set(p, par_ptr[tiles[p][0]]);
    for (int d = 0; d < n_gpus; d++) {
        bf16 *out = d_out[d];
        int dev = d, cs = chunk, np = leaders;
        auto lp = leadp;
        int nwg = (cs + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);
        queues[d].parallel_for(
            sycl::nd_range<1>(nwg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int gid = it.get_global_linear_id();
                int base = dev * cs;
#pragma unroll
                for (int e = 0; e < EPT; e++) {
                    int idx = gid * EPT + e;
                    if (idx >= cs)
                        break;
                    float acc = 0.0f;
#pragma unroll
                    for (int p = 0; p < NUM_DEVICES; p++)
                        if (p < np)
                            acc += static_cast<float>(lp[p][base + idx]);
                    out[idx] = static_cast<bf16>(acc);
                }
            });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_destination_split(std::vector<sycl::queue> &queues,
                                  const std::vector<bf16 *> &d_in,
                                  const std::vector<bf16 *> &d_out,
                                  const std::vector<bf16 *> &d_par,
                                  int npvc,
                                  const std::vector<std::vector<int>> &tiles,
                                  int n_gpus,
                                  int chunk) {
    const kittens::collective::paired_topology topology{n_gpus};
    std::vector<sycl::event> producer_events(n_gpus);
    for (int p = 0; p < npvc; p++) {
        const int first = tiles[p][0];
        const int second = tiles[p].size() > 1 ? tiles[p][1] : first;
        for (int rank = 0; rank < (int)tiles[p].size(); rank++) {
            const int issuer = tiles[p][rank];
            const bf16 *source_first = d_in[first];
            const bf16 *source_second = d_in[second];
            kittens::collective::fixed_endpoints<bf16 *, NUM_DEVICES>
                destinations;
            for (int destination = rank; destination < n_gpus; destination += 2)
                destinations.push(d_par[destination] + (size_t)p * chunk);
            using v16 = sycl::vec<uint32_t, 4>;
            const int cs = chunk, vector_count = chunk / 8;
            const int nwg =
                (vector_count + RS_B_WG_SIZE * RS_B_VPT - 1) /
                (RS_B_WG_SIZE * RS_B_VPT);
            const int launch_groups =
                kittens::collective::partitioned_workgroup_count(
                    nwg, destinations.size(), RS_B_SPLIT_DEST);
            producer_events[issuer] = queues[issuer].parallel_for(
                sycl::nd_range<1>(launch_groups * RS_B_WG_SIZE, RS_B_WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const int lane = item.get_local_linear_id();
                    const int group = item.get_group_linear_id();
                    const auto partition =
                        kittens::collective::partition_workgroup(
                            group, destinations.size(), RS_B_SPLIT_DEST);
                    const int base = partition.data_group *
                        RS_B_WG_SIZE * RS_B_VPT;
#pragma unroll
                    for (int element = 0; element < RS_B_VPT; element++) {
                        const int vector_index = base + element * RS_B_WG_SIZE + lane;
                        if (vector_index >= vector_count)
                            continue;
                        for (int destination_index = partition.begin;
                             destination_index < partition.end;
                             destination_index++) {
                            const int destination =
                                topology.destination(rank, destination_index);
                            const size_t source_vector =
                                ((size_t)destination * cs) / 8 + vector_index;
                            const v16 first_vector =
                                reinterpret_cast<const v16 *>(source_first)[source_vector];
                            const v16 second_vector =
                                reinterpret_cast<const v16 *>(source_second)[source_vector];
                            v16 reduced;
                            bf16 *first_values = (bf16 *)&first_vector;
                            bf16 *second_values = (bf16 *)&second_vector;
                            bf16 *reduced_values = (bf16 *)&reduced;
#pragma unroll
                            for (int value = 0; value < 8; value++)
                                reduced_values[value] = static_cast<bf16>(
                                    static_cast<float>(first_values[value]) +
                                    static_cast<float>(second_values[value]));
                            reinterpret_cast<v16 *>(
                                destinations[destination_index])[vector_index] = reduced;
                        }
                    }
                });
        }
    }
    sycl::event::wait(producer_events);

    std::vector<sycl::event> finalize_events(n_gpus);
    for (int destination = 0; destination < n_gpus; destination++) {
        const bf16 *partials = d_par[destination];
        bf16 *output = d_out[destination];
        const int cs = chunk, pvc_count = npvc;
        using v16 = sycl::vec<uint32_t, 4>;
        const int vector_count = cs / 8;
        const int nwg =
            (vector_count + RS_B_FINAL_WG_SIZE * RS_B_FINAL_VPT - 1) /
            (RS_B_FINAL_WG_SIZE * RS_B_FINAL_VPT);
        finalize_events[destination] = queues[destination].parallel_for(
            sycl::nd_range<1>(nwg * RS_B_FINAL_WG_SIZE, RS_B_FINAL_WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const int lane = item.get_local_linear_id();
                const int base = item.get_group_linear_id() *
                                 RS_B_FINAL_WG_SIZE * RS_B_FINAL_VPT;
#pragma unroll
                for (int element = 0; element < RS_B_FINAL_VPT; element++) {
                    const int vector_index =
                        base + element * RS_B_FINAL_WG_SIZE + lane;
                    if (vector_index >= vector_count)
                        continue;
                    float sums[8] = {};
#pragma unroll
                    for (int p = 0; p < NUM_DEVICES; p++) {
                        if (p >= pvc_count)
                            continue;
                        const v16 partial = reinterpret_cast<const v16 *>(
                            partials + (size_t)p * cs)[vector_index];
                        const bf16 *values = (const bf16 *)&partial;
#pragma unroll
                        for (int value = 0; value < 8; value++)
                            sums[value] += static_cast<float>(values[value]);
                    }
                    v16 reduced;
                    bf16 *values = (bf16 *)&reduced;
#pragma unroll
                    for (int value = 0; value < 8; value++)
                        values[value] = static_cast<bf16>(sums[value]);
                    reinterpret_cast<v16 *>(output)[vector_index] = reduced;
                }
            });
    }
    sycl::event::wait(finalize_events);
}

inline void run_adaptive(std::vector<sycl::queue> &queues,
                         const std::vector<bf16 *> &d_in,
                         const std::vector<bf16 *> &d_out,
                         const std::vector<bf16 *> &d_par,
                         int npvc,
                         const std::vector<std::vector<int>> &tiles,
                         int n_gpus,
                         int total,
                         int chunk) {
    const size_t total_bytes = static_cast<size_t>(total) * sizeof(bf16);
    switch (select_mode(n_gpus, total_bytes)) {
    case Mode::Fused:
        run_fused(queues, d_in, d_out, n_gpus, chunk);
        break;
    case Mode::DestinationSplit:
        run_destination_split(queues, d_in, d_out, d_par, npvc, tiles, n_gpus,
                              chunk);
        break;
    case Mode::Hierarchical:
    default:
        run_hierarchical(queues, d_in, d_out, d_par, npvc, tiles, n_gpus,
                         total, chunk);
        break;
    }
}

} // namespace kittens::collective::reduce_scatter