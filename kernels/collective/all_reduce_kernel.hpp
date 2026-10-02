#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../include/ops/collective/collective.dp.hpp"
#include "dispatch.hpp"
#include "topology.hpp"

namespace kittens::collective::all_reduce {

using bf16 = sycl::ext::oneapi::bfloat16;
using vec128 = sycl::vec<uint32_t, 4>;

inline constexpr int NUM_DEVICES = 12;
inline constexpr int WG_SIZE = 256;
inline constexpr int EPT = 8;
inline constexpr int VEC_BF16 = 8;
inline constexpr int AR_B_RS_VPT = 1;
inline constexpr int AR_B_PUBLISH_VPT = 1;
inline constexpr int AR_B_REPLICATE_VPT = 8;

enum class Mode {
    DirectFullReduction,
    BalancedReduceScatterAllGather,
};

inline Mode select_mode(int n_gpus, size_t bytes) {
    const auto dispatch = kittens::collective::thresholds(
        kittens::collective::Kind::AllReduce, n_gpus);
    if (bytes <= dispatch.latency_max_bytes)
        return Mode::DirectFullReduction;
    return Mode::BalancedReduceScatterAllGather;
}

inline void run_full_reduction(std::vector<sycl::queue> &queues,
                               const std::vector<bf16 *> &d_in,
                               const std::vector<bf16 *> &d_res,
                               int n_gpus,
                               int N) {
    bf16 *in_ptr[NUM_DEVICES] = {nullptr};
    for (int d = 0; d < n_gpus; d++)
        in_ptr[d] = d_in[d];

    for (int d = 0; d < n_gpus; d++) {
        bf16 *out = d_res[d];
        int ng = n_gpus, n = N;
        bf16 *srcs[NUM_DEVICES];
        for (int s = 0; s < NUM_DEVICES; s++)
            srcs[s] = in_ptr[s];
        int nwg = (n + WG_SIZE * EPT - 1) / (WG_SIZE * EPT);
        queues[d].parallel_for(
            sycl::nd_range<1>(nwg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int gid = it.get_global_linear_id();
#pragma unroll
                for (int e = 0; e < EPT; e++) {
                    int idx = gid * EPT + e;
                    if (idx >= n)
                        break;
                    float acc = 0.0f;
#pragma unroll
                    for (int s = 0; s < NUM_DEVICES; s++)
                        if (s < ng)
                            acc += static_cast<float>(srcs[s][idx]);
                    out[idx] = static_cast<bf16>(acc);
                }
            });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_balanced_rs_ag(std::vector<sycl::queue> &queues,
                               const std::vector<bf16 *> &d_in,
                               const std::vector<bf16 *> &d_res,
                               const std::vector<bf16 *> &d_par,
                               int npvc,
                               const std::vector<std::vector<int>> &tiles,
                               int n_gpus,
                               int chunk) {
    const int n = n_gpus, nP = n / 2;
    const kittens::collective::paired_topology topology{n};
    std::vector<sycl::event> events(n);

    for (int p = 0; p < npvc; p++) {
        const int first = tiles[p][0];
        const int second = tiles[p].size() > 1 ? tiles[p][1] : first;
        for (int rank = 0; rank < static_cast<int>(tiles[p].size()); rank++) {
            const int issuer = tiles[p][rank];
            const bf16 *source_first = d_in[first];
            const bf16 *source_second = d_in[second];
            kittens::collective::fixed_endpoints<bf16 *, NUM_DEVICES>
                destinations;
            for (int destination = rank; destination < n; destination += 2)
                destinations.push(d_par[destination] + (size_t)p * chunk);
            const int vector_count = chunk / VEC_BF16;
            const int nwg =
                (vector_count + WG_SIZE * AR_B_RS_VPT - 1) /
                (WG_SIZE * AR_B_RS_VPT);
            const int launch_groups =
                kittens::collective::partitioned_workgroup_count(
                    nwg, destinations.size(), true);
            const int cs = chunk;
            events[issuer] = queues[issuer].parallel_for(
                sycl::nd_range<1>(launch_groups * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_linear_id();
                    const int group = item.get_group_linear_id();
                    const auto partition =
                        kittens::collective::partition_workgroup(
                            group, destinations.size(), true);
                    const int group_base =
                        partition.data_group * WG_SIZE * AR_B_RS_VPT;
#pragma unroll
                    for (int vector = 0; vector < AR_B_RS_VPT; vector++) {
                        const int vector_index =
                            group_base + vector * WG_SIZE + lid;
                        if (vector_index >= vector_count)
                            continue;
                        for (int destination_index = partition.begin;
                             destination_index < partition.end;
                             destination_index++) {
                            const int destination =
                                topology.destination(rank, destination_index);
                            const size_t source_vector =
                                ((size_t)destination * cs) / VEC_BF16 +
                                vector_index;
                            const vec128 first_vector =
                                reinterpret_cast<const vec128 *>(source_first)
                                    [source_vector];
                            const vec128 second_vector =
                                reinterpret_cast<const vec128 *>(source_second)
                                    [source_vector];
                            vec128 reduced;
                            bf16 *first_values = (bf16 *)&first_vector;
                            bf16 *second_values = (bf16 *)&second_vector;
                            bf16 *reduced_values = (bf16 *)&reduced;
#pragma unroll
                            for (int lane = 0; lane < VEC_BF16; lane++)
                                reduced_values[lane] = static_cast<bf16>(
                                    static_cast<float>(first_values[lane]) +
                                    static_cast<float>(second_values[lane]));
                            reinterpret_cast<vec128 *>(
                                destinations[destination_index])[vector_index] =
                                reduced;
                        }
                    }
                });
        }
    }
    sycl::event::wait(events);

    {
        const int vectors = chunk / VEC_BF16;
        const int publish_groups = std::max(
            1,
            (vectors + WG_SIZE * AR_B_PUBLISH_VPT - 1) /
                (WG_SIZE * AR_B_PUBLISH_VPT));
        const int replicate_groups = std::max(
            1,
            (vectors + WG_SIZE * AR_B_REPLICATE_VPT - 1) /
                (WG_SIZE * AR_B_REPLICATE_VPT));
        for (int t = 0; t < n; t++) {
            const int partner = topology.partner(t), p = topology.group(t);
            const bf16 *partials = d_par[t];
            kittens::collective::fixed_endpoints<bf16 *, 8> destinations;
            destinations.push(d_res[t] + (size_t)t * chunk);
            destinations.push(d_res[partner] + (size_t)t * chunk);
            for (int q = 0; q < nP; q++) {
                if (q == p)
                    continue;
                const int landing = topology.tile(q, topology.rank(t));
                destinations.push(d_res[landing] + (size_t)t * chunk);
            }
            const int vector_count = vectors, nwg = publish_groups;
            const int cs = chunk, pvc_count = npvc;
            const int launch_groups =
                kittens::collective::partitioned_workgroup_count(
                    nwg, destinations.size(), true);
            events[t] = queues[t].parallel_for(
                sycl::nd_range<1>(launch_groups * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_linear_id();
                    const int group = item.get_group_linear_id();
                    const auto partition =
                        kittens::collective::partition_workgroup(
                            group, destinations.size(), true);
                    const int base =
                        partition.data_group * WG_SIZE * AR_B_PUBLISH_VPT;
#pragma unroll
                    for (int element = 0; element < AR_B_PUBLISH_VPT;
                         element++) {
                        const int index = base + element * WG_SIZE + lid;
                        if (index >= vector_count)
                            continue;
                        float sums[VEC_BF16] = {};
#pragma unroll
                        for (int source_pvc = 0; source_pvc < NUM_DEVICES;
                             source_pvc++) {
                            if (source_pvc >= pvc_count)
                                continue;
                            const vec128 partial =
                                reinterpret_cast<const vec128 *>(
                                    partials + (size_t)source_pvc * cs)[index];
                            const bf16 *values = (const bf16 *)&partial;
#pragma unroll
                            for (int lane = 0; lane < VEC_BF16; lane++)
                                sums[lane] += static_cast<float>(values[lane]);
                        }
                        vec128 reduced;
                        bf16 *reduced_values = (bf16 *)&reduced;
#pragma unroll
                        for (int lane = 0; lane < VEC_BF16; lane++)
                            reduced_values[lane] = static_cast<bf16>(sums[lane]);
                        for (int destination = partition.begin;
                             destination < partition.end; destination++)
                            reinterpret_cast<vec128 *>(
                                destinations[destination])[index] = reduced;
                    }
                });
        }
        sycl::event::wait(events);

        for (int t = 0; t < n; t++) {
            const int partner = topology.partner(t);
            const int pvc = topology.group(t), rank = topology.rank(t);
            bf16 *self = d_res[t];
            bf16 *partner_output = d_res[partner];
            kittens::collective::fixed_endpoints<int, 8> source_slots;
            for (int source_pvc = 0; source_pvc < nP; source_pvc++) {
                if (source_pvc == pvc)
                    continue;
                source_slots.push(topology.tile(source_pvc, rank));
            }
            const int vector_count = vectors, nwg = replicate_groups,
                      cs = chunk;
            const int launch_groups =
                kittens::collective::partitioned_workgroup_count(
                    nwg, source_slots.size(), true);
            events[t] = queues[t].parallel_for(
                sycl::nd_range<1>(launch_groups * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_linear_id();
                    const int group = item.get_group_linear_id();
                    const auto partition =
                        kittens::collective::partition_workgroup(
                            group, source_slots.size(), true);
                    const int base =
                        partition.data_group * WG_SIZE * AR_B_REPLICATE_VPT;
#pragma unroll
                    for (int element = 0; element < AR_B_REPLICATE_VPT;
                         element++) {
                        const int index = base + element * WG_SIZE + lid;
                        if (index >= vector_count)
                            continue;
                        for (int source_index = partition.begin;
                             source_index < partition.end; source_index++) {
                            const int slot = source_slots[source_index];
                            const size_t offset = (size_t)slot * cs;
                            reinterpret_cast<vec128 *>(
                                partner_output + offset)[index] =
                                reinterpret_cast<const vec128 *>(self + offset)
                                    [index];
                        }
                    }
                });
        }
        sycl::event::wait(events);
    }
}

inline void run_adaptive(std::vector<sycl::queue> &queues,
                         const std::vector<bf16 *> &d_in,
                         const std::vector<bf16 *> &d_res,
                         const std::vector<bf16 *> &d_par,
                         int npvc,
                         const std::vector<std::vector<int>> &tiles,
                         int n_gpus,
                         int N,
                         int chunk) {
    const size_t bytes = static_cast<size_t>(N) * sizeof(bf16);
    const Mode mode = select_mode(n_gpus, bytes);
    if (mode == Mode::DirectFullReduction)
        run_full_reduction(queues, d_in, d_res, n_gpus, N);
    else
        run_balanced_rs_ag(queues, d_in, d_res, d_par, npvc, tiles, n_gpus,
                           chunk);
}

} // namespace kittens::collective::all_reduce