#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../include/ops/collective/collective.dp.hpp"
#include "dispatch.hpp"
#include "topology.hpp"

namespace kittens::collective::all_gather {

using bf16 = sycl::ext::oneapi::bfloat16;
using vec128 = sycl::vec<uint32_t, 4>;

inline constexpr int WG_SIZE = 256;
inline constexpr int EPT = 8;
inline constexpr int AG_K_WG_SIZE = 256;
inline constexpr int AG_K_VPT = 8;

enum class Mode {
    FusedTwoTier,
    TopologyFused,
    BidirPushPull,
    BalancedSplit,
};

inline Mode select_mode(int n_gpus, size_t total_bytes) {
    const auto dispatch = kittens::collective::thresholds(
        kittens::collective::Kind::AllGather, n_gpus);
    if (total_bytes < dispatch.latency_max_bytes)
        return Mode::FusedTwoTier;
    if (total_bytes < dispatch.large_min_bytes)
        return Mode::TopologyFused;
    if (n_gpus == 12)
        return Mode::BidirPushPull;
    return Mode::BalancedSplit;
}

inline const char *mode_label(Mode mode) {
    switch (mode) {
    case Mode::FusedTwoTier:
        return "AG-adaptive(fused-two-tier)";
    case Mode::TopologyFused:
        return "AG-adaptive(topology-fused)";
    case Mode::BidirPushPull:
        return "AG-adaptive(bidir-push-pull)";
    case Mode::BalancedSplit:
    default:
        return "AG-adaptive(balanced-split)";
    }
}

inline void run_topology_fused(std::vector<sycl::queue> &queues,
                               const std::vector<bf16 *> &d_in,
                               const std::vector<bf16 *> &d_out,
                               int n_gpus,
                               int epd,
                               int vpd,
                               int n_wgs) {
    const int n = n_gpus, nP = n / 2;
    const kittens::collective::paired_topology topology{n};
    // Phase 0/1/2: one kernel per tile, own stripe -> all its targets.
    for (int t = 0; t < n; t++) {
        bf16 *in = d_in[t];
        kittens::collective::fixed_endpoints<bf16 *, 8> dsts;
        int par = topology.partner(t), p = topology.group(t);
        dsts.push(d_out[t] + (size_t)t * epd);      // phase 0 local
        dsts.push(d_out[par] + (size_t)t * epd);    // phase 1 partner
        for (int q = 0; q < nP; q++) {              // phase 2 XeLink
            if (q == p)
                continue;
            int landing = topology.tile(q, topology.rank(p));
            dsts.push(d_out[landing] + (size_t)t * epd);
        }
        int vp = vpd, nw = n_wgs;
        queues[t].parallel_for(sycl::nd_range<1>(nw * WG_SIZE, WG_SIZE),
                               [=](sycl::nd_item<1> item) {
                                   int lid = item.get_local_linear_id();
                                   int wg_base =
                                       item.get_group_linear_id() * WG_SIZE * EPT;
                                   auto *sp = reinterpret_cast<const vec128 *>(in);
                                   for (int d = 0; d < dsts.size(); d++) {
                                       auto *dp = reinterpret_cast<vec128 *>(dsts[d]);
#pragma unroll
                                       for (int e = 0; e < EPT; e++) {
                                           int vi = wg_base + e * WG_SIZE + lid;
                                           if (vi < vp)
                                               dp[vi] = sp[vi];
                                       }
                                   }
                               });
    }
    for (auto &q : queues)
        q.wait();

    // Phase 3: one kernel per tile, replicate landed remote blocks to partner.
    for (int t = 0; t < n; t++) {
        int par = topology.partner(t);
        bf16 *own = d_out[t];
        bf16 *pbuf = d_out[par];
        kittens::collective::fixed_endpoints<int, 8> offs;
        int q3 = topology.group(t), b = topology.rank(t);
        for (int pp = 0; pp < nP; pp++) {
            if (pp == q3 || (pp & 1) != b)
                continue;
            offs.push(topology.tile(pp, 0)); // block start slot
        }
        int vp = vpd, nw = n_wgs, ep = epd;
        queues[t].parallel_for(sycl::nd_range<1>(nw * WG_SIZE, WG_SIZE),
                               [=](sycl::nd_item<1> item) {
                                   int lid = item.get_local_linear_id();
                                   int wg_base =
                                       item.get_group_linear_id() * WG_SIZE * EPT;
                                   for (int bidx = 0; bidx < offs.size(); bidx++) {
                                       int slot = offs[bidx];
#pragma unroll
                                       for (int half = 0; half < 2; half++) {
                                           auto *sp = reinterpret_cast<const vec128 *>(
                                               own + (size_t)(slot + half) * ep);
                                           auto *dp = reinterpret_cast<vec128 *>(
                                               pbuf + (size_t)(slot + half) * ep);
#pragma unroll
                                           for (int e = 0; e < EPT; e++) {
                                               int vi = wg_base + e * WG_SIZE + lid;
                                               if (vi < vp)
                                                   dp[vi] = sp[vi];
                                           }
                                       }
                                   }
                               });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_balanced_split(std::vector<sycl::queue> &queues,
                               const std::vector<bf16 *> &d_in,
                               const std::vector<bf16 *> &d_out,
                               int n_gpus,
                               int epd,
                               int vpd,
                               int n_wgs) {
    const int n = n_gpus, nP = n / 2;
    const int vhalf = vpd / 2;
    // Phase 0/1/2: local + partner (full) + inter-PVC split-halves.
    for (int t = 0; t < n; t++) {
        bf16 *in = d_in[t];
        bf16 *dstp[16];
        int voff[16];
        int vend[16];
        int cnt = 0;
        int par = t ^ 1, p = t / 2;
        dstp[cnt] = d_out[t] + (size_t)t * epd;
        voff[cnt] = 0;
        vend[cnt] = vpd;
        cnt++;
        dstp[cnt] = d_out[par] + (size_t)t * epd;
        voff[cnt] = 0;
        vend[cnt] = vpd;
        cnt++;
        for (int q = 0; q < nP; q++) {
            if (q == p)
                continue;
            dstp[cnt] = d_out[2 * q] + (size_t)t * epd;
            voff[cnt] = 0;
            vend[cnt] = vhalf;
            cnt++;
            dstp[cnt] = d_out[2 * q + 1] + (size_t)t * epd;
            voff[cnt] = vhalf;
            vend[cnt] = vpd;
            cnt++;
        }
        int nw = n_wgs;
        queues[t].parallel_for(sycl::nd_range<1>(nw * WG_SIZE, WG_SIZE),
                               [=](sycl::nd_item<1> item) {
                                   int lid = item.get_local_linear_id();
                                   int wg_base =
                                       item.get_group_linear_id() * WG_SIZE * EPT;
                                   auto *sp = reinterpret_cast<const vec128 *>(in);
                                   for (int d = 0; d < cnt; d++) {
                                       auto *dp = reinterpret_cast<vec128 *>(dstp[d]);
                                       int lo = voff[d], hi = vend[d];
#pragma unroll
                                       for (int e = 0; e < EPT; e++) {
                                           int vi = wg_base + e * WG_SIZE + lid;
                                           if (vi >= lo && vi < hi)
                                               dp[vi] = sp[vi];
                                       }
                                   }
                               });
    }
    for (auto &q : queues)
        q.wait();

    // Phase 3: complementary-half swap with partner for the 10 remote slots.
    for (int t = 0; t < n; t++) {
        int par = t ^ 1, q = t / 2, b = t & 1;
        bf16 *self = d_out[t];
        bf16 *pbuf = d_out[par];
        int slots[16];
        int cnt = 0;
        for (int s = 0; s < n; s++) {
            if (s == 2 * q || s == 2 * q + 1)
                continue;
            slots[cnt++] = s;
        }
        int lo = (b == 0) ? vhalf : 0;
        int hi = (b == 0) ? vpd : vhalf;
        int ep = epd, nw = n_wgs;
        queues[t].parallel_for(sycl::nd_range<1>(nw * WG_SIZE, WG_SIZE),
                               [=](sycl::nd_item<1> item) {
                                   int lid = item.get_local_linear_id();
                                   int wg_base =
                                       item.get_group_linear_id() * WG_SIZE * EPT;
                                   for (int si = 0; si < cnt; si++) {
                                       int slot = slots[si];
                                       auto *sp = reinterpret_cast<const vec128 *>(
                                           pbuf + (size_t)slot * ep);
                                       auto *dp = reinterpret_cast<vec128 *>(
                                           self + (size_t)slot * ep);
#pragma unroll
                                       for (int e = 0; e < EPT; e++) {
                                           int vi = wg_base + e * WG_SIZE + lid;
                                           if (vi >= lo && vi < hi)
                                               dp[vi] = sp[vi];
                                       }
                                   }
                               });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_fused_two_tier(std::vector<sycl::queue> &queues,
                               const std::vector<bf16 *> &d_in,
                               const std::vector<bf16 *> &d_out,
                               int n_gpus,
                               int epd,
                               int vpd,
                               int n_wgs) {
    const int n = n_gpus, nP = n / 2;
    const kittens::collective::paired_topology topology{n};
    for (int t = 0; t < n; t++) {
        int par = topology.partner(t), q = topology.group(t), b = topology.rank(t);
        using route_t = kittens::collective::pull_replicate_endpoint<bf16>;
        kittens::collective::fixed_endpoints<route_t, 16> routes;
        routes.push({d_in[t], d_out[t] + (size_t)t * epd, nullptr});
        routes.push({d_in[par], d_out[t] + (size_t)par * epd, nullptr});
        for (int q2 = 0; q2 < nP; q2++) {
            if (q2 == q)
                continue;
            int s = topology.tile(q2, b);
            routes.push(
                {d_in[s], d_out[t] + (size_t)s * epd, d_out[par] + (size_t)s * epd});
        }
        int nw = n_wgs, vp = vpd;
        queues[t].parallel_for(sycl::nd_range<1>(nw * WG_SIZE, WG_SIZE),
                               [=](sycl::nd_item<1> item) {
                                   int lid = item.get_local_linear_id();
                                   int wg_base =
                                       item.get_group_linear_id() * WG_SIZE * EPT;
                                   for (int d = 0; d < routes.size(); d++) {
                                       auto *sp = reinterpret_cast<const vec128 *>(
                                           routes[d].source);
                                       auto *dp = reinterpret_cast<vec128 *>(
                                           routes[d].self_destination);
                                       auto *pp = routes[d].partner_destination
                                           ? reinterpret_cast<vec128 *>(
                                                 routes[d].partner_destination)
                                           : nullptr;
#pragma unroll
                                       for (int e = 0; e < EPT; e++) {
                                           int vi = wg_base + e * WG_SIZE + lid;
                                           if (vi < vp) {
                                               vec128 v = sp[vi];
                                               dp[vi] = v;
                                               if (pp)
                                                   pp[vi] = v;
                                           }
                                       }
                                   }
                               });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_bidir_push_pull(std::vector<sycl::queue> &queues,
                                const std::vector<bf16 *> &d_in,
                                const std::vector<bf16 *> &d_out,
                                int n_gpus,
                                int epd,
                                int vpd) {
    const int n = n_gpus, nP = n / 2;
    const kittens::collective::paired_topology topology{n};
    for (int t = 0; t < n; t++) {
        const int par = topology.partner(t);
        const int p = topology.group(t), b = topology.rank(t);
        const bf16 *partner_source = d_in[par];
        bf16 *own_output = d_out[t];
        const bf16 *native_source = d_in[t];
        const int own_slot = t, partner_slot = par;
        const int vp = vpd;
        kittens::collective::push_pull_endpoints<bf16, 8> routes;
        for (int q = 0; q < nP; q++) {
            if (q == p)
                continue;
            const int owner = topology.exchange_owner(p, q);
            if (owner != p)
                continue;
            const int source = topology.tile(q, b);
            routes.push(d_in[source], d_out[t] + (size_t)source * epd,
                        d_out[par] + (size_t)source * epd,
                        d_out[source] + (size_t)t * epd);
        }
        const int vector_begin = 0;
        const int vector_end = vp;
        const int vector_count = vector_end - vector_begin;
        const int nw = std::max(
            1, (vector_count + AG_K_WG_SIZE * AG_K_VPT - 1) /
                   (AG_K_WG_SIZE * AG_K_VPT));
        queues[t].parallel_for(
            sycl::nd_range<1>(nw * AG_K_WG_SIZE, AG_K_WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_linear_id();
                const int base =
                    vector_begin + item.get_group_linear_id() * AG_K_WG_SIZE * AG_K_VPT;
                auto *native_vectors = reinterpret_cast<const vec128 *>(native_source);
                auto *partner_vectors = reinterpret_cast<const vec128 *>(partner_source);
                auto *own_slot_vectors =
                    reinterpret_cast<vec128 *>(own_output + (size_t)own_slot * epd);
                auto *partner_slot_vectors =
                    reinterpret_cast<vec128 *>(own_output + (size_t)partner_slot * epd);
#pragma unroll
                for (int element = 0; element < AG_K_VPT; element++) {
                    const int index = base + element * AG_K_WG_SIZE + lid;
                    if (index >= vector_end)
                        continue;
                    const vec128 native = native_vectors[index];
                    own_slot_vectors[index] = native;
                    partner_slot_vectors[index] = partner_vectors[index];
                    for (int operation = 0; operation < routes.size(); operation++) {
                        reinterpret_cast<vec128 *>(routes.push_destinations[operation])
                            [index] = native;
                        const vec128 pulled = reinterpret_cast<const vec128 *>(
                            routes.pull_sources[operation])[index];
                        reinterpret_cast<vec128 *>(routes.pull_self[operation])[index] =
                            pulled;
                        reinterpret_cast<vec128 *>(routes.pull_partner[operation])[index] =
                            pulled;
                    }
                }
            });
    }
    for (auto &q : queues)
        q.wait();

    for (int t = 0; t < n; t++) {
        const int par = topology.partner(t);
        const int p = topology.group(t), b = topology.rank(t);
        bf16 *self = d_out[t];
        bf16 *partner = d_out[par];
        const int vp = vpd;
        kittens::collective::fixed_endpoints<int, 8> slots;
        for (int q = 0; q < nP; q++) {
            if (q == p)
                continue;
            const int owner = topology.exchange_owner(p, q);
            if (owner == p)
                continue;
            slots.push(topology.tile(q, b));
        }
        const int vector_begin = 0;
        const int vector_end = vp;
        const int vector_count = vector_end - vector_begin;
        const int nw = std::max(
            1, (vector_count + AG_K_WG_SIZE * AG_K_VPT - 1) /
                   (AG_K_WG_SIZE * AG_K_VPT));
        const int launch_groups = kittens::collective::partitioned_workgroup_count(
            nw, slots.size(), true);
        queues[t].parallel_for(
            sycl::nd_range<1>(launch_groups * AG_K_WG_SIZE, AG_K_WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_linear_id();
                const int group = item.get_group_linear_id();
                const auto partition = kittens::collective::partition_workgroup(
                    group, slots.size(), true);
                const int vector_group = partition.data_group;
                const int base =
                    vector_begin + vector_group * AG_K_WG_SIZE * AG_K_VPT;
#pragma unroll
                for (int element = 0; element < AG_K_VPT; element++) {
                    const int index = base + element * AG_K_WG_SIZE + lid;
                    if (index >= vector_end)
                        continue;
                    for (int operation = partition.begin; operation < partition.end;
                         operation++) {
                        const size_t offset = (size_t)slots[operation] * epd;
                        reinterpret_cast<vec128 *>(partner + offset)[index] =
                            reinterpret_cast<const vec128 *>(self + offset)[index];
                    }
                }
            });
    }
    for (auto &q : queues)
        q.wait();
}

inline void run_mode(Mode mode,
                     std::vector<sycl::queue> &queues,
                     const std::vector<bf16 *> &d_in,
                     const std::vector<bf16 *> &d_out,
                     int n_gpus,
                     int epd,
                     int vpd,
                     int n_wgs) {
    switch (mode) {
    case Mode::FusedTwoTier:
        run_fused_two_tier(queues, d_in, d_out, n_gpus, epd, vpd, n_wgs);
        break;
    case Mode::TopologyFused:
        run_topology_fused(queues, d_in, d_out, n_gpus, epd, vpd, n_wgs);
        break;
    case Mode::BidirPushPull:
        run_bidir_push_pull(queues, d_in, d_out, n_gpus, epd, vpd);
        break;
    case Mode::BalancedSplit:
    default:
        run_balanced_split(queues, d_in, d_out, n_gpus, epd, vpd, n_wgs);
        break;
    }
}

inline void run_adaptive(std::vector<sycl::queue> &queues,
                         const std::vector<bf16 *> &d_in,
                         const std::vector<bf16 *> &d_out,
                         int n_gpus,
                         int epd,
                         int vpd,
                         int n_wgs,
                         size_t total_bytes) {
    run_mode(select_mode(n_gpus, total_bytes), queues, d_in, d_out, n_gpus, epd,
             vpd, n_wgs);
}

} // namespace kittens::collective::all_gather