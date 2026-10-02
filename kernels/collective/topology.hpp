/**
 * @file topology.hpp
 * @brief Topology-aware collective planner for intra-node PVC fabrics.
 *
 * Encodes the MEASURED node model:
 *   - tiles are grouped into PVCs (2 tiles per physical GPU) that share a fast
 *     on-package link (~152 GB/s);
 *   - inter-PVC XeLink edges are ~19 GB/s and, critically, a tile's total
 *     XeLink EGRESS saturates at ~one link (~19 GB/s) no matter how many
 *     destinations it fans out to, while INGRESS scales linearly;
 *   - there is no measurable even/odd or cross-socket asymmetry.
 *
 * The performance-limiting resource is therefore per-tile XeLink egress. Given
 * ANY set of participating tiles the planner emits the schedule that minimises
 * the maximum per-tile egress: combine within a PVC over the fast link, then
 * deliver each block to each remote PVC exactly ONCE (never once-per-tile),
 * balanced across the sending PVC's tiles, then replicate inside the receiving
 * PVC over the fast link. This is provably egress-optimal for one-hop delivery
 * (multi-hop only adds egress, and ingress is already free).
 */
#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>

#include "ops/collective/topology.dp.hpp"

namespace kittens {
namespace topo {

// Measured node link model. Defaults come from PVC profiling; override
// per node if the profiler reports different numbers.
struct LinkModel {
    double intra_pvc_gbps  = 152.0; // on-package tile<->tile
    double xelink_gbps     = 19.0;  // inter-PVC directed edge
    double egress_cap_gbps = 19.0;  // per-tile TOTAL XeLink egress (saturates)
};

// One copy in the schedule. All ops in the same `phase` may run concurrently;
// consecutive phases are separated by a barrier. `block` is the participant
// index whose stripe is being moved; the destination slot is always `block`.
enum class CopyDirection : uint8_t { Push, Pull };

struct CopyOp {
    int  src;          // participant index that issues the copy (owns the data)
    int  dst;          // participant index receiving into slot `block`
    int  block;        // logical stripe id == owning participant index
    int  phase;        // 0..nphases-1
    bool fast;         // true = intra-PVC on-package link; false = XeLink
    bool from_output;  // false: read src's input; true: read src's output slot
    CopyDirection dir = CopyDirection::Push; // endpoint that issues the transfer
};

struct Plan {
    std::vector<CopyOp> ops;
    int nphases = 0;
    int max_egress_stripes = 0; // worst-case XeLink stripes a single tile sends
    int max_bidir_edge_load = 0; // max stripes in either direction on one PVC pair
};

// Group participating tiles into PVCs. Two tiles share a PVC when their device
// UUIDs match except for the trailing tile byte (last 2 hex chars).
inline std::vector<int> group_pvc(const std::vector<std::string>& uuids) {
    return kittens::collective::discover_pvc_topology(uuids).group_of;
}

// Build an egress-optimal All-Gather plan for participants 0..n-1 with the
// given PVC group id per participant. Works for any tile subset and any PVC
// sizes (including a lone tile or uneven groups).
inline Plan plan_all_gather(const std::vector<int>& pvc_of) {
    const int n = (int)pvc_of.size();
    int npvc = 0; for (int p : pvc_of) npvc = std::max(npvc, p + 1);
    std::vector<std::vector<int>> tiles(npvc);
    for (int t = 0; t < n; t++) tiles[pvc_of[t]].push_back(t);

    Plan plan;
    // Phase 0a: each tile lands its own stripe in its own slot (local copy).
    for (int t = 0; t < n; t++)
        plan.ops.push_back({t, t, t, 0, true, false});
    // Phase 0b: replicate native stripes to every peer in the same PVC (fast).
    for (int p = 0; p < npvc; p++)
        for (int t : tiles[p])
            for (int u : tiles[p]) if (u != t)
                plan.ops.push_back({t, u, t, 0, true, false});

    // Phase 0c: inter-PVC dedup delivery (XeLink). Each block is sent to each
    // remote PVC exactly once; the owning tile egresses its own block, and the
    // landing tile within the remote PVC is chosen round-robin to spread
    // arrivals. Record who holds each (remote block, PVC) for the replicate.
    std::vector<int> egress(n, 0);
    // holder[q][b] = tile in PVC q that received remote block b (or -1).
    std::vector<std::vector<int>> holder(npvc, std::vector<int>(n, -1));
    for (int p = 0; p < npvc; p++) {
        for (int q = 0; q < npvc; q++) {
            if (q == p) continue;
            int rr = 0;
            for (int b : tiles[p]) {                // each native block of P
                int landing = tiles[q][rr % (int)tiles[q].size()];
                rr++;
                plan.ops.push_back({b, landing, b, 0, false, false});
                egress[b]++;
                holder[q][b] = landing;
            }
        }
    }

    // Phase 1: within each receiving PVC, replicate the remote blocks it landed
    // to all other tiles of that PVC over the fast on-package link.
    for (int q = 0; q < npvc; q++)
        for (int b = 0; b < n; b++) {
            int h = holder[q][b];
            if (h < 0) continue;
            for (int u : tiles[q]) if (u != h)
                plan.ops.push_back({h, u, b, 1, true, true});
        }

    plan.nphases = 2;
    for (int e : egress) plan.max_egress_stripes = std::max(plan.max_egress_stripes, e);
    return plan;
}

// Build the byte-optimal bidirectional schedule used by the large-payload path.
// Each PVC pair chooses one owner. That owner pushes its native stripes and
// pulls the reverse stripes, so both physical directions remain active while
// every block still crosses into each remote PVC exactly once.
inline Plan plan_all_gather_bidir(const std::vector<int>& pvc_of) {
    const int n = (int)pvc_of.size();
    int npvc = 0; for (int p : pvc_of) npvc = std::max(npvc, p + 1);
    std::vector<std::vector<int>> tiles(npvc);
    for (int tile = 0; tile < n; tile++) tiles[pvc_of[tile]].push_back(tile);

    Plan plan;
    for (int tile = 0; tile < n; tile++)
        plan.ops.push_back({tile, tile, tile, 0, true, false});
    for (int p = 0; p < npvc; p++)
        for (int source : tiles[p])
            for (int destination : tiles[p]) if (destination != source)
                plan.ops.push_back({source, destination, source, 0, true, false});

    std::vector<int> issued(n, 0);
    std::vector<std::vector<int>> holder(npvc, std::vector<int>(n, -1));
    for (int left = 0; left < npvc; left++) {
        for (int right = left + 1; right < npvc; right++) {
            if (tiles[left].empty() || tiles[right].empty()) continue;
            const int owner = ((left + right) & 1) ? right : left;
            plan.max_bidir_edge_load = std::max(
                plan.max_bidir_edge_load,
                std::max((int)tiles[left].size(), (int)tiles[right].size()));
            for (size_t rank = 0; rank < tiles[left].size(); rank++) {
                const int source = tiles[left][rank];
                const int destination = tiles[right][rank % tiles[right].size()];
                const CopyDirection direction = owner == left
                    ? CopyDirection::Push : CopyDirection::Pull;
                plan.ops.push_back({source, destination, source, 0, false, false,
                                    direction});
                issued[direction == CopyDirection::Push ? source : destination]++;
                holder[right][source] = destination;
            }
            for (size_t rank = 0; rank < tiles[right].size(); rank++) {
                const int source = tiles[right][rank];
                const int destination = tiles[left][rank % tiles[left].size()];
                const CopyDirection direction = owner == right
                    ? CopyDirection::Push : CopyDirection::Pull;
                plan.ops.push_back({source, destination, source, 0, false, false,
                                    direction});
                issued[direction == CopyDirection::Push ? source : destination]++;
                holder[left][source] = destination;
            }
        }
    }

    for (int p = 0; p < npvc; p++)
        for (int block = 0; block < n; block++) {
            const int landing = holder[p][block];
            if (landing < 0) continue;
            for (int destination : tiles[p]) if (destination != landing)
                plan.ops.push_back({landing, destination, block, 1, true, true});
        }
    plan.nphases = 2;
    for (int count : issued)
        plan.max_egress_stripes = std::max(plan.max_egress_stripes, count);
    return plan;
}

} // namespace topo
} // namespace kittens
