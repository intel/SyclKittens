# Collective Construction API

SyclKittens exposes device-hot-path building blocks and host-side topology
discovery for writing topology-aware multi-device collectives. The API
abstracts endpoint storage, endpoint-parallel work decomposition, physical
device grouping, and ring construction. It deliberately does not hide
collective algorithms, memory ownership, synchronization, or reduction
semantics.

The production users are:

| Kernel | API use |
|:--|:--|
| AG-K | paired topology, balanced exchange ownership, by-value push/pull routes, split MDFI replication |
| RS-B | by-value destinations, paired destination mapping, destination-split reduction pushes |
| AR-B | paired topology, by-value publication tables, destination-split reduction/publication/replication |
| A2A-S | by-value destinations and one-endpoint-per-workgroup decomposition |

## Endpoint tables

`fixed_endpoints<Endpoint, Capacity>` is a fixed-capacity table intended for
host construction followed by by-value kernel capture. It has no allocation,
indirection, ownership, or bounds check.

```cpp
kittens::collective::fixed_endpoints<bf16 *, 12> destinations;
for (int tile = 0; tile < world; tile++)
    destinations.push(outputs[tile] + offset);

queue.parallel_for(range, [=](sycl::nd_item<1> item) {
    destinations[target][index] = value;
});
```

`push_pull_endpoints<T, Capacity>` stores the four pointers required by a fused
bidirectional exchange: pull source, local pull destination, partner pull
destination, and push destination. AG-K uses it to keep route construction out
of the vector-copy loop while retaining a trivially copied kernel closure.

The caller must ensure that the number of inserted endpoints does not exceed
`Capacity`. This is intentional: a checked or dynamically allocated table would
add control flow or shared-memory traffic to the collective hot path.

## Endpoint-parallel workgroups

`split_workgroup_count(data_groups, endpoints)` returns the number of
workgroups for a Cartesian launch over data groups and endpoints.
`split_workgroup_id(group, endpoints)` maps a linear workgroup to those two
coordinates.

```cpp
const int groups = split_workgroup_count(vector_groups, destinations.size());
queue.parallel_for(sycl::nd_range<1>(groups * wg_size, wg_size),
    [=](sycl::nd_item<1> item) {
        auto split = split_workgroup_id(
            item.get_group_linear_id(), destinations.size());
        copy_vector_group(split.data_group, destinations[split.endpoint]);
    });
```

This is the A2A-S decomposition: each workgroup owns one independent endpoint,
instead of serially visiting every endpoint. `partition_workgroup()` is the
compatible form used by kernels that retain a compile-time switch between a
split launch and a workgroup that loops over all endpoints.

## Paired topology

`grouped_topology<TilesPerGroup>` maps a flat tile index to a local group and
rank. `paired_topology` is the two-tiles-per-PVC specialization:

```cpp
kittens::collective::paired_topology topology{world};
int pvc = topology.group(tile);
int rank = topology.rank(tile);
int partner = topology.partner(tile);
int remote = topology.tile(remote_pvc, rank);
```

`exchange_owner(first, second)` implements the balanced tournament used by
AG-K. Exactly one PVC owns each pairwise exchange, with ownership distributed
across the six PVCs. `destination(rank, index)` maps a tile rank to its
destination-split sequence.

Use this static API only when device enumeration is known to place members of
each local group contiguously.

## Runtime topology discovery

For arbitrary device subsets or enumeration orders, the public host-side API
discovers physical groups and assigns a stable rank within each group. The
generic overload accepts one group key and one local-rank key per participant:

```cpp
auto topology = kittens::collective::discover_topology(
    physical_group_keys, physical_local_rank_keys);
```

`discover_pvc_topology()` specializes this operation for Intel PVC UUIDs. It
groups sibling tiles by the UUID prefix and orders them by the final-byte tile
suffix. The SYCL overload is available through `kittens.dp.hpp`:

```cpp
std::vector<sycl::device> devices = select_devices();
auto topology = kittens::collective::discover_pvc_topology(devices);

int pvc = topology.group(participant);
int tile_rank = topology.rank(participant);
int sibling = topology.participant(pvc, 1 - tile_rank);
```

Group indices follow first appearance in the input. Local ranks do not depend
on input order. Duplicate local-rank keys within a group are rejected.

`plan_lane_rings()` constructs one disjoint ring per stable local rank. Every
physical group must have the same number of participants, and each ring visits
one participant from every group:

```cpp
auto rings = kittens::collective::plan_lane_rings(topology);
for (int lane = 0; lane < rings.lane_count(); ++lane) {
    for (int rank = 0; rank < rings.ring_size(); ++rank) {
        int participant = rings.participant(lane, rank);
        int next = rings.next(participant);
        int previous = rings.previous(participant);
    }
}
```

Discovery and planning allocate and sort only during host setup. The resulting
integer routes can be used to build the existing fixed-capacity endpoint tables
before kernel submission. They add no work to the device hot path.

## Performance contract

The device-side helpers are `constexpr`, header-only aggregates and arithmetic.
They do not submit kernels, allocate memory, synchronize queues, choose
algorithms, or introduce runtime polymorphism. Endpoint tables are captured by
value because shared-USM dispatch tables previously caused page migration and a
severe latency regression. Group size is a template parameter so the compiler
sees the paired-PVC layout at compile time. Runtime discovery is a host-only
setup operation and does not alter that device-side contract.

Changes to this API require:

1. compiling `tests/collective_api_unit.cpp` with C++20;
2. compiling AG-K, RS-B, AR-B, and A2A-S with the production `-O3` flags;
3. full-buffer validation for every payload;
4. interleaved before/after measurements on the same devices for every payload
   before claiming no performance regression.