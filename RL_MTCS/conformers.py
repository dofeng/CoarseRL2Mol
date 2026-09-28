"""Bounded hex conformers for CSV trees; connectivity is never inferred."""

from collections import deque
from dataclasses import replace
from functools import lru_cache

from .geometry import HEX_DIRECTIONS, axial_add, transform_axial, hex_edge_valid
from .models import InverseInputError, TopologyUnit


def geometry_signature(unit: TopologyUnit):
    return (
        tuple((n.node_id, n.local_axial) for n in unit.nodes),
        tuple((p.port_id, p.direction) for p in unit.ports),
    )


def _ring_node_sequences(unit: TopologyUnit):
    by_coordinate = {tuple(node.local_axial): int(node.node_id) for node in unit.nodes}
    sequences = []
    for ring in unit.local_rings:
        try:
            sequence = tuple(by_coordinate[tuple(value)] for value in ring)
        except KeyError:
            return ()
        sequences.append(sequence)
    return tuple(sequences)


def _candidate_ring_coordinates(
    original: TopologyUnit,
    candidate: TopologyUnit,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    sequences = _ring_node_sequences(original)
    coords = {int(node.node_id): tuple(node.local_axial) for node in candidate.nodes}
    return tuple(tuple(coords[node_id] for node_id in sequence) for sequence in sequences)


def validate_conformer(original: TopologyUnit, candidate: TopologyUnit) -> None:
    """Only generated coordinates and their derived port vectors may change."""
    if len(candidate.nodes) != len(original.nodes) or len(candidate.ports) != len(original.ports):
        raise InverseInputError("conformer changed node or port count")
    restored = replace(
        candidate,
        nodes=tuple(replace(n, local_axial=o.local_axial)
                    for n, o in zip(candidate.nodes, original.nodes)),
        ports=tuple(replace(p, direction=o.direction)
                    for p, o in zip(candidate.ports, original.ports)),
        local_rings=original.local_rings,
    )
    if restored != original:
        raise InverseInputError("conformer changed topology identity or connectivity")
    if geometry_signature(original) == geometry_signature(candidate):
        if candidate.local_rings != original.local_rings:
            raise InverseInputError("conformer changed ring identity without changing embedding")
        return
    generated = dict(original.metadata).get("geometry_source") == "generated_from_connectivity"
    if not generated:
        raise InverseInputError("cannot change an explicitly fixed geometry")
    coords = {n.node_id: n.local_axial for n in candidate.nodes}
    if len(set(coords.values())) != len(coords):
        raise InverseInputError("conformer contains overlapping nodes")
    for edge in candidate.internal_edges:
        a, b = coords[edge.source_node_id], coords[edge.target_node_id]
        if not hex_edge_valid(a, b):
            raise InverseInputError("conformer contains a non-hex edge")
    types = {n.node_id: n.su_type for n in candidate.nodes}
    ordinary = [e for e in candidate.internal_edges
                if all(types[n] <= 25 and types[n] != 25
                       for n in (e.source_node_id, e.target_node_id))]
    # Local coordinates are normalized to a node at (0,0). Account for the
    # removed lattice origin instead of testing normalized phase directly.
    if not any(all(hex_edge_valid(
        axial_add(coords[e.source_node_id], (phase, 0)),
        axial_add(coords[e.target_node_id], (phase, 0)), honeycomb=True,
    ) for e in ordinary) for phase in range(3)):
        raise InverseInputError("conformer violates ordinary honeycomb directions")
    virtual = []
    for port in candidate.ports:
        if port.direction not in HEX_DIRECTIONS:
            raise InverseInputError("conformer contains a non-hex port")
        virtual.append(axial_add(coords[port.node_id], port.direction))
    if len(set(virtual)) != len(virtual) or set(virtual) & set(coords.values()):
        raise InverseInputError("conformer contains overlapping boundary sites")
    if original.local_rings:
        original_sizes = tuple(len(ring) for ring in original.local_rings)
        candidate_rings = _candidate_ring_coordinates(original, candidate)
        if tuple(tuple(tuple(point) for point in ring) for ring in candidate.local_rings) != candidate_rings:
            raise InverseInputError("conformer ring coordinates do not match node embedding")
        if tuple(len(ring) for ring in candidate_rings) != original_sizes:
            raise InverseInputError("conformer changed ring topology")
        edge_keys = {
            frozenset((int(edge.source_node_id), int(edge.target_node_id)))
            for edge in original.internal_edges
        }
        for ring in candidate_rings:
            if len(ring) < 3 or len(set(ring)) != len(ring):
                raise InverseInputError("conformer contains an invalid ring")
            for index, source in enumerate(ring):
                target = ring[(index + 1) % len(ring)]
                if not hex_edge_valid(source, target):
                    raise InverseInputError("conformer contains a non-hex ring edge")
        # Every consecutive ring pair must still be one of the prescribed
        # graph edges.  This checks geometry without adding or removing any
        # node/edge identity.
        sequences = _ring_node_sequences(original)
        for sequence in sequences:
            for index, left in enumerate(sequence):
                right = sequence[(index + 1) % len(sequence)]
                if frozenset((left, right)) not in edge_keys:
                    raise InverseInputError("conformer ring is not supported by internal edges")


def _port_direction_variants(unit: TopologyUnit, coords, *, preferred_first=True):
    occupied = set(coords.values())
    choices = []
    for port in unit.ports:
        node_coord = coords[int(port.node_id)]
        directions = list(HEX_DIRECTIONS)
        if preferred_first and port.direction in directions:
            directions.remove(port.direction)
            directions.insert(0, port.direction)
        choices.append([
            direction for direction in directions
            if axial_add(node_coord, direction) not in occupied
        ])
    result = []
    used = set()

    def solve(index, selected):
        if index >= len(unit.ports):
            result.append(tuple(selected))
            return
        port = unit.ports[index]
        node_coord = coords[int(port.node_id)]
        for direction in choices[index]:
            virtual = axial_add(node_coord, direction)
            if virtual in used:
                continue
            used.add(virtual)
            solve(index + 1, selected + [direction])
            used.remove(virtual)

    solve(0, [])
    return result


def _ring_conformers(unit: TopologyUnit, limit: int):
    """Enumerate bounded embeddings for generated ring graphs.

    The graph is authoritative: only prescribed edges are followed, and
    every accepted embedding is validated before it is returned.  The root
    coordinate is fixed to keep the search finite; global rotations and
    translations remain the responsibility of the placement search.
    """
    node_ids = [int(node.node_id) for node in unit.nodes]
    if not node_ids:
        return (unit,), True
    coords0 = {int(node.node_id): tuple(node.local_axial) for node in unit.nodes}
    root = node_ids[0]
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in unit.internal_edges:
        left, right = int(edge.source_node_id), int(edge.target_node_id)
        adjacency[left].add(right)
        adjacency[right].add(left)
    results = []
    seen = set()

    def append(coords):
        for directions in _port_direction_variants(unit, coords):
            candidate_rings = tuple(
                tuple(coords[int(node_id)] for node_id in sequence)
                for sequence in _ring_node_sequences(unit)
            )
            candidate = replace(
                unit,
                nodes=tuple(replace(node, local_axial=coords[int(node.node_id)])
                            for node in unit.nodes),
                ports=tuple(replace(port, direction=direction)
                            for port, direction in zip(unit.ports, directions)),
                local_rings=candidate_rings,
            )
            signature = geometry_signature(candidate)
            if signature in seen:
                continue
            try:
                validate_conformer(unit, candidate)
            except InverseInputError:
                continue
            seen.add(signature)
            results.append(candidate)
            if len(results) >= limit:
                return True
        return False

    # Preserve the source pose whenever it is legal.
    if append(coords0):
        return tuple(results[:limit]), False

    assigned = {root: coords0[root]}

    def backtrack():
        if len(results) >= limit:
            return
        unassigned = [node_id for node_id in node_ids if node_id not in assigned]
        if not unassigned:
            append(dict(assigned))
            return
        node_id = min(
            unassigned,
            key=lambda value: (-sum(neighbor in assigned for neighbor in adjacency[value]), value),
        )
        neighbors = [neighbor for neighbor in adjacency[node_id] if neighbor in assigned]
        if not neighbors:
            return
        candidates = None
        for neighbor in sorted(neighbors):
            local = [axial_add(assigned[neighbor], direction) for direction in HEX_DIRECTIONS]
            candidates = local if candidates is None else [value for value in candidates if value in set(local)]
        for value in candidates or ():
            if value in assigned.values():
                continue
            assigned[node_id] = value
            valid = True
            for other in adjacency[node_id] & assigned.keys():
                if other == node_id or not hex_edge_valid(value, assigned[other]):
                    valid = False
                    break
            if valid:
                backtrack()
            assigned.pop(node_id, None)

    backtrack()
    # If all embeddings were visited before reaching the limit the pool is
    # complete; otherwise the caller must treat it as bounded/truncated.
    return tuple(results[:limit]), len(results) < limit


# Variants are immutable, but each entry still retains every node coordinate in
# the bounded pool. Keep the cache small enough that repeated Flex retries do
# not become an unbounded second state tree.
@lru_cache(maxsize=64)
def topology_conformer_info(unit: TopologyUnit, limit: int = 12):
    limit = max(1, int(limit))
    generated = dict(unit.metadata).get("geometry_source") == "generated_from_connectivity"
    if not generated:
        return (unit,), True
    # A CSV flex_chain already contains its complete endpoint geometry.  The
    # two boundary ports are the two ends of one prescribed 11-23...-11 path;
    # treating them as independent tree leaves creates bent pseudo-chains
    # whose 13-13 displacement is not one of the legal hex-grid paths.  Global
    # rotation/reflection is handled by the placement search, so keep this
    # local geometry fixed.
    if (
        str(dict(unit.metadata).get("source_product_type", "")).lower()
        == "flex_chain"
        and len(unit.ports) == 2
    ):
        return (unit,), True
    if limit <= 1 or any(n.su_type == 25 for n in unit.nodes):
        return (unit,), False
    if unit.local_rings:
        return _ring_conformers(unit, limit)
    return _tree_topology_conformer_info(unit, limit)


def topology_conformers(unit: TopologyUnit, limit: int = 12):
    """Rotate connected tree branches, including virtual aromatic endpoints.

    No edges are added based on spatial adjacency. The original pose is first;
    each subsequent pose preserves the complete labelled graph. This is a
    bounded candidate generator, not an exhaustive proof of embeddability.
    """
    return topology_conformer_info(unit, limit)[0]


def _tree_topology_conformer_info(unit: TopologyUnit, limit: int = 12):
    """Legacy tree generator with an explicit completeness flag."""
    generated = dict(unit.metadata).get("geometry_source") == "generated_from_connectivity"
    if not generated:
        return (unit,), True
    if limit <= 1 or unit.local_rings or any(n.su_type == 25 for n in unit.nodes):
        return (unit,), False
    # Tagged keys keep virtual ports distinct from real (possibly negative) IDs.
    coords = {("node", n.node_id): n.local_axial for n in unit.nodes}
    adjacency = {key: set() for key in coords}
    for edge in unit.internal_edges:
        a, b = ("node", edge.source_node_id), ("node", edge.target_node_id)
        adjacency[a].add(b)
        adjacency[b].add(a)
    for p in unit.ports:
        if p.direction is None:
            return (unit,), False
        a, b = ("node", p.node_id), ("port", p.port_id)
        coords[b] = axial_add(coords[a], p.direction)
        adjacency[b] = {a}
        adjacency[a].add(b)
    if not coords or sum(map(len, adjacency.values())) != 2 * (len(coords) - 1):
        return (unit,), False
    root = ("node", unit.ports[0].node_id if unit.ports else unit.nodes[0].node_id)
    parent = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                order.append(neighbor)
    if len(order) != len(coords):
        return (unit,), False
    descendants = {node: {node} for node in order}
    for node in reversed(order[1:]):
        descendants[parent[node]].update(descendants[node])
    pivots = sorted(order[1:], key=lambda n: (-len(descendants[n]), n))
    results = [unit]
    seen = {geometry_signature(unit)}
    queue = deque([coords])
    # Each accepted conformer is immutable and can be replayed/audited later.
    while queue and len(results) < limit:
        base = queue.popleft()
        for rotation in (1, 5, 2, 4, 3):
            for child in pivots:
                pivot = base[parent[child]]
                trial = dict(base)
                for node in descendants[child]:
                    q, r = base[node]
                    trial[node] = axial_add(pivot, transform_axial(
                        (q - pivot[0], r - pivot[1]), rotation))
                if len(set(trial.values())) != len(trial):
                    continue
                candidate = replace(
                    unit,
                    nodes=tuple(replace(n, local_axial=trial[("node", n.node_id)])
                                for n in unit.nodes),
                    ports=tuple(replace(p, direction=(
                        trial[("port", p.port_id)][0] - trial[("node", p.node_id)][0],
                        trial[("port", p.port_id)][1] - trial[("node", p.node_id)][1],
                    )) for p in unit.ports),
                )
                signature = geometry_signature(candidate)
                if signature in seen:
                    continue
                try:
                    validate_conformer(unit, candidate)
                except InverseInputError:
                    continue
                seen.add(signature)
                results.append(candidate)
                queue.append(trial)
                if len(results) >= limit:
                    return tuple(results), False
    return tuple(results), True
