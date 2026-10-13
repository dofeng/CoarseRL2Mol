"""Lazy honeycomb conformers preserving input topology."""

from collections import deque
from dataclasses import replace

from RL_MTCS.geometry import (HEX_DIRECTIONS, axial_add, hex_edge_valid,
    topology_site_valid, topology_edge_valid, su25_center_branch_errors,
    transform_axial)
from RL_MTCS.mdp.types import InverseInputError, TopologyUnit
from RL_MTCS.reporting.diagnostics import note_rejection


def iter_conformer_states(state, uid, feasible=None):
    from RL_MTCS.actions.conformers import iter_topology_conformers
    for variant in iter_topology_conformers(state.conformer_sources.get(uid,state.unit(uid))):
        if feasible is not None and not feasible(variant):
            note_rejection('mandatory_attachment_geometry', uid)
            continue
        trial = state.clone()
        trial.select_conformer(variant)
        yield trial


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
    branch_errors = su25_center_branch_errors(candidate)
    if branch_errors:
        raise InverseInputError(str(branch_errors))
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
    if original.attaches_to_aromatic_edge:
        coords = {n.node_id: n.local_axial for n in candidate.nodes}
        anchors = [axial_add(coords[p.node_id], p.direction) for p in candidate.ports]
        if len(anchors) != 2 or not hex_edge_valid(*anchors):
            raise InverseInputError("ring conformer requires adjacent aromatic anchors")
    generated = dict(original.metadata).get("geometry_source") == "generated_from_connectivity"
    if not generated and geometry_signature(original) != geometry_signature(candidate):
        raise InverseInputError("cannot change an explicitly fixed geometry")
    coords = {n.node_id: n.local_axial for n in candidate.nodes}
    if len(set(coords.values())) != len(coords):
        raise InverseInputError("conformer contains overlapping nodes")
    for edge in candidate.internal_edges:
        a, b = coords[edge.source_node_id], coords[edge.target_node_id]
        if not hex_edge_valid(a, b):
            raise InverseInputError("conformer contains a non-hex edge")
    # Local coordinates are normalized to a node at (0,0). Account for the
    # removed lattice origin instead of testing normalized phase directly.
    if not _embedding_phases(candidate, coords, [p.direction for p in candidate.ports]):
        raise InverseInputError("conformer violates ordinary honeycomb directions")
    from RL_MTCS.validation import _segment_geometry_errors
    if _segment_geometry_errors(coords, candidate.internal_edges):
        raise InverseInputError("conformer contains intersecting bonds or a node on a bond")
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


def _port_direction_variants(unit: TopologyUnit, coords, *, preferred_first=True, fixed=None):
    if unit.attaches_to_aromatic_edge:
        yield tuple(port.direction for port in unit.ports)
        return
    occupied = set(coords.values())
    choices = []
    for port in unit.ports:
        node_coord = coords[int(port.node_id)]
        directions = ([fixed[port.port_id]] if fixed and port.port_id in fixed
                      else list(HEX_DIRECTIONS))
        if preferred_first and port.direction in directions:
            directions.remove(port.direction)
            directions.insert(0, port.direction)
        choices.append([
            direction for direction in directions
            if axial_add(node_coord, direction) not in occupied
        ])
    used = set()

    def solve(index, selected):
        if index >= len(unit.ports):
            if _embedding_phases(unit, coords, selected):
                yield tuple(selected)
            return
        port = unit.ports[index]
        node_coord = coords[int(port.node_id)]
        for direction in choices[index]:
            virtual = axial_add(node_coord, direction)
            if virtual in used:
                continue
            used.add(virtual)
            yield from solve(index + 1, selected + [direction])
            used.remove(virtual)

    yield from solve(0, [])


def _embedding_phases(unit, coords, directions=()):
    """Necessary honeycomb constraint on a partial embedding; no scan budget.

    Local coordinates may be normalized to a face centre. All ordinary edges
    and external port stubs must admit ONE common global lattice phase.
    """
    edges = [(e.source_node_id,e.target_node_id) for e in unit.internal_edges
             if e.source_node_id in coords and e.target_node_id in coords]
    ports = [(p.node_id,d) for p,d in zip(unit.ports,directions)
             if p.node_id in coords and d is not None]
    center_parents = {child: parent for parent, child in unit.su25_center_branches}
    return tuple(phase for phase in range(3) if
                 all(topology_site_valid(n, axial_add(p, (phase, 0)), center_parents)
                     for n, p in coords.items()) and
                 all(topology_edge_valid(a, b, axial_add(coords[a],(phase,0)),
                                    axial_add(coords[b],(phase,0)), center_parents)
                     for a,b in edges) and
                 all(hex_edge_valid(axial_add(coords[n],(phase,0)),
                                    axial_add(axial_add(coords[n],d),(phase,0)),honeycomb=True)
                     for n,d in ports))


def _iter_embeddings(unit, coords, *, fixed_directions=None, point_allowed=None,
                     preferred_coordinates=None):
    nodes = [n.node_id for n in unit.nodes]
    adjacency = {n: set() for n in nodes}
    for edge in unit.internal_edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)
    partial_directions = tuple((fixed_directions or {}).get(p.port_id) for p in unit.ports)

    def points(node):
        neighbors = sorted(adjacency[node] & coords.keys())
        directions = list(HEX_DIRECTIONS)
        if preferred_coordinates is not None:
            a, b = preferred_coordinates[neighbors[0]], preferred_coordinates[node]
            preferred = (b[0]-a[0], b[1]-a[1])
            if preferred in directions:
                directions.remove(preferred)
                directions.insert(0, preferred)
        for direction in directions:
            point = axial_add(coords[neighbors[0]], direction)
            if point in coords.values() or any(not hex_edge_valid(point, coords[n]) for n in neighbors):
                continue
            if point_allowed is not None and not point_allowed(node, point, coords, adjacency):
                continue
            coords[node] = point
            feasible = _embedding_phases(unit, coords, partial_directions)
            del coords[node]
            if feasible:
                yield point

    def visit():
        remaining = [n for n in nodes if n not in coords]
        if not remaining:
            for directions in _port_direction_variants(unit, coords, fixed=fixed_directions):
                candidate = replace(unit,
                    nodes=tuple(replace(n, local_axial=coords[n.node_id]) for n in unit.nodes),
                    ports=tuple(replace(p, direction=d) for p,d in zip(unit.ports, directions)),
                    local_rings=tuple(tuple(coords[n] for n in ring) for ring in _ring_node_sequences(unit)))
                try:
                    validate_conformer(unit, candidate)
                except InverseInputError:
                    continue
                yield candidate
            return
        frontier = [n for n in remaining if adjacency[n] & coords.keys()]
        if not frontier:
            raise InverseInputError("topology unit has disconnected internal graph")
        if point_allowed is None:
            node = min(frontier, key=lambda n: (-len(adjacency[n] & coords.keys()), n))
            choices = points(node)
        else:
            # Resolve the most constrained branch first, using current space.
            domains = [(n, tuple(points(n))) for n in frontier]
            if any(not domain for _, domain in domains):
                return
            node, choices = min(domains, key=lambda row: (
                len(row[1]), -len(adjacency[row[0]] & coords.keys()),
                -len(adjacency[row[0]]), row[0]))
        for point in choices:
            coords[node] = point
            yield from visit()
            del coords[node]
    yield from visit()


def iter_attached_conformers(state, unit, port, position, site, *, blocked_units=None):
    """Embed a single-ended topology in actual free space at a fixed entry.

    Generated conformers use world directions and a local root at (0,0).
    Fixed input geometry keeps its coordinates and uses rigid poses only.
    The caller still submits through the complete shared placement/audit path.
    """
    unavailable = dict(unit.metadata).get('assembly_unavailable_reason')
    if unavailable:
        note_rejection(unavailable, unit.unit_id)
        return
    if su25_center_branch_errors(unit):
        note_rejection('invalid_su25_center_branch', unit.unit_id)
        return
    toward_site = (state.node_axial[site][0]-position[0],
                   state.node_axial[site][1]-position[1])
    if not state._edge_geometry_valid(port.node_id, site, position, state.node_axial[site]):
        note_rejection('external_endpoint_phase_mismatch', unit.unit_id)
        return
    if dict(unit.metadata).get('geometry_source') != 'generated_from_connectivity':
        try:
            validate_conformer(unit, unit)
        except InverseInputError:
            return
        root = unit.node_map()[port.node_id].local_axial
        for reflection in (False, True):
            for rotation in range(6):
                if transform_axial(port.direction, rotation, reflection) != toward_site:
                    continue
                local = transform_axial(root, rotation, reflection)
                translation = (position[0]-local[0], position[1]-local[1])
                yield unit, rotation, translation, reflection
        return

    def allowed(node, point, coords, adjacency):
        world = axial_add(position, point)
        if not state._node_site_valid(node, world) or world in state.occupied_axial:
            if blocked_units is not None and world in state.occupied_axial:
                blocked_units.update(state._blocking_units({node:world}))
            note_rejection('occupied_conformer_site', unit.unit_id)
            return False
        for other, local in coords.items():
            other_world = axial_add(position, local)
            if other in adjacency[node]:
                if not state._edge_geometry_valid(node, other, world, other_world):
                    return False
        positions = {n: axial_add(position, p) for n, p in coords.items()}
        positions[node] = world
        edges = [e for e in unit.internal_edges
                 if e.source_node_id in positions and e.target_node_id in positions]
        errors = state._proposed_geometry_errors(positions, edges, (frozenset((port.node_id, site)),))
        if errors:
            if blocked_units is not None:
                blocked_units.update(state._blocking_units(positions, errors))
            note_rejection('conformer_edge_obstruction', unit.unit_id)
            return False
        return True

    if not allowed(port.node_id, (0, 0), {}, {}):
        return
    fixed = {port.port_id: toward_site}
    rotation = next(r for r in range(6) if transform_axial(port.direction, r) == toward_site)
    preferred = {n.node_id: transform_axial(n.local_axial, rotation) for n in unit.nodes}
    for variant in _iter_embeddings(unit, {port.node_id: (0, 0)},
                                   fixed_directions=fixed, point_allowed=allowed,
                                   preferred_coordinates=preferred):
        yield variant, 0, position, False


def iter_topology_conformers(unit):
    """Lazily enumerate finite connected embeddings; only legal placements cap use."""
    if dict(unit.metadata).get('assembly_unavailable_reason'):
        note_rejection(dict(unit.metadata)['assembly_unavailable_reason'], unit.unit_id)
        return
    if su25_center_branch_errors(unit):
        note_rejection('invalid_su25_center_branch', unit.unit_id)
        return
    try:
        validate_conformer(unit, unit)
    except InverseInputError:
        pass
    else:
        yield unit
    generated = dict(unit.metadata).get("geometry_source") == "generated_from_connectivity"
    if not generated:
        return
    nodes = [n.node_id for n in unit.nodes]
    if not nodes:
        return
    adjacency = {n: set() for n in nodes}
    for edge in unit.internal_edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)
    coords0 = {n.node_id: n.local_axial for n in unit.nodes}
    core = {nodes[0]}
    if unit.attaches_to_aromatic_edge:
        start, end = (p.node_id for p in unit.ports)
        parents, queue = {start: None}, deque([start])
        while queue and end not in parents:
            node = queue.popleft()
            for other in sorted(adjacency[node]):
                if other not in parents:
                    parents[other] = node
                    queue.append(other)
        node = end
        while node is not None:
            core.add(node)
            node = parents[node]
        core.update(n for ring in _ring_node_sequences(unit) for n in ring)
    original_key = geometry_signature(unit)
    for candidate in _iter_embeddings(unit, {n: coords0[n] for n in core}):
        if geometry_signature(candidate) != original_key:
            yield candidate
