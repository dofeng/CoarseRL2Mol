"""Read-only aromatic template and attachment-site queries."""
from collections import Counter
from typing import Any, Dict, List, Set
from RL_MTCS.geometry import hex_edge_valid
from RL_MTCS.mdp.types import InverseAssemblyInput, TopologyUnit


def aromatic_template_errors(problem: InverseAssemblyInput) -> List[Dict[str, Any]]:
    """Validate ring membership independently of the current placement."""
    errors = []
    for unit in problem.units:
        if unit.phase != "aromatic":
            continue
        by_coord = {tuple(node.local_axial): node.node_id for node in unit.nodes}
        ring_coords = {tuple(p) for ring in unit.local_rings for p in ring}
        edges = {frozenset((e.source_node_id, e.target_node_id)) for e in unit.internal_edges}
        ring_edges = set()
        for ring in unit.local_rings:
            if len(ring) < 3 or len(set(ring)) != len(ring):
                errors.append(dict(unit_id=unit.unit_id, reason="invalid_aromatic_ring"))
                continue
            for index, point in enumerate(ring):
                other = ring[(index + 1) % len(ring)]
                if tuple(point) not in by_coord or tuple(other) not in by_coord:
                    errors.append(dict(unit_id=unit.unit_id, reason="aromatic_ring_site_missing"))
                    continue
                pair = frozenset((by_coord[tuple(point)], by_coord[tuple(other)]))
                ring_edges.add(pair)
                if pair not in edges or not hex_edge_valid(point, other):
                    errors.append(dict(unit_id=unit.unit_id, reason="invalid_aromatic_ring_edge",
                                       node_ids=sorted(pair)))
        outside = sorted(n.node_id for n in unit.nodes if tuple(n.local_axial) not in ring_coords)
        if outside:
            errors.append(dict(unit_id=unit.unit_id, reason="aromatic_nodes_outside_rings",
                               node_ids=outside, node_count=len(unit.nodes), ring_site_count=len(ring_coords)))
        if edges - ring_edges:
            errors.append(dict(unit_id=unit.unit_id, reason="aromatic_edges_outside_rings",
                               edges=sorted(sorted(pair) for pair in edges - ring_edges)))
    return errors


def aromatic_perimeter_ids(unit: TopologyUnit) -> Set[int]:
    ring_coords = {tuple(p) for ring in unit.local_rings for p in ring}
    degree = Counter(n for e in unit.internal_edges for n in (e.source_node_id, e.target_node_id))
    return {n.node_id for n in unit.nodes if n.su_type != 12
            and tuple(n.local_axial) in ring_coords and degree[n.node_id] == 2}


def hetero_staging_ids(state) -> Set[int]:
    return {n for request in state.problem.hetero_requests
            for n in (request.target_node_id, request.synthetic_node_id) if n is not None}
