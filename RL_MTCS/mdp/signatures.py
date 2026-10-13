"""Search state equivalence, preserving policy-relevant orientation.

Keep unit identity, actual edges, occupied port roles and substitution requests.
Ignore free occupant GIDs and lattice-preserving parking translations, while
retaining world orientation and the last successful connection axes.
Only rigid states use this key; flex and later stages retain exact signatures.
"""
from functools import lru_cache

@lru_cache(maxsize=256)
def _canonical_component(nodes, edges):
    # Parking translations preserve lattice phase. World orientation now
    # controls H/V scheduling and must not disappear during deduplication.
    left = min(p[0] for p, _ in nodes)
    bottom = min(p[1] for p, _ in nodes)
    points = [(p[0]-left, p[1]-bottom) for p, _ in nodes]
    vertices = tuple(sorted((points[i], color) for i, (_, color) in enumerate(nodes)))
    bonds = tuple(sorted((tuple(sorted((points[a], points[b]))), kind)
                         for a, b, kind in edges))
    return ((left+bottom) % 3, vertices, bonds)


def rigid_geometry_signature(state):
    from RL_MTCS.mdp.sites import aromatic_perimeter_ids
    owner = state.problem.ownership_map()
    requests = {}
    for request in state.problem.hetero_requests:
        for role, node in (('target', request.target_node_id), ('synthetic', request.synthetic_node_id)):
            if node is not None:
                requests[node] = (role, request.target_node_id, request.target_su, request.ring_size,
                                  tuple(request.hop1_ids))
    used_roles = {}
    for uid in state.placements:
        unit = state.unit(uid)
        for port in unit.ports:
            used = f'{uid}:{port.port_id}' in state.used_ports or port.used
            if used or port.required_su_types:
                used_roles.setdefault(port.node_id, []).append(
                    (port.phase, port.role, used, tuple(port.required_su_types)))
    perimeter = {n for uid in state.placements for n in aromatic_perimeter_ids(state.unit(uid))}
    remaining = set(state.placements)
    components = []
    while remaining:
        group = state.component_unit_ids(min(remaining))
        remaining.difference_update(group)
        ids = sorted(n for n in state.node_axial if owner[n] in group)
        index = {n: i for i, n in enumerate(ids)}
        nodes = tuple((state.node_axial[n], (owner[n], state.current_su_types[n], n in perimeter,
                       requests.get(n, ()), tuple(sorted(used_roles.get(n, ()))))) for n in ids)
        edges = tuple(sorted((index[e.source_node_id], index[e.target_node_id], e.kind)
                             for e in state.edges if e.source_node_id in index and e.target_node_id in index))
        components.append(_canonical_component(nodes, edges))
    # Contract IDs identify consumed tasks, even when the occupied sites are
    # symmetric. Request colors above preserve pinned substitution identities.
    return ('rigid_geometry', state.main_anchor, state.last_rigid_axis, state.last_flex_axis,
            tuple(sorted(state.excluded_unit_ids)),
            tuple(sorted(components)), tuple(sorted(state.satisfied_contracts)))


def search_signature(state, phase):
    # Non-molecular adapters used by integrations keep their own exact keys.
    if phase == 'rigid' and hasattr(state, 'problem'):
        return rigid_geometry_signature(state)
    return state.state_signature()
