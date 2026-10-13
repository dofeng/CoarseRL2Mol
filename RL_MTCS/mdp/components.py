"""Main component identity and post-flex resource scope."""
from RL_MTCS.geometry import axial_to_cartesian, convex_hull, unit_sort_key


def phase_of(unit):
    return 'rigid' if unit.phase == 'aromatic' else unit.phase


def anchor_for(state):
    """Read the anchored identity, or select a stable largest aromatic group."""
    if state.main_anchor in state.placements:
        return state.main_anchor
    groups = []
    seen = set()
    for uid in sorted(state.placements, key=unit_sort_key):
        if uid in seen or state.unit(uid).phase != 'aromatic':
            continue
        group = state.component_unit_ids(uid)
        seen.update(group)
        aromatic = [u for u in group if state.unit(u).phase == 'aromatic']
        groups.append((-len(aromatic), -sum(len(state.unit(u).nodes) for u in group),
                       unit_sort_key(uid), uid))
    return min(groups)[-1] if groups else None


def choose_anchor(state):
    state.main_anchor = anchor_for(state)
    return state.main_anchor


def main_units(state):
    anchor = anchor_for(state)
    return state.component_unit_ids(anchor) if anchor is not None else set()


def finish_flex(state):
    """Restrict downstream geometry only after the flex scheduler is exhausted."""
    current = state.clone()
    choose_anchor(current)
    keep = main_units(current)
    current.excluded_unit_ids.update(set(current.placements) - keep)
    current.placements = {u:p for u,p in current.placements.items() if u in keep}
    owner = current.problem.ownership_map()
    current.node_axial = {n:p for n,p in current.node_axial.items() if owner[n] in keep}
    current.final_node_positions = {n:p for n,p in current.final_node_positions.items() if n in current.node_axial}
    current.occupied_axial = {p:n for n,p in current.node_axial.items()}
    current.edges = [e for e in current.edges if e.source_node_id in current.node_axial and e.target_node_id in current.node_axial]
    current.satisfied_contracts.intersection_update(e.contract_id for e in current.edges if e.contract_id)
    current.used_ports = {f'{uid}:{pid}' for c in current.problem.contracts
        if c.contract_id in current.satisfied_contracts
        for uid, pid in ((c.source_unit_id,c.source_port_id),(c.target_unit_id,c.target_port_id))}
    current.backbone_hull = convex_hull([axial_to_cartesian(p) for p in current.node_axial.values()])
    current.stage = 'ring'
    return current


def stage_disconnected_components(state):
    """Keep unused clusters available, beyond the reachable active workspace.

    Staging translations are multiples of three, preserving honeycomb phase.
    They are bookkeeping poses, never removed resources or attachment bounds.
    """
    from dataclasses import replace
    active = main_units(state)
    active_nodes = {n.node_id for uid in active for n in state.unit(uid).nodes}
    active_points = [p for n,p in state.node_axial.items() if n in active_nodes]
    reach = 3 * (1 + sum(len(u.nodes) + max((abs(v) for n in u.nodes for v in n.local_axial),default=0)
                         for u in state.problem.units))
    base = 3 * ((max((p[0] for p in active_points),default=0)+reach)//3+1)
    seen = set(active)
    for uid in sorted(state.placements,key=unit_sort_key):
        if uid in seen:
            continue
        group = state.component_unit_ids(uid)
        seen.update(group)
        nodes = {n.node_id for u in group for n in state.unit(u).nodes}
        left = min(state.node_axial[n][0] for n in nodes)
        dq = 3 * ((base-left)//3+1)
        for n in nodes:
            q,r = state.node_axial[n]
            state.node_axial[n]=(q+dq,r)
            q,r = state.final_node_positions.get(n,(q,r))
            state.final_node_positions[n]=(q+dq,r)
        for u in group:
            p=state.placements[u]
            state.placements[u]=replace(p,translation=(p.translation[0]+dq,p.translation[1]))
        base=max(state.node_axial[n][0] for n in nodes)+reach
    state.occupied_axial={p:n for n,p in state.node_axial.items()}
