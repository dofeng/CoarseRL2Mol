"""Rigid SU10 action enumeration and site binding."""

from collections import defaultdict
from itertools import combinations

from RL_MTCS.geometry import aromatic_slot_direction, unit_sort_key
from RL_MTCS.mdp.types import InverseInputError, UnitPort
from RL_MTCS.mdp.sites import aromatic_perimeter_ids
from RL_MTCS.mdp.transition import relabel_aromatic_sites
from RL_MTCS.actions.ordering import ordered, round_robin
from RL_MTCS.geometry import connection_axis


def _used_endpoints(state):
    return {n for c in state.problem.contracts if c.contract_id in state.satisfied_contracts
            for n in (c.source_node_id, c.target_node_id)}


def _protected(state):
    return _used_endpoints(state) | {
        n for request in state.problem.hetero_requests
        for n in (request.target_node_id, request.synthetic_node_id) if n is not None
    }


def free_ring_sites(state):
    protected = _protected(state)
    return {
        u.unit_id: sorted(aromatic_perimeter_ids(u) - protected)
        for u in state.problem.units if u.phase == 'aromatic'
    }


def bind_su10_sites(state, assignments):
    """Apply a bijective occupant permutation; template slots never move."""
    planned = state.problem.node_map()
    if any(planned.get(gid) != 10 for gid in assignments):
        raise InverseInputError('rigid binding requires logical SU10 nodes')
    if set(assignments) & _protected(state):
        raise InverseInputError('rigid binding cannot reuse a committed endpoint')
    return relabel_aromatic_sites(
        state, assignments, state.action_history,
        protected_aromatic_ids=_protected(state), replay_contract_kinds=('rigid',), stage='rigid',
    )


def iter_rigid_outcomes(state, *, axis=None, seed=None):
    from RL_MTCS.actions.rigid_geometry import RigidPoseGenerator, _rigid_unit_components
    ports = sorted(n for n, su in state.problem.node_su_types if su == 10)
    unused = [n for n in ports if n not in _used_endpoints(state)]
    if len(unused) < 2:
        return
    # SU10 identities have no distinct chemical constraints. Canonical IDs
    # remove factorial permutations while all physical slot choices remain.
    left_gid, right_gid = unused[:2]
    sites = free_ring_sites(state)
    components = _rigid_unit_components(state)
    membership = {u: index for index, (group, _) in enumerate(components) for u in group}
    pairs = [(a, b) for a, b in combinations(sorted(sites, key=unit_sort_key), 2)
             if sites[a] and sites[b] and membership[a] != membership[b]]
    pairs = ordered(pairs, seed, 'rigid-pairs')
    # Fair enumeration avoids always extending the lexically first
    # aromatic unit. Only child geometry, never pair identity, is scored.
    # Round-robin slot pairs across unit pairs and outward direction groups.
    site_groups = {}
    for uid, ids in sites.items():
        grouped = defaultdict(list)
        for site in ids:
            grouped[aromatic_slot_direction(state.unit(uid), site)].append(site)
        site_groups[uid] = grouped

    def slot_pairs(a, b):
        # Preserve direction round-robin order without materializing sa*sb.
        groups = [(left, right) for left in site_groups[a].values()
                  for right in site_groups[b].values()]
        groups = ordered(groups, seed, f'rigid-slots:{a}:{b}')
        for index in range(max((len(left)*len(right) for left,right in groups), default=0)):
            for left, right in groups:
                if index < len(left)*len(right):
                    yield a, b, left[index//len(right)], right[index%len(right)]

    def outcomes(a, b):
        for pair in slot_pairs(a, b):
            a, b, sa, sb = pair
            pa = UnitPort(f'probe:{sa}', sa, 'rigid', 'rigid', aromatic_slot_direction(state.unit(a), sa))
            pb = UnitPort(f'probe:{sb}', sb, 'rigid', 'rigid', aromatic_slot_direction(state.unit(b), sb))
            actions = RigidPoseGenerator(state)._candidate_for_pair(
                state.unit(a), pa, state.unit(b), pb)
            if not actions:
                continue
            bound = bind_su10_sites(state, {left_gid: sa, right_gid: sb})
            owners = bound.problem.ownership_map()
            units = [bound.unit(owners[n]) for n in (left_gid, right_gid)]
            bound_ports = [next(p for p in u.ports if p.phase == 'rigid' and p.node_id == n)
                           for u, n in zip(units, (left_gid, right_gid))]
            actions = RigidPoseGenerator(bound)._candidate_for_pair(
                units[0], bound_ports[0], units[1], bound_ports[1])
            for action in actions:
                stationary = (action['target_unit_id'] if action.get('move_component_unit_id')
                              == action['source_unit_id'] else action['source_unit_id'])
                node = (action['target_node_id'] if stationary == action['target_unit_id']
                        else action['source_node_id'])
                port = next(p for p in bound.unit(stationary).ports if p.node_id == node and p.phase == 'rigid')
                direction = bound.global_port_direction(stationary, port)
                if axis is not None and connection_axis(direction) != axis:
                    continue
                action['contract_id'] = f'rigid:dynamic:{left_gid}-{right_gid}'
                action['site_bindings'] = {
                    str(left_gid): [a, next(i for i, n in enumerate(state.unit(a).nodes) if n.node_id == sa)],
                    str(right_gid): [b, next(i for i, n in enumerate(state.unit(b).nodes) if n.node_id == sb)],
                }
                child = bound.clone()
                ok, _ = child.connect(action)
                if ok:
                    yield child
    yield from round_robin(outcomes(a, b) for a, b in pairs)
