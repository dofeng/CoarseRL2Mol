"""Bounded PUCT over SU10 bindings, direct ring bonds and component poses."""

from collections import Counter, defaultdict
from itertools import combinations, zip_longest

from .geometry import aromatic_slot_direction, transform_axial, axial_add, unit_sort_key, coordinate_aspect
from .models import InverseInputError, UnitPort
from .search import PersistentMCTS, SearchBudget, SearchObservation
from .state import aromatic_perimeter_ids, aromatic_template_errors, relabel_aromatic_sites


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


def search_dynamic_rigid(stage, *, candidate_k, beam_width, max_steps,
                         iterations, c_puct, seed, search_guidance=None, max_tree_nodes=128):
    from .rigid import (TopologyPlacementStage, SearchResult, _rigid_unit_components,
                        _rigid_component_metrics, _pack_rigid_components)

    initial = stage.state.clone()
    errors = aromatic_template_errors(initial.problem)
    if errors:
        raise InverseInputError('invalid aromatic template', {'errors': errors})
    ports = sorted(n for n, su in initial.problem.node_su_types if su == 10)
    target = len(ports) // 2
    aromatic_count = sum(u.phase == 'aromatic' for u in initial.problem.units)
    if len(ports) % 2 or target >= aromatic_count:
        raise InverseInputError('SU10 count cannot form the requested aromatic forest')
    width = min(32, max(2, candidate_k, beam_width))
    ideal = aromatic_count / max(1, aromatic_count - target)
    best = initial
    diagnostics = {'pose_checks': 0, 'candidate_truncations': 0, 'binding_rejections': 0}

    def degree(state):
        return Counter(u for c in state.problem.contracts
                       if c.kind == 'rigid' and c.contract_id in state.satisfied_contracts
                       for u in (c.source_unit_id, c.target_unit_id))

    def rank(state):
        groups = _rigid_unit_components(state)
        sizes = [len(g) for g, _ in groups]
        degrees = degree(state)
        free = free_ring_sites(state)
        # Empty outward neighbours are a cheap lookahead, not an infeasibility proof.
        accessible = 0
        for uid, ids in free.items():
            placement = state.placements[uid]
            for gid in ids:
                direction = transform_axial(aromatic_slot_direction(state.unit(uid), gid),
                                            placement.rotation, placement.reflection)
                accessible += axial_add(state.node_axial[gid], direction) not in state.occupied_axial
        shape_cost = 0.0
        for group, _ in groups:
            # Separate components are staged far apart; their artificial gaps
            # must never influence shape preference. PCA is rotation invariant.
            shape = coordinate_aspect(state.node_axial[n.node_id]
                                     for uid in group for n in state.unit(uid).nodes)
            if len(group) > 2:
                shape_cost += max(0.0, shape['principal_axis_ratio'] - 2.5) ** 2
        return (sum(degrees.values()) // 2,
                -sum(max(0, size - ideal) ** 2 for size in sizes),
                -sum(max(0, d - 4) ** 2 for d in degrees.values()),
                sum(3 <= d <= 4 for d in degrees.values()), accessible, -shape_cost)

    def observe(state):
        nonlocal best
        if rank(state) > rank(best):
            best = state

    def expand(state):
        unused = [n for n in ports if n not in _used_endpoints(state)]
        if len(unused) < 2 or len(state.satisfied_contracts) >= max_steps:
            return
        # SU10 identities have no distinct chemical constraints. Canonical IDs
        # remove factorial permutations while all physical slot choices remain.
        left_gid, right_gid = unused[:2]
        sites = free_ring_sites(state)
        components = _rigid_unit_components(state)
        membership = {u: index for index, (group, _) in enumerate(components) for u in group}
        loads = degree(state)
        pairs = [(a, b) for a, b in combinations(sorted(sites, key=unit_sort_key), 2)
                 if sites[a] and sites[b] and membership[a] != membership[b]]
        def pair_rank(pair):
            a, b = pair
            merged = len(components[membership[a]][0]) + len(components[membership[b]][0])
            branch = int(loads[a] == 2) + int(loads[b] == 2)
            excess = sum(max(0, loads[u] + 1 - 4) ** 2 for u in pair)
            return max(0, merged - ideal) ** 2, excess, -branch, len(sites[a]) * len(sites[b]), pair
        pairs.sort(key=pair_rank)
        # Round-robin slot pairs across unit pairs and outward direction groups.
        streams = []
        for a, b in pairs:
            grouped = defaultdict(list)
            for sa in sites[a]:
                da = aromatic_slot_direction(state.unit(a), sa)
                for sb in sites[b]:
                    db = aromatic_slot_direction(state.unit(b), sb)
                    grouped[(da, db)].append((a, b, sa, sb))
            streams.append(iter(row for layer in zip_longest(*grouped.values())
                                for row in layer if row is not None))
        checks = 0
        emitted = 0
        for layer in zip_longest(*streams):
            for pair in layer:
                if pair is None:
                    continue
                a, b, sa, sb = pair
                checks += 1
                if checks > max(128, width * 64):
                    diagnostics['candidate_truncations'] += 1
                    return
                pa = UnitPort(f'probe:{sa}', sa, 'rigid', 'rigid', aromatic_slot_direction(state.unit(a), sa))
                pb = UnitPort(f'probe:{sb}', sb, 'rigid', 'rigid', aromatic_slot_direction(state.unit(b), sb))
                actions = TopologyPlacementStage(state, 'rigid')._candidate_for_pair(
                    state.unit(a), pa, state.unit(b), pb)
                diagnostics['pose_checks'] += 1
                if not actions:
                    continue
                try:
                    bound = bind_su10_sites(state, {left_gid: sa, right_gid: sb})
                except InverseInputError:
                    diagnostics['binding_rejections'] += 1
                    continue
                owners = bound.problem.ownership_map()
                units = [bound.unit(owners[n]) for n in (left_gid, right_gid)]
                bound_ports = [next(p for p in u.ports if p.phase == 'rigid' and p.node_id == n)
                               for u, n in zip(units, (left_gid, right_gid))]
                actions = TopologyPlacementStage(bound, 'rigid')._candidate_for_pair(
                    units[0], bound_ports[0], units[1], bound_ports[1])
                for action in actions:
                    action['contract_id'] = f'rigid:dynamic:{left_gid}-{right_gid}'
                    action['site_bindings'] = {
                        str(left_gid): [a, next(i for i, n in enumerate(state.unit(a).nodes) if n.node_id == sa)],
                        str(right_gid): [b, next(i for i, n in enumerate(state.unit(b).nodes) if n.node_id == sb)],
                    }
                    child = bound.clone()
                    ok, _ = child.connect(action)
                    if ok:
                        yield child
                        emitted += 1
                    if emitted >= width * 4:
                        diagnostics['candidate_truncations'] += 1
                        return

    def evaluate(state):
        r = rank(state)
        return min(0.99, 0.85 * r[0] / max(1, target)
                   + 0.04 / (1 - r[1]) + 0.03 * min(1, r[3] / 3)
                   + 0.04 * min(1, r[4] / max(1, aromatic_count * 3))
                   + 0.04 / (1 - r[5]))

    tree = PersistentMCTS(
        initial, expand=expand, accept=lambda s: s.phase_done('rigid'),
        signature=lambda s: (s.state_signature(), tuple(s.problem.ownership)),
        evaluate=evaluate, observe=observe, action=lambda p, c: c.action_history[-1],
        guidance=(search_guidance.adapted(lambda s: SearchObservation('rigid', s))
                  if search_guidance else None),
        max_nodes=max_tree_nodes, c_puct=c_puct, seed=seed,
    )
    budget = SearchBudget(min(20000, max(1, iterations) * max(1, target) * 16))
    # Seed one depth-first feasibility rollout. The path is inserted into the
    # persistent tree with every parent, so PUCT can still replace early bonds.
    probe_limit = min(max_tree_nodes - 1, max(1, target * 4), budget.limit // 3)
    probe_visits = 0
    probe_path = []
    def probe(state, path):
        nonlocal probe_visits, probe_path
        if rank(state) > rank(probe_path[-1] if probe_path else initial):
            probe_path = list(path)
        if state.phase_done('rigid'):
            return True
        if probe_visits >= probe_limit:
            return False
        children = expand(state)
        try:
            for child in children:
                probe_visits += 1
                if probe(child, [*path, child]):
                    return True
                if probe_visits >= probe_limit:
                    break
        finally:
            children.close()
        return False
    probe(initial, [])
    budget.simulations += probe_visits
    budget.visited += probe_visits
    tree.seed_path(probe_path)
    tree.run(budget, solution_limit=width)
    tree.record()
    retained = sorted(tree.solutions or [best], key=rank, reverse=True)[:width]
    for state in retained:
        _pack_rigid_components(state, _rigid_unit_components(state))
    current = retained[0]
    row = TopologyPlacementStage(current, 'rigid').get_result()
    row.update(status='complete' if row['done'] else 'budget_exhausted' if budget.exhausted
               else 'candidate_space_exhausted', failure_scope='none' if row['done'] else 'rigid_search',
               candidate_space_complete=False, completed_contract_count=row['contracts_done'],
               pairing_mode='dynamic_su10_sites', search_algorithm='persistent_puct_with_bounded_probe',
               component_sizes=sorted((len(g) for g, _ in _rigid_unit_components(current)), reverse=True),
               rigid_component_metrics=_rigid_component_metrics(current, _rigid_unit_components(current)),
               retained_candidate_count=len(retained), pairing_search=tree.diagnostics(),
               search_diagnostics=diagnostics, aromatic_neighbor_degrees=dict(degree(current)))
    row['search_diagnostics']['probe_visits'] = probe_visits
    # Release lazy candidate streams even when success stops the tree early.
    tree._close_subtree(tree.root)
    return SearchResult(current, evaluate(current), list(current.action_history),
                        {'phase': 'rigid', 'result': row}, retained)
