"""Occupied-space branch embedding and diversity within the legal action cap."""
from dataclasses import replace

import pytest

from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.actions.conformers import iter_attached_conformers, iter_topology_conformers
from RL_MTCS.actions.side import iter_side_outcomes, _side_pose_key
from RL_MTCS.geometry import axial_add, aromatic_slot_direction, convex_hull, axial_to_cartesian, transform_axial
from RL_MTCS.io.input import _make_cluster_template
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import TopologyUnit, UnitNode, UnitPort, UnitEdge, ExternalContract, InverseAssemblyInput


def branch_unit(uid='branch', first=100):
    return TopologyUnit(uid, 'side_chain', tuple(
        UnitNode(first+i, su, point, metadata=(('target_hop1_degree', str(degree)),))
        for i, (su, point, degree) in enumerate((
            (24, (0, 0), 3), (23, (0, 1), 2), (22, (1, 2), 1),
            (23, (1, 0), 2), (22, (1, -1), 1)))),
        tuple(UnitEdge(first+a, first+b) for a, b in ((0, 1), (1, 2), (0, 3), (3, 4))),
        (UnitPort('end', first, 'topology_boundary', 'side', (-1, -1)),),
        metadata=(('geometry_source', 'generated_from_connectivity'),))


def aromatic_unit(uid, kind='benzene', first=0):
    template = _make_cluster_template(kind)
    return TopologyUnit(uid, kind, tuple(
        UnitNode(first+i, 12 if shared else 13, tuple(point))
        for i, (point, shared) in enumerate(zip(template.coords, template.shared))),
        tuple(UnitEdge(first+a, first+b, 'aromatic') for a, b in template.edges),
        local_rings=tuple(template.rings))


def side_space_fixture(*, obstacle=False, side_count=1):
    aromatic = aromatic_unit('aromatic:0', 'chrysene' if side_count > 1 else 'benzene')
    available = [n.node_id for n in aromatic.nodes if n.su_type == 13]
    if side_count == 1:
        available = [next(n.node_id for n in aromatic.nodes if n.local_axial == (1, 1))]
    sides = tuple(branch_unit(f'branch:{i}', 100+i*10) for i in range(side_count))
    contracts = tuple(ExternalContract(f'side:{i}', u.unit_id, u.nodes[0].node_id,
        aromatic.unit_id, available[i], 'side', source_port_id='end', target_port_id=f'p:{i}')
        for i, u in enumerate(sides))
    aromatic = replace(aromatic, ports=tuple(UnitPort(c.target_port_id, c.target_node_id,
        'aromatic_boundary', 'side', aromatic_slot_direction(aromatic, c.target_node_id))
        for c in contracts))
    units = (aromatic, *sides)
    if obstacle:
        units += (aromatic_unit('aromatic:1', first=20),)
    plan = {n.node_id: n.su_type for u in units for n in u.nodes}
    plan.update((c.target_node_id, 11) for c in contracts)
    state = HexAssemblyState(InverseAssemblyInput(units, contracts, tuple(plan.items()),
        tuple((n.node_id, u.unit_id) for u in units for n in u.nodes)))
    assert state.place_unit(aromatic.unit_id, 0, (-1, -2) if side_count == 1 else (0, 0))
    if obstacle:
        assert state.place_unit('aromatic:1', 0, (3, 3))
    state.main_anchor = aromatic.unit_id
    state.backbone_hull = convex_hull([axial_to_cartesian(p) for p in state.node_axial.values()])
    state.stage = 'side'
    assert state.audit(diagnostics=False)['legal']
    return state


def attached_coordinates(state, unit, position, site):
    for variant, rotation, translation, reflection in iter_attached_conformers(
            state, unit, unit.ports[0], position, site):
        coords = state._placement_coordinates(variant, rotation, translation, reflection)
        yield variant, rotation, translation, reflection, coords


def test_terminal_bends_left_while_root_and_other_branch_stay_fixed():
    state = side_space_fixture(obstacle=True)
    before = state.state_signature()
    unit = state.unit('branch:0')
    site = state.problem.contracts[0].target_node_id
    blocked = state.can_place(unit.unit_id, 0, (1, 0), attachment_edge=(100, site))
    assert not blocked[0] and blocked[1].startswith('axial_collision:')
    expected = {100: (1, 0), 101: (1, 1), 102: (0, 1), 103: (2, 0), 104: (2, -1)}
    outcomes = list(attached_coordinates(state, unit, (1, 0), site))
    assert any(row[-1] == expected for row in outcomes)
    for variant, rotation, translation, reflection, coords in outcomes:
        trial = state.clone()
        trial.select_conformer(variant)
        assert trial.can_place(unit.unit_id, rotation, translation,
            attachment_edge=(100, site), reflection=reflection)[0]
        assert coords[100] == (1, 0)
    committed = next(child for child in iter_side_outcomes(state, unit)
                     if {n: child.node_axial[n] for n in expected} == expected)
    assert committed.audit(diagnostics=False)['legal']
    assert 'side:0' in committed.satisfied_contracts
    assert state.state_signature() == before


def test_space_pruning_preserves_every_legal_embedding_at_the_entry():
    for obstacle in (False, True):
        state = side_space_fixture(obstacle=obstacle)
        unit = state.unit('branch:0')
        site = state.problem.contracts[0].target_node_id
        expected = set()
        for variant in iter_topology_conformers(unit):
            trial = state.clone()
            trial.select_conformer(variant)
            for reflection in (False, True):
                for rotation in range(6):
                    if transform_axial(variant.ports[0].direction, rotation, reflection) != (-1, -1):
                        continue
                    root = transform_axial(variant.node_map()[100].local_axial, rotation, reflection)
                    translation = (1-root[0], -root[1])
                    if trial.can_place(unit.unit_id, rotation, translation,
                            attachment_edge=(100, site), reflection=reflection)[0]:
                        expected.add(tuple(sorted(trial._placement_coordinates(
                            variant, rotation, translation, reflection).items())))
        actual = {tuple(sorted(row[-1].items()))
                  for row in attached_coordinates(state, unit, (1, 0), site)}
        assert actual == expected and actual


def test_ten_candidates_include_different_bends_for_multiple_topologies():
    state = side_space_fixture(side_count=12)
    before = state.state_signature()
    outcomes = list(connection_states(state, 'side', 10))
    assert len(outcomes) == 10
    ids = [child.action_history[-1]['topology_unit_id'] for child in outcomes]
    assert len(set(ids)) == 10
    for uid, child in zip(ids, outcomes):
        assert child.audit(diagnostics=False)['legal']
        assert child.source_problem is state.source_problem
        assert child.unit(uid).internal_edges == state.unit(uid).internal_edges
    assert state.state_signature() == before


def test_equivalent_geometry_does_not_replace_different_entries():
    state = side_space_fixture()
    uid = 'branch:0'
    outcomes = list(connection_states(state, 'side', 10))
    keys = [_side_pose_key(child.unit(uid), child.node_axial) for child in outcomes]
    assert len(keys) == len(set(keys))
    roots = [child.node_axial[100] for child in outcomes]
    assert roots[0] != roots[1]
    assert len(set(roots)) > 1
    assert len(set(roots)) < len(roots)
    assert list(connection_states(state, 'side', 10, seed=7))[0].state_signature() == list(
        connection_states(state, 'side', 10, seed=7))[0].state_signature()


def test_invalid_entries_do_not_consume_the_ten_legal_candidate_slots(monkeypatch):
    import RL_MTCS.actions.flex as flex
    state = side_space_fixture()
    slots = flex._component_slot_records(state, ['aromatic:0'], ())
    site = state.problem.contracts[0].target_node_id
    valid = next(row for row in slots if row[1] == site)
    point = state.node_axial[site]
    # This wrong outward direction points into an occupied aromatic neighbor.
    blocked = ('aromatic:0', site, point, (-1, 0))
    assert axial_add(point, blocked[3]) in state.occupied_axial
    monkeypatch.setattr(flex, '_component_slot_records', lambda *_args: [blocked]*11+[valid])
    child = next(connection_states(state, 'side', 1))
    assert child.node_axial[100] == (1, 0)
    assert child.audit(diagnostics=False)['legal']


def test_fixed_input_geometry_is_not_bent_by_space_search():
    state = side_space_fixture(obstacle=True)
    unit = replace(state.unit('branch:0'), metadata=())
    state.problem = replace(state.problem, units=tuple(
        unit if u.unit_id == unit.unit_id else u for u in state.problem.units))
    state.source_problem = state.problem
    state.__post_init__()
    site = state.problem.contracts[0].target_node_id
    rows = list(attached_coordinates(state, unit, (1, 0), site))
    assert rows and all(row[0] == unit for row in rows)
    for child in connection_states(state, 'side', 10):
        assert child.unit(unit.unit_id) == unit
        assert child.audit(diagnostics=False)['legal']


def test_space_selected_conformer_replays_as_one_atomic_connection():
    from RL_MTCS.storage.state_delta import encode_transition, replay_transition
    state = side_space_fixture(obstacle=True)
    before = state.state_signature()
    child = next(connection_states(state, 'side', 1))
    restored = replay_transition(state, encode_transition(state, child))
    assert restored.state_signature() == child.state_signature()
    assert restored.problem == child.problem
    assert restored.action_history == child.action_history
    assert restored.audit(diagnostics=False)['legal']
    assert state.state_signature() == before


def test_quick_forward_uses_same_space_embedding_without_mutating_source():
    from RL_MTCS.mdp.components import finish_flex
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    state = finish_flex(side_space_fixture())
    before = state.state_signature()
    formal = next(connection_states(state, 'side', 1))
    quick = fast_rollout_remaining(state)
    assert quick.audit['legal']
    assert quick.report['stages']['side']['completed'] == 1
    assert quick.state.node_axial == formal.node_axial
    assert state.state_signature() == before


@pytest.mark.parametrize('rotation', range(6))
@pytest.mark.parametrize('reflection', [False, True])
def test_space_embedding_uses_actual_phase_in_every_world_orientation(rotation, reflection):
    state = side_space_fixture()
    assert state.transform_component('aromatic:0', rotation, (3, 0), reflection=reflection)
    state.backbone_hull = convex_hull([axial_to_cartesian(p) for p in state.node_axial.values()])
    from RL_MTCS.actions.flex import _component_slot_records
    site = state.problem.contracts[0].target_node_id
    slot = next(row for row in _component_slot_records(state, ['aromatic:0'], ()) if row[1] == site)
    position = axial_add(slot[2], slot[3])
    unit = state.unit('branch:0')
    rows = list(attached_coordinates(state, unit, position, site))
    assert rows
    for variant, r, translation, reflected, coords in rows:
        assert all(sum(point) % 3 != 0 for point in coords.values())
        trial = state.clone()
        trial.select_conformer(variant)
        assert trial.can_place(unit.unit_id, r, translation,
            attachment_edge=(100, site), reflection=reflected)[0]


def test_su25_fixed_center_branch_and_nonfirst_attachment_node_are_preserved():
    from test_rl_mtcs_su25_center_branch import star_unit
    state = side_space_fixture()
    source = star_unit()
    unit = replace(source, unit_id='branch:0',
        nodes=tuple(replace(n, node_id=n.node_id+99) for n in source.nodes),
        internal_edges=tuple(replace(e, source_node_id=e.source_node_id+99,
            target_node_id=e.target_node_id+99) for e in source.internal_edges),
        ports=(UnitPort('end', 101, 'topology_boundary', 'side', (1, 1)),),
        su25_center_branches=((100, 104),))
    units = tuple(unit if u.unit_id == unit.unit_id else u for u in state.problem.units)
    plan = {n.node_id: n.su_type for u in units for n in u.nodes}
    plan[state.problem.contracts[0].target_node_id] = 11
    problem = replace(state.problem, units=units, node_su_types=tuple(plan.items()),
        ownership=tuple((n.node_id, u.unit_id) for u in units for n in u.nodes),
        contracts=(replace(state.problem.contracts[0], source_node_id=101),))
    state = HexAssemblyState(problem, placements=state.placements,
        node_axial=state.node_axial, occupied_axial=state.occupied_axial,
        final_node_positions=state.final_node_positions, edges=state.edges,
        main_anchor=state.main_anchor, backbone_hull=state.backbone_hull, stage='side')
    before = state.state_signature()
    outcomes = list(connection_states(state, 'side', 10))
    assert outcomes
    for child in outcomes:
        assert sum(child.node_axial[104]) % 3 == 0
        assert all(sum(child.node_axial[n]) % 3 != 0 for n in (100, 101, 102, 103))
        assert child.unit(unit.unit_id).su25_center_branches == ((100, 104),)
        assert child.audit(diagnostics=False)['legal']
    assert state.state_signature() == before
