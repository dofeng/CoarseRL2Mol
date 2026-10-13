"""One fixed SU22 centre branch; all other nodes and bonds stay ordinary."""
import csv
from dataclasses import replace
from pathlib import Path

import pytest

from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.actions.conformers import iter_topology_conformers, validate_conformer
from RL_MTCS.geometry import hex_edge_valid
from RL_MTCS.io.input import initialize, parse_topology_notation
from RL_MTCS.io.layout import _layout_topology_tree, _su25_center_branches
from RL_MTCS.io.output import export_final_state
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import InverseAssemblyInput, InverseInputError, TopologyUnit, UnitNode, UnitEdge


def star_unit():
    return TopologyUnit('star', 'side_chain', (
        UnitNode(1, 25, (1, 0)), UnitNode(2, 23, (2, 0)),
        UnitNode(3, 22, (1, 1)), UnitNode(4, 23, (0, -1)),
        UnitNode(5, 22, (0, 0))),
        tuple(UnitEdge(1, n) for n in (2, 3, 4, 5)),
        metadata=(('geometry_source', 'generated_from_connectivity'),),
        su25_center_branches=((1, 5),))


def state_for(unit):
    return HexAssemblyState(InverseAssemblyInput((unit,), (),
        tuple((n.node_id, n.su_type) for n in unit.nodes),
        tuple((n.node_id, unit.unit_id) for n in unit.nodes)))


@pytest.fixture(scope='module')
def ym_source():
    inputs = Path(__file__).resolve().parents[1] / 'test_results/YM-3/final_outputs'
    return initialize(inputs / 'final_nodes_by_global_id.csv', inputs / 'final_topology_specs.csv')['state']


@pytest.mark.parametrize('row', range(2, 9))
def test_supplied_bridge_connects_atomically_in_formal_and_quick_modes(ym_source, row):
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    uid = f'topology:{row}'
    contracts = tuple(c for c in ym_source.problem.contracts if c.source_unit_id == uid)
    aromatic_ids = sorted({c.target_unit_id for c in contracts})
    unit_ids = {uid, *aromatic_ids}
    units = tuple(u for u in ym_source.problem.units if u.unit_id in unit_ids)
    gids = {n.node_id for u in units for n in u.nodes}
    problem = InverseAssemblyInput(units, contracts,
        tuple((n, su) for n, su in ym_source.problem.node_su_types if n in gids),
        tuple((n, u) for n, u in ym_source.problem.ownership if n in gids))
    state = HexAssemblyState(problem, current_su_types={n: ym_source.current_su_types[n] for n in gids})
    state.stage = 'flex'
    for i, aryl in enumerate(aromatic_ids):
        assert state.place_unit(aryl, 0, (30*i, 0))
    before = state.state_signature()
    formal = next(connection_states(state, 'flex', 1), None)
    assert formal is not None
    assert len(formal.satisfied_contracts) == len(contracts)
    assert formal.audit()['legal']
    if row != 8:
        assert formal.component_count() == 1
    quick = fast_rollout_remaining(state)
    assert quick.audit['legal'] and quick.report['stages']['flex']['completed'] == 1
    assert len(quick.state.satisfied_contracts) == len(contracts)
    assert state.state_signature() == before
    if row == 8:
        assert sum(quick.state.node_axial[299]) % 3 == 0
        assert sum(quick.state.node_axial[433]) % 3 != 0


@pytest.mark.parametrize('rotation', range(6))
@pytest.mark.parametrize('reflection', [False, True])
def test_exact_fixed_su22_is_allowed_in_every_rigid_pose(rotation, reflection):
    unit = star_unit()
    validate_conformer(unit, unit)
    state = state_for(unit)
    assert state.place_unit(unit.unit_id, rotation, (3, 0), reflection=reflection)
    assert state.node_axial[5] == (3, 0)
    assert sum(state.node_axial[3]) % 3 != 0
    assert state.audit()['legal']
    assert state.audit()['su25_center_branches_valid']
    assert state.transform_component(unit.unit_id, 1, (0, 3), reflection=True)
    assert state.audit()['legal']


def test_two_su22_branches_bind_only_one_before_layout():
    parsed = parse_topology_notation('10-1(-3)(-5)-20')
    types = {10: 23, 1: 25, 3: 22, 5: 22, 20: 23}
    pairs = _su25_center_branches(parsed, types, {1: (5, 3, 10, 20)})
    assert pairs == ((1, 5),)
    coords = _layout_topology_tree(parsed, types, pairs)
    assert sum(coords[5]) % 3 == 0
    assert sum(coords[3]) % 3 != 0
    assert hex_edge_valid(coords[1], coords[3], honeycomb=True)
    assert not hex_edge_valid(coords[1], coords[5], honeycomb=True)


@pytest.mark.parametrize('change', ['second_center', 'fixed_ordinary', 'nonadjacent'])
def test_branch_role_cannot_change_with_pose(change):
    unit = star_unit()
    gid, point = {'second_center': (3, (1, -1)),
                  'fixed_ordinary': (5, (-1, -1)),
                  'nonadjacent': (5, (3, 0))}[change]
    candidate = replace(unit, nodes=tuple(replace(n, local_axial=point) if n.node_id == gid else n
                                         for n in unit.nodes))
    with pytest.raises(InverseInputError):
        validate_conformer(unit, candidate)
    state = state_for(candidate)
    assert not state.place_unit(candidate.unit_id, 0, (0, 0))


@pytest.mark.parametrize('pairs', [(), ((1, 5), (1, 3)), ((2, 5),), ((1, 2),)])
def test_missing_duplicate_or_wrong_branch_declaration_is_rejected(pairs):
    unit = replace(star_unit(), su25_center_branches=pairs)
    with pytest.raises(InverseInputError):
        validate_conformer(unit, unit)
    assert not next(iter_topology_conformers(unit), None)
    state = state_for(unit)
    assert not state.place_unit(unit.unit_id, 0, (0, 0))
    assert not state.audit()['legal']


def test_swapping_fixed_and_normal_su22_identity_cannot_pass_audit():
    state = state_for(star_unit())
    assert state.place_unit('star', 0, (0, 0))
    state.problem = replace(state.problem, units=(replace(state.unit('star'), su25_center_branches=((1, 3),)),))
    state.__post_init__()
    audit = state.audit()
    assert not audit['legal']
    assert any(e['reason'] == 'su25_center_branch_identity_changed'
               for e in audit['su25_center_branch_errors'])


def test_declared_center_is_not_a_collision_exemption():
    state = state_for(star_unit())
    state.occupied_axial[(0, 0)] = 999
    ok, reason = state.can_place('star', 0, (0, 0))
    assert not ok and reason.startswith('axial_collision:')


def test_center_branch_export_explains_its_nonordinary_bond(tmp_path):
    state = state_for(star_unit())
    assert state.place_unit('star', 0, (0, 0))
    outputs = export_final_state(state, tmp_path)
    with open(outputs['nodes_csv']) as handle:
        nodes = {int(r['node_id']): r for r in csv.DictReader(handle)}
    assert nodes[5]['su25_center_parent_id'] == '1'
    assert nodes[3]['su25_center_parent_id'] == ''
    with open(outputs['edges_csv']) as handle:
        edges = list(csv.DictReader(handle))
    special = [r for r in edges if r['su25_center_branch_edge'] == 'True']
    assert len(special) == 1 and special[0]['target_node_id'] == '5'
    assert special[0]['honeycomb_edge_valid'] == 'False'
    assert all(r['actual_geometry_valid'] == 'True' for r in edges)
