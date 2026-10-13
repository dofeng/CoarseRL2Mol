"""Regressions for logical identity, actual labels and strict honeycomb placement."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.actions.conformers import iter_topology_conformers, validate_conformer
from RL_MTCS.actions.substitution import SubstitutionCandidate, apply_candidate, _candidate_rows
from RL_MTCS.geometry import HEX_VERTEX_OFFSETS, hex_edge_valid
from RL_MTCS.io.input import initialize, build_csv_inverse_input, load_topology_rows
from RL_MTCS.io.output import export_final_state
from RL_MTCS.mdp.components import finish_flex
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import InverseAssemblyInput, InverseInputError, UnitNode, UnitEdge, UnitPort, TopologyUnit
from RL_MTCS.reporting.completion import completion_report
from model.shared.port_rules import HOP1_PORT_COMBINATIONS, can_match_ports_exact, port_rule_nodes
from test_rl_mtcs_sites import ring_side_fixture


INPUTS = Path(__file__).resolve().parents[1] / 'test_results/YM-3/final_outputs'


@pytest.fixture(scope='module')
def ym_state():
    return initialize(INPUTS / 'final_nodes_by_global_id.csv', INPUTS / 'final_topology_specs.csv')['state']


def substitute_five(state):
    request = next(r for r in state.problem.hetero_requests if r.ring_size == 5)
    return apply_candidate(state, _candidate_rows(state, request)[0])


@pytest.mark.parametrize('target,site', [(501, 129), (502, 240)])
def test_substitution_moves_logical_identity_without_retargeting_pending_tasks(ym_state, target, site):
    before = ym_state.state_signature()
    owner = ym_state.problem.ownership_map()
    nodes = {n.node_id: n for u in ym_state.problem.units for n in u.nodes}
    child = apply_candidate(ym_state, SubstitutionCandidate(26, 6, owner[site], (site,),
                                                          representative_node_id=target))
    assert ym_state.state_signature() == before
    assert child.source_problem is ym_state.source_problem
    assert child.node_axial[target] == ym_state.node_axial[site]
    assert child.node_axial[site] == ym_state.node_axial[target]
    assert child.current_su_types[target] == 26 and child.current_su_types[site] == 13
    for original, current in zip(ym_state.problem.contracts, child.problem.contracts):
        assert (current.source_node_id, current.target_node_id) == (original.source_node_id, original.target_node_id)
        for uid, pid, gid in ((current.source_unit_id, current.source_port_id, current.source_node_id),
                              (current.target_unit_id, current.target_port_id, current.target_node_id)):
            assert child._ports[f'{uid}:{pid}'].node_id == gid
            assert child.problem.ownership_map()[gid] == uid
    assert child.unit(owner[site]).node_map()[target].slot_id == nodes[site].slot_id
    assert child.audit()['legal']
    assert not child.audit()['identity_errors']


def test_five_member_collapse_removes_only_request_synthetic_identity(ym_state):
    child = substitute_five(ym_state)
    request = next(r for r in ym_state.problem.hetero_requests if r.ring_size == 5)
    assert set(ym_state.problem.node_map()) - set(child.problem.node_map()) == {request.synthetic_node_id}
    assert all(c.source_node_id != request.synthetic_node_id and c.target_node_id != request.synthetic_node_id
               for c in child.problem.contracts)
    assert child.problem.contracts == ym_state.problem.contracts or all(
        (a.contract_id, a.source_node_id, a.target_node_id) == (b.contract_id, b.source_node_id, b.target_node_id)
        for a, b in zip(child.problem.contracts, ym_state.problem.contracts))
    assert child.audit()['legal']


@pytest.mark.parametrize('gid', [129, 240])
def test_planned_label_without_actual_connection_is_illegal(ym_state, gid):
    child = ym_state.clone()
    child.current_su_types[gid] = child.problem.node_map()[gid]
    audit = child.audit()
    assert audit['geometry_valid'] and audit['identity_valid']
    assert not audit['label_valid'] and not audit['legal']
    assert any(e['node_id'] == gid for e in audit['label_errors'])


def test_pending_aromatic_tasks_keep_su13(ym_state):
    assert all(ym_state.current_su_types[n] == 13 for n in (129, 240, 292, 456))
    assert not ym_state.audit_label_errors()
    assert not ym_state.audit()['input_valid']
    from RL_MTCS.io.output import composition_report
    report = composition_report(ym_state, ym_state.source_problem)
    assert report['composition_reconciled']
    assert {292, 456} <= set(report['pending_label_ids'])


@pytest.mark.parametrize('su', [19, 22, 25, 27, 28, 29])
def test_ordinary_nodes_cannot_be_placed_at_ring_centers(su):
    unit = TopologyUnit('body', 'side_chain', (UnitNode(1, su, (0, 0)),))
    state = HexAssemblyState(InverseAssemblyInput((unit,), (), ((1, su),), ((1, 'body'),)))
    assert not state.place_unit('body', 0, (0, 0))
    assert state.place_unit('body', 0, (1, 0))


def test_unchanged_straight_hetero_conformer_is_not_a_valid_embedding():
    unit = TopologyUnit('body', 'side_chain', tuple(UnitNode(i, 29, (i, 0)) for i in range(3)),
                        (UnitEdge(0, 1), UnitEdge(1, 2)))
    with pytest.raises(InverseInputError, match='honeycomb'):
        validate_conformer(unit, unit)
    assert not list(iter_topology_conformers(unit))


def test_normalized_origin_requires_one_phase_for_nodes_and_all_ports():
    valid = TopologyUnit('body', 'side_chain', (UnitNode(1, 23, (0, 0)),),
                        ports=(UnitPort('a', 1, direction=(1, 0)), UnitPort('b', 1, direction=(0, 1))))
    validate_conformer(valid, valid)
    invalid = replace(valid, ports=(valid.ports[0], replace(valid.ports[1], direction=(1, 1))))
    with pytest.raises(InverseInputError, match='honeycomb'):
        validate_conformer(invalid, invalid)


def test_internal_nonedge_neighbors_preserve_the_explicit_open_chain():
    unit = TopologyUnit('body', 'side_chain', tuple(UnitNode(i, 23, p) for i, p in enumerate(HEX_VERTEX_OFFSETS)),
                        tuple(UnitEdge(i, i+1) for i in range(5)))
    validate_conformer(unit, unit)
    state = HexAssemblyState(InverseAssemblyInput((unit,), (), tuple((i, 23) for i in range(6)),
                                                 tuple((i, 'body') for i in range(6))))
    assert state.place_unit('body', 0, (0, 0))
    assert len(state.edges) == 5
    assert not any({e.source_node_id, e.target_node_id} == {0, 5} for e in state.edges)
    assert state.audit()['legal']


def test_five_member_exception_does_not_disable_other_aromatic_template_audits(ym_state):
    child = substitute_five(ym_state)
    modified = child.action_history[-1]['unit_id']
    other = next(u for u in child.problem.units if u.phase == 'aromatic' and u.unit_id != modified)
    child.problem = replace(child.problem, units=tuple(replace(u, local_rings=()) if u == other else u
                                                      for u in child.problem.units))
    child.__post_init__()
    audit = child.audit()
    assert not audit['legal'] and not audit['geometry_valid']
    assert any(e['unit_id'] == other.unit_id for e in audit['aromatic_template_errors'])


@pytest.mark.parametrize('wrong_node', [False, True])
def test_missing_or_mismatched_port_cannot_commit_adjacent_contract(wrong_node):
    original = next(connection_states(finish_flex(ring_side_fixture()), 'side', 1))
    contract = next(c for c in original.problem.contracts if c.kind == 'side')
    state = original.clone()
    state.edges = [e for e in state.edges if e.contract_id != contract.contract_id]
    state.satisfied_contracts.remove(contract.contract_id)
    state.used_ports.clear()
    units = []
    for u in state.problem.units:
        if u.unit_id == contract.source_unit_id:
            ports = tuple(replace(p, node_id=999999) for p in u.ports) if wrong_node else ()
            u = replace(u, ports=ports)
        units.append(u)
    state.problem = replace(state.problem, units=tuple(units))
    state.__post_init__()
    before = state.state_signature()
    assert not state.connect_adjacent_contract(contract, 'invalid_test')
    assert state.state_signature() == before


def test_completed_contract_requires_its_real_edge_and_used_ports():
    state = next(connection_states(finish_flex(ring_side_fixture()), 'side', 1))
    assert state.audit()['legal']
    contract = next(c for c in state.problem.contracts if c.kind == 'side')
    state.edges = [e for e in state.edges if e.contract_id != contract.contract_id]
    audit = state.audit()
    assert not audit['topology_valid'] and not audit['legal']
    assert any(e['reason'] == 'satisfied_contract_edge_mismatch' for e in audit['contract_errors'])


def test_supplied_bridges_and_su25_are_available_without_chemical_revalidation(ym_state):
    report = completion_report(ym_state)
    assert report['total'] == 162 and report['stages']['flex']['total'] == 15
    assert report['stages']['ring']['total'] == 12
    issues = [json.loads(row) for row in ym_state.problem.input_issues]
    assert {r.get('node_id') for r in issues if r['reason'] == 'input_task_coverage_error'} == {292, 456}
    unavailable = [u for u in ym_state.problem.units if dict(u.metadata).get('assembly_unavailable_reason')]
    assert not unavailable
    assert 299 in ym_state.problem.node_map()
    assert len(issues) == 2
    for row in range(2, 9):
        unit = ym_state.unit(f'topology:{row}')
        validate_conformer(unit, unit)
        assert next(iter_topology_conformers(unit)) == unit
    assert ym_state.unit('topology:8').su25_center_branches == ((499, 299),)


@pytest.mark.parametrize('change,error', [('type', 'type mismatch'), ('shape', 'connectivity mismatch'),
                                        ('degree', 'degree mismatch')])
def test_input_su_and_gid_notations_must_agree(monkeypatch, change, error):
    import RL_MTCS.io.input as module
    rows = [dict(r) for r in load_topology_rows(INPUTS / 'final_topology_specs.csv')]
    branched = next(r for r in rows if 'single20d3' in r['actual_su'])
    if change == 'type':
        rows[0]['actual_su'] = rows[0]['actual_su'].replace('5-', '6-', 1)
    elif change == 'shape':
        branched['actual_su'] = '11-23-single20d3-0-9-23-11'
    else:
        branched['actual_su'] = branched['actual_su'].replace('d3', 'd2')
    monkeypatch.setattr(module, 'load_topology_rows', lambda _: rows)
    with pytest.raises(InverseInputError, match=error):
        build_csv_inverse_input(INPUTS / 'final_nodes_by_global_id.csv', INPUTS / 'final_topology_specs.csv')


def test_shared_rules_validate_actual_degree_and_anchor_mode():
    labels = {1: 19, 2: 29, 3: 23}
    adjacency = {1: {2, 3}, 2: {1}, 3: {1}}
    nodes = port_rule_nodes(labels, adjacency, {1: {'su_label': 'double19d2'}})
    assert not can_match_ports_exact([nodes[2], nodes[3]], HOP1_PORT_COMBINATIONS[19], nodes[1])
    nodes = port_rule_nodes(labels, adjacency, {1: {'su_label': 'single19d2'}})
    assert can_match_ports_exact([nodes[2], nodes[3]], HOP1_PORT_COMBINATIONS[19], nodes[1])


def test_exports_expose_identity_and_real_honeycomb_edge_checks(tmp_path):
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    state = fast_rollout_remaining(ring_side_fixture()).state
    outputs = export_final_state(state, tmp_path)
    with Path(outputs['nodes_csv']).open() as handle:
        nodes = list(csv.DictReader(handle))
    assert all(n['logical_gid'] == n['node_id'] and int(n['actual_degree']) > 0 for n in nodes)
    with Path(outputs['edges_csv']).open() as handle:
        edges = list(csv.DictReader(handle))
    assert edges and 'hex_neighbor' not in edges[0]
    assert all(e['honeycomb_edge_valid'] == 'True' and e['actual_geometry_valid'] == 'True' for e in edges)


def test_actual_coordinate_changes_cannot_hide_behind_valid_integer_snapshot():
    state = ring_side_fixture()
    gid = next(iter(state.node_axial))
    p = state.final_node_positions[gid]
    state.final_node_positions[gid] = (p[0] + .1, p[1])
    audit = state.audit()
    assert not audit['legal']
    assert any(e['reason'] == 'ordinary_final_position_changed' for e in audit['actual_geometry_errors'])


def test_nmr_descriptor_uses_only_committed_nodes_edges_and_labels():
    from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
    state = ring_side_fixture()
    scorer = NMRGuidanceScorer.__new__(NMRGuidanceScorer)
    data = scorer._graph_descriptor(state)
    assert set(data['node_ids']) == set(state.node_axial)
    assert data['su_indices'].tolist() == [13] * 6
    assert data['edge_index'].shape[1] == 6
    assert 200 not in data['node_ids'] and 100 not in data['node_ids']


def test_inverse_audit_retains_its_own_port_rule_diagnostics(monkeypatch):
    from types import SimpleNamespace
    from model.inverse import inverse_pipeline as module
    pipeline = module.InversePipelineV3.__new__(module.InversePipelineV3)
    nodes = [SimpleNamespace(global_id=gid, su_type=su, hop1_ids=neighbors)
             for gid, su, neighbors in [(1, 5, [2]), (2, 2, [1, 3]), (3, 2, [2, 4]), (4, 5, [3])]]
    pipeline._final_stage3_completion_diagnostics = {
        'stage3_complete': True, 'products': [{'fixed_source_node_ids': [1, 2, 3, 4],
            'expected_su_type_by_node_id': {1: 5, 2: 2, 3: 2, 4: 5}}]}
    monkeypatch.setattr(module, 'validate_snapshot_state', lambda *a, **k: {'errors': []})
    audit = pipeline._audit_final_su_identity(nodes)
    assert not audit['ok']
    assert {e['node_id'] for e in audit['port_rule_errors']} == {2, 3}


def test_explicitly_unassembled_resources_prevent_all_complete():
    from RL_MTCS.mdp.types import UnassembledClosureResource
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    state = fast_rollout_remaining(ring_side_fixture()).state
    state.problem = replace(state.problem, unassembled_closure_resources=(UnassembledClosureResource(999, 22),))
    report = completion_report(state)
    assert report['completed'] == report['total']
    assert not report['all_complete'] and not state.is_complete()


def test_adjacent_unconnected_units_do_not_create_edges_or_change_labels():
    units = (TopologyUnit('left', 'side_chain', (UnitNode(348, 19, (0, 0)),)),
             TopologyUnit('right', 'side_chain', (UnitNode(578, 29, (0, 0)),)))
    state = HexAssemblyState(InverseAssemblyInput(units, (), ((348, 19), (578, 29)),
        ((348, 'left'), (578, 'right'))))
    assert state.place_unit('left', 0, (1, 0))
    assert state.place_unit('right', 0, (2, 0))
    assert state.audit()['legal']
    assert not state.edges
    assert state.current_su_types == {348: 19, 578: 29}


@pytest.mark.parametrize('changed', [{1,2},{3,4},{1,2,3,4}])
def test_incremental_and_final_checks_detect_actual_edge_intersections(changed):
    from RL_MTCS.validation import _segment_geometry_errors
    points = {1:(0,0),2:(2,2),3:(0,2),4:(2,0)}
    edges = [UnitEdge(1,2),UnitEdge(3,4)]
    incremental = _segment_geometry_errors(points,edges,changed)
    final = _segment_geometry_errors(points,edges)
    assert len(incremental)==len(final)==1
    assert {frozenset(pair) for pair in incremental[0]['edges']}=={
        frozenset(pair) for pair in final[0]['edges']}
    assert incremental[0]['reason']==final[0]['reason']
    assert _segment_geometry_errors(points,edges)[0]['reason']=='edge_intersection'


def test_new_unbonded_node_on_an_existing_edge_is_still_rejected():
    from RL_MTCS.validation import _segment_geometry_errors
    errors = _segment_geometry_errors({1:(0,0),2:(2,2),3:(1,1)},[UnitEdge(1,2)],{3})
    assert errors==[dict(reason='node_on_edge',node=3,edge=[1,2])]
