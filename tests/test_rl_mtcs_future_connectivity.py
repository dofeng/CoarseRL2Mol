from itertools import islice
"""Future capacity, staging phase and transactional external-bond regressions."""
from dataclasses import replace
from pathlib import Path

import pytest

from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.geometry import axial_add, hex_edge_valid
from RL_MTCS.mdp.types import (ExternalContract, InverseAssemblyInput, TopologyUnit,
                            UnitNode, UnitPort)
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import AssemblyEdge, UnitPlacement
from test_rl_mtcs_sites import ring_side_fixture, aromatic_fixture
from test_rl_mtcs_three_terminal import three_terminal_fixture


def external_fixture(source=(24, -2), target=(24, -3)):
    direction = (target[0]-source[0], target[1]-source[1])
    chain = TopologyUnit('chain', 'flex_chain', (UnitNode(269, 23, (0, 0)),),
                        ports=(UnitPort('p', 269, phase='flex', direction=direction),))
    ring = TopologyUnit('aryl', 'side_chain', (UnitNode(227, 22, (0, 0)),),
                       ports=(UnitPort('q', 227, phase='flex',
                                       direction=(-direction[0], -direction[1])),))
    contract = ExternalContract('edge', 'chain', 269, 'aryl', 227, 'flex',
                                source_port_id='p', target_port_id='q')
    state = HexAssemblyState(InverseAssemblyInput((chain, ring), (contract,),
        ((269, 23), (227, 22)), ((269, 'chain'), (227, 'aryl'))))
    assert state.place_unit('chain', 0, source)
    if sum(target)%3:
        assert state.place_unit('aryl', 0, target, attachment_edge=(269, 227))
    else:
        # Deliberately corrupt a snapshot to test replay/audit defenses too.
        assert not state.place_unit('aryl', 0, target, attachment_edge=(269, 227))
        state.placements['aryl']=UnitPlacement('aryl',0,target)
        state.node_axial[227]=target
        state.occupied_axial[target]=227
    action = dict(kind='flex', contract_id='edge', source_unit_id='chain',
                  source_node_id=269, source_port_id='p', target_unit_id='aryl',
                  target_node_id=227, target_port_id='q')
    return state, contract, action


def test_wrong_phase_external_edge_rejected_everywhere_and_rollback():
    state, contract, action = external_fixture()
    assert hex_edge_valid(state.node_axial[269], state.node_axial[227])
    before = state.state_signature()
    assert not state.connect(action)[0]
    assert not state.connect_adjacent_contract(contract, 'test')
    assert state.state_signature() == before
    state.edges.append(AssemblyEdge(269, 227, 'flex', 'edge'))
    state.satisfied_contracts.add('edge')
    assert state.audit_geometry_errors()
    assert state.audit_port_direction_errors()


def test_legal_external_edge_commits_and_audits():
    state, _, action = external_fixture((1, 0), (2, 0))
    assert state.connect(action)[0]
    assert not state.audit_geometry_errors()
    assert not state.audit_port_direction_errors()


def test_forward_audit_is_complete_continuation_not_bounded_matching():
    state=ring_side_fixture()
    signature=state.state_signature()
    audit=fast_rollout_remaining(state).to_dict()
    assert audit['forward_completion']==1
    assert audit['stages']['ring']['completed']==1
    assert audit['stages']['side']['completed']==1
    assert state.state_signature()==signature


def test_no_slots_returns_partial_without_rejecting_structure():
    state=ring_side_fixture()
    for node in state.unit('aromatic:0').nodes:
        state.current_su_types[node.node_id]=12
    audit=fast_rollout_remaining(state).to_dict()
    assert audit['status']=='partial'
    assert audit['stages']['ring']['completed']==0
    assert audit['stages']['side']['completed']==0
    assert audit['total']==3


def test_audit_ignores_source_pairing_and_action_history():
    state = ring_side_fixture()
    first = fast_rollout_remaining(state).to_dict()
    state.problem = replace(state.problem, source_rigid_contracts=(
        ExternalContract('unrelated', 'bogus', 999, 'bogus2', 1000, 'rigid'),))
    state.action_history = [{'kind': 'rigid', 'site_bindings': {'irrelevant': 1}}]
    assert fast_rollout_remaining(state).to_dict() == first


def test_three_terminal_complete_rollout_does_not_mutate_state():
    state,_,_=three_terminal_fixture()
    before=state.state_signature()
    audit=fast_rollout_remaining(state).to_dict()
    assert audit['stages']['flex']['completed']==1
    assert audit['forward_completion']==1
    assert state.state_signature()==before


def test_component_staging_distance_does_not_change_completion():
    state,_,_=three_terminal_fixture()
    before=fast_rollout_remaining(state).to_dict()
    assert state.transform_component('aromatic:2',0,(300,0))
    after=fast_rollout_remaining(state).to_dict()
    assert before['forward_completion']==after['forward_completion']==1


def test_disconnected_staging_keeps_all_external_site_phases():
    from RL_MTCS.mdp.components import stage_disconnected_components
    from RL_MTCS.actions.flex import _component_slot_records
    state=aromatic_fixture(3)
    stage_disconnected_components(state)
    for _,_,axial,outward in _component_slot_records(state,list(state.placements),()):
        assert hex_edge_valid(axial,axial_add(axial,outward),honeycomb=True)
    assert len(state.placements)==3 and not state.excluded_unit_ids


def test_ym3_initialization_preserves_endpoint_phase():
    from RL_MTCS.io.input import initialize
    from RL_MTCS.actions.flex import _component_slot_records
    root=Path(__file__).resolve().parents[1]/'test_results/YM-3/final_outputs'
    if not root.exists():pytest.skip('YM-3 local input unavailable')
    state=initialize(root/'final_nodes_by_global_id.csv',root/'final_topology_specs.csv')['state']
    slots=_component_slot_records(state,list(state.placements),())
    assert slots
    assert all(hex_edge_valid(axial,axial_add(axial,outward),honeycomb=True) for _,_,axial,outward in slots)


def test_three_terminal_bad_branch_port_is_rejected_atomically():
    from RL_MTCS.actions import flex
    from test_rl_mtcs_three_terminal import anchored
    state, _, _ = three_terminal_fixture()
    unit = state.unit('topology:0')
    # The last port now points back into the body. Independent first/second
    # endpoint checks must not admit a partially connected three-end unit.
    ports = (*unit.ports[:-1], replace(unit.ports[-1], direction=(-1, -1)))
    broken = replace(unit, ports=ports)
    state.problem = replace(state.problem, units=tuple(
        broken if u.unit_id == unit.unit_id else u for u in state.problem.units))
    state.__post_init__()
    candidate = anchored(state)
    before = candidate.state.state_signature()
    assert not list(islice(flex._backbone_site_outcomes(candidate, unit.unit_id, [1, 2], flex._rigid_components(state)), 4))
    assert candidate.state.state_signature() == before
    assert not candidate.state.satisfied_contracts


def test_side_probe_uses_actual_committed_edges_and_ports():
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    final=fast_rollout_remaining(ring_side_fixture()).state
    contract=next(c for c in final.problem.contracts if c.kind=='side')
    assert contract.contract_id in final.satisfied_contracts
    assert any(e.kind=='side' and e.contract_id==contract.contract_id for e in final.edges)
    assert not final.audit_port_direction_errors()
