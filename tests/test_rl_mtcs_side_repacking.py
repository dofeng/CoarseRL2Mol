"""A real occupied entrance can be freed without moving its backbone."""
from dataclasses import replace

from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.actions.side import iter_side_outcomes
from RL_MTCS.geometry import aromatic_slot_direction, axial_to_cartesian, convex_hull
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import TopologyUnit, UnitNode, UnitPort, UnitEdge, ExternalContract, InverseAssemblyInput
from RL_MTCS.storage.state_delta import encode_transition, replay_transition
from test_rl_mtcs_side_space import aromatic_unit


def blocked_entry_fixture():
    aromatic = aromatic_unit('aromatic:0')
    aromatic = replace(aromatic, ports=tuple(UnitPort(f'p{i}', i, 'aromatic_boundary', 'side',
        aromatic_slot_direction(aromatic, i)) for i in (0, 1)))
    chain = TopologyUnit('old', 'side_chain', tuple(
        UnitNode(100+i, 22 if i==3 else 23, p,
                 metadata=(('target_hop1_degree', str(1 if i==3 else 2)),))
        for i,p in enumerate(((0,0),(1,0),(1,-1),(0,-2)))),
        tuple(UnitEdge(100+i,101+i) for i in range(3)),
        (UnitPort('end',100,'topology_boundary','side',(-1,-1)),),
        metadata=(('geometry_source','generated_from_connectivity'),))
    new = TopologyUnit('new', 'side_chain',
        (UnitNode(200,28,(0,0),metadata=(('target_hop1_degree','1'),)),),
        ports=(UnitPort('end',200,'topology_boundary','side',(-1,0)),),
        metadata=(('geometry_source','generated_from_connectivity'),))
    units = (aromatic, chain, new)
    contracts = (ExternalContract('old-c','old',100,'aromatic:0',1,'side',
                    source_port_id='end',target_port_id='p1'),
                 ExternalContract('new-c','new',200,'aromatic:0',0,'side',
                    source_port_id='end',target_port_id='p0'))
    plan = {n.node_id:n.su_type for u in units for n in u.nodes}
    plan.update({0:5,1:11})
    state = HexAssemblyState(InverseAssemblyInput(units,contracts,tuple(plan.items()),
        tuple((n.node_id,u.unit_id) for u in units for n in u.nodes)))
    assert state.place_unit('aromatic:0',0,(-1,-2))
    state.main_anchor = 'aromatic:0'
    state.backbone_hull = convex_hull([axial_to_cartesian(p) for p in state.node_axial.values()])
    state.stage = 'side'
    return next(child for child in iter_side_outcomes(state,chain)
                if child.node_axial[100]==(1,0) and child.node_axial[103]==(1,-2)
                and child.node_axial[101]==(2,0) and child.node_axial[102]==(2,-1))


def restrict_to_blocked_entry(monkeypatch):
    import RL_MTCS.actions.flex as module
    original = module._component_slot_records
    monkeypatch.setattr(module,'_component_slot_records',lambda *a:
        [slot for slot in original(*a) if slot[2]==(0,-2)])


def test_atomic_repacking_adds_one_unit_and_preserves_anchor_identity(monkeypatch):
    state = blocked_entry_fixture()
    before = state.state_signature()
    restrict_to_blocked_entry(monkeypatch)
    assert next(connection_states(state,'side',1),None) is None
    child = next(connection_states(state,'side',1,allow_side_repacking=True))
    assert len(child.placements)==len(state.placements)+1
    assert child.node_axial[200]==(1,-2)
    assert child.node_axial[100]==state.node_axial[100]
    assert child.node_axial[103]!=state.node_axial[103]
    assert child.problem.contracts==state.problem.contracts
    assert child.unit('old').internal_edges==state.unit('old').internal_edges
    assert child.satisfied_contracts=={'old-c','new-c'}
    assert child.action_history[-1]['side_replacements'][0]['topology_unit_id']=='old'
    assert child.audit()['legal']
    restored = replay_transition(state,encode_transition(state,child))
    assert restored.state_signature()==child.state_signature()
    assert restored.audit()['legal']
    assert state.state_signature()==before


def test_failed_anchor_change_does_not_mutate_the_state():
    state = blocked_entry_fixture()
    before = state.state_signature()
    assert not state.reshape_side(state.unit('old'),0,(4,0))
    assert state.state_signature()==before


def test_quick_forward_keeps_direct_attachment_policy(monkeypatch):
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    state = blocked_entry_fixture()
    restrict_to_blocked_entry(monkeypatch)
    before = state.state_signature()
    result = fast_rollout_remaining(state)
    assert result.report['stages']['side']['completed']==1
    assert result.audit['legal']
    assert state.state_signature()==before
