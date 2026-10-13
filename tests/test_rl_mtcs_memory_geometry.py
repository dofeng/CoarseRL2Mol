"""Memory lifetime and geometry regressions for the staged search."""
from dataclasses import replace
from types import SimpleNamespace
import gc
import tracemalloc

import pytest

from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.mdp.components import finish_flex, choose_anchor, stage_disconnected_components
from RL_MTCS.geometry import hull_region, convex_hull
from RL_MTCS.geometry import rotate_axial, axial_to_cartesian, aromatic_slot_direction
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.mdp.types import (TopologyUnit, UnitNode, UnitEdge,
                            UnitPort, ExternalContract, InverseAssemblyInput)
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.policy.mcts import MCTS, SearchGuidance
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.storage.state_delta import encode_transition, replay_transition
from RL_MTCS.rewards.scoring import shape_quality
from RL_MTCS.actions import flex
from test_rl_mtcs_sites import aromatic_fixture, ring_side_fixture
from test_rl_mtcs_three_terminal import three_terminal_fixture


def test_exhaustive_tree_releases_closed_nodes_and_streams_records():
    recorded = set()
    tree = MCTS((),
        expand=lambda s: (s+(i,) for i in range(10)), accept=lambda s: len(s)==4,
        signature=lambda s:s, evaluate=lambda s:sum(s)/100, seed=7,
        guidance=SearchGuidance(record=lambda s, children, value: recorded.add(s)))
    assert tree.run() == (9,9,9,9)
    stats = tree.diagnostics()
    assert stats['visited_states'] == len(recorded) == 11111
    assert stats['terminal_count'] == 10000
    assert stats['retained_nodes'] == 1
    assert stats['transition_bytes'] == 0
    assert not tree.root.children and tree.root.pending is None
    assert stats['peak_retained_nodes'] < stats['visited_states']
    tree.run()  # Finished rounds do not emit duplicate learning records.
    assert len(recorded)==11111


def test_large_payload_is_not_retained_at_each_search_node():
    gc.collect()
    tracemalloc.start()
    try:
        def expand(s):
            depth, path, _ = s
            for i in range(6):
                yield depth+1, path+(i,), bytearray(200_000)
        tree = MCTS((0,(),bytearray(200_000)), expand=expand,
            accept=lambda s:s[0]==4, signature=lambda s:s[:2],
            evaluate=lambda s:sum(s[1])/100, seed=7)
        assert tree.run()[1] == (5,5,5,5)
        _, peak = tracemalloc.get_traced_memory()
        # A retention regression would require >300 MB for these 1555 nodes.
        assert peak < 30_000_000
        assert tree.diagnostics()['visited_states']==1555
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize('phase', ['rigid','flex','ring','side','substitution'])
def test_exact_transition_replay_all_stages(phase):
    from RL_MTCS.mdp.types import HeteroAromaticRequest
    if phase=='rigid':
        state=aromatic_fixture(3)
    elif phase=='flex':
        state=three_terminal_fixture()[0]
    elif phase=='substitution':
        unit=ring_side_fixture().unit('aromatic:0')
        a,b=unit.internal_edges[0].source_node_id,unit.internal_edges[0].target_node_id
        problem=InverseAssemblyInput((replace(unit,ports=()),),(),
            tuple((n.node_id,30 if n.node_id==a else 13) for n in unit.nodes),
            tuple((n.node_id,unit.unit_id) for n in unit.nodes),
            hetero_requests=(HeteroAromaticRequest(30,5,a,b),))
        state=HexAssemblyState(problem)
        assert state.place_unit(unit.unit_id,0,(0,0))
    else:
        state=finish_flex(ring_side_fixture())
    before=state.clone()
    children=list(connection_states(state,phase,limit=2))
    assert children
    for child in children:
        delta=encode_transition(state,child)
        restored=replay_transition(state,delta)
        assert restored.problem==child.problem
        assert restored.state_signature()==child.state_signature()
        assert restored.action_history==child.action_history
        assert restored.audit()==child.audit()
        restored.node_axial.clear()
        assert state.state_signature()==before.state_signature()


def branch_fixture():
    state=ring_side_fixture()
    branch=TopologyUnit('topology:1','side_chain',
        tuple(UnitNode(200+i,20 if i==0 else 23,p)
              for i,p in enumerate(((0,0),(1,0),(0,1),(-1,-1)))),
        tuple(UnitEdge(200,200+i) for i in range(1,4)),
        (UnitPort('end',201,'topology_boundary','side',(1,0)),))
    units=tuple(branch if u.unit_id==branch.unit_id else u for u in state.problem.units)
    types={n.node_id:n.su_type for u in units for n in u.nodes}
    types.update({4:11,5:11,0:5})
    problem=replace(state.problem,units=units,node_su_types=tuple(types.items()),
        ownership=tuple((n.node_id,u.unit_id) for u in units for n in u.nodes),
        contracts=tuple(replace(c,source_node_id=201) if c.kind=='side' else c
                        for c in state.problem.contracts))
    result=HexAssemblyState(problem)
    assert result.place_unit('aromatic:0',0,(0,0))
    return finish_flex(result)


def test_ordinary_branch_cannot_occupy_face_centres():
    state=branch_fixture()
    assert not state.place_unit('topology:1',0,(-3,-3))
    # Its supplied fixed embedding has no valid honeycomb placement; it must
    # not become legal merely because it contains a degree-three carbon.
    assert not list(connection_states(state,'side',10))
    state.node_axial[200]=(-3,-3)
    assert any(e['reason']=='non_honeycomb_site' for e in state.audit_geometry_errors())


def chain_pose_fixture(length, rotation):
    template=ring_side_fixture().unit('aromatic:0')
    path=[(1,0)]
    for i in range(length+1):
        dq,dr=(1,0) if i%2==0 else (1,1)
        path.append((path[-1][0]+dq,path[-1][1]+dr))
    outward=(path[-2][0]-path[-1][0],path[-2][1]-path[-1][1])
    center=(path[-1][0]-outward[0],path[-1][1]-outward[1])
    units=[];contracts=[];ports=[];types={}
    for i,point in enumerate(((1,0),outward)):
        unit=replace(template,unit_id=f'aromatic:{i}',ports=(),
            nodes=tuple(replace(n,node_id=n.node_id+i*10) for n in template.nodes),
            internal_edges=tuple(replace(e,source_node_id=e.source_node_id+i*10,
                target_node_id=e.target_node_id+i*10) for e in template.internal_edges))
        boundary=next(n.node_id for n in unit.nodes if n.local_axial==point)
        body=100 if i==0 else 100+length-1
        direction=(-1,0) if i==0 else (-outward[0],-outward[1])
        ports.append(UnitPort(f'p{i}',body,'topology_boundary','flex',direction))
        unit=replace(unit,ports=(UnitPort('p',boundary,'aromatic_boundary','flex',
                                        aromatic_slot_direction(unit,boundary)),))
        units.append(unit)
        types.update({n.node_id:11 if n.node_id==boundary else n.su_type for n in unit.nodes})
        contracts.append(ExternalContract(f'flex:{i}','chain',body,unit.unit_id,boundary,
                                          'flex',source_port_id=f'p{i}',target_port_id='p'))
    origin=path[1]
    chain=TopologyUnit('chain','flex_chain',tuple(UnitNode(100+i,23,(q-origin[0],r-origin[1]))
        for i,(q,r) in enumerate(path[1:-1])),tuple(UnitEdge(100+i,101+i) for i in range(length-1)),tuple(ports))
    units.append(chain);types.update({n.node_id:n.su_type for n in chain.nodes})
    state=HexAssemblyState(InverseAssemblyInput(tuple(units),tuple(contracts),tuple(types.items()),
                            tuple((n.node_id,u.unit_id) for u in units for n in u.nodes)))
    assert state.place_unit('aromatic:0',rotation,(0,0))
    assert state.place_unit('aromatic:1',rotation,rotate_axial(center,rotation))
    candidate=flex._SiteSearchCandidate(state,frozenset(),{},frozenset({0,1}),[])
    return candidate,flex._rigid_components(state)


@pytest.mark.parametrize('length',[3,6])
@pytest.mark.parametrize('rotation',range(6))
def test_internal_chain_uses_actual_geometry_for_all_rotations(length,rotation):
    candidate,components=chain_pose_fixture(length,rotation)
    poses=list(flex._interior_site_poses(candidate,'chain',components))
    assert poses
    for pose in poses:
        committed=flex._place_interior_pose(candidate,'chain',pose)
        assert committed is not None
        assert not committed.state.audit_geometry_errors()


def test_shape_ignores_staging_and_global_rotation():
    for phase in ('rigid','flex'):
        state=aromatic_fixture(3);state.stage=phase;choose_anchor(state)
        expected=shape_quality(state)
        stage_disconnected_components(state)
        assert shape_quality(state)==pytest.approx(expected)
        for rotation in range(6):
            trial=state.clone()
            trial.node_axial={n:rotate_axial(p,rotation) for n,p in state.node_axial.items()}
            trial.final_node_positions=dict(trial.node_axial)
            assert shape_quality(trial)==pytest.approx(expected)


def test_invalid_forward_cannot_win_fallback(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    state=aromatic_fixture(1);bad=state.clone();good=state.clone()
    good.stage='valid'
    controller=StageController(AssemblyConfig(verbose=False,forward_max_rollbacks=1))
    choices=iter([(bad,{}),(good,{})])
    monkeypatch.setattr(controller,'search',lambda *a,**kw:next(choices))
    monkeypatch.setattr(controller,'reward',lambda s,f,**kw:{'total':.5 if s.stage=='valid' else .9})
    monkeypatch.setattr(module,'fast_rollout_remaining',lambda s:SimpleNamespace(to_dict=lambda:{
        'forward_completion':.7 if s.stage=='valid' else .9,
        'audit':{'legal':s.stage=='valid'}}))
    selected,info=controller.checkpoint(state,'rigid')
    assert selected is good
    assert info['retry_selection']['fallback']
    assert info['retry_selection']['selected_attempt']==1
    assert len(controller.forward_cache)==1


def test_hull_regions_distinguish_crossing_from_inside():
    hull=convex_hull([axial_to_cartesian(p) for p in ((-2,-2),(2,-2),(2,2),(-2,2))])
    assert hull_region(hull,[(0,0),(1,0)])=='inside'
    assert hull_region(hull,[(0,0),(3,0)])=='crossing'
    assert hull_region(hull,[(3,0),(4,0)])=='outside'


def test_checkpoint_rejects_all_invalid_forward_results(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    state = aromatic_fixture(1)
    controller = StageController(AssemblyConfig(verbose=False, forward_max_rollbacks=0))
    monkeypatch.setattr(controller, 'search', lambda *args, **kwargs: (state, {}))
    monkeypatch.setattr(controller, 'reward', lambda *args, **kwargs: {'total': 1.0})
    monkeypatch.setattr(module, 'fast_rollout_remaining', lambda s: SimpleNamespace(
        to_dict=lambda: {'forward_completion': 1.0, 'audit': {'legal': False}}))
    with pytest.raises(RuntimeError, match='audit failure'):
        controller.checkpoint(state, 'rigid')
    assert not controller.checkpoints


def test_transition_preserves_mapping_iteration_order():
    state = aromatic_fixture(1)
    child = state.clone()
    child.node_axial = dict(reversed(list(child.node_axial.items())))
    child.action_history = [{'nested': {'a': 1, 'b': 2}}]
    state.action_history = [{'nested': {'b': 2, 'a': 1}}]
    restored = replay_transition(state, encode_transition(state, child))
    assert list(restored.node_axial) == list(child.node_axial)
    assert list(restored.action_history[0]['nested']) == ['a', 'b']


def test_forward_progress_is_observational_and_covers_all_stages():
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    state = ring_side_fixture()
    signature = state.state_signature()
    events = []
    result = fast_rollout_remaining(state, progress=events.append)
    assert [(e['phase'], e['event']) for e in events] == [
        (phase, event) for phase in ('flex', 'ring', 'side', 'substitution')
        for event in ('start', 'no_actions')]
    assert result.to_dict() == fast_rollout_remaining(state).to_dict()
    assert state.state_signature() == signature


def test_ym3_keeps_flex_and_ring_in_separate_inventory():
    from pathlib import Path
    from RL_MTCS import MCTSRunner
    directory = Path(__file__).resolve().parents[1] / 'test_results/YM-3/final_outputs'
    if not directory.is_dir():
        pytest.skip('local YM-3 data unavailable')
    runner = MCTSRunner(directory / 'final_nodes_by_global_id.csv',
        config=AssemblyConfig(verbose=False),
        topology_specs_csv=directory / 'final_topology_specs.csv')
    state = runner.initial_state
    assert len(state.unplaced_units('flex')) == 15
    assert len(state.unplaced_units('ring')) == 12
    diagnostics = dict(state.problem.diagnostics)
    assert int(diagnostics['flex_contract_count']) == 31
    assert int(diagnostics['ring_contract_count']) == 24
    assert state.audit()['geometry_valid']


def test_one_candidate_has_one_actual_geometry_audit(monkeypatch):
    import RL_MTCS.validation as module
    original=module.actual_geometry_errors;calls=[]
    def audit(s):
        calls.append(len(s.node_axial));return original(s)
    monkeypatch.setattr(module,'actual_geometry_errors',audit)
    state=finish_flex(ring_side_fixture())
    child=next(connection_states(state,'side',1))
    assert calls.count(len(child.node_axial))==1
