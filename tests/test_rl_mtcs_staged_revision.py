"""Acceptance tests for complete lookahead and checkpoint-local MCTS replay."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.policy.mcts import MCTS
from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.mdp.components import finish_flex, phase_of
from RL_MTCS.geometry import outside_hull
from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.reporting.completion import completion_report, ensure_inventory
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.geometry import honeycomb_ball, axial_distance
from RL_MTCS.rewards.scoring import su13_clearance_profile
from test_rl_mtcs_sites import ring_side_fixture, aromatic_fixture
from test_rl_mtcs_three_terminal import three_terminal_fixture


def config(**kwargs):
    return AssemblyConfig(verbose=False,save_stage_images=False,connection_candidate_limit=2,
                          **kwargs)


def test_honeycomb_clearance_layers_and_occupied_sites():
    layers = [set(honeycomb_ball((1,0),r)) for r in (1,2,3)]
    assert [len(layers[0]),len(layers[1]-layers[0]),len(layers[2]-layers[1])] == [3,9,12]
    assert all((q+r)%3 for q,r in layers[-1])
    state = ring_side_fixture()
    result = su13_clearance_profile(state)
    row = result['sites'][0]
    cells = set(honeycomb_ball(state.node_axial[row['site_id']],3))
    assert row['clearance_cells'] == len(cells-set(state.occupied_axial))
    assert row['theoretical_cells'] == 24


def test_exhausted_tree_selects_partial_by_measured_reward_and_excludes_whole_result():
    tree = {():[(1,),(2,)],(1,):[(1,3)],(2,):[(2,3),(2,4)]}
    values = {(1,3):.2,(2,3):.9,(2,4):.5}
    def search(excluded=()):
        result = MCTS((),expand=lambda s:iter(tree.get(s,[])),
            accept=lambda s:False,signature=lambda s:s,evaluate=lambda s:values.get(s,0),excluded=excluded)
        chosen=result.run()
        assert result.diagnostics()['search_exhausted']
        return chosen
    assert search()==(2,3)
    assert search({(2,3)})==(2,4)  # Reuse individual action 2 in a different combination.
    assert search(set(values)) is None


def test_complete_rollout_keeps_original_state_and_separates_ring():
    state = ring_side_fixture()
    sig = state.state_signature()
    result = fast_rollout_remaining(state)
    assert state.state_signature() == sig
    assert result.forward_completion == 1
    assert result.report['stages']['flex']['total'] == 0
    assert result.report['stages']['ring']['completed'] == 1
    assert result.report['stages']['side']['completed'] == 1
    assert result.audit['geometry_valid']
    formal,_ = StageController(config(forward_max_rollbacks=0)).run(state,'flex')
    assert not formal.is_placed('topology:0')
    for phase in ('ring','side'):
        formal = next(connection_states(formal,phase,1))
    assert formal.node_axial == result.state.node_axial


def test_three_terminal_merge_is_atomic_and_never_early_isolated():
    state = three_terminal_fixture()[0]
    sig = state.state_signature()
    child = next(connection_states(state,'flex',1))
    assert state.state_signature()==sig
    assert child.component_count()==1
    assert len(child.satisfied_contracts)==3
    assert not child.excluded_unit_ids
    assert child.action_history[-1]['merged_component_count']==3
    assert fast_rollout_remaining(state).forward_completion==1


def test_only_finished_flex_excludes_clusters_and_preserves_denominator(tmp_path):
    state = aromatic_fixture(3)
    # No chains; rigid clusters are all still resources until flex finishes.
    ensure_inventory(state)
    assert len(state.placements)==3
    final = finish_flex(state)
    report = completion_report(final)
    assert report['total']==3 and report['completed']==1
    assert final.excluded_unit_ids=={'aromatic:1','aromatic:2'}
    assert len(final.problem.units)==3
    from RL_MTCS.runner import MCTSRunner
    runner=MCTSRunner.__new__(MCTSRunner)
    # Restrict to the already finished partial structure for downstream export.
    runner.initial_state=final;runner.config=config(forward_max_rollbacks=0);runner.nmr_scorer=None
    from RL_MTCS.io.output import export_final_state,composition_report
    export_final_state(final,tmp_path)
    summary=composition_report(final,state.problem,output_dir=tmp_path)
    assert summary['unconnected_node_ids']
    assert (tmp_path/'final_nodes.csv').exists()
    assert report['completion']==pytest.approx(1/3)


class CounterState:
    def __init__(self,n=0,path=()):
        self.n,self.path,self.stage=n,path,'flex'
        from RL_MTCS.mdp.types import InverseAssemblyInput
        self.source_problem = InverseAssemblyInput((),(),(),())
        self.placements = {}
    def clone(self):
        return CounterState(self.n,self.path)
    def state_signature(self):
        return self.n,self.path
    def forward_signature(self):
        return self.state_signature()


@pytest.mark.parametrize('count,expected',[(0,[0]),(3,[3]),(4,[4]),(7,[4,7]),(8,[4,8])])
def test_checkpoint_schedule_including_zero_and_short_final(monkeypatch,count,expected):
    import RL_MTCS.policy.checkpoints as module
    controller=StageController(config())
    monkeypatch.setattr(module,'ensure_inventory',lambda s:None)
    monkeypatch.setattr(module,'finish_flex',lambda s:s)
    monkeypatch.setattr(module,'connection_states',lambda s,*a,**k:iter([CounterState(s.n+1)]) if s.n<count else iter(()))
    monkeypatch.setattr(module,'completion_report',lambda s:dict(stages={'flex':dict(completed=s.n,total=count)},all_complete=s.n==count))
    monkeypatch.setattr(CounterState,'audit',lambda s:{'legal':True},raising=False)
    monkeypatch.setattr(CounterState,'main_anchor',None,raising=False)
    monkeypatch.setattr(CounterState,'excluded_unit_ids',set(),raising=False)
    seen=[]
    def checkpoint(s,phase):
        n=min(count,s.n+4);seen.append(n)
        return CounterState(n),{'rewards':{'total':.5}}
    monkeypatch.setattr(controller,'checkpoint',checkpoint)
    controller.run(CounterState(),'flex')
    assert seen==expected


@pytest.mark.parametrize('phase',['rigid','flex'])
def test_retry_restores_checkpoint_and_excludes_groups_with_weighted_fallback(monkeypatch,phase):
    import RL_MTCS.policy.checkpoints as module
    controller=StageController(config(forward_max_rollbacks=4))
    starts=[];rollouts=[]
    def search(initial,phase,excluded,**kwargs):
        starts.append(initial.state_signature())
        assert excluded=={(8,(i,)) for i in range(len(starts)-1)}
        return CounterState(8,(len(starts)-1,)),{'search_exhausted':True}
    def rollout(s):
        rollouts.append(s.path)
        return SimpleNamespace(to_dict=lambda:dict(forward_completion=.5,audit={'legal':True}))
    monkeypatch.setattr(controller,'search',search)
    monkeypatch.setattr(controller,'count',lambda s,p:s.n)
    monkeypatch.setattr(controller,'reward',lambda s,f,**kw:dict(total=1-abs(s.path[0]-2)/10))
    monkeypatch.setattr(module,'fast_rollout_remaining',rollout)
    chosen,info=controller.checkpoint(CounterState(4,('accepted',)),phase)
    assert starts==[(4,('accepted',))]*5
    assert chosen.path==(2,) and info['retry_selection']['selected_attempt']==2
    assert len(rollouts)==5


def test_threshold_accepts_immediately_and_cache_uses_state_not_count(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    controller=StageController(config())
    chosen=CounterState(4,('a',));calls=[]
    monkeypatch.setattr(controller,'search',lambda *a,**kw:(chosen,{}))
    monkeypatch.setattr(controller,'count',lambda s,p:s.n)
    monkeypatch.setattr(controller,'reward',lambda *a,**kw:dict(total=.6))
    monkeypatch.setattr(module,'fast_rollout_remaining',lambda s:(calls.append(s.path) or SimpleNamespace(
        to_dict=lambda:dict(forward_completion=.9,audit={'legal':True}))))
    controller.checkpoint(CounterState(),'flex')
    controller.checkpoint(CounterState(),'flex')
    chosen=CounterState(4,('b',))
    controller.checkpoint(CounterState(),'flex')
    assert calls==[('a',),('b',)]


def test_no_new_result_uses_best_evaluated_and_propagates_program_errors(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    controller=StageController(config())
    calls=[]
    def search(*args,**kwargs):
        calls.append(1)
        return (CounterState(4),{}) if len(calls)==1 else (None,{})
    monkeypatch.setattr(controller,'search',search)
    monkeypatch.setattr(controller,'count',lambda s,p:s.n)
    monkeypatch.setattr(controller,'reward',lambda *a,**kw:dict(total=.6))
    monkeypatch.setattr(module,'fast_rollout_remaining',lambda s:SimpleNamespace(
        to_dict=lambda:dict(forward_completion=.1,audit={'legal':True})))
    chosen,info=controller.checkpoint(CounterState(),'flex')
    assert chosen.n==4 and len(info['retry_selection']['attempts'])==1
    import RL_MTCS.policy.rollout as fast
    def broken(*a,**k):
        raise RuntimeError('programming error')
    monkeypatch.setattr(fast,'connection_states',broken)
    with pytest.raises(RuntimeError,match='programming error'):
        fast.fast_rollout_remaining(ring_side_fixture())


def test_legal_candidate_limit_does_not_stop_after_invalid_positions(monkeypatch):
    import RL_MTCS.actions.rigid as rigid
    state=aromatic_fixture(2)
    from RL_MTCS.actions.catalog import connection_states
    calls=[]
    original = rigid.iter_rigid_outcomes
    def outcomes(s, **kwargs):
        legal = next(original(s, **kwargs))
        for i in range(11):
            child=legal.clone()
            # Ten colliding proposals, followed by a fully legal state.
            if i<10:
                gids=list(child.node_axial)
                child.node_axial[gids[0]]=child.node_axial[gids[1]]
                child.final_node_positions[gids[0]]=child.final_node_positions[gids[1]]
            calls.append(i)
            yield child
    monkeypatch.setattr(rigid,'iter_rigid_outcomes',outcomes)
    assert len(list(connection_states(state,'rigid',1)))==1
    assert len(calls)==11


def test_ring_outside_backbone_hull_and_main_anchor_stable():
    state=finish_flex(ring_side_fixture())
    hull=state.backbone_hull
    child=next(connection_states(state,'ring',1))
    assert child.backbone_hull==hull
    assert child.main_anchor==state.main_anchor
    assert all(outside_hull(hull,child.node_axial[n.node_id]) for n in child.unit('topology:0').nodes)


def test_runner_exports_separate_partial_capable_stages(tmp_path):
    from RL_MTCS.runner import MCTSRunner
    runner=MCTSRunner.__new__(MCTSRunner)
    runner.initial_state=ring_side_fixture();runner.config=config();runner.nmr_scorer=None
    result=runner.run_all_stages(tmp_path)
    assert result['ok'] and result['all_complete']
    for phase in ('rigid','flex','ring','side','substitution'):
        summary=json.loads((tmp_path/phase/f'{phase}_summary.json').read_text())
        assert summary['legal'] and summary['stage_no_actions']
    assert result['stages']['flex']['result']['units_total']==0
    assert result['stages']['ring']['result']['units_total']==1


def chain_inventory_fixture(chains=4,clusters=4):
    from RL_MTCS.mdp.types import TopologyUnit,UnitNode,UnitPort,ExternalContract,InverseAssemblyInput
    from RL_MTCS.geometry import aromatic_slot_direction
    from RL_MTCS.mdp.state import HexAssemblyState
    template=ring_side_fixture().unit('aromatic:0')
    aromatic=[]
    for i in range(clusters):
        offset=10*i
        aromatic.append(replace(template,unit_id=f'aromatic:{i}',ports=(),
            nodes=tuple(replace(n,node_id=n.node_id+offset) for n in template.nodes),
            internal_edges=tuple(replace(e,source_node_id=e.source_node_id+offset,target_node_id=e.target_node_id+offset)
                                 for e in template.internal_edges)))
    units=list(aromatic);contracts=[];planned={n.node_id:13 for u in units for n in u.nodes}
    slots=[(i,n.node_id) for i,u in enumerate(aromatic) for n in u.nodes]
    ports={i:[] for i in range(clusters)}
    for j in range(chains):
        uid=f'chain:{j}'
        body=100+j
        chain_ports=[]
        for endpoint,direction in enumerate(((-1,-1),(1,0))):
            i,boundary=slots[2*j+endpoint]
            pid=f'chain:{j}:{endpoint}'
            ports[i].append(UnitPort(pid,boundary,'aromatic_boundary','flex',aromatic_slot_direction(aromatic[i],boundary)))
            chain_ports.append(UnitPort(pid,body,'topology_boundary','flex',direction))
            contracts.append(ExternalContract(pid,uid,body,aromatic[i].unit_id,boundary,'flex',
                                              source_port_id=pid,target_port_id=pid))
            planned[boundary]=11
        units.append(TopologyUnit(uid,'flex_chain',(UnitNode(body,23,(0,0)),),ports=tuple(chain_ports)))
        planned[body]=23
    units[:clusters]=[replace(u,ports=tuple(ports[i])) for i,u in enumerate(aromatic)]
    state=HexAssemblyState(InverseAssemblyInput(tuple(units),tuple(contracts),tuple(planned.items()),
                              tuple((n.node_id,u.unit_id) for u in units for n in u.nodes)))
    for i,u in enumerate(aromatic):assert state.place_unit(u.unit_id,0,(i*30,0))
    return state


def test_four_remaining_chains_three_outside_clusters_continue_before_isolation():
    state=chain_inventory_fixture()
    result=fast_rollout_remaining(state)
    report=result.report
    assert report['stages']['rigid']['completed']==4
    assert not result.state.excluded_unit_ids
    actions=[a for a in result.state.action_history if a.get('stage')=='flex']
    assert all(a.get('merged_component_count',0)>=2 for a in actions[:3])
    assert result.state.component_count()==1
    assert next(connection_states(result.state,'flex',1),None) is None
    assert len(actions)==4
    assert actions[-1]["connection_mode"] == "internal"
    assert report["stages"]["flex"]["completed"]==4


def test_skip_zero_domain_chain_and_recheck_after_state_change(monkeypatch):
    import RL_MTCS.actions.catalog as connections
    state=chain_inventory_fixture(chains=2,clusters=3)
    real_outcomes=connections.iter_flex_outcomes
    visited=[]
    def outcomes(s,u,merge,**kwargs):
        visited.append((u.unit_id,merge))
        if u.unit_id=='chain:0':return
        yield from real_outcomes(s,u,merge,**kwargs)
    monkeypatch.setattr(connections,'iter_flex_outcomes',outcomes)
    child=next(connection_states(state,'flex',1))
    assert child.is_placed('chain:1') and not child.is_placed('chain:0')
    assert visited[:2]==[('chain:0',True),('chain:1',True)]


@pytest.mark.parametrize('target_su,ring_size',[(26,6),(26,5),(30,5)])
def test_rollout_includes_substitution_and_audits_actual_positions(target_su,ring_size):
    from RL_MTCS.mdp.types import HeteroAromaticRequest,InverseAssemblyInput
    from RL_MTCS.mdp.state import HexAssemblyState
    template=replace(ring_side_fixture().unit('aromatic:0'),ports=())
    target=template.nodes[0].node_id
    synthetic=template.nodes[1].node_id if ring_size==5 else None
    request=HeteroAromaticRequest(target_su,ring_size,target,synthetic)
    problem=InverseAssemblyInput((template,),(),tuple((n.node_id,target_su if n.node_id==target else 13) for n in template.nodes),
                                 tuple((n.node_id,template.unit_id) for n in template.nodes),hetero_requests=(request,))
    state=HexAssemblyState(problem)
    assert state.place_unit(template.unit_id,0,(0,0))
    sig=state.state_signature()
    result=fast_rollout_remaining(state)
    assert state.state_signature()==sig
    assert result.report['stages']['substitution']['completed']==1
    assert result.forward_completion==1
    assert result.state.is_complete() and result.audit['complete']
    from RL_MTCS.io.output import composition_report
    assert composition_report(result.state, problem, state)['assembly_complete']
    assert result.state.current_su_types[target]==target_su
    assert result.audit['geometry_valid']
    assert result.report['total']==2
    assert not su13_clearance_profile(result.state)['sites'] or all(
        row['site_id']!=target for row in su13_clearance_profile(result.state)['sites'])


def test_only_selected_mcts_result_gets_forward_evaluation(monkeypatch):
    import RL_MTCS.policy.checkpoints as stages
    controller=StageController(config(forward_max_rollbacks=0))
    calls=[]
    original=stages.fast_rollout_remaining
    def rollout(s):
        calls.append(s.state_signature())
        return original(s)
    monkeypatch.setattr(stages,'fast_rollout_remaining',rollout)
    _,info=controller.run(aromatic_fixture(3),'rigid')
    assert info['retry_selection']['attempts'][0]['search']['terminal_count']>1
    assert len(calls)==1


def test_zero_flex_final_checkpoint_reuses_identical_rigid_forward(monkeypatch):
    import RL_MTCS.policy.checkpoints as stages
    calls=[]
    original=stages.fast_rollout_remaining
    def rollout(s):
        calls.append(s.stage)
        return original(s)
    monkeypatch.setattr(stages,'fast_rollout_remaining',rollout)
    controller=StageController(config(forward_max_rollbacks=0))
    rigid,_=controller.run(ring_side_fixture(),'rigid')
    flex,_=controller.run(rigid,'flex')
    assert calls==['rigid']
    assert len(controller.checkpoints)==2
    assert controller.checkpoints[-1]['start_count']==controller.checkpoints[-1]['end_count']==0


def test_actual_coordinate_audit_detects_crossing_and_node_on_edge():
    from RL_MTCS.mdp.types import AssemblyEdge
    from RL_MTCS.validation import actual_geometry_errors
    state=ring_side_fixture()
    ids=sorted(state.node_axial)
    state.final_node_positions={n:(10+i,20+i) for i,n in enumerate(ids)}
    state.final_node_positions.update(dict(zip(ids[:4],[(0,0),(2,2),(0,2),(2,0)])))
    state.edges=[AssemblyEdge(ids[0],ids[1],'aromatic'),AssemblyEdge(ids[2],ids[3],'aromatic')]
    assert any(e['reason']=='edge_intersection' for e in actual_geometry_errors(state))
    state.final_node_positions[ids[4]]=(1,1)
    assert any(e['reason']=='node_on_edge' for e in actual_geometry_errors(state))


def test_rigid_contract_completion_does_not_require_one_component():
    from RL_MTCS.mdp.types import UnitPort
    from RL_MTCS.geometry import aromatic_slot_direction
    from RL_MTCS.mdp.state import HexAssemblyState
    from RL_MTCS.mdp.sites import aromatic_perimeter_ids
    initial = aromatic_fixture(4)
    units, types = [], {}
    for unit in initial.problem.units:
        site = min(aromatic_perimeter_ids(unit))
        port = UnitPort(f'rigid:{site}',site,'rigid','rigid',
                        aromatic_slot_direction(unit,site))
        units.append(replace(unit, ports=(port,)))
        types.update({n.node_id:10 if n.node_id == site else n.su_type for n in unit.nodes})
    initial = HexAssemblyState(replace(initial.problem,units=tuple(units),
                                      node_su_types=tuple(sorted(types.items()))))
    for i,unit in enumerate(units):
        assert initial.place_unit(unit.unit_id,0,(i*99,0))
    state,info = StageController(config(forward_max_rollbacks=0)).run(initial,'rigid')
    assert state.component_count() == 2
    assert info['status'] == 'complete'
    assert info['contracts_done'] == info['contracts_total'] == 2
    assert info['units_done'] == info['units_total'] == 4
    assert info['completion_report']['stages']['rigid']['completed'] == max(
        len(state.component_unit_ids(uid)) for uid in state.placements)
    assert info['completion_report']['total'] == 4
    assert not info['all_complete']
