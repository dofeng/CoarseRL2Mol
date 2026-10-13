"""Geometry regression fixtures are synthetic; the invalid HSQ input is not patched."""
import json
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RL_MTCS.io.input import _make_cluster_template, initialize, build_csv_inverse_input
from RL_MTCS.mdp.types import InverseAssemblyInput, UnitNode, UnitPort, UnitEdge, TopologyUnit, InverseInputError, ExternalContract
from RL_MTCS.geometry import aromatic_slot_direction
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.sites import aromatic_template_errors, aromatic_perimeter_ids
from RL_MTCS.actions.rigid import bind_su10_sites
from RL_MTCS.actions.rigid_geometry import RigidPoseGenerator
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.io.output import export_rigid_views

HSQ = Path(__file__).resolve().parents[1] / 'test_results/HSQ-1/outer_cycle_1/final_outputs'


def synthetic_hsq_fixture():
    """Large topology test fixture, NOT a repair of the HSQ inverse snapshot.

    Remove the five-member substitution request and its temporary occupant,
    assigning the pendant SU10 to that slot. This explicitly changes staging
    semantics for tests only; production rejects this inconsistent input.
    """
    raw = build_csv_inverse_input(HSQ / 'final_nodes_by_global_id.csv', HSQ / 'final_topology_specs.csv')
    unit = raw.units[0]
    dummy = next(n for n in unit.nodes if n.node_id == -1000000)
    nodes = tuple(replace(n, local_axial=dummy.local_axial) if n.node_id == 78 else n
                  for n in unit.nodes if n.node_id != dummy.node_id)
    edges = tuple(replace(e, source_node_id=78 if e.source_node_id == dummy.node_id else e.source_node_id,
                           target_node_id=78 if e.target_node_id == dummy.node_id else e.target_node_id)
                  for e in unit.internal_edges if {e.source_node_id, e.target_node_id} != {22, 78})
    unit = replace(unit, nodes=nodes, internal_edges=edges)
    unit = replace(unit, ports=tuple(replace(p, direction=aromatic_slot_direction(unit, p.node_id)) for p in unit.ports))
    problem = replace(raw, units=(unit, *raw.units[1:]),
                      node_su_types=tuple(row for row in raw.node_su_types if row[0] != dummy.node_id),
                      ownership=tuple(row for row in raw.ownership if row[0] != dummy.node_id),
                      hetero_requests=tuple(r for r in raw.hetero_requests if r.synthetic_node_id is None))
    state = HexAssemblyState(problem)
    for i, unit in enumerate(problem.units):
        if unit.phase == 'aromatic':
            assert state.place_unit(unit.unit_id, 0, (i * 99, 0))
    assert not aromatic_template_errors(problem)
    return {'state': state, 'units': list(problem.units), 'problem': problem,
            'contracts': list(problem.contracts),
            'aromatic_units': [u for u in problem.units if u.phase == 'aromatic'],
            'topology_units': [u for u in problem.units if u.phase != 'aromatic']}


def aromatic_fixture(count=5):
    units, types = [], {}
    for i in range(count):
        template = _make_cluster_template('chrysene' if i == 0 else 'benzene')
        nodes = tuple(UnitNode(i * 100 + j, 12 if shared else 13, tuple(coord))
                      for j, (coord, shared) in enumerate(zip(template.coords, template.shared)))
        unit = TopologyUnit(f'aromatic:{i}', template.kind, nodes,
                            tuple(UnitEdge(nodes[a].node_id, nodes[b].node_id, 'aromatic')
                                  for a, b in template.edges), local_rings=tuple(template.rings))
        # At most two input SU10 per unit, yet the physical center may bind four.
        port_nodes = sorted(aromatic_perimeter_ids(unit))[:2] if i < count - 1 else []
        ports = tuple(UnitPort(f'rigid:{n}', n, 'rigid', 'rigid', aromatic_slot_direction(unit, n))
                      for n in port_nodes)
        units.append(replace(unit, ports=ports))
        types.update({n.node_id: 10 if n.node_id in port_nodes else n.su_type for n in nodes})
    problem = InverseAssemblyInput(tuple(units), (), tuple(sorted(types.items())),
                                   tuple((n.node_id, u.unit_id) for u in units for n in u.nodes))
    state = HexAssemblyState(problem)
    for i, unit in enumerate(units):
        assert state.place_unit(unit.unit_id, 0, (i * 99, 0))
    return state


def test_hsq_bad_ring_site_is_rejected_before_search():
    raw = build_csv_inverse_input(HSQ / 'final_nodes_by_global_id.csv', HSQ / 'final_topology_specs.csv')
    errors = aromatic_template_errors(raw)
    assert any(e.get('node_ids') == [78] for e in errors)
    with pytest.raises(InverseInputError, match='aromatic_nodes_outside_rings'):
        initialize(HSQ / 'final_nodes_by_global_id.csv', HSQ / 'final_topology_specs.csv')


def test_su10_binding_changes_owner_and_preserves_template_and_ids():
    state = aromatic_fixture()
    ports = [n for n, su in state.problem.node_su_types if su == 10]
    center = sorted(aromatic_perimeter_ids(state.unit('aromatic:0')))
    bound = bind_su10_sites(state, dict(zip(ports[:4], center[:4])))
    assert set(bound.problem.node_map()) == set(state.problem.node_map())
    assert all(bound.problem.ownership_map()[n] == 'aromatic:0' for n in ports[:4])
    assert not aromatic_template_errors(bound.problem)
    assert bound.audit()['geometry_valid']
    for before in state.problem.units:
        after = bound.unit(before.unit_id)
        assert before.local_rings == after.local_rings
        assert {n.local_axial for n in before.nodes} == {n.local_axial for n in after.nodes}


def test_one_aromatic_unit_connects_four_distinct_neighbors():
    state = aromatic_fixture()
    original = state.clone()
    ports = [n for n, su in state.problem.node_su_types if su == 10]
    for index, leaf in enumerate(['aromatic:1', 'aromatic:2', 'aromatic:3', 'aromatic:4']):
        used = {n for c in state.problem.contracts for n in (c.source_node_id, c.target_node_id)}
        found = None
        for site in sorted(aromatic_perimeter_ids(state.unit('aromatic:0')) - used):
            for target in sorted(aromatic_perimeter_ids(state.unit(leaf)) - used):
                bound = bind_su10_sites(state, {ports[index * 2]: site, ports[index * 2 + 1]: target})
                a, b = ports[index * 2:index * 2 + 2]
                left, right = bound.unit('aromatic:0'), bound.unit(leaf)
                pa = next(p for p in left.ports if p.node_id == a)
                pb = next(p for p in right.ports if p.node_id == b)
                for action in RigidPoseGenerator(bound)._candidate_for_pair(left, pa, right, pb):
                    trial = bound.clone()
                    action['contract_id'] = f'rigid:dynamic:{min(a,b)}-{max(a,b)}'
                    ok, _ = trial.connect(action)
                    if ok:
                        found = trial
                        break
                if found:
                    break
            if found:
                break
        assert found is not None, f'cannot attach neighbor {index + 1}'
        state = found
    assert state.phase_done('rigid')
    assert len(state.problem.contracts) == 4
    assert state.audit()['geometry_valid']
    assert len(state.component_unit_ids('aromatic:0')) == 5
    assert original.problem.contracts == ()  # Backtracking retains the old binding.
    assert not original.phase_done('rigid')


def test_initial_images_use_only_staging_types(tmp_path, monkeypatch):
    state = aromatic_fixture(2)
    labels = []
    import RL_MTCS.io.output as output
    real = output.save_state_png
    def capture(s, path, **kwargs):
        labels.extend((kwargs.get('su_types') or s.current_su_types).values())
        assert kwargs.get('show_node_ids') is False
        return real(s, path, **kwargs)
    monkeypatch.setattr(output, 'save_state_png', capture)
    export_rigid_views(state, tmp_path, initial=True)
    assert set(labels) <= {12, 13}
    assert len(list(tmp_path.glob('aromatic_*.png'))) == 2


def test_dynamic_site_mcts_selects_one_result():
    from RL_MTCS import AssemblyConfig
    state, info = StageController(AssemblyConfig(
        connection_candidate_limit=2,forward_max_rollbacks=0,verbose=False)).run(aromatic_fixture(3), 'rigid')
    assert isinstance(state, HexAssemblyState)
    assert state.audit()['geometry_valid']
    assert info['done']


def ring_side_fixture():
    template = _make_cluster_template('benzene')
    aromatic = TopologyUnit('aromatic:0', 'benzene',
        tuple(UnitNode(i, 13, tuple(p)) for i, p in enumerate(template.coords)),
        tuple(UnitEdge(a, b, 'aromatic') for a, b in template.edges), local_rings=tuple(template.rings))
    # Explicit exterior path closes a second hexagon via the prescribed
    # aromatic edge (4,5). No edge is inferred from spatial proximity.
    ring = TopologyUnit('topology:0', 'aliphatic_ring',
        tuple(UnitNode(100+i, 23, p) for i, p in enumerate(((0,0),(1,1),(1,2),(0,2)))),
        tuple(UnitEdge(100+i,101+i) for i in range(3)),
        (UnitPort('start',100,'topology_boundary','flex',(-1,0)),
         UnitPort('end',103,'topology_boundary','flex',(-1,-1))),
        metadata=(('ring_geometry_mode','external_contract_ring'),))
    side = TopologyUnit('topology:1','side_chain',(UnitNode(200,22,(0,0)),),
                         ports=(UnitPort('end',200,'topology_boundary','side',(1,1)),))
    contracts = (
        ExternalContract('flex:a','topology:0',100,'aromatic:0',4,'flex',source_port_id='start',target_port_id='a'),
        ExternalContract('flex:b','topology:0',103,'aromatic:0',5,'flex',source_port_id='end',target_port_id='b'),
        ExternalContract('side:c','topology:1',200,'aromatic:0',0,'side',source_port_id='end',target_port_id='c'),
    )
    aromatic = replace(aromatic, ports=tuple(UnitPort(c.target_port_id,c.target_node_id,'aromatic_boundary',c.kind,
                             aromatic_slot_direction(aromatic,c.target_node_id)) for c in contracts))
    units = (aromatic, ring, side)
    types = {n.node_id:n.su_type for u in units for n in u.nodes}
    types.update({4:11,5:11,0:11})
    problem = InverseAssemblyInput(units, contracts, tuple(sorted(types.items())),
                                   tuple((n.node_id,u.unit_id) for u in units for n in u.nodes))
    state = HexAssemblyState(problem)
    assert state.place_unit('aromatic:0',0,(0,0))
    return state


def test_ring_then_side_runs_and_exports_consistent_stages(tmp_path):
    from RL_MTCS.runner import MCTSRunner
    from RL_MTCS import AssemblyConfig
    runner = MCTSRunner.__new__(MCTSRunner)
    runner.initial_state = ring_side_fixture()
    runner.config = AssemblyConfig(connection_candidate_limit=2,forward_max_rollbacks=0,verbose=False)
    runner.nmr_scorer = None
    runner.inverse_result = None
    runner.init_data = {'diagnostics': {'fixture': 'explicit external ring and side'}}
    result = runner.run_all_stages(str(tmp_path))
    assert result['ok'], result.get('result', result.get('audit'))
    assert result['audit']['geometry_valid']
    assert 'subst_png' not in result['outputs']
    assert not (tmp_path/'subst_result.png').exists()
    for name in ('rigid','flex','ring','side','substitution'):
        trace=json.loads((tmp_path/name/f'{name}_search_trace.json').read_text())
        assert trace['actions'] == json.loads(json.dumps(result['stages'][name]['state'].action_history))
        assert 'plan' not in trace and 'rigid_search_attempts' not in trace
        if name in ('rigid','flex'):
            assert trace['checkpoints']
        else:
            diagnostics = trace['diagnostics']
            assert diagnostics['stop_reason'] in {'tree_exhausted', 'result_candidates_ready'}
            assert 1 <= diagnostics['result_candidate_count'] <= 2
    for name in ('flex','ring','side','substitution'):
        assert (tmp_path/name/f'{name}_molecule.png').is_file()
        molecule=json.loads((tmp_path/name/f'{name}_molecule.json').read_text())
        assert molecule['audit']['geometry_valid']
    ring_summary=json.loads((tmp_path/'ring/ring_summary.json').read_text())
    assert ring_summary['units_done'] == 1
    assert result['stages']['side']['result']['units_done'] == 1


def test_flex_driver_does_not_retry_retained_rigid_skeletons(monkeypatch):
    from RL_MTCS import AssemblyConfig
    from RL_MTCS.policy.checkpoints import StageController
    from RL_MTCS.runner import MCTSRunner
    calls=[]
    def run(self,state,phase):
        calls.append((state,phase))
        return state,{'rewards':{'total':.5},'legal':True,'status':'partial'}
    monkeypatch.setattr(StageController,'run',run)
    state=ring_side_fixture()
    runner=MCTSRunner.__new__(MCTSRunner)
    runner.config=AssemblyConfig(verbose=False)
    runner.nmr_scorer=None
    result=runner.run_flex_stage({'state':state})
    assert calls==[(state,'flex')]
    assert result['state'] is state
    assert 'best_candidate' not in result and 'summaries' not in result


def test_puct_can_replace_an_early_dead_end():
    from RL_MTCS.policy.mcts import MCTS
    graph={0:[1,3],1:[2],2:[],3:[4],4:[]}
    tree=MCTS(0,expand=lambda s:iter(graph[s]),accept=lambda s:s==4,
        signature=lambda s:s,evaluate=lambda s:.1*s,seed=7)
    assert tree.run()==4
    assert tree.diagnostics()['visited_states']==5
    assert tree.diagnostics()['retained_nodes']==1
    assert tree.diagnostics()['search_exhausted']


def test_partial_flex_snapshot_commits_actual_contract_endpoints():
    from RL_MTCS.actions.flex import _append_site_contract
    from RL_MTCS.mdp.transition import relabel_aromatic_sites
    state = ring_side_fixture()
    # Deliberately bind the logical start to a different physical ring atom.
    state = relabel_aromatic_sites(state, {4:0,0:4}, [])
    contract = state.problem.contracts[0]
    state.place_unit('topology:0', 0, (2,0), attachment_edges=[(100,0),(103,5)])
    assert state.is_placed('topology:0')
    _append_site_contract(state, contract, 100, 0)
    _append_site_contract(state, state.problem.contracts[1], 103, 5)
    assert state.audit_port_direction_errors()
    committed = relabel_aromatic_sites(state, {4:0,5:5}, [])
    snapshot = committed.clone()
    assert snapshot.audit()['geometry_valid']
    assert not snapshot.audit()['complete']  # Side still pending.
    assert len(snapshot.satisfied_contracts) == 2
