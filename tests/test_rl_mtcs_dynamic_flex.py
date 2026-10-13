from RL_MTCS.policy.rollout import fast_rollout_remaining
from itertools import islice
"""Replay a real retained rigid layout without changing the inverse snapshot."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from RL_MTCS.io.input import initialize
from RL_MTCS.mdp.types import ExternalContract
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.transition import relabel_aromatic_sites
from RL_MTCS.actions import flex

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'test_results/HSQ-3/rl_mtcs_nmr/20260928_193835_040928'
INPUT = ROOT / 'test_results/HSQ-3/outer_cycle_1/final_outputs'


def saved_rigid_state():
    if not (RUN / 'rigid/rigid_molecule.json').is_file():
        pytest.skip('local HSQ-3 regression snapshot unavailable')
    molecule = json.loads((RUN / 'rigid/rigid_molecule.json').read_text())
    state = initialize(INPUT / 'final_nodes_by_global_id.csv',
                       INPUT / 'final_topology_specs.csv')['state']
    slots = {(u.unit_id, tuple(n.local_axial)): n.node_id
             for u in state.problem.units if u.phase == 'aromatic' for n in u.nodes}
    assignments = {int(gid): slots[(row['unit_id'], tuple(row['local_axial']))]
                   for gid, row in molecule['aromatic_site_bindings'].items()}
    state = relabel_aromatic_sites(state, {a:b for a,b in assignments.items() if a != b}, [], stage='rigid')
    contracts = []
    for row in molecule['rigid_contracts']:
        ports = [next(p.port_id for p in state.unit(row[f'{end}_unit_id']).ports
                      if p.node_id == row[f'{end}_node_id'] and p.phase == 'rigid')
                 for end in ('source', 'target')]
        contracts.append(ExternalContract(
            **{k:v for k,v in row.items() if k != 'satisfied'}, kind='rigid',
            source_port_id=ports[0], target_port_id=ports[1]))
    state = HexAssemblyState(replace(state.problem, contracts=(*state.problem.contracts, *contracts)))
    contacts = [(c.source_node_id,c.target_node_id) for c in contracts]
    for uid, row in molecule['placements'].items():
        assert state.place_unit(uid, row['rotation'], tuple(row['translation']),
                                reflection=row['reflection'], attachment_edges=contacts)
    for c in contracts:
        assert state.connect_adjacent_contract(c, 'snapshot_replay')
    assert state.node_axial == {int(n['node_id']):tuple(n['axial']) for n in molecule['nodes']}
    assert state.audit()['geometry_valid']
    return state


def test_static_boundary_ownership_does_not_block_dynamic_backbone():
    state = saved_rigid_state()
    components = flex._rigid_components(state)
    anchor_nodes = {n.node_id for uid in components[0] for n in state.unit(uid).nodes}
    trial = state.clone()
    trial.occupied_axial = {a:n for n,a in trial.node_axial.items() if n in anchor_nodes}
    probe = flex._SiteSearchCandidate(trial, frozenset(), {}, frozenset({0}), [])
    for target in (4, 5):
        outcomes = list(islice(flex._backbone_site_outcomes(probe, 'topology:0', [target], components), 4))
        assert outcomes
        assert all(len(o.state.satisfied_contracts) == 13 for o in outcomes)
        assert all(o.active_component_indices == {0,target} for o in outcomes)


def test_audit_completes_remaining_stages_in_a_copy():
    from test_rl_mtcs_sites import ring_side_fixture
    state=ring_side_fixture()
    signature=state.state_signature()
    audit=fast_rollout_remaining(state).to_dict()
    assert audit['forward_completion']==1
    assert state.state_signature()==signature


def test_no_node_eviction_or_discarded_unexplored_branches():
    from RL_MTCS.policy.mcts import MCTS
    def expand(state):
        depth,branch=state
        if branch or depth==10:return
        for index in range(30):yield depth+1,index
    tree=MCTS((0,0),expand=expand,accept=lambda s:s==(10,0),signature=lambda s:s,
        evaluate=lambda s:s[0]/11 if s[1]==0 else -.8,seed=7)
    assert tree.run()==(10,0)
    assert tree.diagnostics()['visited_states']==301
    assert tree.diagnostics()['retained_nodes']==1


def test_search_retains_ancestors_and_counts_states():
    from RL_MTCS.policy.mcts import MCTS
    tree=MCTS(0,expand=lambda n:iter([n+1]) if n<8 else iter([]),
        accept=lambda n:n==8,signature=lambda n:n,evaluate=lambda n:n/10)
    assert tree.run()==8
    assert tree.diagnostics()['visited_states']==9
    assert tree.diagnostics()['retained_nodes']==1
    assert tree.diagnostics()['transition_bytes']==0


def test_search_has_no_depth_budget():
    from RL_MTCS.policy.mcts import MCTS
    tree=MCTS(0,expand=lambda s:iter([s+1]) if s<120 else iter(()),
        accept=lambda s:s==120,signature=lambda s:s,evaluate=lambda s:s/120)
    assert tree.run()==120
    assert tree.max_depth==120 and tree.diagnostics()['visited_states']==121


def test_stage_reports_legal_partial_separately_from_completion():
    from RL_MTCS.policy.checkpoints import StageController
    from RL_MTCS.config import AssemblyConfig
    from test_rl_mtcs_sites import aromatic_fixture
    state,info=StageController(AssemblyConfig(verbose=False,forward_max_rollbacks=0)).run(aromatic_fixture(3),'flex')
    assert info['legal'] and info['stage_no_actions'] and not info['all_complete']


def test_ring_cannot_merge_components_left_disconnected_by_chains(tmp_path):
    from test_rl_mtcs_sites import ring_side_fixture
    from RL_MTCS.mdp.types import UnitNode, UnitEdge, UnitPort
    from RL_MTCS.runner import MCTSRunner
    from RL_MTCS import AssemblyConfig

    original = ring_side_fixture().problem
    aromatic, ring, side = original.units
    aromatic_units = [aromatic]
    for index in (1, 2):
        offset = index * 1000
        aromatic_units.append(replace(aromatic, unit_id=f'aromatic:{index}', ports=(),
            nodes=tuple(replace(n, node_id=n.node_id+offset) for n in aromatic.nodes),
            internal_edges=tuple(replace(e, source_node_id=e.source_node_id+offset,
                                        target_node_id=e.target_node_id+offset)
                                 for e in aromatic.internal_edges)))
    chain = replace(ring, unit_id='topology:2', product_type='flex_chain',
                    nodes=(UnitNode(300,29,(0,0)),), internal_edges=(),
                    ports=(UnitPort('a',300,'topology_boundary','flex',(-1,-1)),
                           UnitPort('b',300,'topology_boundary','flex',(1,0))), metadata=())
    contracts = list(original.contracts)
    contracts.extend(ExternalContract(f'chain:{p}',chain.unit_id,300,aromatic.unit_id,n,'flex',
                                     source_port_id=p,target_port_id=f'chain:{p}')
                     for p,n in [('a',1),('b',2)])
    aromatic_units[0] = replace(aromatic, ports=(*aromatic.ports,
        UnitPort('chain:a',1,'aromatic_boundary','flex',(-1,0)),
        UnitPort('chain:b',2,'aromatic_boundary','flex',(0,-1))))
    units = (*aromatic_units,ring,side,chain)
    types = {n.node_id:n.su_type for u in units for n in u.nodes}
    types.update({0:11,1:5,2:5,4:11,5:11})
    state = HexAssemblyState(replace(original,units=units,contracts=tuple(contracts),
        node_su_types=tuple(sorted(types.items())),
        ownership=tuple((n.node_id,u.unit_id) for u in units for n in u.nodes)))
    for i,u in enumerate(aromatic_units):
        assert state.place_unit(u.unit_id,0,(i*30,0))
    runner = MCTSRunner.__new__(MCTSRunner)
    runner.initial_state = state
    runner.config = AssemblyConfig(connection_candidate_limit=2,forward_max_rollbacks=0,verbose=False)
    runner.nmr_scorer = None
    runner.inverse_result = None
    runner.init_data = {'diagnostics': {'fixture':'dynamic stage barrier'}}
    result = runner.run_all_stages(str(tmp_path))
    assert result['ok']
    assert result['audit']['component_count'] == 1
    assert result['completion_report']['excluded_units']
    assert (tmp_path/'side').exists()
