from itertools import islice
"""Three-aromatic flexible trees must never become two-site ring attachments."""
from dataclasses import replace
from pathlib import Path

from RL_MTCS.actions import flex
from RL_MTCS.geometry import hex_edge_valid, aromatic_slot_direction
from RL_MTCS.io.input import parse_topology_notation, build_csv_inverse_input
from RL_MTCS.io.layout import _layout_topology_tree
from RL_MTCS.mdp.types import TopologyUnit, UnitNode, UnitEdge, UnitPort, ExternalContract, InverseAssemblyInput
from RL_MTCS.mdp.state import HexAssemblyState
from test_rl_mtcs_sites import ring_side_fixture


def three_terminal_fixture():
    graph = parse_topology_notation('227-269-430(-0-113)-271-228')
    types = {227:11,269:23,430:20,0:0,113:9,271:23,228:11}
    coords = _layout_topology_tree(graph, types)
    anchors = (227,113,228)
    body = (269,430,0,271)
    ports, contracts, units = [], [], []
    template = ring_side_fixture().unit('aromatic:0')
    planned = {n:types[n] for n in body}
    for i,(boundary,inner) in enumerate(zip(anchors,(269,0,271))):
        uid = f'aromatic:{i}'
        nodes = tuple(replace(n,node_id=1000+i*10+n.node_id) for n in template.nodes)
        edges = tuple(replace(e,source_node_id=1000+i*10+e.source_node_id,
                             target_node_id=1000+i*10+e.target_node_id) for e in template.internal_edges)
        aromatic = replace(template, unit_id=uid,nodes=nodes,internal_edges=edges,ports=())
        target=nodes[0].node_id
        cid=f'flex:{i}'
        pid=f'body:{i}'
        ports.append(UnitPort(pid,inner,'topology_boundary','flex',
                     (coords[boundary][0]-coords[inner][0],coords[boundary][1]-coords[inner][1])))
        aromatic=replace(aromatic,ports=(UnitPort(f'aryl:{i}',target,'aromatic_boundary','flex',
                          aromatic_slot_direction(aromatic,target)),))
        units.append(aromatic)
        planned.update({n.node_id:13 for n in nodes});planned[target]=types[boundary]
        contracts.append(ExternalContract(cid,'topology:0',inner,uid,target,'flex',
                                          source_port_id=pid,target_port_id=f'aryl:{i}'))
    unit=TopologyUnit('topology:0','flex_chain',tuple(UnitNode(n,types[n],coords[n]) for n in body),
          tuple(UnitEdge(a,b) for a,b in graph.edges if a in body and b in body),tuple(ports),
          metadata=(('geometry_source','generated_from_connectivity'),))
    units.append(unit)
    state=HexAssemblyState(InverseAssemblyInput(tuple(units),tuple(contracts),tuple(planned.items()),
                            tuple((n.node_id,u.unit_id) for u in units for n in u.nodes)))
    for i in range(3):
        assert state.place_unit(f'aromatic:{i}',0,(30*i,0))
    return state,graph,coords


def anchored(state):
    trial=state.clone()
    ids=set(trial.unit('aromatic:0').node_map())
    trial.occupied_axial={p:n for n,p in trial.node_axial.items() if n in ids}
    return flex._SiteSearchCandidate(trial,frozenset(),{},frozenset({0}),[])


def test_ym_branch_coordinates_preserved():
    state,graph,coords=three_terminal_fixture()
    assert coords[0]==(3,1) and coords[113]==(4,1)
    assert all(hex_edge_valid(coords[a],coords[b],honeycomb=True) for a,b in graph.edges)
    assert len(flex._flex_port_groups(state,'topology:0'))==3


def test_three_distinct_components_and_atomic_rollback():
    from RL_MTCS.actions.conformers import iter_topology_conformers
    state,_,_=three_terminal_fixture()
    probe=anchored(state)
    components=flex._rigid_components(state)
    outcomes=[]
    for variant in iter_topology_conformers(state.unit('topology:0')):
        probe=anchored(state)
        probe.state.select_conformer(variant)
        signature=probe.state.state_signature()
        outcomes=list(islice(flex._backbone_site_outcomes(probe, 'topology:0', [1, 2], components), 4))
        assert probe.state.state_signature()==signature
        if outcomes:
            break
    assert outcomes
    assert probe.state.state_signature()==signature
    for child in outcomes:
        assert child.active_component_indices=={0,1,2}
        assert len(child.actions)==1
        assert len(child.state.satisfied_contracts)==3
        assert child.state.component_count()==1
        assert len(child.actions[0]['component_moves'])==2
        assert len(child.boundary_site_ids)==3
        assert not child.state.audit_geometry_errors()
    # Block every physical port of the final component: no partial commit.
    blocked=frozenset(state.unit('aromatic:2').node_map())
    failed=replace(probe,used_site_ids=blocked)
    assert not list(islice(flex._backbone_site_outcomes(failed, 'topology:0', [1, 2], components), 4))
    assert failed.state.state_signature()==signature
    assert not failed.state.satisfied_contracts
    assert not list(islice(flex._backbone_site_outcomes(probe, 'topology:0', [1, 1], components), 4))

    # Keep two aromatic poses fixed and attach just the third component.
    solved=outcomes[0].state
    mixed=HexAssemblyState(probe.state.problem)
    for i in range(3):
        placement=solved.placements[f'aromatic:{i}']
        assert mixed.place_unit(f'aromatic:{i}',placement.rotation,placement.translation,
                                reflection=placement.reflection)
    mixed.occupied_axial={p:n for n,p in mixed.node_axial.items()
                          if mixed.problem.ownership_map()[n]!='aromatic:2'}
    candidate=flex._SiteSearchCandidate(mixed,frozenset(),{},frozenset({0,1}),[])
    matches=list(islice(flex._backbone_site_outcomes(candidate, 'topology:0', [2], components), 4))
    assert matches
    assert all(len(c.actions[0]['component_moves'])==1 and len(c.state.satisfied_contracts)==3 for c in matches)


def test_puct_connects_all_three_terminals():
    from RL_MTCS.config import AssemblyConfig
    state,_,_=three_terminal_fixture()
    from RL_MTCS.policy.checkpoints import StageController
    state,info=StageController(AssemblyConfig(
        connection_candidate_limit=2,forward_max_rollbacks=0,verbose=False)).run(state,'flex')
    assert info['done']
    assert state.component_count()==1
    action=next(a for a in state.action_history if a.get('topology_unit_id')=='topology:0')
    assert action['merged_component_count']==3 and len(action['contract_ids'])==3


def test_three_terminal_priority_and_proximity_independence():
    state,_,_=three_terminal_fixture()
    original=state.unit('topology:0')
    # A virtual proximity change must not alter endpoint ownership semantics.
    changed=replace(original,ports=tuple(replace(p,direction=(1,0)) for p in original.ports))
    double=replace(original,unit_id='topology:1',ports=original.ports[:2])
    trial=HexAssemblyState(replace(state.problem,units=(*state.problem.units[:-1],changed,double)))
    assert len(flex._flex_port_groups(trial,'topology:0'))==3
    assert flex._topology_complexity_key(trial.unit('topology:0'))<flex._topology_complexity_key(trial.unit('topology:1'))


def test_ym_failed_export_snapshot_loads():
    import pytest
    root=Path(__file__).resolve().parents[1]/'test_results/YM-2'
    paths=sorted(root.glob('.final_outputs-*.tmp/final_topology_specs.csv'))
    if not paths:
        pytest.skip('local YM export snapshot unavailable')
    problem=build_csv_inverse_input(paths[-1].with_name('final_nodes_by_global_id.csv'),paths[-1],
                                    use_authoritative_plan=False)
    unit=next(u for u in problem.units if set((269,430,0,271))<=set(u.node_map()))
    assert unit.product_type=='flex_chain' and len(unit.ports)==3
