from itertools import islice
"""Ring chemistry, scorer coverage and source inventory regression tests."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from RL_MTCS.io.input import initialize, parse_topology_notation
from RL_MTCS.geometry import hex_edge_valid, rotate_axial, coordinate_aspect, hull_region, axial_to_cartesian
from RL_MTCS.actions.conformers import iter_topology_conformers, validate_conformer
from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.actions.ring import iter_ring_outcomes
from RL_MTCS.mdp.components import finish_flex
from RL_MTCS.io.output import composition_report
from RL_MTCS.actions import flex
from test_rl_mtcs_sites import ring_side_fixture


def test_fused_brackets_preserve_shared_edge():
    graph = parse_topology_notation('316-165-130-[-199-322-331-216-]-203-214-289')
    assert (130, 203) in graph.closure_edges
    assert len(graph.edges) == len(graph.nodes)
    assert len(set(graph.nodes)) == 10


def test_hsq_ring_geometry_and_conformers():
    directory = Path(__file__).resolve().parents[1] / 'test_results/HSQ-3/outer_cycle_1/final_outputs'
    if not directory.exists():
        pytest.skip('local inverse snapshot unavailable')
    state = initialize(directory/'final_nodes_by_global_id.csv', directory/'final_topology_specs.csv')['state']
    rings = [u for u in state.problem.units if u.attaches_to_aromatic_edge]
    assert len(rings) == 14
    fused = state.unit('topology:9')
    assert any({e.source_node_id, e.target_node_id} == {130, 203} for e in fused.internal_edges)
    assert len(fused.local_rings) == 1
    for unit in rings:
        variants = list(islice(iter_topology_conformers(unit), 4))
        assert variants
        for variant in variants:
            validate_conformer(unit, variant)
            coords = {n.node_id: n.local_axial for n in variant.nodes}
            anchors = [(coords[p.node_id][0]+p.direction[0], coords[p.node_id][1]+p.direction[1])
                       for p in variant.ports]
            assert hex_edge_valid(*anchors)
            assert {(e.source_node_id,e.target_node_id) for e in variant.internal_edges} == {
                (e.source_node_id,e.target_node_id) for e in unit.internal_edges}


def test_ring_candidates_require_same_aromatic_edge():
    state = ring_side_fixture()
    components = flex._rigid_components(state)
    probe = flex._SiteSearchCandidate(state, frozenset(), {}, frozenset({0}), [])
    poses = list(flex._interior_site_poses(probe, 'topology:0', components))
    assert poses
    assert all(state.aromatic_edge_sites(tuple(p[4])) for p in poses)
    assert list(islice(flex._backbone_site_outcomes(probe, 'topology:0', [1], components), 4)) == []
    aromatic = state.unit('aromatic:0')
    edge = aromatic.internal_edges[0]
    assert state.aromatic_edge_sites((edge.source_node_id, edge.target_node_id))
    nodes = [n.node_id for n in aromatic.nodes]
    nonedge = next((a,b) for a in nodes for b in nodes if a != b
                  and not any({a,b}=={e.source_node_id,e.target_node_id} for e in aromatic.internal_edges))
    assert not state.aromatic_edge_sites(nonedge)


def test_ring_spatial_preferences_share_one_legal_candidate_cap():
    state = finish_flex(ring_side_fixture())
    state.backbone_hull = ((-100., -100.), (100., -100.), (100., 0.), (-100., 0.))
    before = state.state_signature()
    children = list(connection_states(state, 'ring', 10))
    regions = [hull_region(state.backbone_hull,
                          [child.node_axial[n.node_id] for n in child.unit('topology:0').nodes])
               for child in children]
    assert len(children) == 10
    assert set(regions) == {'outside', 'crossing', 'inside'}
    assert regions == sorted(regions, key=('outside', 'crossing', 'inside').index)
    for child in children:
        assert child.audit()['legal']
        assert {'flex:a', 'flex:b'} <= child.satisfied_contracts
        assert child.backbone_hull == state.backbone_hull
        assert child.main_anchor == state.main_anchor
    assert state.state_signature() == before


@pytest.mark.parametrize('region', ['crossing', 'inside'])
def test_ring_fallback_passes_full_legality_checks(region):
    state = finish_flex(ring_side_fixture())
    state.backbone_hull = ((-100., -100.), (100., -100.), (100., 0.), (-100., 0.))
    before = state.state_signature()
    child = next(iter_ring_outcomes(state, state.unit('topology:0'), region=region))
    coordinates = [child.node_axial[n.node_id] for n in child.unit('topology:0').nodes]
    assert hull_region(state.backbone_hull, coordinates) == region
    assert child.audit()['legal']
    assert state.state_signature() == before


@pytest.mark.parametrize('top,region', [(2., 'crossing'), (100., 'inside')])
def test_formal_ring_connection_falls_back_when_no_exterior_candidate(top, region):
    state = finish_flex(ring_side_fixture())
    state.backbone_hull = ((-100., -100.), (100., -100.), (100., top), (-100., top))
    before = state.state_signature()
    child = next(connection_states(state, 'ring', 1))
    coordinates = [child.node_axial[n.node_id] for n in child.unit('topology:0').nodes]
    assert hull_region(state.backbone_hull, coordinates) == region
    assert child.audit()['legal']
    assert {'flex:a', 'flex:b'} <= child.satisfied_contracts
    assert state.state_signature() == before


def test_ring_node_on_hull_boundary_is_not_a_legality_rejection():
    state = finish_flex(ring_side_fixture())
    state.backbone_hull = ((-100., -100.), (100., -100.), (100., -1.), (-100., -1.))
    child = next(iter_ring_outcomes(state, state.unit('topology:0'), region='inside'))
    coordinates = [child.node_axial[n.node_id] for n in child.unit('topology:0').nodes]
    assert max(axial_to_cartesian(p)[1] for p in coordinates) == -1.
    assert hull_region(state.backbone_hull, coordinates) == 'inside'
    assert child.audit()['legal']


def test_principal_axis_metric_is_invariant_under_global_rotation():
    state = ring_side_fixture()
    expected = coordinate_aspect(state.occupied_axial)['principal_axis_ratio']
    for rotation in range(6):
        trial = state.clone()
        trial.occupied_axial = {rotate_axial(p, rotation):gid for p,gid in state.occupied_axial.items()}
        assert coordinate_aspect(trial.occupied_axial)['principal_axis_ratio'] == pytest.approx(expected)


def test_composition_reports_idle_separately_from_missing(tmp_path):
    state = ring_side_fixture()
    source = replace(state.problem, idle_aromatic_nodes=((999,12),))
    report = composition_report(state, source, output_dir=tmp_path)
    assert report['idle_aromatic_count'] == 1
    assert 999 not in report['missing_required_ids']
    assert 100 in report['unconnected_node_ids']
    assert report['composition_reconciled']
    assert (tmp_path/'final_node_disposition.csv').exists()


def fake_scorer():
    import torch
    from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
    scorer = NMRGuidanceScorer.__new__(NMRGuidanceScorer)
    scorer._graph_cache, scorer._node_cache, scorer._records = {}, {}, []
    scorer._stats = defaultdict(int)
    scorer._stage_stats = defaultdict(lambda: defaultdict(int))
    scorer.graph_cache_limit, scorer.record_limit = 4, 20
    descriptor = dict(graph_key='fixture', carbon_count=3, is_carbon=torch.tensor([True]*3),
                      su_indices=torch.tensor([13,23,23]))
    scorer._graph_descriptor = lambda *_args: descriptor
    scorer._root_key = lambda _d, center: ('root',center)
    scorer._predict_missing_roots = lambda *_args, **_kw: None
    scorer._spectrum_metrics = lambda shifts, su: ({'loss':0.25, 'carbon_centers':len(shifts)}, shifts)
    return scorer


def test_nmr_missing_predictions_never_become_zero_loss_or_fake_peaks():
    scorer = fake_scorer()
    scorer._node_cache[('root',0)] = 100.0
    result = scorer.score_candidates([object()], stage='flex_chain', depth=1)[0]
    assert not result['scored']
    assert result['loss'] is None
    assert result['coverage'] == pytest.approx(1/3)
    assert not scorer._graph_cache
    scorer._node_cache.update({('root',1):30., ('root',2):40.})
    result = scorer.score_candidates([object()], stage='side', depth=1)[0]
    assert result['scored'] and result['coverage'] == 1
    assert result['carbon_centers'] == 3
    with pytest.raises(ValueError, match='rigid'):
        scorer.score_candidates([object()], stage='rigid', depth=0)


def test_nmr_ring_flags_exclude_side_branches():
    from model.forward.g2s.rl_mtcs_utils import _cycle_edges
    cycle = {(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)}
    assert _cycle_edges(cycle | {(2,6),(6,7)}) == cycle


def test_nmr_only_scores_formal_states_not_forward_rollout():
    from RL_MTCS.policy.checkpoints import StageController
    from RL_MTCS.config import AssemblyConfig
    from RL_MTCS.policy.rollout import fast_rollout_remaining
    from types import SimpleNamespace
    calls=[]
    scorer=SimpleNamespace(score_state=lambda state,stage:(calls.append(stage) or {'loss':1,'scored':True}))
    controller=StageController(AssemblyConfig(verbose=False),scorer)
    state=ring_side_fixture()
    for stage in ('rigid','flex','ring','side'):
        state.stage=stage
        controller.reward(state)
    assert calls==['flex_chain','aliphatic_ring','side']
    fast_rollout_remaining(state)
    assert len(calls)==3
