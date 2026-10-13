"""Space demand, compactness and identical checkpoint scoring contexts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from RL_MTCS.config import AssemblyConfig
from RL_MTCS.geometry import axial_add, transform_axial
from RL_MTCS.io.input import initialize
from RL_MTCS.mdp.components import finish_flex
from RL_MTCS.mdp.types import ExternalContract, TopologyUnit, UnitNode
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.reporting.completion import completion_report
from RL_MTCS.rewards.demand import spatial_demand
from RL_MTCS.rewards.scoring import score_state, shape_profile, su13_clearance_profile
from test_rl_mtcs_sites import aromatic_fixture, ring_side_fixture


def test_low_demand_releases_only_clearance_weight_and_nmr_still_normalizes():
    state = ring_side_fixture()
    state.stage = 'ring'
    row = score_state(state, completion=0.5, forward=0.8)
    assert 0 < row['spatial_demand']['pressure'] < 1
    assert row['weights'][3] < row['base_weights'][3] / .85
    assert row['weights'][4] > row['base_weights'][4] / .85
    assert row['weights'][:3] == pytest.approx([.4/.85, 0, .2/.85])
    assert sum(row['weights'][3:]) == pytest.approx(.25/.85)
    assert sum(row['weighted_terms'].values()) == pytest.approx(row['total'])
    with_nmr = score_state(state, completion=0.5, forward=0.8, nmr=0.7)
    assert with_nmr['weights'][:3] == pytest.approx([.4, .15, .2])


def test_high_volume_or_port_demand_preserves_clearance_weight():
    state = ring_side_fixture()
    topology = next(u for u in state.problem.units if u.phase == 'side')
    bulky = replace(topology, unit_id='bulky',
                    nodes=tuple(UnitNode(200+i, 22, (i,0)) for i in range(50)))
    state.source_problem = replace(state.source_problem, units=state.source_problem.units+(bulky,))
    row = score_state(state)
    assert row['spatial_demand']['pressure'] == 1
    assert row['weights'][3:] == pytest.approx([.2/.85, .05/.85])
    small = aromatic_fixture(1)
    units = tuple(TopologyUnit(f'side:{i}', 'side_chain', (UnitNode(100+i,22,(0,0)),))
                  for i in range(3))
    contracts = tuple(ExternalContract(f'c:{i}', 'aromatic:0',0,u.unit_id,100+i,'side')
                      for i,u in enumerate(units))
    small.source_problem = replace(small.source_problem,
                                  units=small.source_problem.units+units, contracts=contracts)
    demand = spatial_demand(small)
    assert demand.remaining_nodes < demand.reference_backbone_nodes
    # Reduce only the synthetic original slot capacity to expose port pressure.
    scarce = replace(demand, reference_su13_sites=2)
    assert score_state(small,demand=scarce)['spatial_demand']['pressure'] == 1


def test_scans_exclusions_and_candidate_rebinding_cannot_lower_frozen_demand():
    state = ring_side_fixture()
    state.stage = 'ring'
    demand = spatial_demand(state)
    changed = state.clone()
    changed.excluded_unit_ids.update(u.unit_id for u in state.problem.units)
    changed.used_ports.update(f'aromatic_site:{n}' for n in changed.node_axial)
    changed.problem = replace(changed.problem, units=())  # Original input remains authoritative.
    assert spatial_demand(changed) == demand
    assert score_state(state,demand=demand)['spatial_demand'] == demand.to_dict()
    assert su13_clearance_profile(state)['site_count'] > demand.remaining_ports


def test_ended_phases_stop_reserving_space_without_erasing_unfinished_tasks():
    state = ring_side_fixture()
    total = completion_report(state)['total']
    state.stage = 'substitution'
    row = score_state(state)
    assert row['spatial_demand']['remaining_units'] == 0
    assert row['weights'][3] == 0
    assert row['weights'][4] == pytest.approx(.25/.85)
    assert completion_report(state)['total'] == total
    assert completion_report(state)['stages']['side']['unconnected']
    assert su13_clearance_profile(state)['site_count'] > 0


def test_zero_capacity_positive_demand_does_not_remove_space_protection():
    state = ring_side_fixture()
    demand = replace(spatial_demand(state), reference_backbone_nodes=0,reference_su13_sites=0)
    row = score_state(state,demand=demand)
    assert row['spatial_demand']['pressure'] == 1
    assert all(0 <= w <= 1 for w in row['weights'])


def test_compactness_distinguishes_isotropic_expansion_and_is_rotation_invariant():
    compact = aromatic_fixture(1)
    hollow = compact.clone()
    hollow.node_axial = {n:(q*4,r*4) for n,(q,r) in compact.node_axial.items()}
    hollow.final_node_positions = dict(hollow.node_axial)
    a,b = shape_profile(compact),shape_profile(hollow)
    assert a['aspect_quality'] == pytest.approx(b['aspect_quality'])
    assert a['quality'] > b['quality']
    assert a['normalized_radius_of_gyration'] < b['normalized_radius_of_gyration']
    for reflection in (False,True):
        for rotation in range(6):
            trial = hollow.clone()
            trial.node_axial = {n:axial_add((900,-600), transform_axial(p,rotation,reflection))
                                for n,p in hollow.node_axial.items()}
            trial.final_node_positions = dict(trial.node_axial)
            assert shape_profile(trial)['quality'] == pytest.approx(b['quality'])


def test_shape_uses_final_physical_coordinates():
    state = aromatic_fixture(1)
    before = shape_profile(state)
    state.final_node_positions = {n:(q*4,r*4) for n,(q,r) in state.node_axial.items()}
    assert shape_profile(state)['quality'] < before['quality']


def test_shape_is_independent_of_mapping_iteration_order():
    from RL_MTCS.actions.catalog import connection_states
    state = next(connection_states(aromatic_fixture(3),'rigid',limit=1))
    before = shape_profile(state)
    state.node_axial = dict(reversed(list(state.node_axial.items())))
    state.final_node_positions = dict(reversed(list(state.final_node_positions.items())))
    state.placements = dict(reversed(list(state.placements.items())))
    assert shape_profile(state) == before


def test_each_retry_and_fallback_share_the_starting_demand(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    initial = ring_side_fixture()
    initial.stage = 'flex'
    controller = StageController(AssemblyConfig(verbose=False,forward_max_rollbacks=1))
    contexts, score_contexts = [],[]
    def search(state,phase,excluded,*,demand):
        contexts.append(demand)
        selected = state.clone()
        selected.last_flex_axis = 'H' if not excluded else 'V'
        selected.placements['topology:0'] = state.placements['aromatic:0']
        return selected,{}
    original_reward = controller.reward
    def reward(state,forward=None,*,demand=None):
        score_contexts.append(demand)
        return original_reward(state,forward,demand=demand)
    monkeypatch.setattr(controller,'search',search)
    monkeypatch.setattr(controller,'reward',reward)
    monkeypatch.setattr(module,'fast_rollout_remaining',lambda s:SimpleNamespace(
        to_dict=lambda:dict(forward_completion=.5,audit={'legal':True})))
    _,info = controller.checkpoint(initial,'flex')
    assert len(contexts) == len(score_contexts) == 2
    assert all(c is contexts[0] for c in contexts+score_contexts)
    attempts = info['retry_selection']['attempts']
    assert attempts[0]['rewards']['weights'] == attempts[1]['rewards']['weights']
    assert info['retry_selection']['spatial_demand'] == contexts[0].to_dict()


def test_search_and_stage_summary_use_the_same_frozen_weights(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    initial = finish_flex(ring_side_fixture())
    seen = []
    controller = StageController(AssemblyConfig(verbose=False,connection_candidate_limit=2))
    original = controller.reward
    def reward(state,forward=None,*,demand=None):
        row = original(state,forward,demand=demand)
        seen.append(row)
        return row
    monkeypatch.setattr(controller,'reward',reward)
    # Formal ring search must not accidentally introduce a branch rollout.
    monkeypatch.setattr(module,'fast_rollout_remaining',lambda *a:pytest.fail('unexpected forward'))
    _,info = controller.run(initial,'ring')
    assert seen
    assert all(row['weights'] == info['rewards']['weights'] for row in seen)
    assert info['remaining_spatial_demand']['stages'].keys() == {'side'}


def test_entrance_ties_saturate_and_finished_tasks_have_no_advantage(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    state = ring_side_fixture()
    state.stage = 'ring'
    demand = spatial_demand(state)
    monkeypatch.setattr(module,'entry_profile',lambda *a,**kw:dict(available_entries=100))
    assert StageController.entrance_tie_break(state,demand) == demand.remaining_ports
    monkeypatch.setattr(module,'entry_profile',lambda *a,**kw:dict(available_entries=200))
    assert StageController.entrance_tie_break(state,demand) == demand.remaining_ports
    finished = state.clone()
    finished.placements.update({u.unit_id:state.placements['aromatic:0']
                               for u in state.problem.units})
    monkeypatch.setattr(module,'entry_profile',lambda *a,**kw:pytest.fail('unneeded entrance scan'))
    assert StageController.entrance_tie_break(finished,demand) == 0


def test_actual_samples_have_distinct_demand_with_identical_base_weights():
    root = Path(__file__).resolve().parents[1] / 'test_results'
    rows = {}
    for sample in ('YM-3','LY-2'):
        inputs = root / sample / 'final_outputs'
        state = initialize(inputs/'final_nodes_by_global_id.csv',inputs/'final_topology_specs.csv')['state']
        # Observe the ring/side requirement with a fixed original backbone.
        state.stage = 'ring'
        rows[sample] = score_state(state)
    ym,ly = rows['YM-3'],rows['LY-2']
    assert ym['base_weights'] == ly['base_weights']
    assert ym['spatial_demand']['pressure'] == pytest.approx(306/310)
    assert ly['spatial_demand']['pressure'] == pytest.approx(84/533)
    assert ym['spatial_demand']['reference_su13_sites'] == 206
    assert ly['spatial_demand']['reference_su13_sites'] == 304
    assert ym['weights'][3] > ly['weights'][3]
    assert ym['weights'][4] < ly['weights'][4]
