"""Shared scheduling, atomic actions and reproducible checkpoint diversity."""
from dataclasses import replace
import pickle
import zlib

import pytest

from RL_MTCS import AssemblyConfig
from RL_MTCS.actions import catalog, rigid
from RL_MTCS.geometry import connection_axis
from RL_MTCS.mdp.sites import aromatic_perimeter_ids
from RL_MTCS.mdp.transition import apply_action, relabel_aromatic_sites
from RL_MTCS.mdp.types import InverseInputError, TopologyUnit, UnitNode
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.policy.mcts import MCTS
from RL_MTCS.reporting.diagnostics import entry_profile
from RL_MTCS.storage.state_delta import encode_transition, replay_transition
from test_rl_mtcs_sites import aromatic_fixture, ring_side_fixture
from test_rl_mtcs_staged_revision import chain_inventory_fixture


@pytest.mark.parametrize('direction,axis', [((0,1),'V'),((0,-1),'V'),
    ((1,1),'H'),((1,0),'H'),((-1,0),'H'),((-1,-1),'H')])
def test_axis_uses_global_hex_directions(direction, axis):
    assert connection_axis(direction) == axis


def test_rigid_successful_connections_alternate_axes():
    initial = state = aromatic_fixture(5)
    axes = []
    for _ in range(3):
        before = state.state_signature()
        action = next(catalog.legal_actions(state, 'rigid', 1))
        child = apply_action(state, action)
        assert state.state_signature() == before
        assert child.audit()['geometry_valid']
        axes.append(child.last_rigid_axis)
        state = child
    assert axes == ['H', 'V', 'H']
    assert initial.last_rigid_axis is None
    assert state.last_flex_axis is None


def test_flex_alternates_merges_then_places_remaining_chain():
    state = chain_inventory_fixture()
    anchor = None
    axes = []
    for index in range(4):
        action = next(catalog.legal_actions(state, 'flex', 1))
        assert len(action.endpoint_bindings) == 2
        assert action.conformer.unit_id.startswith('chain:')
        child = apply_action(state, action)
        if anchor is None:
            anchor = child.main_anchor
        assert child.main_anchor == anchor
        if index < 3:
            assert child.component_count() == state.component_count() - 1
        else:
            assert child.action_history[-1]["connection_mode"] == "internal"
        assert child.audit()['geometry_valid']
        axes.append(child.last_flex_axis)
        state = child
    assert axes == ['H', 'V', 'H', 'V']
    assert not state.excluded_unit_ids


def test_axis_fallback_is_used_only_after_preferred_legal_actions_exhausted(monkeypatch):
    state = aromatic_fixture(2)
    original = rigid.iter_rigid_outcomes
    calls = []
    def outcomes(state, *, axis, seed=None):
        calls.append(axis)
        if axis == 'H':
            invalid = next(original(state, axis=axis, seed=seed))
            nodes = list(invalid.node_axial)
            invalid.node_axial[nodes[0]] = invalid.node_axial[nodes[1]]
            invalid.final_node_positions[nodes[0]] = invalid.final_node_positions[nodes[1]]
            yield invalid
        else:
            yield from original(state, axis=axis, seed=seed)
    monkeypatch.setattr(rigid, 'iter_rigid_outcomes', outcomes)
    result = next(catalog.connection_states(state, 'rigid', 1))
    assert calls == ['H', 'V']
    assert result.last_rigid_axis == 'V'
    assert result.audit()['geometry_valid']


def test_axis_survives_clone_replay_relabel_and_rollback():
    initial = aromatic_fixture(3)
    child = next(catalog.connection_states(initial, 'rigid', 1))
    rebuilt = relabel_aromatic_sites(child, {}, child.action_history, stage='rigid')
    restored = replay_transition(initial, encode_transition(initial, child))
    assert rebuilt.last_rigid_axis == restored.last_rigid_axis == child.clone().last_rigid_axis == 'H'
    restored.last_flex_axis = 'V'
    restored._restore(child)
    assert restored.state_signature() == child.state_signature()
    assert restored.to_dict()['last_rigid_axis'] == 'H'


def test_stale_action_fails_without_mutating_input():
    state = aromatic_fixture(3)
    action = next(catalog.legal_actions(state, 'rigid', 1))
    changed = state.clone()
    changed.last_flex_axis = 'V'
    before = changed.state_signature()
    with pytest.raises(InverseInputError, match='stale connection action'):
        apply_action(changed, action)
    assert changed.state_signature() == before


def test_corrupt_atomic_effect_is_detected_without_partial_commit():
    state = aromatic_fixture(3)
    action = next(catalog.legal_actions(state, 'rigid', 1))
    edits = pickle.loads(zlib.decompress(action.payload))
    edits['last_rigid_axis'] = ('value', 'V')
    corrupt = replace(action, payload=zlib.compress(pickle.dumps(edits, protocol=5)))
    before = state.state_signature()
    with pytest.raises(InverseInputError, match='validated target'):
        apply_action(state, corrupt)
    assert state.state_signature() == before


def test_formal_retry_changes_seed_and_stays_reproducible():
    state = aromatic_fixture(4)
    config = AssemblyConfig(connection_candidate_limit=2, verbose=False)
    one = StageController(config)
    first, first_info = one.search(state, 'rigid')
    from RL_MTCS.mdp.signatures import search_signature
    excluded = {search_signature(first, 'rigid')}
    second, second_info = one.search(state, 'rigid', excluded)
    repeat, repeat_info = StageController(config).search(state, 'rigid', excluded)
    assert first_info['search_seed'] != second_info['search_seed']
    assert repeat_info['search_seed'] == second_info['search_seed']
    assert repeat.state_signature() == second.state_signature()
    assert search_signature(second, 'rigid') not in excluded


def test_complex_sides_search_global_exterior_before_fallback(monkeypatch):
    state = ring_side_fixture()
    extra = TopologyUnit('complex', 'side_chain', tuple(UnitNode(100+i, 23, (i,0)) for i in range(3)))
    other = replace(extra, unit_id='complex:1', nodes=tuple(replace(n, node_id=n.node_id+100) for n in extra.nodes))
    state.problem = replace(state.problem, units=state.problem.units + (extra, other))
    state.__post_init__()
    visited = []
    def outcomes(state, unit, *, region, seed=None, blocked_units=None):
        visited.append((unit.unit_id, region))
        return iter(())
    monkeypatch.setattr(catalog, 'iter_side_outcomes', outcomes)
    assert list(catalog.legal_actions(state, 'side')) == []
    assert visited[:6] == [(uid, region) for region in ('outside','crossing','inside')
                          for uid in ('complex','complex:1')]
    assert all(not uid.startswith('complex') for uid, region in visited[6:])


def test_rings_search_global_exterior_before_spatial_fallback(monkeypatch):
    state = ring_side_fixture()
    other = replace(state.unit('topology:0'), unit_id='topology:2',
                    nodes=tuple(replace(n, node_id=n.node_id+200) for n in state.unit('topology:0').nodes),
                    internal_edges=(), ports=())
    state.problem = replace(state.problem, units=state.problem.units + (other,))
    state.__post_init__()
    visited = []
    def outcomes(state, unit, *, region, seed=None):
        visited.append((unit.unit_id, region))
        return iter(())
    monkeypatch.setattr(catalog, 'iter_ring_outcomes', outcomes)
    assert list(catalog.legal_actions(state, 'ring')) == []
    assert visited == [(uid, region) for region in ('outside', 'crossing', 'inside')
                       for uid in ('topology:0', 'topology:2')]


def test_excluded_result_has_visits_but_no_selectable_reward_sample():
    tree = MCTS(0, expand=lambda s: iter([1,2]) if s == 0 else iter(()),
        accept=lambda s: False, signature=lambda s:s, evaluate=lambda s:s/2,
        excluded={2}, result_limit=1)
    assert tree.run() == 1
    assert tree.result_rewards == [.5]


def test_entry_diagnostics_default_to_counts_only():
    state = ring_side_fixture()
    compact = entry_profile(state)
    detailed = entry_profile(state, include_sites=True)
    assert 'sites' not in compact
    assert len(detailed['sites']) == compact['free_su13']
    assert detailed['available_entries'] == compact['available_entries']


@pytest.mark.parametrize('name,value', [('connection_candidate_limit', True),
    ('connection_candidate_limit', 1.5), ('forward_max_rollbacks', -1),
    ('forward_checkpoint_interval', 0), ('clearance_radius', 0),
    ('c_puct', float('nan')), ('forward_min_completion', float('nan'))])
def test_invalid_configuration_is_not_silently_clamped(name, value):
    with pytest.raises(ValueError):
        AssemblyConfig(**{name:value})


def test_initialization_report_uses_migrated_loader_and_exporter(tmp_path):
    from pathlib import Path
    from RL_MTCS.reporting.initialization import inspect_csv_initialization
    inputs = Path(__file__).resolve().parents[1] / 'test_results/YM-3/final_outputs'
    report = inspect_csv_initialization(inputs/'final_nodes_by_global_id.csv',
                                       inputs/'final_topology_specs.csv', tmp_path)
    assert report['ok'] and report['validation']['ok']
    assert report['summary']['unit_count'] == 159
    assert (tmp_path/'contracts.csv').is_file()


def test_main_component_query_and_report_do_not_choose_an_anchor_in_input():
    from RL_MTCS.mdp.components import main_units
    from RL_MTCS.reporting.completion import completion_report
    state = aromatic_fixture(3)
    before = state.state_signature()
    assert main_units(state) == {'aromatic:0'}
    assert completion_report(state)['main_anchor'] == 'aromatic:0'
    assert state.state_signature() == before
    assert not state.task_inventory


def test_shared_audit_checks_port_partner_type_requirements():
    state = next(catalog.connection_states(aromatic_fixture(3), 'rigid', 1))
    contract = next(c for c in state.problem.contracts if c.kind == 'rigid')
    unit = state.unit(contract.source_unit_id)
    restricted = replace(unit, ports=tuple(replace(p, required_su_types=(23,))
        if p.port_id == contract.source_port_id else p for p in unit.ports))
    state.problem = replace(state.problem, units=tuple(
        restricted if u.unit_id == unit.unit_id else u for u in state.problem.units))
    state.__post_init__()
    assert any(e['reason'] == 'port_partner_su_type_mismatch'
               for e in state.audit_port_direction_errors())
    assert not state.audit()['geometry_valid']


def test_nmr_committed_view_never_replaces_actual_labels_with_input_plan():
    from types import SimpleNamespace
    from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
    state = SimpleNamespace(current_node_map=lambda: {1:13, 2:30},
                            problem=SimpleNamespace(node_map=lambda: {1:11, 2:13}),
                            is_complete=lambda: True)
    assert NMRGuidanceScorer._effective_su_map(state) == {1:13, 2:30}
