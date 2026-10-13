from dataclasses import replace

import pytest

from RL_MTCS.actions import flex
from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.mdp.components import finish_flex
from RL_MTCS.actions.flex import _mandatory_aromatic_footprint, _attachment_footprints_clear
from RL_MTCS.actions.conformers import iter_topology_conformers
from RL_MTCS.reporting.diagnostics import CandidateDiagnostics, collect_candidates, entry_profile
from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.geometry import transform_axial, axial_add, hull_region
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.mdp.types import (HeteroAromaticRequest,
    TopologyUnit, UnitNode, UnitEdge, UnitPort)
from RL_MTCS.policy.mcts import MCTS
from RL_MTCS.mdp.signatures import rigid_geometry_signature, search_signature
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.mdp.state import HexAssemblyState
from test_rl_mtcs_sites import aromatic_fixture, ring_side_fixture
from test_rl_mtcs_memory_geometry import chain_pose_fixture


def test_two_terminal_initial_pose_failure_does_not_hide_legal_conformer():
    candidate, components = chain_pose_fixture(3, 0)
    chain = candidate.state.unit('chain')
    chain = replace(chain, ports=(chain.ports[0], replace(chain.ports[1], direction=(1, 0))),
        metadata=(('geometry_source', 'generated_from_connectivity'), ('source_product_type', 'flex_chain')))
    problem = replace(candidate.state.problem, units=tuple(
        chain if u.unit_id == 'chain' else u for u in candidate.state.problem.units))
    state = HexAssemblyState(problem)
    for uid, placement in candidate.state.placements.items():
        assert state.place_unit(uid, placement.rotation, placement.translation,
                                reflection=placement.reflection)
    probe = replace(candidate, state=state)
    assert next(flex._interior_site_poses(probe, 'chain', components), None) is None
    found = None
    for variant in iter_topology_conformers(chain):
        trial = state.clone()
        trial.select_conformer(variant)
        current = replace(probe, state=trial)
        pose = next(flex._interior_site_poses(current, 'chain', components), None)
        if pose is not None:
            found = flex._place_interior_pose(current, 'chain', pose)
            break
    assert found is not None and found.state.audit()['geometry_valid']
    assert found.state.unit('chain').internal_edges == chain.internal_edges


def test_coarse_endpoints_must_put_both_aromatic_footprints_on_one_lattice():
    footprint = _mandatory_aromatic_footprint(ring_side_fixture())
    chain = TopologyUnit('coarse', 'flex_chain',
        tuple(UnitNode(100+i, 27, (i, 0)) for i in range(3)),
        (UnitEdge(100, 101), UnitEdge(101, 102)),
        (UnitPort('a', 100, direction=(-1, 0)), UnitPort('b', 102, direction=(1, 1))))
    assert not _attachment_footprints_clear(chain, footprint)
    compatible = replace(chain, ports=(chain.ports[0], replace(chain.ports[1], direction=(1, 0))))
    assert not _attachment_footprints_clear(compatible, footprint)


def test_rigid_key_ignores_parking_but_preserves_world_orientation():
    state = next(connection_states(aromatic_fixture(4), 'rigid', 1))
    expected = rigid_geometry_signature(state)
    group = state.component_unit_ids('aromatic:0')
    nodes = {n.node_id for uid in group for n in state.unit(uid).nodes}
    orientations = set()
    for reflected in (False, True):
        for rotation in range(6):
            trial = state.clone()
            trial.node_axial = {n: axial_add((120, -60), transform_axial(p, rotation, reflected))
                                if n in nodes else p for n, p in state.node_axial.items()}
            orientations.add(rigid_geometry_signature(trial))
            if not reflected and rotation == 0:
                assert rigid_geometry_signature(trial) == expected
    assert len(orientations) > 1
    changed = state.clone()
    changed.last_rigid_axis = 'V' if state.last_rigid_axis == 'H' else 'H'
    assert rigid_geometry_signature(changed) != expected
    assert search_signature(state, 'flex') == state.state_signature()


def test_rigid_key_preserves_substitution_roles_edges_and_unit_identity():
    state = next(connection_states(aromatic_fixture(3), 'rigid', 1))
    original = rigid_geometry_signature(state)
    changed = state.clone()
    changed.edges = changed.edges[:-1]
    assert rigid_geometry_signature(changed) != original
    target = next(n for n, su in state.current_su_types.items() if su == 13)
    changed = state.clone()
    changed.problem = replace(state.problem, hetero_requests=(HeteroAromaticRequest(26, 6, target),))
    assert rigid_geometry_signature(changed) != original
    other = changed.clone()
    other.problem = replace(changed.problem, hetero_requests=(HeteroAromaticRequest(30, 6, target),))
    assert rigid_geometry_signature(other) != rigid_geometry_signature(changed)


def test_rigid_candidates_are_geometrically_distinct_and_retry_excludes_geometry():
    initial = aromatic_fixture(3)
    candidates = list(connection_states(initial, 'rigid', 10))
    keys = [rigid_geometry_signature(s) for s in candidates]
    assert 1 < len(keys) <= 10 and len(set(keys)) == len(keys)
    controller = StageController(AssemblyConfig(connection_candidate_limit=2, verbose=False))
    first, _ = controller.search(initial, 'rigid')
    second, _ = controller.search(initial, 'rigid', {rigid_geometry_signature(first)})
    assert rigid_geometry_signature(first) != rigid_geometry_signature(second)
    assert first.audit()['geometry_valid'] and second.audit()['geometry_valid']


@pytest.mark.parametrize('rewards,expected', [({1: .5, 2: .5}, 2), ({1: .6, 2: .5}, 1)])
def test_entry_tie_break_never_overrides_higher_weighted_reward(rewards, expected):
    tree = MCTS(0, expand=lambda s: iter([1, 2]) if s == 0 else iter(()),
        accept=lambda s: False, signature=lambda s: s, evaluate=lambda s: rewards.get(s, 0),
        result_limit=2, tie_break=lambda s: s)
    assert tree.run() == expected


def test_candidate_diagnostics_are_observational_and_context_does_not_leak():
    state = finish_flex(ring_side_fixture())
    before = state.state_signature()
    recorder = CandidateDiagnostics()
    with collect_candidates(recorder):
        candidate = next(connection_states(state, 'side', 1))
    report = recorder.report()
    assert report
    assert next(connection_states(state, 'side', 1)).state_signature() == candidate.state_signature()
    assert recorder.report() == report
    assert state.state_signature() == before


def test_forward_falls_back_to_inside_ring_without_changing_state(monkeypatch):
    import RL_MTCS.policy.rollout as module
    state = finish_flex(ring_side_fixture())
    # Enclose every possible ring attachment in an intentionally huge hull.
    def fixed_hull(s):
        result = finish_flex(s)
        result.backbone_hull = ((-100., -100.), (100., -100.), (100., 100.), (-100., 100.))
        return result
    monkeypatch.setattr(module, 'finish_flex', fixed_hull)
    before = state.state_signature()
    result = fast_rollout_remaining(state)
    row = result.to_dict()['diagnostics']['ring']
    assert row['connected'] == 1
    assert row['unconnected'] == {}
    coordinates = [result.state.node_axial[n.node_id] for n in result.state.unit('topology:0').nodes]
    assert hull_region(result.state.backbone_hull, coordinates) == 'inside'
    assert {'flex:a', 'flex:b'} <= result.state.satisfied_contracts
    assert row['entries_before']['free_su13'] >= row['entries_before']['available_entries']
    assert result.audit['legal']
    assert state.state_signature() == before


def test_entry_profile_distinguishes_empty_space_from_blocked_entrance():
    state = finish_flex(ring_side_fixture())
    before = entry_profile(state)
    assert before['available_entries'] > 0
    # A placed side chain consumes a site; profiling itself does not attach it.
    next_state = next(connection_states(state, 'side', 1))
    after = entry_profile(next_state)
    assert after['free_su13'] == before['free_su13'] - 1
    assert after['available_entries'] + after['blocked_entries'] == after['free_su13']
