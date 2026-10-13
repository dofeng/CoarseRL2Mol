"""HSQ-2 regression: complete chains before narrowing the count window."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from model.inverse.topology.stage3.phase_eval import build_stage3_phase_evaluation
from model.inverse.topology.stage3.workflow import plan_stage3_resource_adjustments
from model.inverse.topology.stage3.state_machine import (
    plan_stage3_state_machine, validate_rebuilt_flex_window,
)
from model.inverse.topology.assembly.state.snapshot import validate_snapshot_state
from model.inverse.layer4.inverse_layer4_block_c import _stage3_runtime_snapshot
from model.inverse.topology.stage3_models import Stage3AdjustmentPlan


SNAPSHOT = Path(__file__).resolve().parents[1] / 'test_results/HSQ-2/stage3_input_snapshot'


@pytest.fixture
def first_round():
    if not SNAPSHOT.is_dir():
        pytest.skip('local HSQ-2 first-round snapshot unavailable')
    return (
        torch.load(SNAPSHOT / 'nodes.pt', map_location='cpu', weights_only=False),
        torch.load(SNAPSHOT / 'H.pt', map_location='cpu', weights_only=False),
        json.loads((SNAPSHOT / 'fixed_partition_meta.json').read_text()),
    )


def test_first_round_completes_five_supplemental_chains_before_conversion(first_round):
    nodes, h, meta = first_round
    before = [(n.global_id, n.su_type, list(n.hop1_ids)) for n in nodes]
    plan = plan_stage3_resource_adjustments(
        h, nodes, fixed_partition_meta=meta, flex_max_primary_adjustments=12,
        enable_post_external_spectrum=False,
    )
    assert plan.ok and plan.plan_kind == 'narrow_flex_window'
    phase = plan.runtime['phase_evaluation']
    assert phase.flex_audit['explicit_flex_count'] == 9
    assert len(phase.flex_audit['selected_candidate_groups']) == 5
    assert phase.flex_counts['final'] == 14
    assert plan.diagnostics['completed_flex_count_before_rebuild'] == 14
    assert plan.histogram_adjustment_plan.primary_adjustment_count == 2
    after = plan.diagnostics['realized_flex_after']
    assert (after['X'], after['Y'], after['flex_lower'], after['flex_upper']) == (26, 13, 14, 20)
    assert after['flex_window_width'] == 6
    assert 'real_like24_ring_assembly' not in phase.completed_phases
    assert 'external_fatty_ring_closure' not in phase.completed_phases
    assert 'remaining_external_halfclosed_closure' not in phase.completed_phases
    assert before == [(n.global_id, n.su_type, list(n.hop1_ids)) for n in nodes]
    assert validate_snapshot_state(nodes, _stage3_runtime_snapshot(plan), h)['ok']


def test_no_candidates_does_not_count_unbuilt_chains_or_convert_ports(first_round):
    nodes, h, meta = first_round
    from model.inverse.topology.classification.classifier import classify_topology_units

    classification = classify_topology_units(nodes, fixed_partition_meta=meta)
    phase = build_stage3_phase_evaluation(nodes, classification=classification)
    phase.flex_audit.update(selected_candidate_groups=[], selected_candidate_fragment_ids=[],
                            hard_selected_candidate_fragment_ids=[], preferred_selected_candidate_fragment_ids=[])
    plan = plan_stage3_state_machine(phase_eval=phase, H=h, nodes=nodes,
                                    classification=classification, enable_post_external_spectrum=False)
    assert not plan.ok
    assert plan.plan_kind != 'narrow_flex_window'
    assert plan.histogram_adjustment_plan is None or not plan.histogram_adjustment_plan.groups
    assert plan.runtime['phase_evaluation'].flex_counts['final'] < 14


def test_early_failure_keeps_all_111_unspent_resources(first_round):
    nodes, h, meta = first_round
    phase = build_stage3_phase_evaluation(nodes, fixed_partition_meta=meta)
    # Simulate the pre-assembly failure boundary; no iteration ledger exists.
    from model.inverse.topology.stage3.phase_eval import _minimal_resource_ledger_from_classification
    ledger = _minimal_resource_ledger_from_classification(phase.classification_view)
    phase.assembly_view = {'resource_ledger': ledger}
    plan = Stage3AdjustmentPlan(runtime={'phase_evaluation': phase})
    snapshot = _stage3_runtime_snapshot(plan)
    resources = snapshot['resource_ledger']
    assert sum(resources['remaining_counts'].values()) == 111
    assert resources['remaining_counts'] == resources['initial_counts']
    assert not any(resources['consumed_counts'].values())
    assert validate_snapshot_state(nodes, snapshot, h)['ok']


def test_explicit_empty_live_inventory_stays_empty(first_round):
    nodes, _h, meta = first_round
    phase = build_stage3_phase_evaluation(nodes, fixed_partition_meta=meta)
    phase.assembly_view = {'resource_ledger': {'remaining_ids_by_su': {'23': [400]}}}
    plan = Stage3AdjustmentPlan(runtime={'phase_evaluation': phase,
                                        'iteration_topology_ledger': {'available_ids_by_su': {}}})
    assert _stage3_runtime_snapshot(plan)['resource_ledger']['remaining_ids_by_su'] == {}


def test_rebuild_validation_recounts_and_rejects_wide_window(first_round):
    nodes, _h, meta = first_round
    result = validate_rebuilt_flex_window(nodes, meta)
    assert not result['ok']
    assert result['reason'] == 'rebuilt_flex_window_invalid'


@pytest.mark.parametrize('count, valid', [(9, False), (14, True), (20, True), (21, False)])
def test_rebuild_validation_uses_fresh_materialized_count(first_round, monkeypatch, count, valid):
    nodes, _h, meta = first_round
    import model.inverse.topology.stage3.state_machine as sm
    import model.inverse.topology.stage3.phase_eval as pe

    phase = build_stage3_phase_evaluation(nodes, fixed_partition_meta=meta)
    phase.flex_audit.update(flex_upper=20, flex_window_width=6, flex_window_width_ok=True)
    phase.flex_counts = {'final': count}
    phase.completed_phases = ['external_flex_closure']
    monkeypatch.setattr(pe, 'build_stage3_phase_evaluation', lambda *a, **k: deepcopy(phase))
    monkeypatch.setattr(sm, '_evaluate_external_phases', lambda **k: k['phase_eval'])
    monkeypatch.setattr(sm, '_external_product_materialization_errors', lambda p: [])
    result = validate_rebuilt_flex_window(nodes, meta)
    assert result['ok'] is valid
    assert result['reason'] == ('ok' if valid else 'rebuilt_flex_count_invalid')
    assert result['completed_flex_count'] == count
