"""LY regression: a narrower window can require additional physical chains."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.inverse.layer4.inverse_layer4_block_c import _stage3_runtime_snapshot
from model.inverse.topology.assembly.state.snapshot import validate_snapshot_state
from model.inverse.topology.resources.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.stage3.state_machine import (
    _narrow_flex_window_plan,
    _preview_narrowed_flex_completion,
    validate_rebuilt_flex_window,
)
from model.inverse.topology.stage3.workflow import plan_stage3_resource_adjustments


SNAPSHOT = Path(__file__).resolve().parents[1] / 'test_results/LY-2/stage3_input_snapshot'


@pytest.fixture(scope='module')
def ly_round():
    if not SNAPSHOT.is_dir():
        pytest.skip('local LY first-round snapshot unavailable')
    nodes = torch.load(SNAPSHOT / 'nodes.pt', map_location='cpu', weights_only=False)
    h = torch.load(SNAPSHOT / 'H.pt', map_location='cpu', weights_only=False)
    meta = json.loads((SNAPSHOT / 'fixed_partition_meta.json').read_text())
    plan = plan_stage3_resource_adjustments(
        h, nodes, fixed_partition_meta=meta, flex_max_primary_adjustments=12,
        enable_post_external_spectrum=False,
    )
    return nodes, h, meta, plan


def test_ly_narrowing_materializes_two_extra_chains_before_acceptance(ly_round):
    nodes, h, _meta, plan = ly_round
    assert plan.ok and plan.plan_kind == 'narrow_flex_window'
    assert plan.histogram_adjustment_plan.primary_adjustment_count == 10
    assert (int(h[10]), int(h[12])) == (38, 142)
    assert (int(plan.histogram_adjustment_plan.H_after[10]),
            int(plan.histogram_adjustment_plan.H_after[12])) == (28, 152)
    window = plan.diagnostics['realized_flex_after']
    assert (window['flex_lower'], window['flex_upper'], window['flex_window_width']) == (22, 28, 6)
    preview = plan.diagnostics['supplemental_flex_completion']
    assert preview['ok'] and preview['commit_scope'] == 'candidate_preview'
    assert preview['completed_flex_count_before'] == 20
    assert preview['completed_flex_count_after'] == 22
    assert preview['assembled_additional_flex_count'] == 2
    assert len(preview['products']) == 2
    assert not preview['pending_flex_contracts']
    assert preview['consumed_resource_counts'] == {'11': 4, '23': 4}
    actual_types = {n.global_id: n.su_type for n in nodes}
    for product in preview['products']:
        assert product['closed'] and not product['provisional']
        assert 'OPEN' not in product['topology_notation']
        assert 'OPEN' not in product['node_id_notation']
        assert all(actual_types[int(key)] == su
                   for key, su in product['expected_su_type_by_node_id'].items())
    # The accepted generation keeps its own products/pool until rebuild.
    phase = plan.runtime['phase_evaluation']
    assert phase.flex_counts['final'] == 20
    assert len(phase.external_assembler.products) == 20
    assert validate_snapshot_state(nodes, _stage3_runtime_snapshot(plan), h)['ok']


def test_partial_supplement_does_not_leak_products_or_resource_claims(ly_round):
    _nodes, h, _meta, plan = ly_round
    phase = deepcopy(plan.runtime['phase_evaluation'])
    live = phase.external_assembler
    # Leave enough claimable SU23 for one of the two required chains.
    free23 = live.allocator.remaining_ids_by_su['23']
    live.allocator._generic_claim_excluded_node_ids.update(free23[2:])
    before = deepcopy(live.__dict__)
    preview = _preview_narrowed_flex_completion(phase, plan.diagnostics['realized_flex_after'], h)
    assert not preview['ok']
    assert preview['completed_flex_count_after'] == 21
    assert preview['assembled_additional_flex_count'] == 1
    assert len(live.products) == len(before['products']) == 20
    assert live.allocator.snapshot() == before['allocator'].snapshot()
    assert live.fragment_state_by_id == before['fragment_state_by_id']
    assert live.pending_contracts == before['pending_contracts']


def test_budget_and_rejected_candidates_still_gate_the_transaction(ly_round):
    nodes, h, _meta, plan = ly_round
    phase = plan.runtime['phase_evaluation']
    limited = _narrow_flex_window_plan(
        phase_eval=phase, H=h, nodes=nodes, max_primary_adjustments=8,
        special_degree_meta=None,
    )
    assert not limited.ok and limited.histogram_adjustment_plan is None
    assert all(a['rejection_reason'] == 'flex_window_invalid'
               for a in limited.phase_result.infeasibility_proof['attempts'])
    signature = plan.histogram_adjustment_plan.groups[0].scoring_diagnostics['candidate_bundle_signature']
    alternate = _narrow_flex_window_plan(
        phase_eval=phase, H=h, nodes=nodes, max_primary_adjustments=12,
        special_degree_meta=None, forbidden_signatures=[signature],
    )
    assert alternate.ok
    assert alternate.histogram_adjustment_plan.primary_adjustment_count == 12
    assert alternate.diagnostics['completed_flex_count_after_supplement'] == 22
    assert phase.flex_counts['final'] == 20


def test_missing_runtime_cannot_replace_physical_chains_with_a_count():
    phase = SimpleNamespace(flex_counts={'final': 20})
    preview = _preview_narrowed_flex_completion(
        phase, {'flex_lower': 22, 'flex_upper': 28}, torch.zeros(33, dtype=torch.long),
    )
    assert not preview['ok']
    assert preview['reason'] == 'flex_completion_runtime_unavailable'
    assert preview['assembled_additional_flex_count'] == 0


def test_consumed_canonical_groups_do_not_hide_remaining_chain_sources(ly_round):
    _nodes, h, _meta, plan = ly_round
    phase = deepcopy(plan.runtime['phase_evaluation'])
    phase.external_assembler.flex_audit['supplemental_flex_candidate_groups'] = []
    preview = _preview_narrowed_flex_completion(phase, plan.diagnostics['realized_flex_after'], h)
    assert preview['ok']
    assert preview['completed_flex_count_after'] == 22
    assert not preview['selected_candidate_group_ids']
    assert len(preview['products']) == 2


def test_rebuild_check_reassembles_against_the_converted_graph(ly_round):
    nodes, h, meta, plan = ly_round
    applied = apply_stage3_node_adjustments(nodes, h, plan)
    assert applied.changed and not applied.rejected_ops
    result = validate_rebuilt_flex_window(applied.nodes, meta)
    assert result['ok'], result
    assert result['completed_flex_count'] == 22
    assert (result['flex_lower'], result['flex_upper']) == (22, 28)
