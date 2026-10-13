"""Template residuals and inverse-only SU10 pair coordination."""

import csv
import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL_MTCS.io.input import (
    build_csv_inverse_input, initialize, load_node_rows, serialize_aromatic_plan,
)
from RL_MTCS.mdp.types import InverseInputError
from RL_MTCS.mdp.sites import aromatic_template_errors
from model.shared.aromatic_plan import exact_aromatic_cluster_plan, flex_count_window
from model.inverse.topology.resources.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.resources.conversion_scoring import _flex_window_from_H
from model.inverse.topology.stage3.state_machine import _narrow_flex_window_plan


HSQ = Path(__file__).resolve().parents[1] / 'test_results/HSQ-1/outer_cycle_1/final_outputs'


@pytest.fixture
def hsq_nodes():
    return [SimpleNamespace(global_id=row.node_id, su_type=row.su_type,
                            hop1_ids=list(row.hop1_ids))
            for row in load_node_rows(HSQ / 'final_nodes_by_global_id.csv').values()]


def coordinate(nodes, *, limit=10, rejected=()):
    h = torch.bincount(torch.tensor([n.su_type for n in nodes]), minlength=40)
    audit = {**flex_count_window(17, 28), 'explicit_flex_count': 9, 'enforced': True}
    phase = SimpleNamespace(flex_audit=audit, flex_counts={'final': 9})
    plan = _narrow_flex_window_plan(
        phase_eval=phase, H=h, nodes=nodes, max_primary_adjustments=limit,
        special_degree_meta=None, forbidden_signatures=rejected,
    )
    return h, plan


def test_bounded_residual_uses_benzene_and_never_appends_sites():
    assert exact_aromatic_cluster_plan(0, 5)['idle_aromatic_count'] == 5
    six = exact_aromatic_cluster_plan(0, 6)
    assert six['kinds'] == ['benzene']
    assert six['idle_aromatic_count'] == 0
    with pytest.raises(ValueError, match='too many non-bridgehead'):
        exact_aromatic_cluster_plan(0, 5, max_idle_aromatic=4)


def test_hsq_new_assignment_preserves_ports_and_substitution():
    problem = build_csv_inverse_input(HSQ / 'final_nodes_by_global_id.csv',
                                      HSQ / 'final_topology_specs.csv', use_authoritative_plan=False)
    assert not aromatic_template_errors(problem)
    assert len(problem.idle_aromatic_nodes) == 1
    assert all(su == 13 for _, su in problem.idle_aromatic_nodes)
    assert 78 in problem.node_map()
    assert sum(su == 10 for su in problem.node_map().values()) == 28
    assert len(problem.source_rigid_contracts) == 14
    assert sum(c.kind == 'flex' for c in problem.contracts) == 18
    assert sum(c.kind == 'ring' for c in problem.contracts) == 26
    assert len([u for u in problem.units if u.phase == 'aromatic']) == 17
    assert any(r.target_node_id == 493 and r.synthetic_node_id == -1000000
               for r in problem.hetero_requests)
    source = load_node_rows(HSQ / 'final_nodes_by_global_id.csv')
    assert set(problem.input_node_map()) == set(source) | {-1000000}
    assert not (set(dict(problem.idle_aromatic_nodes)) & set(problem.ownership_map()))


def test_formal_export_replaces_stale_plan_and_roundtrips_idle(tmp_path):
    from model.inverse.inverse_pipeline import InversePipelineV3

    for name in ('final_nodes_by_global_id.csv', 'final_topology_specs.csv',
                 'final_aromatic_clusters.json', 'final_topology_node_ownership.csv',
                 'final_topology_resource_pool.csv', 'final_topology_summary.csv'):
        shutil.copy2(HSQ / name, tmp_path / name)
    source_bytes = (tmp_path / 'final_nodes_by_global_id.csv').read_bytes()
    exporter = object.__new__(InversePipelineV3)
    first = exporter._save_final_aromatic_plan(output_dir=str(tmp_path))
    data = initialize(tmp_path / 'final_nodes_by_global_id.csv', tmp_path / 'final_topology_specs.csv')
    assert data['state'].audit()['resource_partition_valid']
    assert data['state'].audit()['idle_aromatic_count'] == 1
    assert data['state'].audit()['geometry_valid']
    assert serialize_aromatic_plan(data['problem'])['idle_aromatic_nodes'] == first['idle_aromatic_nodes']
    second = exporter._save_final_aromatic_plan(output_dir=str(tmp_path))
    assert second['clusters'] == first['clusters']
    assert (tmp_path / 'final_nodes_by_global_id.csv').read_bytes() == source_bytes
    with (tmp_path / 'final_topology_node_ownership.csv').open() as handle:
        idle = [r for r in csv.DictReader(handle) if r['state'] == 'idle_aromatic']
    assert len(idle) == 1
    assert idle[0]['resource_pool'].lower() == 'false'
    assert idle[0]['completion_tolerated'].lower() == 'true'
    assert first['completion_mode'] == 'bounded_idle_aromatic'

    # A port cannot be hidden in the residual ledger to satisfy coverage.
    first['idle_aromatic_nodes'][0]['node_id'] = 78
    (tmp_path / 'final_aromatic_clusters.json').write_text(json.dumps(first))
    with pytest.raises(InverseInputError, match='invalid idle aromatic'):
        initialize(tmp_path / 'final_nodes_by_global_id.csv', tmp_path / 'final_topology_specs.csv')


def test_minimal_pair_conversion_is_atomic_and_does_not_mutate_source(hsq_nodes):
    h, plan = coordinate(hsq_nodes)
    assert plan.ok and plan.phase_result.rebuild_required
    assert plan.histogram_adjustment_plan.primary_adjustment_count == 6
    after = plan.diagnostics['realized_flex_after']
    assert (after['X'], after['Y'], after['flex_lower'], after['flex_upper']) == (17, 11, 7, 13)
    assert after['flex_window_width'] == 6
    ids = set(plan.phase_result.transactions[0].converted_node_ids)
    assert len(ids) == 6
    assert len(plan.diagnostics['retired_su10_pairs']) == 3
    applied = apply_stage3_node_adjustments(hsq_nodes, h, plan)
    assert applied.changed, applied.rejected_ops
    assert (int(applied.H[10]), int(applied.H[12])) == (22, 106)
    assert {n.global_id for n in applied.nodes} == {n.global_id for n in hsq_nodes}
    assert all(n.su_type == 10 for n in hsq_nodes if n.global_id in ids)
    assert all(n.su_type == 12 for n in applied.nodes if n.global_id in ids)
    assert torch.equal(h, plan.histogram_adjustment_plan.H_before)


def test_conversion_budget_cannot_commit_partial_pairs(hsq_nodes):
    h, plan = coordinate(hsq_nodes, limit=5)
    assert not plan.ok
    assert plan.blocked_reason == 'flex_window_coordination_exhausted'
    assert plan.histogram_adjustment_plan is None
    assert int(h[10]) == 28


def test_rejected_rebuild_selects_next_bounded_quantity(hsq_nodes):
    from model.inverse.layer4.inverse_layer4_block_c import (
        _register_rejected_resource_candidate, _rollback_stage3_plan_transactions,
    )

    _h, first = coordinate(hsq_nodes)
    rejected = set()
    signature = _register_rejected_resource_candidate(first, first.histogram_adjustment_plan, rejected)
    assert signature in rejected
    _rollback_stage3_plan_transactions(first, 'test_rebuild_rejection')
    assert not first.phase_result.transactions[0].committed
    assert not first.phase_result.transactions[0].converted_node_ids
    _h, second = coordinate(hsq_nodes, rejected=rejected)
    assert second.ok
    assert second.histogram_adjustment_plan.primary_adjustment_count == 8
    after = second.diagnostics['realized_flex_after']
    assert after['X'] == 16  # Cluster count must be replanned, never cached at 17.
    assert (after['flex_lower'], after['flex_upper']) == (7, 12)


def test_window_width_does_not_replace_actual_chain_count_check(hsq_nodes):
    h, _plan = coordinate(hsq_nodes)
    h[10] -= 14
    h[12] += 14
    window = _flex_window_from_H(h, (), 9)
    assert window['flex_window_width_ok']
    assert window['scenario'] == 'below_range'
    assert window['flex_lower'] == 10


def test_missing_real_su10_pair_does_not_invent_donors(hsq_nodes):
    for node in hsq_nodes:
        if node.su_type == 10:
            node.hop1_ids = []
    _h, plan = coordinate(hsq_nodes)
    assert not plan.ok
    assert plan.diagnostics['available_pair_count'] == 0


def test_wide_window_does_not_block_chain_phase():
    from collections import Counter
    from model.shared.inverse_common import _NodeV3
    from model.inverse.topology.stage3.phase_eval import build_stage3_phase_evaluation

    nodes = []
    with (HSQ / 'final_nodes_by_global_id.csv').open() as handle:
        for row in csv.DictReader(handle):
            node = _NodeV3(int(row['global_id']), int(row['su_type']),
                           target_hop1_degree=int(row['target_hop1_degree']))
            node.hop1_ids = ast.literal_eval(row['hop1_ids'])
            node.hop1_su = Counter(ast.literal_eval(row['hop1_ms']))
            node.hop2_su = Counter(ast.literal_eval(row['hop2_ms']))
            node.template_key = ast.literal_eval(row['template_key'])
            nodes.append(node)
    phase = build_stage3_phase_evaluation(nodes)
    assert not phase.flex_reconciliation_block_reason
    assert phase.flex_audit['flex_window_width'] == 9


def test_su10_stays_out_of_generic_conversion_pool(hsq_nodes):
    from model.inverse.topology.resources.conversion_scoring import ConversionContext, score_conversion_bundle
    from model.inverse.topology.resources.histogram import _h_total
    from model.inverse.topology.stage3_models import HistogramConversion

    h, _plan = coordinate(hsq_nodes)
    context = ConversionContext(inventory={10: 28}, demand={}, current_H=h)
    score = score_conversion_bundle(context, [HistogramConversion(from_su=10, to_su=12, count=2)],
                                    h_total_fn=_h_total)
    assert not score.feasible


def test_invalid_real_donor_rolls_back_entire_histogram(hsq_nodes):
    h, plan = coordinate(hsq_nodes)
    op = plan.histogram_adjustment_plan.groups[0].primary_op
    op.node_ids[-1] = 999999
    result = apply_stage3_node_adjustments(hsq_nodes, h, plan)
    assert not result.changed
    assert torch.equal(result.H, h)
    assert [n.su_type for n in result.nodes] == [n.su_type for n in hsq_nodes]


def test_su10_with_external_topology_neighbor_is_protected(hsq_nodes):
    _h, baseline = coordinate(hsq_nodes)
    protected = baseline.diagnostics['retired_su10_pairs'][0][0]
    external = next(n.global_id for n in hsq_nodes if n.su_type == 23)
    next(n for n in hsq_nodes if n.global_id == protected).hop1_ids.append(external)
    _h, plan = coordinate(hsq_nodes)
    assert plan.ok
    assert protected not in plan.phase_result.transactions[0].converted_node_ids
