"""Regression coverage for the HSQ equivalent-generation rebuild loop."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
from model.inverse.layer4 import inverse_layer4_block_c as block_c
from model.inverse.layer4.stage3_convergence import (
    Stage3BudgetExceeded, Stage3SearchBudget, Stage3StateMemory,
)
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup, HistogramAdjustmentPlan, HistogramConversion,
    Stage3AdjustmentPlan, Stage3WorkflowResult,
)
from model.shared.inverse_common import _NodeV3


def graph_nodes(graph, *, offset=0):
    nodes = []
    for index in graph:
        node = _NodeV3(index + offset, 13, mu=125., target_hop1_degree=2)
        node.hop1_ids = [other + offset for other in graph[index]]
        node.template_key = (13, (13, 13), (13, 13))
        nodes.append(node)
    return nodes


def histogram(nodes):
    H = torch.zeros(33, dtype=torch.long)
    for node in nodes:
        H[node.su_type] += 1
    return H


def test_relabelling_preserves_failure_history_without_changing_physical_ids():
    memory = Stage3StateMemory()
    original = graph_nodes(nx.cycle_graph(6))
    renamed = graph_nodes(nx.cycle_graph(6), offset=100)
    H = histogram(original)
    key, seen = memory.identify(original, H, remember=True)
    assert not seen
    memory.rejected_candidates[key].add('failed conversion')
    renamed_key, seen = memory.identify(renamed, H, remember=True)
    assert seen and renamed_key == key and len(memory) == 1
    assert memory.rejected_candidates[renamed_key] == {'failed conversion'}
    assert [node.global_id for node in renamed] == list(range(100, 106))


def test_equal_histogram_and_graph_hash_do_not_merge_different_topologies():
    memory = Stage3StateMemory()
    ring = graph_nodes(nx.cycle_graph(6))
    triangles = graph_nodes(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
    H = histogram(ring)
    memory.identify(ring, H, remember=True)
    _, seen = memory.identify(triangles, H, remember=True)
    assert not seen and len(memory) == 2


@pytest.mark.parametrize('attribute,value', [
    ('special_anchor_partition', 'ether'),
    ('special_anchor_mode', 'double'),
    ('target_fixed_anchor_count', 1),
    ('target_hop1_degree', 3),
    ('mu', 126.),
    ('template_key', (13, (12, 13), (13,))),
])
def test_chemical_and_anchor_constraints_remain_distinct(attribute, value):
    memory = Stage3StateMemory()
    nodes = graph_nodes(nx.cycle_graph(6))
    H = histogram(nodes)
    memory.identify(nodes, H, remember=True)
    changed = deepcopy(nodes)
    setattr(changed[0], attribute, value)
    assert not memory.identify(changed, H)[1]


def test_fixed_edges_remain_distinct():
    memory = Stage3StateMemory()
    nodes = graph_nodes(nx.cycle_graph(6))
    H = histogram(nodes)
    memory.identify(nodes, H, remember=True)
    nodes[0].fixed_hop1_ids.add(1)
    assert not memory.identify(nodes, H)[1]


def test_hsq_port_balance_keeps_the_aromatic_adjustment():
    # The actual HSQ continuation histogram immediately before its looping
    # narrow-window transaction. No checkpoints or template library needed.
    before = torch.tensor([0, 6, 5, 7, 0, 32, 1, 2, 0, 12, 26, 24, 75, 132,
                           1, 9, 4, 2, 2, 16, 1, 0, 21, 62, 18, 2, 1, 1, 11, 20, 1, 1, 0])
    nodes = [_NodeV3(index, su) for index, su in enumerate(
        su for su, count in enumerate(before.tolist()) for _ in range(count))]
    donor = next(node.global_id for node in nodes if node.su_type == 12)
    after = before.clone()
    after[12] -= 1
    after[13] += 1
    operation = HistogramConversion(from_su=12, to_su=13, count=1,
                                    node_ids=[donor], reason='flex_window_coordination')
    group = AtomicAdjustmentGroup(primary_op=operation, combined_delta={12: -1, 13: 1})
    batch = HistogramAdjustmentPlan(
        groups=[group], H_before=before.clone(), H_after=after.clone(),
        inventory_before=dict(Counter(node.su_type for node in nodes)),
        primary_adjustment_count=1, physical_conversion_count=1,
    )
    plan = SimpleNamespace(
        plan_kind='narrow_flex_window', runtime={}, phase_result=None,
        diagnostics={'window_coordination_before': {'flex_window_ok': False},
                     'completed_flex_count_before_rebuild': 16,
                     'supplemental_flex_completion': {'resource_continuation_required': True}},
    )
    special = {19: {1: 6, 2: 4, 3: 6}, 20: {1: 1, 2: 0, 3: 0}, 21: {2: 0, 3: 0}}
    ok, proof = block_c._append_stage3_parity_balance(
        plan=plan, histogram_plan=batch, nodes=nodes,
        special_degree_meta=special, balance_cursor=0,
        E_target=torch.tensor([460., 400., 60., 2., 2., 0.]),
    )
    assert ok, proof
    assert batch.H_after[12] <= after[12]
    assert proof['after']['port_roles_ok']
    assert proof['flex_window_objective']['ok']
    assert batch.port_role_balance_op.from_su == 12
    assert batch.port_role_balance_op.to_su == 11
    assert torch.equal(before, torch.tensor([0, 6, 5, 7, 0, 32, 1, 2, 0, 12, 26, 24, 75, 132,
                                           1, 9, 4, 2, 2, 16, 1, 0, 21, 62, 18, 2, 1, 1, 11, 20, 1, 1, 0]))


def test_shared_budget_counts_failed_rebuilds_and_does_not_refund_them():
    budget = Stage3SearchBudget(max_rebuilds=2, max_planner_attempts=5,
                               max_attempts_per_generation=4, max_no_progress_rebuilds=2)
    assert budget.plan(1)
    budget.rebuild()
    budget.accepted()
    budget.rebuild()
    with pytest.raises(Stage3BudgetExceeded, match='stage3_rebuild_limit'):
        budget.rebuild()
    assert budget.rebuilds == 2
    assert not budget.plan(2)


@pytest.mark.parametrize('name,value', [('max_rebuilds', 0), ('max_planner_attempts', True),
                                       ('max_attempts_per_generation', 1.5),
                                       ('max_no_progress_rebuilds', -1)])
def test_invalid_limits_are_rejected(name, value):
    options = dict(max_rebuilds=8, max_planner_attempts=20,
                   max_attempts_per_generation=10, max_no_progress_rebuilds=3)
    options[name] = value
    with pytest.raises(ValueError, match=name):
        Stage3SearchBudget(**options)


@pytest.mark.parametrize('topology_changes', [False, True])
def test_real_workflow_stops_equivalent_rebuilds_in_both_output_modes(monkeypatch, topology_changes):
    def seed_nodes(offset=0, graph=None):
        result = graph_nodes(nx.cycle_graph(6) if graph is None else graph, offset=offset)
        result[0].su_type = 12
        return result

    outcomes = []
    for detailed in (False, True):
        adjuster = Layer4Adjuster(device=torch.device('cpu'))
        nodes = seed_nodes()
        H = histogram(nodes)
        calls = {'plans': 0, 'rebuilds': 0}
        events = []

        def plan(**options):
            calls['plans'] += 1
            groups = [AtomicAdjustmentGroup(primary_op=HistogramConversion(
                from_su=source, to_su=target, count=1, node_ids=[node_id],
            )) for source, target, node_id in ((12, 13, 0), (13, 12, 1))]
            groups[-1].scoring_diagnostics['candidate_bundle_signature'] = f'variant:{calls["plans"]}'
            return Stage3AdjustmentPlan(
                ok=True, reason='stage3_rebuild_required', phase='flex_range_reconciliation',
                histogram_adjustment_plan=HistogramAdjustmentPlan(
                    groups=groups, H_before=H.clone(), H_after=H.clone(),
                ), diagnostics={'candidate_alternatives_enabled': True},
            )

        def apply(**options):
            return SimpleNamespace(changed=True, nodes=deepcopy(nodes), H=H.clone(),
                                   applied_ops=[], rejected_ops=[], request_ops=[], unresolved_counts={})

        def rebuild(**options):
            calls['rebuilds'] += 1
            graph = (nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
                     if topology_changes else nx.cycle_graph(6))
            # The first structural change keeps every ID and SU assignment:
            # an edge-free assignment hash must not veto this real progress.
            offset = 0 if topology_changes and calls['rebuilds'] == 1 else 100 * calls['rebuilds']
            return (seed_nodes(offset=offset, graph=graph),
                    H.clone(), {'r2': .7}, .7)

        monkeypatch.setattr(adjuster, 'plan_topology_resource_adjustments', plan)
        monkeypatch.setattr(adjuster, 'apply_topology_resource_adjustments', apply)
        monkeypatch.setattr(block_c, '_append_stage3_parity_balance',
                            lambda **options: (True, {'enforced': False, 'balance_applied': False}))
        result = adjuster.run_block_c_resource_adjustment_loop(
            nodes=deepcopy(nodes), H=H.clone(), E_target=None, difference_spectrum={'r2': .7},
            rebuild_fn=rebuild, validate_fn=lambda **options: {'ok': True, 'reason': 'ok'},
            enable_post_external_spectrum=False, max_no_progress_rebuilds=3,
            checkpoint_callback=(lambda **event: events.append(event)) if detailed else None,
        )
        assert not result.ok
        assert result.stop_reason == 'stage3_no_progress_rebuild_limit'
        assert calls['rebuilds'] == 3 + int(topology_changes)
        assert result.accepted_cycles == int(topology_changes)
        assert any(row.stop_reason == 'stage3_equivalent_state_detected' for row in result.cycles)
        assert [node.global_id for node in result.nodes] == list(range(6))
        assert set(result.nodes[0].hop1_ids) == ({1, 2} if topology_changes else {1, 5})
        assert torch.equal(result.H, H) and result.diff['r2'] == .7
        assert result.diagnostics['search_budget']['rebuild_count'] == 3 + int(topology_changes)
        if detailed:
            assert events[-1]['stage'] == 'stage3_search_stopped'
        outcomes.append(result.to_dict())
    assert outcomes[0] == outcomes[1]


def test_budget_rolls_back_provisional_external_conversions(monkeypatch):
    adjuster = Layer4Adjuster(device=torch.device('cpu'))
    nodes = graph_nodes(nx.cycle_graph(6))
    H = histogram(nodes)
    converted = deepcopy(nodes)
    converted[0].su_type = 12
    converted_H = histogram(converted)
    group = AtomicAdjustmentGroup(primary_op=HistogramConversion(
        from_su=13, to_su=12, count=1, node_ids=[0],
    ))
    plan = Stage3AdjustmentPlan(
        ok=True, reason='stage3_rebuild_required', phase='external_topology_assembly',
        histogram_adjustment_plan=HistogramAdjustmentPlan(
            groups=[group], H_before=H.clone(), H_after=converted_H.clone(),
        ),
    )
    monkeypatch.setattr(adjuster, 'plan_topology_resource_adjustments', lambda **options: plan)
    monkeypatch.setattr(adjuster, 'apply_topology_resource_adjustments', lambda **options:
                        SimpleNamespace(changed=True, nodes=converted, H=converted_H,
                                        applied_ops=[], rejected_ops=[], request_ops=[], unresolved_counts={}))
    result = adjuster.run_block_c_resource_adjustment_loop(
        H=H, nodes=nodes, E_target=None, difference_spectrum={'r2': .8},
        enable_post_external_spectrum=False, max_planner_attempts=1,
    )
    assert not result.ok and result.stop_reason == 'stage3_planner_attempt_limit'
    assert torch.equal(result.H, H)
    assert all(node.su_type == 13 for node in result.nodes)
    assert result.diff == {'r2': .8}
    terminal = result.diagnostics['terminal_snapshot']
    assert not terminal['stage3_complete']
    assert terminal['infeasibility_proof']['failure_domain'] == 'convergence'


def test_su24_followup_cannot_reaccept_an_old_graph_or_keep_its_runtime(monkeypatch):
    memory = Stage3StateMemory()
    nodes = graph_nodes(nx.cycle_graph(6))
    H = histogram(nodes)
    memory.identify(nodes, H, remember=True)
    adjuster = SimpleNamespace(_stage3_external_assembler='baseline assembler')
    contexts = []

    def nested(adjuster, **options):
        adjuster._stage3_external_assembler = 'provisional assembler'
        return Stage3WorkflowResult(
            ok=True, nodes=graph_nodes(nx.cycle_graph(6), offset=100),
            H=H.clone(), diff={'r2': .8},
        )

    monkeypatch.setattr(block_c, 'run_block_c_resource_adjustment_loop_impl', nested)
    result, accepted = block_c._evaluate_su24_next_cycle(
        adjuster, nodes=nodes, H=H, difference={'r2': .9}, baseline_r2=.8,
        baseline_context='baseline context', candidate_context='candidate context',
        context_restore_fn=contexts.append, cycle_number=1, generation=2,
        options={}, state_memory=memory,
    )
    assert not accepted
    assert result.diagnostics['su24_evaluation']['reason'] == 'su24_next_cycle_equivalent_state'
    assert adjuster._stage3_external_assembler == 'baseline assembler'
    assert contexts[-1] == 'baseline context'


@pytest.mark.parametrize('spectrum_finalized', [False, True])
def test_pipeline_saves_a_replayable_budget_stop_checkpoint(monkeypatch, tmp_path, spectrum_finalized):
    from model.inverse.inverse_pipeline import InversePipelineV3

    pipeline = InversePipelineV3(torch.nn.Identity(), device='cpu')
    nodes = graph_nodes(nx.cycle_graph(6))
    H = histogram(nodes)
    difference = {'source_stage': 'layer2', 'spectral_contract': 'template_center_mu',
                  'r2': .8}
    target = torch.zeros(2400)
    elements = torch.zeros(6)
    workflow = Stage3WorkflowResult(
        nodes=nodes, H=H, diff=difference, ok=False,
        stop_reason='stage3_rebuild_limit',
        diagnostics={'search_budget': {'stop_reason': 'stage3_rebuild_limit'},
                     'spectrum_rebalance_finalized': spectrum_finalized},
    )
    monkeypatch.setattr(pipeline.layer4_adjuster, 'run_block_c_resource_adjustment_loop',
                        lambda **options: workflow)
    _, returned_H, _, meta = pipeline._run_topology_stage3_resource_cycles(
        nodes=nodes, H=H, current_diff=difference, S_target=target, E_target=elements,
        eval_lib_path='unused.pt', layer1_runtime_cfg={}, outer_idx=0,
        output_dir=str(tmp_path), max_rebuilds=2,
    )
    saved = Path(meta['resume_snapshot_dir'])
    manifest = json.loads((saved / 'snapshot_manifest.json').read_text())
    assert manifest['format'] == 'graphnmr_stage3_input_snapshot'
    for name, checksum in manifest['sha256'].items():
        assert pipeline._stage3_snapshot_sha256(saved / name) == checksum
    saved_H = pipeline._load_stage3_snapshot_torch(saved / 'H.pt', map_location='cpu')
    assert torch.equal(saved_H, H) and torch.equal(returned_H, H)
    config = json.loads((saved / 'stage3_resource_config.json').read_text())
    assert config['enable_post_external_spectrum'] is (not spectrum_finalized)
    assert config['max_rebuilds'] == 2
    assert not meta['ok'] and meta['completion_diagnostics'] == {}
    assert meta['terminal_diagnostics']['resume_snapshot_dir'] == str(saved)
