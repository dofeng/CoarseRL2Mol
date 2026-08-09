from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from model.inverse.topology.classification.classifier import classify_topology_units
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.phase_eval import PHASE_SEQUENCE, _compute_flex_audit, build_stage3_phase_evaluation
from model.inverse.topology.round_context import round_context_invariants
from model.inverse.topology.spectrum_rebalance import (
    DEFAULT_SPECTRUM_WINDOWS,
    apply_spectrum_candidate,
    generate_spectrum_candidates,
    spectrum_candidate_balance_penalty,
)
from model.inverse.topology.stage3_state_machine import (
    _filter_spectrum_candidates_by_stage3_hard_constraint,
    _partition_active_external_contracts,
    _residual_contracts,
    plan_stage3_state_machine,
)
from model.inverse.topology.stage3_models import (
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    Stage3ProgressPhase,
    Stage3SpectrumCandidate,
)
from model.inverse.topology.workflow import plan_stage3_resource_adjustments
from model.inverse.layer4.inverse_layer4_block_c import run_block_c_resource_adjustment_loop_impl
from tests.inverse.topology.helpers import connect, make_node


def _histogram(nodes):
    H = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1
    return H


def _nodes_from_histogram(H):
    nodes = []
    global_id = 0
    for su_type, count in enumerate(torch.clamp(H.detach().cpu(), min=0).long().tolist()):
        for _ in range(int(count)):
            nodes.append(make_node(global_id, su_type))
            global_id += 1
    return nodes


def test_phase_evaluation_is_pure_and_builds_round_ownership() -> None:
    nodes = [make_node(1, 11), make_node(2, 12), make_node(3, 24)]
    classification = classify_topology_units(nodes)
    evaluation = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
        generation_id=7,
    )

    assert evaluation.generation_id == 7
    assert evaluation.assembly_attempted is False
    assert evaluation.assembly_view["products"] == []
    assert evaluation.assembly_view["diagnostics"]["evaluation_only"] is True
    assert evaluation.closure_pool_ids_by_su[11] == [1]
    assert evaluation.closure_pool_ids_by_su[24] == [3]
    assert evaluation.aromatic_donor_ids_by_su[12] == [2]
    assert round_context_invariants(evaluation.round_context) == []


def test_phase_sequence_places_spectrum_before_su24_and_residual() -> None:
    assert PHASE_SEQUENCE == (
        "flex_range_reconciliation",
        "external_flex_closure",
        "external_fatty_ring_closure",
        "external_side_chain_closure",
        "remaining_pool_spectrum_rebalance",
        "resource_su24_ring_closure",
        "residual_su24_aromatization",
        "residual_resource_self_closure",
        "complete",
    )


def test_stage3_progress_rejects_string_phase_compatibility() -> None:
    nodes = [make_node(1, 22)]
    classification = classify_topology_units(nodes)
    evaluation = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )

    with pytest.raises(TypeError, match="Stage3ProgressPhase"):
        plan_stage3_state_machine(
            phase_eval=evaluation,
            H=_histogram(nodes),
            nodes=nodes,
            classification=classification,
            progress_phase="residual_resource_self_closure",
        )


def test_post_external_progress_never_reassembles_completed_external_phases(
    monkeypatch,
) -> None:
    nodes = [make_node(1, 22)]
    classification = classify_topology_units(nodes)
    evaluation = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **_kwargs: pytest.fail("completed external phases were re-executed"),
    )

    plan = plan_stage3_state_machine(
        phase_eval=evaluation,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
    )

    assert plan.reason == "stage3_complete"
    restored_evaluation = plan.runtime["phase_evaluation"]
    assert restored_evaluation.diagnostics[
        "external_stage_restored_from_committed_products"
    ] is True


def test_later_side_contract_is_dormant_while_ring_phase_is_active() -> None:
    phase = Stage3PhaseResult(active_phase="external_fatty_ring_closure")
    phase.active_phase = "external_fatty_ring_closure"
    contracts = [
        {"contract_id": 300, "target_role": "fatty_ring"},
        {"contract_id": 401, "target_role": "side_chain"},
    ]

    active, dormant, active_phase, active_role = _partition_active_external_contracts(
        phase,
        contracts,
    )

    assert [item["contract_id"] for item in active] == [300]
    assert [item["contract_id"] for item in dormant] == [401]
    assert active_phase == "external_fatty_ring_closure"
    assert active_role == "fatty_ring"


def test_flex_upper_uses_exact_point_eight_formula_without_lower_bound_raise(monkeypatch) -> None:
    nodes = [make_node(index, 10) for index in range(4)]
    classification = classify_topology_units(nodes)
    monkeypatch.setattr(
        "model.inverse.topology.phase_eval._count_clusters_from_generator",
        lambda *_args, **_kwargs: (10, "test", ""),
    )

    audit = _compute_flex_audit(nodes, classification)

    assert audit["X"] == 10
    assert audit["Y"] == 2
    assert audit["Z"] == 8
    assert audit["flex_lower"] == 9
    assert audit["flex_upper"] == 8
    assert audit["flex_window_ok"] is False
    assert audit["blocked_reason"] == "flex_window_invalid"


def test_spectrum_candidates_use_only_remaining_ids_and_do_not_reserve_su24() -> None:
    H = torch.zeros(40, dtype=torch.long)
    H[23] = 2
    H[24] = 1
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 32.0) & (ppm <= 48.0)] = 1.0
    diff[(ppm >= 22.0) & (ppm <= 36.0)] -= 1.0

    candidates, audit = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.5},
        closure_pool_ids_by_su={11: [], 22: [], 23: [101, 102], 24: [103]},
        aromatic_donor_ids_by_su={10: [], 12: [], 13: []},
        candidate_limit=8,
        max_primary_adjustments=10,
        E_target=torch.tensor([500.0, 520.0]),
    )

    assert candidates
    atomic = candidates[0]
    assert all(int(op["node_id"]) in {101, 102} for candidate in candidates for op in candidate.operations)
    assert all(
        int(op["from_su"]) == 23 and int(op["to_su"]) == 24
        for candidate in candidates for op in candidate.operations
    )
    assert atomic.primary_adjustment_count == 1
    assert atomic.diagnostics["bundle_kind"] == "atomic"
    assert audit["does_not_reserve_su24"] is True


def test_spectrum_windows_match_authoritative_su_ranges() -> None:
    assert DEFAULT_SPECTRUM_WINDOWS == {
        11: (139.0, 165.0, 139.0, 165.0),
        12: (126.0, 142.0, 126.0, 142.0),
        13: (100.0, 131.0, 100.0, 131.0),
        22: (8.0, 25.0, 8.0, 25.0),
        23: (10.0, 45.0, 10.0, 45.0),
        24: (15.0, 60.0, 15.0, 60.0),
    }


def test_spectrum_bundle_uses_atomic_progressive_and_full_candidates() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13] = 460
    H[22] = 20
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 136.0) & (ppm <= 145.0)] = 8.0
    diff[(ppm >= 10.0) & (ppm <= 21.0)] = -8.0

    candidates, audit = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.5},
        closure_pool_ids_by_su={22: list(range(100, 120))},
        aromatic_donor_ids_by_su={},
        candidate_limit=4,
        max_primary_adjustments=10,
        E_target=torch.tensor([500.0, 520.0]),
    )

    assert [candidate.primary_adjustment_count for candidate in candidates] == [1, 4, 7, 10]
    assert all(int(op["from_su"]) == 22 for candidate in candidates for op in candidate.operations)
    assert all(int(op["to_su"]) in {11, 12} for candidate in candidates for op in candidate.operations)
    assert candidates[0].diagnostics["bundle_strategy"] == "direct_difference_spectrum_proportion"
    assert candidates[0].diagnostics["bundle_kind"] == "atomic"
    assert candidates[-1].diagnostics["bundle_kind"] == "full"
    assert audit["carbon_batch_cap"] == 10
    assert audit["effective_batch_cap"] == 10
    assert audit["ranking_policy"] == "atomic_then_progressive_direct_proportional_bundles"
    assert audit["spectrum_first_ranking"] is True
    assert audit["h_soft_window_aware_ranking"] is True


def test_spectrum_bundle_compensates_low_h_with_remaining_su11_only_outside_window() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[11] = 8
    H[13] = 474
    H[22] = 10
    H[23] = 8
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 136.0) & (ppm <= 145.0)] = 8.0
    diff[(ppm >= 10.0) & (ppm <= 21.0)] = -8.0

    candidates, audit = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.5},
        closure_pool_ids_by_su={
            11: list(range(200, 208)),
            22: list(range(100, 110)),
        },
        aromatic_donor_ids_by_su={},
        candidate_limit=2,
        max_primary_adjustments=10,
        max_h_compensation_ops=8,
        E_target=torch.tensor([500.0, 520.0]),
    )

    assert len(candidates) == 2
    atomic, full = candidates
    assert full.candidate_id.startswith("spectrum:proportional_full:10:")
    assert full.primary_adjustment_count == 10
    assert full.diagnostics["h_compensation_count"] == 8
    assert full.diagnostics["h_compensation"]["h_after_primary"] == pytest.approx(490.0)
    assert full.diagnostics["h_compensation"]["h_after_compensation"] == pytest.approx(506.0)
    assert full.diagnostics["h_compensation"]["complete"] is True
    assert len(full.operations) == 18
    assert all(
        int(op["from_su"]) == 11 and int(op["to_su"]) == 23
        for op in full.operations[10:]
    )
    assert len({int(op["node_id"]) for op in full.operations}) == 18

    assert atomic.primary_adjustment_count == 1
    assert atomic.diagnostics["h_compensation_count"] == 0
    assert atomic.diagnostics["h_compensation"]["h_after_primary"] == pytest.approx(517.0)
    assert atomic.diagnostics["h_compensation"]["reason"] == "not_required"
    assert audit["max_h_compensation_ops"] == 8


def test_spectrum_peak_prominence_can_dominate_mixed_signed_window() -> None:
    H = torch.zeros(40, dtype=torch.long)
    H[22] = 2
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 129.0) & (ppm <= 145.0)] = -1.0
    diff[np.argmin(np.abs(ppm - 140.0))] = 50.0
    diff[(ppm >= 10.0) & (ppm <= 21.0)] = -2.0

    candidates, audit = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.5},
        closure_pool_ids_by_su={22: [100, 101]},
        aromatic_donor_ids_by_su={},
        candidate_limit=1,
        E_target=torch.tensor([500.0, 520.0]),
    )

    assert len(candidates) == 1
    assert all(int(op["from_su"]) == 22 for op in candidates[0].operations)
    assert all(int(op["to_su"]) in {11, 12} for op in candidates[0].operations)
    assert audit["urgency_by_su"]["11"]["direction"] == "deficit"
    assert audit["urgency_by_su"]["11"]["deficit_score"] > audit["urgency_by_su"]["11"]["surplus_score"]


def test_spectrum_bundle_apportions_multiple_sources_and_targets() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[22] = 20
    H[23] = 20
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 10.0) & (ppm <= 21.0)] = -8.0
    diff[(ppm >= 26.0) & (ppm <= 29.0)] = -4.0
    diff[(ppm >= 136.0) & (ppm <= 145.0)] = 8.0
    diff[(ppm >= 105.0) & (ppm <= 121.0)] = 4.0

    candidates, _ = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff},
        closure_pool_ids_by_su={
            22: list(range(100, 120)),
            23: list(range(200, 220)),
        },
        aromatic_donor_ids_by_su={},
        candidate_limit=2,
        max_primary_adjustments=10,
        E_target=torch.tensor([500.0, 520.0]),
    )

    full = candidates[-1]
    source_counts = full.diagnostics["source_counts"]
    target_counts = full.diagnostics["target_counts"]
    assert full.primary_adjustment_count == 10
    assert int(source_counts["22"]) > int(source_counts["23"])
    assert int(target_counts["11"]) >= int(target_counts["13"])
    assert sum(int(value) for value in source_counts.values()) == 10
    assert sum(int(value) for value in target_counts.values()) == 10
    assert sum(int(value) for value in full.diagnostics["conversion_counts"].values()) == 10


def test_spectrum_candidate_application_changes_su_only_and_preserves_hop1() -> None:
    node = make_node(101, 23)
    node.hop1_ids = [88]
    node.hop1_su = {11: 1}
    H = _histogram([node])
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 32.0) & (ppm <= 48.0)] = 1.0
    diff[(ppm >= 22.0) & (ppm <= 36.0)] -= 1.0
    candidates, _ = generate_spectrum_candidates(
        H=H,
        difference_spectrum={"ppm": ppm, "diff": diff},
        closure_pool_ids_by_su={23: [101], 24: []},
        aromatic_donor_ids_by_su={},
        candidate_limit=1,
        E_target=torch.tensor([50.0, 0.0]),
    )
    before_hop1_ids = deepcopy(node.hop1_ids)
    before_hop1_su = deepcopy(node.hop1_su)

    nodes_after, H_after, proof = apply_spectrum_candidate(nodes=[node], H=H, candidate=candidates[0])

    assert proof == {}
    assert nodes_after[0].su_type == 24
    assert nodes_after[0].hop1_ids == before_hop1_ids
    assert nodes_after[0].hop1_su == before_hop1_su
    assert H_after[23].item() == 0 and H_after[24].item() == 1


def test_residual_flex_contract_prefers_three_then_uses_legal_one_su23_fallback() -> None:
    pool = {11: [1, 2, 3, 4], 22: [], 23: [10, 11, 12, 13], 24: []}
    contracts = _residual_contracts(
        pool,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        max_flex_chains=2,
    )
    flex = [item for item in contracts if item.diagnostics.get("product_role") == "residual_flex_chain"]

    assert flex
    lengths = [int(item.required_counts[23]) for item in flex]
    assert lengths[0] == 3
    assert lengths == [3, 1]


def test_above_range_uses_authorized_donor_then_requires_rebuild() -> None:
    nodes = []
    for offset in (0, 10, 20):
        center = make_node(1 + offset, 3)
        left = make_node(2 + offset, 9)
        right = make_node(3 + offset, 9)
        connect(center, left)
        connect(center, right)
        nodes.extend([center, left, right])
    donor = make_node(100, 12)
    nodes.append(donor)
    before_hop1 = {
        int(node.global_id): (set(node.hop1_ids), dict(node.hop1_su))
        for node in nodes
    }

    phase = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    assert phase.flex_audit["scenario"] == "above_range"
    assert phase.aromatic_donor_ids_by_su[12] == [100]
    phase.flex_reconciliation_block_reason = ""
    phase.flex_audit["blocked_reason"] = ""
    phase.flex_audit["above_range_excess"] = 1

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
    )
    applied = apply_stage3_node_adjustments(nodes, _histogram(nodes), plan)

    assert plan.phase == "flex_range_reconciliation"
    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase_result.rebuild_required is True
    assert plan.phase_result.primary_adjustment_count <= 10
    assert {int(item.node_id) for item in plan.planned_conversions if item.node_id is not None} == {100}
    assert applied.changed is True
    assert next(node for node in applied.nodes if node.global_id == 100).su_type == 11
    assert {
        int(node.global_id): (set(node.hop1_ids), dict(node.hop1_su))
        for node in applied.nodes
    } == before_hop1


def test_one_remaining_su24_is_aromatized_with_rebuild() -> None:
    nodes = [make_node(1, 24)]
    plan = plan_stage3_resource_adjustments(
        _histogram(nodes),
        nodes,
        enable_post_external_spectrum=False,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.plan_kind == "residual_su24_aromatization"
    assert plan.phase == "residual_su24_aromatization"
    assert plan.phase_result.rebuild_required is True
    assert plan.histogram_adjustment_plan.H_after[24].item() == 0
    assert plan.histogram_adjustment_plan.H_after[12].item() == 1


def _external_budget_fixture():
    nodes = [make_node(10, 12), make_node(11, 12), make_node(12, 13)]
    phase = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase.flex_reconciliation_block_reason = ""
    phase.flex_audit = {"enforced": False}
    phase.canonical_contracts = [
        {
            "contract_id": index,
            "contract_kind": "mainline_closure",
            "target_role": role,
            "source_fragment_ids": [],
            "claimed_source_node_ids": [],
            "context_node_ids": [],
            "mainline_slots": [],
            "branch_slots": [],
            "required_counts": {"11": 1},
            "expected_topology": f"external-{index}",
            "reason": "test_external_budget",
            "blocker_kind": "",
            "diagnostics": {},
        }
        for index, role in enumerate(
            ("flex_chain", "fatty_ring", "side_chain"),
            start=1,
        )
    ]
    return nodes, phase


def test_external_resource_conversions_are_batched_before_fresh_physical_assembly(
    monkeypatch,
) -> None:
    nodes, phase = _external_budget_fixture()
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **kwargs: kwargs["phase_eval"],
    )

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
        max_primary_adjustments_per_stage=10,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "external_topology_assembly"
    assert plan.histogram_adjustment_plan.primary_adjustment_count == 3
    assert plan.phase_result.phase_completed is False
    assert plan.diagnostics["external_stage_completed"] is False
    assert plan.diagnostics["physical_materialization_required_after_rebuild"] is True
    assert plan.diagnostics["external_reserved_after_conversion"] == 3
    assert plan.diagnostics["external_completed_after_conversion"] == 0
    assert plan.diagnostics["external_rebuild_reason"] == (
        "resource_conversion_requires_fresh_assembly"
    )
    candidate_transactions = [
        transaction
        for transaction in plan.phase_result.transactions
        if transaction.transaction_origin == "current_plan_candidate"
    ]
    assert len(candidate_transactions) == 3
    claimed_ids = [
        int(node_id)
        for transaction in candidate_transactions
        for node_ids in transaction.claimed_resource_ids.values()
        for node_id in node_ids
    ]
    assert len(claimed_ids) == 3
    assert len(set(claimed_ids)) == len(claimed_ids)


def test_available_resources_report_topology_materialization_failure(monkeypatch) -> None:
    nodes = [make_node(1, 11)]
    phase = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase.flex_reconciliation_block_reason = ""
    phase.flex_audit = {"enforced": False}
    phase.canonical_contracts = [{
        "contract_id": 7,
        "contract_kind": "mainline_closure",
        "target_role": "flex_chain",
        "source_fragment_ids": [],
        "claimed_source_node_ids": [],
        "context_node_ids": [],
        "mainline_slots": [{
            "slot_id": "external:7:right:F1",
            "kind": "F1",
            "anchor_node_id": -1,
            "required_su_sequence": [11],
            "state": "OPEN",
        }],
        "branch_slots": [],
        "required_counts": {"11": 1},
        "expected_topology": "OPEN-11",
        "reason": "main_pool_shortage",
        "blocker_kind": "",
        "diagnostics": {"materialization_failure": {"validation_reason": "test"}},
    }]
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **kwargs: kwargs["phase_eval"],
    )

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
    )

    assert plan.reason == "phase_resource_infeasible"
    failed = plan.phase_result.infeasibility_proof["failed_contract"]
    assert failed["proof"]["reason"] == "external_topology_materialization_invalid"
    assert failed["proof"]["failure_domain"] == "topology_materialization"
    assert plan.blocked_reason == "stage3_topology_materialization_invalid"
    assert plan.plan_kind == "sequential_phase_topology_materialization_invalid"
    assert failed["proof"]["missing_counts"] == {}
    assert plan.diagnostics["external_stage_completed"] is False


def test_open_or_provisional_external_product_cannot_complete_stage3(monkeypatch) -> None:
    nodes = [make_node(1, 11)]
    phase = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase.flex_reconciliation_block_reason = ""
    phase.flex_audit = {"enforced": False}
    phase.assembly_view = {
        **dict(phase.assembly_view or {}),
        "products": [{
            "product_id": "provisional:1",
            "product_type": "flex_chain",
            "fixed_source_node_ids": [],
            "claimed_resource_ids_by_su": {"11": [1]},
            "topology_notation": "OPEN-11",
            "node_id_notation": "OPEN-1",
            "closed": False,
            "provisional": True,
        }],
    }
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **kwargs: kwargs["phase_eval"],
    )

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
    )

    assert plan.reason == "phase_resource_infeasible"
    assert plan.blocked_reason == "external_physical_product_validation_failed"
    errors = plan.phase_result.infeasibility_proof["materialization_errors"]
    assert errors[0]["reasons"] == [
        "product_not_closed",
        "provisional_product_forbidden",
        "topology_not_fully_materialized",
        "node_identity_not_fully_materialized",
    ]


def test_external_conversion_budget_does_not_mark_pending_roles_complete(monkeypatch) -> None:
    nodes, phase = _external_budget_fixture()
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **kwargs: kwargs["phase_eval"],
    )

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
        max_primary_adjustments_per_stage=2,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "external_topology_assembly"
    assert plan.histogram_adjustment_plan.primary_adjustment_count == 2
    assert plan.phase_result.phase_completed is False
    assert plan.diagnostics["external_stage_completed"] is False
    assert plan.diagnostics["external_remaining_pending"] == 1
    assert plan.diagnostics["external_reserved_after_conversion"] == 2
    assert plan.diagnostics["external_completed_after_conversion"] == 0
    assert plan.diagnostics["external_conversion_budget_exhausted"] is True
    assert plan.diagnostics["external_rebuild_reason"] == (
        "resource_conversion_requires_fresh_assembly"
    )


def test_residual_does_not_synthesize_products_from_only_aromatic_donors(monkeypatch) -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 9)]
    phase = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase.flex_reconciliation_block_reason = ""
    phase.flex_audit = {"enforced": False}
    phase.flex_bounds = {}
    phase.flex_counts = {}
    phase.canonical_contracts = []
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **kwargs: kwargs["phase_eval"],
    )

    plan = plan_stage3_state_machine(
        phase_eval=phase,
        H=_histogram(nodes),
        nodes=nodes,
        E_target=torch.tensor([0.0, 1.0]),
        enable_post_external_spectrum=False,
        max_primary_adjustments_per_stage=10,
    )

    assert plan.reason == "stage3_complete"
    assert plan.phase == "complete"
    assert plan.histogram_adjustment_plan is None
    assert plan.phase_result.primary_adjustment_count == 0


def _spectrum_fixture():
    ppm = np.linspace(0.0, 200.0, 2001)
    diff = np.zeros_like(ppm)
    diff[(ppm >= 32.0) & (ppm <= 48.0)] = 1.0
    diff[(ppm >= 22.0) & (ppm <= 36.0)] -= 1.0
    improved = diff * 0.25
    return ppm, diff, improved


class _SpectrumAdjuster:
    def __init__(self, candidate):
        self.candidates = list(candidate) if isinstance(candidate, (list, tuple)) else [candidate]

    def plan_topology_resource_adjustments(self, **kwargs):
        if kwargs.get("progress_phase") is not Stage3ProgressPhase.SPECTRUM_REBALANCE:
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    stop_reason="stage3_complete",
                ),
            )
        return Stage3AdjustmentPlan(
            ok=True,
            reason="stage3_spectrum_candidates_ready",
            phase="remaining_pool_spectrum_rebalance",
            plan_kind="remaining_pool_spectrum_candidate_set",
            spectrum_candidates=list(self.candidates),
            phase_result=Stage3PhaseResult(
                active_phase="remaining_pool_spectrum_rebalance",
                stop_reason="stage3_spectrum_candidates_ready",
            ),
        )


def test_block_c_accepts_only_significant_spectrum_candidate() -> None:
    ppm, diff, improved = _spectrum_fixture()
    nodes = [make_node(101, 23), make_node(102, 23)]
    H = _histogram(nodes)
    candidate = Stage3SpectrumCandidate(
        candidate_id="strong",
        histogram_signature="hist-strong",
        inverse_signature="inverse-strong",
        operations=[
            {"node_id": 101, "from_su": 23, "to_su": 24},
            {"node_id": 102, "from_su": 23, "to_su": 24},
        ],
        target_su_types=[23, 24],
        primary_adjustment_count=2,
    )

    def rebuild(**kwargs):
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "ppm": ppm,
                "diff": improved,
                "r2": 0.56,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.56,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        nodes,
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.02,
    )

    assert result.accepted_cycles == 1
    assert int(result.H[23].item()) == 0
    assert int(result.H[24].item()) == 2
    assert result.cycles[0].acceptance_reason == "layer2_r2_gain_threshold_met"
    assert result.cycles[0].diagnostics["baseline_layer2_r2"] == 0.50
    assert "baseline_build_variant" not in result.cycles[0].diagnostics


def test_block_c_does_not_rebuild_odd_port_spectrum_candidate() -> None:
    ppm, diff, _ = _spectrum_fixture()
    node = make_node(101, 23)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="odd-port",
        histogram_signature="odd-port-histogram",
        inverse_signature="odd-port-inverse",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )
    rebuild_calls = []

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        [node],
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.accepted_cycles == 0
    assert rebuild_calls == []
    rejected = result.cycles[0].plan["diagnostics"]["rejected_spectrum_candidates"]
    assert rejected[0]["valid"] is False
    assert rejected[0]["significant"] is False
    assert rejected[0]["pre_rebuild_port_feasibility"]["ok"] is False


def test_block_c_balances_odd_spectrum_candidate_from_remaining_pool() -> None:
    ppm, diff, improved = _spectrum_fixture()
    nodes = [
        make_node(101, 23),
        make_node(102, 23),
        make_node(103, 23),
        make_node(104, 13),
    ]
    H = _histogram(nodes)
    candidate = Stage3SpectrumCandidate(
        candidate_id="odd-but-balanceable",
        histogram_signature="odd-but-balanceable-histogram",
        inverse_signature="odd-but-balanceable-inverse",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )

    class BalanceableSpectrumAdjuster(_SpectrumAdjuster):
        def __init__(self, value):
            super().__init__(value)
            self._stage3_parity_balance_cursor = 0

        def plan_topology_resource_adjustments(self, **kwargs):
            plan = super().plan_topology_resource_adjustments(**kwargs)
            if plan.reason == "stage3_spectrum_candidates_ready":
                plan.runtime = {
                    "iteration_topology_ledger": {
                        "available_ids_by_su": {
                            23: [101, 102, 103],
                            13: [104],
                        },
                        "ownership_by_node_id": {
                            node_id: {"state": "remaining_closure_resource"}
                            for node_id in (101, 102, 103, 104)
                        },
                    }
                }
            return plan

    rebuild_histograms = []

    def rebuild(**kwargs):
        H_seed = kwargs["H_seed"]
        rebuild_histograms.append(H_seed.detach().clone())
        return (
            _nodes_from_histogram(H_seed),
            H_seed,
            {
                "ppm": ppm,
                "diff": improved,
                "r2": 0.60,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.60,
        )

    adjuster = BalanceableSpectrumAdjuster(candidate)
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        nodes,
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.02,
    )

    assert len(rebuild_histograms) == 1
    assert int(rebuild_histograms[0][23].item()) == 2
    assert int(rebuild_histograms[0][24].item()) == 1
    assert int(rebuild_histograms[0][11].item()) == 1
    assert int(rebuild_histograms[0][12].item()) == 0
    assert int(rebuild_histograms[0][13].item()) == 0
    assert result.accepted_cycles == 1
    assert result.cycles[0].parity_balance_count == 2
    assert adjuster._stage3_parity_balance_cursor == 1


def test_block_c_rejects_post_layer3_spectrum_as_layer2_baseline() -> None:
    ppm, diff, _ = _spectrum_fixture()
    node = make_node(101, 23)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="wrong-baseline-stage",
        histogram_signature="wrong-baseline-stage-histogram",
        inverse_signature="wrong-baseline-stage-inverse",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )
    rebuild_calls = []

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        [node],
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.80,
            "source_stage": "post_layer3",
        },
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs) or (),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.accepted_cycles == 0
    assert rebuild_calls == []
    baseline = result.cycles[0].plan["diagnostics"]["spectrum_layer2_baseline"]
    assert baseline["valid"] is False
    assert baseline["reason"] == "spectrum_baseline_is_not_layer2"


def test_block_c_rejects_mixed_layer2_spectral_contract() -> None:
    ppm, diff, _ = _spectrum_fixture()
    node = make_node(101, 23)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="mixed-contract",
        histogram_signature="mixed-contract-histogram",
        inverse_signature="mixed-contract-inverse",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )
    rebuild_calls = []

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        [node],
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.80,
            "source_stage": "layer2",
            "spectral_contract": "g2s_layer3",
        },
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs) or (),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.accepted_cycles == 0
    assert rebuild_calls == []
    baseline = result.cycles[0].plan["diagnostics"]["spectrum_layer2_baseline"]
    assert baseline["valid"] is False
    assert baseline["reason"] == "spectrum_baseline_contract_mismatch"


def test_block_c_restores_context_for_selected_best_spectrum_candidate() -> None:
    ppm, diff, improved = _spectrum_fixture()
    nodes = [make_node(101, 23), make_node(102, 23)]
    H = _histogram(nodes)
    best = Stage3SpectrumCandidate(
        candidate_id="best",
        histogram_signature="hist-best",
        inverse_signature="inverse-best",
        operations=[
            {"node_id": 101, "from_su": 23, "to_su": 24},
            {"node_id": 102, "from_su": 23, "to_su": 24},
        ],
        target_su_types=[23, 24],
        primary_adjustment_count=2,
    )
    last = Stage3SpectrumCandidate(
        candidate_id="last",
        histogram_signature="hist-last",
        inverse_signature="inverse-last",
        operations=[
            {"node_id": 101, "from_su": 23, "to_su": 22},
            {"node_id": 102, "from_su": 23, "to_su": 22},
        ],
        target_su_types=[22, 23],
        primary_adjustment_count=2,
    )
    context = {"candidate": "baseline"}

    def rebuild(**kwargs):
        if int(kwargs["H_seed"][24].item()) == 2:
            context["candidate"] = "best"
            r2 = 0.60
        else:
            context["candidate"] = "last"
            r2 = 0.55
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "ppm": ppm,
                "diff": improved,
                "r2": r2,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            r2,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster([best, last]),
        H,
        nodes,
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=lambda snapshot: (context.clear(), context.update(snapshot)),
        spectrum_min_r2_gain=0.02,
    )

    assert result.accepted_cycles == 1
    assert result.nodes[0].su_type == 24
    assert context == {"candidate": "best"}


def test_block_c_rolls_back_weak_spectrum_candidate_and_continues() -> None:
    ppm, diff, improved = _spectrum_fixture()
    nodes = [make_node(101, 23), make_node(102, 23)]
    H = _histogram(nodes)
    candidate = Stage3SpectrumCandidate(
        candidate_id="weak",
        histogram_signature="hist-weak",
        inverse_signature="inverse-weak",
        operations=[
            {"node_id": 101, "from_su": 23, "to_su": 24},
            {"node_id": 102, "from_su": 23, "to_su": 24},
        ],
        target_su_types=[23, 24],
        primary_adjustment_count=2,
    )

    context = {"candidate": "baseline"}

    def weak_rebuild(**kwargs):
        context["candidate"] = "weak"
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "ppm": ppm,
                "diff": improved,
                "r2": 0.501,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.501,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        nodes,
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=weak_rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=lambda snapshot: (context.clear(), context.update(snapshot)),
        spectrum_min_r2_gain=0.02,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 0
    assert result.nodes[0].su_type == 23
    assert context == {"candidate": "baseline"}
    rejected = result.cycles[0].plan["diagnostics"]["rejected_spectrum_candidates"]
    assert rejected[0]["significant"] is False


def test_block_c_uses_saved_layer2_baseline_without_noop_rebuild() -> None:
    ppm, diff, improved = _spectrum_fixture()
    nodes = [make_node(101, 23), make_node(102, 23)]
    H = _histogram(nodes)
    candidate = Stage3SpectrumCandidate(
        candidate_id="variant-only",
        histogram_signature="variant-only-histogram",
        inverse_signature="variant-only-inverse",
        operations=[
            {"node_id": 101, "from_su": 23, "to_su": 24},
            {"node_id": 102, "from_su": 23, "to_su": 24},
        ],
        target_su_types=[23, 24],
        primary_adjustment_count=2,
    )

    rebuild_phases = []

    def rebuild(**kwargs):
        rebuild_phases.append(kwargs["phase"])
        assert "build_variant" not in kwargs
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "ppm": ppm,
                "diff": improved,
                "r2": 0.56,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.56,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        nodes,
        E_target=None,
        difference_spectrum={
            "ppm": ppm,
            "diff": diff,
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.02,
    )

    assert result.accepted_cycles == 1
    assert rebuild_phases == ["remaining_pool_spectrum_rebalance"]
    accepted = result.cycles[0].diagnostics["evaluated_candidates"][0]
    assert accepted["r2_before"] == 0.50
    assert accepted["r2_after"] == 0.56
    assert accepted["r2_gain"] == pytest.approx(0.06)


def test_block_c_spectrum_gain_outranks_h_balance_penalty(
    monkeypatch,
) -> None:
    nodes = [make_node(100 + index, 22) for index in range(20)]
    H = torch.zeros(600, dtype=torch.long)
    H[13] = 460
    H[22] = 20
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:22:11:10",
        histogram_signature="h-loss-histogram",
        inverse_signature="h-loss-inverse",
        operations=[
            {"node_id": 100 + index, "from_su": 22, "to_su": 11}
            for index in range(10)
        ],
        target_su_types=[11, 22],
        primary_adjustment_count=10,
    )
    monkeypatch.setattr(
        "model.inverse.layer4.inverse_layer4_block_c.spectrum_window_loss",
        lambda diff, _types: 0.0 if dict(diff or {}).get("trial") else 10.0,
    )

    def rebuild(**kwargs):
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "r2": 0.56,
                "trial": True,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.56,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        nodes,
        E_target=torch.tensor([0.0, 520.0]),
        difference_spectrum={
            "r2": 0.50,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.05,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert int(result.H[22].item()) == 10
    assert int(result.H[11].item()) == 10
    diagnostics = result.cycles[0].diagnostics
    assert diagnostics["r2_gain"] > 0.05
    assert diagnostics["balance_penalty"] > 0.0
    assert diagnostics["balanced_r2_gain"] < 0.05
    assert diagnostics["required_r2_gain"] == 0.05


def test_spectrum_h_balance_is_unpenalized_inside_three_percent_window() -> None:
    H_before = torch.zeros(600, dtype=torch.long)
    H_before[13] = 502
    H_before[22] = 5
    H_after_four = H_before.detach().clone()
    H_after_four[22] -= 4
    H_after_four[11] += 4
    H_after_five = H_before.detach().clone()
    H_after_five[22] -= 5
    H_after_five[11] += 5

    inside = spectrum_candidate_balance_penalty(
        H_before=H_before,
        H_after=H_after_four,
        E_target=torch.tensor([0.0, 520.0]),
    )
    outside = spectrum_candidate_balance_penalty(
        H_before=H_before,
        H_after=H_after_five,
        E_target=torch.tensor([0.0, 520.0]),
    )

    assert inside["h_before"] == 517.0
    assert inside["h_after"] == 505.0
    assert inside["h_penalty"] == 0.0
    assert inside["h_compensation_suppressed_within_window"] is True
    assert outside["h_after"] == 502.0
    assert outside["h_penalty"] > 0.0
    assert outside["h_compensation_required"] is True


def test_spectrum_plus_five_percent_keeps_soft_compensation_diagnostic() -> None:
    H_before = torch.zeros(600, dtype=torch.long)
    H_before[13] = 100
    H_after = H_before.detach().clone()
    H_after[13] = 105

    balance = spectrum_candidate_balance_penalty(
        H_before=H_before,
        H_after=H_after,
        E_target=torch.tensor([0.0, 100.0]),
    )

    assert balance["h_after"] == 105.0
    assert balance["h_compensation_required"] is True
    assert balance["spectrum_h_soft_lower_ratio"] == -0.03
    assert balance["spectrum_h_soft_upper_ratio"] == 0.03


def test_stage3_hard_filter_accepts_plus_five_and_rejects_plus_nine_percent() -> None:
    candidates = [
        Stage3SpectrumCandidate(
            candidate_id="plus-five",
            diagnostics={"preflight_balance": {"h_after": 105.0}},
        ),
        Stage3SpectrumCandidate(
            candidate_id="plus-nine",
            diagnostics={"preflight_balance": {"h_after": 109.0}},
        ),
    ]

    valid, rejected = _filter_spectrum_candidates_by_stage3_hard_constraint(
        candidates,
        target_h=100.0,
        h_tolerance=0.08,
    )

    assert [candidate.candidate_id for candidate in valid] == ["plus-five"]
    assert rejected == [
        {
            "candidate_id": "plus-nine",
            "h_after": 109.0,
            "reason": "stage3_hard_constraint_violation",
        }
    ]
