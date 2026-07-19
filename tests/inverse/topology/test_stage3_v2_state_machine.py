from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from model.inverse.topology.classification.classifier import classify_topology_units
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.phase_eval import PHASE_SEQUENCE, _compute_flex_audit, build_stage3_phase_evaluation
from model.inverse.topology.round_context import round_context_invariants
from model.inverse.topology.spectrum_rebalance import apply_spectrum_candidate, generate_spectrum_candidates
from model.inverse.topology.stage3_state_machine import _residual_contracts, plan_stage3_state_machine
from model.inverse.topology.stage3_models import (
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
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
        "residual_resource_self_closure",
        "complete",
    )


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
    )

    assert candidates
    first = candidates[0]
    assert all(int(op["node_id"]) in {101, 102} for op in first.operations)
    assert all(int(op["from_su"]) == 23 and int(op["to_su"]) == 24 for op in first.operations)
    assert audit["does_not_reserve_su24"] is True


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
    )
    before_hop1_ids = deepcopy(node.hop1_ids)
    before_hop1_su = deepcopy(node.hop1_su)

    nodes_after, H_after, proof = apply_spectrum_candidate(nodes=[node], H=H, candidate=candidates[0])

    assert proof == {}
    assert nodes_after[0].su_type == 24
    assert nodes_after[0].hop1_ids == before_hop1_ids
    assert nodes_after[0].hop1_su == before_hop1_su
    assert H_after[23].item() == 0 and H_after[24].item() == 1


def test_residual_flex_contract_prefers_three_and_never_uses_one_su23() -> None:
    pool = {11: [1, 2, 3, 4], 22: [], 23: [10, 11, 12, 13, 14], 24: []}
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
    assert all(length >= 2 for length in lengths)


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


def test_remaining_su24_requires_rebuild_before_completion() -> None:
    nodes = [make_node(1, 24)]
    plan = plan_stage3_resource_adjustments(
        _histogram(nodes),
        nodes,
        enable_post_external_spectrum=False,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.plan_kind == "residual_su24_to_su23_terminalization"
    assert plan.phase == "residual_resource_self_closure"
    assert plan.phase_result.rebuild_required is True


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
        if kwargs.get("skip_post_external_spectrum"):
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
    node = make_node(101, 23)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="strong",
        histogram_signature="hist-strong",
        inverse_signature="inverse-strong",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        [node],
        E_target=None,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.50},
        rebuild_fn=lambda **kwargs: (
            kwargs["identity_seed_nodes"],
            kwargs["H_seed"],
            {"ppm": ppm, "diff": improved, "r2": 0.56},
            0.56,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
        max_stage3_iterations=1,
        spectrum_min_r2_gain=0.02,
        spectrum_min_window_loss_gain=0.1,
    )

    assert result.accepted_cycles == 1
    assert result.nodes[0].su_type == 24
    assert result.cycles[0].acceptance_reason == "significant_r2_and_window_gain"


def test_block_c_restores_context_for_selected_best_spectrum_candidate() -> None:
    ppm, diff, improved = _spectrum_fixture()
    node = make_node(101, 23)
    H = _histogram([node])
    best = Stage3SpectrumCandidate(
        candidate_id="best",
        histogram_signature="hist-best",
        inverse_signature="inverse-best",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )
    last = Stage3SpectrumCandidate(
        candidate_id="last",
        histogram_signature="hist-last",
        inverse_signature="inverse-last",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 22}],
        target_su_types=[22, 23],
        primary_adjustment_count=1,
    )
    context = {"candidate": "baseline"}

    def rebuild(**kwargs):
        if int(kwargs["H_seed"][24].item()) == 1:
            context["candidate"] = "best"
            r2 = 0.60
        else:
            context["candidate"] = "last"
            r2 = 0.55
        return (
            kwargs["identity_seed_nodes"],
            kwargs["H_seed"],
            {"ppm": ppm, "diff": improved, "r2": r2},
            r2,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster([best, last]),
        H,
        [node],
        E_target=None,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.50},
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=lambda snapshot: (context.clear(), context.update(snapshot)),
        max_stage3_iterations=1,
        spectrum_min_r2_gain=0.02,
        spectrum_min_window_loss_gain=0.1,
    )

    assert result.accepted_cycles == 1
    assert result.nodes[0].su_type == 24
    assert context == {"candidate": "best"}


def test_block_c_rolls_back_weak_spectrum_candidate_and_continues() -> None:
    ppm, diff, improved = _spectrum_fixture()
    node = make_node(101, 23)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="weak",
        histogram_signature="hist-weak",
        inverse_signature="inverse-weak",
        operations=[{"node_id": 101, "from_su": 23, "to_su": 24}],
        target_su_types=[23, 24],
        primary_adjustment_count=1,
    )

    context = {"candidate": "baseline"}

    def weak_rebuild(**kwargs):
        context["candidate"] = "weak"
        return (
            kwargs["identity_seed_nodes"],
            kwargs["H_seed"],
            {"ppm": ppm, "diff": improved, "r2": 0.501},
            0.501,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        _SpectrumAdjuster(candidate),
        H,
        [node],
        E_target=None,
        difference_spectrum={"ppm": ppm, "diff": diff, "r2": 0.50},
        rebuild_fn=weak_rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=lambda snapshot: (context.clear(), context.update(snapshot)),
        max_stage3_iterations=1,
        spectrum_min_r2_gain=0.02,
        spectrum_min_window_loss_gain=0.1,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 0
    assert result.nodes[0].su_type == 23
    assert context == {"candidate": "baseline"}
    rejected = result.cycles[0].plan["diagnostics"]["rejected_spectrum_candidates"]
    assert rejected[0]["significant"] is False
