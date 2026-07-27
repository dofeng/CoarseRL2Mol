from __future__ import annotations

from collections import Counter
import inspect
from types import SimpleNamespace

import torch

from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
import model.inverse.layer4.inverse_layer4 as inverse_layer4_module
from model.inverse.layer4.inverse_layer4_block_c import (
    _print_stage3_histogram_adjustments,
    _print_stage3_plan_summary,
    _rollback_stage3_plan_transactions,
    _stage3_terminal_infeasibility_proof,
    run_block_c_resource_adjustment_loop_impl,
)
from model.inverse.layer1.inverse_layer1 import Layer1Assigner
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.histogram_adjustment import _h_total, apply_histogram_adjustment_plan
from model.inverse.topology.iteration_ledger import build_iteration_ledger
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    Stage3AdjustmentBundle,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    Stage3SpectrumCandidate,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyResourceTransaction,
)
from model.inverse.topology.stage3_state_machine import (
    _evaluate_external_phases,
    _iteration_ledger_for_phase_eval,
    _ledger_transaction,
    _plan_next_residual_product,
    _residual_orphan_terminalization_plan,
    _resource_ids,
    _residual_contracts,
    _transaction_plan,
    build_su24_resource_role_profiles,
    plan_stage3_state_machine,
)
from model.inverse.topology.phase_eval import (
    PHASE_SEQUENCE,
    build_stage3_phase_evaluation,
    classify_topology_units,
)
from model.inverse.topology.workflow import plan_stage3_resource_adjustments
from model.inverse.topology.resource_profiles import (
    raw_su24_segment_support,
    raw_su24_support_counts,
)

from .helpers import connect, make_node


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


def _next_residual_product(
    pool,
    H,
    *,
    E_target=None,
    remaining_flex_chains=0,
    remaining_flex_capacity=None,
    max_conversions=10,
    max_h_compensation_ops=8,
):
    nodes = [
        make_node(int(node_id), int(su_type))
        for su_type, node_ids in dict(pool or {}).items()
        for node_id in list(node_ids or [])
    ]
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase_eval = _evaluate_external_phases(
        phase_eval=phase_eval,
        nodes=nodes,
    )
    available = _resource_ids(phase_eval)
    ledger = _iteration_ledger_for_phase_eval(
        phase_eval,
        nodes,
        available,
        [],
    )
    return _plan_next_residual_product(
        phase_eval=phase_eval,
        iteration_ledger=ledger,
        H=H,
        E_target=E_target,
        h_tolerance=0.06,
        special_degree_meta=None,
        su11_rotation_index=0,
        remaining_flex_chains=remaining_flex_chains,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        max_conversions=max_conversions,
        max_h_compensation_ops=max_h_compensation_ops,
        contract_id=2_000_000,
        remaining_flex_capacity=remaining_flex_capacity,
    )


def test_stage3_defaults_have_no_global_iteration_or_spectrum_acceptance_budget() -> None:
    layer4_defaults = inspect.signature(
        Layer4Adjuster.run_block_c_resource_adjustment_loop
    ).parameters
    assert "max_stage3_iterations" not in layer4_defaults
    assert "spectrum_max_accepted_rounds" not in layer4_defaults
    assert layer4_defaults["spectrum_candidate_limit"].default == 2
    assert layer4_defaults["spectrum_max_primary_adjustments"].default == 10

    for function in (plan_stage3_resource_adjustments, plan_stage3_state_machine):
        defaults = inspect.signature(function).parameters
        assert defaults["spectrum_candidate_limit"].default == 2
        assert defaults["spectrum_max_primary_adjustments"].default == 10


def test_layer4_wrapper_forwards_block_c_resource_configuration(monkeypatch) -> None:
    captured = {}

    def fake_impl(adjuster, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(inverse_layer4_module, "run_block_c_resource_adjustment_loop_impl", fake_impl)
    Layer4Adjuster().run_block_c_resource_adjustment_loop(
        H=torch.zeros(40, dtype=torch.long),
        nodes=[],
        E_target=torch.zeros(6),
        max_primary_adjustments_per_stage=6,
        max_h_compensation_ops_per_adjustment=8,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        h_tolerance=0.06,
        iteration_offset=10,
    )
    assert "max_stage3_iterations" not in captured
    assert captured["max_primary_adjustments_per_stage"] == 6
    assert captured["max_h_compensation_ops_per_adjustment"] == 8
    assert captured["residual_flex_preferred_23"] == 3
    assert captured["residual_flex_min_23"] == 2
    assert captured["residual_side_preferred_23"] == 2
    assert captured["residual_side_min_23"] == 1
    assert captured["h_tolerance"] == 0.06
    assert captured["iteration_offset"] == 10


def test_layer4_stage_router_forwards_complete_block_c_configuration(monkeypatch) -> None:
    captured = {}

    def fake_impl(adjuster, H, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            H=H,
            cycles=[],
            to_dict=lambda: {},
        )

    monkeypatch.setattr(inverse_layer4_module, "run_block_c_resource_adjustment_loop_impl", fake_impl)
    Layer4Adjuster().adjust_by_stage(
        H=torch.zeros(40, dtype=torch.long),
        ppm=None,
        diff=None,
        E_target=torch.zeros(6),
        stage="block_c_resource",
        nodes=[],
        max_h_compensation_ops_per_adjustment=7,
        residual_flex_preferred_23=4,
        residual_flex_min_23=3,
        residual_side_preferred_23=3,
        residual_side_min_23=2,
        h_tolerance=0.05,
        outer_cycle=6,
        iteration_offset=9,
    )

    assert captured["max_h_compensation_ops_per_adjustment"] == 7
    assert captured["residual_flex_preferred_23"] == 4
    assert captured["residual_flex_min_23"] == 3
    assert captured["residual_side_preferred_23"] == 3
    assert captured["residual_side_min_23"] == 2
    assert captured["h_tolerance"] == 0.05
    assert captured["outer_cycle"] == 6
    assert captured["iteration_offset"] == 9


def test_layer4_does_not_forward_removed_block_b_tail_configuration(monkeypatch) -> None:
    del monkeypatch
    source = inspect.getsource(inverse_layer4_module.adjust_block_b_hetero_anchor_impl)
    assert "tail_spectrum" not in source


def test_stage3_phase_sequence_is_complete_and_ordered() -> None:
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


def test_residual_side_contracts_are_not_blocked_by_optional_flex_target() -> None:
    pool = {11: [1], 12: [], 13: [], 22: [2], 23: [3, 4], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    direct = _residual_contracts(pool, max_flex_chains=1)
    assert len(direct) == 1
    assert direct[0].expected_topology == "11-23-23-22"
    assert direct[0].diagnostics["product_role"] == "residual_side_chain"


def test_symbolic_product_operation_is_not_applied_to_node_graph() -> None:
    nodes = [make_node(1, 19), make_node(2, 23), make_node(3, 11)]
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        adjustment_bundles=[Stage3AdjustmentBundle(bundle_ops=[{
            "op": "symbolic_topology_product",
            "unit_id": 1,
            "target_role": "flex_chain",
            "source_node_ids": [1],
            "claimed_resource_ids_by_su": {"23": [2], "11": [3]},
        }])],
    )
    result = apply_stage3_node_adjustments(nodes, _histogram(nodes), plan)
    assert result.changed is False
    assert not any(item.get("op") == "symbolic_topology_product" for item in result.applied_ops)
    assert all(not list(getattr(node, "hop1_ids", []) or []) for node in result.nodes)
    assert all(not set(getattr(node, "fixed_hop1_ids", set()) or set()) for node in result.nodes)
    assert all(getattr(node, "stage3_topology_owner_id", None) is None for node in result.nodes)


def test_histogram_adjustment_cannot_consume_fixed_global_resource() -> None:
    nodes = [make_node(index, 24) for index in range(1, 6)]
    H = _histogram(nodes)
    plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(24, 23, 1, "test"),
            combined_delta={24: -1, 23: 1},
        )],
        inventory_before={24: 0, 23: 0},
    )
    result = apply_histogram_adjustment_plan(H, plan, nodes=nodes)
    assert result.changed is False
    assert result.rejected_ops[0]["reason"] == "donor_not_in_authorized_stage3_pool"


def test_su24_resources_only_convert_the_count_above_residual_tolerance() -> None:
    nodes = [make_node(7, 24), make_node(3, 24), make_node(9, 24), make_node(11, 24)]
    plan = plan_stage3_resource_adjustments(H=_histogram(nodes), nodes=nodes)
    groups = plan.histogram_adjustment_plan.groups
    assert len(groups) == 1
    conversions = [group.primary_op for group in groups]
    assert all(conversion.from_su == 24 for conversion in conversions)
    assert [conversion.to_su for conversion in conversions] == [12]
    assert [conversion.via_su for conversion in conversions] == [11]
    assert [conversion.node_ids[0] for conversion in conversions] == [3]
    assert plan.histogram_adjustment_plan.inventory_after[24] == 3
    assert plan.phase_result.primary_adjustment_count == 1
    assert plan.phase_result.primary_adjustment_count <= 10


def test_optional_residual_conversion_does_not_push_h_outside_soft_window() -> None:
    pool = {11: [1, 2], 12: [], 13: [], 22: [], 23: [3, 4, 5], 24: [6]}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)
    E_target = torch.tensor([6, 7, 0, 0, 0, 0], dtype=torch.float32)

    result = _next_residual_product(
        pool,
        H,
        E_target=E_target,
        remaining_flex_chains=0,
        max_conversions=4,
        max_h_compensation_ops=4,
    )

    assert result["mode"] == "none"
    assert any(
        item.get("reason") == "optional_conversion_worsens_h_soft_balance"
        for item in result["attempts"]
    )


def test_residual_next_product_balances_h_when_filling_su11_gap() -> None:
    pool = {11: [], 12: [12], 13: [], 22: [22], 23: [23, 24], 24: [240]}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    result = _next_residual_product(
        pool,
        H,
        E_target=torch.tensor([0.0, 8.0]),
        remaining_flex_chains=0,
        max_conversions=1,
        max_h_compensation_ops=0,
    )

    assert result["mode"] == "conversion"
    conversions = [
        group.primary_op
        for group in result["plan"].histogram_adjustment_plan.groups
    ]
    assert [(item.from_su, item.to_su) for item in conversions] == [(12, 11)]
    assert result["plan"].diagnostics["su11_rotation_index_after"] == 1


def test_residual_template_fit_returns_direct_zero_conversion_candidate() -> None:
    pool = {11: [1, 2], 12: [], 13: [], 22: [9], 23: [3, 4, 5], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    result = _next_residual_product(
        pool,
        H,
        remaining_flex_chains=1,
        max_conversions=0,
        max_h_compensation_ops=0,
    )

    assert result["mode"] == "direct"
    assert result["product_role"] == "residual_flex_chain"
    assert result["required_counts"] == {11: 2, 23: 3}


def test_residual_short_flex_is_used_before_any_conversion() -> None:
    pool = {11: [1, 2], 12: [12], 13: [], 22: [], 23: [3], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    result = _next_residual_product(
        pool,
        H,
        remaining_flex_chains=1,
        remaining_flex_capacity=1,
        max_conversions=4,
        max_h_compensation_ops=4,
    )

    assert result["mode"] == "direct"
    assert result["required_counts"] == {11: 2, 23: 1}
    assert result["chain_23_count"] == 1
    assert result["selection_reason"] == "all_direct_templates_checked_before_conversion"


def test_residual_two_node_side_chain_can_convert_one_of_four_su22() -> None:
    pool = {11: [], 12: [], 13: [], 22: [1, 2, 3, 4], 23: [], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    H[22] = 4

    result = _next_residual_product(
        pool,
        H,
        remaining_flex_chains=0,
        remaining_flex_capacity=0,
        max_conversions=10,
        max_h_compensation_ops=0,
    )

    assert result["mode"] == "conversion"
    assert result["product_role"] == "residual_side_chain"
    assert result["chain_23_count"] == 0
    groups = result["plan"].histogram_adjustment_plan.groups
    assert [(group.primary_op.from_su, group.primary_op.to_su) for group in groups] == [
        (22, 11)
    ]


def test_residual_side_chain_can_fill_two_slots_from_four_su23() -> None:
    pool = {11: [], 12: [], 13: [], 22: [], 23: [1, 2, 3, 4], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    H[23] = 4

    result = _next_residual_product(
        pool,
        H,
        remaining_flex_chains=0,
        remaining_flex_capacity=0,
        max_conversions=10,
        max_h_compensation_ops=0,
    )

    assert result["mode"] == "conversion"
    assert result["product_role"] == "residual_side_chain"
    assert result["chain_23_count"] == 2
    groups = result["plan"].histogram_adjustment_plan.groups
    assert [(group.primary_op.from_su, group.primary_op.to_su) for group in groups] == [
        (23, 11),
        (23, 22),
    ]


def test_residual_orphan_terminalization_never_creates_closure_compensation() -> None:
    nodes = [make_node(node_id, 22) for node_id in range(1, 6)]
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase_eval = _evaluate_external_phases(phase_eval=phase_eval, nodes=nodes)
    pool = _resource_ids(phase_eval)
    H = _histogram(nodes)

    plan = _residual_orphan_terminalization_plan(
        phase_eval=phase_eval,
        pool=pool,
        H=H,
        E_target=torch.tensor([0.0, 20.0]),
        h_tolerance=0.06,
        special_degree_meta=None,
        su11_rotation_index=0,
        max_conversions=2,
    )

    assert plan is not None
    groups = plan.histogram_adjustment_plan.groups
    assert len(groups) == 2
    assert all(group.h_compensation_ops == [] for group in groups)
    assert all(group.primary_op.to_su in {12, 13} for group in groups)


def test_residual_uses_existing_resources_before_terminalizing_single_orphan() -> None:
    pool = {
        11: [1, 2, 3],
        22: [20, 21],
        23: [30, 31, 32],
        24: [],
    }

    contracts = _residual_contracts(pool, max_flex_chains=1)

    assert [item.expected_topology for item in contracts] == [
        "11-23-23-23-11",
        "11-22",
    ]
    consumed = Counter()
    for contract in contracts:
        consumed.update(contract.required_counts)
    assert dict(consumed) == {11: 3, 23: 3, 22: 1}
    assert len(pool[22]) - int(consumed[22]) == 1


def test_optional_side_fit_does_not_worsen_existing_high_h_state() -> None:
    H = torch.zeros(40, dtype=torch.long)
    for su_type, count in {
        0: 1, 1: 10, 2: 11, 3: 12, 5: 64, 6: 2, 9: 23, 10: 30,
        11: 115, 14: 1, 15: 8, 16: 3, 17: 2, 18: 2, 19: 105,
        20: 2, 22: 18, 23: 87, 24: 3, 25: 1, 26: 3, 27: 2,
        28: 50, 29: 61,
    }.items():
        H[int(su_type)] = int(count)
    pool = {
        11: list(range(240, 268)),
        12: [],
        13: [],
        22: [],
        23: list(range(460, 475)),
        24: [],
    }

    result = _next_residual_product(
        pool,
        H,
        E_target=torch.tensor([500, 520, 166, 6, 0, 0], dtype=torch.float32),
        remaining_flex_chains=0,
        max_conversions=12,
        max_h_compensation_ops=8,
    )

    assert result["mode"] == "none"
    assert any(
        item.get("reason") == "optional_conversion_worsens_h_soft_balance"
        for item in result["attempts"]
    )


def test_stage3_transaction_records_effective_h_not_histogram_node_count() -> None:
    nodes = [make_node(1, 22), make_node(7, 24), make_node(3, 24), make_node(9, 24)]
    H = _histogram(nodes)
    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)

    assert plan.phase_result.transactions
    first = plan.phase_result.transactions[0]
    assert first.h_before == _h_total(H, None)
    assert first.h_before != float(H.sum().item())
    assert first.h_after == _h_total(plan.histogram_adjustment_plan.H_after, None)


def test_su24_profile_does_not_claim_or_consume_resource() -> None:
    nodes = [make_node(4, 24), make_node(2, 24)]
    pool = {24: [2, 4]}
    profiles = build_su24_resource_role_profiles(nodes, pool)
    assert profiles == []
    assert pool == {24: [2, 4]}


def test_raw_su24_profiles_are_exact_and_role_specific() -> None:
    a24, a11, a23, a0 = make_node(100, 24), make_node(101, 11), make_node(102, 23), make_node(103, 0)
    b24, b11, b22, b23 = make_node(200, 24), make_node(201, 11), make_node(202, 22), make_node(203, 23)
    c24, c23, c0, c2 = make_node(300, 24), make_node(301, 23), make_node(302, 0), make_node(303, 2)
    d24, d1, d23, d3 = make_node(400, 24), make_node(401, 1), make_node(402, 23), make_node(403, 3)
    bad24, bad11a, bad11b, bad23 = make_node(500, 24), make_node(501, 11), make_node(502, 11), make_node(503, 23)
    for center, neighbors in (
        (a24, (a11, a23, a0)),
        (b24, (b11, b22, b23)),
        (c24, (c23, c0, c2)),
        (d24, (d1, d23, d3)),
        (bad24, (bad11a, bad11b, bad23)),
    ):
        for neighbor in neighbors:
            connect(center, neighbor)

    nodes = [
        a24, a11, a23, a0,
        b24, b11, b22, b23,
        c24, c23, c0, c2,
        d24, d1, d23, d3,
        bad24, bad11a, bad11b, bad23,
    ]
    profiles = build_su24_resource_role_profiles(nodes, {24: [100, 200, 300, 400, 500]})
    by_node = {int(item.node_id): item for item in profiles}

    assert {node_id: profile.role for node_id, profile in by_node.items()} == {
        100: "A",
        200: "B",
        300: "C",
        400: "D",
    }
    assert 500 not in by_node
    assert all(profile.roles == [profile.role] for profile in profiles)
    assert all(profile.available_roles == [profile.role] for profile in profiles)

    a_support = raw_su24_segment_support(by_node[100], skeleton_kind="side", slot_role="left_anchor")
    assert a_support["support_specific_ids_by_su"] == {11: [101]}
    assert a_support["support_required_counts"] == {23: 1, 22: 1}
    assert a_support["needs_h"] is True

    b_support = raw_su24_segment_support(by_node[200], skeleton_kind="side", slot_role="left_anchor")
    assert b_support["support_specific_ids_by_su"] == {11: [201], 22: [202]}
    assert b_support["support_required_counts"] == {}
    assert b_support["existing_branch_stubs"][0]["topology_notation"] == "-22"

    assert raw_su24_support_counts(by_node[300], "side", "middle") == {23: 1, 22: 1}
    d_support = raw_su24_segment_support(by_node[400], skeleton_kind="side", slot_role="middle")
    assert d_support["support_required_counts"] == {}
    assert d_support["existing_branch_stubs"][0]["topology_notation"] == "-1"


def test_block_c_stops_with_specific_phase_resource_infeasible() -> None:
    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason="current_contract_resource_infeasible",
                phase="external_fatty_ring_closure",
                unresolved_counts={"24": 1},
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
    )
    assert result.stop_reason == "phase_resource_infeasible"
    assert result.accepted_cycles == 0
    proof = result.cycles[-1].plan["phase_result"]["infeasibility_proof"]
    assert proof["phase"] == "external_fatty_ring_closure"
    assert proof["reason"] == "current_contract_resource_infeasible"
    assert proof["missing_counts"] == {"24": 1}
    assert "failed_contract" in proof
    assert "attempted_conversions" in proof
    assert "h_or_element_rejections" in proof
    assert "template_checks" in proof


def test_terminal_infeasibility_proof_accepts_legacy_string_failed_contract() -> None:
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="phase_resource_infeasible",
        blocked_reason="residual_terminal_conversion_infeasible",
        phase="residual_resource_self_closure",
        plan_kind="residual_terminal_conversion_infeasible",
        phase_result=Stage3PhaseResult(
            active_phase="residual_resource_self_closure",
            stop_reason="phase_resource_infeasible",
            infeasibility_proof={
                "phase": "residual_resource_self_closure",
                "reason": "residual_terminal_conversion_infeasible",
                "failed_contract": "orphan_22_23_via_11_then_aromatic_rotation",
            },
        ),
    )

    proof = _stage3_terminal_infeasibility_proof(plan)

    assert proof["reason"] == "residual_terminal_conversion_infeasible"
    assert proof["failed_contract"] == {
        "target_role": "orphan_22_23_via_11_then_aromatic_rotation"
    }


def test_block_c_exports_legacy_residual_failure_without_diagnostic_crash() -> None:
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="phase_resource_infeasible",
        blocked_reason="residual_terminal_conversion_infeasible",
        phase="residual_resource_self_closure",
        plan_kind="residual_terminal_conversion_infeasible",
        phase_result=Stage3PhaseResult(
            active_phase="residual_resource_self_closure",
            stop_reason="phase_resource_infeasible",
            infeasibility_proof={
                "phase": "residual_resource_self_closure",
                "reason": "residual_terminal_conversion_infeasible",
                "failed_contract": "orphan_22_23_via_11_then_aromatic_rotation",
                "available_counts": {"22": 1},
            },
        ),
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **_kwargs):
            return plan

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"], kwargs["H"], kwargs["plan"]
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
    )

    assert result.stop_reason == "phase_resource_infeasible"
    proof = result.cycles[-1].plan["phase_result"]["infeasibility_proof"]
    assert proof["failed_contract"]["target_role"] == (
        "orphan_22_23_via_11_then_aromatic_rotation"
    )


def test_zero_conversion_fallback_cannot_commit_without_iteration_ledger() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 11),
        make_node(3, 23),
        make_node(4, 23),
        make_node(5, 23),
    ]
    H = _histogram(nodes)
    pool = {11: [1, 2], 22: [], 23: [3, 4, 5], 24: []}
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    contract = _residual_contracts(pool, max_flex_chains=1)[0]

    plan = _transaction_plan(
        phase_eval=phase_eval,
        contract=contract,
        pool=pool,
        H=H,
        E_target=None,
        max_primary_adjustments=10,
        max_h_compensation_ops=8,
        h_tolerance=0.06,
        special_degree_meta=None,
        iteration_ledger=None,
    )

    transaction = plan.phase_result.transactions[-1]
    assert plan.reason == "phase_resource_infeasible"
    assert plan.phase_result.symbolic_topology_count == 0
    assert transaction.committed is False
    assert transaction.claimed_resource_ids == {}
    assert transaction.product_node_ids == []
    assert transaction.infeasibility_proof["reason"] == (
        "zero_conversion_transaction_must_use_iteration_ledger"
    )
    assert transaction.infeasibility_proof["attempted_claimed_resource_ids"] == {
        11: [1, 2],
        23: [3, 4, 5],
    }


def test_external_contracts_cannot_be_completed_by_ledger_reservation() -> None:
    nodes = [
        make_node(100, 3),
        make_node(101, 3),
        make_node(1, 11),
        make_node(2, 11),
    ]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1, 2]},
        claimed_source_ids=[100, 101],
    )

    def contract(contract_id: int, source_id: int) -> TopologyClosureContract:
        return TopologyClosureContract(
            contract_id=int(contract_id),
            contract_kind="mainline_closure",
            phase="external_flex_closure",
            target_role="flex_chain",
            source_fragment_ids=[int(source_id)],
            source_node_ids=[int(source_id)],
            claimed_source_node_ids=[int(source_id)],
            mainline_slots=[TopologyClosureSlot(
                slot_id=f"external:{contract_id}:right:F1",
                kind="F1",
                anchor_node_id=int(source_id),
                required_su_sequence=[11],
            )],
            required_counts={11: 1},
            expected_topology="OPEN-3-11",
        )

    unchanged, first, first_proof = _ledger_transaction(
        ledger=ledger,
        contract=contract(100, 100),
        phase="external_flex_closure",
        H=_histogram(nodes),
        special_degree_meta=None,
    )
    unchanged, second, second_proof = _ledger_transaction(
        ledger=unchanged,
        contract=contract(101, 101),
        phase="external_flex_closure",
        H=_histogram(nodes),
        special_degree_meta=None,
    )

    assert first is None and second is None
    assert first_proof["reason"] == "external_contract_requires_physical_assembler"
    assert second_proof["reason"] == "external_contract_requires_physical_assembler"
    assert unchanged.available_ids_by_su[11] == [1, 2]
    assert unchanged.completed_products == []


def test_external_contract_without_slots_still_requires_physical_assembler() -> None:
    nodes = [make_node(100, 3), make_node(1, 11)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1]},
        claimed_source_ids=[100],
    )
    contract = TopologyClosureContract(
        contract_id=100,
        contract_kind="mainline_closure",
        phase="external_flex_closure",
        target_role="flex_chain",
        source_node_ids=[100],
        claimed_source_node_ids=[100],
        required_counts={11: 1},
    )

    unchanged, transaction, proof = _ledger_transaction(
        ledger=ledger,
        contract=contract,
        phase="external_flex_closure",
        H=_histogram(nodes),
        special_degree_meta=None,
    )

    assert transaction is None
    assert proof["reason"] == "external_contract_requires_physical_assembler"
    assert unchanged.available_ids_by_su[11] == [1]


def test_transaction_plan_never_reintroduces_ids_absent_from_ledger_pool() -> None:
    nodes = [make_node(1, 11), make_node(2, 11)]
    H = _histogram(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    ledger = build_iteration_ledger(nodes, resource_ids_by_su={11: [1]})
    contract = TopologyClosureContract(
        contract_id=77,
        contract_kind="residual_conversion",
        phase="residual_resource_self_closure",
        target_role="residual_resource_conversion",
        mainline_slots=[TopologyClosureSlot(
            slot_id="residual:77:mainline",
            kind="RESOURCE_PATH",
            required_su_sequence=[11, 11],
        )],
        required_counts={11: 2},
        expected_topology="11-11",
    )

    plan = _transaction_plan(
        phase_eval=phase_eval,
        contract=contract,
        pool={11: [1, 2]},
        H=H,
        E_target=None,
        max_primary_adjustments=0,
        max_h_compensation_ops=0,
        h_tolerance=0.06,
        special_degree_meta=None,
        iteration_ledger=ledger,
    )

    proof = plan.phase_result.infeasibility_proof
    transaction = plan.phase_result.transactions[-1]
    assert plan.reason == "phase_resource_infeasible"
    assert proof["available_counts"][11] == 1
    assert proof["slot_assignment_proof"]["reason"] == "slot_resource_deficit"
    assert transaction.committed is False
    assert transaction.claimed_resource_ids == {}
    assert 2 not in plan.runtime["iteration_topology_ledger"].available_ids_by_su[11]


def test_block_c_iteration_offset_keeps_post_rebuild_audit_ids_unique() -> None:
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=99,
        target_role="residual_resource_conversion",
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(transactions=[transaction]),
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        outer_cycle=4,
        iteration_offset=10,
    )

    exported = result.cycles[0].plan["phase_result"]["transactions"][0]
    assert result.cycles[0].cycle == 11
    assert exported["outer_cycle"] == 4
    assert exported["iteration"] == 11


def test_block_c_generation_id_is_unique_across_outer_cycles() -> None:
    captured = []

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            captured.append(int(kwargs["generation_id"]))
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(stop_reason="stage3_complete"),
            )

    run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        outer_cycle=4,
        iteration_offset=10,
    )

    assert captured == [4_000_011]


def test_block_c_final_read_only_audit_promotes_complete_after_last_rebuild() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            if int(kwargs["H"][11].item()) == 1:
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
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                histogram_adjustment_plan=histogram_plan,
                phase_result=Stage3PhaseResult(rebuild_required=True),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert len(result.cycles) == 2
    assert result.cycles[-1].applied is False
    assert result.cycles[-1].acceptance_reason == "stage3_complete"


def test_block_c_accepts_valid_fresh_rebuild_without_cross_round_metric_gate() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    before_metrics = {
        "subphase_rank": 5,
        "pending_count": 0,
        "unresolved_total": 0,
        "shortage_total": 0,
        "flex_gap": 1,
        "product_count": 5,
    }
    after_metrics = {
        **before_metrics,
        "flex_gap": 2,
        "product_count": 4,
    }
    histogram_plan = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
        su11_rotation_index_before=4,
        su11_rotation_index_after=5,
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=7,
        target_role="residual_resource_conversion",
        converted_node_ids=[7],
        conversion_path=[{"from_su": 12, "to_su": 11, "node_ids": [7]}],
        committed=True,
    )
    before_plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        histogram_adjustment_plan=histogram_plan,
        phase_result=Stage3PhaseResult(transactions=[transaction], rebuild_required=True),
        runtime={"phase_evaluation": SimpleNamespace(topology_metrics=before_metrics)},
    )
    after_plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_complete",
        phase="complete",
        runtime={"phase_evaluation": SimpleNamespace(topology_metrics=after_metrics)},
    )

    class Adjuster:
        _stage3_su11_rotation_index = 4

        def plan_topology_resource_adjustments(self, **kwargs):
            return after_plan if int(kwargs["H"][11].item()) == 1 else before_plan

        def apply_topology_resource_adjustments(self, **kwargs):
            self._stage3_su11_rotation_index = 5
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    context = {"generation": "before"}

    def rebuild(**kwargs):
        context["generation"] = "candidate"
        return _nodes_from_histogram(kwargs["H_seed"]), kwargs["H_seed"], {"r2": 0.7}, 0.7

    def restore(snapshot):
        context.clear()
        context.update(snapshot)

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=restore,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert int(result.H[12].item()) == 0
    assert int(result.H[11].item()) == 1
    assert int(result.nodes[0].su_type) == 11
    assert context == {"generation": "candidate"}
    assert adjuster._stage3_su11_rotation_index == 5
    cycle = result.cycles[0]
    assert cycle.accepted is True
    assert cycle.accepted_state_retained is True
    assert cycle.acceptance_reason == "valid_histogram_fresh_rebuild"


def test_block_c_repeated_physical_histogram_exports_cycle_proof() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            H_before = kwargs["H"].detach().clone()
            node_id = int(kwargs["nodes"][0].global_id)
            source, target = (12, 11) if int(H_before[12].item()) else (11, 12)
            H_after = H_before.detach().clone()
            H_after[source] -= 1
            H_after[target] += 1
            histogram_plan = HistogramAdjustmentPlan(
                phase="external_flex_closure",
                inventory_before={source: 1, target: 0},
                demand_before={target: 1},
                H_before=H_before,
                H_after=H_after,
                primary_adjustment_count=1,
                physical_conversion_count=1,
                groups=[AtomicAdjustmentGroup(
                    primary_op=HistogramConversion(source, target, 1, "cycle", node_ids=[node_id]),
                    combined_delta={source: -1, target: 1},
                )],
            )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                plan_kind="external_flex_resource_rebalance",
                hard_required_counts={str(target): 1},
                planned_conversions=[
                    {"op": "convert", "from_su": source, "to_su": target, "node_id": node_id}
                ],
                histogram_adjustment_plan=histogram_plan,
                phase_result=Stage3PhaseResult(rebuild_required=True),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is False
    assert result.stop_reason == "phase_resource_infeasible"
    assert result.accepted_cycles == 1
    assert len(result.cycles) == 2
    terminal = result.cycles[-1]
    proof = terminal.plan["phase_result"]["infeasibility_proof"]
    assert terminal.applied is False
    assert terminal.blocked_reason == "physical_histogram_cycle_detected"
    assert terminal.plan["reason"] == "phase_resource_infeasible"
    assert proof["reason"] == "physical_histogram_cycle_detected"
    assert proof["phase"] == "external_flex_closure"
    assert proof["required_counts"] == {"12": 1}
    assert proof["attempted_conversions"][0]["node_id"] == 0


def test_block_c_does_not_rebuild_an_attached_empty_histogram_plan() -> None:
    node = make_node(7, 22)
    H = _histogram([node])
    empty_histogram = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        H_before=H.detach().clone(),
        H_after=H.detach().clone(),
        blocked_reason="hydrogen_soft_direction_incompatible_with_resource_demand",
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason="hydrogen_soft_direction_incompatible_with_resource_demand",
                phase="residual_resource_self_closure",
                planned_conversions=[{"from_su": 22, "to_su": 11, "reason": "blocked"}],
                histogram_adjustment_plan=empty_histogram,
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    rebuild_calls = []
    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=torch.tensor([0.0, 100.0]),
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs),
        validate_fn=lambda **kwargs: (True, "ok"),
    )

    assert rebuild_calls == []
    assert result.accepted_cycles == 0
    assert result.stop_reason == "phase_resource_infeasible"


def test_block_c_rejects_fresh_rebuild_histogram_drift() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                histogram_adjustment_plan=histogram_plan,
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    validation_calls = []
    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (_nodes_from_histogram(kwargs["H_seed"]), H.detach().clone(), {}, 0.0),
        validate_fn=lambda **kwargs: validation_calls.append(kwargs) or (True, "ok"),
    )

    assert validation_calls == []
    assert result.accepted_cycles == 0
    assert result.stop_reason == "all_histogram_adjustment_variants_invalid"
    assert result.cycles[0].blocked_reason == "fresh_rebuild_histogram_mismatch"


def test_block_c_exports_invalid_fresh_rebuild_transactions_as_rolled_back() -> None:
    node = make_node(7, 13)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[13] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        inventory_before={13: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(13, 11, 1, "test", node_ids=[7]),
            combined_delta={13: -1, 11: 1},
        )],
        su11_rotation_index_before=4,
        su11_rotation_index_after=5,
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=3_000_007,
        target_role="residual_resource_conversion",
        converted_node_ids=[7],
        conversion_path=[{"from_su": 13, "to_su": 11, "node_ids": [7]}],
        claimed_resource_ids={11: [7]},
        product_node_ids=[7],
        commit_scope="physical_histogram",
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        histogram_adjustment_plan=histogram_plan,
        diagnostics={"su11_rotation_index_after": 5},
        phase_result=Stage3PhaseResult(transactions=[transaction]),
    )

    class Adjuster:
        _stage3_su11_rotation_index = 4

        def plan_topology_resource_adjustments(self, **kwargs):
            return plan

        def apply_topology_resource_adjustments(self, **kwargs):
            self._stage3_su11_rotation_index = 5
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (_nodes_from_histogram(kwargs["H_seed"]), H.detach().clone(), {}, 0.0),
        validate_fn=lambda **kwargs: (True, "ok"),
    )

    exported = result.cycles[0].plan["phase_result"]["transactions"][0]
    assert exported["committed"] is False
    assert exported["rollback_reason"] == "fresh_rebuild_histogram_mismatch"
    assert exported["converted_node_ids"] == []
    assert exported["claimed_resource_ids"] == {}
    attempted = exported["infeasibility_proof"]["attempted_transaction"]
    assert attempted["converted_node_ids"] == [7]
    assert attempted["claimed_resource_ids"] == {"11": [7]}
    assert adjuster._stage3_su11_rotation_index == 4


def test_block_c_retries_next_primary_transition_after_validation_rejection() -> None:
    nodes = [make_node(1, 13), make_node(2, 12)]
    H = _histogram(nodes)

    class Adjuster:
        def __init__(self):
            self.calls = []
            self._stage3_su11_rotation_index = 0

        def plan_topology_resource_adjustments(self, **kwargs):
            forbidden = set(kwargs.get("forbidden_resource_conversion_signatures", []) or [])
            self.calls.append(sorted(forbidden))
            if int(kwargs["H"][11].item()) == 1:
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
            source, node_id = ((12, 2) if "13>11" in forbidden else (13, 1))
            H_after = kwargs["H"].detach().clone()
            H_after[source] -= 1
            H_after[11] += 1
            histogram = HistogramAdjustmentPlan(
                phase="external_topology_assembly",
                groups=[AtomicAdjustmentGroup(
                    primary_op=HistogramConversion(
                        source,
                        11,
                        1,
                        "candidate_fallback",
                        node_ids=[node_id],
                    ),
                    combined_delta={source: -1, 11: 1},
                )],
                primary_adjustment_count=1,
                physical_conversion_count=1,
                H_before=kwargs["H"].detach().clone(),
                H_after=H_after,
                inventory_before={12: 1, 13: 1},
                inventory_after={12: int(source != 12), 13: int(source != 13), 11: 1},
            )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_topology_assembly",
                histogram_adjustment_plan=histogram,
                diagnostics={"candidate_alternatives_enabled": True},
                phase_result=Stage3PhaseResult(
                    active_phase="external_topology_assembly",
                    rebuild_required=True,
                    primary_adjustment_count=1,
                    stop_reason="stage3_rebuild_required",
                ),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"],
                kwargs["H"],
                kwargs["plan"],
            )

    adjuster = Adjuster()
    validation_count = 0

    def validate(**_kwargs):
        nonlocal validation_count
        validation_count += 1
        return (validation_count > 1, "first_transition_rejected")

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        nodes,
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=validate,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert adjuster.calls[:2] == [[], ["13>11"]]
    assert int(result.H[13].item()) == 1
    assert int(result.H[12].item()) == 0
    assert int(result.H[11].item()) == 1
    assert result.cycles[0].diagnostics[
        "rejected_resource_conversion_signature"
    ] == "13>11"


def test_failed_plan_rolls_back_direct_candidate_but_keeps_assembly_snapshot() -> None:
    snapshot = TopologyResourceTransaction(
        contract_id=100,
        claimed_resource_ids={23: [1]},
        product_node_ids=[1],
        slot_assignments=[{"slot_id": "existing"}],
        transaction_origin="assembly_snapshot",
        committed=True,
    )
    candidate = TopologyResourceTransaction(
        contract_id=2_000_000,
        claimed_resource_ids={11: [2, 3], 23: [4, 5]},
        product_node_ids=[2, 3, 4, 5],
        slot_assignments=[{"slot_id": "residual:2000000:mainline"}],
        transaction_origin="current_plan_candidate",
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        phase_result=Stage3PhaseResult(transactions=[snapshot, candidate]),
    )

    exported, rolled_back = _rollback_stage3_plan_transactions(
        plan,
        "phase_resource_infeasible",
    )

    transactions = exported["phase_result"]["transactions"]
    assert rolled_back == [2_000_000]
    assert transactions[0]["committed"] is True
    assert transactions[0]["claimed_resource_ids"] == {"23": [1]}
    assert transactions[1]["committed"] is False
    assert transactions[1]["claimed_resource_ids"] == {}
    assert transactions[1]["slot_assignments"] == []
    attempted = transactions[1]["infeasibility_proof"]["attempted_transaction"]
    assert attempted["claimed_resource_ids"] == {"11": [2, 3], "23": [4, 5]}


def test_block_c_replans_fresh_after_fresh_rebuild() -> None:
    node = make_node(7, 13)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[13] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        inventory_before={13: 1},
        inventory_after={11: 1},
        H_before=H,
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                13, 11, 1, "residual_test", node_ids=[7]
            ),
            combined_delta={13: -1, 11: 1},
        )],
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=2_000_000,
        target_role="residual_resource_conversion",
        converted_node_ids=[7],
        conversion_path=[{"from_su": 13, "to_su": 11, "node_ids": [7]}],
        claimed_resource_ids={11: [7]},
        product_node_ids=[7],
        committed=True,
    )
    conversion_plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        histogram_adjustment_plan=histogram_plan,
        phase_result=Stage3PhaseResult(
            rebuild_required=True,
            transactions=[transaction],
        ),
        runtime={"iteration_topology_ledger": SimpleNamespace(completed_products=[])},
    )

    class Adjuster:
        def __init__(self):
            self.calls = []

        def plan_topology_resource_adjustments(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return conversion_plan
            assert "prior_committed_products" not in kwargs
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                runtime={
                    "iteration_topology_ledger": SimpleNamespace(
                        completed_products=[]
                    ),
                },
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"], kwargs["H"], kwargs["plan"]
            )

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=lambda **kwargs: (True, "ok"),
    )

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert len(adjuster.calls) == 2
    assert all("prior_committed_products" not in call for call in adjuster.calls)
    assert "committed_residual_products" not in result.diagnostics


def test_block_c_outer_cycles_share_nodes_but_not_topology_products() -> None:
    node = make_node(7, 13)
    H = _histogram([node])

    class Adjuster:
        def __init__(self):
            self.calls = []

        def plan_topology_resource_adjustments(self, **kwargs):
            self.calls.append(kwargs)
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                runtime={
                    "iteration_topology_ledger": SimpleNamespace(
                        completed_products=[{"product_id": f"fresh:{len(self.calls)}"}]
                    )
                },
            )

    adjuster = Adjuster()
    first = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        outer_cycle=0,
    )
    second = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        first.H,
        first.nodes,
        E_target=None,
        outer_cycle=1,
    )

    assert first.ok is True and second.ok is True
    assert adjuster.calls[0]["generation_id"] == 1
    assert adjuster.calls[1]["generation_id"] == 1_000_001
    assert all("prior_committed_products" not in call for call in adjuster.calls)


def test_block_c_exports_exact_aggregate_preflight_rejection_and_rolls_back() -> None:
    node = make_node(7, 23)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[23] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={23: 1},
        inventory_after={11: 1},
        aggregate_inventory_before={23: 1},
        aggregate_inventory_after={23: 1},
        H_before=H,
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                23, 11, 1, "test_preflight", node_ids=[7]
            ),
            combined_delta={23: -1, 11: 1},
        )],
    )
    transaction = TopologyResourceTransaction(
        phase="external_side_chain_closure",
        contract_id=77,
        converted_node_ids=[7],
        conversion_path=[{"from_su": 23, "to_su": 11, "node_ids": [7]}],
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="external_side_chain_closure",
        histogram_adjustment_plan=histogram_plan,
        planned_conversions=[
            {"op": "convert", "from_su": 23, "to_su": 11, "node_id": 7}
        ],
        phase_result=Stage3PhaseResult(
            rebuild_required=True,
            transactions=[transaction],
        ),
    )

    class Adjuster:
        _stage3_su11_rotation_index = 0

        def plan_topology_resource_adjustments(self, **kwargs):
            return plan

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"], kwargs["H"], kwargs["plan"]
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
    )

    cycle = result.cycles[0]
    assert result.stop_reason == (
        "histogram_apply_rejected:aggregate_resource_preflight_failed"
    )
    assert cycle.primary_adjustment_count == 1
    assert cycle.diagnostics["apply_rejection_reason"] == (
        "aggregate_resource_preflight_failed"
    )
    assert cycle.diagnostics["rejected_ops"]
    exported = cycle.plan["phase_result"]["transactions"][0]
    assert exported["committed"] is False
    assert exported["converted_node_ids"] == []
    assert exported["infeasibility_proof"]["attempted_transaction"][
        "converted_node_ids"
    ] == [7]


def test_user_supply_demand_example_converts_lowest_su24_to_su23_first() -> None:
    nodes = [
        make_node(10, 11),
        make_node(20, 23), make_node(21, 23), make_node(22, 23),
        make_node(30, 22), make_node(31, 22), make_node(32, 22),
        make_node(40, 24),
    ]
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 4, "23": 7, "22": 3, "24": 0},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_ids_by_su": {
                        "11": [10], "23": [20, 21, 22], "22": [30, 31, 32], "24": [40],
                    },
                    "remaining_counts": {"11": 1, "23": 3, "22": 3, "24": 1},
                },
            },
        },
    )
    from model.inverse.topology.histogram_adjustment import build_histogram_adjustment_plan

    histogram_plan = build_histogram_adjustment_plan(
        plan=plan,
        H=_histogram(nodes),
        max_primary_adjustments=1,
    )
    assert histogram_plan.primary_adjustment_count == 1
    primary = histogram_plan.groups[0].primary_op
    assert (primary.from_su, primary.to_su) == (24, 23)
    assert histogram_plan.groups[0].demand_after == {11: 3, 22: 0, 23: 3, 24: 0}
    assert histogram_plan.physical_conversion_count >= histogram_plan.primary_adjustment_count


def test_below_soft_h_window_does_not_synthesize_aromatic_only_topology() -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 5)]
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase_eval.flex_audit = {"enforced": False}
    H = torch.zeros(140, dtype=torch.long)
    H[12], H[13] = 4, 95

    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=H,
        nodes=nodes,
        E_target=torch.tensor([0.0, 100.0]),
        enable_post_external_spectrum=False,
        post_external_resume_phase="resource_su24_ring_closure",
        h_tolerance=0.06,
    )

    assert plan.reason == "stage3_complete"
    assert plan.phase == "complete"
    assert plan.planned_conversions == []
    assert plan.histogram_adjustment_plan is None
    assert int(H[11].item()) == 0
    assert int(H[23].item()) == 0


def test_su11_external_anchor_errors_are_hard_consistency_failures() -> None:
    errors = [
        "节点273(SU11): 缺少必需外接锚点",
        "节点273(SU11): 固定锚点边数不足",
        "节点273(SU11): 固定锚点边数超过目标上限",
    ]
    assigner = Layer1Assigner()

    accepted, diagnostics = assigner.assess_consistency_acceptance(errors)

    assert accepted is False
    assert diagnostics["n_soft"] == 0
    assert diagnostics["hard_errors"] == errors


def test_stage3_plan_logging_includes_resource_h_and_flex_scores(capsys) -> None:
    H = torch.zeros(40, dtype=torch.long)
    H[13], H[23] = 1, 1
    H_after = H.detach().clone()
    H_after[13] -= 1
    H_after[11] += 1
    group = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(
            13,
            11,
            1,
            "external_contract_resource_gap",
            node_ids=[7],
        ),
        combined_delta={13: -1, 11: 1},
        demand_before={11: 1},
        demand_after={},
        h_before=101.0,
        h_after_primary=100.0,
        h_after_final=100.0,
        scoring_diagnostics={
            "h": {"soft_penalty": 1.25, "drift_penalty": 0.5},
            "flex": {
                "before": {"X": 8, "Y": 2, "Z": 6, "flex_lower": 7, "flex_upper": 6, "minimum_flex_gap": 1},
                "after": {"X": 8, "Y": 2, "Z": 6, "flex_lower": 7, "flex_upper": 6, "minimum_flex_gap": 1},
                "penalty": 2.0,
            },
            "resource_gap_before": {11: 1},
            "resource_gap_after": {},
            "total_score": -95.0,
        },
    )
    histogram = HistogramAdjustmentPlan(
        phase="external_topology_assembly",
        groups=[group],
        H_before=H,
        H_after=H_after,
    )
    plan = Stage3AdjustmentPlan(
        reason="stage3_rebuild_required",
        phase="external_topology_assembly",
        plan_kind="external_contract_bundle",
        active_subphase="external_flex_closure",
        histogram_adjustment_plan=histogram,
    )

    _print_stage3_plan_summary(3, plan, H, torch.tensor([0.0, 100.0]))
    _print_stage3_histogram_adjustments(3, histogram)
    output = capsys.readouterr().out

    assert "[Stage3 Plan] cycle=3" in output
    assert "active_subphase=external_flex_closure" in output
    assert "SU13=1" in output and "SU11=1" in output
    assert "nodes=[7]" in output
    assert "H_soft_penalty=1.2500" in output
    assert "[Flex Impact]" in output


def test_spectrum_trial_with_orphan_su11_is_rejected_and_labeled(
    monkeypatch,
    capsys,
) -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:12:11:1",
        histogram_signature="candidate-histogram",
        inverse_signature="candidate-inverse",
        operations=[{
            "op": "convert",
            "node_id": 7,
            "from_su": 12,
            "to_su": 11,
            "reason": "remaining_pool_difference_spectrum_rebalance",
        }],
        target_su_types=[11, 12],
        primary_adjustment_count=1,
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            if kwargs["post_external_resume_phase"] == "remaining_pool_spectrum_rebalance":
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    plan_kind="remaining_pool_spectrum_candidate_set",
                    spectrum_candidates=[candidate],
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(stop_reason="stage3_complete"),
            )

    monkeypatch.setattr(
        "model.inverse.layer4.inverse_layer4_block_c.spectrum_window_loss",
        lambda diff, _types: 0.0 if dict(diff or {}).get("trial") else 10.0,
    )
    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=torch.tensor([0.0, 100.0]),
        difference_spectrum={
            "r2": 0.1,
            "source_stage": "layer2",
            "spectral_contract": "template_center_mu",
            "evaluation_hwhm": 1.0,
        },
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {
                "r2": 0.9,
                "trial": True,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
                "evaluation_hwhm": 1.0,
            },
            0.9,
        ),
        validate_fn=lambda **kwargs: (False, "节点7(SU11): 缺少必需外接锚点"),
        spectrum_min_r2_gain=0.01,
    )
    output = capsys.readouterr().out

    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 0
    assert int(result.H[12].item()) == 1
    assert int(result.H[11].item()) == 0
    assert "[Spectrum Trial Start]" in output
    assert "graph_valid=False" in output
    assert "accepted=False state_retained=False" in output
    assert "[Spectrum Trial Commit]" not in output


def test_blocker_only_terminal_proof_does_not_report_committed_claims_as_missing() -> None:
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="phase_resource_infeasible",
        phase="external_topology_assembly",
        hard_required_counts={"11": 29, "23": 4, "24": 1},
        phase_result=Stage3PhaseResult(
            active_phase="external_topology_assembly",
            phase_completed=False,
            stop_reason="phase_resource_infeasible",
            infeasibility_proof={
                "reason": "stage3_completion_invariants_failed",
                "required_counts": {},
                "missing_counts": {},
                "failed_contract": {
                    "contract_id": 1,
                    "contract_kind": "structural_blocker",
                    "reason": "flex_range_below_minimum",
                    "required_counts": {},
                },
            },
        ),
    )

    proof = _stage3_terminal_infeasibility_proof(plan)

    assert proof["required_counts"] == {}
    assert proof["missing_counts"] == {}
