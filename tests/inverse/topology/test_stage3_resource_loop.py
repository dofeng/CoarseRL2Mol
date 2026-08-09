from __future__ import annotations

from collections import Counter
import inspect
from types import SimpleNamespace

import pytest
import torch

from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
import model.inverse.layer4.inverse_layer4 as inverse_layer4_module
from model.inverse.layer4.inverse_layer4_block_c import (
    _committed_products_from_plan,
    _last_resource_step_signature,
    _print_stage3_histogram_adjustments,
    _print_stage3_plan_summary,
    _rollback_stage3_plan_transactions,
    _register_rejected_resource_candidate,
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
    Stage3ProgressPhase,
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
    _residual_su24_aromatization_plan,
    _resource_ids,
    _residual_contracts,
    _transaction_plan,
    build_su24_resource_role_profiles,
    plan_stage3_state_machine,
)
from model.inverse.topology.phase_eval import (
    EXTERNAL_SUBPHASE_ORDER,
    PHASE_SEQUENCE,
    build_stage3_phase_evaluation,
    classify_topology_units,
)
from model.inverse.topology.classification import (
    TopologyClassificationResult,
    TopologyFragment,
)
from model.inverse.topology.workflow import plan_stage3_resource_adjustments
from model.inverse.topology.resource_profiles import (
    raw_su24_segment_support,
    raw_su24_support_counts,
)
from model.inverse.topology.resource_accounting import (
    build_resource_accounting_snapshot,
)
from model.inverse.topology.round_context import normalize_stage3_product

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


def _parity_rebuild_plan(
    nodes,
    *,
    primary_node_id,
    primary_source=23,
    primary_target=11,
):
    H_before = _histogram(nodes)
    H_after = H_before.detach().clone()
    H_after[int(primary_source)] -= 1
    H_after[int(primary_target)] += 1
    group = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(
            int(primary_source),
            int(primary_target),
            1,
            "test_primary_resource_adjustment",
            node_ids=[int(primary_node_id)],
        ),
        combined_delta={int(primary_source): -1, int(primary_target): 1},
    )
    aggregate_before = {}
    for node in nodes:
        aggregate_before.setdefault(int(node.su_type), []).append(int(node.global_id))
    claimed = {int(primary_target): [int(primary_node_id)]}
    final_remaining = {
        int(su_type): [
            int(node_id)
            for node_id in node_ids
            if int(node_id) != int(primary_node_id)
        ]
        for su_type, node_ids in aggregate_before.items()
    }
    final_remaining = {
        int(key): list(values)
        for key, values in final_remaining.items()
        if values
    }
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_before,
        authorized_ids_before_by_su=aggregate_before,
        final_remaining_ids_by_su=final_remaining,
        committed_claimed_ids_by_su=claimed,
        groups=[group],
    )
    assert accounting["ok"] is True
    histogram = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        groups=[group],
        primary_adjustment_count=1,
        physical_conversion_count=1,
        H_before=H_before,
        H_after=H_after,
        inventory_before=accounting["authorized_inventory_before"],
        inventory_after=accounting["authorized_inventory_after"],
        aggregate_inventory_before=accounting["aggregate_inventory_before"],
        aggregate_inventory_after=accounting["aggregate_inventory_after"],
        aggregate_required_counts=accounting["aggregate_required_counts"],
        accounting_diagnostics=accounting,
    )
    transaction = TopologyResourceTransaction(
        phase="external_side_chain_closure",
        contract_id=700,
        target_role="test_parity_rebuild",
        converted_node_ids=[int(primary_node_id)],
        conversion_path=[group.primary_op.to_dict()],
        claimed_resource_ids=claimed,
        committed=True,
        transaction_origin="current_plan_candidate",
    )
    return Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="external_side_chain_closure",
        plan_kind="test_parity_rebuild",
        histogram_adjustment_plan=histogram,
        diagnostics={"candidate_alternatives_enabled": False},
        phase_result=Stage3PhaseResult(
            active_phase="external_side_chain_closure",
            rebuild_required=True,
            primary_adjustment_count=1,
            transactions=[transaction],
        ),
    )


def _next_residual_product(
    pool,
    H,
    *,
    E_target=None,
    remaining_flex_chains=0,
    remaining_flex_capacity=None,
    max_conversions=10,
    max_h_compensation_ops=8,
    allow_h_outside_nonworsening=False,
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
        h_tolerance=0.08,
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
        allow_h_outside_nonworsening=allow_h_outside_nonworsening,
    )


def test_stage3_defaults_have_no_global_iteration_or_spectrum_acceptance_budget() -> None:
    layer4_defaults = inspect.signature(
        Layer4Adjuster.run_block_c_resource_adjustment_loop
    ).parameters
    assert "max_stage3_iterations" not in layer4_defaults
    assert "spectrum_max_accepted_rounds" not in layer4_defaults
    assert layer4_defaults["spectrum_candidate_limit"].default == 2
    assert layer4_defaults["spectrum_max_primary_adjustments"].default == 10
    assert "resource_max_r2_drop" not in layer4_defaults

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
    assert "resource_max_r2_drop" not in captured
    assert captured["iteration_offset"] == 10


def test_block_c_resource_bundle_limit_is_hard_capped_at_ten() -> None:
    captured = {}

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            captured.update(kwargs)
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        max_primary_adjustments_per_stage=25,
    )

    assert result.ok is True
    assert captured["max_primary_adjustments_per_stage"] == 10


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


def test_layer4_stage_router_rejects_removed_block_c_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="block_c_resource contains unknown keys: resource_max_r2_drop",
    ):
        Layer4Adjuster().adjust_by_stage(
            H=torch.zeros(40, dtype=torch.long),
            ppm=None,
            diff=None,
            E_target=torch.zeros(6),
            stage="block_c_resource",
            nodes=[],
            resource_max_r2_drop=0.02,
        )

    with pytest.raises(
        ValueError,
        match="block_c_resource contains unknown keys: resource_max_r2_drop",
    ):
        Layer4Adjuster().run_block_c_resource_adjustment_loop(
            H=torch.zeros(40, dtype=torch.long),
            nodes=[],
            E_target=torch.zeros(6),
            resource_max_r2_drop=0.02,
        )


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
        "residual_su24_aromatization",
        "residual_resource_self_closure",
        "complete",
    )
    assert [
        EXTERNAL_SUBPHASE_ORDER[phase]
        for phase in PHASE_SEQUENCE
    ] == list(range(len(PHASE_SEQUENCE)))


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


def test_all_residual_su24_resources_are_aromatized_before_tolerance() -> None:
    nodes = [make_node(7, 24), make_node(3, 24), make_node(9, 24), make_node(11, 24)]
    plan = plan_stage3_resource_adjustments(H=_histogram(nodes), nodes=nodes)
    groups = plan.histogram_adjustment_plan.groups
    assert len(groups) == 4
    conversions = [group.primary_op for group in groups]
    assert all(conversion.from_su == 24 for conversion in conversions)
    assert [conversion.to_su for conversion in conversions] == [12, 13, 13, 12]
    assert [conversion.via_su for conversion in conversions] == [None] * 4
    assert [conversion.node_ids[0] for conversion in conversions] == [3, 7, 9, 11]
    assert plan.histogram_adjustment_plan.inventory_after.get(24, 0) == 0
    assert plan.phase_result.primary_adjustment_count == 4
    assert plan.diagnostics["terminal_h_exemption"] is True
    assert plan.phase_result.primary_adjustment_count <= 10


def test_terminal_su24_h_exemption_is_preserved_in_workflow_result(
    monkeypatch,
) -> None:
    nodes = [make_node(0, 24)]
    H = _histogram(nodes)
    monkeypatch.setattr(
        "model.inverse.layer4.inverse_layer4_block_c._append_stage3_parity_balance",
        lambda **_kwargs: (
            True,
            {"enforced": False, "reason": "isolated_terminal_h_test"},
        ),
    )

    result = Layer4Adjuster().run_block_c_resource_adjustment_loop(
        H=H,
        nodes=nodes,
        E_target=torch.tensor([0.0, 1.0]),
        enable_post_external_spectrum=False,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"].detach().clone(),
            {
                "r2": -1.0,
                "source_stage": "layer2",
                "spectral_contract": "template_center_mu",
            },
            -1.0,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert result.diagnostics["terminal_h_exemption"] is True
    assert result.diagnostics["terminal_h_exemption_only"] is True
    assert (
        result.diagnostics["terminal_h_outside_origin"]
        == "residual_su24_aromatization"
    )
    assert result.diagnostics["terminal_h_final_audit"] == {
        "target_h": 1.0,
        "current_h": 0.0,
        "hard_lower": 0.92,
        "hard_upper": 1.08,
        "outside_hard_bounds": True,
        "pre_terminal_outside_hard_bounds": False,
        "ordinary_conversions_must_not_worsen": True,
    }
    records = result.diagnostics["terminal_h_exemption_records"]
    assert len(records) == 1
    assert records[0]["converted_su24_node_ids"] == [0]


def test_terminal_h_exemption_rejects_residual_conversion_that_worsens_h() -> None:
    pool = {
        11: [],
        12: [],
        13: [],
        22: [1, 2],
        23: [3, 4],
        24: [],
    }
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)
    h_before = _h_total(H, None)

    result = _next_residual_product(
        pool,
        H,
        E_target=torch.tensor([0.0, 15.0]),
        remaining_flex_chains=0,
        max_conversions=4,
        max_h_compensation_ops=0,
        allow_h_outside_nonworsening=True,
    )

    assert h_before == 10.0
    assert result["mode"] == "none"
    assert any(
        attempt.get("reason")
        in {
            "phase_resource_infeasible",
            "optional_conversion_violates_h_hard_constraint",
        }
        for attempt in result["attempts"]
    )


def test_residual_conversion_does_not_use_su24_to_satisfy_h_constraint() -> None:
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


def test_residual_next_product_never_uses_aromatic_or_su24_donors() -> None:
    pool = {
        11: [],
        12: [12],
        13: [13],
        22: [22, 25],
        23: [23, 24],
        24: [240],
    }
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    result = _next_residual_product(
        pool,
        H,
        E_target=None,
        remaining_flex_chains=0,
        max_conversions=10,
        max_h_compensation_ops=8,
    )

    assert result["mode"] == "conversion"
    operations = [
        operation
        for group in result["plan"].histogram_adjustment_plan.groups
        for operation in [group.primary_op, *group.h_compensation_ops]
    ]
    assert (operations[0].from_su, operations[0].to_su) == (22, 11)
    assert all(
        int(operation.from_su) in {11, 22, 23}
        and int(operation.to_su) in {11, 22, 23}
        for operation in operations
    )


def test_residual_su11_gap_is_infeasible_with_only_aromatic_and_su24_donors() -> None:
    pool = {11: [], 12: [12], 13: [13], 22: [22], 23: [], 24: [240]}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    result = _next_residual_product(
        pool,
        H,
        E_target=None,
        remaining_flex_chains=0,
        max_conversions=10,
        max_h_compensation_ops=8,
    )

    assert result["mode"] == "none"


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


def test_two_su23_and_two_su22_convert_one_su22_for_side_chain() -> None:
    pool = {11: [], 12: [], 13: [], 22: [1, 2], 23: [3, 4], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    H[22] = 2
    H[23] = 2

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
    assert result["required_counts"] == {11: 1, 22: 1, 23: 2}
    groups = result["plan"].histogram_adjustment_plan.groups
    assert [
        (group.primary_op.from_su, group.primary_op.to_su)
        for group in groups
    ] == [(22, 11)]
    assert groups[0].primary_op.node_ids == [1]


def test_residual_loop_persists_multiple_converted_products_across_rebuilds() -> None:
    nodes = [make_node(node_id, 22) for node_id in range(8)]
    H = _histogram(nodes)

    def rebuild(**kwargs):
        rebuilt = [
            make_node(int(node.global_id), int(node.su_type))
            for node in kwargs["nodes_seed"]
        ]
        return rebuilt, kwargs["H_seed"].detach().clone(), {}, 0.0

    result = Layer4Adjuster().run_block_c_resource_adjustment_loop(
        H=H,
        nodes=nodes,
        E_target=None,
        enable_post_external_spectrum=False,
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 3
    products = result.diagnostics["committed_topology_products"]
    assert [product["product_id"] for product in products] == [
        "residual:2000000",
        "residual:2000001",
        "residual:2000002",
    ]
    claimed_sets = [
        {
            int(node_id)
            for values in product["claimed_resource_ids_by_su"].values()
            for node_id in values
        }
        for product in products
    ]
    assert claimed_sets == [{0, 1}, {2, 3}, {4, 5}]
    assert not any(
        left & right
        for index, left in enumerate(claimed_sets)
        for right in claimed_sets[index + 1:]
    )
    terminal_plan = result.cycles[-1].plan
    assert terminal_plan["diagnostics"]["tolerated_remaining_ids_by_su"] == {
        "22": [6, 7]
    }
    assert result.diagnostics["tolerated_remaining_ids_by_su"] == {
        "22": [6, 7]
    }
    assert result.diagnostics["tolerated_remaining_count"] == 2


def test_committed_product_capture_excludes_pending_residual_ledger_product() -> None:
    external = {
        "product_id": "external:1",
        "phase": "external_side_chain_closure",
        "closed": True,
        "provisional": False,
        "fixed_source_node_ids": [1],
        "claimed_resource_ids_by_su": {"22": [2]},
    }
    pending_residual = {
        "product_id": "residual:2000000",
        "phase": "residual_resource_self_closure",
        "closed": True,
        "provisional": False,
        "claimed_resource_ids_by_su": {"11": [3], "22": [4]},
    }
    committed_residual = {
        **pending_residual,
        "product_id": "residual:1999999",
        "claimed_resource_ids_by_su": {"11": [5], "22": [6]},
    }
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_spectrum_candidates_ready",
        diagnostics={
            "external_stage_completed": True,
            "committed_residual_products": [committed_residual],
        },
        runtime={
            "iteration_topology_ledger": SimpleNamespace(
                completed_products=[external, pending_residual]
            )
        },
    )

    products = _committed_products_from_plan(plan)

    assert [product["product_id"] for product in products] == [
        "external:1",
        "residual:1999999",
    ]


def test_committed_product_capture_includes_phase_evaluation_external_products() -> None:
    external = {
        "product_id": "external:assembly:1",
        "phase": "external_flex_closure",
        "closed": True,
        "provisional": False,
        "source_fragment_ids": [101],
        "fixed_source_node_ids": [1],
    }
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_spectrum_candidates_ready",
        diagnostics={"external_stage_completed": True},
        runtime={
            "phase_evaluation": SimpleNamespace(
                assembly_view={"products": [external]}
            ),
            "iteration_topology_ledger": SimpleNamespace(
                completed_products=[]
            ),
        },
    )

    products = _committed_products_from_plan(plan)

    assert [product["product_id"] for product in products] == [
        "external:assembly:1"
    ]


def test_committed_product_capture_uses_normalized_external_product_phase() -> None:
    normalized = normalize_stage3_product(
        {
            "product_id": "external:normalized:1",
            "product_type": "flex_chain",
            "closed": True,
            "source_fragment_ids": [101],
            "fixed_source_node_ids": [1],
        },
        generation_id=8,
    ).to_dict()
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_spectrum_candidates_ready",
        diagnostics={"external_stage_completed": True},
        runtime={
            "iteration_topology_ledger": SimpleNamespace(
                completed_products=[normalized]
            )
        },
    )

    products = _committed_products_from_plan(plan)

    assert normalized["phase"] == "external_flex_closure"
    assert [product["source_fragment_ids"] for product in products] == [[101]]


def test_external_fragment_coverage_survives_committed_product_rebuild() -> None:
    center = make_node(10, 3)
    left = make_node(11, 9)
    right = make_node(12, 9)
    connect(center, left)
    connect(center, right)
    nodes = [center, left, right]
    H = _histogram(nodes)

    external_plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
    )
    committed = _committed_products_from_plan(external_plan)
    rebuilt_plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        committed_topology_products=committed,
    )

    assert [product["phase"] for product in committed] == [
        "external_flex_closure"
    ]
    assert rebuilt_plan.reason == "stage3_complete"
    assert rebuilt_plan.diagnostics["fragment_coverage"] == {
        "classified_fragment_ids": [10],
        "completed_fragment_ids": [10],
        "pending_fragment_ids": [],
        "uncovered_fragment_ids": [],
        "unknown_completed_fragment_ids": [],
        "unknown_pending_fragment_ids": [],
    }


def test_post_external_rebuild_fragment_does_not_reopen_external_stage() -> None:
    nodes = [make_node(10, 3), make_node(20, 3)]
    classification = TopologyClassificationResult(
        fragments=[
            TopologyFragment(
                fragment_id=10,
                family="persisted",
                class_type="closed",
                topology_template="closed",
                source_node_ids=[10],
                claimed_node_ids=[10],
                evidence={"fixed_source_node_ids": [10]},
            ),
            TopologyFragment(
                fragment_id=20,
                family="post_external_transient",
                class_type="temporary",
                topology_template="temporary",
                source_node_ids=[20],
                claimed_node_ids=[20],
                evidence={"fixed_source_node_ids": [20]},
            ),
        ],
        resource_pool={
            "remaining_ids_by_su": {"11": [], "22": [], "23": [], "24": []},
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        },
    )
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}
    committed = [{
        "product_id": "external:10",
        "phase": "external_flex_closure",
        "closed": True,
        "provisional": False,
        "source_fragment_ids": [10],
        "fixed_source_node_ids": [10],
    }]

    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        committed_topology_products=committed,
    )

    assert plan.reason == "stage3_complete"
    assert plan.diagnostics["fragment_coverage"]["uncovered_fragment_ids"] == []
    assert plan.diagnostics["post_external_transient_fragment_ids"] == [20]


def test_residual_conversion_retries_and_claims_the_same_converted_node() -> None:
    nodes = [
        make_node(1, 22),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}
    first = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        max_h_compensation_ops_per_adjustment=0,
    )

    pending = first.diagnostics["pending_residual_product_candidate"]
    assert pending["converted_node_ids"] == [1]
    rebuilt_nodes = [
        make_node(1, 11),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    rebuilt_classification = classify_topology_units(rebuilt_nodes)
    rebuilt_eval = build_stage3_phase_evaluation(
        rebuilt_nodes,
        classification=rebuilt_classification,
    )
    rebuilt_eval.flex_audit = {"enforced": False}
    retry = plan_stage3_state_machine(
        phase_eval=rebuilt_eval,
        H=first.histogram_adjustment_plan.H_after,
        nodes=rebuilt_nodes,
        classification=rebuilt_classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        pending_residual_product_candidate=pending,
        max_h_compensation_ops_per_adjustment=0,
    )

    assert retry.reason == "stage3_complete"
    resolved = retry.diagnostics["resolved_pending_residual_product"]
    assert resolved["converted_nodes_claimed"] is True
    assert 1 in {
        int(value)
        for values in resolved["claimed_resource_ids"].values()
        for value in values
    }


def test_residual_retry_prioritizes_converted_node_over_balance_node() -> None:
    nodes = [
        make_node(0, 11),
        make_node(10, 11),
        make_node(20, 22),
        make_node(30, 23),
        make_node(31, 23),
    ]
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}
    pending = {
        "product_role": "residual_side_chain",
        "chain_23_count": 2,
        "required_counts": {11: 1, 22: 1, 23: 2},
        "conversion_count": 1,
        "converted_node_ids": [10],
    }

    retry = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        pending_residual_product_candidate=pending,
        max_h_compensation_ops_per_adjustment=0,
    )

    assert retry.reason == "stage3_complete"
    resolved = retry.diagnostics["resolved_pending_residual_product"]
    assert resolved["converted_nodes_claimed"] is True
    assert resolved["claimed_resource_ids"]["11"] == [10]
    assert retry.diagnostics["tolerated_remaining_ids_by_su"] == {"11": [0]}


def test_residual_retry_restores_converted_node_taken_by_rebuild_fragment() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    classification = TopologyClassificationResult(
        fragments=[TopologyFragment(
            fragment_id=1,
            family="rebuilt_transient",
            class_type="fixed_core",
            topology_template="temporary",
            source_node_ids=[1],
            claimed_node_ids=[1],
            evidence={"fixed_source_node_ids": [1]},
        )],
        resource_pool={
            "remaining_ids_by_su": {
                "11": [],
                "22": [2],
                "23": [3, 4],
                "24": [],
            },
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        },
    )
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}
    pending = {
        "product_role": "residual_side_chain",
        "chain_23_count": 2,
        "required_counts": {11: 1, 22: 1, 23: 2},
        "conversion_count": 1,
        "converted_node_ids": [1],
    }

    retry = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        pending_residual_product_candidate=pending,
        max_h_compensation_ops_per_adjustment=0,
    )

    assert retry.reason == "stage3_complete"
    resolved = retry.diagnostics["resolved_pending_residual_product"]
    assert resolved["converted_nodes_claimed"] is True
    assert resolved["claimed_resource_ids"]["11"] == [1]


def test_residual_retry_rejects_a_conversion_node_not_claimed_by_product() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}
    pending = {
        "product_role": "residual_side_chain",
        "chain_23_count": 2,
        "required_counts": {11: 1, 22: 1, 23: 2},
        "conversion_count": 1,
        "converted_node_ids": [999],
    }

    retry = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.RESIDUAL_SELF_CLOSURE,
        pending_residual_product_candidate=pending,
    )

    assert retry.reason == "phase_resource_infeasible"
    proof = retry.phase_result.infeasibility_proof
    assert proof["reason"] == "pending_residual_product_not_materialized"
    assert proof["required_converted_node_ids"] == [999]
    assert proof["converted_nodes_claimed"] is False


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


def test_residual_su24_aromatization_never_creates_closure_compensation() -> None:
    nodes = [make_node(node_id, 24) for node_id in range(1, 6)]
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase_eval = _evaluate_external_phases(phase_eval=phase_eval, nodes=nodes)
    pool = _resource_ids(phase_eval)
    H = _histogram(nodes)

    plan = _residual_su24_aromatization_plan(
        phase_eval=phase_eval,
        pool=pool,
        H=H,
        special_degree_meta=None,
        su11_rotation_index=0,
        max_conversions=2,
    )

    assert plan is not None
    groups = plan.histogram_adjustment_plan.groups
    assert len(groups) == 2
    assert all(group.h_compensation_ops == [] for group in groups)
    assert all(group.primary_op.from_su == 24 for group in groups)
    assert all(group.primary_op.to_su in {12, 13} for group in groups)
    assert plan.diagnostics["remaining_su24_node_ids"] == [3, 4, 5]


def test_terminal_su24_retry_restores_nodes_taken_by_rebuild_fragments() -> None:
    nodes = [make_node(node_id, 24) for node_id in range(1, 4)]
    classification = TopologyClassificationResult(
        fragments=[TopologyFragment(
            fragment_id=1,
            family="rebuilt_transient",
            class_type="fixed_core",
            topology_template="temporary",
            source_node_ids=[1, 2, 3],
            claimed_node_ids=[1, 2, 3],
            evidence={"fixed_source_node_ids": [1, 2, 3]},
        )],
        resource_pool={
            "remaining_ids_by_su": {
                "11": [], "22": [], "23": [], "24": []
            },
            "su11_donor_ids_by_su": {
                "10": [], "12": [], "13": []
            },
        },
    )
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}

    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.SU24_AROMATIZATION,
        pending_terminal_su24_node_ids=[1, 2, 3],
        max_primary_adjustments_per_stage=10,
    )

    assert plan.reason == "stage3_rebuild_required", plan.to_dict()
    assert plan.phase == "residual_su24_aromatization"
    assert plan.histogram_adjustment_plan.H_after[24].item() == 0
    assert set(plan.diagnostics["converted_su24_node_ids"]) == {1, 2, 3}


def test_terminal_phase_restores_all_uncommitted_physical_su24() -> None:
    nodes = [make_node(node_id, 24) for node_id in range(1, 4)]
    classification = TopologyClassificationResult(
        fragments=[TopologyFragment(
            fragment_id=1,
            family="rebuilt_transient",
            class_type="fixed_core",
            topology_template="temporary",
            source_node_ids=[1, 2, 3],
            claimed_node_ids=[1, 2, 3],
            evidence={"fixed_source_node_ids": [1, 2, 3]},
        )],
        resource_pool={
            "remaining_ids_by_su": {
                "11": [], "22": [], "23": [], "24": []
            },
            "su11_donor_ids_by_su": {
                "10": [], "12": [], "13": []
            },
        },
    )
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    )
    phase_eval.flex_audit = {"enforced": False}

    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=_histogram(nodes),
        nodes=nodes,
        classification=classification,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.SU24_AROMATIZATION,
        max_primary_adjustments_per_stage=10,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.histogram_adjustment_plan.H_after[24].item() == 0
    assert set(plan.diagnostics["converted_su24_node_ids"]) == {1, 2, 3}


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


def test_optional_side_fit_is_allowed_inside_stage3_hard_window() -> None:
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

    assert result["mode"] == "conversion"
    histogram = result["plan"].histogram_adjustment_plan
    assert abs(_h_total(histogram.H_after, None) - 520.0) / 520.0 <= 0.08


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


def test_terminal_infeasibility_proof_rejects_noncanonical_failed_contract() -> None:
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

    with pytest.raises(TypeError, match="failed_contract must be a mapping"):
        _stage3_terminal_infeasibility_proof(plan)


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


def _run_one_parity_rebuild_plan(
    plan,
    nodes,
    *,
    parity_cursor=0,
    rebuild_fn=None,
    validate_fn=None,
    difference_spectrum=None,
):
    class Adjuster:
        def __init__(self):
            self.calls = 0
            self._stage3_su11_rotation_index = 0
            self._stage3_parity_balance_cursor = int(parity_cursor)

        def plan_topology_resource_adjustments(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return plan
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

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"], kwargs["H"], kwargs["plan"]
            )

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        _histogram(nodes),
        nodes,
        E_target=None,
        rebuild_fn=(
            rebuild_fn
            if rebuild_fn is not None
            else lambda **kwargs: (
                _nodes_from_histogram(kwargs["H_seed"]),
                kwargs["H_seed"],
                {},
                0.0,
            )
        ),
        validate_fn=(
            validate_fn
            if validate_fn is not None
            else lambda **_kwargs: (True, "ok")
        ),
        difference_spectrum=difference_spectrum,
    )
    return adjuster, result


def test_block_c_balances_odd_degree_with_first_su13_to_su12_rotation() -> None:
    nodes = [make_node(1, 23), make_node(2, 13), make_node(3, 23)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)
    rebuild_calls = []

    def rebuild(**kwargs):
        rebuild_calls.append(kwargs["H_seed"].detach().clone())
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        )

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        rebuild_fn=rebuild,
    )

    assert result.ok is True
    assert len(rebuild_calls) == 1
    assert int(rebuild_calls[0][11].item()) == 2
    assert int(rebuild_calls[0][12].item()) == 0
    assert int(rebuild_calls[0][13].item()) == 0
    assert int(rebuild_calls[0][23].item()) == 1
    assert int(rebuild_calls[0][24].item()) == 0
    assert adjuster._stage3_parity_balance_cursor == 1
    first_cycle = result.cycles[0]
    audit = first_cycle.plan["diagnostics"]["pre_rebuild_port_feasibility"]
    assert audit["before"]["total_port_degree"] == 7
    assert audit["after"]["total_port_degree"] == 8
    assert "su12_self_pair_is_even" not in audit["before"]
    assert "su12_self_pair_is_even" not in audit["after"]
    assert audit["applied_direction"] == [13, 12]
    assert audit["node_id"] == 2
    histogram = first_cycle.plan["histogram_adjustment_plan"]
    assert histogram["physical_conversion_count"] == 3
    assert histogram["parity_balance_op"]["from_su"] == 13
    assert histogram["parity_balance_op"]["to_su"] == 12
    assert histogram["port_role_balance_op"]["from_su"] == 12
    assert histogram["port_role_balance_op"]["to_su"] == 11
    assert histogram["port_role_balance_op"]["node_ids"] == [2]
    assert histogram["groups"][0]["h_compensation_ops"] == []
    assert histogram["accounting_diagnostics"]["ok"] is True
    assert first_cycle.h_compensation_count == 0
    assert first_cycle.parity_balance_count == 2
    transaction = first_cycle.plan["phase_result"]["transactions"][0]
    assert transaction["compensation_node_ids"] == []
    assert "parity_balance_node_ids" not in transaction
    assert [item["reason"] for item in transaction["conversion_path"]] == [
        "test_primary_resource_adjustment"
    ]


@pytest.mark.parametrize(
    "candidate_phase",
    ["resource_su24_ring_closure", "residual_resource_self_closure"],
)
def test_block_c_rolls_failed_topology_retry_back_to_candidate_start(
    candidate_phase,
) -> None:
    nodes = [make_node(1, 23), make_node(2, 13), make_node(3, 23)]
    H_before = _histogram(nodes)
    conversion_plan = _parity_rebuild_plan(nodes, primary_node_id=1)
    conversion_plan.phase = candidate_phase
    conversion_plan.histogram_adjustment_plan.phase = candidate_phase
    conversion_plan.phase_result.active_phase = candidate_phase
    for transaction in conversion_plan.phase_result.transactions:
        transaction.phase = candidate_phase
    is_ring = candidate_phase == "resource_su24_ring_closure"
    candidate = (
        {
            "candidate_signature": "fused:test-node-bound-candidate",
            "selected_su24_ids": [100, 101, 102, 103],
            "product_kind": "fatty_ring_fused",
            "skeleton_kind": "fused",
        }
        if is_ring
        else {
            "product_role": "residual_side_chain",
            "chain_23_count": 2,
            "required_counts": {11: 1, 22: 1, 23: 2},
            "conversion_count": 1,
            "converted_node_ids": [1],
        }
    )
    diagnostic_key = (
        "pending_su24_ring_candidate"
        if is_ring
        else "pending_residual_product_candidate"
    )
    planner_key = diagnostic_key
    blocked_reason = (
        "pending_su24_ring_candidate_not_materialized"
        if is_ring
        else "pending_residual_product_not_materialized"
    )
    conversion_plan.diagnostics.update({
        "candidate_alternatives_enabled": True,
        diagnostic_key: candidate,
    })
    planner_calls = []

    class Adjuster:
        def __init__(self):
            self._stage3_su11_rotation_index = 0
            self._stage3_parity_balance_cursor = 0

        def plan_topology_resource_adjustments(self, **kwargs):
            planner_calls.append(kwargs)
            if len(planner_calls) == 1:
                return conversion_plan
            if len(planner_calls) == 2:
                assert kwargs[planner_key] == candidate
                return Stage3AdjustmentPlan(
                    ok=False,
                    reason="phase_resource_infeasible",
                    phase=candidate_phase,
                    plan_kind="pending_topology_candidate_infeasible",
                    blocked_reason=blocked_reason,
                    phase_result=Stage3PhaseResult(
                        active_phase=candidate_phase,
                        phase_completed=False,
                        stop_reason="phase_resource_infeasible",
                        infeasibility_proof={
                            "phase": candidate_phase,
                            "reason": blocked_reason,
                        },
                    ),
                )
            assert kwargs[planner_key] is None
            assert kwargs["progress_phase"] is Stage3ProgressPhase(candidate_phase)
            assert kwargs["forbidden_resource_conversion_signatures"]
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

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"], kwargs["H"], kwargs["plan"]
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H_before,
        nodes,
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
    assert result.accepted_cycles == 0
    assert torch.equal(result.H, H_before)
    assert [int(node.global_id) for node in result.nodes] == [1, 2, 3]
    assert result.cycles[0].stop_reason == "rolled_back_by_pending_topology_failure"
    assert result.cycles[0].accepted is False
    assert result.cycles[1].diagnostics["restored_to_candidate_start"] is True


def test_block_c_port_balance_uses_second_rotation_direction() -> None:
    nodes = [make_node(1, 23), make_node(2, 12), make_node(3, 22)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        parity_cursor=1,
    )

    assert result.ok is True
    audit = result.cycles[0].plan["diagnostics"][
        "pre_rebuild_port_feasibility"
    ]
    assert audit["required_direction"] == [12, 13]
    assert audit["applied_direction"] == [12, 13]
    assert adjuster._stage3_parity_balance_cursor == 2


def test_block_c_port_balance_does_not_reverse_when_rotation_donor_is_missing() -> None:
    nodes = [make_node(1, 23), make_node(2, 12), make_node(3, 22)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)
    rebuild_calls = []

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        parity_cursor=0,
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs),
    )

    assert result.ok is False
    assert result.stop_reason == "pre_rebuild_port_feasibility_failed"
    assert rebuild_calls == []
    assert adjuster._stage3_parity_balance_cursor == 0
    audit = result.cycles[0].plan["diagnostics"][
        "pre_rebuild_port_feasibility"
    ]
    assert audit["required_direction"] == [13, 12]
    assert audit["attempted_directions"] == [{
        "kind": "total_port_parity",
        "from_su": 13,
        "to_su": 12,
        "candidate_node_ids": [],
    }]


@pytest.mark.parametrize("failure_mode", ["rebuild", "validation"])
def test_block_c_failed_candidate_does_not_advance_port_balance_cursor(
    failure_mode,
) -> None:
    nodes = [make_node(1, 23), make_node(2, 13), make_node(3, 23)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)

    def rebuild(**kwargs):
        if failure_mode == "rebuild":
            raise RuntimeError("test rebuild failure")
        return (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {},
            0.0,
        )

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (
            (False, "test validation failure")
            if failure_mode == "validation"
            else (True, "ok")
        ),
    )

    assert result.accepted_cycles == 0
    assert adjuster._stage3_parity_balance_cursor == 0
    assert torch.equal(result.H, _histogram(nodes))
    transaction = result.cycles[0].plan["phase_result"]["transactions"][0]
    assert transaction["committed"] is False
    attempted = transaction["infeasibility_proof"]["attempted_transaction"]
    assert attempted["compensation_node_ids"] == []
    assert "parity_balance_node_ids" not in attempted
    assert result.cycles[0].plan["histogram_adjustment_plan"][
        "parity_balance_op"
    ]["node_ids"] == [2]
    assert result.cycles[0].plan["histogram_adjustment_plan"][
        "port_role_balance_op"
    ]["node_ids"] == [2]


def test_block_c_rejects_odd_port_batch_before_rebuild_without_unused_donor() -> None:
    nodes = [make_node(1, 23)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)
    rebuild_calls = []

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs),
    )

    assert rebuild_calls == []
    assert result.stop_reason == "pre_rebuild_port_feasibility_failed"
    assert result.cycles[0].blocked_reason == (
        "odd_port_degree_no_unused_safe_balance_donor"
    )
    assert adjuster._stage3_parity_balance_cursor == 0


def test_block_c_even_total_still_balances_odd_layer1_port_roles() -> None:
    nodes = [make_node(1, 12), make_node(2, 23), make_node(3, 23)]
    plan = _parity_rebuild_plan(
        nodes,
        primary_node_id=2,
        primary_source=23,
        primary_target=11,
    )

    adjuster, result = _run_one_parity_rebuild_plan(plan, nodes)

    assert result.ok is True
    audit = result.cycles[0].plan["diagnostics"][
        "pre_rebuild_port_feasibility"
    ]
    assert audit["reason"] == "pre_rebuild_ports_balanced"
    assert audit["balance_applied"] is True
    assert audit["before"]["degree_is_even"] is True
    assert audit["before"]["port_roles_ok"] is False
    assert audit["after"]["port_roles_ok"] is True
    histogram = result.cycles[0].plan["histogram_adjustment_plan"]
    assert histogram["parity_balance_op"] is None
    assert histogram["port_role_balance_op"]["from_su"] == 12
    assert histogram["port_role_balance_op"]["to_su"] == 11
    assert adjuster._stage3_parity_balance_cursor == 0


def test_block_c_final_read_only_audit_promotes_complete_after_last_rebuild() -> None:
    node = make_node(7, 12)
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
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
        nodes,
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
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
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
        nodes,
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
    assert cycle.acceptance_reason == "valid_topology_histogram_fresh_rebuild"


def test_block_c_only_stops_after_the_full_physical_state_repeats() -> None:
    node = make_node(7, 22)
    nodes = [node, make_node(8, 22)]
    H = _histogram(nodes)
    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            H_before = kwargs["H"].detach().clone()
            source, target = (
                (22, 24) if int(H_before[24].item()) == 0 else (24, 22)
            )
            node_id = next(
                int(value.global_id)
                for value in kwargs["nodes"]
                if int(value.su_type) == int(source)
            )
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
        nodes,
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
    assert result.stop_reason == "stage3_state_cycle_detected"
    # The original H recurs in cycle 2, but its rebuilt node IDs differ from
    # the initial graph. It is therefore accepted. Cycle 3 reconstructs the
    # exact state already accepted in cycle 1 and is the real loop.
    assert result.accepted_cycles == 2
    assert len(result.cycles) == 3
    terminal = result.cycles[-1]
    assert terminal.applied is True
    assert terminal.rebuild_valid is True
    assert terminal.blocked_reason == "stage3_state_cycle_detected"
    assert terminal.diagnostics["histogram_was_previously_seen"] is True
    assert terminal.diagnostics["candidate_state_signature"] == terminal.state_signature


def test_block_c_accepts_valid_topology_despite_large_r2_regression() -> None:
    nodes = [make_node(1, 23), make_node(2, 13), make_node(3, 23)]
    plan = _parity_rebuild_plan(nodes, primary_node_id=1)

    adjuster, result = _run_one_parity_rebuild_plan(
        plan,
        nodes,
        rebuild_fn=lambda **kwargs: (
            _nodes_from_histogram(kwargs["H_seed"]),
            kwargs["H_seed"],
            {"r2": 0.70},
            0.70,
        ),
        difference_spectrum={"r2": 0.80},
    )

    assert adjuster.calls == 2
    assert result.ok is True
    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert result.diff["r2"] == pytest.approx(0.70)
    assert "best_checkpoint_r2" not in result.diagnostics
    accepted = result.cycles[0]
    assert accepted.accepted is True
    assert accepted.acceptance_reason == "valid_topology_histogram_fresh_rebuild"
    assert "r2_drop" not in accepted.diagnostics


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
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
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
        nodes,
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
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
    H_after = H.detach().clone()
    H_after[13] -= 1
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
        nodes,
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
    nodes = [make_node(1, 13), make_node(2, 12), make_node(3, 23)]
    H = _histogram(nodes)

    class Adjuster:
        def __init__(self):
            self.calls = []
            self._stage3_su11_rotation_index = 0

        def plan_topology_resource_adjustments(self, **kwargs):
            forbidden = set(kwargs.get("forbidden_resource_conversion_signatures", []) or [])
            self.calls.append(sorted(forbidden))
            if int(kwargs["H"][11].item()) >= 1:
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
            source, node_id = (
                (12, 2)
                if any(value.endswith("|bundle:13>11") for value in forbidden)
                else (13, 1)
            )
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
                inventory_after={
                    12: int(source != 12),
                    13: int(source != 13),
                    11: 1,
                },
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
    assert adjuster.calls[0] == []
    assert len(adjuster.calls[1]) == 1
    assert adjuster.calls[1][0].endswith("|bundle:13>11")
    assert adjuster.calls[2] == []
    assert int(result.H[13].item()) == 0
    assert int(result.H[12].item()) == 0
    assert int(result.H[11].item()) == 2
    assert result.cycles[0].diagnostics[
        "rejected_resource_conversion_signature"
    ].endswith("|bundle:13>11")


def test_resource_step_signature_ignores_node_ids_and_keeps_parity_separate() -> None:
    H = torch.zeros(40, dtype=torch.long)
    H[13] = 2
    H[24] = 2
    histogram = HistogramAdjustmentPlan(
        H_before=H,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                13,
                11,
                1,
                "resource_rebalance",
                node_ids=[225],
            ),
            h_compensation_ops=[
                HistogramConversion(24, 11, 1, "soft_h_balance_compensation", node_ids=[488]),
                HistogramConversion(24, 11, 1, "soft_h_balance_compensation", node_ids=[489]),
            ],
        )],
        parity_balance_op=HistogramConversion(
            13, 12, 1, "pre_rebuild_port_parity_balance", node_ids=[226]
        ),
    )

    assert _last_resource_step_signature(histogram).endswith(
        "|bundle:13>11|24>11,24>11"
    )


def test_rejected_resource_steps_remain_state_scoped_without_primary_escalation() -> None:
    plan = SimpleNamespace(
        diagnostics={"candidate_alternatives_enabled": True},
    )
    rejected = set()
    for compensation_count in range(3):
        histogram = HistogramAdjustmentPlan(
            H_before=torch.zeros(40, dtype=torch.long),
            groups=[AtomicAdjustmentGroup(
                primary_op=HistogramConversion(13, 11, 1, "resource_rebalance"),
                h_compensation_ops=[
                    HistogramConversion(24, 11, 1, "soft_h_balance_compensation")
                    for _ in range(compensation_count)
                ],
            )],
        )
        _register_rejected_resource_candidate(plan, histogram, rejected)

    assert len(rejected) == 3
    assert all(value.startswith("state:") for value in rejected)
    assert "13>11" not in rejected


def test_multi_primary_failure_rejects_only_last_step_at_its_intermediate_state() -> None:
    plan = SimpleNamespace(
        diagnostics={"candidate_alternatives_enabled": True},
    )
    histogram = HistogramAdjustmentPlan(
        H_before=torch.tensor([0] * 12 + [1, 1] + [0] * 26),
        groups=[
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(13, 11, 1, "resource_rebalance"),
            ),
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(12, 11, 1, "resource_rebalance"),
            ),
        ],
    )
    rejected = set()

    signature = _register_rejected_resource_candidate(plan, histogram, rejected)

    assert signature.endswith("|bundle:12>11")
    assert "bundle:13>11" not in signature
    assert rejected == {signature}


def test_multi_primary_validation_failure_keeps_fixed_primary_limit(
    monkeypatch,
) -> None:
    nodes = [make_node(1, 13), make_node(2, 12)]
    H = _histogram(nodes)

    monkeypatch.setattr(
        "model.inverse.layer4.inverse_layer4_block_c._append_stage3_parity_balance",
        lambda **_kwargs: (True, {"enforced": False, "reason": "test_disabled"}),
    )

    class Adjuster:
        def __init__(self):
            self.calls = []

        def plan_topology_resource_adjustments(self, **kwargs):
            limit = int(kwargs["max_primary_adjustments_per_stage"])
            forbidden = sorted(
                kwargs.get("forbidden_resource_conversion_signatures", []) or []
            )
            self.calls.append((limit, forbidden))
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

            specs = (
                [(13, 1)]
                if forbidden
                else [(13, 1), (12, 2)]
            )
            H_after = kwargs["H"].detach().clone()
            groups = []
            for source, node_id in specs:
                H_after[source] -= 1
                H_after[11] += 1
                groups.append(AtomicAdjustmentGroup(
                    primary_op=HistogramConversion(
                        source,
                        11,
                        1,
                        "fixed_limit_candidate_retry",
                        node_ids=[node_id],
                    ),
                    combined_delta={source: -1, 11: 1},
                ))
            inventory_before = {12: 1, 13: 1, 11: 0}
            inventory_after = dict(inventory_before)
            for source, _ in specs:
                inventory_after[source] -= 1
                inventory_after[11] += 1
            histogram = HistogramAdjustmentPlan(
                phase="external_topology_assembly",
                groups=groups,
                primary_adjustment_count=len(groups),
                physical_conversion_count=len(groups),
                H_before=kwargs["H"].detach().clone(),
                H_after=H_after,
                inventory_before=inventory_before,
                inventory_after=inventory_after,
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
                    primary_adjustment_count=len(groups),
                    stop_reason="stage3_rebuild_required",
                ),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(
                kwargs["nodes"],
                kwargs["H"],
                kwargs["plan"],
            )

    validation_count = 0

    def validate(**_kwargs):
        nonlocal validation_count
        validation_count += 1
        return validation_count > 1, "combined_batch_rejected"

    adjuster = Adjuster()
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

    assert result.ok is True
    assert result.accepted_cycles == 1
    assert [call[0] for call in adjuster.calls[:3]] == [10, 10, 10]
    assert adjuster.calls[0][1] == []
    assert len(adjuster.calls[1][1]) == 1
    assert adjuster.calls[1][1][0].endswith("|bundle:12>11")
    assert adjuster.calls[2][1] == []
    assert int(result.H[11].item()) == 1
    failed_cycle = result.cycles[0]
    assert failed_cycle.diagnostics[
        "rejected_resource_conversion_signature"
    ].endswith("|bundle:12>11")
    assert failed_cycle.diagnostics["primary_adjustment_limit"] == 10


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
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
    H_after = H.detach().clone()
    H_after[13] -= 1
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
        nodes,
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
    nodes = [node, make_node(8, 13), make_node(9, 22)]
    H = _histogram(nodes)
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
        nodes,
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
        progress_phase=Stage3ProgressPhase.SU24_RING_CLOSURE,
        h_tolerance=0.06,
    )

    assert plan.reason == "stage3_complete"
    assert plan.phase == "complete"
    assert plan.planned_conversions == []
    assert plan.histogram_adjustment_plan is None
    assert int(H[11].item()) == 0
    assert int(H[23].item()) == 0


def test_layer1_has_no_soft_consistency_acceptance_path() -> None:
    assigner = Layer1Assigner()
    assert not hasattr(assigner, "assess_consistency_acceptance")


def test_layer1_stage3_identity_seed_preserves_global_ids_across_type_changes() -> None:
    seed_nodes = [
        make_node(0, 23),
        make_node(1, 11),
        make_node(2, 23),
        make_node(3, 22),
    ]
    H = _histogram(seed_nodes)

    rebuilt = Layer1Assigner()._initialize_node_pool(
        H,
        identity_seed_nodes=seed_nodes,
    )

    assert [int(node.global_id) for node in rebuilt] == [0, 1, 2, 3]
    assert [int(node.su_type) for node in rebuilt] == [23, 11, 23, 22]
    assert all(rebuilt[index].global_id == index for index in range(len(rebuilt)))


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
            "h": {"hard_bound_penalty": 1.25, "drift_penalty": 0.5},
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
    assert "H_hard_bound_penalty=1.2500" in output
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
            if kwargs["progress_phase"] is Stage3ProgressPhase.SPECTRUM_REBALANCE:
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
