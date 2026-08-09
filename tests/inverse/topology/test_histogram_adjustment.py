import torch

from model.inverse.topology.histogram_adjustment import (
    _conversion_bundle_signature,
    apply_histogram_adjustment_plan,
    build_histogram_adjustment_plan,
)
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.conversion_scoring import (
    ConversionContext,
    _flex_transition_penalty,
    evaluate_flex_impact,
    score_conversion_bundle,
)
from model.inverse.topology.resource_profiles import h_hard_bounds
from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
)


def test_most_surplus_resource_fills_most_deficient_resource() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[11], H[23], H[22], H[24] = 1, 3, 3, 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 4, 23: 7, 22: 3, 24: 0},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts_by_su": {11: 1, 23: 3, 22: 3, 24: 1}}
            }
        },
    )

    histogram_plan = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=1)

    assert histogram_plan.primary_adjustment_count == 1
    assert histogram_plan.groups[0].primary_op.from_su == 24
    assert histogram_plan.groups[0].primary_op.to_su == 23
    assert histogram_plan.groups[0].demand_after == {11: 3, 23: 3, 22: 0, 24: 0}


def test_histogram_application_does_not_mutate_nodes() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[24] = 1
    node = type("Node", (), {"su_type": 24, "hop1_ids": [8], "hop1_su": [11], "global_id": 7})()
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        planned_conversions=[{"from_su": 24, "to_su": 23, "node_id": 7, "reason": "residual"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"24": 1, "23": 0},
                },
            },
        },
    )
    histogram_plan = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=1)

    result = apply_histogram_adjustment_plan(H, histogram_plan, nodes=[node])

    assert result.changed is True
    assert int(result.H[24]) == 0
    assert int(result.H[23]) == 1
    assert node.su_type == 24
    assert node.hop1_ids == [8]
    assert node.hop1_su == [11]
    assert node.global_id == 7


def test_hard_valid_resource_conversion_does_not_add_h_compensation() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[11], H[22], H[23], H[24] = 1, 10, 1, 3
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 1, 22: 0, 23: 1, 24: 0},
        planned_conversions=[{"from_su": 24, "to_su": 23, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "convertible_counts": {11: 1, 22: 10, 23: 1, 24: 3},
                },
            },
        },
    )
    target_elements = torch.tensor([0.0, 34.0])

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=target_elements,
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
        h_tolerance=0.11,
    )

    assert histogram_plan.primary_adjustment_count == 1
    assert histogram_plan.groups[0].primary_op.from_su == 24
    assert histogram_plan.groups[0].primary_op.to_su == 23
    assert histogram_plan.groups[0].h_after_final == histogram_plan.groups[0].h_after_primary
    assert histogram_plan.groups[0].h_compensation_ops == []


def test_stage3_hard_bounds_are_symmetric_eight_percent() -> None:
    assert h_hard_bounds(100.0) == (92.0, 108.0)


def test_stage3_resource_candidate_at_plus_five_percent_is_hard_valid() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[13] = 106
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        planned_conversions=[{"from_su": 13, "to_su": 11, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"convertible_counts": {11: 0, 13: 106}}
            }
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=0,
        h_tolerance=0.08,
    )

    assert result.primary_adjustment_count == 1
    assert result.groups[0].h_after_final == 105.0


def test_stage3_resource_candidate_at_plus_nine_percent_is_hard_invalid() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[13] = 110
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        planned_conversions=[{"from_su": 13, "to_su": 11, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"convertible_counts": {11: 0, 13: 110}}
            }
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=0,
        h_tolerance=0.08,
    )

    assert result.primary_adjustment_count == 0
    assert torch.equal(result.H_after, H)


def test_stage3_resource_h_compensation_is_suppressed_inside_hard_window() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[13], H[24] = 516, 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 0, 22: 0, 23: 0, 24: 0},
        planned_conversions=[{"from_su": 13, "to_su": 11, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "convertible_counts": {11: 0, 13: 516, 22: 0, 23: 0, 24: 1},
                },
            },
        },
    )

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
        h_tolerance=0.08,
    )

    assert histogram_plan.primary_adjustment_count == 1
    assert histogram_plan.groups[0].h_after_primary == 516.0
    assert histogram_plan.groups[0].h_compensation_ops == []
    assert histogram_plan.accounting_diagnostics["effective_h_tolerance"] == 0.08
    assert histogram_plan.accounting_diagnostics["stage3_hard_lower_ratio"] == -0.08


def test_stage3_resource_does_not_compensate_inside_eight_percent_hard_window() -> None:
    H = torch.zeros(33, dtype=torch.long)
    # Effective H includes both SU13 (H=1) and SU24 (H=1): 494 + 10 = 504.
    H[13], H[24] = 494, 10
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 0, 22: 0, 23: 0, 24: 0},
        planned_conversions=[{"from_su": 13, "to_su": 11, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "convertible_counts": {11: 0, 13: 494, 22: 0, 23: 0, 24: 10},
                },
            },
        },
    )

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=8,
        h_tolerance=0.08,
    )

    group = histogram_plan.groups[0]
    assert group.h_after_primary == 503.0
    assert group.h_after_final == 503.0
    assert group.h_compensation_ops == []


def test_flex_scoring_accounts_for_product_created_by_current_transaction() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13] = 6, 24
    impact = evaluate_flex_impact(
        H,
        H,
        {
            "bridgehead_info": (0, 0, 0, 0, 0, 0, 6),
            "histogram_su12_count": 6,
            "explicit_flex_count": 1,
            "effective_flex_count": 4,
            "expected_flex_product_gain": 1,
        },
    )

    assert impact.before["explicit_flex_count"] == 4
    assert impact.after["explicit_flex_count"] == 5


def test_flex_gap_reduction_outweighs_soft_h_margin_preference() -> None:
    stalled = _flex_transition_penalty(
        {
            "deviation": 1,
            "scenario": "below_range",
            "minimum_flex_gap": 1,
            "maximum_flex_capacity": 8,
            "X": 30,
        },
        {
            "deviation": 1,
            "scenario": "below_range",
            "minimum_flex_gap": 1,
            "maximum_flex_capacity": 8,
            "X": 31,
        },
    )
    converged = _flex_transition_penalty(
        {
            "deviation": 1,
            "scenario": "below_range",
            "minimum_flex_gap": 1,
            "maximum_flex_capacity": 8,
            "X": 30,
        },
        {
            "deviation": 0,
            "scenario": "in_range",
            "minimum_flex_gap": 0,
            "maximum_flex_capacity": 8,
            "X": 30,
        },
    )

    assert stalled == 0.0
    assert converged <= -120.0


def _flex_lower_cap_fixture(phase: str) -> tuple[torch.Tensor, Stage3AdjustmentPlan]:
    H = torch.zeros(40, dtype=torch.long)
    counts = {
        0: 1,
        1: 10,
        2: 11,
        3: 12,
        5: 80,
        6: 2,
        9: 23,
        10: 30,
        11: 64,
        12: 44,
        13: 36,
        14: 1,
        15: 8,
        16: 3,
        17: 2,
        18: 2,
        19: 83,
        20: 2,
        22: 11,
        23: 58,
        24: 16,
        25: 1,
        26: 3,
        27: 2,
        28: 62,
        29: 49,
    }
    for su_type, count in counts.items():
        H[int(su_type)] = int(count)
    plan = Stage3AdjustmentPlan(
        phase=phase,
        planned_conversions=[
            {"from_su": 12, "to_su": 11, "reason": "test_flex_lower_cap"}
            for _ in range(3)
        ],
        runtime={
            "flex_audit": {
                "enforced": True,
                "bridgehead_info": (0, 0, 0, 0, 0, 0, 44),
                "histogram_su12_count": 44,
                "explicit_flex_count": 15,
                "effective_flex_count": 15,
                "flex_lower": 17,
            },
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "convertible_counts": {
                        11: 64,
                        12: 44,
                        13: 36,
                        22: 11,
                        23: 58,
                        24: 16,
                    },
                },
            },
        },
    )
    return H, plan


def test_su12_to_su11_allows_small_but_blocks_large_cumulative_flex_lower_increase() -> None:
    H, plan = _flex_lower_cap_fixture("external_topology_assembly")

    result = build_histogram_adjustment_plan(
        plan,
        H,
        max_primary_adjustments=3,
        max_h_compensation_ops=0,
    )

    assert result.primary_adjustment_count == 2
    assert int(result.H_after[12].item()) == 42
    assert int(result.H_after[11].item()) == 66
    rejections = result.accounting_diagnostics["flex_lower_increase_rejections"]
    assert rejections
    assert rejections[-1]["baseline_flex_lower"] == 17
    assert rejections[-1]["maximum_allowed_flex_lower"] == 18
    assert rejections[-1]["candidate_flex_lower"] == 19


def test_flex_reconciliation_can_exceed_general_flex_lower_increase_cap() -> None:
    H, plan = _flex_lower_cap_fixture("flex_range_reconciliation")

    result = build_histogram_adjustment_plan(
        plan,
        H,
        max_primary_adjustments=3,
        max_h_compensation_ops=0,
    )

    assert result.primary_adjustment_count == 3
    assert int(result.H_after[12].item()) == 41
    assert result.accounting_diagnostics["flex_lower_increase_rejections"] == []


def test_conversion_scoring_balanced_h_prefers_su12_over_hungry_su22_to_su11() -> None:
    from model.inverse.topology.histogram_adjustment import _h_total

    H = torch.zeros(33, dtype=torch.long)
    H[12], H[22] = 1, 1
    context = ConversionContext(
        inventory={11: 0, 12: 1, 22: 1, 23: 0, 24: 0},
        demand={11: 1, 22: 0, 23: 0, 24: 0},
        current_H=H,
        target_h=3.0,
    )

    aromatic = score_conversion_bundle(
        context,
        [HistogramConversion(12, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )
    terminal = score_conversion_bundle(
        context,
        [HistogramConversion(22, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )

    assert aromatic.total_score < terminal.total_score


def test_conversion_scoring_high_h_allows_h_lowering_su22_to_su11() -> None:
    from model.inverse.topology.histogram_adjustment import _h_total

    H = torch.zeros(33, dtype=torch.long)
    H[12], H[22] = 1, 2
    context = ConversionContext(
        inventory={11: 0, 12: 1, 22: 2, 23: 0, 24: 0},
        demand={11: 1, 22: 0, 23: 0, 24: 0},
        current_H=H,
        target_h=3.0,
    )

    aromatic = score_conversion_bundle(
        context,
        [HistogramConversion(12, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )
    terminal = score_conversion_bundle(
        context,
        [HistogramConversion(22, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )

    assert terminal.total_score < aromatic.total_score


def test_explicit_primary_cannot_consume_phase_required_resource() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23] = 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 0, 22: 0, 23: 1, 24: 0},
        planned_conversions=[{"from_su": 23, "to_su": 11, "reason": "invalid_primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"convertible_counts": {11: 0, 22: 0, 23: 1, 24: 0}},
            },
        },
    )

    histogram_plan = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=1)

    assert histogram_plan.primary_adjustment_count == 0
    assert torch.equal(histogram_plan.H_before, histogram_plan.H_after)


def test_executor_atomically_rejects_bundle_that_worsens_phase_deficit() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22], H[23], H[24] = 1, 1, 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={11: 0, 22: 1, 23: 1, 24: 1},
        demand_before={11: 0, 22: 0, 23: 1, 24: 0},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(24, 22, 1, "primary"),
            h_compensation_ops=[HistogramConversion(23, 11, 1, "invalid_compensation")],
            combined_delta={24: -1, 22: 1, 23: -1, 11: 1},
        )],
    )

    result = apply_histogram_adjustment_plan(H, histogram_plan)

    assert result.changed is False
    assert torch.equal(result.H, H)
    assert result.applied_ops == []
    assert result.rejected_ops[0]["reason"] == "bundle_would_worsen_phase_resource_deficit"


def test_resource_conversion_without_hard_valid_bundle_is_rejected() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23], H[24] = 1, 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        minimum_resource_request={11: 0, 22: 0, 23: 1, 24: 0},
        planned_conversions=[{"from_su": 24, "to_su": 23, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"convertible_counts": {11: 0, 22: 0, 23: 1, 24: 1}},
            },
        },
    )

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 0.5]),
        max_primary_adjustments=1,
        h_tolerance=0.0,
    )

    assert histogram_plan.primary_adjustment_count == 0
    assert torch.equal(histogram_plan.H_after, H)


def test_primary_conversion_below_hard_window_is_blocked() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22], H[23] = 1, 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={11: 1},
        planned_conversions=[{"from_su": 22, "to_su": 11, "reason": "primary"}],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"convertible_counts": {11: 0, 22: 1, 23: 1, 24: 0}},
            },
        },
    )

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 5.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=0,
        h_tolerance=0.0,
    )

    assert histogram_plan.primary_adjustment_count == 0
    assert torch.equal(histogram_plan.H_after, H)


def test_hard_demand_precedes_preferred_demand() -> None:
    import torch
    from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
    from model.inverse.topology.histogram_adjustment import build_histogram_adjustment_plan

    H = torch.zeros(33, dtype=torch.long)
    H[22] = 2
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1},
        preferred_required_counts={"11": 1, "23": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 0, "22": 2, "23": 0, "24": 0}}
            }
        },
    )

    result = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=1)

    assert result.primary_adjustment_count == 1
    assert int(result.groups[0].primary_op.to_su) == 11


def test_gap_recomputed_after_each_primary_adjustment() -> None:
    import torch
    from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
    from model.inverse.topology.histogram_adjustment import build_histogram_adjustment_plan

    H = torch.zeros(33, dtype=torch.long)
    H[11], H[23], H[22], H[24] = 1, 6, 9, 3
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 4, "23": 7, "22": 3, "24": 0},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 1, "23": 6, "22": 9, "24": 3}}
            }
        },
    )

    result = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=4)
    targets = [int(group.primary_op.to_su) for group in result.groups]

    assert targets[:2] == [11, 11]
    assert result.unresolved_counts == {}
    assert result.inventory_after[11] >= 4
    assert result.inventory_after[23] >= 7


def test_rejected_deterministic_primary_does_not_loop_forever(monkeypatch) -> None:
    from model.inverse.topology import histogram_adjustment as module

    H = torch.zeros(33, dtype=torch.long)
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"11": 0, "12": 0, "13": 0, "22": 1, "23": 0, "24": 0}
                }
            }
        },
    )
    monkeypatch.setattr(
        module,
        "_choose_bundle_from_gap",
        lambda H_work, inventory, demand, aromatic_cursor, **kwargs: (
            [module.HistogramConversion(22, 11, 1, "deterministic_invalid")],
            aromatic_cursor,
            None,
        ),
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        max_primary_adjustments=4,
    )

    assert result.primary_adjustment_count == 0
    assert result.blocked_reason == "repeated_primary_candidate_no_progress"


def test_external_shortage_cover_ops_are_physical_conversions() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1},
        planned_conversions=[
            {"from_su": 22, "to_su": 11, "reason": "external_only_shortage_cover"},
        ],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 0, "22": 1, "23": 0, "24": 0}}
            }
        },
    )

    result = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=1)

    assert result.primary_adjustment_count == 1
    assert int(result.groups[0].primary_op.from_su) == 22
    assert int(result.groups[0].primary_op.to_su) == 11


def test_explicit_operation_is_not_replayed_after_gap_is_filled() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 2
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1},
        planned_conversions=[
            {"from_su": 22, "to_su": 11, "reason": "shortage_cover"},
        ],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 0, "22": 2, "23": 0, "24": 0}},
            },
        },
    )

    result = build_histogram_adjustment_plan(plan, H, max_primary_adjustments=2)

    assert result.primary_adjustment_count == 1
    assert int(result.H_after[11].item()) == 1
    assert int(result.H_after[22].item()) == 1


def test_topology_conversion_without_element_target_uses_available_donor() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 1
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 0, "22": 1, "23": 0, "24": 0}}
            }
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        max_primary_adjustments=1,
    )

    assert result.primary_adjustment_count == 1
    assert result.groups[0].primary_op.from_su == 22
    assert result.groups[0].primary_op.to_su == 11
    assert result.groups[0].h_after_final < result.groups[0].h_before


def test_histogram_su11_gap_selection_is_stable_without_element_target() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 2, 13: 1, 22: 9, 23: 12, 24: 9}
    demand = {11: 1, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22], H[23], H[24] = 2, 1, 9, 12, 9

    lower_op, _ = module._choose_from_gap(H, inventory, demand, 0)
    balanced_op, _ = module._choose_from_gap(H, inventory, demand, 0)

    assert (lower_op.from_su, lower_op.to_su) == (12, 11)
    assert (balanced_op.from_su, balanced_op.to_su) == (12, 11)


def test_histogram_su11_selection_uses_rotation_without_element_target() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 10, 13: 10, 22: 1, 23: 3, 24: 2}
    demand = {11: 1, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22], H[23], H[24] = 10, 10, 1, 3, 2

    op, cursor = module._choose_from_gap(
        H,
        inventory,
        demand,
        0,
    )

    assert (op.from_su, op.to_su) == (12, 11)
    assert cursor == 1


def test_histogram_su11_donor_h_drift_precedes_rotation() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 2, 13: 2, 22: 0, 23: 0, 24: 0}
    demand = {11: 2, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 2
    H[13] = 2

    first, cursor = module._choose_from_gap(H, inventory, demand, 0)
    inventory[int(first.from_su)] -= 1
    inventory[11] += 1
    H[int(first.from_su)] -= 1
    H[11] += 1
    second, cursor = module._choose_from_gap(H, inventory, demand, cursor)

    assert (first.from_su, first.to_su) == (12, 11)
    assert (second.from_su, second.to_su) == (12, 11)
    assert cursor == 4


def test_histogram_su11_rotation_does_not_override_lower_h_drift() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 2, 13: 3, 22: 0, 23: 0, 24: 0}
    demand = {11: 5, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 2
    H[13] = 3
    cursor = 0
    selected_sources = []

    for _ in range(5):
        op, cursor = module._choose_from_gap(H, inventory, demand, cursor)
        assert op is not None
        selected_sources.append(int(op.from_su))
        inventory[int(op.from_su)] -= 1
        inventory[11] += 1
        H[int(op.from_su)] -= 1
        H[11] += 1

    assert selected_sources == [12, 12, 13, 13, 13]
    assert cursor == 8


def test_su11_aromatic_donor_h_scores_are_equal_inside_hard_window() -> None:
    from model.inverse.topology.histogram_adjustment import _h_total

    high_h = torch.zeros(140, dtype=torch.long)
    high_h[12] = 2
    high_h[13] = 105
    high_context = ConversionContext(
        inventory={11: 0, 12: 2, 13: 105},
        demand={11: 1},
        current_H=high_h,
        target_h=100.0,
        su11_rotation_index=0,
    )
    high_12 = score_conversion_bundle(
        high_context,
        [HistogramConversion(12, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )
    high_13 = score_conversion_bundle(
        high_context,
        [HistogramConversion(13, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )

    low_h = torch.zeros(140, dtype=torch.long)
    low_h[12] = 2
    low_h[13] = 98
    low_context = ConversionContext(
        inventory={11: 0, 12: 2, 13: 98},
        demand={11: 1},
        current_H=low_h,
        target_h=100.0,
        su11_rotation_index=1,
    )
    low_12 = score_conversion_bundle(
        low_context,
        [HistogramConversion(12, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )
    low_13 = score_conversion_bundle(
        low_context,
        [HistogramConversion(13, 11, 1, "candidate")],
        h_total_fn=_h_total,
    )

    assert high_12.diagnostics["h"]["hard_bound_penalty"] == 0.0
    assert high_13.diagnostics["h"]["hard_bound_penalty"] == 0.0
    assert low_12.diagnostics["h"]["hard_bound_penalty"] == 0.0
    assert low_13.diagnostics["h"]["hard_bound_penalty"] == 0.0


def test_residual_su11_to_aromatic_uses_same_three_step_cycle() -> None:
    from model.inverse.topology.histogram_adjustment import next_su11_aromatic_target

    cursor = 0
    targets = []
    for _ in range(5):
        target, cursor = next_su11_aromatic_target(cursor)
        targets.append(target)

    assert targets == [12, 13, 13, 12, 13]
    assert cursor == 5


def test_histogram_non_su11_target_selects_topology_candidate_without_element_target() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 3, 13: 3, 22: 5, 23: 0, 24: 1}
    demand = {11: 0, 22: 0, 23: 1, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22], H[24] = 3, 3, 5, 1

    op, _ = module._choose_from_gap(H, inventory, demand, 0)

    assert op.to_su == 23
    assert op.from_su == 22
    assert op.via_su is None


def test_su11_rotation_advances_only_after_successful_real_adjustment() -> None:
    from types import SimpleNamespace

    from model.inverse.layer4.inverse_layer4 import Layer4Adjuster

    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    adjuster._stage3_su11_rotation_index = 0
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 1
    node = SimpleNamespace(global_id=101, su_type=12)
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={12: 1},
        demand_before={11: 1},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "su11_rotation", node_ids=[101]),
            combined_delta={12: -1, 11: 1},
        )],
    )
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        histogram_adjustment_plan=histogram_plan,
        diagnostics={"su11_rotation_index_after": 1},
    )

    result = adjuster.apply_topology_resource_adjustments([node], H, plan)

    assert result.changed is True
    assert adjuster._stage3_su11_rotation_index == 1

    rejected_plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        histogram_adjustment_plan=HistogramAdjustmentPlan(
            phase="external_side_chain_closure",
            inventory_before={12: 1},
            demand_before={11: 1},
            groups=[AtomicAdjustmentGroup(
                primary_op=HistogramConversion(12, 11, 1, "su11_rotation", node_ids=[999]),
                combined_delta={12: -1, 11: 1},
            )],
        ),
        diagnostics={"su11_rotation_index_after": 2},
    )
    rejected = adjuster.apply_topology_resource_adjustments([node], H, rejected_plan)

    assert rejected.changed is False
    assert adjuster._stage3_su11_rotation_index == 1


def test_real_node_v3_supports_persistent_stage3_conversion_history() -> None:
    from model.shared.inverse_common import _NodeV3

    node = _NodeV3(global_id=7, su_type=12)
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 1
    plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={12: 1},
        demand_before={11: 1},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "rotation", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    result = apply_histogram_adjustment_plan(H, plan, nodes=[node])

    assert result.changed is True
    assert result.nodes[0].global_id == 7
    assert result.nodes[0].su_type == 11
    assert result.nodes[0].stage3_conversion_history == [
        {"from_su": 12, "to_su": 11, "reason": "rotation"}
    ]


def test_low_h_residual_conversion_does_not_use_aromatic_donor() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22] = 1, 94, 1
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        hard_required_counts={"11": 1},
        planned_conversions=[
            {"from_su": 22, "to_su": 11, "reason": "legacy_residual_request"},
        ],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"11": 0, "12": 1, "13": 0, "22": 1, "23": 0, "24": 0},
                    "su11_donor_counts": {"12": 1, "13": 0},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
    )

    assert result.primary_adjustment_count == 1
    operations = [
        operation
        for group in result.groups
        for operation in [group.primary_op, *group.h_compensation_ops]
    ]
    assert (operations[0].from_su, operations[0].to_su) == (22, 11)
    assert all(
        int(operation.from_su) in {11, 22, 23}
        and int(operation.to_su) in {11, 22, 23}
        for operation in operations
    )


def test_low_h_su23_deficit_uses_surplus_su11() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[11], H[13] = 2, 97
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1, "23": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 2, "23": 0, "22": 0, "24": 0}},
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
    )

    primary = result.groups[0].primary_op
    assert (primary.from_su, primary.to_su, primary.via_su) == (11, 23, None)
    assert result.groups[0].h_after_final == 99.0


def test_low_h_double_deficit_uses_two_aromatic_donors() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13] = 2, 97
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"11": 1, "23": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"11": 0, "12": 2, "13": 0, "23": 0, "22": 0, "24": 0},
                    "su11_donor_counts": {"12": 2, "13": 0},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=2,
    )

    assert result.primary_adjustment_count == 2
    assert result.unresolved_counts == {}
    assert (result.groups[0].primary_op.from_su, result.groups[0].primary_op.to_su) == (12, 23)
    assert result.groups[0].primary_op.via_su == 11
    assert (result.groups[1].primary_op.from_su, result.groups[1].primary_op.to_su) == (12, 11)
    assert all(not group.h_compensation_ops for group in result.groups)
    assert result.inventory_after[11] == 1
    assert result.inventory_after[23] == 1
    assert result.physical_conversion_count == 3


def test_low_h_atomic_su13_path_improves_h_via_su11() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[13] = 97
    plan = Stage3AdjustmentPlan(
        phase="external_side_chain_closure",
        hard_required_counts={"23": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"11": 0, "12": 0, "13": 1, "23": 0, "22": 0, "24": 0},
                    "su11_donor_counts": {"12": 0, "13": 1},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
    )

    primary = result.groups[0].primary_op
    assert (primary.from_su, primary.via_su, primary.to_su) == (13, 11, 23)
    assert result.groups[0].h_before == 97.0
    assert result.groups[0].h_after_primary == 96.0
    assert result.groups[0].h_after_final == 98.0


def test_ym10_su11_gap_does_not_spend_protected_su23_on_h_compensation() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13] = 519
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {
                        "11": 0,
                        "13": 1,
                        "22": 22,
                        "23": 13,
                        "24": 2,
                    },
                    "su11_donor_counts": {"12": 0, "13": 1},
                    "protected_counts": {"23": 20},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=8,
    )

    assert result.primary_adjustment_count == 1
    group = result.groups[0]
    assert (group.primary_op.from_su, group.primary_op.to_su) == (13, 11)
    assert group.h_after_final == 518.0
    assert not any(
        (op.from_su, op.to_su) == (23, 22)
        for op in group.h_compensation_ops
    )


def test_resource_conversion_inside_hard_window_does_not_create_nondeficit_su22() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13], H[23], H[22] = 483, 11, 0
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {
                        "11": 0,
                        "13": 1,
                        "22": 5,
                        "23": 10,
                        "24": 0,
                    },
                    "su11_donor_counts": {"12": 0, "13": 1},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
    )

    assert result.primary_adjustment_count == 1
    group = result.groups[0]
    assert (group.primary_op.from_su, group.primary_op.to_su) == (13, 11)
    assert group.h_compensation_ops == []
    assert group.h_before == 505.0
    assert group.h_after_primary == 504.0
    assert group.h_after_final == 504.0
    assert result.inventory_after[22] == 5


def test_low_h_compensation_cannot_spend_protected_su23() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13], H[23] = 483, 11
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {
                        "11": 0,
                        "13": 1,
                        "22": 5,
                        "23": 10,
                        "24": 0,
                    },
                    "su11_donor_counts": {"12": 0, "13": 1},
                    "protected_counts": {"23": 10},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
    )

    assert result.primary_adjustment_count == 1
    group = result.groups[0]
    assert (group.primary_op.from_su, group.primary_op.to_su) == (13, 11)
    assert not group.h_compensation_ops
    assert group.h_after_final == 504.0


def test_high_h_primary_inside_hard_window_does_not_add_compensation() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13], H[23], H[24] = 514, 10, 1
    plan = Stage3AdjustmentPlan(
        phase="resource_su24_ring_closure",
        planned_conversions=[
            {"from_su": 24, "to_su": 22, "reason": "ring_support_shortage"},
        ],
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {
                        "11": 0,
                        "13": 514,
                        "22": 0,
                        "23": 10,
                        "24": 1,
                    },
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
    )

    assert result.primary_adjustment_count == 1
    group = result.groups[0]
    assert (group.primary_op.from_su, group.primary_op.to_su) == (24, 22)
    assert group.h_compensation_ops == []
    assert group.h_before == 535.0
    assert group.h_after_primary == 537.0
    assert group.h_after_final == 537.0


def test_forbidden_hard_valid_primary_bundle_is_not_replayed() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13], H[23], H[24] = 514, 10, 1
    runtime = {
        "assembly_before_adjustment": {
            "resource_ledger": {
                "remaining_counts": {
                    "11": 0,
                    "13": 514,
                    "22": 0,
                    "23": 10,
                    "24": 1,
                },
            },
        },
    }
    plan = Stage3AdjustmentPlan(
        phase="resource_su24_ring_closure",
        planned_conversions=[
            {"from_su": 24, "to_su": 22, "reason": "ring_support_shortage"},
        ],
        runtime=runtime,
    )
    first = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
    )
    first_ops = [first.groups[0].primary_op] + first.groups[0].h_compensation_ops
    first_signature = _conversion_bundle_signature(first_ops)

    plan.diagnostics = {
        "forbidden_primary_conversion_signatures": [first_signature],
    }
    second = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
    )
    assert first_signature == "bundle:24>22"
    assert second.primary_adjustment_count == 0


def test_hard_valid_resource_choice_does_not_add_compensation() -> None:
    H = torch.zeros(600, dtype=torch.long)
    H[13], H[22] = 500, 20
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {
                        "11": 0,
                        "13": 1,
                        "22": 10,
                        "23": 0,
                        "24": 0,
                    },
                    "su11_donor_counts": {"12": 0, "13": 1},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([500.0, 520.0]),
        max_primary_adjustments=1,
        max_h_compensation_ops=2,
    )

    assert result.primary_adjustment_count == 1
    group = result.groups[0]
    assert (group.primary_op.from_su, group.primary_op.to_su) == (13, 11)
    assert group.h_compensation_ops == []
    assert result.inventory_after[11] == 1
    assert result.inventory_after[24] == 0


def test_primary_with_unchanged_soft_h_margin_does_not_enable_compensation() -> None:
    from model.inverse.topology.conversion_scoring import (
        ConversionContext,
        score_conversion_bundle,
    )
    from model.inverse.topology.histogram_adjustment import (
        _compensation_improves_h_state,
        _h_total,
    )

    H = torch.zeros(40, dtype=torch.long)
    H[12], H[23] = 1, 256
    context = ConversionContext(
        inventory={11: 0, 12: 1, 22: 0, 23: 1, 24: 0},
        demand={11: 1},
        current_H=H,
        target_h=520.0,
    )
    primary = HistogramConversion(12, 11, 1, "topology_primary")
    compensation = HistogramConversion(23, 22, 1, "soft_h_balance_compensation")
    primary_score = score_conversion_bundle(
        context,
        [primary],
        h_total_fn=_h_total,
    )
    compensated_score = score_conversion_bundle(
        context,
        [primary, compensation],
        h_total_fn=_h_total,
    )

    assert primary_score.h_before == 512.0
    assert primary_score.h_after == 512.0
    assert compensated_score.h_after == 513.0
    assert not _compensation_improves_h_state(
        primary_score=primary_score,
        bundle_score=compensated_score,
        target_h=520.0,
    )


def test_topology_gap_reduction_is_recorded_as_primary_not_h_compensation() -> None:
    H = torch.zeros(40, dtype=torch.long)
    H[13], H[22] = 100, 2
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 2},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_counts": {"11": 0, "22": 2},
                },
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=2,
        max_h_compensation_ops=8,
    )

    assert result.primary_adjustment_count == 2
    assert [
        (group.primary_op.from_su, group.primary_op.to_su)
        for group in result.groups
    ] == [(22, 11), (22, 11)]
    assert all(not group.h_compensation_ops for group in result.groups)


def test_high_h_su11_deficit_prefers_h_lowering_su22() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[13], H[22] = 101, 1
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 1},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {"remaining_counts": {"11": 0, "22": 1, "23": 0, "24": 0}},
            },
        },
    )

    result = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
    )

    assert (result.groups[0].primary_op.from_su, result.groups[0].primary_op.to_su) == (22, 11)
    assert result.groups[0].h_after_final == 101.0


def test_attached_empty_histogram_plan_is_not_rebuilt_without_h_policy() -> None:
    from model.shared.inverse_common import _NodeV3

    H = torch.zeros(33, dtype=torch.long)
    H[22] = 1
    node = _NodeV3(global_id=7, su_type=22)
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        planned_conversions=[{"from_su": 22, "to_su": 11, "reason": "blocked"}],
        histogram_adjustment_plan=HistogramAdjustmentPlan(
            phase="residual_resource_self_closure",
            H_before=H.detach().clone(),
            H_after=H.detach().clone(),
            blocked_reason="hydrogen_soft_direction_incompatible_with_resource_demand",
        ),
    )

    result = apply_stage3_node_adjustments([node], H, plan)

    assert result.changed is False
    assert torch.equal(result.H, H)
    assert result.nodes[0].su_type == 22


def test_atomic_via_su11_conversion_updates_node_history_as_one_bundle() -> None:
    from model.shared.inverse_common import _NodeV3

    H = torch.zeros(33, dtype=torch.long)
    H[13] = 1
    expected = H.detach().clone()
    expected[13] = 0
    expected[23] = 1
    node = _NodeV3(global_id=17, su_type=13)
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={13: 1, 23: 0},
        demand_before={23: 1},
        H_before=H.detach().clone(),
        H_after=expected,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                13,
                23,
                1,
                "atomic_aromatic_via_su11_to_su23",
                node_ids=[17],
                via_su=11,
            ),
            combined_delta={13: -1, 23: 1},
        )],
    )

    result = apply_histogram_adjustment_plan(H, histogram_plan, nodes=[node])

    assert result.changed is True
    assert result.nodes[0].su_type == 23
    assert result.nodes[0].stage3_conversion_history == [
        {"from_su": 13, "to_su": 11, "reason": "atomic_aromatic_via_su11_to_su23"},
        {"from_su": 11, "to_su": 23, "reason": "atomic_aromatic_via_su11_to_su23"},
    ]


def test_parity_balance_rejects_non_12_13_conversion_contract() -> None:
    from model.shared.inverse_common import _NodeV3

    H = torch.zeros(33, dtype=torch.long)
    H[11] = 1
    node = _NodeV3(global_id=9, su_type=11)
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_side_chain_closure",
        inventory_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H.detach().clone(),
        parity_balance_op=HistogramConversion(
            11,
            12,
            1,
            "invalid_test_parity_conversion",
            node_ids=[9],
        ),
    )

    result = apply_histogram_adjustment_plan(H, histogram_plan, nodes=[node])

    assert result.changed is False
    assert result.nodes[0].su_type == 11
    assert result.rejected_ops[0]["reason"] == "aggregate_resource_preflight_failed"
    assert result.rejected_ops[0]["accounting"]["errors"][0]["reason"] == (
        "invalid_parity_balance_contract"
    )
