import torch

from model.inverse.topology.histogram_adjustment import (
    apply_histogram_adjustment_plan,
    build_histogram_adjustment_plan,
)
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
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
        phase="external_structural_closure",
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
        planned_conversions=[{"from_su": 24, "to_su": 23, "reason": "residual"}],
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


def test_high_h_state_rejects_explicit_conversion_that_raises_h_further() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[11], H[22], H[23], H[24] = 1, 10, 1, 3
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
    target_elements = torch.tensor([0.0, 30.0])

    histogram_plan = build_histogram_adjustment_plan(
        plan,
        H,
        E_target=target_elements,
        max_primary_adjustments=1,
        max_h_compensation_ops=1,
        h_tolerance=0.11,
    )

    assert histogram_plan.primary_adjustment_count == 0
    assert histogram_plan.blocked_reason == "hydrogen_soft_direction_incompatible_with_resource_demand"


def test_explicit_primary_cannot_consume_phase_required_resource() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23] = 1
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
        phase="external_structural_closure",
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


def test_missing_surplus_h_compensation_reports_tolerance_requirement() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23], H[24] = 1, 1
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
    assert histogram_plan.blocked_reason == "hydrogen_soft_direction_incompatible_with_resource_demand"


def test_primary_conversion_cannot_push_h_below_tolerance_without_compensation() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22], H[23] = 1, 1
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
    assert histogram_plan.blocked_reason == "hydrogen_soft_direction_incompatible_with_resource_demand"


def test_hard_demand_precedes_preferred_demand() -> None:
    import torch
    from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
    from model.inverse.topology.histogram_adjustment import build_histogram_adjustment_plan

    H = torch.zeros(33, dtype=torch.long)
    H[22] = 2
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
        phase="external_structural_closure",
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
    assert 23 in targets[2:]


def test_rejected_deterministic_primary_does_not_loop_forever(monkeypatch) -> None:
    from model.inverse.topology import histogram_adjustment as module

    H = torch.zeros(33, dtype=torch.long)
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
        "_choose_from_gap",
        lambda H_work, inventory, demand, aromatic_cursor, **kwargs: (
            module.HistogramConversion(22, 11, 1, "deterministic_invalid"),
            aromatic_cursor,
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
        phase="external_structural_closure",
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
        phase="external_structural_closure",
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


def test_low_hydrogen_state_blocks_conversion_that_would_lower_h_further() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 1
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
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
        E_target=torch.tensor([0.0, 100.0]),
        max_primary_adjustments=1,
        h_tolerance=0.06,
    )

    assert result.primary_adjustment_count == 0
    assert result.blocked_reason == "hydrogen_soft_direction_incompatible_with_resource_demand"


def test_histogram_su11_rotation_uses_aromatic_sources_at_h_window_boundaries() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 2, 13: 1, 22: 9, 23: 9, 24: 9}
    demand = {11: 1, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22], H[23], H[24] = 2, 1, 9, 9, 9

    lower_op, _ = module._choose_from_gap(H, inventory, demand, 0, current_h=90.0, target_h=100.0)
    upper_op, _ = module._choose_from_gap(H, inventory, demand, 0, current_h=103.0, target_h=100.0)

    assert (lower_op.from_su, lower_op.to_su) == (12, 11)
    assert (upper_op.from_su, upper_op.to_su) == (13, 11)


def test_histogram_h_window_rotation_advances_one_step_per_choice() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 2, 13: 2, 22: 0, 23: 0, 24: 0}
    demand = {11: 1, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 2
    H[13] = 2

    first, cursor = module._choose_from_gap(
        H, inventory, demand, 0, current_h=100.0, target_h=100.0
    )
    second, cursor = module._choose_from_gap(
        H, inventory, demand, cursor, current_h=100.0, target_h=100.0
    )

    assert (first.from_su, first.to_su) == (13, 11)
    assert (second.from_su, second.to_su) == (13, 11)
    assert cursor == 2


def test_histogram_su11_aromatic_rotation_repeats_three_step_cycle() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 1, 13: 4, 22: 0, 23: 0, 24: 0}
    demand = {11: 5, 22: 0, 23: 0, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 1
    H[13] = 4
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

    assert selected_sources == [13, 13, 12, 13, 13]
    assert cursor == 5


def test_residual_su11_to_aromatic_uses_same_three_step_cycle() -> None:
    from model.inverse.topology.histogram_adjustment import next_su11_aromatic_target

    cursor = 0
    targets = []
    for _ in range(5):
        target, cursor = next_su11_aromatic_target(cursor)
        targets.append(target)

    assert targets == [13, 13, 12, 13, 13]
    assert cursor == 5


def test_histogram_non_su11_target_keeps_surplus_donor_policy() -> None:
    from model.inverse.topology import histogram_adjustment as module

    inventory = {11: 0, 12: 3, 13: 3, 22: 5, 23: 0, 24: 1}
    demand = {11: 0, 22: 0, 23: 1, 24: 0}
    H = torch.zeros(33, dtype=torch.long)
    H[12], H[13], H[22], H[24] = 3, 3, 5, 1

    op, _ = module._choose_from_gap(H, inventory, demand, 0, current_h=95.0, target_h=100.0)

    assert op.to_su == 23
    assert op.from_su in {22, 24}


def test_su11_rotation_advances_only_after_successful_real_adjustment() -> None:
    from types import SimpleNamespace

    from model.inverse.layer4.inverse_layer4 import Layer4Adjuster

    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    adjuster._stage3_su11_rotation_index = 0
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 1
    node = SimpleNamespace(global_id=101, su_type=12)
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_structural_closure",
        inventory_before={12: 1},
        demand_before={11: 1},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "su11_rotation", node_ids=[101]),
            combined_delta={12: -1, 11: 1},
        )],
    )
    plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
        histogram_adjustment_plan=histogram_plan,
        diagnostics={"su11_rotation_index_after": 1},
    )

    result = adjuster.apply_topology_resource_adjustments([node], H, plan)

    assert result.changed is True
    assert adjuster._stage3_su11_rotation_index == 1

    rejected_plan = Stage3AdjustmentPlan(
        phase="external_structural_closure",
        histogram_adjustment_plan=HistogramAdjustmentPlan(
            phase="external_structural_closure",
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
        phase="external_structural_closure",
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


def test_low_h_replans_explicit_su22_to_su11_with_su12_donor() -> None:
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
    assert (result.groups[0].primary_op.from_su, result.groups[0].primary_op.to_su) == (12, 11)
    assert result.groups[0].h_after_final == result.groups[0].h_before


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
    assert (result.groups[0].primary_op.from_su, result.groups[0].primary_op.to_su) == (12, 23)
    assert result.groups[0].primary_op.via_su == 11
    assert (result.groups[1].primary_op.from_su, result.groups[1].primary_op.to_su) == (12, 11)
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
