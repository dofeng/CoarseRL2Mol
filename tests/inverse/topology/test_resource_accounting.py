from __future__ import annotations

import torch

from model.inverse.topology.histogram_adjustment import apply_histogram_adjustment_plan
from model.inverse.topology.resource_accounting import (
    build_resource_accounting_snapshot,
)
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
)
from model.shared.inverse_common import _NodeV3


def _conversion_group(node_id: int = 2) -> AtomicAdjustmentGroup:
    return AtomicAdjustmentGroup(
        primary_op=HistogramConversion(
            from_su=23,
            to_su=11,
            count=1,
            reason="test_accounting",
            node_ids=[node_id],
        ),
        combined_delta={23: -1, 11: 1},
    )


def test_resource_accounting_partitions_remaining_and_committed_claims() -> None:
    snapshot = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={11: [1], 23: [2], 22: [3]},
        authorized_ids_before_by_su={23: [2], 22: [3]},
        final_remaining_ids_by_su={11: [2], 22: [3]},
        committed_claimed_ids_by_su={11: [1]},
        groups=[_conversion_group()],
    )

    assert snapshot["ok"] is True
    assert snapshot["aggregate_inventory_before"] == {11: 1, 23: 1, 22: 1}
    assert snapshot["aggregate_inventory_after"] == {11: 2, 22: 1}
    assert snapshot["authorized_inventory_before"] == {23: 1, 22: 1}
    assert snapshot["authorized_inventory_after"] == {11: 1, 22: 1}
    assert snapshot["aggregate_required_counts"] == {11: 1}


def test_resource_accounting_rejects_duplicate_product_owner() -> None:
    snapshot = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={11: [1]},
        authorized_ids_before_by_su={},
        final_remaining_ids_by_su={},
        committed_claimed_ids_by_su={11: [1, 1]},
        groups=[],
    )

    assert snapshot["ok"] is False
    assert any(
        item["reason"] == "resource_claimed_by_multiple_products"
        for item in snapshot["errors"]
    )


def test_executor_accepts_zero_omission_in_aggregate_inventory() -> None:
    node = _NodeV3(global_id=2, su_type=23)
    H = torch.zeros(40, dtype=torch.long)
    H[23] = 1
    H_after = H.detach().clone()
    H_after[23] = 0
    H_after[11] = 1
    group = _conversion_group()
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={23: [2]},
        authorized_ids_before_by_su={23: [2]},
        final_remaining_ids_by_su={11: [2]},
        committed_claimed_ids_by_su={},
        groups=[group],
    )
    plan = HistogramAdjustmentPlan(
        groups=[group],
        H_before=H,
        H_after=H_after,
        inventory_before={23: 1},
        inventory_after={11: 1},
        aggregate_inventory_before={23: 1},
        aggregate_inventory_after={11: 1},
        accounting_diagnostics=accounting,
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes=[node])

    assert applied.changed is True
    assert applied.rejected_ops == []
    assert applied.nodes[0].su_type == 11


def test_executor_rejects_planning_accounting_error() -> None:
    H = torch.zeros(40, dtype=torch.long)
    plan = HistogramAdjustmentPlan(
        H_before=H,
        H_after=H,
        aggregate_inventory_before={11: 1},
        aggregate_inventory_after={11: 1},
        accounting_diagnostics={
            "ok": False,
            "errors": [{"reason": "resource_claimed_by_multiple_products"}],
        },
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes=[])

    assert applied.changed is False
    assert applied.rejected_ops[0]["reason"] == "aggregate_resource_preflight_failed"
