from __future__ import annotations

from typing import Dict, List

import torch

from ....shared.inverse_common import _NodeV3
from ..stage3_models import Stage3AdjustmentApplyResult, Stage3AdjustmentPlan


def apply_stage3_node_adjustments(
    nodes: List[_NodeV3],
    H: torch.Tensor,
    plan: Stage3AdjustmentPlan,
) -> Stage3AdjustmentApplyResult:
    if not isinstance(plan, Stage3AdjustmentPlan):
        raise TypeError("plan must be a Stage3AdjustmentPlan")
    requested_counts: Dict[str, int] = {
        str(key): int(value)
        for key, value in dict(plan.requested_counts or {}).items()
        if int(value) > 0
    }
    unresolved_counts: Dict[str, int] = {
        str(key): int(value)
        for key, value in dict(plan.unresolved_counts or {}).items()
        if int(value) > 0
    }
    from .histogram import build_histogram_adjustment_plan

    histogram_plan = plan.histogram_adjustment_plan
    if histogram_plan is None:
        histogram_plan = build_histogram_adjustment_plan(
            plan=plan,
            H=H,
            max_primary_adjustments=max(1, len(list(plan.planned_conversions or []))),
        )
    from .histogram import apply_histogram_adjustment_plan

    result = apply_histogram_adjustment_plan(H=H, histogram_plan=histogram_plan, nodes=nodes)
    result.request_ops = []
    result.requested_counts = dict(requested_counts or result.requested_counts or {})
    result.unresolved_counts = dict(unresolved_counts or result.unresolved_counts or {})
    return result
