from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List

import torch

from ...shared.inverse_common import _NodeV3
from .stage3_models import Stage3AdjustmentApplyResult, Stage3AdjustmentPlan

def _normalize_plan(plan: Any) -> Stage3AdjustmentPlan:
    if isinstance(plan, Stage3AdjustmentPlan):
        return plan
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()
    plan_dict = dict(plan or {})
    field_names = set(Stage3AdjustmentPlan.__dataclass_fields__)
    kwargs = {key: value for key, value in plan_dict.items() if key in field_names}
    kwargs["resource_deficit"] = {
        int(key): int(value)
        for key, value in dict(plan_dict.get("resource_deficit", {}) or {}).items()
    }
    kwargs["resource_surplus"] = {
        int(key): int(value)
        for key, value in dict(plan_dict.get("resource_surplus", {}) or {}).items()
    }
    return Stage3AdjustmentPlan(**kwargs)


def _histogram_counts(nodes: List[_NodeV3], size: int) -> Dict[str, int]:
    counts = Counter()
    for node in list(nodes or []):
        try:
            su_type = int(getattr(node, "su_type", -1))
        except Exception:
            continue
        if 0 <= su_type < size:
            counts[str(su_type)] += 1
    return {str(key): int(value) for key, value in sorted(counts.items())}


def _rebuild_histogram_from_nodes(nodes: List[_NodeV3], H_template: torch.Tensor) -> torch.Tensor:
    H_out = torch.zeros_like(torch.clamp(H_template.detach().clone(), min=0).long())
    for su_key, count in _histogram_counts(nodes, int(H_out.numel())).items():
        H_out[int(su_key)] = int(count)
    return H_out


def _conservation_summary(nodes: List[_NodeV3], H: torch.Tensor) -> Dict[str, Any]:
    node_counts = _histogram_counts(nodes, int(H.numel()))
    histogram_counts = {
        str(index): int(H[index].item())
        for index in range(int(H.numel()))
        if int(H[index].item()) != 0
    }
    return {
        "node_count": int(len(list(nodes or []))),
        "histogram_total": int(torch.clamp(H, min=0).long().sum().item()),
        "node_counts": node_counts,
        "histogram_counts": histogram_counts,
        "consistent": node_counts == histogram_counts,
    }


def apply_stage3_node_adjustments(
    nodes: List[_NodeV3],
    H: torch.Tensor,
    plan: Any,
) -> Stage3AdjustmentApplyResult:
    plan_obj = _normalize_plan(plan)
    requested_counts: Dict[str, int] = {
        str(key): int(value)
        for key, value in dict(plan_obj.requested_counts or {}).items()
        if int(value) > 0
    }
    unresolved_counts: Dict[str, int] = {
        str(key): int(value)
        for key, value in dict(plan_obj.unresolved_counts or {}).items()
        if int(value) > 0
    }
    from .histogram_adjustment import build_histogram_adjustment_plan

    histogram_plan = getattr(plan_obj, "histogram_adjustment_plan", None)
    if histogram_plan is None:
        histogram_plan = build_histogram_adjustment_plan(
            plan=plan_obj,
            H=H,
            max_primary_adjustments=max(1, len(list(plan_obj.planned_conversions or []))),
        )
    from .histogram_adjustment import apply_histogram_adjustment_plan

    result = apply_histogram_adjustment_plan(H=H, histogram_plan=histogram_plan, nodes=nodes)
    result.request_ops = []
    result.requested_counts = dict(requested_counts or result.requested_counts or {})
    result.unresolved_counts = dict(unresolved_counts or result.unresolved_counts or {})
    return result
