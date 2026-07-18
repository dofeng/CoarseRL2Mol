from __future__ import annotations

from typing import Any, Dict, Optional

from .histogram_adjustment import build_histogram_adjustment_plan
from .phase_eval import build_stage3_phase_evaluation, classify_topology_units
from .stage3_models import Stage3AdjustmentPlan
from .stage3_state_machine import plan_stage3_state_machine


def plan_stage3_resource_adjustments(
    H: Any,
    nodes: Any,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    max_primary_adjustments_per_stage: int = 10,
    max_h_compensation_ops_per_adjustment: int = 8,
    h_tolerance: float = 0.06,
    chain_23_limit: int = 6,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
) -> Stage3AdjustmentPlan:
    classification = classify_topology_units(
        list(nodes or []),
        fixed_partition_meta=fixed_partition_meta,
    )
    phase_eval = build_stage3_phase_evaluation(
        nodes=list(nodes or []),
        classification=classification,
        fixed_partition_meta=fixed_partition_meta,
        S_target=S_target,
        E_target=E_target,
    )
    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=H,
        nodes=list(nodes or []),
        E_target=E_target,
        max_primary_adjustments_per_stage=max_primary_adjustments_per_stage,
        max_h_compensation_ops_per_adjustment=max_h_compensation_ops_per_adjustment,
        h_tolerance=h_tolerance,
        chain_23_limit=chain_23_limit,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    plan.runtime.setdefault("classification_before_adjustment", classification)
    plan.runtime.setdefault("classification_before_adjustment_dict", classification.to_dict())
    plan.runtime.setdefault("phase_evaluation", phase_eval)
    plan.runtime.setdefault("phase_evaluation_dict", phase_eval.to_dict())
    if plan.histogram_adjustment_plan is None and list(plan.planned_conversions or []):
        plan.histogram_adjustment_plan = build_histogram_adjustment_plan(
            plan=plan,
            H=H,
            E_target=E_target,
            max_primary_adjustments=max_primary_adjustments_per_stage,
            max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
        )
    if plan.histogram_adjustment_plan is not None:
        plan.diagnostics["su11_rotation_index_after"] = int(
            plan.histogram_adjustment_plan.su11_rotation_index_after
        )
    return plan
