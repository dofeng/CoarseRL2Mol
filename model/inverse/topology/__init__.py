"""Public Stage3 topology classification and assembly API."""

from .classification import classify_topology_units
from .resources.conversion_solver import apply_stage3_node_adjustments
from .stage3.phase_eval import assemble_topology_units
from .stage3_models import (
    Stage3AdjustmentApplyResult,
    Stage3AdjustmentPlan,
    Stage3ProgressPhase,
    Stage3WorkflowCycleRecord,
    Stage3WorkflowResult,
)
from .stage3.workflow import audit_stage3_topology, plan_stage3_resource_adjustments

__all__ = [
    "Stage3AdjustmentApplyResult",
    "Stage3AdjustmentPlan",
    "Stage3ProgressPhase",
    "Stage3WorkflowCycleRecord",
    "Stage3WorkflowResult",
    "apply_stage3_node_adjustments",
    "assemble_topology_units",
    "audit_stage3_topology",
    "classify_topology_units",
    "plan_stage3_resource_adjustments",
]
