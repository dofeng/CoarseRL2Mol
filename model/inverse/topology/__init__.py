from .classification import (
    ClassificationToken,
    NeighborToken,
    PoolClaim,
    TopologyClassificationResult,
    TopologyFragment,
    TopologyRef,
    build_node_token,
    build_node_tokens,
    classify_topology_units,
    normalize_subtype,
)
from .rl_mtcs_export import build_rl_mtcs_payload

__all__ = [
    "TopologyClassificationResult",
    "TopologyRef",
    "PoolClaim",
    "TopologyFragment",
    "ClassificationToken",
    "NeighborToken",
    "build_node_token",
    "build_node_tokens",
    "normalize_subtype",
    "classify_topology_units",
    "build_rl_mtcs_payload",
]

try:
    from .conversion_solver import apply_stage3_node_adjustments
    from .phase_eval import assemble_topology_units, build_stage3_phase_evaluation
    from .stage3_models import (
        Stage3AdjustmentBundle,
        Stage3AdjustmentApplyResult,
        Stage3AdjustmentOp,
        Stage3AdjustmentPlan,
        Stage3FlexDecision,
        Stage3PhaseEvaluation,
        Stage3ProgressPhase,
        Stage3WorkflowCycleRecord,
        Stage3WorkflowResult,
        TopologyAssemblyResult,
    )
    from .workflow import audit_stage3_topology, plan_stage3_resource_adjustments
except ModuleNotFoundError as exc:
    if str(exc).strip() != "No module named 'torch'":
        raise
else:
    __all__.extend([
        "TopologyAssemblyResult",
        "Stage3PhaseEvaluation",
        "Stage3ProgressPhase",
        "Stage3AdjustmentOp",
        "Stage3AdjustmentBundle",
        "Stage3AdjustmentApplyResult",
        "Stage3AdjustmentPlan",
        "Stage3FlexDecision",
        "Stage3WorkflowCycleRecord",
        "Stage3WorkflowResult",
        "assemble_topology_units",
        "build_stage3_phase_evaluation",
        "plan_stage3_resource_adjustments",
        "audit_stage3_topology",
        "apply_stage3_node_adjustments",
    ])
