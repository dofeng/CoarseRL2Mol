from __future__ import annotations

from typing import Any, Dict, Optional

from .histogram_adjustment import build_histogram_adjustment_plan
from .phase_eval import build_stage3_phase_evaluation, classify_topology_units
from .resource_accounting import build_resource_accounting_snapshot
from .stage3_models import Stage3AdjustmentPlan
from .stage3_state_machine import plan_stage3_state_machine


def _audit_claimed_ids_by_su(products: Any) -> Dict[int, list[int]]:
    claimed: Dict[int, list[int]] = {}
    for raw in list(products or []):
        product = dict(raw or {})
        for key, values in dict(
            product.get("claimed_resource_ids_by_su", {})
            or product.get("claimed_resource_ids", {})
            or {}
        ).items():
            if not str(key).lstrip("-").isdigit():
                continue
            claimed.setdefault(int(key), []).extend(
                int(value) for value in list(values or [])
            )
    return {int(key): sorted(values) for key, values in claimed.items()}


def _audit_ownership_view(ledger: Dict[str, Any]) -> Dict[str, Any]:
    state_names = (
        "fixed_topology_core",
        "pending_topology",
        "completed_topology",
        "claimed_closure_resource",
        "remaining_closure_resource",
        "remaining_aromatic_donor",
        "support_context",
        "unclassified_or_conflict",
    )
    states = {str(state): [] for state in state_names}
    for raw in dict(ledger.get("ownership_by_node_id", {}) or {}).values():
        owner = dict(raw or {})
        state = str(owner.get("state", "unclassified_or_conflict") or "unclassified_or_conflict")
        states.setdefault(state, []).append(int(owner.get("node_id", -1)))
    states = {
        str(key): sorted(set(int(value) for value in values))
        for key, values in states.items()
    }
    available = {
        int(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    return {
        "states": states,
        "closure_pool_ids_by_su": {
            str(su_type): list(available.get(int(su_type), []))
            for su_type in (11, 22, 23, 24)
        },
        "aromatic_donor_ids_by_su": {
            str(su_type): list(available.get(int(su_type), []))
            for su_type in (10, 12, 13)
        },
        "invariant_errors": [
            dict(item or {}) for item in list(ledger.get("conflicts", []) or [])
        ],
    }


def _audit_resource_ledger(
    accounting: Dict[str, Any],
    products: Any,
) -> Dict[str, Any]:
    initial = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(accounting.get("aggregate_ids_before_by_su", {}) or {}).items()
    }
    remaining = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(accounting.get("final_remaining_ids_by_su", {}) or {}).items()
    }
    consumed = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(accounting.get("committed_claimed_ids_by_su", {}) or {}).items()
    }
    residual_claimed: Dict[str, set[int]] = {}
    for raw in list(products or []):
        product = dict(raw or {})
        role = str(product.get("product_role", product.get("product_type", "")) or "")
        if not role.startswith("residual_"):
            continue
        for key, values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).items():
            residual_claimed.setdefault(str(key), set()).update(
                int(value) for value in list(values or [])
            )
    keys = sorted(
        set(initial) | set(remaining) | set(consumed),
        key=lambda value: (0, int(value)) if str(value).lstrip("-").isdigit() else (1, str(value)),
    )
    residual = {
        str(key): sorted(
            set(consumed.get(str(key), [])) & residual_claimed.get(str(key), set())
        )
        for key in keys
    }
    topology = {
        str(key): sorted(set(consumed.get(str(key), [])) - set(residual.get(str(key), [])))
        for key in keys
    }
    return {
        "initial_ids_by_su": {str(key): list(initial.get(str(key), [])) for key in keys},
        "consumed_ids_by_su": {str(key): list(consumed.get(str(key), [])) for key in keys},
        "remaining_ids_by_su": {str(key): list(remaining.get(str(key), [])) for key in keys},
        "initial_counts": {str(key): len(initial.get(str(key), [])) for key in keys},
        "consumed_counts": {str(key): len(consumed.get(str(key), [])) for key in keys},
        "topology_consumed_counts": {str(key): len(topology.get(str(key), [])) for key in keys},
        "residual_consumed_counts": {str(key): len(residual.get(str(key), [])) for key in keys},
        "remaining_counts": {str(key): len(remaining.get(str(key), [])) for key in keys},
        "shortage_counts": {},
        "invariant_errors": [
            dict(item or {}) for item in list(accounting.get("errors", []) or [])
        ],
        "snapshot_stage": "fresh_stage3_audit",
    }


def plan_stage3_resource_adjustments(
    H: Any,
    nodes: Any,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    max_primary_adjustments_per_stage: int = 10,
    max_h_compensation_ops_per_adjustment: int = 8,
    h_tolerance: float = 0.08,
    residual_flex_preferred_23: int = 3,
    residual_flex_min_23: int = 2,
    residual_side_preferred_23: int = 2,
    residual_side_min_23: int = 1,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
    generation_id: int = 0,
    difference_spectrum: Optional[Dict[str, Any]] = None,
    enable_post_external_spectrum: bool = True,
    post_external_resume_phase: str = "remaining_pool_spectrum_rebalance",
    spectrum_candidate_limit: int = 2,
    spectrum_max_primary_adjustments: int = 10,
    rejected_spectrum_histogram_signatures: Optional[Any] = None,
    forbidden_spectrum_inverse_signatures: Optional[Any] = None,
    forbidden_resource_conversion_signatures: Optional[Any] = None,
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
        generation_id=int(generation_id),
    )
    plan = plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=H,
        nodes=list(nodes or []),
        classification=classification,
        S_target=S_target,
        E_target=E_target,
        difference_spectrum=difference_spectrum,
        enable_post_external_spectrum=enable_post_external_spectrum,
        post_external_resume_phase=str(post_external_resume_phase),
        spectrum_candidate_limit=spectrum_candidate_limit,
        spectrum_max_primary_adjustments=spectrum_max_primary_adjustments,
        rejected_spectrum_histogram_signatures=rejected_spectrum_histogram_signatures,
        forbidden_spectrum_inverse_signatures=forbidden_spectrum_inverse_signatures,
        forbidden_resource_conversion_signatures=forbidden_resource_conversion_signatures,
        max_primary_adjustments_per_stage=max_primary_adjustments_per_stage,
        max_h_compensation_ops_per_adjustment=max_h_compensation_ops_per_adjustment,
        h_tolerance=h_tolerance,
        residual_flex_preferred_23=residual_flex_preferred_23,
        residual_flex_min_23=residual_flex_min_23,
        residual_side_preferred_23=residual_side_preferred_23,
        residual_side_min_23=residual_side_min_23,
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


def audit_stage3_topology(
    *,
    H: Any,
    nodes: Any,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    allow_conversions: bool = False,
    enable_spectrum_rebalance: bool = False,
    generation_id: int = 0,
    h_tolerance: float = 0.08,
    residual_flex_preferred_23: int = 3,
    residual_flex_min_23: int = 2,
    residual_side_preferred_23: int = 2,
    residual_side_min_23: int = 1,
) -> Dict[str, Any]:
    """Run one read-only Stage3 pass from the supplied current graph state."""
    max_primary = 10 if bool(allow_conversions) else 0
    plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=list(nodes or []),
        fixed_partition_meta=fixed_partition_meta,
        S_target=S_target,
        E_target=E_target,
        max_primary_adjustments_per_stage=max_primary,
        max_h_compensation_ops_per_adjustment=(8 if bool(allow_conversions) else 0),
        h_tolerance=h_tolerance,
        residual_flex_preferred_23=residual_flex_preferred_23,
        residual_flex_min_23=residual_flex_min_23,
        residual_side_preferred_23=residual_side_preferred_23,
        residual_side_min_23=residual_side_min_23,
        generation_id=int(generation_id),
        enable_post_external_spectrum=bool(enable_spectrum_rebalance),
        post_external_resume_phase="remaining_pool_spectrum_rebalance",
    )
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    phase_eval_dict = (
        phase_eval.to_dict()
        if phase_eval is not None and hasattr(phase_eval, "to_dict")
        else dict(runtime.get("phase_evaluation_dict", {}) or {})
    )
    classification_obj = runtime.get("classification_before_adjustment")
    classification = (
        classification_obj.to_dict()
        if classification_obj is not None and hasattr(classification_obj, "to_dict")
        else dict(runtime.get("classification_before_adjustment_dict", {}) or {})
    )
    if not classification:
        classification = dict(phase_eval_dict.get("classification_view", {}) or {})
    assembly_snapshot = dict(
        runtime.get("assembly_before_adjustment_dict", {})
        or phase_eval_dict.get("assembly_view", {})
        or {}
    )
    ledger_obj = runtime.get("iteration_topology_ledger")
    ledger = (
        ledger_obj.to_dict()
        if ledger_obj is not None and hasattr(ledger_obj, "to_dict")
        else dict(ledger_obj or {})
    )
    products = [
        dict(item or {})
        for item in list(ledger.get("completed_products", []) or [])
        if not bool(dict(item or {}).get("provisional", False))
        and bool(dict(item or {}).get("closed", False))
    ]
    if not products:
        products = [
            dict(item or {})
            for item in list(assembly_snapshot.get("products", []) or [])
        ]

    assembly_pending = [
        dict(item or {}) for item in list(assembly_snapshot.get("pending_units", []) or [])
    ]
    pending_contracts = [
        dict(item or {}) for item in assembly_pending
    ] or [
        dict(item or {}) for item in list(ledger.get("pending_contracts", []) or [])
    ] or [
        dict(item or {}) for item in list(getattr(plan, "pending_contracts", []) or [])
    ]
    pending_contracts = [
        {
            **dict(item or {}),
            "real": str(
                dict(item or {}).get("real", "")
                or dict(item or {}).get("real_topology_partial", "")
                or dict(item or {}).get("expected_topology", "")
                or ""
            ),
            "real_gid": str(
                dict(item or {}).get("real_gid", "")
                or dict(item or {}).get("real_node_partial", "")
                or ""
            ),
            "source_node_ids": [
                int(value)
                for value in list(
                    dict(item or {}).get("source_node_ids", [])
                    or dict(item or {}).get("claimed_source_node_ids", [])
                    or []
                )
            ],
        }
        for item in pending_contracts
    ]

    resource_pool = dict(classification.get("resource_pool", {}) or {})
    aggregate_id_sets: Dict[int, set[int]] = {}
    for field in ("initial_ids_by_su", "su11_donor_ids_by_su"):
        for key, values in dict(resource_pool.get(field, {}) or {}).items():
            if not str(key).lstrip("-").isdigit():
                continue
            aggregate_id_sets.setdefault(int(key), set()).update(
                int(value) for value in list(values or [])
            )
    aggregate_ids = {
        int(key): sorted(values) for key, values in aggregate_id_sets.items()
    }
    assembly_resource_ledger = dict(assembly_snapshot.get("resource_ledger", {}) or {})
    remaining_ids = {
        int(key): [int(value) for value in list(values or [])]
        for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    if not remaining_ids:
        remaining_ids = {
            int(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(
                assembly_resource_ledger.get("remaining_ids_by_su", {}) or {}
            ).items()
            if str(key).lstrip("-").isdigit()
        }
    claimed_ids = _audit_claimed_ids_by_su(products)
    histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
    groups = list(getattr(histogram_plan, "groups", []) or [])
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_ids,
        authorized_ids_before_by_su=aggregate_ids,
        final_remaining_ids_by_su=remaining_ids,
        committed_claimed_ids_by_su=claimed_ids,
        groups=groups,
    ) if (aggregate_ids or remaining_ids or claimed_ids) else {
        "ok": False,
        "errors": [],
        "skipped": True,
        "reason": "fresh_audit_resource_snapshot_unavailable",
    }
    phase_result = getattr(plan, "phase_result", None)
    proof = dict(
        getattr(phase_result, "infeasibility_proof", {})
        or dict(getattr(plan, "diagnostics", {}) or {}).get("infeasibility_proof", {})
        or {}
    )
    physical_transactions = [
        transaction
        for transaction in list(getattr(phase_result, "transactions", []) or [])
        if (
            list(getattr(transaction, "converted_node_ids", []) or [])
            or list(getattr(transaction, "conversion_path", []) or [])
            or list(getattr(transaction, "compensation_node_ids", []) or [])
        )
    ]
    requires_conversion = bool(
        groups
        or physical_transactions
        or list(getattr(plan, "planned_conversions", []) or [])
        or bool(getattr(phase_result, "rebuild_required", False))
        or str(getattr(plan, "reason", "") or "") == "stage3_rebuild_required"
    )
    pending_resource_shortage = any(
        int(value) > 0
        for contract in pending_contracts
        for value in dict(
            dict(contract or {}).get("infeasibility_proof", {}).get(
                "missing_counts", {}
            )
            or dict(contract or {}).get("missing_counts", {})
            or {}
        ).values()
    )
    if not bool(allow_conversions) and pending_resource_shortage:
        requires_conversion = True
    proof_reason = str(proof.get("reason", "") or "")
    if (
        not bool(allow_conversions)
        and str(getattr(plan, "reason", "") or "") != "stage3_complete"
        and (
            "conversion" in proof_reason
            or "terminalization" in proof_reason
            or any(
                dict(item or {}).get("unresolved_counts")
                for item in list(proof.get("residual_candidate_attempts", []) or [])
            )
        )
    ):
        requires_conversion = True
    ownership = _audit_ownership_view(ledger)
    if not ledger_obj:
        assembly_ownership = dict(
            assembly_snapshot.get("ownership_ledger", {})
            or dict(assembly_snapshot.get("diagnostics", {}) or {}).get(
                "ownership_ledger", {}
            )
            or {}
        )
        if assembly_ownership:
            ownership = {
                "states": {
                    str(key): sorted({int(value) for value in list(values or [])})
                    for key, values in dict(assembly_ownership.get("states", {}) or {}).items()
                },
                "closure_pool_ids_by_su": {
                    str(key): sorted({int(value) for value in list(values or [])})
                    for key, values in dict(
                        assembly_ownership.get("closure_pool_ids_by_su", {}) or {}
                    ).items()
                },
                "aromatic_donor_ids_by_su": {
                    str(key): sorted({int(value) for value in list(values or [])})
                    for key, values in dict(
                        assembly_ownership.get("aromatic_donor_ids_by_su", {}) or {}
                    ).items()
                },
                "invariant_errors": [
                    dict(item or {})
                    for item in list(assembly_ownership.get("invariant_errors", []) or [])
                ],
            }
    ownership_conflicts = list(ownership.get("invariant_errors", []) or [])
    unclassified_ids = list(
        dict(ownership.get("states", {}) or {}).get("unclassified_or_conflict", [])
        or []
    )
    complete = bool(
        str(getattr(plan, "reason", "") or "") == "stage3_complete"
        and not requires_conversion
        and not pending_contracts
        and not ownership_conflicts
        and not unclassified_ids
        and bool(accounting.get("ok", False))
    )
    reason = "stage3_complete" if complete else str(
        getattr(plan, "reason", "") or "fresh_stage3_audit_failed"
    )
    if not complete and bool(accounting.get("errors", [])):
        reason = "fresh_audit_resource_accounting_failed"
    elif not complete and requires_conversion and not bool(allow_conversions):
        reason = "fresh_audit_requires_conversion"
    return {
        "ok": bool(complete),
        "reason": str(reason),
        "allow_conversions": bool(allow_conversions),
        "enable_spectrum_rebalance": bool(enable_spectrum_rebalance),
        "requires_conversion": bool(requires_conversion),
        "products": products,
        "pending_contracts": pending_contracts,
        "resource_shortage": {
            str(key): int(value)
            for key, value in dict(getattr(plan, "unresolved_counts", {}) or {}).items()
            if int(value) > 0
        },
        "classification": classification,
        "phase_evaluation": phase_eval_dict,
        "external_assembly": dict(phase_eval_dict.get("assembly_view", {}) or {}),
        "ownership_ledger": ownership,
        "iteration_topology_ledger": ledger,
        "resource_ledger": _audit_resource_ledger(accounting, products),
        "resource_accounting": accounting,
        "infeasibility_proof": proof,
        "plan": plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {}),
    }
