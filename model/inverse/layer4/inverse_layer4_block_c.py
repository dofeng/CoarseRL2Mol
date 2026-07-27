"""Authoritative Layer4 Block-C Stage3 workflow loop.

Detailed topology rules live under ``model.inverse.topology``.
"""

from copy import deepcopy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..topology import Stage3WorkflowCycleRecord, Stage3WorkflowResult
from ..topology.histogram_adjustment import _h_total
from ..topology.resource_profiles import h_soft_bounds
from ..topology.spectrum_rebalance import (
    apply_spectrum_candidate,
    spectrum_candidate_balance_penalty,
    spectrum_window_loss,
)


def _stage3_metrics_from_plan(plan: Any) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    if phase_eval is not None:
        return dict(getattr(phase_eval, "topology_metrics", {}) or {})
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    return dict(phase_eval_dict.get("topology_metrics", {}) or {})


def _stage3_signature_from_plan(plan: Any) -> str:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    if phase_eval is not None:
        return str(getattr(phase_eval, "topology_signature", "") or "")
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    return str(phase_eval_dict.get("topology_signature", "") or "")


def _stage3_flex_audit_from_plan(plan: Any) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    if phase_eval is not None:
        return dict(getattr(phase_eval, "flex_audit", {}) or {})
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    return dict(phase_eval_dict.get("flex_audit", {}) or {})


def _histogram_dict(H: torch.Tensor) -> Dict[str, int]:
    values = torch.clamp(H.detach().cpu(), min=0).long().tolist()
    return {str(index): int(value) for index, value in enumerate(values) if int(value) != 0}


def _resource_gap(inventory: Dict[int, int], demand: Dict[int, int]) -> Dict[str, int]:
    keys = set(int(key) for key in inventory) | set(int(key) for key in demand)
    return {
        str(key): int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0))
        for key in sorted(keys)
        if int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0)) != 0
    }


def _mapping_payload(value: Any, *, value_key: str = "value") -> Dict[str, Any]:
    if value is None or (isinstance(value, str) and not value):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return dict(converted)
    return {str(value_key): value}


def _target_h(E_target: Optional[torch.Tensor]) -> Optional[float]:
    try:
        value = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        return None
    return value if value > 0.0 else None


def _selected_histogram_counts(H: Optional[torch.Tensor]) -> str:
    values = (
        torch.clamp(H.detach().cpu(), min=0).long()
        if isinstance(H, torch.Tensor)
        else torch.zeros(25, dtype=torch.long)
    )
    return " ".join(
        f"SU{su}={int(values[su].item()) if su < int(values.numel()) else 0}"
        for su in (11, 12, 13, 22, 23, 24)
    )


def _format_conversion(op: Any) -> str:
    via = (
        f"->SU{int(getattr(op, 'via_su'))}"
        if getattr(op, "via_su", None) is not None
        else ""
    )
    return (
        f"SU{int(getattr(op, 'from_su', -1))}{via}"
        f"->SU{int(getattr(op, 'to_su', -1))}"
        f" x{int(getattr(op, 'count', 1) or 1)}"
        f" nodes={list(getattr(op, 'node_ids', []) or [])}"
        f" reason={str(getattr(op, 'reason', '') or '')}"
    )


def _spectrum_trial_operations(candidate: Any) -> Tuple[str, Dict[int, int]]:
    descriptions: List[str] = []
    delta: Dict[int, int] = {}
    for raw in list(getattr(candidate, "operations", []) or []):
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        node_id = int(op.get("node_id", -1))
        descriptions.append(f"node={node_id}:SU{source}->SU{target}")
        delta[source] = int(delta.get(source, 0)) - 1
        delta[target] = int(delta.get(target, 0)) + 1
    return ", ".join(descriptions) or "none", {
        int(key): int(value) for key, value in sorted(delta.items()) if int(value) != 0
    }


def _print_stage3_plan_summary(
    cycle_number: int,
    plan: Any,
    H_current: torch.Tensor,
    E_target: Optional[torch.Tensor],
    special_degree_meta: Optional[Dict[str, Any]] = None,
) -> None:
    histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
    groups = list(getattr(histogram_plan, "groups", []) or [])
    H_after = getattr(histogram_plan, "H_after", None)
    if not isinstance(H_after, torch.Tensor):
        H_after = H_current
    h_before = (
        float(getattr(groups[0], "h_before", 0.0))
        if groups
        else float(_h_total(H_current, special_degree_meta))
    )
    h_after = (
        float(getattr(groups[-1], "h_after_final", h_before))
        if groups
        else float(_h_total(H_after, special_degree_meta))
    )
    target_h = _target_h(E_target)
    lower, upper = h_soft_bounds(target_h)
    compensation_count = sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in groups
    )
    phase = str(
        getattr(plan, "phase", "")
        or dict(getattr(plan, "diagnostics", {}) or {}).get("stage3_phase", "")
    )
    print(
        f"[Stage3 Plan] cycle={int(cycle_number)} phase={phase or 'unknown'} "
        f"active_subphase={str(getattr(plan, 'active_subphase', '') or '')} "
        f"kind={str(getattr(plan, 'plan_kind', '') or '')} "
        f"reason={str(getattr(plan, 'reason', '') or '')} "
        f"primary={len(groups)} compensation={int(compensation_count)}",
        flush=True,
    )
    print(
        "  [Stage3 H] "
        f"current={h_before:.1f} target={target_h if target_h is not None else 'none'} "
        f"soft=[{lower if lower is not None else 'none'},"
        f"{upper if upper is not None else 'none'}] planned={h_after:.1f}",
        flush=True,
    )
    print(
        f"  [Stage3 Histogram] before: {_selected_histogram_counts(H_current)}",
        flush=True,
    )
    print(
        f"  [Stage3 Histogram] after:  {_selected_histogram_counts(H_after)}",
        flush=True,
    )


def _print_stage3_histogram_adjustments(cycle_number: int, histogram_plan: Any) -> None:
    groups = list(getattr(histogram_plan, "groups", []) or [])
    if not groups:
        print(f"[Block C:resource #{cycle_number}] no physical histogram adjustment", flush=True)
        return
    print(
        f"[Block C:resource #{cycle_number}] structure-unit adjustments: {len(groups)} primary bundle(s)",
        flush=True,
    )
    for group_index, group in enumerate(groups, start=1):
        primary = getattr(group, "primary_op", None)
        compensation = list(getattr(group, "h_compensation_ops", []) or [])
        scoring = dict(getattr(group, "scoring_diagnostics", {}) or {})
        h_scoring = dict(scoring.get("h", {}) or {})
        flex_scoring = dict(scoring.get("flex", {}) or {})
        print(
            f"  [Adjustment {group_index}] primary={_format_conversion(primary)} "
            f"delta={dict(getattr(group, 'combined_delta', {}) or {})} "
            f"H={float(getattr(group, 'h_before', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_primary', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_final', 0.0)):.1f}",
            flush=True,
        )
        for compensation_index, op in enumerate(compensation, start=1):
            print(
                f"    [H Compensation {compensation_index}] {_format_conversion(op)}",
                flush=True,
            )
        if not compensation:
            print("    [H Compensation] none", flush=True)
        print(
            "    [Adjustment Score] "
            f"resource_gap={dict(scoring.get('resource_gap_before', getattr(group, 'demand_before', {})) or {})}"
            f"->{dict(scoring.get('resource_gap_after', getattr(group, 'demand_after', {})) or {})} "
            f"H_soft_penalty={float(h_scoring.get('soft_penalty', 0.0) or 0.0):.4f} "
            f"H_drift_penalty={float(h_scoring.get('drift_penalty', 0.0) or 0.0):.4f} "
            f"flex_penalty={float(flex_scoring.get('penalty', 0.0) or 0.0):.4f} "
            f"total={float(scoring.get('total_score', 0.0) or 0.0):.4f}",
            flush=True,
        )
        if flex_scoring:
            flex_before = dict(flex_scoring.get("before", {}) or {})
            flex_after = dict(flex_scoring.get("after", {}) or {})
            print(
                "    [Flex Impact] "
                f"X/Y/Z={flex_before.get('X', 0)}/{flex_before.get('Y', 0)}/{flex_before.get('Z', 0)}"
                f"->{flex_after.get('X', 0)}/{flex_after.get('Y', 0)}/{flex_after.get('Z', 0)} "
                f"range={flex_before.get('flex_lower', 0)}-{flex_before.get('flex_upper', 0)}"
                f"->{flex_after.get('flex_lower', 0)}-{flex_after.get('flex_upper', 0)} "
                f"gap={flex_before.get('minimum_flex_gap', 0)}"
                f"->{flex_after.get('minimum_flex_gap', 0)}",
                flush=True,
            )


def _histogram_h_compensation_count(histogram_plan: Any) -> int:
    return sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in list(getattr(histogram_plan, "groups", []) or [])
    )


def _primary_conversion_signature(histogram_plan: Any) -> str:
    groups = list(getattr(histogram_plan, "groups", []) or [])
    if not groups:
        return ""
    primary = getattr(groups[0], "primary_op", None)
    if primary is None:
        return ""
    via = (
        f"@{int(getattr(primary, 'via_su'))}"
        if getattr(primary, "via_su", None) is not None
        else ""
    )
    return (
        f"{int(getattr(primary, 'from_su', -1))}>"
        f"{int(getattr(primary, 'to_su', -1))}{via}"
    )


def _register_rejected_resource_variant(
    plan: Any,
    histogram_plan: Any,
    rejected_signatures: set[str],
) -> str:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    if not bool(diagnostics.get("candidate_alternatives_enabled", False)):
        return ""
    signature = _primary_conversion_signature(histogram_plan)
    if not signature or signature in rejected_signatures:
        return ""
    rejected_signatures.add(signature)
    return signature


def _rollback_stage3_plan_transactions(plan: Any, reason: str) -> Tuple[Dict[str, Any], List[int]]:
    phase_result = getattr(plan, "phase_result", None)
    rolled_back_contract_ids: List[int] = []
    for transaction in list(getattr(phase_result, "transactions", []) or []):
        is_current_candidate = (
            str(getattr(transaction, "transaction_origin", "") or "")
            == "current_plan_candidate"
        )
        has_rebuild_effect = bool(
            list(getattr(transaction, "converted_node_ids", []) or [])
            or list(getattr(transaction, "conversion_path", []) or [])
            or list(getattr(transaction, "compensation_node_ids", []) or [])
        )
        if not has_rebuild_effect and not is_current_candidate:
            continue
        attempted = {
            "converted_node_ids": list(getattr(transaction, "converted_node_ids", []) or []),
            "conversion_path": list(getattr(transaction, "conversion_path", []) or []),
            "compensation_node_ids": list(getattr(transaction, "compensation_node_ids", []) or []),
            "claimed_resource_ids": dict(getattr(transaction, "claimed_resource_ids", {}) or {}),
            "product_node_ids": list(getattr(transaction, "product_node_ids", []) or []),
            "slot_assignments": list(getattr(transaction, "slot_assignments", []) or []),
        }
        transaction.committed = False
        transaction.rollback_reason = str(reason)
        transaction.infeasibility_proof = {
            **dict(getattr(transaction, "infeasibility_proof", {}) or {}),
            "reason": str(reason),
            "rolled_back": True,
            "attempted_transaction": attempted,
        }
        transaction.converted_node_ids = []
        transaction.conversion_path = []
        transaction.compensation_node_ids = []
        transaction.claimed_resource_ids = {}
        transaction.product_node_ids = []
        transaction.slot_assignments = []
        rolled_back_contract_ids.append(int(getattr(transaction, "contract_id", -1)))
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostics["phase_result"] = phase_result.to_dict() if hasattr(phase_result, "to_dict") else {}
    diagnostics["rolled_back_contract_ids"] = list(rolled_back_contract_ids)
    diagnostics["rollback_reason"] = str(reason)
    plan.diagnostics = diagnostics
    return plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {}), rolled_back_contract_ids


def _restore_stage3_rotation_cursor(adjuster: Any, cursor_before: Optional[int]) -> None:
    if cursor_before is not None and hasattr(adjuster, "_stage3_su11_rotation_index"):
        adjuster._stage3_su11_rotation_index = int(cursor_before)


def _stage3_terminal_infeasibility_proof(
    plan: Any,
    *,
    default_reason: str = "current_stage3_plan_infeasible",
) -> Dict[str, Any]:
    histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
    phase_result = getattr(plan, "phase_result", None)
    prior_proof = _mapping_payload(
        getattr(phase_result, "infeasibility_proof", {}) or {},
        value_key="reason",
    )
    slot_proof = _mapping_payload(
        prior_proof.get("slot_assignment_proof", {}) or {},
        value_key="reason",
    )
    failed_contract = _mapping_payload(
        prior_proof.get("failed_contract", {}) or {},
        value_key="target_role",
    )
    has_explicit_pending_requirement = bool(
        "required_counts" in prior_proof
        or "required_counts" in failed_contract
    )
    required_source = (
        prior_proof.get("required_counts", {})
        if "required_counts" in prior_proof
        else failed_contract.get("required_counts", {})
        if "required_counts" in failed_contract
        else getattr(plan, "hard_required_counts", {})
        or getattr(histogram_plan, "demand_before", {})
        or {}
    )
    required_counts = {
        str(key): int(value)
        for key, value in dict(required_source or {}).items()
        if int(value) > 0
    }
    available_counts = {
        str(key): int(value)
        for key, value in dict(
            prior_proof.get("available_counts", {})
            or getattr(histogram_plan, "inventory_before", {})
            or {}
        ).items()
    }
    runtime = dict(getattr(plan, "runtime", {}) or {})
    ledger = runtime.get("iteration_topology_ledger")
    if not available_counts and ledger is not None:
        pools = (
            getattr(ledger, "available_ids_by_su", None)
            if not isinstance(ledger, dict)
            else dict(ledger).get("available_ids_by_su", {})
        )
        available_counts = {
            str(key): len(list(values or []))
            for key, values in dict(pools or {}).items()
        }
    missing_counts = {
        str(key): int(value)
        for key, value in dict(
            prior_proof.get("missing_counts", {})
            or prior_proof.get("unresolved_counts", {})
            or getattr(plan, "unresolved_counts", {})
            or {}
        ).items()
        if int(value) > 0
    }
    if not missing_counts and not (
        has_explicit_pending_requirement and not required_counts
    ):
        missing_counts = {
            str(key): max(0, int(required) - int(available_counts.get(str(key), 0)))
            for key, required in required_counts.items()
            if int(required) > int(available_counts.get(str(key), 0))
        }
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    pending_contracts = list(getattr(plan, "pending_contracts", []) or [])
    infeasible_contracts = list(diagnostics.get("infeasible_contracts", []) or [])
    selected_contracts = list(
        dict(diagnostics.get("resource_request_breakdown", {}) or {}).get(
            "selected_active_contracts", []
        )
        or []
    )
    current_contract = _mapping_payload(
        diagnostics.get("current_contract", {}) or {},
        value_key="target_role",
    )
    if not failed_contract and current_contract:
        failed_contract = current_contract
    for candidates in (pending_contracts, infeasible_contracts, selected_contracts):
        if candidates:
            failed_contract = _mapping_payload(
                candidates[0],
                value_key="target_role",
            )
            break
    if not failed_contract:
        failed_contract = {
            "phase": str(getattr(plan, "phase", "") or "unknown"),
            "plan_kind": str(getattr(plan, "plan_kind", "") or "unknown"),
        }
    planned_conversions = list(prior_proof.get("attempted_conversions", []) or [])
    if not planned_conversions:
        planned_conversions = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item or {})
            for item in list(getattr(plan, "planned_conversions", []) or [])
        ]
    rejection_reasons = [
        value
        for value in (
            str(prior_proof.get("reason", "") or ""),
            str(slot_proof.get("reason", "") or ""),
            str(getattr(plan, "blocked_reason", "") or ""),
            str(getattr(histogram_plan, "blocked_reason", "") or ""),
        )
        if value
    ]
    proof = {
        **prior_proof,
        "phase": str(getattr(plan, "phase", "") or "unknown"),
        "reason": str(
            prior_proof.get("reason")
            or slot_proof.get("reason")
            or getattr(plan, "blocked_reason", "")
            or getattr(histogram_plan, "blocked_reason", "")
            or default_reason
        ),
        "failed_contract": failed_contract,
        "required_counts": required_counts,
        "available_counts": available_counts,
        "missing_counts": missing_counts,
        "attempted_conversions": planned_conversions,
        "h_or_element_rejections": list(
            prior_proof.get("h_or_element_rejections", []) or rejection_reasons
        ),
        "template_checks": {
            **_mapping_payload(
                prior_proof.get("template_checks", {}) or {},
                value_key="reason",
            ),
            "next_plan_reason": str(getattr(plan, "reason", "") or ""),
            "next_plan_kind": str(getattr(plan, "plan_kind", "") or ""),
            "rebuild_required": bool(getattr(phase_result, "rebuild_required", False)),
            "spectrum_candidate_count": len(
                list(getattr(plan, "spectrum_candidates", []) or [])
            ),
        },
    }
    return proof


def _stage3_infeasible_plan_dict(
    plan: Any,
    proof: Dict[str, Any],
    *,
    plan_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan_dict = (
        deepcopy(dict(plan_dict or {}))
        if plan_dict is not None
        else plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
    )
    plan_dict["ok"] = False
    plan_dict["reason"] = "phase_resource_infeasible"
    plan_dict["blocked_reason"] = str(
        proof.get("reason", "current_stage3_plan_infeasible")
    )
    phase_result = dict(plan_dict.get("phase_result", {}) or {})
    phase_result.update({
        "active_phase": str(plan_dict.get("phase", "") or "unknown"),
        "phase_completed": False,
        "stop_reason": "phase_resource_infeasible",
        "infeasibility_proof": dict(proof),
    })
    plan_dict["phase_result"] = phase_result
    diagnostics = dict(plan_dict.get("diagnostics", {}) or {})
    diagnostics["infeasibility_proof"] = dict(proof)
    plan_dict["diagnostics"] = diagnostics
    return plan_dict


def run_block_c_resource_adjustment_loop_impl(
    adjuster: Any,
    H: torch.Tensor,
    nodes: Optional[List[Any]],
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    rebuild_fn: Any = None,
    validate_fn: Any = None,
    output_dir: Optional[str] = None,
    max_primary_adjustments_per_stage: int = 10,
    difference_spectrum: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Stage3WorkflowResult:
    primary_limit = max(0, min(10, int(max_primary_adjustments_per_stage)))
    compensation_limit = int(kwargs.pop("max_h_compensation_ops_per_adjustment", 8))
    residual_flex_preferred_23 = int(kwargs.pop("residual_flex_preferred_23", 3))
    residual_flex_min_23 = int(kwargs.pop("residual_flex_min_23", 2))
    residual_side_preferred_23 = int(kwargs.pop("residual_side_preferred_23", 2))
    residual_side_min_23 = int(kwargs.pop("residual_side_min_23", 1))
    h_tolerance = float(kwargs.pop("h_tolerance", 0.06))
    outer_cycle = int(kwargs.pop("outer_cycle", 0))
    iteration_offset = max(0, int(kwargs.pop("iteration_offset", 0)))
    enable_post_external_spectrum = bool(kwargs.pop("enable_post_external_spectrum", True))
    spectrum_candidate_limit = int(kwargs.pop("spectrum_candidate_limit", 2))
    spectrum_max_primary_adjustments = int(kwargs.pop("spectrum_max_primary_adjustments", 10))
    spectrum_min_r2_gain = float(kwargs.pop("spectrum_min_r2_gain", 0.002))
    context_snapshot_fn = kwargs.pop("context_snapshot_fn", None)
    context_restore_fn = kwargs.pop("context_restore_fn", None)
    nodes_current = list(nodes or [])
    H_current = torch.clamp(H.detach().clone(), min=0).long()
    diff_current: Dict[str, Any] = dict(difference_spectrum or {})
    cycles: List[Stage3WorkflowCycleRecord] = []
    accepted_cycles = 0
    stop_reason = "stage3_not_completed"
    accepted_spectrum_histogram_signatures: set[str] = set()
    rejected_spectrum_histogram_signatures: set[str] = set()
    forbidden_spectrum_inverse_signatures: set[str] = set()
    rejected_resource_conversion_signatures: set[str] = set()
    visited_histograms = {
        tuple(int(value) for value in H_current.detach().cpu().long().tolist())
    }

    cycle_idx = 0
    while True:
        cycle_idx += 1
        cycle_number = int(iteration_offset) + int(cycle_idx)
        generation_id = int(max(0, outer_cycle)) * 1_000_000 + int(cycle_number)
        cycle_context_before = (
            context_snapshot_fn()
            if callable(context_snapshot_fn)
            else None
        )
        rotation_cursor_before = (
            int(getattr(adjuster, "_stage3_su11_rotation_index", 0) or 0)
            if hasattr(adjuster, "_stage3_su11_rotation_index")
            else None
        )
        special_degree_meta = (
            adjuster._get_special_degree_meta(H_current)
            if callable(getattr(adjuster, "_get_special_degree_meta", None))
            else None
        )
        plan_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] topology plan start", flush=True)
        plan = adjuster.plan_topology_resource_adjustments(
            H=H_current.detach().clone(),
            nodes=list(nodes_current or []),
            S_target=S_target,
            E_target=E_target,
            max_primary_adjustments_per_stage=primary_limit,
            max_h_compensation_ops_per_adjustment=compensation_limit,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            h_tolerance=h_tolerance,
            generation_id=int(generation_id),
            difference_spectrum=dict(diff_current or {}),
            enable_post_external_spectrum=bool(enable_post_external_spectrum),
            post_external_resume_phase="remaining_pool_spectrum_rebalance",
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            rejected_spectrum_histogram_signatures=sorted(
                accepted_spectrum_histogram_signatures | rejected_spectrum_histogram_signatures
            ),
            forbidden_spectrum_inverse_signatures=sorted(forbidden_spectrum_inverse_signatures),
            forbidden_resource_conversion_signatures=sorted(
                rejected_resource_conversion_signatures
            ),
        )
        _print_stage3_plan_summary(
            cycle_number,
            plan,
            H_current,
            E_target,
            special_degree_meta=special_degree_meta,
        )
        if str(getattr(plan, "reason", "") or "") == "stage3_spectrum_candidates_ready":
            spectrum_plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
            spectrum_H_before = H_current.detach().clone()
            spectrum_nodes_before = deepcopy(nodes_current)
            baseline_r2 = float(dict(diff_current or {}).get("r2", float("nan")))
            baseline_diff = dict(diff_current or {})
            baseline_source = str(baseline_diff.get("source_stage", "") or "")
            baseline_contract = str(baseline_diff.get("spectral_contract", "") or "")
            baseline_hwhm = baseline_diff.get("evaluation_hwhm")
            baseline_build_variant = baseline_diff.get("effective_build_variant")
            spectrum_context_before = (
                deepcopy(cycle_context_before)
            )
            spectrum_flex_audit = _stage3_flex_audit_from_plan(plan)
            baseline_valid = bool(
                np.isfinite(float(baseline_r2))
                and baseline_source == "layer2"
                and baseline_contract == "template_center_mu"
                and rebuild_fn is not None
                and validate_fn is not None
            )
            baseline_reason = (
                "ok"
                if baseline_valid
                else "spectrum_baseline_is_not_layer2"
                if baseline_source != "layer2"
                else "spectrum_baseline_contract_mismatch"
                if baseline_contract != "template_center_mu"
                else "spectrum_baseline_r2_invalid"
                if not np.isfinite(float(baseline_r2))
                else "candidate_rebuild_unavailable"
            )
            print(
                f"[Spectrum Layer2 Baseline] cycle={cycle_number} "
                f"valid={baseline_valid} reason={baseline_reason} "
                f"r2={baseline_r2:.4f} "
                f"histogram={_selected_histogram_counts(spectrum_H_before)}",
                flush=True,
            )
            evaluated_candidates: List[Dict[str, Any]] = []
            accepted_options: List[
                Tuple[float, float, float, str, Any, Any, Any, float, Any, Any, Dict[str, Any]]
            ] = []
            for candidate in list(getattr(plan, "spectrum_candidates", []) or []):
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(spectrum_context_before))
                candidate_diagnostics = dict(getattr(candidate, "diagnostics", {}) or {})
                if not baseline_valid:
                    evaluated_candidates.append({
                        "candidate": (
                            candidate.to_dict()
                            if hasattr(candidate, "to_dict")
                            else dict(candidate or {})
                        ),
                        "accepted": False,
                        "valid": False,
                        "valid_reason": str(baseline_reason),
                        "reason": "saved_layer2_baseline_unavailable",
                        "significant": False,
                    })
                    continue
                candidate_nodes, candidate_H, apply_proof = apply_spectrum_candidate(
                    nodes=spectrum_nodes_before,
                    H=spectrum_H_before,
                    candidate=candidate,
                )
                trial_operations, trial_delta = _spectrum_trial_operations(candidate)
                candidate_h_value = (
                    float(_h_total(candidate_H, special_degree_meta))
                    if isinstance(candidate_H, torch.Tensor)
                    else float("nan")
                )
                print(
                    f"[Spectrum Trial Start] cycle={cycle_number} "
                    f"candidate_id={getattr(candidate, 'candidate_id', '')} "
                    f"spectral_rank={candidate_diagnostics.get('spectral_priority_rank', 'n/a')} "
                    f"bundle={candidate_diagnostics.get('bundle_kind', 'unknown')} "
                    f"batch={int(candidate_diagnostics.get('batch_size', len(list(getattr(candidate, 'operations', []) or []))) or 0)} "
                    f"conversions={dict(candidate_diagnostics.get('conversion_counts', {}) or {})} "
                    f"h_compensation={dict(candidate_diagnostics.get('h_compensation_conversion_counts', {}) or {})} "
                    f"operations=[{trial_operations}] delta={trial_delta} "
                    f"H={float(_h_total(H_current, special_degree_meta)):.1f}->{candidate_h_value:.1f}",
                    flush=True,
                )
                record: Dict[str, Any] = {
                    "candidate": candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate or {}),
                    "accepted": False,
                    "apply_proof": dict(apply_proof or {}),
                }
                if candidate_nodes is None or candidate_H is None or rebuild_fn is None or validate_fn is None:
                    record["reason"] = str(dict(apply_proof or {}).get("reason", "candidate_apply_or_rebuild_unavailable"))
                    rejected_spectrum_histogram_signatures.add(str(getattr(candidate, "histogram_signature", "") or ""))
                    evaluated_candidates.append(record)
                    print(
                        f"[Spectrum Trial Result] candidate_id={getattr(candidate, 'candidate_id', '')} "
                        f"graph_valid=False valid_reason={record['reason']} "
                        "accepted=False state_retained=False",
                        flush=True,
                    )
                    continue
                try:
                    cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                        H_seed=candidate_H,
                        phase="remaining_pool_spectrum_rebalance",
                        cycle_idx=cycle_number - 1,
                        build_variant=(
                            int(baseline_build_variant)
                            if baseline_build_variant is not None
                            else None
                        ),
                    )
                    valid, valid_reason = validate_fn(
                        nodes=cand_nodes,
                        H=cand_H,
                        tag=f"BlockC:spectrum #{cycle_number}/{getattr(candidate, 'candidate_id', '')}",
                    )
                    candidate_source = str(dict(cand_diff or {}).get("source_stage", "") or "")
                    candidate_contract = str(
                        dict(cand_diff or {}).get("spectral_contract", "") or ""
                    )
                    candidate_hwhm = dict(cand_diff or {}).get("evaluation_hwhm")
                    if candidate_source != "layer2":
                        valid = False
                        valid_reason = "spectrum_candidate_is_not_layer2"
                    elif candidate_contract != "template_center_mu":
                        valid = False
                        valid_reason = "spectrum_candidate_contract_mismatch"
                    elif (
                        baseline_hwhm is not None
                        and candidate_hwhm is not None
                        and not np.isclose(float(candidate_hwhm), float(baseline_hwhm))
                    ):
                        valid = False
                        valid_reason = "spectrum_candidate_evaluation_hwhm_mismatch"
                except Exception as exc:
                    valid = False
                    valid_reason = f"spectrum_candidate_rebuild_failed:{exc}"
                    cand_nodes = cand_H = cand_diff = None
                    cand_r2 = float("nan")
                target_types = list(getattr(candidate, "target_su_types", []) or [])
                before_loss = spectrum_window_loss(baseline_diff, target_types)
                after_loss = spectrum_window_loss(cand_diff, target_types) if cand_diff is not None else float("inf")
                r2_gain = float(cand_r2) - float(baseline_r2) if np.isfinite(cand_r2) and np.isfinite(baseline_r2) else float("-inf")
                window_gain = float(before_loss) - float(after_loss)
                candidate_histogram = (
                    tuple(int(value) for value in cand_H.detach().cpu().long().tolist())
                    if cand_H is not None else ()
                )
                repeated_histogram = bool(
                    candidate_histogram and candidate_histogram in visited_histograms
                )
                balance = (
                    spectrum_candidate_balance_penalty(
                        H_before=spectrum_H_before,
                        H_after=cand_H,
                        E_target=E_target,
                        special_degree_meta=special_degree_meta,
                        flex_audit=spectrum_flex_audit,
                    )
                    if isinstance(cand_H, torch.Tensor)
                    else {
                        "penalty": 0.0,
                        "h_penalty": 0.0,
                        "flex_penalty": 0.0,
                        "reason": "candidate_histogram_unavailable",
                    }
                )
                balance_penalty = max(0.0, float(balance.get("penalty", 0.0) or 0.0))
                balanced_r2_gain = float(r2_gain) - float(balance_penalty)
                required_r2_gain = float(spectrum_min_r2_gain)
                significant = bool(
                    valid
                    and not repeated_histogram
                    and float(r2_gain) >= float(required_r2_gain)
                    and float(r2_gain) > 0.0
                )
                record.update({
                    "valid": bool(valid),
                    "valid_reason": str(valid_reason),
                    "r2_before": float(baseline_r2),
                    "r2_after": float(cand_r2),
                    "r2_gain": float(r2_gain),
                    "window_loss_before": float(before_loss),
                    "window_loss_after": float(after_loss),
                    "window_loss_gain": float(window_gain),
                    "balance_penalty": float(balance_penalty),
                    "balanced_r2_gain": float(balanced_r2_gain),
                    "required_r2_gain": float(required_r2_gain),
                    "balance_diagnostics": dict(balance),
                    "repeated_histogram": bool(repeated_histogram),
                    "significant": bool(significant),
                })
                print(
                    f"[Spectrum Trial Result] candidate_id={getattr(candidate, 'candidate_id', '')} "
                    f"graph_valid={bool(valid)} valid_reason={str(valid_reason)} "
                    f"r2={baseline_r2:.4f}->{float(cand_r2):.4f} "
                    f"r2_gain={float(r2_gain):.4f} window_gain={float(window_gain):.4f} "
                    f"balance_penalty={float(balance_penalty):.4f} "
                    f"balanced_r2_gain={float(balanced_r2_gain):.4f} "
                    f"required_r2_gain={float(required_r2_gain):.4f} "
                    f"selection_eligible={bool(significant)} accepted=False state_retained=False",
                    flush=True,
                )
                if significant:
                    candidate_context = (
                        context_snapshot_fn()
                        if callable(context_snapshot_fn)
                        else None
                    )
                    accepted_options.append((
                        float(r2_gain),
                        float(window_gain),
                        -float(balance_penalty),
                        str(getattr(candidate, "candidate_id", "") or ""),
                        cand_nodes,
                        cand_H,
                        cand_diff,
                        float(cand_r2),
                        candidate,
                        candidate_context,
                        dict(balance),
                    ))
                else:
                    rejected_spectrum_histogram_signatures.add(str(getattr(candidate, "histogram_signature", "") or ""))
                evaluated_candidates.append(record)
            if accepted_options:
                selected = max(
                    accepted_options,
                    key=lambda item: (item[0], item[1], item[2], item[3]),
                )
                (
                    r2_gain,
                    window_gain,
                    _negative_balance_penalty,
                    _,
                    cand_nodes,
                    cand_H,
                    cand_diff,
                    cand_r2,
                    candidate,
                    candidate_context,
                    selected_balance,
                ) = selected
                selected_balance_penalty = max(
                    0.0,
                    float(selected_balance.get("penalty", 0.0) or 0.0),
                )
                balanced_r2_gain = float(r2_gain) - float(selected_balance_penalty)
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(candidate_context))
                nodes_current = list(cand_nodes or [])
                H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
                diff_current = dict(cand_diff or {})
                accepted_cycles += 1
                visited_histograms.add(
                    tuple(int(value) for value in H_current.detach().cpu().long().tolist())
                )
                accepted_spectrum_histogram_signatures.add(
                    str(getattr(candidate, "histogram_signature", "") or "")
                )
                forbidden_spectrum_inverse_signatures.add(str(getattr(candidate, "inverse_signature", "") or ""))
                for record in evaluated_candidates:
                    candidate_data = dict(record.get("candidate", {}) or {})
                    if str(candidate_data.get("candidate_id", "")) == str(getattr(candidate, "candidate_id", "")):
                        record["accepted"] = True
                        record["reason"] = "layer2_r2_gain_threshold_met"
                cycles.append(Stage3WorkflowCycleRecord(
                    cycle=cycle_number,
                    phase="remaining_pool_spectrum_rebalance",
                    applied=True,
                    stop_reason="accepted",
                    accepted_state_retained=True,
                    rebuild_valid=True,
                    plan=spectrum_plan_dict,
                    layer2_r2=float(cand_r2),
                    accepted=True,
                    acceptance_reason="layer2_r2_gain_threshold_met",
                    primary_adjustment_count=int(getattr(candidate, "primary_adjustment_count", 0) or 0),
                    histogram_before=_histogram_dict(spectrum_H_before),
                    histogram_after=_histogram_dict(H_current),
                    fresh_rebuild=True,
                    diagnostics={
                        "selected_candidate": candidate.to_dict(),
                        "baseline_source": str(baseline_source),
                        "baseline_spectral_contract": str(baseline_contract),
                        "baseline_layer2_r2": float(baseline_r2),
                        "baseline_reason": str(baseline_reason),
                        "baseline_evaluation_hwhm": baseline_hwhm,
                        "baseline_build_variant": baseline_build_variant,
                        "r2_gain": float(r2_gain),
                        "balanced_r2_gain": float(balanced_r2_gain),
                        "balance_penalty": float(selected_balance_penalty),
                        "required_r2_gain": float(spectrum_min_r2_gain),
                        "balance_diagnostics": dict(selected_balance),
                        "window_loss_gain": float(window_gain),
                        "evaluated_candidates": evaluated_candidates,
                    },
                ))
                print(
                    f"[Spectrum Trial Commit] cycle={cycle_number} "
                    f"candidate_id={getattr(candidate, 'candidate_id', '')} "
                    f"r2={baseline_r2:.4f}->{float(cand_r2):.4f} "
                    f"balanced_r2_gain={float(balanced_r2_gain):.4f} "
                    "accepted=True state_retained=True",
                    flush=True,
                )
                continue
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(spectrum_context_before))
            plan = adjuster.plan_topology_resource_adjustments(
                H=H_current.detach().clone(),
                nodes=list(nodes_current or []),
                S_target=S_target,
                E_target=E_target,
                max_primary_adjustments_per_stage=primary_limit,
                max_h_compensation_ops_per_adjustment=compensation_limit,
                residual_flex_preferred_23=residual_flex_preferred_23,
                residual_flex_min_23=residual_flex_min_23,
                residual_side_preferred_23=residual_side_preferred_23,
                residual_side_min_23=residual_side_min_23,
                h_tolerance=h_tolerance,
                generation_id=int(generation_id),
                difference_spectrum=dict(diff_current or {}),
                enable_post_external_spectrum=bool(enable_post_external_spectrum),
                post_external_resume_phase="resource_su24_ring_closure",
                spectrum_candidate_limit=int(spectrum_candidate_limit),
                spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
                rejected_spectrum_histogram_signatures=sorted(
                    accepted_spectrum_histogram_signatures | rejected_spectrum_histogram_signatures
                ),
                forbidden_spectrum_inverse_signatures=sorted(forbidden_spectrum_inverse_signatures),
                forbidden_resource_conversion_signatures=sorted(
                    rejected_resource_conversion_signatures
                ),
            )
            plan.diagnostics["rejected_spectrum_candidates"] = evaluated_candidates
            plan.diagnostics["spectrum_rebalance_rolled_back"] = True
            plan.diagnostics["spectrum_layer2_baseline"] = {
                "valid": bool(baseline_valid),
                "reason": str(baseline_reason),
                "source_stage": str(baseline_source),
                "spectral_contract": str(baseline_contract),
                "r2": float(baseline_r2),
                "evaluation_hwhm": baseline_hwhm,
                "effective_build_variant": baseline_build_variant,
            }
            _print_stage3_plan_summary(
                cycle_number,
                plan,
                H_current,
                E_target,
                special_degree_meta=special_degree_meta,
            )
        phase_result = getattr(plan, "phase_result", None)
        for transaction in list(getattr(phase_result, "transactions", []) or []):
            transaction.outer_cycle = int(outer_cycle)
            transaction.iteration = int(cycle_number)
        print(
            f"[Block C:resource #{cycle_number}] topology plan done "
            f"({time.perf_counter() - plan_started:.2f}s)",
            flush=True,
        )
        phase = str(getattr(plan, "phase", "") or dict(getattr(plan, "diagnostics", {}) or {}).get("stage3_phase", ""))
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
        before_metrics = _stage3_metrics_from_plan(plan)
        before_signature = _stage3_signature_from_plan(plan)
        histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
        phase_inventory = dict(getattr(histogram_plan, "inventory_before", {}) or {})
        phase_demand = dict(getattr(histogram_plan, "demand_before", {}) or {})
        aggregate_inventory_before = dict(
            getattr(histogram_plan, "aggregate_inventory_before", {}) or {}
        )
        aggregate_inventory_after = dict(
            getattr(histogram_plan, "aggregate_inventory_after", {}) or {}
        )
        aggregate_required_counts = dict(
            getattr(histogram_plan, "aggregate_required_counts", {}) or {}
        )
        accounting_diagnostics = dict(
            getattr(histogram_plan, "accounting_diagnostics", {}) or {}
        )
        histogram_before = _histogram_dict(H_current)
        planned_conversions = list(plan_dict.get('planned_conversions', []) or [])
        adjustment_bundles = list(plan_dict.get('adjustment_bundles', []) or [])
        state_signature = repr((
            tuple(int(value) for value in H_current.detach().cpu().long().tolist()),
            phase,
            str(getattr(plan, "active_subphase", "") or ""),
            before_signature,
            tuple(sorted((int(key), int(value)) for key, value in phase_demand.items())),
            tuple(sorted((int(key), int(value)) for key, value in phase_inventory.items())),
        ))
        phase_eval_dict = dict(dict(getattr(plan, "runtime", {}) or {}).get("phase_evaluation_dict", {}) or {})
        flex_audit = dict(phase_eval_dict.get("flex_audit", {}) or {})
        assembly_view = dict(phase_eval_dict.get("assembly_view", {}) or {})
        flex_target_plan = dict(assembly_view.get("flex_target_plan", {}) or {})
        protected_counts = dict(assembly_view.get("protected_resource_counts", {}) or {})
        print(
            "[Flex Target] "
            f"M={int(flex_audit.get('flex_lower', 0) or 0)} "
            f"N={int(flex_audit.get('flex_upper', 0) or 0)} "
            f"T={int(flex_audit.get('flex_target', flex_audit.get('target_flex_count', 0)) or 0)} "
            f"current={int(dict(phase_eval_dict.get('flex_counts', {}) or {}).get('final', 0) or 0)} "
            f"external_capacity={int(flex_target_plan.get('external_candidate_capacity', 0) or 0)}"
            f" X/Y/Z={int(flex_audit.get('X', 0) or 0)}/{int(flex_audit.get('Y', 0) or 0)}/{int(flex_audit.get('Z', 0) or 0)}"
        )
        if phase in {
            "external_topology_assembly",
            "external_flex_closure",
            "external_fatty_ring_closure",
            "external_side_chain_closure",
        }:
            preferred_demand = dict(plan_dict.get("preferred_required_counts", {}) or {})
            print(
                f"[External Demand] hard={phase_demand} preferred={preferred_demand} "
                f"available={phase_inventory} gap={_resource_gap(phase_inventory, phase_demand)}"
            )
            if aggregate_inventory_before or aggregate_required_counts:
                print(
                    f"[Aggregate Resources] required={aggregate_required_counts} "
                    f"before={aggregate_inventory_before} after={aggregate_inventory_after} "
                    f"gap={_resource_gap(aggregate_inventory_after, aggregate_required_counts)}"
                )
            resource_breakdown = dict(plan_dict.get("resource_request_breakdown", {}) or {})
            selected_contracts = list(resource_breakdown.get("selected_active_contracts", []) or [])
            external_diagnostics = dict(plan_dict.get("diagnostics", {}) or {})
            print(
                "[External Contracts] "
                f"seen={int(external_diagnostics.get('external_contracts_seen', len(selected_contracts)) or 0)} "
                f"direct={int(external_diagnostics.get('external_directly_completed', 0) or 0)} "
                f"converted={int(external_diagnostics.get('external_completed_after_conversion', 0) or 0)} "
                f"blocked={int(external_diagnostics.get('external_blocked', 0) or 0)} "
                f"remaining={int(external_diagnostics.get('external_remaining_pending', 0) or 0)} "
                f"primary={int(external_diagnostics.get('external_primary_count', 0) or 0)}/"
                f"{int(primary_limit)} "
                f"complete={bool(external_diagnostics.get('external_stage_completed', False))}"
            )
            for contract in selected_contracts:
                print(
                    "  [External Contract] "
                    f"contract={int(dict(contract or {}).get('contract_id', 0) or 0)} "
                    f"kind={str(dict(contract or {}).get('contract_kind', '') or '')} "
                    f"fragments={list(dict(contract or {}).get('source_fragment_ids', []) or [])} "
                    f"nodes={list(dict(contract or {}).get('claimed_source_node_ids', []) or [])} "
                    f"required={dict(dict(contract or {}).get('required_counts', {}) or {})}"
                )
        elif phase == "residual_resource_self_closure":
            print(
                "[Residual Flex] "
                f"minimum_gap={int(flex_target_plan.get('remaining_minimum_gap', 0) or 0)} "
                f"preferred_gap={int(flex_target_plan.get('remaining_preferred_gap', 0) or 0)}"
            )
        if str(getattr(plan, "reason", "")) == "stage3_complete":
            stop_reason = "stage3_complete"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase="complete",
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                accepted=True,
                acceptance_reason=stop_reason,
                state_signature=state_signature,
            ))
            return Stage3WorkflowResult(
                nodes=nodes_current,
                H=H_current,
                diff=diff_current,
                ok=True,
                stop_reason=stop_reason,
                accepted_cycles=accepted_cycles,
                cycles=cycles,
                diagnostics={"stage": "topology_resource_adjustment"},
            )
        _print_stage3_histogram_adjustments(cycle_number, histogram_plan)
        apply_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] histogram apply start", flush=True)
        apply_result = adjuster.apply_topology_resource_adjustments(
            nodes=list(nodes_current or []),
            H=H_current.detach().clone(),
            plan=plan,
        )
        print(
            f"[Block C:resource #{cycle_number}] histogram apply done "
            f"({time.perf_counter() - apply_started:.2f}s)",
            flush=True,
        )
        requested_counts = dict(getattr(apply_result, "requested_counts", {}) or {})
        unresolved_counts = dict(getattr(apply_result, "unresolved_counts", {}) or {})
        applied_ops = list(getattr(apply_result, "applied_ops", []) or [])
        request_ops = list(getattr(apply_result, "request_ops", []) or [])
        rejected_ops = list(getattr(apply_result, "rejected_ops", []) or [])
        histogram_changed = not torch.equal(
            torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long(),
            torch.clamp(H_current.detach().cpu(), min=0).long(),
        )
        if not bool(getattr(apply_result, "changed", False)):
            rejection_reason = str(
                dict(rejected_ops[0] or {}).get("reason", "")
                if rejected_ops
                else ""
            )
            plan_blocked_reason = str(
                getattr(plan, "blocked_reason", "")
                or getattr(histogram_plan, "blocked_reason", "")
                or getattr(plan, "reason", "")
                or ""
            )
            if rejection_reason:
                stop_reason = f"histogram_apply_rejected:{rejection_reason}"
            elif list(getattr(histogram_plan, "groups", []) or []):
                stop_reason = "histogram_apply_no_effect"
            elif request_ops or unresolved_counts or str(getattr(plan, "reason", "")) == "phase_resource_infeasible":
                stop_reason = "phase_resource_infeasible"
            else:
                stop_reason = plan_blocked_reason or "histogram_adjustment_impossible"
            terminal_proof: Dict[str, Any] = {}
            if stop_reason == "phase_resource_infeasible":
                terminal_proof = _stage3_terminal_infeasibility_proof(plan)
                print(
                    f"[Block C:resource #{cycle_number}] infeasible "
                    f"phase={str(terminal_proof.get('phase', phase) or phase)} "
                    f"reason={str(terminal_proof.get('reason', stop_reason) or stop_reason)} "
                    f"missing={dict(terminal_proof.get('missing_counts', {}) or {})} "
                    f"failed_contract={terminal_proof.get('failed_contract', {})}",
                    flush=True,
                )
                rolled_back_plan, _ = _rollback_stage3_plan_transactions(
                    plan,
                    stop_reason,
                )
                plan_dict = _stage3_infeasible_plan_dict(
                    plan,
                    terminal_proof,
                    plan_dict=rolled_back_plan,
                )
            elif rejection_reason or stop_reason == "histogram_apply_no_effect":
                plan_dict, _ = _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(
                    terminal_proof.get("reason", "")
                    or getattr(plan, "blocked_reason", "")
                    or stop_reason
                ),
                accepted=False,
                acceptance_reason=stop_reason,
                primary_adjustment_count=int(
                    getattr(histogram_plan, "primary_adjustment_count", 0) or 0
                ),
                physical_conversion_count=int(
                    getattr(histogram_plan, "physical_conversion_count", 0) or 0
                ),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                histogram_before=histogram_before,
                histogram_after=_histogram_dict(getattr(apply_result, "H")),
                phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
                phase_demand={str(key): int(value) for key, value in phase_demand.items()},
                gap_before=_resource_gap(phase_inventory, phase_demand),
                gap_after=_resource_gap(phase_inventory, phase_demand),
                state_signature=state_signature,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "apply_rejection_reason": rejection_reason,
                    "apply_changed": bool(getattr(apply_result, "changed", False)),
                    "histogram_changed": bool(histogram_changed),
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "aggregate_inventory_before": aggregate_inventory_before,
                    "aggregate_inventory_after": aggregate_inventory_after,
                    "aggregate_required_counts": aggregate_required_counts,
                    "protected_resource_counts": dict(
                        getattr(histogram_plan, "protected_resource_counts", {}) or {}
                    ),
                    "resource_accounting": accounting_diagnostics,
                    "infeasibility_proof": terminal_proof,
                },
            ))
            break

        next_histogram_signature = tuple(
            int(value)
            for value in torch.clamp(
                getattr(apply_result, "H").detach().cpu(), min=0
            ).long().tolist()
        )
        if next_histogram_signature in visited_histograms:
            cycle_reason = "physical_histogram_cycle_detected"
            proof = _stage3_terminal_infeasibility_proof(
                plan,
                default_reason=cycle_reason,
            )
            proof["reason"] = cycle_reason
            plan_dict = _stage3_infeasible_plan_dict(plan, proof, plan_dict=plan_dict)
            stop_reason = "phase_resource_infeasible"
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                accepted=False,
                acceptance_reason=cycle_reason,
                blocked_reason=cycle_reason,
                state_signature=state_signature,
                diagnostics={
                    "repeated_histogram": list(next_histogram_signature),
                    "next_pass": "fresh_stage3",
                    "infeasibility_proof": dict(proof),
                },
            ))
            break

        if rebuild_fn is None or validate_fn is None:
            stop_reason = "layer12_rebuild_invalid"
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=stop_reason,
                accepted=False,
                acceptance_reason=stop_reason,
                diagnostics={"rolled_back_contract_ids": list(rolled_back_contract_ids)},
            ))
            break

        try:
            rebuild_started = time.perf_counter()
            print(f"[Block C:resource #{cycle_number}] fresh Layer1/2 rebuild start", flush=True)
            H_seed = torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long()
            cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                H_seed=getattr(apply_result, "H"),
                phase=phase,
                cycle_idx=cycle_number - 1,
            )
            print(
                f"[Block C:resource #{cycle_number}] fresh Layer1/2 rebuild done "
                f"({time.perf_counter() - rebuild_started:.2f}s, r2={float(cand_r2):.4f})",
                flush=True,
            )
        except Exception as exc:
            stop_reason = "layer12_rebuild_invalid"
            rollback_reason = f"{stop_reason}:{exc}"
            retry_signature = _register_rejected_resource_variant(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, rollback_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(exc),
                accepted=False,
                acceptance_reason=stop_reason,
                primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
                physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
                phase_demand={str(key): int(value) for key, value in phase_demand.items()},
                histogram_before=histogram_before,
                histogram_after=_histogram_dict(getattr(apply_result, 'H')),
                gap_before=_resource_gap(phase_inventory, phase_demand),
                state_signature=state_signature,
                fresh_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "rebuild_exception": repr(exc),
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                    "rejected_resource_conversion_signature": str(retry_signature),
                },
            ))
            if retry_signature:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after rebuild failure: rejected={retry_signature}",
                    flush=True,
                )
                continue
            break
        candidate_histogram = torch.clamp(cand_H.detach().cpu(), min=0).long()
        if not torch.equal(candidate_histogram, H_seed):
            valid_reason = "fresh_rebuild_histogram_mismatch"
            retry_signature = _register_rejected_resource_variant(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, valid_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                layer2_r2=float(cand_r2),
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=valid_reason,
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                fresh_rebuild=True,
                diagnostics={
                    "expected_histogram": _histogram_dict(H_seed),
                    "rebuilt_histogram": _histogram_dict(candidate_histogram),
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                    "rejected_resource_conversion_signature": str(retry_signature),
                },
            ))
            if retry_signature:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after histogram mismatch: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break
        validate_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] rebuilt state validation start", flush=True)
        valid, valid_reason = validate_fn(nodes=cand_nodes, H=cand_H, tag=f"BlockC:resource #{cycle_number}")
        print(
            f"[Block C:resource #{cycle_number}] rebuilt state validation done "
            f"({time.perf_counter() - validate_started:.2f}s, valid={bool(valid)})",
            flush=True,
        )
        if not valid:
            retry_signature = _register_rejected_resource_variant(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, str(valid_reason))
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                layer2_r2=float(cand_r2),
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(valid_reason),
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                fresh_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                    "rejected_resource_conversion_signature": str(retry_signature),
                },
            ))
            if retry_signature:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after validation failure: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break

        after_inventory = dict(getattr(histogram_plan, 'inventory_after', {}) or {})
        after_demand = dict(getattr(histogram_plan, 'demand_after', {}) or {})
        acceptance_reason = "valid_histogram_fresh_rebuild"

        nodes_current = list(cand_nodes or [])
        H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
        diff_current = dict(cand_diff or {})
        visited_histograms.add(next_histogram_signature)
        accepted_cycles += 1
        cycles.append(Stage3WorkflowCycleRecord(
            cycle=cycle_number,
            phase=phase,
            applied=True,
            stop_reason="accepted",
            accepted_state_retained=True,
            rebuild_valid=True,
            plan=plan_dict,
            before_metrics=before_metrics,
            after_metrics={},
            layer2_r2=float(cand_r2),
            topology_signature="",
            requested_counts=requested_counts,
            unresolved_counts=unresolved_counts,
            accepted=True,
            acceptance_reason=acceptance_reason,
            primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
            physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
            h_compensation_count=_histogram_h_compensation_count(histogram_plan),
            phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
            phase_demand={str(key): int(value) for key, value in phase_demand.items()},
            histogram_before=histogram_before,
            histogram_after=_histogram_dict(cand_H),
            gap_before=_resource_gap(phase_inventory, phase_demand),
            gap_after=_resource_gap(after_inventory, after_demand),
            state_signature=state_signature,
            fresh_rebuild=True,
            diagnostics={
                "applied_ops": applied_ops,
                "request_ops": request_ops,
                "rejected_ops": rejected_ops,
                "planned_conversions": planned_conversions,
                "adjustment_bundles": adjustment_bundles,
                "before_requirements": dict(plan_dict.get('resource_deficit', {}) or {}),
                "conversion_pattern_key": str(plan_dict.get('plan_kind', '') or ''),
                "flex_target_plan": dict(flex_target_plan),
                "protected_resource_counts": dict(protected_counts),
                "fresh_replan_deferred_to_next_cycle": True,
            },
        ))

    return Stage3WorkflowResult(
        nodes=nodes_current,
        H=H_current,
        diff=diff_current,
        ok=stop_reason == "stage3_complete",
        stop_reason=stop_reason,
        accepted_cycles=accepted_cycles,
        cycles=cycles,
        diagnostics={
            "stage": "topology_resource_adjustment",
            "output_dir": output_dir,
        },
    )
