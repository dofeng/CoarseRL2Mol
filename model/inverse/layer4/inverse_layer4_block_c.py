"""Authoritative Layer4 Block-C Stage3 workflow loop.

Detailed topology rules live under ``model.inverse.topology``.
"""

from copy import deepcopy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..topology import Stage3WorkflowCycleRecord, Stage3WorkflowResult
from ..topology.spectrum_rebalance import apply_spectrum_candidate, spectrum_window_loss


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


def _stage3_plan_fingerprint(plan: Any) -> Tuple[Any, ...]:
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
    concrete_ops: List[Tuple[Any, ...]] = []
    for bundle in list(plan_dict.get("adjustment_bundles", []) or []):
        for op in list(dict(bundle or {}).get("bundle_ops", []) or []):
            op_dict = dict(op or {})
            if op_dict.get("node_id") is None and op_dict.get("delta") is None:
                continue
            concrete_ops.append((
                str(op_dict.get("op", "")),
                op_dict.get("from_su"),
                op_dict.get("to_su"),
                op_dict.get("node_id"),
                tuple(sorted((str(key), int(value)) for key, value in dict(op_dict.get("delta", {}) or {}).items())),
            ))
    return (
        str(plan_dict.get("phase", "")),
        str(plan_dict.get("active_subphase", "")),
        _stage3_signature_from_plan(plan),
        tuple(concrete_ops),
    )


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
        primary_text = (
            f"SU{int(getattr(primary, 'from_su', -1))}->"
            + (
                f"SU{int(getattr(primary, 'via_su'))}->"
                if getattr(primary, "via_su", None) is not None
                else ""
            )
            + f"SU{int(getattr(primary, 'to_su', -1))}"
            f" x{int(getattr(primary, 'count', 1) or 1)}"
        )
        compensation_text = ", ".join(
            f"SU{int(getattr(op, 'from_su', -1))}->SU{int(getattr(op, 'to_su', -1))}"
            f" x{int(getattr(op, 'count', 1) or 1)}"
            for op in compensation
        ) or "none"
        print(
            f"  [Adjustment {group_index}] primary={primary_text} "
            f"H_compensation={compensation_text} "
            f"delta={dict(getattr(group, 'combined_delta', {}) or {})} "
            f"H={float(getattr(group, 'h_before', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_primary', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_final', 0.0)):.1f}",
            flush=True,
        )


def _histogram_h_compensation_count(histogram_plan: Any) -> int:
    return sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in list(getattr(histogram_plan, "groups", []) or [])
    )


def _topology_strictly_improved(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, str]:
    if not before or not after:
        return False, "topology_metrics_missing"
    if int(after.get("subphase_rank", 0)) > int(before.get("subphase_rank", 0)):
        return True, "subphase_advanced"
    if int(after.get("pending_count", 0)) < int(before.get("pending_count", 0)):
        return True, "pending_reduced"
    if int(after.get("unresolved_total", 0)) < int(before.get("unresolved_total", 0)):
        return True, "unresolved_shortage_reduced"
    if int(after.get("shortage_total", 0)) < int(before.get("shortage_total", 0)):
        return True, "resource_shortage_reduced"
    if (
        int(after.get("flex_gap", 0)) < int(before.get("flex_gap", 0))
        and int(after.get("unresolved_total", 0)) <= int(before.get("unresolved_total", 0))
    ):
        return True, "flex_gap_reduced"
    if (
        int(after.get("product_count", 0)) > int(before.get("product_count", 0))
        and int(after.get("pending_count", 0)) <= int(before.get("pending_count", 0))
        and int(after.get("shortage_total", 0)) <= int(before.get("shortage_total", 0))
    ):
        return True, "product_count_increased"
    return False, "topology_not_improved"


def _rollback_stage3_plan_transactions(plan: Any, reason: str) -> Tuple[Dict[str, Any], List[int]]:
    phase_result = getattr(plan, "phase_result", None)
    rolled_back_contract_ids: List[int] = []
    for transaction in list(getattr(phase_result, "transactions", []) or []):
        has_rebuild_effect = bool(
            list(getattr(transaction, "converted_node_ids", []) or [])
            or list(getattr(transaction, "conversion_path", []) or [])
            or list(getattr(transaction, "compensation_node_ids", []) or [])
        )
        if not has_rebuild_effect:
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
    prior_proof = dict(getattr(phase_result, "infeasibility_proof", {}) or {})
    slot_proof = dict(prior_proof.get("slot_assignment_proof", {}) or {})
    required_counts = {
        str(key): int(value)
        for key, value in dict(
            prior_proof.get("required_counts", {})
            or getattr(plan, "hard_required_counts", {})
            or getattr(histogram_plan, "demand_before", {})
            or {}
        ).items()
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
    if not missing_counts:
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
    failed_contract: Dict[str, Any] = dict(prior_proof.get("failed_contract", {}) or {})
    current_contract = dict(diagnostics.get("current_contract", {}) or {})
    if not failed_contract and current_contract:
        failed_contract = current_contract
    for candidates in (pending_contracts, infeasible_contracts, selected_contracts):
        if candidates:
            failed_contract = dict(candidates[0] or {})
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
            **dict(prior_proof.get("template_checks", {}) or {}),
            "next_plan_reason": str(getattr(plan, "reason", "") or ""),
            "next_plan_kind": str(getattr(plan, "plan_kind", "") or ""),
            "rebuild_required": bool(getattr(phase_result, "rebuild_required", False)),
            "spectrum_candidate_count": len(
                list(getattr(plan, "spectrum_candidates", []) or [])
            ),
        },
    }
    return proof


def _stage3_iteration_budget_proof(
    plan: Any,
    *,
    max_stage3_iterations: int,
    accepted_cycles: int,
) -> Dict[str, Any]:
    proof = _stage3_terminal_infeasibility_proof(
        plan,
        default_reason="stage3_iteration_budget_exhausted",
    )
    proof.update({
        "reason": "stage3_iteration_budget_exhausted",
        "max_stage3_iterations": int(max_stage3_iterations),
        "accepted_cycles": int(accepted_cycles),
    })
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
    diagnostics["iteration_budget_exhausted"] = bool(
        proof.get("reason") == "stage3_iteration_budget_exhausted"
    )
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
    max_stage3_iterations: int = 20,
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
    spectrum_candidate_limit = int(kwargs.pop("spectrum_candidate_limit", 10))
    spectrum_max_primary_adjustments = int(kwargs.pop("spectrum_max_primary_adjustments", 8))
    spectrum_min_r2_gain = float(kwargs.pop("spectrum_min_r2_gain", 0.002))
    spectrum_min_window_loss_gain = float(kwargs.pop("spectrum_min_window_loss_gain", 0.1))
    spectrum_max_accepted_rounds = max(0, int(kwargs.pop("spectrum_max_accepted_rounds", 2)))
    context_snapshot_fn = kwargs.pop("context_snapshot_fn", None)
    context_restore_fn = kwargs.pop("context_restore_fn", None)
    nodes_current = list(nodes or [])
    H_current = torch.clamp(H.detach().clone(), min=0).long()
    diff_current: Dict[str, Any] = dict(difference_spectrum or {})
    cycles: List[Stage3WorkflowCycleRecord] = []
    accepted_cycles = 0
    stop_reason = "max_stage3_iterations_reached"
    accepted_state_signatures = set()
    accepted_spectrum_histogram_signatures: set[str] = set()
    rejected_spectrum_histogram_signatures: set[str] = set()
    forbidden_spectrum_inverse_signatures: set[str] = set()
    accepted_spectrum_rounds = 0

    for cycle_idx in range(max(0, int(max_stage3_iterations))):
        cycle_number = int(iteration_offset) + int(cycle_idx) + 1
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
            enable_post_external_spectrum=bool(
                enable_post_external_spectrum
                and int(accepted_spectrum_rounds) < int(spectrum_max_accepted_rounds)
            ),
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            rejected_spectrum_histogram_signatures=sorted(
                accepted_spectrum_histogram_signatures | rejected_spectrum_histogram_signatures
            ),
            forbidden_spectrum_inverse_signatures=sorted(forbidden_spectrum_inverse_signatures),
        )
        if str(getattr(plan, "reason", "") or "") == "stage3_spectrum_candidates_ready":
            spectrum_plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
            spectrum_H_before = H_current.detach().clone()
            baseline_r2 = float(dict(diff_current or {}).get("r2", float("nan")))
            spectrum_context_before = (
                deepcopy(cycle_context_before)
            )
            evaluated_candidates: List[Dict[str, Any]] = []
            accepted_options: List[Tuple[float, float, str, Any, Any, Any, float, Any, Any]] = []
            for candidate in list(getattr(plan, "spectrum_candidates", []) or []):
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(spectrum_context_before))
                candidate_nodes, candidate_H, apply_proof = apply_spectrum_candidate(
                    nodes=nodes_current,
                    H=H_current,
                    candidate=candidate,
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
                    continue
                try:
                    cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                        H_seed=candidate_H,
                        phase="remaining_pool_spectrum_rebalance",
                        cycle_idx=cycle_number - 1,
                        identity_seed_nodes=candidate_nodes,
                    )
                    valid, valid_reason = validate_fn(
                        nodes=cand_nodes,
                        H=cand_H,
                        tag=f"BlockC:spectrum #{cycle_number}/{getattr(candidate, 'candidate_id', '')}",
                    )
                except Exception as exc:
                    valid = False
                    valid_reason = f"spectrum_candidate_rebuild_failed:{exc}"
                    cand_nodes = cand_H = cand_diff = None
                    cand_r2 = float("nan")
                target_types = list(getattr(candidate, "target_su_types", []) or [])
                before_loss = spectrum_window_loss(diff_current, target_types)
                after_loss = spectrum_window_loss(cand_diff, target_types) if cand_diff is not None else float("inf")
                r2_gain = float(cand_r2) - float(baseline_r2) if np.isfinite(cand_r2) and np.isfinite(baseline_r2) else float("-inf")
                window_gain = float(before_loss) - float(after_loss)
                significant = bool(
                    valid
                    and float(r2_gain) >= float(spectrum_min_r2_gain)
                    and float(window_gain) >= float(spectrum_min_window_loss_gain)
                    and float(r2_gain) > 0.0
                    and float(window_gain) > 0.0
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
                    "significant": bool(significant),
                })
                if significant:
                    candidate_context = (
                        context_snapshot_fn()
                        if callable(context_snapshot_fn)
                        else None
                    )
                    accepted_options.append((
                        float(r2_gain),
                        float(window_gain),
                        str(getattr(candidate, "candidate_id", "") or ""),
                        cand_nodes,
                        cand_H,
                        cand_diff,
                        float(cand_r2),
                        candidate,
                        candidate_context,
                    ))
                else:
                    rejected_spectrum_histogram_signatures.add(str(getattr(candidate, "histogram_signature", "") or ""))
                evaluated_candidates.append(record)
            if accepted_options:
                selected = max(accepted_options, key=lambda item: (item[0], item[1], item[2]))
                (
                    r2_gain,
                    window_gain,
                    _,
                    cand_nodes,
                    cand_H,
                    cand_diff,
                    cand_r2,
                    candidate,
                    candidate_context,
                ) = selected
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(candidate_context))
                nodes_current = list(cand_nodes or [])
                H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
                diff_current = dict(cand_diff or {})
                accepted_cycles += 1
                accepted_spectrum_rounds += 1
                accepted_spectrum_histogram_signatures.add(
                    str(getattr(candidate, "histogram_signature", "") or "")
                )
                forbidden_spectrum_inverse_signatures.add(str(getattr(candidate, "inverse_signature", "") or ""))
                for record in evaluated_candidates:
                    candidate_data = dict(record.get("candidate", {}) or {})
                    if str(candidate_data.get("candidate_id", "")) == str(getattr(candidate, "candidate_id", "")):
                        record["accepted"] = True
                        record["reason"] = "significant_r2_and_window_gain"
                cycles.append(Stage3WorkflowCycleRecord(
                    cycle=cycle_number,
                    phase="remaining_pool_spectrum_rebalance",
                    applied=True,
                    stop_reason="accepted",
                    accepted_state_retained=False,
                    rebuild_valid=True,
                    plan=spectrum_plan_dict,
                    layer2_r2=float(cand_r2),
                    accepted=True,
                    acceptance_reason="significant_r2_and_window_gain",
                    primary_adjustment_count=int(getattr(candidate, "primary_adjustment_count", 0) or 0),
                    histogram_before=_histogram_dict(spectrum_H_before),
                    histogram_after=_histogram_dict(H_current),
                    cold_rebuild=True,
                    diagnostics={
                        "selected_candidate": candidate.to_dict(),
                        "r2_gain": float(r2_gain),
                        "window_loss_gain": float(window_gain),
                        "evaluated_candidates": evaluated_candidates,
                    },
                ))
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
                skip_post_external_spectrum=True,
                spectrum_candidate_limit=int(spectrum_candidate_limit),
                spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
                rejected_spectrum_histogram_signatures=sorted(
                    accepted_spectrum_histogram_signatures | rejected_spectrum_histogram_signatures
                ),
                forbidden_spectrum_inverse_signatures=sorted(forbidden_spectrum_inverse_signatures),
            )
            plan.diagnostics["rejected_spectrum_candidates"] = evaluated_candidates
            plan.diagnostics["spectrum_rebalance_rolled_back"] = True
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
            "external_flex_closure",
            "external_fatty_ring_closure",
            "external_side_chain_closure",
        }:
            preferred_demand = dict(plan_dict.get("preferred_required_counts", {}) or {})
            print(
                f"[External Demand] hard={phase_demand} preferred={preferred_demand} "
                f"available={phase_inventory} gap={_resource_gap(phase_inventory, phase_demand)}"
            )
            resource_breakdown = dict(plan_dict.get("resource_request_breakdown", {}) or {})
            selected_contracts = list(resource_breakdown.get("selected_active_contracts", []) or [])
            print(
                "[External Contracts] "
                f"before={int(resource_breakdown.get('active_contract_count_before_overlap_filter', 0) or 0)} "
                f"after={int(resource_breakdown.get('active_contract_count_after_overlap_filter', 0) or 0)}"
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
        if state_signature in accepted_state_signatures:
            stop_reason = "repeated_histogram_phase_state"
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
                acceptance_reason=stop_reason,
                state_signature=state_signature,
            ))
            break

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
            plan_blocked_reason = str(
                getattr(plan, "blocked_reason", "")
                or getattr(histogram_plan, "blocked_reason", "")
                or getattr(plan, "reason", "")
                or ""
            )
            stop_reason = (
                "phase_resource_infeasible"
                if request_ops or unresolved_counts or str(getattr(plan, "reason", "")) == "phase_resource_infeasible"
                else plan_blocked_reason or "histogram_adjustment_impossible"
            )
            terminal_proof: Dict[str, Any] = {}
            if stop_reason == "phase_resource_infeasible":
                terminal_proof = _stage3_terminal_infeasibility_proof(plan)
                plan_dict = _stage3_infeasible_plan_dict(plan, terminal_proof)
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
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "infeasibility_proof": terminal_proof,
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
            print(f"[Block C:resource #{cycle_number}] cold Layer1/2 rebuild start", flush=True)
            H_seed = torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long()
            cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                H_seed=getattr(apply_result, "H"),
                phase=phase,
                cycle_idx=cycle_number - 1,
                identity_seed_nodes=list(getattr(apply_result, "nodes", []) or []),
            )
            print(
                f"[Block C:resource #{cycle_number}] cold Layer1/2 rebuild done "
                f"({time.perf_counter() - rebuild_started:.2f}s, r2={float(cand_r2):.4f})",
                flush=True,
            )
        except Exception as exc:
            stop_reason = "layer12_rebuild_invalid"
            rollback_reason = f"{stop_reason}:{exc}"
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
                cold_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "rebuild_exception": repr(exc),
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            break
        candidate_histogram = torch.clamp(cand_H.detach().cpu(), min=0).long()
        if not torch.equal(candidate_histogram, H_seed):
            valid_reason = "cold_rebuild_histogram_mismatch"
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
                cold_rebuild=True,
                diagnostics={
                    "expected_histogram": _histogram_dict(H_seed),
                    "rebuilt_histogram": _histogram_dict(candidate_histogram),
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
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
                cold_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break

        replan_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] rebuilt topology replan start", flush=True)
        after_plan = adjuster.plan_topology_resource_adjustments(
            H=torch.clamp(cand_H.detach().clone(), min=0).long(),
            nodes=list(cand_nodes or []),
            S_target=S_target,
            E_target=E_target,
            max_primary_adjustments_per_stage=primary_limit,
            max_h_compensation_ops_per_adjustment=compensation_limit,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            h_tolerance=h_tolerance,
        )
        print(
            f"[Block C:resource #{cycle_number}] rebuilt topology replan done "
            f"({time.perf_counter() - replan_started:.2f}s)",
            flush=True,
        )
        after_metrics = _stage3_metrics_from_plan(after_plan)
        after_signature = _stage3_signature_from_plan(after_plan)
        after_histogram_plan = getattr(after_plan, 'histogram_adjustment_plan', None)
        after_inventory = dict(getattr(after_histogram_plan, 'inventory_before', {}) or {})
        after_demand = dict(getattr(after_histogram_plan, 'demand_before', {}) or {})
        topology_improved, topology_improvement_reason = _topology_strictly_improved(
            before_metrics,
            after_metrics,
        )
        metrics_comparable = bool(before_metrics and after_metrics)
        if metrics_comparable and not topology_improved:
            rollback_reason = "topology_progress_not_improved"
            rolled_back_plan, rolled_back_contract_ids = _rollback_stage3_plan_transactions(
                plan,
                rollback_reason,
            )
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            proof = _stage3_terminal_infeasibility_proof(
                plan,
                default_reason=rollback_reason,
            )
            proof.update({
                "reason": rollback_reason,
                "topology_improvement_reason": str(topology_improvement_reason),
                "before_metrics": dict(before_metrics),
                "after_metrics": dict(after_metrics),
            })
            plan_dict = _stage3_infeasible_plan_dict(
                plan,
                proof,
                plan_dict=rolled_back_plan,
            )
            stop_reason = "phase_resource_infeasible"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                layer2_r2=float(cand_r2),
                topology_signature=before_signature,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=rollback_reason,
                accepted=False,
                acceptance_reason=rollback_reason,
                primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
                physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
                phase_demand={str(key): int(value) for key, value in phase_demand.items()},
                histogram_before=histogram_before,
                histogram_after=histogram_before,
                gap_before=_resource_gap(phase_inventory, phase_demand),
                gap_after=_resource_gap(after_inventory, after_demand),
                state_signature=state_signature,
                cold_rebuild=True,
                diagnostics={
                    "infeasibility_proof": dict(proof),
                    "attempted_histogram_after": _histogram_dict(cand_H),
                    "topology_improvement_reason": str(topology_improvement_reason),
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            break

        acceptance_reason = (
            str(topology_improvement_reason)
            if metrics_comparable
            else "valid_histogram_cold_rebuild"
        )

        nodes_current = list(cand_nodes or [])
        H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
        diff_current = dict(cand_diff or {})
        accepted_cycles += 1
        accepted_state_signatures.add(state_signature)
        cycles.append(Stage3WorkflowCycleRecord(
            cycle=cycle_number,
            phase=phase,
            applied=True,
            stop_reason="accepted",
            accepted_state_retained=False,
            rebuild_valid=True,
            plan=plan_dict,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            layer2_r2=float(cand_r2),
            topology_signature=after_signature,
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
            cold_rebuild=True,
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
            },
        ))

    if stop_reason == "max_stage3_iterations_reached":
        audit_cycle = int(iteration_offset) + max(0, int(max_stage3_iterations)) + 1
        audit_generation_id = int(max(0, outer_cycle)) * 1_000_000 + int(audit_cycle)
        audit_plan = adjuster.plan_topology_resource_adjustments(
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
            generation_id=int(audit_generation_id),
            difference_spectrum=dict(diff_current or {}),
            enable_post_external_spectrum=bool(
                enable_post_external_spectrum
                and int(accepted_spectrum_rounds) < int(spectrum_max_accepted_rounds)
            ),
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            rejected_spectrum_histogram_signatures=sorted(
                accepted_spectrum_histogram_signatures | rejected_spectrum_histogram_signatures
            ),
            forbidden_spectrum_inverse_signatures=sorted(forbidden_spectrum_inverse_signatures),
        )
        for transaction in list(
            getattr(getattr(audit_plan, "phase_result", None), "transactions", []) or []
        ):
            transaction.outer_cycle = int(outer_cycle)
            transaction.iteration = int(audit_cycle)
        audit_metrics = _stage3_metrics_from_plan(audit_plan)
        audit_signature = _stage3_signature_from_plan(audit_plan)
        audit_state_signature = repr(_stage3_plan_fingerprint(audit_plan))
        if str(getattr(audit_plan, "reason", "") or "") == "stage3_complete":
            stop_reason = "stage3_complete"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=audit_cycle,
                phase="complete",
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=audit_plan.to_dict() if hasattr(audit_plan, "to_dict") else dict(audit_plan or {}),
                before_metrics=audit_metrics,
                after_metrics=audit_metrics,
                topology_signature=audit_signature,
                accepted=True,
                acceptance_reason="final_read_only_audit_complete",
                state_signature=audit_state_signature,
            ))
        else:
            proof = _stage3_iteration_budget_proof(
                audit_plan,
                max_stage3_iterations=max_stage3_iterations,
                accepted_cycles=accepted_cycles,
            )
            stop_reason = "phase_resource_infeasible"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=audit_cycle,
                phase=str(getattr(audit_plan, "phase", "") or "unknown"),
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=_stage3_infeasible_plan_dict(audit_plan, proof),
                before_metrics=audit_metrics,
                after_metrics=audit_metrics,
                topology_signature=audit_signature,
                accepted=False,
                acceptance_reason="stage3_iteration_budget_exhausted",
                blocked_reason="stage3_iteration_budget_exhausted",
                state_signature=audit_state_signature,
                diagnostics={"infeasibility_proof": dict(proof)},
            ))

    return Stage3WorkflowResult(
        nodes=nodes_current,
        H=H_current,
        diff=diff_current,
        ok=stop_reason == "stage3_complete",
        stop_reason=stop_reason,
        accepted_cycles=accepted_cycles,
        cycles=cycles,
        diagnostics={"stage": "topology_resource_adjustment", "output_dir": output_dir},
    )
