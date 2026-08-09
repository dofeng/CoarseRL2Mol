from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .classification.classifier import classify_topology_units
from .assembly_external import assemble_pool_su24_rings
from .phase_eval import assemble_topology_units
from .histogram_adjustment import (
    _h_total,
    build_histogram_adjustment_plan,
    next_su11_aromatic_target,
)
from .conversion_scoring import (
    ConversionContext,
    apply_conversions_to_H,
    apply_conversions_to_inventory,
    h_within_context_hard_bounds,
    rank_conversions,
    score_conversion_bundle,
)
from .resource_profiles import (
    DEFAULT_STAGE3_H_HARD_TOLERANCE,
    MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    audit_raw_su24_resource_profiles,
    build_raw_su24_resource_profiles,
    h_hard_bounds,
    h_within_hard_bounds,
)
from .resource_accounting import build_resource_accounting_snapshot
from .iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_priority,
    contract_source_status,
    normalize_slot,
    required_counts_from_slots,
    seed_ledger_topology_state,
)
from .round_context import normalize_stage3_product
from .spectrum_rebalance import generate_spectrum_candidates
from .stage3_models import (
    AtomicAdjustmentGroup,
    HistogramConversion,
    HistogramAdjustmentPlan,
    Stage3AdjustmentBundle,
    Stage3AdjustmentOp,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    Stage3OwnershipState,
    Stage3ProgressPhase,
    SU24ResourceRoleProfile,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyResourceTransaction,
    merge_stage3_contracts_by_owner,
    stage3_contract_owner_key,
)

RESOURCE_TYPES = (11, 22, 23, 24)
RESIDUAL_SELF_CLOSURE_TYPES = (11, 22, 23)
EXTERNAL_ROLE_ORDER = (
    ("external_flex_closure", "flex_chain"),
    ("external_fatty_ring_closure", "fatty_ring"),
    ("external_side_chain_closure", "side_chain"),
)
EXTERNAL_STAGE = "external_topology_assembly"
EXTERNAL_PHASE_NAMES = frozenset(phase for phase, _ in EXTERNAL_ROLE_ORDER)


def _h_hard_rule_diagnostics(h_tolerance: float) -> Dict[str, Any]:
    effective_tolerance = min(1.0, max(0.0, float(h_tolerance)))
    return {
        "h_tolerance_mode": "stage3_symmetric_hard_bound",
        "effective_h_tolerance": float(effective_tolerance),
        "stage3_hard_lower_ratio": -float(effective_tolerance),
        "stage3_hard_upper_ratio": float(effective_tolerance),
    }


def _annotate_h_hard_rule(plan: Stage3AdjustmentPlan, h_tolerance: float) -> Stage3AdjustmentPlan:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostics.setdefault("h_constraint", _h_hard_rule_diagnostics(h_tolerance))
    plan.diagnostics = diagnostics
    return plan


def _filter_spectrum_candidates_by_stage3_hard_constraint(
    candidates: Sequence[Any],
    *,
    target_h: Optional[float],
    h_tolerance: float,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    valid: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in list(candidates or []):
        balance = dict(
            dict(getattr(candidate, "diagnostics", {}) or {}).get(
                "preflight_balance", {}
            )
            or {}
        )
        candidate_h = balance.get("h_after")
        if (
            target_h is None
            or candidate_h is None
            or h_within_hard_bounds(float(candidate_h), target_h, h_tolerance)
        ):
            valid.append(candidate)
            continue
        rejected.append({
            "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
            "h_after": candidate_h,
            "reason": "stage3_hard_constraint_violation",
        })
    return valid, rejected


def _evaluate_external_phases(
    *,
    phase_eval: Any,
    nodes: Sequence[Any],
    classification: Any = None,
    S_target: Any = None,
    E_target: Any = None,
) -> Any:
    classification_obj = classification or classify_topology_units(list(nodes or []))
    assembly_obj = assemble_topology_units(
        nodes=list(nodes or []),
        classification=classification_obj,
        S_target=S_target,
        E_target=E_target,
        flex_audit=dict(getattr(phase_eval, "flex_audit", {}) or {}),
        generation_id=int(getattr(phase_eval, "generation_id", 0) or 0),
    )
    assembly = assembly_obj.to_dict()
    symbolic_product_objects = [
        normalize_stage3_product(
            raw,
            generation_id=int(getattr(phase_eval, "generation_id", 0) or 0),
            index=index,
        )
        for index, raw in enumerate(list(assembly.get("products", []) or []))
    ]
    symbolic_products = [item.to_dict() for item in symbolic_product_objects]
    assembly["products"] = symbolic_products
    diagnostics = dict(assembly.get("diagnostics", {}) or {})
    pending_contracts = [
        dict(item or {})
        for item in list(diagnostics.get("pending_contracts", []) or [])
        if dict(item or {})
    ]
    working = deepcopy(phase_eval)
    working.assembly_attempted = True
    working.assembly_view = assembly
    working.canonical_contracts = sorted(
        merge_stage3_contracts_by_owner(pending_contracts),
        key=contract_priority,
    )
    ownership = dict(
        assembly.get("ownership_ledger", {})
        or diagnostics.get("ownership_ledger", {})
        or {}
    )
    working.ownership_ledger = ownership
    working.closure_pool_ids_by_su = {
        int(key): [int(value) for value in list(values or [])]
        for key, values in dict(ownership.get("closure_pool_ids_by_su", {}) or {}).items()
    }
    working.aromatic_donor_ids_by_su = {
        int(key): [int(value) for value in list(values or [])]
        for key, values in dict(ownership.get("aromatic_donor_ids_by_su", {}) or {}).items()
    }
    working.external_closure_ok = not bool(assembly.get("pending_units", []) or pending_contracts)
    working.active_phase = str(
        assembly.get("active_phase", "")
        or diagnostics.get("stage3_phase", "")
        or "external_flex_closure"
    )
    working.phase = str(working.active_phase)
    working.active_external_subphase = str(
        diagnostics.get("active_external_subphase", "")
        or ("remaining_pool_spectrum_rebalance" if working.external_closure_ok else "external_flex_closure")
    )
    working.flex_bounds = dict(assembly.get("flex_bounds", working.flex_bounds) or {})
    working.flex_counts = dict(assembly.get("flex_counts", working.flex_counts) or {})
    working.flex_decision = dict(assembly.get("flex_decision", {}) or {})
    working.topology_metrics = dict(assembly.get("topology_metrics", {}) or {})
    working.topology_signature = str(assembly.get("topology_signature", "") or "")
    working.phase_signature = str(
        assembly.get("phase_signature", "") or working.topology_signature
    )
    working.requested_counts = {
        str(key): int(value)
        for key, value in dict(assembly.get("requested_counts", {}) or {}).items()
        if int(value) > 0
    }
    working.unresolved_counts = {
        str(key): int(value)
        for key, value in dict(assembly.get("unresolved_counts", {}) or {}).items()
        if int(value) > 0
    }
    working.blocked_reason = str(assembly.get("blocked_reason", "") or "")
    working.completed_phases = [
        str(value) for value in list(assembly.get("completed_phases", []) or [])
    ]
    working.phase_metrics = dict(assembly.get("phase_metrics", {}) or {})
    working.external_subphase_status = dict(
        diagnostics.get("external_subphase_status", {}) or {}
    )
    working.flex_range_reconciliation_ok = bool(
        working.active_phase != "flex_range_reconciliation"
    )
    working.diagnostics = {
        **dict(getattr(working, "diagnostics", {}) or {}),
        "external_assembly": diagnostics,
        "external_closure_ok": bool(working.external_closure_ok),
        "external_pending_contract_count": len(pending_contracts),
    }
    return working


def _resource_ids(phase_eval: Any) -> Dict[int, List[int]]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    classification = dict(getattr(phase_eval, "classification_view", {}) or {})
    pool = dict(classification.get("resource_pool", {}) or {})
    raw = (
        ledger.get("remaining_ids_by_su")
        or pool.get("remaining_ids_by_su")
        or pool.get("initial_ids_by_su")
        or {}
    )
    result = {
        int(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(raw or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    donor_raw = ledger.get("su11_donor_ids_by_su") or pool.get("su11_donor_ids_by_su") or {}
    for key, values in dict(donor_raw or {}).items():
        if not str(key).lstrip("-").isdigit():
            continue
        result.setdefault(int(key), [])
        result[int(key)] = sorted(set(result[int(key)]) | {int(value) for value in list(values or [])})
    for su_type in (10, 11, 12, 13, 22, 23, 24):
        result.setdefault(int(su_type), [])
    return result


def _remaining_closure_ids_by_su(
    pool: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    return {
        int(su_type): [int(value) for value in list(pool.get(int(su_type), []) or [])]
        for su_type in RESIDUAL_SELF_CLOSURE_TYPES
        if list(pool.get(int(su_type), []) or [])
    }


def _tolerated_remaining_closure_ids(
    pool: Dict[int, List[int]],
    *,
    remaining_flex_goal: int,
) -> Dict[int, List[int]]:
    by_su = _remaining_closure_ids_by_su(pool)
    total = sum(len(values) for values in by_su.values())
    if (
        int(remaining_flex_goal) <= 0
        and int(total) <= int(MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES)
    ):
        return by_su
    return {}


def _consume(pool: Dict[int, List[int]], required: Dict[int, int]) -> Optional[Dict[int, List[int]]]:
    if any(len(pool.get(int(su), [])) < int(count) for su, count in required.items()):
        return None
    claimed: Dict[int, List[int]] = {}
    for su_type, count in sorted(required.items()):
        selected = list(pool.get(int(su_type), [])[: int(count)])
        claimed[int(su_type)] = selected
        pool[int(su_type)] = list(pool.get(int(su_type), [])[int(count):])
    return claimed


def _deficit(pool: Dict[int, List[int]], required: Dict[int, int]) -> Dict[int, int]:
    return {
        int(su): int(count) - len(pool.get(int(su), []))
        for su, count in required.items()
        if int(count) > len(pool.get(int(su), []))
    }


def _contract_id(contract: Dict[str, Any]) -> int:
    value = dict(contract or {}).get("contract_id")
    if value is None:
        raise ValueError(f"canonical topology contract is missing contract_id: {contract}")
    return int(value)


def _closure_contract(contract: Dict[str, Any], phase: str) -> TopologyClosureContract:
    data = dict(contract or {})
    contract_kind = str(data.get("contract_kind", "") or "")
    target_role = str(data.get("target_role", "") or "")
    if not contract_kind or not target_role:
        raise ValueError(f"non-canonical topology contract: {data}")
    mainline_slots = [
        normalize_slot(value)
        for value in list(data.get("mainline_slots", []) or [])
    ]
    branch_slots = [normalize_slot(value) for value in list(data.get("branch_slots", []) or [])]
    required = {
        int(key): int(value)
        for key, value in dict(data.get("required_counts", {}) or {}).items()
        if int(value) > 0
    }
    claimed_source_ids = [
        int(value)
        for value in list(data.get("claimed_source_node_ids", []) or [])
    ]
    return TopologyClosureContract(
        contract_id=_contract_id(data),
        contract_kind=str(contract_kind),
        phase=str(phase),
        priority=int(contract_priority(data)[0]),
        target_role=str(target_role),
        source_fragment_ids=[int(value) for value in list(data.get("source_fragment_ids", []) or [])],
        source_fragment_classes=[str(value) for value in list(data.get("source_fragment_classes", []) or [])],
        source_node_ids=list(claimed_source_ids),
        claimed_source_node_ids=list(claimed_source_ids),
        context_node_ids=[int(value) for value in list(data.get("context_node_ids", []) or [])],
        mainline_slots=mainline_slots,
        branch_slots=branch_slots,
        slot_requirements=[slot.to_dict() for slot in mainline_slots + branch_slots],
        required_counts={int(key): int(value) for key, value in required.items()},
        required_roles=[],
        expected_topology=str(data.get("expected_topology", "") or ""),
        reason=str(data.get("reason", "") or ""),
        blocker_kind=str(data.get("blocker_kind", "") or ""),
        diagnostics=dict(data.get("diagnostics", {}) or {}),
    )


def _external_contracts(phase_eval: Any, nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    del nodes
    contracts = [
        dict(item or {})
        for item in list(getattr(phase_eval, "canonical_contracts", []) or [])
        if dict(item or {})
    ]
    return sorted(merge_stage3_contracts_by_owner(contracts), key=contract_priority)


def _partition_active_external_contracts(
    phase_eval: Any,
    contracts: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    all_contracts = [dict(contract or {}) for contract in list(contracts or [])]
    active_phase = str(getattr(phase_eval, "active_phase", "") or "")
    active_role = next(
        (
            str(role)
            for phase_name, role in EXTERNAL_ROLE_ORDER
            if str(phase_name) == active_phase
        ),
        "",
    )
    if not active_role:
        return all_contracts, [], active_phase, ""
    active = [
        contract
        for contract in all_contracts
        if str(contract.get("target_role", "") or "") == active_role
    ]
    dormant = [
        contract
        for contract in all_contracts
        if str(contract.get("target_role", "") or "") != active_role
    ]
    return active, dormant, active_phase, active_role


def _external_product_materialization_errors(phase_eval: Any) -> List[Dict[str, Any]]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    errors: List[Dict[str, Any]] = []
    for index, raw in enumerate(list(assembly.get("products", []) or [])):
        product = dict(raw or {})
        reasons: List[str] = []
        if not bool(product.get("closed", False)):
            reasons.append("product_not_closed")
        if bool(product.get("provisional", False)):
            reasons.append("provisional_product_forbidden")
        topology = str(
            product.get("topology_notation", product.get("actual_topology_su", ""))
            or ""
        )
        node_topology = str(
            product.get("node_id_notation", product.get("actual_topology_gid", ""))
            or ""
        )
        if not topology or "OPEN" in topology.upper():
            reasons.append("topology_not_fully_materialized")
        if not node_topology or "OPEN" in node_topology.upper():
            reasons.append("node_identity_not_fully_materialized")
        if reasons:
            errors.append({
                "product_index": int(index),
                "product_id": str(product.get("product_id", "") or ""),
                "product_type": str(
                    product.get("product_type", product.get("product_kind", "")) or ""
                ),
                "reasons": reasons,
                "product": product,
            })
    return errors


def build_su24_resource_role_profiles(nodes: Sequence[Any], pool: Dict[int, List[int]]) -> List[SU24ResourceRoleProfile]:
    return build_raw_su24_resource_profiles(nodes, pool)


def _pool_su24_ring_conversion_contract(
    attempt: Dict[str, Any],
    *,
    contract_id: int = 1_000_000,
) -> TopologyClosureContract:
    generic_sequence = [
        int(value)
        for value in list(attempt.get("generic_required_sequence", []) or [])
    ]
    source_ids = sorted({
        int(value)
        for value in list(attempt.get("selected_source_node_ids", []) or [])
    })
    slots = []
    if generic_sequence:
        slots.append(TopologyClosureSlot(
            slot_id=f"pool_su24:{int(contract_id)}:support",
            kind="RESOURCE_PATH",
            anchor_node_id=int(source_ids[-1]) if source_ids else -1,
            anchor_role="ring_support",
            required_su_sequence=list(generic_sequence),
        ))
    return TopologyClosureContract(
        contract_id=int(contract_id),
        contract_kind="ring_partner",
        phase="resource_su24_ring_closure",
        priority=int(attempt.get("template_priority", 0) or 0),
        target_role="fatty_ring",
        source_node_ids=list(source_ids),
        claimed_source_node_ids=list(source_ids),
        mainline_slots=slots,
        required_counts={
            int(key): int(value)
            for key, value in dict(attempt.get("required_counts", {}) or {}).items()
            if int(value) > 0
        },
        expected_topology=str(attempt.get("ring_template", "pool_su24_ring")),
        diagnostics={
            "conversion_only": True,
            "pool_su24_ring_attempt": {
                str(key): value
                for key, value in dict(attempt or {}).items()
                if str(key) != "ordered_segments"
            },
        },
    )


def _residual_contracts(
    pool: Dict[int, List[int]],
    residual_flex_preferred_23: int = 3,
    residual_flex_min_23: int = 2,
    residual_side_preferred_23: int = 2,
    residual_side_min_23: int = 1,
    max_flex_chains: Optional[int] = None,
) -> List[TopologyClosureContract]:
    contracts: List[TopologyClosureContract] = []
    pool_work = {
        int(key): list(values or []) for key, values in dict(pool or {}).items()
    }
    contract_id = 2_000_000
    flex_goal_left = (
        sum(len(values) for values in pool_work.values())
        if max_flex_chains is None
        else max(0, int(max_flex_chains))
    )
    while True:
        selected = None
        for product_role, chain_23_count, required_counts in _residual_template_specs(
            remaining_flex_chains=flex_goal_left,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
        ):
            if not _deficit(pool_work, required_counts):
                selected = (product_role, chain_23_count, required_counts)
                break
        if selected is None:
            break
        product_role, chain_23_count, required_counts = selected
        contract = _residual_product_contract(
            product_role=product_role,
            chain_23_count=chain_23_count,
            required_counts=required_counts,
            contract_id=contract_id,
        )
        contracts.append(contract)
        if _consume(pool_work, required_counts) is None:
            break
        contract_id += 1
        if product_role == "residual_flex_chain":
            flex_goal_left = max(0, int(flex_goal_left) - 1)
    return contracts


def _residual_product_sequence(product_role: str, chain_23_count: int) -> List[int]:
    if str(product_role) == "residual_flex_chain":
        return [11] + [23] * int(chain_23_count) + [11]
    return [11] + [23] * int(chain_23_count) + [22]


def _residual_product_topology(product_role: str, chain_23_count: int) -> str:
    return "-".join(str(value) for value in _residual_product_sequence(product_role, chain_23_count))


def _residual_template_specs(
    *,
    remaining_flex_chains: int,
    residual_flex_preferred_23: int,
    residual_flex_min_23: int,
    residual_side_preferred_23: int,
    residual_side_min_23: int,
    remaining_flex_capacity: Optional[int] = None,
) -> List[Tuple[str, int, Dict[int, int]]]:
    requirements: List[Tuple[str, int, Dict[int, int]]] = []
    allow_flex = bool(
        int(remaining_flex_chains) > 0
        or (
            remaining_flex_capacity is not None
            and int(remaining_flex_capacity) > 0
        )
    )
    if allow_flex:
        preferred_flex_23 = max(int(residual_flex_min_23), int(residual_flex_preferred_23))
        # The configured minimum controls the normal preference, not topology
        # legality. One SU23 is still a valid last-resort flex chain.
        for chain_nodes in range(int(preferred_flex_23), 0, -1):
            requirements.append((
                "residual_flex_chain",
                int(chain_nodes),
                {11: 2, 23: int(chain_nodes)},
            ))

    preferred_side_23 = max(int(residual_side_min_23), int(residual_side_preferred_23))
    # 11-22 is a complete side chain and must remain available when no SU23
    # is left in the residual pool.
    for chain_nodes in range(int(preferred_side_23), -1, -1):
        requirements.append((
            "residual_side_chain",
            int(chain_nodes),
            {11: 1, 22: 1, 23: int(chain_nodes)},
        ))
    return requirements


def _residual_product_contract(
    *,
    product_role: str,
    chain_23_count: int,
    required_counts: Dict[int, int],
    contract_id: int,
) -> TopologyClosureContract:
    sequence = _residual_product_sequence(product_role, chain_23_count)
    return TopologyClosureContract(
        contract_id=int(contract_id),
        contract_kind="residual_conversion",
        phase="residual_resource_self_closure",
        priority=0 if product_role == "residual_flex_chain" else 1,
        target_role="residual_resource_conversion",
        mainline_slots=[TopologyClosureSlot(
            slot_id=f"residual:{int(contract_id)}:mainline",
            kind="RESOURCE_PATH",
            anchor_node_id=-1,
            anchor_role="resource_path",
            required_su_sequence=list(sequence),
        )],
        required_counts=dict(required_counts),
        expected_topology=_residual_product_topology(product_role, chain_23_count),
        diagnostics={
            "product_role": str(product_role),
            "chain_23_count": int(chain_23_count),
            "residual_stepwise": True,
        },
    )


def _plan_next_residual_product(
    *,
    phase_eval: Any,
    iteration_ledger: Any,
    H: torch.Tensor,
    E_target: Any,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    remaining_flex_chains: int,
    residual_flex_preferred_23: int,
    residual_flex_min_23: int,
    residual_side_preferred_23: int,
    residual_side_min_23: int,
    max_conversions: int,
    max_h_compensation_ops: int,
    contract_id: int,
    remaining_flex_capacity: Optional[int] = None,
    prefix_groups: Optional[Sequence[Any]] = None,
    accounting_initial_pool: Optional[Dict[int, List[int]]] = None,
    forbidden_primary_conversion_signatures: Optional[Sequence[str]] = None,
    allow_h_outside_nonworsening: bool = False,
) -> Dict[str, Any]:
    pool = deepcopy(getattr(iteration_ledger, "available_ids_by_su", {}) or {})
    required_flex = int(remaining_flex_chains) > 0
    template_specs = _residual_template_specs(
        remaining_flex_chains=remaining_flex_chains,
        residual_flex_preferred_23=residual_flex_preferred_23,
        residual_flex_min_23=residual_flex_min_23,
        residual_side_preferred_23=residual_side_preferred_23,
        residual_side_min_23=residual_side_min_23,
        remaining_flex_capacity=remaining_flex_capacity,
    )
    if required_flex:
        template_specs = [
            item for item in template_specs if item[0] == "residual_flex_chain"
        ]
    attempts: List[Dict[str, Any]] = []

    # First inspect every legal template against the unchanged ledger. A
    # preferred long template must never trigger conversions before a shorter
    # template that can already be assembled from real residual resources.
    direct_candidates: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    for template_index, (product_role, chain_23_count, required_counts) in enumerate(template_specs):
        contract = _residual_product_contract(
            product_role=product_role,
            chain_23_count=chain_23_count,
            required_counts=required_counts,
            contract_id=int(contract_id),
        )
        updated_ledger, transaction, direct_proof = _ledger_transaction(
            ledger=iteration_ledger,
            contract=contract,
            phase="residual_resource_self_closure",
            H=H,
            special_degree_meta=special_degree_meta,
        )
        if transaction is not None:
            direct_candidates.append((
                (
                    -sum(int(value) for value in required_counts.values()),
                    0 if product_role == "residual_flex_chain" else 1,
                    int(template_index),
                ),
                {
                    "mode": "direct",
                    "ledger": updated_ledger,
                    "transaction": transaction,
                    "product_role": product_role,
                    "chain_23_count": int(chain_23_count),
                    "required_counts": dict(required_counts),
                },
            ))
            continue
        attempts.append({
            "product_role": product_role,
            "chain_23_count": int(chain_23_count),
            "required_counts": dict(required_counts),
            "reason": str(direct_proof.get("reason", "direct_resource_shortage") or "direct_resource_shortage"),
            "mode": "direct",
        })

    if direct_candidates:
        _, selected = min(direct_candidates, key=lambda item: item[0])
        selected["attempts"] = list(attempts)
        selected["selection_reason"] = "all_direct_templates_checked_before_conversion"
        return selected

    total_available = sum(
        len(pool.get(int(su_type), []) or [])
        for su_type in (11, 22, 23)
    )
    conversion_candidates: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []
    target_h: Optional[float] = None
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    h_lower, h_upper = h_hard_bounds(target_h, h_tolerance)

    for template_index, (product_role, chain_23_count, required_counts) in enumerate(template_specs):
        direct_coverage = sum(
            min(len(pool.get(int(su_type), []) or []), int(count))
            for su_type, count in required_counts.items()
        )
        template_size = sum(int(value) for value in required_counts.values())
        if not required_flex and int(direct_coverage) <= 0:
            attempts.append({
                "product_role": product_role,
                "chain_23_count": int(chain_23_count),
                "required_counts": dict(required_counts),
                "reason": "optional_template_has_no_existing_matching_resource",
                "mode": "conversion",
            })
            continue
        if sum(required_counts.values()) > int(total_available):
            attempts.append({
                "product_role": product_role,
                "chain_23_count": int(chain_23_count),
                "required_counts": dict(required_counts),
                "reason": "insufficient_total_residual_nodes",
                "mode": "conversion",
            })
            continue
        contract = _residual_product_contract(
            product_role=product_role,
            chain_23_count=chain_23_count,
            required_counts=required_counts,
            contract_id=int(contract_id),
        )
        candidate_plan = _transaction_plan(
            phase_eval=phase_eval,
            contract=contract,
            pool=pool,
            H=H,
            E_target=E_target,
            max_primary_adjustments=max_conversions,
            max_h_compensation_ops=max_h_compensation_ops,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
            iteration_ledger=iteration_ledger,
            prefix_groups=prefix_groups,
            accounting_initial_pool=accounting_initial_pool,
            forbidden_primary_conversion_signatures=forbidden_primary_conversion_signatures,
            allow_h_outside_nonworsening=allow_h_outside_nonworsening,
            protected_resource_counts={
                11: 2 * max(0, int(remaining_flex_chains)),
                23: max(0, int(remaining_flex_chains)),
            },
        )
        attempts.append({
            "product_role": product_role,
            "chain_23_count": int(chain_23_count),
            "required_counts": dict(required_counts),
            "reason": str(candidate_plan.reason),
            "blocked_reason": str(candidate_plan.blocked_reason or ""),
            "unresolved_counts": dict(candidate_plan.unresolved_counts or {}),
            "mode": "conversion",
        })
        if candidate_plan.ok:
            histogram = candidate_plan.histogram_adjustment_plan
            primary_count = int(getattr(histogram, "primary_adjustment_count", 0) or 0)
            h_after = _h_total(histogram.H_after, special_degree_meta)
            h_before = _h_total(H, special_degree_meta)
            h_after_primary = (
                float(histogram.groups[-1].h_after_primary)
                if list(histogram.groups or [])
                else float(h_after)
            )
            if h_lower is None or h_upper is None:
                h_violation = 0.0
            elif float(h_after) < float(h_lower):
                h_violation = float(h_lower) - float(h_after)
            elif float(h_after) > float(h_upper):
                h_violation = float(h_after) - float(h_upper)
            else:
                h_violation = 0.0
            if h_lower is None or h_upper is None:
                h_violation_before = 0.0
            elif float(h_before) < float(h_lower):
                h_violation_before = float(h_lower) - float(h_before)
            elif float(h_before) > float(h_upper):
                h_violation_before = float(h_before) - float(h_upper)
            else:
                h_violation_before = 0.0
            h_error = (
                abs(float(h_after) - float(target_h))
                if target_h is not None
                else 0.0
            )
            optional_h_invalid = bool(
                not required_flex
                and h_lower is not None
                and h_upper is not None
                and (
                    float(h_violation) > float(h_violation_before) + 1e-12
                    if bool(allow_h_outside_nonworsening)
                    else float(h_violation) > 1e-12
                )
            )
            missing_slot_count = max(0, int(template_size) - int(direct_coverage))
            if not required_flex and (
                int(primary_count) > int(missing_slot_count)
                or bool(optional_h_invalid)
            ):
                attempts.append({
                    "product_role": product_role,
                    "chain_23_count": int(chain_23_count),
                    "required_counts": dict(required_counts),
                    "reason": (
                        "optional_conversion_violates_h_hard_constraint"
                        if optional_h_invalid
                        else "optional_template_uses_excess_conversions"
                    ),
                    "mode": "conversion",
                    "h_before": float(h_before),
                    "h_after_primary": float(h_after_primary),
                    "h_after": float(h_after),
                })
                continue
            if required_flex:
                score = (
                    float(primary_count),
                    float(h_violation),
                    float(h_error),
                    float(-direct_coverage),
                    float(-int(chain_23_count)),
                    float(template_index),
                )
            else:
                score = (
                    float(primary_count),
                    float(h_violation),
                    float(h_error),
                    float(-direct_coverage),
                    float(-template_size),
                    float(template_index),
                )
            candidate_plan.plan_kind = "residual_single_product_conversion"
            candidate_plan.diagnostics.update({
                "residual_product_role": product_role,
                "residual_chain_23_count": int(chain_23_count),
                "residual_required_counts": dict(required_counts),
                "residual_candidate_attempts": list(attempts),
                "residual_candidate_score": list(score),
                "residual_selection_rule": (
                    "minimum_conversion_then_h_then_existing_resource_use"
                    if required_flex
                    else "single_missing_slot_then_h_then_existing_resource_use"
                ),
            })
            conversion_candidates.append((score, {
                "mode": "conversion",
                "plan": candidate_plan,
                "product_role": product_role,
                "chain_23_count": int(chain_23_count),
                "required_counts": dict(required_counts),
            }))
    if conversion_candidates:
        _, selected = min(conversion_candidates, key=lambda item: item[0])
        selected["attempts"] = list(attempts)
        selected["selection_reason"] = (
            "ranked_required_flex_conversion_candidates"
            if required_flex
            else "ranked_single_gap_optional_conversion_candidates"
        )
        return selected
    return {"mode": "none", "attempts": attempts}


def _attach_conversion_only_accounting(
    *,
    phase_eval: Any,
    histogram: HistogramAdjustmentPlan,
    initial_pool: Dict[int, List[int]],
    final_pool: Dict[int, List[int]],
    committed_transactions: Optional[Sequence[TopologyResourceTransaction]] = None,
) -> Dict[str, Any]:
    aggregate_ids = _round_resource_ids(phase_eval)
    # A pending transaction can restore real nodes that a fresh classifier
    # temporarily claimed as fragment core. They remain part of this round's
    # physical inventory and must be visible to the accounting audit.
    for su_type, node_ids in dict(initial_pool or {}).items():
        aggregate_ids.setdefault(int(su_type), [])
        aggregate_ids[int(su_type)] = sorted({
            *[int(value) for value in aggregate_ids[int(su_type)]],
            *[int(value) for value in list(node_ids or [])],
        })
    groups = list(histogram.groups or [])
    transactions = (
        list(committed_transactions)
        if committed_transactions is not None
        else _assembly_product_transactions(phase_eval, histogram.H_before, None)
    )
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_ids,
        authorized_ids_before_by_su=initial_pool,
        final_remaining_ids_by_su=final_pool,
        committed_claimed_ids_by_su=_committed_occupied_ids_by_su(
            transactions,
            aggregate_ids_by_su=aggregate_ids,
            groups=groups,
        ),
        groups=groups,
    )
    histogram.inventory_before = {
        int(key): int(value)
        for key, value in dict(
            accounting.get("authorized_inventory_before", {}) or {}
        ).items()
    }
    histogram.inventory_after = {
        int(key): int(value)
        for key, value in dict(
            accounting.get("authorized_inventory_after", {}) or {}
        ).items()
    }
    histogram.aggregate_inventory_before = {
        int(key): int(value)
        for key, value in dict(
            accounting.get("aggregate_inventory_before", {}) or {}
        ).items()
    }
    histogram.aggregate_inventory_after = {
        int(key): int(value)
        for key, value in dict(
            accounting.get("aggregate_inventory_after", {}) or {}
        ).items()
    }
    histogram.aggregate_required_counts = {
        int(key): int(value)
        for key, value in dict(
            accounting.get("aggregate_required_counts", {}) or {}
        ).items()
    }
    histogram.protected_resource_counts = _phase_protected_resource_counts(phase_eval)
    histogram.accounting_diagnostics = dict(accounting)
    return accounting


def _residual_su24_aromatization_plan(
    *,
    phase_eval: Any,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    max_conversions: int,
    prefix_transactions: Optional[Sequence[TopologyResourceTransaction]] = None,
) -> Optional[Stage3AdjustmentPlan]:
    limit = max(0, int(max_conversions))
    operations: List[Stage3AdjustmentOp] = []
    cursor = int(su11_rotation_index)
    for node_id in list(pool.get(24, []) or []):
        if len(operations) >= limit:
            break
        target, cursor_after = next_su11_aromatic_target(cursor)
        if target is None:
            break
        operations.append(Stage3AdjustmentOp(
            op="convert",
            from_su=24,
            to_su=int(target),
            node_id=int(node_id),
            reason="residual_su24_aromatic_rotation",
        ))
        cursor = int(cursor_after)
    if not operations:
        return None
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="residual_su24_aromatization_pending",
        phase="residual_su24_aromatization",
        plan_kind="residual_su24_aromatization",
        active_subphase="residual_su24_aromatization",
        planned_conversions=operations,
        runtime=_runtime_with_pool(phase_eval, pool),
    )
    histogram = build_histogram_adjustment_plan(
        plan=plan,
        H=H,
        # Residual SU24 terminalization is topology-mandated.  It deliberately
        # bypasses the Stage3 H hard bound and never creates compensation ops.
        E_target=None,
        max_primary_adjustments=len(operations),
        max_h_compensation_ops=0,
        h_tolerance=1.0,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    if not list(histogram.groups or []):
        return None
    pool_after = _bind_real_node_ids(histogram, pool)
    accounting = _attach_conversion_only_accounting(
        phase_eval=phase_eval,
        histogram=histogram,
        initial_pool=pool,
        final_pool=pool_after,
        committed_transactions=prefix_transactions,
    )
    if not bool(accounting.get("ok", False)):
        return None
    all_ops = [
        op
        for group in list(histogram.groups or [])
        for op in [group.primary_op] + list(group.h_compensation_ops or [])
    ]
    transaction = TopologyResourceTransaction(
        phase="residual_su24_aromatization",
        contract_id=2_900_000,
        target_role="residual_su24_aromatization",
        converted_node_ids=[
            int(value) for op in all_ops for value in list(op.node_ids or [])
        ],
        conversion_path=[op.to_dict() for op in all_ops],
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=float(histogram.groups[0].h_after_primary),
        h_after=_h_total(histogram.H_after, special_degree_meta),
        commit_scope="physical_histogram",
        committed=True,
    )
    plan.ok = True
    plan.reason = "stage3_rebuild_required"
    plan.histogram_adjustment_plan = histogram
    plan.phase_result = Stage3PhaseResult(
        active_phase="residual_su24_aromatization",
        phase_completed=False,
        rebuild_required=True,
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        transactions=[transaction],
        stop_reason="stage3_rebuild_required",
    )
    h_before = _h_total(H, special_degree_meta)
    h_after = _h_total(histogram.H_after, special_degree_meta)
    plan.diagnostics = {
        "stage3_phase": "residual_su24_aromatization",
        "terminalization": "all_remaining_su24_via_12_13_13_rotation",
        "terminal_h_exemption": True,
        "h_before": float(h_before),
        "h_after": float(h_after),
        "h_delta": float(h_after - h_before),
        "converted_su24_node_ids": [
            int(op.node_id) for op in operations if op.node_id is not None
        ],
        "remaining_su24_count": len(list(pool_after.get(24, []) or [])),
        "remaining_su24_node_ids": [
            int(value) for value in list(pool_after.get(24, []) or [])
        ],
        "primary_conversion_limit": int(limit),
        "su11_rotation_index_after": int(histogram.su11_rotation_index_after),
        "resource_accounting": dict(accounting),
    }
    histogram.accounting_diagnostics.update({
        "terminal_h_exemption": True,
        "h_before": float(h_before),
        "h_after": float(h_after),
    })
    return plan


def _flex_excess_reduction_plan(
    *,
    phase_eval: Any,
    H: torch.Tensor,
    E_target: Any,
    max_primary_adjustments: int,
    max_h_compensation_ops: int,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
) -> Stage3AdjustmentPlan:
    audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
    pool = _resource_ids(phase_eval)
    excess = max(0, int(audit.get("above_range_excess", 0) or 0))
    limit = max(0, int(max_primary_adjustments))
    operations: List[Stage3AdjustmentOp] = []
    target_h = None
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    inventory_work = {
        int(su_type): len(list(pool.get(int(su_type), []) or []))
        for su_type in (10, 11, 12, 13, 22, 23, 24)
    }
    H_work = H.detach().clone().long()
    used_node_ids: Set[int] = set()
    selected_excess_units = 0
    while selected_excess_units < int(excess) and len(operations) < int(limit):
        candidate_bundles: List[Tuple[List[HistogramConversion], int]] = []
        for node_id in list(pool.get(12, []) or []):
            if int(node_id) in used_node_ids or len(operations) + 1 > int(limit):
                continue
            candidate_bundles.append((
                [HistogramConversion(
                    from_su=12,
                    to_su=11,
                    count=1,
                    reason="flex_excess_reduce_su12_to_su11",
                    node_ids=[int(node_id)],
                )],
                1,
            ))
        su10_ids = [
            int(node_id)
            for node_id in list(pool.get(10, []) or [])
            if int(node_id) not in used_node_ids
        ]
        for pair_index in range(len(su10_ids) // 2):
            if len(operations) + 2 > int(limit):
                break
            pair_ids = su10_ids[2 * pair_index:2 * pair_index + 2]
            candidate_bundles.append((
                [
                    HistogramConversion(
                        from_su=10,
                        to_su=11,
                        count=1,
                        reason="flex_excess_reduce_su10_pair_to_su11",
                        node_ids=[int(node_id)],
                    )
                    for node_id in pair_ids
                ],
                1,
            ))
        if not candidate_bundles:
            break
        context = ConversionContext(
            inventory=dict(inventory_work),
            demand={},
            current_H=H_work,
            target_h=target_h,
            h_tolerance=float(h_tolerance),
            flex_audit=dict(audit or {}),
            su11_rotation_index=int(su11_rotation_index),
            phase="flex_range_reconciliation",
            reason="flex_excess_reduction",
        )
        scored = []
        for bundle, unit_count in candidate_bundles:
            score = score_conversion_bundle(
                context,
                bundle,
                h_total_fn=_h_total,
                special_degree_meta=special_degree_meta,
            )
            if score.feasible and h_within_context_hard_bounds(
                context, score.h_after, score.h_before
            ):
                scored.append((score.sort_key, bundle, int(unit_count), score))
        if not scored:
            break
        _, selected_bundle, unit_count, _ = min(scored, key=lambda item: item[0])
        next_H = apply_conversions_to_H(H_work, selected_bundle)
        next_inventory = apply_conversions_to_inventory(inventory_work, selected_bundle)
        if next_H is None or next_inventory is None:
            break
        H_work = next_H
        inventory_work = next_inventory
        selected_excess_units += int(unit_count)
        for conversion in selected_bundle:
            used_node_ids.update(int(value) for value in list(conversion.node_ids or []))
            operations.append(Stage3AdjustmentOp(
                op="convert",
                from_su=int(conversion.from_su),
                to_su=int(conversion.to_su),
                node_id=int(conversion.node_ids[0]) if list(conversion.node_ids or []) else None,
                reason=str(conversion.reason),
                via_su=conversion.via_su,
            ))

    proof = {
        "phase": "flex_range_reconciliation",
        "reason": "flex_excess_reduction_resource_infeasible",
        "flex_excess": int(excess),
        "available_donor_ids_by_su": {
            "10": list(pool.get(10, []) or []),
            "12": list(pool.get(12, []) or []),
        },
        "attempted_conversions": [item.to_dict() for item in operations],
        "primary_conversion_limit": int(limit),
    }
    if not operations:
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="flex_range_reconciliation",
            plan_kind="flex_excess_reduction_infeasible",
            active_subphase="flex_range_reconciliation",
            phase_result=Stage3PhaseResult(
                active_phase="flex_range_reconciliation",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={"stage3_phase": "flex_range_reconciliation", "infeasibility_proof": proof},
            runtime=_runtime_with_pool(phase_eval, pool),
        ), h_tolerance)

    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="flex_excess_reduction_pending",
        phase="flex_range_reconciliation",
        plan_kind="flex_excess_reduction",
        active_subphase="flex_range_reconciliation",
        planned_conversions=list(operations),
        runtime=_runtime_with_pool(phase_eval, pool),
    )
    histogram = build_histogram_adjustment_plan(
        plan=plan,
        H=H,
        E_target=E_target,
        max_primary_adjustments=limit,
        max_h_compensation_ops=max_h_compensation_ops,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    if not list(histogram.groups or []):
        proof["reason"] = "flex_excess_reduction_h_or_element_infeasible"
        proof["histogram_unresolved_counts"] = dict(histogram.unresolved_counts or {})
        plan.reason = "phase_resource_infeasible"
        plan.blocked_reason = str(proof["reason"])
        plan.phase_result = Stage3PhaseResult(
            active_phase="flex_range_reconciliation",
            phase_completed=False,
            rebuild_required=False,
            stop_reason="phase_resource_infeasible",
            infeasibility_proof=proof,
        )
        plan.diagnostics = {
            "stage3_phase": "flex_range_reconciliation",
            "infeasibility_proof": proof,
        }
        return _annotate_h_hard_rule(plan, h_tolerance)

    all_ops = [
        op
        for group in list(histogram.groups or [])
        for op in [group.primary_op] + list(group.h_compensation_ops or [])
    ]
    pool_after = _bind_real_node_ids(histogram, pool)
    accounting = _attach_conversion_only_accounting(
        phase_eval=phase_eval,
        histogram=histogram,
        initial_pool=pool,
        final_pool=pool_after,
    )
    if not bool(accounting.get("ok", False)):
        proof["reason"] = "stage3_resource_accounting_invariant_failed"
        proof["resource_accounting"] = dict(accounting)
        plan.reason = "phase_resource_infeasible"
        plan.blocked_reason = str(proof["reason"])
        plan.histogram_adjustment_plan = histogram
        plan.phase_result = Stage3PhaseResult(
            active_phase="flex_range_reconciliation",
            phase_completed=False,
            rebuild_required=False,
            stop_reason="phase_resource_infeasible",
            infeasibility_proof=proof,
        )
        plan.diagnostics = {
            "stage3_phase": "flex_range_reconciliation",
            "infeasibility_proof": proof,
        }
        return _annotate_h_hard_rule(plan, h_tolerance)
    transaction = TopologyResourceTransaction(
        phase="flex_range_reconciliation",
        contract_id=-1002,
        target_role="flex_excess_reduction",
        converted_node_ids=[
            int(value) for op in all_ops for value in list(op.node_ids or [])
        ],
        conversion_path=[op.to_dict() for op in all_ops],
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=float(histogram.groups[0].h_after_primary),
        h_after=_h_total(histogram.H_after, special_degree_meta),
        commit_scope="physical_histogram",
        committed=True,
    )
    plan.ok = True
    plan.reason = "stage3_rebuild_required"
    plan.histogram_adjustment_plan = histogram
    plan.planned_conversions = [
        Stage3AdjustmentOp(
            op="convert",
            from_su=int(op.from_su),
            to_su=int(op.to_su),
            node_id=int(op.node_ids[0]) if list(op.node_ids or []) else None,
            reason=str(op.reason),
            via_su=op.via_su,
        )
        for op in all_ops
    ]
    plan.phase_result = Stage3PhaseResult(
        active_phase="flex_range_reconciliation",
        phase_completed=False,
        rebuild_required=True,
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        transactions=[transaction],
        stop_reason="stage3_rebuild_required",
    )
    plan.diagnostics = {
        "stage3_phase": "flex_range_reconciliation",
        "flex_excess_before": int(excess),
        "primary_conversion_limit": int(limit),
        "su11_rotation_index_after": int(histogram.su11_rotation_index_after),
        "resource_accounting": dict(accounting),
    }
    return _annotate_h_hard_rule(plan, h_tolerance)


def _runtime_with_pool(
    phase_eval: Any,
    pool: Dict[int, List[int]],
    *,
    protected_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    assembly = deepcopy(dict(getattr(phase_eval, "assembly_view", {}) or {}))
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    ledger["remaining_ids_by_su"] = {str(key): list(values) for key, values in pool.items()}
    ledger["remaining_counts"] = {str(key): len(values) for key, values in pool.items()}
    ledger["remaining_counts_by_su"] = dict(ledger["remaining_counts"])
    ledger["convertible_counts"] = dict(ledger["remaining_counts"])
    ledger["su11_donor_ids_by_su"] = {
        str(su_type): list(pool.get(int(su_type), []) or [])
        for su_type in (10, 12, 13)
    }
    ledger["su11_donor_counts"] = {
        str(su_type): len(list(pool.get(int(su_type), []) or []))
        for su_type in (10, 12, 13)
    }
    merged_protected = {
        int(key): max(0, int(value))
        for key, value in dict(ledger.get("protected_counts", {}) or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    for key, value in dict(protected_counts or {}).items():
        merged_protected[int(key)] = max(
            int(merged_protected.get(int(key), 0)),
            max(0, int(value)),
        )
    ledger["protected_counts"] = {
        str(key): int(value)
        for key, value in sorted(merged_protected.items())
        if int(value) > 0
    }
    assembly["resource_ledger"] = ledger
    return {
        "phase_evaluation": phase_eval,
        "assembly_before_adjustment": assembly,
    }


def _protect_future_su11_aromatic_donors(
    pool: Dict[int, List[int]],
    *,
    current_required_11: int,
    future_required_11: int,
    aromatic_cursor: int,
) -> Dict[int, int]:
    available_11_after_current = max(
        0,
        len(list(pool.get(11, []) or [])) - max(0, int(current_required_11)),
    )
    future_donor_need = max(
        0,
        int(future_required_11) - int(available_11_after_current),
    )
    current_11_gap = max(
        0,
        int(current_required_11) - len(list(pool.get(11, []) or [])),
    )
    available = {
        12: len(list(pool.get(12, []) or [])),
        13: len(list(pool.get(13, []) or [])),
    }
    aromatic_total = sum(available.values())
    protected_total = min(
        int(future_donor_need),
        max(0, int(aromatic_total) - int(current_11_gap)),
    )
    spendable_total = max(0, int(aromatic_total) - int(protected_total))
    spendable = {12: 0, 13: 0}
    remaining = dict(available)
    cursor = int(aromatic_cursor)
    for _ in range(int(spendable_total)):
        target, cursor_after = next_su11_aromatic_target(
            cursor,
            available_targets=[
                su_type for su_type in (12, 13)
                if int(remaining.get(int(su_type), 0)) > 0
            ],
        )
        if target is None:
            break
        spendable[int(target)] += 1
        remaining[int(target)] -= 1
        cursor = int(cursor_after)
    return {
        int(su_type): int(available[int(su_type)] - spendable[int(su_type)])
        for su_type in (12, 13)
        if int(available[int(su_type)] - spendable[int(su_type)]) > 0
    }


def _iteration_ledger_for_phase_eval(
    phase_eval: Any,
    nodes: Sequence[Any],
    pool: Dict[int, List[int]],
    contracts: Sequence[Dict[str, Any]],
) -> Any:
    classification = dict(getattr(phase_eval, "classification_view", {}) or {})
    support_ids = [
        int(dict(item or {}).get("node_id"))
        for item in list(classification.get("support_units_only", []) or [])
        if dict(item or {}).get("node_id") is not None
    ]
    claimed_source_ids = [
        int(value)
        for contract in list(contracts or [])
        for value in list(dict(contract or {}).get("claimed_source_node_ids", []) or [])
    ]
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su=pool,
        support_context_node_ids=support_ids,
        claimed_source_ids=claimed_source_ids,
    )
    round_context = getattr(phase_eval, "round_context", None)
    round_ownership = dict(
        getattr(round_context, "ownership_by_node_id", {}) or {}
    )
    if round_ownership:
        ledger.ownership_by_node_id = deepcopy(round_ownership)
        ledger.support_context_node_ids = sorted(
            int(node_id)
            for node_id, owner in round_ownership.items()
            if str(getattr(owner, "state", "") or "")
            == Stage3OwnershipState.SUPPORT_CONTEXT.value
        )
        ledger.conflict_node_ids = sorted(
            int(node_id)
            for node_id, owner in round_ownership.items()
            if str(getattr(owner, "state", "") or "")
            == Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
        )
        ledger.conflicts = deepcopy(
            list(getattr(round_context, "conflicts", []) or [])
        )
    ledger.generation_id = int(getattr(phase_eval, "generation_id", 0) or 0)
    completed_products: List[Dict[str, Any]] = []
    product_ids = set()
    for raw in list(assembly.get("products", []) or []):
        product = dict(raw or {})
        if not product:
            continue
        product_id = str(product.get("product_id", "") or "")
        if product_id and product_id in product_ids:
            continue
        if product_id:
            product_ids.add(product_id)
        completed_products.append(product)
    return seed_ledger_topology_state(
        ledger,
        completed_products=completed_products,
        pending_contracts=list(contracts or []),
    )


def _prioritize_pending_converted_ids(
    ledger: Any,
    converted_node_ids: Sequence[int],
) -> Any:
    working = deepcopy(ledger)
    preferred_order = [
        int(value) for value in list(converted_node_ids or [])
    ]
    for su_type, raw_ids in dict(
        getattr(working, "available_ids_by_su", {}) or {}
    ).items():
        available = [int(value) for value in list(raw_ids or [])]
        preferred = [
            int(node_id)
            for node_id in preferred_order
            if int(node_id) in available
        ]
        preferred_set = set(preferred)
        working.available_ids_by_su[int(su_type)] = preferred + [
            int(node_id)
            for node_id in available
            if int(node_id) not in preferred_set
        ]
    return working


def _reserve_pending_candidate_nodes(
    ledger: Any,
    nodes: Sequence[Any],
    *,
    pending_ring_candidate: Optional[Dict[str, Any]] = None,
    pending_residual_candidate: Optional[Dict[str, Any]] = None,
    pending_terminal_su24_node_ids: Optional[Sequence[int]] = None,
    restore_uncommitted_su24: bool = False,
) -> tuple[Any, Dict[str, Any]]:
    """Restore the current transaction's real nodes to the post-rebuild pool."""
    working = deepcopy(ledger)
    ring = dict(pending_ring_candidate or {})
    residual = dict(pending_residual_candidate or {})
    converted_ids = {
        int(value)
        for candidate in (ring, residual)
        for value in list(candidate.get("converted_node_ids", []) or [])
    }
    selected_ring_ids = {
        int(value)
        for value in list(ring.get("selected_su24_ids", []) or [])
    }
    terminal_su24_ids = {
        int(value) for value in list(pending_terminal_su24_node_ids or [])
    }
    node_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
    }
    uncommitted_su24_ids = (
        {
            int(node_id)
            for node_id, node in node_by_id.items()
            if int(getattr(node, "su_type", -1)) == 24
        }
        if bool(restore_uncommitted_su24)
        else set()
    )
    requested_ids = (
        converted_ids
        | selected_ring_ids
        | terminal_su24_ids
        | uncommitted_su24_ids
    )
    audit: Dict[str, Any] = {
        "requested_node_ids": sorted(requested_ids),
        "converted_node_ids": sorted(converted_ids),
        "selected_su24_node_ids": sorted(selected_ring_ids),
        "terminal_su24_node_ids": sorted(terminal_su24_ids),
        "uncommitted_su24_node_ids": sorted(uncommitted_su24_ids),
        "restored_node_ids": [],
        "missing_node_ids": [],
        "committed_owner_conflicts": [],
        "ownership_before": {},
        "ownership_after": {},
    }
    if not requested_ids:
        audit["ok"] = True
        return working, audit

    for node_id in sorted(requested_ids):
        node = node_by_id.get(int(node_id))
        owner = working.ownership_by_node_id.get(int(node_id))
        if node is None or owner is None:
            audit["missing_node_ids"].append(int(node_id))
            continue
        audit["ownership_before"][str(node_id)] = {
            "su_type": int(getattr(owner, "su_type", -1)),
            "state": str(getattr(owner, "state", "") or ""),
            "owner_fragment_id": getattr(owner, "owner_fragment_id", None),
            "owner_contract_id": getattr(owner, "owner_contract_id", None),
            "owner_product_id": str(getattr(owner, "owner_product_id", "") or ""),
        }
        if str(owner.state) in {
            Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
            Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
        } and str(owner.owner_product_id or ""):
            audit["committed_owner_conflicts"].append({
                "node_id": int(node_id),
                "owner_product_id": str(owner.owner_product_id),
                "state": str(owner.state),
            })
            continue

        su_type = int(getattr(node, "su_type", -1))
        if int(node_id) in (selected_ring_ids | terminal_su24_ids) and su_type != 24:
            audit["missing_node_ids"].append(int(node_id))
            continue
        if su_type not in set(RESOURCE_TYPES) | {10, 12, 13}:
            audit["missing_node_ids"].append(int(node_id))
            continue

        for values in working.available_ids_by_su.values():
            while int(node_id) in values:
                values.remove(int(node_id))
        working.available_ids_by_su.setdefault(int(su_type), [])
        working.available_ids_by_su[int(su_type)].insert(0, int(node_id))
        if int(node_id) in working.support_context_node_ids:
            working.support_context_node_ids.remove(int(node_id))
        if int(node_id) in working.conflict_node_ids:
            working.conflict_node_ids.remove(int(node_id))
        working.conflicts = [
            conflict
            for conflict in list(working.conflicts or [])
            if int(dict(conflict or {}).get("node_id", -1)) != int(node_id)
        ]
        owner.su_type = int(su_type)
        owner.state = (
            Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value
            if su_type in RESOURCE_TYPES
            else Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value
        )
        owner.owner_fragment_id = None
        owner.owner_contract_id = None
        owner.owner_product_id = ""
        owner.owner_slot_id = ""
        owner.role = str(owner.state)
        owner.reason = "reserved_by_pending_stage3_transaction"
        audit["restored_node_ids"].append(int(node_id))
        audit["ownership_after"][str(node_id)] = {
            "su_type": int(su_type),
            "state": str(owner.state),
        }

    audit["ok"] = not bool(
        audit["missing_node_ids"] or audit["committed_owner_conflicts"]
    )
    return working, audit


def _ledger_transaction(
    *,
    ledger: Any,
    contract: TopologyClosureContract,
    phase: str,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
) -> tuple[Any, Optional[TopologyResourceTransaction], Dict[str, Any]]:
    if str(phase) in EXTERNAL_PHASE_NAMES:
        return ledger, None, {
            "reason": "external_contract_requires_physical_assembler",
            "contract_id": int(contract.contract_id),
            "phase": str(phase),
        }
    updated, slots, claimed, proof = assign_contract_slots(ledger, contract)
    if updated is None:
        return ledger, None, dict(proof or {})
    product = dict(updated.completed_products[-1] or {})
    product_node_ids = sorted({
        int(value)
        for value in list(product.get("fixed_source_node_ids", []) or [])
        + [
            int(value)
            for values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).values()
            for value in list(values or [])
        ]
    })
    h_value = _h_total(H, special_degree_meta)
    transaction = TopologyResourceTransaction(
        phase=str(phase),
        contract_id=int(contract.contract_id),
        target_role=str(contract.target_role),
        source_node_ids=list(contract.claimed_source_node_ids or contract.source_node_ids),
        claimed_resource_ids=dict(claimed),
        product_node_ids=product_node_ids,
        slot_assignments=[slot.to_dict() for slot in slots],
        h_before=float(h_value),
        h_after_primary=float(h_value),
        h_after=float(h_value),
        commit_scope="iteration_ledger",
        transaction_origin="current_plan_candidate",
        committed=True,
    )
    return updated, transaction, {}


def _assembly_product_transactions(
    phase_eval: Any,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    completed_products: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[TopologyResourceTransaction]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    h_value = _h_total(H, special_degree_meta)
    transactions: List[TopologyResourceTransaction] = []
    products = (
        list(completed_products or [])
        if completed_products is not None
        else list(assembly.get("products", []) or [])
    )
    for index, raw in enumerate(products):
        product = dict(raw or {})
        product_kind = str(product.get("product_kind", product.get("product_type", "")) or "")
        phase = str(product.get("phase", "") or "") or (
            "residual_resource_self_closure"
            if str(product.get("product_role", "") or "").startswith("residual_")
            else "external_fatty_ring_closure"
            if "ring" in product_kind
            else "external_side_chain_closure"
            if "side" in product_kind
            else "external_flex_closure"
        )
        fragment_ids = [int(value) for value in list(product.get("source_fragment_ids", []) or [])]
        fixed_source_ids = [
            int(value) for value in list(product.get("fixed_source_node_ids", []) or [])
        ]
        claimed_resource_ids = {
            int(key): [int(value) for value in list(values or [])]
            for key, values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).items()
            if str(key).lstrip("-").isdigit()
        }
        product_ids = fixed_source_ids + [
            int(value) for values in claimed_resource_ids.values() for value in values
        ]
        transactions.append(TopologyResourceTransaction(
            phase=phase,
            contract_id=int(product.get("contract_id", min(fragment_ids or [index + 1])) or index + 1),
            target_role=str(product.get("target_role", "") or product_kind),
            source_node_ids=fixed_source_ids,
            claimed_resource_ids=claimed_resource_ids,
            product_node_ids=sorted(set(product_ids)),
            slot_assignments=(
                [dict(item or {}) for item in list(product.get("mainline_slots", []) or [])]
                + [dict(item or {}) for item in list(product.get("branch_slots", []) or [])]
            ),
            h_before=float(h_value),
            h_after_primary=float(h_value),
            h_after=float(h_value),
            commit_scope="iteration_ledger",
            transaction_origin="assembly_snapshot",
            committed=True,
        ))
    return transactions


def _bind_real_node_ids(
    histogram_plan: Any,
    pool: Dict[int, List[int]],
    *,
    protected_node_ids: Optional[Sequence[int]] = None,
) -> Dict[int, List[int]]:
    work = {int(key): list(values) for key, values in pool.items()}
    protected = {int(value) for value in list(protected_node_ids or [])}
    for group in list(getattr(histogram_plan, "groups", []) or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            candidates = list(work.get(int(op.from_su), []))
            requested = [int(value) for value in list(op.node_ids or [])]
            eligible = [
                int(value) for value in candidates if int(value) not in protected
            ]
            selected = (
                requested[: int(op.count)]
                if len(requested) >= int(op.count)
                and set(requested[: int(op.count)]).issubset(set(eligible))
                else eligible[: int(op.count)]
            )
            op.node_ids = selected
            selected_set = set(selected)
            work[int(op.from_su)] = [
                int(value) for value in candidates if int(value) not in selected_set
            ]
            work.setdefault(int(op.to_su), []).extend(selected)
            work[int(op.to_su)] = sorted(work[int(op.to_su)])
    return work


def _ledger_after_histogram_conversions(
    ledger: Any,
    histogram_plan: Any,
    *,
    protected_node_ids: Optional[Sequence[int]] = None,
) -> Any:
    working = deepcopy(ledger)
    working.available_ids_by_su = _bind_real_node_ids(
        histogram_plan,
        deepcopy(getattr(working, "available_ids_by_su", {}) or {}),
        protected_node_ids=protected_node_ids,
    )
    for group in list(getattr(histogram_plan, "groups", []) or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            for node_id in list(op.node_ids or []):
                owner = working.ownership_by_node_id.get(int(node_id))
                if owner is None:
                    continue
                owner.su_type = int(op.to_su)
                owner.state = (
                    Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value
                    if int(op.to_su) in RESOURCE_TYPES
                    else Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value
                )
                owner.owner_contract_id = None
                owner.owner_product_id = ""
                owner.owner_slot_id = ""
                owner.role = str(owner.state)
    return working


def _round_resource_ids(phase_eval: Any) -> Dict[int, List[int]]:
    context = getattr(phase_eval, "round_context", None)
    if context is None:
        return _resource_ids(phase_eval)
    result: Dict[int, List[int]] = {}
    for raw in (
        getattr(context, "closure_pool_ids_by_su", {}) or {},
        getattr(context, "aromatic_donor_ids_by_su", {}) or {},
    ):
        for key, values in dict(raw or {}).items():
            result.setdefault(int(key), [])
            result[int(key)].extend(int(value) for value in list(values or []))
    return {
        int(key): sorted(set(values))
        for key, values in result.items()
    }


def _post_conversion_resource_types(
    aggregate_ids_by_su: Dict[int, List[int]],
    groups: Sequence[Any],
) -> Dict[int, int]:
    resource_types = {
        int(node_id): int(su_type)
        for su_type, node_ids in dict(aggregate_ids_by_su or {}).items()
        for node_id in list(node_ids or [])
    }
    for group in list(groups or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            for node_id in list(op.node_ids or []):
                if int(node_id) in resource_types:
                    resource_types[int(node_id)] = int(op.to_su)
    return resource_types


def _committed_occupied_ids_by_su(
    transactions: Sequence[TopologyResourceTransaction],
    *,
    aggregate_ids_by_su: Dict[int, List[int]],
    groups: Sequence[Any],
) -> Dict[int, List[int]]:
    occupied: Dict[int, List[int]] = {}
    resource_types = _post_conversion_resource_types(
        aggregate_ids_by_su,
        groups,
    )
    for transaction in list(transactions or []):
        if not bool(getattr(transaction, "committed", False)):
            continue
        transaction_ids = {
            int(value)
            for value in (
                list(getattr(transaction, "source_node_ids", []) or [])
                + list(getattr(transaction, "product_node_ids", []) or [])
                + [
                    int(node_id)
                    for values in dict(
                        getattr(transaction, "claimed_resource_ids", {}) or {}
                    ).values()
                    for node_id in list(values or [])
                ]
            )
            if int(value) in resource_types
        }
        for node_id in sorted(transaction_ids):
            su_type = int(resource_types[int(node_id)])
            occupied.setdefault(su_type, []).append(int(node_id))
    return occupied


def _ledger_occupied_ids_by_su(
    ledger: Any,
    *,
    aggregate_ids_by_su: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    aggregate_ids = {
        int(node_id)
        for node_ids in dict(aggregate_ids_by_su or {}).values()
        for node_id in list(node_ids or [])
    }
    occupied: Dict[int, List[int]] = {}
    for raw_product in list(getattr(ledger, "completed_products", []) or []):
        product = dict(raw_product or {})
        product_ids = {
            int(node_id)
            for node_id in (
                list(product.get("fixed_source_node_ids", []) or [])
                + [
                    int(value)
                    for values in dict(
                        product.get("claimed_resource_ids_by_su", {})
                        or product.get("claimed_resource_ids", {})
                        or {}
                    ).values()
                    for value in list(values or [])
                ]
            )
            if int(node_id) in aggregate_ids
        }
        for node_id in sorted(product_ids):
            owner = getattr(ledger, "ownership_by_node_id", {}).get(int(node_id))
            if owner is None:
                continue
            occupied.setdefault(int(owner.su_type), []).append(int(node_id))
    return occupied


def _phase_protected_resource_counts(phase_eval: Any) -> Dict[int, int]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    diagnostics = dict(assembly.get("diagnostics", {}) or {})
    raw = (
        assembly.get("protected_resource_counts")
        or ledger.get("protected_counts")
        or diagnostics.get("protected_resource_counts")
        or {}
    )
    return {
        int(key): max(0, int(value))
        for key, value in dict(raw or {}).items()
        if str(key).lstrip("-").isdigit()
    }


def _group_operations(groups: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        op.to_dict()
        for group in list(groups or [])
        for op in [group.primary_op] + list(group.h_compensation_ops or [])
    ]


def _h_compensation_count(groups: Sequence[Any]) -> int:
    return sum(len(list(group.h_compensation_ops or [])) for group in list(groups or []))


def _finalize_phase_plan(
    *,
    phase_eval: Any,
    phase: str,
    H_before: torch.Tensor,
    H_after: torch.Tensor,
    initial_pool: Dict[int, List[int]],
    final_pool: Dict[int, List[int]],
    groups: List[Any],
    transactions: List[TopologyResourceTransaction],
    rotation_before: int,
    rotation_after: int,
    h_tolerance: float,
    infeasible_contracts: Optional[List[Dict[str, Any]]] = None,
    stage_diagnostics: Optional[Dict[str, Any]] = None,
) -> Stage3AdjustmentPlan:
    phase_infeasible_contracts = [
        dict(item or {})
        for item in list(infeasible_contracts or [])
    ]
    phase_transactions = [
        item
        for item in list(transactions or [])
        if (
            str(getattr(item, "phase", "") or "") == str(phase)
            or (
                str(phase) == EXTERNAL_STAGE
                and str(getattr(item, "phase", "") or "") in EXTERNAL_PHASE_NAMES
            )
        )
    ]
    requirement_contracts: List[Dict[str, Any]] = []
    for transaction in phase_transactions:
        required_counts = {
            int(key): len(list(values or []))
            for key, values in dict(
                getattr(transaction, "claimed_resource_ids", {}) or {}
            ).items()
            if list(values or [])
        }
        if not required_counts:
            required_counts = {
                int(key): int(value)
                for key, value in required_counts_from_slots(
                    list(getattr(transaction, "slot_assignments", []) or [])
                ).items()
                if int(value) > 0
            }
        requirement_contracts.append({
            "contract_id": int(getattr(transaction, "contract_id", -1)),
            "contract_kind": "topology_closure",
            "target_role": str(getattr(transaction, "target_role", "") or ""),
            "source_fragment_ids": [],
            "claimed_source_node_ids": [
                int(value)
                for value in list(getattr(transaction, "source_node_ids", []) or [])
            ],
            "required_counts": {
                str(key): int(value) for key, value in required_counts.items()
            },
        })
    aggregate_ids = _round_resource_ids(phase_eval)
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_ids,
        authorized_ids_before_by_su=initial_pool,
        final_remaining_ids_by_su=final_pool,
        committed_claimed_ids_by_su=_committed_occupied_ids_by_su(
            transactions,
            aggregate_ids_by_su=aggregate_ids,
            groups=groups,
        ),
        groups=groups,
    )
    aggregate_required_counts = {
        int(key): int(value)
        for key, value in dict(accounting.get("aggregate_required_counts", {}) or {}).items()
        if int(value) > 0
    }
    demand_before: Dict[int, int] = {}
    histogram = HistogramAdjustmentPlan(
        phase=str(phase),
        groups=list(groups),
        primary_adjustment_count=len(list(groups)),
        physical_conversion_count=sum(
            (2 if group.primary_op.via_su is not None else 1)
            + sum(2 if op.via_su is not None else 1 for op in list(group.h_compensation_ops or []))
            for group in list(groups)
        ),
        H_before=H_before.detach().clone(),
        H_after=H_after.detach().clone(),
        inventory_before={
            int(key): int(value)
            for key, value in dict(accounting.get("authorized_inventory_before", {}) or {}).items()
        },
        inventory_after={
            int(key): int(value)
            for key, value in dict(accounting.get("authorized_inventory_after", {}) or {}).items()
        },
        aggregate_inventory_before={
            int(key): int(value)
            for key, value in dict(accounting.get("aggregate_inventory_before", {}) or {}).items()
        },
        aggregate_inventory_after={
            int(key): int(value)
            for key, value in dict(accounting.get("aggregate_inventory_after", {}) or {}).items()
        },
        aggregate_required_counts=dict(aggregate_required_counts),
        protected_resource_counts=_phase_protected_resource_counts(phase_eval),
        accounting_diagnostics=dict(accounting),
        demand_before=dict(demand_before),
        demand_after=dict(demand_before),
        su11_rotation_index_before=int(rotation_before),
        su11_rotation_index_after=int(rotation_after),
    )
    stage_completed = True
    for key in (
        "external_stage_completed",
        "su24_stage_completed",
        "residual_stage_completed",
    ):
        if key in dict(stage_diagnostics or {}):
            stage_completed = bool(dict(stage_diagnostics or {}).get(key))
            break
    if not bool(accounting.get("ok", False)):
        phase_infeasible_contracts.insert(0, {
            "phase": str(phase),
            "contract_id": -1,
            "target_role": "resource_accounting",
            "source_node_ids": [],
            "required_counts": dict(aggregate_required_counts),
            "proof": {
                "reason": "stage3_resource_accounting_invariant_failed",
                "accounting": dict(accounting),
            },
        })
    if phase_infeasible_contracts:
        first_contract_proof = dict(
            phase_infeasible_contracts[0].get("proof", {}) or {}
        )
        first_failure_domain = str(
            first_contract_proof.get("failure_domain", "") or ""
        )
        phase_failure_reason = (
            "stage3_topology_materialization_invalid"
            if first_failure_domain == "topology_materialization"
            else "stage3_completion_invariants_failed"
        )
        pending_required_counts = {
            str(key): int(value)
            for key, value in dict(
                phase_infeasible_contracts[0].get("required_counts", {}) or {}
            ).items()
            if int(value) > 0
        }
        proof = {
            "phase": str(phase),
            "reason": str(phase_failure_reason),
            "failure_domain": str(first_failure_domain),
            "failed_contract": dict(phase_infeasible_contracts[0]),
            "required_counts": dict(
                phase_infeasible_contracts[0].get("required_counts", {}) or {}
            ),
            "available_counts": {
                str(key): len(values) for key, values in dict(final_pool or {}).items()
            },
            "missing_counts": {},
            "attempted_conversions": list(phase_infeasible_contracts),
            "h_or_element_rejections": [
                dict(item or {}).get("proof", {})
                for item in list(phase_infeasible_contracts or [])
            ],
            "template_checks": {"phase_has_unresolved_blockers": True},
        }
        phase_result = Stage3PhaseResult(
            active_phase=str(phase),
            phase_completed=False,
            rebuild_required=False,
            primary_adjustment_count=int(histogram.primary_adjustment_count),
            h_compensation_count=_h_compensation_count(histogram.groups),
            symbolic_topology_count=sum(1 for item in transactions if bool(item.committed)),
            transactions=list(transactions),
            stop_reason="phase_resource_infeasible",
            infeasibility_proof=proof,
        )
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase=str(phase),
            plan_kind=(
                "sequential_phase_topology_materialization_invalid"
                if first_failure_domain == "topology_materialization"
                else "sequential_phase_infeasible_after_progress"
            ),
            active_subphase=str(phase),
            requirement_contracts=list(requirement_contracts),
            pending_contracts=list(phase_infeasible_contracts),
            hard_required_counts=dict(pending_required_counts),
            minimum_resource_request=dict(pending_required_counts),
            requested_counts=dict(pending_required_counts),
            histogram_adjustment_plan=histogram,
            phase_result=phase_result,
            diagnostics={
                "stage3_phase": str(phase),
                "sequential_single_topology": True,
                "candidate_alternatives_enabled": True,
                "phase_result": phase_result.to_dict(),
                "histogram_adjustment_plan": histogram.to_dict(),
                "infeasible_contracts": list(phase_infeasible_contracts),
                **dict(stage_diagnostics or {}),
            },
            runtime=_runtime_with_pool(phase_eval, final_pool),
        ), h_tolerance)
    phase_result = Stage3PhaseResult(
        active_phase=str(phase),
        phase_completed=bool(stage_completed),
        rebuild_required=True,
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        symbolic_topology_count=sum(1 for item in transactions if bool(item.committed)),
        transactions=list(transactions),
        stop_reason="stage3_rebuild_required",
    )
    return _annotate_h_hard_rule(Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase=str(phase),
        plan_kind="sequential_phase_transactions",
        active_subphase=str(phase),
        requirement_contracts=list(requirement_contracts),
        pending_contracts=[],
        # Completed-product claims are aggregate accounting evidence, not a
        # new demand against the already reduced authorized inventory.
        hard_required_counts={},
        minimum_resource_request={},
        requested_counts={},
        resource_request_breakdown={
            "active_contract_count_before_overlap_filter": len(requirement_contracts),
            "active_contract_count_after_overlap_filter": len(requirement_contracts),
            "selected_active_contracts": list(requirement_contracts),
        },
        adjustment_bundles=[Stage3AdjustmentBundle(
            bundle_id=f"{phase}:transactions",
            phase=str(phase),
            bundle_ops=_group_operations(groups),
            diagnostics={"sequential_single_topology": True},
        )],
        histogram_adjustment_plan=histogram,
        planned_conversions=[
            Stage3AdjustmentOp(
                op="convert",
                from_su=int(op.from_su),
                to_su=int(op.to_su),
                node_id=int(list(op.node_ids or [-1])[0]),
                reason=str(op.reason),
                via_su=op.via_su,
            )
            for group in groups
            for op in [group.primary_op] + list(group.h_compensation_ops or [])
            if list(op.node_ids or [])
        ],
        phase_result=phase_result,
        diagnostics={
            "stage3_phase": str(phase),
            "sequential_single_topology": True,
            "candidate_alternatives_enabled": True,
            "phase_result": phase_result.to_dict(),
            "su11_rotation_index_after": int(rotation_after),
            "resource_request_breakdown": {
                "active_contract_count_before_overlap_filter": len(requirement_contracts),
                "active_contract_count_after_overlap_filter": len(requirement_contracts),
                "selected_active_contracts": list(requirement_contracts),
            },
            "resource_accounting": dict(accounting),
            **dict(stage_diagnostics or {}),
        },
        runtime=_runtime_with_pool(phase_eval, initial_pool),
    ), h_tolerance)


def _transaction_plan(
    *,
    phase_eval: Any,
    contract: TopologyClosureContract,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    E_target: Any,
    max_primary_adjustments: int,
    max_h_compensation_ops: int,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int = 0,
    iteration_ledger: Any = None,
    prefix_groups: Optional[Sequence[Any]] = None,
    protected_resource_counts: Optional[Dict[int, int]] = None,
    accounting_initial_pool: Optional[Dict[int, List[int]]] = None,
    forbidden_primary_conversion_signatures: Optional[Sequence[str]] = None,
    reserve_contract_after_conversion: bool = True,
    allow_h_outside_nonworsening: bool = False,
) -> Stage3AdjustmentPlan:
    if iteration_ledger is not None:
        pool = deepcopy(iteration_ledger.available_ids_by_su)
    contract_dict = contract.to_dict()
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="phase_resource_infeasible",
        phase=str(contract.phase),
        plan_kind="single_topology_transaction",
        active_subphase=str(contract.phase),
        requirement_contracts=[contract_dict],
        pending_contracts=[contract_dict],
        hard_required_counts={str(key): int(value) for key, value in contract.required_counts.items()},
        minimum_resource_request={str(key): int(value) for key, value in contract.required_counts.items()},
        resource_deficit=_deficit(pool, contract.required_counts),
        resource_request_breakdown={
            "active_contract_count_before_overlap_filter": 1,
            "active_contract_count_after_overlap_filter": 1,
            "selected_active_contracts": [contract_dict],
        },
        runtime={
            **_runtime_with_pool(
                phase_eval,
                pool,
                protected_counts=protected_resource_counts,
            ),
        },
        diagnostics={
            "candidate_alternatives_enabled": True,
            "allow_h_outside_nonworsening": bool(
                allow_h_outside_nonworsening
            ),
            "forbidden_primary_conversion_signatures": sorted({
                str(value)
                for value in list(forbidden_primary_conversion_signatures or [])
                if str(value)
            }),
        },
    )
    histogram = build_histogram_adjustment_plan(
        plan=plan,
        H=H,
        E_target=E_target,
        max_primary_adjustments=max_primary_adjustments,
        max_h_compensation_ops=max_h_compensation_ops,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    pool_after = _bind_real_node_ids(
        histogram,
        pool,
        protected_node_ids=(
            contract.claimed_source_node_ids or contract.source_node_ids
        ),
    )
    ledger_after_conversion = (
        _ledger_after_histogram_conversions(
            iteration_ledger,
            histogram,
            protected_node_ids=(
                contract.claimed_source_node_ids or contract.source_node_ids
            ),
        )
        if iteration_ledger is not None
        else None
    )
    assigned_slots: List[Any] = []
    proof: Dict[str, Any] = {}
    claimed: Optional[Dict[int, List[int]]] = None
    if ledger_after_conversion is not None and bool(reserve_contract_after_conversion):
        updated_ledger, assigned_slots, claimed_map, proof = assign_contract_slots(
            ledger_after_conversion,
            contract,
        )
        if updated_ledger is not None:
            ledger_after_conversion = updated_ledger
            claimed = claimed_map
    elif ledger_after_conversion is None and bool(reserve_contract_after_conversion):
        claimed = _consume(pool_after, contract.required_counts)
    else:
        claimed = {}
    accounting: Dict[str, Any] = {}
    if claimed is not None and ledger_after_conversion is not None:
        aggregate_ids = _round_resource_ids(phase_eval)
        accounting_groups = list(prefix_groups or []) + list(histogram.groups or [])
        accounting = build_resource_accounting_snapshot(
            aggregate_ids_before_by_su=aggregate_ids,
            authorized_ids_before_by_su=(
                accounting_initial_pool
                if accounting_initial_pool is not None
                else pool
            ),
            final_remaining_ids_by_su=ledger_after_conversion.available_ids_by_su,
            committed_claimed_ids_by_su=_ledger_occupied_ids_by_su(
                ledger_after_conversion,
                aggregate_ids_by_su=aggregate_ids,
            ),
            groups=accounting_groups,
        )
        histogram.inventory_before = {
            int(key): int(value)
            for key, value in dict(
                accounting.get("authorized_inventory_before", {}) or {}
            ).items()
        }
        histogram.inventory_after = {
            int(key): int(value)
            for key, value in dict(
                accounting.get("authorized_inventory_after", {}) or {}
            ).items()
        }
        histogram.aggregate_inventory_before = {
            int(key): int(value)
            for key, value in dict(
                accounting.get("aggregate_inventory_before", {}) or {}
            ).items()
        }
        histogram.aggregate_inventory_after = {
            int(key): int(value)
            for key, value in dict(
                accounting.get("aggregate_inventory_after", {}) or {}
            ).items()
        }
        histogram.aggregate_required_counts = {
            int(key): int(value)
            for key, value in dict(
                accounting.get("aggregate_required_counts", {}) or {}
            ).items()
        }
        phase_protected = _phase_protected_resource_counts(phase_eval)
        histogram.protected_resource_counts = {
            int(key): max(
                int(phase_protected.get(int(key), 0)),
                int(dict(protected_resource_counts or {}).get(int(key), 0)),
            )
            for key in (
                set(phase_protected)
                | {int(value) for value in dict(protected_resource_counts or {})}
            )
            if max(
                int(phase_protected.get(int(key), 0)),
                int(dict(protected_resource_counts or {}).get(int(key), 0)),
            ) > 0
        }
        histogram.accounting_diagnostics = dict(accounting)
    conversions = []
    compensation_ids: List[int] = []
    converted_ids: List[int] = []
    for group in list(histogram.groups or []):
        primary = group.primary_op
        converted_ids.extend(int(value) for value in list(primary.node_ids or []))
        conversions.append(primary.to_dict())
        for compensation in list(group.h_compensation_ops or []):
            compensation_ids.extend(int(value) for value in list(compensation.node_ids or []))
            conversions.append(compensation.to_dict())
    feasible = (
        claimed is not None
        and not dict(histogram.unresolved_counts or {})
        and (not accounting or bool(accounting.get("ok", False)))
    )
    requires_rebuild = bool(feasible and histogram.primary_adjustment_count > 0)
    zero_conversion_proof = (
        {
            "reason": "zero_conversion_transaction_must_use_iteration_ledger",
            "contract_id": int(contract.contract_id),
            "attempted_claimed_resource_ids": dict(claimed or {}),
            "attempted_slot_assignments": [
                slot.to_dict() for slot in list(assigned_slots or [])
            ],
        }
        if feasible and not requires_rebuild
        else {}
    )
    h_after_primary = _h_total(H, special_degree_meta)
    if list(histogram.groups or []):
        h_after_primary = float(histogram.groups[-1].h_after_primary)
    transaction = TopologyResourceTransaction(
        phase=str(contract.phase),
        contract_id=int(contract.contract_id),
        target_role=str(contract.target_role),
        transaction_origin="current_plan_candidate",
        source_node_ids=(
            list(contract.source_node_ids)
            if bool(reserve_contract_after_conversion)
            else []
        ),
        converted_node_ids=converted_ids,
        conversion_path=conversions,
        compensation_node_ids=compensation_ids,
        claimed_resource_ids=(
            dict(claimed or {})
            if requires_rebuild and bool(reserve_contract_after_conversion)
            else {}
        ),
        product_node_ids=(
            sorted(set(
                contract.source_node_ids
                + [
                    value
                    for values in dict(claimed or {}).values()
                    for value in values
                ]
            ))
            if requires_rebuild and bool(reserve_contract_after_conversion) else []
        ),
        slot_assignments=(
            [slot.to_dict() for slot in list(assigned_slots or [])]
            if requires_rebuild and bool(reserve_contract_after_conversion) else []
        ),
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=float(h_after_primary),
        h_after=_h_total(getattr(histogram, "H_after", H), special_degree_meta),
        commit_scope="physical_histogram" if requires_rebuild else "iteration_ledger",
        committed=bool(requires_rebuild),
        rollback_reason=(
            ""
            if requires_rebuild
            else str(
                zero_conversion_proof.get("reason")
                or histogram.blocked_reason
                or proof.get("reason", "current_contract_resource_infeasible")
            )
        ),
        infeasibility_proof=(
            {}
            if requires_rebuild
            else dict(
                zero_conversion_proof
                or ({
                    "reason": "stage3_resource_accounting_invariant_failed",
                    "accounting": dict(accounting),
                    "attempted_claimed_resource_ids": dict(claimed or {}),
                } if accounting and not bool(accounting.get("ok", False)) else {})
                or proof
                or {}
            )
        ),
    )
    ops = [dict(value) for value in conversions]
    plan.ok = bool(requires_rebuild)
    plan.reason = "stage3_rebuild_required" if plan.ok else "phase_resource_infeasible"
    plan.histogram_adjustment_plan = histogram
    plan.planned_conversions = [
        Stage3AdjustmentOp(
            op="convert",
            from_su=int(value["from_su"]),
            to_su=int(value["to_su"]),
            node_id=int(list(value.get("node_ids", []) or [-1])[0]),
            reason=str(value.get("reason", "")),
        )
        for value in conversions
        if value.get("from_su") is not None and list(value.get("node_ids", []) or [])
    ]
    plan.adjustment_bundles = [Stage3AdjustmentBundle(
        bundle_id=f"{contract.phase}:{contract.contract_id}",
        phase=str(contract.phase),
        bundle_ops=ops,
        diagnostics={"sequential_single_topology": True, "contract_id": int(contract.contract_id)},
    )]
    plan.phase_result = Stage3PhaseResult(
        active_phase=str(contract.phase),
        phase_completed=False,
        rebuild_required=bool(plan.ok),
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        symbolic_topology_count=(
            1 if requires_rebuild and bool(reserve_contract_after_conversion) else 0
        ),
        transactions=[transaction],
        stop_reason=str(plan.reason),
        infeasibility_proof={} if plan.ok else {
            "contract_id": int(contract.contract_id),
            "required_counts": dict(contract.required_counts),
            "available_counts": {int(key): len(values) for key, values in pool.items()},
            "unresolved_counts": dict(histogram.unresolved_counts or _deficit(pool, contract.required_counts)),
            "slot_assignment_proof": dict(zero_conversion_proof or proof or {}),
            "resource_accounting": dict(accounting),
        },
    )
    plan.diagnostics = {
        "stage3_phase": str(contract.phase),
        "sequential_single_topology": True,
        "candidate_alternatives_enabled": True,
        "forbidden_primary_conversion_signatures": sorted({
            str(value)
            for value in list(forbidden_primary_conversion_signatures or [])
            if str(value)
        }),
        "current_contract": contract.to_dict(),
        "phase_result": plan.phase_result.to_dict(),
        "histogram_adjustment_plan": histogram.to_dict(),
        "resource_accounting": dict(accounting),
        "su11_rotation_index_after": int(
            histogram.su11_rotation_index_after
        ),
    }
    plan.unresolved_counts = {str(key): int(value) for key, value in dict(histogram.unresolved_counts or {}).items()}
    plan.requested_counts = {str(key): int(value) for key, value in contract.required_counts.items()}
    if ledger_after_conversion is not None:
        plan.runtime["iteration_topology_ledger"] = ledger_after_conversion
    return _annotate_h_hard_rule(plan, h_tolerance)


def plan_stage3_state_machine(
    *,
    phase_eval: Any,
    H: torch.Tensor,
    nodes: Sequence[Any],
    classification: Any = None,
    S_target: Any = None,
    E_target: Any = None,
    difference_spectrum: Optional[Dict[str, Any]] = None,
    enable_post_external_spectrum: bool = True,
    progress_phase: Stage3ProgressPhase = Stage3ProgressPhase.SPECTRUM_REBALANCE,
    spectrum_candidate_limit: int = 2,
    spectrum_max_primary_adjustments: int = 10,
    rejected_spectrum_histogram_signatures: Optional[Sequence[str]] = None,
    forbidden_spectrum_inverse_signatures: Optional[Sequence[str]] = None,
    max_primary_adjustments_per_stage: int = 10,
    max_h_compensation_ops_per_adjustment: int = 8,
    h_tolerance: float = DEFAULT_STAGE3_H_HARD_TOLERANCE,
    residual_flex_preferred_23: int = 3,
    residual_flex_min_23: int = 2,
    residual_side_preferred_23: int = 2,
    residual_side_min_23: int = 1,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
    su24_nonfused_cursor: int = 0,
    allow_h_outside_nonworsening: bool = False,
    protected_terminal_aromatic_node_ids: Optional[Sequence[int]] = None,
    pending_su24_ring_candidate: Optional[Dict[str, Any]] = None,
    pending_residual_product_candidate: Optional[Dict[str, Any]] = None,
    pending_terminal_su24_node_ids: Optional[Sequence[int]] = None,
    committed_topology_products: Optional[Sequence[Dict[str, Any]]] = None,
    forbidden_resource_conversion_signatures: Optional[Sequence[str]] = None,
) -> Stage3AdjustmentPlan:
    max_primary_adjustments_per_stage = min(
        10,
        max(0, int(max_primary_adjustments_per_stage)),
    )
    if H is None:
        histogram_size = max(
            40,
            max((int(getattr(node, "su_type", -1)) for node in nodes), default=-1) + 1,
        )
        H = torch.zeros(histogram_size, dtype=torch.long)
        for node in nodes:
            su_type = int(getattr(node, "su_type", -1))
            if 0 <= su_type < histogram_size:
                H[su_type] += 1
    if int(residual_flex_min_23) < 1 or int(residual_flex_preferred_23) < int(residual_flex_min_23):
        raise ValueError("residual flex configuration requires preferred >= minimum >= 1")
    if int(residual_side_min_23) < 0 or int(residual_side_preferred_23) < int(residual_side_min_23):
        raise ValueError("residual side configuration requires preferred >= minimum >= 0")
    if not isinstance(progress_phase, Stage3ProgressPhase):
        raise TypeError("progress_phase must be a Stage3ProgressPhase")
    external_phases_pending = (
        progress_phase is Stage3ProgressPhase.SPECTRUM_REBALANCE
    )
    flex_block = str(getattr(phase_eval, "flex_reconciliation_block_reason", "") or "")
    if external_phases_pending and flex_block:
        flex_contracts = [
            dict(item or {})
            for item in list(getattr(phase_eval, "flex_reconciliation_contracts", []) or [])
            if dict(item or {})
        ]
        phase_result = Stage3PhaseResult(
            active_phase="flex_range_reconciliation",
            phase_completed=False,
            rebuild_required=False,
            stop_reason="phase_resource_infeasible",
            infeasibility_proof={
                "reason": flex_block,
                "flex_bounds": dict(getattr(phase_eval, "flex_bounds", {}) or {}),
                "flex_counts": dict(getattr(phase_eval, "flex_counts", {}) or {}),
            },
        )
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=flex_block,
            phase="flex_range_reconciliation",
            plan_kind="flex_range_infeasible",
            active_subphase="flex_range_reconciliation",
            requirement_contracts=flex_contracts,
            pending_contracts=flex_contracts,
            phase_result=phase_result,
            diagnostics={
                "stage3_phase": "flex_range_reconciliation",
                "phase_result": phase_result.to_dict(),
            },
            runtime=_runtime_with_pool(phase_eval, _resource_ids(phase_eval)),
        ), h_tolerance)
    if (
        external_phases_pending
        and str(
            dict(getattr(phase_eval, "flex_audit", {}) or {}).get(
                "scenario", ""
            )
            or ""
        )
        == "above_range"
    ):
        return _flex_excess_reduction_plan(
            phase_eval=phase_eval,
            H=H,
            E_target=E_target,
            max_primary_adjustments=max_primary_adjustments_per_stage,
            max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
        )
    if classification is None:
        classification = classify_topology_units(list(nodes or []))
    if external_phases_pending:
        phase_eval = _evaluate_external_phases(
            phase_eval=phase_eval,
            nodes=nodes,
            classification=classification,
            S_target=S_target,
            E_target=E_target,
        )
    else:
        phase_eval = deepcopy(phase_eval)
        phase_eval.external_closure_ok = True
        phase_eval.flex_range_reconciliation_ok = True
        phase_eval.active_phase = progress_phase.value
        phase_eval.phase = progress_phase.value
        phase_eval.active_external_subphase = progress_phase.value
        phase_eval.canonical_contracts = []
        phase_eval.completed_phases = [
            "flex_range_reconciliation",
            *[phase for phase, _role in EXTERNAL_ROLE_ORDER],
        ]
        phase_eval.diagnostics = {
            **dict(getattr(phase_eval, "diagnostics", {}) or {}),
            "external_stage_restored_from_committed_products": True,
            "external_closure_ok": True,
        }
    materialization_errors = (
        _external_product_materialization_errors(phase_eval)
        if external_phases_pending
        else []
    )
    if materialization_errors:
        proof = {
            "phase": EXTERNAL_STAGE,
            "reason": "external_physical_product_validation_failed",
            "materialization_errors": materialization_errors,
        }
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase=EXTERNAL_STAGE,
            plan_kind="external_physical_product_validation_failed",
            active_subphase=EXTERNAL_STAGE,
            phase_result=Stage3PhaseResult(
                active_phase=EXTERNAL_STAGE,
                phase_completed=False,
                rebuild_required=False,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": EXTERNAL_STAGE,
                "external_stage_completed": False,
                "infeasibility_proof": proof,
            },
            runtime=_runtime_with_pool(phase_eval, _resource_ids(phase_eval)),
        ), h_tolerance)
    protected_terminal_ids = {
        int(node_id)
        for node_id in list(protected_terminal_aromatic_node_ids or [])
    }
    pool = _resource_ids(phase_eval)
    if protected_terminal_ids:
        pool = {
            int(su_type): [
                int(node_id)
                for node_id in list(node_ids or [])
                if int(node_id) not in protected_terminal_ids
            ]
            for su_type, node_ids in pool.items()
        }
    H_work = H.detach().clone()
    all_contracts = _external_contracts(phase_eval, nodes)
    (
        active_contracts,
        dormant_contracts,
        active_external_phase,
        active_external_role,
    ) = _partition_active_external_contracts(phase_eval, all_contracts)
    contracts = list(all_contracts or [])
    dormant_contract_owner_keys = {
        stage3_contract_owner_key(contract)
        for contract in dormant_contracts
    }
    iteration_ledger = _iteration_ledger_for_phase_eval(
        phase_eval,
        nodes,
        pool,
        all_contracts,
    )
    persisted_products = [
        dict(product or {})
        for product in list(committed_topology_products or [])
        if dict(product or {})
    ]
    if persisted_products:
        iteration_ledger = seed_ledger_topology_state(
            iteration_ledger,
            completed_products=persisted_products,
        )
    iteration_ledger, pending_node_reservation_audit = (
        _reserve_pending_candidate_nodes(
            iteration_ledger,
            nodes,
            pending_ring_candidate=pending_su24_ring_candidate,
            pending_residual_candidate=pending_residual_product_candidate,
            pending_terminal_su24_node_ids=pending_terminal_su24_node_ids,
            restore_uncommitted_su24=not bool(external_phases_pending),
        )
    )
    pool = deepcopy(iteration_ledger.available_ids_by_su)
    initial_pool = deepcopy(pool)
    committed_product_conflicts = [
        dict(item or {})
        for item in list(iteration_ledger.conflicts or [])
        if str(dict(item or {}).get("reason", "") or "").startswith(
            ("completed_product_", "duplicate_completed_product")
        )
    ]
    if committed_product_conflicts:
        proof = {
            "phase": "resource_accounting",
            "reason": "committed_product_resource_identity_conflict",
            "conflicts": committed_product_conflicts,
            "available_counts": {
                str(key): len(values) for key, values in pool.items()
            },
        }
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="resource_accounting",
            plan_kind="committed_product_resource_identity_conflict",
            active_subphase="resource_accounting",
            phase_result=Stage3PhaseResult(
                active_phase="resource_accounting",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "resource_accounting",
                "infeasibility_proof": proof,
            },
            runtime={
                **_runtime_with_pool(phase_eval, pool),
                "iteration_topology_ledger": iteration_ledger,
            },
        ), h_tolerance)
    completed_phases: List[str] = ["flex_range_reconciliation"]
    direct_transactions: List[TopologyResourceTransaction] = _assembly_product_transactions(
        phase_eval,
        H_work,
        special_degree_meta,
        completed_products=list(iteration_ledger.completed_products or []),
    )
    phase_groups: List[Any] = []
    infeasible_contracts: List[Dict[str, Any]] = []
    stale_identity_contracts: List[Dict[str, Any]] = []
    rotation_cursor = int(su11_rotation_index)
    external_contracts_seen = 0
    materialized_external_products = [
        dict(product or {})
        for product in list(iteration_ledger.completed_products or [])
        if str(dict(product or {}).get("phase", "") or "") in EXTERNAL_PHASE_NAMES
        and bool(dict(product or {}).get("closed", False))
        and not bool(dict(product or {}).get("provisional", False))
    ]
    external_directly_completed = len(materialized_external_products)
    external_completed_after_conversion = 0
    external_superseded = 0
    external_status: Dict[str, Dict[str, Any]] = {}
    for role_index, (phase, role) in enumerate(EXTERNAL_ROLE_ORDER):
        role_seen = 0
        role_direct = sum(
            1
            for product in materialized_external_products
            if str(product.get("target_role", product.get("product_role", "")) or "")
            == str(role)
        )
        role_converted = 0
        role_blocked = 0
        role_deferred = 0
        phase_contracts = [item for item in contracts if str(item.get("target_role", "")) == role]
        for contract_index, raw in enumerate(phase_contracts):
            external_contracts_seen += 1
            role_seen += 1
            contract = _closure_contract(raw, phase)
            source_status, source_proof = contract_source_status(
                iteration_ledger,
                list(contract.claimed_source_node_ids or contract.source_node_ids or []),
            )
            if source_status == "superseded":
                external_superseded += 1
                continue
            blocker_reason = str(raw.get("blocker_kind", raw.get("blocked_reason", "")) or "")
            if blocker_reason and not contract.required_counts:
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": {},
                    "proof": {
                        "contract_id": int(contract.contract_id),
                        "reason": blocker_reason,
                        "source_node_ids": list(contract.source_node_ids),
                    },
                })
                role_blocked += 1
                continue
            if source_status == "conflict":
                stale_proof = {
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "reason": "stale_pending_identity",
                    **dict(source_proof or {}),
                }
                stale_identity_contracts.append(stale_proof)
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": dict(stale_proof),
                })
                role_blocked += 1
                continue
            missing_counts = _deficit(pool, contract.required_counts)
            if not missing_counts:
                if stage3_contract_owner_key(raw) in dormant_contract_owner_keys:
                    # Resource deficits may be batched across later phases,
                    # but physical materialization remains sequential.  A
                    # resource-satisfied later contract has not failed; its
                    # assembler subphase simply has not run yet.
                    role_deferred += 1
                    continue
                assembler_diagnostics = dict(raw.get("diagnostics", {}) or {})
                materialization_failure = dict(
                    assembler_diagnostics.get("materialization_failure", {}) or {}
                )
                assembler_reason = str(
                    materialization_failure.get("validation_reason", "")
                    or raw.get("reason", contract.reason)
                    or "external_product_not_created"
                )
                materialization_proof = {
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "reason": "external_topology_materialization_invalid",
                    "failure_domain": "topology_materialization",
                    "assembler_reason": str(assembler_reason),
                    "source_fragment_ids": list(contract.source_fragment_ids),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "available_counts": {
                        int(key): len(list(values or []))
                        for key, values in pool.items()
                    },
                    "missing_counts": {},
                    "mainline_slots": [
                        slot.to_dict() for slot in list(contract.mainline_slots or [])
                    ],
                    "branch_slots": [
                        slot.to_dict() for slot in list(contract.branch_slots or [])
                    ],
                    "assembler_diagnostics": assembler_diagnostics,
                }
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": materialization_proof,
                })
                role_blocked += 1
                continue
            direct_proof = {
                "reason": "external_resource_deficit",
                "missing_counts": dict(missing_counts),
            }
            remaining_limit = max(0, int(max_primary_adjustments_per_stage) - len(phase_groups))
            future_contracts = list(phase_contracts[int(contract_index) + 1:])
            for _, future_role in EXTERNAL_ROLE_ORDER[int(role_index) + 1:]:
                future_contracts.extend(
                    item
                    for item in all_contracts
                    if str(item.get("target_role", "")) == str(future_role)
                )
            future_required_11 = sum(
                max(
                    0,
                    int(
                        dict(item.get("required_counts", {}) or {}).get(
                            11,
                            dict(item.get("required_counts", {}) or {}).get("11", 0),
                        )
                        or 0
                    ),
                )
                for item in future_contracts
            )
            current_required_11 = int(contract.required_counts.get(11, 0) or 0)
            protected_aromatic_donors = _protect_future_su11_aromatic_donors(
                pool,
                current_required_11=current_required_11,
                future_required_11=future_required_11,
                aromatic_cursor=rotation_cursor,
            )
            candidate_plan = _transaction_plan(
                phase_eval=phase_eval,
                contract=contract,
                pool=pool,
                H=H_work,
                E_target=E_target,
                max_primary_adjustments=remaining_limit,
                max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
                h_tolerance=h_tolerance,
                special_degree_meta=special_degree_meta,
                su11_rotation_index=rotation_cursor,
                iteration_ledger=iteration_ledger,
                prefix_groups=phase_groups,
                protected_resource_counts=protected_aromatic_donors,
                accounting_initial_pool=initial_pool,
                forbidden_primary_conversion_signatures=forbidden_resource_conversion_signatures,
                reserve_contract_after_conversion=True,
            )
            if not candidate_plan.ok:
                if phase_groups:
                    budget_exhausted = int(remaining_limit) <= 0
                    return _finalize_phase_plan(
                        phase_eval=phase_eval,
                        phase=EXTERNAL_STAGE,
                        H_before=H,
                        H_after=H_work,
                        initial_pool=initial_pool,
                        final_pool=pool,
                        groups=phase_groups,
                        transactions=direct_transactions,
                        rotation_before=su11_rotation_index,
                        rotation_after=rotation_cursor,
                        h_tolerance=h_tolerance,
                        stage_diagnostics={
                            "external_stage_completed": False,
                            "external_rebuild_reason": (
                                "resource_conversion_requires_fresh_assembly"
                                if budget_exhausted
                                else "external_blocker_after_conversion"
                            ),
                            "external_contracts_seen": int(external_contracts_seen),
                            "external_directly_completed": int(external_directly_completed),
                            "external_reserved_after_conversion": int(
                                external_completed_after_conversion
                            ),
                            "external_completed_after_conversion": 0,
                            "external_blocked": 0 if budget_exhausted else 1,
                            "external_remaining_pending": max(
                                1,
                                len(contracts) - int(external_contracts_seen) + 1,
                            ),
                            "external_primary_count": len(phase_groups),
                            "external_remaining_budget": max(
                                0,
                                int(max_primary_adjustments_per_stage) - len(phase_groups),
                            ),
                            "external_conversion_budget_exhausted": bool(
                                budget_exhausted
                            ),
                            "physical_materialization_required_after_rebuild": True,
                        },
                    )
                failed_transactions = list(candidate_plan.phase_result.transactions or [])
                direct_transactions.extend(failed_transactions)
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": dict(
                        candidate_plan.phase_result.infeasibility_proof
                        or direct_proof
                        or {}
                    ),
                })
                role_blocked += 1
                continue
            histogram = candidate_plan.histogram_adjustment_plan
            candidate_ledger = dict(getattr(candidate_plan, "runtime", {}) or {}).get(
                "iteration_topology_ledger"
            )
            if candidate_ledger is not None:
                iteration_ledger = candidate_ledger
                pool = deepcopy(iteration_ledger.available_ids_by_su)
            else:
                pool_after = _bind_real_node_ids(histogram, pool)
                claimed = _consume(pool_after, contract.required_counts)
                if claimed is None:
                    return candidate_plan
                pool = pool_after
            H_work = histogram.H_after.detach().clone()
            phase_groups.extend(list(histogram.groups or []))
            rotation_cursor = int(histogram.su11_rotation_index_after)
            direct_transactions.extend(list(candidate_plan.phase_result.transactions or []))
            external_completed_after_conversion += 1
            role_converted += 1
        external_status[str(role)] = {
            "phase": str(phase),
            "contracts_seen": int(role_seen),
            "directly_completed": int(role_direct),
            "completed_after_conversion": int(role_converted),
            "blocked": int(role_blocked),
            "deferred": int(role_deferred),
            "completed": bool(
                role_blocked == 0
                and role_converted == 0
                and role_deferred == 0
            ),
        }
        completed_phases.append(phase)

    external_diagnostics = {
        "external_contracts_seen": int(external_contracts_seen),
        "external_directly_completed": int(external_directly_completed),
        "external_completed_after_conversion": int(external_completed_after_conversion),
        "external_superseded": int(external_superseded),
        "external_materialized_product_count": len(materialized_external_products),
        "external_blocked": len(infeasible_contracts),
        "external_remaining_pending": len(infeasible_contracts) + len(dormant_contracts),
        "external_primary_count": len(phase_groups),
        "external_remaining_budget": max(
            0,
            int(max_primary_adjustments_per_stage) - len(phase_groups),
        ),
        "external_subphase_status": dict(external_status),
        "external_stage_completed": not bool(infeasible_contracts or dormant_contracts),
        "active_external_phase": str(active_external_phase),
        "active_external_role": str(active_external_role),
        "dormant_external_contract_count": len(dormant_contracts),
        "dormant_external_contract_ids": [
            int(dict(contract or {}).get("contract_id", -1) or -1)
            for contract in dormant_contracts
        ],
    }
    if phase_groups:
        # Resource reservations above are round-local.  They let the planner
        # aggregate all classified contract deficits into one histogram batch,
        # but only a fresh Layer1/2 graph may materialize the products.
        external_diagnostics["external_stage_completed"] = False
        external_diagnostics["external_reserved_after_conversion"] = int(
            external_completed_after_conversion
        )
        external_diagnostics["external_completed_after_conversion"] = 0
        external_diagnostics["external_remaining_pending"] = len(all_contracts)
        external_diagnostics["physical_materialization_required_after_rebuild"] = True
        external_diagnostics["external_rebuild_reason"] = (
            "external_blocker_after_conversion"
            if infeasible_contracts
            else "resource_conversion_requires_fresh_assembly"
        )
        return _finalize_phase_plan(
            phase_eval=phase_eval,
            phase=EXTERNAL_STAGE,
            H_before=H,
            H_after=H_work,
            initial_pool=initial_pool,
            final_pool=pool,
            groups=phase_groups,
            transactions=direct_transactions,
            rotation_before=su11_rotation_index,
            rotation_after=rotation_cursor,
            h_tolerance=h_tolerance,
            stage_diagnostics=external_diagnostics,
        )
    if infeasible_contracts:
        return _finalize_phase_plan(
            phase_eval=phase_eval,
            phase=EXTERNAL_STAGE,
            H_before=H,
            H_after=H_work,
            initial_pool=initial_pool,
            final_pool=pool,
            groups=[],
            transactions=direct_transactions,
            rotation_before=su11_rotation_index,
            rotation_after=rotation_cursor,
            h_tolerance=h_tolerance,
            infeasible_contracts=infeasible_contracts,
            stage_diagnostics=external_diagnostics,
        )

    if (
        progress_phase is Stage3ProgressPhase.SPECTRUM_REBALANCE
        and bool(enable_post_external_spectrum)
    ):
        spectrum_candidates, spectrum_audit = generate_spectrum_candidates(
            H=H_work,
            difference_spectrum=difference_spectrum,
            closure_pool_ids_by_su={
                int(su_type): list(pool.get(int(su_type), []) or [])
                for su_type in (11, 22, 23, 24)
            },
            aromatic_donor_ids_by_su={
                int(su_type): list(pool.get(int(su_type), []) or [])
                for su_type in (10, 12, 13)
            },
            candidate_limit=int(spectrum_candidate_limit),
            max_primary_adjustments=min(
                int(max_primary_adjustments_per_stage),
                int(spectrum_max_primary_adjustments),
            ),
            max_h_compensation_ops=int(max_h_compensation_ops_per_adjustment),
            rejected_histogram_signatures=rejected_spectrum_histogram_signatures,
            forbidden_inverse_signatures=forbidden_spectrum_inverse_signatures,
            E_target=E_target,
            special_degree_meta=special_degree_meta,
            flex_audit=dict(getattr(phase_eval, "flex_audit", {}) or {}),
        )
        target_h: Optional[float]
        try:
            target_h = float(E_target.detach().cpu().float()[1].item())
        except Exception:
            target_h = None
        spectrum_candidates, hard_rejected_candidates = (
            _filter_spectrum_candidates_by_stage3_hard_constraint(
                spectrum_candidates,
                target_h=target_h,
                h_tolerance=h_tolerance,
            )
        )
        spectrum_audit = dict(spectrum_audit or {})
        spectrum_audit["stage3_hard_tolerance"] = float(h_tolerance)
        spectrum_audit["stage3_hard_rejected_candidates"] = hard_rejected_candidates
        if spectrum_candidates:
            phase_result = Stage3PhaseResult(
                active_phase="remaining_pool_spectrum_rebalance",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="stage3_spectrum_candidates_ready",
            )
            return _annotate_h_hard_rule(Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_spectrum_candidates_ready",
                phase="remaining_pool_spectrum_rebalance",
                plan_kind="remaining_pool_spectrum_candidate_set",
                active_subphase="remaining_pool_spectrum_rebalance",
                phase_result=phase_result,
                spectrum_candidates=list(spectrum_candidates),
                diagnostics={
                    "stage3_phase": "remaining_pool_spectrum_rebalance",
                    "completed_phases": list(completed_phases),
                    "spectrum_audit": dict(spectrum_audit or {}),
                    "external_products_are_round_local": True,
                    **dict(external_diagnostics),
                },
                runtime={
                    **_runtime_with_pool(phase_eval, pool),
                    "iteration_topology_ledger": iteration_ledger,
                },
            ), h_tolerance)
    completed_phases.append("remaining_pool_spectrum_rebalance")
    if progress_phase is Stage3ProgressPhase.SPECTRUM_REBALANCE:
        progress_phase = Stage3ProgressPhase.SU24_RING_CLOSURE

    su24_initial_pool = deepcopy(pool)
    su24_groups: List[Any] = []
    su24_products: List[Dict[str, Any]] = []
    su24_ring_attempts: List[Dict[str, Any]] = []
    su24_conversion_rejections: List[Dict[str, Any]] = []
    su24_profile_records: Dict[int, Dict[str, Any]] = {}
    su24_rejected_profiles: List[Dict[str, Any]] = []
    su24_stop_reason = "resource_su24_ring_closure_skipped"
    su24_remaining_count = len(list(pool.get(24, []) or []))
    su24_nonfused_cursor_work = max(0, int(su24_nonfused_cursor))
    required_ring_candidate = dict(pending_su24_ring_candidate or {})
    required_ring_signature = str(
        required_ring_candidate.get("candidate_signature", "") or ""
    )
    required_ring_su24_ids = sorted({
        int(node_id)
        for node_id in list(
            required_ring_candidate.get("selected_su24_ids", []) or []
        )
    })
    required_ring_resolved: Dict[str, Any] = {}
    pending_ring_candidate_after: Dict[str, Any] = dict(required_ring_candidate)
    exhausted_ring_candidate_signatures: Set[str] = set()
    if progress_phase is Stage3ProgressPhase.SU24_RING_CLOSURE:
        while su24_remaining_count > 1:
            su24_profile_audit = audit_raw_su24_resource_profiles(nodes, pool)
            su24_profiles = list(su24_profile_audit.get("profiles", []) or [])
            for profile in su24_profiles:
                su24_profile_records[int(profile.node_id)] = profile.to_dict()
            su24_rejected_profiles.extend(
                dict(item or {})
                for item in list(su24_profile_audit.get("rejected", []) or [])
            )
            if not su24_profiles:
                su24_stop_reason = "pool_su24_ring_no_compatible_profiles"
                break

            su24_direct_result = assemble_pool_su24_rings(
                nodes,
                classification,
                pool,
                generation_id=int(getattr(phase_eval, "generation_id", 0) or 0),
                preferred_nonfused=(
                    "side"
                    if int(su24_nonfused_cursor_work) % 2 == 0
                    else "upper_lower"
                ),
                required_candidate_signature=required_ring_signature,
                required_su24_node_ids=required_ring_su24_ids,
                excluded_candidate_signatures=sorted(
                    exhausted_ring_candidate_signatures
                ),
            )
            su24_ring_attempts.extend(
                dict(item or {})
                for item in list(su24_direct_result.get("attempts", []) or [])
            )
            direct_su24_products = [
                dict(item or {})
                for item in list(su24_direct_result.get("products", []) or [])
                if dict(item or {})
            ]
            if direct_su24_products:
                if bool(su24_direct_result.get("required_candidate_resolved", False)):
                    required_ring_resolved = dict(
                        su24_direct_result.get("resolved_required_candidate", {}) or {}
                    )
                    required_ring_signature = ""
                    required_ring_su24_ids = []
                    pending_ring_candidate_after = {}
                iteration_ledger = seed_ledger_topology_state(
                    iteration_ledger,
                    completed_products=direct_su24_products,
                )
                pool = deepcopy(iteration_ledger.available_ids_by_su)
                su24_products.extend(direct_su24_products)
                direct_transactions.extend(
                    _assembly_product_transactions(
                        phase_eval,
                        H_work,
                        special_degree_meta,
                        completed_products=direct_su24_products,
                    )
                )
                product_kind = str(
                    direct_su24_products[-1].get("product_kind", "") or ""
                )
                if product_kind in {
                    "fatty_ring_side",
                    "fatty_ring_upper_lower",
                }:
                    su24_nonfused_cursor_work += 1

            su24_remaining_count = len(list(pool.get(24, []) or []))
            su24_stop_reason = str(
                su24_direct_result.get("stop_reason", "")
                or "pool_su24_ring_templates_exhausted"
            )
            if su24_remaining_count <= 1:
                break

            if required_ring_candidate and not required_ring_resolved:
                su24_stop_reason = str(
                    su24_direct_result.get("stop_reason", "")
                    or "pool_su24_required_candidate_not_materialized"
                )
                break

            blocked_attempts = [
                dict(item or {})
                for item in list(
                    su24_direct_result.get("blocked_attempts", [])
                    or [su24_direct_result.get("blocked_attempt", {})]
                )
                if dict(item or {})
                and dict(dict(item or {}).get("missing_counts", {}) or {})
            ]
            remaining_budget = max(
                0,
                int(max_primary_adjustments_per_stage) - len(su24_groups),
            )
            if not blocked_attempts or remaining_budget <= 0:
                if remaining_budget <= 0:
                    su24_stop_reason = "pool_su24_ring_conversion_budget_reached"
                break

            accepted_conversion_plan: Optional[Stage3AdjustmentPlan] = None
            accepted_attempt: Dict[str, Any] = {}
            for attempt_index, blocked_attempt in enumerate(blocked_attempts):
                contract = _pool_su24_ring_conversion_contract(
                    blocked_attempt,
                    contract_id=(
                        1_000_000
                        + len(su24_groups) * 1_000
                        + int(attempt_index)
                    ),
                )
                candidate_plan = _transaction_plan(
                    phase_eval=phase_eval,
                    contract=contract,
                    pool=pool,
                    H=H_work,
                    E_target=E_target,
                    max_primary_adjustments=int(remaining_budget),
                    max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
                    h_tolerance=h_tolerance,
                    special_degree_meta=special_degree_meta,
                    su11_rotation_index=rotation_cursor,
                    iteration_ledger=iteration_ledger,
                    prefix_groups=su24_groups,
                    accounting_initial_pool=su24_initial_pool,
                    protected_resource_counts={
                        int(key): len(list(values or []))
                        for key, values in dict(
                            blocked_attempt.get(
                                "protected_source_ids_by_su", {}
                            )
                            or {}
                        ).items()
                        if str(key).lstrip("-").isdigit()
                        and list(values or [])
                    },
                    forbidden_primary_conversion_signatures=forbidden_resource_conversion_signatures,
                    reserve_contract_after_conversion=False,
                )
                histogram = candidate_plan.histogram_adjustment_plan
                if candidate_plan.ok and list(histogram.groups or []):
                    accepted_conversion_plan = candidate_plan
                    accepted_attempt = dict(blocked_attempt)
                    break
                su24_conversion_rejections.append({
                    "candidate_signature": str(
                        blocked_attempt.get("candidate_signature", "")
                    ),
                    "skeleton_kind": str(blocked_attempt.get("skeleton_kind", "")),
                    "missing_counts": dict(
                        blocked_attempt.get("missing_counts", {}) or {}
                    ),
                    "proof": dict(
                        candidate_plan.phase_result.infeasibility_proof or {}
                    ),
                })

            if accepted_conversion_plan is None:
                newly_exhausted = {
                    str(attempt.get("candidate_signature", ""))
                    for attempt in blocked_attempts
                    if str(attempt.get("candidate_signature", ""))
                } - exhausted_ring_candidate_signatures
                exhausted_ring_candidate_signatures.update(newly_exhausted)
                if newly_exhausted:
                    su24_ring_attempts.append({
                        "exhausted_candidate_signatures": sorted(
                            newly_exhausted
                        ),
                        "reason": "all_conversion_paths_rejected",
                    })
                    continue
                su24_stop_reason = (
                    "pool_su24_ring_all_conversion_candidates_rejected"
                )
                break

            histogram = accepted_conversion_plan.histogram_adjustment_plan
            candidate_ledger = dict(
                getattr(accepted_conversion_plan, "runtime", {}) or {}
            ).get("iteration_topology_ledger")
            if candidate_ledger is None:
                return accepted_conversion_plan
            iteration_ledger = candidate_ledger
            pool = deepcopy(iteration_ledger.available_ids_by_su)
            H_work = histogram.H_after.detach().clone()
            rotation_cursor = int(histogram.su11_rotation_index_after)
            su24_groups.extend(list(histogram.groups or []))
            direct_transactions.extend(
                list(accepted_conversion_plan.phase_result.transactions or [])
            )
            su24_ring_attempts.append({
                "accepted_conversion_for_candidate": str(
                    accepted_attempt.get("candidate_signature", "")
                ),
                "conversion_group_count": len(list(histogram.groups or [])),
            })
            pending_ring_candidate_after = {
                "candidate_signature": str(
                    accepted_attempt.get("candidate_signature", "") or ""
                ),
                "selected_su24_ids": sorted(
                    int(value)
                    for value in list(
                        accepted_attempt.get("selected_su24_ids", []) or []
                    )
                ),
                "product_kind": str(
                    accepted_attempt.get("product_kind", "") or ""
                ),
                "skeleton_kind": str(
                    accepted_attempt.get("skeleton_kind", "") or ""
                ),
                "required_counts": dict(
                    accepted_attempt.get("required_counts", {}) or {}
                ),
                "source_ids_by_su": dict(
                    accepted_attempt.get("source_ids_by_su", {}) or {}
                ),
                "converted_node_ids": sorted({
                    int(node_id)
                    for transaction in list(
                        accepted_conversion_plan.phase_result.transactions or []
                    )
                    for node_id in list(transaction.converted_node_ids or [])
                }),
            }
            # A conversion is physical only after Layer1/2 rebuild.  Stop here
            # so the next pass can materialize this exact node-bound ring.
            su24_stop_reason = "pool_su24_ring_conversion_requires_rebuild"
            break

        if required_ring_candidate and not required_ring_resolved:
            proof = {
                "phase": "resource_su24_ring_closure",
                "reason": "pending_su24_ring_candidate_not_materialized",
                "pending_ring_candidate": dict(required_ring_candidate),
                "available_counts": {
                    str(key): len(values) for key, values in pool.items()
                },
                "ring_attempts": list(su24_ring_attempts),
                "ring_stop_reason": str(su24_stop_reason),
            }
            return _annotate_h_hard_rule(Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason=str(proof["reason"]),
                phase="resource_su24_ring_closure",
                plan_kind="pending_su24_ring_candidate_infeasible",
                active_subphase="resource_su24_ring_closure",
                phase_result=Stage3PhaseResult(
                    active_phase="resource_su24_ring_closure",
                    phase_completed=False,
                    rebuild_required=False,
                    stop_reason="phase_resource_infeasible",
                    infeasibility_proof=proof,
                ),
                diagnostics={
                    "stage3_phase": "resource_su24_ring_closure",
                    "pending_su24_ring_candidate": dict(required_ring_candidate),
                    "infeasibility_proof": proof,
                },
                runtime={
                    **_runtime_with_pool(phase_eval, pool),
                    "iteration_topology_ledger": iteration_ledger,
                },
            ), h_tolerance)

        su24_stage_diagnostics = {
            "su24_stage_completed": bool(
                su24_stop_reason != "pool_su24_ring_conversion_budget_reached"
            ),
            "su24_rebuild_reason": str(su24_stop_reason),
            "su24_direct_product_count": len(su24_products),
            "su24_ring_attempts": list(su24_ring_attempts),
            "su24_conversion_rejections": list(su24_conversion_rejections),
            "su24_remaining_count": int(len(list(pool.get(24, []) or []))),
            "su24_resource_profiles": list(su24_profile_records.values()),
            "su24_rejected_profiles": list(su24_rejected_profiles),
            "su24_nonfused_cursor_after": int(su24_nonfused_cursor_work),
            "pending_su24_ring_candidate": dict(pending_ring_candidate_after),
            "resolved_pending_su24_ring_candidate": dict(required_ring_resolved),
            "exhausted_su24_ring_candidate_signatures": sorted(
                exhausted_ring_candidate_signatures
            ),
            "committed_su24_products": list(su24_products),
        }
        if su24_groups:
            rebuild_plan = _finalize_phase_plan(
                phase_eval=phase_eval,
                phase="resource_su24_ring_closure",
                H_before=H,
                H_after=H_work,
                initial_pool=su24_initial_pool,
                final_pool=pool,
                groups=list(su24_groups),
                transactions=list(direct_transactions or []),
                rotation_before=su11_rotation_index,
                rotation_after=rotation_cursor,
                h_tolerance=h_tolerance,
                stage_diagnostics=dict(su24_stage_diagnostics),
            )
            rebuild_plan.phase_result.phase_completed = False
            rebuild_plan.phase_result.symbolic_topology_count = len(
                su24_products
            )
            rebuild_plan.diagnostics["phase_result"] = (
                rebuild_plan.phase_result.to_dict()
            )
            return rebuild_plan
    else:
        su24_stage_diagnostics = {
            "su24_stage_completed": True,
            "su24_rebuild_reason": "resource_su24_ring_closure_skipped",
            "su24_direct_product_count": 0,
            "su24_ring_attempts": [],
            "su24_conversion_rejections": [],
            "su24_remaining_count": int(len(list(pool.get(24, []) or []))),
            "su24_resource_profiles": [],
            "su24_rejected_profiles": [],
            "su24_nonfused_cursor_after": int(su24_nonfused_cursor_work),
            "pending_su24_ring_candidate": dict(required_ring_candidate),
            "resolved_pending_su24_ring_candidate": {},
            "exhausted_su24_ring_candidate_signatures": [],
            "committed_su24_products": [],
        }
    completed_phases.append("resource_su24_ring_closure")

    remaining_su24_ids = list(pool.get(24, []) or [])
    if remaining_su24_ids:
        terminal_plan = _residual_su24_aromatization_plan(
            phase_eval=phase_eval,
            pool=pool,
            H=H_work,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=rotation_cursor,
            max_conversions=max_primary_adjustments_per_stage,
            prefix_transactions=direct_transactions,
        )
        if terminal_plan is None:
            proof = {
                "phase": "residual_su24_aromatization",
                "reason": "residual_su24_aromatization_infeasible",
                "remaining_su24_node_ids": [
                    int(value) for value in remaining_su24_ids
                ],
                "available_counts": {
                    str(key): len(values) for key, values in pool.items()
                },
                "primary_conversion_limit": int(
                    max_primary_adjustments_per_stage
                ),
            }
            return Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason=str(proof["reason"]),
                phase="residual_su24_aromatization",
                plan_kind="residual_su24_aromatization_infeasible",
                active_subphase="residual_su24_aromatization",
                phase_result=Stage3PhaseResult(
                    active_phase="residual_su24_aromatization",
                    phase_completed=False,
                    rebuild_required=False,
                    stop_reason="phase_resource_infeasible",
                    infeasibility_proof=proof,
                ),
                diagnostics={
                    "stage3_phase": "residual_su24_aromatization",
                    "terminal_h_exemption": True,
                    "infeasibility_proof": proof,
                    **dict(su24_stage_diagnostics),
                },
                runtime={
                    **_runtime_with_pool(phase_eval, pool),
                    "iteration_topology_ledger": iteration_ledger,
                },
            )
        terminal_plan.diagnostics.update({
            **dict(su24_stage_diagnostics),
            "completed_phases": list(completed_phases),
        })
        return terminal_plan
    completed_phases.append("residual_su24_aromatization")

    residual_initial_pool = deepcopy(pool)
    residual_converted_product_count = 0
    flex_bounds = dict(getattr(phase_eval, "flex_bounds", {}) or {})
    flex_counts = dict(getattr(phase_eval, "flex_counts", {}) or {})
    assembly_view = dict(getattr(phase_eval, "assembly_view", {}) or {})
    flex_target_plan = dict(assembly_view.get("flex_target_plan", {}) or {})
    maximum_flex_capacity = max(
        0,
        int(flex_bounds.get("max", 0) or 0)
        - int(flex_counts.get("final", 0) or 0)
    )
    requested_flex_gap = max(
        int(flex_target_plan.get("remaining_minimum_gap", 0) or 0),
        int(flex_target_plan.get("remaining_preferred_gap", 0) or 0),
    )
    required_flex_goal = min(int(maximum_flex_capacity), int(requested_flex_gap))
    direct_flex_count = 0
    direct_residual_template_fits: List[Dict[str, Any]] = []
    residual_candidate_attempts: List[Dict[str, Any]] = []
    committed_residual_products: List[Dict[str, Any]] = []
    remaining_flex_goal = int(required_flex_goal)
    remaining_flex_capacity = int(maximum_flex_capacity)
    pending_residual_candidate = dict(
        pending_residual_product_candidate or {}
    )
    resolved_pending_residual_candidate: Dict[str, Any] = {}

    persisted_residual_product_count = sum(
        1
        for product in persisted_products
        if str(product.get("phase", "") or "")
        == "residual_resource_self_closure"
    )
    residual_contract_id = (
        2_000_000 + int(persisted_residual_product_count)
    )
    if pending_residual_candidate:
        pending_role = str(
            pending_residual_candidate.get("product_role", "") or ""
        )
        pending_chain_23_count = int(
            pending_residual_candidate.get("chain_23_count", 0) or 0
        )
        pending_required_counts = {
            int(key): int(value)
            for key, value in dict(
                pending_residual_candidate.get("required_counts", {}) or {}
            ).items()
            if str(key).lstrip("-").isdigit() and int(value) > 0
        }
        pending_contract = _residual_product_contract(
            product_role=pending_role,
            chain_23_count=pending_chain_23_count,
            required_counts=pending_required_counts,
            contract_id=residual_contract_id,
        )
        required_converted_ids = {
            int(value)
            for value in list(
                pending_residual_candidate.get("converted_node_ids", []) or []
            )
        }
        pending_input_ledger = _prioritize_pending_converted_ids(
            iteration_ledger,
            sorted(required_converted_ids),
        )
        (
            pending_ledger,
            pending_transaction,
            pending_proof,
        ) = _ledger_transaction(
            ledger=pending_input_ledger,
            contract=pending_contract,
            phase="residual_resource_self_closure",
            H=H_work,
            special_degree_meta=special_degree_meta,
        )
        claimed_pending_ids = {
            int(value)
            for values in dict(
                getattr(pending_transaction, "claimed_resource_ids", {}) or {}
            ).values()
            for value in list(values or [])
        } if pending_transaction is not None else set()
        converted_ids_claimed = required_converted_ids.issubset(
            claimed_pending_ids
        )
        if pending_transaction is None or not converted_ids_claimed:
            proof = {
                "phase": "residual_resource_self_closure",
                "reason": "pending_residual_product_not_materialized",
                "pending_residual_product_candidate": dict(
                    pending_residual_candidate
                ),
                "required_counts": dict(pending_required_counts),
                "available_counts": {
                    str(key): len(values) for key, values in pool.items()
                },
                "missing_counts": _deficit(pool, pending_required_counts),
                "required_converted_node_ids": sorted(required_converted_ids),
                "claimed_resource_node_ids": sorted(claimed_pending_ids),
                "converted_nodes_claimed": bool(converted_ids_claimed),
                "slot_assignment_proof": dict(pending_proof or {}),
                "pending_node_reservation_audit": dict(
                    pending_node_reservation_audit
                ),
            }
            return _annotate_h_hard_rule(Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason=str(proof["reason"]),
                phase="residual_resource_self_closure",
                plan_kind="pending_residual_product_infeasible",
                active_subphase="residual_resource_self_closure",
                phase_result=Stage3PhaseResult(
                    active_phase="residual_resource_self_closure",
                    phase_completed=False,
                    rebuild_required=False,
                    stop_reason="phase_resource_infeasible",
                    infeasibility_proof=proof,
                ),
                diagnostics={
                    "stage3_phase": "residual_resource_self_closure",
                    "pending_residual_product_candidate": dict(
                        pending_residual_candidate
                    ),
                    "infeasibility_proof": proof,
                    **dict(su24_stage_diagnostics),
                },
                runtime={
                    **_runtime_with_pool(phase_eval, pool),
                    "iteration_topology_ledger": iteration_ledger,
                },
            ), h_tolerance)
        iteration_ledger = pending_ledger
        if list(getattr(pending_ledger, "completed_products", []) or []):
            committed_residual_products.append(dict(
                pending_ledger.completed_products[-1] or {}
            ))
        pool = deepcopy(iteration_ledger.available_ids_by_su)
        direct_transactions.append(pending_transaction)
        resolved_pending_residual_candidate = {
            **dict(pending_residual_candidate),
            "claimed_resource_ids": {
                str(key): list(values)
                for key, values in dict(
                    pending_transaction.claimed_resource_ids or {}
                ).items()
            },
            "converted_nodes_claimed": True,
        }
        direct_residual_template_fits.append({
            "product_role": pending_role,
            "chain_23_count": int(pending_chain_23_count),
            "required_counts": dict(pending_required_counts),
            "conversion_count": int(
                pending_residual_candidate.get("conversion_count", 0) or 0
            ),
            "materialized_after_rebuild": True,
        })
        if pending_role == "residual_flex_chain":
            direct_flex_count += 1
            remaining_flex_goal = max(0, int(remaining_flex_goal) - 1)
            remaining_flex_capacity = max(
                0,
                int(remaining_flex_capacity) - 1,
            )
        residual_contract_id += 1

    while True:
        tolerated_remaining_ids_by_su = _tolerated_remaining_closure_ids(
            pool,
            # A final pool of at most three real closure resources is allowed
            # even when the preferred flex target could not be reached.
            remaining_flex_goal=0,
        )
        if tolerated_remaining_ids_by_su or not _remaining_closure_ids_by_su(pool):
            break
        residual_remaining_budget = max(
            0,
            int(max_primary_adjustments_per_stage),
        )
        next_product = _plan_next_residual_product(
            phase_eval=phase_eval,
            iteration_ledger=iteration_ledger,
            H=H_work,
            E_target=E_target,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=rotation_cursor,
            remaining_flex_chains=remaining_flex_goal,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            max_conversions=residual_remaining_budget,
            max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
            contract_id=residual_contract_id,
            remaining_flex_capacity=remaining_flex_capacity,
            prefix_groups=[],
            accounting_initial_pool=residual_initial_pool,
            forbidden_primary_conversion_signatures=forbidden_resource_conversion_signatures,
            allow_h_outside_nonworsening=allow_h_outside_nonworsening,
        )
        residual_candidate_attempts.extend(
            list(next_product.get("attempts", []) or [])
        )
        if str(next_product.get("mode", "")) == "direct":
            iteration_ledger = next_product["ledger"]
            if list(
                getattr(iteration_ledger, "completed_products", []) or []
            ):
                committed_residual_products.append(dict(
                    iteration_ledger.completed_products[-1] or {}
                ))
            pool = deepcopy(iteration_ledger.available_ids_by_su)
            direct_transactions.append(next_product["transaction"])
            direct_residual_template_fits.append({
                "product_role": str(next_product.get("product_role", "")),
                "chain_23_count": int(next_product.get("chain_23_count", 0) or 0),
                "required_counts": dict(next_product.get("required_counts", {}) or {}),
                "conversion_count": 0,
            })
            if str(next_product.get("product_role", "")) == "residual_flex_chain":
                direct_flex_count += 1
                remaining_flex_goal = max(0, int(remaining_flex_goal) - 1)
                remaining_flex_capacity = max(0, int(remaining_flex_capacity) - 1)
            residual_contract_id += 1
            continue
        if str(next_product.get("mode", "")) == "conversion":
            residual_plan = next_product["plan"]
            residual_histogram = residual_plan.histogram_adjustment_plan
            selected_product = {
                "product_role": str(next_product.get("product_role", "")),
                "chain_23_count": int(next_product.get("chain_23_count", 0) or 0),
                "required_counts": dict(next_product.get("required_counts", {}) or {}),
                "conversion_count": int(residual_histogram.primary_adjustment_count),
                "converted_node_ids": sorted({
                    int(node_id)
                    for transaction in list(
                        residual_plan.phase_result.transactions or []
                    )
                    for node_id in list(
                        transaction.converted_node_ids or []
                    )
                }),
            }
            residual_plan.diagnostics.update({
                **dict(su24_stage_diagnostics),
                "residual_stage_completed": False,
                "residual_rebuild_reason": "converted_product_requires_fresh_rebuild",
                "residual_converted_product_count": 1,
                "remaining_required_flex_goal": int(remaining_flex_goal),
                "selected_residual_product": selected_product,
                "pending_residual_product_candidate": selected_product,
                "resolved_pending_residual_product": dict(
                    resolved_pending_residual_candidate
                ),
                "direct_residual_template_fits": list(direct_residual_template_fits),
                "residual_candidate_attempts": list(residual_candidate_attempts),
                "committed_residual_products": list(
                    committed_residual_products
                ),
                "completed_phases": list(completed_phases),
            })
            return residual_plan
        break
    remaining_closure_ids_by_su = _remaining_closure_ids_by_su(pool)
    remaining_closure_ids = [
        int(node_id)
        for values in remaining_closure_ids_by_su.values()
        for node_id in values
    ]
    tolerated_remaining_ids_by_su = _tolerated_remaining_closure_ids(
        pool,
        remaining_flex_goal=0,
    )
    if remaining_closure_ids and not tolerated_remaining_ids_by_su:
        proof = {
            "phase": "residual_resource_self_closure",
            "reason": "residual_topology_conversion_infeasible",
            "failed_contract": {
                "phase": "residual_resource_self_closure",
                "contract_kind": "residual_topology_conversion",
                "target_role": "residual_flex_or_side_chain",
                "source_node_ids": list(remaining_closure_ids),
            },
            "required_counts": {},
            "available_counts": {str(key): len(values) for key, values in pool.items()},
            "missing_counts": {},
            "attempted_conversions": list(residual_candidate_attempts),
            "template_checks": {"residual_template_executable": False},
            "direct_residual_template_fits": list(direct_residual_template_fits),
            "residual_candidate_attempts": list(residual_candidate_attempts),
        }
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="residual_resource_self_closure",
            plan_kind="residual_topology_conversion_infeasible",
            active_subphase="residual_resource_self_closure",
            phase_result=Stage3PhaseResult(
                active_phase="residual_resource_self_closure",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "residual_resource_self_closure",
                "infeasibility_proof": proof,
                "resolved_pending_residual_product": dict(
                    resolved_pending_residual_candidate
                ),
                "committed_residual_products": list(
                    committed_residual_products
                ),
                **dict(su24_stage_diagnostics),
            },
            runtime={
                **_runtime_with_pool(phase_eval, pool),
                "iteration_topology_ledger": iteration_ledger,
            },
        ), h_tolerance)
    completed_phases.append("residual_resource_self_closure")

    tolerated_remaining_ids_by_su = _tolerated_remaining_closure_ids(
        pool,
        remaining_flex_goal=0,
    )

    profiles = build_su24_resource_role_profiles(nodes, pool)
    classification_view = dict(getattr(phase_eval, "classification_view", {}) or {})
    classification_failures = {
        "unclassified_nodes": list(classification_view.get("unclassified_nodes", []) or []),
        "classification_rejects": list(classification_view.get("classification_rejects", []) or []),
        "claim_conflicts": list(classification_view.get("claim_conflicts", []) or []),
    }
    classification_ok = not any(classification_failures.values())
    pending_contracts_final = list(getattr(iteration_ledger, "pending_contracts", []) or [])
    ownership_conflicts = list(getattr(iteration_ledger, "conflicts", []) or [])
    classified_fragment_ids = {
        int(dict(item or {}).get("fragment_id"))
        for item in list(classification_view.get("fragments", []) or [])
        if dict(item or {}).get("fragment_id") is not None
    }
    current_fragment_node_ids: Dict[int, Set[int]] = {}
    for raw_fragment in list(classification_view.get("fragments", []) or []):
        fragment = dict(raw_fragment or {})
        fragment_id = fragment.get("fragment_id")
        if fragment_id is None:
            continue
        current_fragment_node_ids[int(fragment_id)] = {
            int(value)
            for value in (
                list(fragment.get("source_node_ids", []) or [])
                + list(fragment.get("claimed_node_ids", []) or [])
            )
        }
    historical_completed_fragment_ids: Set[int] = set()
    completed_fragment_ids: Set[int] = set()
    unknown_completed_fragment_ids: Set[int] = set()
    fragment_id_remap: Dict[str, List[int]] = {}
    for raw_product in list(getattr(iteration_ledger, "completed_products", []) or []):
        product = dict(raw_product or {})
        historical_ids = {
            int(value)
            for value in list(product.get("source_fragment_ids", []) or [])
        }
        historical_completed_fragment_ids.update(historical_ids)
        product_node_ids = {
            int(value)
            for value in (
                list(product.get("fixed_source_node_ids", []) or [])
                + [
                    node_id
                    for values in dict(
                        product.get("claimed_resource_ids_by_su", {}) or {}
                    ).values()
                    for node_id in list(values or [])
                ]
            )
        }
        mapped_ids = {
            int(fragment_id)
            for fragment_id, fragment_node_ids in current_fragment_node_ids.items()
            if fragment_node_ids and bool(fragment_node_ids & product_node_ids)
        }
        direct_ids = (
            set()
            if product_node_ids
            else historical_ids & classified_fragment_ids
        )
        completed_fragment_ids.update(direct_ids | mapped_ids)
        if historical_ids and not (direct_ids or mapped_ids):
            unknown_completed_fragment_ids.update(historical_ids)
        for historical_id in sorted(historical_ids - classified_fragment_ids):
            if mapped_ids:
                fragment_id_remap[str(historical_id)] = sorted(mapped_ids)
    pending_fragment_ids = {
        int(value)
        for contract in pending_contracts_final
        for value in list(dict(contract or {}).get("source_fragment_ids", []) or [])
    }
    post_external_transient_fragment_ids = (
        classified_fragment_ids - completed_fragment_ids - pending_fragment_ids
        if not external_phases_pending
        else set()
    )
    audited_classified_fragment_ids = (
        classified_fragment_ids - post_external_transient_fragment_ids
    )
    fragment_coverage = {
        "classified_fragment_ids": sorted(audited_classified_fragment_ids),
        "completed_fragment_ids": sorted(completed_fragment_ids),
        "pending_fragment_ids": sorted(pending_fragment_ids),
        "uncovered_fragment_ids": sorted(
            audited_classified_fragment_ids
            - completed_fragment_ids
            - pending_fragment_ids
        ),
        "unknown_completed_fragment_ids": sorted(
            unknown_completed_fragment_ids
        ),
        "unknown_pending_fragment_ids": sorted(pending_fragment_ids - classified_fragment_ids),
    }
    fragment_coverage_ok = not any(
        list(fragment_coverage[key] or [])
        for key in (
            "uncovered_fragment_ids",
            "unknown_completed_fragment_ids",
            "unknown_pending_fragment_ids",
        )
    )
    flex_enforced = bool(dict(getattr(phase_eval, "flex_audit", {}) or {}).get("enforced", False))
    flex_final = (
        int(flex_counts.get("final", 0) or 0)
        + int(direct_flex_count)
    )
    tolerated_remaining_count = sum(
        len(values) for values in tolerated_remaining_ids_by_su.values()
    )
    residual_release_allowed = bool(
        int(tolerated_remaining_count)
        <= int(MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES)
    )
    flex_ok = bool(
        not flex_enforced
        or residual_release_allowed
        or (
            int(flex_bounds.get("min", 0) or 0)
            <= int(flex_final)
            <= int(flex_bounds.get("max", 0) or 0)
        )
    )
    remaining_su24_after_terminalization = list(pool.get(24, []) or [])
    su24_terminalization_ok = not remaining_su24_after_terminalization
    failed_invariants = {
        "classification_ok": bool(classification_ok),
        "pending_contracts_empty": not bool(pending_contracts_final),
        "ownership_conflicts_empty": not bool(ownership_conflicts),
        "fragment_coverage_ok": bool(fragment_coverage_ok),
        "flex_ok": bool(flex_ok),
        "su24_terminalization_ok": bool(su24_terminalization_ok),
    }
    if (
        not classification_ok
        or pending_contracts_final
        or ownership_conflicts
        or not fragment_coverage_ok
        or not flex_ok
        or not su24_terminalization_ok
    ):
        proof = {
            "phase": "complete",
            "reason": "stage3_completion_invariants_failed",
            "failed_contract": dict(pending_contracts_final[0]) if pending_contracts_final else {},
            "required_counts": dict(dict(pending_contracts_final[0]).get("required_counts", {}) or {}) if pending_contracts_final else {},
            "available_counts": {str(key): len(values) for key, values in pool.items()},
            "missing_counts": (
                _deficit(
                    pool,
                    {
                        int(key): int(value)
                        for key, value in dict(dict(pending_contracts_final[0]).get("required_counts", {}) or {}).items()
                    },
                )
                if pending_contracts_final else {}
            ),
            "attempted_conversions": [
                dict(item or {}) for item in list(infeasible_contracts or [])
            ],
            "h_or_element_rejections": [
                dict(item or {}).get("proof", {}) for item in list(infeasible_contracts or [])
            ],
            "template_checks": {
                "su24_ring_executable": False,
                "residual_template_executable": False,
            },
            "classification_failures": classification_failures,
            "ownership_conflicts": ownership_conflicts,
            "fragment_coverage": fragment_coverage,
            "post_external_transient_fragment_ids": sorted(
                post_external_transient_fragment_ids
            ),
            "flex_bounds": dict(flex_bounds),
            "flex_final": int(flex_final),
            "remaining_su24_node_ids": [
                int(value) for value in remaining_su24_after_terminalization
            ],
            "direct_residual_template_fits": list(direct_residual_template_fits),
            "failed_invariants": dict(failed_invariants),
        }
        return _annotate_h_hard_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="complete",
            plan_kind="stage3_completion_infeasible",
            active_subphase="complete",
            pending_contracts=[dict(item or {}) for item in pending_contracts_final],
            phase_result=Stage3PhaseResult(
                active_phase="complete",
                phase_completed=False,
                rebuild_required=False,
                transactions=list(direct_transactions),
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "complete",
                "completed_phases": completed_phases,
                "infeasibility_proof": proof,
                **dict(su24_stage_diagnostics),
            },
            runtime={
                **_runtime_with_pool(phase_eval, pool),
                "iteration_topology_ledger": iteration_ledger,
            },
        ), h_tolerance)
    phase_result = Stage3PhaseResult(
        active_phase="complete",
        phase_completed=True,
        rebuild_required=False,
        symbolic_topology_count=len(direct_transactions),
        transactions=list(direct_transactions),
        stop_reason="stage3_complete",
        infeasibility_proof={},
    )
    return _annotate_h_hard_rule(Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_complete",
        phase="complete",
        plan_kind="stage3_state_machine_complete",
        active_subphase="complete",
        phase_result=phase_result,
        diagnostics={
            "stage3_phase": "complete",
            "completed_phases": completed_phases,
            "sequential_single_topology": True,
            **dict(external_diagnostics),
            **dict(su24_stage_diagnostics),
            "su24_resource_profiles": [item.to_dict() for item in profiles],
            "residual_flex_preferred_23": int(residual_flex_preferred_23),
            "residual_flex_min_23": int(residual_flex_min_23),
            "residual_side_preferred_23": int(residual_side_preferred_23),
            "residual_side_min_23": int(residual_side_min_23),
            "tolerated_unassembled_resource_limit": int(
                MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES
            ),
            "tolerated_remaining_ids_by_su": {
                str(key): list(values)
                for key, values in tolerated_remaining_ids_by_su.items()
            },
            "tolerated_remaining_count": sum(
                len(values) for values in tolerated_remaining_ids_by_su.values()
            ),
            "remaining_su24_count": int(
                len(remaining_su24_after_terminalization)
            ),
            "protected_terminal_aromatic_node_ids": sorted(
                protected_terminal_ids
            ),
            "resolved_pending_residual_product": dict(
                resolved_pending_residual_candidate
            ),
            "committed_residual_products": list(
                committed_residual_products
            ),
            "infeasible_contracts": list(infeasible_contracts),
            "stale_identity_contracts": list(stale_identity_contracts),
            "residual_infeasibility_proof": {},
            "tolerated_remaining_proof": (
                {
                    "reason": "tolerated_unassembled_closure_resources",
                    "remaining_ids_by_su": {
                        str(key): list(values)
                        for key, values in tolerated_remaining_ids_by_su.items()
                    },
                    "remaining_count": sum(
                        len(values)
                        for values in tolerated_remaining_ids_by_su.values()
                    ),
                }
                if tolerated_remaining_ids_by_su
                else {}
            ),
            "direct_residual_template_fits": list(direct_residual_template_fits),
            "fragment_coverage": fragment_coverage,
            "post_external_transient_fragment_ids": sorted(
                post_external_transient_fragment_ids
            ),
        },
        runtime={
            **_runtime_with_pool(phase_eval, pool),
            "iteration_topology_ledger": iteration_ledger,
        },
    ), h_tolerance)
