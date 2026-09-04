from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from ...layer1.port_role_balance import audit_layer1_port_roles
from ..classification.classifier import classify_topology_units
from .phase_eval import assemble_topology_units
from ..resources.histogram import (
    _conversion_signature,
    _h_total,
    _stateful_conversion_bundle_signature,
    build_histogram_adjustment_plan,
    next_su11_aromatic_target,
)
from ..resources.conversion_scoring import (
    apply_conversions_to_H,
    evaluate_flex_impact,
)
from ..resources.profiles import (
    DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    MIN_STAGE3_AROMATIC_RESOURCE_COUNT,
    MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    h_soft_bounds,
)
from ..resources.accounting import build_resource_accounting_snapshot
from ..resources.types import (
    EXTERNAL_ADJUSTMENT_RESOURCE_TYPES,
    GENERIC_CLOSURE_RESOURCE_TYPES,
    INTERNAL_CONVERSION_SOURCE_TYPES,
    STAGE3_ACCOUNTING_SU_TYPES,
    SU24_RING_RESOURCE_TYPE,
)
from ..assembly.state.iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_priority,
    contract_source_status,
    normalize_slot,
    required_counts_from_slots,
    seed_ledger_topology_state,
)
from ..assembly.state.round_context import (
    build_stage3_round_context,
    normalize_stage3_product,
)
from ..resources.spectrum_rebalance import (
    fixed_flex_capacity_spectrum_audit,
    generate_spectrum_candidates,
)
from ..stage3_models import (
    AtomicAdjustmentGroup,
    HistogramConversion,
    HistogramAdjustmentPlan,
    Stage3AdjustmentBundle,
    Stage3AdjustmentOp,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    Stage3OwnershipState,
    Stage3ProgressPhase,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyResourceTransaction,
    merge_stage3_contracts_by_owner,
    stage3_contract_owner_key,
)

RESIDUAL_SELF_CLOSURE_TYPES = (11, 22, 23)
EXTERNAL_ROLE_ORDER = (
    ("external_flex_closure", "flex_chain"),
    ("external_fatty_ring_closure", "fatty_ring"),
    ("external_side_chain_closure", "side_chain"),
)
EXTERNAL_STAGE = "external_topology_assembly"
EXTERNAL_PHASE_NAMES = frozenset(phase for phase, _ in EXTERNAL_ROLE_ORDER)
FIXED_FLEX_CAPACITY_DONOR_ORDER = (23, 12, 11)
MAX_DIAGNOSTIC_ATTEMPT_SAMPLES = 8


def _classification_with_live_allocator_pool(
    classification: Any,
    assembler: Any,
) -> Any:
    """Overlay the allocator's current concrete inventory on classification.

    Classification describes the fresh generation only. During continuation,
    ordinary SU conversions move concrete IDs between buckets while the
    classifier object remains immutable; round-context construction therefore
    needs this small runtime overlay to avoid resurrecting the old inventory.
    """
    allocator = getattr(assembler, "allocator", None)
    if allocator is None:
        return classification
    remaining = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(
            getattr(allocator, "remaining_ids_by_su", {}) or {}
        ).items()
    }
    if not remaining:
        return classification
    effective = deepcopy(classification)
    pool = deepcopy(
        dict(getattr(effective, "resource_pool", {}) or {})
    )
    pool["remaining_ids_by_su"] = dict(remaining)
    donor_map = {
        str(su_type): list(remaining.get(str(su_type), []) or [])
        for su_type in (12, 13)
    }
    pool["su11_donor_ids_by_su"] = donor_map
    pool["su11_donor_counts"] = {
        str(key): len(values) for key, values in donor_map.items()
    }
    effective.resource_pool = pool
    return effective


def _diagnostic_attempt_samples(
    attempts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items = [dict(item or {}) for item in list(attempts or []) if dict(item or {})]
    limit = int(MAX_DIAGNOSTIC_ATTEMPT_SAMPLES)
    if len(items) <= limit:
        return items
    head_count = limit // 2
    return items[:head_count] + items[-(limit - head_count):]


def _h_soft_rule_diagnostics(h_tolerance: float) -> Dict[str, Any]:
    effective_tolerance = min(1.0, max(0.0, float(h_tolerance)))
    return {
        "h_tolerance_mode": "stage3_symmetric_soft_window",
        "effective_h_tolerance": float(effective_tolerance),
        "stage3_soft_lower_ratio": -float(effective_tolerance),
        "stage3_soft_upper_ratio": float(effective_tolerance),
        "topology_gate": False,
    }


def _annotate_h_soft_rule(plan: Stage3AdjustmentPlan, h_tolerance: float) -> Stage3AdjustmentPlan:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostics.setdefault("h_constraint", _h_soft_rule_diagnostics(h_tolerance))
    plan.diagnostics = diagnostics
    return plan


def _evaluate_external_phases(
    *,
    phase_eval: Any,
    nodes: Sequence[Any],
    H: Any = None,
    classification: Any = None,
    allow_ring_recovery: bool = False,
    external_assembler: Any = None,
) -> Any:
    classification_obj = classification or classify_topology_units(list(nodes or []))
    generation_id = int(getattr(phase_eval, "generation_id", 0) or 0)
    live_assembler = external_assembler
    if live_assembler is not None and int(
        getattr(live_assembler, "generation_id", generation_id)
    ) == generation_id:
        live_assembler.allow_ring_recovery = bool(allow_ring_recovery)
        if callable(getattr(live_assembler, "sync_after_resource_adjustments", None)):
            live_assembler.sync_after_resource_adjustments(
                list(nodes or []),
                H=H,
            )
        assembly_obj = live_assembler.continue_assembly()
    else:
        assembly_obj = assemble_topology_units(
            nodes=list(nodes or []),
            classification=classification_obj,
            flex_audit=dict(getattr(phase_eval, "flex_audit", {}) or {}),
            generation_id=generation_id,
            allow_ring_recovery=bool(allow_ring_recovery),
        )
        live_assembler = getattr(assembly_obj, "_runtime_assembler", None)
        if live_assembler is not None and callable(
            getattr(live_assembler, "sync_after_resource_adjustments", None)
        ):
            live_assembler.sync_after_resource_adjustments(
                list(nodes or []),
                H=H,
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
    # The assembler owns the live resource identity after a materialization
    # pass.  Reconcile the serialized contract snapshots once more at this
    # boundary so a contract cannot claim a resource that was consumed by a
    # product created in the same continuation.  This is deliberately a
    # snapshot update only; it does not claim, release, or reorder resources.
    if live_assembler is not None:
        live_allocator = getattr(live_assembler, "allocator", None)
        if live_allocator is not None:
            live_raw_counts = {
                str(key): int(value)
                for key, value in dict(
                    live_allocator.available_counts() or {}
                ).items()
            }
            live_claimable_counts = {
                str(key): int(value)
                for key, value in dict(
                    live_allocator.claimable_counts() or {}
                ).items()
            }
            for contract in pending_contracts:
                required = dict(contract.get("required_counts", {}) or {})
                live_missing = {
                    str(key): max(
                        0,
                        int(value)
                        - int(live_claimable_counts.get(str(key), 0)),
                    )
                    for key, value in required.items()
                    if int(value) > int(live_claimable_counts.get(str(key), 0))
                }
                contract["raw_available_counts"] = dict(live_raw_counts)
                contract["available_counts"] = dict(live_claimable_counts)
                contract["claimable_available_counts"] = dict(
                    live_claimable_counts
                )
                contract["missing_counts"] = dict(live_missing)
                proof = dict(contract.get("proof", {}) or {})
                proof.update({
                    "raw_available_counts": dict(live_raw_counts),
                    "available_counts": dict(live_claimable_counts),
                    "claimable_available_counts": dict(live_claimable_counts),
                    "missing_counts": dict(live_missing),
                })
                contract["proof"] = proof
                contract_diagnostics = dict(contract.get("diagnostics", {}) or {})
                contract_diagnostics.update({
                    "raw_available_counts": dict(live_raw_counts),
                    "claimable_available_counts": dict(live_claimable_counts),
                    "missing_counts": dict(live_missing),
                })
                contract["diagnostics"] = contract_diagnostics
            # Keep the public assembly view consistent with the canonical
            # contracts consumed below.  Without this mirror, later
            # diagnostics would still serialize the pre-reconciliation
            # ``claimable_available_counts`` even though planning used the
            # live allocator values.
            diagnostics["pending_contracts"] = [
                dict(item or {}) for item in pending_contracts if dict(item or {})
            ]
            assembly["diagnostics"] = {
                **dict(assembly.get("diagnostics", {}) or {}),
                "pending_contracts": list(diagnostics["pending_contracts"]),
            }
            assembly["pending_contracts"] = list(diagnostics["pending_contracts"])
            # ``working`` is created below, so retain the reconciled view in a
            # local object that is assigned after the deepcopy.
    working = deepcopy(phase_eval)
    # Dynamic runtime handle; Stage3AdjustmentPlan.to_dict intentionally omits
    # phase-evaluation runtime objects and therefore cannot leak this object.
    working.external_assembler = live_assembler
    working.assembly_attempted = True
    working.assembly_view = assembly
    released_resource_ids_by_su = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(
            dict(assembly.get("resource_ledger", {}) or {}).get(
                "released_resource_ids_by_su", {}
            )
            or {}
        ).items()
        if list(values or [])
    }
    recycled_conversion_rows = list(
        assembly.get("planned_conversions", [])
        or dict(assembly.get("resource_ledger", {}) or {}).get(
            "recycled_resource_conversion_ops", []
        )
        or []
    )
    recycled_su24_to_23_ids = sorted({
        int(dict(item or {}).get("node_id"))
        for item in recycled_conversion_rows
        if dict(item or {}).get("from_su") is not None
        and int(dict(item or {}).get("from_su")) == 24
        and dict(item or {}).get("to_su") is not None
        and int(dict(item or {}).get("to_su")) == 23
        and dict(item or {}).get("node_id") is not None
    })
    # Rebuild the round context from classification plus the assembler's
    # explicit recovery ledger. Keeping the pre-assembly context here would
    # leave dismantled fixed nodes marked as fragment-owned resources.
    # The external assembler works on a transaction-local node copy. Apply
    # only the reported reclassification to a context copy so ownership sees
    # the same effective SU type without mutating the caller graph.
    context_nodes = deepcopy(list(nodes or []))
    context_nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in context_nodes
        if int(getattr(node, "global_id", -1)) >= 0
    }
    for node_id in recycled_su24_to_23_ids:
        node = context_nodes_by_id.get(int(node_id))
        if node is not None:
            setattr(node, "su_type", 23)
    context_classification = _classification_with_live_allocator_pool(
        classification_obj,
        live_assembler,
    )
    working.round_context = build_stage3_round_context(
        nodes=context_nodes,
        classification=context_classification,
        flex_audit=dict(getattr(phase_eval, "flex_audit", {}) or {}),
        generation_id=int(getattr(phase_eval, "generation_id", 0) or 0),
        released_resource_ids_by_su=released_resource_ids_by_su,
        recycled_su24_to_23_ids=recycled_su24_to_23_ids,
    )
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
    working.generic_closure_pool_ids_by_su = {
        int(key): [int(value) for value in list(values or [])]
        for key, values in dict(
            ownership.get("generic_closure_pool_ids_by_su", {}) or {}
        ).items()
    }
    working.su24_ring_resource_ids = [
        int(value)
        for value in list(ownership.get("su24_ring_resource_ids", []) or [])
    ]
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


def _merge_released_resource_diagnostics(
    phase_eval: Any = None,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge ring-recovery provenance without exposing a SU24 donor pool."""
    merged: Dict[str, Any] = dict(existing or {})
    released_by_su: Dict[str, Set[int]] = {}

    def collect(raw: Any) -> None:
        for raw_su, raw_ids in dict(raw or {}).items():
            try:
                su_key = str(int(raw_su))
            except (TypeError, ValueError):
                continue
            released_by_su.setdefault(su_key, set()).update(
                int(value) for value in list(raw_ids or [])
            )

    collect(merged.get("released_resource_ids_by_su", {}))
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    collect(ledger.get("released_resource_ids_by_su", {}))

    released_su24: Set[int] = {
        int(value)
        for value in list(merged.get("released_su24_ids", []) or [])
    }
    released_su24.update(
        int(value) for value in list(ledger.get("released_su24_ids", []) or [])
    )
    released_su24.update(released_by_su.get("24", set()))
    recycled = {
        int(value)
        for value in list(merged.get("recycled_su24_to_23_ids", []) or [])
    }
    recycled.update(
        int(value)
        for value in list(ledger.get("recycled_su24_to_23_ids", []) or [])
    )
    # Release provenance is normalized to the effective ordinary SU23 pool.
    recycled.update(released_su24)
    released_by_su.setdefault("24", set()).difference_update(recycled)
    released_by_su.setdefault("23", set()).update(recycled)

    merged["released_su24_ids"] = sorted(released_su24)
    merged["recycled_su24_to_23_ids"] = sorted(recycled)
    merged["released_resource_ids_by_su"] = {
        str(key): sorted(values)
        for key, values in sorted(released_by_su.items())
        if values
    }
    return merged


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
    recycled_resource_ids: Optional[Set[int]] = None,
) -> Dict[int, List[int]]:
    by_su = _remaining_closure_ids_by_su(pool)
    recycled_ids = {int(value) for value in list(recycled_resource_ids or set())}
    if recycled_ids & {
        int(value) for values in by_su.values() for value in values
    }:
        return {}
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


def _contract_claimable_counts(contract: Dict[str, Any]) -> Dict[int, int]:
    data = dict(contract or {})
    diagnostics = dict(data.get("diagnostics", {}) or {})
    proof = dict(data.get("proof", {}) or {})
    raw = (
        data.get("claimable_available_counts")
        or proof.get("claimable_available_counts")
        or diagnostics.get("claimable_available_counts")
        or proof.get("available_counts")
        or {}
    )
    return {
        int(key): max(0, int(value))
        for key, value in dict(raw or {}).items()
        if str(key).lstrip("-").isdigit()
    }


def _contract_reported_missing_counts(contract: Dict[str, Any]) -> Dict[int, int]:
    """Read the materializer's concrete claim failure, when present."""
    data = dict(contract or {})
    diagnostics = dict(data.get("diagnostics", {}) or {})
    proof = dict(data.get("proof", {}) or {})
    # ``missing_counts`` is a current contract snapshot.  An empty snapshot
    # is meaningful after a successful resource conversion and must not fall
    # back to the historical claim failure retained for diagnostics.
    raw: Any = {}
    for source in (data, proof, diagnostics):
        if "missing_counts" in source:
            raw = source.get("missing_counts") or {}
            break
    else:
        raw = diagnostics.get("claim_failure_missing_counts") or {}
    return {
        int(key): max(0, int(value))
        for key, value in dict(raw or {}).items()
        if str(key).lstrip("-").isdigit() and int(value) > 0
    }


def _deficit_from_counts(
    available_counts: Dict[int, int],
    required: Dict[int, int],
) -> Dict[int, int]:
    return {
        int(su_type): int(count) - int(available_counts.get(int(su_type), 0))
        for su_type, count in dict(required or {}).items()
        if int(count) > int(available_counts.get(int(su_type), 0))
    }


def _deficit_after_protection(
    pool: Dict[int, List[int]],
    required: Dict[int, int],
    protected_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    protected = {
        int(key): max(0, int(value))
        for key, value in dict(protected_counts or {}).items()
    }
    return {
        int(su_type): int(count) - max(
            0,
            len(list(pool.get(int(su_type), []) or []))
            - int(protected.get(int(su_type), 0)),
        )
        for su_type, count in required.items()
        if int(count) > max(
            0,
            len(list(pool.get(int(su_type), []) or []))
            - int(protected.get(int(su_type), 0)),
        )
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


def _residual_product_sequence(product_role: str, chain_23_count: int) -> List[int]:
    if str(product_role) == "residual_flex_chain":
        return [11] + [23] * int(chain_23_count) + [11]
    return [11] + [23] * int(chain_23_count) + [22]


def _residual_product_topology(product_role: str, chain_23_count: int) -> str:
    return "-".join(str(value) for value in _residual_product_sequence(product_role, chain_23_count))


def _plan_internal_su24_aromatic_conversion(
    *,
    phase_eval: Any,
    iteration_ledger: Any,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    contract_id: int,
    max_conversions: int,
) -> Optional[Dict[str, Any]]:
    """Create the internal-only conversion for an isolated SU24 resource.

    SU24 is a ring skeleton during external assembly and may not be used as a
    generic donor there.  Once internal residual closure is reached, an
    isolated SU24 cannot form a ring by itself; it becomes an aromatic donor
    and is allowed to convert to SU12/SU13.  This is deliberately represented
    as a concrete one-node histogram transaction so H, the ledger and the
    external budget machinery use the same identity.
    """
    if int(max_conversions) <= 0:
        return None
    pool = dict(getattr(iteration_ledger, "available_ids_by_su", {}) or {})
    su24_ids = sorted({
        int(value) for value in list(pool.get(SU24_RING_RESOURCE_TYPE, []) or [])
    })
    if not su24_ids:
        return None
    node_id = int(su24_ids[0])
    if int(H.numel()) <= SU24_RING_RESOURCE_TYPE:
        return None
    # Keep the two aromatic buckets balanced while preserving the current H
    # direction.  Either target is a legal internal aromatic donor.
    h12 = int(H[12].item()) if int(H.numel()) > 12 else 0
    h13 = int(H[13].item()) if int(H.numel()) > 13 else 0
    target = 12 if h12 <= h13 else 13
    if int(H.numel()) <= int(target):
        return None
    op = HistogramConversion(
        from_su=SU24_RING_RESOURCE_TYPE,
        to_su=int(target),
        count=1,
        node_ids=[int(node_id)],
        reason="internal_isolated_su24_to_aromatic_donor",
    )
    H_after = apply_conversions_to_H(H, [op])
    if H_after is None:
        return None
    inventory_before = {
        int(key): len(list(values or []))
        for key, values in pool.items()
    }
    inventory_after = dict(inventory_before)
    inventory_after[SU24_RING_RESOURCE_TYPE] = max(
        0,
        int(inventory_after.get(SU24_RING_RESOURCE_TYPE, 0)) - 1,
    )
    inventory_after[int(target)] = int(inventory_after.get(int(target), 0)) + 1
    group = AtomicAdjustmentGroup(
        primary_op=op,
        combined_delta={24: -1, int(target): 1},
        demand_before={},
        demand_after={},
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=_h_total(H_after, special_degree_meta),
        h_after_final=_h_total(H_after, special_degree_meta),
        scoring_diagnostics={"internal_only": True},
    )
    histogram = HistogramAdjustmentPlan(
        phase="internal_topology",
        groups=[group],
        primary_adjustment_count=1,
        physical_conversion_count=1,
        H_before=H.detach().clone(),
        H_after=H_after.detach().clone(),
        inventory_before=inventory_before,
        inventory_after=inventory_after,
        accounting_diagnostics={
            "mode": "internal_isolated_su24_reclassification",
            "allow_unreleased_su24_conversion": True,
            "node_id": int(node_id),
            "target_su": int(target),
        },
    )
    ledger_after = _ledger_after_histogram_conversions(
        iteration_ledger,
        histogram,
    )
    transaction = TopologyResourceTransaction(
        phase="internal_topology",
        contract_id=int(contract_id),
        target_role="internal_su24_aromatic_donor",
        source_node_ids=[],
        converted_node_ids=[int(node_id)],
        conversion_path=[op.to_dict()],
        product_node_ids=[],
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=_h_total(H_after, special_degree_meta),
        h_after=_h_total(H_after, special_degree_meta),
        commit_scope="physical_histogram",
        transaction_origin="internal_su24_reclassification",
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="internal_su24_aromatic_conversion",
        phase="internal_topology",
        plan_kind="internal_su24_aromatic_conversion",
        active_subphase="residual_resource_self_closure",
        planned_conversions=[Stage3AdjustmentOp(
            op="convert",
            from_su=24,
            to_su=int(target),
            node_id=int(node_id),
            reason=str(op.reason),
        )],
        adjustment_bundles=[Stage3AdjustmentBundle(
            bundle_id=f"internal_topology:{int(contract_id)}:su24",
            phase="internal_topology",
            bundle_ops=[op.to_dict()],
            diagnostics={"internal_only": True},
        )],
        histogram_adjustment_plan=histogram,
        phase_result=Stage3PhaseResult(
            active_phase="internal_topology",
            phase_completed=False,
            rebuild_required=False,
            primary_adjustment_count=1,
            transactions=[transaction],
            stop_reason="internal_su24_aromatic_conversion",
        ),
        diagnostics={
            "stage3_phase": "internal_topology",
            "internal_only_su24_conversion": True,
            "internal_su24_node_id": int(node_id),
            "internal_su24_target_su": int(target),
        },
        runtime={
            **_runtime_with_pool(
                phase_eval,
                ledger_after.available_ids_by_su,
            ),
            "iteration_topology_ledger": ledger_after,
        },
    )
    return {
        "mode": "conversion",
        "plan": plan,
        "product_role": "internal_su24_aromatic_donor",
        "chain_23_count": 0,
        "required_counts": {},
        "attempt_count": 0,
        "attempts": [],
    }


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
        selected["attempt_count"] = len(attempts)
        selected["attempts"] = _diagnostic_attempt_samples(attempts)
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
    h_lower, h_upper = h_soft_bounds(target_h, h_tolerance)

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
            # Released SU24 is a donor only in this internal residual phase;
            # the histogram scorer still binds it by concrete released ID.
            forbidden_conversion_su_types=[],
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
            "missing_counts": dict(
                dict(candidate_plan.phase_result.infeasibility_proof or {}).get(
                    "missing_counts", {}
                )
                or {}
            ),
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
            h_error = (
                abs(float(h_after) - float(target_h))
                if target_h is not None
                else 0.0
            )
            missing_slot_count = max(0, int(template_size) - int(direct_coverage))
            if not required_flex and int(primary_count) > int(missing_slot_count):
                attempts.append({
                    "product_role": product_role,
                    "chain_23_count": int(chain_23_count),
                    "required_counts": dict(required_counts),
                    "reason": "optional_template_uses_excess_conversions",
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
        selected["attempt_count"] = len(attempts)
        selected["attempts"] = _diagnostic_attempt_samples(attempts)
        selected["selection_reason"] = (
            "ranked_required_flex_conversion_candidates"
            if required_flex
            else "ranked_single_gap_optional_conversion_candidates"
        )
        return selected
    # Ordinary 11/12/13/22/23 conversion has been exhausted.  Only now may
    # an isolated SU24 be reclassified for the internal aromatic pool.
    su24_conversion = _plan_internal_su24_aromatic_conversion(
        phase_eval=phase_eval,
        iteration_ledger=iteration_ledger,
        H=H,
        special_degree_meta=special_degree_meta,
        contract_id=contract_id,
        max_conversions=max_conversions,
    )
    if su24_conversion is not None:
        return su24_conversion
    return {
        "mode": "none",
        "attempt_count": len(attempts),
        "attempts": _diagnostic_attempt_samples(attempts),
    }


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
    committed_claimed_ids = _committed_occupied_ids_by_su(
        transactions,
        aggregate_ids_by_su=aggregate_ids,
        groups=groups,
    )
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_ids,
        authorized_ids_before_by_su=initial_pool,
        final_remaining_ids_by_su=_accounting_remaining_pool(final_pool),
        committed_claimed_ids_by_su=committed_claimed_ids,
        groups=groups,
    )
    accounting = _merge_released_resource_diagnostics(phase_eval, accounting)
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


def _fixed_flex_capacity_candidate(
    *,
    H: torch.Tensor,
    audit: Dict[str, Any],
    pool: Dict[int, List[int]],
    spectrum_audit: Dict[str, Any],
    max_primary_adjustments: int,
    special_degree_meta: Optional[Dict[str, Any]],
    forbidden_primary_conversion_signatures: Optional[Sequence[str]] = None,
) -> Tuple[List[Stage3AdjustmentOp], Dict[str, Any]]:
    explicit_count = int(audit.get("explicit_flex_count", 0) or 0)
    before_impact = evaluate_flex_impact(H, H, audit)
    before = dict(before_impact.before or {})
    before_upper = int(before.get("flex_upper", audit.get("flex_upper", 0)) or 0)
    before_excess = max(0, int(explicit_count) - int(before_upper))
    su23_spectrum_eligible = bool(spectrum_audit.get("eligible", False))
    attempts: List[Dict[str, Any]] = []
    forbidden_signatures = {
        str(value)
        for value in list(forbidden_primary_conversion_signatures or [])
        if str(value)
    }
    limit = max(0, int(max_primary_adjustments))
    available_by_su: Dict[int, List[int]] = {}
    balance_pool_by_su = {
        int(su_type): sorted({
            int(value) for value in list(pool.get(int(su_type), []) or [])
        })
        for su_type in (11, 12, 13)
    }
    for priority, source_su in enumerate(FIXED_FLEX_CAPACITY_DONOR_ORDER):
        source_ids = [
            int(value) for value in list(pool.get(int(source_su), []) or [])
        ]
        if int(source_su) == 23 and not su23_spectrum_eligible:
            attempts.append({
                "source_su": int(source_su),
                "priority": int(priority),
                "reason": (
                    "su23_requires_layer2_aliphatic_surplus_and_su13_deficit"
                ),
                "available_count": len(source_ids),
                "layer2_contract": bool(
                    spectrum_audit.get("layer2_contract", False)
                ),
                "aliphatic_surplus": bool(
                    spectrum_audit.get("aliphatic_surplus", False)
                ),
                "su13_deficit": bool(
                    spectrum_audit.get("su13_deficit", False)
                ),
            })
            continue
        physical_count = (
            int(H[int(source_su)].item())
            if 0 <= int(source_su) < int(H.numel())
            else 0
        )
        if int(source_su) in {12, 13}:
            physical_count = max(
                0,
                int(physical_count) - int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
            )
        available_count = min(
            int(limit),
            len(source_ids),
            int(physical_count),
        )
        if available_count <= 0:
            minimum_preserved = bool(
                int(source_su) in {12, 13}
                and bool(source_ids)
                and int(H[int(source_su)].item())
                <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
            )
            attempts.append({
                "source_su": int(source_su),
                "priority": int(priority),
                "reason": (
                    "aromatic_minimum_preserved"
                    if minimum_preserved
                    else "no_real_unclaimed_donor"
                ),
                "available_count": len(source_ids),
            })
            continue
        available_by_su[int(source_su)] = source_ids[: int(available_count)]

    eligible_sources = [
        int(source_su)
        for source_su in FIXED_FLEX_CAPACITY_DONOR_ORDER
        if available_by_su.get(int(source_su))
    ]
    candidates_evaluated = 0
    best_rank: Optional[Tuple[Any, ...]] = None
    best_payload: Optional[Dict[str, Any]] = None

    def _role_parities_repairable(candidate: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        role_audit = audit_layer1_port_roles(candidate, special_degree_meta)
        if bool(role_audit.get("ok", False)):
            return True, {
                "role_audit": role_audit,
                "role_balance_required": False,
            }
        if (
            not bool(role_audit.get("total_even", False))
            or not bool(role_audit.get("role_parity_consistent", False))
        ):
            return False, {
                "role_audit": role_audit,
                "role_balance_required": False,
            }
        for balance_source, balance_target in ((11, 12), (12, 11)):
            if int(candidate[int(balance_source)].item()) <= 0:
                continue
            if (
                int(balance_source) in {12, 13}
                and int(candidate[int(balance_source)].item())
                <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
            ):
                continue
            balanced = candidate.detach().clone()
            balanced[int(balance_source)] -= 1
            balanced[int(balance_target)] += 1
            balanced_audit = audit_layer1_port_roles(
                balanced,
                special_degree_meta,
            )
            if bool(balanced_audit.get("ok", False)):
                return True, {
                    "role_audit": role_audit,
                    "role_balance_required": True,
                    "role_balance_from_su": int(balance_source),
                    "role_balance_to_su": int(balance_target),
                    "role_balance_audit": balanced_audit,
                }
        return False, {
            "role_audit": role_audit,
            "role_balance_required": True,
            "role_balance_reason": "su11_su12_role_balance_unavailable",
        }

    def _evaluate_counts(counts: Dict[int, int]) -> None:
        nonlocal candidates_evaluated, best_rank, best_payload
        total_count = sum(int(value) for value in counts.values())
        if total_count <= 0 or total_count > int(limit):
            return
        conversions: List[HistogramConversion] = []
        for source_su in FIXED_FLEX_CAPACITY_DONOR_ORDER:
            source_ids = available_by_su.get(int(source_su), [])
            selected_ids = source_ids[: int(counts.get(int(source_su), 0))]
            conversions.extend(
                HistogramConversion(
                    from_su=int(source_su),
                    to_su=13,
                    count=1,
                    node_ids=[int(node_id)],
                    reason=(
                        f"fixed_flex_capacity_expand_su{int(source_su)}_to_su13"
                    ),
                )
                for node_id in selected_ids
            )
        # Reject a previously failed real conversion state while enumerating
        # candidates. This keeps retry logic at the planner boundary instead
        # of letting histogram accounting rediscover the same type-level plan.
        candidate_signature = _stateful_conversion_bundle_signature(
            H, conversions
        )
        if (
            candidate_signature in forbidden_signatures
            or any(
                _conversion_signature(conversion) in forbidden_signatures
                for conversion in conversions
            )
        ):
            attempts.append({
                "counts": dict(counts),
                "reason": "forbidden_resource_conversion_signature",
            })
            return
        candidate_H = apply_conversions_to_H(H, conversions)
        if candidate_H is None:
            return
        candidates_evaluated += 1
        impact = evaluate_flex_impact(H, candidate_H, audit)
        after = dict(impact.after or {})
        after_lower = int(after.get("flex_lower", 0) or 0)
        after_upper = int(after.get("flex_upper", 0) or 0)
        after_excess = max(0, int(explicit_count) - int(after_upper))
        valid_progress = bool(
            str(after.get("scenario", "") or "") != "invalid"
            and int(after_lower) <= int(explicit_count)
            and int(after_upper) > int(before_upper)
            and int(after_excess) < int(before_excess)
        )
        if not valid_progress:
            return
        resolved = bool(
            int(after_lower) <= int(explicit_count) <= int(after_upper)
        )
        parity_balance_options: List[Dict[str, Any]] = []
        role_repair: Dict[str, Any] = {}
        role_balance_ids: List[int] = []
        selected_primary_node_ids = {
            int(node_id)
            for conversion in conversions
            for node_id in list(conversion.node_ids or [])
        }

        def _available_balance_ids(
            source_su: int,
            excluded_ids: Set[int],
        ) -> List[int]:
            if (
                int(source_su) in {12, 13}
                and int(candidate_H[int(source_su)].item())
                <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
            ):
                return []
            return [
                int(node_id)
                for node_id in list(balance_pool_by_su.get(int(source_su), []) or [])
                if int(node_id) not in excluded_ids
            ]

        def _role_balance_ids(
            repair: Dict[str, Any],
            excluded_ids: Set[int],
        ) -> List[int]:
            if not bool(repair.get("role_balance_required", False)):
                return []
            source_su = int(repair.get("role_balance_from_su", -1) or -1)
            return _available_balance_ids(source_su, excluded_ids)

        degree_parity_flips = sum(
            int(counts.get(int(source_su), 0))
            for source_su in (10, 11, 12)
        )
        if int(degree_parity_flips) % 2 != 0:
            for balance_source, balance_target in ((12, 13), (13, 12)):
                if int(candidate_H[int(balance_source)].item()) <= 0:
                    continue
                if (
                    int(candidate_H[int(balance_source)].item())
                    <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
                ):
                    continue
                balanced_H = candidate_H.detach().clone()
                balanced_H[int(balance_source)] -= 1
                balanced_H[int(balance_target)] += 1
                balanced_impact = evaluate_flex_impact(H, balanced_H, audit)
                balanced_after = dict(balanced_impact.after or {})
                balanced_lower = int(
                    balanced_after.get("flex_lower", 0) or 0
                )
                balanced_upper = int(
                    balanced_after.get("flex_upper", 0) or 0
                )
                balanced_excess = max(
                    0, int(explicit_count) - int(balanced_upper)
                )
                preserves_progress = bool(
                    str(balanced_after.get("scenario", "") or "") != "invalid"
                    and int(balanced_lower) <= int(explicit_count)
                    and int(balanced_upper) > int(before_upper)
                    and int(balanced_excess) < int(before_excess)
                    and (
                        not resolved
                        or int(balanced_lower)
                        <= int(explicit_count)
                        <= int(balanced_upper)
                    )
                )
                roles_repairable, candidate_role_repair = (
                    _role_parities_repairable(balanced_H)
                )
                if not preserves_progress or not roles_repairable:
                    continue
                total_balance_ids = _available_balance_ids(
                    int(balance_source),
                    selected_primary_node_ids,
                )
                if not total_balance_ids:
                    attempts.append({
                        "counts": dict(counts),
                        "reason": "real_balance_donor_unavailable",
                        "from_su": int(balance_source),
                        "to_su": int(balance_target),
                    })
                    continue
                role_balance_ids = _role_balance_ids(
                    candidate_role_repair,
                    selected_primary_node_ids | {int(total_balance_ids[0])},
                )
                if bool(candidate_role_repair.get("role_balance_required", False)) and not role_balance_ids:
                    attempts.append({
                        "counts": dict(counts),
                        "reason": "real_role_balance_donor_unavailable",
                    })
                    continue
                parity_balance_options.append({
                    "from_su": int(balance_source),
                    "to_su": int(balance_target),
                    "flex_after": balanced_after,
                    "excess_after": int(balanced_excess),
                    "role_repair": candidate_role_repair,
                    "node_ids": [int(total_balance_ids[0])],
                    "role_balance_node_ids": (
                        [int(role_balance_ids[0])]
                        if role_balance_ids
                        else []
                    ),
                })
            if not parity_balance_options:
                return
            role_repair = dict(parity_balance_options[0]["role_repair"])
            role_balance_ids = list(
                parity_balance_options[0].get("role_balance_node_ids", [])
            )
        else:
            roles_repairable, role_repair = _role_parities_repairable(
                candidate_H
            )
            if not roles_repairable:
                return
            role_balance_ids = _role_balance_ids(
                role_repair,
                selected_primary_node_ids,
            )
            if bool(role_repair.get("role_balance_required", False)) and not role_balance_ids:
                attempts.append({
                    "counts": dict(counts),
                    "reason": "real_role_balance_donor_unavailable",
                })
                return
        donor_priority = tuple(
            -int(counts.get(int(source_su), 0))
            for source_su in FIXED_FLEX_CAPACITY_DONOR_ORDER
        )
        rank = (
            0 if resolved else 1,
            int(total_count) if resolved else -int(total_count),
            donor_priority,
            int(after_excess),
            -int(after_upper),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_payload = {
                "conversions": conversions,
                "counts": {
                    int(key): int(value)
                    for key, value in counts.items()
                    if int(value) > 0
                },
                "after": after,
                "after_excess": int(after_excess),
                "resolved": bool(resolved),
                "total_count": int(total_count),
                "degree_parity_flips": int(degree_parity_flips),
                "parity_balance_options": parity_balance_options,
                "role_repair": role_repair,
                "selected_primary_node_ids": sorted(selected_primary_node_ids),
                "role_balance_node_ids": [int(role_balance_ids[0])] if role_balance_ids else [],
            }

    def _enumerate_counts(
        source_index: int,
        remaining_budget: int,
        counts: Dict[int, int],
    ) -> None:
        if source_index >= len(eligible_sources):
            _evaluate_counts(counts)
            return
        source_su = int(eligible_sources[source_index])
        maximum = min(
            int(remaining_budget),
            len(available_by_su.get(int(source_su), [])),
        )
        for count in range(0, int(maximum) + 1):
            counts[int(source_su)] = int(count)
            _enumerate_counts(
                int(source_index) + 1,
                int(remaining_budget) - int(count),
                counts,
            )
        counts.pop(int(source_su), None)

    _enumerate_counts(0, int(limit), {})
    if best_payload is not None:
        conversions = list(best_payload["conversions"])
        operations = [
            Stage3AdjustmentOp(
                op="convert",
                from_su=int(conversion.from_su),
                to_su=int(conversion.to_su),
                node_id=int(conversion.node_ids[0]),
                reason=str(conversion.reason),
            )
            for conversion in conversions
        ]
        selected_counts = dict(best_payload["counts"])
        selected_sources = [
            int(source_su)
            for source_su in FIXED_FLEX_CAPACITY_DONOR_ORDER
            if int(selected_counts.get(int(source_su), 0)) > 0
        ]
        return operations, {
            "selected_source_su": selected_sources[0] if selected_sources else None,
            "selected_source_sus": selected_sources,
            "selected_counts_by_su": selected_counts,
            "selected_conversion_count": int(best_payload["total_count"]),
            "selected_completed_flex_range": bool(best_payload["resolved"]),
            "selection_stop_reason": (
                "fixed_flex_count_entered_valid_range"
                if bool(best_payload["resolved"])
                else "primary_adjustment_limit_reached"
                if int(best_payload["total_count"]) >= int(limit)
                else "available_valid_donors_exhausted"
            ),
            "su23_spectrum_eligible": bool(su23_spectrum_eligible),
            "selected_aromatic_fallback": bool(
                any(int(source_su) != 23 for source_su in selected_sources)
            ),
            "authorized_adjustment_su_types": list(
                EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
            ),
            "predicted_role_repair": dict(best_payload["role_repair"]),
            "predicted_flex_before": dict(before),
            "predicted_flex_after": dict(best_payload["after"]),
            "predicted_excess_before": int(before_excess),
            "predicted_excess_after": int(best_payload["after_excess"]),
            "predicted_degree_parity_flips": int(
                best_payload["degree_parity_flips"]
            ),
            "predicted_parity_balance_options": list(
                best_payload["parity_balance_options"]
            ),
            "predicted_parity_balance_node_ids": list(
                (best_payload["parity_balance_options"] or [{}])[0].get("node_ids", [])
            ),
            "predicted_role_balance_node_ids": list(
                best_payload.get("role_balance_node_ids", [])
            ),
            "selected_primary_node_ids": list(
                best_payload.get("selected_primary_node_ids", [])
            ),
            "candidate_count_vectors_evaluated": int(candidates_evaluated),
            "candidate_attempt_count": len(attempts),
            "candidate_attempts": _diagnostic_attempt_samples(attempts),
        }

    return [], {
        "selected_source_su": None,
        "selected_conversion_count": 0,
        "su23_spectrum_eligible": bool(su23_spectrum_eligible),
        "predicted_flex_before": dict(before),
        "predicted_flex_after": {},
        "candidate_count_vectors_evaluated": int(candidates_evaluated),
        "candidate_attempt_count": len(attempts),
        "candidate_attempts": _diagnostic_attempt_samples(attempts),
    }


def _fixed_flex_capacity_expansion_plan(
    *,
    phase_eval: Any,
    H: torch.Tensor,
    E_target: Any,
    difference_spectrum: Optional[Dict[str, Any]],
    max_primary_adjustments: int,
    max_h_compensation_ops: int,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    forbidden_primary_conversion_signatures: Optional[Sequence[str]] = None,
) -> Stage3AdjustmentPlan:
    audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
    pool = {
        int(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(
            phase_eval.flex_reconciliation_ids_by_su or {}
        ).items()
    }
    pool_source = "preassembly_flex_reconciliation_pool"
    excess = max(0, int(audit.get("above_range_excess", 0) or 0))
    limit = max(0, int(max_primary_adjustments))
    spectrum_audit = fixed_flex_capacity_spectrum_audit(difference_spectrum)
    proof = {
        "phase": "flex_range_reconciliation",
        "reason": "fixed_flex_capacity_expansion_infeasible",
        "fixed_flex_excess": int(excess),
        "fixed_flex_count": int(audit.get("explicit_flex_count", 0) or 0),
        "flex_bounds": {
            "lower": int(audit.get("flex_lower", 0) or 0),
            "upper": int(audit.get("flex_upper", 0) or 0),
        },
        "spectrum_audit": dict(spectrum_audit),
        "available_donor_ids_by_su": {
            str(su_type): list(pool.get(int(su_type), []) or [])
            for su_type in FIXED_FLEX_CAPACITY_DONOR_ORDER
        },
        "primary_conversion_limit": int(limit),
        "donor_pool_source": str(pool_source),
    }
    proof["su23_spectrum_eligible"] = bool(
        spectrum_audit.get("eligible", False)
    )
    proof["su23_spectrum_rule"] = (
        "current_layer2_0_90ppm_surplus_and_su13_deficit"
    )
    proof["aromatic_fallback_order"] = [12, 11]

    operations, selection = _fixed_flex_capacity_candidate(
        H=H,
        audit=audit,
        pool=pool,
        spectrum_audit=spectrum_audit,
        max_primary_adjustments=limit,
        special_degree_meta=special_degree_meta,
        forbidden_primary_conversion_signatures=forbidden_primary_conversion_signatures,
    )
    proof["candidate_selection"] = dict(selection)
    if not operations:
        proof["reason"] = (
            "fixed_flex_capacity_all_real_node_candidates_exhausted"
        )

    if not operations:
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="flex_range_reconciliation",
            plan_kind="fixed_flex_capacity_expansion_infeasible",
            active_subphase="flex_range_reconciliation",
            phase_result=Stage3PhaseResult(
                active_phase="flex_range_reconciliation",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "flex_range_reconciliation",
            },
            runtime=_runtime_with_pool(phase_eval, pool),
        ), h_tolerance)

    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="fixed_flex_capacity_expansion_pending",
        phase="flex_range_reconciliation",
        plan_kind="fixed_flex_capacity_expansion",
        active_subphase="flex_range_reconciliation",
        planned_conversions=list(operations),
        runtime=_runtime_with_pool(phase_eval, pool),
        diagnostics={
            "stage3_phase": "flex_range_reconciliation",
            "fixed_flex_capacity_expansion": dict(selection),
            "spectrum_audit": dict(spectrum_audit),
            "donor_pool_source": str(pool_source),
            "candidate_alternatives_enabled": True,
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
        max_primary_adjustments=limit,
        # The fixed-flex candidate already binds the real primary and port
        # balance donors.  Generic H compensation can consume those donors
        # after preflight and make the accepted candidate unrebuildable. H is
        # a soft objective here, so keep this phase to its explicit feedback
        # conversions and perform port balancing separately in Block C.
        max_h_compensation_ops=0,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    if not list(histogram.groups or []):
        proof["reason"] = (
            "fixed_flex_capacity_all_real_node_candidates_exhausted"
        )
        proof["histogram_unresolved_counts"] = dict(histogram.unresolved_counts or {})
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
            **dict(getattr(plan, "diagnostics", {}) or {}),
            "stage3_phase": "flex_range_reconciliation",
        }
        return _annotate_h_soft_rule(plan, h_tolerance)

    realized_impact = evaluate_flex_impact(H, histogram.H_after, audit)
    realized_before = dict(realized_impact.before or {})
    realized_after = dict(realized_impact.after or {})
    fixed_count = int(audit.get("explicit_flex_count", 0) or 0)
    realized_before_excess = max(
        0,
        int(fixed_count) - int(realized_before.get("flex_upper", 0) or 0),
    )
    realized_after_excess = max(
        0,
        int(fixed_count) - int(realized_after.get("flex_upper", 0) or 0),
    )
    realized_valid = bool(
        str(realized_after.get("scenario", "") or "") != "invalid"
        and int(realized_after.get("flex_lower", 0) or 0) <= int(fixed_count)
        and int(realized_after.get("flex_upper", 0) or 0)
        > int(realized_before.get("flex_upper", 0) or 0)
        and int(realized_after_excess) < int(realized_before_excess)
    )
    if not realized_valid:
        proof.update({
            "reason": "fixed_flex_capacity_histogram_did_not_expand_range",
            "realized_flex_before": dict(realized_before),
            "realized_flex_after": dict(realized_after),
        })
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
            **dict(getattr(plan, "diagnostics", {}) or {}),
            "stage3_phase": "flex_range_reconciliation",
        }
        return _annotate_h_soft_rule(plan, h_tolerance)

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
            **dict(getattr(plan, "diagnostics", {}) or {}),
            "stage3_phase": "flex_range_reconciliation",
        }
        return _annotate_h_soft_rule(plan, h_tolerance)

    transaction = TopologyResourceTransaction(
        phase="flex_range_reconciliation",
        contract_id=-1002,
        target_role="fixed_flex_capacity_expansion",
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
        "fixed_flex_excess_before": int(excess),
        "fixed_flex_excess_after_prediction": int(realized_after_excess),
        "fixed_flex_capacity_expansion": dict(selection),
        "spectrum_audit": dict(spectrum_audit),
        "realized_flex_before": dict(realized_before),
        "realized_flex_after": dict(realized_after),
        "primary_conversion_limit": int(limit),
        "donor_pool_source": str(pool_source),
        "su11_rotation_index_after": int(histogram.su11_rotation_index_after),
        "fresh_layer12_rebuild_required": True,
        "restart_stage3_after_rebuild": True,
        "candidate_alternatives_enabled": True,
        "forbidden_primary_conversion_signatures": sorted({
            str(value)
            for value in list(forbidden_primary_conversion_signatures or [])
            if str(value)
        }),
    }
    return _annotate_h_soft_rule(plan, h_tolerance)


def _runtime_with_pool(
    phase_eval: Any,
    pool: Dict[int, List[int]],
    *,
    protected_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    # Runtime lives only for the current plan/apply transaction.  Copy the one
    # ledger that is amended here instead of cloning the complete assembly,
    # which contains every product, pending contract and classification view.
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = deepcopy(dict(assembly.get("resource_ledger", {}) or {}))
    # Normalize legacy recovery snapshots at the runtime boundary as well as
    # in RoundContext.  A caller may pass a stale pool that still lists a
    # dismantled SU24 node under bucket 24; recovered nodes are ordinary SU23
    # resources and must never be available to a later ring candidate.
    recycled_su24_ids = {
        int(value)
        for value in list(
            ledger.get("recycled_su24_to_23_ids", []) or []
        )
    }
    recycled_su24_ids.update(
        int(value)
        for value in list(ledger.get("released_su24_ids", []) or [])
    )
    released_map = dict(ledger.get("released_resource_ids_by_su", {}) or {})
    recycled_su24_ids.update(
        int(value)
        for value in list(
            released_map.get("24", []) or released_map.get(24, []) or []
        )
    )
    if recycled_su24_ids:
        normalized_pool = {
            int(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(pool or {}).items()
        }
        recycled_present = recycled_su24_ids & set(
            normalized_pool.get(SU24_RING_RESOURCE_TYPE, [])
        )
        if recycled_present:
            normalized_pool[SU24_RING_RESOURCE_TYPE] = [
                int(value)
                for value in normalized_pool.get(SU24_RING_RESOURCE_TYPE, [])
                if int(value) not in recycled_present
            ]
            normalized_pool.setdefault(23, [])
            normalized_pool[23] = sorted(
                set(normalized_pool[23]) | recycled_present
            )
        pool = normalized_pool
        ledger["recycled_su24_to_23_ids"] = sorted(recycled_su24_ids)
        normalized_released = {
            str(key): {
                int(value) for value in list(values or [])
            }
            for key, values in released_map.items()
        }
        normalized_released.setdefault("24", set()).difference_update(
            recycled_su24_ids
        )
        normalized_released.setdefault("23", set()).update(recycled_su24_ids)
        ledger["released_resource_ids_by_su"] = {
            str(key): sorted(values)
            for key, values in normalized_released.items()
            if values
        }
    ledger["remaining_ids_by_su"] = {str(key): list(values) for key, values in pool.items()}
    ledger["remaining_counts"] = {str(key): len(values) for key, values in pool.items()}
    ledger["remaining_counts_by_su"] = dict(ledger["remaining_counts"])
    ledger["convertible_counts"] = {
        str(su_type): len(list(pool.get(int(su_type), []) or []))
        for su_type in EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
    }
    ledger["convertible_ids_by_su"] = {
        str(su_type): sorted({
            int(value) for value in list(pool.get(int(su_type), []) or [])
        })
        for su_type in EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
    }
    ledger["recycled_su24_to_23_ids"] = sorted({
        int(value)
        for value in list(ledger.get("recycled_su24_to_23_ids", []) or [])
    })
    ledger["recycled_resource_conversion_ops"] = [
        dict(item or {})
        for item in list(ledger.get("recycled_resource_conversion_ops", []) or [])
        if dict(item or {})
    ]
    ledger["su11_donor_ids_by_su"] = {
        str(su_type): list(pool.get(int(su_type), []) or [])
        for su_type in (12, 13)
    }
    ledger["su11_donor_counts"] = {
        str(su_type): len(list(pool.get(int(su_type), []) or []))
        for su_type in (12, 13)
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
        "external_assembler": getattr(phase_eval, "external_assembler", None),
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
        support_context_node_ids=sorted(set(support_ids)),
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
    # A caller may provide a serialized/stale round context. Re-apply the
    # explicit recovery ledger after that ownership overlay so released fixed
    # classifier nodes cannot remain falsely owned by their old fragment.
    raw_released_resource_ids_by_su = dict(
        dict(assembly.get("resource_ledger", {}) or {}).get(
            "released_resource_ids_by_su", {}
        )
        or {}
    )
    recycled_su24_ids = {
        int(value)
        for value in list(
            dict(assembly.get("resource_ledger", {}) or {}).get(
                "recycled_su24_to_23_ids", []
            )
            or []
        )
    }
    released_resource_ids_by_su: Dict[int, Set[int]] = {}
    for raw_su, raw_ids in raw_released_resource_ids_by_su.items():
        if not str(raw_su).lstrip("-").isdigit():
            continue
        su_type = int(raw_su)
        # Legacy snapshots may still label a recycled node as SU24. Normalize
        # it before rebuilding the iteration ledger so it cannot re-enter the
        # ring resource bucket.
        target_su = 23 if su_type == SU24_RING_RESOURCE_TYPE else su_type
        ids = {int(value) for value in list(raw_ids or [])}
        released_resource_ids_by_su.setdefault(target_su, set()).update(ids)
        if su_type == SU24_RING_RESOURCE_TYPE:
            recycled_su24_ids.update(ids)
    released_resource_ids_by_su.setdefault(23, set()).update(recycled_su24_ids)
    completed_product_node_ids = {
        int(value)
        for product in list(assembly.get("products", []) or [])
        for value in (
            list(dict(product or {}).get("fixed_source_node_ids", []) or [])
            + [
                int(value)
                for values in dict(
                    dict(product or {}).get("claimed_resource_ids_by_su", {})
                    or {}
                ).values()
                for value in list(values or [])
            ]
        )
    }
    node_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
    }
    for su_type, node_ids in released_resource_ids_by_su.items():
        for node_id in node_ids:
            if int(node_id) in completed_product_node_ids:
                continue
            node = node_by_id.get(int(node_id))
            owner = ledger.ownership_by_node_id.get(int(node_id))
            if node is None or owner is None:
                continue
            actual_su = (
                23
                if int(node_id) in recycled_su24_ids
                else int(getattr(node, "su_type", -1))
            )
            if actual_su not in set(
                GENERIC_CLOSURE_RESOURCE_TYPES
            ) | {SU24_RING_RESOURCE_TYPE}:
                continue
            for values in ledger.available_ids_by_su.values():
                while int(node_id) in values:
                    values.remove(int(node_id))
            ledger.available_ids_by_su.setdefault(actual_su, []).append(int(node_id))
            ledger.available_ids_by_su[actual_su] = sorted(
                set(ledger.available_ids_by_su[actual_su])
            )
            owner.su_type = actual_su
            owner.state = (
                Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value
                if actual_su == SU24_RING_RESOURCE_TYPE
                else Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value
            )
            owner.owner_fragment_id = None
            owner.owner_contract_id = None
            owner.owner_product_id = ""
            owner.owner_slot_id = ""
            owner.role = str(owner.state)
            owner.reason = "recycled_pure_resource_ring" if int(node_id) in recycled_su24_ids else "released_pure_resource_ring"
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


def _restore_uncommitted_ring_pool_nodes(
    ledger: Any,
    nodes: Sequence[Any],
    *,
    restore_resource_ids_by_su: Optional[
        Dict[int, Sequence[int]]
    ] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Restore explicitly supplied uncommitted resource IDs.

    SU24 ring resources are owned by the round context. They must not be
    rediscovered and injected into the pool when the state machine enters the
    internal phase; only an explicit release/recovery path may add IDs.
    """
    working = deepcopy(ledger)
    restored_resource_ids = {
        int(value)
        for values in dict(restore_resource_ids_by_su or {}).values()
        for value in list(values or [])
    }
    node_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
    }
    committed_states = {
        Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
        Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
        Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
    }
    restored_resource_ids = {
        int(node_id)
        for node_id in restored_resource_ids
        if str(
            getattr(
                working.ownership_by_node_id.get(int(node_id)),
                "state",
                "",
            )
            or ""
        ) not in committed_states
    }
    requested_ids = set(restored_resource_ids)
    committed_product_node_ids: Set[int] = set()
    for raw_product in list(getattr(working, "completed_products", []) or []):
        product = dict(raw_product or {})
        committed_product_node_ids.update(
            int(value)
            for value in list(product.get("fixed_source_node_ids", []) or [])
        )
        committed_product_node_ids.update(
            int(value)
            for values in dict(
                product.get("claimed_resource_ids_by_su", {})
                or {}
            ).values()
            for value in list(values or [])
        )
    requested_ids.difference_update(committed_product_node_ids)
    audit: Dict[str, Any] = {
        "requested_node_ids": sorted(requested_ids),
        "uncommitted_su24_node_ids": [],
        "restored_resource_node_ids": sorted(restored_resource_ids),
        "restored_node_ids": [],
        "missing_node_ids": [],
        "committed_owner_conflicts": [],
        "committed_product_node_ids": sorted(committed_product_node_ids),
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
            Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
        } and str(owner.owner_product_id or ""):
            audit["committed_owner_conflicts"].append({
                "node_id": int(node_id),
                "owner_product_id": str(owner.owner_product_id),
                "state": str(owner.state),
            })
            continue

        su_type = int(getattr(node, "su_type", -1))
        if su_type not in STAGE3_ACCOUNTING_SU_TYPES:
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
            if su_type in GENERIC_CLOSURE_RESOURCE_TYPES
            else Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value
            if su_type == SU24_RING_RESOURCE_TYPE
            else Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value
        )
        owner.owner_fragment_id = None
        owner.owner_contract_id = None
        owner.owner_product_id = ""
        owner.owner_slot_id = ""
        owner.role = str(owner.state)
        owner.reason = "restored_for_current_generation_ring_pool"
        audit["restored_node_ids"].append(int(node_id))
        audit["ownership_after"][str(node_id)] = {
            "su_type": int(su_type),
            "state": str(owner.state),
        }

    audit["ok"] = not bool(
        audit["missing_node_ids"] or audit["committed_owner_conflicts"]
    )
    return working, audit


def _uncommitted_physical_su24_ids(
    ledger: Any,
    nodes: Sequence[Any],
) -> List[int]:
    """Return real SU24 nodes not owned by a committed topology product."""
    committed_states = {
        Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
        Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
        Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
    }
    ownership = dict(getattr(ledger, "ownership_by_node_id", {}) or {})
    return sorted({
        int(getattr(node, "global_id", -1))
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
        and int(getattr(node, "su_type", -1)) == 24
        and str(
            getattr(
                ownership.get(int(getattr(node, "global_id", -1))),
                "state",
                "",
            )
            or ""
        ) not in committed_states
    })


def _all_physical_su24_ids(nodes: Sequence[Any]) -> List[int]:
    """Return the temporary spectrum authorization view for physical SU24."""
    return sorted({
        int(getattr(node, "global_id", -1))
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
        and int(getattr(node, "su_type", -1)) == 24
    })


def _spectrum_trial_node_ids(
    nodes: Sequence[Any],
    pool: Dict[int, Sequence[int]],
) -> Dict[int, List[int]]:
    spectrum_pool = {
        int(su_type): sorted({
            int(value)
            for value in list(pool.get(int(su_type), []) or [])
        })
        for su_type in (11, 22, 23, 24)
    }
    # Spectrum trials rebuild Layer1 and Stage3 from scratch when accepted, so
    # every physical SU24 is a valid temporary donor, including product-owned
    # nodes omitted from the canonical assembly pool.
    spectrum_pool[24] = _all_physical_su24_ids(nodes)
    return spectrum_pool


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
        source_node_ids=list(contract.claimed_source_node_ids),
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
    released_su24_ids = {
        int(value)
        for value in list(
            dict(getattr(histogram_plan, "accounting_diagnostics", {}) or {}).get(
                "released_su24_ids", []
            )
            or []
        )
    }
    accounting_diagnostics = dict(
        getattr(histogram_plan, "accounting_diagnostics", {}) or {}
    )
    released_map = dict(
        accounting_diagnostics.get("released_resource_ids_by_su", {}) or {}
    )
    released_su24_ids.update(
        int(value)
        for value in list(
            released_map.get("24", []) or released_map.get(24, []) or []
        )
    )
    for group in list(getattr(histogram_plan, "groups", []) or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            all_candidates = list(work.get(int(op.from_su), []))
            candidates = list(all_candidates)
            if int(op.from_su) == SU24_RING_RESOURCE_TYPE and not (
                bool(accounting_diagnostics.get("allow_unreleased_su24_conversion", False))
                and int(op.to_su) in {12, 13}
            ):
                candidates = [
                    int(value) for value in candidates
                    if int(value) in released_su24_ids
                ]
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
                int(value) for value in all_candidates if int(value) not in selected_set
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
                    if int(op.to_su) in GENERIC_CLOSURE_RESOURCE_TYPES
                    else Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value
                    if int(op.to_su) == SU24_RING_RESOURCE_TYPE
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
        getattr(context, "generic_closure_pool_ids_by_su", {}) or {},
        getattr(context, "aromatic_donor_ids_by_su", {}) or {},
    ):
        for key, values in dict(raw or {}).items():
            result.setdefault(int(key), [])
            result[int(key)].extend(int(value) for value in list(values or []))
    result[SU24_RING_RESOURCE_TYPE] = sorted({
        int(value)
        for value in list(
            getattr(context, "su24_ring_resource_ids", []) or []
        )
    })
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


def _accounting_remaining_pool(
    pool: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    return {
        int(su_type): sorted({int(value) for value in list(node_ids or [])})
        for su_type, node_ids in dict(pool or {}).items()
    }


def _group_operations(groups: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        op.to_dict()
        for group in list(groups or [])
        for op in [group.primary_op] + list(group.h_compensation_ops or [])
    ]


def _h_compensation_count(groups: Sequence[Any]) -> int:
    return sum(len(list(group.h_compensation_ops or [])) for group in list(groups or []))


def _phase_infeasibility_labels(failure_domain: str) -> Tuple[str, str]:
    """Map a concrete failure domain to stable public Stage3 labels."""

    if str(failure_domain or "") == "topology_materialization":
        return (
            "stage3_topology_materialization_invalid",
            "sequential_phase_topology_materialization_invalid",
        )
    return (
        "stage3_completion_invariants_failed",
        "sequential_phase_infeasible_after_progress",
    )


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
    committed_claimed_ids = _committed_occupied_ids_by_su(
        transactions,
        aggregate_ids_by_su=aggregate_ids,
        groups=groups,
    )
    accounting = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su=aggregate_ids,
        authorized_ids_before_by_su=initial_pool,
        final_remaining_ids_by_su=_accounting_remaining_pool(final_pool),
        committed_claimed_ids_by_su=committed_claimed_ids,
        groups=groups,
    )
    accounting = _merge_released_resource_diagnostics(phase_eval, accounting)
    aggregate_required_counts = {
        int(key): int(value)
        for key, value in dict(accounting.get("aggregate_required_counts", {}) or {}).items()
        if int(value) > 0
    }
    demand_before: Dict[int, int] = {}
    histogram = HistogramAdjustmentPlan(
        phase=str(phase),
        groups=list(groups),
        # External budget accounting is per concrete primary node, not per
        # histogram group.  This keeps finalized multi-node batches aligned
        # with the incremental planner and the Block-C budget gate.
        primary_adjustment_count=_external_primary_node_count(groups),
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
        accounting_diagnostics=_merge_released_resource_diagnostics(
            phase_eval,
            accounting,
        ),
        demand_before=dict(demand_before),
        demand_after=dict(demand_before),
        su11_rotation_index_before=int(rotation_before),
        su11_rotation_index_after=int(rotation_after),
    )
    stage_completed = True
    for key in (
        "external_stage_completed",
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
            "claimed_source_node_ids": [],
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
            first_contract_proof.get("failure_domain", "")
            or phase_infeasible_contracts[0].get("failure_domain", "")
            or ""
        )
        phase_failure_reason, phase_plan_kind = _phase_infeasibility_labels(
            first_failure_domain
        )
        contract_available_counts = (
            first_contract_proof.get("claimable_available_counts")
            or first_contract_proof.get("available_counts")
            or {}
        )
        contract_missing_counts = dict(
            first_contract_proof.get("missing_counts", {}) or {}
        )
        if (
            not contract_missing_counts
            and first_failure_domain in {"resource_capacity", "resource_accounting"}
        ):
            contract_required = dict(
                first_contract_proof.get("required_counts", {})
                or phase_infeasible_contracts[0].get("required_counts", {})
                or {}
            )
            contract_missing_counts = {
                str(key): max(
                    0,
                    int(value)
                    - int(
                        dict(contract_available_counts or {}).get(
                            str(key),
                            dict(contract_available_counts or {}).get(int(key), 0),
                        )
                        or 0
                    ),
                )
                for key, value in contract_required.items()
                if int(value)
                > int(
                    dict(contract_available_counts or {}).get(
                        str(key),
                        dict(contract_available_counts or {}).get(int(key), 0),
                    )
                    or 0
                )
            }
        pending_required_counts = {
            str(key): int(value)
            for key, value in dict(
                phase_infeasible_contracts[0].get("required_counts", {}) or {}
            ).items()
            if int(value) > 0
        }
        infeasible_contract_samples = _diagnostic_attempt_samples(
            phase_infeasible_contracts
        )
        proof = {
            "phase": str(phase),
            "reason": str(phase_failure_reason),
            "failure_domain": str(first_failure_domain),
            "failed_contract": dict(phase_infeasible_contracts[0]),
            "required_counts": dict(
                phase_infeasible_contracts[0].get("required_counts", {}) or {}
            ),
            "available_counts": (
                {
                    str(key): int(value)
                    for key, value in dict(contract_available_counts or {}).items()
                }
                if contract_available_counts
                else {
                    str(key): len(values)
                    for key, values in dict(final_pool or {}).items()
                }
            ),
            "raw_available_counts": dict(
                first_contract_proof.get("raw_available_counts", {}) or {}
            ),
            "claimable_available_counts": dict(
                first_contract_proof.get("claimable_available_counts", {}) or {}
            ),
            "protected_counts": dict(
                first_contract_proof.get("protected_counts", {}) or {}
            ),
            "excluded_node_ids": list(
                first_contract_proof.get("excluded_node_ids", []) or []
            ),
            "claim_trace": list(
                first_contract_proof.get("claim_trace", []) or []
            ),
            "missing_counts": {
                str(key): int(value)
                for key, value in contract_missing_counts.items()
                if int(value) > 0
            },
            "conversion_capacity_proof": dict(
                first_contract_proof.get("conversion_capacity_proof", {}) or {}
            ),
            "attempted_conversion_count": len(phase_infeasible_contracts),
            "attempted_conversions": infeasible_contract_samples,
            "h_or_element_rejections": [
                dict(item or {}).get("proof", {})
                for item in infeasible_contract_samples
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
            stop_reason=(
                "phase_topology_materialization_invalid"
                if first_failure_domain == "topology_materialization"
                else "phase_resource_infeasible"
            ),
            infeasibility_proof=proof,
        )
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            ok=False,
            reason=(
                "phase_topology_materialization_invalid"
                if first_failure_domain == "topology_materialization"
                else "phase_resource_infeasible"
            ),
            blocked_reason=str(proof["reason"]),
            phase=str(phase),
            plan_kind=str(phase_plan_kind),
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
                **dict(stage_diagnostics or {}),
            },
            runtime=_runtime_with_pool(phase_eval, final_pool),
        ), h_tolerance)
    external_phase = str(phase).startswith("external")
    external_budget_boundary = bool(
        external_phase
        and dict(stage_diagnostics or {}).get(
            "external_conversion_budget_exhausted", False
        )
    )
    external_continuation = bool(
        external_phase and not external_budget_boundary
    )
    continuation_reason = (
        "external_conversion_continuation"
        if external_continuation
        else "external_conversion_budget_boundary"
        if external_budget_boundary
        else "stage3_rebuild_required"
    )
    phase_result = Stage3PhaseResult(
        active_phase=str(phase),
        phase_completed=bool(stage_completed),
        rebuild_required=bool(external_budget_boundary or not external_phase),
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        symbolic_topology_count=sum(1 for item in transactions if bool(item.committed)),
        transactions=list(transactions),
        stop_reason=continuation_reason,
    )
    return _annotate_h_soft_rule(Stage3AdjustmentPlan(
        ok=True,
        reason=continuation_reason,
        phase=str(phase),
        plan_kind=(
            "external_conversion_continuation"
            if external_continuation
            else "external_conversion_budget_boundary"
            if external_budget_boundary
            else "sequential_phase_transactions"
        ),
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
            "su11_rotation_index_after": int(rotation_after),
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
    forbidden_conversion_su_types: Optional[Sequence[int]] = None,
    reserve_contract_after_conversion: bool = True,
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
            "forbidden_primary_conversion_signatures": sorted({
                str(value)
                for value in list(forbidden_primary_conversion_signatures or [])
                if str(value)
            }),
            "forbidden_conversion_su_types": sorted({
                int(value)
                for value in list(forbidden_conversion_su_types or [])
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
        protected_node_ids=contract.claimed_source_node_ids,
    )
    ledger_after_conversion = (
        _ledger_after_histogram_conversions(
            iteration_ledger,
            histogram,
            protected_node_ids=contract.claimed_source_node_ids,
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
        committed_claimed_ids = _ledger_occupied_ids_by_su(
            ledger_after_conversion,
            aggregate_ids_by_su=aggregate_ids,
        )
        accounting = build_resource_accounting_snapshot(
            aggregate_ids_before_by_su=aggregate_ids,
            authorized_ids_before_by_su=(
                accounting_initial_pool
                if accounting_initial_pool is not None
                else pool
            ),
            final_remaining_ids_by_su=_accounting_remaining_pool(
                ledger_after_conversion.available_ids_by_su,
            ),
            committed_claimed_ids_by_su=committed_claimed_ids,
            groups=accounting_groups,
        )
        accounting = _merge_released_resource_diagnostics(phase_eval, accounting)
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
    materialization_required_proof = (
        {
            "reason": "live_assembler_materialization_required",
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
            list(contract.claimed_source_node_ids)
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
                contract.claimed_source_node_ids
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
        commit_scope=(
            "physical_histogram"
            if requires_rebuild
            else "uncommitted_live_assembler"
        ),
        committed=bool(requires_rebuild),
        rollback_reason=(
            ""
            if requires_rebuild
            else str(
                materialization_required_proof.get("reason")
                or histogram.blocked_reason
                or proof.get("reason", "current_contract_resource_infeasible")
            )
        ),
        infeasibility_proof=(
            {}
            if requires_rebuild
            else dict(
                materialization_required_proof
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
    external_continuation = bool(
        plan.ok and str(contract.phase).startswith("external")
    )
    plan.reason = (
        "external_conversion_continuation"
        if external_continuation
        else "stage3_rebuild_required"
        if plan.ok
        else "phase_resource_infeasible"
    )
    if external_continuation:
        plan.plan_kind = "external_conversion_continuation"
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
        rebuild_required=bool(plan.ok and not external_continuation),
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
            "slot_assignment_proof": dict(
                materialization_required_proof or proof or {}
            ),
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
        "forbidden_conversion_su_types": sorted({
            int(value)
            for value in list(forbidden_conversion_su_types or [])
        }),
        "current_contract": contract.to_dict(),
        "su11_rotation_index_after": int(
            histogram.su11_rotation_index_after
        ),
    }
    plan.unresolved_counts = {str(key): int(value) for key, value in dict(histogram.unresolved_counts or {}).items()}
    plan.requested_counts = {str(key): int(value) for key, value in contract.required_counts.items()}
    if ledger_after_conversion is not None:
        plan.runtime["iteration_topology_ledger"] = ledger_after_conversion
    return _annotate_h_soft_rule(plan, h_tolerance)


def _physical_conversion_count(groups: Sequence[Any]) -> int:
    return sum(
        (2 if group.primary_op.via_su is not None else 1)
        + sum(
            2 if op.via_su is not None else 1
            for op in list(group.h_compensation_ops or [])
        )
        for group in list(groups or [])
    )


def _external_primary_node_count(groups: Sequence[Any]) -> int:
    """Count concrete primary conversion nodes, excluding compensation ops."""
    node_ids: List[int] = []
    for group in list(groups or []):
        primary = getattr(group, "primary_op", None)
        if primary is None:
            continue
        ids = [int(value) for value in list(getattr(primary, "node_ids", []) or [])]
        expected = max(0, int(getattr(primary, "count", 0) or 0))
        if expected <= 0 or len(ids) != expected:
            raise ValueError(
                "primary_conversion_requires_concrete_node_ids: "
                f"expected={expected} provided={len(ids)}"
            )
        if any(int(value) < 0 for value in ids):
            raise ValueError("primary_conversion_contains_invalid_node_id")
        node_ids.extend(ids)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("primary_conversion_reuses_concrete_node_id")
    return len(node_ids)


def _internal_conversion_is_allowed(histogram: Any) -> bool:
    """Validate internal conversion sources and the isolated SU24 exception."""
    for group in list(getattr(histogram, "groups", []) or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            if int(op.from_su) == SU24_RING_RESOURCE_TYPE:
                # An unassembled SU24 is internal-only and may become an
                # aromatic donor, but it must never be converted back to a
                # ring-resource identity.
                if int(op.to_su) not in {12, 13}:
                    return False
            elif int(op.from_su) not in set(INTERNAL_CONVERSION_SOURCE_TYPES):
                return False
            if int(op.to_su) not in set(EXTERNAL_ADJUSTMENT_RESOURCE_TYPES):
                return False
            if int(op.to_su) == SU24_RING_RESOURCE_TYPE:
                return False
            if op.via_su is not None and int(op.via_su) == SU24_RING_RESOURCE_TYPE:
                return False
    return True


def _recycled_su24_to_23_ops(
    phase_eval: Any,
    nodes: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Return only still-pending concrete SU24->SU23 recovery operations.

    ``recycled_resource_conversion_ops`` is durable provenance.  It remains in
    the assembler diagnostics after the operation has been accepted, so it
    cannot by itself be used as a pending queue.  A node is pending only while
    the current generation still exposes it as physical SU24.
    """
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    live_assembler = getattr(phase_eval, "external_assembler", None)
    raw_ops = list(
        getattr(live_assembler, "pending_recycled_resource_conversion_ops", [])
        or ledger.get("recycled_resource_conversion_ops", [])
        or assembly.get("planned_conversions", [])
        or []
    )
    ops: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    node_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
    }
    for raw in raw_ops:
        item = dict(raw or {})
        node_id = item.get("node_id")
        if node_id is None:
            node_ids = list(item.get("node_ids", []) or [])
            node_id = node_ids[0] if node_ids else None
        if node_id is None:
            continue
        if int(item.get("from_su", -1)) != 24 or int(item.get("to_su", -1)) != 23:
            continue
        node_id = int(node_id)
        if node_id in seen:
            continue
        current_node = node_by_id.get(node_id)
        if current_node is not None and int(
            getattr(current_node, "su_type", -1)
        ) != 24:
            # The recovery was already applied in this generation.  Keep the
            # row in diagnostics, but do not emit it as a second transaction.
            continue
        seen.add(node_id)
        ops.append({
            "from_su": 24,
            "to_su": 23,
            "node_id": node_id,
            "count": 1,
            "reason": str(item.get("reason", "recycle_pure_resource_ring") or "recycle_pure_resource_ring"),
        })
    return sorted(ops, key=lambda item: int(item["node_id"]))


def _plan_recycled_su24_to_23(
    *,
    phase_eval: Any,
    nodes: Optional[Sequence[Any]] = None,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    h_tolerance: float,
    external_primary_adjustments_used: int = 0,
    external_max_primary_adjustments: int = 10,
) -> Optional[Stage3AdjustmentPlan]:
    """Build the forced 24->23 commit emitted by pure-ring recovery."""
    assembly_view = dict(getattr(phase_eval, "assembly_view", {}) or {})
    pending_roles = {
        str(dict(item or {}).get("target_role", "") or "")
        for item in list(assembly_view.get("pending_units", []) or [])
        if dict(item or {})
    }
    # External roles are strictly ordered.  If a fatty-ring contract is still
    # pending, recovery belongs to that contract even when later side-chain
    # contracts are also present in the diagnostic snapshot.
    recovery_subphase = (
        "external_fatty_ring_closure"
        if "fatty_ring" in pending_roles
        else "external_side_chain_closure"
    )
    operations = _recycled_su24_to_23_ops(phase_eval, nodes=nodes)
    if not operations:
        return None
    used = max(0, int(external_primary_adjustments_used))
    limit = max(0, int(external_max_primary_adjustments))
    if used >= limit or len(operations) > max(0, limit - used):
        if len(operations) > int(limit):
            proof = {
                "phase": recovery_subphase,
                "reason": "recovery_exceeds_external_budget",
                "external_primary_adjustments_used": int(used),
                "external_max_primary_adjustments": int(limit),
                "requested_recovery_adjustments": len(operations),
                "recovery_is_atomic": True,
                "graph_state_unchanged": True,
            }
            return _annotate_h_soft_rule(Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                phase="external_topology",
                plan_kind="recovery_exceeds_external_budget",
                active_subphase=recovery_subphase,
                blocked_reason=str(proof["reason"]),
                phase_result=Stage3PhaseResult(
                    active_phase="external_topology",
                    phase_completed=False,
                    rebuild_required=False,
                    stop_reason=str(proof["reason"]),
                    infeasibility_proof=proof,
                ),
                diagnostics={
                    "stage3_phase": "external_topology",
                    "external_stage_completed": False,
                    "recovery_exceeds_external_budget": True,
                    "external_primary_adjustments_used": int(used),
                    "external_remaining_budget": max(0, int(limit) - int(used)),
                },
                histogram_adjustment_plan=HistogramAdjustmentPlan(
                    phase="external_topology",
                    groups=[],
                    primary_adjustment_count=0,
                    physical_conversion_count=0,
                    H_before=torch.clamp(H.detach().clone(), min=0).long(),
                    H_after=torch.clamp(H.detach().clone(), min=0).long(),
                ),
                runtime={"phase_evaluation": phase_eval},
            ), h_tolerance)
        proof = {
            "phase": recovery_subphase,
            "reason": "external_conversion_budget_exhausted",
            "external_primary_adjustments_used": int(used),
            "external_max_primary_adjustments": int(limit),
            "requested_recovery_adjustments": len(operations),
        }
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            # A recovery ring is atomic: its SU24 nodes cannot be split across
            # generations.  When the shared external budget cannot accommodate
            # the whole ring, stop this generation cleanly and let Block-C run
            # the normal port-balance/fresh-rebuild boundary without applying a
            # partial recovery.
            ok=True,
            reason="stage3_complete",
            phase="external_topology",
            plan_kind="recycled_su24_to_23_budget_exhausted",
            active_subphase=recovery_subphase,
            blocked_reason=str(proof["reason"]),
            phase_result=Stage3PhaseResult(
                active_phase="external_topology",
                phase_completed=False,
                rebuild_required=False,
                stop_reason=str(proof["reason"]),
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "external_topology",
                "external_stage_completed": False,
                "external_generation_rebuild_required": True,
                "external_generation_rebuild_reason": (
                    "external_conversion_budget_exhausted_before_atomic_recovery"
                ),
                "external_primary_adjustments_used": int(used),
                "external_remaining_budget": max(0, int(limit) - int(used)),
                "external_conversion_budget_exhausted": True,
                "external_stage_completed": False,
            },
            histogram_adjustment_plan=HistogramAdjustmentPlan(
                phase="external_topology",
                groups=[],
                primary_adjustment_count=0,
                physical_conversion_count=0,
                H_before=torch.clamp(H.detach().clone(), min=0).long(),
                H_after=torch.clamp(H.detach().clone(), min=0).long(),
            ),
            runtime={"phase_evaluation": phase_eval},
        ), h_tolerance)
    node_ids = [int(item["node_id"]) for item in operations]
    if int(H.numel()) <= 24 or int(H[24].item()) < len(node_ids):
        proof = {
            "phase": recovery_subphase,
            "reason": "recycled_su24_histogram_underflow",
            "requested_count": len(node_ids),
            "available_su24": int(H[24].item()) if int(H.numel()) > 24 else 0,
            "node_ids": node_ids,
        }
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            phase="external_topology",
            plan_kind="recycled_su24_to_23_invalid",
            active_subphase=recovery_subphase,
            blocked_reason=str(proof["reason"]),
            phase_result=Stage3PhaseResult(
                active_phase="external_topology",
                phase_completed=False,
                rebuild_required=False,
                stop_reason=str(proof["reason"]),
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "external_topology",
                "recycled_su24_to_23_ids": node_ids,
                "external_stage_completed": False,
            },
            runtime={
                "phase_evaluation": phase_eval,
            },
        ), h_tolerance)

    H_before = torch.clamp(H.detach().clone(), min=0).long()
    H_after = H_before.detach().clone()
    H_after[24] -= len(node_ids)
    H_after[23] += len(node_ids)
    op = HistogramConversion(
        from_su=24,
        to_su=23,
        count=len(node_ids),
        node_ids=list(node_ids),
        reason="recycle_pure_resource_ring",
    )
    group = AtomicAdjustmentGroup(
        primary_op=op,
        combined_delta={24: -len(node_ids), 23: len(node_ids)},
        h_before=_h_total(H_before, special_degree_meta),
        h_after_primary=_h_total(H_after, special_degree_meta),
        h_after_final=_h_total(H_after, special_degree_meta),
    )
    histogram = HistogramAdjustmentPlan(
        phase=recovery_subphase,
        groups=[group],
        # Recovery is forced, and every concrete SU24 node consumes one unit
        # of the shared external primary-conversion budget.
        primary_adjustment_count=len(node_ids),
        physical_conversion_count=len(node_ids),
        H_before=H_before,
        H_after=H_after,
        inventory_before={24: len(node_ids)},
        inventory_after={23: len(node_ids)},
        accounting_diagnostics={
            "mode": "pure_resource_ring_recycle",
            "recycled_su24_to_23_ids": list(node_ids),
        },
    )
    transaction = TopologyResourceTransaction(
        phase=recovery_subphase,
        contract_id=-240023,
        target_role="recycled_ring_resource",
        converted_node_ids=list(node_ids),
        conversion_path=[dict(item) for item in operations],
        h_before=_h_total(H_before, special_degree_meta),
        h_after_primary=_h_total(H_after, special_degree_meta),
        h_after=_h_total(H_after, special_degree_meta),
        commit_scope="physical_histogram",
        transaction_origin="recycled_ring_resource",
        committed=True,
    )
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    pool = _round_resource_ids(phase_eval)
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="external_recovery_continuation",
        phase="external_topology",
        plan_kind="external_recovery_continuation",
        active_subphase=recovery_subphase,
        planned_conversions=[
            Stage3AdjustmentOp(
                op="convert",
                from_su=24,
                to_su=23,
                node_id=int(item["node_id"]),
                reason=str(item["reason"]),
            )
            for item in operations
        ],
        histogram_adjustment_plan=histogram,
        phase_result=Stage3PhaseResult(
            active_phase="external_topology",
            phase_completed=False,
            rebuild_required=False,
            transactions=[transaction],
            stop_reason="external_recovery_continuation",
        ),
        diagnostics={
            "stage3_phase": "external_topology",
            "external_stage_completed": False,
            "external_recovery_continuation": True,
            "recycled_su24_to_23_ids": list(node_ids),
            "recycled_resource_conversion_ops": [dict(item) for item in operations],
            "primary_adjustment_count": len(node_ids),
            "external_primary_adjustments_used": int(used),
            "external_remaining_budget": max(0, int(limit) - int(used) - len(node_ids)),
        },
        runtime={
            **_runtime_with_pool(phase_eval, pool),
            "phase_evaluation": phase_eval,
        },
    )
    return _annotate_h_soft_rule(plan, h_tolerance)


def _internal_ledger_invariant_errors(
    *,
    ledger: Any,
    H: torch.Tensor,
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    available = {
        int(key): [int(value) for value in list(values or [])]
        for key, values in dict(
            getattr(ledger, "available_ids_by_su", {}) or {}
        ).items()
    }
    owners = dict(getattr(ledger, "ownership_by_node_id", {}) or {})
    seen_pool: Dict[int, int] = {}
    for su_type, node_ids in available.items():
        for node_id in node_ids:
            if node_id in seen_pool:
                errors.append({
                    "reason": "resource_id_in_multiple_su_buckets",
                    "node_id": int(node_id),
                    "su_types": [int(seen_pool[node_id]), int(su_type)],
                })
            seen_pool[node_id] = int(su_type)
            owner = owners.get(int(node_id))
            if owner is None or int(getattr(owner, "su_type", -1)) != int(su_type):
                errors.append({
                    "reason": "resource_bucket_owner_type_mismatch",
                    "node_id": int(node_id),
                    "bucket_su": int(su_type),
                    "owner_su": (
                        None if owner is None else int(getattr(owner, "su_type", -1))
                    ),
                })

    product_owner: Dict[int, str] = {}
    for product in list(getattr(ledger, "completed_products", []) or []):
        payload = dict(product or {})
        product_id = str(payload.get("product_id", "") or "")
        product_ids = {
            int(value)
            for value in list(payload.get("fixed_source_node_ids", []) or [])
        } | {
            int(value)
            for values in dict(
                payload.get(
                    "claimed_resource_ids_by_su",
                    {},
                )
                or {}
            ).values()
            for value in list(values or [])
        }
        for node_id in product_ids:
            if node_id in seen_pool:
                errors.append({
                    "reason": "completed_product_node_still_available",
                    "node_id": int(node_id),
                    "product_id": product_id,
                })
            if node_id in product_owner and product_owner[node_id] != product_id:
                errors.append({
                    "reason": "completed_products_overlap",
                    "node_id": int(node_id),
                    "product_ids": [product_owner[node_id], product_id],
                })
            product_owner[node_id] = product_id

    expected = torch.zeros_like(torch.clamp(H.detach().cpu(), min=0).long())
    for owner in owners.values():
        su_type = int(getattr(owner, "su_type", -1))
        if 0 <= su_type < int(expected.numel()):
            expected[su_type] += 1
    actual = torch.clamp(H.detach().cpu(), min=0).long()
    if not torch.equal(expected, actual):
        errors.append({
            "reason": "ledger_histogram_mismatch",
            "expected": expected.tolist(),
            "actual": actual.tolist(),
        })
    for conflict in list(getattr(ledger, "conflicts", []) or []):
        errors.append({
            "reason": "ownership_ledger_conflict",
            "conflict": dict(conflict or {}),
        })
    if errors:
        # Keep the invariant failure actionable in long CPU runs.  This is
        # diagnostic only; the caller still rejects the state atomically.
        try:
            import json
            with open("/tmp/graphnmr_internal_ledger_invariant.json", "w") as handle:
                json.dump(errors, handle, sort_keys=True)
        except Exception:
            pass
    return errors


def _plan_internal_topology_stage(
    *,
    phase_eval: Any,
    H: torch.Tensor,
    iteration_ledger: Any,
    initial_transactions: Sequence[TopologyResourceTransaction],
    E_target: Any,
    h_tolerance: float,
    max_primary_adjustments: int,
    max_h_compensation_ops: int,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    residual_flex_preferred_23: int,
    residual_flex_min_23: int,
    residual_side_preferred_23: int,
    residual_side_min_23: int,
    forbidden_primary_conversion_signatures: Optional[Sequence[str]],
) -> Stage3AdjustmentPlan:
    initial_pool = deepcopy(iteration_ledger.available_ids_by_su)
    H_work = H.detach().clone()
    groups: List[Any] = []
    transactions = list(initial_transactions or [])
    residual_products: List[Dict[str, Any]] = []
    residual_attempts: List[Dict[str, Any]] = []
    residual_attempt_count = 0
    cursor = int(su11_rotation_index)
    budget = max(0, int(max_primary_adjustments))
    stop_subphase = "residual_self_closure"
    remaining_budget = budget
    flex_bounds = dict(getattr(phase_eval, "flex_bounds", {}) or {})
    flex_counts = dict(getattr(phase_eval, "flex_counts", {}) or {})
    remaining_flex_capacity = max(
        0,
        int(flex_bounds.get("max", 0) or 0)
        - int(flex_counts.get("final", 0) or 0),
    )
    residual_contract_id = 2_000_000
    assembly_view = dict(getattr(phase_eval, "assembly_view", {}) or {})
    resource_ledger = dict(assembly_view.get("resource_ledger", {}) or {})
    recycled_resource_ids = {
        int(value)
        for values in dict(
            resource_ledger.get("released_resource_ids_by_su", {}) or {}
        ).values()
        for value in list(values or [])
    }
    recycled_resource_ids.update(
        int(value)
        for value in list(
            resource_ledger.get("recycled_su24_to_23_ids", []) or []
        )
    )
    while remaining_budget >= 0:
        remaining_closure = _remaining_closure_ids_by_su(
            iteration_ledger.available_ids_by_su
        )
        remaining_su24 = list(
            iteration_ledger.available_ids_by_su.get(
                SU24_RING_RESOURCE_TYPE,
                [],
            )
            or []
        )
        tolerated = _tolerated_remaining_closure_ids(
            iteration_ledger.available_ids_by_su,
            remaining_flex_goal=0,
            recycled_resource_ids=recycled_resource_ids,
        )
        # A singleton SU24 is not an acceptable terminal residual. Keep the
        # internal loop alive so it can be reclassified to SU12/SU13; the
        # ordinary <=3 residual tolerance applies only after SU24 is removed.
        if (tolerated or not remaining_closure) and not remaining_su24:
            break
        next_product = _plan_next_residual_product(
            phase_eval=phase_eval,
            iteration_ledger=iteration_ledger,
            H=H_work,
            E_target=E_target,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=cursor,
            remaining_flex_chains=0,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            max_conversions=remaining_budget,
            max_h_compensation_ops=max_h_compensation_ops,
            contract_id=residual_contract_id,
            remaining_flex_capacity=remaining_flex_capacity,
            prefix_groups=groups,
            accounting_initial_pool=initial_pool,
            forbidden_primary_conversion_signatures=forbidden_primary_conversion_signatures,
        )
        attempts = list(next_product.get("attempts", []) or [])
        residual_attempt_count += int(
            next_product.get("attempt_count", len(attempts)) or len(attempts)
        )
        residual_attempts = _diagnostic_attempt_samples(
            [*residual_attempts, *attempts]
        )
        mode = str(next_product.get("mode", "") or "")
        if mode == "direct":
            iteration_ledger = next_product["ledger"]
            transactions.append(next_product["transaction"])
            if list(iteration_ledger.completed_products or []):
                residual_products.append(dict(iteration_ledger.completed_products[-1] or {}))
            if str(next_product.get("product_role", "")) == "residual_flex_chain":
                remaining_flex_capacity = max(0, remaining_flex_capacity - 1)
            residual_contract_id += 1
            continue
        if mode != "conversion" or remaining_budget <= 0:
            break
        conversion_plan = next_product["plan"]
        histogram = conversion_plan.histogram_adjustment_plan
        step_count = _physical_conversion_count(
            list(getattr(histogram, "groups", []) or [])
        )
        if (
            histogram is None
            or step_count <= 0
            or step_count > remaining_budget
            or not _internal_conversion_is_allowed(histogram)
        ):
            break
        candidate_ledger = dict(
            getattr(conversion_plan, "runtime", {}) or {}
        ).get("iteration_topology_ledger")
        if candidate_ledger is None:
            break
        iteration_ledger = candidate_ledger
        H_work = histogram.H_after.detach().clone()
        groups.extend(list(histogram.groups or []))
        cursor = int(histogram.su11_rotation_index_after)
        transactions.extend(list(conversion_plan.phase_result.transactions or []))
        if list(iteration_ledger.completed_products or []):
            residual_products.append(dict(iteration_ledger.completed_products[-1] or {}))
        if str(next_product.get("product_role", "")) == "residual_flex_chain":
            remaining_flex_capacity = max(0, remaining_flex_capacity - 1)
        residual_contract_id += 1
        remaining_budget = budget - _physical_conversion_count(groups)

    invariant_errors = _internal_ledger_invariant_errors(
        ledger=iteration_ledger,
        H=H_work,
    )
    preserved_unassembled_su24_ids = sorted(
        int(value)
        for value in list(
            iteration_ledger.available_ids_by_su.get(
                SU24_RING_RESOURCE_TYPE,
                [],
            )
            or []
        )
    )
    diagnostics = {
        "internal_stage_completed": bool(
            not remaining_closure and not preserved_unassembled_su24_ids
        ),
        "internal_stop_subphase": str(stop_subphase),
        "internal_conversion_budget": int(budget),
        "internal_physical_conversion_count": _physical_conversion_count(groups),
        "internal_conversion_budget_remaining": max(
            0, budget - _physical_conversion_count(groups)
        ),
        "internal_budget_exhausted": bool(
            _physical_conversion_count(groups) >= budget and bool(groups)
        ),
        "internal_ring_count": 0,
        "internal_residual_product_count": len(residual_products),
        "internal_ring_attempt_count": 0,
        "internal_ring_attempts": [],
        # Only SU24 owned by completed ring products may survive Stage3.  IDs
        # still present in this available pool are physically unassembled and
        # therefore block the terminal completion state below.
        "su24_remaining_count": len(preserved_unassembled_su24_ids),
        "preserved_unassembled_su24_ids": preserved_unassembled_su24_ids,
        "internal_residual_attempt_count": int(residual_attempt_count),
        "internal_residual_attempts": list(residual_attempts),
        "committed_residual_products": list(residual_products),
        "remaining_ids_by_su": {
            str(key): list(values)
            for key, values in dict(iteration_ledger.available_ids_by_su or {}).items()
        },
        "resource_invariant_errors": list(invariant_errors),
    }
    if invariant_errors:
        proof = {
            "phase": "internal_topology",
            "reason": "internal_resource_ledger_invariant_failed",
            "errors": list(invariant_errors),
        }
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="internal_topology",
            plan_kind="internal_resource_ledger_invariant_failed",
            active_subphase=str(stop_subphase),
            phase_result=Stage3PhaseResult(
                active_phase="internal_topology",
                phase_completed=False,
                rebuild_required=False,
                transactions=transactions,
                stop_reason="phase_resource_infeasible",
                infeasibility_proof=proof,
            ),
            diagnostics=diagnostics,
            runtime={
                **_runtime_with_pool(phase_eval, iteration_ledger.available_ids_by_su),
                "iteration_topology_ledger": iteration_ledger,
            },
        ), h_tolerance)
    if groups:
        result = _finalize_phase_plan(
            phase_eval=phase_eval,
            phase="internal_topology",
            H_before=H,
            H_after=H_work,
            initial_pool=initial_pool,
            final_pool=iteration_ledger.available_ids_by_su,
            groups=groups,
            transactions=transactions,
            rotation_before=su11_rotation_index,
            rotation_after=cursor,
            h_tolerance=h_tolerance,
            stage_diagnostics=diagnostics,
        )
        result.runtime["iteration_topology_ledger"] = iteration_ledger
        result.active_subphase = str(stop_subphase)
        result.phase_result.active_phase = "internal_topology"
        result.phase_result.phase_completed = False
        return result

    if preserved_unassembled_su24_ids:
        diagnostics["internal_stage_completed"] = False
        proof = {
            "phase": "internal_topology",
            "reason": "internal_su24_conversion_budget_exhausted",
            "count": len(preserved_unassembled_su24_ids),
            "node_ids": list(preserved_unassembled_su24_ids),
            "available_counts": {
                str(key): len(list(values or []))
                for key, values in dict(
                    iteration_ledger.available_ids_by_su or {}
                ).items()
            },
        }
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=str(proof["reason"]),
            phase="internal_topology",
            plan_kind="internal_su24_conversion_budget_exhausted",
            active_subphase=str(stop_subphase),
            phase_result=Stage3PhaseResult(
                active_phase="internal_topology",
                phase_completed=False,
                rebuild_required=False,
                transactions=transactions,
                stop_reason="internal_su24_conversion_budget_exhausted",
                infeasibility_proof=proof,
            ),
            diagnostics={
                "stage3_phase": "internal_topology",
                **diagnostics,
            },
            runtime={
                **_runtime_with_pool(
                    phase_eval,
                    iteration_ledger.available_ids_by_su,
                ),
                "iteration_topology_ledger": iteration_ledger,
            },
        ), h_tolerance)

    return _annotate_h_soft_rule(Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_complete",
        phase="complete",
        plan_kind="stage3_state_machine_complete",
        active_subphase="complete",
        phase_result=Stage3PhaseResult(
            active_phase="complete",
            phase_completed=True,
            rebuild_required=False,
            symbolic_topology_count=len(transactions),
            transactions=transactions,
            stop_reason="stage3_complete",
        ),
        diagnostics={
            "stage3_phase": "complete",
            **diagnostics,
        },
        runtime={
            **_runtime_with_pool(phase_eval, iteration_ledger.available_ids_by_su),
            "iteration_topology_ledger": iteration_ledger,
        },
    ), h_tolerance)


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
    progress_phase: Stage3ProgressPhase = Stage3ProgressPhase.FLEX_RECONCILIATION,
    spectrum_candidate_limit: int = 8,
    spectrum_max_primary_adjustments: int = 10,
    spectrum_profile_stats: Optional[Dict[int, Dict[str, Any]]] = None,
    spectrum_hop1_template_options: Optional[Dict[int, Any]] = None,
    rejected_spectrum_histogram_signatures: Optional[Sequence[str]] = None,
    rejected_spectrum_bundle_signatures: Optional[Sequence[str]] = None,
    forbidden_spectrum_inverse_signatures: Optional[Sequence[str]] = None,
    flex_max_primary_adjustments: int = 10,
    external_max_primary_adjustments: int = 10,
    external_primary_adjustments_used: int = 0,
    allow_ring_recovery: bool = False,
    external_assembler: Any = None,
    internal_max_primary_adjustments: int = 30,
    max_h_compensation_ops_per_adjustment: int = 8,
    h_tolerance: float = DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    residual_flex_preferred_23: int = 3,
    residual_flex_min_23: int = 2,
    residual_side_preferred_23: int = 2,
    residual_side_min_23: int = 1,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
    forbidden_resource_conversion_signatures: Optional[Sequence[str]] = None,
) -> Stage3AdjustmentPlan:
    flex_max_primary_adjustments = max(0, int(flex_max_primary_adjustments))
    external_max_primary_adjustments = max(0, int(external_max_primary_adjustments))
    external_primary_adjustments_used = max(0, int(external_primary_adjustments_used))
    internal_max_primary_adjustments = max(0, int(internal_max_primary_adjustments))
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
    external_phases_pending = progress_phase in {
        Stage3ProgressPhase.FLEX_RECONCILIATION,
        Stage3ProgressPhase.EXTERNAL_TOPOLOGY,
        Stage3ProgressPhase.SPECTRUM_REBALANCE,
    }
    restore_external_resource_context = progress_phase is Stage3ProgressPhase.INTERNAL_TOPOLOGY
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
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
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
        return _fixed_flex_capacity_expansion_plan(
            phase_eval=phase_eval,
            H=H,
            E_target=E_target,
            difference_spectrum=difference_spectrum,
            max_primary_adjustments=flex_max_primary_adjustments,
            max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
            forbidden_primary_conversion_signatures=(
                forbidden_resource_conversion_signatures
            ),
        )
    if classification is None:
        classification = classify_topology_units(list(nodes or []))
    if external_phases_pending:
        phase_eval = _evaluate_external_phases(
            phase_eval=phase_eval,
            nodes=nodes,
            H=H,
            classification=classification,
            allow_ring_recovery=bool(allow_ring_recovery),
            external_assembler=external_assembler,
        )
    else:
        # The internal phase consumes the exact external assembly snapshot
        # from this generation. Re-running the assembler here would recreate
        # SU24 rings after ring recovery and lose released-resource
        # ownership. A Layer1/2 rebuild resets progress to FLEX_RECONCILIATION
        # and is the only point at which external assembly runs again.
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
            "external_resource_context_preserved": True,
            "external_closure_ok": True,
        }
    materialization_errors = (
        _external_product_materialization_errors(phase_eval)
        if external_phases_pending
        else []
    )
    if materialization_errors:
        infeasible_contract_samples = _diagnostic_attempt_samples(
            infeasible_contracts
        )
        proof = {
            "phase": EXTERNAL_STAGE,
            "reason": "external_physical_product_validation_failed",
            "materialization_errors": materialization_errors,
        }
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
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
            },
            runtime=_runtime_with_pool(phase_eval, _resource_ids(phase_eval)),
        ), h_tolerance)
    # A recovery-enabled external pass reports its forced SU24->SU23
    # operations in the assembly ledger. The caller applies that operation as
    # part of the current external transaction; it is deliberately not an
    # immediate rebuild trigger because the current ring/side contract must
    # retry against the released pool first.
    if external_phases_pending:
        recycled_plan = _plan_recycled_su24_to_23(
            phase_eval=phase_eval,
            nodes=nodes,
            H=H,
            special_degree_meta=special_degree_meta,
            h_tolerance=h_tolerance,
            external_primary_adjustments_used=external_primary_adjustments_used,
            external_max_primary_adjustments=external_max_primary_adjustments,
        )
        if recycled_plan is not None:
            return recycled_plan
        # A generation that changed any concrete external SU must not enter
        # spectrum rebalance or internal topology on the stale Layer1/2 graph.
        # Assembly continuation is allowed until the last external contract is
        # physically closed; this explicit boundary then hands the accepted H
        # to Block C for one port-balance pass and a fresh Layer1/2 rebuild.
        if (
            bool(getattr(phase_eval, "external_closure_ok", False))
            and int(external_primary_adjustments_used) > 0
        ):
            boundary_H = torch.clamp(H.detach().clone(), min=0).long()
            budget_exhausted = bool(
                int(external_primary_adjustments_used)
                >= int(external_max_primary_adjustments)
            )
            boundary_diagnostics = {
                "stage3_phase": EXTERNAL_STAGE,
                "external_stage_completed": True,
                "external_generation_rebuild_required": True,
                "external_generation_rebuild_reason": (
                    "external_completed_after_primary_conversion"
                ),
                "external_conversion_budget_exhausted": bool(budget_exhausted),
                "external_primary_adjustments_used": int(
                    external_primary_adjustments_used
                ),
                "external_remaining_budget": max(
                    0,
                    int(external_max_primary_adjustments)
                    - int(external_primary_adjustments_used),
                ),
                "physical_materialization_required_after_rebuild": True,
            }
            return _annotate_h_soft_rule(Stage3AdjustmentPlan(
                ok=True,
            reason="stage3_complete",
                phase=EXTERNAL_STAGE,
                plan_kind="external_generation_complete_rebuild_boundary",
                active_subphase=EXTERNAL_STAGE,
                phase_result=Stage3PhaseResult(
                    active_phase=EXTERNAL_STAGE,
                    phase_completed=True,
                    rebuild_required=True,
                    primary_adjustment_count=0,
                    stop_reason="external_generation_rebuild_required",
                ),
                histogram_adjustment_plan=HistogramAdjustmentPlan(
                    phase=EXTERNAL_STAGE,
                    groups=[],
                    primary_adjustment_count=0,
                    physical_conversion_count=0,
                    H_before=boundary_H,
                    H_after=boundary_H.detach().clone(),
                ),
                diagnostics=boundary_diagnostics,
                runtime={
                    **_runtime_with_pool(
                        phase_eval,
                        _round_resource_ids(phase_eval),
                    ),
                    "phase_evaluation": phase_eval,
                },
            ), h_tolerance)
        # The shared external budget is generation-local.  A previous
        # continuation may have consumed the final primary-conversion slot
        # while leaving the current contract pending; on the next planner
        # entry there is no histogram batch to return, but the generation is
        # still stale and must cross the same port-balance/fresh-rebuild
        # boundary.  Emit an explicit boundary plan instead of misclassifying
        # this state as an ordinary resource infeasibility.
        if (
            int(external_primary_adjustments_used) > 0
            and int(external_primary_adjustments_used)
            >= int(external_max_primary_adjustments)
        ):
            boundary_H = torch.clamp(H.detach().clone(), min=0).long()
            boundary_phase_eval = phase_eval
            boundary_diagnostics = {
                "stage3_phase": EXTERNAL_STAGE,
                "external_stage_completed": False,
                "external_conversion_budget_exhausted": True,
                "external_primary_adjustments_used": int(
                    external_primary_adjustments_used
                ),
                "external_remaining_budget": 0,
                "external_rebuild_reason": (
                    "resource_conversion_requires_fresh_assembly"
                ),
                "physical_materialization_required_after_rebuild": True,
            }
            return _annotate_h_soft_rule(Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase=EXTERNAL_STAGE,
                plan_kind="external_conversion_budget_exhausted_boundary",
                active_subphase=str(
                    getattr(boundary_phase_eval, "active_external_subphase", "")
                    or EXTERNAL_STAGE
                ),
                pending_contracts=[
                    dict(item or {})
                    for item in list(
                        getattr(boundary_phase_eval, "canonical_contracts", [])
                        or []
                    )
                    if dict(item or {})
                ],
                phase_result=Stage3PhaseResult(
                    active_phase=EXTERNAL_STAGE,
                    phase_completed=False,
                    rebuild_required=True,
                    primary_adjustment_count=0,
                    stop_reason="external_conversion_budget_exhausted",
                    infeasibility_proof={
                        "reason": "external_conversion_budget_exhausted",
                        "external_primary_adjustments_used": int(
                            external_primary_adjustments_used
                        ),
                        "external_max_primary_adjustments": int(
                            external_max_primary_adjustments
                        ),
                    },
                ),
                histogram_adjustment_plan=HistogramAdjustmentPlan(
                    phase=EXTERNAL_STAGE,
                    groups=[],
                    primary_adjustment_count=0,
                    physical_conversion_count=0,
                    H_before=boundary_H,
                    H_after=boundary_H.detach().clone(),
                ),
                diagnostics=boundary_diagnostics,
                runtime={
                    **_runtime_with_pool(
                        boundary_phase_eval,
                        _round_resource_ids(boundary_phase_eval),
                    ),
                    "phase_evaluation": boundary_phase_eval,
                },
            ), h_tolerance)
    # The round context is authoritative after every fresh classification.
    # Reading the assembly ledger here can reintroduce nodes already owned by
    # external fragments and later let the ring allocator capture them.
    pool = _round_resource_ids(phase_eval)
    H_work = H.detach().clone()
    all_contracts = _external_contracts(phase_eval, nodes)
    (
        active_contracts,
        dormant_contracts,
        active_external_phase,
        active_external_role,
    ) = _partition_active_external_contracts(phase_eval, all_contracts)
    # Only the current assembler phase may request conversions.  Later
    # contracts remain diagnostic/dormant until continuation has completed the
    # current contract and advanced the live assembler to their phase.
    contracts = list(active_contracts or [])
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
    # ``iteration_ledger`` is reconstructed from the live assembler snapshot.
    # Feeding an output product cache back here would create a second, stale
    # ownership authority and could duplicate products after continuation.
    iteration_ledger, _ring_pool_restore_audit = (
        _restore_uncommitted_ring_pool_nodes(
            iteration_ledger,
            nodes,
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
        return _annotate_h_soft_rule(Stage3AdjustmentPlan(
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
                list(contract.claimed_source_node_ids),
            )
            if source_status == "superseded":
                external_superseded += 1
                continue
            blocker_reason = str(raw.get("blocker_kind", raw.get("blocked_reason", "")) or "")
            if blocker_reason and not contract.required_counts:
                failure_domain = str(
                    raw.get("failure_domain", "")
                    or "topology_materialization"
                )
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                    "required_counts": {},
                    "proof": {
                        "contract_id": int(contract.contract_id),
                        "reason": blocker_reason,
                        "failure_domain": str(failure_domain),
                        "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                        **{
                            key: raw[key]
                            for key in (
                                "assembler_reason",
                                "branch_anchor_index",
                                "contract_base_path_labels",
                                "contract_base_path_node_ids",
                                "unexpected_fixed_edges",
                                "offending_fixed_edges",
                                "terminal_node_ids",
                                "missing_node_ids",
                                "mainline_node_ids",
                                "structural_boundary",
                            )
                            if key in raw
                        },
                        "assembler_diagnostics": dict(
                            raw.get("diagnostics", {}) or {}
                        ),
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
                    "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": dict(stale_proof),
                })
                role_blocked += 1
                continue
            # Raw SU24 cap/terminal IDs were removed when the iteration
            # ledger was built, so this pool already contains only resources
            # that the current external contract may claim.
            claimable_counts = _contract_claimable_counts(raw)
            reported_missing_counts = _contract_reported_missing_counts(raw)
            missing_counts = (
                dict(reported_missing_counts)
                if reported_missing_counts
                else _deficit_from_counts(claimable_counts, contract.required_counts)
                if claimable_counts
                else _deficit(pool, contract.required_counts)
            )
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
                declared_failure_domain = str(
                    raw.get("failure_domain", "")
                    or assembler_diagnostics.get("failure_domain", "")
                    or ""
                )
                resource_reason = any(
                    token in assembler_reason.lower()
                    for token in ("shortage", "resource", "pool", "missing")
                )
                if declared_failure_domain and declared_failure_domain != "topology_materialization":
                    infeasible_contracts.append({
                        "phase": str(phase),
                        "contract_id": int(contract.contract_id),
                        "target_role": str(contract.target_role),
                        "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                        "required_counts": dict(contract.required_counts),
                        "proof": {
                            "reason": str(assembler_reason),
                            "failure_domain": str(declared_failure_domain),
                            "required_counts": dict(contract.required_counts),
                            "available_counts": dict(claimable_counts),
                            "missing_counts": {},
                            "assembler_diagnostics": assembler_diagnostics,
                        },
                    })
                    role_blocked += 1
                    continue
                if resource_reason:
                    infeasible_contracts.append({
                        "phase": str(phase),
                        "contract_id": int(contract.contract_id),
                        "target_role": str(contract.target_role),
                        "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                        "required_counts": dict(contract.required_counts),
                        "proof": {
                            "reason": "resource_contract_claimability_inconsistent",
                            "failure_domain": "resource_accounting",
                            "assembler_reason": str(assembler_reason),
                            "required_counts": dict(contract.required_counts),
                            "available_counts": dict(claimable_counts),
                            "missing_counts": {},
                            "assembler_diagnostics": assembler_diagnostics,
                        },
                    })
                    role_blocked += 1
                    continue
                materialization_proof = {
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "reason": "external_topology_materialization_invalid",
                    "failure_domain": "topology_materialization",
                    "assembler_reason": str(assembler_reason),
                    "source_fragment_ids": list(contract.source_fragment_ids),
                    "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "available_counts": (
                        dict(claimable_counts)
                        if claimable_counts
                        else {
                            int(key): max(0, len(list(values or [])))
                            for key, values in pool.items()
                        }
                    ),
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
                    "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": materialization_proof,
                })
                role_blocked += 1
                continue
            direct_proof = {
                "reason": "external_resource_deficit",
                "missing_counts": dict(missing_counts),
            }
            remaining_limit = max(
                0,
                int(external_max_primary_adjustments)
                - int(external_primary_adjustments_used)
                - _external_primary_node_count(phase_groups),
            )
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
            protected_for_conversion = {
                int(su_type): min(
                    len(list(pool.get(int(su_type), []) or [])),
                    int(protected_aromatic_donors.get(int(su_type), 0)),
                )
                for su_type in set(protected_aromatic_donors)
            }
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
                protected_resource_counts=protected_for_conversion,
                accounting_initial_pool=initial_pool,
                forbidden_primary_conversion_signatures=forbidden_resource_conversion_signatures,
                # External topology conversion must never consume SU24,
                # including SU24 released by a previously dismantled ring.
                forbidden_conversion_su_types=[24],
                reserve_contract_after_conversion=True,
            )
            if not candidate_plan.ok:
                candidate_histogram = candidate_plan.histogram_adjustment_plan
                partial_groups = list(
                    getattr(candidate_histogram, "groups", []) or []
                )
                partial_count = _external_primary_node_count(partial_groups)
                # A legal prefix is still useful when the current contract
                # needs more resources than this planner pass could provide.
                # Commit that concrete prefix and continue the same live
                # assembler; do not roll it back merely because the contract
                # is not complete yet.  The next pass may then use another
                # ordinary conversion or open the ring-recovery gate after a
                # genuine capacity proof.
                if partial_groups and int(partial_count) > 0:
                    candidate_ledger = dict(
                        getattr(candidate_plan, "runtime", {}) or {}
                    ).get("iteration_topology_ledger")
                    if candidate_ledger is not None:
                        iteration_ledger = candidate_ledger
                        pool = deepcopy(iteration_ledger.available_ids_by_su)
                    else:
                        pool = _bind_real_node_ids(candidate_histogram, pool)
                    H_work = candidate_histogram.H_after.detach().clone()
                    phase_groups.extend(partial_groups)
                    rotation_cursor = int(
                        candidate_histogram.su11_rotation_index_after
                    )
                    partial_transactions = list(
                        candidate_plan.phase_result.transactions or []
                    )
                    for transaction in partial_transactions:
                        transaction.committed = True
                        transaction.commit_scope = "physical_histogram"
                        transaction.rollback_reason = ""
                        transaction.claimed_resource_ids = {}
                        transaction.product_node_ids = []
                        transaction.slot_assignments = []
                    direct_transactions.extend(partial_transactions)
                    budget_exhausted = bool(
                        int(external_primary_adjustments_used)
                        + int(_external_primary_node_count(phase_groups))
                        >= int(external_max_primary_adjustments)
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
                        stage_diagnostics={
                            "external_stage_completed": False,
                            "external_rebuild_reason": (
                                "external_conversion_budget_exhausted"
                                if budget_exhausted
                                else "external_conversion_continuation_required"
                            ),
                            "external_contracts_seen": int(external_contracts_seen),
                            "external_directly_completed": int(external_directly_completed),
                            "external_reserved_after_conversion": int(
                                external_completed_after_conversion
                            ),
                            "external_completed_after_conversion": 0,
                            "external_blocked": 0,
                            "external_remaining_pending": max(
                                1,
                                len(contracts) - int(external_contracts_seen) + 1,
                            ),
                            "external_primary_count": _external_primary_node_count(phase_groups),
                            "external_primary_adjustments_used": int(
                                external_primary_adjustments_used
                                + _external_primary_node_count(phase_groups)
                            ),
                            "external_remaining_budget": max(
                                0,
                                int(external_max_primary_adjustments)
                                - int(external_primary_adjustments_used)
                                - _external_primary_node_count(phase_groups),
                            ),
                            "external_conversion_budget_exhausted": bool(
                                budget_exhausted
                            ),
                            "physical_materialization_required_after_rebuild": bool(
                                budget_exhausted
                            ),
                            "partial_contract_conversion_committed": True,
                            "external_continuation_required": not bool(
                                budget_exhausted
                            ),
                        },
                    )
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
                            "external_primary_count": _external_primary_node_count(phase_groups),
                            "external_primary_adjustments_used": int(
                                external_primary_adjustments_used
                                + _external_primary_node_count(phase_groups)
                            ),
                            "external_remaining_budget": max(
                                0,
                                int(external_max_primary_adjustments)
                                - int(external_primary_adjustments_used)
                                - _external_primary_node_count(phase_groups),
                            ),
                            "external_conversion_budget_exhausted": bool(
                                budget_exhausted
                            ),
                            "physical_materialization_required_after_rebuild": True,
                        },
                    )
                failed_transactions = list(candidate_plan.phase_result.transactions or [])
                direct_transactions.extend(failed_transactions)
                candidate_proof = dict(
                    candidate_plan.phase_result.infeasibility_proof
                    or direct_proof
                    or {}
                )
                candidate_failure_domain = str(
                    candidate_proof.get("failure_domain", "") or ""
                )
                if not candidate_failure_domain:
                    candidate_failure_domain = "resource_capacity"
                    candidate_proof["failure_domain"] = candidate_failure_domain
                candidate_proof["conversion_capacity_proof"] = {
                    "search_exhausted": True,
                    "ordinary_conversion_feasible": False,
                    "missing_counts": {
                        str(key): int(value)
                        for key, value in dict(missing_counts or {}).items()
                        if int(value) > 0
                    },
                    "remaining_primary_budget": int(remaining_limit),
                    "allowed_donor_su_types": [11, 12, 13, 22, 23],
                    "failure_domain": str(candidate_failure_domain),
                }
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "claimed_source_node_ids": list(contract.claimed_source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": candidate_proof,
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
            planned_count = _external_primary_node_count(phase_groups)
            budget_exhausted = bool(
                int(external_primary_adjustments_used) + int(planned_count)
                >= int(external_max_primary_adjustments)
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
                stage_diagnostics={
                    "external_stage_completed": False,
                    "external_rebuild_reason": (
                        "external_conversion_budget_exhausted"
                        if budget_exhausted
                        else "external_conversion_continuation_required"
                    ),
                    "external_contracts_seen": int(external_contracts_seen),
                    "external_directly_completed": int(external_directly_completed),
                    "external_reserved_after_conversion": 1,
                    "external_completed_after_conversion": 0,
                    "external_blocked": 0,
                    "external_remaining_pending": max(1, len(all_contracts)),
                    "external_primary_count": int(planned_count),
                    "external_primary_adjustments_used": int(
                        external_primary_adjustments_used + planned_count
                    ),
                    "external_remaining_budget": max(
                        0,
                        int(external_max_primary_adjustments)
                        - int(external_primary_adjustments_used)
                        - int(planned_count),
                    ),
                    "external_conversion_budget_exhausted": bool(
                        budget_exhausted
                    ),
                    "external_continuation_required": not bool(budget_exhausted),
                    "physical_materialization_required_after_rebuild": bool(
                        budget_exhausted
                    ),
                },
            )
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
        "external_primary_count": _external_primary_node_count(phase_groups),
        "external_remaining_budget": max(
            0,
            int(external_max_primary_adjustments)
            - int(external_primary_adjustments_used)
            - _external_primary_node_count(phase_groups),
        ),
        "external_primary_adjustments_used": int(
            external_primary_adjustments_used
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
        progress_phase in {
            Stage3ProgressPhase.FLEX_RECONCILIATION,
            Stage3ProgressPhase.EXTERNAL_TOPOLOGY,
            Stage3ProgressPhase.SPECTRUM_REBALANCE,
        }
        and bool(enable_post_external_spectrum)
    ):
        spectrum_candidates, spectrum_audit = generate_spectrum_candidates(
            H=H_work,
            nodes=nodes,
            difference_spectrum=difference_spectrum,
            profile_stats=spectrum_profile_stats,
            hop1_template_options=spectrum_hop1_template_options,
            spectrum_node_ids_by_su=_spectrum_trial_node_ids(nodes, pool),
            aromatic_donor_ids_by_su={
                int(su_type): list(pool.get(int(su_type), []) or [])
                for su_type in (12, 13)
            },
            candidate_limit=int(spectrum_candidate_limit),
            max_primary_adjustments=int(spectrum_max_primary_adjustments),
            max_h_compensation_ops=0,
            rejected_histogram_signatures=rejected_spectrum_histogram_signatures,
            rejected_bundle_signatures=rejected_spectrum_bundle_signatures,
            forbidden_inverse_signatures=forbidden_spectrum_inverse_signatures,
            E_target=E_target,
            special_degree_meta=special_degree_meta,
            flex_audit=dict(getattr(phase_eval, "flex_audit", {}) or {}),
        )
        spectrum_audit = dict(spectrum_audit or {})
        spectrum_audit["forbidden_inverse_signature_count"] = len(
            list(forbidden_spectrum_inverse_signatures or [])
        )
        spectrum_audit["physical_su24_authorization_ids"] = (
            _all_physical_su24_ids(nodes)
        )
        spectrum_audit["stage3_h_soft_tolerance"] = float(h_tolerance)
        if spectrum_candidates:
            phase_result = Stage3PhaseResult(
                active_phase="remaining_pool_spectrum_rebalance",
                phase_completed=False,
                rebuild_required=False,
                stop_reason="stage3_spectrum_candidates_ready",
            )
            return _annotate_h_soft_rule(Stage3AdjustmentPlan(
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
                    "spectrum_stage_evaluated": True,
                    "external_products_are_round_local": True,
                    **dict(external_diagnostics),
                },
                runtime={
                    **_runtime_with_pool(phase_eval, pool),
                    "iteration_topology_ledger": iteration_ledger,
                },
            ), h_tolerance)
    completed_phases.append("remaining_pool_spectrum_rebalance")
    internal_plan = _plan_internal_topology_stage(
        phase_eval=phase_eval,
        H=H_work,
        iteration_ledger=iteration_ledger,
        initial_transactions=direct_transactions,
        E_target=E_target,
        h_tolerance=h_tolerance,
        max_primary_adjustments=internal_max_primary_adjustments,
        max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=rotation_cursor,
        residual_flex_preferred_23=residual_flex_preferred_23,
        residual_flex_min_23=residual_flex_min_23,
        residual_side_preferred_23=residual_side_preferred_23,
        residual_side_min_23=residual_side_min_23,
        forbidden_primary_conversion_signatures=(
            forbidden_resource_conversion_signatures
        ),
    )
    internal_plan.diagnostics = {
        **dict(getattr(internal_plan, "diagnostics", {}) or {}),
        "spectrum_stage_evaluated": bool(enable_post_external_spectrum),
    }
    return internal_plan
