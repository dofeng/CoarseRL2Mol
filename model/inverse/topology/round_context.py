from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .stage3_models import (
    STAGE3_OWNERSHIP_STATES,
    Stage3NodeOwnership,
    Stage3OwnershipState,
    Stage3RoundContext,
    Stage3TopologyProduct,
)


CLOSURE_RESOURCE_TYPES = (11, 22, 23, 24)
AROMATIC_DONOR_TYPES = (10, 12, 13)


def _ids(values: Iterable[Any]) -> Set[int]:
    result: Set[int] = set()
    for value in list(values or []):
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _audit_node_ids(rows: Iterable[Dict[str, Any]]) -> Set[int]:
    result: Set[int] = set()
    for raw in list(rows or []):
        row = dict(raw or {})
        for key in ("node_id", "global_id"):
            value = row.get(key)
            if value is None:
                continue
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                pass
    return result


def _pool_ids(pool: Dict[str, Any], key: str, su_types: Sequence[int]) -> Dict[int, List[int]]:
    raw = dict(dict(pool or {}).get(str(key), {}) or {})
    return {
        int(su_type): sorted(_ids(raw.get(str(su_type), raw.get(int(su_type), [])) or []))
        for su_type in su_types
    }


def _fixed_fragment_ids(fragment: Dict[str, Any]) -> Tuple[Set[int], Set[int]]:
    evidence = dict(fragment.get("evidence", {}) or {})
    source_ids = _ids(fragment.get("source_node_ids", []) or [])
    claimed_ids = _ids(fragment.get("claimed_node_ids", []) or [])
    context_ids = _ids(fragment.get("context_node_ids", []) or [])
    fixed_ids = _ids(evidence.get("fixed_hop1_ids", []) or [])
    fixed_ids.update(_ids(evidence.get("fixed_source_node_ids", []) or []))
    fixed_ids.update(_ids(evidence.get("center_node_ids", []) or []))
    # atomic_group_ids names groups; ownership follows their explicit members.
    fixed_ids.update(_ids(evidence.get("atomic_group_node_ids", []) or []))
    for raw_group in list(evidence.get("atomic_groups", []) or []):
        fixed_ids.update(_ids(dict(raw_group or {}).get("node_ids", []) or []))
    matched_neighbors = [
        dict(item or {})
        for item in list(evidence.get("matched_neighbors", []) or [])
    ]
    nonfixed_neighbor_ids = {
        int(item.get("node_id"))
        for item in matched_neighbors
        if item.get("node_id") is not None and not bool(item.get("is_fixed", False))
    }
    fixed_ids.update(claimed_ids - nonfixed_neighbor_ids)
    fixed_ids.update(
        int(item.get("node_id"))
        for item in matched_neighbors
        if item.get("node_id") is not None and bool(item.get("is_fixed", False))
    )

    # A fragment center is always owned. Neighbor IDs are owned only when the
    # classifier explicitly claimed them or marked them fixed/atomic.
    fragment_id = fragment.get("fragment_id")
    if fragment_id is not None:
        try:
            fixed_ids.add(int(fragment_id))
        except (TypeError, ValueError):
            pass
    fixed_ids.intersection_update(source_ids | claimed_ids | fixed_ids)
    context_ids.update(nonfixed_neighbor_ids)
    context_ids.difference_update(fixed_ids)
    return fixed_ids, context_ids


def build_stage3_round_context(
    *,
    nodes: Sequence[Any],
    classification: Any,
    flex_audit: Optional[Dict[str, Any]] = None,
    generation_id: int = 0,
) -> Stage3RoundContext:
    classification_dict = (
        classification.to_dict()
        if hasattr(classification, "to_dict")
        else deepcopy(dict(classification or {}))
    )


    node_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
    }
    pool = dict(classification_dict.get("resource_pool", {}) or {})
    closure_pool = _pool_ids(pool, "remaining_ids_by_su", CLOSURE_RESOURCE_TYPES)
    donor_pool = _pool_ids(pool, "su11_donor_ids_by_su", AROMATIC_DONOR_TYPES)

    fragment_owner_by_id: Dict[int, int] = {}
    fixed_ids: Set[int] = set()
    context_ids: Set[int] = set()
    conflicts: List[Dict[str, Any]] = []
    for raw_fragment in list(classification_dict.get("fragments", []) or []):
        fragment = dict(raw_fragment or {})
        fragment_id = int(fragment.get("fragment_id", -1) or -1)
        fragment_fixed, fragment_context = _fixed_fragment_ids(fragment)
        context_ids.update(fragment_context)
        for node_id in sorted(fragment_fixed):
            existing = fragment_owner_by_id.get(int(node_id))
            if existing is not None and int(existing) != int(fragment_id):
                conflicts.append({
                    "node_id": int(node_id),
                    "reason": "duplicate_fixed_topology_owner",
                    "owner_fragment_ids": sorted({int(existing), int(fragment_id)}),
                })
                continue
            fragment_owner_by_id[int(node_id)] = int(fragment_id)
            fixed_ids.add(int(node_id))

    conflict_ids = _audit_node_ids(classification_dict.get("unclassified_nodes", []) or [])
    conflict_ids.update(_audit_node_ids(classification_dict.get("classification_rejects", []) or []))
    conflict_ids.update(_audit_node_ids(classification_dict.get("claim_conflicts", []) or []))
    conflict_ids.update(int(item["node_id"]) for item in conflicts)
    support_ids = _audit_node_ids(classification_dict.get("support_units_only", []) or [])
    support_ids.update(context_ids)

    closure_ids = {node_id for values in closure_pool.values() for node_id in values}
    donor_ids = {node_id for values in donor_pool.values() for node_id in values}
    closure_ids.difference_update(fixed_ids | conflict_ids)
    donor_ids.difference_update(fixed_ids | conflict_ids)
    support_ids.difference_update(fixed_ids | closure_ids | donor_ids | conflict_ids)

    normalized_closure: Dict[int, List[int]] = defaultdict(list)
    normalized_donor: Dict[int, List[int]] = defaultdict(list)
    ownership: Dict[int, Stage3NodeOwnership] = {}
    for node_id, node in sorted(node_by_id.items()):
        su_type = int(getattr(node, "su_type", -1))
        if int(node_id) in conflict_ids:
            state = Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
            reason = "classification_conflict_or_reject"
        elif int(node_id) in fixed_ids:
            state = Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value
            reason = "classifier_fixed_fragment_claim"
        elif int(node_id) in closure_ids:
            state = Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value
            normalized_closure[int(su_type)].append(int(node_id))
            reason = "remaining_closure_pool"
        elif int(node_id) in donor_ids:
            state = Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value
            normalized_donor[int(su_type)].append(int(node_id))
            reason = "remaining_aromatic_donor_pool"
        elif int(node_id) in support_ids:
            state = Stage3OwnershipState.SUPPORT_CONTEXT.value
            reason = "classification_support_context"
        else:
            state = Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
            reason = "node_not_covered_by_classification_partition"
            conflict_ids.add(int(node_id))
        ownership[int(node_id)] = Stage3NodeOwnership(
            node_id=int(node_id),
            su_type=int(su_type),
            state=str(state),
            owner_fragment_id=fragment_owner_by_id.get(int(node_id)),
            role=str(state),
            reason=str(reason),
        )

    for su_type in CLOSURE_RESOURCE_TYPES:
        normalized_closure.setdefault(int(su_type), [])
    for su_type in AROMATIC_DONOR_TYPES:
        normalized_donor.setdefault(int(su_type), [])
    return Stage3RoundContext(
        generation_id=int(generation_id),
        classification=classification_dict,
        ownership_by_node_id=ownership,
        closure_pool_ids_by_su={
            int(key): sorted(values) for key, values in normalized_closure.items()
        },
        aromatic_donor_ids_by_su={
            int(key): sorted(values) for key, values in normalized_donor.items()
        },
        flex_audit=deepcopy(dict(flex_audit or {})),
        conflicts=list(conflicts),
    )


def normalize_stage3_product(
    raw: Any,
    *,
    generation_id: int,
    index: int = 0,
) -> Stage3TopologyProduct:
    if isinstance(raw, Stage3TopologyProduct):
        product = deepcopy(raw)
        product.generation_id = int(generation_id)
        return product
    data = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
    claimed = {
        str(key): sorted(_ids(values or []))
        for key, values in dict(
            data.get("claimed_resource_ids_by_su", data.get("claimed_resource_ids", {})) or {}
        ).items()
        if list(values or [])
    }
    claimed_ids = {
        int(node_id) for values in claimed.values() for node_id in list(values or [])
    }
    source_ids = _ids(
        data.get("fixed_source_node_ids", [])
        or data.get("source_node_ids", [])
        or []
    )
    slots = [dict(item or {}) for item in list(data.get("slot_assignments", []) or [])]
    mainline_slots = [dict(item or {}) for item in list(data.get("mainline_slots", []) or [])]
    branch_slots = [dict(item or {}) for item in list(data.get("branch_slots", []) or [])]
    for slot in slots:
        descriptor = " ".join((
            str(slot.get("slot_id", "")),
            str(slot.get("kind", "")),
            str(slot.get("anchor_role", "")),
        )).lower()
        if "branch" in descriptor or str(slot.get("kind", "")) in {"H", "H1"}:
            branch_slots.append(slot)
        else:
            mainline_slots.append(slot)
    product_type = str(
        data.get("product_type", data.get("product_kind", data.get("target_role", ""))) or ""
    )
    product_id = str(
        data.get("product_id", "")
        or f"g{int(generation_id)}:{product_type or 'topology'}:{int(index)}"
    )
    node_id_notation = str(
        data.get("node_id_notation", "")
        or data.get("actual_topology_gid", "")
        or data.get("gid", "")
    )
    if not node_id_notation:
        ordered_ids: List[int] = []
        for slot in mainline_slots + branch_slots:
            ordered_ids.extend(
                int(value)
                for value in list(dict(slot or {}).get("existing_node_ids", []) or [])
                + list(dict(slot or {}).get("assigned_resource_ids", []) or [])
            )
        ordered_ids.extend(sorted(source_ids | claimed_ids))
        node_id_notation = "-".join(
            str(value) for value in dict.fromkeys(ordered_ids)
        )
    else:
        owned_ids = source_ids | claimed_ids
        node_id_notation = re.sub(
            r"\d+",
            lambda match: (
                match.group(0)
                if int(match.group(0)) in owned_ids
                else "OPEN"
            ),
            node_id_notation,
        )
    return Stage3TopologyProduct(
        product_id=product_id,
        generation_id=int(generation_id),
        product_type=product_type,
        source_fragment_ids=[
            int(value) for value in list(data.get("source_fragment_ids", []) or [])
        ],
        fixed_source_node_ids=sorted(source_ids - claimed_ids),
        claimed_resource_ids_by_su=claimed,
        mainline_slots=mainline_slots,
        branch_slots=branch_slots,
        topology_notation=str(
            data.get("topology_notation", "")
            or data.get("actual_topology_su", "")
            or data.get("actual_su", "")
        ),
        node_id_notation=str(node_id_notation),
        closed=bool(data.get("closed", True)),
        reason=str(data.get("reason", "") or ""),
    )


def claim_round_product(
    context: Stage3RoundContext,
    product: Stage3TopologyProduct,
) -> Tuple[Optional[Stage3RoundContext], Dict[str, Any]]:
    working = deepcopy(context)
    fixed_ids = _ids(product.fixed_source_node_ids)
    resource_ids = {
        int(node_id)
        for values in dict(product.claimed_resource_ids_by_su or {}).values()
        for node_id in list(values or [])
    }
    all_ids = fixed_ids | resource_ids
    unavailable: List[Dict[str, Any]] = []
    for node_id in sorted(all_ids):
        owner = working.ownership_by_node_id.get(int(node_id))
        allowed = (
            {Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value}
            if int(node_id) in fixed_ids
            else {
                Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value,
                Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value,
            }
        )
        if owner is None or str(owner.state) not in allowed:
            unavailable.append({
                "node_id": int(node_id),
                "state": None if owner is None else str(owner.state),
                "allowed_states": sorted(allowed),
            })
    if unavailable:
        return None, {"reason": "product_ownership_unavailable", "nodes": unavailable}

    for node_id in sorted(fixed_ids):
        owner = working.ownership_by_node_id[int(node_id)]
        owner.state = Stage3OwnershipState.COMPLETED_TOPOLOGY.value
        owner.owner_product_id = str(product.product_id)
        owner.role = "fixed_source"
    for su_key, values in dict(product.claimed_resource_ids_by_su or {}).items():
        su_type = int(su_key)
        for node_id in list(values or []):
            owner = working.ownership_by_node_id[int(node_id)]
            owner.state = Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
            owner.owner_product_id = str(product.product_id)
            owner.role = "symbolic_closure_resource"
            if int(node_id) in working.closure_pool_ids_by_su.get(int(su_type), []):
                working.closure_pool_ids_by_su[int(su_type)].remove(int(node_id))
            if int(node_id) in working.aromatic_donor_ids_by_su.get(int(su_type), []):
                working.aromatic_donor_ids_by_su[int(su_type)].remove(int(node_id))
    working.products.append(deepcopy(product))
    return working, {}


def mark_round_pending_contracts(
    context: Stage3RoundContext,
    contracts: Sequence[Dict[str, Any]],
) -> Stage3RoundContext:
    working = deepcopy(context)
    for raw in list(contracts or []):
        contract = dict(raw or {})
        if not contract:
            continue
        contract_id = int(contract.get("contract_id", contract.get("unit_id", -1)) or -1)
        fragment_ids = [
            int(value) for value in list(contract.get("source_fragment_ids", []) or [])
        ]
        source_ids = _ids(
            contract.get("claimed_source_node_ids", [])
            or contract.get("core_node_ids", [])
            or contract.get("source_node_ids", [])
            or []
        )
        invalid: List[Dict[str, Any]] = []
        for node_id in sorted(source_ids):
            owner = working.ownership_by_node_id.get(int(node_id))
            if owner is None or str(owner.state) not in {
                Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value,
                Stage3OwnershipState.PENDING_TOPOLOGY.value,
            }:
                invalid.append({
                    "node_id": int(node_id),
                    "state": None if owner is None else str(owner.state),
                })
        if invalid:
            working.conflicts.append({
                "reason": "pending_contract_ownership_unavailable",
                "contract_id": int(contract_id),
                "nodes": invalid,
            })
            continue
        for node_id in sorted(source_ids):
            owner = working.ownership_by_node_id[int(node_id)]
            owner.state = Stage3OwnershipState.PENDING_TOPOLOGY.value
            owner.owner_fragment_id = min(fragment_ids) if fragment_ids else owner.owner_fragment_id
            owner.owner_contract_id = int(contract_id)
            owner.role = "fixed_source"
            owner.reason = "pending_symbolic_topology_contract"
        normalized = deepcopy(contract)
        normalized["contract_id"] = int(contract_id)
        normalized["claimed_source_node_ids"] = sorted(source_ids)
        working.pending_contracts.append(normalized)
    return working


def round_context_invariants(context: Stage3RoundContext) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    valid_states = set(STAGE3_OWNERSHIP_STATES)
    closure_ids = {
        int(node_id)
        for values in dict(context.closure_pool_ids_by_su or {}).values()
        for node_id in list(values or [])
    }
    donor_ids = {
        int(node_id)
        for values in dict(context.aromatic_donor_ids_by_su or {}).values()
        for node_id in list(values or [])
    }
    overlap = closure_ids & donor_ids
    if overlap:
        errors.append({"reason": "closure_donor_pool_overlap", "node_ids": sorted(overlap)})
    for node_id, owner in dict(context.ownership_by_node_id or {}).items():
        if str(owner.state) not in valid_states:
            errors.append({
                "reason": "invalid_ownership_state",
                "node_id": int(node_id),
                "state": str(owner.state),
            })
        if int(node_id) in closure_ids and str(owner.state) != Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value:
            errors.append({"reason": "closure_pool_state_mismatch", "node_id": int(node_id), "state": str(owner.state)})
        if int(node_id) in donor_ids and str(owner.state) != Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value:
            errors.append({"reason": "donor_pool_state_mismatch", "node_id": int(node_id), "state": str(owner.state)})
    errors.extend(deepcopy(list(context.conflicts or [])))
    return errors


def ownership_public_view(context: Stage3RoundContext) -> Dict[str, Any]:
    by_state: Dict[str, List[int]] = defaultdict(list)
    for node_id, owner in dict(context.ownership_by_node_id or {}).items():
        by_state[str(owner.state)].append(int(node_id))
    return {
        "generation_id": int(context.generation_id),
        "states": {str(key): sorted(values) for key, values in sorted(by_state.items())},
        "closure_pool_ids_by_su": {
            str(key): sorted(int(value) for value in values)
            for key, values in sorted(dict(context.closure_pool_ids_by_su or {}).items())
        },
        "aromatic_donor_ids_by_su": {
            str(key): sorted(int(value) for value in values)
            for key, values in sorted(dict(context.aromatic_donor_ids_by_su or {}).items())
        },
        "invariant_errors": round_context_invariants(context),
    }
