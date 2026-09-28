"""Iteration-level closure and ownership ledger helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...stage3_models import (
    Stage3NodeOwnership,
    Stage3OwnershipLedger,
    Stage3OwnershipState,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologySlotKind,
)
from ...resources.profiles import closure_sequences
from ...resources.types import (
    GENERIC_CLOSURE_RESOURCE_TYPES,
    STAGE3_ACCOUNTING_SU_TYPES,
    SU24_RING_RESOURCE_TYPE,
)
from .round_context import normalize_stage3_product


def slot_sequence(kind: str) -> List[int]:
    if str(kind) == TopologySlotKind.CLOSED_TERMINAL.value:
        return []
    options = closure_sequences(str(kind))
    if not options:
        return []
    return [int(value) for value in list(options[0] or [])]


def normalize_slot(slot: Any) -> TopologyClosureSlot:
    if isinstance(slot, TopologyClosureSlot):
        result = deepcopy(slot)
    else:
        data = dict(slot or {})
        raw_anchor_node_id = data.get("anchor_node_id", -1)
        result = TopologyClosureSlot(
            slot_id=str(data.get("slot_id", "")),
            kind=str(data.get("kind", "")),
            anchor_node_id=(
                -1
                if raw_anchor_node_id is None or raw_anchor_node_id == ""
                else int(raw_anchor_node_id)
            ),
            anchor_role=str(data.get("anchor_role", "")),
            required_su_sequence=[int(value) for value in list(data.get("required_su_sequence", []) or [])],
            existing_node_ids=[int(value) for value in list(data.get("existing_node_ids", []) or [])],
            assigned_resource_ids=[int(value) for value in list(data.get("assigned_resource_ids", []) or [])],
            state=str(data.get("state", "OPEN")),
        )
    if not result.required_su_sequence:
        result.required_su_sequence = slot_sequence(result.kind)
    if str(result.kind) == TopologySlotKind.CLOSED_TERMINAL.value:
        result.state = TopologySlotKind.CLOSED_TERMINAL.value
    return result


def contract_slots(contract: TopologyClosureContract) -> List[TopologyClosureSlot]:
    slots = [normalize_slot(slot) for slot in list(contract.mainline_slots or []) + list(contract.branch_slots or [])]
    if slots or not dict(contract.required_counts or {}):
        return slots
    sequence: List[int] = []
    for su_type, count in sorted(dict(contract.required_counts or {}).items()):
        sequence.extend([int(su_type)] * int(count))
    anchor_ids = list(contract.claimed_source_node_ids)
    return [TopologyClosureSlot(
        slot_id=f"contract:{int(contract.contract_id)}:resource_path",
        kind="RESOURCE_PATH",
        anchor_node_id=int(anchor_ids[-1]) if anchor_ids else -1,
        anchor_role="resource_path",
        required_su_sequence=sequence,
    )]


def required_counts_from_slots(slots: Sequence[TopologyClosureSlot]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for slot in list(slots or []):
        normalized = normalize_slot(slot)
        if normalized.is_complete():
            continue
        for su_type in normalized.required_su_sequence:
            counts[int(su_type)] = int(counts.get(int(su_type), 0)) + 1
    return counts


def build_iteration_ledger(
    nodes: Sequence[Any],
    *,
    resource_ids_by_su: Optional[Dict[Any, Iterable[int]]] = None,
    support_context_node_ids: Optional[Iterable[int]] = None,
    claimed_source_ids: Optional[Iterable[int]] = None,
) -> Stage3OwnershipLedger:
    resources = {
        int(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(resource_ids_by_su or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    for su_type in STAGE3_ACCOUNTING_SU_TYPES:
        resources.setdefault(int(su_type), [])
    support = {int(value) for value in list(support_context_node_ids or [])}
    claimed = {int(value) for value in list(claimed_source_ids or [])}
    resource_ids = {int(value) for values in resources.values() for value in values}
    ownership: Dict[int, Stage3NodeOwnership] = {}
    conflicts: List[int] = []
    for node in list(nodes or []):
        node_id = int(getattr(node, "global_id", -1))
        su_type = int(getattr(node, "su_type", -1))
        if node_id in claimed:
            state = Stage3OwnershipState.PENDING_TOPOLOGY.value
            role = "fragment_source"
        elif node_id in resource_ids:
            state = (
                Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value
                if int(su_type) == SU24_RING_RESOURCE_TYPE
                else Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value
                if int(su_type) in GENERIC_CLOSURE_RESOURCE_TYPES
                else Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value
            )
            role = str(state)
        elif node_id in support:
            state = Stage3OwnershipState.SUPPORT_CONTEXT.value
            role = Stage3OwnershipState.SUPPORT_CONTEXT.value
        else:
            state = Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
            role = Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
            conflicts.append(node_id)
        ownership[node_id] = Stage3NodeOwnership(
            node_id=node_id,
            su_type=su_type,
            state=state,
            role=role,
        )
    return Stage3OwnershipLedger(
        ownership_by_node_id=ownership,
        available_ids_by_su=resources,
        support_context_node_ids=sorted(support),
        conflict_node_ids=sorted(conflicts),
    )


def seed_ledger_topology_state(
    ledger: Stage3OwnershipLedger,
    *,
    completed_products: Optional[Sequence[Dict[str, Any]]] = None,
    pending_contracts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Stage3OwnershipLedger:
    working = deepcopy(ledger)
    for index, raw in enumerate(list(completed_products or [])):
        product = dict(raw or {})
        product_id = str(product.get("product_id", f"product:{index}"))
        fragment_ids = [int(value) for value in list(product.get("source_fragment_ids", []) or [])]
        fixed_node_ids = [
            int(value)
            for value in list(product.get("fixed_source_node_ids", []) or [])
        ]
        resource_node_ids = [
            int(value)
            for values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).values()
            for value in list(values or [])
        ]
        claimed_su_by_node_id = {
            int(node_id): int(su_type)
            for su_type, values in dict(
                product.get(
                    "claimed_resource_ids_by_su",
                    {},
                )
                or {}
            ).items()
            if str(su_type).lstrip("-").isdigit()
            for node_id in list(values or [])
        }
        resource_id_set = set(resource_node_ids)
        conflict_count = len(working.conflicts)
        if len(resource_node_ids) != len(resource_id_set) or resource_id_set.intersection(fixed_node_ids):
            working.conflicts.append({
                "reason": "completed_product_duplicate_node",
                "product_id": product_id,
            })
        expected_su = {
            int(node_id): int(su_type)
            for node_id, su_type in dict(product.get("expected_su_type_by_node_id", {}) or {}).items()
        }
        for node_id, claimed_su in claimed_su_by_node_id.items():
            if node_id in expected_su and expected_su[node_id] != claimed_su:
                working.conflicts.append({
                    "node_id": node_id,
                    "reason": "completed_product_su_declaration_conflict",
                    "product_id": product_id,
                })
            expected_su[node_id] = claimed_su
        for node_id in sorted(set(fixed_node_ids + resource_node_ids)):
            owner = working.ownership_by_node_id.get(int(node_id))
            if owner is None:
                working.conflicts.append({"node_id": int(node_id), "reason": "completed_product_node_missing"})
                continue
            if owner.state in {
                Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
                Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
                Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
            } and owner.owner_product_id != product_id:
                working.conflicts.append({
                    "node_id": int(node_id),
                    "reason": "duplicate_completed_product",
                    "owners": [str(owner.owner_product_id), product_id],
                })
                continue
            claimed_su = expected_su.get(int(node_id))
            if claimed_su is not None and int(owner.su_type) != int(claimed_su):
                working.conflicts.append({
                    "node_id": int(node_id),
                    "reason": "completed_product_su_type_mismatch",
                    "product_id": product_id,
                    "claimed_su_type": int(claimed_su),
                    "actual_su_type": int(owner.su_type),
                })
                continue
        # A product is indivisible. Never publish a completed product with
        # only a subset of its resources claimed after an identity failure.
        if len(working.conflicts) != conflict_count:
            continue
        product["expected_su_type_by_node_id"] = {
            int(node_id): int(working.ownership_by_node_id[int(node_id)].su_type)
            for node_id in set(fixed_node_ids + resource_node_ids)
        }
        for node_id in sorted(set(fixed_node_ids + resource_node_ids)):
            owner = working.ownership_by_node_id[int(node_id)]
            for values in working.available_ids_by_su.values():
                while int(node_id) in values:
                    values.remove(int(node_id))
            if int(node_id) in working.conflict_node_ids:
                working.conflict_node_ids.remove(int(node_id))
            owner.state = (
                Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value
                if int(node_id) in resource_id_set
                and int(owner.su_type) == SU24_RING_RESOURCE_TYPE
                else Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
                if int(node_id) in resource_id_set
                else Stage3OwnershipState.COMPLETED_TOPOLOGY.value
            )
            owner.owner_fragment_id = min(fragment_ids) if fragment_ids else None
            owner.owner_product_id = product_id
            owner.role = (
                "su24_ring_skeleton_resource"
                if owner.state
                == Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value
                else "symbolic_closure_resource"
                if owner.state == Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
                else "fixed_source"
            )
        working.completed_products.append(product)
    for index, raw in enumerate(list(pending_contracts or [])):
        contract = dict(raw or {})
        if "contract_id" not in contract:
            raise ValueError(f"canonical topology contract is missing contract_id: {contract}")
        contract_id = int(contract["contract_id"])
        fragment_ids = [int(value) for value in list(contract.get("source_fragment_ids", []) or [])]
        node_ids = [
            int(value)
            for value in list(contract.get("claimed_source_node_ids", []) or [])
        ]
        source_status, source_proof = contract_source_status(
            working,
            node_ids,
            current_contract_id=int(contract_id),
        )
        if source_status == "superseded":
            continue
        if source_status == "pending":
            working.conflicts.append({
                "reason": "pending_contract_overlap",
                "contract_id": int(contract_id),
                **{
                    str(key): deepcopy(value)
                    for key, value in dict(source_proof or {}).items()
                    if str(key) != "reason"
                },
            })
            continue
        if source_status == "conflict":
            working.conflicts.append({
                "reason": str(source_proof.get("reason", "pending_contract_overlap")),
                "contract_id": int(contract_id),
                **{
                    str(key): deepcopy(value)
                    for key, value in dict(source_proof or {}).items()
                    if str(key) != "reason"
                },
            })
            continue
        for node_id in sorted(set(node_ids)):
            owner = working.ownership_by_node_id.get(int(node_id))
            if owner is None:
                working.conflicts.append({"node_id": int(node_id), "reason": "pending_contract_node_missing"})
                continue
            if owner.state in {
                Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
                Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
                Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
            }:
                working.conflicts.append({
                    "node_id": int(node_id),
                    "reason": "completed_pending_overlap",
                    "contract_id": contract_id,
                })
                continue
            for values in working.available_ids_by_su.values():
                if int(node_id) in values:
                    values.remove(int(node_id))
            if int(node_id) in working.conflict_node_ids:
                working.conflict_node_ids.remove(int(node_id))
            owner.state = Stage3OwnershipState.PENDING_TOPOLOGY.value
            owner.owner_fragment_id = min(fragment_ids) if fragment_ids else None
            owner.owner_contract_id = contract_id
            owner.role = "fragment_source"
        normalized = dict(contract)
        normalized["contract_id"] = contract_id
        normalized["claimed_source_node_ids"] = sorted(set(node_ids))
        # Re-seeding the same live contract refreshes its demand/proof; it
        # must not append another pending row for the same physical sources.
        existing_index = next((
            index for index, item in enumerate(working.pending_contracts)
            if int(item["contract_id"]) == contract_id
            and sorted(set(item.get("claimed_source_node_ids", []) or []))
            == normalized["claimed_source_node_ids"]
        ), None)
        if existing_index is None:
            working.pending_contracts.append(normalized)
        else:
            working.pending_contracts[existing_index] = normalized
    return working


def contract_source_status(
    ledger: Stage3OwnershipLedger,
    claimed_source_node_ids: Sequence[int],
    *,
    current_contract_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    node_ids = sorted({
        int(value) for value in list(claimed_source_node_ids or [])
    })
    if not node_ids:
        return "available", {}
    owners = [ledger.ownership_by_node_id.get(int(node_id)) for node_id in node_ids]
    missing_ids = [node_id for node_id, owner in zip(node_ids, owners) if owner is None]
    if missing_ids:
        return "conflict", {"reason": "source_node_missing", "node_ids": missing_ids}
    # Check every committed owner before allowing continuation of pending
    # sources. One pending source must not hide another source that has already
    # been consumed by a product (including a claimed closure/ring resource).
    completed_states = {
        Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
        Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
        Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
    }
    completed = [
        owner for owner in owners
        if owner.state in completed_states or owner.owner_product_id
    ]
    if completed:
        product_ids = {str(owner.owner_product_id or "") for owner in completed}
        if len(completed) == len(owners) and len(product_ids) == 1 and "" not in product_ids:
            return "superseded", {
                "reason": "source_contract_superseded",
                "claimed_source_node_ids": node_ids,
                "owner_product_id": next(iter(product_ids)),
            }
        return "conflict", {
            "reason": "contract_identity_conflict",
            "claimed_source_node_ids": node_ids,
            "completed_node_ids": sorted(int(owner.node_id) for owner in completed),
            "available_node_ids": sorted(
                int(owner.node_id) for owner in owners
                if owner.state not in completed_states and not owner.owner_product_id
            ),
            "owner_product_ids": sorted(product_ids),
        }
    pending = [
        owner for owner in owners
        if owner is not None
        and owner.state == Stage3OwnershipState.PENDING_TOPOLOGY.value
    ]
    if pending:
        pending_contract_ids = sorted({
            int(owner.owner_contract_id)
            for owner in pending
            if owner.owner_contract_id is not None
        })
        if (
            current_contract_id is not None
            and (
                not pending_contract_ids
                or set(pending_contract_ids) == {int(current_contract_id)}
            )
        ):
            return "available", {
                "reason": "source_contract_pending_current",
                "claimed_source_node_ids": node_ids,
                "pending_contract_ids": pending_contract_ids,
            }
        return "pending", {
            "reason": "source_contract_pending",
            "claimed_source_node_ids": node_ids,
            "pending_contract_ids": pending_contract_ids,
        }
    return "available", {}


def assign_contract_slots(
    ledger: Stage3OwnershipLedger,
    contract: TopologyClosureContract,
) -> Tuple[Optional[Stage3OwnershipLedger], List[TopologyClosureSlot], Dict[int, List[int]], Dict[str, Any]]:
    working = deepcopy(ledger)
    slots = deepcopy(contract_slots(contract))
    claimed: Dict[int, List[int]] = {}
    used_ids: set[int] = set()
    source_ids = list(contract.claimed_source_node_ids)
    source_status, source_proof = contract_source_status(
        working,
        source_ids,
        current_contract_id=int(contract.contract_id),
    )
    if source_status != "available":
        return None, slots, {}, dict(source_proof or {})
    for source_id in source_ids:
        owner = working.ownership_by_node_id.get(int(source_id))
        if owner is None or owner.owner_product_id or owner.state in {
            Stage3OwnershipState.COMPLETED_TOPOLOGY.value,
            Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value,
            Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
        }:
            return None, slots, {}, {
                "reason": "source_node_unavailable",
                "node_id": int(source_id),
                "state": None if owner is None else str(owner.state),
            }
        if owner.state in {
            Stage3OwnershipState.REMAINING_CLOSURE_RESOURCE.value,
            Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value,
            Stage3OwnershipState.REMAINING_AROMATIC_DONOR.value,
        }:
            resource_ids = working.available_ids_by_su.get(int(owner.su_type), [])
            if int(source_id) in resource_ids:
                resource_ids.remove(int(source_id))
        owner.state = Stage3OwnershipState.PENDING_TOPOLOGY.value
        owner.owner_fragment_id = min(list(contract.source_fragment_ids or [-1]))
        owner.owner_contract_id = int(contract.contract_id)
        owner.role = "fragment_source"
    for slot in slots:
        if slot.is_complete():
            continue
        assigned: List[int] = []
        for su_type in slot.required_su_sequence:
            available = [
                int(node_id)
                for node_id in list(working.available_ids_by_su.get(int(su_type), []) or [])
                if int(node_id) not in used_ids
            ]
            if not available:
                return None, slots, {}, {
                    "reason": "slot_resource_deficit",
                    "slot_id": str(slot.slot_id),
                    "slot_kind": str(slot.kind),
                    "missing_su": int(su_type),
                    "required_counts": required_counts_from_slots(slots),
                    "available_counts": {
                        int(key): len(list(values or []))
                        for key, values in working.available_ids_by_su.items()
                    },
                }
            node_id = int(available[0])
            owner = working.ownership_by_node_id.get(node_id)
            if owner is None or int(owner.su_type) != int(su_type) or owner.owner_product_id:
                return None, slots, {}, {
                    "reason": "slot_resource_identity_conflict",
                    "node_id": node_id,
                    "required_su": int(su_type),
                }
            working.available_ids_by_su[int(su_type)].remove(node_id)
            used_ids.add(node_id)
            assigned.append(node_id)
            claimed.setdefault(int(su_type), []).append(node_id)
            owner = working.ownership_by_node_id[node_id]
            owner.state = Stage3OwnershipState.PENDING_TOPOLOGY.value
            owner.owner_contract_id = int(contract.contract_id)
            owner.owner_slot_id = str(slot.slot_id)
            owner.role = str(slot.kind)
        slot.assigned_resource_ids = list(assigned)
        slot.state = "FILLED"
    product_id = (
        f"residual:{int(contract.contract_id)}"
        if str(contract.contract_kind) == "residual_conversion"
        else str(contract.expected_topology or contract.contract_id)
    )
    product_nodes = sorted({
        int(value)
        for value in list(contract.claimed_source_node_ids)
        + [value for values in claimed.values() for value in values]
        + [value for slot in slots for value in list(slot.existing_node_ids or [])]
    })
    claimed_id_set = {value for values in claimed.values() for value in values}
    for node_id in product_nodes:
        owner = working.ownership_by_node_id.get(int(node_id))
        if owner is None:
            return None, slots, {}, {"reason": "product_node_missing", "node_id": int(node_id)}
        owner.state = (
            Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value
            if int(node_id) in claimed_id_set
            and int(owner.su_type) == SU24_RING_RESOURCE_TYPE
            else Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
            if int(node_id) in claimed_id_set
            else Stage3OwnershipState.COMPLETED_TOPOLOGY.value
        )
        owner.owner_contract_id = int(contract.contract_id)
        owner.owner_product_id = product_id
        owner.role = (
            "su24_ring_skeleton_resource"
            if owner.state
            == Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value
            else "symbolic_closure_resource"
            if owner.state == Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
            else "fixed_source"
        )
    provisional = bool(
        dict(contract.diagnostics or {}).get("provisional_requires_rebuild", False)
    )
    product = {
        "product_id": product_id,
        "contract_id": int(contract.contract_id),
        "contract_kind": str(contract.contract_kind),
        "phase": str(contract.phase),
        "target_role": str(contract.target_role),
        "product_role": str(
            dict(contract.diagnostics or {}).get("product_role", "") or ""
        ),
        "chain_23_count": int(
            dict(contract.diagnostics or {}).get("chain_23_count", 0) or 0
        ),
        "source_fragment_ids": list(contract.source_fragment_ids),
        "source_fragment_classes": list(contract.source_fragment_classes),
        "fixed_source_node_ids": list(contract.claimed_source_node_ids),
        "claimed_resource_ids_by_su": {str(key): list(values) for key, values in claimed.items()},
        "expected_su_type_by_node_id": {
            int(node_id): int(working.ownership_by_node_id[int(node_id)].su_type)
            for node_id in product_nodes
        },
        "mainline_slots": [
            slot.to_dict() for slot in slots[: len(list(contract.mainline_slots or []))]
        ],
        "branch_slots": [
            slot.to_dict() for slot in slots[len(list(contract.mainline_slots or [])):]
        ],
        "topology_notation": str(contract.expected_topology),
        "closed": not provisional,
        "reason": "provisional_requires_rebuild" if provisional else "",
        "provisional": bool(provisional),
    }
    normalized_product = normalize_stage3_product(
        product,
        generation_id=int(getattr(working, "generation_id", 0) or 0),
        index=len(list(working.completed_products or [])),
    ).to_dict()
    normalized_product.update({
        "contract_id": int(contract.contract_id),
        "contract_kind": str(contract.contract_kind),
        "phase": str(contract.phase),
        "target_role": str(contract.target_role),
        "product_role": str(product.get("product_role", "") or contract.target_role),
        "chain_23_count": int(product.get("chain_23_count", 0) or 0),
        "source_fragment_classes": list(contract.source_fragment_classes),
        "provisional": bool(provisional),
    })
    product = normalized_product
    working.completed_products.append(product)
    working.pending_contracts = [
        item for item in list(working.pending_contracts or [])
        if int(dict(item or {}).get("contract_id", -1)) != int(contract.contract_id)
    ]
    return working, slots, claimed, {}


def contract_priority(contract: Dict[str, Any]) -> Tuple[int, int, int]:
    data = dict(contract or {})
    target_role = str(data.get("target_role", "") or "")
    if not str(data.get("contract_kind", "") or "") or not target_role:
        raise ValueError(f"non-canonical topology contract: {data}")
    priority = {"flex_chain": 0, "fatty_ring": 1, "side_chain": 2}.get(target_role, 3)
    fragment_ids = [int(value) for value in list(data.get("source_fragment_ids", []) or [])]
    node_ids = [int(value) for value in list(data.get("claimed_source_node_ids", []) or [])]
    return int(priority), min(fragment_ids or [10**12]), min(node_ids or [10**12])
