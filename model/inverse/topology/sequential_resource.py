from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RESOURCE_SU_TYPES = (10, 11, 12, 13, 22, 23, 24)
DIRECT_CLOSURE_SU_TYPES = (11, 22, 23, 24)


@dataclass
class NodeOwnership:
    node_id: int
    su_type: int
    state: str
    owner_topology_id: Optional[int] = None
    owner_fragment_id: Optional[int] = None
    reason: str = ""


@dataclass
class OwnershipLedger:
    by_node_id: Dict[int, NodeOwnership] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def ids_by_su(self, state: str = "remaining_resource") -> Dict[int, List[int]]:
        result = {int(su): [] for su in RESOURCE_SU_TYPES}
        for ownership in self.by_node_id.values():
            if str(ownership.state) != str(state):
                continue
            if int(ownership.su_type) in result:
                result[int(ownership.su_type)].append(int(ownership.node_id))
        return {int(key): sorted(values) for key, values in result.items()}

    def claim(self, node_id: int, topology_id: int, reason: str) -> bool:
        ownership = self.by_node_id.get(int(node_id))
        if ownership is None or ownership.state != "remaining_resource":
            return False
        ownership.state = "pending_topology"
        ownership.owner_topology_id = int(topology_id)
        ownership.reason = str(reason)
        return True

    def summary(self) -> Dict[str, Any]:
        states: Dict[str, List[int]] = {}
        for ownership in self.by_node_id.values():
            states.setdefault(str(ownership.state), []).append(int(ownership.node_id))
        return {
            "states": {
                str(state): sorted(int(node_id) for node_id in node_ids)
                for state, node_ids in states.items()
            },
            "conflicts": [dict(item or {}) for item in list(self.conflicts or [])],
        }


def _owner_rows(
    items: Iterable[Dict[str, Any]],
    state: str,
    node_su_by_id: Optional[Dict[int, int]] = None,
) -> List[Tuple[int, int, Optional[int], str]]:
    rows: List[Tuple[int, int, Optional[int], str]] = []
    for item in list(items or []):
        data = dict(item or {})
        fragment_ids = [int(value) for value in list(data.get("source_fragment_ids", []) or [])]
        owner_fragment_id = min(fragment_ids) if fragment_ids else None
        owner_node_ids = [int(value) for value in list(data.get("source_node_ids", []) or [])]
        topology_id = int(data.get("unit_id", data.get("topology_id", -1)) or -1)
        if topology_id < 0:
            topology_id = int(owner_fragment_id if owner_fragment_id is not None else min(owner_node_ids or [-1]))
        if str(state) == "pending_topology":
            explicit_core_ids = [int(value) for value in list(data.get("core_node_ids", []) or [])]
            owner_node_ids = explicit_core_ids or [
                int(value) for value in list(data.get("source_node_ids", []) or [])
            ]
            owner_node_ids.extend(
                int(value)
                for value in list(data.get("source_node_ids", []) or [])
                if int(dict(node_su_by_id or {}).get(int(value), -1)) in DIRECT_CLOSURE_SU_TYPES
            )
            if not owner_node_ids and int(topology_id) >= 0:
                owner_node_ids = [int(topology_id)]
            owner_node_ids.extend(
                int(value)
                for values in dict(data.get("claimed_resource_ids_by_su", {}) or {}).values()
                for value in list(values or [])
            )
        for node_id in sorted(set(owner_node_ids)):
            rows.append((int(node_id), topology_id, owner_fragment_id, str(state)))
    return rows


def build_ownership_ledger(
    nodes: Sequence[Any],
    *,
    completed_products: Sequence[Dict[str, Any]],
    pending_topologies: Sequence[Dict[str, Any]],
    support_only_node_ids: Optional[Sequence[int]] = None,
    ignore_stage3_node_owners: bool = False,
) -> OwnershipLedger:
    ledger = OwnershipLedger()
    node_by_id = {int(getattr(node, "global_id", -1)): node for node in list(nodes or [])}
    support_ids = {int(value) for value in list(support_only_node_ids or [])}
    for node_id, node in node_by_id.items():
        stage3_owner = None if bool(ignore_stage3_node_owners) else getattr(node, "stage3_topology_owner_id", None)
        ledger.by_node_id[int(node_id)] = NodeOwnership(
            node_id=int(node_id),
            su_type=int(getattr(node, "su_type", -1)),
            state=(
                "completed_topology"
                if stage3_owner is not None
                else (
                    "support_only"
                    if int(node_id) in support_ids
                    else (
                    "remaining_resource"
                    if int(getattr(node, "su_type", -1)) in RESOURCE_SU_TYPES
                    else "unassigned_nonresource"
                    )
                )
            ),
            owner_topology_id=int(stage3_owner) if stage3_owner is not None else None,
            reason="stage3_materialized_topology" if stage3_owner is not None else "",
        )
    node_su_by_id = {
        int(node_id): int(getattr(node, "su_type", -1))
        for node_id, node in node_by_id.items()
    }
    for node_id, topology_id, fragment_id, state in _owner_rows(completed_products, "completed_topology", node_su_by_id) + _owner_rows(pending_topologies, "pending_topology", node_su_by_id):
        ownership = ledger.by_node_id.get(int(node_id))
        if ownership is None:
            if int(node_id) != int(topology_id):
                ledger.conflicts.append({"node_id": int(node_id), "reason": "owner_node_missing"})
            continue
        if str(state) == "pending_topology" and ownership.state == "completed_topology":
            continue
        if ownership.state in {"completed_topology", "pending_topology"} and ownership.owner_topology_id != int(topology_id):
            ledger.conflicts.append({
                "node_id": int(node_id),
                "reason": "duplicate_topology_ownership",
                "owners": [ownership.owner_topology_id, int(topology_id)],
            })
            continue
        ownership.state = str(state)
        ownership.owner_topology_id = int(topology_id)
        ownership.owner_fragment_id = fragment_id
    return ledger


def contract_priority(contract: Dict[str, Any]) -> Tuple[int, int, int]:
    data = dict(contract or {})
    target_role = str(data.get("target_role", "") or "")
    if not str(data.get("contract_kind", "") or "") or not target_role:
        raise ValueError(f"non-canonical topology contract: {data}")
    if target_role == "flex_chain":
        priority = 0
    elif target_role == "fatty_ring":
        priority = 1
    else:
        priority = 2
    fragment_ids = [int(value) for value in list(data.get("source_fragment_ids", []) or [])]
    node_ids = [int(value) for value in list(data.get("claimed_source_node_ids", []) or [])]
    return priority, min(fragment_ids or [10**12]), min(node_ids or [10**12])


def select_next_contract(contracts: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized = [dict(item or {}) for item in list(contracts or []) if dict(item or {})]
    if not normalized:
        return None
    return dict(sorted(normalized, key=contract_priority)[0])


def contract_required_counts(contract: Dict[str, Any]) -> Dict[int, int]:
    data = dict(contract or {})
    if not str(data.get("contract_kind", "") or "") or not str(data.get("target_role", "") or ""):
        raise ValueError(f"non-canonical topology contract: {data}")
    explicit = dict(data.get("required_counts", {}) or {})
    return {int(key): max(0, int(value)) for key, value in explicit.items() if int(value) > 0}


def claim_requirement(
    ledger: OwnershipLedger,
    contract: Dict[str, Any],
    required: Dict[int, int],
) -> Dict[str, Any]:
    snapshot = deepcopy(ledger.by_node_id)
    topology_id = int(dict(contract or {}).get("contract_id", -1) or -1)
    claimed: Dict[int, List[int]] = {}
    pool = ledger.ids_by_su()
    for su_type, count in sorted(required.items()):
        available = list(pool.get(int(su_type), []) or [])
        if len(available) < int(count):
            ledger.by_node_id = snapshot
            return {
                "ok": False,
                "claimed_ids_by_su": {},
                "deficit": {int(su_type): int(count) - len(available)},
            }
        selected = available[:int(count)]
        for node_id in selected:
            if not ledger.claim(node_id, topology_id, "single_topology_resource_claim"):
                ledger.by_node_id = snapshot
                return {"ok": False, "claimed_ids_by_su": {}, "deficit": {int(su_type): int(count)}}
        claimed[int(su_type)] = selected
    return {"ok": True, "claimed_ids_by_su": claimed, "deficit": {}}
