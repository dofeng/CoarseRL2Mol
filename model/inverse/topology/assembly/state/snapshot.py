"""Read-only identity checks shared by completion and failure exports."""

from collections import Counter
from typing import Any, Dict, Sequence


# States that represent a valid terminal ownership assignment.  Claimed
# closure and SU24 ring resources are already consumed by a completed product;
# they must not be mistaken for unowned nodes during the final audit.
COMPLETION_OWNERSHIP_STATES = frozenset({
    "completed_topology",
    "support_context",
    "remaining_aromatic_donor",
    "remaining_closure_resource",
    "claimed_closure_resource",
    "claimed_su24_ring_resource",
})


def validate_snapshot_state(
    nodes: Sequence[Any], snapshot: Dict[str, Any], H: Any = None,
    *, require_complete: bool = False,
) -> Dict[str, Any]:
    actual = {int(node.global_id): int(node.su_type) for node in nodes}
    errors = []
    if len(actual) != len(nodes) or any(node_id < 0 for node_id in actual):
        errors.append({"reason": "duplicate_or_invalid_node_id"})
    if H is not None:
        counts = Counter(actual.values())
        histogram = [int(value) for value in H.tolist()]
        if any(su < 0 or su >= len(histogram) for su in counts) or any(
            counts[su] != count for su, count in enumerate(histogram)
        ):
            errors.append({"reason": "node_histogram_mismatch"})

    def check(node_id, expected_su, reason):
        node_id, expected_su = int(node_id), int(expected_su)
        if actual.get(node_id) != expected_su:
            errors.append({"reason": reason, "node_id": node_id,
                           "expected_su": expected_su, "actual_su": actual.get(node_id)})

    ledger = dict(snapshot.get("iteration_topology_ledger", {}) or {})
    if require_complete or snapshot.get("stage3_complete", False):
        owners = dict(ledger.get("ownership_by_node_id", {}) or {})
        states = dict(dict(snapshot.get("ownership_ledger", {}) or {}).get("states", {}) or {})
        state_by_id = {int(node_id): state for state, ids in states.items() for node_id in ids}
        state_by_id.update({int(node_id): owner.get("state", "") for node_id, owner in owners.items()})
        missing = sorted(
            node_id
            for node_id in actual
            if state_by_id.get(node_id) not in COMPLETION_OWNERSHIP_STATES
        )
        if missing:
            errors.append({"reason": "completion_ownership_incomplete", "node_ids": missing})
        if snapshot.get("pending_contracts"):
            errors.append({"reason": "completion_has_pending_contracts"})
    for node_id, owner in dict(ledger.get("ownership_by_node_id", {}) or {}).items():
        check(node_id, owner["su_type"], "owner_su_type_mismatch")
    pools = []
    if "available_ids_by_su" in ledger:
        pools.append(ledger["available_ids_by_su"])
    resource_ledger = dict(snapshot.get("resource_ledger", {}) or {})
    resource_maps: Dict[str, Dict[int, set[int]]] = {}
    for field in (
        "initial_ids_by_su",
        "consumed_ids_by_su",
        "topology_consumed_ids_by_su",
        "residual_consumed_ids_by_su",
        "remaining_ids_by_su",
    ):
        raw_mapping = resource_ledger.get(field)
        if raw_mapping is None:
            continue
        if not isinstance(raw_mapping, dict):
            errors.append({"reason": "invalid_resource_id_map", "field": field})
            continue
        normalized: Dict[int, set[int]] = {}
        assigned_su_by_id: Dict[int, int] = {}
        for raw_su, raw_ids in raw_mapping.items():
            try:
                su_type = int(raw_su)
            except (TypeError, ValueError):
                errors.append({
                    "reason": "invalid_resource_su_key",
                    "field": field,
                    "su_type": raw_su,
                })
                continue
            values = [int(node_id) for node_id in list(raw_ids or [])]
            if len(values) != len(set(values)):
                errors.append({
                    "reason": "duplicate_resource_node",
                    "field": field,
                    "su_type": su_type,
                })
            normalized.setdefault(su_type, set()).update(values)
            for node_id in values:
                previous_su = assigned_su_by_id.get(int(node_id))
                if previous_su is not None and int(previous_su) != su_type:
                    errors.append({
                        "reason": "resource_node_assigned_to_multiple_su_types",
                        "field": field,
                        "node_id": int(node_id),
                        "su_types": sorted({int(previous_su), su_type}),
                    })
                else:
                    assigned_su_by_id[int(node_id)] = su_type
                check(node_id, su_type, "resource_su_type_mismatch")
        resource_maps[field] = normalized

        count_field = field.replace("_ids_by_su", "_counts")
        raw_counts = resource_ledger.get(count_field)
        if isinstance(raw_counts, dict):
            count_keys = set(normalized) | {
                int(key)
                for key in raw_counts
                if str(key).lstrip("-").isdigit()
            }
            mismatches = {
                str(su_type): {
                    "ids": len(normalized.get(su_type, set())),
                    "count": int(raw_counts.get(str(su_type), raw_counts.get(su_type, 0)) or 0),
                }
                for su_type in count_keys
                if len(normalized.get(su_type, set()))
                != int(raw_counts.get(str(su_type), raw_counts.get(su_type, 0)) or 0)
            }
            if mismatches:
                errors.append({
                    "reason": "resource_id_count_mismatch",
                    "field": field,
                    "mismatches": mismatches,
                })

    partition_fields = {
        "initial_ids_by_su",
        "consumed_ids_by_su",
        "remaining_ids_by_su",
    }
    if partition_fields & set(resource_maps):
        missing_fields = sorted(partition_fields - set(resource_maps))
        if missing_fields:
            errors.append({
                "reason": "resource_partition_fields_missing",
                "fields": missing_fields,
            })
        else:
            initial_ids = {
                node_id
                for values in resource_maps["initial_ids_by_su"].values()
                for node_id in values
            }
            consumed_ids = {
                node_id
                for values in resource_maps["consumed_ids_by_su"].values()
                for node_id in values
            }
            remaining_ids = {
                node_id
                for values in resource_maps["remaining_ids_by_su"].values()
                for node_id in values
            }
            if consumed_ids & remaining_ids:
                errors.append({
                    "reason": "consumed_remaining_resource_overlap",
                    "node_ids": sorted(consumed_ids & remaining_ids),
                })
            if initial_ids != consumed_ids | remaining_ids:
                errors.append({
                    "reason": "initial_resource_partition_invalid",
                    "missing_node_ids": sorted(initial_ids - consumed_ids - remaining_ids),
                    "extra_node_ids": sorted((consumed_ids | remaining_ids) - initial_ids),
                })

    category_fields = {
        "topology_consumed_ids_by_su",
        "residual_consumed_ids_by_su",
    }
    if category_fields & set(resource_maps):
        missing_fields = sorted(category_fields - set(resource_maps))
        if missing_fields or "consumed_ids_by_su" not in resource_maps:
            errors.append({
                "reason": "consumed_resource_category_fields_missing",
                "fields": missing_fields,
            })
        else:
            consumed_ids = {
                node_id
                for values in resource_maps["consumed_ids_by_su"].values()
                for node_id in values
            }
            topology_ids = {
                node_id
                for values in resource_maps["topology_consumed_ids_by_su"].values()
                for node_id in values
            }
            residual_ids = {
                node_id
                for values in resource_maps["residual_consumed_ids_by_su"].values()
                for node_id in values
            }
            if topology_ids & residual_ids:
                errors.append({
                    "reason": "topology_residual_resource_overlap",
                    "node_ids": sorted(topology_ids & residual_ids),
                })
            if consumed_ids != topology_ids | residual_ids:
                errors.append({
                    "reason": "consumed_resource_category_partition_invalid",
                    "missing_node_ids": sorted(consumed_ids - topology_ids - residual_ids),
                    "extra_node_ids": sorted((topology_ids | residual_ids) - consumed_ids),
                })

    if "remaining_ids_by_su" in resource_ledger:
        pools.append(resource_ledger["remaining_ids_by_su"])
    resource_ids = set()
    resource_su_by_node: Dict[int, set[int]] = {}
    for pool in pools:
        seen = set()
        for su, ids in pool.items():
            try:
                su_type = int(su)
            except (TypeError, ValueError):
                errors.append({"reason": "invalid_resource_su_key", "su_type": su})
                continue
            for node_id in ids:
                check(node_id, su_type, "resource_su_type_mismatch")
                if int(node_id) in seen:
                    errors.append({"reason": "duplicate_resource_node", "node_id": int(node_id)})
                seen.add(int(node_id))
                resource_su_by_node.setdefault(int(node_id), set()).add(su_type)
        resource_ids.update(seen)
    for node_id, su_types in resource_su_by_node.items():
        if len(su_types) > 1:
            errors.append({
                "reason": "resource_node_assigned_to_multiple_su_types",
                "node_id": int(node_id),
                "su_types": sorted(su_types),
            })

    # The ownership ledger is the authority for whether an available resource
    # is still free.  A resource pool entry with a completed/pending owner is
    # a stale or contradictory snapshot and must never pass the final audit.
    owner_state_by_id: Dict[int, str] = {}
    for raw_id, raw_owner in dict(ledger.get("ownership_by_node_id", {}) or {}).items():
        owner = dict(raw_owner or {})
        try:
            owner_state_by_id[int(raw_id)] = str(owner.get("state", "") or "")
        except (TypeError, ValueError):
            errors.append({"reason": "invalid_ownership_node_id", "node_id": raw_id})
    for state, ids in dict(
        dict(snapshot.get("ownership_ledger", {}) or {}).get("states", {}) or {}
    ).items():
        for raw_id in list(ids or []):
            try:
                owner_state_by_id.setdefault(int(raw_id), str(state))
            except (TypeError, ValueError):
                errors.append({"reason": "invalid_ownership_node_id", "node_id": raw_id})
    expected_resource_states = {
        10: "remaining_aromatic_donor",
        12: "remaining_aromatic_donor",
        13: "remaining_aromatic_donor",
        11: "remaining_closure_resource",
        22: "remaining_closure_resource",
        23: "remaining_closure_resource",
        24: "remaining_su24_ring_resource",
    }
    for node_id, su_types in resource_su_by_node.items():
        expected_states = {
            expected_resource_states[su_type]
            for su_type in su_types
            if su_type in expected_resource_states
        }
        actual_state = owner_state_by_id.get(int(node_id), "")
        if expected_states and actual_state and actual_state not in expected_states:
            errors.append({
                "reason": "resource_owner_state_mismatch",
                "node_id": int(node_id),
                "su_types": sorted(su_types),
                "expected_states": sorted(expected_states),
                "actual_state": actual_state,
            })
    for product in snapshot.get("products", []):
        for node_id, su in product.get("expected_su_type_by_node_id", {}).items():
            check(node_id, su, "product_su_type_mismatch")
            if int(node_id) in resource_ids:
                errors.append({"reason": "product_node_in_resource_pool", "node_id": int(node_id)})
    for raw_error in list(resource_ledger.get("invariant_errors", []) or []):
        errors.append(
            dict(raw_error)
            if isinstance(raw_error, dict)
            else {
                "reason": "resource_ledger_invariant_error",
                "detail": str(raw_error),
            }
        )
    errors.extend(list(ledger.get("conflicts", []) or []))
    errors.extend(list(snapshot.get("snapshot_errors", []) or []))
    return {"ok": not errors, "checked": True, "node_count": len(actual), "errors": errors}
