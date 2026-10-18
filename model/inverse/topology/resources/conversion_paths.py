"""Ordered physical-node identities shared by Stage3 planning and execution."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class ConversionPathError(ValueError):
    def __init__(self, proof: Dict[str, Any]):
        self.proof = proof
        errors = list(proof.get("errors", []) or [])
        super().__init__(str(errors[0]["reason"] if errors else "invalid_conversion_path"))


def audit_conversion_paths(
    groups: Sequence[Any],
    *,
    initial_ids_by_su: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    current_types: Dict[int, int] = {}
    errors = []
    if initial_ids_by_su is not None:
        for su_type, node_ids in dict(initial_ids_by_su).items():
            for node_id in list(node_ids or []):
                node_id = int(node_id)
                if node_id in current_types:
                    errors.append({"reason": "duplicate_authorized_resource_ids", "node_id": node_id})
                current_types[node_id] = int(su_type)
    paths: Dict[str, Any] = {}
    primary_ids = set()
    compensation_ids = set()
    physical_count = 0
    for group_index, group in enumerate(list(groups or [])):
        primary = getattr(group, "primary_op", None)
        operations = [("primary", primary)] + [
            ("h_compensation", operation)
            for operation in list(getattr(group, "h_compensation_ops", []) or [])
        ]
        for role, operation in operations:
            if operation is None:
                errors.append({"reason": "conversion_operation_missing", "group_index": group_index})
                continue
            source = int(getattr(operation, "from_su", -1))
            target = int(getattr(operation, "to_su", -1))
            count = int(getattr(operation, "count", 0) or 0)
            node_ids = [int(value) for value in list(getattr(operation, "node_ids", []) or [])]
            via = getattr(operation, "via_su", None)
            via = int(via) if via is not None else None
            context = {"group_index": group_index, "role": role, "from_su": source, "to_su": target}
            if count <= 0 or len(node_ids) != count:
                errors.append({
                    **context, "reason": "conversion_requires_concrete_node_ids",
                    "expected": count, "node_ids": node_ids,
                })
                continue
            if len(node_ids) != len(set(node_ids)):
                errors.append({
                    **context, "reason": "duplicate_real_conversion_node_id_in_operation",
                    "node_ids": node_ids,
                })
                continue
            if source < 0 or target < 0 or source == target or (via is not None and via < 0):
                errors.append({**context, "reason": "invalid_conversion_types", "via_su": via})
                continue
            physical_count += count * (2 if via is not None else 1)
            for node_id in node_ids:
                if node_id < 0:
                    errors.append({**context, "reason": "conversion_contains_invalid_node_id", "node_id": node_id})
                    continue
                if node_id not in current_types and initial_ids_by_su is None:
                    current_types[node_id] = source
                if current_types.get(node_id) != source:
                    errors.append({
                        **context, "reason": "non_contiguous_real_conversion_path", "node_id": node_id,
                        "expected_from_su": current_types.get(node_id), "actual_from_su": source,
                    })
                    continue
                path = paths.setdefault(str(node_id), {
                    "node_id": node_id, "initial_su": source, "final_su": source,
                    "su_path": [source], "steps": [], "primary_group_indexes": [],
                })
                path["steps"].append({**context, "via_su": via})
                path["su_path"].extend(([via] if via is not None else []) + [target])
                path["final_su"] = target
                current_types[node_id] = target
                if role == "primary":
                    primary_ids.add(node_id)
                    path["primary_group_indexes"].append(group_index)
                else:
                    compensation_ids.add(node_id)
    return {
        "ok": not errors,
        "errors": errors,
        "node_paths": paths,
        "primary_node_ids": sorted(primary_ids),
        "h_compensation_node_ids": sorted(compensation_ids),
        "primary_adjustment_count": len(primary_ids),
        "physical_conversion_count": physical_count,
    }


def require_conversion_paths(
    groups: Sequence[Any],
    *,
    initial_ids_by_su: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    proof = audit_conversion_paths(groups, initial_ids_by_su=initial_ids_by_su)
    if not proof["ok"]:
        raise ConversionPathError(proof)
    return proof


def bind_conversion_paths(
    groups: Sequence[Any],
    pool: Mapping[Any, Any],
    *,
    protected_node_ids: Optional[Sequence[int]] = None,
    donor_ids_by_conversion: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    working = {int(key): list(values) for key, values in dict(pool).items()}
    protected = {int(value) for value in list(protected_node_ids or [])}
    for group in list(groups or []):
        for operation in [group.primary_op] + list(group.h_compensation_ops or []):
            source = int(operation.from_su)
            target = int(operation.to_su)
            eligible = [int(value) for value in working.get(source, []) if int(value) not in protected]
            allowed = dict(donor_ids_by_conversion or {}).get((source, target))
            if allowed is not None:
                allowed_ids = {int(value) for value in allowed}
                eligible = [node_id for node_id in eligible if node_id in allowed_ids]
            requested = [int(value) for value in list(operation.node_ids or [])]
            if requested and (
                len(requested) != int(operation.count)
                or len(set(requested)) != len(requested)
                or not set(requested).issubset(set(eligible))
            ):
                raise ConversionPathError({"ok": False, "errors": [{
                    "reason": "requested_conversion_donor_not_authorized",
                    "from_su": source, "to_su": target,
                    "node_ids": requested, "eligible_node_ids": eligible,
                }]})
            selected = requested if requested else eligible[:max(0, int(operation.count))]
            operation.node_ids = selected
            selected_ids = set(selected)
            working[source] = [value for value in working.get(source, []) if int(value) not in selected_ids]
            working.setdefault(target, []).extend(selected)
            working[target].sort()
    proof = require_conversion_paths(groups, initial_ids_by_su=pool)
    return working, proof


def internal_primary_adjustment_counts(groups: Sequence[Any]) -> Tuple[int, int]:
    ordinary_ids = set()
    reclassification_ids = set()
    ordinary_unbound = 0
    reclassification_unbound = 0
    for group in list(groups or []):
        operation = getattr(group, "primary_op", None)
        if operation is None:
            continue
        node_ids = [int(value) for value in list(getattr(operation, "node_ids", []) or [])]
        kind = dict(getattr(group, "scoring_diagnostics", {}) or {}).get("plan_kind", "")
        if kind == "residual_1122_reclassification":
            reclassification_ids.update(node_ids)
            if not node_ids:
                reclassification_unbound += max(0, int(getattr(operation, "count", 0) or 0))
        else:
            ordinary_ids.update(node_ids)
            if not node_ids:
                ordinary_unbound += max(0, int(getattr(operation, "count", 0) or 0))
    return (
        len(ordinary_ids) + ordinary_unbound,
        len(reclassification_ids - ordinary_ids) + reclassification_unbound,
    )
