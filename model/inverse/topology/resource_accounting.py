from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


RESOURCE_TYPES = (11, 22, 23, 24)
STAGE3_RESOURCE_TYPES = (10, 11, 12, 13, 22, 23, 24)


def normalize_counts(values: Mapping[Any, Any]) -> Dict[int, int]:
    return {
        int(key): max(0, int(value))
        for key, value in dict(values or {}).items()
        if str(key).lstrip("-").isdigit()
    }


def positive_resource_gap(
    demand: Mapping[Any, Any],
    inventory: Mapping[Any, Any],
    *,
    resource_types: Iterable[int] = RESOURCE_TYPES,
) -> Dict[int, int]:
    required = normalize_counts(demand)
    available = normalize_counts(inventory)
    return {
        int(su_type): max(
            0,
            int(required.get(int(su_type), 0)) - int(available.get(int(su_type), 0)),
        )
        for su_type in tuple(int(value) for value in resource_types)
    }


def resource_gap_does_not_worsen(
    demand: Mapping[Any, Any],
    inventory_before: Mapping[Any, Any],
    inventory_after: Mapping[Any, Any],
    *,
    resource_types: Iterable[int] = RESOURCE_TYPES,
) -> bool:
    before = positive_resource_gap(
        demand,
        inventory_before,
        resource_types=resource_types,
    )
    after = positive_resource_gap(
        demand,
        inventory_after,
        resource_types=resource_types,
    )
    return all(
        int(after.get(int(su_type), 0)) <= int(before.get(int(su_type), 0))
        for su_type in tuple(int(value) for value in resource_types)
    )


def apply_count_delta(
    inventory: Mapping[Any, Any],
    delta: Mapping[Any, Any],
) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    result = normalize_counts(inventory)
    errors: List[Dict[str, Any]] = []
    for key, value in dict(delta or {}).items():
        su_type = int(key)
        result[su_type] = int(result.get(su_type, 0)) + int(value)
        if int(result[su_type]) < 0:
            errors.append({
                "reason": "negative_resource_inventory",
                "su_type": int(su_type),
                "count": int(result[su_type]),
            })
    return result, errors


def combined_group_delta(groups: Sequence[Any]) -> Dict[int, int]:
    delta: Counter = Counter()
    for group in list(groups or []):
        delta.update({
            int(key): int(value)
            for key, value in dict(getattr(group, "combined_delta", {}) or {}).items()
        })
    return {int(key): int(value) for key, value in delta.items() if int(value) != 0}


def _counts_equal(left: Mapping[Any, Any], right: Mapping[Any, Any]) -> bool:
    left_counts = normalize_counts(left)
    right_counts = normalize_counts(right)
    keys = set(left_counts) | set(right_counts)
    return all(
        int(left_counts.get(int(key), 0)) == int(right_counts.get(int(key), 0))
        for key in keys
    )


def _normalize_ids_by_su(values: Mapping[Any, Any]) -> Dict[int, List[int]]:
    normalized: Dict[int, List[int]] = {}
    for key, raw_ids in dict(values or {}).items():
        if not str(key).lstrip("-").isdigit():
            continue
        normalized[int(key)] = sorted({int(value) for value in list(raw_ids or [])})
    return normalized


def _duplicate_ids(values: Mapping[Any, Any]) -> List[int]:
    counts: Counter = Counter()
    for key, raw_ids in dict(values or {}).items():
        if not str(key).lstrip("-").isdigit():
            continue
        counts.update(int(value) for value in list(raw_ids or []))
    return sorted(int(node_id) for node_id, count in counts.items() if int(count) > 1)


def _node_types_from_ids(
    ids_by_su: Mapping[Any, Any],
) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    node_types: Dict[int, int] = {}
    errors: List[Dict[str, Any]] = []
    for su_type, node_ids in _normalize_ids_by_su(ids_by_su).items():
        for node_id in node_ids:
            prior = node_types.get(int(node_id))
            if prior is not None and int(prior) != int(su_type):
                errors.append({
                    "reason": "resource_node_has_multiple_types",
                    "node_id": int(node_id),
                    "su_types": sorted({int(prior), int(su_type)}),
                })
                continue
            node_types[int(node_id)] = int(su_type)
    return node_types, errors


def _ids_by_type(node_types: Mapping[int, int]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for node_id, su_type in dict(node_types or {}).items():
        grouped[int(su_type)].append(int(node_id))
    return {
        int(key): sorted(values)
        for key, values in grouped.items()
    }


def build_resource_accounting_snapshot(
    *,
    aggregate_ids_before_by_su: Mapping[Any, Any],
    authorized_ids_before_by_su: Mapping[Any, Any],
    final_remaining_ids_by_su: Mapping[Any, Any],
    committed_claimed_ids_by_su: Mapping[Any, Any],
    groups: Sequence[Any],
) -> Dict[str, Any]:
    aggregate_types, errors = _node_types_from_ids(aggregate_ids_before_by_su)
    authorized_types, authorized_errors = _node_types_from_ids(authorized_ids_before_by_su)
    errors.extend(authorized_errors)
    duplicate_aggregate_ids = _duplicate_ids(aggregate_ids_before_by_su)
    duplicate_authorized_ids = _duplicate_ids(authorized_ids_before_by_su)
    duplicate_claimed_ids = _duplicate_ids(committed_claimed_ids_by_su)
    if duplicate_aggregate_ids:
        errors.append({
            "reason": "duplicate_aggregate_resource_ids",
            "node_ids": duplicate_aggregate_ids,
        })
    if duplicate_authorized_ids:
        errors.append({
            "reason": "duplicate_authorized_resource_ids",
            "node_ids": duplicate_authorized_ids,
        })
    if duplicate_claimed_ids:
        errors.append({
            "reason": "resource_claimed_by_multiple_products",
            "node_ids": duplicate_claimed_ids,
        })
    aggregate_ids_before = _ids_by_type(aggregate_types)
    authorized_ids_before = _ids_by_type(authorized_types)

    seen_conversion_ids: set[int] = set()
    for group in list(groups or []):
        operations = [getattr(group, "primary_op", None)] + list(
            getattr(group, "h_compensation_ops", []) or []
        )
        for op in operations:
            if op is None:
                continue
            source = int(getattr(op, "from_su", -1))
            target = int(getattr(op, "to_su", -1))
            node_ids = [int(value) for value in list(getattr(op, "node_ids", []) or [])]
            if len(node_ids) != max(1, int(getattr(op, "count", 1) or 1)):
                errors.append({
                    "reason": "authorized_real_donor_ids_missing",
                    "from_su": int(source),
                    "to_su": int(target),
                    "node_ids": list(node_ids),
                })
                continue
            for node_id in node_ids:
                if int(node_id) in seen_conversion_ids:
                    errors.append({
                        "reason": "resource_node_converted_more_than_once",
                        "node_id": int(node_id),
                    })
                    continue
                seen_conversion_ids.add(int(node_id))
                if int(authorized_types.get(int(node_id), -1)) != int(source):
                    errors.append({
                        "reason": "conversion_donor_not_authorized",
                        "node_id": int(node_id),
                        "expected_su": int(source),
                        "authorized_su": int(authorized_types.get(int(node_id), -1)),
                    })
                    continue
                if int(aggregate_types.get(int(node_id), -1)) != int(source):
                    errors.append({
                        "reason": "aggregate_conversion_source_mismatch",
                        "node_id": int(node_id),
                        "expected_su": int(source),
                        "aggregate_su": int(aggregate_types.get(int(node_id), -1)),
                    })
                    continue
                aggregate_types[int(node_id)] = int(target)
                authorized_types[int(node_id)] = int(target)

    aggregate_ids_after = _ids_by_type(aggregate_types)
    authorized_ids_after = _ids_by_type(authorized_types)
    remaining_types, remaining_errors = _node_types_from_ids(final_remaining_ids_by_su)
    claimed_types, claimed_errors = _node_types_from_ids(committed_claimed_ids_by_su)
    errors.extend(remaining_errors)
    errors.extend(claimed_errors)

    overlap = sorted(set(remaining_types) & set(claimed_types))
    if overlap:
        errors.append({
            "reason": "remaining_and_claimed_resource_overlap",
            "node_ids": overlap,
        })
    partition_types = dict(remaining_types)
    for node_id, su_type in claimed_types.items():
        prior = partition_types.get(int(node_id))
        if prior is not None and int(prior) != int(su_type):
            errors.append({
                "reason": "claimed_resource_type_conflict",
                "node_id": int(node_id),
                "remaining_su": int(prior),
                "claimed_su": int(su_type),
            })
            continue
        partition_types[int(node_id)] = int(su_type)

    missing_ids = sorted(set(aggregate_types) - set(partition_types))
    extra_ids = sorted(set(partition_types) - set(aggregate_types))
    type_mismatches = [
        {
            "node_id": int(node_id),
            "aggregate_su": int(aggregate_types[node_id]),
            "partition_su": int(partition_types[node_id]),
        }
        for node_id in sorted(set(aggregate_types) & set(partition_types))
        if int(aggregate_types[node_id]) != int(partition_types[node_id])
    ]
    if missing_ids:
        errors.append({"reason": "aggregate_resource_ids_unaccounted", "node_ids": missing_ids})
    if extra_ids:
        errors.append({"reason": "partition_contains_unknown_resource_ids", "node_ids": extra_ids})
    if type_mismatches:
        errors.append({"reason": "aggregate_partition_type_mismatch", "items": type_mismatches})

    aggregate_before_counts = {
        int(key): len(values) for key, values in aggregate_ids_before.items()
    }
    aggregate_after_counts = {
        int(key): len(values) for key, values in aggregate_ids_after.items()
    }
    authorized_before_counts = {
        int(key): len(values) for key, values in authorized_ids_before.items()
    }
    authorized_after_counts = {
        int(key): len(values) for key, values in authorized_ids_after.items()
    }
    required_counts = Counter(int(value) for value in claimed_types.values())
    aggregate_required_counts = {
        int(key): int(value) for key, value in required_counts.items()
    }
    required_gap = positive_resource_gap(
        aggregate_required_counts,
        aggregate_after_counts,
    )
    if any(int(value) > 0 for value in required_gap.values()):
        errors.append({
            "reason": "aggregate_committed_demand_unsatisfied",
            "missing_counts": {
                int(key): int(value)
                for key, value in required_gap.items()
                if int(value) > 0
            },
        })

    return {
        "ok": not bool(errors),
        "errors": errors,
        "aggregate_ids_before_by_su": aggregate_ids_before,
        "aggregate_ids_after_by_su": aggregate_ids_after,
        "authorized_ids_before_by_su": authorized_ids_before,
        "authorized_ids_after_by_su": authorized_ids_after,
        "final_remaining_ids_by_su": _ids_by_type(remaining_types),
        "committed_claimed_ids_by_su": _ids_by_type(claimed_types),
        "aggregate_inventory_before": aggregate_before_counts,
        "aggregate_inventory_after": aggregate_after_counts,
        "authorized_inventory_before": authorized_before_counts,
        "authorized_inventory_after": authorized_after_counts,
        "aggregate_required_counts": aggregate_required_counts,
        "combined_delta": combined_group_delta(groups),
    }


def preflight_histogram_accounting(plan: Any) -> Tuple[bool, Dict[str, Any]]:
    authorized_before = normalize_counts(
        getattr(plan, "inventory_before", {}) or {}
    )
    authorized_after_expected = normalize_counts(
        getattr(plan, "inventory_after", {}) or {}
    )
    aggregate_before = normalize_counts(
        getattr(plan, "aggregate_inventory_before", {}) or {}
    )
    aggregate_after_expected = normalize_counts(
        getattr(plan, "aggregate_inventory_after", {}) or {}
    )
    aggregate_required = normalize_counts(
        getattr(plan, "aggregate_required_counts", {}) or {}
    )
    if not aggregate_before and not aggregate_after_expected and not aggregate_required:
        return True, {"mode": "authorized_inventory_only"}

    planning_accounting = dict(
        getattr(plan, "accounting_diagnostics", {}) or {}
    )
    errors: List[Dict[str, Any]] = []
    if planning_accounting and not bool(planning_accounting.get("ok", True)):
        errors.append({
            "reason": "planning_resource_accounting_failed",
            "errors": list(planning_accounting.get("errors", []) or []),
        })

    aggregate_after, delta_errors = apply_count_delta(
        aggregate_before,
        combined_group_delta(list(getattr(plan, "groups", []) or [])),
    )
    errors.extend(delta_errors)
    authorized_after, authorized_errors = apply_count_delta(
        authorized_before,
        combined_group_delta(list(getattr(plan, "groups", []) or [])),
    )
    errors.extend(authorized_errors)
    if authorized_after_expected and not _counts_equal(
        authorized_after,
        authorized_after_expected,
    ):
        errors.append({
            "reason": "authorized_inventory_differs_from_plan",
            "expected": authorized_after_expected,
            "actual": authorized_after,
        })
    if aggregate_after_expected and not _counts_equal(
        aggregate_after,
        aggregate_after_expected,
    ):
        errors.append({
            "reason": "aggregate_inventory_differs_from_plan",
            "expected": aggregate_after_expected,
            "actual": aggregate_after,
        })
    missing = positive_resource_gap(aggregate_required, aggregate_after)
    if any(int(value) > 0 for value in missing.values()):
        errors.append({
            "reason": "aggregate_required_counts_unsatisfied",
            "missing_counts": {
                int(key): int(value)
                for key, value in missing.items()
                if int(value) > 0
            },
        })
    return not bool(errors), {
        "mode": "aggregate_resource_accounting",
        "aggregate_inventory_before": aggregate_before,
        "aggregate_inventory_after": aggregate_after,
        "aggregate_required_counts": aggregate_required,
        "authorized_inventory_before": authorized_before,
        "authorized_inventory_after": authorized_after,
        "errors": errors,
    }
