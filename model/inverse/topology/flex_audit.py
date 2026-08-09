from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Sequence, Tuple

from ...shared.inverse_common import _NodeV3
from .aromatic_clusters import generate_aromatic_clusters
from .classification.classification_models import TopologyClassificationResult
from .iteration_ledger import required_counts_from_slots


SUPPLEMENTAL_FLEX_CANDIDATE_KINDS = {"b_like_open", "d_like_open"}


def _count_clusters_from_generator(
    su_counts: Dict[int, int],
    bridgehead_info: Tuple[int, ...],
) -> Tuple[int, str, str]:
    clusters = generate_aromatic_clusters(
        dict(su_counts or {}),
        bridgehead_info=tuple(bridgehead_info or ()),
    )
    return int(len(clusters)), "topology_aromatic_cluster_generator", ""


def _derive_bridgehead_info_for_flex_window(
    nodes: Sequence[_NodeV3],
) -> Tuple[int, int, int, int, int, int, int]:
    x = y = z = m = n = p = q = 0
    any_hop2 = False
    node_list = list(nodes or [])
    nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in node_list
    }
    for node in node_list:
        try:
            if int(getattr(node, "su_type", -1)) != 12:
                continue
        except Exception:
            continue
        node_id = int(getattr(node, "global_id", -1))
        hop1_ids = [int(value) for value in list(getattr(node, "hop1_ids", []) or [])]
        hop1_su = dict(getattr(node, "hop1_su", {}) or {})
        hop1_12 = int(hop1_su.get(12, 0) or 0)
        hop2 = list(getattr(node, "hop2", []) or [])
        if not hop2:
            hop2_counter: Counter[int] = Counter()
            for neighbor_id in hop1_ids:
                neighbor = nodes_by_id.get(int(neighbor_id))
                if neighbor is None:
                    continue
                for hop2_id in list(getattr(neighbor, "hop1_ids", []) or []):
                    if int(hop2_id) == node_id:
                        continue
                    hop2_node = nodes_by_id.get(int(hop2_id))
                    if hop2_node is not None:
                        hop2_counter[int(getattr(hop2_node, "su_type", -1))] += 1
            hop2 = [
                int(su)
                for su, count in hop2_counter.items()
                for _ in range(int(count))
            ]
        if hop2:
            any_hop2 = True
        hop2_12 = int(sum(1 for su in hop2 if int(su) == 12))
        if len(hop1_ids) == 3 and hop1_12 == 3:
            x += 1
        elif hop1_12 == 2 and hop2_12 >= 3:
            y += 1
        elif hop1_12 == 2 and hop2_12 == 2:
            z += 1
        elif hop2_12 >= 2:
            m += 1
        elif hop1_12 == 2 and hop2_12 == 1:
            n += 1
        elif hop2_12 == 1:
            p += 1
        else:
            q += 1
    if not any_hop2:
        q = sum(1 for node in node_list if int(getattr(node, "su_type", -1)) == 12)
        x = y = z = m = n = p = 0
    return int(x), int(y), int(z), int(m), int(n), int(p), int(q)


def _fragment_flex_reconcile_profile(fragment: Any) -> Dict[str, Any]:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    try:
        priority = int(evidence.get("flex_reconcile_priority", 99) or 99)
    except Exception:
        priority = 99
    mainline_resource_template = {
        str(key): int(value)
        for key, value in required_counts_from_slots(
            list(getattr(fragment, "open_mainline_slots", []) or [])
        ).items()
        if int(value) > 0
    }
    branch_resource_template = {
        str(key): int(value)
        for key, value in required_counts_from_slots(
            list(getattr(fragment, "branch_contracts", []) or [])
        ).items()
        if int(value) > 0
    }
    total_resource_template = Counter(mainline_resource_template)
    total_resource_template.update(branch_resource_template)
    return {
        "fragment_id": int(getattr(fragment, "fragment_id", -1)),
        "class_type": str(getattr(fragment, "class_type", "") or ""),
        "main_kind": str(evidence.get("flex_reconcile_main_kind", "") or ""),
        "preferred_product_type": str(
            evidence.get("flex_reconcile_preferred_product_type", "") or ""
        ),
        "roles": [
            str(value)
            for value in list(evidence.get("flex_reconcile_roles", []) or [])
            if str(value)
        ],
        "priority": priority,
        "count_role": str(evidence.get("flex_range_count_role", "") or ""),
        "candidate_kind": str(evidence.get("flex_range_candidate_kind", "") or ""),
        "counts_as_explicit": bool(evidence.get("flex_range_counts_as_explicit", False)),
        "can_supplement": bool(evidence.get("flex_range_can_supplement", False)),
        "can_demote": bool(evidence.get("flex_range_can_demote", False)),
        "mainline_closed": bool(evidence.get("flex_range_mainline_closed", False)),
        "mainline_open": bool(evidence.get("flex_range_mainline_open", False)),
        "mainline_source": str(evidence.get("flex_range_mainline_source", "") or ""),
        "required_mainline_closure_modes": [
            str(value)
            for value in list(
                evidence.get("flex_range_required_mainline_closure_modes", []) or []
            )
            if str(value)
        ],
        "fixed_branch_contract_count": int(
            evidence.get("flex_range_fixed_branch_contract_count", 0) or 0
        ),
        "open_fixed_branch_contract_count": int(
            evidence.get("flex_range_open_fixed_branch_contract_count", 0) or 0
        ),
        "supports_explicit_mainline": bool(
            evidence.get("flex_range_supports_explicit_mainline", False)
        ),
        "supports_supplemental_mainline": bool(
            evidence.get("flex_range_supports_supplemental_mainline", False)
        ),
        "resource_template": dict(total_resource_template),
        "mainline_resource_template": dict(mainline_resource_template),
        "branch_resource_template": dict(branch_resource_template),
        "source_node_ids": [
            int(value) for value in list(getattr(fragment, "source_node_ids", []) or [])
        ],
    }


def _supplemental_flex_candidate_groups(
    profiles: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    b_like = [
        dict(item or {})
        for item in list(profiles or [])
        if str(dict(item or {}).get("candidate_kind", "")) == "b_like_open"
    ]
    d_like = [
        dict(item or {})
        for item in list(profiles or [])
        if str(dict(item or {}).get("candidate_kind", "")) == "d_like_open"
    ]
    groups: List[Dict[str, Any]] = []
    while len(b_like) >= 2:
        pair = [b_like.pop(0), b_like.pop(0)]
        resource_template: Counter[str] = Counter()
        for item in pair:
            resource_template.update(dict(item.get("branch_resource_template", {}) or {}))
        fragment_ids = [int(item.get("fragment_id", -1)) for item in pair]
        groups.append({
            "group_id": "b_pair:" + ":".join(str(value) for value in fragment_ids),
            "strategy": "b_plus_b",
            "fragment_ids": fragment_ids,
            "contribution_count": 1,
            "resource_template": dict(resource_template),
        })
    for item in d_like:
        fragment_id = int(item.get("fragment_id", -1))
        groups.append({
            "group_id": f"d_single:{fragment_id}",
            "strategy": "d_double_closure",
            "fragment_ids": [fragment_id],
            "contribution_count": 1,
            "resource_template": dict(item.get("resource_template", {}) or {}),
        })
    for item in b_like:
        fragment_id = int(item.get("fragment_id", -1))
        groups.append({
            "group_id": f"b_single:{fragment_id}",
            "strategy": "b_single_closure",
            "fragment_ids": [fragment_id],
            "contribution_count": 1,
            "resource_template": dict(item.get("resource_template", {}) or {}),
        })
    return groups


def _profile_counts_as_strict_explicit_flex(profile: Dict[str, Any]) -> bool:
    item = dict(profile or {})
    return bool(
        bool(item.get("counts_as_explicit", False))
        and bool(item.get("supports_explicit_mainline", False))
        and bool(item.get("mainline_closed", False))
        and int(item.get("open_fixed_branch_contract_count", 0) or 0) == 0
        and not dict(item.get("branch_resource_template", {}) or {})
        and str(item.get("preferred_product_type", "") or "") == "flex_chain"
        and str(item.get("candidate_kind", "") or "") == "explicit_flex"
    )


def _compute_flex_audit(
    nodes: Sequence[_NodeV3],
    classification: TopologyClassificationResult,
    *,
    enforce: bool = True,
) -> Dict[str, Any]:
    counts: Counter[int] = Counter()
    node_ids_by_su: Dict[int, List[int]] = {}
    for node in list(nodes or []):
        try:
            su_type = int(getattr(node, "su_type", -1))
            node_id = int(getattr(node, "global_id", -1))
        except Exception:
            continue
        counts[su_type] += 1
        node_ids_by_su.setdefault(su_type, []).append(node_id)
    for ids in node_ids_by_su.values():
        ids.sort()

    bridgehead_info = _derive_bridgehead_info_for_flex_window(nodes)
    cluster_count_before_trim, cluster_source_before_trim, cluster_error_before_trim = (
        _count_clusters_from_generator(dict(counts), bridgehead_info)
    )
    cluster_count = int(cluster_count_before_trim)
    source = str(cluster_source_before_trim)
    error = str(cluster_error_before_trim)
    rigid10_pairs = max(0, int(counts.get(10, 0)) // 2)
    y_value = rigid10_pairs
    z_value = cluster_count - y_value
    flex_lower = z_value + 1
    flex_upper_raw = int(math.floor(0.8 * float(max(0, cluster_count))))
    window_ok = flex_lower <= flex_upper_raw
    window_applicable = cluster_count > 0

    flex_profiles = [
        _fragment_flex_reconcile_profile(fragment)
        for fragment in list(getattr(classification, "fragments", []) or [])
    ]
    hard_explicit_ids = sorted({
        int(profile["fragment_id"])
        for profile in flex_profiles
        if _profile_counts_as_strict_explicit_flex(profile)
    })
    supplemental_candidate_profiles = sorted(
        [
            {
                "fragment_id": int(profile.get("fragment_id", -1)),
                "class_type": str(profile.get("class_type", "")),
                "main_kind": str(profile.get("main_kind", "")),
                "preferred_product_type": str(profile.get("preferred_product_type", "")),
                "roles": [str(value) for value in list(profile.get("roles", []) or []) if str(value)],
                "priority": int(profile.get("priority", 99) or 99),
                "candidate_kind": str(profile.get("candidate_kind", "") or ""),
                "count_role": str(profile.get("count_role", "") or ""),
                "resource_template": {
                    str(key): int(value)
                    for key, value in dict(profile.get("resource_template", {}) or {}).items()
                    if int(value) > 0
                },
                "mainline_resource_template": {
                    str(key): int(value)
                    for key, value in dict(profile.get("mainline_resource_template", {}) or {}).items()
                    if int(value) > 0
                },
                "branch_resource_template": {
                    str(key): int(value)
                    for key, value in dict(profile.get("branch_resource_template", {}) or {}).items()
                    if int(value) > 0
                },
                "source_node_ids": [
                    int(value) for value in list(profile.get("source_node_ids", []) or [])
                ],
            }
            for profile in flex_profiles
            if bool(profile.get("can_supplement", False))
            and str(profile.get("candidate_kind", "") or "")
            in SUPPLEMENTAL_FLEX_CANDIDATE_KINDS
        ],
        key=lambda item: (int(item.get("priority", 99) or 99), int(item.get("fragment_id", -1))),
    )
    supplemental_candidate_ids = [
        int(profile.get("fragment_id", -1))
        for profile in supplemental_candidate_profiles
        if int(profile.get("fragment_id", -1)) >= 0
    ]
    supplemental_candidate_groups = _supplemental_flex_candidate_groups(
        supplemental_candidate_profiles
    )
    explicit_flex_count = len(hard_explicit_ids)
    if not enforce:
        scenario = "in_range"
    elif not window_applicable:
        scenario = "not_applicable"
    elif explicit_flex_count < flex_lower:
        scenario = "below_range"
    elif explicit_flex_count > flex_upper_raw:
        scenario = "above_range"
    else:
        scenario = "in_range"
    flex_target = flex_lower if window_ok else explicit_flex_count
    target_flex_count = (
        explicit_flex_count
        if scenario != "below_range"
        else max(flex_lower, min(flex_upper_raw, flex_target))
    )
    selected_profiles: List[Dict[str, Any]] = []
    selected_groups: List[Dict[str, Any]] = []
    blocked_reason = ""
    if enforce and window_applicable and not window_ok:
        blocked_reason = "flex_window_invalid"
        target_flex_count = explicit_flex_count
    if scenario == "below_range" and blocked_reason != "flex_window_invalid":
        selected_groups = supplemental_candidate_groups[
            :max(0, target_flex_count - explicit_flex_count)
        ]
        selected_group_fragment_ids = [
            int(fragment_id)
            for group in selected_groups
            for fragment_id in list(group.get("fragment_ids", []) or [])
        ]
        profile_by_id = {
            int(item.get("fragment_id", -1)): dict(item)
            for item in supplemental_candidate_profiles
        }
        selected_profiles = [
            profile_by_id[fragment_id]
            for fragment_id in selected_group_fragment_ids
            if fragment_id in profile_by_id
        ]

    selected_fragment_ids = [int(item.get("fragment_id", -1)) for item in selected_profiles]
    minimum_selected_group_count = min(
        len(selected_groups),
        max(0, flex_lower - explicit_flex_count),
    )
    hard_selected_fragment_ids = [
        int(fragment_id)
        for group in selected_groups[:minimum_selected_group_count]
        for fragment_id in list(group.get("fragment_ids", []) or [])
    ]
    preferred_selected_fragment_ids = [
        int(fragment_id)
        for group in selected_groups[minimum_selected_group_count:]
        for fragment_id in list(group.get("fragment_ids", []) or [])
    ]
    selected_node_ids = sorted({
        int(node_id)
        for item in selected_profiles
        for node_id in list(item.get("source_node_ids", []) or [])
    })
    required_resource_counts: Counter[str] = Counter()
    for item in selected_groups:
        required_resource_counts.update({
            str(key): int(value)
            for key, value in dict(item.get("resource_template", {}) or {}).items()
            if int(value) > 0
        })

    return {
        "X": cluster_count,
        "X_before_trim": int(cluster_count_before_trim),
        "X_after_trim": cluster_count,
        "Y": y_value,
        "Z": z_value,
        "flex_lower": flex_lower,
        "flex_upper": flex_upper_raw,
        "flex_upper_raw": flex_upper_raw,
        "flex_upper_after_window_raise": flex_upper_raw,
        "flex_window_ok": bool(window_ok),
        "flex_window_applicable": bool(window_applicable),
        "flex_window_source": source,
        "flex_window_error": error,
        "flex_window_source_before_trim": str(cluster_source_before_trim),
        "flex_window_error_before_trim": str(cluster_error_before_trim),
        "flex_window_source_after_trim": source,
        "flex_window_error_after_trim": error,
        "bridgehead_info": tuple(int(value) for value in bridgehead_info),
        "histogram_su12_count": int(counts.get(12, 0)),
        "explicit_flex_count": explicit_flex_count,
        "explicit_flex_count_raw": explicit_flex_count,
        "hard_explicit_flex_fragment_ids": hard_explicit_ids,
        "supplemental_flex_candidate_ids": supplemental_candidate_ids,
        "supplemental_flex_candidate_profiles": supplemental_candidate_profiles,
        "supplemental_flex_candidate_groups": supplemental_candidate_groups,
        "selected_candidate_fragment_ids": selected_fragment_ids,
        "selected_candidate_groups": selected_groups,
        "hard_selected_candidate_fragment_ids": hard_selected_fragment_ids,
        "preferred_selected_candidate_fragment_ids": preferred_selected_fragment_ids,
        "selected_candidate_node_ids": selected_node_ids,
        "required_resource_counts": dict(required_resource_counts),
        "unresolved_resource_counts": dict(required_resource_counts) if blocked_reason else {},
        "flex_profiles": flex_profiles,
        "rigid10_pairs_before": rigid10_pairs,
        "rigid10_pairs_after": rigid10_pairs,
        "scenario": scenario,
        "target_flex_count": target_flex_count,
        "flex_target": flex_target,
        "minimum_flex_gap": max(0, flex_lower - explicit_flex_count),
        "preferred_flex_gap": max(0, target_flex_count - explicit_flex_count),
        "maximum_flex_capacity": max(0, flex_upper_raw - explicit_flex_count),
        "external_candidate_capacity": len(supplemental_candidate_groups),
        "external_planned_addition": len(selected_groups),
        "external_capacity_exhausted": bool(
            len(supplemental_candidate_groups)
            < max(0, target_flex_count - explicit_flex_count)
        ),
        "above_range_excess": max(0, explicit_flex_count - flex_upper_raw),
        "blocked_reason": blocked_reason,
        "candidate_node_ids_by_su": {
            str(key): [int(value) for value in values]
            for key, values in sorted(node_ids_by_su.items())
        },
        "enforced": bool(enforce and window_applicable),
    }


def _flex_bounds_from_audit(flex_audit: Dict[str, Any]) -> Dict[str, Any]:
    audit = dict(flex_audit or {})
    flex_min = int(audit.get("flex_lower", 0) or 0)
    flex_max = int(audit.get("flex_upper", 0) or 0)
    flex_target = int(audit.get("target_flex_count", 0) or 0)
    if flex_min <= flex_max:
        flex_target = max(flex_min, min(flex_max, flex_target))
    return {
        "min": flex_min,
        "max": flex_max,
        "target": flex_target,
        "preferred_target": int(audit.get("flex_target", flex_target) or flex_target),
        "enforced": bool(audit.get("enforced", False)),
        "window_source": "flex_audit",
        "audit_X": int(audit.get("X", 0) or 0),
        "audit_Y": int(audit.get("Y", 0) or 0),
        "audit_Z": int(audit.get("Z", 0) or 0),
    }


def _flex_precheck_required(flex_audit: Dict[str, Any]) -> bool:
    audit = dict(flex_audit or {})
    return bool(
        str(audit.get("scenario", "") or "") == "above_range"
        or str(audit.get("blocked_reason", "") or "") == "flex_window_invalid"
    )
