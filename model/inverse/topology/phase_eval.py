from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...shared.inverse_common import _NodeV3
from .aromatic_clusters import generate_aromatic_clusters
from .classification.classifier import classify_topology_units
from .classification.classification_models import TopologyClassificationResult
from .round_context import build_stage3_round_context, ownership_public_view
from .stage3_models import (
    Stage3FlexDecision,
    Stage3PhaseEvaluation,
    TopologyAssemblyResult,
    merge_stage3_contracts_by_owner,
    stage3_contract_owner_key,
)


PHASE_SEQUENCE = (
    "flex_range_reconciliation",
    "external_flex_closure",
    "external_fatty_ring_closure",
    "external_side_chain_closure",
    "remaining_pool_spectrum_rebalance",
    "resource_su24_ring_closure",
    "residual_resource_self_closure",
    "complete",
)

SUPPLEMENTAL_FLEX_CANDIDATE_KINDS = {"b_like_open", "d_like_open"}

EXTERNAL_SUBPHASE_ORDER = {
    "flex_range_reconciliation": 0,
    "external_flex_closure": 1,
    "external_fatty_ring_closure": 2,
    "external_side_chain_closure": 3,
    "remaining_pool_spectrum_rebalance": 4,
    "resource_su24_ring_closure": 5,
    "residual_resource_self_closure": 6,
    "complete": 7,
}


def build_stage3_topology_metrics(
    classification: Any,
    assembly: Any,
    flex_decision: Any,
) -> Dict[str, Any]:
    classification_dict = classification.to_dict() if hasattr(classification, "to_dict") else dict(classification or {})
    assembly_dict = assembly.to_dict() if hasattr(assembly, "to_dict") else dict(assembly or {})
    decision_dict = flex_decision.to_dict() if hasattr(flex_decision, "to_dict") else dict(flex_decision or {})
    diagnostics = dict(assembly_dict.get("diagnostics", {}) or {})
    ledger = dict(assembly_dict.get("resource_ledger", {}) or {})
    shortage = {
        str(key): int(value)
        for key, value in dict(ledger.get("shortage_counts", {}) or {}).items()
        if int(value) > 0
    }
    unresolved = {
        str(key): int(value)
        for key, value in dict(assembly_dict.get("unresolved_counts", shortage) or {}).items()
        if int(value) > 0
    }
    scenario = str(decision_dict.get("scenario", "in_range") or "in_range")
    effective = int(decision_dict.get("effective_flex_count", 0) or 0)
    lower = int(decision_dict.get("lower_bound", 0) or 0)
    upper = int(decision_dict.get("upper_bound", 0) or 0)
    flex_gap = max(0, lower - effective) if scenario == "below_range" else max(0, effective - upper)
    active_subphase = str(
        assembly_dict.get("active_phase")
        or assembly_dict.get("active_subphase")
        or diagnostics.get("active_external_subphase")
        or diagnostics.get("stage3_phase")
        or "flex_range_reconciliation"
    )
    products = list(assembly_dict.get("products", []) or [])
    return {
        "active_subphase": active_subphase,
        "subphase_rank": int(EXTERNAL_SUBPHASE_ORDER.get(active_subphase, 0)),
        "product_count": int(len(products)),
        "product_types": sorted(str(dict(item or {}).get("product_type", "")) for item in products),
        "pending_count": int(len(list(assembly_dict.get("pending_units", []) or []))),
        "shortage_total": int(sum(shortage.values())),
        "unresolved_total": int(sum(unresolved.values())),
        "flex_gap": int(flex_gap),
        "unclassified_count": int(len(list(classification_dict.get("unclassified_nodes", []) or []))),
        "reject_count": int(len(list(classification_dict.get("classification_rejects", []) or []))),
        "consumed_counts": dict(ledger.get("consumed_counts", {}) or {}),
        "remaining_counts": dict(ledger.get("remaining_counts", {}) or {}),
    }


def build_stage3_topology_signature(
    classification: Any,
    assembly: Any,
    flex_decision: Any,
    H: Any = None,
) -> str:
    classification_dict = classification.to_dict() if hasattr(classification, "to_dict") else dict(classification or {})
    assembly_dict = assembly.to_dict() if hasattr(assembly, "to_dict") else dict(assembly or {})
    decision_dict = flex_decision.to_dict() if hasattr(flex_decision, "to_dict") else dict(flex_decision or {})
    histogram = []
    if H is not None:
        histogram = [int(value) for value in H.detach().cpu().long().tolist()]
    payload = {
        "fragments": sorted(
            (
                str(dict(item or {}).get("class_type", "")),
                tuple(sorted(int(value) for value in list(dict(item or {}).get("claimed_node_ids", []) or []))),
            )
            for item in list(classification_dict.get("fragments", []) or [])
        ),
        "flex_decision": decision_dict,
        "active_phase": str(assembly_dict.get("active_phase", "") or dict(assembly_dict.get("diagnostics", {}) or {}).get("stage3_phase", "")),
        "active_subphase": str(assembly_dict.get("active_subphase", "") or dict(assembly_dict.get("diagnostics", {}) or {}).get("active_external_subphase", "")),
        "products": list(assembly_dict.get("products", []) or []),
        "pending_units": list(assembly_dict.get("pending_units", []) or []),
        "resource_ledger": dict(assembly_dict.get("resource_ledger", {}) or {}),
        "requested_counts": dict(assembly_dict.get("requested_counts", {}) or {}),
        "unresolved_counts": dict(assembly_dict.get("unresolved_counts", {}) or {}),
        "H": histogram,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_clusters_from_generator(
    su_counts: Dict[int, int],
    bridgehead_info: Tuple[int, ...],
) -> Tuple[int, str, str]:
    clusters = generate_aromatic_clusters(
        dict(su_counts or {}),
        bridgehead_info=tuple(bridgehead_info or ()),
    )
    return int(len(clusters)), "topology_aromatic_cluster_generator", ""


def _derive_bridgehead_info_for_flex_window(nodes: Sequence[_NodeV3]) -> Tuple[int, int, int, int, int, int, int]:
    x = y = z = m = n = p = q = 0
    any_hop2 = False
    for node in list(nodes or []):
        try:
            if int(getattr(node, "su_type", -1)) != 12:
                continue
        except Exception:
            continue
        hop1_ids = [int(value) for value in list(getattr(node, "hop1_ids", []) or [])]
        hop1_su = dict(getattr(node, "hop1_su", {}) or {})
        hop1_12 = int(hop1_su.get(12, 0) or 0)
        hop2 = list(getattr(node, "hop2", []) or [])
        if not hop2:
            hop2_counter = Counter()
            for neighbor_id in hop1_ids:
                neighbor = next(
                    (candidate for candidate in list(nodes or []) if int(getattr(candidate, "global_id", -1)) == int(neighbor_id)),
                    None,
                )
                if neighbor is None:
                    continue
                for hop2_id in list(getattr(neighbor, "hop1_ids", []) or []):
                    if int(hop2_id) == int(getattr(node, "global_id", -1)):
                        continue
                    hop2_node = next(
                        (candidate for candidate in list(nodes or []) if int(getattr(candidate, "global_id", -1)) == int(hop2_id)),
                        None,
                    )
                    if hop2_node is None:
                        continue
                    hop2_counter[int(getattr(hop2_node, "su_type", -1))] += 1
            hop2 = [int(su) for su, count in hop2_counter.items() for _ in range(int(count))]
        if hop2:
            any_hop2 = True
        hop2_12 = int(sum(1 for su in hop2 if int(su) == 12))
        if len(hop1_ids) == 3 and int(hop1_12) == 3:
            x += 1
        elif int(hop1_12) == 2 and int(hop2_12) >= 3:
            y += 1
        elif int(hop1_12) == 2 and int(hop2_12) == 2:
            z += 1
        elif int(hop2_12) >= 2:
            m += 1
        elif int(hop1_12) == 2 and int(hop2_12) == 1:
            n += 1
        elif int(hop2_12) == 1:
            p += 1
        else:
            q += 1
    if not any_hop2:
        q = int(sum(1 for node in list(nodes or []) if int(getattr(node, "su_type", -1)) == 12))
        x = y = z = m = n = p = 0
    return int(x), int(y), int(z), int(m), int(n), int(p), int(q)


def _fragment_flex_reconcile_profile(fragment: Any) -> Dict[str, Any]:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    try:
        priority = int(evidence.get("flex_reconcile_priority", 99) or 99)
    except Exception:
        priority = 99
    return {
        "fragment_id": int(getattr(fragment, "fragment_id", -1)),
        "class_type": str(getattr(fragment, "class_type", "") or ""),
        "main_kind": str(evidence.get("flex_reconcile_main_kind", "") or ""),
        "preferred_product_type": str(evidence.get("flex_reconcile_preferred_product_type", "") or ""),
        "roles": [
            str(value)
            for value in list(evidence.get("flex_reconcile_roles", []) or [])
            if str(value)
        ],
        "priority": int(priority),
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
            for value in list(evidence.get("flex_range_required_mainline_closure_modes", []) or [])
            if str(value)
        ],
        "fixed_branch_contract_count": int(evidence.get("flex_range_fixed_branch_contract_count", 0) or 0),
        "open_fixed_branch_contract_count": int(evidence.get("flex_range_open_fixed_branch_contract_count", 0) or 0),
        "supports_explicit_mainline": bool(evidence.get("flex_range_supports_explicit_mainline", False)),
        "supports_supplemental_mainline": bool(evidence.get("flex_range_supports_supplemental_mainline", False)),
        "resource_template": {
            str(key): int(value)
            for key, value in dict(evidence.get("flex_range_resource_template", {}) or {}).items()
            if int(value) > 0
        },
        "source_node_ids": [
            int(value)
            for value in list(getattr(fragment, "source_node_ids", []) or [])
        ],
    }


def _profile_counts_as_strict_explicit_flex(profile: Dict[str, Any]) -> bool:
    item = dict(profile or {})
    return bool(
        bool(item.get("counts_as_explicit", False))
        and bool(item.get("supports_explicit_mainline", False))
        and bool(item.get("mainline_closed", False))
        and str(item.get("preferred_product_type", "") or "") == "flex_chain"
        and str(item.get("candidate_kind", "") or "") == "explicit_flex"
    )


def _compute_flex_audit(
    nodes: Sequence[_NodeV3],
    classification: TopologyClassificationResult,
    *,
    enforce: bool = True,
) -> Dict[str, Any]:
    counts = Counter()
    node_ids_by_su: Dict[int, List[int]] = {}
    for node in list(nodes or []):
        try:
            su_type = int(getattr(node, "su_type", -1))
            node_id = int(getattr(node, "global_id", -1))
        except Exception:
            continue
        counts[su_type] += 1
        node_ids_by_su.setdefault(int(su_type), []).append(int(node_id))
    for ids in node_ids_by_su.values():
        ids.sort()

    bridgehead_info = _derive_bridgehead_info_for_flex_window(nodes)
    cluster_count_before_trim, cluster_source_before_trim, cluster_error_before_trim = _count_clusters_from_generator(
        dict(counts),
        bridgehead_info,
    )

    cluster_count = int(cluster_count_before_trim)
    source = str(cluster_source_before_trim)
    error = str(cluster_error_before_trim)
    rigid10_pairs = max(0, int(counts.get(10, 0)) // 2)
    y_value = int(rigid10_pairs)
    z_value = int(cluster_count) - int(y_value)
    flex_lower = int(z_value + 1)
    flex_upper_raw = int(math.floor(0.8 * float(max(0, int(cluster_count)))))
    window_ok = bool(int(flex_lower) <= int(flex_upper_raw))
    window_applicable = bool(int(cluster_count) > 0)

    fragments = list(getattr(classification, "fragments", []) or [])
    flex_profiles = [
        _fragment_flex_reconcile_profile(fragment)
        for fragment in fragments
    ]
    hard_explicit_ids = sorted({
        int(profile["fragment_id"])
        for profile in list(flex_profiles or [])
        if _profile_counts_as_strict_explicit_flex(dict(profile or {}))
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
                "source_node_ids": [
                    int(value)
                    for value in list(profile.get("source_node_ids", []) or [])
                ],
            }
            for profile in list(flex_profiles or [])
            if (
                bool(profile.get("can_supplement", False))
                and str(profile.get("candidate_kind", "") or "") in SUPPLEMENTAL_FLEX_CANDIDATE_KINDS
            )
        ],
        key=lambda item: (int(item.get("priority", 99) or 99), int(item.get("fragment_id", -1))),
    )
    supplemental_candidate_ids = [
        int(profile.get("fragment_id", -1))
        for profile in list(supplemental_candidate_profiles or [])
        if int(profile.get("fragment_id", -1)) >= 0
    ]
    explicit_flex_count_raw = int(len(hard_explicit_ids))

    flex_upper = int(flex_upper_raw)
    explicit_flex_count = int(explicit_flex_count_raw)
    if not bool(enforce):
        scenario = "in_range"
    elif not bool(window_applicable):
        scenario = "not_applicable"
    elif int(explicit_flex_count) < int(flex_lower):
        scenario = "below_range"
    elif int(explicit_flex_count) > int(flex_upper):
        scenario = "above_range"
    else:
        scenario = "in_range"
    flex_target = int(flex_lower) if window_ok else int(explicit_flex_count)
    target_flex_count = int(
        explicit_flex_count if scenario != "below_range"
        else max(int(flex_lower), min(int(flex_upper), int(flex_target)))
    )
    selected_profiles: List[Dict[str, Any]] = []
    blocked_reason = ""
    if not bool(enforce):
        blocked_reason = ""
    elif not bool(window_applicable):
        blocked_reason = ""
    elif not bool(window_ok):
        blocked_reason = "flex_window_invalid"
        target_flex_count = int(explicit_flex_count)
    if str(scenario) == "below_range" and str(blocked_reason) != "flex_window_invalid":
        selected_profiles = list(supplemental_candidate_profiles or [])[
            :max(0, int(target_flex_count) - int(explicit_flex_count))
        ]
        # Candidate exhaustion is not a pre-assembly blocker. The remaining
        # minimum/preferred gap is carried into residual planning.

    selected_fragment_ids = [int(item.get("fragment_id", -1)) for item in selected_profiles]
    minimum_selected_count = min(
        len(selected_profiles),
        max(0, int(flex_lower) - int(explicit_flex_count)),
    )
    hard_selected_fragment_ids = [
        int(item.get("fragment_id", -1))
        for item in selected_profiles[:minimum_selected_count]
    ]
    preferred_selected_fragment_ids = [
        int(item.get("fragment_id", -1))
        for item in selected_profiles[minimum_selected_count:]
    ]
    selected_node_ids = sorted({
        int(node_id)
        for item in selected_profiles
        for node_id in list(item.get("source_node_ids", []) or [])
    })
    required_resource_counts: Counter = Counter()
    for item in selected_profiles:
        required_resource_counts.update({
            str(key): int(value)
            for key, value in dict(item.get("resource_template", {}) or {}).items()
            if int(value) > 0
        })

    return {
        "X": int(cluster_count),
        "X_before_trim": int(cluster_count_before_trim),
        "X_after_trim": int(cluster_count),
        "Y": int(y_value),
        "Z": int(z_value),
        "flex_lower": int(flex_lower),
        "flex_upper": int(flex_upper_raw),
        "flex_upper_raw": int(flex_upper_raw),
        "flex_upper_after_window_raise": int(flex_upper_raw),
        "flex_window_ok": bool(window_ok),
        "flex_window_applicable": bool(window_applicable),
        "flex_window_source": str(source),
        "flex_window_error": str(error),
        "flex_window_source_before_trim": str(cluster_source_before_trim),
        "flex_window_error_before_trim": str(cluster_error_before_trim),
        "flex_window_source_after_trim": str(source),
        "flex_window_error_after_trim": str(error),
        "bridgehead_info": tuple(int(x) for x in bridgehead_info),
        "explicit_flex_count": int(explicit_flex_count),
        "explicit_flex_count_raw": int(explicit_flex_count_raw),
        "hard_explicit_flex_fragment_ids": list(hard_explicit_ids),
        "supplemental_flex_candidate_ids": list(supplemental_candidate_ids),
        "supplemental_flex_candidate_profiles": list(supplemental_candidate_profiles),
        "selected_candidate_fragment_ids": list(selected_fragment_ids),
        "hard_selected_candidate_fragment_ids": list(hard_selected_fragment_ids),
        "preferred_selected_candidate_fragment_ids": list(preferred_selected_fragment_ids),
        "selected_candidate_node_ids": list(selected_node_ids),
        "required_resource_counts": dict(required_resource_counts),
        "unresolved_resource_counts": dict(required_resource_counts) if blocked_reason else {},
        "flex_profiles": list(flex_profiles),
        "rigid10_pairs_before": int(rigid10_pairs),
        "rigid10_pairs_after": int(rigid10_pairs),
        "scenario": str(scenario),
        "target_flex_count": int(target_flex_count),
        "flex_target": int(flex_target),
        "minimum_flex_gap": max(0, int(flex_lower) - int(explicit_flex_count)),
        "preferred_flex_gap": max(0, int(target_flex_count) - int(explicit_flex_count)),
        "maximum_flex_capacity": max(0, int(flex_upper) - int(explicit_flex_count)),
        "external_candidate_capacity": int(len(supplemental_candidate_profiles)),
        "external_planned_addition": int(len(selected_profiles)),
        "external_capacity_exhausted": bool(
            int(len(supplemental_candidate_profiles)) < max(0, int(target_flex_count) - int(explicit_flex_count))
        ),
        "above_range_excess": max(0, int(explicit_flex_count) - int(flex_upper)),
        "blocked_reason": str(blocked_reason),
        "candidate_node_ids_by_su": {
            str(key): [int(value) for value in list(values or [])]
            for key, values in sorted(node_ids_by_su.items(), key=lambda item: int(item[0]))
        },
        "enforced": bool(enforce and window_applicable),
    }


def _flex_bounds_from_audit(flex_audit: Dict[str, Any]) -> Dict[str, Any]:
    audit = dict(flex_audit or {})
    flex_min = int(audit.get("flex_lower", 0) or 0)
    flex_max = int(audit.get("flex_upper", 0) or 0)
    flex_target = int(audit.get("target_flex_count", 0) or 0)
    if int(flex_min) <= int(flex_max):
        flex_target = max(int(flex_min), min(int(flex_max), int(flex_target)))
    return {
        "min": int(flex_min),
        "max": int(flex_max),
        "target": int(flex_target),
        "preferred_target": int(audit.get("flex_target", flex_target) or flex_target),
        "enforced": bool(audit.get("enforced", False)),
        "window_source": "flex_audit",
        "audit_X": int(audit.get("X", 0) or 0),
        "audit_Y": int(audit.get("Y", 0) or 0),
        "audit_Z": int(audit.get("Z", 0) or 0),
    }


def _flex_precheck_required(flex_audit: Dict[str, Any]) -> bool:
    audit = dict(flex_audit or {})
    scenario = str(audit.get("scenario", "") or "")
    blocked_reason = str(audit.get("blocked_reason", "") or "")
    return bool(
        str(scenario) == "above_range"
        or str(blocked_reason) == "flex_window_invalid"
    )


def _minimal_resource_ledger_from_classification(classification_dict: Dict[str, Any]) -> Dict[str, Any]:
    resource_pool = dict(dict(classification_dict or {}).get("resource_pool", {}) or {})
    remaining_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("remaining_ids_by_su", {}) or {}).items()
    }
    initial_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("initial_ids_by_su", remaining_ids_by_su) or {}).items()
    }
    all_keys = sorted({
        str(key)
        for key in set(initial_ids_by_su.keys()) | set(remaining_ids_by_su.keys())
    }, key=int)
    initial_ids_by_su = {
        str(key): [int(value) for value in list(initial_ids_by_su.get(str(key), []) or [])]
        for key in all_keys
    }
    remaining_ids_by_su = {
        str(key): [int(value) for value in list(remaining_ids_by_su.get(str(key), []) or [])]
        for key in all_keys
    }
    consumed_ids_by_su = {str(key): [] for key in all_keys}
    initial_counts = {str(key): int(len(list(values or []))) for key, values in initial_ids_by_su.items()}
    remaining_counts = {str(key): int(len(list(values or []))) for key, values in remaining_ids_by_su.items()}
    consumed_counts = {str(key): 0 for key in all_keys}
    su11_donor_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("su11_donor_ids_by_su", {}) or {}).items()
    }
    return {
        "initial_ids_by_su": initial_ids_by_su,
        "remaining_ids_by_su": remaining_ids_by_su,
        "consumed_ids_by_su": consumed_ids_by_su,
        "initial_counts": initial_counts,
        "remaining_counts": remaining_counts,
        "consumed_counts": consumed_counts,
        "shortage_counts": {},
        "su11_donor_ids_by_su": su11_donor_ids_by_su,
        "su11_donor_counts": {
            str(key): int(len(values or []))
            for key, values in su11_donor_ids_by_su.items()
        },
    }


def _preassembly_pending_units(
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    reason = str(blocked_reason or ("flex_range_reconciliation_pending" if scenario else "flex_range_reconciliation"))
    hard_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
    }
    supplemental_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
    }
    pending_ids: List[int] = []
    pending_kind_by_id: Dict[int, str] = {}
    if str(blocked_reason) == "flex_window_invalid":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
        for fragment_id in sorted(supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = "supplemental_flex_candidate"
            pending_ids.append(int(fragment_id))
    elif str(scenario) == "below_range":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
        for fragment_id in sorted(supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = "supplemental_flex_candidate"
            pending_ids.append(int(fragment_id))
    elif str(scenario) == "above_range":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
    else:
        for fragment_id in sorted(hard_ids | supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = (
                "explicit_flex" if int(fragment_id) in hard_ids else "supplemental_flex_candidate"
            )
            pending_ids.append(int(fragment_id))
    fragments_by_id = {
        int(getattr(fragment, "fragment_id", -1)): fragment
        for fragment in list(getattr(classification, "fragments", []) or [])
    }
    pending_units: List[Dict[str, Any]] = []
    for fragment_id in pending_ids:
        fragment = fragments_by_id.get(int(fragment_id))
        if fragment is None:
            continue
        class_type = str(getattr(fragment, "class_type", "") or "")
        pending_kind = str(pending_kind_by_id.get(int(fragment_id), "supplemental_flex_candidate"))
        pending_units.append({
            "unit_id": int(fragment_id),
            "pending_kind": str(pending_kind),
            "real_topology_partial": str(getattr(fragment, "topology_template", "") or class_type),
            "real_node_partial": "",
            "source_fragment_ids": [int(fragment_id)],
            "source_fragment_classes": [str(class_type)],
            "source_node_ids": [int(value) for value in list(getattr(fragment, "source_node_ids", []) or [])],
            "reason": str(reason),
        })
    return pending_units


def _flex_reconciliation_contracts(
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    explicit_count = int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0)
    target_count = int(dict(flex_audit or {}).get("target_flex_count", 0) or 0)
    flex_gap = max(0, int(target_count) - int(explicit_count))
    contracts: List[Dict[str, Any]] = []
    if str(blocked_reason) == "flex_window_invalid":
        contracts.append({
            "contract_id": -1000,
            "contract_kind": "structural_blocker",
            "target_role": "flex_chain",
            "source_fragment_ids": [],
            "source_fragment_classes": [],
            "claimed_source_node_ids": [],
            "context_node_ids": [],
            "mainline_slots": [],
            "branch_slots": [],
            "required_counts": {},
            "expected_topology": "",
            "reason": "flex_window_invalid",
            "blocker_kind": "flex_window_invalid",
            "diagnostics": {
                "scenario": str(scenario or "in_range"),
                "X": int(dict(flex_audit or {}).get("X", 0) or 0),
                "Y": int(dict(flex_audit or {}).get("Y", 0) or 0),
                "Z": int(dict(flex_audit or {}).get("Z", 0) or 0),
                "flex_lower": int(dict(flex_audit or {}).get("flex_lower", 0) or 0),
                "flex_upper": int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
            },
        })
    if scenario == "below_range":
        fragments_by_id = {
            int(getattr(fragment, "fragment_id", -1)): fragment
            for fragment in list(getattr(classification, "fragments", []) or [])
        }
        candidate_profiles = [
            dict(item or {})
            for item in list(dict(flex_audit or {}).get("supplemental_flex_candidate_profiles", []) or [])
            if dict(item or {})
        ]
        selected_ids = {
            int(value)
            for value in list(dict(flex_audit or {}).get("selected_candidate_fragment_ids", []) or [])
        }
        for item in list(candidate_profiles or []):
            fragment_id = int(item.get("fragment_id", -1))
            if fragment_id < 0:
                continue
            fragment = fragments_by_id.get(int(fragment_id))
            required_counts = (
                {
                    str(key): int(value)
                    for key, value in dict(item.get("resource_template", {}) or {}).items()
                    if int(value) > 0
                }
                if int(fragment_id) in selected_ids else {}
            )
            contracts.append({
                "contract_id": int(fragment_id),
                "contract_kind": "mainline_closure" if required_counts else "structural_blocker",
                "target_role": "flex_chain",
                "source_fragment_ids": [int(fragment_id)],
                "source_fragment_classes": [str(getattr(fragment, "class_type", item.get("class_type", "")))],
                "claimed_source_node_ids": [
                    int(value)
                    for value in list(getattr(fragment, "claimed_node_ids", []) or item.get("source_node_ids", []) or [])
                ],
                "context_node_ids": [int(value) for value in list(getattr(fragment, "context_node_ids", []) or [])],
                "mainline_slots": [
                    slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
                    for slot in list(getattr(fragment, "open_mainline_slots", []) or [])
                ],
                "branch_slots": [
                    slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
                    for slot in list(getattr(fragment, "branch_contracts", []) or [])
                ],
                "required_counts": dict(required_counts),
                "expected_topology": str(getattr(fragment, "topology_template", "") or ""),
                "reason": "supplemental_flex_candidate" if required_counts else "supplemental_flex_candidate_not_selected",
                "blocker_kind": "" if required_counts else "supplemental_flex_candidate_not_selected",
                "diagnostics": {
                    "scenario": str(scenario),
                    "candidate_main_kind": str(item.get("main_kind", "")),
                    "candidate_count_role": str(item.get("count_role", "")),
                    "candidate_kind": str(item.get("candidate_kind", "")),
                    "candidate_priority": int(item.get("priority", 99) or 99),
                    "selected_for_target": bool(int(fragment_id) in selected_ids),
                },
            })
        if int(flex_gap) > int(len(selected_ids)):
            contracts.append({
                "contract_id": -1001,
                "contract_kind": "structural_blocker",
                "target_role": "flex_chain",
                "source_fragment_ids": [],
                "source_fragment_classes": [],
                "claimed_source_node_ids": [],
                "context_node_ids": [],
                "mainline_slots": [],
                "branch_slots": [],
                "required_counts": {},
                "expected_topology": "",
                "reason": "supplemental_flex_candidate_shortfall",
                "blocker_kind": "supplemental_flex_candidate_shortfall",
                "diagnostics": {
                    "scenario": str(scenario),
                    "flex_gap": int(flex_gap),
                    "selected_candidate_count": int(len(selected_ids)),
                    "uncovered_gap": int(max(0, int(flex_gap) - int(len(selected_ids)))),
                },
            })
        return contracts
    if scenario == "above_range":
        contracts.append({
            "contract_id": -1002,
            "contract_kind": "structural_blocker",
            "target_role": "flex_chain",
            "source_fragment_ids": [],
            "source_fragment_classes": [],
            "claimed_source_node_ids": [],
            "context_node_ids": [],
            "mainline_slots": [],
            "branch_slots": [],
            "required_counts": {},
            "expected_topology": "",
            "reason": "explicit_flex_reduction",
            "blocker_kind": "explicit_flex_reduction",
            "diagnostics": {
                "scenario": str(scenario),
                "explicit_count": int(explicit_count),
                "flex_upper": int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
                "flex_excess": int(max(0, int(explicit_count) - int(dict(flex_audit or {}).get("flex_upper", 0) or 0))),
                "allowed_conversion_order": ["12_to_11", "10_pair_to_11"],
            },
        })
        return contracts
    return contracts


def _group_contracts_by_subphase(
    contracts: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for contract in merge_stage3_contracts_by_owner(list(contracts or [])):
        item = dict(contract or {})
        subphase = str(item.get("subphase", "") or _external_phase_for_role(item.get("target_role", "")))
        grouped.setdefault(str(subphase), []).append(item)
    return grouped


def _external_phase_for_role(target_role: str) -> str:
    return {
        "flex_chain": "external_flex_closure",
        "fatty_ring": "external_fatty_ring_closure",
        "side_chain": "external_side_chain_closure",
    }.get(str(target_role or ""), "external_side_chain_closure")


def _preassembly_gate_assembly_dict(
    classification: TopologyClassificationResult,
    classification_dict: Dict[str, Any],
    flex_audit: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    if scenario == "above_range" and str(blocked_reason) != "flex_window_invalid":
        result_reason = "external_flex_range_exceeded"
    else:
        result_reason = "external_flex_range_pending"
    pending_units = _preassembly_pending_units(classification, flex_audit)
    flex_contracts = _flex_reconciliation_contracts(classification, flex_audit)
    contract_counts: Dict[str, int] = {}
    for contract in list(flex_contracts or []):
        required_counts = dict(contract.get("required_counts", {}) or {})
        for key, value in required_counts.items():
            contract_counts[str(key)] = int(contract_counts.get(str(key), 0) or 0) + int(value or 0)
    return {
        "ok": False,
        "reason": str(result_reason),
        "feasible": bool(
            len(list(getattr(classification, "unclassified_nodes", []) or [])) == 0
            and len(list(getattr(classification, "claim_conflicts", []) or [])) == 0
            and len(list(getattr(classification, "classification_rejects", []) or [])) == 0
        ),
        "products": [],
        "pending_units": list(pending_units),
        "flex_bounds": _flex_bounds_from_audit(flex_audit),
        "flex_counts": {
            "explicit_total": int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
            "final": 0,
        },
        "resource_ledger": _minimal_resource_ledger_from_classification(classification_dict),
        "diagnostics": {
            "stage3_phase": "flex_range_reconciliation",
            "external_subphases": [],
            "active_external_subphase": "flex_range_reconciliation",
            "external_subphase_status": {
                "explicit_branch_closure": {"ok": True},
                "real_like24_ring_assembly": {"ok": True},
                "remaining_external_halfclosed_closure": {"ok": True},
            },
            "explicit_flex_count": int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
            "flex_audit": dict(flex_audit or {}),
            "pending_contracts": list(flex_contracts),
            "total_shortage": dict(contract_counts or {}),
        },
        "planned_conversions": [],
    }


def _derive_external_closure_requirements(
    assembly_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diagnostics = dict(dict(assembly_dict or {}).get("diagnostics", {}) or {})
    if str(diagnostics.get("stage3_phase", "") or "") == "flex_range_reconciliation":
        empty = {
            "contracts": [],
            "contracts_by_subphase": {},
            "subphase_status": dict(diagnostics.get("external_subphase_status", {}) or {}),
        }
        return empty, {
            "contracts": [],
            "subphase_status": dict(diagnostics.get("external_subphase_status", {}) or {}),
        }
    pending_contracts = [
        dict(item or {})
        for item in list(diagnostics.get("pending_contracts", []) or [])
        if dict(item or {})
    ]
    contracts: List[Dict[str, Any]] = []
    blockers: Dict[str, Any] = {"contracts": []}
    external_subphase_status = dict(diagnostics.get("external_subphase_status", {}) or {})
    if pending_contracts:
        for contract in list(pending_contracts or []):
            contract_kind = str(contract.get("contract_kind", "") or "")
            blocker_kind = str(contract.get("blocker_kind", "") or "")
            if contract_kind == "structural_blocker" or blocker_kind:
                blockers["contracts"].append(dict(contract or {}))
                continue
            contracts.append(dict(contract or {}))
        contracts = merge_stage3_contracts_by_owner(contracts)
        requirements = {
            "contracts": contracts,
            "contracts_by_subphase": _group_contracts_by_subphase(contracts),
            "subphase_status": external_subphase_status,
        }
        blockers["subphase_status"] = dict(external_subphase_status or {})
        return requirements, blockers
    requirements = {
        "contracts": contracts,
        "contracts_by_subphase": _group_contracts_by_subphase(contracts),
        "subphase_status": external_subphase_status,
    }
    blockers["subphase_status"] = dict(external_subphase_status or {})
    return requirements, blockers


def _contracts_by_contract_kind(
    contracts: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for contract in list(contracts or []):
        item = dict(contract or {})
        contract_kind = str(item.get("contract_kind", "") or "unknown_contract")
        grouped.setdefault(str(contract_kind), []).append(item)
    return grouped


def _canonical_stage3_contracts(
    *,
    flex_contracts: Sequence[Dict[str, Any]],
    assembly_contracts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flex_by_owner = {
        stage3_contract_owner_key(dict(contract or {})): dict(contract or {})
        for contract in list(flex_contracts or [])
        if dict(contract or {})
    }
    canonical: List[Dict[str, Any]] = []
    for raw_contract in list(assembly_contracts or []):
        contract = dict(raw_contract or {})
        if not contract:
            continue
        owner_key = stage3_contract_owner_key(contract)
        flex_contract = flex_by_owner.pop(owner_key, None)
        if flex_contract is None:
            canonical.append(contract)
            continue
        flex_diagnostics = dict(flex_contract.get("diagnostics", {}) or {})
        if (
            str(contract.get("contract_kind", "") or "") == "structural_blocker"
            or bool(str(contract.get("blocker_kind", "") or ""))
        ):
            contract["diagnostics"] = {
                **dict(contract.get("diagnostics", {}) or {}),
                **flex_diagnostics,
            }
            canonical.append(contract)
            continue
        if bool(flex_diagnostics.get("selected_for_target", False)):
            contract["diagnostics"] = {
                **dict(contract.get("diagnostics", {}) or {}),
                **flex_diagnostics,
            }
            canonical.append(contract)
        else:
            canonical.append(flex_contract)
    canonical.extend(flex_by_owner.values())
    return merge_stage3_contracts_by_owner(canonical)


def _requirement_summary_by_phase(
    *,
    flex_contracts: List[Dict[str, Any]],
    external_requirements: Dict[str, Any],
    external_blockers: Dict[str, Any],
) -> Dict[str, Any]:
    external_contracts = [
        dict(item or {})
        for item in list(external_requirements.get("contracts", []) or [])
        if dict(item or {})
    ]
    blocker_contracts = [
        dict(item or {})
        for item in list(external_blockers.get("contracts", []) or [])
        if dict(item or {})
    ]
    contracts_by_subphase = {
        str(key): [
            dict(item or {})
            for item in list(value or [])
            if dict(item or {})
        ]
        for key, value in dict(external_requirements.get("contracts_by_subphase", {}) or {}).items()
    }
    summary = {
        "flex_range_reconciliation": {
            "contract_count": int(len(list(flex_contracts or []))),
            "contracts_by_contract_kind": _contracts_by_contract_kind(list(flex_contracts or [])),
        },
    }
    for target_role in ("flex_chain", "fatty_ring", "side_chain"):
        phase = _external_phase_for_role(target_role)
        phase_contracts = [
            item for item in external_contracts
            if str(item.get("target_role", "") or "") == str(target_role)
        ]
        phase_blockers = [
            item for item in blocker_contracts
            if str(item.get("target_role", "") or "") == str(target_role)
        ]
        summary[str(phase)] = {
            "contract_count": int(len(phase_contracts)),
            "blocker_count": int(len(phase_blockers)),
            "contracts_by_subphase": {
                str(key): list(values)
                for key, values in contracts_by_subphase.items()
                if any(item in phase_contracts for item in values)
            },
            "contracts_by_contract_kind": _contracts_by_contract_kind(phase_contracts + phase_blockers),
        }
    return summary


def _external_phase_structured_views(
    external_requirements: Dict[str, Any],
    external_blockers: Dict[str, Any],
) -> Dict[str, Any]:
    requirements = dict(external_requirements or {})
    blockers = dict(external_blockers or {})
    contracts_by_subphase = {
        str(key): [
            dict(item or {})
            for item in list(value or [])
            if dict(item or {})
        ]
        for key, value in dict(requirements.get("contracts_by_subphase", {}) or {}).items()
    }
    blocker_contracts = [
        dict(item or {})
        for item in list(blockers.get("contracts", []) or [])
        if dict(item or {})
    ]
    blockers_by_kind = _contracts_by_contract_kind(blocker_contracts)
    external_contracts = [
        dict(item or {})
        for item in list(requirements.get("contracts", []) or [])
        if dict(item or {})
    ]
    role_order = {"flex_chain": 0, "fatty_ring": 1, "side_chain": 2}
    active_role = min(
        (str(item.get("target_role", "side_chain") or "side_chain") for item in external_contracts),
        key=lambda role: int(role_order.get(role, 2)),
        default="",
    )
    active_external_subphase = {
        "flex_chain": "external_flex_closure",
        "fatty_ring": "external_fatty_ring_closure",
        "side_chain": "external_side_chain_closure",
    }.get(active_role, "resource_su24_ring_closure")
    return {
        "active_external_subphase": str(active_external_subphase),
        "external_contracts_by_subphase": dict(contracts_by_subphase or {}),
        "external_blockers_by_kind": dict(blockers_by_kind or {}),
        "external_partner_request_summary": {},
        "external_partner_match_requests": {},
    }


def _pending_summary_by_phase(
    *,
    flex_audit: Dict[str, Any],
    pending_units: List[Dict[str, Any]],
    external_pending_by_subphase: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    pending_by_kind: Dict[str, int] = {}
    for item in list(pending_units or []):
        pending_kind = str(dict(item or {}).get("pending_kind", "") or "unknown_pending")
        pending_by_kind[str(pending_kind)] = int(pending_by_kind.get(str(pending_kind), 0) or 0) + 1
    summary = {
        "flex_range_reconciliation": {
            "scenario": str(dict(flex_audit or {}).get("scenario", "") or ""),
            "blocked_reason": str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
            "explicit_fragment_ids": [
                int(value)
                for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
            ],
            "supplemental_candidate_ids": [
                int(value)
                for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
            ],
            "selected_candidate_fragment_ids": [
                int(value)
                for value in list(dict(flex_audit or {}).get("selected_candidate_fragment_ids", []) or [])
            ],
        },
    }
    for phase in (
        "external_flex_closure",
        "external_fatty_ring_closure",
        "external_side_chain_closure",
    ):
        phase_pending = [
            dict(item or {}) for item in list(pending_units or [])
            if str(dict(item or {}).get("phase", "") or "") == str(phase)
        ]
        summary[str(phase)] = {
            "pending_unit_count": int(len(phase_pending)),
            "pending_unit_count_by_kind": {
                str(kind): sum(
                    1 for item in phase_pending
                    if str(item.get("pending_kind", "") or "unknown_pending") == str(kind)
                )
                for kind in pending_by_kind
            },
            "pending_by_subphase": {
                str(key): int(len(list(value or [])))
                for key, value in dict(external_pending_by_subphase or {}).items()
                if any(dict(item or {}) in phase_pending for item in list(value or []))
            },
        }
    return summary


def _build_front_half_step_trace(
    *,
    phase: str,
    flex_audit: Dict[str, Any],
    assembly_attempted: bool,
    assembly_dict: Dict[str, Any],
    external_requirements: Dict[str, Any],
    external_ok: bool,
) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = [{
        "step": "classification",
        "status": "completed",
    }]
    flex_required = bool(str(phase) == "flex_range_reconciliation")
    trace.append({
        "step": "flex_range_reconciliation",
        "status": "pending" if flex_required else "completed",
        "scenario": str(dict(flex_audit or {}).get("scenario", "") or ""),
        "blocked_reason": str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        "target_flex_count": int(dict(flex_audit or {}).get("target_flex_count", 0) or 0),
    })
    subphase_status = {
        str(key): dict(value or {})
        for key, value in dict(external_requirements.get("subphase_status", {}) or {}).items()
    }
    step_to_subphase = (
        ("explicit_branch_closure", "explicit_branch_closure"),
        ("real_like24_ring_assembly", "real_like24_ring_assembly"),
        ("remaining_external_halfclosed_closure", "remaining_external_halfclosed_closure"),
    )
    blocked_seen = False
    for step_name, subphase_name in step_to_subphase:
        status = dict(subphase_status.get(str(subphase_name), {}) or {})
        contracts_for_subphase = [
            dict(item or {})
            for item in list(dict(external_requirements.get("contracts_by_subphase", {}) or {}).get(str(subphase_name), []) or [])
            if dict(item or {})
        ]
        if flex_required:
            step_status = "pending"
        elif bool(external_ok) or bool(status.get("ok", False)):
            step_status = "completed"
        elif not bool(assembly_attempted):
            step_status = "pending"
        elif not blocked_seen:
            step_status = "active"
            blocked_seen = True
        else:
            step_status = "pending"
        trace.append({
            "step": str(step_name),
            "status": str(step_status),
            "subphase": str(subphase_name),
            "requirement_count": int(len(list(contracts_for_subphase or []))),
            "blocking_reason": str(status.get("reason", "") or status.get("blocked_reason", "") or ""),
        })
    trace.append({
        "step": "external_phases_complete",
        "status": "completed" if str(phase) == "complete" else "pending",
    })
    return trace


def _assemble_external_only(
    *,
    nodes: List[_NodeV3],
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
    S_target: Any = None,
    E_target: Any = None,
) -> TopologyAssemblyResult:
    from .assembly_external import assemble_external_topologies

    return assemble_external_topologies(
        nodes=list(nodes or []),
        classification=classification,
        S_target=S_target,
        E_target=E_target,
        flex_audit=flex_audit,
    )


def assemble_topology_units(
    nodes: List[_NodeV3],
    classification: TopologyClassificationResult,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
) -> TopologyAssemblyResult:
    del fixed_partition_meta
    classification_dict = classification.to_dict()
    flex_audit = _compute_flex_audit(
        list(nodes or []),
        classification,
        enforce=True,
    )
    decision = Stage3FlexDecision(
        scenario=str(flex_audit.get("scenario", "in_range") or "in_range"),
        cluster_count=int(flex_audit.get("X", 0) or 0),
        su10_pair_count=int(flex_audit.get("Y", 0) or 0),
        effective_flex_count=int(flex_audit.get("explicit_flex_count", 0) or 0),
        lower_bound=int(flex_audit.get("flex_lower", 0) or 0),
        upper_bound=int(flex_audit.get("flex_upper", 0) or 0),
        target_count=int(flex_audit.get("target_flex_count", 0) or 0),
        selected_candidate_fragment_ids=list(flex_audit.get("selected_candidate_fragment_ids", []) or []),
        selected_candidate_node_ids=list(flex_audit.get("selected_candidate_node_ids", []) or []),
        required_resource_counts=dict(flex_audit.get("required_resource_counts", {}) or {}),
        unresolved_resource_counts=dict(flex_audit.get("unresolved_resource_counts", {}) or {}),
        blocked_reason=str(flex_audit.get("blocked_reason", "") or ""),
    )
    if _flex_precheck_required(flex_audit):
        result = TopologyAssemblyResult(**_preassembly_gate_assembly_dict(
            classification,
            classification_dict,
            flex_audit,
        ))
    else:
        result = _assemble_external_only(
            nodes=list(nodes or []),
            classification=classification,
            S_target=S_target,
            E_target=E_target,
            flex_audit=flex_audit,
        )
    result.flex_decision = decision.to_dict()
    result.requested_counts = {
        str(key): int(value) for key, value in dict(decision.required_resource_counts or {}).items()
    }
    ledger = result.resource_ledger.to_dict() if hasattr(result.resource_ledger, "to_dict") else dict(result.resource_ledger or {})
    shortages = {
        str(key): int(value)
        for key, value in dict(ledger.get("shortage_counts", {}) or {}).items()
        if int(value) > 0
    }
    result.unresolved_counts = dict(shortages or decision.unresolved_resource_counts)
    result.blocked_reason = str(decision.blocked_reason or ("external_resource_shortage" if shortages else ""))
    result.active_subphase = str(
        dict(result.diagnostics or {}).get("active_external_subphase", "")
        or dict(result.diagnostics or {}).get("stage3_phase", "")
        or ("complete" if result.ok else "flex_range_reconciliation")
    )
    result.active_phase = str(
        getattr(result, "active_phase", "")
        or dict(result.diagnostics or {}).get("stage3_phase", "")
        or result.active_subphase
    )
    result.topology_metrics = build_stage3_topology_metrics(classification, result, decision)
    result.topology_signature = build_stage3_topology_signature(classification, result, decision)
    result.phase_signature = str(result.topology_signature)
    return result


def build_stage3_phase_evaluation(
    nodes: List[_NodeV3],
    classification: Optional[TopologyClassificationResult] = None,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    generation_id: int = 0,
) -> Stage3PhaseEvaluation:
    del S_target, E_target
    classification_obj = classification or classify_topology_units(
        list(nodes or []),
        fixed_partition_meta=fixed_partition_meta,
    )
    flex_audit = _compute_flex_audit(
        list(nodes or []),
        classification_obj,
        enforce=True,
    )
    classification_dict = classification_obj.to_dict()
    round_context = build_stage3_round_context(
        nodes=list(nodes or []),
        classification=classification_obj,
        flex_audit=flex_audit,
        generation_id=int(generation_id),
    )
    ownership_view = ownership_public_view(round_context)
    flex_bounds = _flex_bounds_from_audit(flex_audit)
    explicit_count = int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0)
    flex_counts = {
        "explicit_total": int(explicit_count),
        "final": int(explicit_count),
    }
    flex_enforced = bool(flex_audit.get("enforced", False))
    flex_reconciliation_required = bool(_flex_precheck_required(flex_audit))
    flex_contracts = _flex_reconciliation_contracts(classification_obj, flex_audit)
    phase = "flex_range_reconciliation" if flex_reconciliation_required else "external_flex_closure"
    minimal_ledger = _minimal_resource_ledger_from_classification(classification_dict)
    minimal_ledger["ownership"] = ownership_view
    assembly_dict = {
        "ok": False,
        "reason": "stage3_round_not_executed",
        "products": [],
        "pending_units": [],
        "resource_ledger": minimal_ledger,
        "flex_bounds": dict(flex_bounds),
        "flex_counts": dict(flex_counts),
        "active_phase": str(phase),
        "active_subphase": str(phase),
        "completed_phases": [],
        "diagnostics": {
            "stage3_phase": str(phase),
            "evaluation_only": True,
            "pending_contracts": list(flex_contracts or []),
        },
    }

    diagnostics = {
        "phase_sequence": list(PHASE_SEQUENCE),
        "phase_status": {
            "classification": True,
            "ownership_partition": not bool(ownership_view.get("invariant_errors", [])),
            "flex_range_reconciliation": not bool(flex_reconciliation_required),
        },
        "resource_pool_remaining_counts": {
            str(key): len(values)
            for key, values in dict(round_context.closure_pool_ids_by_su or {}).items()
        },
        "unclassified_non_resource_count": int(
            dict(classification_dict.get("summary", {}) or {}).get("unclassified_non_resource_count", 0) or 0
        ),
        "claim_conflict_count": int(dict(classification_dict.get("summary", {}) or {}).get("claim_conflict_count", 0) or 0),
        "support_units_only_count": int(dict(classification_dict.get("summary", {}) or {}).get("support_units_only_count", 0) or 0),
        "pending_unit_count": int(len(flex_contracts)),
        "reject_count": int(dict(classification_dict.get("summary", {}) or {}).get("reject_count", 0) or 0),
        "open_fragment_count": int(dict(classification_dict.get("summary", {}) or {}).get("open_fragment_count", 0) or 0),
        "flex_audit": dict(flex_audit or {}),
        "classification_ready": True,
        "assembly_attempted": False,
        "evaluation_only": True,
        "flex_enforced": bool(flex_enforced),
        "flex_reconciliation_required": bool(flex_reconciliation_required),
        "flex_reconciliation_scenario": str(dict(flex_audit or {}).get("scenario", "") or ""),
        "flex_reconciliation_block_reason": str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        "flex_reconciliation_contracts": list(flex_contracts or []),
        "ownership_ledger": ownership_view,
        "generation_id": int(generation_id),
    }
    flex_decision = Stage3FlexDecision(
        scenario=str(dict(flex_audit or {}).get("scenario", "in_range") or "in_range"),
        cluster_count=int(dict(flex_audit or {}).get("X", 0) or 0),
        su10_pair_count=int(dict(flex_audit or {}).get("Y", 0) or 0),
        effective_flex_count=int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
        lower_bound=int(dict(flex_audit or {}).get("flex_lower", 0) or 0),
        upper_bound=int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
        target_count=int(dict(flex_audit or {}).get("target_flex_count", 0) or 0),
        selected_candidate_fragment_ids=[
            int(value) for value in list(dict(flex_audit or {}).get("selected_candidate_fragment_ids", []) or [])
        ],
        selected_candidate_node_ids=[
            int(value) for value in list(dict(flex_audit or {}).get("selected_candidate_node_ids", []) or [])
        ],
        required_resource_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("required_resource_counts", {}) or {}).items()
        },
        unresolved_resource_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("unresolved_resource_counts", {}) or {}).items()
        },
        blocked_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
    )
    topology_metrics = build_stage3_topology_metrics(classification_obj, assembly_dict, flex_decision)
    topology_signature = build_stage3_topology_signature(classification_obj, assembly_dict, flex_decision)
    return Stage3PhaseEvaluation(
        generation_id=int(generation_id),
        phase=str(phase),
        done=False,
        classification_ready=True,
        assembly_attempted=False,
        flex_range_reconciliation_ok=not bool(flex_reconciliation_required),
        external_closure_ok=False,
        flex_reconciliation_required=bool(flex_reconciliation_required),
        flex_reconciliation_scenario=str(dict(flex_audit or {}).get("scenario", "") or ""),
        flex_reconciliation_block_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        flex_bounds=dict(flex_bounds or {}),
        flex_counts=dict(flex_counts or {}),
        flex_audit=dict(flex_audit or {}),
        flex_range_case=str(dict(flex_audit or {}).get("scenario", "") or ""),
        flex_target_count=int(dict(flex_audit or {}).get("target_flex_count", 0) or 0),
        explicit_flex_count=int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
        explicit_flex_fragment_ids=[
            int(value)
            for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
        ],
        supplemental_flex_candidate_ids=[
            int(value)
            for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
        ],
        supplemental_flex_candidate_profiles=[
            dict(item or {})
            for item in list(dict(flex_audit or {}).get("supplemental_flex_candidate_profiles", []) or [])
            if dict(item or {})
        ],
        flex_reconciliation_contracts=list(flex_contracts or []),
        canonical_contracts=list(flex_contracts or []),
        flex_range_action_class=str(dict(flex_audit or {}).get("scenario", "") or "in_range"),
        active_external_subphase="external_flex_closure" if not flex_reconciliation_required else "",
        dormant_phases=[],
        dormant_phase_reasons={},
        assembly_view=assembly_dict,
        classification_view=classification_dict,
        diagnostics=diagnostics,
        flex_decision=flex_decision,
        topology_metrics=topology_metrics,
        topology_signature=topology_signature,
        requested_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("required_resource_counts", {}) or {}).items()
        },
        unresolved_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("unresolved_resource_counts", {}) or {}).items()
        },
        blocked_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        active_phase=str(phase),
        completed_phases=[],
        phase_signature=str(topology_signature),
        round_context=round_context,
        ownership_ledger=ownership_view,
        closure_pool_ids_by_su=deepcopy(dict(round_context.closure_pool_ids_by_su or {})),
        aromatic_donor_ids_by_su=deepcopy(dict(round_context.aromatic_donor_ids_by_su or {})),
    )
