"""Project classified topology fragments into assembly views."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .....shared.inverse_common import _NodeV3
from ...classification.classification_models import TopologyClassificationResult, TopologyFragment
from ...classification.classification_tokens import (
    AROMATIC_SU_TYPES,
    OPEN_ALIPHATIC_BASE_SU_TYPES,
    OPEN_ALIPHATIC_SUBTYPES,
    TERMINAL_BASE_SU_TYPES,
    TERMINAL_SUBTYPES,
    is_abstract_open_nonfixed_semantic,
    normalize_subtype,
)
from ...stage3.flex_audit import _compute_flex_audit
from ...fixed_edges import nodes_have_fixed_edge as _nodes_have_fixed_edge
from ...resources.types import PHYSICAL_TOPOLOGY_RESOURCE_TYPES
from ...stage3_models import (
    TopologyAssemblyResult,
    TopologyPathRender,
    TopologyPendingUnit,
    TopologyPoolLedger,
)
from ..core.models import (
    BranchStub,
)
from .validation import (
    canonical_view_render_node_ids as _canonical_view_render_node_ids,
    fragment_atomic_groups as _fragment_atomic_groups,
    route_unrendered_claimed_fixed_components as _route_unrendered_claimed_fixed_components,
    validate_fragment_view_projection as _validate_fragment_view_projection,
    view_render_route_node_ids as _view_render_route_node_ids,
)
from ..core.fragment import (
    AssemblyFragmentView,
    clean_ring_segment_text as _clean_ring_segment_text,
    is_aromatic_label as _is_aromatic_label,
    render_linear_path as _render_linear_path,
    render_token_sequence as _render_token_sequence,
    strip_open_markers as _strip_open_markers,
)
from ..core.rules import (
    CANONICAL_LABEL_BY_CLASS,
    TERMINAL_ENDPOINT_LABELS,
    main_kind_for_class as _main_kind_for_class,
)
from ..state.iteration_ledger import normalize_slot, required_counts_from_slots
from ...resources.profiles import (
    build_raw_su24_resource_profiles,
    raw_su24_segment_support,
)


def _slot_required_counts(slot_dicts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = required_counts_from_slots([
        normalize_slot(dict(slot or {}))
        for slot in list(slot_dicts or [])
        if dict(slot or {})
    ])
    return {str(key): int(value) for key, value in counts.items() if int(value) > 0}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_su_key(value: Any) -> str:
    return str(_safe_int(value))


def _dedupe_rendered_route_with_maps(
    labels: Sequence[Any],
    node_ids: Sequence[Optional[int]],
    branch_map: Dict[int, List[str]],
    branch_id_map: Dict[int, List[str]],
) -> Tuple[List[str], List[Optional[int]], Dict[int, List[str]], Dict[int, List[str]]]:
    """Normalize a route and preserve branch anchor indexes."""
    out_labels: List[str] = []
    out_ids: List[Optional[int]] = []
    old_to_new: Dict[int, int] = {}
    last_node_id: Optional[int] = None
    for index, (raw_label, raw_node_id) in enumerate(
        zip(list(labels or []), list(node_ids or []))
    ):
        node_id = None if raw_node_id is None else int(raw_node_id)
        if node_id is not None and node_id == last_node_id:
            old_to_new[int(index)] = len(out_ids) - 1
            continue
        old_to_new[int(index)] = len(out_ids)
        out_labels.append(str(raw_label))
        out_ids.append(node_id)
        last_node_id = node_id

    def _remap(values: Dict[int, List[str]]) -> Dict[int, List[str]]:
        remapped: Dict[int, List[str]] = {}
        for raw_index, items in dict(values or {}).items():
            index = int(raw_index)
            target = int(old_to_new.get(index, index))
            remapped.setdefault(target, []).extend(str(item) for item in list(items or []))
        return remapped

    return out_labels, out_ids, _remap(branch_map), _remap(branch_id_map)


def _strip_node_ids_from_path(path: str, excluded_node_ids: Iterable[int]) -> str:
    """Remove already-rendered mainline IDs from a branch node notation."""
    excluded = {int(value) for value in list(excluded_node_ids or [])}
    if not excluded:
        return str(path or "")

    def _replace(match: re.Match[str]) -> str:
        return "" if int(match.group(0)) in excluded else match.group(0)

    filtered = re.sub(r"\d+", _replace, str(path or ""))
    while "--" in filtered:
        filtered = filtered.replace("--", "-")
    filtered = filtered.replace("(-)", "")
    filtered = filtered.replace("()", "")
    return filtered


def _node_label(node: _NodeV3) -> str:
    subtype = str(normalize_subtype(node))
    try:
        su_type = int(getattr(node, "su_type", -1))
    except Exception:
        su_type = -1
    return subtype if subtype not in {str(su_type), ""} else str(su_type)


def _is_aromatic_token(token: Dict[str, Any]) -> bool:
    return _safe_int(token.get("su_type", -1), -1) in set(AROMATIC_SU_TYPES)


def _is_terminal_token(token: Dict[str, Any]) -> bool:
    return (
        _safe_int(token.get("su_type", -1), -1) in set(TERMINAL_BASE_SU_TYPES)
        or str(token.get("subtype", "")) in set(TERMINAL_SUBTYPES)
    )


def _is_open_token(token: Dict[str, Any]) -> bool:
    return (
        _safe_int(token.get("su_type", -1), -1) in set(OPEN_ALIPHATIC_BASE_SU_TYPES)
        or str(token.get("subtype", "")) in set(OPEN_ALIPHATIC_SUBTYPES)
    )


def _is_generic_open_token(token: Dict[str, Any]) -> bool:
    return bool(is_abstract_open_nonfixed_semantic(token))


def _normalize_ring_backbone_labels(values: Sequence[Any]) -> List[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _normalize_ring_backbone_node_ids(values: Sequence[Any]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for value in list(values or []):
        if value is None:
            out.append(None)
            continue
        try:
            out.append(int(value))
        except Exception:
            out.append(None)
    return out


def _compute_flex_bounds(
    *,
    explicit_flex_count: int,
    flex_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    audit = dict(flex_audit or {})
    if not audit:
        return {
            "min": int(max(0, explicit_flex_count)),
            "max": int(max(0, explicit_flex_count)),
            "target": int(max(0, explicit_flex_count)),
            "enforced": False,
            "ordinary_aliphatic_c": 0,
            "budget_xn": 0,
            "window_source": "explicit_flex_fallback",
        }
    audit_enforced = bool(audit.get("enforced", False))
    if not bool(audit_enforced):
        return {
            "min": int(max(0, explicit_flex_count)),
            "max": int(max(0, explicit_flex_count)),
            "target": int(max(0, explicit_flex_count)),
            "enforced": False,
            "ordinary_aliphatic_c": 0,
            "budget_xn": 0,
            "window_source": "flex_audit",
            "audit_X": int(audit.get("X", 0) or 0),
            "audit_Y": int(audit.get("Y", 0) or 0),
            "audit_Z": int(audit.get("Z", 0) or 0),
        }
    flex_min = int(audit.get("flex_lower", max(0, int(explicit_flex_count))) or 0)
    flex_max = int(audit.get("flex_upper", explicit_flex_count) or explicit_flex_count)
    flex_target = int(audit.get("target_flex_count", explicit_flex_count) or explicit_flex_count)
    if int(flex_min) <= int(flex_max):
        flex_target = max(int(flex_min), min(int(flex_max), int(flex_target)))
    else:
        flex_target = int(max(0, explicit_flex_count))
    return {
        "min": int(flex_min),
        "max": int(flex_max),
        "target": int(flex_target),
        "enforced": True,
        "ordinary_aliphatic_c": 0,
        "budget_xn": 0,
        "window_source": "flex_audit",
        "audit_X": int(audit.get("X", 0) or 0),
        "audit_Y": int(audit.get("Y", 0) or 0),
        "audit_Z": int(audit.get("Z", 0) or 0),
    }


def _flex_preassembly_gate_result(
    *,
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
    flex_bounds: Dict[str, Any],
) -> TopologyAssemblyResult:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    if scenario == "above_range" and str(blocked_reason) != "flex_window_invalid":
        result_reason = "external_flex_range_exceeded"
    else:
        result_reason = "external_flex_range_pending"
    pending_units: List[TopologyPendingUnit] = []
    fragments_by_id = {
        int(getattr(fragment, "fragment_id", -1)): fragment
        for fragment in list(getattr(classification, "fragments", []) or [])
    }
    hard_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
    }
    supplemental_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
    }
    pending_kind_by_id: Dict[int, str] = {}
    pending_ids: List[int] = []
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
    for fragment_id in list(pending_ids or []):
        fragment = fragments_by_id.get(int(fragment_id))
        if fragment is None:
            continue
        pending_units.append(TopologyPendingUnit(
            unit_id=int(fragment_id),
            pending_kind=str(pending_kind_by_id.get(int(fragment_id), "supplemental_flex_candidate")),
            real_topology_partial=str(getattr(fragment, "topology_template", "") or getattr(fragment, "class_type", "")),
            real_node_partial="",
            source_fragment_ids=[int(fragment_id)],
            source_fragment_classes=[str(getattr(fragment, "class_type", "") or "")],
            claimed_source_node_ids=[
                int(value)
                for value in list(
                    getattr(fragment, "claimed_node_ids", []) or []
                )
            ],
            reason=str(blocked_reason or "flex_range_reconciliation_pending"),
        ))
    resource_pool = dict(getattr(classification, "resource_pool", {}) or {})
    initial_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("initial_ids_by_su", {}) or {}).items()
    }
    remaining_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("remaining_ids_by_su", initial_ids_by_su) or {}).items()
    }
    all_keys = sorted({str(key) for key in set(initial_ids_by_su) | set(remaining_ids_by_su)}, key=int) if (initial_ids_by_su or remaining_ids_by_su) else []
    ledger = TopologyPoolLedger(
        initial_ids_by_su={str(key): list(initial_ids_by_su.get(str(key), []) or []) for key in all_keys},
        remaining_ids_by_su={str(key): list(remaining_ids_by_su.get(str(key), []) or []) for key in all_keys},
        consumed_ids_by_su={str(key): [] for key in all_keys},
        initial_counts={str(key): int(len(list(initial_ids_by_su.get(str(key), []) or []))) for key in all_keys},
        consumed_counts={str(key): 0 for key in all_keys},
        remaining_counts={str(key): int(len(list(remaining_ids_by_su.get(str(key), []) or []))) for key in all_keys},
        shortage_counts={},
    )
    return TopologyAssemblyResult(
        ok=False,
        reason=str(result_reason),
        feasible=bool(
            len(list(getattr(classification, "unclassified_nodes", []) or [])) == 0
            and len(list(getattr(classification, "claim_conflicts", []) or [])) == 0
            and len(list(getattr(classification, "classification_rejects", []) or [])) == 0
        ),
        products=[],
        pending_units=list(pending_units or []),
        flex_bounds=dict(flex_bounds or {}),
        flex_counts={
            "explicit_total": int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
            "final": 0,
        },
        resource_ledger=ledger,
        diagnostics={
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
            "pending_contracts": [],
            "total_shortage": {},
        },
        planned_conversions=[],
    )


def _build_node_lookup(nodes: Sequence[_NodeV3]) -> Dict[int, _NodeV3]:
    out: Dict[int, _NodeV3] = {}
    for node in list(nodes or []):
        try:
            out[int(getattr(node, "global_id"))] = node
        except Exception:
            continue
    return out


def _flex_reconcile_roles_from_evidence(fragment: TopologyFragment) -> List[str]:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    return sorted({
        str(role)
        for role in list(evidence.get("flex_reconcile_roles", []) or [])
        if str(role)
    })


def _flex_reconcile_priority_from_evidence(fragment: TopologyFragment) -> int:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    try:
        return int(evidence.get("flex_reconcile_priority", 99))
    except Exception:
        return 99


def _ring_roles_from_evidence(fragment: TopologyFragment) -> List[str]:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    return sorted({
        str(role)
        for role in list(evidence.get("ring_roles", []) or [])
        if str(role)
    })


def _ring_like24_units_from_evidence(fragment: TopologyFragment) -> int:
    evidence = dict(getattr(fragment, "evidence", {}) or {})
    try:
        return max(0, int(evidence.get("ring_like24_units", 0)))
    except Exception:
        return 0


def _skeleton_allowed_name(skeleton_kind: str) -> str:
    mapping = {
        "side_ring_plan": "side",
        "upper_lower_ring_plan": "upper_lower",
        "fused_ring_plan": "fused",
    }
    return str(mapping.get(str(skeleton_kind), str(skeleton_kind)))


def _pick_center_node(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> Optional[_NodeV3]:
    evidence = dict(fragment.evidence or {})
    center_ids = list(evidence.get("center_node_ids", []) or [])
    if center_ids:
        return nodes_by_id.get(_safe_int(center_ids[0], -1))
    if evidence.get("center_node_id") is not None:
        return nodes_by_id.get(_safe_int(evidence.get("center_node_id"), -1))
    if list(fragment.main_path or []):
        node_id = getattr(fragment.main_path[0], "node_id", None)
        if node_id is not None:
            return nodes_by_id.get(int(node_id))
    if list(fragment.source_node_ids or []):
        return nodes_by_id.get(int(fragment.source_node_ids[0]))
    return None


def _canonical_labels_for_class(class_type: str, fallback: Sequence[str]) -> List[str]:
    labels = list(CANONICAL_LABEL_BY_CLASS.get(str(class_type), []) or [])
    return labels if labels else list(fallback or [])


def _dict_label(token: Dict[str, Any]) -> str:
    return str(token.get("subtype") or token.get("su_type") or "")


def _dict_node_id(token: Dict[str, Any]) -> int:
    return _safe_int(token.get("node_id"), -1)


def _dict_su(token: Dict[str, Any]) -> int:
    return _safe_int(token.get("su_type"), -1)


def _sort_token_dicts(tokens: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [dict(token or {}) for token in list(tokens or [])],
        key=lambda item: (_dict_node_id(item), _dict_label(item)),
    )


def _terminal_priority(token: Dict[str, Any]) -> Tuple[int, int, str]:
    su_type = _dict_su(token)
    subtype = _dict_label(token)
    if su_type in {1, 4, 16, 18, 28, 32} or subtype in {"19d1", "20d1", "21d2"}:
        return (0, _dict_node_id(token), subtype)
    if su_type == 22:
        return (1, _dict_node_id(token), subtype)
    return (2, _dict_node_id(token), subtype)


def _source_fragment_classes(source_views: Sequence["AssemblyFragmentView"]) -> List[str]:
    return [str(view.class_type) for view in list(source_views or [])]


def _collect_source_metadata(source_views: Sequence["AssemblyFragmentView"]) -> Tuple[List[int], List[str], List[int]]:
    source_fragment_ids = sorted({int(view.fragment_id) for view in list(source_views or [])})
    source_fragment_classes = _source_fragment_classes(source_views)
    source_node_ids = sorted({
        int(node_id)
        for view in list(source_views or [])
        for node_id in list(view.source_node_ids or [])
    })
    return source_fragment_ids, source_fragment_classes, source_node_ids


def _fixed_source_ids_from_views(
    source_views: Sequence["AssemblyFragmentView"],
    *,
    assembler: Optional[Any] = None,
) -> List[int]:
    fixed_ids: Set[int] = set()
    source_fragment_ids = {
        int(getattr(view, "fragment_id", -1))
        for view in list(source_views or [])
    }
    owner_map = {
        int(node_id): int(fragment_id)
        for node_id, fragment_id in dict(
            getattr(assembler, "fixed_node_owner_by_id", {}) or {}
        ).items()
    } if assembler is not None else {}
    nodes_by_id = dict(getattr(assembler, "nodes_by_id", {}) or {}) if assembler is not None else {}

    def _keep_source_id(node_id: int) -> bool:
        owner = owner_map.get(int(node_id))
        if owner is not None:
            return int(owner) in source_fragment_ids
        node = nodes_by_id.get(int(node_id))
        if node is not None and int(
            getattr(node, "su_type", -1)
        ) in set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES):
            return False
        return True

    for view in list(source_views or []):
        evidence = dict(getattr(view, "evidence", {}) or {})
        fixed_ids.update(
            int(value)
            for value in list(getattr(view, "source_node_ids", []) or [])
            if _keep_source_id(int(value))
        )
        for key in (
            "center_node_ids",
            "atomic_group_node_ids",
            "fixed_source_node_ids",
            "fixed_hop1_ids",
        ):
            fixed_ids.update(
                int(value)
                for value in list(evidence.get(key, []) or [])
                if _keep_source_id(int(value))
            )
        fixed_ids.add(int(view.fragment_id))
    return sorted(fixed_ids)


def _should_materialize_bridge_literal(view: "AssemblyFragmentView", bridge_label: str) -> bool:
    label = str(bridge_label or "")
    if label in {
        "0",
        "2",
        "3",
        "14",
        "23",
        "24",
        "25",
    }:
        return False
    if label in {
        "15",
        "17",
        "single19d2",
        "single20d2",
        "double19d2",
        "double20d2",
        "21d2",
    }:
        return bool(
            getattr(view, "preferred_product_type", "") == "side_chain"
            or any(
                role in {
                    "b_like_open",
                    "d_like_open",
                    "single_end_flex_source",
                    "double_end_flex_source",
                }
                for role in list(getattr(view, "flex_reconcile_roles", []) or [])
            )
        )
    return False


def _first_neighbor_by_su(
    center: Optional[_NodeV3],
    *,
    su_types: Set[int],
    exclude_ids: Optional[Set[int]] = None,
    nodes_by_id: Optional[Dict[int, _NodeV3]] = None,
) -> Optional[_NodeV3]:
    if center is None or nodes_by_id is None:
        return None
    blocked = {int(x) for x in list(exclude_ids or set())}
    candidates: List[_NodeV3] = []
    for node_id in list(getattr(center, "hop1_ids", []) or []):
        node_id_i = _safe_int(node_id, -1)
        if node_id_i in blocked:
            continue
        neighbor = nodes_by_id.get(node_id_i)
        if neighbor is None:
            continue
        if int(getattr(neighbor, "su_type", -1)) in set(su_types):
            candidates.append(neighbor)
    candidates.sort(key=lambda item: int(getattr(item, "global_id", 10**9)))
    return candidates[0] if candidates else None


def _direct_join_bridge_tokens(
    *,
    left_bridge: Optional[Tuple[str, int]],
    right_bridge: Optional[Tuple[str, int]],
    left_view: "AssemblyFragmentView",
    right_view: "AssemblyFragmentView",
) -> Tuple[List[str], List[int]]:
    def _normalize(
        bridge: Optional[Tuple[str, int]],
        view: "AssemblyFragmentView",
    ) -> Optional[Tuple[str, int]]:
        if bridge is None:
            return None
        label, node_id = bridge
        bridge_label = str(label or "")
        bridge_node_id = int(node_id)
        if bridge_label == "23":
            return ("23", bridge_node_id)
        if _should_materialize_bridge_literal(view, bridge_label):
            return (bridge_label, bridge_node_id)
        return None

    left_token = _normalize(left_bridge, left_view)
    right_token = _normalize(right_bridge, right_view)

    if left_token is not None and right_token is not None:
        if left_token[0] == "23" and right_token[0] != "23":
            left_token = None
        elif right_token[0] == "23" and left_token[0] != "23":
            right_token = None

    if left_token is not None and right_token is not None:
        if left_token == right_token:
            return [str(left_token[0])], [int(left_token[1])]
        if left_token[0] == "23" and right_token[0] == "23":
            return ["23", "23"], [int(left_token[1]), int(right_token[1])]
        return [str(left_token[0]), str(right_token[0])], [int(left_token[1]), int(right_token[1])]
    if left_token is not None:
        return [str(left_token[0])], [int(left_token[1])]
    if right_token is not None:
        return [str(right_token[0])], [int(right_token[1])]
    return [], []


def _ordered_ring_backbone_map(
    *,
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    allowed_kinds: Sequence[str],
) -> Dict[str, List[Any]]:
    normalized_labels = [str(value) for value in list(labels or [])]
    normalized_node_ids = [None if value is None else int(value) for value in list(node_ids or [])]
    out: Dict[str, List[Any]] = {}
    for kind in list(allowed_kinds or []):
        out[str(kind)] = list(normalized_labels)
        out[f"{str(kind)}_ids"] = list(normalized_node_ids)
    return out


def _open_basic_ring_projection(
    *,
    class_type: str,
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    open_left: bool,
    open_right: bool,
    evidence: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[Optional[int]]],
    Dict[str, str],
    Dict[str, int],
    Dict[str, List[str]],
    List[str],
    Dict[str, Any],
]:
    """Expose existing B/D paths to ring templates without reclassifying them."""
    resolved_evidence = dict(evidence or {})
    class_name = str(class_type)
    path_labels = [str(value) for value in list(labels or []) if str(value)]
    path_node_ids = _normalize_ring_backbone_node_ids(node_ids)
    if (
        class_name not in {"B", "B+", "D", "D+"}
        or not path_labels
        or not bool(open_left or open_right)
    ):
        return {}, {}, {}, {}, {}, [], resolved_evidence

    if class_name in {"B", "B+"}:
        allowed = ["side"]
        capabilities = {
            "side": ["terminal_pair" if len(path_labels) > 1 else "terminal_single"],
        }
        backbone_kind = {"side": "side_terminal_path"}
        ring_roles = ["side_anchor"]
    else:
        allowed = ["side", "upper_lower"]
        pair_path = len(path_labels) > 1
        capabilities = {
            "side": ["inner_pair" if pair_path else "inner_single"],
            "upper_lower": ["path_pair" if pair_path else "path_single"],
        }
        backbone_kind = {
            "side": "side_middle_path",
            "upper_lower": "upper_path",
        }
        ring_roles = ["side_middle", "upper_core"]

    resolved_evidence["ring_allowed_skeletons"] = list(allowed)
    resolved_evidence["ring_path_capabilities"] = {
        str(key): list(values) for key, values in capabilities.items()
    }
    resolved_evidence["ring_roles"] = list(ring_roles)
    backbones = _ordered_ring_backbone_map(
        labels=path_labels,
        node_ids=path_node_ids,
        allowed_kinds=allowed,
    )
    backbone_labels = {
        str(kind): [str(value) for value in list(backbones.get(str(kind), []) or [])]
        for kind in allowed
    }
    backbone_ids = {
        str(kind): _normalize_ring_backbone_node_ids(
            backbones.get(f"{str(kind)}_ids", [])
        )
        for kind in allowed
    }
    backbone_span = {
        str(kind): len(list(backbone_labels.get(str(kind), []) or []))
        for kind in allowed
    }
    return (
        backbone_labels,
        backbone_ids,
        backbone_kind,
        backbone_span,
        capabilities,
        ring_roles,
        resolved_evidence,
    )


def _expand_atomic_groups_in_mainline(
    *,
    fragment: TopologyFragment,
    labels: Sequence[str],
    node_ids: Sequence[int],
    nodes_by_id: Dict[int, _NodeV3],
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """Replace a routed atomic-group member with the complete ordered group."""
    expanded_ids = [int(value) for value in list(node_ids or [])]
    expansions: List[Dict[str, Any]] = []
    for group in _fragment_atomic_groups(fragment):
        group_ids = [
            int(value)
            for value in list(group.get("node_ids", []) or [])
            if int(value) in nodes_by_id
        ]
        if not group_ids or set(group_ids).issubset(set(expanded_ids)):
            continue
        routed_members = [node_id for node_id in group_ids if node_id in expanded_ids]
        if len(routed_members) != 1:
            continue
        attachment_id = int(routed_members[0])
        attachment_index = expanded_ids.index(attachment_id)
        other_ids = sorted(
            (node_id for node_id in group_ids if node_id != attachment_id),
            key=lambda node_id: (
                2 if int(getattr(nodes_by_id.get(int(node_id)), "su_type", -1)) in {16, 18} else 0,
                0 if int(getattr(nodes_by_id.get(int(node_id)), "su_type", -1)) == 14 else 1,
                int(getattr(nodes_by_id.get(int(node_id)), "su_type", -1)),
                int(node_id),
            ),
        )
        if attachment_index == 0:
            replacement = list(other_ids) + [attachment_id]
        elif attachment_index == len(expanded_ids) - 1:
            replacement = [attachment_id] + list(other_ids)
        else:
            attachment_neighbors = {
                int(value)
                for value in list(
                    getattr(nodes_by_id.get(attachment_id), "hop1_ids", []) or []
                )
            }
            previous_id = int(expanded_ids[attachment_index - 1])
            next_id = int(expanded_ids[attachment_index + 1])
            if previous_id in attachment_neighbors and next_id not in attachment_neighbors:
                replacement = [attachment_id] + list(other_ids)
            elif next_id in attachment_neighbors and previous_id not in attachment_neighbors:
                replacement = list(other_ids) + [attachment_id]
            else:
                replacement = [attachment_id] + list(other_ids)
        expanded_ids[attachment_index:attachment_index + 1] = replacement
        expansions.append({
            "group_id": int(group.get("group_id", min(group_ids))),
            "attachment_node_id": int(attachment_id),
            "rendered_node_ids": list(replacement),
        })
    expanded_labels = [
        _node_label(nodes_by_id[node_id])
        if node_id in nodes_by_id
        else str(labels[index]) if index < len(labels) else str(node_id)
        for index, node_id in enumerate(expanded_ids)
    ]
    return expanded_labels, expanded_ids, expansions


def _project_basic_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    class_type = str(fragment.class_type)
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    center_su = int(getattr(center, "su_type", -1))
    matched_neighbors = list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
    claimed_ids = {int(value) for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])}
    # An unsaturated atomic pair such as SU17-SU18 is a branch unit when it is
    # attached to a basic center through SU17.  SU18 is a terminal endpoint;
    # putting the pair on the center's mainline would force a route such as
    # ``18-17-3`` and place the terminal in the middle after closure resources
    # are prepended.  Keep the pair together as one flat branch payload and
    # let the center's remaining open ports form the mainline.
    atomic_branch_paths: List[List[int]] = []
    atomic_branch_member_ids: Set[int] = set()
    atomic_group_records = list(
        dict(fragment.evidence or {}).get("context_atomic_groups", []) or []
    ) + list(dict(fragment.evidence or {}).get("atomic_groups", []) or [])
    for raw_group in atomic_group_records:
        group = dict(raw_group or {})
        group_ids = [
            int(value)
            for value in list(group.get("node_ids", []) or [])
            if int(value) in nodes_by_id and int(value) in claimed_ids
        ]
        if len(group_ids) != 2:
            continue
        if bool(group.get("center_owned", False)) or int(getattr(center, "global_id", -1)) in set(group_ids):
            continue
        terminal_ids = [
            int(node_id)
            for node_id in group_ids
            if _node_label(nodes_by_id[int(node_id)]) in set(TERMINAL_ENDPOINT_LABELS)
        ]
        if len(terminal_ids) != 1:
            continue
        terminal_id = int(terminal_ids[0])
        open_ids = [int(node_id) for node_id in group_ids if int(node_id) != terminal_id]
        if len(open_ids) != 1:
            continue
        open_id = int(open_ids[0])
        if not any(
            _dict_node_id(dict(item or {})) == int(open_id)
            for item in matched_neighbors
        ):
            continue
        atomic_branch_paths.append([int(open_id), int(terminal_id)])
        atomic_branch_member_ids.update({int(open_id), int(terminal_id)})

    raw_open_neighbor_ids = [
        _dict_node_id(dict(item or {}))
        for item in matched_neighbors
        if _is_open_token(dict(item or {}))
    ]
    if atomic_branch_member_ids:
        matched_neighbors = [
            dict(item or {})
            for item in matched_neighbors
            if _dict_node_id(dict(item or {})) not in atomic_branch_member_ids
        ]
    center_atomic_node_ids = {
        int(value)
        for value in list(
            dict(fragment.evidence or {}).get("center_atomic_group_node_ids", []) or []
        )
    }
    aromatic = _sort_token_dicts([nb for nb in matched_neighbors if _is_aromatic_token(nb)])
    open_neighbors = _sort_token_dicts([nb for nb in matched_neighbors if _is_open_token(nb)])
    generic_open_neighbors = _sort_token_dicts([
        nb for nb in matched_neighbors
        if _is_generic_open_token(nb) and _dict_node_id(dict(nb or {})) in claimed_ids
        and _dict_node_id(dict(nb or {})) not in center_atomic_node_ids
    ])
    special_open_neighbors = _sort_token_dicts([nb for nb in open_neighbors if not _is_generic_open_token(nb)])
    terminal = sorted(
        [dict(nb or {}) for nb in matched_neighbors if _is_terminal_token(nb)],
        key=_terminal_priority,
    )
    existing_bridge: Dict[str, Any] = {}

    if class_type == "A":
        labels = [_dict_label(aromatic[0]), _node_label(center), _dict_label(aromatic[1])]
        node_ids = [_dict_node_id(aromatic[0]), int(center.global_id), _dict_node_id(aromatic[1])]
        open_left = False
        open_right = False
        preferred = "flex_chain"
    elif class_type == "B":
        labels = [_dict_label(aromatic[0]), _node_label(center)]
        node_ids = [_dict_node_id(aromatic[0]), int(center.global_id)]
        open_left = False
        open_right = True
        preferred = "flex_chain"
    elif class_type == "C":
        labels = [_dict_label(aromatic[0]), _node_label(center), _dict_label(terminal[0])]
        node_ids = [_dict_node_id(aromatic[0]), int(center.global_id), _dict_node_id(terminal[0])]
        open_left = False
        open_right = False
        preferred = "side_chain"
    elif class_type == "D":
        if center_su in {15, 17}:
            pair_su_types = {14, 15, 16} if center_su == 15 else {17, 18}
            pair_neighbor = next(
                dict(nb or {})
                for nb in matched_neighbors
                if _dict_su(dict(nb or {})) in pair_su_types
            )
            labels = [_node_label(center), _dict_label(pair_neighbor)]
            node_ids = [int(center.global_id), _dict_node_id(pair_neighbor)]
        else:
            labels = [_node_label(center)]
            node_ids = [int(center.global_id)]
        remaining_open_ids = [
            int(node_id)
            for node_id in raw_open_neighbor_ids
            if int(node_id) not in atomic_branch_member_ids
        ]
        if len(remaining_open_ids) >= 2:
            open_left = True
            open_right = True
        elif len(remaining_open_ids) == 1:
            # Preserve the side of the original open port.  The atomic branch
            # consumes the other port and must not create a second mainline
            # closure slot.
            atomic_positions = [
                index
                for index, node_id in enumerate(raw_open_neighbor_ids)
                if int(node_id) in atomic_branch_member_ids
            ]
            if atomic_positions and atomic_positions[0] == 0:
                open_left = False
                open_right = True
            else:
                open_left = True
                open_right = False
        else:
            open_left = False
            open_right = False
        preferred = "flex_chain"
    elif class_type == "E":
        if center_su == 3:
            labels = [_dict_label(terminal[0]), _node_label(center)]
            node_ids = [_dict_node_id(terminal[0]), int(center.global_id)]
            open_left = False
            open_right = True
        elif center_su == 15:
            other_15 = next(
                dict(nb or {})
                for nb in matched_neighbors
                if _dict_su(dict(nb or {})) == 15
            )
            labels = [_dict_label(terminal[0]), _node_label(center), _dict_label(other_15)]
            node_ids = [_dict_node_id(terminal[0]), int(center.global_id), _dict_node_id(other_15)]
            open_left = False
            open_right = True
        elif center_su in {27, 29, 31} and special_open_neighbors:
            labels = [_dict_label(terminal[0]), _node_label(center)]
            node_ids = [_dict_node_id(terminal[0]), int(center.global_id)]
            open_left = False
            open_right = True
        elif center_su in {0, 2} and special_open_neighbors and any(_dict_su(nb) == 22 for nb in terminal):
            labels = [_dict_label(terminal[0]), _node_label(center)]
            node_ids = [_dict_node_id(terminal[0]), int(center.global_id)]
            open_left = False
            open_right = True
        else:
            labels = [_dict_label(terminal[0]), _node_label(center)]
            node_ids = [_dict_node_id(terminal[0]), int(center.global_id)]
            open_left = False
            open_right = True
        preferred = "side_chain"
    elif class_type == "F":
        labels = [_dict_label(aromatic[0]), _node_label(center)]
        node_ids = [_dict_node_id(aromatic[0]), int(center.global_id)]
        open_left = False
        open_right = False
        preferred = "side_chain"
    elif class_type == "G":
        if center_su in {16, 18}:
            pair_su_types = {14, 15, 16} if center_su == 16 else {17, 18}
            pair_neighbor = next(
                dict(nb or {})
                for nb in matched_neighbors
                if _dict_su(dict(nb or {})) in pair_su_types
            )
            labels = [_dict_label(pair_neighbor), _node_label(center)]
            node_ids = [_dict_node_id(pair_neighbor), int(center.global_id)]
        else:
            fixed_terminal_bridge_subtype = {
                28: "single19d2",
                32: "21d2",
            }.get(center_su)
            existing_bridge = (
                next(
                    (
                        dict(candidate or {})
                        for candidate in list(matched_neighbors or [])
                        if _dict_node_id(dict(candidate or {})) in claimed_ids
                        and str(_dict_label(dict(candidate or {}))) == str(fixed_terminal_bridge_subtype)
                    ),
                    {},
                )
                if fixed_terminal_bridge_subtype is not None
                else {}
            )
            if existing_bridge:
                labels = [_dict_label(existing_bridge), _node_label(center)]
                node_ids = [_dict_node_id(existing_bridge), int(center.global_id)]
            else:
                labels = [_node_label(center)]
                node_ids = [int(center.global_id)]
        open_left = True
        open_right = False
        preferred = "side_chain"
    else:
        labels = [_node_label(center)]
        node_ids = [int(center.global_id)]
        open_left = bool(list(fragment.open_slots or []))
        open_right = bool(list(fragment.open_slots or []))
        preferred = "unknown"
    endpoint_profiles = [
        dict(item or {})
        for item in list(dict(fragment.evidence or {}).get("basic_open_endpoints", []) or [])
    ]
    if atomic_branch_member_ids:
        endpoint_profiles = [
            dict(item or {})
            for item in endpoint_profiles
            if _safe_int(dict(item or {}).get("node_id"), -1)
            not in atomic_branch_member_ids
        ]
    endpoint_by_side = {
        str(item.get("side", "")): item
        for item in endpoint_profiles
        if str(item.get("side", "")) in {"left", "right"}
    }

    def _owned_endpoint_token(side: str) -> Dict[str, Any]:
        profile = dict(endpoint_by_side.get(str(side), {}) or {})
        node_id = _safe_int(profile.get("node_id"), -1)
        if not bool(profile.get("owned_literal", False)) or node_id not in claimed_ids:
            return {}
        token = next(
            (
                dict(item or {})
                for item in matched_neighbors
                if _dict_node_id(dict(item or {})) == int(node_id)
            ),
            {},
        )
        return token or {
            "node_id": int(node_id),
            "su_type": _safe_int(profile.get("su_type"), -1),
            "subtype": str(profile.get("subtype", "") or ""),
            "is_fixed": bool(profile.get("is_fixed", False)),
        }

    left_port_token = _owned_endpoint_token("left")
    right_port_token = _owned_endpoint_token("right")
    left_port_id = _dict_node_id(left_port_token) if left_port_token else -1
    if left_port_id in claimed_ids and left_port_id not in set(int(value) for value in node_ids if value is not None):
        labels = [_dict_label(left_port_token)] + list(labels)
        node_ids = [int(left_port_id)] + list(node_ids)
    right_port_id = _dict_node_id(right_port_token) if right_port_token else -1
    if right_port_id in claimed_ids and right_port_id not in set(int(value) for value in node_ids if value is not None):
        labels = list(labels) + [_dict_label(right_port_token)]
        node_ids = list(node_ids) + [int(right_port_id)]
    labels, node_ids, atomic_path_expansions = _expand_atomic_groups_in_mainline(
        fragment=fragment,
        labels=labels,
        node_ids=node_ids,
        nodes_by_id=nodes_by_id,
    )
    main_kind = _main_kind_for_class(class_type)
    (
        ring_backbone_labels,
        ring_backbone_node_ids,
        ring_backbone_kind,
        ring_backbone_span,
        ring_template_roles,
        ring_roles,
        view_evidence,
    ) = _open_basic_ring_projection(
        class_type=class_type,
        labels=labels,
        node_ids=node_ids,
        open_left=bool(open_left),
        open_right=bool(open_right),
        evidence=dict(fragment.evidence or {}),
    )
    ring_anchor_profile = {"left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")), "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else ""))}
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
    capability_evidence = dict(
        dict(view_evidence or {}).get("assembly_capabilities", {}) or {}
    )
    # Product preference is an ordering hint; explicit classification
    # capabilities preserve legal flex/side/ring fallback routes.
    can_flex = bool(capability_evidence.get(
        "flex_allowed",
        ("flex_preferred" in set(flex_reconcile_roles)) or preferred == "flex_chain",
    ))
    can_side = bool(capability_evidence.get(
        "side_allowed", preferred == "side_chain" or class_type in {"B", "D"}
    ))
    can_ring = bool(capability_evidence.get("ring_allowed", bool(ring_roles))) and bool(ring_roles)
    branch_anchor_index = (
        list(node_ids).index(int(center.global_id))
        if int(center.global_id) in set(int(value) for value in node_ids)
        else max(0, len(labels) // 2)
    )
    atomic_branch_stubs = [
        BranchStub(
            topology_notation="-" + "-".join(
                _node_label(nodes_by_id[int(node_id)]) for node_id in path
            ),
            node_id_notation="-" + "-".join(str(int(node_id)) for node_id in path),
            source_node_ids=[int(node_id) for node_id in path],
            branch_kind="actual_atomic_terminal_branch",
            mode="basic_atomic_terminal_branch",
        )
        for path in atomic_branch_paths
    ]
    branch_stubs_by_anchor = (
        {int(branch_anchor_index): list(atomic_branch_stubs)}
        if atomic_branch_stubs
        else {}
    )
    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=class_type,
        topology_template=str(fragment.topology_template),
        source_node_ids=list(fragment.source_node_ids or []),
        context_node_ids=list(fragment.context_node_ids or []),
        main_labels=labels,
        main_node_ids=node_ids,
        canonical_labels=_canonical_labels_for_class(class_type, labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=int(branch_anchor_index),
        unresolved_branch_kinds=[],
        actual_branch_stubs=list(atomic_branch_stubs),
        branch_stubs_by_anchor=branch_stubs_by_anchor,
        real_branch_stubs_by_anchor=branch_stubs_by_anchor,
        main_kind=main_kind,
        preferred_product_type=preferred,
        ring_roles=ring_roles,
        ring_backbone_labels=ring_backbone_labels,
        ring_backbone_node_ids=ring_backbone_node_ids,
        ring_backbone_kind=ring_backbone_kind,
        ring_backbone_span=ring_backbone_span,
        ring_like24_units=_ring_like24_units_from_evidence(fragment),
        ring_anchor_profile=ring_anchor_profile,
        ring_template_roles=ring_template_roles,
        flex_reconcile_roles=flex_reconcile_roles,
        flex_reconcile_priority=int(flex_reconcile_priority),
        can_flex=can_flex,
        can_side=can_side,
        can_ring=can_ring,
        target_roles=sorted({
            *(["flex_chain"] if can_flex else []),
            *(["side_chain"] if can_side else []),
            *(["fatty_ring"] if can_ring else []),
        }),
        evidence=view_evidence,
        diagnostics={
            "source_fragment_family": str(fragment.family),
            "matched_neighbors": [dict(token or {}) for token in matched_neighbors],
            "aromatic_neighbors": [dict(token or {}) for token in aromatic],
            "terminal_neighbors": [dict(token or {}) for token in terminal],
            "special_open_neighbors": [dict(token or {}) for token in special_open_neighbors],
            "generic_open_neighbors": [dict(token or {}) for token in generic_open_neighbors],
            "left_open_bridge": dict(left_port_token or {}),
            "right_open_bridge": dict(right_port_token or {}),
            "basic_open_endpoints": endpoint_profiles,
            "context_open_endpoints": [
                dict(item or {})
                for item in endpoint_profiles
                if bool(dict(item or {}).get("context_only", False))
            ],
            "atomic_path_expansions": atomic_path_expansions,
            "atomic_terminal_branch_paths": [
                [int(node_id) for node_id in path]
                for path in atomic_branch_paths
            ],
        },
    )


def _project_double_d2_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    class_type = str(fragment.class_type)
    evidence = dict(fragment.evidence or {})
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    profiles = [dict(x or {}) for x in list(evidence.get("connector_profiles", []) or [])]

    def _segment(profile: Dict[str, Any]) -> Tuple[List[str], List[int]]:
        labels: List[str] = []
        node_ids: List[int] = []
        for node_id in list(profile.get("remaining_node_ids", []) or []):
            node = nodes_by_id.get(_safe_int(node_id, -1))
            labels.append(_node_label(node) if node is not None else str(node_id))
            node_ids.append(_safe_int(node_id, -1))
        connector_id = _safe_int(profile.get("connector_node_id"), -1)
        connector = nodes_by_id.get(connector_id)
        labels.append(_node_label(connector) if connector is not None else str(profile.get("connector_su_type", "")))
        node_ids.append(int(connector_id))
        return labels, node_ids

    ordered = sorted(
        profiles,
        key=lambda item: (
            {"M": 0, "N": 1, "P": 2}.get(str(item.get("side_class", "")), 9),
            _safe_int(item.get("connector_node_id"), 10**9),
        ),
    )
    if class_type in {"B+", "C+"}:
        ordered = sorted(
            profiles,
            key=lambda item: (
                0 if str(item.get("side_class", "")) == "M" else 1,
                _safe_int(item.get("connector_node_id"), 10**9),
            ),
        )
    if class_type == "E+":
        ordered = sorted(
            profiles,
            key=lambda item: (
                0 if str(item.get("side_class", "")) == "N" else 1,
                _safe_int(item.get("connector_node_id"), 10**9),
            ),
        )
    left_labels, left_ids = _segment(ordered[0])
    right_labels, right_ids = _segment(ordered[1]) if len(ordered) > 1 else ([], [])
    labels = list(left_labels) + [_node_label(center)] + list(reversed(right_labels))
    node_ids = list(left_ids) + [int(center.global_id)] + list(reversed(right_ids))
    open_left = class_type in {"D+", "E+"}
    open_right = class_type in {"B+", "D+"}
    preferred = "flex_chain" if class_type in {"A+", "B+", "D+"} else "side_chain"
    (
        ring_backbone_labels,
        ring_backbone_node_ids,
        ring_backbone_kind,
        ring_backbone_span,
        ring_template_roles,
        ring_roles,
        evidence,
    ) = _open_basic_ring_projection(
        class_type=class_type,
        labels=labels,
        node_ids=node_ids,
        open_left=bool(open_left),
        open_right=bool(open_right),
        evidence=evidence,
    )
    ring_anchor_profile = {"left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")), "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else ""))}
    main_kind = _main_kind_for_class(class_type)
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
    capability_evidence = dict(evidence.get("assembly_capabilities", {}) or {})
    # Product preference is an ordering hint; explicit classification
    # capabilities preserve legal flex/side/ring fallback routes.
    can_flex = bool(capability_evidence.get(
        "flex_allowed", preferred == "flex_chain" or class_type in {"C+"}
    ))
    can_side = bool(capability_evidence.get(
        "side_allowed",
        preferred == "side_chain" or class_type in {"B+", "D+", "E+"},
    ))
    can_ring = bool(capability_evidence.get("ring_allowed", bool(ring_roles))) and bool(ring_roles)
    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=class_type,
        topology_template=str(fragment.topology_template),
        source_node_ids=list(fragment.source_node_ids or []),
        main_labels=labels,
        main_node_ids=node_ids,
        canonical_labels=_canonical_labels_for_class(class_type, labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=len(left_labels),
        unresolved_branch_kinds=[],
        actual_branch_stubs=[],
        fixed_connection_contracts=[
            dict(item or {})
            for item in list(evidence.get("fixed_connection_contracts", []) or [])
            if dict(item or {})
        ],
        main_kind=main_kind,
        preferred_product_type=preferred,
        ring_roles=ring_roles,
        ring_backbone_labels=ring_backbone_labels,
        ring_backbone_node_ids=ring_backbone_node_ids,
        ring_backbone_kind=ring_backbone_kind,
        ring_backbone_span=ring_backbone_span,
        ring_like24_units=_ring_like24_units_from_evidence(fragment),
        ring_anchor_profile=ring_anchor_profile,
        ring_template_roles=ring_template_roles,
        flex_reconcile_roles=flex_reconcile_roles,
        flex_reconcile_priority=int(flex_reconcile_priority),
        can_flex=can_flex,
        can_side=can_side,
        can_ring=can_ring,
        target_roles=sorted({
            *(["flex_chain"] if can_flex else []),
            *(["side_chain"] if can_side else []),
            *(["fatty_ring"] if can_ring else []),
        }),
        evidence=evidence,
        diagnostics={"connector_profiles": profiles},
    )


def _neighbor_semantic_dict(center: _NodeV3, neighbor: _NodeV3) -> Dict[str, Any]:
    fixed_ids = {int(x) for x in list(getattr(center, "fixed_hop1_ids", set()) or set())}
    reverse_fixed_ids = {
        int(x) for x in list(getattr(neighbor, "fixed_hop1_ids", set()) or set())
    }
    center_id = int(getattr(center, "global_id", -1))
    neighbor_id = int(getattr(neighbor, "global_id", -1))
    return {
        "node_id": neighbor_id,
        "su_type": int(getattr(neighbor, "su_type", -1)),
        "subtype": _node_label(neighbor),
        "is_fixed": bool(neighbor_id in fixed_ids and center_id in reverse_fixed_ids),
    }


def _node_has_complete_materialized_degree(
    node: Optional[_NodeV3],
    nodes_by_id: Dict[int, _NodeV3],
    *,
    materialized_neighbor_ids: Iterable[int] = (),
) -> bool:
    if node is None:
        return False
    target_degree = _safe_int(getattr(node, "target_hop1_degree", 0), 0)
    if target_degree <= 0:
        return False
    explicit_neighbors = {
        int(value) for value in list(materialized_neighbor_ids or [])
    }
    materialized_degree = sum(
        1
        for raw_neighbor_id in list(getattr(node, "hop1_ids", []) or [])
        if (
            (neighbor_id := _safe_int(raw_neighbor_id, -1)) in explicit_neighbors
            or (
                (neighbor := nodes_by_id.get(int(neighbor_id))) is not None
                and _nodes_have_fixed_edge(node, neighbor)
            )
        )
    )
    return int(materialized_degree) >= int(target_degree)


def _make_custom_branch_stub(
    *,
    topology_notation: str,
    node_id_notation: str,
    node_id: Optional[int] = None,
    mode: str,
) -> BranchStub:
    source_node_ids = [int(node_id)] if node_id is not None else []
    return BranchStub(
        topology_notation=str(topology_notation),
        node_id_notation=str(node_id_notation),
        source_node_ids=source_node_ids,
        branch_kind="actual" if node_id is not None else "abstract_open",
        mode=str(mode),
    )


def _build_like24_branch_maps_for_center(
    center: _NodeV3,
    *,
    exclude_ids: Set[int],
    nodes_by_id: Dict[int, _NodeV3],
) -> Tuple[List[BranchStub], List[BranchStub], List[int]]:
    abstract_stubs: List[BranchStub] = []
    real_stubs: List[BranchStub] = []
    suppressed_support_node_ids: List[int] = []
    for node_id in sorted(int(x) for x in list(getattr(center, "hop1_ids", []) or [])):
        if int(node_id) in exclude_ids:
            continue
        neighbor = nodes_by_id.get(int(node_id))
        if neighbor is None:
            continue
        semantics = _neighbor_semantic_dict(center, neighbor)
        label = _node_label(neighbor)
        if int(getattr(neighbor, "su_type", -1)) == 11:
            suppressed_support_node_ids.append(int(node_id))
            continue
        if bool(semantics.get("is_fixed", False)):
            continue
        suppressed_support_node_ids.append(int(node_id))
    return abstract_stubs, real_stubs, sorted({int(x) for x in suppressed_support_node_ids})


def project_like24_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    class_type = str(fragment.class_type)
    evidence = dict(fragment.evidence or {})
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    center_id = int(center.global_id)
    center_ids = [
        int(node_id)
        for node_id in list(evidence.get("center_node_ids", []) or [])
        if nodes_by_id.get(_safe_int(node_id, -1)) is not None
    ] or [center_id]
    center_ids = list(dict.fromkeys(
        list(center_ids)
        + [
            int(node_id)
            for node_id in list(
                evidence.get("center_atomic_group_node_ids", [])
                or (
                    evidence.get("atomic_group_node_ids", [])
                    if center_id in {
                        int(value)
                        for value in list(evidence.get("atomic_group_node_ids", []) or [])
                    }
                    else []
                )
                or []
            )
            if nodes_by_id.get(_safe_int(node_id, -1)) is not None
        ]
    ))
    center_nodes = {
        int(node_id): nodes_by_id.get(int(node_id))
        for node_id in center_ids
        if nodes_by_id.get(int(node_id)) is not None
    }
    profiles = [dict(x or {}) for x in list(evidence.get("connector_profiles", []) or [])]
    fixed_component_node_ids: Set[int] = set(int(value) for value in list(fragment.source_node_ids or []))
    primary = dict(evidence.get("selected_primary_connector", {}) or {})
    secondary_profiles = [dict(x or {}) for x in list(evidence.get("secondary_connector_profiles", []) or [])]
    fixed_connection_contracts = [
        dict(item or {})
        for item in list(evidence.get("fixed_connection_contracts", []) or [])
        if dict(item or {})
    ]
    primary_connector_id = _safe_int(primary.get("connector_node_id"), -1)
    primary_contract = next((
        dict(item or {})
        for item in fixed_connection_contracts
        if _safe_int(dict(item or {}).get("connector_node_id"), -1) == primary_connector_id
    ), {})
    direct_neighbors = [
        nodes_by_id.get(_safe_int(node_id, -1))
        for node_id in list(getattr(center, "hop1_ids", []) or [])
    ]
    direct_neighbors = [node for node in direct_neighbors if node is not None]
    eleven_neighbors = sorted(
        [
            node
            for node in direct_neighbors
            if int(getattr(node, "su_type", -1)) == 11
            and int(getattr(node, "global_id", -1)) in fixed_component_node_ids
        ],
        key=lambda item: int(item.global_id),
    )
    connector_id = _safe_int(primary.get("connector_node_id"), -1)
    connector_node = nodes_by_id.get(connector_id)
    remaining_node_ids = [
        _safe_int(x, -1)
        for x in list(primary_contract.get("base_path_node_ids", []) or [])
        if _safe_int(x, -1) != int(connector_id)
    ]
    remaining_nodes = [nodes_by_id.get(node_id) for node_id in remaining_node_ids if nodes_by_id.get(node_id) is not None]
    raw_su24_role = str(evidence.get("raw_su24_role", "") or "")
    raw_su24_profile = dict(evidence.get("raw_su24_profile", {}) or {})
    fixed_branch_double_open_flex = bool(
        evidence.get("fixed_branch_double_open_flex", False)
    )
    terminal_28_ring_preferred_branch = bool(
        evidence.get(
            "terminal_28_ring_preferred_branch",
            evidence.get("terminal_28_ring_only_branch", False),
        )
    )
    labels: List[str] = []
    node_ids: List[int] = []
    open_left = False
    open_right = False
    preferred = "flex_chain"
    real_degree_complete_center_ids: Set[int] = set()
    exclude_direct_ids: Set[int] = {center_id}
    exclude_by_center: Dict[int, Set[int]] = {
        int(node_id): {int(node_id)}
        for node_id in center_nodes.keys()
    }
    raw_terminal_branch_ids: List[int] = []

    if raw_su24_role in {"A", "B", "C", "D"}:
        cap_ids = [
            int(value)
            for value in list(raw_su24_profile.get("cap_node_ids", []) or [])
            if nodes_by_id.get(_safe_int(value, -1)) is not None
            and int(value) in fixed_component_node_ids
        ]
        terminal_ids = [
            int(value)
            for value in list(raw_su24_profile.get("terminal_node_ids", []) or [])
            if nodes_by_id.get(_safe_int(value, -1)) is not None
            and int(value) in fixed_component_node_ids
        ]
        if raw_su24_role in {"A", "B"}:
            labels = (
                [_node_label(nodes_by_id[cap_ids[0]])] if cap_ids else []
            ) + [_node_label(center)]
            node_ids = (
                [int(cap_ids[0])] if cap_ids else []
            ) + [center_id]
            open_left = not bool(cap_ids)
            open_right = True
            if raw_su24_role == "B" and terminal_ids:
                raw_terminal_branch_ids = [int(terminal_ids[0])]
        else:
            labels = [_node_label(center)]
            node_ids = [center_id]
            open_left = True
            open_right = True
            if raw_su24_role == "D" and terminal_ids:
                raw_terminal_branch_ids = [int(terminal_ids[0])]
        raw_terminal_branch_ids = [
            int(value)
            for value in list(dict.fromkeys(raw_terminal_branch_ids))
            if int(value) in fixed_component_node_ids
            and int(value) not in set(int(node_id) for node_id in node_ids)
        ]
        preferred = "ring_candidate"
        exclude_direct_ids.update(int(value) for value in node_ids)
        exclude_by_center.setdefault(center_id, {center_id}).update(int(value) for value in node_ids)
        exclude_by_center.setdefault(center_id, {center_id}).update(
            raw_terminal_branch_ids
        )
        for terminal_id in terminal_ids:
            if int(terminal_id) not in set(node_ids):
                continue
            exclude_by_center.setdefault(center_id, {center_id}).add(int(terminal_id))
    elif class_type in {"24_A", "24_B"}:
        if eleven_neighbors:
            labels.append(_node_label(eleven_neighbors[0]))
            node_ids.append(int(eleven_neighbors[0].global_id))
            exclude_direct_ids.add(int(eleven_neighbors[0].global_id))
            exclude_by_center.setdefault(center_id, {center_id}).add(int(eleven_neighbors[0].global_id))
        labels.append(_node_label(center))
        node_ids.append(center_id)
        open_left = not bool(eleven_neighbors)
        open_right = True
        preferred = "ring_candidate"
    elif class_type in {"24_C", "24_D"}:
        atomic_group_node_ids = [
            int(value)
            for value in list(evidence.get("atomic_group_node_ids", []) or [])
            if nodes_by_id.get(int(value)) is not None
        ]
        if int(getattr(center, "su_type", -1)) == 14 and atomic_group_node_ids:
            ordered_atomic_ids = sorted(
                atomic_group_node_ids,
                key=lambda node_id: (
                    0 if int(getattr(nodes_by_id[int(node_id)], "su_type", -1)) == 14 else 1,
                    int(node_id),
                ),
            )
            labels = [_node_label(nodes_by_id[node_id]) for node_id in ordered_atomic_ids]
            node_ids = list(ordered_atomic_ids)
            for atomic_id in ordered_atomic_ids:
                exclude_by_center.setdefault(int(atomic_id), {int(atomic_id)}).update(ordered_atomic_ids)
        else:
            labels = [_node_label(center)]
            node_ids = [center_id]
        open_left = True
        open_right = True
        preferred = "ring_candidate"
    elif class_type in {"24_AB_M", "24_AB_N", "24_AB_P"}:
        if eleven_neighbors:
            labels.append(_node_label(eleven_neighbors[0]))
            node_ids.append(int(eleven_neighbors[0].global_id))
            exclude_direct_ids.add(int(eleven_neighbors[0].global_id))
            exclude_by_center.setdefault(center_id, {center_id}).add(int(eleven_neighbors[0].global_id))
        labels.append(_node_label(center))
        node_ids.append(center_id)
        if connector_node is not None:
            labels.append(_node_label(connector_node))
            node_ids.append(connector_id)
            exclude_direct_ids.add(connector_id)
            exclude_by_center.setdefault(center_id, {center_id}).add(connector_id)
        for node in remaining_nodes:
            labels.append(_node_label(node))
            node_ids.append(int(node.global_id))
            exclude_direct_ids.add(int(node.global_id))
            exclude_by_center.setdefault(center_id, {center_id}).add(int(node.global_id))
        open_left = not bool(eleven_neighbors)
        open_right = class_type == "24_AB_N"
        preferred = "flex_chain" if class_type == "24_AB_M" else ("side_chain" if class_type == "24_AB_P" else "flex_chain")
        if class_type == "24_AB_P":
            labels = list(reversed(labels))
            node_ids = list(reversed(node_ids))
            open_left, open_right = bool(open_right), bool(open_left)
    elif class_type in {"24_CD_M", "24_CD_P", "24_CD_N"}:
        labels = [_node_label(center)]
        node_ids = [center_id]
        if connector_node is not None:
            labels.append(_node_label(connector_node))
            node_ids.append(connector_id)
            exclude_direct_ids.add(connector_id)
            exclude_by_center.setdefault(center_id, {center_id}).add(connector_id)
        for node in remaining_nodes:
            labels.append(_node_label(node))
            node_ids.append(int(node.global_id))
            exclude_direct_ids.add(int(node.global_id))
            exclude_by_center.setdefault(center_id, {center_id}).add(int(node.global_id))
        open_left = True
        open_right = class_type == "24_CD_N"
        preferred = "side_chain" if class_type == "24_CD_P" else "flex_chain"
    elif class_type == "24_CD_M_FULL":
        def _segment(profile: Dict[str, Any]) -> Tuple[List[str], List[int]]:
            seg_labels: List[str] = []
            seg_ids: List[int] = []
            for node_id in list(profile.get("remaining_node_ids", []) or []):
                node = nodes_by_id.get(_safe_int(node_id, -1))
                if node is not None:
                    seg_labels.append(_node_label(node))
                    seg_ids.append(int(node.global_id))
            connector_id_i = _safe_int(profile.get("connector_node_id"), -1)
            connector = nodes_by_id.get(connector_id_i)
            if connector is not None:
                seg_labels.append(_node_label(connector))
                seg_ids.append(int(connector.global_id))
            return seg_labels, seg_ids

        ordered = sorted(profiles, key=lambda item: _safe_int(item.get("connector_node_id"), 10**9))
        left_labels, left_ids = _segment(ordered[0]) if ordered else ([], [])
        right_labels, right_ids = _segment(ordered[1]) if len(ordered) > 1 else ([], [])
        labels = list(left_labels) + [_node_label(center)] + list(reversed(right_labels))
        node_ids = list(left_ids) + [center_id] + list(reversed(right_ids))
        exclude_direct_ids.update(int(x) for x in node_ids if x is not None)
        exclude_by_center.setdefault(center_id, {center_id}).update(int(x) for x in node_ids if x is not None)
        preferred = "flex_chain"
    elif class_type in {"24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD"}:
        labels = []
        node_ids = []
        ordered_center_ids = [int(node_id) for node_id in list(center_ids or []) if int(node_id) in center_nodes]
        left_center_id = ordered_center_ids[0] if ordered_center_ids else center_id
        right_center_id = ordered_center_ids[1] if len(ordered_center_ids) > 1 else center_id
        left_center = center_nodes.get(int(left_center_id))
        right_center = center_nodes.get(int(right_center_id))
        left_class = str(evidence.get("left_class", "") or "")
        right_class = str(evidence.get("right_class", "") or "")
        bridge_connector_id = _safe_int(evidence.get("bridge_connector_node_id"), -1)
        bridge_connector = nodes_by_id.get(int(bridge_connector_id)) if bridge_connector_id >= 0 else None

        if left_center is not None and left_class in {"24_A", "24_B", "24_AB_N"}:
            left_anchor = _first_neighbor_by_su(
                left_center,
                su_types={11},
                exclude_ids={int(left_center_id), int(right_center_id), int(bridge_connector_id)},
                nodes_by_id=nodes_by_id,
            )
            if (
                left_anchor is not None
                and int(left_anchor.global_id) in fixed_component_node_ids
            ):
                labels.append(_node_label(left_anchor))
                node_ids.append(int(left_anchor.global_id))
                exclude_by_center.setdefault(int(left_center_id), {int(left_center_id)}).add(int(left_anchor.global_id))
        if left_center is not None:
            labels.append(_node_label(left_center))
            node_ids.append(int(left_center.global_id))
        if bridge_connector is not None:
            labels.append(_node_label(bridge_connector))
            node_ids.append(int(bridge_connector.global_id))
            exclude_by_center.setdefault(int(left_center_id), {int(left_center_id)}).add(int(bridge_connector.global_id))
            exclude_by_center.setdefault(int(right_center_id), {int(right_center_id)}).add(int(bridge_connector.global_id))
        if right_center is not None and int(right_center_id) != int(left_center_id):
            labels.append(_node_label(right_center))
            node_ids.append(int(right_center.global_id))
        if right_center is not None and class_type == "24_AB_N_24_AB" and right_class in {"24_A", "24_B", "24_AB_N"}:
            right_anchor = _first_neighbor_by_su(
                right_center,
                su_types={11},
                exclude_ids={int(left_center_id), int(right_center_id), int(bridge_connector_id)},
                nodes_by_id=nodes_by_id,
            )
            if (
                right_anchor is not None
                and int(right_anchor.global_id) in fixed_component_node_ids
            ):
                labels.append(_node_label(right_anchor))
                node_ids.append(int(right_anchor.global_id))
                exclude_by_center.setdefault(int(right_center_id), {int(right_center_id)}).add(int(right_anchor.global_id))
        for cid in center_nodes.keys():
            exclude_by_center.setdefault(int(cid), {int(cid)}).update(int(x) for x in node_ids if x is not None)
        has_left_anchor = bool(
            node_ids and left_center_id in node_ids and node_ids.index(left_center_id) > 0
        )
        has_right_anchor = bool(
            node_ids and right_center_id in node_ids and node_ids.index(right_center_id) < len(node_ids) - 1
        )
        open_left = class_type == "24_CD_N_24_CD" or not has_left_anchor
        open_right = class_type in {"24_AB_N_24_CD", "24_CD_N_24_CD"} or not has_right_anchor
        preferred = (
            "flex_chain"
            if class_type == "24_AB_N_24_AB"
            else "ring_candidate"
        )
    else:
        labels = [_node_label(center)]
        node_ids = [center_id]

    if preferred == "side_chain" and not bool(open_left or open_right):
        evidence["external_pending_profile"] = {
            "external_pending_kind": "explicit_side",
            "external_pending_reason": "explicit_side_closure",
        }

    for cid, center_node in center_nodes.items():
        if _node_has_complete_materialized_degree(center_node, nodes_by_id):
            real_degree_complete_center_ids.add(int(cid))

    h_branch_contract_counts_by_center: Counter = Counter()
    for raw_contract in list(fixed_connection_contracts or []):
        contract = dict(raw_contract or {})
        role_candidates = {
            str(value)
            for value in list(contract.get("role_candidates", []) or [])
        }
        if "branch_only" not in role_candidates:
            continue
        contract_center_id = _safe_int(contract.get("center_node_id"), center_id)
        h_branch_contract_counts_by_center[int(contract_center_id)] += 1

    exclude_direct_ids.update(int(node_id) for node_id in list(node_ids or []) if node_id is not None)
    exclude_by_center.setdefault(center_id, {center_id}).update(int(node_id) for node_id in list(node_ids or []) if node_id is not None)

    branch_stubs_by_anchor: Dict[int, List[BranchStub]] = {}
    real_branch_stubs_by_anchor: Dict[int, List[BranchStub]] = {}
    suppressed_support_node_ids: List[int] = []
    primary_anchor = max(0, labels.index(_node_label(center)) if _node_label(center) in labels else len(labels) // 2)
    anchor_by_center_node_id: Dict[int, int] = {}
    for cid, center_node in center_nodes.items():
        if center_node is None:
            continue
        if int(cid) not in [int(node_id) for node_id in list(node_ids or [])]:
            continue
        anchor = list(node_ids).index(int(cid))
        anchor_by_center_node_id[int(cid)] = int(anchor)
        abstract_stubs, real_stubs, suppressed_ids = _build_like24_branch_maps_for_center(
            center_node,
            exclude_ids=set(exclude_by_center.get(int(cid), {int(cid)})),
            nodes_by_id=nodes_by_id,
        )
        if int(cid) == int(center_id) and raw_terminal_branch_ids:
            branch_stub = BranchStub(
                topology_notation="-" + "-".join(
                    _node_label(nodes_by_id[int(node_id)])
                    for node_id in raw_terminal_branch_ids
                ),
                node_id_notation="-" + "-".join(
                    str(int(node_id)) for node_id in raw_terminal_branch_ids
                ),
                source_node_ids=[
                    int(node_id) for node_id in raw_terminal_branch_ids
                ],
                branch_kind="actual_raw_su24_terminal",
                mode=f"raw_su24_{raw_su24_role}_fixed_terminal",
            )
            abstract_stubs.append(branch_stub)
            real_stubs.append(branch_stub)
        if abstract_stubs:
            branch_stubs_by_anchor[int(anchor)] = list(abstract_stubs)
        if real_stubs:
            real_branch_stubs_by_anchor[int(anchor)] = list(real_stubs)
        suppressed_support_node_ids.extend(int(x) for x in list(suppressed_ids or []))
        if int(cid) == int(center_id):
            primary_anchor = int(anchor)

    h_branch_satisfaction_counts = Counter(h_branch_contract_counts_by_center)
    for center_node_id, anchor in anchor_by_center_node_id.items():
        h_branch_satisfaction_counts[int(center_node_id)] += len(
            list(real_branch_stubs_by_anchor.get(int(anchor), []) or [])
        )

    canonical_branch_slots: List[TopologyClosureSlot] = []
    for raw_slot in list(fragment.branch_contracts or []):
        slot = normalize_slot(raw_slot)
        if not slot.is_complete():
            canonical_branch_slots.append(slot)
    required_branch_map_raw: Dict[str, List[str]] = {}
    if canonical_branch_slots:
        for slot in canonical_branch_slots:
            anchor_node_id = int(slot.anchor_node_id)
            if anchor_node_id not in anchor_by_center_node_id:
                raise ValueError(
                    f"like24_branch_contract_anchor_invalid:{int(fragment.fragment_id)}:"
                    f"{anchor_node_id}"
                )
            required_branch_map_raw.setdefault(str(anchor_node_id), []).append(
                str(slot.kind)
            )
    unresolved_branch_kinds_by_anchor: Dict[int, List[str]] = {}
    if required_branch_map_raw:
        ordered_center_ids = list(center_nodes.keys())
        remaining_contracts = Counter(h_branch_satisfaction_counts)
        normalized_required_branch_map: Dict[str, List[str]] = {}
        for raw_anchor, values in required_branch_map_raw.items():
            anchor_key = str(raw_anchor)
            mapped_anchor: Optional[int] = None
            if anchor_key == "center":
                mapped_anchor = int(primary_anchor)
            elif _safe_int(raw_anchor, -1) in anchor_by_center_node_id:
                mapped_anchor = int(anchor_by_center_node_id[_safe_int(raw_anchor, -1)])
            else:
                anchor_index = _safe_int(raw_anchor, -1)
                if anchor_index >= 0 and anchor_index < len(ordered_center_ids):
                    mapped_anchor = int(anchor_by_center_node_id.get(int(ordered_center_ids[anchor_index]), primary_anchor))
            if mapped_anchor is None:
                mapped_anchor = int(primary_anchor)
            normalized_values = [str(value) for value in list(values or [])]
            logical_center_id = (
                int(center_id)
                if anchor_key == "center"
                else _safe_int(raw_anchor, -1)
            )
            if int(logical_center_id) in real_degree_complete_center_ids:
                normalized_values = [
                    value for value in normalized_values if str(value) != "H"
                ]
            else:
                retained_values: List[str] = []
                for value in normalized_values:
                    if (
                        str(value) == "H"
                        and int(remaining_contracts.get(int(logical_center_id), 0)) > 0
                    ):
                        remaining_contracts[int(logical_center_id)] -= 1
                        continue
                    retained_values.append(str(value))
                normalized_values = retained_values
            normalized_required_branch_map[str(raw_anchor)] = list(normalized_values)
            unresolved_branch_kinds_by_anchor[int(mapped_anchor)] = normalized_values
        evidence["required_branch_kinds_by_anchor"] = normalized_required_branch_map
    else:
        unresolved_branch_kinds_by_anchor = {}
    primary_real_stubs = list(real_branch_stubs_by_anchor.get(int(primary_anchor), []) or [])
    unresolved_branch_kinds = list(unresolved_branch_kinds_by_anchor.get(int(primary_anchor), []) or [])
    mainline_fixed_connector_ids = {
        int(node_id)
        for node_id in list(node_ids or [])
        if node_id is not None
    }
    branch_closure_contracts_by_anchor: Dict[int, List[Dict[str, Any]]] = {}
    for contract in list(fixed_connection_contracts or []):
        connector_node_id = _safe_int(dict(contract or {}).get("connector_node_id"), -1)
        if connector_node_id < 0 or connector_node_id in mainline_fixed_connector_ids:
            continue
        contract_center_id = _safe_int(dict(contract or {}).get("center_node_id"), center_id)
        contract_anchor = int(anchor_by_center_node_id.get(int(contract_center_id), primary_anchor))
        branch_closure_contracts_by_anchor.setdefault(int(contract_anchor), [])
        branch_closure_contracts_by_anchor[int(contract_anchor)].append(dict(contract or {}))

    ring_roles = _ring_roles_from_evidence(fragment)
    main_kind = _main_kind_for_class(class_type)
    ring_backbone_labels: Dict[str, List[str]] = {}
    ring_backbone_node_ids: Dict[str, List[Optional[int]]] = {}
    ring_backbone_kind: Dict[str, str] = {}
    ring_backbone_span: Dict[str, int] = {}
    ring_anchor_profile: Dict[str, Any] = {
        "left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")),
        "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else "")),
        "touches_aromatic_anchor": bool(evidence.get("touches_aromatic_anchor", False)),
        "fixed_anchor_eligible": bool(evidence.get("fixed_anchor_eligible", False)),
        "bridgehead_role": str(evidence.get("bridgehead_role", "")),
    }
    ring_template_roles: Dict[str, List[str]] = {
        str(kind): [str(item) for item in list(values or [])]
        for kind, values in dict(evidence.get("ring_path_capabilities", {}) or {}).items()
    }
    # Contract eligibility is the authoritative ring standard.  The flat
    # ``ring_allowed_skeletons`` field is retained as a compatibility fallback
    # for older classifier payloads, but must not widen an explicit contract.
    ring_contracts = {
        str(key): dict(value or {})
        for key, value in dict(evidence.get("ring_contracts", {}) or {}).items()
    }
    if ring_contracts:
        allowed_ring_kinds = {
            _skeleton_allowed_name(str(kind))
            for kind, contract in ring_contracts.items()
            if bool(dict(contract or {}).get("eligible", False))
            and _skeleton_allowed_name(str(kind)) in {"side", "upper_lower", "fused"}
        }
    else:
        allowed_ring_kinds = {
            str(kind)
            for kind in list(evidence.get("ring_allowed_skeletons", []) or [])
            if str(kind) in {"side", "upper_lower", "fused"}
        }
    ordered_backbone = _ordered_ring_backbone_map(
        labels=labels,
        node_ids=node_ids,
        allowed_kinds=sorted(allowed_ring_kinds),
    )
    for kind in sorted(allowed_ring_kinds):
        ring_backbone_labels[str(kind)] = [str(value) for value in list(ordered_backbone.get(str(kind), []) or [])]
        ring_backbone_node_ids[str(kind)] = _normalize_ring_backbone_node_ids(ordered_backbone.get(f"{str(kind)}_ids", []))
        ring_backbone_span[str(kind)] = len(list(ring_backbone_labels[str(kind)] or []))
        if str(kind) == "side":
            ring_backbone_kind[str(kind)] = "side_path"
        elif str(kind) == "upper_lower":
            ring_backbone_kind[str(kind)] = "upper_path"
        else:
            ring_backbone_kind[str(kind)] = "fused_path"
    # A fixed terminal SU28 branch on single19d3 is the 24_B+N ring form.  Its
    # aromatic SU11 endpoint is a pool resource, so expose it only on the ring
    # backbone and let ring materialization claim its concrete ID atomically.
    ring_open_pool_positions: Dict[str, Dict[str, int]] = {}
    if (
        (fixed_branch_double_open_flex or terminal_28_ring_preferred_branch)
        and bool(evidence.get("ring_branch_capable", False))
        and bool(evidence.get("ring_branch_open_aromatic_endpoint", False))
        and str(class_type) in {"24_A", "24_B"}
    ):
        for skeleton_kind in ("side", "fused"):
            if skeleton_kind not in allowed_ring_kinds:
                continue
            ring_backbone_labels[skeleton_kind] = ["11"] + list(
                ring_backbone_labels.get(skeleton_kind, []) or []
            )
            ring_backbone_node_ids[skeleton_kind] = [None] + list(
                ring_backbone_node_ids.get(skeleton_kind, []) or []
            )
            ring_backbone_span[skeleton_kind] = len(
                ring_backbone_labels[skeleton_kind]
            )
            ring_open_pool_positions[skeleton_kind] = {"0": 11}
    if "upper_lower" in allowed_ring_kinds and class_type == "24_A" and len(node_ids) >= 2:
        ring_backbone_labels["upper_lower"] = [str(labels[-1])]
        ring_backbone_node_ids["upper_lower"] = [None if node_ids[-1] is None else int(node_ids[-1])]
        ring_backbone_span["upper_lower"] = 1
        ring_backbone_kind["upper_lower"] = "upper_fixed_anchor"
    ring_like24_units = _ring_like24_units_from_evidence(fragment)
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
    if fixed_branch_double_open_flex:
        preferred = "flex_chain"
        # A double-open branch may be a flex source and a ring segment at the
        # same time.  The classifier records that capability explicitly;
        # changing the preferred flex route must not erase the ring contract.
        if bool(evidence.get("ring_branch_capable", False)) and not ring_roles:
            ring_roles = [
                str(role)
                for role in list(evidence.get("ring_branch_roles", []) or [])
                if str(role)
            ]
        flex_reconcile_roles = sorted({
            *flex_reconcile_roles,
            "d_like_open",
            "double_end_flex_source",
            "fixed_branch_double_open_flex",
            "flex_preferred",
        })
        flex_reconcile_priority = 0
    elif terminal_28_ring_preferred_branch:
        # Ring remains the preferred route, but the two open boundaries are a
        # legitimate D-like side-chain fallback (F1 + F2) after ring exhaustion.
        preferred = "ring_candidate"
        flex_reconcile_roles = sorted({
            *(
                role
                for role in flex_reconcile_roles
                if str(role) not in {"flex_preferred", "single_end_flex_source"}
            ),
            "d_like_open",
            "double_end_flex_source",
            "like24_open",
        })
        flex_reconcile_priority = 40
        evidence["external_pending_profile"] = {
            "external_pending_kind": "d_like_open",
            "external_pending_reason": "ring_preferred_side_fallback",
        }
    if class_type in {"24_AB_N_24_CD", "24_CD_N_24_CD"}:
        # Paired mixed/CD forms are ring-first with a side fallback.  Their
        # B/D-like endpoint role is retained for side closure, but they are not
        # supplemental flex sources.
        flex_reconcile_roles = sorted({
            role
            for role in flex_reconcile_roles
            if str(role) not in {
                "flex_preferred",
                "single_end_flex_source",
                "double_end_flex_source",
            }
        })

    canonical_mainline_slots: List[TopologyClosureSlot] = []
    for raw_slot in list(fragment.open_mainline_slots or []):
        slot = normalize_slot(raw_slot)
        if not slot.is_complete():
            canonical_mainline_slots.append(slot)
    if canonical_mainline_slots or bool(evidence.get("edge_semantics_finalized", False)):
        # Class names describe the broad topology family, but the classifier's
        # canonical slots are the authority for the concrete fixed/open
        # boundary.  In particular, 24_CD_N can have one open boundary at the
        # fixed-path endpoint instead of two open mainline ends.
        open_left = False
        open_right = False
        index_by_node_id = {
            int(node_id): int(index)
            for index, node_id in enumerate(list(node_ids or []))
            if node_id is not None
        }
        for slot in canonical_mainline_slots:
            role = str(slot.anchor_role or "")
            anchor_index = index_by_node_id.get(int(slot.anchor_node_id))
            if anchor_index is None:
                raise ValueError(
                    f"like24_mainline_contract_anchor_missing:{int(fragment.fragment_id)}:"
                    f"{int(slot.anchor_node_id)}"
                )
            if role.startswith("left"):
                if len(node_ids) == 1:
                    open_left = True
                elif int(anchor_index) == 0:
                    open_left = True
                elif int(anchor_index) == len(node_ids) - 1:
                    open_right = True
                else:
                    raise ValueError(
                        f"like24_mainline_anchor_not_endpoint:"
                        f"{int(fragment.fragment_id)}:{int(slot.anchor_node_id)}"
                    )
                continue
            if role.startswith("right"):
                if len(node_ids) == 1:
                    open_right = True
                elif int(anchor_index) == len(node_ids) - 1:
                    open_right = True
                elif int(anchor_index) == 0:
                    open_left = True
                else:
                    raise ValueError(
                        f"like24_mainline_anchor_not_endpoint:"
                        f"{int(fragment.fragment_id)}:{int(slot.anchor_node_id)}"
                    )
                continue
            if anchor_index == 0 and anchor_index != len(node_ids) - 1:
                open_left = True
            elif anchor_index == len(node_ids) - 1 and anchor_index != 0:
                open_right = True
            elif anchor_index == 0:
                if not open_left:
                    open_left = True
                else:
                    open_right = True
            else:
                raise ValueError(
                    f"like24_mainline_contract_anchor_invalid:{int(fragment.fragment_id)}:"
                    f"{int(slot.anchor_node_id)}"
                )
    assembly_capabilities = dict(evidence.get("assembly_capabilities", {}) or {})
    legacy_can_flex = bool(
        "flex_preferred" in set(flex_reconcile_roles)
        or preferred == "flex_chain"
    )
    legacy_can_side = bool(preferred == "side_chain")
    legacy_can_ring = bool(ring_roles) and not str(
        evidence.get("ring_block_reason", "")
    )
    can_flex = bool(
        assembly_capabilities.get("flex_allowed", legacy_can_flex)
        or fixed_branch_double_open_flex
    )
    can_side = bool(
        assembly_capabilities.get(
            "side_allowed",
            evidence.get("side_allowed", legacy_can_side),
        )
    )
    can_ring = bool(
        assembly_capabilities.get("ring_allowed", legacy_can_ring)
        and not str(evidence.get("ring_block_reason", ""))
    )

    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=class_type,
        topology_template=str(fragment.topology_template),
        source_node_ids=sorted({
            int(value)
            for value in list(getattr(fragment, "claimed_node_ids", []) or getattr(fragment, "source_node_ids", []) or [])
        }),
        context_node_ids=sorted({
            int(value)
            for value in list(getattr(fragment, "context_node_ids", []) or [])
        }),
        main_labels=labels,
        main_node_ids=node_ids,
        canonical_labels=_canonical_labels_for_class(class_type, labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=int(primary_anchor),
        unresolved_branch_kinds=unresolved_branch_kinds,
        actual_branch_stubs=primary_real_stubs,
        branch_stubs_by_anchor=branch_stubs_by_anchor,
        unresolved_branch_kinds_by_anchor=unresolved_branch_kinds_by_anchor,
        real_branch_stubs_by_anchor=real_branch_stubs_by_anchor,
        fixed_connection_contracts=fixed_connection_contracts,
        branch_closure_contracts_by_anchor=branch_closure_contracts_by_anchor,
        branch_closure_blockers_by_anchor={},
        resolved_fixed_mainline_connector_id=connector_id if connector_id >= 0 else None,
        suppressed_support_node_ids=sorted({int(x) for x in suppressed_support_node_ids}),
        main_kind=main_kind,
        preferred_product_type=preferred,
        ring_roles=ring_roles,
        ring_backbone_labels=ring_backbone_labels,
        ring_backbone_node_ids=ring_backbone_node_ids,
        ring_backbone_kind=ring_backbone_kind,
        ring_backbone_span=ring_backbone_span,
        ring_like24_units=ring_like24_units,
        ring_anchor_profile=ring_anchor_profile,
        ring_template_roles=ring_template_roles,
        flex_reconcile_roles=flex_reconcile_roles,
        flex_reconcile_priority=int(flex_reconcile_priority),
        can_flex=bool(can_flex),
        can_side=bool(can_side),
        can_ring=bool(can_ring),
        target_roles=sorted({
            *(["flex_chain"] if can_flex else []),
            *(["side_chain"] if can_side else []),
            *(["fatty_ring"] if can_ring else []),
        }),
        evidence=evidence,
        diagnostics={
            "connector_profiles": profiles,
            "secondary_connector_profiles": secondary_profiles,
            "realization_modes": list(dict(fragment.evidence or {}).get("realization_modes", []) or []),
            "suppressed_support_node_ids": sorted({int(x) for x in suppressed_support_node_ids}),
            "required_branch_closures_by_anchor_in_ring": dict(dict(fragment.evidence or {}).get("required_branch_closures_by_anchor_in_ring", {}) or {}),
            "ring_slot_capabilities": dict(dict(fragment.evidence or {}).get("ring_slot_capabilities", {}) or {}),
            "ring_open_pool_positions": dict(ring_open_pool_positions),
        },
    )


def _project_su25_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    owned_node_ids = {
        int(value)
        for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])
    }
    neighbors = [
        nodes_by_id.get(_safe_int(node_id, -1))
        for node_id in list(getattr(center, "hop1_ids", []) or [])
        if _safe_int(node_id, -1) in owned_node_ids
    ]
    neighbors = [node for node in neighbors if node is not None]
    fixed_22 = sorted(
        [node for node in neighbors if int(getattr(node, "su_type", -1)) == 22],
        key=lambda item: int(item.global_id),
    )
    elevens = sorted(
        [node for node in neighbors if int(getattr(node, "su_type", -1)) == 11],
        key=lambda item: int(item.global_id),
    )
    labels: List[str] = []
    node_ids: List[int] = []
    open_left = False
    open_right = True
    if str(fragment.class_type) == "25_AB":
        if elevens:
            labels.append(_node_label(elevens[0]))
            node_ids.append(int(elevens[0].global_id))
        else:
            open_left = True
        labels.append(_node_label(center))
        node_ids.append(int(center.global_id))
    else:
        labels.append(_node_label(center))
        node_ids.append(int(center.global_id))
        open_left = True
    ring_backbone_labels: Dict[str, List[str]] = {}
    ring_backbone_node_ids: Dict[str, List[Optional[int]]] = {}
    ring_backbone_kind: Dict[str, str] = {}
    ring_backbone_span: Dict[str, int] = {}
    ring_anchor_profile = {"left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")), "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else ""))}
    ring_template_roles: Dict[str, List[str]] = {}
    main_kind = _main_kind_for_class(str(fragment.class_type))
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
    branch_anchor_index = (
        list(node_ids).index(int(center.global_id))
        if int(center.global_id) in set(int(value) for value in node_ids)
        else max(0, len(labels) - 1)
    )
    fixed_22_stubs = [
        _make_custom_branch_stub(
            topology_notation="-22",
            node_id_notation=f"-{int(node.global_id)}",
            node_id=int(node.global_id),
            mode="existing_fixed22",
        )
        for node in fixed_22[:1]
    ]
    unresolved_branch_kinds = ([] if fixed_22_stubs else ["FIXED22"]) + ["H"]
    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=str(fragment.class_type),
        topology_template=str(fragment.topology_template),
        source_node_ids=list(fragment.source_node_ids or []),
        context_node_ids=list(fragment.context_node_ids or []),
        main_labels=labels,
        main_node_ids=node_ids,
        canonical_labels=_canonical_labels_for_class(str(fragment.class_type), labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=int(branch_anchor_index),
        unresolved_branch_kinds=list(unresolved_branch_kinds),
        actual_branch_stubs=list(fixed_22_stubs),
        branch_stubs_by_anchor={int(branch_anchor_index): list(fixed_22_stubs)},
        unresolved_branch_kinds_by_anchor={int(branch_anchor_index): list(unresolved_branch_kinds)},
        real_branch_stubs_by_anchor={int(branch_anchor_index): list(fixed_22_stubs)},
        real_unresolved_branch_kinds_by_anchor={int(branch_anchor_index): list(unresolved_branch_kinds)},
        branch_closure_contracts_by_anchor={},
        main_kind=main_kind,
        preferred_product_type="flex_chain",
        ring_roles=[],
        ring_backbone_labels=ring_backbone_labels,
        ring_backbone_node_ids=ring_backbone_node_ids,
        ring_backbone_kind=ring_backbone_kind,
        ring_backbone_span=ring_backbone_span,
        ring_like24_units=_ring_like24_units_from_evidence(fragment),
        ring_anchor_profile=ring_anchor_profile,
        ring_template_roles=ring_template_roles,
        flex_reconcile_roles=flex_reconcile_roles,
        flex_reconcile_priority=int(flex_reconcile_priority),
        can_flex=True,
        can_side=True,
        can_ring=False,
        target_roles=["flex_chain", "side_chain"],
        evidence=dict(fragment.evidence or {}),
        diagnostics={
            "real_mainline_support_node_ids": [
                int(value)
                for value in list(node_ids or [])
                if int(value) != int(center.global_id)
            ],
            "real_h_branch_prefix_node_id": None,
            "unassigned_real_neighbor_ids": [],
        },
    )


def _project_atomic_fallback_fragment(
    fragment: TopologyFragment,
    nodes_by_id: Dict[int, _NodeV3],
) -> AssemblyFragmentView:
    node_ids = [int(value) for value in list(fragment.mainline_node_ids or [])]
    if not node_ids:
        raise ValueError(f"atomic fallback fragment {fragment.fragment_id} has no mainline nodes")
    missing_node_ids = [node_id for node_id in node_ids if int(node_id) not in nodes_by_id]
    if missing_node_ids:
        raise ValueError(
            f"atomic fallback fragment {fragment.fragment_id} references missing nodes: {missing_node_ids}"
        )
    labels = [str(label) for label in list(fragment.mainline_labels or [])]
    if len(labels) != len(node_ids):
        labels = [_node_label(nodes_by_id[int(node_id)]) for node_id in node_ids]

    index_by_node_id = {int(node_id): index for index, node_id in enumerate(node_ids)}
    open_left = False
    open_right = False
    for raw_slot in list(fragment.open_mainline_slots or []):
        slot = normalize_slot(raw_slot)
        if slot.is_complete():
            continue
        anchor_index = index_by_node_id.get(int(slot.anchor_node_id))
        if str(slot.anchor_role).startswith("left") or anchor_index == 0:
            open_left = True
        if str(slot.anchor_role).startswith("right") or anchor_index == len(node_ids) - 1:
            open_right = True

    unresolved_branch_kinds_by_anchor: Dict[int, List[str]] = {}
    for raw_slot in list(fragment.branch_contracts or []):
        slot = normalize_slot(raw_slot)
        if slot.is_complete():
            continue
        anchor_index = index_by_node_id.get(int(slot.anchor_node_id))
        if anchor_index is None:
            raise ValueError(
                f"atomic fallback fragment {fragment.fragment_id} has invalid branch anchor "
                f"{slot.anchor_node_id}"
            )
        unresolved_branch_kinds_by_anchor.setdefault(int(anchor_index), []).append(str(slot.kind))

    if open_left and open_right:
        main_kind = "d_like_open"
        preferred = "flex_chain"
        flex_roles = ["d_like_open", "double_end_flex_source", "flex_preferred"]
    elif open_left or open_right:
        main_kind = "e_like_open"
        preferred = "side_chain"
        flex_roles = ["e_like_open"]
    else:
        main_kind = "explicit_flex"
        preferred = "flex_chain"
        flex_roles = ["explicit_flex", "flex_preferred"]

    primary_anchor = min(unresolved_branch_kinds_by_anchor, default=max(0, len(node_ids) // 2))
    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=str(fragment.class_type),
        topology_template=str(fragment.topology_template),
        source_node_ids=sorted({
            int(value)
            for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])
        }),
        context_node_ids=sorted({int(value) for value in list(fragment.context_node_ids or [])}),
        main_labels=list(labels),
        main_node_ids=list(node_ids),
        canonical_labels=list(labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=int(primary_anchor),
        unresolved_branch_kinds=list(unresolved_branch_kinds_by_anchor.get(int(primary_anchor), []) or []),
        unresolved_branch_kinds_by_anchor=unresolved_branch_kinds_by_anchor,
        main_kind=str(main_kind),
        preferred_product_type=str(preferred),
        flex_reconcile_roles=list(flex_roles),
        flex_reconcile_priority=50,
        can_flex=bool(preferred == "flex_chain"),
        can_side=bool(preferred == "side_chain"),
        can_ring=bool(str(fragment.family) == "like24_unsaturated_pair"),
        target_roles=[str(preferred)],
        evidence=dict(fragment.evidence or {}),
        diagnostics={"source_fragment_family": str(fragment.family)},
    )


def project_fragment_views(
    classification: TopologyClassificationResult,
    nodes_by_id: Dict[int, _NodeV3],
) -> List[AssemblyFragmentView]:
    views: List[AssemblyFragmentView] = []
    unsupported = []
    for fragment in list(classification.fragments or []):
        family = str(fragment.family)
        if family == "basic":
            view = _project_basic_fragment(fragment, nodes_by_id)
        elif family == "double_d2_combo":
            view = _project_double_d2_fragment(fragment, nodes_by_id)
        elif family in {"like24", "like24_secondary", "like24_bridge"}:
            view = project_like24_fragment(fragment, nodes_by_id)
        elif family == "su25":
            view = _project_su25_fragment(fragment, nodes_by_id)
        elif family in {"basic_unsaturated_pair", "like24_unsaturated_pair"}:
            view = _project_atomic_fallback_fragment(fragment, nodes_by_id)
        else:
            unsupported.append((int(fragment.fragment_id), family))
            continue
        _route_unrendered_claimed_fixed_components(fragment, view, nodes_by_id)
        _validate_fragment_view_projection(
            fragment,
            view,
            fixed_node_owner_by_id=getattr(
                classification,
                "fixed_node_owner_by_id",
                {},
            ),
            nodes_by_id=nodes_by_id,
        )
        views.append(view)
    if unsupported:
        details = ", ".join(
            f"{fragment_id}:{family}"
            for fragment_id, family in unsupported
        )
        raise ValueError(f"unsupported topology fragment families: {details}")
    fragment_ids = [
        int(fragment.fragment_id)
        for fragment in list(classification.fragments or [])
    ]
    view_ids = [int(view.fragment_id) for view in views]
    if sorted(fragment_ids) != sorted(view_ids):
        raise ValueError(
            "topology fragment projection coverage mismatch: "
            f"fragments={sorted(fragment_ids)}, views={sorted(view_ids)}"
        )
    views.sort(key=lambda item: int(item.fragment_id))
    return views


def preflight_fragment_view_projection(
    nodes: Sequence[_NodeV3],
    classification: TopologyClassificationResult,
) -> Dict[str, Any]:
    """Project every classified fragment without mutating assembly state."""

    nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
        if int(getattr(node, "global_id", -1)) >= 0
    }
    try:
        views = project_fragment_views(classification, nodes_by_id)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "stage3_fragment_projection_invalid",
            "failure_domain": "topology_materialization",
            "error": str(exc),
        }
    fragment_ids = sorted(
        int(getattr(fragment, "fragment_id", -1))
        for fragment in list(getattr(classification, "fragments", []) or [])
    )
    view_ids = sorted(int(view.fragment_id) for view in views)
    if fragment_ids != view_ids:
        return {
            "ok": False,
            "reason": "stage3_fragment_projection_coverage_invalid",
            "failure_domain": "topology_materialization",
            "fragment_ids": fragment_ids,
            "view_ids": view_ids,
        }
    return {
        "ok": True,
        "reason": "ok",
        "failure_domain": "",
        "fragment_count": len(fragment_ids),
        "view_count": len(view_ids),
    }


__all__ = [
    "preflight_fragment_view_projection",
    "project_fragment_views",
    "project_like24_fragment",
]
