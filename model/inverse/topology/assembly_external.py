from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...shared.inverse_common import _NodeV3, estimate_region_carbon_budgets
from .assembly_residual import assemble_residual_resource_chains
from .classification.classification_models import TopologyClassificationResult, TopologyFragment
from .classification.classification_tokens import (
    AROMATIC_SU_TYPES,
    OPEN_ALIPHATIC_BASE_SU_TYPES,
    OPEN_ALIPHATIC_SUBTYPES,
    TERMINAL_BASE_SU_TYPES,
    TERMINAL_SUBTYPES,
    is_abstract_open_nonfixed_semantic,
    is_visible_fixed_connector_semantic,
    is_visible_mainline_nonfixed_semantic,
    is_visible_terminal_semantic,
    normalize_subtype,
)
from .stage3_models import (
    Stage3Serializable,
    TopologyAssemblyProduct,
    TopologyAssemblyResult,
    TopologyPathRender,
    TopologyPendingUnit,
    TopologyPoolLedger,
    stage3_contract_owner_key,
)
from .iteration_ledger import normalize_slot, required_counts_from_slots
from .sequential_resource import build_ownership_ledger


RESOURCE_SU_TYPES = (11, 22, 23, 24)
EXPLICIT_FLEX_CLASSES = {"A", "A+", "24_AB_M", "24_CD_M_FULL", "24_AB_N_24_AB"}
EXPLICIT_SIDE_CLASSES = {"C", "C+", "F", "24_AB_P"}
B_LIKE_OPEN_CLASSES = {"B", "B+", "24_AB_N", "24_CD_M", "24_AB_N_24_CD", "25_AB"}
D_LIKE_OPEN_CLASSES = {"D", "D+", "24_CD_N", "24_CD_N_24_CD", "25_CD"}
E_LIKE_OPEN_CLASSES = {"E", "E+", "G", "24_CD_P"}
RING_ONLY_LIKE24_CLASSES = {"24_A", "24_B", "24_C", "24_D"}
CHAIN_MATERIALIZABLE_LIKE24_CLASSES = {"24_AB_M", "24_AB_P", "24_CD_M_FULL"}
CHAIN_MATERIALIZABLE_OPEN_CLASSES = {
    "B",
    "B+",
    "C",
    "C+",
    "D",
    "D+",
    "E",
    "E+",
    "F",
    "G",
    "24_AB_N",
    "24_CD_M",
    "24_CD_P",
}
TERMINAL_ENDPOINT_LABELS = {"1", "4", "16", "18", "22", "25", "28", "32", "19d1", "20d1", "21d2"}
RING_PRODUCT_KIND_TO_TYPE = {
    "fatty_ring_fused": "aliphatic_ring",
    "fatty_ring_side": "aliphatic_ring",
    "fatty_ring_upper_lower": "aliphatic_ring",
}


def _slot_required_counts(slot_dicts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = required_counts_from_slots([
        normalize_slot(dict(slot or {}))
        for slot in list(slot_dicts or [])
        if dict(slot or {})
    ])
    return {str(key): int(value) for key, value in counts.items() if int(value) > 0}


def _pending_owner_key(
    unit_id: int,
    source_fragment_ids: Sequence[int],
    core_node_ids: Sequence[int],
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    return (
        int(unit_id),
        tuple(sorted({int(value) for value in list(source_fragment_ids or [])})),
        tuple(sorted({int(value) for value in list(core_node_ids or [])})),
    )
CHAIN_PRODUCT_KIND_TO_TYPE = {
}
CANONICAL_LABEL_BY_CLASS = {
    "A": ["11", "23", "11"],
    "B": ["11", "23"],
    "C": ["11", "23", "22"],
    "D": ["23"],
    "E": ["23", "22"],
    "F": ["11", "22"],
    "G": ["23", "22"],
    "A+": ["11", "23", "23", "23", "11"],
    "B+": ["11", "23", "23", "23"],
    "C+": ["11", "23", "23", "22"],
    "D+": ["23", "23", "23"],
    "E+": ["23", "23", "22"],
    "25_AB": ["11", "25"],
    "25_CD": ["25"],
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_su_key(value: Any) -> str:
    return str(_safe_int(value))


def _node_label(node: _NodeV3) -> str:
    subtype = str(normalize_subtype(node))
    try:
        su_type = int(getattr(node, "su_type", -1))
    except Exception:
        su_type = -1
    return subtype if subtype not in {str(su_type), ""} else str(su_type)


def _node_id_label(node_id: Optional[int]) -> str:
    if node_id is None:
        return "?"
    return str(int(node_id))


def _render_linear_path(
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    *,
    open_left: bool = False,
    open_right: bool = False,
    branch_map: Optional[Dict[int, List[str]]] = None,
    branch_id_map: Optional[Dict[int, List[str]]] = None,
    suppressed_support_node_ids: Optional[Sequence[int]] = None,
) -> TopologyPathRender:
    branch_map = dict(branch_map or {})
    branch_id_map = dict(branch_id_map or {})
    topo_parts: List[str] = []
    gid_parts: List[str] = []
    branch_topologies: List[str] = []
    branch_gids: List[str] = []
    for idx, (label, node_id) in enumerate(zip(list(labels or []), list(node_ids or []))):
        topo_token = str(label)
        gid_token = _node_id_label(node_id)
        for branch_topo, branch_gid in zip(
            list(branch_map.get(int(idx), []) or []),
            list(branch_id_map.get(int(idx), []) or []),
        ):
            topo_token += f"({branch_topo})"
            gid_token += f"({branch_gid})"
            branch_topologies.append(str(branch_topo))
            branch_gids.append(str(branch_gid))
        topo_parts.append(topo_token)
        gid_parts.append(gid_token)
    topology_notation = "-".join(x for x in topo_parts if str(x))
    node_id_notation = "-".join(x for x in gid_parts if str(x))
    if open_left:
        topology_notation = "-" + topology_notation
        node_id_notation = "-" + node_id_notation
    if open_right:
        topology_notation += "-"
        node_id_notation += "-"
    return TopologyPathRender(
        topology_notation=topology_notation,
        node_id_notation=node_id_notation,
        main_topology_notation=topology_notation,
        main_node_id_notation=node_id_notation,
        branch_topology_notations=branch_topologies,
        branch_node_id_notations=branch_gids,
        suppressed_support_node_ids=sorted({int(x) for x in list(suppressed_support_node_ids or [])}),
    )


def _render_token_sequence(
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    *,
    branch_map: Optional[Dict[int, List[str]]] = None,
    branch_id_map: Optional[Dict[int, List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    branch_map = dict(branch_map or {})
    branch_id_map = dict(branch_id_map or {})
    topo_tokens: List[str] = []
    gid_tokens: List[str] = []
    for idx, (label, node_id) in enumerate(zip(list(labels or []), list(node_ids or []))):
        topo_token = str(label)
        gid_token = _node_id_label(node_id)
        for branch_topo, branch_gid in zip(
            list(branch_map.get(int(idx), []) or []),
            list(branch_id_map.get(int(idx), []) or []),
        ):
            topo_token += f"({branch_topo})"
            gid_token += f"({branch_gid})"
        topo_tokens.append(topo_token)
        gid_tokens.append(gid_token)
    return topo_tokens, gid_tokens


def _strip_open_markers(text: str) -> str:
    out = str(text or "")
    while out.startswith("-"):
        out = out[1:]
    while out.endswith("-"):
        out = out[:-1]
    return out


def _clean_ring_segment_text(text: str) -> str:
    out = _strip_open_markers(str(text or ""))
    while "()" in out:
        out = out.replace("()", "")
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _is_aromatic_token(token: Dict[str, Any]) -> bool:
    return _safe_int(token.get("su_type", -1), -1) in set(AROMATIC_SU_TYPES)


def _is_aromatic_label(label: str) -> bool:
    value = str(label or "")
    return value == "11" or value in {str(x) for x in AROMATIC_SU_TYPES}


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


def _is_visible_terminal_token(token: Dict[str, Any]) -> bool:
    return bool(is_visible_terminal_semantic(token))


def _is_visible_fixed_connector_token(token: Dict[str, Any]) -> bool:
    return bool(is_visible_fixed_connector_semantic(token))


def _is_visible_mainline_nonfixed_token(token: Dict[str, Any]) -> bool:
    return bool(is_visible_mainline_nonfixed_semantic(token))


def _is_like24_family(view: "AssemblyFragmentView") -> bool:
    return str(view.family) in {"like24", "like24_secondary", "like24_bridge"}


def _is_chain_materializable_view(view: "AssemblyFragmentView") -> bool:
    class_type = str(view.class_type)
    if class_type in set(EXPLICIT_FLEX_CLASSES) | set(EXPLICIT_SIDE_CLASSES):
        return True
    if class_type in set(CHAIN_MATERIALIZABLE_OPEN_CLASSES):
        return True
    if _is_like24_family(view):
        return class_type in (
            set(CHAIN_MATERIALIZABLE_LIKE24_CLASSES)
            | {
                "24_A", "24_B", "24_C", "24_D",
                "24_AB_N", "24_CD_M", "24_CD_N", "24_AB_N_24_CD", "24_CD_N_24_CD",
            }
        )
    return class_type not in {"25_AB", "25_CD"}


def _is_generic_chain_join_view(view: "AssemblyFragmentView") -> bool:
    if not _is_chain_materializable_view(view):
        return False
    if _is_like24_family(view):
        return str(view.class_type) in {"24_AB_N", "24_CD_M"}
    return True


def _validate_chain_product_shape(
    product_kind: str,
    labels: Sequence[str],
    source_views: Sequence["AssemblyFragmentView"],
    node_ids: Optional[Sequence[Optional[int]]] = None,
) -> Optional[str]:
    normalized_kind = str(product_kind)
    token_labels = [str(label) for label in list(labels or []) if str(label)]
    if not token_labels:
        return "empty_topology"
    if normalized_kind not in {"flex_chain", "side_chain"}:
        return None
    materialized_ids = [int(value) for value in list(node_ids or []) if value is not None]
    if len(materialized_ids) != len(set(materialized_ids)):
        duplicate_ids = sorted({value for value in materialized_ids if materialized_ids.count(value) > 1})
        return f"duplicate_mainline_node_ids:{duplicate_ids}"
    left_label = str(token_labels[0])
    right_label = str(token_labels[-1])
    left_is_aromatic = left_label in {str(x) for x in AROMATIC_SU_TYPES} or left_label == "11"
    right_is_aromatic = right_label in {str(x) for x in AROMATIC_SU_TYPES} or right_label == "11"
    left_is_terminal = left_label in set(TERMINAL_ENDPOINT_LABELS)
    right_is_terminal = right_label in set(TERMINAL_ENDPOINT_LABELS)
    if normalized_kind == "flex_chain":
        if not (left_is_aromatic and right_is_aromatic):
            return f"invalid_flex_endpoints:{left_label}|{right_label}"
    if normalized_kind == "side_chain":
        has_su25_source = "25" in token_labels
        if has_su25_source and "25" in {left_label, right_label}:
            return None
        if not (
            (left_is_aromatic and right_is_terminal)
            or (left_is_terminal and right_is_aromatic)
        ):
            return f"invalid_side_endpoints:{left_label}|{right_label}"
    return None


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
    S_target: Any,
    E_target: Any,
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
            source_node_ids=[int(value) for value in list(getattr(fragment, "source_node_ids", []) or [])],
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


def _validate_ring_product_shape(
    product_kind: str,
    labels: Sequence[str],
    topology_notation: str,
    actual_topology_su: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    diag = dict(diagnostics or {})
    normalized_kind = {
        "side": "fatty_ring_side",
        "upper_lower": "fatty_ring_upper_lower",
        "fused": "fatty_ring_fused",
    }.get(str(diag.get("ring_template_kind", "")), str(product_kind))
    if normalized_kind not in {"fatty_ring_fused", "fatty_ring_side", "fatty_ring_upper_lower"}:
        return None
    backbone_labels = _normalize_ring_backbone_labels(diag.get("ring_backbone_labels", []))
    backbone_node_ids = _normalize_ring_backbone_node_ids(diag.get("ring_backbone_node_ids", []))
    like24_count = _safe_int(diag.get("ring_like24_count"), -1)
    actual = str(actual_topology_su or "")
    topology = str(topology_notation or "")
    if not actual or not topology:
        return "ring_empty_render"
    if int(diag.get("unresolved_branch_contract_count", 0) or 0) > 0:
        return "ring_unresolved_branch_contract"
    if "()" in actual or "()" in topology:
        return "ring_empty_branch_render"
    if any(marker in actual for marker in {"H1", "FIXED22"}) or any(marker in topology for marker in {"H1", "FIXED22"}):
        return "ring_unresolved_branch_placeholder"
    if actual.endswith("-") or actual.startswith("--") or "--" in actual:
        return "ring_open_mainline_render"
    if not backbone_labels:
        return "ring_missing_backbone_metadata"
    if len(backbone_labels) != len(backbone_node_ids):
        return "ring_backbone_metadata_mismatch"
    if str(normalized_kind) == "fatty_ring_side":
        if len(backbone_labels) != 6:
            return f"side_ring_backbone_node_count:{len(backbone_labels)}"
        if like24_count < 2 or like24_count > 4:
            return f"side_ring_like24_count:{like24_count}"
        left_label = str(backbone_labels[0])
        right_label = str(backbone_labels[-1])
        if not (_is_aromatic_label(left_label) and _is_aromatic_label(right_label)):
            return f"invalid_side_ring_endpoints:{left_label}|{right_label}"
        if "[" in actual or "]" in actual:
            return "side_ring_inner_loop_not_allowed"
        return None
    if str(normalized_kind) == "fatty_ring_upper_lower":
        if len(backbone_labels) != 6:
            return f"upper_ring_backbone_node_count:{len(backbone_labels)}"
        if like24_count < 2 or like24_count > 3:
            return f"upper_ring_like24_count:{like24_count}"
        has_fixed_anchor = bool(diag.get("ring_has_fixed_anchor"))
        has_closure_target = (
            diag.get("closure_target_fragment_id") is not None
            or diag.get("closure_target_node_id") is not None
        )
        if not has_fixed_anchor:
            return "upper_ring_missing_fixed_anchor"
        if not bool(has_closure_target):
            return "upper_ring_missing_closure_target"
        return None
    outer_labels = _normalize_ring_backbone_labels(diag.get("ring_outer_backbone_labels", []))
    outer_node_ids = _normalize_ring_backbone_node_ids(diag.get("ring_outer_backbone_node_ids", []))
    inner_labels = _normalize_ring_backbone_labels(diag.get("ring_inner_backbone_labels", []))
    inner_node_ids = _normalize_ring_backbone_node_ids(diag.get("ring_inner_backbone_node_ids", []))
    bridgehead_ids = [
        int(value)
        for value in list(diag.get("ring_bridgehead_ids", []) or [])
        if value is not None
    ]
    if "[" not in actual or "]" not in actual:
        return "fused_ring_missing_inner_loop"
    if len(outer_labels) != 6 or len(outer_labels) != len(outer_node_ids):
        return f"fused_ring_outer_backbone_node_count:{len(outer_labels)}"
    if len(inner_labels) != 6 or len(inner_labels) != len(inner_node_ids):
        return f"fused_ring_inner_backbone_node_count:{len(inner_labels)}"
    if like24_count < 4 or like24_count > 6:
        return f"fused_ring_like24_count:{like24_count}"
    if len(bridgehead_ids) != 2:
        return f"fused_ring_bridgehead_count:{len(bridgehead_ids)}"
    if bool(diag.get("ring_pair_bridgehead_mode")) and bool(diag.get("ring_inner_bridgehead_capable")):
        return "fused_bridgehead_template_conflict"
    return None


def _build_node_lookup(nodes: Sequence[_NodeV3]) -> Dict[int, _NodeV3]:
    out: Dict[int, _NodeV3] = {}
    for node in list(nodes or []):
        try:
            out[int(getattr(node, "global_id"))] = node
        except Exception:
            continue
    return out


@dataclass
class BranchStub(Stage3Serializable):
    topology_notation: str = ""
    node_id_notation: str = ""
    source_fragment_ids: List[int] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    branch_kind: str = ""
    mode: str = ""


@dataclass
class BranchClosureResult(Stage3Serializable):
    branch_topology: str = ""
    branch_node_ids: str = ""
    branch_kind: str = ""
    mode: str = ""
    anchor_index: int = 0
    attach_role: str = "center_branch"
    target_path: str = "center_side_branch"
    suppressed_support_node_ids: List[int] = field(default_factory=list)
    source_fragment_ids: List[int] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    main_labels: List[str] = field(default_factory=list)
    main_node_ids: List[Optional[int]] = field(default_factory=list)


@dataclass
class ClosureOperatorSpec(Stage3Serializable):
    name: str = ""
    scope: str = ""
    preferred_fragment_class_types: List[str] = field(default_factory=list)
    preferred_fragment_main_kinds: List[str] = field(default_factory=list)
    pool_templates: List[str] = field(default_factory=list)
    allow_fragment_substitution: bool = False
    fixed_side_compatible_only: bool = False


@dataclass(frozen=True)
class AssemblyEndpoint(Stage3Serializable):
    side: str
    node_id: Optional[int]
    label: str
    is_open: bool
    role: str
    bridge_node_id: Optional[int] = None
    bridge_label: str = ""


@dataclass(frozen=True)
class AssemblyCompositionCandidate:
    left: "AssemblyFragmentView"
    right: "AssemblyFragmentView"
    left_reversed: bool
    right_reversed: bool
    product_kind: str
    reason: str
    score: Tuple[Any, ...]


@dataclass
class AssemblyFragmentView(Stage3Serializable):
    unit_id: int
    fragment_id: int
    family: str
    class_type: str
    topology_template: str
    source_node_ids: List[int]
    main_labels: List[str]
    main_node_ids: List[Optional[int]]
    canonical_labels: List[str]
    open_left: bool
    open_right: bool
    branch_anchor_index: int = 0
    unresolved_branch_kinds: List[str] = field(default_factory=list)
    actual_branch_stubs: List[BranchStub] = field(default_factory=list)
    branch_stubs_by_anchor: Dict[int, List[BranchStub]] = field(default_factory=dict)
    unresolved_branch_kinds_by_anchor: Dict[int, List[str]] = field(default_factory=dict)
    real_branch_stubs_by_anchor: Dict[int, List[BranchStub]] = field(default_factory=dict)
    real_unresolved_branch_kinds_by_anchor: Dict[int, List[str]] = field(default_factory=dict)
    fixed_connection_contracts: List[Dict[str, Any]] = field(default_factory=list)
    branch_closure_contracts_by_anchor: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    branch_closure_blockers_by_anchor: Dict[int, List[str]] = field(default_factory=dict)
    resolved_fixed_mainline_connector_id: Optional[int] = None
    suppressed_support_node_ids: List[int] = field(default_factory=list)
    main_kind: str = ""
    preferred_product_type: str = ""
    ring_roles: List[str] = field(default_factory=list)
    ring_backbone_labels: Dict[str, List[str]] = field(default_factory=dict)
    ring_backbone_node_ids: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    ring_backbone_kind: Dict[str, str] = field(default_factory=dict)
    ring_backbone_span: Dict[str, int] = field(default_factory=dict)
    ring_like24_units: int = 0
    ring_anchor_profile: Dict[str, Any] = field(default_factory=dict)
    ring_template_roles: Dict[str, List[str]] = field(default_factory=dict)
    flex_reconcile_roles: List[str] = field(default_factory=list)
    flex_reconcile_priority: int = 99
    can_flex: bool = False
    can_side: bool = False
    can_ring: bool = False
    target_roles: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    context_node_ids: List[int] = field(default_factory=list)

    def _endpoint_role(self, *, side: str, is_open: bool, label: str) -> str:
        if is_open:
            roles = set(str(value) for value in list(self.flex_reconcile_roles or []))
            if "b_like_open" in roles or str(self.class_type) in B_LIKE_OPEN_CLASSES:
                return "b_like_open_end"
            if str(self.class_type) in E_LIKE_OPEN_CLASSES:
                return "e_like_open_end"
            if "d_like_open" in roles or str(self.class_type) in D_LIKE_OPEN_CLASSES:
                return "double_open_end"
            return "closed_internal_end"
        if _is_aromatic_label(str(label)):
            return "aromatic_closed_end"
        if str(label) in TERMINAL_ENDPOINT_LABELS:
            return "terminal_closed_end"
        return "closed_internal_end"

    def endpoint(self, side: str) -> AssemblyEndpoint:
        side_name = str(side)
        if side_name not in {"left", "right"}:
            raise ValueError(f"invalid endpoint side: {side_name}")
        index = 0 if side_name == "left" else -1
        label = str(self.main_labels[index]) if self.main_labels else ""
        node_id = self.main_node_ids[index] if self.main_node_ids else None
        node_id = None if node_id is None else int(node_id)
        is_open = bool(self.open_left if side_name == "left" else self.open_right)
        token = dict(self.diagnostics.get(f"{side_name}_open_bridge", {}) or {})
        bridge_node_id = _safe_int(token.get("node_id"), -1)
        bridge_label = str(token.get("subtype") or token.get("su_type") or "")
        if is_open and bridge_node_id < 0 and node_id is not None:
            bridge_node_id = int(node_id)
            bridge_label = str(label)
        return AssemblyEndpoint(
            side=side_name,
            node_id=node_id,
            label=str(label),
            is_open=bool(is_open),
            role=self._endpoint_role(side=side_name, is_open=bool(is_open), label=str(label)),
            bridge_node_id=None if bridge_node_id < 0 else int(bridge_node_id),
            bridge_label=str(bridge_label),
        )

    def oriented_endpoint(self, side: str, *, reversed_view: bool) -> AssemblyEndpoint:
        actual_side = str(side)
        if bool(reversed_view):
            actual_side = "right" if str(side) == "left" else "left"
        endpoint = self.endpoint(actual_side)
        return AssemblyEndpoint(
            side=str(side),
            node_id=endpoint.node_id,
            label=endpoint.label,
            is_open=endpoint.is_open,
            role=endpoint.role,
            bridge_node_id=endpoint.bridge_node_id,
            bridge_label=endpoint.bridge_label,
        )

    def oriented_mainline(self, *, reversed_view: bool) -> Tuple[List[str], List[Optional[int]]]:
        if bool(reversed_view):
            return list(reversed(self.main_labels or [])), list(reversed(self.main_node_ids or []))
        return list(self.main_labels or []), list(self.main_node_ids or [])

    @property
    def main_closed(self) -> bool:
        return not bool(self.open_left or self.open_right)

    def branch_stubs_map(self) -> Dict[int, List[BranchStub]]:
        if dict(self.branch_stubs_by_anchor or {}):
            return {
                int(anchor): list(values or [])
                for anchor, values in dict(self.branch_stubs_by_anchor or {}).items()
            }
        return {int(self.branch_anchor_index): list(self.actual_branch_stubs or [])}

    def branch_kinds_by_anchor(self) -> Dict[int, List[str]]:
        if dict(self.unresolved_branch_kinds_by_anchor or {}):
            return {
                int(anchor): [str(value) for value in list(values or [])]
                for anchor, values in dict(self.unresolved_branch_kinds_by_anchor or {}).items()
            }
        return {int(self.branch_anchor_index): [str(value) for value in list(self.unresolved_branch_kinds or [])]}

    def branch_closure_specs(self) -> List[Dict[str, Any]]:
        return [
            dict(item or {})
            for item in list(dict(self.evidence or {}).get("required_branch_closure_specs", []) or [])
            if dict(item or {})
        ]

    def fixed_branch_contracts_by_anchor(self) -> Dict[int, List[Dict[str, Any]]]:
        return {
            int(anchor): [dict(item or {}) for item in list(values or []) if dict(item or {})]
            for anchor, values in dict(self.branch_closure_contracts_by_anchor or {}).items()
        }

    def real_branch_stubs_map(self) -> Dict[int, List[BranchStub]]:
        if dict(self.real_branch_stubs_by_anchor or {}):
            return {
                int(anchor): list(values or [])
                for anchor, values in dict(self.real_branch_stubs_by_anchor or {}).items()
            }
        return self.branch_stubs_map()

    def real_branch_kinds_by_anchor(self) -> Dict[int, List[str]]:
        if dict(self.real_unresolved_branch_kinds_by_anchor or {}):
            return {
                int(anchor): [str(value) for value in list(values or [])]
                for anchor, values in dict(self.real_unresolved_branch_kinds_by_anchor or {}).items()
            }
        return self.branch_kinds_by_anchor()

    def branch_suppression_by_skeleton(self) -> Dict[str, List[str]]:
        raw = dict(dict(self.evidence or {}).get("branch_suppression_by_skeleton", {}) or {})
        return {
            str(key): [str(value) for value in list(values or []) if str(value)]
            for key, values in raw.items()
        }

    def partial_render(self) -> TopologyPathRender:
        return self._render_from_maps(self.branch_stubs_map(), self.branch_kinds_by_anchor())

    def real_render(self) -> TopologyPathRender:
        return self._render_from_maps(self.real_branch_stubs_map(), self.real_branch_kinds_by_anchor())

    def _render_from_maps(
        self,
        branch_stubs_map: Dict[int, List[BranchStub]],
        branch_kinds_map: Dict[int, List[str]],
    ) -> TopologyPathRender:
        def _branch_kind_render(kind: str) -> Tuple[str, str]:
            name = str(kind or "")
            if name in {"H", "H1"}:
                return "", ""
            if name == "FIXED22":
                return "-22", "-22"
            return name, name

        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        all_anchors = sorted({
            int(anchor)
            for anchor in set(branch_stubs_map.keys()) | set(branch_kinds_map.keys())
        })
        for anchor in all_anchors:
            branch_entries: List[Tuple[str, str]] = [
                (str(stub.topology_notation), str(stub.node_id_notation))
                for stub in list(branch_stubs_map.get(int(anchor), []) or [])
            ]
            for kind in list(branch_kinds_map.get(int(anchor), []) or []):
                topo_text, gid_text = _branch_kind_render(str(kind))
                if not str(topo_text) and not str(gid_text):
                    continue
                branch_entries.append((str(topo_text), str(gid_text)))
            branch_map[int(anchor)] = [topo for topo, _ in branch_entries]
            branch_id_map[int(anchor)] = [gid for _, gid in branch_entries]
        return _render_linear_path(
            self.main_labels,
            self.main_node_ids,
            open_left=bool(self.open_left),
            open_right=bool(self.open_right),
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=list(self.suppressed_support_node_ids or []),
        )
@dataclass
class RingSkeletonPlan(Stage3Serializable):
    ring_kind: str = ""
    skeleton_kind: str = ""
    missing_slots: List[str] = field(default_factory=list)
    source_views: List[AssemblyFragmentView] = field(default_factory=list)
    ordered_segments: List[Dict[str, Any]] = field(default_factory=list)
    closure_target_fragment_id: Optional[int] = None
    closure_target_node_id: Optional[int] = None
    like24_count_min: int = 0
    like24_count_max: int = 0
    complete_ready: bool = False


@dataclass
class ViewSegment(Stage3Serializable):
    view: Optional[AssemblyFragmentView] = None
    labels: List[str] = field(default_factory=list)
    node_ids: List[Optional[int]] = field(default_factory=list)
    ring_backbone_labels: List[str] = field(default_factory=list)
    ring_backbone_node_ids: List[Optional[int]] = field(default_factory=list)
    branch_map: Dict[int, List[str]] = field(default_factory=dict)
    branch_id_map: Dict[int, List[str]] = field(default_factory=dict)
    branch_modes: List[str] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    source_views: List[AssemblyFragmentView] = field(default_factory=list)
    suppressed_support_node_ids: List[int] = field(default_factory=list)
    ring_template_kind: str = ""
    ring_template_roles: List[str] = field(default_factory=list)
    required_counts: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RingBackboneMaterialization(Stage3Serializable):
    source_views: List[AssemblyFragmentView] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    node_ids: List[Optional[int]] = field(default_factory=list)
    backbone_labels: List[str] = field(default_factory=list)
    backbone_node_ids: List[Optional[int]] = field(default_factory=list)
    topology_notation: str = ""
    node_id_notation: str = ""
    support_node_ids: List[int] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)


BRANCH_CLOSURE_OPERATOR_SPECS = {
    "H": ClosureOperatorSpec(
        name="H",
        scope="branch",
        preferred_fragment_class_types=[],
        preferred_fragment_main_kinds=[],
        pool_templates=["23-22"],
        allow_fragment_substitution=False,
        fixed_side_compatible_only=False,
    ),
    "H1": ClosureOperatorSpec(
        name="H1",
        scope="branch",
        preferred_fragment_class_types=[],
        preferred_fragment_main_kinds=[],
        pool_templates=["23-22"],
        allow_fragment_substitution=False,
        fixed_side_compatible_only=True,
    ),
    "FIXED22": ClosureOperatorSpec(
        name="FIXED22",
        scope="branch",
        preferred_fragment_class_types=[],
        preferred_fragment_main_kinds=[],
        pool_templates=["22"],
        allow_fragment_substitution=False,
        fixed_side_compatible_only=True,
    ),
}
MAINLINE_CLOSURE_OPERATOR_SPECS = {
    "F1": ClosureOperatorSpec(
        name="F1",
        scope="mainline",
        preferred_fragment_class_types=[],
        preferred_fragment_main_kinds=[],
        pool_templates=["23-11"],
        allow_fragment_substitution=False,
        fixed_side_compatible_only=False,
    ),
    "F2": ClosureOperatorSpec(
        name="F2",
        scope="mainline",
        preferred_fragment_class_types=[],
        preferred_fragment_main_kinds=[],
        pool_templates=["23-22"],
        allow_fragment_substitution=False,
        fixed_side_compatible_only=False,
    ),
}


class PoolAllocator:
    def __init__(self, initial_ids_by_su: Dict[str, Iterable[int]]) -> None:
        normalized = {
            _normalize_su_key(key): [int(x) for x in list(values or [])]
            for key, values in dict(initial_ids_by_su or {}).items()
        }
        self.initial_ids_by_su = {
            str(key): sorted(list(values or []))
            for key, values in normalized.items()
        }
        self.remaining_ids_by_su = deepcopy(self.initial_ids_by_su)
        self.consumed_ids_by_su = {
            str(key): []
            for key in self.initial_ids_by_su.keys()
        }
        self.protected_counts_by_su: Dict[str, int] = {}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "remaining": deepcopy(self.remaining_ids_by_su),
            "consumed": deepcopy(self.consumed_ids_by_su),
        }

    def rollback(self, snapshot: Dict[str, Any]) -> None:
        self.remaining_ids_by_su = deepcopy(dict(snapshot or {}).get("remaining", {}))
        self.consumed_ids_by_su = deepcopy(dict(snapshot or {}).get("consumed", {}))

    def claimed_ids_since(self, snapshot: Dict[str, Any]) -> Dict[str, List[int]]:
        before = {
            str(key): set(int(value) for value in list(values or []))
            for key, values in dict(snapshot or {}).get("remaining", {}).items()
        }
        return {
            str(key): sorted(before.get(str(key), set()) - set(int(value) for value in list(values or [])))
            for key, values in self.remaining_ids_by_su.items()
            if before.get(str(key), set()) - set(int(value) for value in list(values or []))
        }

    def claim(self, su_type: int, count: int = 1) -> Optional[List[int]]:
        su_key = _normalize_su_key(su_type)
        want = max(0, int(count))
        available = list(self.remaining_ids_by_su.get(su_key, []) or [])
        if want <= 0:
            return []
        protected = max(0, int(self.protected_counts_by_su.get(su_key, 0) or 0))
        if len(available) - protected < want:
            return None
        claimed = [int(available.pop(0)) for _ in range(want)]
        self.remaining_ids_by_su[su_key] = list(available)
        self.consumed_ids_by_su.setdefault(su_key, [])
        self.consumed_ids_by_su[su_key].extend(int(x) for x in claimed)
        return claimed

    def claim_specific(self, su_type: int, node_id: int) -> Optional[List[int]]:
        su_key = _normalize_su_key(su_type)
        selected = int(node_id)
        available = list(self.remaining_ids_by_su.get(su_key, []) or [])
        if selected not in available:
            return None
        protected = max(0, int(self.protected_counts_by_su.get(su_key, 0) or 0))
        if len(available) - protected < 1:
            return None
        available.remove(selected)
        self.remaining_ids_by_su[su_key] = available
        self.consumed_ids_by_su.setdefault(su_key, [])
        self.consumed_ids_by_su[su_key].append(selected)
        return [selected]

    def consume_existing_ids(self, ids_by_su: Dict[Any, Iterable[int]]) -> None:
        for raw_su, raw_ids in dict(ids_by_su or {}).items():
            su_key = _normalize_su_key(raw_su)
            selected = {int(value) for value in list(raw_ids or [])}
            if not selected:
                continue
            available = list(self.remaining_ids_by_su.get(su_key, []) or [])
            consumed_now = [int(value) for value in available if int(value) in selected]
            self.remaining_ids_by_su[su_key] = [
                int(value) for value in available if int(value) not in selected
            ]
            self.consumed_ids_by_su.setdefault(su_key, [])
            self.consumed_ids_by_su[su_key].extend(consumed_now)
            self.consumed_ids_by_su[su_key] = sorted(set(self.consumed_ids_by_su[su_key]))

    def set_protected_counts(self, counts: Optional[Dict[Any, Any]] = None) -> None:
        self.protected_counts_by_su = {
            _normalize_su_key(key): max(0, int(value))
            for key, value in dict(counts or {}).items()
            if int(value) > 0
        }

    def build_ledger(self, shortage_counts: Dict[str, int]) -> TopologyPoolLedger:
        initial_counts = {
            str(key): int(len(list(values or [])))
            for key, values in self.initial_ids_by_su.items()
        }
        consumed_counts = {
            str(key): int(len(list(values or [])))
            for key, values in self.consumed_ids_by_su.items()
        }
        remaining_counts = {
            str(key): int(len(list(values or [])))
            for key, values in self.remaining_ids_by_su.items()
        }
        return TopologyPoolLedger(
            initial_ids_by_su={str(k): list(v or []) for k, v in self.initial_ids_by_su.items()},
            remaining_ids_by_su={str(k): list(v or []) for k, v in self.remaining_ids_by_su.items()},
            consumed_ids_by_su={str(k): list(v or []) for k, v in self.consumed_ids_by_su.items()},
            initial_counts=initial_counts,
            consumed_counts=consumed_counts,
            remaining_counts=remaining_counts,
            shortage_counts={str(k): int(v) for k, v in dict(shortage_counts or {}).items()},
        )

    def available_counts(self) -> Dict[str, int]:
        return {
            str(key): int(len(list(values or [])))
            for key, values in self.remaining_ids_by_su.items()
        }

    def usable_counts(self) -> Dict[str, int]:
        return {
            str(key): max(0, int(len(list(values or []))) - int(self.protected_counts_by_su.get(str(key), 0) or 0))
            for key, values in self.remaining_ids_by_su.items()
        }


def _main_kind_for_class(class_type: str) -> str:
    if class_type in EXPLICIT_FLEX_CLASSES:
        return "explicit_flex"
    if class_type in EXPLICIT_SIDE_CLASSES:
        return "explicit_side"
    if class_type in B_LIKE_OPEN_CLASSES:
        return "b_like_open"
    if class_type in D_LIKE_OPEN_CLASSES:
        return "d_like_open"
    if class_type in E_LIKE_OPEN_CLASSES:
        return "e_like_open"
    if class_type in RING_ONLY_LIKE24_CLASSES:
        return "ring_only_like24"
    return "other"


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
    center_ids = list(dict(fragment.evidence or {}).get("center_node_ids", []) or [])
    if center_ids:
        return nodes_by_id.get(_safe_int(center_ids[0], -1))
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


def _project_basic_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    class_type = str(fragment.class_type)
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    center_su = int(getattr(center, "su_type", -1))
    matched_neighbors = list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
    claimed_ids = {int(value) for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])}
    atomic_pair_node_ids = {
        int(value)
        for value in list(dict(fragment.evidence or {}).get("atomic_group_node_ids", []) or [])
    }
    aromatic = _sort_token_dicts([nb for nb in matched_neighbors if _is_aromatic_token(nb)])
    open_neighbors = _sort_token_dicts([nb for nb in matched_neighbors if _is_open_token(nb)])
    generic_open_neighbors = _sort_token_dicts([
        nb for nb in matched_neighbors
        if _is_generic_open_token(nb) and _dict_node_id(dict(nb or {})) in claimed_ids
        and _dict_node_id(dict(nb or {})) not in atomic_pair_node_ids
    ])
    special_open_neighbors = _sort_token_dicts([nb for nb in open_neighbors if not _is_generic_open_token(nb)])
    terminal = sorted(
        [dict(nb or {}) for nb in matched_neighbors if _is_terminal_token(nb)],
        key=_terminal_priority,
    )
    left_bridge_token = generic_open_neighbors[0] if generic_open_neighbors else {}
    right_bridge_token = generic_open_neighbors[-1] if generic_open_neighbors else {}
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
        open_left = True
        open_right = True
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
    left_port_token = {}
    right_port_token = {}
    if class_type == "B":
        right_port_token = dict(special_open_neighbors[0] if special_open_neighbors else right_bridge_token or {})
    elif class_type == "D":
        if center_su in {15, 17} and len(node_ids) == 2:
            left_port_token = dict(generic_open_neighbors[0]) if generic_open_neighbors else {}
            right_port_token = {}
        elif special_open_neighbors:
            left_port_token = dict(special_open_neighbors[0])
            right_port_token = dict(special_open_neighbors[-1])
        else:
            left_port_token = dict(left_bridge_token or {})
            right_port_token = dict(right_bridge_token or {})
    elif class_type == "E":
        right_port_token = dict(special_open_neighbors[0] if special_open_neighbors else right_bridge_token or {})
    elif class_type == "G":
        left_port_token = dict(
            existing_bridge
            or (left_bridge_token or special_open_neighbors[0] if special_open_neighbors else left_bridge_token or {})
        )
    else:
        left_port_token = dict(left_bridge_token or {})
        right_port_token = dict(right_bridge_token or {})
    left_port_id = _dict_node_id(left_port_token) if left_port_token else -1
    if left_port_id in claimed_ids and left_port_id not in set(int(value) for value in node_ids if value is not None):
        labels = [_dict_label(left_port_token)] + list(labels)
        node_ids = [int(left_port_id)] + list(node_ids)
    right_port_id = _dict_node_id(right_port_token) if right_port_token else -1
    if right_port_id in claimed_ids and right_port_id not in set(int(value) for value in node_ids if value is not None):
        labels = list(labels) + [_dict_label(right_port_token)]
        node_ids = list(node_ids) + [int(right_port_id)]
    main_kind = _main_kind_for_class(class_type)
    ring_backbone_labels: Dict[str, List[str]] = {}
    ring_backbone_node_ids: Dict[str, List[Optional[int]]] = {}
    ring_backbone_kind: Dict[str, str] = {}
    ring_backbone_span: Dict[str, int] = {}
    ring_anchor_profile = {"left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")), "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else ""))}
    ring_template_roles: Dict[str, List[str]] = {}
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
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
        branch_anchor_index=max(0, len(labels) // 2),
        unresolved_branch_kinds=[],
        actual_branch_stubs=[],
        main_kind=main_kind,
        preferred_product_type=preferred,
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
        can_flex=bool(("flex_preferred" in set(flex_reconcile_roles)) or preferred == "flex_chain"),
        can_side=bool(preferred == "side_chain"),
        can_ring=False,
        target_roles=["flex_chain"] if preferred == "flex_chain" else ["side_chain"],
        evidence=dict(fragment.evidence or {}),
        diagnostics={
            "source_fragment_family": str(fragment.family),
            "matched_neighbors": [dict(token or {}) for token in matched_neighbors],
            "aromatic_neighbors": [dict(token or {}) for token in aromatic],
            "terminal_neighbors": [dict(token or {}) for token in terminal],
            "special_open_neighbors": [dict(token or {}) for token in special_open_neighbors],
            "generic_open_neighbors": [dict(token or {}) for token in generic_open_neighbors],
            "left_open_bridge": dict(left_port_token or {}),
            "right_open_bridge": dict(right_port_token or {}),
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
    ring_backbone_labels: Dict[str, List[str]] = {}
    ring_backbone_node_ids: Dict[str, List[Optional[int]]] = {}
    ring_backbone_kind: Dict[str, str] = {}
    ring_backbone_span: Dict[str, int] = {}
    ring_anchor_profile = {"left_aromatic": _is_aromatic_label(str(labels[0] if labels else "")), "right_aromatic": _is_aromatic_label(str(labels[-1] if labels else ""))}
    ring_template_roles: Dict[str, List[str]] = {}
    main_kind = _main_kind_for_class(class_type)
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
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
        can_flex=preferred == "flex_chain" or class_type in {"C+"},
        can_side=preferred == "side_chain" or class_type == "E+",
        can_ring=False,
        target_roles=["flex_chain"] if preferred == "flex_chain" else ["side_chain"],
        evidence=evidence,
        diagnostics={"connector_profiles": profiles},
    )


def _neighbor_semantic_dict(center: _NodeV3, neighbor: _NodeV3) -> Dict[str, Any]:
    fixed_ids = {int(x) for x in list(getattr(center, "fixed_hop1_ids", set()) or set())}
    return {
        "node_id": int(getattr(neighbor, "global_id", -1)),
        "su_type": int(getattr(neighbor, "su_type", -1)),
        "subtype": _node_label(neighbor),
        "is_fixed": bool(int(getattr(neighbor, "global_id", -1)) in fixed_ids),
    }


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
        if _is_visible_terminal_token(semantics):
            stub = _make_custom_branch_stub(
                topology_notation=f"-{label}",
                node_id_notation=f"-{_node_id_label(int(node_id))}",
                node_id=int(node_id),
                mode="visible_terminal",
            )
            abstract_stubs.append(stub)
            real_stubs.append(stub)
        elif not _is_generic_open_token(semantics) and not _is_visible_mainline_nonfixed_token(semantics):
            stub = _make_custom_branch_stub(
                topology_notation=f"-{label}",
                node_id_notation=f"-{_node_id_label(int(node_id))}",
                node_id=int(node_id),
                mode="visible_branch",
            )
            abstract_stubs.append(stub)
            real_stubs.append(stub)
        else:
            suppressed_support_node_ids.append(int(node_id))
    return abstract_stubs, real_stubs, sorted({int(x) for x in suppressed_support_node_ids})


def _select_like24_mainline_candidate(
    center: _NodeV3,
    *,
    exclude_ids: Set[int],
    nodes_by_id: Dict[int, _NodeV3],
) -> Optional[_NodeV3]:
    candidates: List[_NodeV3] = []
    for node_id in list(getattr(center, "hop1_ids", []) or []):
        node_id_i = _safe_int(node_id, -1)
        if node_id_i in exclude_ids:
            continue
        neighbor = nodes_by_id.get(node_id_i)
        if neighbor is None:
            continue
        semantics = _neighbor_semantic_dict(center, neighbor)
        if _is_visible_mainline_nonfixed_token(semantics):
            candidates.append(neighbor)
    candidates.sort(key=lambda item: int(item.global_id))
    return candidates[0] if candidates else None


def _project_like24_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
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
    direct_neighbors = [
        nodes_by_id.get(_safe_int(node_id, -1))
        for node_id in list(getattr(center, "hop1_ids", []) or [])
    ]
    direct_neighbors = [node for node in direct_neighbors if node is not None]
    eleven_neighbors = sorted(
        [node for node in direct_neighbors if int(getattr(node, "su_type", -1)) == 11],
        key=lambda item: int(item.global_id),
    )
    connector_id = _safe_int(primary.get("connector_node_id"), -1)
    connector_node = nodes_by_id.get(connector_id)
    remaining_node_ids = [_safe_int(x, -1) for x in list(primary.get("remaining_node_ids", []) or [])]
    remaining_nodes = [nodes_by_id.get(node_id) for node_id in remaining_node_ids if nodes_by_id.get(node_id) is not None]

    labels: List[str] = []
    node_ids: List[int] = []
    open_left = False
    open_right = False
    preferred = "flex_chain"
    exclude_direct_ids: Set[int] = {center_id}
    exclude_by_center: Dict[int, Set[int]] = {
        int(node_id): {int(node_id)}
        for node_id in center_nodes.keys()
    }

    if class_type in {"24_A", "24_B"}:
        if eleven_neighbors:
            labels.append(_node_label(eleven_neighbors[0]))
            node_ids.append(int(eleven_neighbors[0].global_id))
            exclude_direct_ids.add(int(eleven_neighbors[0].global_id))
            exclude_by_center.setdefault(center_id, {center_id}).add(int(eleven_neighbors[0].global_id))
        labels.append(_node_label(center))
        node_ids.append(center_id)
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
        open_right = class_type == "24_AB_N"
        preferred = "flex_chain" if class_type == "24_AB_M" else ("side_chain" if class_type == "24_AB_P" else "flex_chain")
        if class_type == "24_AB_P":
            labels = list(reversed(labels))
            node_ids = list(reversed(node_ids))
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
            if left_anchor is not None:
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
            if right_anchor is not None:
                labels.append(_node_label(right_anchor))
                node_ids.append(int(right_anchor.global_id))
                exclude_by_center.setdefault(int(right_center_id), {int(right_center_id)}).add(int(right_anchor.global_id))
        for cid in center_nodes.keys():
            exclude_by_center.setdefault(int(cid), {int(cid)}).update(int(x) for x in node_ids if x is not None)
        open_left = class_type == "24_CD_N_24_CD"
        open_right = class_type in {"24_AB_N_24_CD", "24_CD_N_24_CD"}
        preferred = "flex_chain" if class_type == "24_AB_N_24_AB" else "flex_chain"
    else:
        labels = [_node_label(center)]
        node_ids = [center_id]

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
        if abstract_stubs:
            branch_stubs_by_anchor[int(anchor)] = list(abstract_stubs)
        if real_stubs:
            real_branch_stubs_by_anchor[int(anchor)] = list(real_stubs)
        suppressed_support_node_ids.extend(int(x) for x in list(suppressed_ids or []))
        if int(cid) == int(center_id):
            primary_anchor = int(anchor)

    if class_type == "24_CD_N_24_CD":
        for anchor in list(anchor_by_center_node_id.values()):
            if list(branch_stubs_by_anchor.get(int(anchor), []) or []):
                continue
            branch_stubs_by_anchor[int(anchor)] = [
                _make_custom_branch_stub(
                    topology_notation="",
                    node_id_notation="",
                    mode="pair_bridge_open",
                ),
            ]

    required_branch_map_raw = dict(evidence.get("required_branch_kinds_by_anchor", {}) or {})
    unresolved_branch_kinds_by_anchor: Dict[int, List[str]] = {}
    if required_branch_map_raw:
        ordered_center_ids = list(center_nodes.keys())
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
            unresolved_branch_kinds_by_anchor[int(mapped_anchor)] = [str(value) for value in list(values or [])]
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

    if class_type in {"24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD"}:
        for center_node_id, anchor in dict(anchor_by_center_node_id or {}).items():
            center_node = center_nodes.get(int(center_node_id))
            if center_node is None:
                continue
            excluded_ids = set(exclude_by_center.get(int(center_node_id), {int(center_node_id)}))
            for neighbor_id in sorted(int(x) for x in list(getattr(center_node, "fixed_hop1_ids", []) or [])):
                if int(neighbor_id) in excluded_ids:
                    continue
                neighbor = nodes_by_id.get(int(neighbor_id))
                if neighbor is None:
                    continue
                base_labels = [_node_label(neighbor)]
                base_node_ids = [int(neighbor.global_id)]
                tail_kind = "none"
                tail_reason = ""
                externally_closed = False
                tail_neighbor = next((
                    nodes_by_id.get(int(hop_id))
                    for hop_id in list(getattr(neighbor, "hop1_ids", []) or [])
                    if int(hop_id) != int(center_node_id) and nodes_by_id.get(int(hop_id)) is not None
                ), None)
                if tail_neighbor is not None:
                    tail_label = _node_label(tail_neighbor)
                    base_labels.append(str(tail_label))
                    base_node_ids.append(int(tail_neighbor.global_id))
                    if str(tail_label).endswith("d2"):
                        tail_kind = "H1"
                        tail_reason = "fixed_branch_d2_requires_H1"
                    elif str(tail_label) in {
                        "single19d3",
                        "double19d3",
                        "single20d3",
                        "double20d3",
                    }:
                        tail_kind = "paired_like24"
                        tail_reason = "fixed_branch_requires_like24_partner"
                externally_closed = False
                contract = {
                    "connector_node_id": int(neighbor.global_id),
                    "connector_su_type": int(neighbor.su_type),
                    "connector_token_subtype": _node_label(neighbor),
                    "side_class": "N" if tail_kind in {"H1", "paired_like24"} else "",
                    "role_candidates": ["mainline_or_branch"],
                    "base_path_labels": list(base_labels),
                    "base_path_node_ids": list(base_node_ids),
                    "tail_closure_kind": str(tail_kind or "none"),
                    "tail_open_reason": str(tail_reason),
                    "externally_closed": bool(externally_closed),
                    "center_node_id": int(center_node_id),
                }
                if int(contract["connector_node_id"]) in mainline_fixed_connector_ids:
                    continue
                existing_ids = {
                    _safe_int(dict(item or {}).get("connector_node_id"), -1)
                    for item in list(branch_closure_contracts_by_anchor.get(int(anchor), []) or [])
                }
                if int(contract["connector_node_id"]) not in existing_ids:
                    branch_closure_contracts_by_anchor.setdefault(int(anchor), [])
                    branch_closure_contracts_by_anchor[int(anchor)].append(contract)

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
    if "upper_lower" in allowed_ring_kinds and class_type == "24_A" and len(node_ids) >= 2:
        ring_backbone_labels["upper_lower"] = [str(labels[-1])]
        ring_backbone_node_ids["upper_lower"] = [None if node_ids[-1] is None else int(node_ids[-1])]
        ring_backbone_span["upper_lower"] = 1
        ring_backbone_kind["upper_lower"] = "upper_fixed_anchor"
    ring_like24_units = _ring_like24_units_from_evidence(fragment)
    flex_reconcile_roles = _flex_reconcile_roles_from_evidence(fragment)
    flex_reconcile_priority = _flex_reconcile_priority_from_evidence(fragment)
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
        can_flex=bool(("flex_preferred" in set(flex_reconcile_roles)) or preferred == "flex_chain"),
        can_side=bool(preferred == "side_chain"),
        can_ring=bool(ring_roles) and not str(evidence.get("ring_block_reason", "")),
        target_roles=sorted({
            *(["flex_chain"] if ("flex_preferred" in set(flex_reconcile_roles) or preferred == "flex_chain") else []),
            *(["side_chain"] if preferred == "side_chain" else []),
            *(["fatty_ring"] if bool(ring_roles) and not str(evidence.get("ring_block_reason", "")) else []),
        }),
        evidence=evidence,
        diagnostics={
            "connector_profiles": profiles,
            "secondary_connector_profiles": secondary_profiles,
            "realization_modes": list(dict(fragment.evidence or {}).get("realization_modes", []) or []),
            "suppressed_support_node_ids": sorted({int(x) for x in suppressed_support_node_ids}),
            "required_branch_closures_by_anchor": dict(dict(fragment.evidence or {}).get("required_branch_closures_by_anchor", {}) or {}),
            "required_branch_closures_by_anchor_in_ring": dict(dict(fragment.evidence or {}).get("required_branch_closures_by_anchor_in_ring", {}) or {}),
            "branch_suppression_by_skeleton": dict(dict(fragment.evidence or {}).get("branch_suppression_by_skeleton", {}) or {}),
            "ring_slot_capabilities": dict(dict(fragment.evidence or {}).get("ring_slot_capabilities", {}) or {}),
        },
    )


def _project_su25_fragment(fragment: TopologyFragment, nodes_by_id: Dict[int, _NodeV3]) -> AssemblyFragmentView:
    center = _pick_center_node(fragment, nodes_by_id)
    if center is None:
        raise ValueError(f"missing center node for fragment {fragment.fragment_id}")
    neighbors = [
        nodes_by_id.get(_safe_int(node_id, -1))
        for node_id in list(getattr(center, "hop1_ids", []) or [])
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
    if elevens:
        labels.append(_node_label(elevens[0]))
        node_ids.append(int(elevens[0].global_id))
    else:
        open_left = True
    labels.append(_node_label(center))
    node_ids.append(int(center.global_id))
    if str(fragment.class_type) == "25_CD":
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
    branch_anchor_index = max(0, len(labels) - 1)
    fixed_22_stubs = [
        _make_custom_branch_stub(
            topology_notation="-22",
            node_id_notation=f"-{int(node.global_id)}",
            node_id=int(node.global_id),
            mode="existing_fixed22",
        )
        for node in fixed_22[:1]
    ]
    return AssemblyFragmentView(
        unit_id=int(fragment.fragment_id),
        fragment_id=int(fragment.fragment_id),
        family=str(fragment.family),
        class_type=str(fragment.class_type),
        topology_template=str(fragment.topology_template),
        source_node_ids=list(fragment.source_node_ids or []),
        main_labels=labels,
        main_node_ids=node_ids,
        canonical_labels=_canonical_labels_for_class(str(fragment.class_type), labels),
        open_left=bool(open_left),
        open_right=bool(open_right),
        branch_anchor_index=int(branch_anchor_index),
        unresolved_branch_kinds=["H"],
        actual_branch_stubs=list(fixed_22_stubs),
        branch_stubs_by_anchor={int(branch_anchor_index): list(fixed_22_stubs)},
        unresolved_branch_kinds_by_anchor={int(branch_anchor_index): ["H"]},
        real_branch_stubs_by_anchor={int(branch_anchor_index): list(fixed_22_stubs)},
        real_unresolved_branch_kinds_by_anchor={int(branch_anchor_index): ["H"]},
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
        diagnostics={},
    )


def project_fragment_views(
    classification: TopologyClassificationResult,
    nodes_by_id: Dict[int, _NodeV3],
) -> List[AssemblyFragmentView]:
    views: List[AssemblyFragmentView] = []
    for fragment in list(classification.fragments or []):
        family = str(fragment.family)
        if family == "basic":
            view = _project_basic_fragment(fragment, nodes_by_id)
        elif family == "double_d2_combo":
            view = _project_double_d2_fragment(fragment, nodes_by_id)
        elif family in {"like24", "like24_secondary", "like24_bridge"}:
            view = _project_like24_fragment(fragment, nodes_by_id)
        elif family == "su25":
            view = _project_su25_fragment(fragment, nodes_by_id)
        else:
            continue
        views.append(view)
    views.sort(key=lambda item: int(item.fragment_id))
    return views


class ExternalTopologyAssembler:
    def __init__(
        self,
        nodes: Sequence[_NodeV3],
        classification: TopologyClassificationResult,
        *,
        S_target: Any = None,
        E_target: Any = None,
        flex_audit: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.nodes = list(nodes or [])
        self.nodes_by_id = _build_node_lookup(self.nodes)
        self.classification = classification
        self.flex_audit = dict(flex_audit or {})
        if not self.flex_audit:
            from .phase_eval import _compute_flex_audit  # local import to avoid module cycle

            self.flex_audit = dict(_compute_flex_audit(self.nodes, classification, enforce=True) or {})
        self.selected_flex_candidate_ids = {
            int(value)
            for value in list(self.flex_audit.get("selected_candidate_fragment_ids", []) or [])
        }
        pool = dict(classification.resource_pool or {})
        self.allocator = PoolAllocator(dict(pool.get("remaining_ids_by_su", {}) or {}))
        self.views = project_fragment_views(classification, self.nodes_by_id)
        self.view_by_id = {int(view.fragment_id): view for view in self.views}
        self.products: List[TopologyAssemblyProduct] = []
        self.product_owner_by_node_id: Dict[int, int] = {}
        self.product_ownership_conflicts: List[Dict[str, Any]] = []
        self.pending_units: List[TopologyPendingUnit] = []
        self.pending_contracts: List[Dict[str, Any]] = []
        self.shortage_counts: Counter = Counter()
        self.fragment_state_by_id: Dict[int, str] = {
            int(view.fragment_id): "available"
            for view in self.views
        }
        self.ring_pending_reasons: Dict[int, List[str]] = {}
        self.ring_pending_attempts: Dict[int, Dict[str, Any]] = {}
        self.last_ring_materialization_reason: str = ""
        self.ring_pending_required_counts: Dict[int, Dict[str, int]] = {}
        self.invalid_topologies: List[Dict[str, Any]] = []
        self.invalid_topology_keys: Set[Tuple[str, str, str]] = set()
        self.composition_attempt_failures: List[Dict[str, Any]] = []
        self.contract_definition_conflicts: List[Dict[str, Any]] = []
        self.explicit_flex_count = int(sum(
            1
            for view in self.views
            if str(self._pending_kind_for_view(view)) == "explicit_flex"
        ))
        self.flex_bounds = _compute_flex_bounds(
            S_target=S_target,
            E_target=E_target,
            explicit_flex_count=int(self.explicit_flex_count),
            flex_audit=self.flex_audit,
        )

    def _commit_product(
        self,
        product: Optional[TopologyAssemblyProduct],
        *,
        allocator_snapshot: Optional[Dict[str, Any]] = None,
        branch_donors: Optional[List[AssemblyFragmentView]] = None,
        donor_snapshot: Optional[List[AssemblyFragmentView]] = None,
    ) -> bool:
        if product is None:
            if allocator_snapshot is not None:
                self.allocator.rollback(allocator_snapshot)
            if branch_donors is not None and donor_snapshot is not None:
                branch_donors[:] = donor_snapshot
            return False
        rendered_node_ids = [
            int(value)
            for value in re.findall(r"\d+", str(product.actual_topology_gid or ""))
        ]
        declared_source_ids = {
            int(value) for value in list(product.source_node_ids or [])
        }
        rendered_source_ids = set(rendered_node_ids)
        missing_claimed_ids = sorted(declared_source_ids - rendered_source_ids)
        duplicate_rendered_ids = sorted({
            int(value)
            for value in rendered_node_ids
            if rendered_node_ids.count(int(value)) > 1
        })
        if missing_claimed_ids or duplicate_rendered_ids:
            self._record_invalid_topology_attempt(
                product_kind=str(product.product_kind),
                actual_topology_su=str(product.actual_topology_su),
                actual_topology_gid=str(product.actual_topology_gid),
                topology_notation=str(product.actual_topology_su),
                validation_reason=(
                    f"claimed_source_not_materialized:{missing_claimed_ids}"
                    if missing_claimed_ids
                    else f"duplicate_materialized_node_ids:{duplicate_rendered_ids}"
                ),
                diagnostics={
                    "declared_source_node_ids": sorted(declared_source_ids),
                    "rendered_source_node_ids": sorted(rendered_source_ids),
                },
            )
            if allocator_snapshot is not None:
                self.allocator.rollback(allocator_snapshot)
            if branch_donors is not None and donor_snapshot is not None:
                branch_donors[:] = donor_snapshot
            return False
        if rendered_node_ids:
            product.source_node_ids = sorted(rendered_source_ids)
            product.materialized_node_ids = list(product.source_node_ids)
        gid_text = str(product.actual_topology_gid or "")
        main_gid_text = re.sub(r"\([^()]*\)", "", gid_text)
        mainline_ids = [int(value) for value in re.findall(r"\d+", main_gid_text)]
        branch_ids = [
            int(value)
            for branch_text in re.findall(r"\(([^()]*)\)", gid_text)
            for value in re.findall(r"\d+", branch_text)
        ]
        product.mainline_node_ids = list(dict.fromkeys(mainline_ids))
        product.branch_node_ids = list(dict.fromkeys(branch_ids))
        edge_plan: List[Dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(mainline_ids, mainline_ids[1:])):
            edge_plan.append({
                "left_node_id": int(left),
                "right_node_id": int(right),
                "slot_id": f"product:{len(self.products)}:main:{index}",
                "edge_role": "assembled_mainline",
                "fixed": False,
            })
        for branch_index, match in enumerate(re.finditer(r"(\d+)\(([^()]*)\)", gid_text)):
            anchor_id = int(match.group(1))
            path_ids = [int(value) for value in re.findall(r"\d+", match.group(2))]
            for edge_index, (left, right) in enumerate(zip([anchor_id] + path_ids, path_ids)):
                edge_plan.append({
                    "left_node_id": int(left),
                    "right_node_id": int(right),
                    "slot_id": f"product:{len(self.products)}:branch:{branch_index}:{edge_index}",
                    "edge_role": "assembled_branch",
                    "fixed": False,
                })
        product.edge_plan = edge_plan
        node_ids = [int(value) for value in list(product.source_node_ids or [])]
        if len(node_ids) != len(set(node_ids)):
            conflict = {
                "reason": "duplicate_node_within_product",
                "product_kind": str(product.product_kind),
                "node_ids": list(node_ids),
            }
            self.product_ownership_conflicts.append(conflict)
            if allocator_snapshot is not None:
                self.allocator.rollback(allocator_snapshot)
            if branch_donors is not None and donor_snapshot is not None:
                branch_donors[:] = donor_snapshot
            return False
        overlaps = sorted(
            int(node_id)
            for node_id in node_ids
            if int(node_id) in self.product_owner_by_node_id
        )
        if overlaps:
            self.product_ownership_conflicts.append({
                "reason": "duplicate_topology_ownership",
                "product_kind": str(product.product_kind),
                "actual_topology_gid": str(product.actual_topology_gid),
                "node_ids": overlaps,
                "existing_product_indices": sorted({
                    int(self.product_owner_by_node_id[node_id]) for node_id in overlaps
                }),
            })
            if allocator_snapshot is not None:
                self.allocator.rollback(allocator_snapshot)
            if branch_donors is not None and donor_snapshot is not None:
                branch_donors[:] = donor_snapshot
            return False
        product_index = int(len(self.products))
        self.products.append(product)
        for node_id in node_ids:
            self.product_owner_by_node_id[int(node_id)] = product_index
        committed_resource_ids: Dict[str, Set[int]] = defaultdict(set)
        existing_claims = {
            str(key): {int(value) for value in list(values or [])}
            for key, values in dict(product.claimed_resource_ids or {}).items()
        }
        fragment_source_ids = {
            int(value)
            for fragment_id in list(product.source_fragment_ids or [])
            for value in list(
                getattr(
                    next((
                        fragment
                        for fragment in list(getattr(self.classification, "fragments", []) or [])
                        if int(getattr(fragment, "fragment_id", -1)) == int(fragment_id)
                    ), None),
                    "claimed_node_ids",
                    [],
                )
                or []
            )
        }
        for node_id in node_ids:
            su_type = int(getattr(self.nodes_by_id.get(int(node_id)), "su_type", -1))
            if su_type in {11, 22, 23, 24}:
                committed_resource_ids[str(su_type)].add(int(node_id))
                if int(node_id) not in fragment_source_ids:
                    existing_claims.setdefault(str(su_type), set()).add(int(node_id))
        product.claimed_resource_ids = {
            str(key): sorted(int(value) for value in values)
            for key, values in existing_claims.items()
            if values
        }
        self.allocator.consume_existing_ids({
            str(key): sorted(int(value) for value in values)
            for key, values in committed_resource_ids.items()
            if values
        })
        return True

    def _reconcile_resource_ownership(
        self,
        ledger: TopologyPoolLedger,
    ) -> TopologyPoolLedger:
        allocator_remaining_ids_by_su = {
            str(key): sorted(set(int(value) for value in list(values or [])))
            for key, values in dict(ledger.remaining_ids_by_su or {}).items()
        }
        resource_types = sorted({10, 11, 12, 13, 22, 23, 24})
        global_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
        for node in list(self.nodes or []):
            su_type = int(getattr(node, "su_type", -1))
            if su_type in resource_types:
                global_ids_by_su[str(su_type)].append(int(getattr(node, "global_id", -1)))
        global_ids_by_su = {
            str(key): sorted(set(int(value) for value in list(values or []) if int(value) >= 0))
            for key, values in global_ids_by_su.items()
        }

        fixed_by_product: Dict[str, Dict[str, Any]] = {}
        fixed_node_ids: Set[int] = set()
        for product_index, product in enumerate(list(self.products or [])):
            product_dict = product.to_dict() if hasattr(product, "to_dict") else dict(product or {})
            source_ids = sorted({int(value) for value in list(product_dict.get("source_node_ids", []) or [])})
            counts = Counter(
                int(getattr(self.nodes_by_id.get(int(node_id)), "su_type", -1))
                for node_id in source_ids
                if int(getattr(self.nodes_by_id.get(int(node_id)), "su_type", -1)) in resource_types
            )
            product_key = f"{str(product_dict.get('product_kind', 'product'))}:{int(product_index)}"
            fixed_by_product[product_key] = {
                "source_node_ids": source_ids,
                "counts_by_su": {str(key): int(value) for key, value in sorted(counts.items())},
            }
            fixed_node_ids.update(source_ids)

        fragment_owned_node_ids: Set[int] = set()
        fixed_fragment_ids = {
            int(fragment_id)
            for product in list(self.products or [])
            for fragment_id in list(getattr(product, "source_fragment_ids", []) or [])
        }
        for fragment in list(getattr(self.classification, "fragments", []) or []):
            if int(getattr(fragment, "fragment_id", -1)) in fixed_fragment_ids:
                continue
            fragment_owned_node_ids.update(
                int(value)
                for value in list(getattr(fragment, "claimed_node_ids", []) or [])
            )
        fragment_owned_node_ids.difference_update(fixed_node_ids)

        fixed_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
        fragment_owned_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
        remaining_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
        for su_type in resource_types:
            for node_id in list(global_ids_by_su.get(str(su_type), []) or []):
                if node_id in fixed_node_ids:
                    fixed_ids_by_su[str(su_type)].append(node_id)
                elif node_id in fragment_owned_node_ids:
                    fragment_owned_ids_by_su[str(su_type)].append(node_id)
                else:
                    remaining_ids_by_su[str(su_type)].append(node_id)

        allocation_state_by_node: Dict[str, str] = {}
        for node_id in fixed_node_ids:
            allocation_state_by_node[str(node_id)] = "fixed"
        for node_id in fragment_owned_node_ids:
            allocation_state_by_node[str(node_id)] = "fragment_owned"
        for values in remaining_ids_by_su.values():
            for node_id in values:
                allocation_state_by_node[str(node_id)] = "remaining"

        allocator_source_ids_by_su = dict(getattr(ledger, "remaining_ids_by_su", {}) or {})
        allocator_remaining_ids_by_su = {
            str(key): sorted(
                set(allocator_source_ids_by_su.get(str(key), []))
                & set(remaining_ids_by_su.get(str(key), []))
            )
            for key in global_ids_by_su
        }
        allocator = getattr(self, "allocator", None)
        protected_counts = {
            str(key): max(0, int(getattr(allocator, "protected_counts_by_su", {}).get(str(key), 0) or 0))
            for key in global_ids_by_su
        }
        convertible_ids_by_su = {
            str(key): list(values[:max(0, len(values) - int(protected_counts.get(str(key), 0)))])
            for key, values in allocator_remaining_ids_by_su.items()
        }
        invariant_errors: List[str] = []
        for key, global_ids in global_ids_by_su.items():
            buckets = [
                set(fixed_ids_by_su.get(str(key), [])),
                set(fragment_owned_ids_by_su.get(str(key), [])),
                set(remaining_ids_by_su.get(str(key), [])),
            ]
            if any(buckets[left] & buckets[right] for left in range(3) for right in range(left + 1, 3)):
                invariant_errors.append(f"su{key}:ownership_overlap")
            if set(global_ids) - set().union(*buckets):
                invariant_errors.append(f"su{key}:unaccounted")

        ledger.global_ids_by_su = global_ids_by_su
        ledger.fixed_ids_by_su = fixed_ids_by_su
        ledger.fragment_owned_ids_by_su = fragment_owned_ids_by_su
        ledger.diagnostic_remaining_ids_by_su = deepcopy(remaining_ids_by_su)
        ledger.allocator_remaining_ids_by_su = deepcopy(allocator_remaining_ids_by_su)
        ledger.remaining_ids_by_su = deepcopy(allocator_remaining_ids_by_su)
        ledger.convertible_ids_by_su = deepcopy(convertible_ids_by_su)
        allocator_initial_source = dict(
            getattr(getattr(self, "allocator", None), "initial_ids_by_su", {}) or {}
        )
        if not allocator_initial_source:
            allocator_initial_source = dict(ledger.initial_ids_by_su or ledger.remaining_ids_by_su or {})
        allocator_initial_ids_by_su = {
            str(key): sorted(set(int(value) for value in list(values or [])))
            for key, values in allocator_initial_source.items()
        }
        ledger.initial_ids_by_su = deepcopy(allocator_initial_ids_by_su)
        ledger.consumed_ids_by_su = {
            str(key): sorted(
                set(allocator_initial_ids_by_su.get(str(key), []) or [])
                - set(allocator_remaining_ids_by_su.get(str(key), []) or [])
            )
            for key in allocator_initial_ids_by_su
        }
        ledger.global_counts = {str(key): int(len(values)) for key, values in global_ids_by_su.items()}
        ledger.fixed_counts_by_su = {str(key): int(len(values)) for key, values in fixed_ids_by_su.items()}
        ledger.fragment_owned_counts_by_su = {str(key): int(len(values)) for key, values in fragment_owned_ids_by_su.items()}
        ledger.diagnostic_remaining_counts = {str(key): int(len(values)) for key, values in remaining_ids_by_su.items()}
        ledger.allocator_remaining_counts = {str(key): int(len(values)) for key, values in allocator_remaining_ids_by_su.items()}
        ledger.remaining_counts = dict(ledger.allocator_remaining_counts)
        ledger.convertible_counts = {
            str(key): int(len(values))
            for key, values in ledger.convertible_ids_by_su.items()
        }
        ledger.protected_counts = dict(protected_counts)
        ledger.unusable_counts = {
            str(key): max(0, int(ledger.diagnostic_remaining_counts.get(str(key), 0)) - int(ledger.convertible_counts.get(str(key), 0)))
            for key in global_ids_by_su
        }
        ledger.invariant_errors = list(invariant_errors)
        ledger.initial_counts = {str(key): int(len(values)) for key, values in ledger.initial_ids_by_su.items()}
        ledger.consumed_counts = {
            str(key): int(len(values))
            for key, values in ledger.consumed_ids_by_su.items()
        }
        ledger.snapshot_stage = "post_assembly_reconciled"
        ledger.fixed_node_ids = sorted(fixed_node_ids)
        ledger.fragment_owned_node_ids = sorted(fragment_owned_node_ids)
        ledger.fixed_by_product = fixed_by_product
        ledger.allocation_state_by_node = allocation_state_by_node
        return ledger

    def _fragment_available(self, fragment_id: int) -> bool:
        return str(self.fragment_state_by_id.get(int(fragment_id), "available")) == "available"

    def _reserve_fragments(self, fragment_ids: Iterable[int]) -> bool:
        selected = sorted({int(value) for value in list(fragment_ids or [])})
        if not selected or any(not self._fragment_available(fragment_id) for fragment_id in selected):
            return False
        for fragment_id in selected:
            self.fragment_state_by_id[int(fragment_id)] = "reserved"
        return True

    def _release_reserved_fragments(self, fragment_ids: Iterable[int]) -> None:
        for fragment_id in sorted({int(value) for value in list(fragment_ids or [])}):
            if str(self.fragment_state_by_id.get(int(fragment_id), "")) == "reserved":
                self.fragment_state_by_id[int(fragment_id)] = "available"

    @staticmethod
    def _pending_identity_overlaps(
        pending: TopologyPendingUnit,
        fragment_ids: Set[int],
        claimed_node_ids: Set[int],
    ) -> bool:
        pending_fragment_ids = {
            int(value) for value in list(getattr(pending, "source_fragment_ids", []) or [])
        }
        pending_claimed_ids = {
            int(value)
            for value in list(
                getattr(pending, "core_node_ids", [])
                or getattr(pending, "source_node_ids", [])
                or []
            )
        }
        return bool(pending_fragment_ids & fragment_ids or pending_claimed_ids & claimed_node_ids)

    @staticmethod
    def _contract_identity_overlaps(
        contract: Dict[str, Any],
        fragment_ids: Set[int],
        claimed_node_ids: Set[int],
    ) -> bool:
        data = dict(contract or {})
        contract_fragment_ids = {
            int(value) for value in list(data.get("source_fragment_ids", []) or [])
        }
        contract_claimed_ids = {
            int(value)
            for value in list(
                data.get("claimed_source_node_ids", [])
                or data.get("core_node_ids", [])
                or data.get("source_node_ids", [])
                or []
            )
        }
        return bool(contract_fragment_ids & fragment_ids or contract_claimed_ids & claimed_node_ids)

    def _invalidate_pending_identity(
        self,
        *,
        fragment_ids: Iterable[int],
        claimed_node_ids: Iterable[int],
    ) -> None:
        fragment_id_set = {int(value) for value in list(fragment_ids or [])}
        claimed_node_id_set = {int(value) for value in list(claimed_node_ids or [])}
        self.pending_units = [
            pending
            for pending in list(self.pending_units or [])
            if not self._pending_identity_overlaps(pending, fragment_id_set, claimed_node_id_set)
        ]
        self.pending_contracts = [
            contract
            for contract in list(self.pending_contracts or [])
            if not self._contract_identity_overlaps(contract, fragment_id_set, claimed_node_id_set)
        ]

    def _mark_fragments_completed(self, fragment_ids: Iterable[int]) -> None:
        completed_fragment_ids = {int(value) for value in list(fragment_ids or [])}
        for fragment_id in completed_fragment_ids:
            frag_id = int(fragment_id)
            self.fragment_state_by_id[frag_id] = "completed"
            self.ring_pending_reasons.pop(frag_id, None)
            self.ring_pending_attempts.pop(frag_id, None)
            self.ring_pending_required_counts.pop(frag_id, None)
        used_node_ids = {
            int(node_id)
            for fragment_id in list(fragment_ids or [])
            for node_id in list(
                getattr(self.view_by_id.get(int(fragment_id)), "source_node_ids", [])
                if self.view_by_id.get(int(fragment_id)) is not None
                else []
            )
        }
        self._invalidate_pending_identity(
            fragment_ids=completed_fragment_ids,
            claimed_node_ids=used_node_ids,
        )

    def _drop_pending_overlapping_product(self, product: TopologyAssemblyProduct) -> None:
        materialized_node_ids = set(
            int(node_id)
            for node_id in list(
                getattr(product, "materialized_node_ids", [])
                or getattr(product, "source_node_ids", [])
                or []
            )
        )
        if not materialized_node_ids:
            return
        self._invalidate_pending_identity(
            fragment_ids=list(getattr(product, "source_fragment_ids", []) or []),
            claimed_node_ids=materialized_node_ids,
        )

    def _available_views(self, predicate: Any) -> List[AssemblyFragmentView]:
        return [
            view for view in self.views
            if self._fragment_available(int(view.fragment_id)) and bool(predicate(view))
        ]

    def _pending_target_role(
        self,
        pending_kind: str,
        views: Sequence[AssemblyFragmentView],
    ) -> str:
        available_roles = {
            str(role)
            for view in list(views or [])
            for role in list(getattr(view, "target_roles", []) or [])
            if str(role)
        }
        required_role = ""
        if str(pending_kind) == "pending_ring_candidate":
            required_role = "fatty_ring"
        elif str(pending_kind) == "pending_su25":
            required_role = "side_chain"
        else:
            chain_roles = available_roles & {"flex_chain", "side_chain"}
            if len(chain_roles) == 1:
                required_role = next(iter(chain_roles))
        if required_role and required_role in available_roles:
            return str(required_role)
        raise ValueError(
            f"pending target role is undefined or ambiguous: kind={pending_kind} roles={sorted(available_roles)}"
        )

    @staticmethod
    def _canonical_contract_kind(
        pending_kind: str,
        reason: str,
        *,
        required_counts: Dict[str, int],
        has_branch_contracts: bool,
    ) -> str:
        reason_name = str(reason or "")
        if reason_name in {"fixed_branch_partner_missing", "fused_bridgehead_missing"}:
            return "ring_partner" if reason_name == "fixed_branch_partner_missing" else "structural_blocker"
        if str(pending_kind) == "pending_ring_candidate":
            return "ring_partner" if not dict(required_counts or {}) else "mainline_closure"
        if str(pending_kind) == "pending_su25":
            return "fixed_su25"
        if bool(has_branch_contracts) or "branch" in reason_name:
            return "branch_closure" if dict(required_counts or {}) else "structural_blocker"
        if dict(required_counts or {}):
            return "mainline_closure"
        return "structural_blocker"

    def _record_pending(
        self,
        *,
        unit_id: int,
        pending_kind: str,
        views: Sequence[AssemblyFragmentView],
        required_counts: Dict[str, int],
        available_counts: Dict[str, int],
        reason: str,
        partial_render: TopologyPathRender,
        actual_render: Optional[TopologyPathRender] = None,
    ) -> None:
        actual = actual_render or partial_render
        source_fragment_ids, source_fragment_classes, source_node_ids = _collect_source_metadata(list(views or []))
        fragment_by_id = {
            int(getattr(fragment, "fragment_id", -1)): fragment
            for fragment in list(getattr(self.classification, "fragments", []) or [])
        }
        core_node_ids = sorted({
            int(node_id)
            for fragment_id in list(source_fragment_ids or [])
            for node_id in list(
                getattr(fragment_by_id.get(int(fragment_id)), "claimed_node_ids", [])
                or []
            )
        })
        rendered_branch_source_ids = {
            int(node_id)
            for view in list(views or [])
            for stub_map in (
                getattr(view, "branch_stubs_by_anchor", {}) or {},
                getattr(view, "real_branch_stubs_by_anchor", {}) or {},
            )
            for stubs in dict(stub_map or {}).values()
            for stub in list(stubs or [])
            for node_id in list(getattr(stub, "source_node_ids", []) or [])
        }
        core_node_ids = sorted(
            set(core_node_ids)
            | {int(value) for value in list(source_node_ids or [])}
            | rendered_branch_source_ids
        )
        if not core_node_ids:
            core_node_ids = sorted(set(int(value) for value in list(source_node_ids or [])))
        context_node_ids = sorted({
            int(node_id)
            for view in list(views or [])
            for node_id in list(getattr(view, "context_node_ids", []) or [])
        })
        source_node_ids = list(core_node_ids)
        rendered_pending_ids = {
            int(value)
            for value in re.findall(r"\d+", str(actual.node_id_notation or ""))
        }
        render_validation_errors: List[str] = []
        suppressed_support_node_ids = sorted(({
            int(value)
            for view in list(views or [])
            for value in list(getattr(view, "suppressed_support_node_ids", []) or [])
        } | {
            int(value)
            for value in list(getattr(actual, "suppressed_support_node_ids", []) or [])
        }) & set(core_node_ids))
        expected_render_ids = set(core_node_ids) - set(suppressed_support_node_ids)
        if expected_render_ids != rendered_pending_ids:
            missing_render_ids = sorted(expected_render_ids - rendered_pending_ids)
            extra_render_ids = sorted(rendered_pending_ids - set(core_node_ids))
            render_validation_errors.append(
                f"render_ownership_mismatch:missing={missing_render_ids}:extra={extra_render_ids}"
            )
        completed_node_ids = {
            int(value)
            for product in list(self.products or [])
            for value in list(
                getattr(product, "materialized_node_ids", [])
                or getattr(product, "source_node_ids", [])
                or []
            )
        }
        if set(core_node_ids) & completed_node_ids:
            return
        open_mainline_slots = [
            slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
            for fragment_id in source_fragment_ids
            for slot in list(getattr(fragment_by_id.get(int(fragment_id)), "open_mainline_slots", []) or [])
        ]
        branch_slots = [
            slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
            for fragment_id in source_fragment_ids
            for slot in list(getattr(fragment_by_id.get(int(fragment_id)), "branch_contracts", []) or [])
        ]
        slot_counts = _slot_required_counts(open_mainline_slots + branch_slots)
        if str(reason) in {"fixed_branch_partner_missing", "fused_bridgehead_missing"}:
            required_counts = {}
        elif slot_counts:
            required_counts = dict(slot_counts)
        target_role = self._pending_target_role(str(pending_kind), list(views or []))
        contract_kind = self._canonical_contract_kind(
            str(pending_kind),
            str(reason),
            required_counts=dict(required_counts or {}),
            has_branch_contracts=bool(branch_slots),
        )
        pending_key = _pending_owner_key(unit_id, source_fragment_ids, core_node_ids)
        for existing in list(self.pending_units or []):
            existing_key = _pending_owner_key(
                int(getattr(existing, "unit_id", 0)),
                list(getattr(existing, "source_fragment_ids", []) or []),
                list(getattr(existing, "core_node_ids", []) or getattr(existing, "source_node_ids", []) or []),
            )
            if existing_key == pending_key:
                existing.attempt_reasons = sorted({
                    str(value)
                    for value in list(getattr(existing, "attempt_reasons", []) or [])
                    + [str(getattr(existing, "reason", "") or ""), str(reason or "")]
                    if str(value)
                })
                existing.validation_errors = sorted({
                    str(value)
                    for value in list(getattr(existing, "validation_errors", []) or [])
                    + list(render_validation_errors or [])
                    if str(value)
                })
                existing.required_counts = {
                    str(key): int(value) for key, value in dict(required_counts or {}).items()
                }
                existing.available_counts = {
                    str(key): int(value) for key, value in dict(available_counts or {}).items()
                }
                existing.infeasibility_proof = {
                    **dict(getattr(existing, "infeasibility_proof", {}) or {}),
                    "reason": str(reason),
                    "required_counts": dict(existing.required_counts),
                    "available_counts": dict(existing.available_counts),
                    "suppressed_support_node_ids": list(suppressed_support_node_ids),
                    "validation_errors": list(existing.validation_errors),
                }
                return
        missing_counts = {
            str(key): max(0, int(required_counts.get(str(key), 0)) - int(available_counts.get(str(key), 0)))
            for key in {str(k) for k in dict(required_counts or {}).keys()}
        }
        self._record_pending_contract(
            unit_id=int(unit_id),
            pending_kind=str(pending_kind),
            views=list(views or []),
            required_counts=dict(required_counts or {}),
            reason=str(reason),
            target_role=str(target_role),
            contract_kind=str(contract_kind),
            expected_topology=str(actual.topology_notation or ""),
            mainline_slots=open_mainline_slots,
            branch_slots=branch_slots,
            claimed_source_node_ids=core_node_ids,
            context_node_ids=context_node_ids,
        )
        self.pending_units.append(TopologyPendingUnit(
            unit_id=int(unit_id),
            pending_kind=str(pending_kind),
            contract_kind=str(contract_kind),
            target_role=str(target_role),
            real_topology_partial=str(actual.topology_notation),
            real_node_partial=str(actual.node_id_notation),
            source_fragment_ids=source_fragment_ids,
            source_fragment_classes=source_fragment_classes,
            source_node_ids=source_node_ids,
            core_node_ids=core_node_ids,
            phase="external_structural_closure",
            subphase=str(self._pending_subphase(pending_kind, reason)),
            required_counts={str(key): int(value) for key, value in dict(required_counts or {}).items()},
            available_counts={str(key): int(value) for key, value in dict(available_counts or {}).items()},
            reason=str(reason),
            context_node_ids=context_node_ids,
            open_mainline_slots=open_mainline_slots,
            branch_slots=branch_slots,
            infeasibility_proof={
                "reason": str(reason),
                "required_counts": {str(key): int(value) for key, value in dict(required_counts or {}).items()},
                "available_counts": {str(key): int(value) for key, value in dict(available_counts or {}).items()},
                "missing_counts": {str(key): int(value) for key, value in missing_counts.items() if int(value) > 0},
                "suppressed_support_node_ids": list(suppressed_support_node_ids),
                "validation_errors": list(render_validation_errors),
            },
            attempt_reasons=[str(reason)] if str(reason) else [],
            validation_errors=list(render_validation_errors),
        ))

    def _consolidate_pending_owners(self) -> None:
        canonical: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], TopologyPendingUnit] = {}
        for pending in list(self.pending_units or []):
            key = _pending_owner_key(
                int(getattr(pending, "unit_id", 0)),
                list(getattr(pending, "source_fragment_ids", []) or []),
                list(getattr(pending, "core_node_ids", []) or getattr(pending, "source_node_ids", []) or []),
            )
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = pending
            elif existing.to_dict() != pending.to_dict():
                self.contract_definition_conflicts.append({
                    "reason": "duplicate_pending_definition",
                    "owner_key": list(key),
                    "existing": existing.to_dict(),
                    "incoming": pending.to_dict(),
                })
        self.pending_units = list(canonical.values())

    def _pending_subphase(
        self,
        pending_kind: str,
        reason: str,
    ) -> str:
        kind = str(pending_kind or "")
        reason_name = str(reason or "")
        if kind == "pending_ring_candidate":
            return "real_like24_ring_assembly"
        if kind in {"explicit_flex", "explicit_side"}:
            return "explicit_branch_closure"
        if reason_name in {"fixed_branch_H1_missing", "fixed_branch_role_conflict"}:
            return "explicit_branch_closure"
        if reason_name == "fixed_branch_partner_missing":
            return "real_like24_ring_assembly"
        return "remaining_external_halfclosed_closure"

    def _view_external_pending_profile(self, view: Optional[AssemblyFragmentView]) -> Dict[str, Any]:
        if view is None:
            return {}
        evidence = dict(getattr(view, "evidence", {}) or {})
        return dict(evidence.get("external_pending_profile", {}) or {})

    def _record_pending_contract(
        self,
        *,
        unit_id: int,
        pending_kind: str,
        views: Sequence[AssemblyFragmentView],
        required_counts: Dict[str, int],
        reason: str,
        target_role: str,
        contract_kind: str,
        expected_topology: str,
        mainline_slots: Sequence[Dict[str, Any]],
        branch_slots: Sequence[Dict[str, Any]],
        claimed_source_node_ids: Sequence[int],
        context_node_ids: Sequence[int],
    ) -> None:
        if not str(target_role):
            raise ValueError("canonical topology contract requires target_role")
        if not str(contract_kind):
            raise ValueError("canonical topology contract requires contract_kind")
        source_fragment_ids, source_fragment_classes, _ = _collect_source_metadata(list(views or []))
        blocker_kind = str(reason) if str(contract_kind) == "structural_blocker" else ""
        contract = {
            "contract_id": int(unit_id),
            "contract_kind": str(contract_kind),
            "target_role": str(target_role),
            "phase": "external_structural_closure",
            "subphase": self._pending_subphase(str(pending_kind), str(reason)),
            "source_fragment_ids": list(source_fragment_ids or []),
            "source_fragment_classes": list(source_fragment_classes or []),
            "claimed_source_node_ids": [int(value) for value in list(claimed_source_node_ids or [])],
            "context_node_ids": [int(value) for value in list(context_node_ids or [])],
            "mainline_slots": [dict(item or {}) for item in list(mainline_slots or [])],
            "branch_slots": [dict(item or {}) for item in list(branch_slots or [])],
            "required_counts": {
                str(key): int(value)
                for key, value in dict(required_counts or {}).items()
                if int(value) > 0
            },
            "expected_topology": str(expected_topology or ""),
            "reason": str(reason or ""),
            "blocker_kind": str(blocker_kind),
            "diagnostics": {"public_pending_kind": str(pending_kind or "")},
        }
        contract_key = stage3_contract_owner_key(contract)
        for existing in list(self.pending_contracts or []):
            if stage3_contract_owner_key(dict(existing or {})) != contract_key:
                continue
            if dict(existing or {}) != contract:
                self.contract_definition_conflicts.append({
                    "reason": "duplicate_contract_definition",
                    "owner_key": list(contract_key),
                    "existing": dict(existing or {}),
                    "incoming": dict(contract),
                })
            return
        self.pending_contracts.append(contract)

    def _branch_from_view(self, view: AssemblyFragmentView) -> BranchClosureResult:
        render = view.real_render()
        branch_topology = _strip_open_markers(render.topology_notation)
        branch_node_ids = _strip_open_markers(render.node_id_notation)
        return BranchClosureResult(
            branch_topology=f"-{branch_topology}" if branch_topology else "",
            branch_node_ids=f"-{branch_node_ids}" if branch_node_ids else "",
            branch_kind=str(view.class_type),
            mode=f"fragment:{view.class_type}",
            anchor_index=int(view.branch_anchor_index),
            suppressed_support_node_ids=list(render.suppressed_support_node_ids or []),
            source_fragment_ids=[int(view.fragment_id)],
            source_node_ids=list(view.source_node_ids or []),
            pool_node_ids_by_su={},
            main_labels=[token for token in _strip_open_markers(render.topology_notation).split("-") if token],
            main_node_ids=[int(token) for token in _strip_open_markers(render.node_id_notation).split("-") if token],
        )

    def _claim_branch_from_pool(self, branch_kind: str) -> Optional[BranchClosureResult]:
        snapshot = self.allocator.snapshot()
        if str(branch_kind) == "FIXED22":
            claimed_22 = self.allocator.claim(22, 1)
            if claimed_22 is None:
                self.allocator.rollback(snapshot)
                return None
            return BranchClosureResult(
                branch_topology="-22",
                branch_node_ids=f"-{_node_id_label(claimed_22[0])}",
                branch_kind="FIXED22",
                mode="fixed22_pool",
                suppressed_support_node_ids=[],
                pool_node_ids_by_su={"22": list(claimed_22)},
                main_labels=["22"],
                main_node_ids=[int(claimed_22[0])],
            )
        if str(branch_kind) == "H1":
            claimed_23 = self.allocator.claim(23, 1)
            claimed_22 = self.allocator.claim(22, 1) if claimed_23 is not None else None
            if claimed_23 is None or claimed_22 is None:
                self.allocator.rollback(snapshot)
                return None
            return BranchClosureResult(
                branch_topology="-23-22",
                branch_node_ids=f"-{claimed_23[0]}-{claimed_22[0]}",
                branch_kind="H1",
                mode="pool:23-22",
                suppressed_support_node_ids=[],
                pool_node_ids_by_su={"23": list(claimed_23), "22": list(claimed_22)},
                main_labels=["23", "22"],
                main_node_ids=[int(claimed_23[0]), int(claimed_22[0])],
            )
        claimed_23 = self.allocator.claim(23, 1)
        claimed_22 = self.allocator.claim(22, 1) if claimed_23 is not None else None
        if claimed_23 is not None and claimed_22 is not None:
            return BranchClosureResult(
                branch_topology="-23-22",
                branch_node_ids=f"-{claimed_23[0]}-{claimed_22[0]}",
                branch_kind="H",
                mode="pool:23-22",
                suppressed_support_node_ids=[],
                pool_node_ids_by_su={"23": list(claimed_23), "22": list(claimed_22)},
                main_labels=["23", "22"],
                main_node_ids=[int(claimed_23[0]), int(claimed_22[0])],
            )
        self.allocator.rollback(snapshot)
        return None

    def _resolve_branch_closure(
        self,
        branch_kind: str,
        branch_donors: List[AssemblyFragmentView],
        *,
        anchor_index: int,
        attach_role: str = "center_branch",
        target_path: str = "center_side_branch",
    ) -> Optional[BranchClosureResult]:
        operator = BRANCH_CLOSURE_OPERATOR_SPECS.get(str(branch_kind), ClosureOperatorSpec(name=str(branch_kind), scope="branch"))
        result = self._claim_branch_from_pool(branch_kind)
        if result is not None:
            result.anchor_index = int(anchor_index)
            result.attach_role = str(attach_role)
            result.target_path = str(target_path)
            return result
        return None

    def _consume_branch_placeholder(
        self,
        branch_kind: str,
        branch_donors: List[AssemblyFragmentView],
        *,
        anchor_index: int,
        attach_role: str = "center_branch",
        target_path: str = "center_side_branch",
    ) -> Optional[BranchClosureResult]:
        return self._resolve_branch_closure(
            str(branch_kind),
            branch_donors,
            anchor_index=int(anchor_index),
            attach_role=str(attach_role),
            target_path=str(target_path),
        )

    def _fixed_contract_required_counts(
        self,
        contract: Dict[str, Any],
    ) -> Dict[str, int]:
        tail_kind = str(dict(contract or {}).get("tail_closure_kind", "none") or "none")
        if tail_kind == "H1":
            return {"23": 1, "22": 1}
        return {}

    def _materialize_fixed_branch_contract(
        self,
        contract: Dict[str, Any],
        *,
        anchor_index: int,
        tail_result: Optional[BranchClosureResult] = None,
    ) -> BranchClosureResult:
        contract_local = dict(contract or {})
        base_labels = [
            str(label)
            for label in list(contract_local.get("base_path_labels", []) or [])
            if str(label)
        ]
        base_node_ids = [
            int(node_id)
            for node_id in list(contract_local.get("base_path_node_ids", []) or [])
            if _safe_int(node_id, -1) >= 0
        ]
        all_labels = list(base_labels)
        all_node_ids = list(base_node_ids)
        pool_ids_by_su: Dict[str, List[int]] = {}
        mode = "fixed_contract:closed"
        if tail_result is not None:
            all_labels.extend(str(label) for label in list(tail_result.main_labels or []) if str(label))
            all_node_ids.extend(int(node_id) for node_id in list(tail_result.main_node_ids or []) if node_id is not None)
            pool_ids_by_su = self._merge_pool_ids([dict(tail_result.pool_node_ids_by_su or {})])
            mode = f"fixed_contract:{str(contract_local.get('tail_closure_kind', 'tail'))}"
        branch_topology = ""
        branch_node_ids = ""
        if all_labels:
            branch_topology = "-" + "-".join(str(label) for label in all_labels)
        if all_node_ids:
            branch_node_ids = "-" + "-".join(str(int(node_id)) for node_id in all_node_ids)
        return BranchClosureResult(
            branch_topology=str(branch_topology),
            branch_node_ids=str(branch_node_ids),
            branch_kind="actual_fixed_connection_contract",
            mode=str(mode),
            anchor_index=int(anchor_index),
            suppressed_support_node_ids=[],
            source_fragment_ids=[],
            source_node_ids=[int(node_id) for node_id in all_node_ids],
            pool_node_ids_by_su=pool_ids_by_su,
            main_labels=list(all_labels),
            main_node_ids=[int(node_id) for node_id in all_node_ids],
        )

    def _resolve_fixed_branch_contract(
        self,
        contract: Dict[str, Any],
        branch_donors: List[AssemblyFragmentView],
        *,
        anchor_index: int,
    ) -> Tuple[Optional[BranchClosureResult], Dict[str, int], str]:
        contract_local = dict(contract or {})
        role_candidates = {
            str(value)
            for value in list(contract_local.get("role_candidates", []) or [])
            if str(value)
        }
        if "mainline_only" in role_candidates:
            return None, {}, "fixed_branch_role_conflict"
        if bool(contract_local.get("externally_closed", False)):
            return self._materialize_fixed_branch_contract(contract_local, anchor_index=int(anchor_index)), {}, ""
        tail_kind = str(contract_local.get("tail_closure_kind", "none") or "none")
        if tail_kind == "H1":
            required_counts = self._fixed_contract_required_counts(contract_local)
            tail_result = self._resolve_branch_closure(
                "H1",
                branch_donors,
                anchor_index=int(anchor_index),
                attach_role="fixed_connection_branch",
                target_path="fixed_connection_branch",
            )
            if tail_result is None:
                return None, dict(required_counts or {}), "fixed_branch_H1_missing"
            return (
                self._materialize_fixed_branch_contract(contract_local, anchor_index=int(anchor_index), tail_result=tail_result),
                dict(required_counts or {}),
                "",
            )
        if tail_kind == "paired_like24":
            return None, {}, "fixed_branch_partner_missing"
        return None, {}, "external_branch_closure_pending"

    def _resolve_branches(
        self,
        view: AssemblyFragmentView,
        branch_donors: List[AssemblyFragmentView],
        *,
        branch_kind_map_override: Optional[Dict[int, List[str]]] = None,
        branch_contract_map_override: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Tuple[Optional[List[BranchClosureResult]], Dict[str, int], Dict[str, Any]]:
        snapshot = self.allocator.snapshot()
        branch_stub_map = dict(view.real_branch_stubs_map() or {})
        branch_kind_map = dict(branch_kind_map_override or view.real_branch_kinds_by_anchor() or {})
        branch_contract_map = dict(branch_contract_map_override or view.fixed_branch_contracts_by_anchor() or {})
        branch_specs = [] if branch_kind_map_override is not None else list(view.branch_closure_specs() or [])
        if branch_specs:
            branch_kind_map = {}
        branch_closures: List[BranchClosureResult] = []
        required = Counter()
        diagnostics = {
            "branch_modes": [],
            "branch_specs": list(branch_specs or []),
            "branch_contract_blockers": {},
        }
        for anchor in sorted(int(value) for value in branch_contract_map.keys()):
            for contract in list(branch_contract_map.get(int(anchor), []) or []):
                contract_result, contract_required, blocker = self._resolve_fixed_branch_contract(
                    dict(contract or {}),
                    branch_donors,
                    anchor_index=int(anchor),
                )
                for key, value in dict(contract_required or {}).items():
                    required[str(key)] += int(value)
                if contract_result is None:
                    self.allocator.rollback(snapshot)
                    if str(blocker):
                        diagnostics["branch_contract_blockers"].setdefault(str(anchor), [])
                        diagnostics["branch_contract_blockers"][str(anchor)].append(str(blocker))
                    diagnostics["unresolved_branch_contract_count"] = int(sum(
                        len(list(values or []))
                        for values in dict(diagnostics.get("branch_contract_blockers", {}) or {}).values()
                    ))
                    return None, {str(k): int(v) for k, v in required.items()}, diagnostics
                branch_closures.append(contract_result)
                diagnostics["branch_modes"].append(str(contract_result.mode))
        for spec in list(branch_specs or []):
            branch_kind = str(dict(spec or {}).get("closure_kind", ""))
            anchor_raw = dict(spec or {}).get("anchor", "center")
            attach_role = str(dict(spec or {}).get("attach_role", "center_branch"))
            target_path = str(dict(spec or {}).get("target_path", "center_side_branch"))
            anchor_index = int(view.branch_anchor_index) if str(anchor_raw) == "center" else _safe_int(anchor_raw, int(view.branch_anchor_index))
            if not branch_kind:
                continue
            if str(branch_kind) == "FIXED22":
                required["22"] += 1
            elif str(branch_kind) == "H1":
                required["23"] += 1
                required["22"] += 1
            else:
                required["23"] += 1
                required["22"] += 1
            branch_result = self._consume_branch_placeholder(
                str(branch_kind),
                branch_donors,
                anchor_index=int(anchor_index),
                attach_role=str(attach_role),
                target_path=str(target_path),
            )
            if branch_result is None:
                self.allocator.rollback(snapshot)
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            branch_closures.append(branch_result)
            diagnostics["branch_modes"].append(str(branch_result.mode))
        all_anchors = sorted({
            int(anchor)
            for anchor in set(branch_stub_map.keys()) | set(branch_kind_map.keys()) | set(branch_contract_map.keys())
        })
        for anchor in all_anchors:
            for branch_kind in list(branch_kind_map.get(int(anchor), []) or []):
                if str(branch_kind) == "FIXED22":
                    required["22"] += 1
                elif str(branch_kind) == "H1":
                    required["23"] += 1
                    required["22"] += 1
                else:
                    required["23"] += 1
                    required["22"] += 1
                branch_result = self._consume_branch_placeholder(
                    str(branch_kind),
                    branch_donors,
                    anchor_index=int(anchor),
                )
                if branch_result is None:
                    self.allocator.rollback(snapshot)
                    return None, {str(k): int(v) for k, v in required.items()}, diagnostics
                branch_closures.append(branch_result)
                diagnostics["branch_modes"].append(str(branch_result.mode))
            for actual_stub in list(branch_stub_map.get(int(anchor), []) or []):
                branch_closures.append(BranchClosureResult(
                    branch_topology=str(actual_stub.topology_notation),
                    branch_node_ids=str(actual_stub.node_id_notation),
                    branch_kind=str(actual_stub.branch_kind),
                    mode=str(actual_stub.mode),
                    anchor_index=int(anchor),
                    suppressed_support_node_ids=[],
                    source_fragment_ids=list(actual_stub.source_fragment_ids or []),
                    source_node_ids=list(actual_stub.source_node_ids or []),
                    pool_node_ids_by_su={},
                    main_labels=[str(actual_stub.topology_notation)],
                    main_node_ids=[int(actual_stub.node_id_notation)] if str(actual_stub.node_id_notation).isdigit() else [],
                ))
        diagnostics["unresolved_branch_contract_count"] = 0
        return list(branch_closures), {str(k): int(v) for k, v in required.items()}, diagnostics

    def _claim_f1(self) -> Optional[BranchClosureResult]:
        snapshot = self.allocator.snapshot()
        short_23 = self.allocator.claim(23, 1)
        short_11 = self.allocator.claim(11, 1) if short_23 is not None else None
        if short_23 is not None and short_11 is not None:
            return BranchClosureResult(
                branch_topology="11-23",
                branch_node_ids=f"{short_11[0]}-{short_23[0]}",
                branch_kind="F1",
                mode="pool:F1_short",
                suppressed_support_node_ids=[],
                pool_node_ids_by_su={"23": list(short_23), "11": list(short_11)},
                main_labels=["11", "23"],
                main_node_ids=[int(short_11[0]), int(short_23[0])],
            )
        self.allocator.rollback(snapshot)
        return None

    def _claim_f2(self) -> Optional[BranchClosureResult]:
        snapshot = self.allocator.snapshot()
        short_23 = self.allocator.claim(23, 1)
        short_22 = self.allocator.claim(22, 1) if short_23 is not None else None
        if short_23 is not None and short_22 is not None:
            return BranchClosureResult(
                branch_topology="23-22",
                branch_node_ids=f"{short_23[0]}-{short_22[0]}",
                branch_kind="F2",
                mode="pool:F2_short",
                suppressed_support_node_ids=[],
                pool_node_ids_by_su={"23": list(short_23), "22": list(short_22)},
                main_labels=["23", "22"],
                main_node_ids=[int(short_23[0]), int(short_22[0])],
            )
        self.allocator.rollback(snapshot)
        return None

    def _resolve_mainline_closure(self, closure_mode: str) -> Optional[BranchClosureResult]:
        operator = MAINLINE_CLOSURE_OPERATOR_SPECS.get(str(closure_mode), ClosureOperatorSpec(name=str(closure_mode), scope="mainline"))
        if operator.name == "F1":
            return self._claim_f1()
        if operator.name == "F2":
            return self._claim_f2()
        return None

    def _render_product(
        self,
        labels: Sequence[str],
        node_ids: Sequence[Optional[int]],
        branch_map: Dict[int, List[str]],
        branch_id_map: Dict[int, List[str]],
        *,
        open_left: bool = False,
        open_right: bool = False,
        suppressed_support_node_ids: Optional[Sequence[int]] = None,
    ) -> TopologyPathRender:
        return _render_linear_path(
            labels=labels,
            node_ids=node_ids,
            open_left=bool(open_left),
            open_right=bool(open_right),
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
        )

    def _closure_sequence(
        self,
        closure: BranchClosureResult,
        orientation: str,
    ) -> Tuple[List[str], List[int]]:
        labels = list(closure.main_labels or str(closure.branch_topology).split("-"))
        node_ids = list(closure.main_node_ids or [int(x) for x in str(closure.branch_node_ids).split("-") if str(x)])
        if str(orientation) == "left":
            return labels, [int(x) for x in node_ids]
        return list(reversed(labels)), [int(x) for x in reversed(node_ids)]

    def _invalid_product_reason(
        self,
        product_kind: str,
        labels: Sequence[str],
        source_views: Sequence[AssemblyFragmentView],
        node_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> Optional[str]:
        return _validate_chain_product_shape(
            str(product_kind),
            labels,
            source_views,
            node_ids=node_ids,
        )

    def _merge_pool_ids(self, chunks: Iterable[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        merged: Dict[str, List[int]] = {}
        for chunk in list(chunks or []):
            for key, values in dict(chunk or {}).items():
                merged.setdefault(str(key), [])
                merged[str(key)].extend(int(x) for x in list(values or []))
        for key in list(merged.keys()):
            merged[key] = sorted(list(merged[key]))
        return merged

    def _make_product(
        self,
        *,
        product_kind: str,
        reason: str,
        source_views: Sequence[AssemblyFragmentView],
        labels: Sequence[str],
        node_ids: Sequence[Optional[int]],
        branch_map: Dict[int, List[str]],
        branch_id_map: Dict[int, List[str]],
        suppressed_support_node_ids: Optional[Sequence[int]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[TopologyAssemblyProduct]:
        render = self._render_product(
            labels,
            node_ids,
            branch_map,
            branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
        )
        source_fragment_ids, source_fragment_classes, source_node_ids = _collect_source_metadata(source_views)
        materialized_node_ids = {
            int(value)
            for value in list(node_ids or [])
            if value is not None and int(value) >= 0
        }
        for values in dict(branch_id_map or {}).values():
            for token in list(values or []):
                for raw_value in str(token).replace("(", "-").replace(")", "-").split("-"):
                    if str(raw_value).isdigit():
                        materialized_node_ids.add(int(raw_value))
        source_node_ids = sorted(set(int(value) for value in source_node_ids) | materialized_node_ids)
        diagnostics_payload = dict(diagnostics or {})
        product_type = str(
            RING_PRODUCT_KIND_TO_TYPE.get(
                str(product_kind),
                CHAIN_PRODUCT_KIND_TO_TYPE.get(str(product_kind), product_kind),
            )
        )
        topology_notation = str(render.topology_notation)
        if str(product_kind).startswith("fatty_ring_"):
            topology_notation = str(
                diagnostics_payload.get("topology_notation")
                or diagnostics_payload.get("canonical_topology_notation")
                or render.topology_notation
            )
        elif "canonical_labels" in diagnostics_payload:
            canonical_labels = list(diagnostics_payload.get("canonical_labels", list(labels or [])) or [])
            canonical_render = self._render_product(
                canonical_labels,
                [None for _ in canonical_labels],
                {},
                {},
                suppressed_support_node_ids=suppressed_support_node_ids,
            )
            topology_notation = str(canonical_render.topology_notation)
        actual_topology_su = str(diagnostics_payload.get("actual_topology_su") or render.topology_notation)
        actual_topology_gid = str(diagnostics_payload.get("actual_topology_gid") or render.node_id_notation)
        ring_validation_reason = _validate_ring_product_shape(
            str(product_kind),
            labels,
            topology_notation,
            actual_topology_su,
            diagnostics_payload,
        )
        if ring_validation_reason is not None:
            self._record_invalid_topology_attempt(
                product_kind=str(product_kind),
                actual_topology_su=actual_topology_su,
                actual_topology_gid=actual_topology_gid,
                topology_notation=topology_notation,
                validation_reason=str(ring_validation_reason),
                diagnostics=diagnostics_payload,
            )
            return None
        return TopologyAssemblyProduct(
            product_kind=str(product_kind),
            product_type=product_type,
            actual_topology_su=actual_topology_su,
            actual_topology_gid=actual_topology_gid,
            source_fragment_ids=source_fragment_ids,
            source_fragment_classes=source_fragment_classes,
            source_node_ids=source_node_ids,
            materialized_node_ids=sorted(materialized_node_ids),
            closed=True,
            reason=str(reason),
        )

    def _record_invalid_topology_attempt(
        self,
        *,
        product_kind: str,
        actual_topology_su: str,
        actual_topology_gid: str,
        topology_notation: str,
        validation_reason: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = (str(product_kind), str(actual_topology_su), str(validation_reason))
        if key in self.invalid_topology_keys:
            return
        self.invalid_topology_keys.add(key)
        diag = dict(diagnostics or {})
        entry = {
            "product_kind": str(product_kind),
            "actual_topology_su": str(actual_topology_su or ""),
            "actual_topology_gid": str(actual_topology_gid or ""),
            "topology_notation": str(topology_notation or ""),
            "reason": str(validation_reason or ""),
            "ring_template_kind": str(diag.get("ring_template_kind", "")),
            "ring_like24_count": _safe_int(diag.get("ring_like24_count"), -1),
            "ring_backbone_labels": _normalize_ring_backbone_labels(diag.get("ring_backbone_labels", [])),
            "ring_backbone_node_ids": _normalize_ring_backbone_node_ids(diag.get("ring_backbone_node_ids", [])),
        }
        outer_labels = _normalize_ring_backbone_labels(diag.get("ring_outer_backbone_labels", []))
        inner_labels = _normalize_ring_backbone_labels(diag.get("ring_inner_backbone_labels", []))
        if outer_labels:
            entry["ring_outer_backbone_labels"] = outer_labels
        if inner_labels:
            entry["ring_inner_backbone_labels"] = inner_labels
        self.invalid_topologies.append(entry)

    def _collect_suppressed_support_node_ids(
        self,
        source_views: Sequence[AssemblyFragmentView],
        branch_results: Optional[Sequence[BranchClosureResult]] = None,
        extra_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        suppressed: Set[int] = set()
        for view in list(source_views or []):
            suppressed.update(int(x) for x in list(getattr(view, "suppressed_support_node_ids", []) or []))
        for branch in list(branch_results or []):
            suppressed.update(int(x) for x in list(getattr(branch, "suppressed_support_node_ids", []) or []))
        suppressed.update(int(x) for x in list(extra_ids or []))
        return sorted(suppressed)

    def _claim_endcap(self, su_type: int, label: str, mode: str) -> Optional[BranchClosureResult]:
        claimed = self.allocator.claim(int(su_type), 1)
        if claimed is None:
            return None
        return BranchClosureResult(
            branch_topology=str(label),
            branch_node_ids=str(claimed[0]),
            branch_kind=str(label),
            mode=str(mode),
            pool_node_ids_by_su={str(int(su_type)): list(claimed)},
            main_labels=[str(label)],
            main_node_ids=[int(claimed[0])],
        )

    def _open_bridge_token(self, view: AssemblyFragmentView, side: str) -> Optional[Tuple[str, int]]:
        endpoint = view.endpoint(str(side))
        if not endpoint.is_open or endpoint.bridge_node_id is None:
            return None
        return str(endpoint.bridge_label or endpoint.label), int(endpoint.bridge_node_id)

    def _open_bridge_token_oriented(
        self,
        view: AssemblyFragmentView,
        side: str,
        *,
        reversed_view: bool = False,
    ) -> Optional[Tuple[str, int]]:
        if not bool(reversed_view):
            return self._open_bridge_token(view, side)
        actual_side = "right" if str(side) == "left" else "left"
        return self._open_bridge_token(view, actual_side)

    def _build_main_closure(
        self,
        view: AssemblyFragmentView,
        *,
        side: str,
        closure_mode: str,
    ) -> Tuple[Optional[Tuple[List[str], List[int]]], Dict[str, int]]:
        bridge = self._open_bridge_token(view, side)
        if bridge is not None:
            bridge_label, bridge_id = bridge
            if str(bridge_label) in {
                "23",
                "single19d2",
                "single20d2",
                "double19d2",
                "double20d2",
                "21d2",
            }:
                endcap = (
                    self._claim_endcap(11, "11", "pool:11")
                    if str(closure_mode) == "F1"
                    else self._claim_endcap(22, "22", "pool:22")
                )
                if endcap is None:
                    return None, {"11": 1} if str(closure_mode) == "F1" else {"22": 1}
                bridge_already_materialized = int(bridge_id) in {
                    int(node_id)
                    for node_id in list(view.main_node_ids or [])
                }
                if str(side) == "left":
                    labels = list(endcap.main_labels or []) + [str(bridge_label)]
                    node_ids = list(endcap.main_node_ids or []) + [int(bridge_id)]
                else:
                    labels = [str(bridge_label)] + list(endcap.main_labels or [])
                    node_ids = [int(bridge_id)] + list(endcap.main_node_ids or [])
                if bridge_already_materialized:
                    labels = list(endcap.main_labels or [])
                    node_ids = list(endcap.main_node_ids or [])
                return (labels, node_ids), {}
            closure = self._resolve_mainline_closure(str(closure_mode))
            if closure is None:
                if str(closure_mode) == "F1":
                    return None, {"23": 1, "11": 1}
                return None, {"23": 1, "22": 1}
            labels, node_ids = self._closure_sequence(closure, side)
            bridge_already_materialized = int(bridge_id) in {
                int(node_id)
                for node_id in list(view.main_node_ids or [])
                if node_id is not None
            }
            if _should_materialize_bridge_literal(view, str(bridge_label)) and not bridge_already_materialized:
                if str(side) == "left":
                    labels = list(labels) + [str(bridge_label)]
                    node_ids = list(node_ids) + [int(bridge_id)]
                else:
                    labels = [str(bridge_label)] + list(labels)
                    node_ids = [int(bridge_id)] + list(node_ids)
            return (labels, node_ids), {}
        closure = self._resolve_mainline_closure(str(closure_mode))
        if closure is None:
            if str(closure_mode) == "F1":
                return None, {"23": 1, "11": 1}
            return None, {"23": 1, "22": 1}
        labels, node_ids = self._closure_sequence(closure, side)
        return (labels, node_ids), {}

    def _build_branch_maps(
        self,
        view: AssemblyFragmentView,
        branch_results: Optional[Sequence[BranchClosureResult]],
        *,
        reverse: bool = False,
        anchor_offset: int = 0,
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        label_count = len(list(view.main_labels or []))
        for item in list(branch_results or []):
            anchor_index = int(getattr(item, "anchor_index", int(view.branch_anchor_index)))
            if reverse:
                anchor_index = label_count - 1 - int(anchor_index)
            anchor_index += int(anchor_offset)
            branch_map.setdefault(int(anchor_index), [])
            branch_map[int(anchor_index)].append(str(item.branch_topology))
            branch_id_map.setdefault(int(anchor_index), [])
            branch_id_map[int(anchor_index)].append(str(item.branch_node_ids))
        return branch_map, branch_id_map

    def _collect_source_views_from_branches(
        self,
        base_views: Sequence[AssemblyFragmentView],
        branch_results: Optional[Sequence[BranchClosureResult]],
    ) -> List[AssemblyFragmentView]:
        return list(base_views or []) + [
            self.view_by_id[x]
            for item in list(branch_results or [])
            for x in list(item.source_fragment_ids or [])
            if x in self.view_by_id
        ]

    def _collect_claimed_fragment_ids(
        self,
        base_views: Sequence[AssemblyFragmentView],
        branch_results: Optional[Sequence[BranchClosureResult]],
    ) -> List[int]:
        return [int(view.fragment_id) for view in list(base_views or [])] + [
            int(x)
            for item in list(branch_results or [])
            for x in list(item.source_fragment_ids or [])
        ]

    def _record_invalid_materialization_failure(
        self,
        *,
        snapshot: Any,
        donor_snapshot: Sequence[AssemblyFragmentView],
        branch_donors: List[AssemblyFragmentView],
        product_kind: str,
        source_views: Sequence[AssemblyFragmentView],
        labels: Sequence[str],
        node_ids: Sequence[Optional[int]],
        branch_map: Dict[int, List[str]],
        branch_id_map: Dict[int, List[str]],
        suppressed_support_node_ids: Optional[Sequence[int]],
        validation_reason: str,
    ) -> bool:
        rolled_back_resource_ids = self.allocator.claimed_ids_since(snapshot)
        attempted_render = self._render_product(
            labels,
            node_ids,
            branch_map,
            branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
        )
        self.allocator.rollback(snapshot)
        branch_donors[:] = list(donor_snapshot or [])
        self.composition_attempt_failures.append({
            "attempted_product_kind": str(product_kind),
            "source_fragment_ids": sorted({
                int(view.fragment_id) for view in list(source_views or [])
            }),
            "left_endpoint": {},
            "right_endpoint": {},
            "validation_reason": str(validation_reason),
            "attempted_resource_counts": {
                str(key): int(len(list(values or [])))
                for key, values in dict(rolled_back_resource_ids or {}).items()
            },
            "rolled_back_resource_ids": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(rolled_back_resource_ids or {}).items()
            },
        })
        return False

    def _validate_external_branch_closure_state(
        self,
        diagnostics: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        diag = dict(diagnostics or {})
        blockers = {
            str(anchor): [str(value) for value in list(values or []) if str(value)]
            for anchor, values in dict(diag.get("branch_contract_blockers", {}) or {}).items()
            if list(values or [])
        }
        if not blockers and int(diag.get("unresolved_branch_contract_count", 0) or 0) <= 0:
            return None
        for values in blockers.values():
            if list(values or []):
                return str(values[0])
        return "external_branch_closure_pending"

    def _branch_pending_reason(
        self,
        diagnostics: Optional[Dict[str, Any]],
        *,
        default: str,
    ) -> str:
        return str(self._validate_external_branch_closure_state(diagnostics) or str(default))

    def _is_external_branch_closure_reason(
        self,
        reason: str,
    ) -> bool:
        return str(reason) in {
            "external_branch_closure_pending",
            "fixed_branch_H1_missing",
            "fixed_branch_partner_missing",
            "fixed_branch_role_conflict",
        }

    def _has_pending_contracts_for_subphase(
        self,
        subphase: str,
    ) -> bool:
        subphase_name = str(subphase or "")
        return any(
            str(dict(item or {}).get("subphase", "") or "") == str(subphase_name)
            for item in list(self.pending_contracts or [])
        )

    def _external_subphase_status_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "explicit_branch_closure": {
                "ok": not self._has_pending_contracts_for_subphase("explicit_branch_closure"),
            },
            "real_like24_ring_assembly": {
                "ok": not self._has_pending_contracts_for_subphase("real_like24_ring_assembly"),
            },
            "remaining_external_halfclosed_closure": {
                "ok": not self._has_pending_contracts_for_subphase("remaining_external_halfclosed_closure"),
            },
        }

    def _contracts_by_subphase(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in list(self.pending_contracts or []):
            contract = dict(item or {})
            grouped[str(contract.get("subphase", "external_structural_closure") or "external_structural_closure")].append(contract)
        return {str(key): list(value or []) for key, value in grouped.items()}

    def _view_flex_reconcile_roles(
        self,
        view: Optional[AssemblyFragmentView],
    ) -> Set[str]:
        if view is None:
            return set()
        return {
            str(role)
            for role in list(getattr(view, "flex_reconcile_roles", []) or [])
            if str(role)
        }

    def _view_has_flex_reconcile_role(
        self,
        view: Optional[AssemblyFragmentView],
        *roles: str,
    ) -> bool:
        available = self._view_flex_reconcile_roles(view)
        return any(str(role) in available for role in list(roles or []))

    def _view_flex_reconcile_priority(
        self,
        view: Optional[AssemblyFragmentView],
    ) -> int:
        if view is None:
            return 99
        return _safe_int(getattr(view, "flex_reconcile_priority", 99), 99)

    def _pending_kind_for_view(
        self,
        view: Optional[AssemblyFragmentView],
    ) -> str:
        if view is None:
            return "other"
        profile = self._view_external_pending_profile(view)
        pending_kind = str(profile.get("external_pending_kind", "") or "")
        if pending_kind:
            return str(pending_kind)
        if self._view_has_flex_reconcile_role(view, "d_like_open"):
            return "d_like_open"
        if self._view_has_flex_reconcile_role(view, "b_like_open"):
            return "b_like_open"
        if self._is_side_pending_view(view):
            return "explicit_side" if str(getattr(view, "preferred_product_type", "")) == "side_chain" else "e_like_open"
        preferred = str(getattr(view, "preferred_product_type", "") or "")
        if preferred == "flex_chain":
            return "explicit_flex"
        if preferred == "side_chain":
            return "explicit_side"
        if preferred == "ring_candidate":
            return "ring_only_like24"
        return "other"

    def _is_side_pending_view(self, view: Optional[AssemblyFragmentView]) -> bool:
        if view is None:
            return False
        return bool(
            str(getattr(view, "class_type", "")) in E_LIKE_OPEN_CLASSES
            or str(getattr(view, "preferred_product_type", "")) == "side_chain"
        )

    def _is_explicit_mainline_view(self, view: Optional[AssemblyFragmentView]) -> bool:
        if view is None:
            return False
        pending_kind = self._pending_kind_for_view(view)
        return str(pending_kind) in {"explicit_flex", "explicit_side"}

    def _available_remaining_side_views(
        self,
        *,
        require_generic_join: bool,
    ) -> List[AssemblyFragmentView]:
        return self._available_views(
            lambda item: self._is_side_pending_view(item) and (not require_generic_join or _is_generic_chain_join_view(item))
        )

    def _available_remaining_b_open_views(
        self,
        *,
        require_generic_join: bool,
    ) -> List[AssemblyFragmentView]:
        return self._available_views(
            lambda item: self._view_has_flex_reconcile_role(item, "b_like_open") and (not require_generic_join or _is_generic_chain_join_view(item))
        )

    def _available_branch_donors(self) -> List[AssemblyFragmentView]:
        return self._available_views(lambda item: self._is_side_pending_view(item))

    def _view_masked_ring_open_sides(self, view: AssemblyFragmentView) -> Set[str]:
        return {
            str(item)
            for item in list(dict(view.evidence or {}).get("masked_ring_open_sides", []) or [])
        }

    def _view_is_structurally_explicit(self, view: AssemblyFragmentView, *, max_suppressed: int) -> bool:
        suppressed_count = int(len(list(getattr(view, "suppressed_support_node_ids", []) or [])))
        masked_open_count = int(len(self._view_masked_ring_open_sides(view)))
        effective_suppressed = max(0, int(suppressed_count - masked_open_count))
        return int(effective_suppressed) <= int(max_suppressed)

    def _view_is_ring_core_explicit(self, view: AssemblyFragmentView) -> bool:
        if self._view_has_slot_profile(view, "fused_pair_inner_segment", "upper_lower_pair_core_segment", "side_pair_core_segment"):
            return True
        bridgehead_role = self._view_bridgehead_role(view)
        if bridgehead_role in {"fused_bridgehead_substitute_14", "fused_bridgehead_substitute_pair"}:
            return True
        return self._view_is_structurally_explicit(view, max_suppressed=0)

    def _view_required_branch_kinds_for_skeleton(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
    ) -> Dict[int, List[str]]:
        if view is None:
            return {}
        raw = dict(
            dict(dict(view.evidence or {}).get("required_branch_closures_by_anchor_in_ring", {}) or {}).get(str(skeleton_kind), {}) or {}
        )
        if not raw:
            return self._view_required_branch_kinds(view)
        out: Dict[int, List[str]] = {}
        for anchor, values in raw.items():
            if str(anchor) == "center":
                anchor_idx = int(view.branch_anchor_index)
            else:
                anchor_idx = _safe_int(anchor, int(view.branch_anchor_index))
            out[int(anchor_idx)] = [str(value) for value in list(values or [])]
        return out

    def _view_ring_contract(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
    ) -> Dict[str, Any]:
        contracts = dict(dict(view.evidence or {}).get("ring_contracts", {}) or {})
        return dict(contracts.get(str(skeleton_kind), {}) or {})

    def _view_has_ring_contract_role(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
        *roles: str,
    ) -> bool:
        contract = self._view_ring_contract(view, str(skeleton_kind))
        slot_roles = {str(item) for item in list(contract.get("slot_roles", []) or [])}
        return any(str(role) in slot_roles for role in list(roles or []))

    def _view_ring_contract_priority(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
    ) -> int:
        contract = self._view_ring_contract(view, str(skeleton_kind))
        return _safe_int(contract.get("priority"), 99)

    def _view_ring_path_capabilities(
        self,
        view: Optional[AssemblyFragmentView],
        skeleton_kind: str,
    ) -> List[str]:
        if view is None:
            return []
        raw = dict(dict(view.evidence or {}).get("ring_path_capabilities", {}) or {})
        return [str(item) for item in list(raw.get(str(skeleton_kind), []) or []) if str(item)]

    def _view_ring_backbone_labels(
        self,
        view: Optional[AssemblyFragmentView],
        *,
        skeleton_kind: str,
        reverse: bool = False,
    ) -> List[str]:
        if view is None:
            return []
        labels = list(dict(getattr(view, "ring_backbone_labels", {}) or {}).get(str(skeleton_kind), []) or [])
        if not labels:
            return []
        normalized = [str(value) for value in labels if str(value)]
        return list(reversed(normalized)) if reverse else normalized

    def _view_ring_backbone_node_ids(
        self,
        view: Optional[AssemblyFragmentView],
        *,
        skeleton_kind: str,
        reverse: bool = False,
    ) -> List[Optional[int]]:
        if view is None:
            return []
        node_ids = _normalize_ring_backbone_node_ids(
            dict(getattr(view, "ring_backbone_node_ids", {}) or {}).get(str(skeleton_kind), [])
        )
        return list(reversed(node_ids)) if reverse else list(node_ids)

    def _view_ring_backbone_span(
        self,
        view: Optional[AssemblyFragmentView],
        *,
        skeleton_kind: str,
    ) -> int:
        if view is None:
            return 0
        span = _safe_int(dict(getattr(view, "ring_backbone_span", {}) or {}).get(str(skeleton_kind)), -1)
        if span >= 0:
            return int(span)
        return int(len(self._view_ring_backbone_labels(view, skeleton_kind=str(skeleton_kind), reverse=False)))

    def _claim_ring_pool_segment(
        self,
        su_type: int,
        *,
        mode: str,
        node_id: Optional[int] = None,
        skeleton_kind: str = "",
        role: str = "pool",
        support_required_counts: Optional[Dict[int, int]] = None,
    ) -> Optional[ViewSegment]:
        snapshot = self.allocator.snapshot()
        claimed_support_ids: Dict[int, List[int]] = {}
        for support_su, support_count in dict(support_required_counts or {}).items():
            claimed_support = self.allocator.claim(int(support_su), int(support_count))
            if claimed_support is None:
                self.allocator.rollback(snapshot)
                return None
            claimed_support_ids[int(support_su)] = [int(node_id) for node_id in list(claimed_support or [])]
        claimed = (
            self.allocator.claim_specific(int(su_type), int(node_id))
            if node_id is not None
            else self.allocator.claim(int(su_type), 1)
        )
        if claimed is None:
            self.allocator.rollback(snapshot)
            return None
        label = str(int(su_type))
        labels = [label]
        node_ids = [int(claimed[0])]
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        materialized_support_ids: Set[int] = set()
        if int(su_type) == 24:
            cap_ids = list(claimed_support_ids.get(11, []) or [])
            if str(role) in {"left_outer_anchor", "right_outer_anchor"} and cap_ids:
                if str(role) == "left_outer_anchor":
                    labels = ["11", "24"]
                    node_ids = [int(cap_ids[0]), int(claimed[0])]
                    branch_anchor = 1
                else:
                    labels = ["24", "11"]
                    node_ids = [int(claimed[0]), int(cap_ids[0])]
                    branch_anchor = 0
                materialized_support_ids.add(int(cap_ids[0]))
            else:
                branch_anchor = 0
            h23_ids = list(claimed_support_ids.get(23, []) or [])
            h22_ids = list(claimed_support_ids.get(22, []) or [])
            if h23_ids and h22_ids and str(role) not in {"left_bridgehead", "right_bridgehead"}:
                branch_map[int(branch_anchor)] = ["-23-22"]
                branch_id_map[int(branch_anchor)] = [f"-{int(h23_ids[0])}-{int(h22_ids[0])}"]
                materialized_support_ids.update({int(h23_ids[0]), int(h22_ids[0])})
        pool_node_ids_by_su = {str(int(su_type)): [int(claimed[0])]}
        for support_su, support_ids in dict(claimed_support_ids or {}).items():
            pool_node_ids_by_su.setdefault(str(int(support_su)), [])
            pool_node_ids_by_su[str(int(support_su))].extend(int(node_id) for node_id in list(support_ids or []))
        return ViewSegment(
            view=None,
            labels=labels,
            node_ids=node_ids,
            ring_backbone_labels=labels,
            ring_backbone_node_ids=node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            branch_modes=[str(mode)],
            pool_node_ids_by_su=pool_node_ids_by_su,
            source_views=[],
            suppressed_support_node_ids=sorted({
                int(node_id)
                for values in dict(claimed_support_ids or {}).values()
                for node_id in list(values or [])
                if int(node_id) not in materialized_support_ids
            }),
            ring_template_kind="pool",
            ring_template_roles=["pool"],
            diagnostics={
                "branch_render": _render_linear_path(
                    labels=labels,
                    node_ids=node_ids,
                    branch_map=branch_map,
                    branch_id_map=branch_id_map,
                ).topology_notation,
                "branch_render_gid": _render_linear_path(
                    labels=labels,
                    node_ids=node_ids,
                    branch_map=branch_map,
                    branch_id_map=branch_id_map,
                ).node_id_notation,
                "raw_su24_role": str(role),
                "skeleton_kind": str(skeleton_kind),
                "support_required_counts": {
                    str(int(key)): int(value)
                    for key, value in dict(support_required_counts or {}).items()
                    if int(value) > 0
                },
            },
        )

    def _view_bridgehead_role(self, view: AssemblyFragmentView) -> str:
        return str(dict(view.evidence or {}).get("bridgehead_role", ""))

    def _view_ring_like24_units(self, view: Optional[AssemblyFragmentView]) -> int:
        if view is None:
            return 0
        return max(0, _safe_int(getattr(view, "ring_like24_units", 0), 0))

    def _view_fixed_branch_contracts_for_skeleton(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        contract_map = {
            int(anchor): [dict(item or {}) for item in list(values or []) if dict(item or {})]
            for anchor, values in dict(getattr(view, "branch_closure_contracts_by_anchor", {}) or {}).items()
        }
        if str(skeleton_kind) and str(view.class_type) in {"24_AB_N", "24_CD_N"}:
            primary_id = _safe_int(getattr(view, "resolved_fixed_mainline_connector_id", None), -1)
            primary_contract = next(
                (
                    dict(item or {})
                    for item in list(getattr(view, "fixed_connection_contracts", []) or [])
                    if _safe_int(dict(item or {}).get("connector_node_id"), -1) == int(primary_id)
                ),
                {},
            )
            role_candidates = {
                str(value)
                for value in list(primary_contract.get("role_candidates", []) or [])
                if str(value)
            }
            backbone_node_ids = {
                int(node_id)
                for node_id in self._view_ring_backbone_node_ids(
                    view,
                    skeleton_kind=str(skeleton_kind),
                    reverse=False,
                )
                if node_id is not None
            }
            if primary_contract and "mainline_or_branch" in role_candidates:
                existing_ids = {
                    _safe_int(dict(item or {}).get("connector_node_id"), -1)
                    for item in list(contract_map.get(int(view.branch_anchor_index), []) or [])
                }
                if primary_id >= 0 and primary_id not in existing_ids and primary_id not in backbone_node_ids:
                    contract_map.setdefault(int(view.branch_anchor_index), [])
                    contract_map[int(view.branch_anchor_index)].append(dict(primary_contract or {}))
        return contract_map

    def _view_ring_branch_anchor_index(
        self,
        view: AssemblyFragmentView,
        skeleton_kind: str,
    ) -> int:
        center_ids = [
            int(node_id)
            for node_id in list(dict(view.evidence or {}).get("center_node_ids", []) or [])
        ]
        backbone_ids = self._view_ring_backbone_node_ids(view, skeleton_kind=str(skeleton_kind), reverse=False)
        for center_id in center_ids:
            if int(center_id) in [int(node_id) for node_id in list(backbone_ids or []) if node_id is not None]:
                return list(backbone_ids).index(int(center_id))
        return min(int(view.branch_anchor_index), max(0, len(backbone_ids) - 1))

    def _view_required_branch_kinds(self, view: AssemblyFragmentView) -> Dict[int, List[str]]:
        if view is None:
            return {}
        out: Dict[int, List[str]] = {}
        for anchor, values in dict(dict(view.evidence or {}).get("required_branch_kinds_by_anchor", {}) or {}).items():
            if str(anchor) == "center":
                anchor_idx = int(view.branch_anchor_index)
            else:
                anchor_idx = _safe_int(anchor, int(view.branch_anchor_index))
            out[int(anchor_idx)] = [str(value) for value in list(values or [])]
        return out

    def _render_resolved_view(
        self,
        view: AssemblyFragmentView,
        branch_resolved: Sequence[BranchClosureResult],
        *,
        reverse: bool = False,
        labels_override: Optional[Sequence[str]] = None,
        node_ids_override: Optional[Sequence[Optional[int]]] = None,
        center_anchor_override: Optional[int] = None,
    ) -> TopologyPathRender:
        labels = (
            [str(value) for value in list(labels_override or [])]
            if labels_override is not None
            else list(view.main_labels or [])
        )
        node_ids = (
            [None if value is None else int(value) for value in list(node_ids_override or [])]
            if node_ids_override is not None
            else list(view.main_node_ids or [])
        )
        if reverse:
            labels = list(reversed(labels))
            node_ids = list(reversed(node_ids))
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        for item in list(branch_resolved or []):
            anchor_index = int(getattr(item, "anchor_index", int(view.branch_anchor_index)))
            if center_anchor_override is not None and int(anchor_index) == int(view.branch_anchor_index):
                anchor_index = int(center_anchor_override)
            if reverse:
                anchor_index = len(labels) - 1 - int(anchor_index)
            branch_map.setdefault(int(anchor_index), [])
            branch_map[int(anchor_index)].append(str(item.branch_topology))
            branch_id_map.setdefault(int(anchor_index), [])
            branch_id_map[int(anchor_index)].append(str(item.branch_node_ids))
        suppressed_support_node_ids = self._collect_suppressed_support_node_ids([view], branch_resolved)
        return self._render_product(
            labels,
            node_ids,
            branch_map,
            branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
        )

    def _view_is_upper_fixed_anchor(self, view: AssemblyFragmentView) -> bool:
        return bool(
            (
                self._view_has_ring_contract_role(view, "upper_lower_ring_plan", "upper_lower_fixed_anchor")
                or (
                    str(view.class_type) == "24_A"
                    and bool(dict(view.evidence or {}).get("fixed_anchor_eligible", False))
                    and self._view_has_slot_profile(view, "upper_lower_fixed_anchor")
                )
            )
        )

    def _side_span_priority(
        self,
        view: AssemblyFragmentView,
    ) -> Tuple[int, int, int]:
        class_type = str(getattr(view, "class_type", ""))
        priority_by_class = {
            "24_AB_N_24_CD": 0,
            "24_AB_N_24_AB": 1,
            "24_A": 2,
            "24_B": 3,
            "24_CD_N_24_CD": 4,
            "24_CD_N": 5,
            "24_CD_M": 6,
            "24_C": 7,
            "24_D": 8,
        }
        return (
            int(priority_by_class.get(class_type, 50)),
            self._view_ring_contract_priority(view, "side_ring_plan"),
            int(view.fragment_id),
        )

    def _upper_path_priority(
        self,
        view: AssemblyFragmentView,
    ) -> Tuple[int, int, int]:
        class_type = str(getattr(view, "class_type", ""))
        priority_by_class = {
            "24_CD_N_24_CD": 0,
            "24_CD_N": 1,
            "24_D": 2,
            "24_C": 3,
            "24_A": 4,
            "24_B": 5,
        }
        return (
            int(priority_by_class.get(class_type, 50)),
            self._view_ring_contract_priority(view, "upper_lower_ring_plan"),
            int(view.fragment_id),
        )

    def _build_ring_template_segment(
        self,
        *,
        skeleton_kind: str,
        role: str,
        candidate_id: str,
        view: Optional[AssemblyFragmentView] = None,
        reverse: bool = False,
        pool_su: Optional[int] = None,
        priority: Tuple[Any, ...] = (),
    ) -> Optional[Dict[str, Any]]:
        if view is not None:
            backbone_labels = self._view_ring_backbone_labels(
                view,
                skeleton_kind=str(skeleton_kind),
                reverse=bool(reverse),
            )
            backbone_node_ids = self._view_ring_backbone_node_ids(
                view,
                skeleton_kind=str(skeleton_kind),
                reverse=bool(reverse),
            )
            if not backbone_labels:
                return None
            return {
                "kind": "view",
                "role": str(role),
                "candidate_id": str(candidate_id),
                "fragment_id": int(view.fragment_id),
                "view": view,
                "reverse": bool(reverse),
                "pool_su": None,
                "backbone_span": int(len(backbone_labels)),
                "backbone_labels": list(backbone_labels or []),
                "backbone_node_ids": list(backbone_node_ids or []),
                "like24_units": self._view_ring_like24_units(view),
                "priority": tuple(priority),
            }
        if pool_su is None:
            return None
        label = str(int(pool_su))
        return {
            "kind": "pool",
            "role": str(role),
            "candidate_id": str(candidate_id),
            "fragment_id": None,
            "view": None,
            "reverse": False,
            "pool_su": int(pool_su),
            "pool_node_id": None,
            "backbone_span": 1,
            "backbone_labels": [label],
            "backbone_node_ids": [None],
            "like24_units": 1 if int(pool_su) == 24 else 0,
            "support_required_counts": {
                11: 1
            } if int(pool_su) == 24 and str(skeleton_kind) == "upper_lower" and str(role) == "path" else {},
            "priority": tuple(priority),
        }

    def _build_pool_ring_template_segments(
        self,
        *,
        skeleton_kind: str,
        role: str,
        allowed_su: Sequence[int],
        priority_by_su: Optional[Dict[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        counts = self.allocator.available_counts()
        for su_type in list(allowed_su or []):
            if int(su_type) == 24:
                from .stage3_state_machine import build_su24_resource_role_profiles

                remaining_ids = [
                    int(value)
                    for value in list(self.allocator.remaining_ids_by_su.get("24", []) or [])
                ]
                profiles = build_su24_resource_role_profiles(
                    self.nodes,
                    {24: remaining_ids},
                )
                compatible_profiles = [
                    profile
                    for profile in profiles
                    if f"{str(skeleton_kind)}:{str(role)}" in set(profile.allowed_ring_slots or [])
                ]
                role_counts = Counter(
                    str(role_name)
                    for profile in compatible_profiles
                    for role_name in list(profile.available_roles or [])
                )
                for profile in compatible_profiles:
                    preferred_role = min(
                        list(profile.available_roles or ["D"]),
                        key=lambda role_name: (
                            -int(role_counts.get(str(role_name), 0)),
                            {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(role_name), 9),
                        ),
                    )
                    candidate = self._build_ring_template_segment(
                        skeleton_kind=str(skeleton_kind),
                        role=str(role),
                        candidate_id=f"pool:{str(skeleton_kind)}:{str(role)}:24:{int(profile.node_id)}",
                        pool_su=24,
                        priority=(
                            int(dict(priority_by_su or {}).get(24, 99)),
                            -int(role_counts.get(str(preferred_role), 0)),
                            {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(preferred_role), 9),
                            int(profile.node_id),
                        ),
                    )
                    if candidate is not None:
                        candidate["pool_node_id"] = int(profile.node_id)
                        candidate["raw_su24_role"] = str(preferred_role)
                        if str(role) in {"left_outer_anchor", "right_outer_anchor"}:
                            candidate["support_required_counts"] = {11: 1, 23: 1, 22: 1}
                            candidate["backbone_span"] = 2
                            candidate["backbone_labels"] = ["11", "24"] if str(role) == "left_outer_anchor" else ["24", "11"]
                            candidate["backbone_node_ids"] = [None, int(profile.node_id)] if str(role) == "left_outer_anchor" else [int(profile.node_id), None]
                        elif str(role) in {"left_bridgehead", "right_bridgehead"}:
                            candidate["support_required_counts"] = {}
                        else:
                            candidate["support_required_counts"] = {23: 1, 22: 1}
                        candidates.append(candidate)
                continue
            count = int(counts.get(str(int(su_type)), 0) or 0)
            for idx in range(count):
                candidate = self._build_ring_template_segment(
                    skeleton_kind=str(skeleton_kind),
                    role=str(role),
                    candidate_id=f"pool:{str(skeleton_kind)}:{str(role)}:{int(su_type)}:{int(idx)}",
                    pool_su=int(su_type),
                    priority=(
                        int(dict(priority_by_su or {}).get(int(su_type), 99)),
                        int(su_type),
                        int(idx),
                    ),
                )
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _resolve_planned_ring_segment(
        self,
        segment_spec: Dict[str, Any],
        branch_donors: List[AssemblyFragmentView],
        *,
        skeleton_kind: str,
    ) -> Optional[ViewSegment]:
        if str(segment_spec.get("kind", "")) == "pool":
            return self._claim_ring_pool_segment(
                int(segment_spec.get("pool_su", -1)),
                mode=f"ring:{str(skeleton_kind)}:{str(segment_spec.get('role', 'pool'))}",
                node_id=(
                    int(segment_spec.get("pool_node_id"))
                    if segment_spec.get("pool_node_id") is not None
                    else None
                ),
                skeleton_kind=str(skeleton_kind),
                role=str(segment_spec.get("role", "pool")),
                support_required_counts={
                    int(key): int(value)
                    for key, value in dict(segment_spec.get("support_required_counts", {}) or {}).items()
                    if int(value) > 0
                },
            )
        view = segment_spec.get("view")
        if view is None and segment_spec.get("fragment_id") is not None:
            view = self.view_by_id.get(int(segment_spec.get("fragment_id")))
        if view is None:
            return None
        segment, _, diag = self._resolve_view_segment(
            view,
            branch_donors,
            reverse=bool(segment_spec.get("reverse", False)),
            skeleton_kind=str(skeleton_kind),
        )
        if segment is None and not str(self.last_ring_materialization_reason):
            reason = str(self._validate_external_branch_closure_state(diag) or "")
            if not reason:
                reason = str(dict(diag or {}).get("external_branch_closure_reason", "") or "")
            self.last_ring_materialization_reason = str(reason or "external_branch_closure_pending")
        return segment

    def _planned_ring_source_views(
        self,
        ordered_segments: Sequence[Dict[str, Any]],
    ) -> List[AssemblyFragmentView]:
        source: Dict[int, AssemblyFragmentView] = {}
        for segment in list(ordered_segments or []):
            view = segment.get("view")
            if view is None and segment.get("fragment_id") is not None:
                view = self.view_by_id.get(int(segment.get("fragment_id")))
            if view is None:
                continue
            source[int(view.fragment_id)] = view
        return list(source.values())

    def _planned_ring_like24_count(
        self,
        ordered_segments: Sequence[Dict[str, Any]],
    ) -> int:
        total = 0
        seen_fragment_ids: Set[int] = set()
        for segment in list(ordered_segments or []):
            fragment_id = segment.get("fragment_id")
            if fragment_id is not None:
                fragment_id_i = int(fragment_id)
                if fragment_id_i in seen_fragment_ids:
                    continue
                seen_fragment_ids.add(fragment_id_i)
            total += max(0, int(segment.get("like24_units", 0) or 0))
        return int(total)

    def _build_side_ring_template_candidates(
        self,
        views: Sequence[AssemblyFragmentView],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        left_anchors: List[Dict[str, Any]] = []
        right_anchors: List[Dict[str, Any]] = []
        middle_segments: List[Dict[str, Any]] = []
        for view in list(views or []):
            capabilities = set(self._view_ring_path_capabilities(view, "side"))
            if not capabilities:
                continue
            reverse_modes = [False, True] if capabilities & {"terminal_single", "terminal_pair", "terminal_core_pair"} else [False]
            if capabilities & {"terminal_single", "terminal_pair", "terminal_core_pair"}:
                terminal_role = (
                    "terminal_core_pair"
                    if "terminal_core_pair" in capabilities
                    else ("terminal_pair" if "terminal_pair" in capabilities else "terminal_single")
                )
                terminal_rank = {"terminal_core_pair": 0, "terminal_pair": 1, "terminal_single": 2}.get(str(terminal_role), 9)
                for reverse in reverse_modes:
                    backbone_labels = self._view_ring_backbone_labels(
                        view,
                        skeleton_kind="side",
                        reverse=bool(reverse),
                    )
                    if not backbone_labels:
                        continue
                    if _is_aromatic_label(str(backbone_labels[0])):
                        candidate = self._build_ring_template_segment(
                            skeleton_kind="side",
                            role="left_anchor",
                            candidate_id=f"view:side:left:{int(view.fragment_id)}:{1 if reverse else 0}",
                            view=view,
                            reverse=bool(reverse),
                            priority=(int(terminal_rank), *self._side_span_priority(view), 0 if not reverse else 1),
                        )
                        if candidate is not None:
                            left_anchors.append(candidate)
                    if _is_aromatic_label(str(backbone_labels[-1])):
                        candidate = self._build_ring_template_segment(
                            skeleton_kind="side",
                            role="right_anchor",
                            candidate_id=f"view:side:right:{int(view.fragment_id)}:{1 if reverse else 0}",
                            view=view,
                            reverse=bool(reverse),
                            priority=(int(terminal_rank), *self._side_span_priority(view), 0 if not reverse else 1),
                        )
                        if candidate is not None:
                            right_anchors.append(candidate)
            if capabilities & {"inner_single", "inner_pair"}:
                inner_role = "inner_pair" if "inner_pair" in capabilities else "inner_single"
                inner_rank = 0 if inner_role == "inner_pair" else 1
                candidate = self._build_ring_template_segment(
                    skeleton_kind="side",
                    role="middle",
                    candidate_id=f"view:side:middle:{int(view.fragment_id)}",
                    view=view,
                    reverse=False,
                    priority=(int(inner_rank), *self._side_span_priority(view)),
                )
                if candidate is not None:
                    middle_segments.append(candidate)
        middle_segments.extend(
            self._build_pool_ring_template_segments(
                skeleton_kind="side",
                role="middle",
                allowed_su=[23, 24],
                priority_by_su={23: 20, 24: 21},
            )
        )
        return (
            sorted(left_anchors, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
            sorted(right_anchors, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
            sorted(middle_segments, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
        )

    def _build_upper_ring_template_candidates(
        self,
        views: Sequence[AssemblyFragmentView],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        anchors: List[Dict[str, Any]] = []
        path_segments: List[Dict[str, Any]] = []
        for view in list(views or []):
            capabilities = set(self._view_ring_path_capabilities(view, "upper_lower"))
            if self._view_is_upper_fixed_anchor(view):
                candidate = self._build_ring_template_segment(
                    skeleton_kind="upper_lower",
                    role="fixed_anchor",
                    candidate_id=f"view:upper:anchor:{int(view.fragment_id)}",
                    view=view,
                    reverse=False,
                    priority=(0, self._view_ring_contract_priority(view, "upper_lower_ring_plan"), int(view.fragment_id)),
                )
                if candidate is not None:
                    anchors.append(candidate)
            if capabilities & {"path_single", "path_pair"}:
                path_role = "path_pair" if "path_pair" in capabilities else "path_single"
                path_rank = 0 if path_role == "path_pair" else 1
                candidate = self._build_ring_template_segment(
                    skeleton_kind="upper_lower",
                    role="path",
                    candidate_id=f"view:upper:path:{int(view.fragment_id)}",
                    view=view,
                    reverse=False,
                    priority=(int(path_rank), *self._upper_path_priority(view)),
                )
                if candidate is not None:
                    path_segments.append(candidate)
        path_segments.extend(
            self._build_pool_ring_template_segments(
                skeleton_kind="upper_lower",
                role="path",
                allowed_su=[23, 24, 3],
                priority_by_su={23: 20, 24: 21, 3: 22},
            )
        )
        return (
            sorted(anchors, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
            sorted(path_segments, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
        )

    def _plan_side_ring_templates(
        self,
        preferred_views: Sequence[AssemblyFragmentView],
    ) -> RingSkeletonPlan:
        available = [
            view for view in list(preferred_views or [])
            if self._fragment_available(int(view.fragment_id))
        ]
        left_anchors, right_anchors, middle_segments = self._build_side_ring_template_candidates(available)
        missing_slots: List[str] = []
        ordered_segments: List[Dict[str, Any]] = []

        if not left_anchors or not right_anchors:
            missing_slots.append("side_terminal_missing")
        else:
            for left_anchor in list(left_anchors or []):
                left_fragment_id = left_anchor.get("fragment_id")
                for right_anchor in list(right_anchors or []):
                    right_fragment_id = right_anchor.get("fragment_id")
                    if left_fragment_id is not None and right_fragment_id is not None and int(left_fragment_id) == int(right_fragment_id):
                        continue
                    base_like24 = int(left_anchor.get("like24_units", 0) or 0) + int(right_anchor.get("like24_units", 0) or 0)
                    if base_like24 > 4:
                        continue
                    remaining_span = 6 - int(left_anchor.get("backbone_span", 0) or 0) - int(right_anchor.get("backbone_span", 0) or 0)
                    if remaining_span < 0:
                        continue
                    middle = self._search_ring_middle_segments(
                        middle_segments,
                        remaining_span=int(remaining_span),
                        claimed_fragment_ids={
                            int(fragment_id)
                            for fragment_id in [left_fragment_id, right_fragment_id]
                            if fragment_id is not None
                        },
                        used_candidate_ids={
                            str(left_anchor.get("candidate_id", "")),
                            str(right_anchor.get("candidate_id", "")),
                        },
                        current_like24=int(base_like24),
                        min_like24=2,
                        max_like24=4,
                    )
                    if middle is None:
                        continue
                    candidate_segments = [dict(left_anchor)] + list(middle or []) + [dict(right_anchor)]
                    like24_count = self._planned_ring_like24_count(candidate_segments)
                    if like24_count < 2 or like24_count > 4:
                        continue
                    ordered_segments = candidate_segments
                    break
                if ordered_segments:
                    break
        if not ordered_segments:
            if not missing_slots:
                if not middle_segments:
                    missing_slots.append("side_inner_slot_missing")
                else:
                    missing_slots.append("ring_backbone_nodes_insufficient")
        source_views = self._planned_ring_source_views(ordered_segments)
        return RingSkeletonPlan(
            ring_kind="fatty_ring_side",
            skeleton_kind="side_ring_plan",
            source_views=source_views,
            ordered_segments=list(ordered_segments or []),
            missing_slots=sorted({str(item) for item in list(missing_slots or []) if str(item)}),
            like24_count_min=2,
            like24_count_max=4,
            complete_ready=bool(ordered_segments) and not bool(missing_slots),
        )

    def _plan_upper_ring_templates(
        self,
        preferred_views: Sequence[AssemblyFragmentView],
    ) -> RingSkeletonPlan:
        available = [
            view for view in list(preferred_views or [])
            if self._fragment_available(int(view.fragment_id))
        ]
        anchors, path_segments = self._build_upper_ring_template_candidates(available)
        missing_slots: List[str] = []
        ordered_segments: List[Dict[str, Any]] = []
        closure_target_fragment_id: Optional[int] = None
        closure_target_node_id: Optional[int] = None

        if not anchors:
            missing_slots.append("fixed_top_anchor_missing")
        else:
            for anchor in list(anchors or []):
                anchor_view = anchor.get("view")
                remaining_span = 6 - int(anchor.get("backbone_span", 0) or 0)
                if remaining_span < 0:
                    continue
                path = self._search_ring_middle_segments(
                    path_segments,
                    remaining_span=int(remaining_span),
                    claimed_fragment_ids={
                        int(anchor.get("fragment_id"))
                        for _ in [0]
                        if anchor.get("fragment_id") is not None
                    },
                    used_candidate_ids={str(anchor.get("candidate_id", ""))},
                    current_like24=int(anchor.get("like24_units", 0) or 0),
                    min_like24=2,
                    max_like24=3,
                )
                if path is None:
                    continue
                candidate_segments = [dict(anchor)] + list(path or [])
                like24_count = self._planned_ring_like24_count(candidate_segments)
                if like24_count < 2 or like24_count > 3:
                    continue
                ordered_segments = candidate_segments
                if isinstance(anchor_view, AssemblyFragmentView):
                    closure_target_fragment_id = int(anchor_view.fragment_id)
                    anchor_node_ids = self._view_ring_backbone_node_ids(
                        anchor_view,
                        skeleton_kind="upper_lower",
                        reverse=False,
                    )
                    if anchor_node_ids:
                        closure_target_node_id = None if anchor_node_ids[-1] is None else int(anchor_node_ids[-1])
                break
        if not ordered_segments:
            if not missing_slots:
                if not path_segments:
                    missing_slots.append("upper_path_slot_missing")
                else:
                    missing_slots.append("upper_ring_path_span_unresolved")
        source_views = self._planned_ring_source_views(ordered_segments)
        return RingSkeletonPlan(
            ring_kind="fatty_ring_upper_lower",
            skeleton_kind="upper_lower_ring_plan",
            source_views=source_views,
            ordered_segments=list(ordered_segments or []),
            closure_target_fragment_id=closure_target_fragment_id,
            closure_target_node_id=closure_target_node_id,
            missing_slots=sorted({str(item) for item in list(missing_slots or []) if str(item)}),
            like24_count_min=2,
            like24_count_max=3,
            complete_ready=bool(ordered_segments) and not bool(missing_slots),
        )

    def _plan_fused_ring_templates(
        self,
        preferred_views: Sequence[AssemblyFragmentView],
    ) -> RingSkeletonPlan:
        available = [
            view for view in list(preferred_views or [])
            if self._fragment_available(int(view.fragment_id))
        ]
        anchors = [
            view for view in available
            if "fused" in self._view_ring_allowed_skeletons(view)
            and (
                self._view_has_ring_contract_role(view, "fused_ring_plan", "fused_outer_anchor")
                or self._view_has_slot_profile(view, "fused_outer_anchor")
            )
            and bool(dict(view.evidence or {}).get("touches_aromatic_anchor", False))
            and self._view_is_structurally_explicit(view, max_suppressed=1)
        ]
        bridgeheads = [
            view for view in available
            if "fused" in self._view_ring_allowed_skeletons(view)
            and (self._view_is_fused_pair_bridgehead(view) or self._view_supports_fused_bridgehead_single(view))
            and self._view_is_ring_core_explicit(view)
        ]
        inner_cores = [
            view for view in available
            if self._view_supports_fused_inner(view)
            and not self._view_is_fused_pair_bridgehead(view)
            and self._view_is_ring_core_explicit(view)
        ]
        missing_slots: List[str] = []
        if len(anchors) < 2:
            missing_slots.append("fused_outer_anchor_missing")
        if not bridgeheads:
            missing_slots.append("fused_bridgehead_missing")
        if not inner_cores:
            missing_slots.append("fused_inner_core_missing")
        anchors.sort(key=lambda item: (self._view_ring_contract_priority(item, "fused_ring_plan"), int(item.fragment_id)))
        bridgeheads.sort(key=lambda item: (
            0 if self._view_is_fused_pair_bridgehead(item) else 1,
            self._view_ring_contract_priority(item, "fused_ring_plan"),
            int(item.fragment_id),
        ))
        pair_bridge = next((item for item in bridgeheads if self._view_is_fused_pair_bridgehead(item)), None)
        inner_cores.sort(
            key=lambda item: self._fused_inner_priority(
                item,
                pair_bridgehead_mode=bool(pair_bridge is not None),
            )
        )
        left_anchor = anchors[0] if anchors else None
        right_anchor = next((item for item in anchors[1:] if left_anchor is not None and int(item.fragment_id) != int(left_anchor.fragment_id)), None)
        single_bridgeheads = [item for item in bridgeheads if not self._view_is_fused_pair_bridgehead(item)]
        bridge = pair_bridge if pair_bridge is not None else (single_bridgeheads[0] if single_bridgeheads else None)
        right_bridge = pair_bridge if pair_bridge is not None else (
            next(
                (
                    item for item in single_bridgeheads[1:]
                    if bridge is not None and int(item.fragment_id) != int(bridge.fragment_id)
                ),
                None,
            )
            if bridge is not None else None
        )
        disallowed_fragment_ids = {
            int(left_anchor.fragment_id) if left_anchor is not None else -1,
            int(right_anchor.fragment_id) if right_anchor is not None else -1,
            int(bridge.fragment_id) if bridge is not None else -1,
            int(right_bridge.fragment_id) if right_bridge is not None else -1,
        }
        core = next(
            (
                item
                for item in inner_cores
                if int(item.fragment_id) not in disallowed_fragment_ids
                and self._fused_inner_pool23_distribution(
                    self._view_ring_backbone_span(item, skeleton_kind="fused")
                ) is not None
            ),
            None,
        )
        if left_anchor is not None and right_anchor is None:
            missing_slots.append("right_fused_outer_anchor_missing")
        if bridge is None:
            missing_slots.append("left_bridgehead_missing")
        if right_bridge is None:
            missing_slots.append("right_bridgehead_missing")
        if bridge is not None and core is None:
            missing_slots.append("inner_ring_slot_unfilled")
        ordered_segments: List[Dict[str, Any]] = []
        if left_anchor is not None and bridge is not None and core is not None and right_bridge is not None and right_anchor is not None:
            left_anchor_spec = self._build_ring_template_segment(
                skeleton_kind="fused",
                role="left_outer_anchor",
                candidate_id=f"view:fused:left_anchor:{int(left_anchor.fragment_id)}",
                view=left_anchor,
                reverse=False,
                priority=(self._view_ring_contract_priority(left_anchor, "fused_ring_plan"), int(left_anchor.fragment_id)),
            )
            left_bridge_spec = self._build_ring_template_segment(
                skeleton_kind="fused",
                role="left_bridgehead",
                candidate_id=f"view:fused:left_bridge:{int(bridge.fragment_id)}",
                view=bridge,
                reverse=False,
                priority=(self._view_ring_contract_priority(bridge, "fused_ring_plan"), int(bridge.fragment_id)),
            )
            inner_spec = self._build_ring_template_segment(
                skeleton_kind="fused",
                role="inner_core",
                candidate_id=f"view:fused:inner:{int(core.fragment_id)}",
                view=core,
                reverse=False,
                priority=self._fused_inner_priority(core, pair_bridgehead_mode=bool(pair_bridge is not None)),
            )
            right_bridge_spec = self._build_ring_template_segment(
                skeleton_kind="fused",
                role="right_bridgehead",
                candidate_id=f"view:fused:right_bridge:{int(right_bridge.fragment_id)}",
                view=right_bridge,
                reverse=False,
                priority=(self._view_ring_contract_priority(right_bridge, "fused_ring_plan"), int(right_bridge.fragment_id)),
            )
            right_anchor_spec = self._build_ring_template_segment(
                skeleton_kind="fused",
                role="right_outer_anchor",
                candidate_id=f"view:fused:right_anchor:{int(right_anchor.fragment_id)}",
                view=right_anchor,
                reverse=True,
                priority=(self._view_ring_contract_priority(right_anchor, "fused_ring_plan"), int(right_anchor.fragment_id)),
            )
            if left_anchor_spec is not None and left_bridge_spec is not None and inner_spec is not None and right_bridge_spec is not None and right_anchor_spec is not None:
                if int(bridge.fragment_id) == int(right_bridge.fragment_id):
                    right_bridge_spec["shared_candidate_id"] = str(left_bridge_spec.get("candidate_id", ""))
                ordered_segments = [
                    dict(left_anchor_spec),
                    dict(left_bridge_spec),
                    dict(inner_spec),
                    dict(right_bridge_spec),
                    dict(right_anchor_spec),
                ]
        if not ordered_segments:
            left_anchor_specs = [
                self._build_ring_template_segment(
                    skeleton_kind="fused",
                    role="left_outer_anchor",
                    candidate_id=f"view:fused:left_anchor:{int(view.fragment_id)}",
                    view=view,
                    reverse=False,
                    priority=(self._view_ring_contract_priority(view, "fused_ring_plan"), int(view.fragment_id)),
                )
                for view in anchors
            ]
            right_anchor_specs = [
                self._build_ring_template_segment(
                    skeleton_kind="fused",
                    role="right_outer_anchor",
                    candidate_id=f"view:fused:right_anchor:{int(view.fragment_id)}",
                    view=view,
                    reverse=True,
                    priority=(self._view_ring_contract_priority(view, "fused_ring_plan"), int(view.fragment_id)),
                )
                for view in anchors
            ]
            left_anchor_specs = [item for item in left_anchor_specs if item is not None]
            right_anchor_specs = [item for item in right_anchor_specs if item is not None]
            left_anchor_specs.extend(self._build_pool_ring_template_segments(
                skeleton_kind="fused",
                role="left_outer_anchor",
                allowed_su=[24],
                priority_by_su={24: 30},
            ))
            right_anchor_specs.extend(self._build_pool_ring_template_segments(
                skeleton_kind="fused",
                role="right_outer_anchor",
                allowed_su=[24],
                priority_by_su={24: 30},
            ))

            bridge_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for left_view in bridgeheads:
                left_spec = self._build_ring_template_segment(
                    skeleton_kind="fused",
                    role="left_bridgehead",
                    candidate_id=f"view:fused:left_bridge:{int(left_view.fragment_id)}",
                    view=left_view,
                    reverse=False,
                    priority=(self._view_ring_contract_priority(left_view, "fused_ring_plan"), int(left_view.fragment_id)),
                )
                if left_spec is None:
                    continue
                if self._view_is_fused_pair_bridgehead(left_view):
                    right_spec = dict(left_spec)
                    right_spec["role"] = "right_bridgehead"
                    right_spec["candidate_id"] = f"view:fused:right_bridge:{int(left_view.fragment_id)}"
                    right_spec["shared_candidate_id"] = str(left_spec.get("candidate_id", ""))
                    bridge_pairs.append((left_spec, right_spec))
                    continue
                for right_view in bridgeheads:
                    if int(right_view.fragment_id) == int(left_view.fragment_id):
                        continue
                    right_spec = self._build_ring_template_segment(
                        skeleton_kind="fused",
                        role="right_bridgehead",
                        candidate_id=f"view:fused:right_bridge:{int(right_view.fragment_id)}",
                        view=right_view,
                        reverse=False,
                        priority=(self._view_ring_contract_priority(right_view, "fused_ring_plan"), int(right_view.fragment_id)),
                    )
                    if right_spec is not None:
                        bridge_pairs.append((left_spec, right_spec))
            pool_left_bridges = self._build_pool_ring_template_segments(
                skeleton_kind="fused",
                role="left_bridgehead",
                allowed_su=[24],
                priority_by_su={24: 30},
            )
            pool_right_bridges = self._build_pool_ring_template_segments(
                skeleton_kind="fused",
                role="right_bridgehead",
                allowed_su=[24],
                priority_by_su={24: 30},
            )
            for left_spec in pool_left_bridges:
                for right_spec in pool_right_bridges:
                    if left_spec.get("pool_node_id") == right_spec.get("pool_node_id"):
                        continue
                    bridge_pairs.append((left_spec, right_spec))

            inner_specs = [
                self._build_ring_template_segment(
                    skeleton_kind="fused",
                    role="inner_core",
                    candidate_id=f"view:fused:inner:{int(view.fragment_id)}",
                    view=view,
                    reverse=False,
                    priority=self._fused_inner_priority(view, pair_bridgehead_mode=False),
                )
                for view in inner_cores
            ]
            inner_specs = [item for item in inner_specs if item is not None]
            inner_specs.extend(self._build_pool_ring_template_segments(
                skeleton_kind="fused",
                role="inner_core",
                allowed_su=[23, 24],
                priority_by_su={23: 20, 24: 30},
            ))

            def _candidate_owner_ids(spec: Dict[str, Any]) -> Set[Tuple[str, int]]:
                out: Set[Tuple[str, int]] = set()
                if spec.get("fragment_id") is not None:
                    out.add(("fragment", int(spec.get("fragment_id"))))
                if spec.get("pool_node_id") is not None:
                    out.add(("pool", int(spec.get("pool_node_id"))))
                return out

            for left_spec in sorted(left_anchor_specs, key=lambda item: tuple(item.get("priority", (99,)))):
                for right_spec in sorted(right_anchor_specs, key=lambda item: tuple(item.get("priority", (99,)))):
                    if _candidate_owner_ids(left_spec) & _candidate_owner_ids(right_spec):
                        continue
                    for left_bridge_spec, right_bridge_spec in bridge_pairs:
                        base_specs = [left_spec, right_spec, left_bridge_spec, right_bridge_spec]
                        owner_sets = [_candidate_owner_ids(item) for item in base_specs]
                        duplicate_owner = any(
                            owner_sets[index] & owner_sets[other]
                            for index in range(len(owner_sets))
                            for other in range(index + 1, len(owner_sets))
                            if not (
                                str(right_bridge_spec.get("shared_candidate_id", ""))
                                and index == 2 and other == 3
                            )
                        )
                        if duplicate_owner:
                            continue
                        outer_span = sum(int(item.get("backbone_span", 0) or 0) for item in base_specs)
                        if outer_span != 6:
                            continue
                        for inner_spec in sorted(inner_specs, key=lambda item: tuple(item.get("priority", (99,)))):
                            if any(_candidate_owner_ids(inner_spec) & owners for owners in owner_sets):
                                continue
                            if self._fused_inner_pool23_distribution(int(inner_spec.get("backbone_span", 0) or 0)) is None:
                                continue
                            candidate_segments = [
                                dict(left_spec),
                                dict(left_bridge_spec),
                                dict(inner_spec),
                                dict(right_bridge_spec),
                                dict(right_spec),
                            ]
                            like24_count = self._planned_ring_like24_count(candidate_segments)
                            if 4 <= like24_count <= 6:
                                ordered_segments = candidate_segments
                                missing_slots = []
                                break
                        if ordered_segments:
                            break
                    if ordered_segments:
                        break
                if ordered_segments:
                    break
        source_views = self._planned_ring_source_views(ordered_segments)
        plan = RingSkeletonPlan(
            ring_kind="fatty_ring_fused",
            skeleton_kind="fused_ring_plan",
            source_views=source_views,
            ordered_segments=list(ordered_segments or []),
            missing_slots=sorted({str(item) for item in missing_slots}),
            like24_count_min=4,
            like24_count_max=6,
            complete_ready=bool(ordered_segments) and not bool(missing_slots),
        )
        return plan

    def _materialize_ring_backbone(
        self,
        plan: RingSkeletonPlan,
        *,
        skeleton_kind: str,
    ) -> RingBackboneMaterialization:
        self.last_ring_materialization_reason = ""
        ring_branch_donors: List[AssemblyFragmentView] = []
        resolved_segments: List[ViewSegment] = []
        for segment_spec in list(plan.ordered_segments or []):
            segment = self._resolve_planned_ring_segment(
                segment_spec,
                ring_branch_donors,
                skeleton_kind=str(skeleton_kind),
            )
            if segment is None:
                raise ValueError(str(self.last_ring_materialization_reason or f"{str(skeleton_kind)}_ring_segment_resolution_failed"))
            if self._segment_has_forbidden_ring_branch_mix(segment):
                self.last_ring_materialization_reason = "ring_forbidden_branch_mix"
                raise ValueError("ring_forbidden_branch_mix")
            resolved_segments.append(segment)
        source_views = list({
            int(view.fragment_id): view
            for segment in list(resolved_segments)
            for view in list(segment.source_views or [])
        }.values())
        backbone_labels: List[str] = []
        backbone_node_ids: List[Optional[int]] = []
        topology_parts: List[str] = []
        node_id_parts: List[str] = []
        for segment in list(resolved_segments):
            backbone_labels.extend([str(value) for value in list(segment.ring_backbone_labels or []) if str(value)])
            backbone_node_ids.extend(_normalize_ring_backbone_node_ids(segment.ring_backbone_node_ids or []))
            segment_topology = _clean_ring_segment_text(segment.diagnostics.get("branch_render", ""))
            segment_node_ids = _clean_ring_segment_text(segment.diagnostics.get("branch_render_gid", ""))
            if not str(segment_topology) or not str(segment_node_ids):
                raise ValueError("resolved ring segment missing rendered topology")
            topology_parts.append(str(segment_topology))
            node_id_parts.append(str(segment_node_ids))
        return RingBackboneMaterialization(
            source_views=source_views,
            labels=list(backbone_labels),
            node_ids=list(backbone_node_ids),
            backbone_labels=backbone_labels,
            backbone_node_ids=backbone_node_ids,
            topology_notation="-".join(str(value) for value in topology_parts if str(value)),
            node_id_notation="-".join(str(value) for value in node_id_parts if str(value)),
            support_node_ids=self._resolved_ring_support_node_ids(resolved_segments),
            pool_node_ids_by_su=self._merge_pool_ids(
                segment.pool_node_ids_by_su for segment in resolved_segments
            ),
        )

    def _resolved_ring_support_node_ids(
        self,
        resolved_segments: Sequence[ViewSegment],
    ) -> List[int]:
        return sorted({
            int(node_id)
            for segment in list(resolved_segments or [])
            for node_id in list(getattr(segment, "suppressed_support_node_ids", []) or [])
        })

    def _render_ring_product_from_backbone(
        self,
        *,
        product_kind: str,
        reason: str,
        materialization: RingBackboneMaterialization,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[TopologyAssemblyProduct]:
        diagnostics_payload = {
            "topology_notation": str(materialization.topology_notation),
            "node_id_notation": str(materialization.node_id_notation),
            "actual_topology_su": str(materialization.topology_notation),
            "actual_topology_gid": str(materialization.node_id_notation),
            "ring_backbone_labels": list(materialization.backbone_labels or []),
            "ring_backbone_node_ids": list(materialization.backbone_node_ids or []),
            **dict(diagnostics or {}),
        }
        product = self._make_product(
            product_kind=str(product_kind),
            reason=str(reason),
            source_views=list(materialization.source_views or []),
            labels=list(materialization.labels or []),
            node_ids=list(materialization.node_ids or []),
            branch_map={},
            branch_id_map={},
            suppressed_support_node_ids=self._collect_suppressed_support_node_ids(
                materialization.source_views,
                extra_ids=materialization.support_node_ids,
            ),
            diagnostics=diagnostics_payload,
        )
        if product is not None:
            product.claimed_resource_ids = {
                str(key): sorted({int(value) for value in list(values or [])})
                for key, values in dict(materialization.pool_node_ids_by_su or {}).items()
                if list(values or [])
            }
        return product

    def _materialize_linear_ring_plan(
        self,
        plan: RingSkeletonPlan,
        branch_donors: List[AssemblyFragmentView],
        *,
        skeleton_kind: str,
        product_kind: str,
        reason: str,
        ring_template_kind: str,
        ring_has_fixed_anchor: bool,
        extra_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not bool(getattr(plan, "complete_ready", False)) or list(plan.missing_slots or []):
            return False
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        try:
            materialization = self._materialize_ring_backbone(
                plan,
                skeleton_kind=str(skeleton_kind),
            )
        except Exception as exc:
            self.last_ring_materialization_reason = str(exc or self.last_ring_materialization_reason or "")
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        like24_count = self._planned_ring_like24_count(plan.ordered_segments)
        if like24_count < int(plan.like24_count_min) or like24_count > int(plan.like24_count_max):
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        diagnostics = {
            "ring_like24_count": int(like24_count),
            "ring_template_kind": str(ring_template_kind),
            "ring_has_fixed_anchor": bool(ring_has_fixed_anchor),
            **dict(extra_diagnostics or {}),
        }
        product = self._render_ring_product_from_backbone(
            product_kind=str(product_kind),
            reason=str(reason),
            materialization=materialization,
            diagnostics=diagnostics,
        )
        if product is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed([int(view.fragment_id) for view in list(materialization.source_views or [])])
        return True

    def _materialize_fused_ring_plan(self, plan: RingSkeletonPlan, branch_donors: List[AssemblyFragmentView]) -> bool:
        if not bool(getattr(plan, "complete_ready", False)):
            return False
        if list(plan.missing_slots or []):
            return False
        spec_by_role = {
            str(item.get("role", "")): dict(item or {})
            for item in list(plan.ordered_segments or [])
        }
        left_anchor_spec = dict(spec_by_role.get("left_outer_anchor", {}) or {})
        left_bridge_spec = dict(spec_by_role.get("left_bridgehead", {}) or {})
        inner_spec = dict(spec_by_role.get("inner_core", {}) or {})
        right_bridge_spec = dict(spec_by_role.get("right_bridgehead", {}) or {})
        right_anchor_spec = dict(spec_by_role.get("right_outer_anchor", {}) or {})
        if not all([left_anchor_spec, left_bridge_spec, inner_spec, right_bridge_spec, right_anchor_spec]):
            return False
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        ring_branch_donors: List[AssemblyFragmentView] = []
        left_anchor_segment = self._resolve_planned_ring_segment(
            left_anchor_spec,
            ring_branch_donors,
            skeleton_kind="fused",
        )
        left_bridge_segment = self._resolve_planned_ring_segment(
            left_bridge_spec,
            ring_branch_donors,
            skeleton_kind="fused",
        )
        inner_segment = self._resolve_planned_ring_segment(
            inner_spec,
            ring_branch_donors,
            skeleton_kind="fused",
        )
        pair_bridgehead_mode = bool(
            str(right_bridge_spec.get("shared_candidate_id", ""))
            and str(right_bridge_spec.get("shared_candidate_id", "")) == str(left_bridge_spec.get("candidate_id", ""))
        )
        right_bridge_segment = left_bridge_segment if pair_bridgehead_mode else self._resolve_planned_ring_segment(
            right_bridge_spec,
            ring_branch_donors,
            skeleton_kind="fused",
        )
        right_anchor_segment = self._resolve_planned_ring_segment(
            right_anchor_spec,
            ring_branch_donors,
            skeleton_kind="fused",
        )
        if left_anchor_segment is None or left_bridge_segment is None or inner_segment is None or right_bridge_segment is None or right_anchor_segment is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        inner_core_span = int(len(list(inner_segment.ring_backbone_labels or [])))
        filler_distribution = self._fused_inner_pool23_distribution(inner_core_span)
        if filler_distribution is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        left_filler_count, right_filler_count = filler_distribution
        filler_total = int(left_filler_count + right_filler_count)
        filler_ids = self.allocator.claim(23, filler_total) if filler_total > 0 else []
        if filler_total > 0 and filler_ids is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        filler_ids = [int(node_id) for node_id in list(filler_ids or [])]
        left_filler_ids = list(filler_ids[:left_filler_count])
        right_filler_ids = list(filler_ids[left_filler_count:])
        like24_count = self._planned_ring_like24_count(plan.ordered_segments)
        if like24_count < int(plan.like24_count_min) or like24_count > int(plan.like24_count_max):
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        inner_view = inner_spec.get("view")
        inner_bridgehead_capable = bool(
            isinstance(inner_view, AssemblyFragmentView)
            and self._view_has_slot_profile(inner_view, "fused_bridgehead")
        )
        outer_left = _clean_ring_segment_text(left_anchor_segment.diagnostics.get("branch_render", ""))
        inner_text = _clean_ring_segment_text(inner_segment.diagnostics.get("branch_render", ""))
        outer_right = _clean_ring_segment_text(right_anchor_segment.diagnostics.get("branch_render", ""))
        outer_left_gid = _clean_ring_segment_text(left_anchor_segment.diagnostics.get("branch_render_gid", ""))
        inner_gid = _clean_ring_segment_text(inner_segment.diagnostics.get("branch_render_gid", ""))
        outer_right_gid = _clean_ring_segment_text(right_anchor_segment.diagnostics.get("branch_render_gid", ""))
        if pair_bridgehead_mode:
            bridge_tokens, bridge_gid_tokens = _render_token_sequence(
                left_bridge_segment.labels,
                left_bridge_segment.node_ids,
                branch_map=left_bridge_segment.branch_map,
                branch_id_map=left_bridge_segment.branch_id_map,
            )
            if len(bridge_tokens) < 2 or len(bridge_gid_tokens) < 2:
                self.allocator.rollback(snapshot)
                branch_donors[:] = donor_snapshot
                return False
            left_bridge_text = _clean_ring_segment_text(bridge_tokens[0])
            right_bridge_text = _clean_ring_segment_text(bridge_tokens[-1])
            left_bridge_gid = _clean_ring_segment_text(bridge_gid_tokens[0])
            right_bridge_gid = _clean_ring_segment_text(bridge_gid_tokens[-1])
            pair_bridge_labels = [str(value) for value in list(left_bridge_segment.ring_backbone_labels or []) if str(value)]
            pair_bridge_node_ids = _normalize_ring_backbone_node_ids(left_bridge_segment.ring_backbone_node_ids or [])
            if len(pair_bridge_labels) < 2 or len(pair_bridge_node_ids) < 2:
                self.allocator.rollback(snapshot)
                branch_donors[:] = donor_snapshot
                return False
            left_bridge_labels = [str(pair_bridge_labels[0])]
            right_bridge_labels = [str(pair_bridge_labels[-1])]
            left_bridge_node_ids = [pair_bridge_node_ids[0]]
            right_bridge_node_ids = [pair_bridge_node_ids[-1]]
        else:
            left_bridge_text = _clean_ring_segment_text(left_bridge_segment.diagnostics.get("branch_render", ""))
            right_bridge_text = _clean_ring_segment_text(right_bridge_segment.diagnostics.get("branch_render", ""))
            left_bridge_gid = _clean_ring_segment_text(left_bridge_segment.diagnostics.get("branch_render_gid", ""))
            right_bridge_gid = _clean_ring_segment_text(right_bridge_segment.diagnostics.get("branch_render_gid", ""))
            left_bridge_labels = [str(value) for value in list(left_bridge_segment.ring_backbone_labels or []) if str(value)]
            right_bridge_labels = [str(value) for value in list(right_bridge_segment.ring_backbone_labels or []) if str(value)]
            left_bridge_node_ids = _normalize_ring_backbone_node_ids(left_bridge_segment.ring_backbone_node_ids or [])
            right_bridge_node_ids = _normalize_ring_backbone_node_ids(right_bridge_segment.ring_backbone_node_ids or [])
        if not left_bridge_text or not right_bridge_text:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        left_filler_labels = ["23"] * int(left_filler_count)
        right_filler_labels = ["23"] * int(right_filler_count)
        inner_loop_tokens = list(left_filler_labels)
        if inner_text:
            inner_loop_tokens.append(inner_text)
        inner_loop_tokens.extend(list(right_filler_labels))
        inner_loop_gid_tokens = [str(node_id) for node_id in list(left_filler_ids or [])]
        if inner_gid:
            inner_loop_gid_tokens.append(inner_gid)
        inner_loop_gid_tokens.extend(str(node_id) for node_id in list(right_filler_ids or []))
        if not inner_loop_tokens or not inner_loop_gid_tokens:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        inner_loop_text = "-".join(str(token) for token in list(inner_loop_tokens or []) if str(token))
        inner_loop_gid = "-".join(str(token) for token in list(inner_loop_gid_tokens or []) if str(token))
        topology_notation = f"{outer_left}-{left_bridge_text}-[-{inner_loop_text}-]-{right_bridge_text}-{outer_right}"
        node_id_notation = f"{outer_left_gid}-{left_bridge_gid}-[-{inner_loop_gid}-]-{right_bridge_gid}-{outer_right_gid}"
        outer_backbone_labels = (
            list(left_anchor_segment.ring_backbone_labels)
            + list(left_bridge_labels)
            + [str(item) for item in list(right_bridge_labels or [])]
            + list(right_anchor_segment.ring_backbone_labels)
        )
        outer_backbone_node_ids = (
            list(left_anchor_segment.ring_backbone_node_ids)
            + _normalize_ring_backbone_node_ids(left_bridge_node_ids)
            + _normalize_ring_backbone_node_ids(right_bridge_node_ids)
            + list(right_anchor_segment.ring_backbone_node_ids)
        )
        inner_backbone_labels = (
            [str(left_bridge_labels[-1])] if list(left_bridge_labels or []) else []
        ) + list(left_filler_labels) + list(inner_segment.ring_backbone_labels) + list(right_filler_labels) + (
            [str(right_bridge_labels[0])] if list(right_bridge_labels or []) else []
        )
        inner_backbone_node_ids = (
            [left_bridge_node_ids[-1]] if list(left_bridge_node_ids or []) else []
        ) + list(left_filler_ids) + list(inner_segment.ring_backbone_node_ids) + list(right_filler_ids) + (
            [right_bridge_node_ids[0]] if list(right_bridge_node_ids or []) else []
        )
        source_views = list({
            int(view.fragment_id): view
            for view in list(left_anchor_segment.source_views)
            + list(left_bridge_segment.source_views)
            + list(inner_segment.source_views)
            + list(right_bridge_segment.source_views)
            + list(right_anchor_segment.source_views)
        }.values())
        materialization = RingBackboneMaterialization(
            source_views=source_views,
            labels=list(outer_backbone_labels),
            node_ids=list(outer_backbone_node_ids),
            backbone_labels=outer_backbone_labels,
            backbone_node_ids=outer_backbone_node_ids,
            topology_notation=topology_notation,
            node_id_notation=node_id_notation,
            support_node_ids=self._resolved_ring_support_node_ids([
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]),
            pool_node_ids_by_su=self._merge_pool_ids([
                left_anchor_segment.pool_node_ids_by_su,
                left_bridge_segment.pool_node_ids_by_su,
                inner_segment.pool_node_ids_by_su,
                right_bridge_segment.pool_node_ids_by_su,
                right_anchor_segment.pool_node_ids_by_su,
                {"23": list(filler_ids)},
            ]),
        )
        product = self._render_ring_product_from_backbone(
            product_kind="fatty_ring_fused",
            reason="ring:fused_template",
            materialization=materialization,
            diagnostics={
                "ring_outer_backbone_labels": list(outer_backbone_labels or []),
                "ring_outer_backbone_node_ids": list(outer_backbone_node_ids or []),
                "ring_inner_backbone_labels": list(inner_backbone_labels or []),
                "ring_inner_backbone_node_ids": list(inner_backbone_node_ids or []),
                "ring_like24_count": int(like24_count),
                "ring_template_kind": "fused",
                "ring_pair_bridgehead_mode": bool(pair_bridgehead_mode),
                "ring_inner_bridgehead_capable": bool(inner_bridgehead_capable),
                "ring_inner_pool23_count": int(filler_total),
                "ring_inner_pool23_distribution": [int(left_filler_count), int(right_filler_count)],
                "ring_bridgehead_ids": [
                    int(left_bridge_node_ids[-1]) if list(left_bridge_node_ids or []) and left_bridge_node_ids[-1] is not None else None,
                    int(right_bridge_node_ids[0]) if list(right_bridge_node_ids or []) and right_bridge_node_ids[0] is not None else None,
                ],
            },
        )
        if product is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed([int(view.fragment_id) for view in list(materialization.source_views or [])])
        return True

    def _assemble_ring(
        self,
        *,
        product_kind: str,
        preferred_views: Sequence[AssemblyFragmentView],
        branch_donors: List[AssemblyFragmentView],
        like24_fill_views: Optional[Sequence[AssemblyFragmentView]] = None,
    ) -> bool:
        plan: Optional[RingSkeletonPlan] = None
        reason = "ring_slot_selection_failed"

        def _build_and_materialize(
            candidate_views: Sequence[AssemblyFragmentView],
        ) -> Tuple[bool, Optional[RingSkeletonPlan], str]:
            local_plan: Optional[RingSkeletonPlan] = None
            local_reason = "ring_slot_selection_failed"
            if str(product_kind) == "fatty_ring_fused":
                local_plan = self._plan_fused_ring_templates(candidate_views)
                if bool(getattr(local_plan, "complete_ready", False)) and not list(local_plan.missing_slots or []) and self._materialize_fused_ring_plan(local_plan, branch_donors):
                    return True, local_plan, ""
                local_reason = str((list(local_plan.missing_slots or []) or [self.last_ring_materialization_reason or local_reason])[0])
            elif str(product_kind) == "fatty_ring_upper_lower":
                local_plan = self._plan_upper_ring_templates(candidate_views)
                if (
                    bool(getattr(local_plan, "complete_ready", False))
                    and not list(local_plan.missing_slots or [])
                    and self._materialize_linear_ring_plan(
                        local_plan,
                        branch_donors,
                        skeleton_kind="upper_lower",
                        product_kind="fatty_ring_upper_lower",
                        reason="ring:upper_lower_template",
                        ring_template_kind="upper_lower",
                        ring_has_fixed_anchor=True,
                        extra_diagnostics={
                            "closure_target_fragment_id": local_plan.closure_target_fragment_id,
                            "closure_target_node_id": local_plan.closure_target_node_id,
                        },
                    )
                ):
                    return True, local_plan, ""
                local_reason = str((list(local_plan.missing_slots or []) or [self.last_ring_materialization_reason or local_reason])[0])
            elif str(product_kind) == "fatty_ring_side":
                local_plan = self._plan_side_ring_templates(candidate_views)
                if (
                    bool(getattr(local_plan, "complete_ready", False))
                    and not list(local_plan.missing_slots or [])
                    and self._materialize_linear_ring_plan(
                        local_plan,
                        branch_donors,
                        skeleton_kind="side",
                        product_kind="fatty_ring_side",
                        reason="ring:side_template",
                        ring_template_kind="side",
                        ring_has_fixed_anchor=False,
                    )
                ):
                    return True, local_plan, ""
                local_reason = str((list(local_plan.missing_slots or []) or [self.last_ring_materialization_reason or local_reason])[0])
            return False, local_plan, local_reason

        created, plan, reason = _build_and_materialize(preferred_views)
        if created:
            return True

        # Classified like24 fragments may supplement an already selected
        # external ring candidate, but are never created from resource nodes.
        fill_views = [
            view
            for view in list(like24_fill_views or [])
            if self._fragment_available(int(view.fragment_id))
            and str(getattr(view, "class_type", "")) in {"24_A", "24_B", "24_C", "24_D"}
        ]
        if fill_views:
            preferred_ids = {int(view.fragment_id) for view in list(preferred_views or [])}
            combined_views = list(preferred_views or []) + [
                view for view in list(fill_views or [])
                if int(view.fragment_id) not in preferred_ids
            ]
            created, fill_plan, fill_reason = _build_and_materialize(combined_views)
            if created:
                return True
            if fill_plan is not None:
                plan = fill_plan
            if str(fill_reason):
                reason = str(fill_reason)

        skeleton_name = {
            "fatty_ring_fused": "fused_ring_plan",
            "fatty_ring_upper_lower": "upper_lower_ring_plan",
            "fatty_ring_side": "side_ring_plan",
        }.get(str(product_kind), "")
        if plan is not None and list(plan.source_views or []):
            relevant_views = list(plan.source_views or [])
        else:
            allowed_name = _skeleton_allowed_name(str(skeleton_name))
            relevant_views = [
                view for view in list(preferred_views or [])
                if str(allowed_name) in self._view_ring_allowed_skeletons(view)
            ]
        for view in list(relevant_views or []):
            if not self._fragment_available(int(view.fragment_id)):
                continue
            self.ring_pending_reasons.setdefault(int(view.fragment_id), [])
            if str(reason) not in self.ring_pending_reasons[int(view.fragment_id)]:
                self.ring_pending_reasons[int(view.fragment_id)].append(str(reason))
            prior = dict(self.ring_pending_attempts.get(int(view.fragment_id), {}) or {})
            merged_missing = sorted({
                str(item)
                for item in list(prior.get("missing_slots", []) or [])
                + ([str(reason)] if str(reason) else [])
                + (list(plan.missing_slots or []) if plan is not None else [])
            })
            merged_attempt = {"missing_slots": merged_missing}
            self.ring_pending_attempts[int(view.fragment_id)] = merged_attempt
            required_counts_payload = dict(self.ring_pending_required_counts.get(int(view.fragment_id), {}) or {})
            if plan is not None:
                for segment in list(plan.ordered_segments or []):
                    for key, value in dict(segment.get("support_required_counts", {}) or {}).items():
                        key_s = str(key)
                        required_counts_payload[key_s] = int(required_counts_payload.get(key_s, 0)) + int(value)
            self.ring_pending_required_counts[int(view.fragment_id)] = required_counts_payload
        return False
    def _view_supports_fused_inner(self, view: AssemblyFragmentView) -> bool:
        return bool(
            "fused" in self._view_ring_allowed_skeletons(view)
            and (
                self._view_has_ring_contract_role(view, "fused_ring_plan", "fused_inner_core")
                or self._view_has_slot_profile(view, "fused_inner_segment", "fused_pair_inner_segment")
            )
        )

    def _view_is_fused_pair_bridgehead(self, view: AssemblyFragmentView) -> bool:
        return bool(
            self._view_has_ring_contract_role(view, "fused_ring_plan", "fused_bridgehead_pair")
            or str(self._view_bridgehead_role(view)) == "fused_bridgehead_substitute_pair"
        )

    def _view_supports_fused_bridgehead_single(self, view: AssemblyFragmentView) -> bool:
        return bool(
            "fused" in self._view_ring_allowed_skeletons(view)
            and (
                self._view_has_ring_contract_role(view, "fused_ring_plan", "fused_bridgehead")
                or self._view_has_slot_profile(view, "fused_bridgehead")
            )
            and not self._view_is_fused_pair_bridgehead(view)
            and self._view_ring_backbone_span(view, skeleton_kind="fused") == 1
        )

    def _fused_inner_priority(
        self,
        view: AssemblyFragmentView,
        *,
        pair_bridgehead_mode: bool,
    ) -> Tuple[int, int, int]:
        class_type = str(getattr(view, "class_type", ""))
        family = str(getattr(view, "family", ""))
        backbone_labels = self._view_ring_backbone_labels(
            view,
            skeleton_kind="fused",
            reverse=False,
        )
        subtype = str((backbone_labels or [None])[0] or "")
        bridgehead_capable = bool(
            self._view_has_slot_profile(view, "fused_bridgehead")
            or self._view_bridgehead_role(view) in {
                "fused_bridgehead_candidate",
                "fused_bridgehead_substitute_14",
                "fused_bridgehead_substitute_pair",
            }
        )
        if pair_bridgehead_mode:
            pair_mode_priority = {
                ("like24", "24_D", "14"): 0,
                ("like24", "24_D", ""): 1,
                ("like24_secondary", "24_CD_N", ""): 2,
                ("like24_secondary", "24_CD_M", ""): 3,
                ("like24", "24_C", "14"): 4,
                ("like24", "24_C", ""): 5,
                ("like24_bridge", "", ""): 9,
            }
            base_rank = 20
            exact_key = (family, class_type, subtype)
            loose_key = (family, class_type, "")
            family_key = (family, "", "")
            if exact_key in pair_mode_priority:
                base_rank = int(pair_mode_priority[exact_key])
            elif loose_key in pair_mode_priority:
                base_rank = int(pair_mode_priority[loose_key])
            elif family_key in pair_mode_priority:
                base_rank = int(pair_mode_priority[family_key])
            bridgehead_penalty = 10 if bool(bridgehead_capable) else 0
            return (
                int(base_rank + bridgehead_penalty),
                self._view_ring_contract_priority(view, "fused_ring_plan"),
                int(view.fragment_id),
            )
        base_priority = {
            "24_C": 3,
            "24_D": 4,
            "24_CD_N": 5,
            "24_AB_N_24_CD": 6,
            "24_CD_N_24_CD": 7,
        }
        return (
            int(base_priority.get(class_type, 20)),
            self._view_ring_contract_priority(view, "fused_ring_plan"),
            int(view.fragment_id),
        )

    def _fused_inner_pool23_distribution(
        self,
        inner_core_span: int,
    ) -> Optional[Tuple[int, int]]:
        target_inner_path_span = 4
        span = int(inner_core_span)
        if span <= 0 or span > int(target_inner_path_span):
            return None
        filler_total = int(target_inner_path_span - span)
        if filler_total < 0:
            return None
        left_fillers = int((filler_total + 1) // 2)
        right_fillers = int(filler_total - left_fillers)
        return left_fillers, right_fillers

    def _search_ring_middle_segments(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        remaining_span: int,
        claimed_fragment_ids: Set[int],
        used_candidate_ids: Set[str],
        current_like24: int,
        min_like24: int,
        max_like24: int,
    ) -> Optional[List[Dict[str, Any]]]:
        if int(remaining_span) == 0:
            if int(current_like24) < int(min_like24) or int(current_like24) > int(max_like24):
                return None
            return []
        ordered = sorted(
            [dict(item or {}) for item in list(candidates or [])],
            key=lambda item: tuple(item.get("priority", (99, 99, 99))),
        )

        def _search(
            span_left: int,
            used_fragments_local: Set[int],
            used_candidates_local: Set[str],
            like24_local: int,
        ) -> Optional[List[Dict[str, Any]]]:
            if int(span_left) == 0:
                if int(like24_local) < int(min_like24) or int(like24_local) > int(max_like24):
                    return None
                return []
            for candidate in list(ordered or []):
                candidate_id = str(candidate.get("candidate_id", ""))
                fragment_id = candidate.get("fragment_id")
                span = int(candidate.get("backbone_span", 0) or 0)
                if span <= 0 or span > int(span_left):
                    continue
                if candidate_id and candidate_id in used_candidates_local:
                    continue
                if fragment_id is not None and int(fragment_id) in used_fragments_local:
                    continue
                if (
                    str(candidate.get("kind", "")) == "pool"
                    and int(candidate.get("pool_su", -1)) == 24
                    and int(like24_local) >= int(min_like24)
                ):
                    continue
                next_like24 = int(like24_local) + int(candidate.get("like24_units", 0) or 0)
                if next_like24 > int(max_like24):
                    continue
                next_fragments = set(used_fragments_local)
                next_candidates = set(used_candidates_local)
                if fragment_id is not None:
                    next_fragments.add(int(fragment_id))
                if candidate_id:
                    next_candidates.add(candidate_id)
                tail = _search(
                    int(span_left) - int(span),
                    next_fragments,
                    next_candidates,
                    next_like24,
                )
                if tail is not None:
                    return [dict(candidate)] + list(tail or [])
            return None

        return _search(
            int(remaining_span),
            set(claimed_fragment_ids or set()),
            set(used_candidate_ids or set()),
            int(current_like24),
        )

    def _segment_has_forbidden_ring_branch_mix(
        self,
        segment: ViewSegment,
    ) -> bool:
        branch_topologies: List[str] = []
        for values in dict(segment.branch_map or {}).values():
            branch_topologies.extend(str(value) for value in list(values or []))
        has_h_closure = any(str(value) == "23-22" for value in branch_topologies)
        has_fixed_terminal = any(str(value).startswith("-22") or str(value) == "-22" for value in branch_topologies)
        return bool(has_h_closure and has_fixed_terminal)

    def _resolve_view_segment(
        self,
        view: AssemblyFragmentView,
        branch_donors: List[AssemblyFragmentView],
        *,
        reverse: bool = False,
        skeleton_kind: Optional[str] = None,
    ) -> Tuple[Optional[ViewSegment], Dict[str, int], Dict[str, Any]]:
        branch_override = (
            self._view_required_branch_kinds_for_skeleton(view, str(skeleton_kind))
            if skeleton_kind
            else None
        )
        branch_contract_override = (
            self._view_fixed_branch_contracts_for_skeleton(view, str(skeleton_kind))
            if skeleton_kind
            else None
        )
        branch_resolved, required_counts, branch_diag = self._resolve_branches(
            view,
            branch_donors,
            branch_kind_map_override=branch_override,
            branch_contract_map_override=branch_contract_override,
        )
        if branch_resolved is None:
            return None, dict(required_counts or {}), dict(branch_diag or {})
        validation_reason = self._validate_external_branch_closure_state(branch_diag)
        if validation_reason is not None:
            branch_diag = dict(branch_diag or {})
            branch_diag["external_branch_closure_reason"] = str(validation_reason)
            return None, dict(required_counts or {}), branch_diag
        if skeleton_kind:
            labels = self._view_ring_backbone_labels(view, skeleton_kind=str(skeleton_kind), reverse=False)
            node_ids = self._view_ring_backbone_node_ids(view, skeleton_kind=str(skeleton_kind), reverse=False)
            center_anchor_override = self._view_ring_branch_anchor_index(view, str(skeleton_kind))
            render_labels = list(labels)
            render_node_ids = list(node_ids)
            if (
                str(skeleton_kind) == "upper_lower"
                and self._view_is_upper_fixed_anchor(view)
                and list(view.main_labels or [])
                and list(view.main_node_ids or [])
            ):
                render_labels = [str(value) for value in list(view.main_labels or []) if str(value)]
                render_node_ids = [
                    None if value is None else int(value)
                    for value in list(view.main_node_ids or [])
                ]
        else:
            labels = list(view.main_labels or [])
            node_ids = list(view.main_node_ids or [])
            center_anchor_override = None
            render_labels = list(labels)
            render_node_ids = list(node_ids)
        render = self._render_resolved_view(
            view,
            branch_resolved,
            reverse=bool(reverse),
            labels_override=render_labels,
            node_ids_override=render_node_ids,
            center_anchor_override=center_anchor_override,
        )
        if reverse:
            labels = list(reversed(labels))
            node_ids = list(reversed(node_ids))
        if skeleton_kind:
            ring_backbone_labels = list(labels)
            ring_backbone_node_ids = [None if value is None else int(value) for value in list(node_ids or [])]
            resolved_skeleton_kind = str(skeleton_kind)
        else:
            ring_backbone_labels = []
            ring_backbone_node_ids = []
            resolved_skeleton_kind = ""
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        for item in list(branch_resolved or []):
            anchor_index = int(getattr(item, "anchor_index", int(view.branch_anchor_index)))
            if center_anchor_override is not None and int(anchor_index) == int(view.branch_anchor_index):
                anchor_index = int(center_anchor_override)
            if reverse:
                anchor_index = len(labels) - 1 - int(anchor_index)
            branch_map.setdefault(int(anchor_index), [])
            branch_map[int(anchor_index)].append(str(item.branch_topology))
            branch_id_map.setdefault(int(anchor_index), [])
            branch_id_map[int(anchor_index)].append(str(item.branch_node_ids))
        source_views = [view] + [
            self.view_by_id[x]
            for item in list(branch_resolved or [])
            for x in list(item.source_fragment_ids or [])
            if x in self.view_by_id
        ]
        pool_ids = self._merge_pool_ids([item.pool_node_ids_by_su for item in list(branch_resolved or [])])
        segment = ViewSegment(
            view=view,
            labels=labels,
            node_ids=node_ids,
            ring_backbone_labels=ring_backbone_labels,
            ring_backbone_node_ids=ring_backbone_node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            branch_modes=[str(item.mode) for item in list(branch_resolved or [])],
            pool_node_ids_by_su=pool_ids,
            source_views=source_views,
            suppressed_support_node_ids=list(render.suppressed_support_node_ids or []),
            ring_template_kind=resolved_skeleton_kind,
            ring_template_roles=list(dict(getattr(view, "ring_template_roles", {}) or {}).get(resolved_skeleton_kind, []) or []),
            diagnostics={
                "branch_render": str(render.topology_notation),
                "branch_render_gid": str(render.node_id_notation),
                "ring_backbone_labels": list(ring_backbone_labels or []),
                "ring_backbone_node_ids": list(ring_backbone_node_ids or []),
                **dict(branch_diag or {}),
            },
        )
        return segment, {}, dict(branch_diag or {})

    def _available_flex_reconcile_views(
        self,
        *,
        require_roles: Sequence[str],
        allow_ring_capable: bool,
        require_materializable: bool = True,
        require_generic_join: bool = False,
    ) -> List[AssemblyFragmentView]:
        roles = {str(role) for role in list(require_roles or []) if str(role)}
        candidates = self._available_views(
            lambda item: (
                (
                    str(self.flex_audit.get("scenario", "") or "") != "below_range"
                    or not self.selected_flex_candidate_ids
                    or int(item.fragment_id) in self.selected_flex_candidate_ids
                )
                and
                (not roles or any(str(role) in self._view_flex_reconcile_roles(item) for role in roles))
                and (bool(allow_ring_capable) or not bool(item.can_ring))
                and (not bool(require_materializable) or _is_chain_materializable_view(item))
                and (not bool(require_generic_join) or _is_generic_chain_join_view(item))
            )
        )
        return sorted(
            list(candidates or []),
            key=lambda item: (
                self._view_flex_reconcile_priority(item),
                int(item.fragment_id),
            ),
        )

    def _finalize_with_two_end_closures(
        self,
        view: AssemblyFragmentView,
        *,
        product_kind: str,
        left_mode: str,
        right_mode: str,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        branch_resolved, branch_required, branch_diag = self._resolve_branches(view, branch_donors)
        if branch_resolved is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=branch_required,
                available_counts=self.allocator.available_counts(),
                reason=self._branch_pending_reason(branch_diag, default="branch_pool_shortage"),
                partial_render=view.partial_render(),
                actual_render=view.real_render(),
            )
            return False
        validation_reason = self._validate_external_branch_closure_state(branch_diag)
        if validation_reason is not None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=branch_required,
                available_counts=self.allocator.available_counts(),
                reason=str(validation_reason),
                partial_render=view.partial_render(),
                actual_render=view.real_render(),
            )
            return False
        left_payload, left_required = self._build_main_closure(view, side="left", closure_mode=str(left_mode))
        right_payload, right_required = self._build_main_closure(view, side="right", closure_mode=str(right_mode))
        if left_payload is None or right_payload is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            required_counts = dict(branch_required or {})
            for chunk in (left_required, right_required):
                for key, value in dict(chunk or {}).items():
                    required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=required_counts,
                available_counts=self.allocator.available_counts(),
                reason="main_pool_shortage",
                partial_render=view.partial_render(),
                actual_render=view.real_render(),
            )
            return False
        branch_map, branch_id_map = self._build_branch_maps(view, branch_resolved)
        left_labels, left_ids = left_payload
        right_labels, right_ids = right_payload
        labels = list(left_labels) + list(view.main_labels or []) + list(right_labels)
        node_ids = list(left_ids) + list(view.main_node_ids or []) + list(right_ids)
        canonical_labels = list(view.canonical_labels or [])
        if str(left_mode) == "F1":
            canonical_labels = ["11", "23"] + canonical_labels
        else:
            canonical_labels = ["23", "22"] + canonical_labels
        if str(right_mode) == "F1":
            canonical_labels = canonical_labels + ["23", "11"]
        else:
            canonical_labels = canonical_labels + ["23", "22"]
        source_views = self._collect_source_views_from_branches([view], branch_resolved)
        validation_reason = self._invalid_product_reason(str(product_kind), labels, source_views, node_ids)
        suppressed_support_node_ids = self._collect_suppressed_support_node_ids(
            source_views,
            branch_resolved,
        )
        if validation_reason is not None:
            return self._record_invalid_materialization_failure(
                snapshot=snapshot,
                donor_snapshot=donor_snapshot,
                branch_donors=branch_donors,
                product_kind=str(product_kind),
                source_views=source_views,
                labels=labels,
                node_ids=node_ids,
                branch_map=branch_map,
                branch_id_map=branch_id_map,
                suppressed_support_node_ids=suppressed_support_node_ids,
                validation_reason=str(validation_reason),
            )
        product = self._make_product(
            product_kind=str(product_kind),
            reason=f"main_close:{left_mode}+{right_mode}",
            source_views=source_views,
            labels=labels,
            node_ids=node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
            diagnostics={"origin_type": str(view.class_type), "canonical_labels": canonical_labels, **dict(branch_diag or {})},
        )
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed(self._collect_claimed_fragment_ids([view], branch_resolved))
        return True

    def _finalize_single_view_product(
        self,
        view: AssemblyFragmentView,
        *,
        product_kind: str,
        reason: str,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        branch_resolved, required_counts, branch_diag = self._resolve_branches(view, branch_donors)
        if branch_resolved is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=required_counts,
                available_counts=self.allocator.available_counts(),
                reason=self._branch_pending_reason(branch_diag, default="branch_pool_shortage"),
                partial_render=view.partial_render(),
                actual_render=view.real_render(),
            )
            return False
        validation_reason = self._validate_external_branch_closure_state(branch_diag)
        if validation_reason is not None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=required_counts,
                available_counts=self.allocator.available_counts(),
                reason=str(validation_reason),
                partial_render=view.partial_render(),
                actual_render=view.real_render(),
            )
            return False
        branch_map, branch_id_map = self._build_branch_maps(view, branch_resolved)
        source_views = self._collect_source_views_from_branches([view], branch_resolved)
        validation_reason = self._invalid_product_reason(
            str(product_kind),
            list(view.main_labels or []),
            source_views,
            list(view.main_node_ids or []),
        )
        suppressed_support_node_ids = self._collect_suppressed_support_node_ids(
            source_views,
            branch_resolved,
        )
        if validation_reason is not None:
            return self._record_invalid_materialization_failure(
                snapshot=snapshot,
                donor_snapshot=donor_snapshot,
                branch_donors=branch_donors,
                product_kind=str(product_kind),
                source_views=source_views,
                labels=list(view.main_labels or []),
                node_ids=list(view.main_node_ids or []),
                branch_map=branch_map,
                branch_id_map=branch_id_map,
                suppressed_support_node_ids=suppressed_support_node_ids,
                validation_reason=str(validation_reason),
            )
        product = self._make_product(
            product_kind=str(product_kind),
            reason=str(reason),
            source_views=source_views,
            labels=list(view.main_labels or []),
            node_ids=list(view.main_node_ids or []),
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
            diagnostics={"origin_type": str(view.class_type), "canonical_labels": list(view.canonical_labels or []), **dict(branch_diag or {})},
        )
        if product is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed(self._collect_claimed_fragment_ids([view], branch_resolved))
        return True

    def _finalize_with_closure(
        self,
        view: AssemblyFragmentView,
        *,
        product_kind: str,
        closure_mode: str,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        branch_resolved, branch_required, branch_diag = self._resolve_branches(view, branch_donors)
        if branch_resolved is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=branch_required,
                available_counts=self.allocator.available_counts(),
                reason=self._branch_pending_reason(branch_diag, default="branch_pool_shortage"),
                partial_render=view.partial_render(),
            )
            return False
        validation_reason = self._validate_external_branch_closure_state(branch_diag)
        if validation_reason is not None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=branch_required,
                available_counts=self.allocator.available_counts(),
                reason=str(validation_reason),
                partial_render=view.partial_render(),
            )
            return False
        closure_payload, closure_required = self._build_main_closure(
            view,
            side="left" if bool(view.open_left) else "right",
            closure_mode=str(closure_mode),
        )
        if closure_payload is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            required_counts = dict(branch_required or {})
            for key, value in dict(closure_required or {}).items():
                required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=self._pending_kind_for_view(view),
                views=[view],
                required_counts=required_counts,
                available_counts=self.allocator.available_counts(),
                reason=f"{closure_mode}_pool_shortage",
                partial_render=view.partial_render(),
            )
            return False
        branch_map, branch_id_map = self._build_branch_maps(view, branch_resolved)
        labels = list(view.main_labels or [])
        node_ids = list(view.main_node_ids or [])
        closure_labels, closure_ids = closure_payload
        if bool(view.open_left):
            labels = list(closure_labels) + labels
            node_ids = list(closure_ids) + node_ids
        else:
            labels = labels + list(closure_labels)
            node_ids = node_ids + list(closure_ids)
        canonical_labels = list(view.canonical_labels or [])
        if str(closure_mode) == "F1":
            if bool(view.open_left):
                canonical_labels = ["11", "23"] + canonical_labels
            else:
                canonical_labels = canonical_labels + ["23", "11"]
        else:
            if bool(view.open_left):
                canonical_labels = ["23", "22"] + canonical_labels
            else:
                canonical_labels = canonical_labels + ["23", "22"]
        source_views = self._collect_source_views_from_branches([view], branch_resolved)
        validation_reason = self._invalid_product_reason(str(product_kind), labels, source_views, node_ids)
        suppressed_support_node_ids = self._collect_suppressed_support_node_ids(
            source_views,
            branch_resolved,
        )
        if validation_reason is not None:
            return self._record_invalid_materialization_failure(
                snapshot=snapshot,
                donor_snapshot=donor_snapshot,
                branch_donors=branch_donors,
                product_kind=str(product_kind),
                source_views=source_views,
                labels=labels,
                node_ids=node_ids,
                branch_map=branch_map,
                branch_id_map=branch_id_map,
                suppressed_support_node_ids=suppressed_support_node_ids,
                validation_reason=str(validation_reason),
            )
        product = self._make_product(
            product_kind=str(product_kind),
            reason=f"main_close:{closure_mode}",
            source_views=source_views,
            labels=labels,
            node_ids=node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
            diagnostics={"origin_type": str(view.class_type), "canonical_labels": canonical_labels, **dict(branch_diag or {})},
        )
        if product is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            return False
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed(self._collect_claimed_fragment_ids([view], branch_resolved))
        return True

    def _composition_path(
        self,
        candidate: AssemblyCompositionCandidate,
    ) -> Tuple[List[str], List[Optional[int]], List[str], List[int]]:
        left_labels, left_ids = candidate.left.oriented_mainline(
            reversed_view=bool(candidate.left_reversed),
        )
        right_labels, right_ids = candidate.right.oriented_mainline(
            reversed_view=bool(candidate.right_reversed),
        )
        left_endpoint = candidate.left.oriented_endpoint(
            "right",
            reversed_view=bool(candidate.left_reversed),
        )
        right_endpoint = candidate.right.oriented_endpoint(
            "left",
            reversed_view=bool(candidate.right_reversed),
        )
        bridge_labels, bridge_ids = _direct_join_bridge_tokens(
            left_bridge=(
                (str(left_endpoint.bridge_label), int(left_endpoint.bridge_node_id))
                if left_endpoint.bridge_node_id is not None
                else None
            ),
            right_bridge=(
                (str(right_endpoint.bridge_label), int(right_endpoint.bridge_node_id))
                if right_endpoint.bridge_node_id is not None
                else None
            ),
            left_view=candidate.left,
            right_view=candidate.right,
        )
        existing_ids = {
            int(value)
            for value in list(left_ids or []) + list(right_ids or [])
            if value is not None
        }
        retained_bridge = [
            (str(label), int(node_id))
            for label, node_id in zip(bridge_labels, bridge_ids)
            if int(node_id) not in existing_ids
        ]
        retained_labels = [label for label, _ in retained_bridge]
        retained_ids = [node_id for _, node_id in retained_bridge]
        return (
            list(left_labels) + retained_labels + list(right_labels),
            list(left_ids) + retained_ids + list(right_ids),
            retained_labels,
            retained_ids,
        )

    def _composition_candidate_score(
        self,
        candidate: AssemblyCompositionCandidate,
        labels: Sequence[str],
    ) -> Tuple[Any, ...]:
        unresolved_branches = sum(
            len(values or [])
            for view in (candidate.left, candidate.right)
            for values in view.branch_kinds_by_anchor().values()
        )
        branch_contracts = sum(
            len(values or [])
            for view in (candidate.left, candidate.right)
            for values in view.fixed_branch_contracts_by_anchor().values()
        )
        terminal_left_penalty = 0
        if str(candidate.product_kind) == "side_chain":
            terminal_left_penalty = 0 if str(labels[0] if labels else "") in TERMINAL_ENDPOINT_LABELS else 1
        return (
            int(unresolved_branches),
            int(branch_contracts),
            int(terminal_left_penalty),
            min(int(candidate.left.fragment_id), int(candidate.right.fragment_id)),
            max(int(candidate.left.fragment_id), int(candidate.right.fragment_id)),
            int(bool(candidate.left_reversed)),
            int(bool(candidate.right_reversed)),
        )

    def _build_composition_candidates(
        self,
        pairs: Sequence[Tuple[AssemblyFragmentView, AssemblyFragmentView]],
        *,
        product_kind: str,
        reason: str,
        allowed_join_roles: Set[str],
    ) -> List[AssemblyCompositionCandidate]:
        candidates: List[AssemblyCompositionCandidate] = []
        seen: Set[Tuple[int, int, bool, bool, str]] = set()
        for first, second in list(pairs or []):
            if int(first.fragment_id) == int(second.fragment_id):
                continue
            if set(int(value) for value in first.source_node_ids) & set(int(value) for value in second.source_node_ids):
                continue
            for left, right in ((first, second), (second, first)):
                for left_reversed in (False, True):
                    for right_reversed in (False, True):
                        key = (
                            int(left.fragment_id),
                            int(right.fragment_id),
                            bool(left_reversed),
                            bool(right_reversed),
                            str(product_kind),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        left_join = left.oriented_endpoint("right", reversed_view=bool(left_reversed))
                        right_join = right.oriented_endpoint("left", reversed_view=bool(right_reversed))
                        if not bool(left_join.is_open and right_join.is_open):
                            continue
                        if {str(left_join.role), str(right_join.role)} != set(allowed_join_roles):
                            continue
                        provisional = AssemblyCompositionCandidate(
                            left=left,
                            right=right,
                            left_reversed=bool(left_reversed),
                            right_reversed=bool(right_reversed),
                            product_kind=str(product_kind),
                            reason=str(reason),
                            score=(),
                        )
                        labels, node_ids, _, _ = self._composition_path(provisional)
                        if self._invalid_product_reason(str(product_kind), labels, [left, right], node_ids) is not None:
                            continue
                        candidates.append(AssemblyCompositionCandidate(
                            left=left,
                            right=right,
                            left_reversed=bool(left_reversed),
                            right_reversed=bool(right_reversed),
                            product_kind=str(product_kind),
                            reason=str(reason),
                            score=self._composition_candidate_score(provisional, labels),
                        ))
        return sorted(candidates, key=lambda item: tuple(item.score))

    def _record_composition_failure(
        self,
        candidate: AssemblyCompositionCandidate,
        *,
        validation_reason: str,
        required_counts: Optional[Dict[str, int]] = None,
        rolled_back_resource_ids: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        left_endpoint = candidate.left.oriented_endpoint(
            "right",
            reversed_view=bool(candidate.left_reversed),
        )
        right_endpoint = candidate.right.oriented_endpoint(
            "left",
            reversed_view=bool(candidate.right_reversed),
        )
        self.composition_attempt_failures.append({
            "attempted_product_kind": str(candidate.product_kind),
            "source_fragment_ids": [int(candidate.left.fragment_id), int(candidate.right.fragment_id)],
            "left_endpoint": left_endpoint.to_dict(),
            "right_endpoint": right_endpoint.to_dict(),
            "validation_reason": str(validation_reason),
            "attempted_resource_counts": {
                str(key): int(value)
                for key, value in dict(required_counts or {}).items()
                if int(value) > 0
            },
            "rolled_back_resource_ids": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(rolled_back_resource_ids or {}).items()
                if list(values or [])
            },
        })

    def _compose_two_views(
        self,
        candidate: AssemblyCompositionCandidate,
        *,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        left = candidate.left
        right = candidate.right
        reserved_fragment_ids = [int(left.fragment_id), int(right.fragment_id)]
        if not self._reserve_fragments(reserved_fragment_ids):
            return False
        snapshot = self.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        left_branch, left_required, left_diag = self._resolve_branches(left, branch_donors)
        if left_branch is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_composition_failure(
                candidate,
                validation_reason=self._branch_pending_reason(left_diag, default="left_branch_pool_shortage"),
                required_counts=left_required,
            )
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        left_validation_reason = self._validate_external_branch_closure_state(left_diag)
        if left_validation_reason is not None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_composition_failure(
                candidate,
                validation_reason=str(left_validation_reason),
                required_counts=left_required,
            )
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        right_branch, right_required, right_diag = self._resolve_branches(right, branch_donors)
        if right_branch is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            req = Counter(left_required)
            req.update(Counter(right_required))
            self._record_composition_failure(
                candidate,
                validation_reason=self._branch_pending_reason(right_diag, default="right_branch_pool_shortage"),
                required_counts={str(k): int(v) for k, v in req.items()},
            )
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        right_validation_reason = self._validate_external_branch_closure_state(right_diag)
        if right_validation_reason is not None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            req = Counter(left_required)
            req.update(Counter(right_required))
            self._record_composition_failure(
                candidate,
                validation_reason=str(right_validation_reason),
                required_counts={str(k): int(v) for k, v in req.items()},
            )
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        labels, node_ids, bridge_labels, _ = self._composition_path(candidate)
        left_labels, _ = left.oriented_mainline(reversed_view=bool(candidate.left_reversed))
        canonical_labels = ["11", "23", "11"] if str(candidate.product_kind) == "flex_chain" else ["11", "23", "22"]
        branch_map, branch_id_map = self._build_branch_maps(
            left,
            left_branch,
            reverse=bool(candidate.left_reversed),
        )
        right_branch_map, right_branch_id_map = self._build_branch_maps(
            right,
            right_branch,
            reverse=bool(candidate.right_reversed),
            anchor_offset=len(left_labels) + len(bridge_labels),
        )
        branch_map.update(right_branch_map)
        branch_id_map.update(right_branch_id_map)
        all_branch_results = list(left_branch or []) + list(right_branch or [])
        source_views = self._collect_source_views_from_branches([left, right], all_branch_results)
        validation_reason = self._invalid_product_reason(
            str(candidate.product_kind),
            labels,
            source_views,
            node_ids,
        )
        suppressed_support_node_ids = self._collect_suppressed_support_node_ids(
            source_views,
            all_branch_results,
        )
        if validation_reason is not None:
            claimed_ids = self.allocator.claimed_ids_since(snapshot)
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._record_composition_failure(
                candidate,
                validation_reason=str(validation_reason),
                rolled_back_resource_ids=claimed_ids,
            )
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        product = self._make_product(
            product_kind=str(candidate.product_kind),
            reason=str(candidate.reason),
            source_views=source_views,
            labels=labels,
            node_ids=node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
            diagnostics={
                "origin_type": f"{left.class_type}+{right.class_type}",
                "canonical_labels": canonical_labels,
                "source_fragment_classes": _source_fragment_classes([left, right]),
                "left_reversed": bool(candidate.left_reversed),
                "right_reversed": bool(candidate.right_reversed),
            },
        )
        if product is None:
            self.allocator.rollback(snapshot)
            branch_donors[:] = donor_snapshot
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        if not self._commit_product(
            product,
            allocator_snapshot=snapshot,
            branch_donors=branch_donors,
            donor_snapshot=donor_snapshot,
        ):
            self._release_reserved_fragments(reserved_fragment_ids)
            return False
        self._drop_pending_overlapping_product(product)
        self._mark_fragments_completed(self._collect_claimed_fragment_ids([left, right], all_branch_results))
        return True

    def _view_ring_allowed_skeletons(self, view: AssemblyFragmentView) -> Set[str]:
        contracts = {
            str(key): dict(value or {})
            for key, value in dict(dict(view.evidence or {}).get("ring_contracts", {}) or {}).items()
        }
        if contracts:
            out: Set[str] = set()
            for skeleton_kind, contract in contracts.items():
                if bool(contract.get("eligible", False)):
                    out.add(_skeleton_allowed_name(str(skeleton_kind)))
            if out:
                return out
        allowed = {
            str(item)
            for item in list(dict(view.evidence or {}).get("ring_allowed_skeletons", []) or [])
        }
        if allowed:
            return allowed
        return self._view_ring_eligibility(view)

    def _view_ring_eligibility(self, view: AssemblyFragmentView) -> Set[str]:
        contracts = {
            str(key): dict(value or {})
            for key, value in dict(dict(view.evidence or {}).get("ring_contracts", {}) or {}).items()
        }
        if contracts:
            out: Set[str] = set()
            for skeleton_kind, contract in contracts.items():
                if bool(contract.get("eligible", False)):
                    out.add(_skeleton_allowed_name(str(skeleton_kind)))
            if out:
                return out
        return {
            str(item)
            for item in list(dict(view.evidence or {}).get("ring_eligibility", []) or [])
        }

    def _is_like24_ring_fill_view(self, view: AssemblyFragmentView) -> bool:
        return bool(
            bool(getattr(view, "can_ring", False))
            and str(getattr(view, "family", "")) in {"like24", "like24_secondary", "like24_bridge"}
            and str(getattr(view, "preferred_product_type", "")) == "ring_candidate"
        )

    def _is_external_ring_view(self, view: AssemblyFragmentView) -> bool:
        return bool(
            bool(getattr(view, "can_ring", False))
            and not self._is_like24_ring_fill_view(view)
        )

    def _view_ring_slot_profiles(self, view: AssemblyFragmentView) -> Set[str]:
        return {
            str(item)
            for item in list(dict(view.evidence or {}).get("ring_slot_profiles", []) or [])
        }

    def _view_has_slot_profile(self, view: AssemblyFragmentView, *profiles: str) -> bool:
        if view is None:
            return False
        available = self._view_ring_slot_profiles(view)
        return any(str(profile) in available for profile in list(profiles or []))

    def _assemble_remaining_external_b_plus_b_flex(
        self,
        branch_donors: List[AssemblyFragmentView],
        *,
        target_count: int,
    ) -> bool:
        progress_made = False
        while int(self._current_flex_product_count()) < int(target_count) and self._can_create_more_flex():
            b_views = self._available_flex_reconcile_views(
                require_roles=["b_like_open"],
                allow_ring_capable=True,
                require_materializable=True,
                require_generic_join=True,
            )
            if len(b_views) < 2:
                break
            candidates = self._build_composition_candidates(
                [
                    (b_views[left_index], b_views[right_index])
                    for left_index in range(len(b_views))
                    for right_index in range(left_index + 1, len(b_views))
                ],
                product_kind="flex_chain",
                reason="B_plus_B",
                allowed_join_roles={"b_like_open_end"},
            )
            created = False
            for candidate in candidates:
                if self._compose_two_views(candidate, branch_donors=branch_donors):
                    created = True
                    break
            if not created:
                break
            progress_made = True
        return progress_made

    def _assemble_remaining_external_b_plus_e_side(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        progress_made = False
        while True:
            b_views = self._available_remaining_b_open_views(require_generic_join=True)
            e_views = self._available_remaining_side_views(require_generic_join=True)
            if not b_views or not e_views:
                break
            candidates = self._build_composition_candidates(
                [(b_view, e_view) for b_view in b_views for e_view in e_views],
                product_kind="side_chain",
                reason="B_plus_E_or_G",
                allowed_join_roles={"b_like_open_end", "e_like_open_end"},
            )
            created = False
            for candidate in candidates:
                if self._compose_two_views(candidate, branch_donors=branch_donors):
                    created = True
                    break
            if not created:
                break
            progress_made = True
        return progress_made

    def _assemble_remaining_external_d_double_flex(
        self,
        branch_donors: List[AssemblyFragmentView],
        *,
        target_count: int,
    ) -> bool:
        progress_made = False
        while int(self._current_flex_product_count()) < int(target_count) and self._can_create_more_flex():
            cycle_progress = False
            for view in self._available_flex_reconcile_views(
                require_roles=["d_like_open", "double_end_flex_source"],
                allow_ring_capable=True,
                require_materializable=True,
                require_generic_join=False,
            ):
                if self._finalize_with_two_end_closures(
                    view,
                    product_kind="flex_chain",
                    left_mode="F1",
                    right_mode="F1",
                    branch_donors=branch_donors,
                ):
                    cycle_progress = True
                    break
            if not cycle_progress:
                break
            progress_made = True
        return progress_made

    def _assemble_remaining_external_single_b_closures_for_flex(
        self,
        branch_donors: List[AssemblyFragmentView],
        *,
        target_count: int,
    ) -> bool:
        progress_made = False
        while int(self._current_flex_product_count()) < int(target_count) and self._can_create_more_flex():
            cycle_progress = False
            for view in self._available_flex_reconcile_views(
                require_roles=["b_like_open", "single_end_flex_source"],
                allow_ring_capable=True,
                require_materializable=True,
                require_generic_join=False,
            ):
                if self._finalize_with_closure(
                    view,
                    product_kind="flex_chain",
                    closure_mode="F1",
                    branch_donors=branch_donors,
                ):
                    cycle_progress = True
                    break
            if not cycle_progress:
                break
            progress_made = True
        return progress_made

    def _assemble_remaining_external_single_e_closures_for_side(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        progress_made = False
        for view in self._available_remaining_side_views(require_generic_join=True):
            if self._finalize_with_closure(
                view,
                product_kind="side_chain",
                closure_mode="F1",
                branch_donors=branch_donors,
            ):
                progress_made = True
        return progress_made

    def _flex_bounds_enforced(self) -> bool:
        return bool(dict(self.flex_bounds or {}).get("enforced", False))

    def _flex_limit(self, name: str) -> int:
        return max(0, _safe_int(dict(self.flex_bounds or {}).get(str(name), 0), 0))

    def _current_flex_product_count(self) -> int:
        return int(sum(
            1
            for item in list(self.products or [])
            if str(getattr(item, "product_kind", "")) in {"flex_chain", "residual_flex_chain"}
        ))

    def _mark_unselected_flex_pending_as_structural(self) -> None:
        target_count = int(dict(self.flex_bounds or {}).get("target", self._flex_limit("min")) or 0)
        final_flex_count = int(self._current_flex_product_count())
        if int(final_flex_count) < int(target_count):
            return
        blocked_fragment_ids = {
            int(view.fragment_id)
            for view in self._available_views(lambda item: bool(item.can_flex) and not bool(item.can_ring))
            if int(view.fragment_id) not in set(self.selected_flex_candidate_ids)
        }
        if not blocked_fragment_ids:
            return
        reason = "supplemental_flex_candidate_not_selected"
        for pending in list(self.pending_units or []):
            fragment_ids = {
                int(value)
                for value in list(getattr(pending, "source_fragment_ids", []) or [])
            }
            if not fragment_ids.intersection(blocked_fragment_ids):
                continue
            if str(getattr(pending, "target_role", "") or "") != "flex_chain":
                continue
            pending.contract_kind = "structural_blocker"
            pending.required_counts = {}
            pending.reason = str(reason)
            pending.attempt_reasons = sorted({
                str(value)
                for value in list(getattr(pending, "attempt_reasons", []) or []) + [str(reason)]
                if str(value)
            })
            pending.infeasibility_proof = {
                **dict(getattr(pending, "infeasibility_proof", {}) or {}),
                "reason": str(reason),
                "required_counts": {},
                "missing_counts": {},
                "flex_target": int(target_count),
                "final_flex_count": int(final_flex_count),
            }
        for contract in list(self.pending_contracts or []):
            fragment_ids = {
                int(value)
                for value in list(dict(contract or {}).get("source_fragment_ids", []) or [])
            }
            if not fragment_ids.intersection(blocked_fragment_ids):
                continue
            if str(dict(contract or {}).get("target_role", "") or "") != "flex_chain":
                continue
            contract["contract_kind"] = "structural_blocker"
            contract["required_counts"] = {}
            contract["reason"] = str(reason)
            contract["blocker_kind"] = str(reason)
            contract["diagnostics"] = {
                **dict(dict(contract or {}).get("diagnostics", {}) or {}),
                "selected_for_target": False,
                "flex_target": int(target_count),
                "final_flex_count": int(final_flex_count),
            }

    def _can_create_more_flex(self) -> bool:
        if not self._flex_bounds_enforced():
            return True
        return int(self._current_flex_product_count()) < int(self._flex_limit("max"))

    def _flex_exceeds_max(self) -> bool:
        if not self._flex_bounds_enforced():
            return False
        return int(self._current_flex_product_count()) > int(self._flex_limit("max"))

    def _flex_below_min(self) -> bool:
        if not self._flex_bounds_enforced():
            return False
        return int(self._current_flex_product_count()) < int(self._flex_limit("min"))

    def _close_explicit_flex_branches_if_needed(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        progress_made = False
        views = sorted(
            self._available_views(lambda item: self._is_explicit_mainline_view(item)),
            key=lambda item: (
                0 if str(self._pending_kind_for_view(item)) == "explicit_flex" else 1,
                int(item.fragment_id),
            ),
        )
        for view in views:
            product_kind = "flex_chain" if str(self._pending_kind_for_view(view)) == "explicit_flex" else "side_chain"
            if self._finalize_single_view_product(
                view,
                product_kind=product_kind,
                reason="explicit_branch_closure",
                branch_donors=branch_donors,
            ):
                progress_made = True
                continue
            break
        return progress_made

    def _assemble_selected_supplemental_flex(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        if not self.selected_flex_candidate_ids:
            return False
        target_count = int(dict(self.flex_bounds or {}).get("target", self._flex_limit("min")) or 0)
        progress_made = self._assemble_remaining_external_b_plus_b_flex(
            branch_donors,
            target_count=target_count,
        )
        progress_made = bool(
            self._assemble_remaining_external_d_double_flex(
                branch_donors,
                target_count=target_count,
            ) or progress_made
        )
        progress_made = bool(
            self._assemble_remaining_external_single_b_closures_for_flex(
                branch_donors,
                target_count=target_count,
            ) or progress_made
        )
        return progress_made

    def _has_unresolved_selected_flex_candidates(self) -> bool:
        if not self._flex_below_min():
            return False
        return any(
            self._fragment_available(int(view.fragment_id))
            and int(view.fragment_id) in self.selected_flex_candidate_ids
            and bool(view.can_flex)
            and _is_chain_materializable_view(view)
            for view in list(self.views or [])
        )

    def _flex_range_violation_reason(self) -> Optional[str]:
        blocked_reason = str(dict(self.flex_audit or {}).get("blocked_reason", "") or "")
        if str(blocked_reason) == "flex_window_invalid":
            return "flex_window_invalid"
        if self._flex_exceeds_max():
            return "flex_range_exceeded"
        return None

    def _assemble_priority_rings(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        ring_views = self._available_views(lambda item: bool(getattr(item, "can_ring", False)))
        if not ring_views:
            return False
        like24_fill_views = self._available_views(lambda item: self._is_like24_ring_fill_view(item))
        return self._assemble_all_rings(
            branch_donors,
            ring_views=ring_views,
            like24_fill_views=like24_fill_views,
            candidate_predicate=lambda item: bool(getattr(item, "can_ring", False)),
        )

    def _assemble_all_rings(
        self,
        branch_donors: List[AssemblyFragmentView],
        *,
        ring_views: Optional[Sequence[AssemblyFragmentView]] = None,
        like24_fill_views: Optional[Sequence[AssemblyFragmentView]] = None,
        candidate_predicate: Optional[Any] = None,
        ring_order: Optional[Sequence[str]] = None,
    ) -> bool:
        resolved_ring_order: List[str] = list(ring_order or [
            "fatty_ring_side",
            "fatty_ring_upper_lower",
            "fatty_ring_fused",
        ])
        predicate = candidate_predicate or self._is_external_ring_view
        progress_made = False
        while True:
            cycle_progress = False
            for product_kind in list(resolved_ring_order):
                candidate_ring_views = [
                    view for view in list(ring_views or self._available_views(lambda item: True))
                    if self._fragment_available(int(view.fragment_id))
                    and bool(predicate(view))
                ]
                if not candidate_ring_views:
                    continue
                product_count_before = len(self.products)
                created = self._assemble_ring(
                    product_kind=str(product_kind),
                    preferred_views=candidate_ring_views,
                    branch_donors=branch_donors,
                    like24_fill_views=like24_fill_views,
                )
                if created and len(self.products) > product_count_before:
                    cycle_progress = True
                    progress_made = True
                    break
            if not cycle_progress:
                break
        return progress_made

    def _infer_ring_pending_attempt(
        self,
        view: AssemblyFragmentView,
        candidate_views: Sequence[AssemblyFragmentView],
    ) -> Dict[str, Any]:
        del candidate_views
        prior = dict(self.ring_pending_attempts.get(int(view.fragment_id), {}) or {})
        missing_slots = [
            str(item)
            for item in list(prior.get("missing_slots", []) or [])
            if str(item)
        ]
        if missing_slots:
            return {"missing_slots": sorted(set(missing_slots))}
        reason_list = [
            str(item)
            for item in list(self.ring_pending_reasons.get(int(view.fragment_id), []) or [])
            if str(item)
        ]
        if reason_list:
            return {"missing_slots": sorted(set(reason_list))}
        return {"missing_slots": []}

    def _assemble_remaining_flex(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        # This stage runs only after explicit branch closure and real like24 ring
        # assembly. It must not be treated as pre-assembly flex reconciliation.
        target_count = int(
            dict(self.flex_bounds or {}).get("target", self._flex_limit("max"))
            if self._flex_bounds_enforced()
            else 10**9
        )
        if self._assemble_remaining_external_b_plus_b_flex(
            branch_donors,
            target_count=target_count,
        ):
            return True
        if self._assemble_remaining_external_d_double_flex(branch_donors, target_count=target_count):
            return True
        return self._assemble_remaining_external_single_b_closures_for_flex(
            branch_donors,
            target_count=target_count,
        )

    def _assemble_remaining_side(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        if self._assemble_remaining_external_b_plus_e_side(branch_donors):
            return True
        return self._assemble_remaining_external_single_e_closures_for_side(branch_donors)

    def _assemble_remaining_su25(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        for view in self._available_views(lambda item: str(item.class_type) in {"25_AB", "25_CD"}):
            closure_mode = "F2" if str(view.class_type) == "25_CD" else "F1"
            if self._finalize_with_closure(
                view,
                product_kind="side_chain",
                closure_mode=closure_mode,
                branch_donors=branch_donors,
            ):
                return True
        return False

    def _assemble_remaining_flex_and_side(
        self,
        branch_donors: List[AssemblyFragmentView],
    ) -> bool:
        progress_made = self._assemble_remaining_flex(branch_donors)
        progress_made = bool(self._assemble_remaining_external_b_plus_e_side(branch_donors) or progress_made)
        progress_made = bool(self._assemble_remaining_su25(branch_donors) or progress_made)
        progress_made = bool(self._assemble_remaining_external_single_e_closures_for_side(branch_donors) or progress_made)
        return progress_made

    def _remaining_pending_kind_and_reason(
        self,
        view: AssemblyFragmentView,
        *,
        flex_violation_reason: Optional[str],
    ) -> Tuple[str, str]:
        if flex_violation_reason is not None:
            return str(self._pending_kind_for_view(view)), str(flex_violation_reason)
        profile = self._view_external_pending_profile(view)
        profile_pending_kind = str(profile.get("external_pending_kind", "") or "")
        profile_reason = str(profile.get("external_pending_reason", "") or "")
        if profile_pending_kind and profile_reason:
            return str(profile_pending_kind), str(profile_reason)
        if str(view.class_type) in {"25_AB", "25_CD"}:
            return "pending_su25", "fixed_connector_unresolved"
        if self._view_has_flex_reconcile_role(view, "b_like_open", "d_like_open"):
            if self._view_has_flex_reconcile_role(view, "d_like_open"):
                return "d_like_open", "remaining_external_halfclosed_open"
            return "b_like_open", "remaining_external_halfclosed_open"
        if self._is_side_pending_view(view):
            if str(getattr(view, "preferred_product_type", "")) == "side_chain":
                return "explicit_side", "remaining_external_side_open"
            return "e_like_open", "remaining_external_side_open"
        if str(getattr(view, "preferred_product_type", "")) == "flex_chain":
            return "explicit_flex", "remaining_external_halfclosed_open"
        if str(getattr(view, "preferred_product_type", "")) == "ring_candidate":
            return "ring_only_like24", "remaining_external_halfclosed_open"
        return "other", "remaining_external_halfclosed_open"

    def _record_remaining_external_pending(
        self,
        *,
        flex_violation_reason: Optional[str],
    ) -> None:
        for view in self._available_views(lambda item: True):
            if bool(view.can_ring):
                pending_kind = "pending_ring_candidate"
                pending_reason = str(
                    flex_violation_reason
                    or "remaining_external_ring_candidate"
                )
            else:
                pending_kind, pending_reason = self._remaining_pending_kind_and_reason(
                    view,
                    flex_violation_reason=flex_violation_reason,
                )
            if any(
                int(getattr(existing, "unit_id", 0)) == int(view.fragment_id)
                and str(getattr(existing, "pending_kind", "")) == str(pending_kind)
                for existing in list(self.pending_units or [])
            ):
                continue
            partial = view.partial_render()
            self._record_pending(
                unit_id=int(view.fragment_id),
                pending_kind=str(pending_kind),
                views=[view],
                required_counts={},
                available_counts=self.allocator.available_counts(),
                reason=str(pending_reason),
                partial_render=partial,
                actual_render=view.real_render(),
            )

    def assemble(self) -> TopologyAssemblyResult:
        if bool(dict(self.flex_audit or {}).get("enforced", False)):
            scenario = str(dict(self.flex_audit or {}).get("scenario", "") or "")
            blocked_reason = str(dict(self.flex_audit or {}).get("blocked_reason", "") or "")
            if scenario == "above_range" or blocked_reason == "flex_window_invalid":
                return _flex_preassembly_gate_result(
                    classification=self.classification,
                    flex_audit=dict(self.flex_audit or {}),
                    flex_bounds=dict(self.flex_bounds or {}),
                )
        branch_donors = self._available_branch_donors()
        external_subphases: List[str] = []
        ring_stage_attempted = False
        remaining_stage_attempted = False
        external_subphases.append("explicit_branch_closure")
        self._close_explicit_flex_branches_if_needed(branch_donors)
        if str(dict(self.flex_audit or {}).get("scenario", "") or "") == "below_range":
            external_subphases.append("supplemental_flex_closure")
            self._assemble_selected_supplemental_flex(branch_donors)
        flex_violation_reason = self._flex_range_violation_reason()
        explicit_branch_stage_ok = not self._has_pending_contracts_for_subphase("explicit_branch_closure")
        unresolved_flex_candidates = self._has_unresolved_selected_flex_candidates()
        if flex_violation_reason is None:
            external_subphases.append("real_like24_ring_assembly")
            ring_stage_attempted = True
            self._assemble_priority_rings(branch_donors)
        ring_stage_ok = not self._has_pending_contracts_for_subphase("real_like24_ring_assembly")
        if (
            flex_violation_reason is None
        ):
            external_subphases.append("remaining_external_halfclosed_closure")
            remaining_stage_attempted = True
            self._assemble_remaining_flex_and_side(branch_donors)
        if ring_stage_attempted:
            ring_candidate_views = self._available_views(lambda item: bool(getattr(item, "can_ring", False)))
            for view in ring_candidate_views:
                reason_list = list(self.ring_pending_reasons.get(int(view.fragment_id), []) or [])
                pending_attempt = dict(self.ring_pending_attempts.get(int(view.fragment_id), {}) or {})
                pending_required_counts = dict(self.ring_pending_required_counts.get(int(view.fragment_id), {}) or {})
                inferred_attempt = self._infer_ring_pending_attempt(view, ring_candidate_views)
                inferred_missing = list(inferred_attempt.get("missing_slots", []) or [])
                closure_reasons = [
                    str(reason)
                    for reason in list(reason_list or [])
                    if self._is_external_branch_closure_reason(str(reason))
                ]
                if not pending_attempt:
                    pending_attempt = dict(inferred_attempt or {})
                else:
                    existing_missing = list(pending_attempt.get("missing_slots", []) or [])
                    pending_attempt["missing_slots"] = sorted({
                        str(item)
                        for item in existing_missing + inferred_missing
                    })
                all_missing_slots = list(pending_attempt.get("missing_slots", []) or reason_list or [])
                if flex_violation_reason is not None:
                    pending_reason = str(flex_violation_reason)
                elif closure_reasons:
                    pending_reason = str(closure_reasons[0])
                elif inferred_missing:
                    pending_reason = str(inferred_missing[0])
                else:
                    pending_reason = str(all_missing_slots[0]) if all_missing_slots else "ring_partner_missing"
                self._record_pending(
                    unit_id=int(view.fragment_id),
                    pending_kind="pending_ring_candidate",
                    views=[view],
                    required_counts=pending_required_counts,
                    available_counts=self.allocator.available_counts(),
                    reason=pending_reason,
                    partial_render=view.partial_render(),
                    actual_render=view.real_render(),
                )

        residual_products: List[TopologyAssemblyProduct] = []
        residual_phase_attempted = False
        protected_resource_counts: Dict[str, int] = {}
        self.allocator.set_protected_counts({})
        if flex_violation_reason is None and not self._flex_exceeds_max():
            residual_phase_attempted = True
            products_before = len(self.products)
            assemble_residual_resource_chains(
                allocator=self.allocator,
                products=self.products,
                current_flex_count=int(self._current_flex_product_count()),
                flex_lower=int(self._flex_limit("min")),
                flex_target=int(dict(self.flex_bounds or {}).get("target", self._flex_limit("min")) or 0),
                flex_upper=int(self._flex_limit("max")),
                commit_product=lambda product: self._commit_product(product),
            )
            residual_products = list(self.products[products_before:])

        if residual_phase_attempted:
            final_available_counts = self.allocator.available_counts()
            for pending in list(self.pending_units or []):
                pending.available_counts = {
                    str(key): int(value)
                    for key, value in dict(final_available_counts or {}).items()
                }
                missing_counts = {
                    str(key): max(
                        0,
                        int(dict(pending.required_counts or {}).get(str(key), 0) or 0)
                        - int(dict(final_available_counts or {}).get(str(key), 0) or 0),
                    )
                    for key in dict(pending.required_counts or {})
                }
                pending.infeasibility_proof = {
                    **dict(pending.infeasibility_proof or {}),
                    "available_counts": dict(pending.available_counts),
                    "missing_counts": {
                        str(key): int(value)
                        for key, value in missing_counts.items()
                        if int(value) > 0
                    },
                    "resource_snapshot": "post_residual",
                }

        # Record every still-available fragment only after all materialization
        # stages have had a chance to consume it.  Pending registration must
        # never prevent raw/residual assembly from running, but it must also
        # not depend on those stages having been entered.
        if self._available_views(lambda item: True):
            self._record_remaining_external_pending(
                flex_violation_reason=flex_violation_reason,
            )

        self._mark_unselected_flex_pending_as_structural()
        self._consolidate_pending_owners()

        completed_product_node_ids = {
            int(node_id)
            for product in list(self.products or [])
            for node_id in list(getattr(product, "source_node_ids", []) or [])
        }
        completed_product_fragment_ids = {
            int(fragment_id)
            for product in list(self.products or [])
            for fragment_id in list(getattr(product, "source_fragment_ids", []) or [])
        }
        self._invalidate_pending_identity(
            fragment_ids=completed_product_fragment_ids,
            claimed_node_ids=completed_product_node_ids,
        )
        ledger = self.allocator.build_ledger({str(k): int(v) for k, v in self.shortage_counts.items()})
        explicit_flex_count = int(self.explicit_flex_count)
        final_flex_count = int(self._current_flex_product_count())
        flex_target_plan = {
            "flex_lower": int(self._flex_limit("min")),
            "flex_upper": int(self._flex_limit("max")),
            "flex_target": int(dict(self.flex_bounds or {}).get("target", self._flex_limit("min")) or 0),
            "explicit_flex_count": int(explicit_flex_count),
            "minimum_gap": max(0, int(self._flex_limit("min")) - int(explicit_flex_count)),
            "preferred_gap": max(0, int(dict(self.flex_bounds or {}).get("target", 0) or 0) - int(explicit_flex_count)),
            "maximum_capacity": max(0, int(self._flex_limit("max")) - int(explicit_flex_count)),
            "external_candidate_capacity": int(len(self.selected_flex_candidate_ids)),
            "external_planned_addition": int(len(self.selected_flex_candidate_ids)),
            "external_materialized_addition": max(0, int(final_flex_count) - int(explicit_flex_count)),
            "external_capacity_exhausted": bool(dict(self.flex_audit or {}).get("external_capacity_exhausted", False)),
            "remaining_minimum_gap": max(0, int(self._flex_limit("min")) - int(final_flex_count)),
            "remaining_preferred_gap": max(0, int(dict(self.flex_bounds or {}).get("target", 0) or 0) - int(final_flex_count)),
        }
        flex_above_max = bool(self._flex_exceeds_max())
        pending_contracts = [
            dict(item or {})
            for item in list(self.pending_contracts or [])
            if dict(item or {})
        ]
        has_external_pending = bool(self.pending_units)
        has_shortage_contract = any(
            int(value) > 0
            for contract in list(pending_contracts or [])
            for value in dict(contract.get("required_counts", {}) or {}).values()
        ) and any(int(value) > 0 for value in dict(self.shortage_counts or {}).values())
        has_branch_closure_contract = any(
            self._is_external_branch_closure_reason(str(contract.get("reason", "") or ""))
            for contract in list(pending_contracts or [])
        )
        ledger = self._reconcile_resource_ownership(ledger)
        residual_consumed_ids = {
            int(node_id)
            for product in list(residual_products or [])
            for node_id in list(
                getattr(product, "materialized_node_ids", [])
                or getattr(product, "source_node_ids", [])
                or []
            )
        }
        ledger.residual_consumed_counts = {
            str(key): int(len(set(values or []) & residual_consumed_ids))
            for key, values in dict(ledger.consumed_ids_by_su or {}).items()
        }
        ledger.topology_consumed_counts = {
            str(key): max(
                0,
                int(ledger.consumed_counts.get(str(key), 0) or 0)
                - int(ledger.residual_consumed_counts.get(str(key), 0) or 0),
            )
            for key in set(ledger.initial_counts) | set(ledger.consumed_counts)
        }
        ledger.snapshot_stage = "post_residual_final"
        remaining_pool_counts = dict(ledger.allocator_remaining_counts or ledger.remaining_counts or {})
        residual_convertible_total = sum(
            int(remaining_pool_counts.get(str(su_type), 0) or 0)
            for su_type in (11, 22, 23, 24)
        )
        if str(flex_violation_reason) == "flex_range_below_minimum":
            result_reason = "external_flex_range_below_minimum"
        elif flex_violation_reason is not None or flex_above_max:
            result_reason = "external_flex_range_exceeded"
        elif not has_external_pending and residual_convertible_total > 0:
            result_reason = "residual_resource_self_closure_pending"
        elif not self.pending_units:
            result_reason = "topology_assembly_complete"
        elif has_shortage_contract:
            result_reason = "external_pool_shortage_pending"
        elif has_branch_closure_contract:
            result_reason = "external_structural_closure_pending"
        elif any(str(unit.pending_kind) == "pending_ring_candidate" for unit in list(self.pending_units or [])):
            result_reason = "external_ring_candidates_pending"
        else:
            result_reason = "external_structural_closure_pending"
        if flex_violation_reason is not None or flex_above_max:
            stage3_phase = "flex_range_reconciliation"
        elif has_external_pending:
            stage3_phase = "external_structural_closure"
        elif residual_convertible_total > 0:
            stage3_phase = "residual_resource_self_closure"
        else:
            stage3_phase = "complete"
        completed_phases = [
            "flex_range_reconciliation",
            "explicit_branch_closure",
            "real_like24_ring_assembly",
            "remaining_external_halfclosed_closure",
        ]
        if residual_phase_attempted and stage3_phase == "complete":
            completed_phases.append("residual_resource_self_closure")
        terminal_resource_counts = {
            str(su_type): int(remaining_pool_counts.get(str(su_type), 0) or 0)
            for su_type in (12, 13)
            if int(remaining_pool_counts.get(str(su_type), 0) or 0) > 0
        }
        terminal_unresolved_counts = {
            str(su_type): int(remaining_pool_counts.get(str(su_type), 0) or 0)
            for su_type in (11, 22, 23, 24)
            if int(remaining_pool_counts.get(str(su_type), 0) or 0) > 0
        }
        diagnostics = {
            "explicit_flex_count": int(explicit_flex_count),
            "total_shortage": {str(k): int(v) for k, v in self.shortage_counts.items()},
            "stage3_phase": stage3_phase,
            "external_subphases": list(external_subphases),
            "active_external_subphase": (
                list(external_subphases or [])[-1]
                if list(external_subphases or []) and stage3_phase == "external_structural_closure"
                else ("flex_range_reconciliation" if stage3_phase == "flex_range_reconciliation" else "")
            ),
            "flex_audit": dict(self.flex_audit or {}),
            "pending_contracts": list(pending_contracts),
            "external_subphase_status": self._external_subphase_status_snapshot(),
            "residual_closure_product_count": int(len(residual_products)),
            "completed_phases": list(completed_phases),
            "protected_resource_counts": dict(protected_resource_counts),
            "claimable_counts": dict(ledger.convertible_counts or {}),
            "allocator_remaining_counts": dict(ledger.allocator_remaining_counts or {}),
            "diagnostic_remaining_counts": dict(ledger.diagnostic_remaining_counts or {}),
            "unusable_counts": dict(ledger.unusable_counts or {}),
            "ledger_invariant_errors": list(ledger.invariant_errors or []),
            "remaining_minimum_flex_gap": int(flex_target_plan["remaining_minimum_gap"]),
            "remaining_preferred_flex_gap": int(flex_target_plan["remaining_preferred_gap"]),
            "product_ownership_conflicts": [
                dict(item or {}) for item in list(self.product_ownership_conflicts or [])
            ],
            "composition_attempt_failures": [
                dict(item or {}) for item in list(self.composition_attempt_failures or [])
            ],
            "contract_definition_conflicts": [
                dict(item or {}) for item in list(self.contract_definition_conflicts or [])
            ],
        }
        if self.invalid_topologies:
            diagnostics["bad_topologies"] = [dict(item or {}) for item in list(self.invalid_topologies or [])]
        support_only_node_ids = [
            int(item.get("node_id", item.get("global_id", -1)))
            for item in list(getattr(self.classification, "support_units_only", []) or [])
            if int(item.get("node_id", item.get("global_id", -1))) >= 0
        ]
        ownership = build_ownership_ledger(
            self.nodes,
            ignore_stage3_node_owners=True,
            completed_products=[
                product.to_dict() if hasattr(product, "to_dict") else dict(product or {})
                for product in list(self.products or [])
            ],
            pending_topologies=[
                pending.to_ownership_dict()
                if hasattr(pending, "to_ownership_dict")
                else (pending.to_dict() if hasattr(pending, "to_dict") else dict(pending or {}))
                for pending in list(self.pending_units or [])
            ],
            support_only_node_ids=support_only_node_ids,
        )
        for atomic_group in list(getattr(self.classification, "atomic_groups", []) or []):
            group = dict(atomic_group or {})
            group_node_ids = [int(value) for value in list(group.get("node_ids", []) or [])]
            group_owners = [ownership.by_node_id.get(int(node_id)) for node_id in group_node_ids]
            if not group_node_ids or any(owner is None for owner in group_owners):
                ownership.conflicts.append({
                    "reason": "atomic_group_node_missing",
                    "atomic_group_id": int(group.get("group_id", -1)),
                    "node_ids": group_node_ids,
                })
                continue
            identities = {
                (str(owner.state), owner.owner_topology_id, owner.owner_fragment_id)
                for owner in group_owners
                if owner is not None
            }
            if len(identities) != 1:
                ownership.conflicts.append({
                    "reason": "atomic_group_identity_split",
                    "atomic_group_id": int(group.get("group_id", -1)),
                    "kind": str(group.get("kind", "")),
                    "node_ids": group_node_ids,
                    "identities": [
                        {
                            "node_id": int(owner.node_id),
                            "state": str(owner.state),
                            "owner_topology_id": owner.owner_topology_id,
                            "owner_fragment_id": owner.owner_fragment_id,
                        }
                        for owner in group_owners
                        if owner is not None
                    ],
                })
        ownership_view = {
            "completed_topology_node_ids": sorted(
                int(node_id) for node_id, item in ownership.by_node_id.items()
                if item.state == "completed_topology"
            ),
            "pending_topology_node_ids": sorted(
                int(node_id) for node_id, item in ownership.by_node_id.items()
                if item.state == "pending_topology"
            ),
            "unassigned_nonresource_node_ids": sorted(
                int(node_id) for node_id, item in ownership.by_node_id.items()
                if item.state == "unassigned_nonresource"
            ),
            "support_only_node_ids": sorted(
                int(node_id) for node_id, item in ownership.by_node_id.items()
                if item.state == "support_only"
            ),
            "remaining_resource_ids_by_su": {
                str(key): list(values)
                for key, values in ownership.ids_by_su().items()
            },
            "conflicts": list(ownership.conflicts or []),
        }
        ownership_remaining_ids = {
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(ownership_view["remaining_resource_ids_by_su"] or {}).items()
        }
        ledger.allocator_remaining_ids_by_su = deepcopy(ownership_remaining_ids)
        ledger.remaining_ids_by_su = deepcopy(ownership_remaining_ids)
        ledger.convertible_ids_by_su = deepcopy(ownership_remaining_ids)
        ledger.allocator_remaining_counts = {
            str(key): int(len(values)) for key, values in ownership_remaining_ids.items()
        }
        ledger.remaining_counts = dict(ledger.allocator_remaining_counts)
        ledger.convertible_counts = dict(ledger.allocator_remaining_counts)
        ledger.protected_counts = {str(key): 0 for key in ownership_remaining_ids}
        ledger.initial_ids_by_su = {
            str(key): sorted(set(int(value) for value in list(values or [])))
            for key, values in dict(getattr(self.allocator, "initial_ids_by_su", {}) or {}).items()
        }
        ledger.consumed_ids_by_su = {
            str(key): sorted(
                set(ledger.initial_ids_by_su.get(str(key), []) or [])
                - set(ownership_remaining_ids.get(str(key), []) or [])
            )
            for key in ledger.initial_ids_by_su
        }
        ledger.initial_counts = {
            str(key): int(len(values)) for key, values in ledger.initial_ids_by_su.items()
        }
        ledger.consumed_counts = {
            str(key): int(len(values)) for key, values in ledger.consumed_ids_by_su.items()
        }
        residual_consumed_ids = {
            int(node_id)
            for product in list(residual_products or [])
            for node_id in list(
                getattr(product, "materialized_node_ids", [])
                or getattr(product, "source_node_ids", [])
                or []
            )
        }
        ledger.residual_consumed_counts = {
            str(key): int(len(set(values or []) & residual_consumed_ids))
            for key, values in dict(ledger.consumed_ids_by_su or {}).items()
        }
        ledger.topology_consumed_counts = {
            str(key): max(
                0,
                int(ledger.consumed_counts.get(str(key), 0) or 0)
                - int(ledger.residual_consumed_counts.get(str(key), 0) or 0),
            )
            for key in set(ledger.initial_counts) | set(ledger.consumed_counts)
        }
        ledger.snapshot_stage = "final_ownership_reconciled"
        diagnostics["ownership_ledger"] = dict(ownership_view)
        completed_node_ids = set(int(x) for x in ownership_view["completed_topology_node_ids"])
        pending_node_ids = set(int(x) for x in ownership_view["pending_topology_node_ids"])
        support_node_ids = set(int(x) for x in ownership_view["support_only_node_ids"])
        resource_node_ids = {
            int(node_id)
            for values in ownership_view["remaining_resource_ids_by_su"].values()
            for node_id in list(values or [])
        }
        unassigned_node_ids = set(int(x) for x in ownership_view["unassigned_nonresource_node_ids"])
        diagnostics["node_coverage"] = {
            "total_node_count": int(len(self.nodes_by_id)),
            "completed_node_count": int(len(completed_node_ids)),
            "pending_node_count": int(len(pending_node_ids)),
            "remaining_resource_node_count": int(len(resource_node_ids)),
            "unassigned_nonresource_node_count": int(len(unassigned_node_ids)),
            "support_only_node_count": int(len(support_node_ids)),
            "covered_node_count": int(len(completed_node_ids | pending_node_ids | support_node_ids | resource_node_ids)),
            "uncovered_node_ids": sorted(
                set(self.nodes_by_id) - completed_node_ids - pending_node_ids - support_node_ids - resource_node_ids - unassigned_node_ids
            ),
            "unassigned_nonresource_node_ids": sorted(unassigned_node_ids),
            "duplicate_node_ids": [],
        }
        ownership_residual_total = sum(
            int(ledger.remaining_counts.get(str(su_type), 0) or 0)
            for su_type in (11, 22, 23, 24)
        )
        residual_convertible_total = int(ownership_residual_total)
        if (
            stage3_phase == "complete"
            and int(ownership_residual_total) > 0
        ):
            stage3_phase = "residual_resource_self_closure"
            result_reason = "residual_resource_self_closure_pending"
            diagnostics["stage3_phase"] = str(stage3_phase)
            diagnostics["completed_phases"] = [
                str(value)
                for value in list(completed_phases or [])
                if str(value) != "residual_resource_self_closure"
            ]
        classification_valid = bool(
            len(self.classification.unclassified_nodes or []) == 0
            and len(getattr(self.classification, "claim_conflicts", []) or []) == 0
            and len(self.classification.classification_rejects or []) == 0
        )
        ownership_valid = bool(
            not list(ownership_view.get("conflicts", []) or [])
            and not list(diagnostics.get("node_coverage", {}).get("uncovered_node_ids", []) or [])
        )
        ledger_valid = bool(not list(ledger.invariant_errors or []))
        topology_complete = bool(
            len(self.pending_units) == 0
            and flex_violation_reason is None
            and not flex_above_max
            and stage3_phase == "complete"
        )
        result_ok = bool(topology_complete and classification_valid and ownership_valid and ledger_valid)
        if not result_ok and result_reason == "topology_assembly_complete":
            if not ownership_valid:
                result_reason = "final_ownership_conflict"
            elif not classification_valid:
                result_reason = "final_classification_invalid"
            elif not ledger_valid:
                result_reason = "final_resource_ledger_invalid"
        result = TopologyAssemblyResult(
            ok=bool(result_ok),
            reason=result_reason,
            feasible=bool(classification_valid and ownership_valid and ledger_valid),
            products=list(self.products or []),
            pending_units=list(self.pending_units or []),
            flex_bounds=dict(self.flex_bounds or {}),
            flex_counts={
                "explicit_total": int(explicit_flex_count),
                "final": int(final_flex_count),
            },
            resource_ledger=ledger,
            diagnostics=diagnostics,
            planned_conversions=[],
            active_phase=str(stage3_phase),
            completed_phases=list(completed_phases),
            residual_closure_products=list(residual_products),
            terminal_resource_counts=dict(terminal_resource_counts),
            terminal_converted_counts={},
            terminal_unresolved_counts=dict(terminal_unresolved_counts),
            phase_metrics={
                "residual_convertible_total": int(residual_convertible_total),
                "residual_product_count": int(len(residual_products)),
            },
            flex_target_plan=dict(flex_target_plan),
            resource_demand={
                "hard_required_counts": {
                    str(key): int(value)
                    for key, value in dict(self.shortage_counts or {}).items()
                    if int(value) > 0
                },
                "preferred_required_counts": {},
                "available_counts": self.allocator.available_counts(),
                "shortage_counts": {
                    str(key): int(value)
                    for key, value in dict(self.shortage_counts or {}).items()
                    if int(value) > 0
                },
                "contracts_by_subphase": self._contracts_by_subphase(),
            },
            ownership_ledger=dict(ownership_view),
            protected_resource_counts=dict(protected_resource_counts),
            referenced_resource_counts={
                str(key): int(value)
                for key, value in dict(getattr(ledger, "fragment_owned_counts_by_su", {}) or {}).items()
            },
            external_candidate_capacity_exhausted=bool(flex_target_plan["external_capacity_exhausted"]),
        )
        return result


def assemble_external_topologies(
    nodes: Sequence[_NodeV3],
    classification: TopologyClassificationResult,
    S_target: Any = None,
    E_target: Any = None,
    flex_audit: Optional[Dict[str, Any]] = None,
) -> TopologyAssemblyResult:
    resolved_flex_audit = dict(flex_audit or {})
    if not resolved_flex_audit:
        from .phase_eval import _compute_flex_audit  # local import to avoid module cycle

        resolved_flex_audit = dict(_compute_flex_audit(
            list(nodes or []),
            classification,
            enforce=True,
        ) or {})
    assembler = ExternalTopologyAssembler(
        nodes=nodes,
        classification=classification,
        S_target=S_target,
        E_target=E_target,
        flex_audit=resolved_flex_audit,
    )
    if bool(dict(assembler.flex_audit or {}).get("enforced", False)):
        scenario = str(dict(assembler.flex_audit or {}).get("scenario", "") or "")
        blocked_reason = str(dict(assembler.flex_audit or {}).get("blocked_reason", "") or "")
        if scenario == "above_range" or blocked_reason in {"flex_window_invalid", "supplemental_flex_candidate_shortfall"}:
            return _flex_preassembly_gate_result(
                classification=classification,
                flex_audit=dict(assembler.flex_audit or {}),
                flex_bounds=dict(assembler.flex_bounds or {}),
            )
    return assembler.assemble()
