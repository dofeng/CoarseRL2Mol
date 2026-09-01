from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..classification.classification_tokens import normalize_subtype
from ..stage3_models import SU24ResourceRoleProfile


CLOSURE_TEMPLATE_SEQUENCES: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "H": ((23, 22), (23, 23, 22)),
    "H1": ((23, 22),),
    "F1": ((23, 11), (23, 23, 11)),
    "F2": ((23, 22), (23, 23, 22)),
    "FIXED22": ((22,),),
}

DEFAULT_STAGE3_H_SOFT_TOLERANCE = 0.08
H_SOFT_LOWER_RATIO = -DEFAULT_STAGE3_H_SOFT_TOLERANCE
H_SOFT_UPPER_RATIO = DEFAULT_STAGE3_H_SOFT_TOLERANCE
# Aromatic donor rotation used when an existing aromatic unit is converted
# through SU11.  This is intentionally different from terminal SU24
# aromatization: the two operations have different chemical roles.
SU11_AROMATIC_ROTATION = (12, 13, 13)
# Terminal SU24 aromatization must preserve more bridgeheads than ordinary
# SU11 donor compensation.
MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES = 3
MAX_FLEX_LOWER_INCREASE_PER_ADJUSTMENT_PLAN = 1
MIN_STAGE3_AROMATIC_RESOURCE_COUNT = 3

RAW_SU24_CAP_SUS = {11}
RAW_SU24_TERMINAL_SUS = {1, 4, 22}
RAW_SU24_OPEN_BASE_SUS = {0, 2, 3, 14, 15, 17, 23, 24, 25}
RAW_SU24_OPEN_SUBTYPES = {
    "single19d2",
    "double19d2",
    "single19d3",
    "double19d3",
    "single20d2",
    "double20d2",
    "single20d3",
    "double20d3",
    "21d2",
    "21d3",
}

RAW_SU24_ALL_RING_SLOTS = {
    "side:left_anchor",
    "side:right_anchor",
    "side:middle",
    "upper_lower:fixed_anchor",
    "upper_lower:path",
    "fused:left_outer_anchor",
    "fused:right_outer_anchor",
    "fused:left_bridgehead",
    "fused:right_bridgehead",
    "fused:inner_core",
}

RAW_SU24_ALLOWED_RING_SLOTS: Dict[str, Tuple[str, ...]] = {
    "A": (
        "side:left_anchor",
        "side:right_anchor",
        "upper_lower:fixed_anchor",
        "fused:left_outer_anchor",
        "fused:right_outer_anchor",
    ),
    "B": (
        "side:left_anchor",
        "side:right_anchor",
        "fused:left_outer_anchor",
        "fused:right_outer_anchor",
    ),
    "C": (
        "side:middle",
        "upper_lower:path",
        "fused:left_bridgehead",
        "fused:right_bridgehead",
        "fused:inner_core",
    ),
    "D": (
        "side:middle",
        "upper_lower:path",
        "fused:inner_core",
    ),
}


def closure_sequences(kind: str) -> List[List[int]]:
    return [
        [int(value) for value in sequence]
        for sequence in CLOSURE_TEMPLATE_SEQUENCES.get(str(kind), ())
    ]


def counts_for_sequence(sequence: Sequence[int]) -> Dict[int, int]:
    counts: Counter = Counter(int(value) for value in list(sequence or []))
    return {int(key): int(value) for key, value in counts.items() if int(value) > 0}


def closure_required_count_options(kind: str) -> List[Dict[int, int]]:
    return [counts_for_sequence(sequence) for sequence in closure_sequences(str(kind))]


def minimum_closure_required_counts(kind: str) -> Dict[int, int]:
    options = closure_required_count_options(str(kind))
    return dict(options[0]) if options else {}


def alternative_closure_required_counts(kind: str) -> List[Dict[int, int]]:
    return [dict(item) for item in closure_required_count_options(str(kind))[1:]]


def h_soft_bounds(
    target_h: Optional[float],
    tolerance: float = DEFAULT_STAGE3_H_SOFT_TOLERANCE,
) -> Tuple[Optional[float], Optional[float]]:
    if target_h is None or float(target_h) <= 0:
        return None, None
    effective_tolerance = min(1.0, max(0.0, float(tolerance)))
    return (
        float(target_h) * (1.0 - float(effective_tolerance)),
        float(target_h) * (1.0 + float(effective_tolerance)),
    )
def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _node_id(node: Any) -> int:
    return _safe_int(getattr(node, "global_id", None), -1)


def _node_su(node: Any) -> int:
    return _safe_int(getattr(node, "su_type", None), -1)


def _node_label(node: Any) -> str:
    if node is None:
        return ""
    try:
        return str(normalize_subtype(node))
    except Exception:
        return str(_node_su(node))


def _integer_counter(value: Any) -> Dict[int, int]:
    try:
        items = dict(value or {}).items()
    except Exception:
        items = []
    return {
        int(key): int(count)
        for key, count in items
        if str(key).lstrip("-").isdigit()
    }


def _neighbors(node: Any, nodes_by_id: Optional[Dict[int, Any]]) -> List[Any]:
    by_id = dict(nodes_by_id or {})
    result: List[Any] = []
    for node_id in list(getattr(node, "hop1_ids", []) or []):
        neighbor = by_id.get(_safe_int(node_id, -1))
        if neighbor is not None:
            result.append(neighbor)
    return sorted(result, key=_node_id)


def classify_raw_su24_semantics(
    *,
    su_type: int,
    neighbors: Sequence[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Classify one raw SU24 from explicit real-neighbor semantics."""
    if int(su_type) != 24:
        return "", {"reason": "not_raw_su24"}

    cap: List[Dict[str, Any]] = []
    terminal: List[Dict[str, Any]] = []
    open_like: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for raw_neighbor in list(neighbors or []):
        neighbor = dict(raw_neighbor or {})
        neighbor_su = _safe_int(neighbor.get("su_type"), -1)
        neighbor_label = str(neighbor.get("subtype") or neighbor_su)
        if neighbor_su in RAW_SU24_CAP_SUS:
            cap.append(neighbor)
        elif neighbor_su in RAW_SU24_TERMINAL_SUS:
            terminal.append(neighbor)
        elif neighbor_su in RAW_SU24_OPEN_BASE_SUS or neighbor_label in RAW_SU24_OPEN_SUBTYPES:
            open_like.append(neighbor)
        else:
            other.append(neighbor)

    counter = Counter(
        _safe_int(dict(neighbor or {}).get("su_type"), -1)
        for neighbor in list(neighbors or [])
    )
    if other:
        return "", {
            "reason": "raw_su24_unsupported_neighbor",
            "unsupported_neighbor_ids": [
                _safe_int(item.get("node_id"), -1)
                for item in other
                if _safe_int(item.get("node_id"), -1) >= 0
            ],
            "hop1_su": dict(counter),
        }

    # Raw SU24 has at most one structural SU11 cap.  Additional SU11
    # neighbours are ordinary open-side neighbours; they must not turn the
    # SU24 into a multi-cap branch topology.
    cap = sorted(
        cap,
        key=lambda item: (
            _safe_int(item.get("node_id"), 10**9),
            str(item.get("subtype") or ""),
        ),
    )
    if len(cap) > 1:
        open_like.extend(cap[1:])
        cap = cap[:1]

    cap_count = len(cap)
    terminal_count = len(terminal)
    open_count = len(open_like)
    classified_neighbor_count = int(cap_count + terminal_count + open_count)
    if classified_neighbor_count != 3:
        role = ""
        reason = "raw_su24_degree_unresolved"
    elif cap_count == 1 and terminal_count == 0 and open_count == 2:
        role = "A"
        reason = ""
    elif cap_count == 1 and terminal_count == 1 and open_count == 1:
        role = "B"
        reason = ""
    elif cap_count == 0 and terminal_count == 0 and open_count == 3:
        role = "C"
        reason = ""
    elif cap_count == 0 and terminal_count == 1 and open_count == 2:
        role = "D"
        reason = ""
    else:
        role = ""
        reason = "raw_su24_role_ambiguous"

    details = {
        "role": str(role),
        "cap_node_ids": sorted(
            _safe_int(item.get("node_id"), -1)
            for item in cap
            if _safe_int(item.get("node_id"), -1) >= 0
        ),
        "terminal_node_ids": sorted(
            _safe_int(item.get("node_id"), -1)
            for item in terminal
            if _safe_int(item.get("node_id"), -1) >= 0
        ),
        "terminal_su_by_node_id": {
            str(_safe_int(item.get("node_id"), -1)): _safe_int(item.get("su_type"), -1)
            for item in terminal
            if _safe_int(item.get("node_id"), -1) >= 0
        },
        "open_neighbor_ids": sorted(
            _safe_int(item.get("node_id"), -1)
            for item in open_like
            if _safe_int(item.get("node_id"), -1) >= 0
        ),
        "hop1_su": dict(counter),
        "cap_count": int(cap_count),
        "terminal_count": int(terminal_count),
        "open_count": int(open_count),
    }
    if not role:
        details["reason"] = str(reason)
    return str(role), details


def classify_raw_su24_role(
    node: Any,
    nodes_by_id: Optional[Dict[int, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    neighbors = [
        {
            "node_id": _node_id(neighbor),
            "su_type": _node_su(neighbor),
            "subtype": _node_label(neighbor),
        }
        for neighbor in _neighbors(node, nodes_by_id)
    ]
    if not neighbors:
        counter = _integer_counter(getattr(node, "hop1_su", {}) or {})
        neighbors = [
            {
                "node_id": -1,
                "su_type": int(su_type),
                "subtype": str(int(su_type)),
            }
            for su_type, count in counter.items()
            for _ in range(max(0, int(count)))
        ]
    return classify_raw_su24_semantics(
        su_type=_node_su(node),
        neighbors=neighbors,
    )


def build_raw_su24_resource_profile(
    node: Any,
    nodes_by_id: Optional[Dict[int, Any]] = None,
) -> Optional[SU24ResourceRoleProfile]:
    role, details = classify_raw_su24_role(node, nodes_by_id)
    if not role:
        return None
    allowed = sorted(RAW_SU24_ALLOWED_RING_SLOTS.get(str(role), ()))
    forbidden = sorted(RAW_SU24_ALL_RING_SLOTS - set(allowed))
    hop1_su = {
        int(key): int(value)
        for key, value in dict(details.get("hop1_su", {}) or {}).items()
    }
    hop1_node_ids = sorted(
        _safe_int(value)
        for value in list(getattr(node, "hop1_ids", []) or [])
        if _safe_int(value) >= 0
    )
    terminal_node_ids = [
        int(value) for value in list(details.get("terminal_node_ids", []) or [])
    ]
    cap_node_ids = [
        int(value) for value in list(details.get("cap_node_ids", []) or [])
    ]
    open_neighbor_ids = [
        int(value) for value in list(details.get("open_neighbor_ids", []) or [])
    ]
    target_degree = _safe_int(getattr(node, "target_hop1_degree", 3) or 3, 3)
    degree = sum(int(value) for value in hop1_su.values())
    return SU24ResourceRoleProfile(
        node_id=_node_id(node),
        role=str(role),
        roles=[str(role)],
        available_roles=[str(role)],
        hop1_su=hop1_su,
        hop1_node_ids=hop1_node_ids,
        fixed_neighbor_ids=list(terminal_node_ids),
        cap_node_ids=list(cap_node_ids),
        terminal_node_ids=list(terminal_node_ids),
        terminal_su_by_node_id={
            str(key): int(value)
            for key, value in dict(details.get("terminal_su_by_node_id", {}) or {}).items()
        },
        open_neighbor_ids=list(open_neighbor_ids),
        fixed_neighbor_count=len(terminal_node_ids),
        remaining_degree=max(0, int(target_degree) - int(degree)),
        branch_h_count=1 if str(role) in {"A", "C"} else 0,
        requires_su11_cap=bool(str(role) in {"A", "B"}),
        allowed_ring_slots=allowed,
        forbidden_ring_slots=forbidden,
    )


def audit_raw_su24_resource_profiles(
    nodes: Sequence[Any],
    pool: Dict[int, List[int]],
) -> Dict[str, Any]:
    by_id = {
        _node_id(node): node
        for node in list(nodes or [])
        if _node_id(node) >= 0
    }
    pool_local = dict(pool or {})
    pool_24_ids = list(pool_local.get(24, []) or []) + list(pool_local.get("24", []) or [])
    profiles: List[SU24ResourceRoleProfile] = []
    rejected: List[Dict[str, Any]] = []
    for node_id in sorted({int(value) for value in list(pool_24_ids or [])}):
        node = by_id.get(int(node_id))
        if node is None:
            rejected.append({
                "node_id": int(node_id),
                "reason": "raw_su24_node_missing",
            })
            continue
        role, details = classify_raw_su24_role(node, by_id)
        if not role:
            rejected.append({
                "node_id": int(node_id),
                **dict(details or {}),
            })
            continue
        profile = build_raw_su24_resource_profile(node, by_id)
        if profile is not None:
            profiles.append(profile)
    return {
        "profiles": profiles,
        "rejected": rejected,
        "pool_su24_ids": sorted({
            int(value) for value in list(pool_24_ids or [])
        }),
    }


def build_raw_su24_resource_profiles(
    nodes: Sequence[Any],
    pool: Dict[int, List[int]],
) -> List[SU24ResourceRoleProfile]:
    return list(
        audit_raw_su24_resource_profiles(nodes, pool).get("profiles", [])
        or []
    )


def raw_su24_slot_needs_h(profile: SU24ResourceRoleProfile, skeleton_kind: str, slot_role: str) -> bool:
    if str(profile.role or next(iter(profile.roles or []), "")) not in {"A", "C"}:
        return False
    if str(slot_role) in {"left_bridgehead", "right_bridgehead", "fixed_anchor"}:
        return False
    if str(skeleton_kind) == "upper_lower" and str(slot_role) == "fixed_anchor":
        return False
    return True


def raw_su24_segment_support(
    profile: SU24ResourceRoleProfile,
    *,
    skeleton_kind: str,
    slot_role: str,
) -> Dict[str, Any]:
    role = str(profile.role or next(iter(profile.roles or []), "") or "")
    support_counts: Counter = Counter()
    support_specific: Dict[int, List[int]] = {}
    existing_branch_stubs: List[Dict[str, Any]] = []
    materialized_cap_ids: List[int] = []

    if role in {"A", "B"}:
        cap_ids = [int(value) for value in list(profile.cap_node_ids or [])]
        if cap_ids:
            support_specific[11] = [int(cap_ids[0])]
            materialized_cap_ids = [int(cap_ids[0])]
        else:
            support_counts[11] += 1

    if role in {"B", "D"}:
        terminal_ids = [int(value) for value in list(profile.terminal_node_ids or [])]
        if terminal_ids:
            terminal_id = int(terminal_ids[0])
            terminal_su = int(
                dict(getattr(profile, "terminal_su_by_node_id", {}) or {}).get(
                    str(terminal_id),
                    -1,
                )
            )
            # Preserve the concrete SU22 identity.  Classification binds it to
            # this raw SU24 when it is a real terminal support; the ring
            # transaction then carries it as a fixed source instead of
            # replacing it with an arbitrary SU22.  Fixed SU1/SU4 terminals
            # remain source topology rather than pool claims.
            if terminal_su == 22:
                support_specific.setdefault(22, []).append(terminal_id)
            existing_branch_stubs.append({
                "topology_notation": f"-{terminal_su}" if terminal_su > 0 else f"-{terminal_id}",
                "node_id_notation": f"-{terminal_id}",
                "node_id": terminal_id,
                "branch_kind": "FIXED22" if terminal_su == 22 else "FIXED_TERMINAL",
                "mode": f"raw_su24_{role}_fixed_terminal",
            })

    if raw_su24_slot_needs_h(profile, skeleton_kind, slot_role):
        for su_type in closure_sequences("H")[0]:
            support_counts[int(su_type)] += 1

    return {
        "support_required_counts": {
            int(key): int(value)
            for key, value in support_counts.items()
            if int(value) > 0
        },
        "support_specific_ids_by_su": {
            int(key): [int(value) for value in list(values or [])]
            for key, values in support_specific.items()
            if list(values or [])
        },
        "existing_branch_stubs": existing_branch_stubs,
        "materialized_cap_ids": materialized_cap_ids,
        "needs_h": raw_su24_slot_needs_h(profile, skeleton_kind, slot_role),
        "role": role,
    }
