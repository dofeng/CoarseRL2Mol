from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from ....shared.inverse_common import _NodeV3
else:
    _NodeV3 = Any
from .classification_models import TopologyClassificationResult, TopologyFragment, TopologyRef
from ..assembly.state.iteration_ledger import normalize_slot
from ..resources.profiles import classify_raw_su24_semantics
from ..resources.types import PHYSICAL_TOPOLOGY_RESOURCE_TYPES
from ..stage3_models import TopologyClosureSlot, TopologySlotKind
from .classification_rules_basic import (
    ALIPHATIC_OPEN_NEIGHBORS,
    BASIC_A_AROMATIC_BY_CENTER,
    BASIC_B_SPECIAL_OPEN_BY_CENTER,
    BASIC_C_SPECIAL_TERMINAL_BY_CENTER,
    BASIC_CHAIN_RULES,
    BASIC_D_SPECIAL_OPEN_BY_CENTER,
    BASIC_E_SPECIAL_SIDE_BY_CENTER,
    BASIC_F_AROMATIC_BY_CENTER,
    BASIC_G_SIDE_BY_CENTER,
    DOUBLE_D2_ALLOWED_CONNECTORS,
    DOUBLE_D2_COMBINATION_RULES,
    DOUBLE_D2_SIDE_RULES,
    LIKE24_ALLOWED_CONNECTORS,
    LIKE24_FIXED_ELIGIBLE_SUBTYPES,
    LIKE24_FIXED_SIDE_RULES,
    SU25_RULES,
    TERMINAL_NEIGHBORS,
)
from .classification_rules_like24 import (
    LIKE24_BASE_RULES,
    LIKE24_PAIR_BRIDGE_RULES,
    LIKE24_SECONDARY_RULES,
)
from .classification_tokens import (
    AROMATIC_SU_TYPES,
    CONNECTOR_SU_TYPES,
    ClassificationToken,
    NeighborToken,
    build_node_tokens,
    is_abstract_open_nonfixed_semantic,
    is_open_boundary_semantic,
    is_visible_fixed_connector_semantic,
    is_visible_terminal_semantic,
)


PHYSICAL_TOPOLOGY_POOL_SU_TYPES = set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES)
SU11_DONOR_SU_TYPES = {10, 12, 13}
DOUBLE_D2_CLASS_TYPES = {"A+", "B+", "C+", "D+", "E+"}
BASIC_CLASS_TYPES = {"A", "B", "C", "D", "E", "F", "G"}
FIXED_SIDE_PRIORITY = {"M": 0, "N": 1, "P": 2}
FLEX_RECONCILE_EXPLICIT_FLEX_CLASSES = {
    "A", "A+", "24_AB_M", "24_CD_M_FULL",
}
FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES = {"C", "C+", "F", "24_AB_P"}
FLEX_RECONCILE_B_LIKE_OPEN_CLASSES = {
    "B", "B+", "24_A", "24_B", "24_AB_N", "24_CD_M", "24_AB_N_24_CD", "25_AB",
}
FLEX_RECONCILE_D_LIKE_OPEN_CLASSES = {
    "D", "D+", "24_C", "24_D", "24_CD_N", "24_CD_N_24_CD",
    "24_AB_N_24_AB", "25_CD",
}
FLEX_RECONCILE_E_LIKE_OPEN_CLASSES = {"E", "E+", "G", "24_CD_P"}
FLEX_RANGE_DEMOTABLE_EXPLICIT_CLASSES = set()
FLEX_RECONCILE_FLEX_CHAIN_CLASSES = {
    "A",
    "B",
    "D",
    "A+",
    "B+",
    "D+",
    "24_AB_M",
    "24_AB_N",
    "24_CD_M",
    "24_CD_N",
    "24_CD_M_FULL",
    "24_AB_N_24_CD",
    "24_CD_N_24_CD",
    "25_AB",
    "25_CD",
}
FLEX_RECONCILE_SIDE_CHAIN_CLASSES = {"C", "E", "F", "G", "C+", "E+", "24_AB_P", "24_CD_P"}
PAIRED_LIKE24_CLASS_TYPES = {"24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD"}

_DEFAULT_ASSEMBLY_CAPABILITIES_BY_FAMILY = {
    "basic": {
        "flex": {"A", "B", "D"},
        "side": {"B", "C", "D", "E", "F", "G"},
        "ring": {"B", "D"},
    },
    "double_d2_combo": {
        "flex": {"A+", "B+", "D+"},
        "side": {"B+", "C+", "D+", "E+"},
        # Double-d2 combinations have mainline closure contracts but no
        # standard fatty-ring skeleton.  Ring capability belongs to Like24
        # contracts and must not be inferred from the ``+`` class suffix.
        "ring": set(),
    },
    "su25": {
        "flex": {"25_AB", "25_CD"},
        "side": {"25_AB", "25_CD"},
        "ring": set(),
    },
}

GENERIC_OPEN_SUBTYPE_LIKE = {"single19d2", "single20d2", "double19d2", "double20d2"}
SUPPORT_ONLY_SU_TYPES = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
CLASSIFIABLE_CENTER_SU_TYPES = {0, 1, 2, 3, 4, 14, 15, 16, 18, 19, 20, 21, 24, 25, 27, 28, 29, 31, 32}
DOUBLE_BOND_SU_TYPES = {14, 15, 16}
TRIPLE_BOND_SU_TYPES = {17, 18}
ALLOWED_DOUBLE_BOND_PAIRS = {
    (14, 14), (14, 15), (14, 16), (15, 15), (15, 16),
}
ALLOWED_TRIPLE_BOND_PAIRS = {(17, 17), (17, 18)}
OPEN_BRIDGE_SUBTYPE_SET = {
    "23",
    "24",
    "25",
    "14",
    "15",
    "17",
    "0",
    "2",
    "3",
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

BASIC_CAP_ONLY_BRIDGE_SUBTYPES = {
    "23",
    "single19d2",
    "single20d2",
    "double19d2",
    "double20d2",
    "21d2",
}

BASIC_RULE_BY_ID = {
    str(rule.get("rule_id")): dict(rule)
    for rule in BASIC_CHAIN_RULES
}
DOUBLE_D2_COMBO_BY_COMPONENTS = {
    tuple(sorted(str(component) for component in tuple(rule.get("components", ()) or ()))): dict(rule)
    for rule in DOUBLE_D2_COMBINATION_RULES
}
LIKE24_SECONDARY_BY_CLASS = {
    str(rule.get("class_type")): dict(rule)
    for rule in LIKE24_SECONDARY_RULES
}
LIKE24_BRIDGE_BY_CLASS = {
    str(rule.get("class_type")): dict(rule)
    for rule in LIKE24_PAIR_BRIDGE_RULES
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _raw_su24_support_binding(
    token: ClassificationToken,
) -> Optional[Dict[str, Any]]:
    """Return mandatory, concrete supports for one raw SU24.

    The open neighbours of a raw SU24 are closure slots and deliberately do
    not belong to this binding.  Only the chemically fixed cap/terminal
    neighbours are made atomic with the SU24 center.
    """
    if int(token.su_type) != 24:
        return None
    role, details = classify_raw_su24_semantics(
        su_type=int(token.su_type),
        neighbors=[neighbor.to_dict() for neighbor in list(token.neighbor_tokens or [])],
    )
    if not role:
        return None

    fixed_by_su: Dict[int, List[int]] = {}
    if str(role) in {"A", "B"}:
        cap_ids = [
            _safe_int(value)
            for value in list(details.get("cap_node_ids", []) or [])
            if _safe_int(value) >= 0
        ]
        if cap_ids:
            fixed_by_su[11] = sorted(set(cap_ids))
    if str(role) in {"B", "D"}:
        terminal_ids = [
            _safe_int(value)
            for value in list(details.get("terminal_node_ids", []) or [])
            if _safe_int(value) >= 0
        ]
        terminal_su_by_id = {
            _safe_int(key): _safe_int(value)
            for key, value in dict(details.get("terminal_su_by_node_id", {}) or {}).items()
            if _safe_int(key) >= 0
        }
        for terminal_id in terminal_ids:
            terminal_su = int(terminal_su_by_id.get(int(terminal_id), -1))
            if terminal_su in {1, 4, 22}:
                fixed_by_su.setdefault(terminal_su, []).append(int(terminal_id))
        fixed_by_su = {
            int(su_type): sorted(set(int(value) for value in values))
            for su_type, values in fixed_by_su.items()
            if values
        }

    fixed_ids = sorted({
        int(value)
        for values in fixed_by_su.values()
        for value in values
    })
    return {
        "su24_node_id": int(token.node_id),
        "role": str(role),
        "fixed_support_ids_by_su": {
            str(int(su_type)): [int(value) for value in values]
            for su_type, values in sorted(fixed_by_su.items())
        },
        "fixed_support_ids": fixed_ids,
        "open_neighbor_ids": [
            int(value)
            for value in list(details.get("open_neighbor_ids", []) or [])
            if _safe_int(value) >= 0
        ],
    }


def _build_resource_pool(
    tokens: Sequence[ClassificationToken],
    consumed: Set[int],
    *,
    excluded_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    remaining_ids_by_su: Dict[str, List[int]] = {
        str(su): [] for su in PHYSICAL_TOPOLOGY_RESOURCE_TYPES
    }
    su11_donor_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in sorted(SU11_DONOR_SU_TYPES)}
    excluded = {int(node_id) for node_id in list(excluded_ids or [])}
    for token in list(tokens or []):
        # SU10/SU12/SU13 are aromatic construction resources, not classified
        # topology ownership. Their current physical type is the full donor
        # contract, even when a fragment references them as 1-hop context.
        if int(token.su_type) in SU11_DONOR_SU_TYPES:
            su11_donor_ids_by_su.setdefault(str(token.su_type), []).append(int(token.node_id))
            continue
        if int(token.node_id) in consumed:
            continue
        if int(token.node_id) in excluded:
            continue
        if int(token.su_type) in PHYSICAL_TOPOLOGY_POOL_SU_TYPES:
            remaining_ids_by_su.setdefault(str(token.su_type), []).append(int(token.node_id))
    for values in list(remaining_ids_by_su.values()) + list(su11_donor_ids_by_su.values()):
        values.sort()
    remaining_counts = {str(k): int(len(v or [])) for k, v in remaining_ids_by_su.items()}
    return {
        "remaining_ids_by_su": remaining_ids_by_su,
        "initial_ids_by_su": {str(k): list(v or []) for k, v in remaining_ids_by_su.items()},
        "remaining_counts": remaining_counts,
        "su11_donor_ids_by_su": su11_donor_ids_by_su,
        "su11_donor_counts": {
            str(key): int(len(values or []))
            for key, values in su11_donor_ids_by_su.items()
        },
        "shortages": [],
    }


def _token_by_id(tokens: Sequence[ClassificationToken]) -> Dict[int, ClassificationToken]:
    return {int(token.node_id): token for token in list(tokens or [])}


def _dedupe_ints(values: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for value in list(values or []):
        try:
            value_i = int(value)
        except Exception:
            continue
        if int(value_i) in seen:
            continue
        seen.add(int(value_i))
        out.append(int(value_i))
    return out


def _union_source_ids(*chunks: Iterable[int]) -> List[int]:
    merged: List[int] = []
    for chunk in chunks:
        merged.extend(int(value) for value in list(chunk or []))
    return sorted(_dedupe_ints(merged))


def _ordered_atomic_pair_ids(
    node_ids: Iterable[int],
    tokens_by_id: Dict[int, ClassificationToken],
) -> List[int]:
    """Order an unsaturated pair by chemical role, not by global ID."""
    return sorted(
        {int(value) for value in list(node_ids or [])},
        key=lambda node_id: (
            2 if int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)) in {16, 18} else 0,
            0 if int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)) == 14 else 1,
            int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)),
            int(node_id),
        ),
    )


def _build_unsaturated_atomic_groups(
    tokens: Sequence[ClassificationToken],
    rejects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    token_by_id = {int(token.node_id): token for token in list(tokens or [])}
    groups: List[Dict[str, Any]] = []
    assigned: Set[int] = set()
    rejected: Set[int] = set()
    for token in sorted(list(tokens or []), key=lambda item: int(item.node_id)):
        node_id = int(token.node_id)
        su_type = int(token.su_type)
        if su_type not in DOUBLE_BOND_SU_TYPES | TRIPLE_BOND_SU_TYPES:
            continue
        family = DOUBLE_BOND_SU_TYPES if su_type in DOUBLE_BOND_SU_TYPES else TRIPLE_BOND_SU_TYPES
        allowed_pairs = ALLOWED_DOUBLE_BOND_PAIRS if su_type in DOUBLE_BOND_SU_TYPES else ALLOWED_TRIPLE_BOND_PAIRS
        partner_ids = sorted(
            int(neighbor.node_id)
            for neighbor in list(token.neighbor_tokens or [])
            if int(neighbor.su_type) in family
        )
        if len(partner_ids) != 1:
            if node_id not in rejected:
                _record_reject(
                    rejects,
                    token,
                    family="unsaturated_atomic_pair",
                    reason="unsaturated_pair_count_mismatch",
                    details={"partner_node_ids": partner_ids, "expected_count": 1},
                )
                rejected.add(node_id)
            continue
        partner_id = int(partner_ids[0])
        partner = token_by_id.get(partner_id)
        reverse_ids = sorted(
            int(neighbor.node_id)
            for neighbor in list(getattr(partner, "neighbor_tokens", []) or [])
            if int(neighbor.su_type) in family
        ) if partner is not None else []
        pair_types = tuple(sorted((su_type, int(getattr(partner, "su_type", -1)))))
        if partner is None or reverse_ids != [node_id] or pair_types not in allowed_pairs:
            if node_id not in rejected:
                _record_reject(
                    rejects,
                    token,
                    family="unsaturated_atomic_pair",
                    reason="invalid_unsaturated_atomic_pair",
                    details={
                        "partner_node_id": partner_id,
                        "partner_su_type": int(getattr(partner, "su_type", -1)),
                        "reverse_partner_node_ids": reverse_ids,
                    },
                )
                rejected.add(node_id)
            continue
        group_node_ids = sorted([node_id, partner_id])
        group_id = int(group_node_ids[0])
        group_kind = "double_bond_pair" if su_type in DOUBLE_BOND_SU_TYPES else "triple_bond_pair"
        metadata = {
            "group_id": group_id,
            "kind": group_kind,
            "node_ids": group_node_ids,
            "su_pair": list(pair_types),
        }
        for member_id in group_node_ids:
            member = token_by_id[member_id]
            member.meta["atomic_group_id"] = group_id
            member.meta["atomic_group_kind"] = group_kind
            member.meta["atomic_group_node_ids"] = list(group_node_ids)
            member.meta["atomic_group_su_pair"] = list(pair_types)
            member.meta["atomic_partner_node_id"] = partner_id if member_id == node_id else node_id
        if not (set(group_node_ids) & assigned):
            groups.append(metadata)
            assigned.update(group_node_ids)
    return groups


def _collect_direct_source_ids(token: ClassificationToken) -> List[int]:
    return _union_source_ids(
        [int(token.node_id)],
        [int(neighbor.node_id) for neighbor in list(token.neighbor_tokens or [])],
    )


def _tokens_have_fixed_edge(
    left_node_id: int,
    right_node_id: int,
    tokens_by_id: Dict[int, ClassificationToken],
) -> bool:
    left = tokens_by_id.get(int(left_node_id))
    right = tokens_by_id.get(int(right_node_id))
    if left is None or right is None:
        return False
    return bool(
        int(right_node_id) in set(int(value) for value in left.fixed_neighbor_ids)
        and int(left_node_id) in set(int(value) for value in right.fixed_neighbor_ids)
    )


def _fixed_reachable_node_ids(
    root_node_ids: Iterable[int],
    tokens_by_id: Dict[int, ClassificationToken],
) -> Set[int]:
    pending = [int(value) for value in list(root_node_ids or []) if int(value) in tokens_by_id]
    reached: Set[int] = set()
    while pending:
        node_id = int(pending.pop())
        if node_id in reached:
            continue
        reached.add(node_id)
        token = tokens_by_id.get(node_id)
        if token is None:
            continue
        for neighbor in list(token.neighbor_tokens or []):
            neighbor_id = int(neighbor.node_id)
            if (
                bool(neighbor.is_fixed)
                and _tokens_have_fixed_edge(node_id, neighbor_id, tokens_by_id)
                and neighbor_id not in reached
            ):
                pending.append(neighbor_id)
    return reached


def _fragment_center_node_ids(fragment: TopologyFragment) -> List[int]:
    center_ids = [
        int(value)
        for value in list(dict(fragment.evidence or {}).get("center_node_ids", []) or [])
    ]
    if not center_ids:
        center_ids = [int(fragment.fragment_id)]
    return sorted(set(center_ids))


def _normalize_fixed_contract_route_from_graph(
    contract: Dict[str, Any],
    *,
    default_center_id: int,
    tokens_by_id: Dict[int, ClassificationToken],
) -> Dict[str, Any]:
    """Build one branch contract as a single route over concrete fixed edges."""

    normalized = dict(contract or {})
    center_id = _safe_int(normalized.get("center_node_id"), default_center_id)
    connector_id = _safe_int(normalized.get("connector_node_id"), -1)
    if connector_id < 0:
        raw_ids = [
            _safe_int(value, -1)
            for value in list(normalized.get("base_path_node_ids", []) or [])
            if _safe_int(value, -1) >= 0
        ]
        connector_id = raw_ids[0] if raw_ids else -1

    route_ids: List[int] = []
    route_labels: List[str] = []
    previous_id = int(center_id)
    current_id = int(connector_id)
    visited = {int(center_id)}

    def _set_boundary(reason: str, **details: Any) -> None:
        normalized["tail_closure_kind"] = "STRUCTURAL_BOUNDARY"
        normalized["tail_open_reason"] = str(reason)
        normalized["structural_boundary"] = {
            "reason": str(reason),
            **details,
        }
        normalized["externally_closed"] = False

    if current_id < 0 or current_id not in tokens_by_id:
        normalized["base_path_node_ids"] = []
        normalized["base_path_labels"] = []
        _set_boundary(
            "fixed_route_node_missing",
            connector_node_id=int(current_id),
        )
        return normalized
    parent_edge_ok = _tokens_have_fixed_edge(
        center_id,
        current_id,
        tokens_by_id,
    )
    if not parent_edge_ok:
        normalized["base_path_node_ids"] = []
        normalized["base_path_labels"] = []
        normalized["path_truncated_at_nonfixed_edge"] = [
            int(center_id),
            int(current_id),
        ]
        normalized["tail_closure_kind"] = "OPEN_BOUNDARY"
        normalized["tail_open_reason"] = "nonfixed_edge_stops_concrete_path"
        normalized["externally_closed"] = False
        return normalized

    while current_id >= 0:
        if current_id in visited:
            _set_boundary(
                "nested_branch_topology_forbidden:fixed_route_cycle",
                connector_node_id=int(connector_id),
                cycle_node_id=int(current_id),
            )
            break
        token = tokens_by_id.get(int(current_id))
        if token is None:
            _set_boundary(
                "fixed_route_node_missing",
                connector_node_id=int(connector_id),
                missing_node_ids=[int(current_id)],
            )
            break
        visited.add(int(current_id))
        route_ids.append(int(current_id))
        route_labels.append(str(token.subtype or token.su_type))

        if (
            len(route_ids) > 1
            and int(token.su_type) in set(AROMATIC_SU_TYPES)
        ):
            normalized["allow_aromatic_endpoint_context"] = True
            normalized["tail_closure_kind"] = "none"
            normalized["tail_open_reason"] = ""
            normalized["externally_closed"] = True
            break

        forward_ids = sorted({
            int(neighbor.node_id)
            for neighbor in list(token.neighbor_tokens or [])
            if int(neighbor.node_id) != int(previous_id)
            and bool(neighbor.is_fixed)
            and _tokens_have_fixed_edge(
                int(current_id),
                int(neighbor.node_id),
                tokens_by_id,
            )
        })
        if is_visible_terminal_semantic(token) and forward_ids:
            _set_boundary(
                f"terminal_has_fixed_continuation:{int(current_id)}",
                connector_node_id=int(connector_id),
                terminal_node_id=int(current_id),
                fixed_child_node_ids=list(forward_ids),
            )
            break
        if len(forward_ids) > 1:
            _set_boundary(
                "nested_branch_topology_forbidden:nonlinear_fixed_component",
                connector_node_id=int(current_id),
                fixed_child_node_ids=list(forward_ids),
            )
            break
        if not forward_ids:
            break
        previous_id, current_id = int(current_id), int(forward_ids[0])

    normalized["base_path_node_ids"] = list(route_ids)
    normalized["base_path_labels"] = list(route_labels)
    # A d2 node can be the tail of a mainline fixed route while still having
    # an unresolved SU23 neighbour.  The H1 label describes the fixed-side
    # branch contract at that node; it does not close the physical mainline
    # endpoint.  Preserve the concrete boundary so the fragment normalizer
    # can expose the required F1 = [23, 11] slot without widening ordinary
    # branch-only H1 contracts.
    if str(normalized.get("tail_closure_kind", "")) in {"H1", "H_AFTER_PREFIX"} and route_ids:
        tail_token = tokens_by_id.get(int(route_ids[-1]))
        tail_boundary_ids = sorted({
            int(neighbor.node_id)
            for neighbor in list(getattr(tail_token, "neighbor_tokens", []) or [])
            if not bool(neighbor.is_fixed)
            and int(neighbor.node_id) not in set(visited)
        }) if tail_token is not None else []
        normalized["tail_open_boundary_node_ids"] = list(tail_boundary_ids)
        normalized["tail_open_boundary_su_types"] = sorted({
            int(tokens_by_id[int(node_id)].su_type)
            for node_id in tail_boundary_ids
            if int(node_id) in tokens_by_id
        })
    return normalized


def _normalize_fragment_fixed_connection_contracts(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    roots = _fragment_center_node_ids(fragment)
    raw_contracts = [
        dict(item or {})
        for item in list(
            dict(fragment.evidence or {}).get("fixed_connection_contracts", [])
            or []
        )
        if dict(item or {})
    ]
    fragment.evidence["fixed_connection_contracts"] = [
        _normalize_fixed_contract_route_from_graph(
            contract,
            default_center_id=roots[0] if roots else -1,
            tokens_by_id=tokens_by_id,
        )
        for contract in raw_contracts
    ]
    return fragment


def _finalize_fragment_edge_semantics(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    roots = _fragment_center_node_ids(fragment)
    reachable = _fixed_reachable_node_ids(roots, tokens_by_id)
    original_claimed = {
        int(value)
        for value in list(fragment.claimed_node_ids or fragment.source_node_ids or roots)
    }
    intrinsic_group_ids = {
        int(value)
        for group in list(dict(fragment.evidence or {}).get("atomic_groups", []) or [])
        if set(int(item) for item in list(dict(group or {}).get("node_ids", []) or [])) & set(roots)
        for value in list(dict(group or {}).get("node_ids", []) or [])
    }
    claimed = (original_claimed & (reachable | intrinsic_group_ids)) | set(roots)
    released = original_claimed - claimed
    context = {
        int(value) for value in list(fragment.context_node_ids or [])
    } | released
    boundaries: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for owner_id in sorted(claimed):
        owner = tokens_by_id.get(int(owner_id))
        if owner is None:
            continue
        for neighbor in list(owner.neighbor_tokens or []):
            if not is_open_boundary_semantic(neighbor):
                continue
            neighbor_id = int(neighbor.node_id)
            if owner_id in intrinsic_group_ids and neighbor_id in intrinsic_group_ids:
                continue
            edge_key = tuple(sorted((int(owner_id), int(neighbor_id))))
            boundaries[edge_key] = {
                "owner_node_id": int(owner_id),
                "boundary_neighbor_id": int(neighbor_id),
                "edge_key": list(edge_key),
                "neighbor_su_type": int(neighbor.su_type),
                "neighbor_subtype": str(neighbor.subtype),
                "role_candidates": ["mainline", "branch"],
                "closure_kind": "OPEN",
            }
            if neighbor_id not in claimed:
                context.add(neighbor_id)

    fragment.claimed_node_ids = sorted(claimed)
    fragment.source_node_ids = sorted(claimed)
    fragment.context_node_ids = sorted(context - claimed)
    fragment.open_boundaries = [boundaries[key] for key in sorted(boundaries)]
    if fragment.mainline_node_ids:
        fragment.mainline_node_ids = [
            int(value)
            for value in list(fragment.mainline_node_ids or [])
            if int(value) in claimed
        ]
    fragment.evidence["open_boundaries"] = [dict(item) for item in fragment.open_boundaries]
    fragment.evidence["released_nonfixed_claim_node_ids"] = sorted(released)

    fixed_contracts_raw = [
        dict(item or {})
        for item in list(
            fragment.evidence.get("fixed_connection_contracts", []) or []
        )
        if dict(item or {})
    ]
    fixed_branch_satisfaction: Counter = Counter()
    for contract in fixed_contracts_raw:
        if "branch_only" not in {
            str(value)
            for value in list(contract.get("role_candidates", []) or [])
        }:
            continue
        contract_center_id = _safe_int(
            contract.get("center_node_id"), roots[0] if roots else -1
        )
        fixed_branch_satisfaction[int(contract_center_id)] += 1

    if not fragment.branch_contracts:
        required_branch_map = dict(
            fragment.evidence.get("required_branch_kinds_by_anchor", {}) or {}
        )
        for raw_anchor, kinds in required_branch_map.items():
            anchor_id = (
                int(roots[0])
                if str(raw_anchor) == "center" and roots
                else _safe_int(raw_anchor, -1)
            )
            if anchor_id in set(roots):
                retained_kinds: List[str] = []
                for kind in [value for value in list(kinds or []) if str(value)]:
                    if (
                        str(kind) == TopologySlotKind.H.value
                        and int(fixed_branch_satisfaction.get(int(anchor_id), 0)) > 0
                    ):
                        fixed_branch_satisfaction[int(anchor_id)] -= 1
                        continue
                    retained_kinds.append(str(kind))
                for index, kind in enumerate(retained_kinds):
                    fragment.branch_contracts.append(TopologyClosureSlot(
                        slot_id=(
                            f"fragment:{int(fragment.fragment_id)}:branch:"
                            f"{int(anchor_id)}:{int(index)}:{str(kind)}"
                        ),
                        kind=str(kind),
                        anchor_node_id=int(anchor_id),
                        anchor_role="center_branch",
                        required_su_sequence=(
                            [23, 22] if str(kind) in {"H", "H1"} else []
                        ),
                    ))
        if not required_branch_map and len(roots) == 1:
            for index, kind in enumerate(
                [value for value in list(fragment.branch_slots or []) if str(value)]
            ):
                fragment.branch_contracts.append(TopologyClosureSlot(
                    slot_id=(
                        f"fragment:{int(fragment.fragment_id)}:branch:"
                        f"{int(roots[0])}:{int(index)}:{str(kind)}"
                    ),
                    kind=str(kind),
                    anchor_node_id=int(roots[0]),
                    anchor_role="center_branch",
                    required_su_sequence=(
                        [23, 22] if str(kind) in {"H", "H1"} else []
                    ),
                ))

    branch_count_by_center: Counter = Counter()
    for raw_slot in list(fragment.branch_contracts or []):
        slot = normalize_slot(raw_slot)
        if not slot.is_complete():
            branch_count_by_center[int(slot.anchor_node_id)] += 1

    selected_primary_id = _safe_int(
        dict(fragment.evidence.get("selected_primary_connector", {}) or {}).get(
            "connector_node_id"
        ),
        -1,
    )

    open_mainline_slots: List[TopologyClosureSlot] = []
    for root_index, root_id in enumerate(roots):
        root = tokens_by_id.get(int(root_id))
        if root is None:
            continue
        concrete_neighbors = {
            int(neighbor.node_id)
            for neighbor in list(root.neighbor_tokens or [])
            if (
                bool(neighbor.is_fixed)
                and _tokens_have_fixed_edge(int(root_id), int(neighbor.node_id), tokens_by_id)
            )
            or (
                int(root_id) in intrinsic_group_ids
                and int(neighbor.node_id) in intrinsic_group_ids
            )
        }
        root_boundaries = [
            dict(boundary or {})
            for boundary in fragment.open_boundaries
            if _safe_int(dict(boundary or {}).get("owner_node_id"), -1) == int(root_id)
        ]
        boundary_count = len(root_boundaries)
        target_degree = int(root.target_degree)
        if target_degree <= 0:
            continue
        declared_branch_count = int(branch_count_by_center.get(int(root_id), 0))
        unresolved_degree = max(
            0, int(target_degree) - len(concrete_neighbors) - declared_branch_count
        )
        nonfixed_su11_count = sum(
            1
            for boundary in root_boundaries
            if _safe_int(boundary.get("neighbor_su_type"), -1) == 11
        )
        fixed_mainline_contract_count = sum(
            1
            for contract in fixed_contracts_raw
            if (
                _safe_int(
                    contract.get("center_node_id"),
                    roots[0] if roots else -1,
                )
                == int(root_id)
                and _safe_int(contract.get("connector_node_id"), -1) >= 0
                and (
                    "mainline_only" in {
                        str(value)
                        for value in list(
                            contract.get("role_candidates", []) or []
                        )
                    }
                    or (
                        _safe_int(contract.get("connector_node_id"), -1)
                        == int(selected_primary_id)
                        and "mainline_or_branch" in {
                            str(value)
                            for value in list(
                                contract.get("role_candidates", []) or []
                            )
                        }
                    )
                )
            )
        )
        # A center has at most two mainline directions. Once fixed contracts
        # occupy both, any remaining open degree is a branch attachment.
        mainline_open_capacity = max(
            0,
            2 - int(fixed_mainline_contract_count),
        )
        open_count = min(
            int(boundary_count),
            int(unresolved_degree),
            int(mainline_open_capacity),
        )
        if nonfixed_su11_count > 0:
            open_count = min(
                int(open_count),
                int(nonfixed_su11_count),
            )

        missing_branch_count = max(
            0,
            int(target_degree)
            - len(concrete_neighbors)
            - int(declared_branch_count)
            - int(open_count),
        )
        missing_branch_count = min(
            int(missing_branch_count),
            max(0, int(boundary_count) - int(open_count)),
        )
        for index in range(int(missing_branch_count)):
            fragment.branch_contracts.append(TopologyClosureSlot(
                slot_id=(
                    f"fragment:{int(fragment.fragment_id)}:branch:"
                    f"{int(root_id)}:inferred:{int(index)}:H"
                ),
                kind=TopologySlotKind.H.value,
                anchor_node_id=int(root_id),
                anchor_role="open_boundary_branch",
                required_su_sequence=[23, 22],
            ))
            branch_count_by_center[int(root_id)] += 1
        for index in range(int(open_count)):
            if len(roots) > 1:
                anchor_role = (
                    "left_mainline"
                    if int(root_index) == 0
                    else (
                        "right_mainline"
                        if int(root_index) == len(roots) - 1
                        else (
                            "left_mainline"
                            if int(index) == 0
                            else "right_mainline"
                        )
                    )
                )
            else:
                anchor_role = (
                    "left_mainline"
                    if int(index) == 0
                    else "right_mainline"
                )
            open_mainline_slots.append(TopologyClosureSlot(
                slot_id=(
                    f"fragment:{int(fragment.fragment_id)}:main:"
                    f"{int(root_id)}:{int(index)}:F1"
                ),
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(root_id),
                anchor_role=str(anchor_role),
                required_su_sequence=[23, 11],
            ))

    _normalize_fragment_fixed_connection_contracts(fragment, tokens_by_id)
    normalized_contracts = [
        dict(item or {})
        for item in list(
            fragment.evidence.get("fixed_connection_contracts", []) or []
        )
    ]
    fragment.evidence["fixed_connection_contracts"] = normalized_contracts

    existing_mainline_anchors = {
        int(normalize_slot(slot).anchor_node_id)
        for slot in open_mainline_slots
    }
    for contract in normalized_contracts:
        role_candidates = {
            str(value)
            for value in list(contract.get("role_candidates", []) or [])
        }
        connector_id = _safe_int(contract.get("connector_node_id"), -1)
        uses_mainline = bool(
            "mainline_only" in role_candidates
            or (
                connector_id == int(selected_primary_id)
                and "mainline_or_branch" in role_candidates
            )
        )
        base_ids = [
            int(value)
            for value in list(contract.get("base_path_node_ids", []) or [])
            if _safe_int(value, -1) >= 0
        ]
        tail_labels = [
            str(value)
            for value in list(contract.get("base_path_labels", []) or [])
            if str(value)
        ]
        tail_is_d2 = bool(
            tail_labels
            and tail_labels[-1] in GENERIC_OPEN_SUBTYPE_LIKE
        )
        tail_boundary_su_types = {
            _safe_int(value, -1)
            for value in list(contract.get("tail_open_boundary_su_types", []) or [])
        }
        # ``H1`` is normally a branch-only closure.  For a mainline fixed
        # route ending in a d2 unit, however, an unresolved SU23 neighbour is
        # the remaining physical endpoint and must be materialized as F1.
        # This is deliberately narrower than treating every H1 as a mainline
        # slot, which would alter genuine branch-only contracts.
        h1_d2_open_mainline = bool(
            uses_mainline
            and str(contract.get("tail_closure_kind", "")) in {"H1", "H_AFTER_PREFIX"}
            and tail_is_d2
            and 23 in tail_boundary_su_types
        )
        if (
            uses_mainline
            and (
                str(contract.get("tail_closure_kind", "")) == "OPEN_BOUNDARY"
                or h1_d2_open_mainline
            )
            and base_ids
            and int(base_ids[-1]) not in existing_mainline_anchors
        ):
            open_mainline_slots.append(TopologyClosureSlot(
                slot_id=(
                    f"fragment:{int(fragment.fragment_id)}:main:"
                    f"{int(base_ids[-1])}:open_boundary:F1"
                ),
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(base_ids[-1]),
                anchor_role="fixed_path_open_boundary",
                required_su_sequence=[23, 11],
            ))
            existing_mainline_anchors.add(int(base_ids[-1]))

    # This is the authoritative, degree-aware mainline contract for like24
    # fragments.  Preserve an empty result as well: an empty list can mean that
    # fixed edges and branch contracts have exhausted the center's degree, not
    # that the legacy contract still needs to be synthesized.
    fragment.open_mainline_slots = list(open_mainline_slots)
    fragment.open_slots = [str(slot.kind) for slot in open_mainline_slots]
    fragment.evidence["edge_semantics_finalized"] = True
    if open_mainline_slots:
        closure_count = len(open_mainline_slots)
        preferred = str(
            fragment.evidence.get("flex_reconcile_preferred_product_type", "")
        )
        main_kind = "other_open"
        if preferred == "flex_chain":
            main_kind = "d_like_open" if closure_count >= 2 else "b_like_open"
            fragment.evidence["flex_reconcile_main_kind"] = str(main_kind)
            fragment.evidence["flex_reconcile_priority"] = (
                40 if closure_count >= 2 else 30
            )
            fragment.evidence["flex_reconcile_roles"] = [
                str(main_kind),
                "double_end_flex_source" if closure_count >= 2 else "single_end_flex_source",
                "flex_preferred",
                "like24_open",
            ]
            fragment.evidence["flex_range_candidate_kind"] = str(main_kind)
            fragment.evidence["flex_range_can_supplement"] = True
        elif preferred == "side_chain":
            main_kind = "e_like_open"
            fragment.evidence["flex_reconcile_main_kind"] = str(main_kind)
            fragment.evidence["flex_reconcile_roles"] = [str(main_kind)]
            fragment.evidence["flex_range_candidate_kind"] = "explicit_side"
        fragment.evidence["required_mainline_closure_modes"] = [
            str(slot.kind) for slot in open_mainline_slots
        ]
        fragment.evidence["flex_range_mainline_closed"] = False
        fragment.evidence["flex_range_mainline_open"] = True
        fragment.evidence["flex_range_counts_as_explicit"] = False
        fragment.evidence["external_pending_profile"] = {
            "external_pending_kind": str(main_kind),
            "external_pending_reason": "nonfixed_mainline_open_boundary",
        }

    canonical_branch_slots: List[TopologyClosureSlot] = []
    for raw_slot in list(fragment.branch_contracts or []):
        slot = normalize_slot(raw_slot)
        if not slot.is_complete():
            canonical_branch_slots.append(slot)
    fragment.branch_contracts = list(canonical_branch_slots)
    fragment.branch_slots = [str(slot.kind) for slot in canonical_branch_slots]
    canonical_branch_map: Dict[str, List[str]] = {}
    for slot in canonical_branch_slots:
        anchor_key = (
            "center"
            if len(roots) == 1 and int(slot.anchor_node_id) == int(roots[0])
            else str(int(slot.anchor_node_id))
        )
        canonical_branch_map.setdefault(anchor_key, []).append(str(slot.kind))
    fragment.evidence["required_branch_kinds_by_anchor"] = canonical_branch_map
    return fragment


def _fragment_edge_semantic_errors(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> List[str]:
    claimed = {int(value) for value in list(fragment.claimed_node_ids or [])}
    context = {int(value) for value in list(fragment.context_node_ids or [])}
    errors: List[str] = []
    overlap = sorted(claimed & context)
    if overlap:
        errors.append(f"claimed_context_overlap:{overlap}")
    roots = _fragment_center_node_ids(fragment)
    intrinsic_group_ids = {
        int(value)
        for group in list(dict(fragment.evidence or {}).get("atomic_groups", []) or [])
        if set(int(item) for item in list(dict(group or {}).get("node_ids", []) or [])) & set(roots)
        for value in list(dict(group or {}).get("node_ids", []) or [])
    }
    unreachable = sorted(
        claimed - _fixed_reachable_node_ids(roots, tokens_by_id) - intrinsic_group_ids
    )
    if unreachable:
        errors.append(f"claimed_not_fixed_reachable:{unreachable}")
    for boundary in list(fragment.open_boundaries or []):
        neighbor_id = _safe_int(dict(boundary or {}).get("boundary_neighbor_id"), -1)
        if neighbor_id in claimed:
            errors.append(f"open_boundary_is_claimed:{neighbor_id}")
    for contract in list(
        dict(fragment.evidence or {}).get("fixed_connection_contracts", []) or []
    ):
        contract_local = dict(contract or {})
        connector_id = _safe_int(contract_local.get("connector_node_id"), -1)
        if not _normalize_fixed_side_class(contract_local.get("side_class")):
            errors.append(f"fixed_contract_side_class_invalid:{connector_id}")
        tail_kind = str(
            contract_local.get("tail_closure_kind", "none") or "none"
        )
        if (
            not bool(contract_local.get("externally_closed", False))
            and tail_kind not in VALID_OPEN_FIXED_TAIL_KINDS
        ):
            errors.append(
                f"fixed_contract_closure_state_invalid:{connector_id}:{tail_kind}"
            )
    intrinsic_group_ids = {
        int(value)
        for group in list(dict(fragment.evidence or {}).get("atomic_groups", []) or [])
        if set(int(item) for item in list(dict(group or {}).get("node_ids", []) or [])) & set(roots)
        for value in list(dict(group or {}).get("node_ids", []) or [])
    }
    for root_id in roots:
        root = tokens_by_id.get(int(root_id))
        if root is None or int(root.target_degree) <= 0:
            continue
        concrete_degree = sum(
            1
            for neighbor in list(root.neighbor_tokens or [])
            if (
                bool(neighbor.is_fixed)
                and _tokens_have_fixed_edge(
                    int(root_id), int(neighbor.node_id), tokens_by_id
                )
            )
            or (
                int(root_id) in intrinsic_group_ids
                and int(neighbor.node_id) in intrinsic_group_ids
            )
        )
        open_mainline_degree = sum(
            1
            for raw_slot in list(fragment.open_mainline_slots or [])
            if (
                not normalize_slot(raw_slot).is_complete()
                and int(normalize_slot(raw_slot).anchor_node_id) == int(root_id)
            )
        )
        open_branch_degree = sum(
            1
            for raw_slot in list(fragment.branch_contracts or [])
            if (
                not normalize_slot(raw_slot).is_complete()
                and int(normalize_slot(raw_slot).anchor_node_id) == int(root_id)
            )
        )
        projected_degree = int(
            concrete_degree + open_mainline_degree + open_branch_degree
        )
        if int(projected_degree) > int(root.target_degree):
            errors.append(
                f"center_degree_contract_overflow:{int(root_id)}:"
                f"projected={int(projected_degree)}:target={int(root.target_degree)}"
            )
    return errors


def _fragment_claimed_node_ids(fragment: TopologyFragment) -> List[int]:
    claimed = list(getattr(fragment, "claimed_node_ids", []) or [])
    if claimed:
        return [int(x) for x in claimed]
    return [int(x) for x in list(getattr(fragment, "source_node_ids", []) or [])]


def _attach_center_atomic_group(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    center_ids = [
        int(value)
        for value in list(dict(fragment.evidence or {}).get("center_node_ids", []) or [])
    ] or [int(fragment.fragment_id)]
    claimed_ids = set(_fragment_claimed_node_ids(fragment))
    candidate_ids = sorted(set(center_ids) | claimed_ids)
    groups_by_id: Dict[int, Dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        token = tokens_by_id.get(int(candidate_id))
        group_node_ids = [
            int(value)
            for value in list(dict(getattr(token, "meta", {}) or {}).get("atomic_group_node_ids", []) or [])
        ]
        if not group_node_ids:
            continue
        group_id = int(dict(token.meta or {}).get("atomic_group_id", min(group_node_ids)))
        if group_id in groups_by_id:
            continue
        groups_by_id[group_id] = {
            "group_id": group_id,
            "kind": str(dict(token.meta or {}).get("atomic_group_kind", "")),
            "node_ids": sorted(group_node_ids),
            "su_pair": [int(value) for value in list(dict(token.meta or {}).get("atomic_group_su_pair", []) or [])],
            "attachment_node_ids": sorted(set(group_node_ids) & claimed_ids),
            "center_owned": bool(set(group_node_ids) & set(center_ids)),
        }
    if not groups_by_id:
        return fragment

    # Atomic groups are ownership units.  Keep each group separate so a group
    # attached through a fragment endpoint cannot be flattened into an
    # unrelated center-owned group.
    atomic_groups = [groups_by_id[group_id] for group_id in sorted(groups_by_id)]
    owned_groups = [
        group
        for group in atomic_groups
        if set(int(value) for value in list(group.get("node_ids", []) or [])) & claimed_ids
    ]
    context_groups = [group for group in atomic_groups if group not in owned_groups]
    atomic_ids = {
        int(value)
        for group in owned_groups
        for value in list(group.get("node_ids", []) or [])
    }
    atomic_group_ids = {
        int(group["group_id"])
        for group in owned_groups
    }
    if not atomic_ids:
        if context_groups:
            fragment.evidence["context_atomic_groups"] = context_groups
        return fragment
    fragment.claimed_node_ids = _union_source_ids(fragment.claimed_node_ids, atomic_ids)
    fragment.source_node_ids = _union_source_ids(fragment.source_node_ids, atomic_ids)
    fragment.evidence["atomic_group_ids"] = sorted(atomic_group_ids)
    fragment.evidence["atomic_groups"] = owned_groups
    fragment.evidence["atomic_group_node_ids"] = sorted(atomic_ids)
    if context_groups:
        fragment.evidence["context_atomic_groups"] = context_groups

    center_groups = [
        group
        for group in owned_groups
        if set(int(value) for value in list(group.get("node_ids", []) or [])) & set(center_ids)
    ]
    center_atomic_ids = {
        int(value)
        for group in center_groups
        for value in list(group.get("node_ids", []) or [])
    }
    fragment.evidence["center_atomic_group_node_ids"] = sorted(center_atomic_ids)
    fragment_token = tokens_by_id.get(int(fragment.fragment_id))
    center_owns_atomic_group = bool(center_atomic_ids)
    if (
        center_owns_atomic_group
        and fragment_token is not None
        and int(fragment_token.su_type) == 14
    ):
        ordered_center_atomic_ids = _ordered_atomic_pair_ids(center_atomic_ids, tokens_by_id)
        fragment.evidence["center_node_ids"] = ordered_center_atomic_ids
        anchor_14_ids = sorted(
            int(node_id)
            for node_id in center_atomic_ids
            if int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)) == 14
        )
        fragment.evidence["required_branch_kinds_by_anchor"] = {
            str(node_id): ["H"]
            for node_id in anchor_14_ids
        }
        fragment.branch_slots = ["H" for _ in anchor_14_ids]
        fragment.branch_contracts = [
            TopologyClosureSlot(
                slot_id=f"atomic:{min(center_atomic_ids)}:H:{node_id}",
                kind=TopologySlotKind.H.value,
                anchor_node_id=int(node_id),
                anchor_role="unsaturated_branch_anchor",
                required_su_sequence=[23, 22],
            )
            for node_id in anchor_14_ids
        ]
    elif center_owns_atomic_group and fragment_token is not None:
        ordered_atomic_ids = _ordered_atomic_pair_ids(center_atomic_ids, tokens_by_id)
        fragment.mainline_node_ids = list(ordered_atomic_ids)
        fragment.mainline_labels = [
            str(int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)))
            for node_id in ordered_atomic_ids
        ]
        fragment.open_mainline_slots = [
            TopologyClosureSlot(
                slot_id=f"atomic:{ordered_atomic_ids[0]}:left:F1",
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(ordered_atomic_ids[0]),
                anchor_role="left_mainline",
                required_su_sequence=[23, 11],
            )
        ]
        right_su = int(getattr(tokens_by_id.get(int(ordered_atomic_ids[-1])), "su_type", -1))
        if right_su in {16, 18}:
            fragment.open_mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{ordered_atomic_ids[0]}:right:terminal",
                kind=TopologySlotKind.CLOSED_TERMINAL.value,
                anchor_node_id=int(ordered_atomic_ids[-1]),
                anchor_role="right_terminal",
                existing_node_ids=[int(ordered_atomic_ids[-1])],
                state=TopologySlotKind.CLOSED_TERMINAL.value,
            ))
        else:
            fragment.open_mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{ordered_atomic_ids[0]}:right:F1",
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(ordered_atomic_ids[-1]),
                anchor_role="right_mainline",
                required_su_sequence=[23, 11],
            ))
    return fragment


def _is_like24_fixed_partner_support(
    token: ClassificationToken,
    token_lookup: Dict[int, ClassificationToken],
) -> bool:
    if int(token.su_type) != 15:
        return False
    for neighbor in list(token.neighbor_tokens or []):
        if int(neighbor.su_type) != 14:
            continue
        primary = token_lookup.get(int(neighbor.node_id))
        if primary is None:
            continue
        if any(
            int(reverse.node_id) == int(token.node_id)
            and int(reverse.su_type) == 15
            for reverse in list(primary.neighbor_tokens or [])
        ):
            return True
    return False


def _make_fragment(
    token: ClassificationToken,
    *,
    family: str,
    class_type: str,
    topology_template: str,
    open_slots: Optional[List[str]] = None,
    branch_slots: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    source_node_ids: Optional[List[int]] = None,
    claimed_node_ids: Optional[List[int]] = None,
    refs: Optional[List[TopologyRef]] = None,
) -> TopologyFragment:
    evidence_payload = _augment_fragment_evidence(
        family=str(family),
        class_type=str(class_type),
        evidence=evidence,
        fixed_connector_inferred=bool(dict(token.meta or {}).get("fixed_connector_inferred", False)),
        topology_template=str(topology_template),
        open_slots=list(open_slots or []),
        branch_slots=list(branch_slots or []),
    )
    return TopologyFragment(
        fragment_id=int(token.node_id),
        family=str(family),
        class_type=str(class_type),
        topology_template=str(topology_template),
        main_path=list(
            refs or [TopologyRef(kind="node", role="center", su_type=int(token.su_type), node_id=int(token.node_id))]
        ),
        source_node_ids=list(source_node_ids or [int(token.node_id)]),
        claimed_node_ids=list(claimed_node_ids or [int(token.node_id)]),
        open_slots=list(open_slots or []),
        branch_slots=list(branch_slots or []),
        evidence=evidence_payload,
    )


def _flex_reconcile_main_kind_for_fragment(class_type: str) -> str:
    if class_type in FLEX_RECONCILE_EXPLICIT_FLEX_CLASSES:
        return "explicit_flex"
    if class_type in FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES:
        return "explicit_side"
    if class_type in FLEX_RECONCILE_B_LIKE_OPEN_CLASSES:
        return "b_like_open"
    if class_type in FLEX_RECONCILE_D_LIKE_OPEN_CLASSES:
        return "d_like_open"
    if class_type in FLEX_RECONCILE_E_LIKE_OPEN_CLASSES:
        return "e_like_open"
    if str(class_type) in {"24_A", "24_B"}:
        return "b_like_open"
    if str(class_type) in {"24_C", "24_D"}:
        return "d_like_open"
    return "other"


def _flex_reconcile_preferred_product_type_for_fragment(class_type: str) -> str:
    # A paired AB bridge is an open, dual-use mainline.  It is ring-first at
    # Stage3, with a side-chain fallback; it is not an explicit flex product.
    if str(class_type) == "24_AB_N_24_AB":
        return "ring_candidate"
    if class_type in FLEX_RECONCILE_FLEX_CHAIN_CLASSES:
        return "flex_chain"
    if class_type in FLEX_RECONCILE_SIDE_CHAIN_CLASSES:
        return "side_chain"
    if str(class_type) in {"24_A", "24_B", "24_C", "24_D"}:
        return "flex_chain"
    return "unknown"


def _flex_reconcile_evidence(
    *,
    family: str,
    class_type: str,
) -> Dict[str, Any]:
    main_kind = _flex_reconcile_main_kind_for_fragment(str(class_type))
    preferred_product_type = _flex_reconcile_preferred_product_type_for_fragment(str(class_type))
    is_like24_open = (
        str(family) in {"like24", "like24_secondary", "like24_bridge"}
        and str(main_kind) in {"b_like_open", "d_like_open"}
    )
    roles: List[str] = []
    if str(main_kind) == "explicit_flex":
        roles.extend(["explicit_flex", "flex_preferred"])
    if str(main_kind) == "b_like_open":
        roles.extend(["b_like_open", "single_end_flex_source"])
    if str(main_kind) == "d_like_open":
        roles.extend(["d_like_open", "double_end_flex_source"])
    if bool(is_like24_open):
        roles.append("like24_open")
    if str(preferred_product_type) == "flex_chain" and "flex_preferred" not in roles:
        roles.append("flex_preferred")
    priority = 99
    if str(main_kind) == "explicit_flex":
        priority = 0
    elif str(main_kind) == "b_like_open" and not bool(is_like24_open):
        priority = 10
    elif str(main_kind) == "d_like_open" and not bool(is_like24_open):
        priority = 20
    elif str(main_kind) == "b_like_open" and bool(is_like24_open):
        priority = 30
    elif str(main_kind) == "d_like_open" and bool(is_like24_open):
        priority = 40
    elif str(preferred_product_type) == "flex_chain":
        priority = 50
    return {
        "flex_reconcile_main_kind": str(main_kind),
        "flex_reconcile_preferred_product_type": str(preferred_product_type),
        "flex_reconcile_roles": sorted({str(role) for role in roles if str(role)}),
        "flex_reconcile_priority": int(priority),
    }


def _flex_range_mainline_evidence(
    *,
    family: str,
    class_type: str,
    topology_template: str,
    open_slots: Sequence[str],
    branch_slots: Sequence[str],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    del class_type
    preferred_product_type = str(evidence.get("flex_reconcile_preferred_product_type", "") or "")
    main_kind = str(evidence.get("flex_reconcile_main_kind", "") or "")
    fixed_connection_contracts = [
        dict(item or {})
        for item in list(evidence.get("fixed_connection_contracts", []) or [])
        if dict(item or {})
    ]
    required_mainline_closure_modes = [
        str(value)
        for value in list(evidence.get("required_mainline_closure_modes", []) or [])
        if str(value)
    ]
    has_open_mainline = bool(list(open_slots or []))
    branch_contracts = [
        dict(contract or {})
        for contract in list(fixed_connection_contracts or [])
        if "branch_only" in set(str(value) for value in list(dict(contract or {}).get("role_candidates", []) or []))
    ]
    open_branch_contracts = [
        dict(contract or {})
        for contract in list(branch_contracts or [])
        if not bool(dict(contract or {}).get("externally_closed", False))
    ]
    mainline_closed = bool(
        preferred_product_type in {"flex_chain", "side_chain"}
        and not bool(has_open_mainline)
        and not list(required_mainline_closure_modes or [])
    )
    if str(family) == "like24_bridge":
        mainline_source = "bridge_mainline" if bool(mainline_closed) else "bridge_open_mainline"
    elif bool(fixed_connection_contracts):
        mainline_source = "fixed_secondary_mainline" if bool(mainline_closed) else "fixed_secondary_open_mainline"
    elif preferred_product_type == "flex_chain":
        mainline_source = "basic_mainline" if bool(mainline_closed) else "basic_open_mainline"
    elif preferred_product_type == "side_chain":
        mainline_source = "side_mainline" if bool(mainline_closed) else "side_open_mainline"
    else:
        mainline_source = "non_chain_fragment"
    return {
        "flex_range_mainline_closed": bool(mainline_closed),
        "flex_range_mainline_open": bool(has_open_mainline),
        "flex_range_mainline_source": str(mainline_source),
        "flex_range_required_mainline_closure_modes": list(required_mainline_closure_modes or []),
        "flex_range_branch_slot_count": int(len(list(branch_slots or []))),
        "flex_range_fixed_branch_contract_count": int(len(list(branch_contracts or []))),
        "flex_range_open_fixed_branch_contract_count": int(len(list(open_branch_contracts or []))),
        "flex_range_supports_explicit_mainline": bool(
            preferred_product_type == "flex_chain"
            and main_kind == "explicit_flex"
        ),
        "flex_range_supports_supplemental_mainline": bool(
            preferred_product_type == "flex_chain"
            and main_kind in {"b_like_open", "d_like_open"}
        ),
        "flex_range_topology_template": str(topology_template or ""),
    }


def _flex_range_evidence(
    *,
    family: str,
    class_type: str,
    topology_template: str,
    open_slots: Sequence[str],
    branch_slots: Sequence[str],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    class_name = str(class_type)
    count_role = "inactive"
    candidate_kind = "other"
    counts_as_explicit = False
    can_supplement = False
    can_demote = False
    mainline_evidence = _flex_range_mainline_evidence(
        family=str(family),
        class_type=str(class_type),
        topology_template=str(topology_template),
        open_slots=list(open_slots or []),
        branch_slots=list(branch_slots or []),
        evidence=dict(evidence or {}),
    )
    mainline_closed = bool(mainline_evidence.get("flex_range_mainline_closed", False))
    mainline_open = bool(mainline_evidence.get("flex_range_mainline_open", False))
    supports_explicit = bool(mainline_evidence.get("flex_range_supports_explicit_mainline", False))
    supports_supplemental = bool(mainline_evidence.get("flex_range_supports_supplemental_mainline", False))

    if supports_explicit and mainline_closed:
        count_role = "explicit_mainline"
        candidate_kind = "explicit_flex"
        counts_as_explicit = True
        can_demote = bool(class_name in FLEX_RANGE_DEMOTABLE_EXPLICIT_CLASSES)
    elif supports_supplemental and mainline_open and class_name in FLEX_RECONCILE_B_LIKE_OPEN_CLASSES:
        count_role = "supplemental_candidate"
        candidate_kind = "b_like_open"
        can_supplement = True
    elif supports_supplemental and mainline_open and class_name in FLEX_RECONCILE_D_LIKE_OPEN_CLASSES:
        count_role = "supplemental_candidate"
        candidate_kind = "d_like_open"
        can_supplement = True
    elif class_name in FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES or class_name in FLEX_RECONCILE_E_LIKE_OPEN_CLASSES:
        count_role = "side_explicit"
        candidate_kind = "explicit_side"
    elif class_name in {"24_A", "24_B", "24_C", "24_D"}:
        count_role = "ring_candidate"
        candidate_kind = "ring_only_like24"

    return {
        "flex_range_count_role": str(count_role),
        "flex_range_candidate_kind": str(candidate_kind),
        "flex_range_counts_as_explicit": bool(counts_as_explicit),
        "flex_range_can_supplement": bool(can_supplement),
        "flex_range_can_demote": bool(can_demote),
        **dict(mainline_evidence or {}),
    }


def _external_pending_profile(
    *,
    family: str,
    class_type: str,
) -> Dict[str, Any]:
    del family
    class_name = str(class_type)
    pending_kind = "other"
    pending_reason = "remaining_external_halfclosed_open"

    if class_name in FLEX_RECONCILE_EXPLICIT_FLEX_CLASSES:
        pending_kind = "explicit_flex"
    elif class_name in {"25_AB", "25_CD"}:
        pending_kind = "pending_su25"
        pending_reason = "su25_closure_not_materialized"
    elif class_name in FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES or class_name in FLEX_RECONCILE_E_LIKE_OPEN_CLASSES:
        pending_kind = "explicit_side"
        pending_reason = "remaining_external_side_open"
    elif class_name in FLEX_RECONCILE_B_LIKE_OPEN_CLASSES:
        pending_kind = "b_like_open"
    elif class_name in FLEX_RECONCILE_D_LIKE_OPEN_CLASSES:
        pending_kind = "d_like_open"

    return {
        "external_pending_kind": str(pending_kind),
        "external_pending_reason": str(pending_reason),
    }


def _default_assembly_capabilities(
    *,
    family: str,
    class_type: str,
) -> Optional[Dict[str, Any]]:
    rules = _DEFAULT_ASSEMBLY_CAPABILITIES_BY_FAMILY.get(str(family))
    if rules is None:
        return None
    class_name = str(class_type)
    return {
        "flex_allowed": class_name in set(rules["flex"]),
        "side_allowed": class_name in set(rules["side"]),
        "ring_allowed": class_name in set(rules["ring"]),
        "ring_roles_by_skeleton": {},
    }


def _ring_roles_from_fragment_evidence(evidence: Dict[str, Any]) -> List[str]:
    ring_contracts = {
        str(key): dict(value or {})
        for key, value in dict(evidence.get("ring_contracts", {}) or {}).items()
    }
    if ring_contracts:
        roles: List[str] = []
        for _, contract in ring_contracts.items():
            if not bool(contract.get("eligible", False)):
                continue
            slot_roles = {str(item) for item in list(contract.get("slot_roles", []) or [])}
            if slot_roles & {"side_anchor", "side_anchor_pair", "side_anchor_core_pair"}:
                roles.append("side_anchor")
            if slot_roles & {"side_core", "side_pair_core", "side_anchor_core_pair"}:
                roles.append("side_middle")
            if "upper_lower_fixed_anchor" in slot_roles:
                roles.append("upper_anchor")
            if slot_roles & {"upper_lower_core", "upper_lower_pair_core"}:
                roles.append("upper_core")
            if "fused_outer_anchor" in slot_roles:
                roles.append("fused_anchor")
            if slot_roles & {"fused_bridgehead", "fused_bridgehead_pair", "fused_inner_core"}:
                roles.append("fused_core")
        if roles:
            return sorted({str(role) for role in roles})
    capabilities = {
        str(key): [str(value) for value in list(values or [])]
        for key, values in dict(evidence.get("ring_slot_capabilities", {}) or {}).items()
    }
    roles: List[str] = []
    if {"side_ring_anchor", "side_ring_anchor_pair", "side_ring_anchor_core_pair"} & set(list(capabilities.get("side", []) or [])):
        roles.append("side_anchor")
    if {"side_ring_core", "side_ring_anchor_core_pair"} & set(list(capabilities.get("side", []) or [])):
        roles.append("side_middle")
    if "upper_lower_fixed_anchor" in list(capabilities.get("upper_lower", []) or []):
        roles.append("upper_anchor")
    if "upper_lower_core" in list(capabilities.get("upper_lower", []) or []):
        roles.append("upper_core")
    if "fused_outer_anchor" in list(capabilities.get("fused", []) or []):
        roles.append("fused_anchor")
    if (
        "fused_inner_core" in list(capabilities.get("fused", []) or [])
        or "fused_bridgehead" in list(capabilities.get("fused", []) or [])
    ):
        roles.append("fused_core")
    return sorted({str(role) for role in roles})


def _ring_like24_units_for_fragment(
    *,
    family: str,
    class_type: str,
    evidence: Dict[str, Any],
) -> int:
    del evidence
    if str(family) not in {"like24", "like24_secondary", "like24_bridge"}:
        return 0
    if str(class_type) in PAIRED_LIKE24_CLASS_TYPES:
        return 2
    return 1


def _augment_fragment_evidence(
    *,
    family: str,
    class_type: str,
    evidence: Optional[Dict[str, Any]],
    fixed_connector_inferred: bool,
    topology_template: str,
    open_slots: Sequence[str],
    branch_slots: Sequence[str],
) -> Dict[str, Any]:
    payload = dict(evidence or {})
    payload.setdefault("fixed_connector_inferred", bool(fixed_connector_inferred))
    default_capabilities = _default_assembly_capabilities(
        family=str(family),
        class_type=str(class_type),
    )
    if default_capabilities is not None:
        payload.setdefault("assembly_capabilities", default_capabilities)
    for key, value in _flex_reconcile_evidence(
        family=str(family),
        class_type=str(class_type),
    ).items():
        payload.setdefault(str(key), value)
    for key, value in _flex_range_evidence(
        family=str(family),
        class_type=str(class_type),
        topology_template=str(topology_template),
        open_slots=list(open_slots or []),
        branch_slots=list(branch_slots or []),
        evidence=payload,
    ).items():
        payload.setdefault(str(key), value)
    payload.setdefault(
        "external_pending_profile",
        _external_pending_profile(
            family=str(family),
            class_type=str(class_type),
        ),
    )
    payload.setdefault("ring_roles", _ring_roles_from_fragment_evidence(payload))
    payload.setdefault(
        "ring_like24_units",
        int(_ring_like24_units_for_fragment(
            family=str(family),
            class_type=str(class_type),
            evidence=payload,
        )),
    )
    return payload


def _append_if_unclaimed(
    fragments: List[TopologyFragment],
    consumed: Set[int],
    fragment: TopologyFragment,
    *,
    claim_conflicts: Optional[List[Dict[str, Any]]] = None,
    record_conflict: bool = False,
) -> bool:
    node_ids = _fragment_claimed_node_ids(fragment)
    blocking = sorted(int(node_id) for node_id in node_ids if int(node_id) in consumed)
    center_ids = {
        int(value)
        for value in list(dict(getattr(fragment, "evidence", {}) or {}).get("center_node_ids", []) or [])
    } or {int(fragment.fragment_id)}
    atomic_ids = {
        int(value)
        for value in list(dict(getattr(fragment, "evidence", {}) or {}).get("atomic_group_node_ids", []) or [])
    }
    su24_fixed_support_ids = {
        int(value)
        for value in list(
            dict(getattr(fragment, "evidence", {}) or {}).get(
                "su24_fixed_support_ids", []
            )
            or []
        )
    }
    center_atomic_ids = {
        int(value)
        for value in list(
            dict(getattr(fragment, "evidence", {}) or {}).get(
                "center_atomic_group_node_ids", []
            )
            or []
        )
    }
    resource_blocking = [
        int(node_id)
        for node_id in blocking
        if int(node_id) not in center_ids
        and int(node_id) not in su24_fixed_support_ids
        and (
            int(node_id) not in atomic_ids
            or int(node_id) not in center_atomic_ids
        )
    ]
    if blocking and len(resource_blocking) == len(blocking):
        fragment.claimed_node_ids = [int(value) for value in node_ids if int(value) not in set(blocking)]
        fragment.source_node_ids = list(fragment.claimed_node_ids)
        fragment.context_node_ids = sorted({
            int(value)
            for value in list(getattr(fragment, "context_node_ids", []) or []) + list(blocking)
        })
        released_atomic_ids = set(blocking) & atomic_ids
        if released_atomic_ids:
            retained_groups = [
                dict(group or {})
                for group in list(
                    dict(getattr(fragment, "evidence", {}) or {}).get(
                        "atomic_groups", []
                    )
                    or []
                )
                if not released_atomic_ids.intersection({
                    int(value)
                    for value in list(dict(group or {}).get("node_ids", []) or [])
                })
            ]
            fragment.evidence["atomic_groups"] = retained_groups
            fragment.evidence["atomic_group_ids"] = sorted(
                int(group.get("group_id"))
                for group in retained_groups
                if group.get("group_id") is not None
            )
            fragment.evidence["atomic_group_node_ids"] = sorted({
                int(value)
                for group in retained_groups
                for value in list(group.get("node_ids", []) or [])
            })
            fragment.evidence["center_atomic_group_node_ids"] = sorted(
                center_atomic_ids - released_atomic_ids
            )
        node_ids = _fragment_claimed_node_ids(fragment)
        required_anchor_ids = {
            int(getattr(slot, "anchor_node_id", -1))
            for slot in list(getattr(fragment, "open_mainline_slots", []) or [])
            + list(getattr(fragment, "branch_contracts", []) or [])
            if int(getattr(slot, "anchor_node_id", -1)) >= 0
        }
        missing_anchor_ids = sorted(required_anchor_ids - set(node_ids))
        if missing_anchor_ids:
            blocking = list(missing_anchor_ids)
        else:
            blocking = []
    if blocking:
        if record_conflict and claim_conflicts is not None:
            claim_conflicts.append({
                "node_id": min(center_ids),
                "global_id": min(center_ids),
                "family": str(fragment.family),
                "class_type": str(fragment.class_type),
                "classification_status": "claim_conflict",
                "reject_bucket": "claim_conflicts",
                "source_node_ids": [int(x) for x in list(fragment.source_node_ids or [])],
                "claimed_node_ids": [int(x) for x in node_ids],
                "blocking_node_ids": [int(x) for x in blocking],
                "reason": "fragment_identity_conflict",
                "atomic_group_node_ids": sorted(atomic_ids),
            })
        return False
    fragments.append(fragment)
    consumed.update(node_ids)
    return True


def _record_reject(
    rejects: List[Dict[str, Any]],
    token: ClassificationToken,
    *,
    family: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    hop1_su = {
        str(k): int(v)
        for k, v in dict(token.hop1_counter or {}).items()
    }
    rejects.append({
        "node_id": int(token.node_id),
        "global_id": int(token.node_id),
        "su_type": int(token.su_type),
        "subtype": str(token.subtype),
        "family": str(family),
        "classification_status": "rule_reject",
        "reject_bucket": "rule_rejects",
        "hop1_ids": [int(x) for x in list(token.hop1_node_ids or [])],
        "hop1_su": hop1_su,
        "target_hop1_degree": int(token.target_degree),
        "target_fixed_anchor_count": int(token.fixed_anchor_count),
        "fixed_hop1_ids": [int(x) for x in list(token.fixed_neighbor_ids or [])],
        "fixed_connector_inferred": bool(dict(token.meta or {}).get("fixed_connector_inferred", False)),
        "anchor_mode": str(token.anchor_mode),
        "anchor_partition": str(token.anchor_partition),
        "reason": str(reason),
        "details": dict(details or {}),
    })


def _is_generic_open_neighbor(neighbor: NeighborToken) -> bool:
    return bool(is_abstract_open_nonfixed_semantic(neighbor))


def _is_terminal_branch_neighbor(neighbor: NeighborToken) -> bool:
    return (
        int(neighbor.su_type) in TERMINAL_NEIGHBORS
        or str(neighbor.subtype) in {"19d1", "20d1", "21d2"}
    )


def _is_like24_pair_neighbor(neighbor: NeighborToken, center_su: int) -> bool:
    if int(center_su) == 19:
        return str(neighbor.subtype) in {"single19d3", "double19d3"}
    if int(center_su) == 20:
        return str(neighbor.subtype) in {"single20d3", "double20d3"}
    return False


def _is_nonterminal_branch_neighbor(neighbor: NeighborToken) -> bool:
    if int(neighbor.su_type) == 11:
        return False
    if _is_terminal_branch_neighbor(neighbor):
        return False
    if is_abstract_open_nonfixed_semantic(neighbor):
        return False
    return True


def _matches_exact_su(neighbor: NeighborToken, allowed: Set[int]) -> bool:
    return int(neighbor.su_type) in set(int(x) for x in allowed)


def _matches_exact_subtype(neighbor: NeighborToken, allowed: Set[str]) -> bool:
    return str(neighbor.subtype) in set(str(x) for x in allowed)


def _matches_neighbor_pair(
    neighbors: Sequence[NeighborToken],
    left_predicate: Any,
    right_predicate: Any,
) -> bool:
    if len(list(neighbors or [])) != 2:
        return False
    left = list(neighbors or [])[0]
    right = list(neighbors or [])[1]
    return bool(
        (bool(left_predicate(left)) and bool(right_predicate(right)))
        or (bool(left_predicate(right)) and bool(right_predicate(left)))
    )


def _matches_all_generic_open(neighbors: Sequence[NeighborToken]) -> bool:
    return len(list(neighbors or [])) == 2 and all(_is_generic_open_neighbor(neighbor) for neighbor in list(neighbors or []))


def _side_rule_matches(
    remaining_neighbors: Sequence[NeighborToken],
    spec: Dict[str, Any],
    *,
    center_su: Optional[int] = None,
) -> bool:
    exact_neighbor_count = spec.get("exact_neighbor_count")
    remaining = list(remaining_neighbors or [])
    if exact_neighbor_count is not None:
        return int(len(remaining)) == int(exact_neighbor_count)
    if len(remaining) != 1:
        return False
    neighbor = remaining[0]
    if spec.get("exact_su_types") is not None and not _matches_exact_su(neighbor, set(spec.get("exact_su_types", set()) or set())):
        return False
    if spec.get("exact_subtypes") is not None and not _matches_exact_subtype(neighbor, set(spec.get("exact_subtypes", set()) or set())):
        return False
    if bool(spec.get("allow_generic_open")) and not _is_generic_open_neighbor(neighbor):
        return False
    if bool(spec.get("allow_like24_pair")) and center_su is not None and not _is_like24_pair_neighbor(neighbor, int(center_su)):
        return False
    return True


def resolve_fixed_connectors(
    token: ClassificationToken,
    *,
    allowed_su: Set[int],
    expected_count: int,
) -> Tuple[List[NeighborToken], Optional[str]]:
    if bool(dict(token.meta or {}).get("fixed_connector_forbidden", False)):
        return [], "fixed_connector_forbidden_by_partition_meta"
    partition = ""
    if int(token.su_type) == 19:
        partition = str(token.anchor_partition or "").strip().lower()
        if not partition:
            return [], "su19_anchor_partition_missing"
        if partition not in {"ether", "thio"}:
            return [], f"su19_anchor_partition_invalid:{partition}"
        partition_allowed = (
            {31}
            if partition == "thio"
            else ({2, 29} if int(token.target_degree) == 1 else {2, 28, 29})
        )
        allowed_su = set(allowed_su) & set(partition_allowed)
    fixed_neighbors = [neighbor for neighbor in list(token.neighbor_tokens or []) if bool(neighbor.is_fixed)]
    partition_incompatible = [
        neighbor
        for neighbor in fixed_neighbors
        if int(token.su_type) == 19
        and int(neighbor.su_type) in set(CONNECTOR_SU_TYPES)
        and int(neighbor.su_type) not in set(allowed_su)
    ]
    if partition_incompatible:
        return [], f"su19_partition_connector_mismatch:{partition}:" + ",".join(
            str(int(neighbor.su_type)) for neighbor in partition_incompatible
        )
    connectors = [neighbor for neighbor in fixed_neighbors if int(neighbor.su_type) in set(allowed_su)]
    connectors.sort(key=lambda item: int(item.node_id))
    if int(len(connectors)) != int(expected_count):
        return [], f"fixed_connector_count_mismatch:{len(connectors)}"
    unsupported_fixed = [
        neighbor
        for neighbor in fixed_neighbors
        if int(neighbor.su_type) not in set(allowed_su)
    ]
    if unsupported_fixed:
        return [], "unsupported_fixed_connector_types:" + ",".join(
            str(int(neighbor.su_type)) for neighbor in unsupported_fixed
        )
    return connectors, None


def _fixed_connector_slot_violation(
    subtype: str,
    connectors: Sequence[NeighborToken],
) -> Optional[str]:
    subtype_name = str(subtype)
    connector_types = [int(connector.su_type) for connector in list(connectors or [])]
    connector_ids = [int(connector.node_id) for connector in list(connectors or [])]
    if len(connector_ids) != len(set(connector_ids)):
        return "duplicate_fixed_connector_identity"
    if subtype_name.startswith("double19"):
        if len(connector_types) != 2:
            return f"double19_fixed_connector_count_mismatch:{len(connector_types)}"
        if any(su_type not in {2, 28, 29, 31} for su_type in connector_types):
            return "double19_fixed_connector_type_invalid"
        if connector_types.count(28) > 1:
            return "double19_second_fixed_slot_disallows_su28"
    elif subtype_name.startswith("double20"):
        if len(connector_types) != 2:
            return f"double20_fixed_connector_count_mismatch:{len(connector_types)}"
        if any(su_type not in {0, 27} for su_type in connector_types):
            return "double20_fixed_connector_type_invalid"
    elif subtype_name.startswith("single19"):
        if len(connector_types) != 1 or connector_types[0] not in {2, 28, 29, 31}:
            return "single19_fixed_connector_contract_invalid"
    elif subtype_name.startswith("single20"):
        if len(connector_types) != 1 or connector_types[0] not in {0, 27}:
            return "single20_fixed_connector_contract_invalid"
    return None


def _fixed_anchor_contract_violation(token: ClassificationToken) -> Optional[str]:
    subtype = str(token.subtype)
    if subtype.startswith("double19") or subtype.startswith("double20"):
        expected_mode = "double"
        expected_count = 2
    elif subtype.startswith("single19") or subtype.startswith("single20"):
        expected_mode = "single"
        expected_count = 1
    else:
        return None
    if str(token.anchor_mode) != expected_mode:
        return f"special_anchor_mode_mismatch:{token.anchor_mode or 'missing'}:{expected_mode}"
    if int(token.fixed_anchor_count) != int(expected_count):
        return f"target_fixed_anchor_count_mismatch:{token.fixed_anchor_count}:{expected_count}"
    return None


def _profile_remaining_neighbors(
    center_token: ClassificationToken,
    connector_token: ClassificationToken,
) -> List[NeighborToken]:
    return [
        neighbor
        for neighbor in list(connector_token.neighbor_tokens or [])
        if int(neighbor.node_id) != int(center_token.node_id)
    ]


VALID_FIXED_SIDE_CLASSES = frozenset({"M", "N", "P"})
VALID_OPEN_FIXED_TAIL_KINDS = frozenset({
    "H1",
    "H_AFTER_PREFIX",
    "paired_like24",
    "OPEN_BOUNDARY",
    "STRUCTURAL_BOUNDARY",
})


def _normalize_fixed_side_class(value: Any) -> str:
    side_class = str(value or "").strip().upper()
    return side_class if side_class in VALID_FIXED_SIDE_CLASSES else ""


def _fixed_side_matches(
    remaining: Sequence[NeighborToken],
    side_rules: Dict[str, Dict[str, Any]],
    *,
    center_su: Optional[int] = None,
) -> List[str]:
    remaining_neighbors = list(remaining or [])
    if (
        len(remaining_neighbors) == 1
        and is_open_boundary_semantic(remaining_neighbors[0])
    ):
        # A non-fixed connector-side edge is an unresolved boundary. Its SU
        # label cannot turn the edge into a concrete M/P path.
        return ["N"]
    return [
        str(side_class)
        for side_class, spec in dict(side_rules or {}).items()
        if _side_rule_matches(
            remaining_neighbors,
            dict(spec or {}),
            center_su=center_su,
        )
        and (
            str(side_class) == "N"
            or all(bool(neighbor.is_fixed) for neighbor in remaining_neighbors)
        )
    ]


def classify_double_d2_side(
    center_token: ClassificationToken,
    connector_neighbor: NeighborToken,
    tokens_by_id: Dict[int, ClassificationToken],
) -> Dict[str, Any]:
    connector_token = tokens_by_id.get(int(connector_neighbor.node_id))
    if connector_token is None:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "missing_connector_token",
        }
    subtype_rules = dict(DOUBLE_D2_SIDE_RULES.get(str(center_token.subtype), {}) or {})
    side_rules = dict(subtype_rules.get(int(connector_neighbor.su_type), {}) or {})
    if not side_rules:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "unsupported_connector_side_rule",
        }
    remaining = _profile_remaining_neighbors(center_token, connector_token)
    matches = _fixed_side_matches(remaining, side_rules)
    if len(matches) != 1:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "connector_side_ambiguous" if matches else "connector_side_unmatched",
            "remaining_neighbors": [neighbor.to_dict() for neighbor in remaining],
        }
    return {
        "connector_node_id": int(connector_neighbor.node_id),
        "connector_su_type": int(connector_neighbor.su_type),
        "side_class": str(matches[0]),
        "connector_token_subtype": str(connector_token.subtype),
        "remaining_node_ids": [int(neighbor.node_id) for neighbor in remaining],
        "remaining_neighbors": [neighbor.to_dict() for neighbor in remaining],
        "reject_reason": "",
    }


def classify_like24_fixed_side(
    center_token: ClassificationToken,
    connector_neighbor: NeighborToken,
    tokens_by_id: Dict[int, ClassificationToken],
) -> Dict[str, Any]:
    connector_token = tokens_by_id.get(int(connector_neighbor.node_id))
    if connector_token is None:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "missing_connector_token",
        }
    remaining = _profile_remaining_neighbors(center_token, connector_token)
    # A terminal SU28 attached to single19d3 is the fixed branch of the
    # 24_B+N ring form (11-24(?)-), not a P-type mainline terminus.  Keeping
    # this distinction here prevents the downstream secondary-rule resolver
    # from turning a ring-priority fragment into 24_AB_P/side_chain.
    if (
        str(center_token.subtype) == "single19d3"
        and int(connector_neighbor.su_type) == 28
        and not remaining
    ):
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": "N",
            "connector_token_subtype": str(connector_token.subtype),
            "remaining_node_ids": [],
            "remaining_neighbors": [],
            "fixed_terminal_branch": True,
            "reject_reason": "",
        }
    center_su = 19 if "19" in str(center_token.subtype) else 20
    family_rules = dict(LIKE24_FIXED_SIDE_RULES.get(int(center_su), {}) or {})
    side_rules = dict(family_rules.get(int(connector_neighbor.su_type), {}) or {})
    if not side_rules:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "unsupported_connector_side_rule",
        }
    matches = _fixed_side_matches(
        remaining,
        side_rules,
        center_su=int(center_su),
    )
    if len(matches) != 1:
        return {
            "connector_node_id": int(connector_neighbor.node_id),
            "connector_su_type": int(connector_neighbor.su_type),
            "side_class": None,
            "reject_reason": "connector_side_ambiguous" if matches else "connector_side_unmatched",
            "remaining_neighbors": [neighbor.to_dict() for neighbor in remaining],
        }
    return {
        "connector_node_id": int(connector_neighbor.node_id),
        "connector_su_type": int(connector_neighbor.su_type),
        "side_class": str(matches[0]),
        "connector_token_subtype": str(connector_token.subtype),
        "remaining_node_ids": [int(neighbor.node_id) for neighbor in remaining],
        "remaining_neighbors": [neighbor.to_dict() for neighbor in remaining],
        "reject_reason": "",
    }


def select_primary_fixed_side(connector_profiles: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for raw_profile in list(connector_profiles or []):
        profile = dict(raw_profile or {})
        side_class = _normalize_fixed_side_class(profile.get("side_class"))
        if not side_class:
            continue
        profile["side_class"] = str(side_class)
        valid.append(profile)
    if not valid:
        return None
    valid.sort(key=lambda item: (
        int(FIXED_SIDE_PRIORITY.get(_normalize_fixed_side_class(item.get("side_class")), 99)),
        -int(len(list(item.get("remaining_node_ids", []) or []))),
        int(item.get("connector_node_id", 10**9)),
    ))
    return dict(valid[0])


def _like24_base_claimed_node_ids(token: ClassificationToken) -> List[int]:
    return _union_source_ids(
        [int(token.node_id)],
        [
            int(neighbor.node_id)
            for neighbor in list(token.neighbor_tokens or [])
            if bool(neighbor.is_fixed)
        ],
    )


def _like24_owned_fixed_component_node_ids(
    token: ClassificationToken,
    tokens_by_id: Dict[int, ClassificationToken],
) -> List[int]:
    """Return fixed-node ownership coverage, never a connection route."""

    # A raw SU24 has a deliberately narrow fixed-support contract: only its
    # structural SU11 cap and (for role B/D) terminal support are owned by the
    # SU24 fragment.  Other 1-hop neighbours, including ``double19d3`` /
    # ``single19d3``, are ordinary open neighbours.  Treating every
    # ``is_fixed`` edge as fragment-owned would incorrectly create a
    # ``24(-double19d3-)`` classification and later allow that branch to be
    # materialized as if it were fixed.
    if int(token.su_type) == 24:
        binding = _raw_su24_support_binding(token)
        return _union_source_ids(
            [int(token.node_id)],
            list((binding or {}).get("fixed_support_ids", []) or []),
        )

    node_ids: Set[int] = set(_like24_base_claimed_node_ids(token))
    for neighbor in list(token.neighbor_tokens or []):
        if not bool(neighbor.is_fixed):
            continue
        connector = tokens_by_id.get(int(neighbor.node_id))
        if connector is None:
            continue
        node_ids.add(int(connector.node_id))
        node_ids.update(
            int(other.node_id)
            for other in list(connector.neighbor_tokens or [])
            if int(other.node_id) != int(token.node_id)
            and bool(other.is_fixed)
            and _tokens_have_fixed_edge(
                int(connector.node_id),
                int(other.node_id),
                tokens_by_id,
            )
        )
    return sorted(node_ids)


def _like24_secondary_claimed_node_ids(
    token: ClassificationToken,
    connector_profiles: Sequence[Dict[str, Any]],
    tokens_by_id: Dict[int, ClassificationToken],
    *,
    fixed_connection_contracts: Sequence[Dict[str, Any]],
    primary_connector_id: int = -1,
) -> List[int]:
    # A generic open neighbor behind a secondary fixed connector belongs to
    # the closure pool.  It is claimed later by that branch transaction.  The
    # same kind of neighbor behind the selected primary connector is part of
    # the materialized mainline and therefore remains fragment-owned.
    return _union_source_ids(
        _like24_owned_fixed_component_node_ids(token, tokens_by_id),
        [int(profile.get("connector_node_id", -1)) for profile in list(connector_profiles or [])],
        [
            int(node_id)
            for contract in list(fixed_connection_contracts or [])
            for node_id in list(dict(contract or {}).get("base_path_node_ids", []) or [])
        ],
    )


LIKE24_FIXED_OPEN_NEIGHBOR_LABELS = {
    "23",
    "24",
    "25",
    "19d2d3",
    "20d2d3",
    "21d2d3",
    "14",
    "15",
    "17",
}
LIKE24_FIXED_MAINLINE_ONLY_CLASSES = {
    "24_AB_M",
    "24_AB_P",
    "24_CD_M",
    "24_CD_M_FULL",
    "24_CD_P",
}
LIKE24_FIXED_MAINLINE_OR_BRANCH_CLASSES = {
    "24_AB_N",
    "24_CD_N",
}


def _profile_neighbor_label(value: Dict[str, Any]) -> str:
    item = dict(value or {})
    return str(item.get("subtype") or item.get("su_type") or "")


def _like24_fixed_role_candidates_for_contract(
    *,
    class_type: str,
    connector_node_id: int,
    side_class: str,
    primary_connector_id: int,
) -> List[str]:
    class_name = str(class_type)
    connector_id = int(connector_node_id)
    primary_id = int(primary_connector_id)
    if class_name == "24_CD_M_FULL":
        return ["mainline_only"]
    if class_name in LIKE24_FIXED_MAINLINE_ONLY_CLASSES:
        if primary_id >= 0 and connector_id >= 0 and connector_id != primary_id:
            return ["branch_only"]
        return ["mainline_only"]
    if class_name in LIKE24_FIXED_MAINLINE_OR_BRANCH_CLASSES:
        if primary_id >= 0 and connector_id >= 0 and connector_id != primary_id:
            return ["branch_only"]
        return ["mainline_or_branch"]
    if class_name in PAIRED_LIKE24_CLASS_TYPES:
        if str(side_class) in {"M", "P"}:
            return ["mainline_only"]
        return ["mainline_or_branch"]
    return []


def _paired_like24_partner_spec(
    *,
    connector_su: int,
    base_labels: Sequence[str],
) -> Dict[str, Any]:
    labels = [str(label) for label in list(base_labels or []) if str(label)]
    if int(connector_su) == 27:
        return {
            "partner_family": "20d3",
            "partner_kind": "paired_like24",
            "allowed_partner_connector_su_types": [27],
            "allowed_partner_tail_labels": ["single20d3", "double20d3"],
            "base_path_labels": list(labels),
        }
    if int(connector_su) in {29, 31}:
        return {
            "partner_family": "19d3",
            "partner_kind": "paired_like24",
            "allowed_partner_connector_su_types": [29, 31],
            "allowed_partner_tail_labels": ["single19d3", "double19d3"],
            "base_path_labels": list(labels),
        }
    return {}


def _build_fixed_connection_contract(
    profile: Dict[str, Any],
    *,
    class_type: str,
    primary_connector_id: int = -1,
) -> Dict[str, Any]:
    connector_node_id = _safe_int(dict(profile or {}).get("connector_node_id"), -1)
    connector_su_type = _safe_int(dict(profile or {}).get("connector_su_type"), -1)
    side_class = _normalize_fixed_side_class(dict(profile or {}).get("side_class"))
    if not side_class:
        raise ValueError("invalid_fixed_connection_side_class")
    contract = {
        "connector_node_id": int(connector_node_id),
        "connector_su_type": int(connector_su_type),
        "connector_token_subtype": str(dict(profile or {}).get("connector_token_subtype") or ""),
        "side_class": str(side_class),
        "role_candidates": _like24_fixed_role_candidates_for_contract(
            class_type=str(class_type),
            connector_node_id=int(connector_node_id),
            side_class=str(side_class),
            primary_connector_id=int(primary_connector_id),
        ),
        "base_path_labels": [],
        "base_path_node_ids": [],
        "tail_closure_kind": "none",
        "tail_open_reason": "",
        "externally_closed": False,
        "partner_spec": {},
        "generic_open_evidence_node_ids": [],
    }
    connector_label = str(contract["connector_token_subtype"] or contract["connector_su_type"] or "")
    connector_node_id = int(contract["connector_node_id"])
    remaining_neighbors = [
        dict(item or {})
        for item in list(dict(profile or {}).get("remaining_neighbors", []) or [])
    ]
    remaining_labels = [_profile_neighbor_label(item) for item in remaining_neighbors]
    remaining_node_ids = [_safe_int(item.get("node_id"), -1) for item in remaining_neighbors]
    contract["generic_open_evidence_node_ids"] = [
        int(node_id)
        for item, node_id in zip(remaining_neighbors, remaining_node_ids)
        if int(node_id) >= 0
        and is_open_boundary_semantic(dict(item or {}))
    ]
    side_class = str(contract["side_class"])
    connector_su = int(contract["connector_su_type"])

    base_labels: List[str] = [connector_label] if connector_label else []
    base_node_ids: List[int] = [connector_node_id] if connector_node_id >= 0 else []
    tail_kind = "none"
    tail_reason = ""
    externally_closed = False

    if side_class in {"M", "P"}:
        for item, label, node_id in zip(remaining_neighbors, remaining_labels, remaining_node_ids):
            if not bool(item.get("is_fixed", False)):
                continue
            if not label or node_id < 0:
                continue
            base_labels.append(str(label))
            base_node_ids.append(int(node_id))
        externally_closed = True
    elif side_class == "N":
        resolved_tail = False
        open_boundary_ids = [
            int(node_id)
            for item, node_id in zip(remaining_neighbors, remaining_node_ids)
            if int(node_id) >= 0 and is_open_boundary_semantic(item)
        ]
        for item, label, node_id in zip(remaining_neighbors, remaining_labels, remaining_node_ids):
            if not bool(item.get("is_fixed", False)):
                continue
            if connector_su in {0, 2} and str(label) in LIKE24_FIXED_OPEN_NEIGHBOR_LABELS:
                base_labels.append(str(label))
                if node_id >= 0:
                    base_node_ids.append(int(node_id))
                tail_kind = "H1"
                tail_reason = "fixed_branch_open_like_requires_H1"
                resolved_tail = True
                break
            if connector_su == 27 and str(label) == "single20d2":
                base_labels.append(str(label))
                if node_id >= 0:
                    base_node_ids.append(int(node_id))
                tail_kind = "H1"
                tail_reason = "fixed_branch_d2_requires_H1"
                resolved_tail = True
                break
            if connector_su in {29, 31} and str(label) == "single19d2":
                base_labels.append(str(label))
                if node_id >= 0:
                    base_node_ids.append(int(node_id))
                tail_kind = "H1"
                tail_reason = "fixed_branch_d2_requires_H1"
                resolved_tail = True
                break
            if connector_su == 27 and str(label) in {"single20d3", "double20d3"}:
                base_labels.append(str(label))
                if node_id >= 0:
                    base_node_ids.append(int(node_id))
                tail_kind = "paired_like24"
                tail_reason = "fixed_branch_requires_like24_partner"
                resolved_tail = True
                break
            if connector_su in {29, 31} and str(label) in {"single19d3", "double19d3"}:
                base_labels.append(str(label))
                if node_id >= 0:
                    base_node_ids.append(int(node_id))
                tail_kind = "paired_like24"
                tail_reason = "fixed_branch_requires_like24_partner"
                resolved_tail = True
                break
        if not resolved_tail and open_boundary_ids:
            tail_kind = "OPEN_BOUNDARY"
            tail_reason = "nonfixed_edge_stops_concrete_path"
            resolved_tail = True
        if not resolved_tail and remaining_labels:
            for item, label, node_id in zip(remaining_neighbors, remaining_labels, remaining_node_ids):
                if not bool(item.get("is_fixed", False)):
                    continue
                if not label or node_id < 0:
                    continue
                base_labels.append(str(label))
                base_node_ids.append(int(node_id))
        externally_closed = tail_kind == "none" and not bool(open_boundary_ids)

    contract["base_path_labels"] = [str(label) for label in base_labels if str(label)]
    contract["base_path_node_ids"] = [int(node_id) for node_id in base_node_ids if int(node_id) >= 0]
    contract["tail_closure_kind"] = str(tail_kind)
    contract["tail_open_reason"] = str(tail_reason)
    contract["externally_closed"] = bool(externally_closed)
    if str(tail_kind) == "paired_like24":
        contract["partner_spec"] = _paired_like24_partner_spec(
            connector_su=int(connector_su),
            base_labels=list(contract["base_path_labels"] or []),
        )
    return contract


def _build_fixed_connection_contracts(
    *,
    class_type: str,
    connector_profiles: Sequence[Dict[str, Any]],
    primary_connector: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    primary_connector_id = _safe_int(dict(primary_connector or {}).get("connector_node_id"), -1)
    return [
        _build_fixed_connection_contract(
            dict(profile or {}),
            class_type=str(class_type),
            primary_connector_id=int(primary_connector_id),
        )
        for profile in list(connector_profiles or [])
        if _normalize_fixed_side_class(dict(profile or {}).get("side_class"))
    ]


def _preferred_fixed_connection_role(
    *,
    class_type: str,
    primary_connector: Optional[Dict[str, Any]],
    fixed_connection_contracts: Sequence[Dict[str, Any]],
    realization_modes: Optional[Sequence[str]] = None,
) -> str:
    del realization_modes
    if not list(fixed_connection_contracts or []):
        return "none"
    primary = dict(primary_connector or {})
    primary_connector_id = _safe_int(primary.get("connector_node_id"), -1)
    contract = next(
        (
            dict(item or {})
            for item in list(fixed_connection_contracts or [])
            if _safe_int(dict(item or {}).get("connector_node_id"), -1) == int(primary_connector_id)
        ),
        {},
    )
    candidates = set(str(value) for value in list(contract.get("role_candidates", []) or []))
    if "mainline_only" in candidates:
        return "mainline"
    if "mainline_or_branch" in candidates:
        return "mainline"
    if "branch_only" in candidates:
        return "branch"
    return "none"


def _neighbor_or_dict_is_aromatic(value: Any) -> bool:
    try:
        if hasattr(value, "su_type"):
            su_type = int(getattr(value, "su_type", -1))
        else:
            su_type = int(dict(value or {}).get("su_type", -1))
    except Exception:
        su_type = -1
    return int(su_type) in set(AROMATIC_SU_TYPES)


def _like24_touches_aromatic_anchor(
    token: ClassificationToken,
    connector_profiles: Sequence[Dict[str, Any]],
) -> bool:
    if any(_neighbor_or_dict_is_aromatic(neighbor) for neighbor in list(token.neighbor_tokens or [])):
        return True
    for profile in list(connector_profiles or []):
        if any(
            _neighbor_or_dict_is_aromatic(neighbor)
            for neighbor in list(dict(profile or {}).get("remaining_neighbors", []) or [])
        ):
            return True
    return False


def _like24_required_mainline_closure_modes(
    *,
    class_type: str,
    preferred_fixed_role: str,
    realization_modes: Sequence[str],
) -> List[str]:
    class_name = str(class_type)
    fixed_role = str(preferred_fixed_role)
    modes = [str(value) for value in list(realization_modes or []) if str(value)]
    if class_name in {"24_AB_M", "24_CD_M_FULL"}:
        return []
    if class_name in {"24_AB_P", "24_CD_P"}:
        return []
    if class_name in {"24_AB_N_24_AB"}:
        return []
    if class_name in {"24_A", "24_B", "24_C", "24_D"}:
        return []
    if class_name in {"24_AB_N", "24_AB_N_24_CD"}:
        if fixed_role == "mainline":
            return ["F1"]
        if "branch_open" in modes:
            return []
        return ["F1"]
    if class_name in {"24_CD_M"}:
        return ["F1"]
    if class_name in {"24_CD_N", "24_CD_N_24_CD"}:
        if fixed_role == "mainline":
            return ["F1", "F1"]
        return ["F1", "F2"]
    return []


def _like24_allowed_pool_fillers_by_slot(
    *,
    class_type: str,
    bridgehead_role: str,
) -> Dict[str, List[str]]:
    class_name = str(class_type)
    bridge_role = str(bridgehead_role)
    fillers: Dict[str, List[str]] = {
        "branch_H": ["23-22"],
        "branch_H1": ["23-22"],
    }
    if class_name in {"24_A", "24_B", "24_AB_N"}:
        fillers["side_outer_anchor"] = ["24+11"]
        fillers["fused_outer_anchor"] = ["24+11"]
    if class_name in {"24_C", "24_D", "24_CD_N", "24_CD_N_24_CD", "24_CD_M", "24_AB_N_24_CD"}:
        fillers["side_core_segment"] = ["24", "23"]
    if class_name in {"24_CD_N", "24_CD_N_24_CD"}:
        fillers["upper_lower_core"] = ["24", "23"]
        fillers["fused_inner_segment"] = ["24", "23"]
    if bridge_role in {"fused_bridgehead_candidate", "fused_bridgehead_substitute_14", "fused_bridgehead_substitute_pair"}:
        fillers["fused_bridgehead"] = ["24"] if bridge_role != "fused_bridgehead_substitute_pair" else []
    return {
        str(key): [str(item) for item in list(values or [])]
        for key, values in fillers.items()
    }


def _like24_ring_skeleton_capabilities(
    *,
    class_type: str,
    ring_allowed_skeletons: Sequence[str],
    ring_slot_capabilities: Dict[str, List[str]],
    preferred_fixed_role: str,
) -> Dict[str, Dict[str, Any]]:
    class_name = str(class_type)
    allowed = {str(item) for item in list(ring_allowed_skeletons or [])}
    fixed_role = str(preferred_fixed_role)
    side_caps = [str(item) for item in list(ring_slot_capabilities.get("side", []) or [])]
    upper_caps = [str(item) for item in list(ring_slot_capabilities.get("upper_lower", []) or [])]
    fused_caps = [str(item) for item in list(ring_slot_capabilities.get("fused", []) or [])]
    out: Dict[str, Dict[str, Any]] = {}
    if "side" in allowed:
        out["side_ring_plan"] = {
            "eligible": True,
            "anchor_slots": [cap for cap in side_caps if "anchor" in cap],
            "core_slots": [cap for cap in side_caps if "core" in cap],
            "requires_real_like24": True,
            "allows_pool_23_bridges": True,
            "allows_pool_24_fill": True,
            "fixed_connection_guard": "ring_branch_only" if class_name in {"24_A", "24_B", "24_AB_N"} else fixed_role,
        }
    if "upper_lower" in allowed:
        out["upper_lower_ring_plan"] = {
            "eligible": True,
            "anchor_slots": [cap for cap in upper_caps if "anchor" in cap],
            "core_slots": [cap for cap in upper_caps if "core" in cap],
            "requires_real_like24": True,
            "allows_pool_23_bridges": True,
            "allows_pool_24_fill": True,
            "fixed_connection_guard": fixed_role,
        }
    if "fused" in allowed:
        out["fused_ring_plan"] = {
            "eligible": True,
            "anchor_slots": [cap for cap in fused_caps if "anchor" in cap],
            "bridgehead_slots": [cap for cap in fused_caps if "bridgehead" in cap],
            "core_slots": [cap for cap in fused_caps if "core" in cap],
            "requires_real_like24": True,
            "allows_pool_23_bridges": True,
            "allows_pool_24_fill": True,
            "fixed_connection_guard": "bridgehead_or_core",
        }
    return out


def _like24_direct_generic_count(token: ClassificationToken) -> int:
    return int(sum(
        1
        for neighbor in list(token.neighbor_tokens or [])
        if (
            _is_generic_open_neighbor(neighbor)
            and not is_visible_terminal_semantic(neighbor)
            and not is_visible_fixed_connector_semantic(neighbor)
        )
    ))


def _like24_expected_mainline_generic_count(
    class_type: str,
    connector_profiles: Sequence[Dict[str, Any]],
) -> int:
    if class_type in {"24_A", "24_B", "24_C", "24_D"}:
        return 1
    if class_type == "24_CD_N":
        # C/D contributes one open mainline edge at the like24 center while
        # the N fixed path contributes the other open endpoint.  Remaining
        # generic center edges are branches, not replacements for that first
        # mainline boundary.
        return 1
    if class_type in {"24_AB_M", "24_AB_N", "24_AB_P", "24_CD_M", "24_CD_M_FULL", "24_CD_N", "24_CD_P"}:
        return 0
    if class_type in PAIRED_LIKE24_CLASS_TYPES:
        return 0
    if list(connector_profiles or []):
        return 0
    return 1


def _like24_required_branch_kinds(
    token: ClassificationToken,
    *,
    class_type: str,
    connector_profiles: Sequence[Dict[str, Any]],
    realization_modes: Sequence[str],
) -> Dict[str, List[str]]:
    class_name = str(class_type)
    subtype = str(getattr(token, "subtype", ""))
    terminal_visible_count = int(sum(
        1 for neighbor in list(token.neighbor_tokens or [])
        if is_visible_terminal_semantic(neighbor)
    ))
    if class_name in {"24_D", "24_CD_P"} and subtype not in {"14"}:
        return {}
    if class_name in {"24_AB_M", "24_CD_M", "24_CD_M_FULL"}:
        has_closed_fixed_branch = any(
            str(dict(profile or {}).get("side_class", "")) in {"M", "P"}
            for profile in list(connector_profiles or [])[1:]
        )
        # ``double19d3`` / ``double20d3`` can already be fully accounted for by
        # their fixed branch contract.  In that case the remaining generic
        # neighbors belong to the main backbone/ring route, not to an extra H
        # branch.  Keep the closed-branch special case ahead of the generic
        # fallback so we do not accidentally manufacture a second side branch.
        if subtype in {"double19d3", "double20d3"} and has_closed_fixed_branch:
            return {}
        generic_count = _like24_direct_generic_count(token)
        if generic_count > 0:
            return {"center": ["H" for _ in range(int(generic_count))]}
        return {"center": ["H"]}
    if class_name in {"24_C", "24_D"} and subtype == "14":
        if terminal_visible_count > 0:
            return {"center": []}
        return {"center": ["H"]}
    if class_name in PAIRED_LIKE24_CLASS_TYPES:
        return {}
    generic_count = _like24_direct_generic_count(token)
    expected_mainline_generic = _like24_expected_mainline_generic_count(class_name, connector_profiles)
    if str(getattr(token, "subtype", "")) == "24":
        expected_mainline_generic = 1 if class_name in {"24_A", "24_B"} else 2
    unresolved_count = max(0, int(generic_count - expected_mainline_generic))
    if unresolved_count <= 0:
        return {}
    del realization_modes
    return {"center": ["H" for _ in range(int(unresolved_count))]}


def _like24_branch_closures_in_ring(
    *,
    class_type: str,
    required_branch_kinds_by_anchor: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    del class_type
    center_branches = [str(value) for value in list(required_branch_kinds_by_anchor.get("center", []) or [])]
    if not center_branches:
        return {"side": {}, "upper_lower": {}, "fused": {}}
    return {
        "side": {"center": list(center_branches)},
        "upper_lower": {"center": list(center_branches)},
        "fused": {"center": list(center_branches)},
    }


def _bridge_fragment_fixed_connection_contracts(
    *,
    class_type: str,
    left_fragment: TopologyFragment,
    right_fragment: TopologyFragment,
) -> List[Dict[str, Any]]:
    del class_type

    def _clone_contracts(fragment: TopologyFragment, center_node_id: int) -> List[Dict[str, Any]]:
        evidence = dict(getattr(fragment, "evidence", {}) or {})
        contracts = [
            dict(item or {})
            for item in list(evidence.get("fixed_connection_contracts", []) or [])
            if dict(item or {})
        ]
        if contracts:
            return [
                {
                    **dict(contract or {}),
                    "center_node_id": int(center_node_id),
                }
                for contract in contracts
            ]
        profiles = [
            dict(item or {})
            for item in list(evidence.get("connector_profiles", []) or [])
            if dict(item or {})
        ]
        if not profiles:
            return []
        return [
            {
                **dict(contract or {}),
                "center_node_id": int(center_node_id),
            }
            for contract in list(_build_fixed_connection_contracts(
                class_type=str(getattr(fragment, "class_type", "") or ""),
                connector_profiles=profiles,
            ) or [])
            if dict(contract or {})
        ]

    left_center_ids = [
        int(node_id)
        for node_id in list(dict(getattr(left_fragment, "evidence", {}) or {}).get("center_node_ids", []) or [])
        if _safe_int(node_id, -1) >= 0
    ]
    right_center_ids = [
        int(node_id)
        for node_id in list(dict(getattr(right_fragment, "evidence", {}) or {}).get("center_node_ids", []) or [])
        if _safe_int(node_id, -1) >= 0
    ]
    left_center_id = int(left_center_ids[0]) if left_center_ids else int(getattr(left_fragment, "fragment_id", -1))
    right_center_id = int(right_center_ids[0]) if right_center_ids else int(getattr(right_fragment, "fragment_id", -1))
    contracts = (
        _clone_contracts(left_fragment, left_center_id)
        + _clone_contracts(right_fragment, right_center_id)
    )
    pair_center_ids = {int(left_center_id), int(right_center_id)}
    for contract in contracts:
        base_ids = {
            int(value)
            for value in list(dict(contract or {}).get("base_path_node_ids", []) or [])
        }
        if (
            str(contract.get("tail_closure_kind", "")) == "paired_like24"
            and bool(base_ids & pair_center_ids)
        ):
            contract["tail_closure_kind"] = "none"
            contract["tail_open_reason"] = ""
            contract["externally_closed"] = True
            contract["paired_like24_resolved"] = True
            partner_spec = dict(contract.get("partner_spec", {}) or {})
            partner_spec["resolved"] = True
            partner_spec["resolved_partner_node_ids"] = sorted(base_ids & pair_center_ids)
            contract["partner_spec"] = partner_spec
    return contracts


def _like24_ring_slot_capabilities(
    *,
    class_type: str,
    ring_allowed_skeletons: Sequence[str],
    ring_slot_profiles: Sequence[str],
    bridgehead_role: str,
) -> Dict[str, List[str]]:
    class_name = str(class_type)
    allowed = {str(item) for item in list(ring_allowed_skeletons or [])}
    profiles = {str(item) for item in list(ring_slot_profiles or [])}
    capabilities: Dict[str, List[str]] = {
        "side": [],
        "upper_lower": [],
        "fused": [],
    }
    if "side" in allowed:
        if "side_outer_anchor" in profiles:
            capabilities["side"].append("side_ring_anchor")
        if "side_core_segment" in profiles or "side_pair_core_segment" in profiles:
            capabilities["side"].append("side_ring_core")
        if "side_anchor_pair" in profiles:
            capabilities["side"].extend(["side_ring_anchor", "side_ring_anchor_pair"])
        if "side_anchor_core_pair_segment" in profiles:
            capabilities["side"].extend(["side_ring_anchor", "side_ring_core", "side_ring_anchor_core_pair"])
    if "upper_lower" in allowed:
        if "upper_lower_fixed_anchor" in profiles:
            capabilities["upper_lower"].append("upper_lower_fixed_anchor")
        if "upper_lower_core_segment" in profiles or "upper_lower_pair_core_segment" in profiles:
            capabilities["upper_lower"].append("upper_lower_core")
    if "fused" in allowed:
        if "fused_outer_anchor" in profiles:
            capabilities["fused"].append("fused_outer_anchor")
        if bridgehead_role in {
            "fused_bridgehead_candidate",
            "fused_bridgehead_substitute_14",
            "fused_bridgehead_substitute_pair",
        }:
            capabilities["fused"].append("fused_bridgehead")
        if "fused_inner_segment" in profiles or "fused_pair_inner_segment" in profiles:
            capabilities["fused"].append("fused_inner_core")
    if class_name in {"24_CD_M", "24_AB_N_24_CD"} and "side" in allowed and "side_ring_core" not in capabilities["side"]:
        capabilities["side"].append("side_ring_core")
    return {
        str(key): sorted({str(value) for value in list(values or [])})
        for key, values in capabilities.items()
    }


def _like24_ring_path_capabilities(
    *,
    class_type: str,
    touches_aromatic_anchor: bool,
    fixed_anchor_eligible: bool,
) -> Dict[str, Any]:
    class_name = str(class_type)
    side_path_roles: List[str] = []
    upper_path_roles: List[str] = []
    fused_path_roles: List[str] = []

    if class_name in {"24_A", "24_B", "24_AB_N"}:
        if bool(touches_aromatic_anchor):
            side_path_roles.append("terminal_single")
            fused_path_roles.append("outer_anchor_single")
    if class_name == "24_AB_N_24_AB":
        # The paired AB mainline is an upper/lower path even when neither end
        # currently carries a fixed aromatic anchor.  Its -23-22 neighbors
        # stay branch slots and are resolved only after the ring candidate is
        # selected.  A side-chain route is still exposed separately through
        # the assembly capability flags.
        upper_path_roles.append("path_pair")
        if bool(touches_aromatic_anchor):
            side_path_roles.append("terminal_pair")
    if class_name in {"24_C", "24_D", "24_CD_N", "24_CD_M"}:
        side_path_roles.append("inner_single")
        fused_path_roles.append("inner_single")
    if class_name == "24_CD_M" and bool(touches_aromatic_anchor):
        side_path_roles.append("terminal_core_pair")
    if class_name == "24_AB_N_24_CD":
        if bool(touches_aromatic_anchor):
            side_path_roles.append("terminal_core_pair")
    if class_name == "24_CD_N_24_CD":
        side_path_roles.append("inner_pair")
        upper_path_roles.append("path_pair")
        fused_path_roles.append("inner_pair")
    if class_name == "24_A" and bool(fixed_anchor_eligible):
        upper_path_roles.append("fixed_top_anchor")
    if class_name in {"24_C", "24_D", "24_CD_N"}:
        upper_path_roles.append("path_single")

    return {
        "side": sorted({str(item) for item in side_path_roles if str(item)}),
        "upper_lower": sorted({str(item) for item in upper_path_roles if str(item)}),
        "fused": sorted({str(item) for item in fused_path_roles if str(item)}),
    }


def _like24_ring_role_priority(
    *,
    class_type: str,
    ring_slot_capabilities: Dict[str, List[str]],
) -> List[str]:
    class_name = str(class_type)
    priority: List[str] = []
    if "fused_bridgehead" in list(ring_slot_capabilities.get("fused", []) or []):
        priority.append("fused_bridgehead")
    if "fused_outer_anchor" in list(ring_slot_capabilities.get("fused", []) or []):
        priority.append("fused_outer_anchor")
    if "fused_inner_core" in list(ring_slot_capabilities.get("fused", []) or []):
        priority.append("fused_inner_core")
    if "upper_lower_fixed_anchor" in list(ring_slot_capabilities.get("upper_lower", []) or []):
        priority.append("upper_lower_fixed_anchor")
    if "upper_lower_core" in list(ring_slot_capabilities.get("upper_lower", []) or []):
        priority.append("upper_lower_core")
    if "side_ring_anchor" in list(ring_slot_capabilities.get("side", []) or []):
        priority.append("side_ring_anchor")
    if "side_ring_core" in list(ring_slot_capabilities.get("side", []) or []):
        priority.append("side_ring_core")
    if class_name in {"24_A", "24_B", "24_AB_N"}:
        priority = [item for item in priority if item != "upper_lower_core"]
    return [str(item) for item in priority]


def _make_ring_contract(
    *,
    eligible: bool = False,
    priority: int = 99,
    slot_roles: Optional[Sequence[str]] = None,
    slot_coverage: Optional[Sequence[str]] = None,
    slot_coverage_mode: str = "single_slot",
    required_real_like24_slots: Optional[Sequence[str]] = None,
    required_branch_closure_slots: Optional[Sequence[str]] = None,
    required_mainline_closure_slots: Optional[Sequence[str]] = None,
    required_branch_kinds_by_slot: Optional[Dict[str, Sequence[str]]] = None,
    optional_pool23_slots: Optional[Sequence[str]] = None,
    optional_pool24_slots: Optional[Sequence[str]] = None,
    pool24_requires_11_slots: Optional[Sequence[str]] = None,
    inner_slot_count: int = 0,
    bridgehead_mode: str = "none",
    bridgehead_kind: str = "none",
    fixed_connector_position: str = "none",
    fixed_connector_ring_eligible: bool = True,
    outer_anchor_must_touch_aromatic: bool = False,
    touches_aromatic_anchor: bool = False,
    render_policy: Optional[Dict[str, Any]] = None,
    blocking_reason: str = "",
) -> Dict[str, Any]:
    return {
        "eligible": bool(eligible),
        "priority": int(priority),
        "slot_roles": sorted({str(item) for item in list(slot_roles or []) if str(item)}),
        "slot_coverage": [str(item) for item in list(slot_coverage or []) if str(item)],
        "slot_coverage_mode": str(slot_coverage_mode or "single_slot"),
        "required_real_like24_slots": [str(item) for item in list(required_real_like24_slots or []) if str(item)],
        "required_branch_closure_slots": [str(item) for item in list(required_branch_closure_slots or []) if str(item)],
        "required_mainline_closure_slots": [str(item) for item in list(required_mainline_closure_slots or []) if str(item)],
        "required_branch_kinds_by_slot": {
            str(key): [str(item) for item in list(values or []) if str(item)]
            for key, values in dict(required_branch_kinds_by_slot or {}).items()
        },
        "optional_pool23_slots": [str(item) for item in list(optional_pool23_slots or []) if str(item)],
        "optional_pool24_slots": [str(item) for item in list(optional_pool24_slots or []) if str(item)],
        "allowed_pool23_slots": [str(item) for item in list(optional_pool23_slots or []) if str(item)],
        "allowed_pool24_slots": [str(item) for item in list(optional_pool24_slots or []) if str(item)],
        "pool24_requires_11_slots": [str(item) for item in list(pool24_requires_11_slots or []) if str(item)],
        "inner_slot_count": int(inner_slot_count),
        "bridgehead_mode": str(bridgehead_mode or "none"),
        "bridgehead_kind": str(bridgehead_kind or "none"),
        "fixed_connector_position": str(fixed_connector_position or "none"),
        "fixed_connector_ring_eligible": bool(fixed_connector_ring_eligible),
        "outer_anchor_must_touch_aromatic": bool(outer_anchor_must_touch_aromatic),
        "touches_aromatic_anchor": bool(touches_aromatic_anchor),
        "render_policy": dict(render_policy or {}),
        "blocking_reason": str(blocking_reason or ""),
    }


def _like24_required_branch_kinds_by_slot(
    *,
    class_type: str,
    branch_closures_in_ring: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    class_name = str(class_type)
    center_side = [str(value) for value in list(dict(branch_closures_in_ring.get("side", {}) or {}).get("center", []) or [])]
    center_upper = [str(value) for value in list(dict(branch_closures_in_ring.get("upper_lower", {}) or {}).get("center", []) or [])]
    center_fused = [str(value) for value in list(dict(branch_closures_in_ring.get("fused", {}) or {}).get("center", []) or [])]
    out: Dict[str, Dict[str, List[str]]] = {
        "side_ring_plan": {},
        "upper_lower_ring_plan": {},
        "fused_ring_plan": {},
    }
    if center_side and class_name not in {"24_A", "24_B", "24_AB_N"}:
        out["side_ring_plan"] = {
            "left_middle": list(center_side),
            "right_middle": list(center_side),
        }
    if center_upper and class_name in {"24_C", "24_D", "24_CD_N", "24_CD_N_24_CD"}:
        out["upper_lower_ring_plan"] = {
            "core_segment": list(center_upper),
        }
    if center_fused and class_name in {"24_C", "24_D", "24_CD_N", "24_CD_N_24_CD"}:
        out["fused_ring_plan"] = {
            "inner_2": list(center_fused),
            "inner_4": list(center_fused),
        }
    return out


def _like24_ring_contracts(
    *,
    class_type: str,
    subtype: str,
    has_11: bool,
    preferred_fixed_role: str,
    bridgehead_role: str,
    ring_block_reason: str,
) -> Dict[str, Dict[str, Any]]:
    class_name = str(class_type)
    subtype_name = str(subtype)
    fixed_role = str(preferred_fixed_role or "none")
    bridge_role = str(bridgehead_role or "none")

    contracts = {
        "side_ring_plan": _make_ring_contract(
            blocking_reason=ring_block_reason,
            render_policy={"suppress_generic_open": True},
        ),
        "upper_lower_ring_plan": _make_ring_contract(
            blocking_reason=ring_block_reason,
            render_policy={"suppress_generic_open": True},
        ),
        "fused_ring_plan": _make_ring_contract(
            blocking_reason=ring_block_reason,
            render_policy={"suppress_generic_open": True},
        ),
    }
    if ring_block_reason:
        return contracts

    if class_name in {"24_A", "24_B"} and has_11:
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=10 if subtype_name == "24" else 20,
            slot_roles=["side_anchor"],
            required_real_like24_slots=["left_anchor", "right_anchor"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        contracts["fused_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=10 if subtype_name == "24" else 20,
            slot_roles=["fused_outer_anchor"],
            required_real_like24_slots=["left_outer_anchor", "right_outer_anchor"],
            optional_pool23_slots=["inner_1", "inner_4"],
            optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            inner_slot_count=4,
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
    if class_name == "24_A" and subtype_name == "24" and has_11:
        contracts["upper_lower_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=0,
            slot_roles=["upper_lower_fixed_anchor"],
            required_real_like24_slots=["fixed_anchor_A", "return_anchor_A"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            fixed_connector_position="fixed_anchor",
            touches_aromatic_anchor=True,
            render_policy={"suppress_generic_open": True, "suppress_branch_closure": True},
        )
    if class_name == "24_AB_N" and has_11:
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=15,
            slot_roles=["side_anchor"],
            required_real_like24_slots=["left_anchor", "right_anchor"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=True,
            render_policy={"suppress_generic_open": True},
        )
    if class_name == "24_AB_N" and fixed_role not in {"branch", "dual"}:
        contracts["fused_ring_plan"] = _make_ring_contract(
            blocking_reason="mainline_anchor_not_ring_outer",
            render_policy={"suppress_generic_open": True},
        )
    if class_name in {"24_C", "24_D"}:
        side_priority = 30 if class_name == "24_C" else 35
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=side_priority,
            slot_roles=["side_core"],
            required_real_like24_slots=["left_middle", "right_middle"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        contracts["fused_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=20 if class_name == "24_C" else 25,
            slot_roles=["fused_inner_core"],
            required_real_like24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool23_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            inner_slot_count=4,
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        if class_name == "24_C" and subtype_name == "24":
            contracts["fused_ring_plan"] = _make_ring_contract(
                eligible=True,
                priority=5,
                slot_roles=["fused_bridgehead", "fused_inner_core"],
                required_real_like24_slots=["left_bridgehead", "right_bridgehead"],
                optional_pool23_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
                optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
                inner_slot_count=4,
                bridgehead_mode="single_24_c",
                fixed_connector_position=fixed_role,
                touches_aromatic_anchor=bool(has_11),
                render_policy={"suppress_generic_open": True, "suppress_branch_closure": True},
            )
        if class_name == "24_C" and subtype_name == "24":
            contracts["upper_lower_ring_plan"] = _make_ring_contract(
                eligible=True,
                priority=15,
                slot_roles=["upper_lower_core"],
                required_real_like24_slots=["core_segment"],
                optional_pool23_slots=["left_bridge", "right_bridge"],
                optional_pool24_slots=["core_segment"],
                fixed_connector_position=fixed_role,
                touches_aromatic_anchor=bool(has_11),
                render_policy={"suppress_generic_open": True},
            )
        elif class_name == "24_D":
            contracts["upper_lower_ring_plan"] = _make_ring_contract(
                eligible=True,
                priority=25,
                slot_roles=["upper_lower_core"],
                required_real_like24_slots=["core_segment"],
                optional_pool23_slots=["left_bridge", "right_bridge"],
                optional_pool24_slots=["core_segment"],
                fixed_connector_position=fixed_role,
                touches_aromatic_anchor=bool(has_11),
                render_policy={"suppress_generic_open": True},
            )
    if class_name == "24_CD_N":
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=45,
            slot_roles=["side_core"],
            required_real_like24_slots=["left_middle", "right_middle"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        contracts["upper_lower_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=20,
            slot_roles=["upper_lower_core"],
            required_real_like24_slots=["core_segment"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["core_segment"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        contracts["fused_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=30,
            slot_roles=["fused_inner_core"],
            required_real_like24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool23_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            inner_slot_count=4,
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
    if class_name == "24_CD_M":
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=55,
            slot_roles=["side_core"],
            required_real_like24_slots=["left_middle", "right_middle"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position=fixed_role,
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
    if class_name == "24_AB_N_24_CD":
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=12,
            slot_roles=["side_anchor", "side_core", "side_anchor_core_pair"],
            slot_coverage=["left_anchor", "left_middle"],
            slot_coverage_mode="anchor_core_pair",
            required_real_like24_slots=["left_anchor", "left_middle"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position="dual",
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
    if class_name == "24_CD_N_24_CD":
        contracts["side_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=5,
            slot_roles=["side_pair_core", "side_core"],
            slot_coverage=["left_middle", "right_middle"],
            slot_coverage_mode="pair_core",
            required_real_like24_slots=["left_middle", "right_middle"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["left_middle", "right_middle"],
            fixed_connector_position="dual",
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        contracts["upper_lower_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=5,
            slot_roles=["upper_lower_core", "upper_lower_pair_core"],
            slot_coverage=["core_segment"],
            slot_coverage_mode="pair_core",
            required_real_like24_slots=["core_segment"],
            optional_pool23_slots=["left_bridge", "right_bridge"],
            optional_pool24_slots=["core_segment"],
            fixed_connector_position="dual",
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True},
        )
        fused_roles = ["fused_inner_core"]
        fused_priority = 18
        fused_mode = "inner_pair_core"
        if bridge_role == "fused_bridgehead_substitute_pair":
            fused_roles = ["fused_bridgehead_pair", "fused_inner_core"]
            fused_priority = 0
            fused_mode = "pair_14_bridgehead"
        contracts["fused_ring_plan"] = _make_ring_contract(
            eligible=True,
            priority=fused_priority,
            slot_roles=fused_roles,
            slot_coverage=(
                ["left_bridgehead", "right_bridgehead"]
                if "fused_bridgehead_pair" in fused_roles
                else ["inner_2", "inner_3"]
            ),
            slot_coverage_mode=(
                "bridgehead_pair"
                if "fused_bridgehead_pair" in fused_roles
                else "pair_inner_core"
            ),
            required_real_like24_slots=["left_bridgehead", "right_bridgehead"] if "fused_bridgehead_pair" in fused_roles else ["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool23_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"],
            inner_slot_count=4,
            bridgehead_mode=fused_mode,
            fixed_connector_position="dual",
            touches_aromatic_anchor=bool(has_11),
            render_policy={"suppress_generic_open": True, "suppress_branch_closure": "fused_bridgehead_pair" in fused_roles},
        )
    return contracts

def _like24_suppressed_support_node_ids(token: ClassificationToken) -> List[int]:
    out: List[int] = []
    for neighbor in list(token.neighbor_tokens or []):
        if is_visible_terminal_semantic(neighbor):
            continue
        if is_visible_fixed_connector_semantic(neighbor):
            continue
        if int(neighbor.su_type) == 11:
            continue
        if is_abstract_open_nonfixed_semantic(neighbor):
            out.append(int(neighbor.node_id))
    return sorted({int(node_id) for node_id in out})


def _like24_ring_evidence(
    token: ClassificationToken,
    *,
    class_type: str,
    topology_template: str,
    connector_profiles: Sequence[Dict[str, Any]],
    primary_connector: Optional[Dict[str, Any]] = None,
    realization_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    subtype = str(token.subtype)
    has_11 = any(int(neighbor.su_type) == 11 for neighbor in list(token.neighbor_tokens or []))
    touches_aromatic_anchor = _like24_touches_aromatic_anchor(token, connector_profiles)
    terminal_visible_count = int(sum(
        1 for neighbor in list(token.neighbor_tokens or [])
        if is_visible_terminal_semantic(neighbor)
    ))
    fixed_connection_contracts = _build_fixed_connection_contracts(
        class_type=str(class_type),
        connector_profiles=connector_profiles,
        primary_connector=primary_connector,
    )
    preferred_fixed_role = _preferred_fixed_connection_role(
        class_type=str(class_type),
        primary_connector=primary_connector,
        fixed_connection_contracts=fixed_connection_contracts,
        realization_modes=realization_modes,
    )
    primary_connector_id = _safe_int(dict(primary_connector or {}).get("connector_node_id"), -1)
    primary_contract = next(
        (
            dict(item or {})
            for item in list(fixed_connection_contracts or [])
            if _safe_int(dict(item or {}).get("connector_node_id"), -1) == int(primary_connector_id)
        ),
        {},
    )
    primary_role_candidates = [
        str(value)
        for value in list(dict(primary_contract or {}).get("role_candidates", []) or [])
        if str(value)
    ]
    ring_eligibility: List[str] = []
    ring_slot_signatures: List[str] = []
    ring_allowed_skeletons: List[str] = []
    ring_forbidden_skeletons: List[str] = []
    bridgehead_role = "none"
    inner_ring_open_slots = 0
    mainline_open_pattern = "closed_like_A"
    ring_plan_role = "non_ring"
    fixed_anchor_eligible = False
    outer_anchor_eligible = False
    pair_core_eligible = False
    bridgehead_eligible = False
    ring_slot_profiles: List[str] = []
    masked_ring_open_sides: List[str] = []
    def _allow(*names: str) -> None:
        for name in list(names or []):
            if str(name) not in ring_allowed_skeletons:
                ring_allowed_skeletons.append(str(name))

    def _forbid(*names: str) -> None:
        for name in list(names or []):
            if str(name) not in ring_forbidden_skeletons:
                ring_forbidden_skeletons.append(str(name))

    if class_type in {"24_A", "24_B", "24_AB_N"}:
        ring_eligibility.extend(["side", "fused"])
        ring_slot_signatures.extend(["side_anchor", "side_anchor_outer", "fused_outer_anchor"])
        ring_slot_profiles.extend(["side_outer_anchor", "fused_outer_anchor"])
        _allow("side", "fused")
        outer_anchor_eligible = True
        ring_plan_role = "ring_outer_anchor"
        mainline_open_pattern = "ring_only_anchor"
        masked_ring_open_sides.append("ring_facing")
    if class_type == "24_A" and has_11:
        ring_eligibility.append("upper_lower")
        ring_slot_signatures.extend(["upper_anchor", "upper_lower_anchor_fixed_candidate", "upper_lower_fixed_anchor"])
        ring_slot_profiles.extend(["upper_lower_fixed_anchor"])
        _allow("upper_lower")
        fixed_anchor_eligible = True
    if class_type == "24_AB_N":
        _forbid("upper_lower")
        mainline_open_pattern = "left_closed_right_open"
    if class_type in {"24_C", "24_D"}:
        ring_eligibility.extend(["side", "fused"])
        ring_slot_signatures.extend([
            "side_core",
            "side_core_primary",
            "fused_inner_core",
            "ring_only_core",
        ])
        ring_slot_profiles.extend([
            "side_core_segment",
            "fused_inner_segment",
        ])
        _allow("side", "fused")
        ring_plan_role = "ring_core"
        mainline_open_pattern = "double_open_ring_core"
        masked_ring_open_sides.extend(["left_ring", "right_ring"])
        if subtype == "14" and class_type == "24_C" and terminal_visible_count == 0:
            bridgehead_role = "fused_bridgehead_substitute_14"
            ring_slot_signatures.extend(["fused_bridgehead_substitute_14", "fused_bridgehead_candidate"])
            ring_slot_profiles.append("fused_bridgehead")
            bridgehead_eligible = True
        elif class_type == "24_C":
            bridgehead_role = "fused_bridgehead_candidate"
            ring_slot_signatures.append("fused_bridgehead_candidate")
            bridgehead_eligible = True
    if class_type == "24_CD_N":
        ring_eligibility.extend(["side", "upper_lower", "fused"])
        ring_slot_signatures.extend([
            "side_core",
            "side_core_primary",
            "upper_lower_core",
            "fused_inner_core",
            "ring_only_core",
        ])
        ring_slot_profiles.extend([
            "side_core_segment",
            "upper_lower_core_segment",
            "fused_inner_segment",
        ])
        _allow("side", "upper_lower", "fused")
        ring_plan_role = "ring_core"
        mainline_open_pattern = "double_open_ring_core"
        masked_ring_open_sides.extend(["left_ring", "right_ring"])
        if subtype == "14" and terminal_visible_count == 0:
            bridgehead_role = "fused_bridgehead_substitute_14"
            ring_slot_signatures.extend(["fused_bridgehead_substitute_14", "fused_bridgehead_candidate"])
            ring_slot_profiles.append("fused_bridgehead")
            bridgehead_eligible = True
        else:
            bridgehead_role = "fused_bridgehead_candidate"
            ring_slot_signatures.append("fused_bridgehead_candidate")
            bridgehead_eligible = True
    if class_type in {"24_CD_M", "24_AB_N_24_CD"}:
        ring_eligibility.append("side")
        ring_slot_signatures.extend(["side_core", "side_pair_core"])
        ring_slot_profiles.extend(["side_core_segment"])
        _allow("side")
        ring_plan_role = "side_core_only"
        mainline_open_pattern = "left_open_like_B"
        masked_ring_open_sides.extend(["left_ring", "right_ring"])
    if class_type == "24_CD_M":
        _forbid("upper_lower", "fused")
    if class_type == "24_CD_N_24_CD":
        ring_eligibility.extend(["side", "upper_lower", "fused"])
        ring_slot_signatures.extend([
            "side_pair_core",
            "upper_lower_core",
            "upper_lower_pair_core",
            "fused_inner_core",
            "fused_pair_core",
            "cd_cd_pair_core",
        ])
        ring_slot_profiles.extend([
            "side_pair_core_segment",
            "upper_lower_pair_core_segment",
            "fused_pair_inner_segment",
        ])
        _allow("side", "upper_lower", "fused")
        inner_ring_open_slots = 1
        pair_core_eligible = True
        ring_plan_role = "pair_core"
        mainline_open_pattern = "double_open_pair_core"
        masked_ring_open_sides.extend(["left_ring", "right_ring"])
    if class_type in {"24_AB_M", "24_AB_P", "24_CD_M_FULL", "24_CD_P"}:
        _forbid("side", "upper_lower", "fused")

    required_branch_kinds_by_anchor = _like24_required_branch_kinds(
        token,
        class_type=str(class_type),
        connector_profiles=connector_profiles,
        realization_modes=list(realization_modes or []),
    )
    try:
        primary_connector_su = int(dict(primary_connector or {}).get("connector_su_type", -1))
    except Exception:
        primary_connector_su = -1
    ring_block_reason = ""
    primary_forces_mainline = (
        "mainline_only" in set(primary_role_candidates)
        or str(preferred_fixed_role) == "mainline"
    )
    if primary_connector_su in {0, 2} and bool(primary_forces_mainline):
        ring_block_reason = "mainline_fixed_connector_0_2"
        ring_eligibility = []
        ring_allowed_skeletons = []
        ring_slot_signatures = []
        ring_slot_profiles = []
        ring_plan_role = "non_ring_blocked"
        bridgehead_role = "none"
        inner_ring_open_slots = 0
        fixed_anchor_eligible = False
        outer_anchor_eligible = False
        pair_core_eligible = False
        bridgehead_eligible = False
        masked_ring_open_sides = []
        _forbid("side", "upper_lower", "fused")

    branch_closures_in_ring = _like24_branch_closures_in_ring(
        class_type=str(class_type),
        required_branch_kinds_by_anchor=required_branch_kinds_by_anchor,
    )
    branch_kinds_by_slot = _like24_required_branch_kinds_by_slot(
        class_type=str(class_type),
        branch_closures_in_ring=branch_closures_in_ring,
    )
    ring_slot_capabilities = _like24_ring_slot_capabilities(
        class_type=str(class_type),
        ring_allowed_skeletons=ring_allowed_skeletons,
        ring_slot_profiles=ring_slot_profiles,
        bridgehead_role=str(bridgehead_role),
    )
    ring_role_priority = _like24_ring_role_priority(
        class_type=str(class_type),
        ring_slot_capabilities=ring_slot_capabilities,
    )
    required_mainline_closure_modes = _like24_required_mainline_closure_modes(
        class_type=str(class_type),
        preferred_fixed_role=str(preferred_fixed_role),
        realization_modes=list(realization_modes or []),
    )
    allowed_pool_fillers_by_slot = _like24_allowed_pool_fillers_by_slot(
        class_type=str(class_type),
        bridgehead_role=str(bridgehead_role),
    )
    ring_skeleton_capabilities = _like24_ring_skeleton_capabilities(
        class_type=str(class_type),
        ring_allowed_skeletons=ring_allowed_skeletons,
        ring_slot_capabilities=ring_slot_capabilities,
        preferred_fixed_role=str(preferred_fixed_role),
    )
    ring_path_capabilities = _like24_ring_path_capabilities(
        class_type=str(class_type),
        touches_aromatic_anchor=bool(touches_aromatic_anchor),
        fixed_anchor_eligible=bool(fixed_anchor_eligible),
    )
    fixed_connection_context = {
        "preferred_role": str(preferred_fixed_role),
        "primary_connector_id": int(primary_connector_id),
        "primary_role_candidates": list(primary_role_candidates or []),
        "selected_primary_connector_side_class": str(dict(primary_connector or {}).get("side_class", "")),
        "mainline_blocked_for_ring": bool(primary_connector_su in {0, 2} and bool(primary_forces_mainline)),
    }
    bridgehead_policy = {
        "role": str(bridgehead_role),
        "eligible": bool(bridgehead_eligible),
        "substitutable_with_14_pair": bool(str(bridgehead_role) in {"fused_bridgehead_substitute_14", "fused_bridgehead_substitute_pair"}),
    }
    ring_contracts = _like24_ring_contracts(
        class_type=str(class_type),
        subtype=str(subtype),
        has_11=bool(has_11),
        preferred_fixed_role=str(preferred_fixed_role),
        bridgehead_role=str(bridgehead_role),
        ring_block_reason=str(ring_block_reason),
    )
    bridgehead_kind = "none"
    if str(bridgehead_role) == "fused_bridgehead_candidate":
        bridgehead_kind = "real_24c"
    elif str(bridgehead_role) in {"fused_bridgehead_substitute_14", "fused_bridgehead_substitute_pair"}:
        bridgehead_kind = "substitute_14_pair"
    for skeleton_kind, contract in list(dict(ring_contracts or {}).items()):
        contract_local = dict(contract or {})
        slot_branch_map = {
            str(key): [str(item) for item in list(values or []) if str(item)]
            for key, values in dict(branch_kinds_by_slot.get(str(skeleton_kind), {}) or {}).items()
        }
        contract_local["required_branch_kinds_by_slot"] = slot_branch_map
        contract_local["required_branch_closure_slots"] = sorted(slot_branch_map.keys())
        contract_local["required_mainline_closure_slots"] = list(required_mainline_closure_modes or [])
        contract_local["fixed_connector_ring_eligible"] = bool(
            not bool(primary_connector_su in {0, 2} and bool(primary_forces_mainline))
            and not bool(contract_local.get("blocking_reason", ""))
        )
        if str(skeleton_kind) in {"side_ring_plan", "fused_ring_plan"}:
            contract_local["outer_anchor_must_touch_aromatic"] = True
        if str(skeleton_kind) == "upper_lower_ring_plan" and str(class_type) == "24_A":
            contract_local["pool24_requires_11_slots"] = ["fixed_anchor_A", "return_anchor_A"]
        elif str(skeleton_kind) in {"side_ring_plan", "fused_ring_plan"} and str(class_type) in {"24_A", "24_B", "24_AB_N"}:
            contract_local["pool24_requires_11_slots"] = ["left_anchor", "right_anchor", "left_outer_anchor", "right_outer_anchor"]
        else:
            contract_local.setdefault("pool24_requires_11_slots", [])
        contract_local["bridgehead_kind"] = str(bridgehead_kind)
        ring_contracts[str(skeleton_kind)] = contract_local
    # Keep assembly capabilities as explicit facts.  preferred_fixed_role and
    # preferred_product_type are ordering hints; they must not erase a valid
    # side/flex fallback for a Like24 fragment.
    ring_roles_by_skeleton = {
        str(skeleton): {
            "allowed": bool(dict(contract or {}).get("eligible", False)),
            "roles": [
                str(role)
                for role in list(dict(contract or {}).get("slot_roles", []) or [])
                if str(role)
            ],
            "slot_coverage": [
                str(slot)
                for slot in list(dict(contract or {}).get("slot_coverage", []) or [])
                if str(slot)
            ],
        }
        for skeleton, contract in dict(ring_contracts or {}).items()
    }
    segment_span = 2 if str(class_type) in PAIRED_LIKE24_CLASS_TYPES else 1
    ring_priority = {
        str(skeleton): int(dict(contract or {}).get("priority", 99))
        for skeleton, contract in dict(ring_contracts or {}).items()
        if bool(dict(contract or {}).get("eligible", False))
    }
    flex_allowed = bool(
        str(class_type) in {"24_AB_M", "24_CD_M", "24_CD_M_FULL"}
        or (
            str(class_type) in {"24_AB_N", "24_CD_N"}
            and str(preferred_fixed_role) == "mainline"
        )
    )
    side_allowed = bool(
        str(class_type) in {
            "24_A", "24_B", "24_C", "24_D",
            "24_AB_M", "24_AB_N", "24_AB_P", "24_CD_M", "24_CD_N", "24_CD_P",
            "24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD",
        }
    )
    for contract in fixed_connection_contracts:
        candidates = {
            str(value)
            for value in list(dict(contract or {}).get("role_candidates", []) or [])
        }
        contract["fixed_connection_role"] = (
            "mainline"
            if "mainline_only" in candidates or (
                "mainline_or_branch" in candidates
                and str(preferred_fixed_role) == "mainline"
            )
            else ("branch" if "branch_only" in candidates else "mainline_or_branch")
        )
    return {
        "fixed_connection_contracts": [dict(item or {}) for item in list(fixed_connection_contracts or [])],
        "required_branch_kinds_by_anchor": required_branch_kinds_by_anchor,
        "required_branch_closures_by_anchor_in_ring": branch_closures_in_ring,
        "ring_eligibility": sorted({str(x) for x in ring_eligibility}),
        "ring_slot_signatures": sorted({str(x) for x in ring_slot_signatures}),
        "ring_allowed_skeletons": sorted({str(x) for x in ring_allowed_skeletons}),
        "ring_forbidden_skeletons": sorted({str(x) for x in ring_forbidden_skeletons}),
        "ring_plan_role": str(ring_plan_role),
        "ring_core_slots": sorted({str(x) for x in ring_slot_signatures if "core" in str(x) or "anchor" in str(x)}),
        "bridgehead_role": str(bridgehead_role),
        "inner_ring_open_slots": int(inner_ring_open_slots),
        "mainline_open_pattern": str(mainline_open_pattern),
        "fixed_anchor_eligible": bool(fixed_anchor_eligible),
        "outer_anchor_eligible": bool(outer_anchor_eligible),
        "pair_core_eligible": bool(pair_core_eligible),
        "bridgehead_eligible": bool(bridgehead_eligible),
        "touches_aromatic_anchor": bool(touches_aromatic_anchor),
        "selected_primary_connector_side_class": str(dict(primary_connector or {}).get("side_class", "")),
        "semantic_bucket": str(dict(token.meta or {}).get("semantic_bucket", "")),
        "ring_slot_profiles": sorted({str(x) for x in ring_slot_profiles}),
        "ring_slot_capabilities": ring_slot_capabilities,
        "ring_skeleton_capabilities": ring_skeleton_capabilities,
        "ring_path_capabilities": ring_path_capabilities,
        "ring_role_priority": list(ring_role_priority or []),
        "masked_ring_open_sides": sorted({str(x) for x in masked_ring_open_sides}),
        "ring_block_reason": str(ring_block_reason),
        "required_mainline_closure_modes": list(required_mainline_closure_modes or []),
        "allowed_pool_fillers_by_slot": allowed_pool_fillers_by_slot,
        "fixed_connection_context": fixed_connection_context,
        "bridgehead_policy": bridgehead_policy,
        "ring_contracts": ring_contracts,
        "ring_allowed": bool(any(
            bool(dict(contract or {}).get("eligible", False))
            for contract in dict(ring_contracts or {}).values()
        )),
        "ring_roles_by_skeleton": ring_roles_by_skeleton,
        "ring_segment_span": {
            str(skeleton): int(segment_span)
            for skeleton, details in ring_roles_by_skeleton.items()
            if bool(details.get("allowed", False))
        },
        "ring_priority": ring_priority,
        "fixed_connection_role": str(preferred_fixed_role),
        "side_allowed": side_allowed,
        "assembly_capabilities": {
            "flex_allowed": bool(flex_allowed),
            "side_allowed": bool(side_allowed),
            "ring_allowed": bool(any(
                bool(dict(contract or {}).get("eligible", False))
                for contract in dict(ring_contracts or {}).values()
            )),
            "ring_roles_by_skeleton": ring_roles_by_skeleton,
            "ring_segment_span": {
                str(skeleton): int(segment_span)
                for skeleton, details in ring_roles_by_skeleton.items()
                if bool(details.get("allowed", False))
            },
            "ring_priority": ring_priority,
            "fixed_connection_role": str(preferred_fixed_role),
        },
        "suppressed_support_node_ids": _like24_suppressed_support_node_ids(token),
    }


def _like24_bridge_ring_evidence(
    class_type: str,
    *,
    left_fragment: TopologyFragment,
    right_fragment: TopologyFragment,
) -> Dict[str, Any]:
    allowed: List[str] = []
    slots: List[str] = []
    slot_profiles: List[str] = []
    bridgehead_role = "none"
    pair_core_eligible = False
    touches_aromatic_anchor = bool(
        dict(left_fragment.evidence or {}).get("touches_aromatic_anchor", False)
        or dict(right_fragment.evidence or {}).get("touches_aromatic_anchor", False)
        or "11" in str(getattr(left_fragment, "topology_template", ""))
        or "11" in str(getattr(right_fragment, "topology_template", ""))
    )
    if class_type == "24_AB_N_24_AB":
        # The two AB-N halves form an open pair bridge.  Its two centers are
        # the upper/lower ring path (the -23-22 H stubs remain independent
        # branches), and the same open path may fall back to a side chain.
        # Do not classify the pair by its historical name as an explicit flex
        # product: product preference and legal assembly capabilities differ.
        allowed.append("upper_lower")
        slots.append("upper_lower_pair_core")
        slot_profiles.append("upper_lower_pair_core_segment")
        pair_core_eligible = True
    elif class_type == "24_AB_N_24_CD":
        allowed.append("side")
        slots.extend(["side_anchor_core_pair"])
        slot_profiles.extend(["side_anchor_core_pair_segment"])
    elif class_type == "24_CD_N_24_CD":
        allowed.extend(["side", "upper_lower", "fused"])
        slots.extend(["side_pair_core", "upper_lower_pair_core", "fused_pair_core", "cd_cd_pair_core"])
        slot_profiles.extend(["side_pair_core_segment", "upper_lower_pair_core_segment", "fused_pair_inner_segment"])
        pair_core_eligible = True
        left_ids = list(dict(left_fragment.evidence or {}).get("center_node_ids", []) or [])
        right_ids = list(dict(right_fragment.evidence or {}).get("center_node_ids", []) or [])
        left_bridge = str(dict(left_fragment.evidence or {}).get("bridgehead_role", ""))
        right_bridge = str(dict(right_fragment.evidence or {}).get("bridgehead_role", ""))
        if (
            left_bridge == "fused_bridgehead_substitute_14"
            and right_bridge == "fused_bridgehead_substitute_14"
            and len(left_ids) == 1
            and len(right_ids) == 1
        ):
            bridgehead_role = "fused_bridgehead_substitute_pair"
            slots.append("fused_bridgehead_substitute_pair")
    ring_slot_capabilities = _like24_ring_slot_capabilities(
        class_type=str(class_type),
        ring_allowed_skeletons=allowed,
        ring_slot_profiles=slot_profiles,
        bridgehead_role=str(bridgehead_role),
    )
    ring_role_priority = _like24_ring_role_priority(
        class_type=str(class_type),
        ring_slot_capabilities=ring_slot_capabilities,
    )
    ring_path_capabilities = _like24_ring_path_capabilities(
        class_type=str(class_type),
        touches_aromatic_anchor=bool(touches_aromatic_anchor),
        fixed_anchor_eligible=False,
    )
    branch_kinds_by_slot = {
        "side_ring_plan": {},
        "upper_lower_ring_plan": {},
        "fused_ring_plan": {},
    }
    bridgehead_kind = "substitute_14_pair" if bridgehead_role == "fused_bridgehead_substitute_pair" else "none"
    ring_priority_summary: Dict[str, int] = {}
    for skeleton_kind in sorted({str(item) for item in allowed}):
        if skeleton_kind == "side":
            ring_priority_summary["side_ring_plan"] = (
                12 if class_type == "24_AB_N_24_CD" else 5
            )
        elif skeleton_kind == "upper_lower":
            ring_priority_summary["upper_lower_ring_plan"] = 5
        elif skeleton_kind == "fused":
            ring_priority_summary["fused_ring_plan"] = (
                0 if bridgehead_role == "fused_bridgehead_substitute_pair" else 18
            )
    return {
        "required_branch_kinds_by_anchor": {},
        "required_branch_closures_by_anchor_in_ring": {"side": {}, "upper_lower": {}, "fused": {}},
        "ring_eligibility": sorted({str(item) for item in allowed}),
        "ring_slot_signatures": sorted({str(item) for item in slots}),
        "ring_allowed_skeletons": sorted({str(item) for item in allowed}),
        "ring_forbidden_skeletons": [],
        "ring_plan_role": "pair_core" if pair_core_eligible else "pair_bridge",
        "ring_core_slots": sorted({str(item) for item in slots}),
        "bridgehead_role": str(bridgehead_role),
        "inner_ring_open_slots": 1 if class_type == "24_CD_N_24_CD" else 0,
        "mainline_open_pattern": "double_open_pair_core",
        "fixed_anchor_eligible": False,
        "outer_anchor_eligible": False,
        "pair_core_eligible": bool(pair_core_eligible),
        "bridgehead_eligible": bool(bridgehead_role == "fused_bridgehead_substitute_pair"),
        "touches_aromatic_anchor": bool(touches_aromatic_anchor),
        "selected_primary_connector_side_class": "",
        "semantic_bucket": "",
        "ring_slot_profiles": sorted({str(item) for item in slot_profiles}),
        "ring_slot_capabilities": ring_slot_capabilities,
        "ring_skeleton_capabilities": _like24_ring_skeleton_capabilities(
            class_type=str(class_type),
            ring_allowed_skeletons=allowed,
            ring_slot_capabilities=ring_slot_capabilities,
            preferred_fixed_role="dual",
        ),
        "ring_path_capabilities": ring_path_capabilities,
        "ring_role_priority": list(ring_role_priority or []),
        "masked_ring_open_sides": ["left_ring", "right_ring"] if allowed else [],
        "ring_block_reason": "",
        "required_mainline_closure_modes": [],
        "allowed_pool_fillers_by_slot": _like24_allowed_pool_fillers_by_slot(
            class_type=str(class_type),
            bridgehead_role=str(bridgehead_role),
        ),
        "fixed_connection_context": {
            "preferred_role": "dual",
            "primary_connector_id": -1,
            "primary_role_candidates": [],
            "selected_primary_connector_side_class": "",
            "mainline_blocked_for_ring": False,
        },
        "bridgehead_policy": {
            "role": str(bridgehead_role),
            "eligible": bool(bridgehead_role == "fused_bridgehead_substitute_pair"),
            "substitutable_with_14_pair": bool(bridgehead_role == "fused_bridgehead_substitute_pair"),
        },
        "ring_contracts": {
            "side_ring_plan": _make_ring_contract(
                eligible=(
                    class_type in {"24_AB_N_24_CD", "24_CD_N_24_CD"}
                    and bridgehead_role != "fused_bridgehead_substitute_pair"
                ),
                priority=12 if class_type == "24_AB_N_24_CD" else (5 if class_type == "24_CD_N_24_CD" else 25),
                slot_roles=(
                    ["side_anchor_pair"]
                    if class_type == "24_AB_N_24_AB"
                    else (["side_anchor", "side_core", "side_anchor_core_pair"] if class_type == "24_AB_N_24_CD" else ["side_pair_core", "side_core"])
                ),
                slot_coverage=(
                    ["left_anchor", "left_middle"]
                    if class_type == "24_AB_N_24_CD"
                    else (["left_middle", "right_middle"] if class_type != "24_AB_N_24_AB" else ["left_anchor", "right_anchor"])
                ),
                slot_coverage_mode=(
                    "pair_core"
                    if class_type == "24_CD_N_24_CD"
                    else ("anchor_pair" if class_type == "24_AB_N_24_AB" else ("anchor_core_pair" if class_type == "24_AB_N_24_CD" else "single_slot"))
                ),
                required_real_like24_slots=(
                    ["left_anchor", "left_middle"]
                    if class_type == "24_AB_N_24_CD"
                    else (["left_middle", "right_middle"] if class_type != "24_AB_N_24_AB" else ["left_anchor", "right_anchor"])
                ),
                required_branch_kinds_by_slot=branch_kinds_by_slot["side_ring_plan"],
                optional_pool23_slots=["left_bridge", "right_bridge"],
                optional_pool24_slots=["left_middle", "right_middle"],
                pool24_requires_11_slots=["left_anchor", "right_anchor"] if class_type == "24_AB_N_24_AB" else [],
                fixed_connector_position="dual",
                outer_anchor_must_touch_aromatic=class_type in {"24_AB_N_24_AB", "24_AB_N_24_CD"},
                touches_aromatic_anchor=bool(touches_aromatic_anchor),
                render_policy={"suppress_generic_open": True},
            ),
            "upper_lower_ring_plan": _make_ring_contract(
                eligible=class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"},
                priority=5,
                slot_roles=["upper_lower_core", "upper_lower_pair_core"] if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else [],
                slot_coverage=["core_segment"] if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else [],
                slot_coverage_mode="pair_core" if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else "single_slot",
                required_real_like24_slots=["core_segment"] if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else [],
                required_branch_kinds_by_slot=branch_kinds_by_slot["upper_lower_ring_plan"],
                optional_pool23_slots=["left_bridge", "right_bridge"] if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else [],
                optional_pool24_slots=["core_segment"] if class_type in {"24_AB_N_24_AB", "24_CD_N_24_CD"} else [],
                fixed_connector_position="dual",
                render_policy={"suppress_generic_open": True},
            ),
            "fused_ring_plan": _make_ring_contract(
                eligible=class_type == "24_CD_N_24_CD",
                priority=0 if bridgehead_role == "fused_bridgehead_substitute_pair" else 18,
                slot_roles=(
                    ["fused_bridgehead_pair", "fused_inner_core"]
                    if bridgehead_role == "fused_bridgehead_substitute_pair"
                    else (["fused_inner_core"] if class_type == "24_CD_N_24_CD" else [])
                ),
                slot_coverage=(
                    ["left_bridgehead", "right_bridgehead"]
                    if bridgehead_role == "fused_bridgehead_substitute_pair"
                    else (["inner_2", "inner_3"] if class_type == "24_CD_N_24_CD" else [])
                ),
                slot_coverage_mode=(
                    "bridgehead_pair"
                    if bridgehead_role == "fused_bridgehead_substitute_pair"
                    else ("pair_inner_core" if class_type == "24_CD_N_24_CD" else "single_slot")
                ),
                required_real_like24_slots=(
                    ["left_bridgehead", "right_bridgehead"]
                    if bridgehead_role == "fused_bridgehead_substitute_pair"
                    else (["inner_1", "inner_2", "inner_3", "inner_4"] if class_type == "24_CD_N_24_CD" else [])
                ),
                required_branch_kinds_by_slot=branch_kinds_by_slot["fused_ring_plan"],
                optional_pool23_slots=["inner_1", "inner_2", "inner_3", "inner_4"] if class_type == "24_CD_N_24_CD" else [],
                optional_pool24_slots=["inner_1", "inner_2", "inner_3", "inner_4"] if class_type == "24_CD_N_24_CD" else [],
                inner_slot_count=4 if class_type == "24_CD_N_24_CD" else 0,
                bridgehead_mode=(
                    "pair_14_bridgehead"
                    if bridgehead_role == "fused_bridgehead_substitute_pair"
                    else ("inner_pair_core" if class_type == "24_CD_N_24_CD" else "none")
                ),
                bridgehead_kind=bridgehead_kind,
                fixed_connector_position="dual",
                outer_anchor_must_touch_aromatic=class_type == "24_CD_N_24_CD",
                render_policy={"suppress_generic_open": True},
            ),
        },
        "ring_allowed": bool(allowed),
        # Every open pair bridge has a legal side-chain fallback.  The H
        # closures on the two centers are branch contracts, not extra
        # mainline endpoints.
        "side_allowed": True,
        "assembly_capabilities": {
            "flex_allowed": False,
            "side_allowed": True,
            "ring_allowed": bool(allowed),
            "ring_roles_by_skeleton": {
                f"{skeleton}_ring_plan": {
                    "allowed": bool(skeleton in allowed),
                    "roles": [
                        str(role)
                        for role in list(ring_slot_capabilities.get(skeleton, []) or [])
                    ],
                    "slot_coverage": [],
                }
                for skeleton in ("side", "upper_lower", "fused")
            },
            "ring_segment_span": {
                f"{skeleton}_ring_plan": 2
                for skeleton in allowed
            },
            # Keep the summary authoritative: ineligible ring contracts must
            # not appear as selectable priorities (e.g. AB-AB has no side
            # ring contract, although it does have a side-chain fallback).
            "ring_priority": {
                str(key): int(value)
                for key, value in dict(ring_priority_summary or {}).items()
                if int(value) >= 0
            },
            "fixed_connection_role": "dual",
        },
        "suppressed_support_node_ids": [],
    }


def _find_double_d2_fragment(
    token: ClassificationToken,
    tokens_by_id: Dict[int, ClassificationToken],
    rejects: List[Dict[str, Any]],
) -> Optional[TopologyFragment]:
    allowed_connectors = set(DOUBLE_D2_ALLOWED_CONNECTORS.get(str(token.subtype), set()) or set())
    if not allowed_connectors:
        return None
    contract_violation = _fixed_anchor_contract_violation(token)
    if contract_violation is not None:
        _record_reject(rejects, token, family="double_d2", reason=contract_violation)
        return None
    connectors, reject_reason = resolve_fixed_connectors(
        token,
        allowed_su=allowed_connectors,
        expected_count=2,
    )
    if reject_reason is not None:
        _record_reject(rejects, token, family="double_d2", reason=reject_reason)
        return None
    slot_violation = _fixed_connector_slot_violation(str(token.subtype), connectors)
    if slot_violation is not None:
        _record_reject(rejects, token, family="double_d2", reason=slot_violation)
        return None
    connector_profiles = [
        classify_double_d2_side(token, connector, tokens_by_id)
        for connector in connectors
    ]
    if any(
        not _normalize_fixed_side_class(profile.get("side_class"))
        for profile in connector_profiles
    ):
        _record_reject(
            rejects,
            token,
            family="double_d2",
            reason="connector_side_classification_failed",
            details={"connector_profiles": connector_profiles},
        )
        return None
    key = tuple(sorted(
        _normalize_fixed_side_class(profile.get("side_class"))
        for profile in connector_profiles
    ))
    rule = dict(DOUBLE_D2_COMBO_BY_COMPONENTS.get(key, {}) or {})
    if not rule:
        _record_reject(
            rejects,
            token,
            family="double_d2",
            reason="unsupported_component_combination",
            details={"connector_profiles": connector_profiles},
        )
        return None
    primary_connector = select_primary_fixed_side(connector_profiles)
    secondary_connectors = [
        dict(profile or {})
        for profile in connector_profiles
        if primary_connector is None or int(profile.get("connector_node_id", -1)) != int(primary_connector.get("connector_node_id", -1))
    ]
    source_node_ids = _union_source_ids(
        [int(token.node_id)],
        [int(profile.get("connector_node_id")) for profile in connector_profiles],
        [
            int(node_id)
            for profile in connector_profiles
            for node_id in list(profile.get("remaining_node_ids", []) or [])
        ],
    )
    evidence = {
        "rule_id": str(rule.get("rule_id")),
        "center_node_ids": [int(token.node_id)],
        "connector_profiles": connector_profiles,
        "selected_primary_connector": dict(primary_connector or {}),
        "secondary_connector_profiles": secondary_connectors,
        "fixed_connection_contracts": _build_fixed_connection_contracts(
            class_type=str(rule.get("class_type")),
            connector_profiles=connector_profiles,
            primary_connector=primary_connector,
        ),
        "realization_modes": [],
        "reject_reason": "",
    }
    refs = [
        TopologyRef(kind="node", role="center", su_type=int(token.su_type), node_id=int(token.node_id)),
    ] + [
        TopologyRef(kind="node", role="connector", su_type=int(profile.get("connector_su_type", -1)), node_id=int(profile.get("connector_node_id", -1)))
        for profile in connector_profiles
    ]
    topology_template = str(rule.get("topology_template"))
    open_slots = ["OPEN"] if not bool(rule.get("closed_main", False)) else []
    return _make_fragment(
        token,
        family=str(rule.get("family")),
        class_type=str(rule.get("class_type")),
        topology_template=topology_template,
        open_slots=open_slots,
        branch_slots=[],
        evidence=evidence,
        source_node_ids=source_node_ids,
        claimed_node_ids=source_node_ids,
        refs=refs,
    )


def _match_like24_base_rule(token: ClassificationToken) -> Optional[Dict[str, Any]]:
    subtype = str(token.subtype)
    if subtype not in {"single19d3", "double19d3", "single20d3", "double20d3", "21d3", "24", "14"}:
        return None
    neighbors = list(token.neighbor_tokens or [])
    if len(neighbors) < 2:
        return None
    has_11 = any(int(neighbor.su_type) == 11 for neighbor in neighbors)
    non_11_neighbors = [
        neighbor
        for neighbor in neighbors
        if int(neighbor.su_type) != 11
    ]
    has_like24_partner_branch = any(
        int(neighbor.su_type) in {14, 15}
        for neighbor in neighbors
        if int(neighbor.su_type) != 11
    )
    has_terminal_branch = any(
        _is_terminal_branch_neighbor(neighbor)
        for neighbor in neighbors
        if int(neighbor.su_type) != 11
    )
    has_nonterminal_branch = any(
        _is_nonterminal_branch_neighbor(neighbor)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) != 11
    )
    if not has_terminal_branch:
        has_terminal_branch = any(
            is_visible_terminal_semantic(neighbor)
            for neighbor in neighbors
            if int(neighbor.su_type) != 11
        )
    if not has_nonterminal_branch:
        has_nonterminal_branch = any(
            (
                int(neighbor.su_type) in {28, 32}
                or (
                    not is_visible_terminal_semantic(neighbor)
                    and not is_abstract_open_nonfixed_semantic(neighbor)
                    and int(neighbor.su_type) != 11
                )
            )
            for neighbor in neighbors
            if int(neighbor.su_type) != 11
        )
    has_any_branch = bool(
        has_terminal_branch
        or has_nonterminal_branch
        or (subtype in {"24", "14"} and bool(non_11_neighbors))
    )
    if subtype == "14" and has_like24_partner_branch:
        has_nonterminal_branch = True
        has_any_branch = True
    if has_11:
        if has_terminal_branch:
            return next(rule for rule in LIKE24_BASE_RULES if str(rule.get("rule_id")) == "24_B")
        if has_nonterminal_branch or (subtype in {"24", "14"} and has_any_branch):
            return next(rule for rule in LIKE24_BASE_RULES if str(rule.get("rule_id")) == "24_A")
    else:
        if has_terminal_branch:
            return next(rule for rule in LIKE24_BASE_RULES if str(rule.get("rule_id")) == "24_D")
        if has_nonterminal_branch or (subtype in {"24", "14"} and has_any_branch):
            return next(rule for rule in LIKE24_BASE_RULES if str(rule.get("rule_id")) == "24_C")
    return None


def _make_like24_base_fragment(
    token: ClassificationToken,
    base_rule: Dict[str, Any],
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    topology_template = str(base_rule.get("topology_template"))
    branch_requirements = _like24_required_branch_kinds(
        token,
        class_type=str(base_rule.get("class_type")),
        connector_profiles=[],
        realization_modes=[],
    )
    evidence = {
        "rule_id": str(base_rule.get("rule_id")),
        "center_node_ids": [int(token.node_id)],
        "connector_profiles": [],
        "selected_primary_connector": {},
        "secondary_connector_profiles": [],
        "fixed_connection_contracts": [],
        "realization_modes": [],
        "reject_reason": "",
        "secondary_skipped_reason": "",
    }
    su24_binding = _raw_su24_support_binding(token)
    if su24_binding is not None:
        raw_role, raw_details = classify_raw_su24_semantics(
            su_type=int(token.su_type),
            neighbors=[neighbor.to_dict() for neighbor in list(token.neighbor_tokens or [])],
        )
        evidence.update({
            "raw_su24_role": str(raw_role),
            "raw_su24_profile": dict(raw_details or {}),
            "raw_su24_real_neighbors": True,
            "su24_support_binding": dict(su24_binding),
            "su24_fixed_support_ids_by_su": dict(
                su24_binding.get("fixed_support_ids_by_su", {}) or {}
            ),
            "su24_fixed_support_ids": list(
                su24_binding.get("fixed_support_ids", []) or []
            ),
        })
    evidence.update(_like24_ring_evidence(
        token,
        class_type=str(base_rule.get("class_type")),
        topology_template=topology_template,
        connector_profiles=[],
        primary_connector=None,
        realization_modes=[],
    ))
    claimed_node_ids = _like24_owned_fixed_component_node_ids(token, tokens_by_id)
    if su24_binding is not None:
        claimed_node_ids = _union_source_ids(
            claimed_node_ids,
            list(su24_binding.get("fixed_support_ids", []) or []),
        )
    evidence["selected_su11_anchor_ids"] = sorted(
        int(node_id)
        for node_id in claimed_node_ids
        if any(
            int(neighbor.node_id) == int(node_id) and int(neighbor.su_type) == 11
            for neighbor in list(token.neighbor_tokens or [])
        )
    )[:1]
    evidence["released_extra_su11_ids"] = sorted(
        int(neighbor.node_id)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) == 11
        and int(neighbor.node_id) not in set(evidence["selected_su11_anchor_ids"])
    )
    return _make_fragment(
        token,
        family=str(base_rule.get("family")),
        class_type=str(base_rule.get("class_type")),
        topology_template=topology_template,
        open_slots=["OPEN"] if str(topology_template).startswith("-") or str(topology_template).endswith("-") else [],
        branch_slots=list(branch_requirements.get("center", []) or []),
        evidence=evidence,
        source_node_ids=claimed_node_ids,
        claimed_node_ids=claimed_node_ids,
    )


def _resolve_like24_secondary_rule(
    base_class: str,
    primary_side_class: str,
    connector_profiles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    primary_side_class = _normalize_fixed_side_class(primary_side_class)
    all_side_classes = [
        _normalize_fixed_side_class(profile.get("side_class"))
        for profile in list(connector_profiles or [])
    ]
    if base_class in {"24_A", "24_B"}:
        class_type = {
            "M": "24_AB_M",
            "N": "24_AB_N",
            "P": "24_AB_P",
        }.get(str(primary_side_class))
        if class_type is None:
            return None
        return dict(LIKE24_SECONDARY_BY_CLASS.get(class_type, {}) or {})
    if base_class in {"24_C", "24_D"}:
        if str(primary_side_class) == "M":
            class_type = "24_CD_M_FULL" if len(all_side_classes) >= 2 and all(side == "M" for side in all_side_classes) else "24_CD_M"
        elif str(primary_side_class) == "N":
            class_type = "24_CD_N"
        elif str(primary_side_class) == "P":
            class_type = "24_CD_P"
        else:
            class_type = None
        if class_type is None:
            return None
        return dict(LIKE24_SECONDARY_BY_CLASS.get(class_type, {}) or {})
    return None


def _match_like24_fragment(
    token: ClassificationToken,
    tokens_by_id: Dict[int, ClassificationToken],
    rejects: List[Dict[str, Any]],
) -> Optional[TopologyFragment]:
    if int(token.su_type) == 24:
        raw_role, raw_details = classify_raw_su24_semantics(
            su_type=int(token.su_type),
            neighbors=[neighbor.to_dict() for neighbor in list(token.neighbor_tokens or [])],
        )
        # Ring-capable raw SU24 nodes remain resources until a concrete ring
        # contract selects them.
        return None
    base_rule = _match_like24_base_rule(token)
    if base_rule is None:
        return None
    base_fragment = _make_like24_base_fragment(token, base_rule, tokens_by_id)
    if str(token.subtype) not in LIKE24_FIXED_ELIGIBLE_SUBTYPES:
        return base_fragment

    contract_violation = _fixed_anchor_contract_violation(token)
    if contract_violation is not None:
        _record_reject(rejects, token, family="like24", reason=contract_violation)
        return None
    allowed_connectors = set(LIKE24_ALLOWED_CONNECTORS.get(int(token.su_type), set()) or set())
    expected_connectors = 2 if str(token.subtype).startswith("double") else 1
    connectors, reject_reason = resolve_fixed_connectors(
        token,
        allowed_su=allowed_connectors,
        expected_count=expected_connectors,
    )
    if reject_reason is not None:
        _record_reject(rejects, token, family="like24", reason=str(reject_reason))
        return None
    slot_violation = _fixed_connector_slot_violation(str(token.subtype), connectors)
    if slot_violation is not None:
        _record_reject(rejects, token, family="like24", reason=slot_violation)
        return None
    connector_profiles = [
        classify_like24_fixed_side(token, connector, tokens_by_id)
        for connector in connectors
    ]
    if any(
        not _normalize_fixed_side_class(profile.get("side_class"))
        for profile in connector_profiles
    ):
        _record_reject(
            rejects,
            token,
            family="like24",
            reason="connector_side_classification_failed",
            details={"connector_profiles": connector_profiles},
        )
        return None
    primary_connector = select_primary_fixed_side(connector_profiles)
    if primary_connector is None:
        _record_reject(rejects, token, family="like24", reason="missing_primary_connector")
        return None
    open_su11_ids = sorted(
        int(neighbor.node_id)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) == 11
        and is_open_boundary_semantic(neighbor)
    )
    terminal_28_ring_branch = bool(
        str(token.subtype) == "single19d3"
        and len(connector_profiles) == 1
        and _safe_int(connector_profiles[0].get("connector_su_type"), -1) == 28
        and _normalize_fixed_side_class(connector_profiles[0].get("side_class")) == "N"
        and bool(connector_profiles[0].get("fixed_terminal_branch", False))
        and not list(connector_profiles[0].get("remaining_node_ids", []) or [])
    )
    if str(token.subtype).startswith("single") and (
        len(open_su11_ids) >= 2 or terminal_28_ring_branch
    ):
        contracts = _build_fixed_connection_contracts(
            class_type=str(base_rule.get("class_type")),
            connector_profiles=connector_profiles,
            primary_connector=None,
        )
        for contract in contracts:
            contract["role_candidates"] = ["branch_only"]
        # This branch has two open mainline boundaries, so it is also a
        # legitimate ring segment.  Keep that fact explicit in the
        # classification contract; assembly must not infer ring eligibility
        # from the preferred flex/side product type.
        ring_contracts = dict(base_fragment.evidence.get("ring_contracts", {}) or {})
        ring_branch_kind_names = {
            "side_ring_plan": "side",
            "upper_lower_ring_plan": "upper_lower",
            "fused_ring_plan": "fused",
        }
        ring_branch_kinds = sorted({
            str(ring_branch_kind_names.get(str(kind), str(kind)))
            for kind, contract in ring_contracts.items()
            if bool(dict(contract or {}).get("eligible", False))
        })
        ring_branch_roles = sorted({
            str(role)
            for contract in ring_contracts.values()
            if bool(dict(contract or {}).get("eligible", False))
            for role in list(dict(contract or {}).get("slot_roles", []) or [])
            if str(role)
        })
        base_fragment.evidence.update({
            "connector_profiles": connector_profiles,
            "selected_primary_connector": {},
            "secondary_connector_profiles": connector_profiles,
            "fixed_connection_contracts": contracts,
            "fixed_branch_double_open_flex": bool(not terminal_28_ring_branch),
            "terminal_28_ring_preferred_branch": bool(terminal_28_ring_branch),
            "ring_branch_capable": bool(ring_branch_kinds),
            "ring_branch_allowed_skeletons": list(ring_branch_kinds),
            "ring_branch_roles": list(ring_branch_roles),
            "ring_branch_open_aromatic_endpoint": bool(terminal_28_ring_branch),
            "selected_su11_anchor_ids": [],
            "released_extra_su11_ids": list(open_su11_ids),
            "secondary_skipped_reason": (
                "terminal_28_prefers_24b_n_ring_branch"
                if terminal_28_ring_branch
                else "two_open_mainline_boundaries_keep_fixed_connection_as_branch"
            ),
        })
        capabilities = dict(
            base_fragment.evidence.get("assembly_capabilities", {}) or {}
        )
        capabilities.update({
            "flex_allowed": bool(not terminal_28_ring_branch),
            "side_allowed": True,
            "ring_allowed": bool(ring_branch_kinds),
        })
        base_fragment.evidence["assembly_capabilities"] = capabilities
        base_fragment.topology_template = "-24(?)-"
        base_fragment.open_slots = ["OPEN", "OPEN"]
        base_fragment.open_mainline_slots = []
        base_fragment.source_node_ids = _union_source_ids(
            _like24_owned_fixed_component_node_ids(token, tokens_by_id),
            [
                int(node_id)
                for contract in contracts
                for node_id in list(dict(contract or {}).get("base_path_node_ids", []) or [])
            ],
        )
        base_fragment.claimed_node_ids = list(base_fragment.source_node_ids)
        return base_fragment
    secondary_rule = _resolve_like24_secondary_rule(
        str(base_rule.get("class_type")),
        str(primary_connector.get("side_class")),
        connector_profiles,
    )
    if not secondary_rule:
        _record_reject(
            rejects,
            token,
            family="like24",
            reason="unsupported_like24_secondary_rule",
            details={"connector_profiles": connector_profiles},
        )
        return None
    secondary_connectors = [
        dict(profile or {})
        for profile in connector_profiles
        if int(profile.get("connector_node_id", -1)) != int(primary_connector.get("connector_node_id", -1))
    ]
    topology_template = str(secondary_rule.get("topology_template"))
    branch_requirements = _like24_required_branch_kinds(
        token,
        class_type=str(secondary_rule.get("class_type")),
        connector_profiles=connector_profiles,
        realization_modes=list(secondary_rule.get("realization_modes", []) or []),
    )
    context_node_ids = _union_source_ids(
        _collect_direct_source_ids(token),
        [
            int(node_id)
            for profile in connector_profiles
            for node_id in list(profile.get("remaining_node_ids", []) or [])
        ],
    )
    evidence = {
        "rule_id": str(secondary_rule.get("rule_id")),
        "center_node_ids": [int(token.node_id)],
        "connector_profiles": connector_profiles,
        "selected_primary_connector": dict(primary_connector),
        "secondary_connector_profiles": secondary_connectors,
        "fixed_connection_contracts": _build_fixed_connection_contracts(
            class_type=str(secondary_rule.get("class_type")),
            connector_profiles=connector_profiles,
            primary_connector=primary_connector,
        ),
        "realization_modes": list(secondary_rule.get("realization_modes", []) or []),
        "reject_reason": "",
        "secondary_skipped_reason": "",
        "context_node_ids": context_node_ids,
    }
    evidence.update(_like24_ring_evidence(
        token,
        class_type=str(secondary_rule.get("class_type")),
        topology_template=topology_template,
        connector_profiles=connector_profiles,
        primary_connector=primary_connector,
        realization_modes=list(secondary_rule.get("realization_modes", []) or []),
    ))
    refs = [
        TopologyRef(kind="node", role="center", su_type=int(token.su_type), node_id=int(token.node_id)),
        TopologyRef(
            kind="node",
            role="primary_connector",
            su_type=int(primary_connector.get("connector_su_type", -1)),
            node_id=int(primary_connector.get("connector_node_id", -1)),
        ),
    ]
    claimed_node_ids = _like24_secondary_claimed_node_ids(
        token,
        connector_profiles,
        tokens_by_id,
        fixed_connection_contracts=list(evidence.get("fixed_connection_contracts", []) or []),
        primary_connector_id=_safe_int(primary_connector.get("connector_node_id"), -1),
    )
    return _make_fragment(
        token,
        family=str(secondary_rule.get("family")),
        class_type=str(secondary_rule.get("class_type")),
        topology_template=topology_template,
        open_slots=["OPEN"] if str(topology_template).startswith("-") or str(topology_template).endswith("-") else [],
        branch_slots=list(branch_requirements.get("center", []) or []),
        evidence=evidence,
        source_node_ids=claimed_node_ids,
        claimed_node_ids=claimed_node_ids,
        refs=refs,
    )


def _resolve_bridge_rule(left_class: str, right_class: str) -> Optional[Dict[str, Any]]:
    for class_type, rule in LIKE24_BRIDGE_BY_CLASS.items():
        if left_class in set(rule.get("left_classes", set()) or set()) and right_class in set(rule.get("right_classes", set()) or set()):
            return dict(rule)
    return None


def _match_like24_pair_bridge_fragments(
    tokens: Sequence[ClassificationToken],
    fragments_by_token: Dict[int, TopologyFragment],
    consumed: Set[int],
) -> List[TopologyFragment]:
    out: List[TopologyFragment] = []
    tokens_by_id = _token_by_id(tokens)
    used_local: Set[int] = set()

    connector_candidates = [
        token
        for token in list(tokens or [])
        if int(token.node_id) not in consumed and int(token.su_type) in {27, 29, 31}
    ]
    for connector in connector_candidates:
        if int(connector.node_id) in used_local:
            continue
        center_ids = [
            int(neighbor.node_id)
            for neighbor in list(connector.neighbor_tokens or [])
            if int(neighbor.node_id) in fragments_by_token and int(neighbor.node_id) not in used_local
            and bool(neighbor.is_fixed)
            and _tokens_have_fixed_edge(
                int(connector.node_id),
                int(neighbor.node_id),
                tokens_by_id,
            )
        ]
        if len(center_ids) != 2:
            continue
        left_id, right_id = sorted(int(node_id) for node_id in center_ids)
        left_fragment = fragments_by_token.get(int(left_id))
        right_fragment = fragments_by_token.get(int(right_id))
        if left_fragment is None or right_fragment is None:
            continue
        left_class = str(getattr(left_fragment, "class_type", ""))
        right_class = str(getattr(right_fragment, "class_type", ""))
        rule = _resolve_bridge_rule(left_class, right_class)
        if rule is None:
            rule = _resolve_bridge_rule(right_class, left_class)
            if rule is None:
                continue
            left_id, right_id = right_id, left_id
            left_fragment, right_fragment = right_fragment, left_fragment
            left_class, right_class = right_class, left_class
        source_node_ids = _union_source_ids(
            list(getattr(left_fragment, "source_node_ids", []) or []),
            list(getattr(right_fragment, "source_node_ids", []) or []),
            [int(connector.node_id)],
        )
        claimed_node_ids = _union_source_ids(
            _fragment_claimed_node_ids(left_fragment),
            _fragment_claimed_node_ids(right_fragment),
            [int(connector.node_id)],
        )
        bridge_fixed_contracts = _bridge_fragment_fixed_connection_contracts(
            class_type=str(rule.get("class_type")),
            left_fragment=left_fragment,
            right_fragment=right_fragment,
        )
        merged_branch_slots = [TopologySlotKind.H.value, TopologySlotKind.H.value]
        merged_branch_contracts = [
            TopologyClosureSlot(
                slot_id=f"pair:{min(left_id, right_id)}:branch:{side}:{node_id}",
                kind=TopologySlotKind.H.value,
                anchor_node_id=int(node_id),
                anchor_role=f"{side}_like24_branch",
                required_su_sequence=[23, 22],
            )
            for side, node_id in (("left", left_id), ("right", right_id))
        ]
        fragment = TopologyFragment(
            fragment_id=min(int(left_id), int(right_id), int(connector.node_id)),
            family=str(rule.get("family")),
            class_type=str(rule.get("class_type")),
            topology_template=str(rule.get("topology_template")),
            main_path=[
                TopologyRef(kind="node", role="left", su_type=int(tokens_by_id[int(left_id)].su_type), node_id=int(left_id)),
                TopologyRef(kind="node", role="bridge_connector", su_type=int(connector.su_type), node_id=int(connector.node_id)),
                TopologyRef(kind="node", role="right", su_type=int(tokens_by_id[int(right_id)].su_type), node_id=int(right_id)),
            ],
            source_node_ids=source_node_ids,
            claimed_node_ids=claimed_node_ids,
            open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
            branch_slots=merged_branch_slots,
            branch_contracts=merged_branch_contracts,
            evidence=_augment_fragment_evidence(
                family=str(rule.get("family")),
                class_type=str(rule.get("class_type")),
                fixed_connector_inferred=False,
                topology_template=str(rule.get("topology_template")),
                open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
                branch_slots=merged_branch_slots,
                evidence={
                    "rule_id": str(rule.get("rule_id")),
                    "bridge_kind": str(rule.get("bridge_kind")),
                    "left_class": left_class,
                    "right_class": right_class,
                    "center_node_ids": [int(left_id), int(right_id)],
                    "bridge_connector_node_id": int(connector.node_id),
                    "fixed_connection_contracts": bridge_fixed_contracts,
                    "realization_modes": [],
                    "reject_reason": "",
                    **_like24_bridge_ring_evidence(
                        str(rule.get("class_type")),
                        left_fragment=left_fragment,
                        right_fragment=right_fragment,
                    ),
                },
            ),
        )
        out.append(fragment)
        used_local.update(claimed_node_ids)

    eligible_direct = [
        token
        for token in list(tokens or [])
        if int(token.node_id) not in consumed
        and int(token.node_id) not in used_local
        and str(token.subtype) == "14"
        and int(token.node_id) in fragments_by_token
    ]
    for token in eligible_direct:
        if int(token.node_id) in used_local:
            continue
        partner = next((
            tokens_by_id.get(int(neighbor.node_id))
            for neighbor in list(token.neighbor_tokens or [])
            if int(neighbor.node_id) in fragments_by_token and int(neighbor.node_id) not in used_local
        ), None)
        if partner is None or str(partner.subtype) != "14":
            continue
        left_fragment = fragments_by_token.get(int(token.node_id))
        right_fragment = fragments_by_token.get(int(partner.node_id))
        if left_fragment is None or right_fragment is None:
            continue
        rule = _resolve_bridge_rule(str(left_fragment.class_type), str(right_fragment.class_type))
        if rule is None:
            continue
        source_node_ids = _union_source_ids(
            list(getattr(left_fragment, "source_node_ids", []) or []),
            list(getattr(right_fragment, "source_node_ids", []) or []),
        )
        claimed_node_ids = _union_source_ids(
            _fragment_claimed_node_ids(left_fragment),
            _fragment_claimed_node_ids(right_fragment),
        )
        direct_branch_slots = [TopologySlotKind.H.value, TopologySlotKind.H.value]
        direct_branch_contracts = [
            TopologyClosureSlot(
                slot_id=(
                    f"pair:{min(int(token.node_id), int(partner.node_id))}:"
                    f"branch:{side}:{int(node_id)}"
                ),
                kind=TopologySlotKind.H.value,
                anchor_node_id=int(node_id),
                anchor_role=f"{side}_like24_branch",
                required_su_sequence=[23, 22],
            )
            for side, node_id in (
                ("left", int(token.node_id)),
                ("right", int(partner.node_id)),
            )
        ]
        out.append(TopologyFragment(
            fragment_id=min(int(token.node_id), int(partner.node_id)),
            family=str(rule.get("family")),
            class_type=str(rule.get("class_type")),
            topology_template=str(rule.get("topology_template")),
            main_path=[
                TopologyRef(kind="node", role="left", su_type=int(token.su_type), node_id=int(token.node_id)),
                TopologyRef(kind="node", role="right", su_type=int(partner.su_type), node_id=int(partner.node_id)),
            ],
            source_node_ids=source_node_ids,
            claimed_node_ids=claimed_node_ids,
            open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
            branch_slots=direct_branch_slots,
            branch_contracts=direct_branch_contracts,
            evidence=_augment_fragment_evidence(
                family=str(rule.get("family")),
                class_type=str(rule.get("class_type")),
                fixed_connector_inferred=False,
                topology_template=str(rule.get("topology_template")),
                open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
                branch_slots=direct_branch_slots,
                evidence={
                    "rule_id": str(rule.get("rule_id")),
                    "bridge_kind": "direct_like24_bridge",
                    "left_class": str(left_fragment.class_type),
                    "right_class": str(right_fragment.class_type),
                    "center_node_ids": [int(token.node_id), int(partner.node_id)],
                    "realization_modes": [],
                    "reject_reason": "",
                    **_like24_bridge_ring_evidence(
                        str(rule.get("class_type")),
                        left_fragment=left_fragment,
                        right_fragment=right_fragment,
                    ),
                },
            ),
        ))
        used_local.update(claimed_node_ids)

    return out


def _basic_fragment(
    token: ClassificationToken,
    rule_id: str,
    evidence: Dict[str, Any],
) -> TopologyFragment:
    rule = dict(BASIC_RULE_BY_ID[str(rule_id)])
    topology_template = str(rule.get("topology_template"))
    direct_source_ids = _collect_direct_source_ids(token)
    initial_open_slots = {
        "B": [TopologySlotKind.F1.value],
        "D": [TopologySlotKind.F1.value, TopologySlotKind.F1.value],
        "E": [TopologySlotKind.F1.value],
        "G": [TopologySlotKind.F1.value],
    }.get(str(rule_id), [])
    return _make_fragment(
        token,
        family=str(rule.get("family")),
        class_type=str(rule.get("class_type")),
        topology_template=topology_template,
        # These preliminary kinds preserve endpoint cardinality for flex
        # selection. Their anchors and exact resource sequences are rebuilt
        # after ownership conflicts have been resolved.
        open_slots=list(initial_open_slots),
        branch_slots=[],
        evidence={
            "rule_id": str(rule_id),
            "center_node_ids": [int(token.node_id)],
            "connector_profiles": [],
            "selected_primary_connector": {},
            "secondary_connector_profiles": [],
            "fixed_connection_contracts": [],
            "realization_modes": [],
            "reject_reason": "",
            **dict(evidence or {}),
        },
        source_node_ids=direct_source_ids,
        claimed_node_ids=direct_source_ids,
    )


def classify_basic_center(token: ClassificationToken) -> Optional[TopologyFragment]:
    neighbors = list(token.neighbor_tokens or [])
    center_su = int(token.su_type)

    aromatic_pattern = BASIC_A_AROMATIC_BY_CENTER.get(center_su)
    if aromatic_pattern is not None:
        expected = sorted(int(su) for su in list(aromatic_pattern or []))
        actual = sorted(int(neighbor.su_type) for neighbor in neighbors)
        if actual == expected:
            return _basic_fragment(token, "A", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    if center_su in {0, 2, 3, 27, 29, 31}:
        if center_su == 0:
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 6):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, {"single20d2"}), lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) == 22, lambda nb: int(nb.su_type) == 6):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, {"20d1"}), lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: _matches_exact_subtype(nb, {"single20d2"})):
                return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) == 22, lambda nb: _matches_exact_subtype(nb, {"single20d2"})):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: _matches_exact_subtype(nb, {"20d1"})):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
        if center_su == 2:
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 5):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, {"single19d2"}), lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) == 22, lambda nb: int(nb.su_type) == 5):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, {"19d1"}), lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: _matches_exact_subtype(nb, {"single19d2"})):
                return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) == 22, lambda nb: _matches_exact_subtype(nb, {"single19d2"})):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: _matches_exact_subtype(nb, {"19d1"})):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
        if center_su == 3:
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) == 22, lambda nb: int(nb.su_type) == 9):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_all_generic_open(neighbors):
                return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 22):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
        if center_su in {27, 29, 31} and len(neighbors) == 2:
            aromatic_expected = BASIC_A_AROMATIC_BY_CENTER.get(center_su, [])
            open_expected = BASIC_B_SPECIAL_OPEN_BY_CENTER.get(center_su, set())
            terminal_expected = BASIC_C_SPECIAL_TERMINAL_BY_CENTER.get(center_su, set())
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, set(open_expected)), lambda nb: int(nb.su_type) == int(aromatic_expected[0])):
                return _basic_fragment(token, "B", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, set(terminal_expected)), lambda nb: int(nb.su_type) == int(aromatic_expected[0])):
                return _basic_fragment(token, "C", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if all(_matches_exact_subtype(neighbor, set(BASIC_D_SPECIAL_OPEN_BY_CENTER.get(center_su, set()) or set())) for neighbor in neighbors):
                return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
            if _matches_neighbor_pair(neighbors, lambda nb: _matches_exact_subtype(nb, set(BASIC_D_SPECIAL_OPEN_BY_CENTER.get(center_su, set()) or set())), lambda nb: _matches_exact_subtype(nb, set(BASIC_E_SPECIAL_SIDE_BY_CENTER.get(center_su, set()) or set()))):
                return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    if center_su == 15 and len(neighbors) == 2:
        if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 15):
            return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})
        if _matches_neighbor_pair(neighbors, lambda nb: int(nb.su_type) in {1, 4, 22}, lambda nb: int(nb.su_type) == 15):
            return _basic_fragment(token, "E", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    if center_su == 17 and len(neighbors) == 2:
        if _matches_neighbor_pair(neighbors, lambda nb: _is_generic_open_neighbor(nb), lambda nb: int(nb.su_type) == 17):
            return _basic_fragment(token, "D", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    if center_su in BASIC_F_AROMATIC_BY_CENTER and len(neighbors) == 1:
        if int(neighbors[0].su_type) in set(BASIC_F_AROMATIC_BY_CENTER.get(center_su, set()) or set()):
            return _basic_fragment(token, "F", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    if center_su in BASIC_G_SIDE_BY_CENTER and len(neighbors) == 1:
        spec = dict(BASIC_G_SIDE_BY_CENTER.get(center_su, {}) or {})
        if _side_rule_matches(neighbors, spec):
            return _basic_fragment(token, "G", {"matched_neighbors": [neighbor.to_dict() for neighbor in neighbors]})

    return None


def _match_su25_fragment(token: ClassificationToken) -> Optional[TopologyFragment]:
    if str(token.subtype) != "25":
        return None
    eleven_neighbor_ids = [
        int(neighbor.node_id)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) == 11
    ]
    has_11 = bool(eleven_neighbor_ids)
    fixed_22_neighbor_ids = [
        int(neighbor.node_id)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) == 22
    ]
    has_branch_capacity = any(
        int(neighbor.su_type) in ({22, 23, 24, 25, 0, 1, 2, 3, 14, 15, 17} | TERMINAL_NEIGHBORS)
        for neighbor in list(token.neighbor_tokens or [])
    )
    if not has_branch_capacity:
        return None
    rule = SU25_RULES[0] if has_11 else SU25_RULES[1]
    branch_slot_kinds = [TopologySlotKind.H.value]
    if not fixed_22_neighbor_ids:
        branch_slot_kinds = [TopologySlotKind.FIXED22.value, TopologySlotKind.H.value]
    return _make_fragment(
        token,
        family=str(rule.get("family")),
        class_type=str(rule.get("class_type")),
        topology_template=str(rule.get("topology_template")),
        open_slots=["OPEN"],
        branch_slots=branch_slot_kinds,
        evidence={
            "rule_id": str(rule.get("rule_id")),
            "center_node_ids": [int(token.node_id)],
            "connector_profiles": [],
            "selected_primary_connector": {},
            "secondary_connector_profiles": [],
            "fixed_connection_contracts": [],
            "realization_modes": [],
            "requires_fixed22_branch": True,
            "fixed22_neighbor_ids": list(fixed_22_neighbor_ids[:1]),
            "fixed22_requires_pool": not bool(fixed_22_neighbor_ids),
            "reject_reason": "",
        },
        source_node_ids=_collect_direct_source_ids(token),
        claimed_node_ids=_union_source_ids(
            [int(token.node_id)],
            list(eleven_neighbor_ids[:1]),
            list(fixed_22_neighbor_ids[:1]),
        ),
    )


def _token_audit_dict(
    token: ClassificationToken,
    *,
    classification_status: str,
    reject_bucket: str,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "global_id": int(token.node_id),
        "node_id": int(token.node_id),
        "su_type": int(token.su_type),
        "subtype": str(token.subtype),
        "classification_status": str(classification_status),
        "reject_bucket": str(reject_bucket),
        "hop1_ids": [int(x) for x in list(token.hop1_node_ids or [])],
        "hop1_su": {str(k): int(v) for k, v in dict(token.hop1_counter or {}).items()},
        "target_hop1_degree": int(token.target_degree),
        "target_fixed_anchor_count": int(token.fixed_anchor_count),
        "fixed_hop1_ids": [int(x) for x in list(token.fixed_neighbor_ids or [])],
        "fixed_connector_inferred": bool(dict(token.meta or {}).get("fixed_connector_inferred", False)),
        "anchor_mode": str(token.anchor_mode),
        "anchor_partition": str(token.anchor_partition),
        "resource_pool_member": bool(
            int(token.su_type) in PHYSICAL_TOPOLOGY_POOL_SU_TYPES
        ),
        "reason": str(reason),
    }
    payload.update(dict(extra or {}))
    return payload


def _atomic_pair_fallback_fragment(
    group: Dict[str, Any],
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    node_ids = _ordered_atomic_pair_ids(group.get("node_ids", []), tokens_by_id)
    tokens = [tokens_by_id[node_id] for node_id in node_ids]
    su_pair = tuple(int(token.su_type) for token in tokens)
    branch_slots: List[TopologyClosureSlot] = []
    mainline_slots: List[TopologyClosureSlot] = []
    if 14 in su_pair:
        class_type = "24_CD_P" if int(tokens[-1].su_type) in {16, 18} else "24_C"
        family = "like24_unsaturated_pair"
        for token in tokens:
            if int(token.su_type) == 14:
                branch_slots.append(TopologyClosureSlot(
                    slot_id=f"atomic:{node_ids[0]}:H:{int(token.node_id)}",
                    kind=TopologySlotKind.H.value,
                    anchor_node_id=int(token.node_id),
                    anchor_role="unsaturated_branch_anchor",
                    required_su_sequence=[23, 22],
                ))
        mainline_slots.append(TopologyClosureSlot(
            slot_id=f"atomic:{node_ids[0]}:left:F1",
            kind=TopologySlotKind.F1.value,
            anchor_node_id=int(node_ids[0]),
            anchor_role="left_mainline",
            required_su_sequence=[23, 11],
        ))
        if int(tokens[-1].su_type) in {16, 18}:
            mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{node_ids[0]}:right:terminal",
                kind=TopologySlotKind.CLOSED_TERMINAL.value,
                anchor_node_id=int(node_ids[-1]),
                anchor_role="right_terminal",
                existing_node_ids=[int(node_ids[-1])],
                state=TopologySlotKind.CLOSED_TERMINAL.value,
            ))
        else:
            mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{node_ids[0]}:right:F1",
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(node_ids[-1]),
                anchor_role="right_mainline",
                required_su_sequence=[23, 11],
            ))
        topology_template = "-" + "-".join(
            f"14(-H)" if int(token.su_type) == 14 else str(int(token.su_type))
            for token in tokens
        ) + ("-" if int(tokens[-1].su_type) != 16 else "")
    else:
        class_type = "G" if int(tokens[-1].su_type) in {16, 18} else "D"
        family = "basic_unsaturated_pair"
        mainline_slots.append(TopologyClosureSlot(
            slot_id=f"atomic:{node_ids[0]}:left:F1",
            kind=TopologySlotKind.F1.value,
            anchor_node_id=int(node_ids[0]),
            anchor_role="left_mainline",
            required_su_sequence=[23, 11],
        ))
        if int(tokens[-1].su_type) in {16, 18}:
            mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{node_ids[0]}:right:terminal",
                kind=TopologySlotKind.CLOSED_TERMINAL.value,
                anchor_node_id=int(node_ids[-1]),
                anchor_role="right_terminal",
                existing_node_ids=[int(node_ids[-1])],
                state=TopologySlotKind.CLOSED_TERMINAL.value,
            ))
        else:
            mainline_slots.append(TopologyClosureSlot(
                slot_id=f"atomic:{node_ids[0]}:right:F1",
                kind=TopologySlotKind.F1.value,
                anchor_node_id=int(node_ids[-1]),
                anchor_role="right_mainline",
                required_su_sequence=[23, 11],
            ))
        topology_template = "-".join(str(value) for value in su_pair)
    # ``preferred_product_type`` is only a routing priority.  Atomic pairs
    # with two open mainline ports (the common D-like 15-15 case) are valid
    # side-chain sources when they are not selected for the flex window.  The
    # old fallback emitted no capability facts, so projection interpreted the
    # flex preference as an exclusive capability and dropped the fragment
    # from both flex and side materialization.
    main_kind = _flex_reconcile_main_kind_for_fragment(str(class_type))
    assembly_capabilities = {
        "flex_allowed": bool(main_kind in {"d_like_open", "explicit_flex"}),
        "side_allowed": bool(main_kind in {"d_like_open", "e_like_open"}),
        # Atomic fallback has no contract-defined ring skeleton.  Ring
        # eligibility must come from a classified standard form, never from
        # the pair shape alone.
        "ring_allowed": False,
        "ring_roles_by_skeleton": {},
    }
    evidence = _augment_fragment_evidence(
        family=family,
        class_type=class_type,
        topology_template=topology_template,
        open_slots=[slot.kind for slot in mainline_slots if not slot.is_complete()],
        branch_slots=[slot.kind for slot in branch_slots if not slot.is_complete()],
        fixed_connector_inferred=False,
        evidence={
            "rule_id": "unsaturated_atomic_pair_fallback",
            "center_node_ids": list(node_ids),
            "atomic_group_id": int(group.get("group_id", node_ids[0])),
            "atomic_group_node_ids": list(node_ids),
            "atomic_group_su_pair": list(su_pair),
            "assembly_capabilities": assembly_capabilities,
        },
    )
    return TopologyFragment(
        fragment_id=int(group.get("group_id", node_ids[0])),
        family=family,
        class_type=class_type,
        topology_template=topology_template,
        main_path=[
            TopologyRef(kind="node", role=f"atomic_{index}", su_type=int(token.su_type), node_id=int(token.node_id))
            for index, token in enumerate(tokens)
        ],
        source_node_ids=list(node_ids),
        claimed_node_ids=list(node_ids),
        open_slots=[slot.kind for slot in mainline_slots if not slot.is_complete()],
        branch_slots=[slot.kind for slot in branch_slots if not slot.is_complete()],
        mainline_node_ids=list(node_ids),
        mainline_labels=[str(int(token.su_type)) for token in tokens],
        open_mainline_slots=mainline_slots,
        branch_contracts=branch_slots,
        context_node_ids=[],
        atomic_group_ids=[int(group.get("group_id", node_ids[0]))],
        evidence=evidence,
    )


def _basic_open_endpoint_profiles(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> List[Dict[str, Any]]:
    """Build the owned/context endpoint contract for a basic fragment."""
    if str(fragment.family) != "basic":
        return []

    class_type = str(fragment.class_type)
    if class_type not in {"B", "D", "E", "G"}:
        return []

    claimed = {
        int(value)
        for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])
    }
    context = {
        int(value) for value in list(fragment.context_node_ids or [])
    }
    matched_neighbors = sorted(
        [
            dict(item or {})
            for item in list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
            if dict(item or {}).get("node_id") is not None
        ],
        key=lambda item: (
            _safe_int(item.get("node_id"), 10**9),
            str(item.get("subtype", "")),
        ),
    )

    def _is_open(item: Dict[str, Any]) -> bool:
        return bool(
            not is_visible_terminal_semantic(item)
            and (
                _safe_int(item.get("su_type"), -1) in ALIPHATIC_OPEN_NEIGHBORS
                or str(item.get("subtype", "")) in OPEN_BRIDGE_SUBTYPE_SET
            )
        )

    open_neighbors = [item for item in matched_neighbors if _is_open(item)]
    center_su = (
        int(fragment.main_path[0].su_type)
        if list(fragment.main_path or []) and fragment.main_path[0].su_type is not None
        else -1
    )
    endpoints: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    if class_type == "D":
        if center_su in {15, 17}:
            pair_su_types = {14, 15, 16} if center_su == 15 else {17, 18}
            generic = [
                item for item in open_neighbors
                if is_abstract_open_nonfixed_semantic(item)
                and _safe_int(item.get("su_type"), -1) not in pair_su_types
            ]
            endpoints = [
                ("left", generic[0] if generic else None),
                ("right", None),
            ]
        else:
            endpoints = [
                ("left", open_neighbors[0] if open_neighbors else None),
                ("right", open_neighbors[-1] if open_neighbors else None),
            ]
    elif class_type in {"B", "E"}:
        endpoints = [("right", open_neighbors[0] if open_neighbors else None)]
    elif class_type == "G":
        endpoints = [("left", open_neighbors[0] if open_neighbors else None)]

    profiles: List[Dict[str, Any]] = []
    for side, token in endpoints:
        token_payload = dict(token or {})
        node_id = _safe_int(token_payload.get("node_id"), -1)
        label = str(
            token_payload.get("subtype")
            or token_payload.get("su_type")
            or ""
        )
        owned_literal = bool(node_id >= 0 and node_id in claimed)
        endpoint_token = tokens_by_id.get(int(node_id)) if node_id >= 0 else None
        open_boundary_node_ids = sorted({
            int(neighbor.node_id)
            for neighbor in list(
                getattr(endpoint_token, "neighbor_tokens", []) or []
            )
            if int(neighbor.su_type) == 23
            and not _tokens_have_fixed_edge(
                int(node_id), int(neighbor.node_id), tokens_by_id
            )
        })
        profiles.append({
            "side": str(side),
            "node_id": None if node_id < 0 else int(node_id),
            "su_type": _safe_int(token_payload.get("su_type"), -1),
            "subtype": str(label),
            "is_fixed": bool(token_payload.get("is_fixed", False)),
            "owned_literal": bool(owned_literal),
            "context_only": bool(node_id >= 0 and node_id in context and not owned_literal),
            "open_boundary_node_ids": open_boundary_node_ids,
            "requires_full_f1": bool(open_boundary_node_ids),
        })
    return profiles


def _basic_open_endpoint_slots(
    fragment: TopologyFragment,
    profiles: Sequence[Dict[str, Any]],
) -> List[TopologyClosureSlot]:
    slots: List[TopologyClosureSlot] = []
    for index, raw_profile in enumerate(list(profiles or [])):
        profile = dict(raw_profile or {})
        side = str(profile.get("side", "right") or "right")
        node_id = _safe_int(profile.get("node_id"), -1)
        label = str(profile.get("subtype", "") or "")
        owned_literal = bool(profile.get("owned_literal", False))
        required = (
            [11]
            if (
                owned_literal
                and label in BASIC_CAP_ONLY_BRIDGE_SUBTYPES
                and not bool(profile.get("requires_full_f1", False))
            )
            else [23, 11]
        )
        anchor_node_id = (
            int(node_id)
            if owned_literal and node_id >= 0
            else int(fragment.fragment_id)
        )
        slots.append(TopologyClosureSlot(
            slot_id=(
                f"fragment:{int(fragment.fragment_id)}:main:"
                f"{int(index)}:{str(side)}:F1"
            ),
            kind=TopologySlotKind.F1.value,
            anchor_node_id=int(anchor_node_id),
            anchor_role=f"{str(side)}_mainline",
            required_su_sequence=list(required),
        ))
    return slots


def _normalize_fragment_contract(
    fragment: TopologyFragment,
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    original_source = sorted({int(value) for value in list(fragment.source_node_ids or [])})
    claimed = sorted({int(value) for value in list(fragment.claimed_node_ids or original_source)})
    fragment.context_node_ids = sorted(
        {
            int(value)
            for value in list(fragment.context_node_ids or [])
        }
        | (set(original_source) - set(claimed))
    )
    fragment.claimed_node_ids = list(claimed)
    fragment.source_node_ids = list(claimed)
    if not fragment.mainline_node_ids:
        fragment.mainline_node_ids = [
            int(ref.node_id)
            for ref in list(fragment.main_path or [])
            if ref.node_id is not None and int(ref.node_id) in set(claimed)
        ]
    if not fragment.mainline_node_ids:
        fragment.mainline_node_ids = list(claimed)
    if not fragment.mainline_labels:
        fragment.mainline_labels = [
            str(ref.label or ref.su_type)
            for ref in list(fragment.main_path or [])
            if ref.node_id is not None
        ]
    if str(fragment.family) == "basic":
        endpoint_profiles = _basic_open_endpoint_profiles(fragment, tokens_by_id)
        fragment.evidence["basic_open_endpoints"] = [
            dict(item or {}) for item in endpoint_profiles
        ]
        if not fragment.open_mainline_slots:
            fragment.open_mainline_slots = _basic_open_endpoint_slots(
                fragment,
                endpoint_profiles,
            )
            fragment.open_slots = [
                str(slot.kind) for slot in fragment.open_mainline_slots
            ]
    center_su = int(fragment.main_path[0].su_type) if list(fragment.main_path or []) else -1
    if str(fragment.class_type) in {"25_AB", "25_CD"}:
        matched_neighbors = [
            dict(item or {})
            for item in list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
        ]
        fixed_22_ids = sorted({
            int(item.get("node_id"))
            for item in matched_neighbors
            if item.get("node_id") is not None
            and int(item.get("su_type", -1)) == 22
            and int(item.get("node_id")) in set(claimed)
        })
        fixed_22_ids = sorted(
            set(fixed_22_ids)
            | {
                int(node_id)
                for node_id in list(
                    dict(fragment.evidence or {}).get("fixed22_neighbor_ids", []) or []
                )
                if int(node_id) in set(claimed)
            }
        )
        # SU25 has four physical ports.  Its public contract must describe
        # every open mainline closure that `_assemble_remaining_su25()` will
        # materialize:
        #
        #   25_AB: right F2 + H branch
        #   25_CD: left F1 + right F2 + H branch
        #
        # The previous implementation exposed a single slot (and used F1 for
        # 25_AB), which under-counted resources and made Stage3 reject a
        # resource-sufficient topology as an invalid materialization.
        if str(fragment.class_type) == "25_CD":
            fragment.open_mainline_slots = [
                TopologyClosureSlot(
                    slot_id=f"fragment:{fragment.fragment_id}:main:left:F1",
                    kind=TopologySlotKind.F1.value,
                    anchor_node_id=int(fragment.fragment_id),
                    anchor_role="left_mainline",
                    required_su_sequence=[23, 11],
                ),
                TopologyClosureSlot(
                    slot_id=f"fragment:{fragment.fragment_id}:main:right:F2",
                    kind=TopologySlotKind.F2.value,
                    anchor_node_id=int(fragment.fragment_id),
                    anchor_role="right_mainline",
                    required_su_sequence=[23, 22],
                ),
            ]
            fragment.open_slots = [
                TopologySlotKind.F1.value,
                TopologySlotKind.F2.value,
            ]
        else:
            fragment.open_mainline_slots = [TopologyClosureSlot(
                slot_id=f"fragment:{fragment.fragment_id}:main:right:F2",
                kind=TopologySlotKind.F2.value,
                anchor_node_id=int(fragment.fragment_id),
                anchor_role="right_mainline",
                required_su_sequence=[23, 22],
            )]
            fragment.open_slots = [TopologySlotKind.F2.value]
        branch_contracts: List[TopologyClosureSlot] = []
        if not fixed_22_ids:
            branch_contracts.append(TopologyClosureSlot(
                slot_id=f"fragment:{fragment.fragment_id}:branch:FIXED22",
                kind=TopologySlotKind.FIXED22.value,
                anchor_node_id=int(fragment.fragment_id),
                anchor_role="fixed22_branch",
                required_su_sequence=[22],
            ))
        fragment.branch_contracts = branch_contracts + [TopologyClosureSlot(
            slot_id=f"fragment:{fragment.fragment_id}:branch:H",
            kind=TopologySlotKind.H.value,
            anchor_node_id=int(fragment.fragment_id),
            anchor_role="branch",
            required_su_sequence=[23, 22],
        )]
        fragment.branch_contracts.extend(
            TopologyClosureSlot(
                slot_id=f"fragment:{fragment.fragment_id}:branch:fixed22:{node_id}",
                kind=TopologySlotKind.CLOSED_TERMINAL.value,
                anchor_node_id=int(fragment.fragment_id),
                anchor_role="closed_terminal_branch",
                existing_node_ids=[int(node_id)],
                state=TopologySlotKind.CLOSED_TERMINAL.value,
            )
            for node_id in fixed_22_ids
        )
        fragment.branch_slots = (
            [TopologySlotKind.H.value]
            if fixed_22_ids
            else [TopologySlotKind.FIXED22.value, TopologySlotKind.H.value]
        )
    fixed_terminal_bridge_subtype = {
        28: "single19d2",
        32: "21d2",
    }.get(center_su)
    if str(fragment.class_type) == "G" and fixed_terminal_bridge_subtype is not None:
        matched_neighbors = [
            dict(item or {})
            for item in list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
        ]
        existing_fixed_bridge = next(
            (
                item
                for item in matched_neighbors
                if str(item.get("subtype", "")) == str(fixed_terminal_bridge_subtype)
                and item.get("node_id") is not None
                and int(item.get("node_id")) in set(claimed)
            ),
            None,
        )
        if existing_fixed_bridge is not None:
            bridge_id = int(existing_fixed_bridge["node_id"])
            fragment.mainline_node_ids = [bridge_id, int(fragment.fragment_id)]
            fragment.mainline_labels = [str(fixed_terminal_bridge_subtype), str(center_su)]
            fragment.open_mainline_slots = [TopologyClosureSlot(
                slot_id=f"fragment:{fragment.fragment_id}:main:0",
                kind=TopologySlotKind.F1.value,
                anchor_node_id=bridge_id,
                anchor_role="left_mainline_existing_bridge",
                required_su_sequence=[11],
            )]
    atomic_ids = list(fragment.atomic_group_ids or [])
    atomic_ids.extend(
        int(value)
        for value in list(dict(fragment.evidence or {}).get("atomic_group_ids", []) or [])
    )
    atomic_group_id = dict(fragment.evidence or {}).get("atomic_group_id")
    if atomic_group_id is not None:
        atomic_ids.append(int(atomic_group_id))
    fragment.atomic_group_ids = sorted(set(atomic_ids))
    fragment.open_mainline_slots = [normalize_slot(slot) for slot in list(fragment.open_mainline_slots or [])]
    fragment.branch_contracts = [normalize_slot(slot) for slot in list(fragment.branch_contracts or [])]
    if (
        not fragment.open_mainline_slots
        and str(fragment.family) != "basic"
        and not bool(dict(fragment.evidence or {}).get("edge_semantics_finalized", False))
    ):
        roles = set(str(value) for value in list(dict(fragment.evidence or {}).get("flex_reconcile_roles", []) or []))
        matched_neighbors = [
            dict(item or {})
            for item in list(dict(fragment.evidence or {}).get("matched_neighbors", []) or [])
        ]
        open_neighbor_ids = sorted({
            int(item.get("node_id"))
            for item in matched_neighbors
            if item.get("node_id") is not None
            and int(item.get("su_type", -1)) in {23, 24}
        })
        desired_open_count = 2 if "double_end_flex_source" in roles else 1 if "single_end_flex_source" in roles else 0
        for index, anchor in enumerate(open_neighbor_ids[:desired_open_count]):
            fragment.open_mainline_slots.append(normalize_slot({
                "slot_id": f"fragment:{fragment.fragment_id}:main:{index}",
                "kind": TopologySlotKind.F1.value,
                "anchor_node_id": int(anchor),
                "anchor_role": "left_mainline" if index == 0 else "right_mainline",
                "required_su_sequence": [11],
            }))
        for index, kind in enumerate(list(fragment.open_slots or [])):
            if fragment.open_mainline_slots:
                break
            slot_kind = str(kind) if str(kind) in {"F1", "F2"} else TopologySlotKind.F1.value
            anchor = fragment.mainline_node_ids[0 if index == 0 else -1] if fragment.mainline_node_ids else -1
            fragment.open_mainline_slots.append(normalize_slot({
                "slot_id": f"fragment:{fragment.fragment_id}:main:{index}",
                "kind": slot_kind,
                "anchor_node_id": int(anchor),
                "anchor_role": "left_mainline" if index == 0 else "right_mainline",
            }))
    if not fragment.branch_contracts:
        anchors = list(fragment.mainline_node_ids or [-1])
        for index, kind in enumerate(list(fragment.branch_slots or [])):
            slot_kind = str(kind) if str(kind) in {item.value for item in TopologySlotKind} else TopologySlotKind.H.value
            fragment.branch_contracts.append(normalize_slot({
                "slot_id": f"fragment:{fragment.fragment_id}:branch:{index}",
                "kind": slot_kind,
                "anchor_node_id": int(anchors[min(index, len(anchors) - 1)]),
                "anchor_role": "branch",
            }))
    return fragment


def classify_topology_units(
    nodes: List[_NodeV3],
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
) -> TopologyClassificationResult:
    tokens = build_node_tokens(list(nodes or []))
    partition_meta = dict(fixed_partition_meta or {})
    partition_by_node = {
        int(key): str(value)
        for key, value in dict(partition_meta.get("anchor_partition_by_node_id", {}) or {}).items()
    }
    forbidden_fixed_connector_node_ids = {
        int(value)
        for value in list(partition_meta.get("forbidden_fixed_connector_node_ids", []) or [])
    }
    if partition_by_node or forbidden_fixed_connector_node_ids:
        for token in tokens:
            if int(token.node_id) in partition_by_node:
                token.anchor_partition = str(partition_by_node[int(token.node_id)]).strip().lower()
                token.meta["fixed_partition_meta_applied"] = True
            if int(token.node_id) in forbidden_fixed_connector_node_ids:
                token.meta["fixed_connector_forbidden"] = True
    tokens_by_id = _token_by_id(tokens)
    raw_su24_fixed_support_ids: Set[int] = set()
    su24_support_bindings: List[Dict[str, Any]] = []
    for token in tokens:
        if int(token.su_type) != 24:
            continue
        binding = _raw_su24_support_binding(token)
        if binding is None:
            continue
        su24_support_bindings.append(dict(binding))
        raw_su24_fixed_support_ids.update(
            int(node_id)
            for node_id in list(binding.get("fixed_support_ids", []) or [])
        )
    consumed: Set[int] = set()
    fragments: List[TopologyFragment] = []
    classification_rejects: List[Dict[str, Any]] = []
    claim_conflicts: List[Dict[str, Any]] = []
    support_owner_by_id: Dict[int, int] = {}
    for binding in list(su24_support_bindings or []):
        center_id = _safe_int(binding.get("su24_node_id"), -1)
        for support_id in list(binding.get("fixed_support_ids", []) or []):
            support_id = int(support_id)
            previous_owner = support_owner_by_id.get(support_id)
            if previous_owner is not None and int(previous_owner) != int(center_id):
                claim_conflicts.append({
                    "node_id": int(support_id),
                    "global_id": int(support_id),
                    "family": "su24_support_binding",
                    "classification_status": "claim_conflict",
                    "reject_bucket": "claim_conflicts",
                    "reason": "su24_support_bound_to_multiple_centers",
                    "support_node_id": int(support_id),
                    "owner_su24_node_ids": sorted({
                        int(previous_owner),
                        int(center_id),
                    }),
                })
            else:
                support_owner_by_id[support_id] = int(center_id)
    fixed_edge_audit_keys: Set[Tuple[int, int]] = set()
    for token in tokens:
        for violation in list(dict(token.meta or {}).get("fixed_edge_violations", []) or []):
            edge_key = tuple(int(value) for value in list(dict(violation or {}).get("edge_key", []) or []))
            if len(edge_key) != 2 or edge_key in fixed_edge_audit_keys:
                continue
            fixed_edge_audit_keys.add(edge_key)
            classification_rejects.append({
                "node_id": int(token.node_id),
                "su_type": int(token.su_type),
                "subtype": str(token.subtype),
                "family": "fixed_edge_audit",
                "reason": "asymmetric_fixed_edge",
                "edge_key": list(edge_key),
            })
    invalid_partition_node_ids: Set[int] = set()
    invalid_resource_node_ids: Set[int] = set()
    for token in tokens:
        if int(token.su_type) != 19:
            continue
        partition = str(token.anchor_partition or "").strip().lower()
        if partition in {"ether", "thio"}:
            continue
        reason = (
            "su19_anchor_partition_missing"
            if not partition
            else f"su19_anchor_partition_invalid:{partition}"
        )
        _record_reject(
            classification_rejects,
            token,
            family="su19_partition",
            reason=reason,
        )
        invalid_partition_node_ids.add(int(token.node_id))
    atomic_groups = _build_unsaturated_atomic_groups(tokens, classification_rejects)
    rejected_atomic_node_ids = {
        int(item.get("node_id", -1))
        for item in classification_rejects
        if str(item.get("family", "")) == "unsaturated_atomic_pair"
    }

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids | invalid_partition_node_ids | invalid_resource_node_ids:
            continue
        fragment = _find_double_d2_fragment(token, tokens_by_id, classification_rejects)
        if fragment is not None:
            fragment = _attach_center_atomic_group(fragment, tokens_by_id)
            _append_if_unclaimed(fragments, consumed, fragment)

    per_token_like24: Dict[int, TopologyFragment] = {}
    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids | invalid_partition_node_ids | invalid_resource_node_ids:
            continue
        fragment = _match_like24_fragment(token, tokens_by_id, classification_rejects)
        if fragment is not None:
            per_token_like24[int(token.node_id)] = fragment

    pair_bridge_fragments = _match_like24_pair_bridge_fragments(tokens, per_token_like24, consumed)
    for fragment in pair_bridge_fragments:
        fragment = _attach_center_atomic_group(fragment, tokens_by_id)
        _append_if_unclaimed(fragments, consumed, fragment)

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids | invalid_partition_node_ids | invalid_resource_node_ids:
            continue
        like24_fragment = per_token_like24.get(int(token.node_id))
        if like24_fragment is not None:
            like24_fragment = _attach_center_atomic_group(like24_fragment, tokens_by_id)
            _append_if_unclaimed(fragments, consumed, like24_fragment)

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in invalid_partition_node_ids | invalid_resource_node_ids:
            continue
        su25_fragment = _match_su25_fragment(token)
        if su25_fragment is not None:
            su25_fragment = _attach_center_atomic_group(su25_fragment, tokens_by_id)
            _append_if_unclaimed(
                fragments,
                consumed,
                su25_fragment,
                claim_conflicts=claim_conflicts,
                record_conflict=True,
            )

    for token in tokens:
        if (
            int(token.node_id) in consumed
            or int(token.node_id) in raw_su24_fixed_support_ids
            or int(token.node_id) in rejected_atomic_node_ids | invalid_partition_node_ids | invalid_resource_node_ids
        ):
            continue
        basic_fragment = classify_basic_center(token)
        if basic_fragment is not None:
            basic_fragment = _attach_center_atomic_group(basic_fragment, tokens_by_id)
            _append_if_unclaimed(
                fragments,
                consumed,
                basic_fragment,
                claim_conflicts=claim_conflicts,
                record_conflict=True,
            )

    for atomic_group in list(atomic_groups or []):
        group_node_ids = {int(value) for value in list(dict(atomic_group or {}).get("node_ids", []) or [])}
        if not group_node_ids or group_node_ids & consumed:
            continue
        fallback_fragment = _atomic_pair_fallback_fragment(dict(atomic_group or {}), tokens_by_id)
        _append_if_unclaimed(
            fragments,
            consumed,
            fallback_fragment,
            claim_conflicts=claim_conflicts,
            record_conflict=True,
        )

    raw_su24_ids = {
        int(token.node_id)
        for token in tokens
        if int(token.su_type) == 24
    }
    reserved_fixed_branch_pool_ids = {
        int(node_id)
        for fragment in fragments
        for contract in list(
            dict(getattr(fragment, "evidence", {}) or {}).get(
                "fixed_connection_contracts", []
            )
            or []
        )
        if "branch_only" in {
            str(value)
            for value in list(dict(contract or {}).get("role_candidates", []) or [])
        }
        for node_id in list(
            dict(contract or {}).get("generic_open_evidence_node_ids", []) or []
        )
        if int(getattr(tokens_by_id.get(_safe_int(node_id, -1)), "su_type", -1))
        in PHYSICAL_TOPOLOGY_POOL_SU_TYPES
    }
    released_pool_context_ids = set(raw_su24_ids) | set(
        reserved_fixed_branch_pool_ids
    )
    if released_pool_context_ids:
        for fragment in fragments:
            claimed = {
                int(value)
                for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])
            }
            released = claimed & released_pool_context_ids
            if not released:
                continue
            fragment.claimed_node_ids = sorted(claimed - released)
            fragment.source_node_ids = list(fragment.claimed_node_ids)
            fragment.context_node_ids = sorted({
                int(value)
                for value in list(fragment.context_node_ids or [])
            } | released)
        consumed.difference_update(released_pool_context_ids)

    # Mandatory SU24 supports are part of the classification ownership
    # partition even for raw SU24 ring resources that remain unmaterialized.
    # They are never generic pool inventory, but the SU24 center itself stays
    # available in the dedicated ring pool.
    consumed.update(raw_su24_fixed_support_ids)

    normalized_fragments: List[TopologyFragment] = []
    for fragment in fragments:
        family = str(fragment.family)
        if family in {"like24", "like24_secondary", "like24_bridge"}:
            fragment = _finalize_fragment_edge_semantics(
                fragment,
                tokens_by_id,
            )
        normalized_fragments.append(
            _normalize_fragment_contract(fragment, tokens_by_id)
        )
    fragments = normalized_fragments
    edge_semantics_audit_errors: List[Dict[str, Any]] = []
    for fragment in fragments:
        if str(fragment.family) not in {"like24", "like24_secondary", "like24_bridge"}:
            continue
        for reason in _fragment_edge_semantic_errors(fragment, tokens_by_id):
            edge_semantics_audit_errors.append({
                "fragment_id": int(fragment.fragment_id),
                "family": str(fragment.family),
                "class_type": str(fragment.class_type),
                "reason": str(reason),
            })
    if edge_semantics_audit_errors:
        raise ValueError(f"topology_edge_semantics_invalid:{edge_semantics_audit_errors}")

    unclassified_nodes: List[Dict[str, Any]] = []
    support_units_only: List[Dict[str, Any]] = []
    class_counts = Counter()
    family_counts = Counter()
    basic_class_counts = Counter()
    double_d2_combo_counts = Counter()
    like24_secondary_counts = Counter()
    realization_mode_count = 0

    for fragment in fragments:
        class_type = str(fragment.class_type)
        family = str(fragment.family)
        class_counts[class_type] += 1
        family_counts[family] += 1
        if class_type in BASIC_CLASS_TYPES:
            basic_class_counts[class_type] += 1
        if class_type in DOUBLE_D2_CLASS_TYPES:
            double_d2_combo_counts[class_type] += 1
        if family == "like24_secondary":
            like24_secondary_counts[class_type] += 1
        realization_mode_count += int(len(dict(fragment.evidence or {}).get("realization_modes", []) or []))

    # Freeze the physical ownership established by classification.  The
    # assembly layer must not reconstruct this mapping from projected views:
    # doing so loses conflicts when a branch later expands a source fragment.
    fixed_node_owner_by_id: Dict[int, int] = {}
    fixed_owner_conflicts: List[Dict[str, Any]] = []
    for fragment in list(fragments or []):
        fragment_id = int(fragment.fragment_id)
        claimed_ids = {
            int(value)
            for value in list(fragment.claimed_node_ids or fragment.source_node_ids or [])
        }
        for node_id in sorted(claimed_ids):
            previous_owner = fixed_node_owner_by_id.get(int(node_id))
            if previous_owner is None:
                fixed_node_owner_by_id[int(node_id)] = fragment_id
                continue
            if int(previous_owner) == fragment_id:
                continue
            fixed_owner_conflicts.append({
                "node_id": int(node_id),
                "owner_fragment_ids": sorted({int(previous_owner), fragment_id}),
                "reason": "fixed_node_claimed_by_multiple_fragments",
                "classification_status": "claim_conflict",
                "reject_bucket": "claim_conflicts",
            })
    if fixed_owner_conflicts:
        claim_conflicts.extend(fixed_owner_conflicts)

    # A raw SU24 binding is attached to the fragment that owns its center when
    # such a fragment exists.  Open SU24 ring centers remain pool resources;
    # their exact supports are still excluded from generic allocation by the
    # binding itself, but are not assigned to an unrelated fragment.
    owner_by_center = {
        int(node_id): int(owner)
        for node_id, owner in fixed_node_owner_by_id.items()
    }
    for binding in list(su24_support_bindings or []):
        center_id = _safe_int(binding.get("su24_node_id"), -1)
        owner = owner_by_center.get(int(center_id))
        if owner is None:
            continue
        for support_id in list(binding.get("fixed_support_ids", []) or []):
            support_id = int(support_id)
            previous_owner = fixed_node_owner_by_id.get(support_id)
            if previous_owner is None:
                fixed_node_owner_by_id[support_id] = int(owner)
            elif int(previous_owner) != int(owner):
                claim_conflicts.append({
                    "node_id": support_id,
                    "owner_fragment_ids": sorted({int(previous_owner), int(owner)}),
                    "reason": "su24_fixed_support_owner_conflict",
                    "classification_status": "claim_conflict",
                    "reject_bucket": "claim_conflicts",
                })
    consumed.update(int(node_id) for node_id in fixed_node_owner_by_id)

    token_lookup = _token_by_id(tokens)
    referenced_source_ids: Set[int] = set()
    for fragment in list(fragments or []):
        referenced_source_ids.update(
            int(node_id)
            for node_id in list(getattr(fragment, "source_node_ids", []) or [])
        )
    referenced_fixed_support_ids = {
        int(neighbor_id)
        for source_id in referenced_source_ids
        for neighbor_id in list(
            getattr(token_lookup.get(int(source_id)), "fixed_neighbor_ids", []) or []
        )
        if int(neighbor_id) not in referenced_source_ids
        and token_lookup.get(int(neighbor_id)) is not None
        and int(getattr(token_lookup[int(neighbor_id)], "su_type", -1))
        not in PHYSICAL_TOPOLOGY_POOL_SU_TYPES
    }
    resource_pool = _build_resource_pool(
        tokens,
        consumed,
        excluded_ids=(
            referenced_source_ids
            | raw_su24_fixed_support_ids
            | invalid_resource_node_ids
        ),
    )
    rejected_node_ids = {int(item.get("node_id", -1)) for item in list(classification_rejects or [])}
    conflict_node_ids = {int(item.get("node_id", -1)) for item in list(claim_conflicts or [])}
    for token in tokens:
        if int(token.node_id) in consumed:
            continue
        if int(token.node_id) in rejected_node_ids:
            continue
        if int(token.node_id) in conflict_node_ids:
            continue
        if int(token.su_type) in PHYSICAL_TOPOLOGY_POOL_SU_TYPES:
            continue
        like24_fixed_partner_support = _is_like24_fixed_partner_support(token, token_lookup)
        if (
            int(token.node_id) in referenced_source_ids
            or int(token.node_id) in referenced_fixed_support_ids
            or int(token.node_id) in raw_su24_fixed_support_ids
            or bool(like24_fixed_partner_support)
            or bool(str(dict(token.meta or {}).get("semantic_bucket", "")))
            or int(token.su_type) in set(SUPPORT_ONLY_SU_TYPES)
            or int(token.su_type) in set(AROMATIC_SU_TYPES)
            or int(token.su_type) not in set(CLASSIFIABLE_CENTER_SU_TYPES)
        ):
            support_units_only.append(_token_audit_dict(
                token,
                classification_status="support_unit_only",
                reject_bucket="support_units_only",
                reason=(
                    "referenced_as_support_unit"
                    if int(token.node_id) in referenced_source_ids
                    else (
                        "referenced_fixed_neighbor_support"
                        if int(token.node_id) in referenced_fixed_support_ids
                        else (
                            "raw_su24_fixed_support"
                            if int(token.node_id) in raw_su24_fixed_support_ids
                            else (
                                "like24_fixed_partner_support"
                                if bool(like24_fixed_partner_support)
                                else (
                                    "semantic_open_support"
                                    if str(dict(token.meta or {}).get("semantic_bucket", ""))
                                    else "support_unit_only"
                                )
                            )
                        )
                    )
                ),
                extra={
                    "semantic_bucket": str(dict(token.meta or {}).get("semantic_bucket", "")),
                },
            ))
            continue
        unclassified_nodes.append(_token_audit_dict(
            token,
            classification_status="no_rule_matched",
            reject_bucket="no_rule_matched",
            reason="no_classification_rule_matched",
        ))

    unresolved_fragment_ids = {
        int(fragment.fragment_id)
        for fragment in fragments
        if any(
            not normalize_slot(slot).is_complete()
            for slot in list(fragment.open_mainline_slots or []) + list(fragment.branch_contracts or [])
        )
    }
    summary = {
        "fragment_count": int(len(fragments)),
        "closed_fragment_count": int(len(fragments) - len(unresolved_fragment_ids)),
        "open_fragment_count": int(len(unresolved_fragment_ids)),
        "unclassified_non_resource_count": int(len(unclassified_nodes)),
        "support_units_only_count": int(len(support_units_only)),
        "class_counts": dict(class_counts),
        "family_counts": dict(family_counts),
        "basic_class_counts": dict(basic_class_counts),
        "double_d2_combo_counts": dict(double_d2_combo_counts),
        "like24_secondary_counts": dict(like24_secondary_counts),
        "reject_count": int(len(classification_rejects) + len(claim_conflicts)),
        "rule_reject_count": int(len(classification_rejects)),
        "claim_conflict_count": int(len(claim_conflicts)),
        "no_rule_matched_count": int(len(unclassified_nodes)),
        "realization_mode_count": int(realization_mode_count),
        "fixed_edge_violation_count": int(len(fixed_edge_audit_keys)),
        "open_boundary_count": int(sum(len(fragment.open_boundaries) for fragment in fragments)),
        "edge_semantics_audit_error_count": int(len(edge_semantics_audit_errors)),
        "fixed_node_owner_count": int(len(fixed_node_owner_by_id)),
        "fixed_owner_conflict_count": int(len(fixed_owner_conflicts)),
    }
    return TopologyClassificationResult(
        fragments=fragments,
        atomic_groups=atomic_groups,
        fixed_node_owner_by_id={
            int(node_id): int(fragment_id)
            for node_id, fragment_id in sorted(fixed_node_owner_by_id.items())
        },
        su24_support_bindings=su24_support_bindings,
        resource_pool=resource_pool,
        consumed_node_ids=sorted(int(x) for x in consumed),
        unclassified_nodes=unclassified_nodes,
        classification_rejects=classification_rejects,
        claim_conflicts=claim_conflicts,
        support_units_only=support_units_only,
        summary=summary,
    )
