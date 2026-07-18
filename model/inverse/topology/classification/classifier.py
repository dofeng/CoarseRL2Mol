from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from ....shared.inverse_common import _NodeV3
else:
    _NodeV3 = Any
from .classification_models import TopologyClassificationResult, TopologyFragment, TopologyRef
from ..iteration_ledger import normalize_slot
from ..stage3_models import TopologyClosureSlot, TopologySlotKind
from .classification_rules_basic import (
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
    SPECIAL_CONNECTOR_NEIGHBORS,
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
    is_visible_fixed_connector_semantic,
    is_visible_mainline_nonfixed_semantic,
    is_visible_terminal_semantic,
)


RESOURCE_POOL_SU_TYPES = {11, 22, 23, 24}
SU11_DONOR_SU_TYPES = {10, 12, 13}
DOUBLE_D2_CLASS_TYPES = {"A+", "B+", "C+", "D+", "E+"}
BASIC_CLASS_TYPES = {"A", "B", "C", "D", "E", "F", "G"}
FIXED_SIDE_PRIORITY = {"M": 0, "N": 1, "P": 2}
FLEX_RECONCILE_EXPLICIT_FLEX_CLASSES = {"A", "A+", "24_AB_M", "24_CD_M_FULL", "24_AB_N_24_AB"}
FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES = {"C", "C+", "F", "24_AB_P"}
FLEX_RECONCILE_B_LIKE_OPEN_CLASSES = {
    "B", "B+", "24_A", "24_B", "24_AB_N", "24_CD_M", "24_AB_N_24_CD", "25_AB",
}
FLEX_RECONCILE_D_LIKE_OPEN_CLASSES = {
    "D", "D+", "24_C", "24_D", "24_CD_N", "24_CD_N_24_CD", "25_CD",
}
FLEX_RECONCILE_E_LIKE_OPEN_CLASSES = {"E", "E+", "G", "24_CD_P"}
FLEX_RANGE_DEMOTABLE_EXPLICIT_CLASSES = {"24_AB_N_24_AB"}
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
    "24_AB_N_24_AB",
    "24_AB_N_24_CD",
    "24_CD_N_24_CD",
    "25_AB",
    "25_CD",
}
FLEX_RECONCILE_SIDE_CHAIN_CLASSES = {"C", "E", "F", "G", "C+", "E+", "24_AB_P", "24_CD_P"}
PAIRED_LIKE24_CLASS_TYPES = {"24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD"}

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


def _build_resource_pool(
    tokens: Sequence[ClassificationToken],
    consumed: Set[int],
    *,
    excluded_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    remaining_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in sorted(RESOURCE_POOL_SU_TYPES)}
    su11_donor_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in sorted(SU11_DONOR_SU_TYPES)}
    excluded = {int(node_id) for node_id in list(excluded_ids or [])}
    for token in list(tokens or []):
        if int(token.node_id) in consumed:
            continue
        if int(token.node_id) in excluded:
            continue
        if int(token.su_type) in RESOURCE_POOL_SU_TYPES:
            remaining_ids_by_su.setdefault(str(token.su_type), []).append(int(token.node_id))
        elif int(token.su_type) in SU11_DONOR_SU_TYPES:
            su11_donor_ids_by_su.setdefault(str(token.su_type), []).append(int(token.node_id))
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


def _neighbor_by_id(token: ClassificationToken, neighbor_id: int) -> Optional[NeighborToken]:
    for neighbor in list(token.neighbor_tokens or []):
        if int(neighbor.node_id) == int(neighbor_id):
            return neighbor
    return None


def _neighbor_su_set(token: ClassificationToken) -> Set[int]:
    return {int(neighbor.su_type) for neighbor in list(token.neighbor_tokens or [])}


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


def _collect_direct_claimed_ids(
    token: ClassificationToken,
    *,
    include_neighbor_ids: Optional[Iterable[int]] = None,
    exclude_neighbor_ids: Optional[Iterable[int]] = None,
    include_aromatic: bool = False,
    include_terminals: bool = False,
    include_fixed: bool = False,
    include_visible_mainline: bool = False,
    include_explicit_nonabstract: bool = False,
) -> List[int]:
    include_set = {int(x) for x in list(include_neighbor_ids or [])}
    exclude_set = {int(x) for x in list(exclude_neighbor_ids or [])}
    out: List[int] = [int(token.node_id)]
    for neighbor in list(token.neighbor_tokens or []):
        node_id = int(neighbor.node_id)
        if node_id in exclude_set:
            continue
        include = False
        if node_id in include_set:
            include = True
        elif include_fixed and is_visible_fixed_connector_semantic(neighbor):
            include = True
        elif include_aromatic and int(neighbor.su_type) in set(AROMATIC_SU_TYPES):
            include = True
        elif include_terminals and is_visible_terminal_semantic(neighbor):
            include = True
        elif include_visible_mainline and is_visible_mainline_nonfixed_semantic(neighbor):
            include = True
        elif include_explicit_nonabstract and not is_abstract_open_nonfixed_semantic(neighbor):
            include = True
        if include:
            out.append(node_id)
    return _union_source_ids(out)


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
    atomic_ids: Set[int] = set()
    atomic_group_ids: Set[int] = set()
    atomic_groups: List[Dict[str, Any]] = []
    for center_id in center_ids:
        token = tokens_by_id.get(int(center_id))
        group_node_ids = [
            int(value)
            for value in list(dict(getattr(token, "meta", {}) or {}).get("atomic_group_node_ids", []) or [])
        ]
        if not group_node_ids:
            continue
        group_id = int(dict(token.meta or {}).get("atomic_group_id", min(group_node_ids)))
        if group_id in atomic_group_ids:
            continue
        atomic_group_ids.add(group_id)
        atomic_ids.update(group_node_ids)
        atomic_groups.append({
            "group_id": group_id,
            "kind": str(dict(token.meta or {}).get("atomic_group_kind", "")),
            "node_ids": sorted(group_node_ids),
            "su_pair": [int(value) for value in list(dict(token.meta or {}).get("atomic_group_su_pair", []) or [])],
        })
    if not atomic_ids:
        return fragment
    fragment.claimed_node_ids = _union_source_ids(fragment.claimed_node_ids, atomic_ids)
    fragment.source_node_ids = _union_source_ids(fragment.source_node_ids, atomic_ids)
    fragment.evidence["atomic_group_ids"] = sorted(atomic_group_ids)
    fragment.evidence["atomic_groups"] = atomic_groups
    fragment.evidence["atomic_group_node_ids"] = sorted(atomic_ids)
    fragment_token = tokens_by_id.get(int(fragment.fragment_id))
    if fragment_token is not None and int(fragment_token.su_type) == 14:
        fragment.evidence["center_node_ids"] = sorted(atomic_ids)
        anchor_14_ids = sorted(
            int(node_id)
            for node_id in atomic_ids
            if int(getattr(tokens_by_id.get(int(node_id)), "su_type", -1)) == 14
        )
        fragment.evidence["required_branch_kinds_by_anchor"] = {
            str(node_id): ["H"]
            for node_id in anchor_14_ids
        }
        fragment.branch_slots = ["H" for _ in anchor_14_ids]
        fragment.branch_contracts = [
            TopologyClosureSlot(
                slot_id=f"atomic:{min(atomic_ids)}:H:{node_id}",
                kind=TopologySlotKind.H.value,
                anchor_node_id=int(node_id),
                anchor_role="unsaturated_branch_anchor",
                required_su_sequence=[23, 22],
            )
            for node_id in anchor_14_ids
        ]
    elif fragment_token is not None:
        ordered_atomic_ids = sorted(atomic_ids)
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
        pool_claims=[],
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
    resource_template: Dict[str, int] = {}
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
        resource_template = {"11": 1, "23": 1}
    elif supports_supplemental and mainline_open and class_name in FLEX_RECONCILE_D_LIKE_OPEN_CLASSES:
        count_role = "supplemental_candidate"
        candidate_kind = "d_like_open"
        can_supplement = True
        resource_template = {"11": 2, "23": 1}
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
        "flex_range_resource_template": {
            str(key): int(value)
            for key, value in dict(resource_template or {}).items()
            if int(value) > 0
        },
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
        pending_reason = "fixed_connector_unresolved"
    elif class_name in FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES or class_name in FLEX_RECONCILE_E_LIKE_OPEN_CLASSES:
        pending_kind = "explicit_side" if class_name in FLEX_RECONCILE_EXPLICIT_SIDE_CLASSES else "e_like_open"
        pending_reason = "remaining_external_side_open"
    elif class_name in FLEX_RECONCILE_B_LIKE_OPEN_CLASSES:
        pending_kind = "b_like_open"
    elif class_name in FLEX_RECONCILE_D_LIKE_OPEN_CLASSES:
        pending_kind = "d_like_open"

    return {
        "external_pending_kind": str(pending_kind),
        "external_pending_reason": str(pending_reason),
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
    resource_blocking = [
        int(node_id)
        for node_id in blocking
        if int(node_id) not in center_ids
        and int(node_id) not in atomic_ids
    ]
    if blocking and len(resource_blocking) == len(blocking):
        fragment.claimed_node_ids = [int(value) for value in node_ids if int(value) not in set(blocking)]
        fragment.source_node_ids = list(fragment.claimed_node_ids)
        fragment.context_node_ids = sorted({
            int(value)
            for value in list(getattr(fragment, "context_node_ids", []) or []) + list(blocking)
        })
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
    if is_abstract_open_nonfixed_semantic(neighbor) or is_visible_mainline_nonfixed_semantic(neighbor):
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
    fixed_neighbors = [neighbor for neighbor in list(token.neighbor_tokens or []) if bool(neighbor.is_fixed)]
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
    matches = [
        str(side_class)
        for side_class, spec in dict(side_rules or {}).items()
        if _side_rule_matches(remaining, dict(spec or {}))
    ]
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
    remaining = _profile_remaining_neighbors(center_token, connector_token)
    matches = [
        str(side_class)
        for side_class, spec in dict(side_rules or {}).items()
        if _side_rule_matches(
            remaining,
            dict(spec or {}),
            center_su=int(center_su),
        )
    ]
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
    valid = [
        dict(profile or {})
        for profile in list(connector_profiles or [])
        if str(dict(profile or {}).get("side_class", ""))
    ]
    if not valid:
        return None
    valid.sort(key=lambda item: (
        int(FIXED_SIDE_PRIORITY.get(str(item.get("side_class")), 99)),
        -int(len(list(item.get("remaining_node_ids", []) or []))),
        int(item.get("connector_node_id", 10**9)),
    ))
    return dict(valid[0])


def _like24_base_claimed_node_ids(token: ClassificationToken) -> List[int]:
    if str(token.subtype) == "24":
        return [int(token.node_id)]
    eleven_ids = sorted(
        int(neighbor.node_id)
        for neighbor in list(token.neighbor_tokens or [])
        if int(neighbor.su_type) == 11
    )
    unselected_eleven_ids = set(eleven_ids[1:])
    return _collect_direct_claimed_ids(
        token,
        include_aromatic=True,
        include_terminals=True,
        include_fixed=True,
        exclude_neighbor_ids=unselected_eleven_ids,
    )


def _like24_fixed_component_node_ids(
    token: ClassificationToken,
    tokens_by_id: Dict[int, ClassificationToken],
) -> List[int]:
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
        )
    return sorted(node_ids)


def _like24_secondary_claimed_node_ids(
    token: ClassificationToken,
    connector_profiles: Sequence[Dict[str, Any]],
) -> List[int]:
    return _union_source_ids(
        _like24_base_claimed_node_ids(token),
        [int(profile.get("connector_node_id", -1)) for profile in list(connector_profiles or [])],
        [
            int(node_id)
            for profile in list(connector_profiles or [])
            for node_id in list(profile.get("remaining_node_ids", []) or [])
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
    side_class = str(dict(profile or {}).get("side_class") or "")
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
    }
    connector_label = str(contract["connector_token_subtype"] or contract["connector_su_type"] or "")
    connector_node_id = int(contract["connector_node_id"])
    remaining_neighbors = [
        dict(item or {})
        for item in list(dict(profile or {}).get("remaining_neighbors", []) or [])
    ]
    remaining_labels = [_profile_neighbor_label(item) for item in remaining_neighbors]
    remaining_node_ids = [_safe_int(item.get("node_id"), -1) for item in remaining_neighbors]
    side_class = str(contract["side_class"])
    connector_su = int(contract["connector_su_type"])

    base_labels: List[str] = [connector_label] if connector_label else []
    base_node_ids: List[int] = [connector_node_id] if connector_node_id >= 0 else []
    tail_kind = "none"
    tail_reason = ""
    externally_closed = False

    if side_class in {"M", "P"}:
        for label, node_id in zip(remaining_labels, remaining_node_ids):
            if not label or node_id < 0:
                continue
            base_labels.append(str(label))
            base_node_ids.append(int(node_id))
        externally_closed = True
    elif side_class == "N":
        resolved_tail = False
        for label, node_id in zip(remaining_labels, remaining_node_ids):
            if connector_su in {0, 2} and str(label) in LIKE24_FIXED_OPEN_NEIGHBOR_LABELS:
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
        if not resolved_tail and remaining_labels:
            for label, node_id in zip(remaining_labels, remaining_node_ids):
                if not label or node_id < 0:
                    continue
                base_labels.append(str(label))
                base_node_ids.append(int(node_id))
        externally_closed = tail_kind == "none"

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
        if str(dict(profile or {}).get("side_class", ""))
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
            and not is_visible_mainline_nonfixed_semantic(neighbor)
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
        if subtype in {"double19d3", "double20d3"} and has_closed_fixed_branch:
            return {}
        return {"center": ["H"]}
    if class_name in {"24_C", "24_D"} and subtype == "14":
        if terminal_visible_count > 0:
            return {"center": []}
        return {"center": ["H"]}
    if class_name in PAIRED_LIKE24_CLASS_TYPES:
        return {}
    generic_count = _like24_direct_generic_count(token)
    expected_mainline_generic = _like24_expected_mainline_generic_count(class_name, connector_profiles)
    unresolved_count = max(0, int(generic_count - expected_mainline_generic))
    if unresolved_count <= 0:
        return {}
    del realization_modes
    return {"center": ["H" for _ in range(int(unresolved_count))]}


def _like24_required_branch_closure_specs(
    token: ClassificationToken,
    *,
    class_type: str,
    connector_profiles: Sequence[Dict[str, Any]],
    realization_modes: Sequence[str],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    branch_map = _like24_required_branch_kinds(
        token,
        class_type=str(class_type),
        connector_profiles=connector_profiles,
        realization_modes=realization_modes,
    )
    center_branch_kinds = [str(value) for value in list(branch_map.get("center", []) or []) if str(value)]
    for index, branch_kind in enumerate(center_branch_kinds):
        specs.append({
            "slot_id": f"center:{index}",
            "anchor": "center",
            "closure_kind": str(branch_kind),
            "attach_role": "center_branch",
            "target_path": "center_side_branch",
        })
    return specs


def _like24_branch_closures_in_ring(
    *,
    class_type: str,
    required_branch_kinds_by_anchor: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    class_name = str(class_type)
    center_branches = [str(value) for value in list(required_branch_kinds_by_anchor.get("center", []) or [])]
    if not center_branches:
        return {"side": {}, "upper_lower": {}, "fused": {}}
    if class_name in {"24_A", "24_B", "24_AB_N"}:
        return {
            "side": {},
            "upper_lower": {},
            "fused": {},
        }
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
    return _clone_contracts(left_fragment, left_center_id) + _clone_contracts(right_fragment, right_center_id)


def _like24_branch_suppression_by_skeleton(class_type: str) -> Dict[str, List[str]]:
    class_name = str(class_type)
    if class_name in {"24_A", "24_B", "24_AB_N"}:
        return {
            "side": ["center"],
            "upper_lower": ["center"] if class_name == "24_A" else [],
            "fused": ["center"],
        }
    return {
        "side": [],
        "upper_lower": [],
        "fused": [],
    }


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
    mainline_fixed_connector_forbidden_for_ring: bool = False,
    requires_preclosed_branch_before_ring: bool = False,
    complete_requires_all_slots_filled: bool = True,
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
        "mainline_fixed_connector_forbidden_for_ring": bool(mainline_fixed_connector_forbidden_for_ring),
        "requires_preclosed_branch_before_ring": bool(requires_preclosed_branch_before_ring),
        "complete_requires_all_slots_filled": bool(complete_requires_all_slots_filled),
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
        if is_abstract_open_nonfixed_semantic(neighbor) or is_visible_mainline_nonfixed_semantic(neighbor):
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
    required_branch_closure_specs = _like24_required_branch_closure_specs(
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
    branch_suppression_by_skeleton = _like24_branch_suppression_by_skeleton(str(class_type))
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
        contract_local["mainline_fixed_connector_forbidden_for_ring"] = bool(
            primary_connector_su in {0, 2} and bool(primary_forces_mainline)
        )
        contract_local["fixed_connector_ring_eligible"] = bool(
            not contract_local["mainline_fixed_connector_forbidden_for_ring"]
            and not bool(contract_local.get("blocking_reason", ""))
        )
        contract_local["requires_preclosed_branch_before_ring"] = bool(slot_branch_map)
        contract_local["complete_requires_all_slots_filled"] = True
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
    return {
        "fixed_connection_contracts": [dict(item or {}) for item in list(fixed_connection_contracts or [])],
        "required_branch_kinds_by_anchor": required_branch_kinds_by_anchor,
        "required_branch_closures_by_anchor": dict(required_branch_kinds_by_anchor or {}),
        "required_branch_closure_specs": required_branch_closure_specs,
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
        "branch_suppression_by_skeleton": branch_suppression_by_skeleton,
        "required_mainline_closure_modes": list(required_mainline_closure_modes or []),
        "allowed_pool_fillers_by_slot": allowed_pool_fillers_by_slot,
        "fixed_connection_context": fixed_connection_context,
        "bridgehead_policy": bridgehead_policy,
        "ring_contracts": ring_contracts,
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
        allowed.append("side")
        slots.extend(["side_double_anchor", "side_anchor_pair"])
        slot_profiles.extend(["side_anchor_pair"])
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
    return {
        "required_branch_kinds_by_anchor": {},
        "required_branch_closures_by_anchor": {},
        "required_branch_closure_specs": [],
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
        "branch_suppression_by_skeleton": {"side": [], "upper_lower": [], "fused": []},
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
                    class_type in PAIRED_LIKE24_CLASS_TYPES
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
                complete_requires_all_slots_filled=True,
                outer_anchor_must_touch_aromatic=class_type in {"24_AB_N_24_AB", "24_AB_N_24_CD"},
                touches_aromatic_anchor=bool(touches_aromatic_anchor),
                render_policy={"suppress_generic_open": True},
            ),
            "upper_lower_ring_plan": _make_ring_contract(
                eligible=class_type == "24_CD_N_24_CD",
                priority=5,
                slot_roles=["upper_lower_core", "upper_lower_pair_core"] if class_type == "24_CD_N_24_CD" else [],
                slot_coverage=["core_segment"] if class_type == "24_CD_N_24_CD" else [],
                slot_coverage_mode="pair_core" if class_type == "24_CD_N_24_CD" else "single_slot",
                required_real_like24_slots=["core_segment"] if class_type == "24_CD_N_24_CD" else [],
                required_branch_kinds_by_slot=branch_kinds_by_slot["upper_lower_ring_plan"],
                optional_pool23_slots=["left_bridge", "right_bridge"] if class_type == "24_CD_N_24_CD" else [],
                optional_pool24_slots=["core_segment"] if class_type == "24_CD_N_24_CD" else [],
                fixed_connector_position="dual",
                complete_requires_all_slots_filled=True,
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
                complete_requires_all_slots_filled=True,
                outer_anchor_must_touch_aromatic=class_type == "24_CD_N_24_CD",
                render_policy={"suppress_generic_open": True},
            ),
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
    if any(not str(profile.get("side_class", "")) for profile in connector_profiles):
        _record_reject(
            rejects,
            token,
            family="double_d2",
            reason="connector_side_classification_failed",
            details={"connector_profiles": connector_profiles},
        )
        return None
    key = tuple(sorted(str(profile.get("side_class")) for profile in connector_profiles))
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
    evidence.update(_like24_ring_evidence(
        token,
        class_type=str(base_rule.get("class_type")),
        topology_template=topology_template,
        connector_profiles=[],
        primary_connector=None,
        realization_modes=[],
    ))
    claimed_node_ids = _like24_fixed_component_node_ids(token, tokens_by_id)
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
    all_side_classes = [str(profile.get("side_class")) for profile in list(connector_profiles or [])]
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
    # A physical/raw SU24 is a free resource until the dedicated raw-SU24
    # ring phase explicitly selects it.  It is not an external half-closed
    # topology fragment.  Genuine like24 branch structures are represented by
    # non-raw centers and continue through the secondary 24_AB_*/24_CD_* rules.
    if int(token.su_type) == 24:
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
    if any(not str(profile.get("side_class", "")) for profile in connector_profiles):
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
    claimed_node_ids = _like24_secondary_claimed_node_ids(token, connector_profiles)
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
            branch_slots=["H"] if "?" in str(rule.get("topology_template", "")) else [],
            pool_claims=[],
            evidence=_augment_fragment_evidence(
                family=str(rule.get("family")),
                class_type=str(rule.get("class_type")),
                fixed_connector_inferred=False,
                topology_template=str(rule.get("topology_template")),
                open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
                branch_slots=["H"] if "?" in str(rule.get("topology_template", "")) else [],
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
            branch_slots=["H"] if "?" in str(rule.get("topology_template", "")) else [],
            pool_claims=[],
            evidence=_augment_fragment_evidence(
                family=str(rule.get("family")),
                class_type=str(rule.get("class_type")),
                fixed_connector_inferred=False,
                topology_template=str(rule.get("topology_template")),
                open_slots=["OPEN"] if str(rule.get("topology_template", "")).endswith("-") else [],
                branch_slots=["H"] if "?" in str(rule.get("topology_template", "")) else [],
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
    matched_neighbors = [
        dict(item or {})
        for item in list(dict(evidence or {}).get("matched_neighbors", []) or [])
    ]
    claim_neighbor_ids = [
        _dict.get("node_id")
        for _dict in matched_neighbors
        if (
            int(_dict.get("su_type", -1)) in set(AROMATIC_SU_TYPES)
            or is_visible_terminal_semantic(_dict)
            or is_visible_fixed_connector_semantic(_dict)
            or is_visible_mainline_nonfixed_semantic(_dict)
            or not is_abstract_open_nonfixed_semantic(_dict)
        )
    ]
    direct_source_ids = _collect_direct_source_ids(token)
    return _make_fragment(
        token,
        family=str(rule.get("family")),
        class_type=str(rule.get("class_type")),
        topology_template=topology_template,
        open_slots=["OPEN"] if not bool(rule.get("closed_main", False)) else [],
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
    has_11 = any(int(neighbor.su_type) == 11 for neighbor in list(token.neighbor_tokens or []))
    has_22_like = any(
        int(neighbor.su_type) in ({22} | TERMINAL_NEIGHBORS | {0, 1, 2, 3, 14, 15, 17, 24, 25})
        for neighbor in list(token.neighbor_tokens or [])
    )
    if not has_22_like:
        return None
    rule = SU25_RULES[0] if has_11 else SU25_RULES[1]
    return _make_fragment(
        token,
        family=str(rule.get("family")),
        class_type=str(rule.get("class_type")),
        topology_template=str(rule.get("topology_template")),
        open_slots=["OPEN"],
        branch_slots=["H"],
        evidence={
            "rule_id": str(rule.get("rule_id")),
            "center_node_ids": [int(token.node_id)],
            "connector_profiles": [],
            "selected_primary_connector": {},
            "secondary_connector_profiles": [],
            "fixed_connection_contracts": [],
            "realization_modes": [],
            "reject_reason": "",
        },
        source_node_ids=_collect_direct_source_ids(token),
        claimed_node_ids=_collect_direct_claimed_ids(
            token,
            include_aromatic=True,
            include_terminals=True,
            include_fixed=True,
            include_visible_mainline=True,
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
        "resource_pool_member": bool(int(token.su_type) in RESOURCE_POOL_SU_TYPES),
        "reason": str(reason),
    }
    payload.update(dict(extra or {}))
    return payload


def _atomic_pair_fallback_fragment(
    group: Dict[str, Any],
    tokens_by_id: Dict[int, ClassificationToken],
) -> TopologyFragment:
    node_ids = sorted(int(value) for value in list(group.get("node_ids", []) or []))
    tokens = [tokens_by_id[node_id] for node_id in node_ids]
    su_pair = tuple(int(token.su_type) for token in tokens)
    branch_slots: List[TopologyClosureSlot] = []
    mainline_slots: List[TopologyClosureSlot] = []
    if 14 in su_pair:
        class_type = "24_CD_M_FULL" if su_pair == (14, 14) else "24_C"
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
        topology_template = "-" + "-".join(
            f"14(-H)" if int(token.su_type) == 14 else str(int(token.su_type))
            for token in tokens
        ) + ("-" if int(tokens[-1].su_type) != 16 else "")
    else:
        class_type = "G"
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
        evidence={
            "rule_id": "unsaturated_atomic_pair_fallback",
            "center_node_ids": list(node_ids),
            "atomic_group_id": int(group.get("group_id", node_ids[0])),
            "atomic_group_node_ids": list(node_ids),
            "atomic_group_su_pair": list(su_pair),
        },
    )


def _normalize_fragment_contract(fragment: TopologyFragment) -> TopologyFragment:
    original_source = sorted({int(value) for value in list(fragment.source_node_ids or [])})
    claimed = sorted({int(value) for value in list(fragment.claimed_node_ids or original_source)})
    fragment.context_node_ids = sorted(set(original_source) - set(claimed))
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
        main_kind = TopologySlotKind.F2.value if str(fragment.class_type) == "25_CD" else TopologySlotKind.F1.value
        main_required = [23, 22] if main_kind == TopologySlotKind.F2.value else [23, 11]
        fragment.open_mainline_slots = [TopologyClosureSlot(
            slot_id=f"fragment:{fragment.fragment_id}:main:0",
            kind=main_kind,
            anchor_node_id=int(fragment.fragment_id),
            anchor_role="left_mainline",
            required_su_sequence=main_required,
        )]
        fragment.branch_contracts = [TopologyClosureSlot(
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
        fragment.open_slots = [main_kind]
        fragment.branch_slots = [TopologySlotKind.H.value]
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
    if not fragment.open_mainline_slots:
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
                token.anchor_partition = str(partition_by_node[int(token.node_id)])
                token.meta["fixed_partition_meta_applied"] = True
            if int(token.node_id) in forbidden_fixed_connector_node_ids:
                token.meta["fixed_connector_forbidden"] = True
    tokens_by_id = _token_by_id(tokens)
    consumed: Set[int] = set()
    fragments: List[TopologyFragment] = []
    classification_rejects: List[Dict[str, Any]] = []
    claim_conflicts: List[Dict[str, Any]] = []
    atomic_groups = _build_unsaturated_atomic_groups(tokens, classification_rejects)
    rejected_atomic_node_ids = {
        int(item.get("node_id", -1))
        for item in classification_rejects
        if str(item.get("family", "")) == "unsaturated_atomic_pair"
    }

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids:
            continue
        fragment = _find_double_d2_fragment(token, tokens_by_id, classification_rejects)
        if fragment is not None:
            fragment = _attach_center_atomic_group(fragment, tokens_by_id)
            _append_if_unclaimed(fragments, consumed, fragment)

    per_token_like24: Dict[int, TopologyFragment] = {}
    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids:
            continue
        fragment = _match_like24_fragment(token, tokens_by_id, classification_rejects)
        if fragment is not None:
            per_token_like24[int(token.node_id)] = fragment

    pair_bridge_fragments = _match_like24_pair_bridge_fragments(tokens, per_token_like24, consumed)
    for fragment in pair_bridge_fragments:
        fragment = _attach_center_atomic_group(fragment, tokens_by_id)
        _append_if_unclaimed(fragments, consumed, fragment)

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids:
            continue
        like24_fragment = per_token_like24.get(int(token.node_id))
        if like24_fragment is not None:
            like24_fragment = _attach_center_atomic_group(like24_fragment, tokens_by_id)
            _append_if_unclaimed(fragments, consumed, like24_fragment)

    for token in tokens:
        if int(token.node_id) in consumed or int(token.node_id) in rejected_atomic_node_ids:
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

    for token in tokens:
        if int(token.node_id) in consumed:
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

    fragments = [_normalize_fragment_contract(fragment) for fragment in fragments]

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

    token_lookup = _token_by_id(tokens)
    referenced_source_ids: Set[int] = set()
    for fragment in list(fragments or []):
        center_ids = [
            int(node_id)
            for node_id in list(dict(getattr(fragment, "evidence", {}) or {}).get("center_node_ids", []) or [])
        ]
        referenced_source_ids.update(
            int(node_id)
            for node_id in list(getattr(fragment, "source_node_ids", []) or [])
        )
    reusable_abstract_su23_ids = {
        int(node_id)
        for node_id in referenced_source_ids
        if int(getattr(token_lookup.get(int(node_id)), "su_type", -1)) == 23
        and is_abstract_open_nonfixed_semantic(token_lookup.get(int(node_id)))
    }
    resource_pool = _build_resource_pool(
        tokens,
        consumed,
        excluded_ids=referenced_source_ids - reusable_abstract_su23_ids,
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
        if int(token.su_type) in RESOURCE_POOL_SU_TYPES:
            continue
        like24_fixed_partner_support = _is_like24_fixed_partner_support(token, token_lookup)
        if (
            int(token.node_id) in referenced_source_ids
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
                        "like24_fixed_partner_support"
                        if bool(like24_fixed_partner_support)
                        else (
                        "semantic_open_support"
                        if str(dict(token.meta or {}).get("semantic_bucket", ""))
                        else "support_unit_only"
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

    summary = {
        "fragment_count": int(len(fragments)),
        "closed_fragment_count": int(sum(1 for fragment in fragments if not list(fragment.open_slots or []))),
        "open_fragment_count": int(sum(1 for fragment in fragments if bool(list(fragment.open_slots or [])))),
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
    }
    return TopologyClassificationResult(
        fragments=fragments,
        atomic_groups=atomic_groups,
        resource_pool=resource_pool,
        consumed_node_ids=sorted(int(x) for x in consumed),
        unclassified_nodes=unclassified_nodes,
        classification_rejects=classification_rejects,
        claim_conflicts=claim_conflicts,
        support_units_only=support_units_only,
        summary=summary,
    )
