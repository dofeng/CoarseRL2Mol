"""Validation of materialized aliphatic-ring topology contracts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

from ...classification.classification_tokens import (
    AROMATIC_SU_TYPES,
    TERMINAL_BASE_SU_TYPES,
    TERMINAL_SUBTYPES,
)
from ...stage3_models import RingTopologyContract


RING_KINDS = {"side", "upper_lower", "fused"}
RING_LIKE24_LIMITS = {
    "side": (2, 4),
    "upper_lower": (2, 3),
    "fused": (4, 6),
}

RECYCLABLE_PURE_RESOURCE_SU_TYPES = frozenset({11, 22, 23, 24})


def _edge(left: int, right: int) -> Tuple[int, int]:
    return tuple(sorted((int(left), int(right))))


def cycle_edges(cycle: Sequence[int]) -> Set[Tuple[int, int]]:
    node_ids = [int(value) for value in list(cycle or [])]
    if len(node_ids) < 2:
        return set()
    return {
        _edge(node_ids[index], node_ids[(index + 1) % len(node_ids)])
        for index in range(len(node_ids))
    }


def _real_node_lookup(nodes: Iterable[Any]) -> Dict[int, Any]:
    lookup: Dict[int, Any] = {}
    for node in list(nodes or []):
        try:
            lookup[int(getattr(node, "global_id"))] = node
        except (TypeError, ValueError, AttributeError):
            continue
    return lookup


def _branch_reason(
    contract: RingTopologyContract,
    ring_node_ids: Set[int],
) -> Optional[str]:
    occupied: Set[int] = set(ring_node_ids)
    for index, raw_path in enumerate(list(contract.branch_paths or [])):
        path = dict(raw_path or {})
        anchor_id = int(path.get("anchor_node_id", -1))
        node_ids = [int(value) for value in list(path.get("node_ids", []) or [])]
        if anchor_id not in ring_node_ids:
            return f"ring_branch_anchor_not_in_cycle:{index}:{anchor_id}"
        if not node_ids:
            return f"ring_empty_branch_path:{index}"
        duplicates = sorted({value for value in node_ids if node_ids.count(value) > 1})
        if duplicates:
            return f"ring_duplicate_branch_node_ids:{index}:{duplicates}"
        terminal_positions = [
            position
            for position, node_id in enumerate(node_ids)
            if int(contract.node_su_by_id.get(int(node_id), -1)) in set(TERMINAL_BASE_SU_TYPES)
            or str(contract.node_labels_by_id.get(int(node_id), "")) in set(TERMINAL_SUBTYPES)
        ]
        if terminal_positions and terminal_positions != [len(node_ids) - 1]:
            return f"ring_branch_terminal_not_at_path_end:{index}:{node_ids}"
        overlap = sorted(set(node_ids) & occupied)
        if overlap:
            return f"ring_branch_node_reused:{index}:{overlap}"
        occupied.update(node_ids)
    replaceable_anchors = {
        int(value)
        for role in ("fixed_anchor", "bridgeheads")
        for value in list(dict(contract.role_node_ids or {}).get(role, []) or [])
    }
    for index, raw_requirement in enumerate(list(contract.branch_requirements or [])):
        requirement = dict(raw_requirement or {})
        state = str(requirement.get("state", ""))
        kind = str(requirement.get("kind", ""))
        anchor_id = int(requirement.get("anchor_node_id", -1))
        if state not in {"FILLED", "REPLACED_BY_RING_BONDS"}:
            return f"ring_branch_requirement_open:{index}:{requirement.get('kind', '')}"
        if state == "REPLACED_BY_RING_BONDS":
            if kind not in {"H", "H1"}:
                return f"ring_non_h_branch_replaced:{index}:{kind}"
            if anchor_id not in replaceable_anchors:
                return f"ring_branch_replacement_role_mismatch:{index}:{anchor_id}"
        if state == "FILLED" and kind in {"H", "H1", "F1", "F2"}:
            if not list(requirement.get("node_ids", []) or []):
                return f"ring_filled_branch_has_no_nodes:{index}:{kind}"
    return None


def audit_ring_topology_contract(
    contract: RingTopologyContract,
    *,
    nodes: Iterable[Any],
    owned_node_ids: Optional[Iterable[int]] = None,
) -> Optional[str]:
    """Return the first physical-contract failure, or ``None`` when valid."""

    kind = str(contract.ring_kind or "")
    if kind not in RING_KINDS:
        return f"unknown_ring_kind:{kind}"
    cycles = [[int(value) for value in list(cycle or [])] for cycle in list(contract.cycles or [])]
    expected_cycle_count = 2 if kind == "fused" else 1
    if len(cycles) != expected_cycle_count:
        return f"ring_cycle_count:{kind}:{len(cycles)}"
    for index, cycle in enumerate(cycles):
        if len(cycle) != 6:
            return f"ring_cycle_node_count:{kind}:{index}:{len(cycle)}"
        if len(set(cycle)) != 6:
            return f"ring_cycle_reuses_node:{kind}:{index}"

    ring_node_ids = {node_id for cycle in cycles for node_id in cycle}
    su_by_id = {
        int(key): int(value)
        for key, value in dict(contract.node_su_by_id or {}).items()
    }
    labels_by_id = {
        int(key): str(value)
        for key, value in dict(contract.node_labels_by_id or {}).items()
    }
    forbidden_terminal_ids = sorted(
        int(node_id)
        for node_id in ring_node_ids
        if int(su_by_id.get(int(node_id), -1)) in set(TERMINAL_BASE_SU_TYPES)
        or str(labels_by_id.get(int(node_id), "")) in set(TERMINAL_SUBTYPES)
    )
    if forbidden_terminal_ids:
        return f"ring_terminal_in_backbone:{forbidden_terminal_ids}"
    profiles = {
        int(key): str(value).upper()
        for key, value in dict(contract.node_profiles_by_id or {}).items()
    }
    if kind == "side":
        labels = dict(contract.node_labels_by_id or {})
        left_id, right_id = int(cycles[0][0]), int(cycles[0][-1])
        left_aromatic = int(su_by_id.get(left_id, -1)) in set(AROMATIC_SU_TYPES)
        right_aromatic = int(su_by_id.get(right_id, -1)) in set(AROMATIC_SU_TYPES)
        if not left_aromatic or not right_aromatic:
            return (
                "side_ring_nonaromatic_endpoints:"
                f"{labels.get(left_id, '')}|{labels.get(right_id, '')}"
            )
    elif kind == "upper_lower":
        fixed_ids = [
            int(value)
            for value in list(dict(contract.role_node_ids or {}).get("fixed_anchor", []) or [])
        ]
        if len(fixed_ids) != 1 or fixed_ids[0] != int(cycles[0][0]):
            return "upper_ring_fixed_anchor_mismatch"
        if profiles.get(int(fixed_ids[0]), "") != "A":
            return f"upper_ring_fixed_anchor_not_profile_a:{int(fixed_ids[0])}"
        attachments = [
            [int(value) for value in list(edge or [])]
            for edge in list(contract.external_attachments or [])
        ]
        fixed_attachments = [edge for edge in attachments if len(edge) == 2 and fixed_ids[0] in edge]
        if len(fixed_attachments) != 1:
            return "upper_ring_external_su11_attachment_count"
        external_id = next(value for value in fixed_attachments[0] if value != fixed_ids[0])
        external_su = int(dict(contract.node_su_by_id or {}).get(external_id, -1))
        if external_su not in set(AROMATIC_SU_TYPES):
            return f"upper_ring_external_anchor_not_aromatic:{external_id}"
        if external_id in ring_node_ids:
            return f"upper_ring_external_anchor_in_cycle:{external_id}"
    else:
        outer, inner = cycles
        bridgeheads = set(outer) & set(inner)
        expected_bridgeheads = {
            int(value)
            for value in list(dict(contract.role_node_ids or {}).get("bridgeheads", []) or [])
        }
        if len(bridgeheads) != 2 or bridgeheads != expected_bridgeheads:
            return f"fused_ring_bridgehead_mismatch:{sorted(bridgeheads)}"
        shared_edges = cycle_edges(outer) & cycle_edges(inner)
        if shared_edges != {_edge(*sorted(bridgeheads))}:
            return f"fused_ring_shared_edge_mismatch:{sorted(shared_edges)}"
        if len(ring_node_ids) != 10:
            return f"fused_ring_unique_node_count:{len(ring_node_ids)}"
        bridgehead_kind = str(contract.bridgehead_kind or "")
        bridgehead_profiles = {
            profiles.get(int(node_id), "") for node_id in bridgeheads
        }
        bridgehead_su = {
            int(dict(contract.node_su_by_id or {}).get(int(node_id), -1))
            for node_id in bridgeheads
        }
        if bridgehead_kind == "real_24c":
            if bridgehead_profiles != {"C"} or bridgehead_su != {24}:
                return "fused_ring_bridgeheads_not_real_24c"
        elif bridgehead_kind == "substitute_14_pair":
            if bridgehead_su != {14}:
                return "fused_ring_bridgeheads_not_14_pair"
        else:
            return f"fused_ring_bridgehead_kind_missing:{bridgehead_kind}"

    like24_count = len({int(value) for value in list(contract.like24_node_ids or [])})
    minimum, maximum = RING_LIKE24_LIMITS[kind]
    if like24_count < minimum or like24_count > maximum:
        return f"ring_like24_count:{kind}:{like24_count}"
    if not set(int(value) for value in list(contract.like24_node_ids or [])) <= ring_node_ids:
        return "ring_like24_node_not_in_cycle"
    missing_profiles = sorted(
        int(value)
        for value in list(contract.like24_node_ids or [])
        if int(value) not in profiles
    )
    if missing_profiles:
        return f"ring_like24_profile_missing:{missing_profiles}"

    real_nodes = _real_node_lookup(nodes)
    declared = set(int(value) for value in contract.declared_node_ids())
    missing_su = sorted(declared - set(su_by_id))
    if missing_su:
        return f"ring_declared_su_missing:{missing_su}"
    missing_real = sorted(declared - set(real_nodes))
    if missing_real:
        return f"ring_declared_node_missing:{missing_real}"
    for node_id, declared_su in dict(contract.node_su_by_id or {}).items():
        actual_su = int(getattr(real_nodes[int(node_id)], "su_type", -1))
        if actual_su != int(declared_su):
            return f"ring_declared_su_mismatch:{int(node_id)}:{int(declared_su)}:{actual_su}"

    required_declared = set(ring_node_ids)
    required_declared.update(
        int(value) for value in list(contract.support_node_ids or [])
    )
    attachment_edges: Set[Tuple[int, int]] = set()
    for index, attachment in enumerate(list(contract.external_attachments or [])):
        edge = [int(value) for value in list(attachment or [])]
        if len(edge) != 2 or edge[0] == edge[1]:
            return f"ring_external_attachment_invalid:{index}:{edge}"
        normalized_edge = _edge(edge[0], edge[1])
        if normalized_edge in attachment_edges:
            return f"ring_external_attachment_duplicate:{index}:{edge}"
        attachment_edges.add(normalized_edge)
        if len(set(edge) & ring_node_ids) != 1:
            return f"ring_external_attachment_cycle_incidence:{index}:{edge}"
        required_declared.update(edge)
    for path in list(contract.branch_paths or []):
        required_declared.add(int(dict(path or {}).get("anchor_node_id", -1)))
        required_declared.update(int(value) for value in list(dict(path or {}).get("node_ids", []) or []))
    if required_declared != declared:
        return (
            "ring_declared_node_set_mismatch:"
            f"missing={sorted(required_declared - declared)}:extra={sorted(declared - required_declared)}"
        )

    branch_reason = _branch_reason(contract, ring_node_ids)
    if branch_reason is not None:
        return branch_reason
    if owned_node_ids is not None:
        owned = {int(value) for value in list(owned_node_ids or [])}
        if declared != owned:
            return (
                "ring_transaction_ownership_mismatch:"
                f"missing={sorted(declared - owned)}:extra={sorted(owned - declared)}"
            )
    if not str(contract.topology_notation or "") or not str(contract.node_id_notation or ""):
        return "ring_standard_render_missing"
    return None


def is_recyclable_pure_resource_ring(
    product: Any,
    *,
    assembler: Optional[Any] = None,
) -> bool:
    """Return whether a completed ring may be dismantled for external resources.

    ``source_fragment_ids`` is intentionally *not* part of this predicate.
    Classifier-fixed 11/22/23 nodes can be embedded in a ring that is still
    physically a pure resource ring.  The decision is made from the complete
    node-bound contract and actual SU types instead.
    """
    if product is None:
        return False
    kind = str(getattr(product, "product_kind", "") or "")
    if not kind.startswith("fatty_ring_"):
        return False
    contract = getattr(product, "ring_contract", None)
    if contract is None or not bool(getattr(contract, "complete", False)):
        return False
    ring_kind = str(getattr(contract, "ring_kind", "") or "")
    if ring_kind not in RING_KINDS:
        return False
    node_su = {
        int(node_id): int(su_type)
        for node_id, su_type in dict(
            getattr(contract, "node_su_by_id", {}) or {}
        ).items()
    }
    declared = {
        int(node_id) for node_id in contract.declared_node_ids()
    }
    if not declared or set(node_su) != declared:
        return False
    if not set(node_su.values()).issubset(RECYCLABLE_PURE_RESOURCE_SU_TYPES):
        return False
    if assembler is not None:
        real_nodes = _real_node_lookup(getattr(assembler, "nodes", []))
        if any(
            int(node_id) not in real_nodes
            or int(getattr(real_nodes[int(node_id)], "su_type", -1))
            != int(su_type)
            for node_id, su_type in node_su.items()
        ):
            return False
    return True
