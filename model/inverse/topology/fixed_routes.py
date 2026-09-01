"""Authoritative fixed-branch contract route validation.

Fragment ownership describes coverage, not connectivity.  A fixed branch route
is therefore resolved only from the concrete node IDs declared by its contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ...shared.inverse_common import _NodeV3
from .classification.classification_tokens import (
    AROMATIC_SU_TYPES,
    TERMINAL_BASE_SU_TYPES,
    TERMINAL_SUBTYPES,
    normalize_subtype,
)
from .fixed_edges import nodes_have_fixed_edge as _nodes_have_fixed_edge


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _node_label(node: _NodeV3) -> str:
    subtype = str(normalize_subtype(node))
    su_type = _safe_int(getattr(node, "su_type", -1), -1)
    return subtype if subtype not in {str(su_type), ""} else str(su_type)


def _is_terminal(node: _NodeV3) -> bool:
    return bool(
        _safe_int(getattr(node, "su_type", -1), -1)
        in set(TERMINAL_BASE_SU_TYPES)
        or str(normalize_subtype(node)) in set(TERMINAL_SUBTYPES)
    )


def _canonical_slot(slot: Any) -> Dict[str, Any]:
    payload = slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
    return {
        "slot_id": str(payload.get("slot_id", "") or ""),
        "kind": str(payload.get("kind", "") or ""),
        "anchor_node_id": _safe_int(payload.get("anchor_node_id"), -1),
        "anchor_role": str(payload.get("anchor_role", "") or ""),
        "required_su_sequence": [
            _safe_int(value, -1)
            for value in list(payload.get("required_su_sequence", []) or [])
        ],
        "existing_node_ids": [
            _safe_int(value, -1)
            for value in list(payload.get("existing_node_ids", []) or [])
        ],
        "state": str(payload.get("state", "") or ""),
    }


def _canonical_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(contract or {})
    boundary = dict(payload.get("structural_boundary", {}) or {})
    return {
        "center_node_id": _safe_int(payload.get("center_node_id"), -1),
        "connector_node_id": _safe_int(payload.get("connector_node_id"), -1),
        "connector_su_type": _safe_int(payload.get("connector_su_type"), -1),
        "side_class": str(payload.get("side_class", "") or ""),
        "role_candidates": sorted({
            str(value)
            for value in list(payload.get("role_candidates", []) or [])
            if str(value)
        }),
        "base_path_labels": [
            str(value)
            for value in list(payload.get("base_path_labels", []) or [])
        ],
        "base_path_node_ids": [
            _safe_int(value, -1)
            for value in list(payload.get("base_path_node_ids", []) or [])
        ],
        "tail_closure_kind": str(payload.get("tail_closure_kind", "") or ""),
        "tail_open_reason": str(payload.get("tail_open_reason", "") or ""),
        "externally_closed": bool(payload.get("externally_closed", False)),
        "structural_boundary": {
            str(key): boundary[key]
            for key in sorted(boundary)
        },
    }


def fragment_structure_snapshot(fragment: Any) -> Dict[str, Any]:
    """Return the planner/assembler structural contract for one fragment."""

    evidence = dict(getattr(fragment, "evidence", {}) or {})
    contracts = [
        _canonical_contract(dict(value or {}))
        for value in list(evidence.get("fixed_connection_contracts", []) or [])
    ]
    contracts.sort(key=lambda item: (
        int(item["center_node_id"]),
        int(item["connector_node_id"]),
        tuple(item["base_path_node_ids"]),
        tuple(item["role_candidates"]),
    ))
    open_slots = [
        _canonical_slot(value)
        for value in list(getattr(fragment, "open_mainline_slots", []) or [])
    ]
    branch_slots = [
        _canonical_slot(value)
        for value in list(getattr(fragment, "branch_contracts", []) or [])
    ]
    slot_key = lambda item: (
        int(item["anchor_node_id"]),
        str(item["anchor_role"]),
        str(item["kind"]),
        str(item["slot_id"]),
    )
    open_slots.sort(key=slot_key)
    branch_slots.sort(key=slot_key)
    return {
        "fragment_id": _safe_int(getattr(fragment, "fragment_id", -1), -1),
        "family": str(getattr(fragment, "family", "") or ""),
        "class_type": str(getattr(fragment, "class_type", "") or ""),
        "mainline_node_ids": [
            _safe_int(value, -1)
            for value in list(getattr(fragment, "mainline_node_ids", []) or [])
        ],
        "mainline_labels": [
            str(value)
            for value in list(getattr(fragment, "mainline_labels", []) or [])
        ],
        "open_mainline_slots": open_slots,
        "branch_slots": branch_slots,
        "fixed_connection_contracts": contracts,
    }


def fragment_structure_signature(fragment: Any) -> str:
    payload = json.dumps(
        fragment_structure_snapshot(fragment),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FixedRouteResolution:
    labels: Tuple[str, ...] = ()
    node_ids: Tuple[int, ...] = ()
    source_fragment_ids: Tuple[int, ...] = ()
    terminal_node_id: Optional[int] = None
    reason: str = ""
    failure_domain: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not bool(self.reason)


def resolve_fixed_contract_route(
    contract: Mapping[str, Any],
    *,
    nodes_by_id: Mapping[int, _NodeV3],
    mainline_node_ids: Sequence[int] = (),
    node_owner_by_id: Optional[Mapping[int, int]] = None,
) -> FixedRouteResolution:
    """Validate one concrete branch route without expanding fragment ownership."""

    payload = dict(contract or {})
    labels = tuple(
        str(value)
        for value in list(payload.get("base_path_labels", []) or [])
        if str(value)
    )
    node_ids = tuple(
        _safe_int(value, -1)
        for value in list(payload.get("base_path_node_ids", []) or [])
        if _safe_int(value, -1) >= 0
    )
    owner_map = {
        _safe_int(node_id, -1): _safe_int(fragment_id, -1)
        for node_id, fragment_id in dict(node_owner_by_id or {}).items()
        if _safe_int(node_id, -1) >= 0 and _safe_int(fragment_id, -1) >= 0
    }

    def _failure(reason: str, **details: Any) -> FixedRouteResolution:
        return FixedRouteResolution(
            labels=labels,
            node_ids=node_ids,
            source_fragment_ids=tuple(sorted({
                int(owner_map[node_id])
                for node_id in node_ids
                if node_id in owner_map
            })),
            reason=str(reason),
            failure_domain="topology_materialization",
            diagnostics={
                "contract_base_path_labels": list(labels),
                "contract_base_path_node_ids": list(node_ids),
                **details,
            },
        )

    if not node_ids:
        return _failure("fixed_route_empty")
    if len(labels) != len(node_ids):
        return _failure(
            "fixed_route_label_node_count_mismatch",
            label_count=len(labels),
            node_count=len(node_ids),
        )
    if len(set(node_ids)) != len(node_ids):
        return _failure("fixed_route_duplicate_node")

    missing_node_ids = [node_id for node_id in node_ids if node_id not in nodes_by_id]
    if missing_node_ids:
        return _failure("fixed_route_node_missing", missing_node_ids=missing_node_ids)

    for left_id, right_id in zip(node_ids, node_ids[1:]):
        if not _nodes_have_fixed_edge(
            nodes_by_id[int(left_id)],
            nodes_by_id[int(right_id)],
        ):
            return _failure(
                f"fixed_route_edge_missing:{int(left_id)}-{int(right_id)}",
                offending_fixed_edges=[[int(left_id), int(right_id)]],
            )

    terminal_positions = [
        index
        for index, node_id in enumerate(node_ids)
        if _is_terminal(nodes_by_id[int(node_id)])
    ]
    if terminal_positions and terminal_positions != [len(node_ids) - 1]:
        return _failure(
            "branch_terminal_not_at_path_end",
            terminal_node_ids=[int(node_ids[index]) for index in terminal_positions],
        )

    route_set = set(node_ids)
    allowed_parent_ids = {
        _safe_int(value, -1)
        for value in list(mainline_node_ids or [])
        if _safe_int(value, -1) >= 0
    }
    center_id = _safe_int(payload.get("center_node_id"), -1)
    if center_id >= 0:
        allowed_parent_ids.add(int(center_id))

    unexpected_edges = []
    for index, node_id in enumerate(node_ids):
        node = nodes_by_id[int(node_id)]
        expected_ids = set(route_set)
        if index == 0:
            expected_ids.update(allowed_parent_ids)
        for candidate_id, candidate in nodes_by_id.items():
            candidate_id_i = _safe_int(candidate_id, -1)
            if candidate_id_i < 0 or candidate_id_i == int(node_id):
                continue
            if candidate_id_i in expected_ids:
                continue
            if _nodes_have_fixed_edge(node, candidate):
                if (
                    index == len(node_ids) - 1
                    and bool(payload.get("allow_aromatic_endpoint_context", False))
                    and _safe_int(getattr(node, "su_type", -1), -1)
                    in set(AROMATIC_SU_TYPES)
                ):
                    continue
                unexpected_edges.append([int(node_id), int(candidate_id_i)])

    if unexpected_edges:
        offending_parent_ids = {edge[0] for edge in unexpected_edges}
        terminal_ids = {
            int(node_id)
            for node_id in node_ids
            if _is_terminal(nodes_by_id[int(node_id)])
        }
        reason = (
            f"terminal_has_fixed_continuation:{min(offending_parent_ids & terminal_ids)}"
            if offending_parent_ids & terminal_ids
            else "nested_branch_topology_forbidden:nonlinear_fixed_component"
        )
        return _failure(
            reason,
            mainline_node_ids=sorted(allowed_parent_ids),
            unexpected_fixed_edges=sorted(unexpected_edges),
        )

    terminal_node_id = (
        int(node_ids[-1])
        if _is_terminal(nodes_by_id[int(node_ids[-1])])
        else None
    )
    source_fragment_ids = tuple(sorted({
        int(owner_map[node_id])
        for node_id in node_ids
        if node_id in owner_map
    }))
    return FixedRouteResolution(
        labels=labels,
        node_ids=node_ids,
        source_fragment_ids=source_fragment_ids,
        terminal_node_id=terminal_node_id,
        diagnostics={
            "contract_base_path_labels": list(labels),
            "contract_base_path_node_ids": list(node_ids),
        },
    )


def preflight_fixed_contracts(
    nodes: Sequence[_NodeV3],
    contracts: Iterable[Mapping[str, Any]],
    *,
    mainline_node_ids: Sequence[int] = (),
    node_owner_by_id: Optional[Mapping[int, int]] = None,
) -> Tuple[FixedRouteResolution, ...]:
    nodes_by_id = {
        _safe_int(getattr(node, "global_id", -1), -1): node
        for node in list(nodes or [])
        if _safe_int(getattr(node, "global_id", -1), -1) >= 0
    }
    return tuple(
        resolve_fixed_contract_route(
            contract,
            nodes_by_id=nodes_by_id,
            mainline_node_ids=mainline_node_ids,
            node_owner_by_id=node_owner_by_id,
        )
        for contract in list(contracts or [])
    )


def fragment_branch_contracts(fragment: Any) -> Tuple[Dict[str, Any], ...]:
    """Return contracts that will be attached as branches, not mainline paths."""

    evidence = dict(getattr(fragment, "evidence", {}) or {})
    selected_primary_id = _safe_int(
        dict(evidence.get("selected_primary_connector", {}) or {}).get(
            "connector_node_id"
        ),
        -1,
    )
    contracts = []
    for raw_contract in list(evidence.get("fixed_connection_contracts", []) or []):
        contract = dict(raw_contract or {})
        roles = {
            str(value)
            for value in list(contract.get("role_candidates", []) or [])
            if str(value)
        }
        connector_id = _safe_int(contract.get("connector_node_id"), -1)
        uses_mainline = bool(
            "mainline_only" in roles
            or (
                connector_id == selected_primary_id
                and "mainline_or_branch" in roles
            )
        )
        if not uses_mainline:
            contracts.append(contract)
    return tuple(contracts)


def preflight_classification_fixed_routes(
    nodes: Sequence[_NodeV3],
    classification: Any,
) -> Dict[int, Dict[str, Any]]:
    """Build per-fragment structural status used by planning and assembly."""

    nodes_by_id = {
        _safe_int(getattr(node, "global_id", -1), -1): node
        for node in list(nodes or [])
        if _safe_int(getattr(node, "global_id", -1), -1) >= 0
    }
    owner_map = dict(
        getattr(classification, "fixed_node_owner_by_id", {}) or {}
    )
    status_by_fragment_id: Dict[int, Dict[str, Any]] = {}
    for fragment in list(getattr(classification, "fragments", []) or []):
        fragment_id = _safe_int(getattr(fragment, "fragment_id", -1), -1)
        contracts = fragment_branch_contracts(fragment)
        resolutions = tuple(
            resolve_fixed_contract_route(
                contract,
                nodes_by_id=nodes_by_id,
                mainline_node_ids=list(
                    getattr(fragment, "mainline_node_ids", []) or []
                ),
                node_owner_by_id=owner_map,
            )
            for contract in contracts
        )
        failures = [resolution for resolution in resolutions if not resolution.ok]
        status_by_fragment_id[int(fragment_id)] = {
            "ok": not bool(failures),
            "contract_count": len(contracts),
            "failure_domain": (
                "topology_materialization" if failures else ""
            ),
            "reasons": [str(item.reason) for item in failures],
            "failures": [
                {
                    "reason": str(item.reason),
                    "failure_domain": str(item.failure_domain),
                    **dict(item.diagnostics or {}),
                }
                for item in failures
            ],
        }
    return status_by_fragment_id


def stage3_fixed_route_preflight(
    nodes: Sequence[_NodeV3],
    classification: Any,
    flex_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    """Check fixed routes that the current flex plan must materialize."""

    status_by_fragment_id = preflight_classification_fixed_routes(
        nodes,
        classification,
    )
    audit = dict(flex_audit or {})
    required_fragment_ids = {
        _safe_int(value, -1)
        for value in (
            list(audit.get("hard_explicit_flex_fragment_ids", []) or [])
            + list(audit.get("selected_candidate_fragment_ids", []) or [])
            + list(audit.get("invalid_explicit_flex_fragment_ids", []) or [])
        )
        if _safe_int(value, -1) >= 0
    }
    failures = {
        int(fragment_id): dict(
            status_by_fragment_id.get(int(fragment_id), {}) or {}
        )
        for fragment_id in sorted(required_fragment_ids)
        if not bool(
            dict(status_by_fragment_id.get(int(fragment_id), {}) or {}).get(
                "ok", True
            )
        )
    }
    return {
        "ok": not bool(failures),
        "reason": (
            "stage3_fixed_route_materialization_invalid" if failures else "ok"
        ),
        "failure_domain": "topology_materialization" if failures else "",
        "required_fragment_ids": sorted(required_fragment_ids),
        "failures_by_fragment_id": {
            str(key): value for key, value in failures.items()
        },
    }
