"""Projected fragment view and canonical topology rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import AssemblyEndpoint, BranchStub
from .rules import (
    B_LIKE_OPEN_CLASSES,
    D_LIKE_OPEN_CLASSES,
    E_LIKE_OPEN_CLASSES,
    TERMINAL_ENDPOINT_LABELS,
)
from ...classification.classification_tokens import AROMATIC_SU_TYPES
from ...stage3_models import Stage3Serializable, TopologyPathRender


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def is_aromatic_label(label: str) -> bool:
    value = str(label or "")
    return value == "11" or value in {str(su_type) for su_type in AROMATIC_SU_TYPES}


def render_linear_path(
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    *,
    open_left: bool = False,
    open_right: bool = False,
    branch_map: Optional[Dict[int, List[str]]] = None,
    branch_id_map: Optional[Dict[int, List[str]]] = None,
    suppressed_support_node_ids: Optional[Sequence[int]] = None,
) -> TopologyPathRender:
    branches = dict(branch_map or {})
    branch_ids = dict(branch_id_map or {})
    topology_parts: List[str] = []
    node_id_parts: List[str] = []
    branch_topologies: List[str] = []
    branch_node_ids: List[str] = []
    for index, (label, node_id) in enumerate(
        zip(list(labels or []), list(node_ids or []))
    ):
        topology_token = str(label)
        node_id_token = "?" if node_id is None else str(int(node_id))
        for branch_topology, branch_node_id in zip(
            list(branches.get(int(index), []) or []),
            list(branch_ids.get(int(index), []) or []),
        ):
            topology_token += f"({branch_topology})"
            node_id_token += f"({branch_node_id})"
            branch_topologies.append(str(branch_topology))
            branch_node_ids.append(str(branch_node_id))
        topology_parts.append(topology_token)
        node_id_parts.append(node_id_token)
    topology_notation = "-".join(value for value in topology_parts if str(value))
    node_id_notation = "-".join(value for value in node_id_parts if str(value))
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
        branch_node_id_notations=branch_node_ids,
        suppressed_support_node_ids=sorted({
            int(value) for value in list(suppressed_support_node_ids or [])
        }),
    )


def render_token_sequence(
    labels: Sequence[str],
    node_ids: Sequence[Optional[int]],
    *,
    branch_map: Optional[Dict[int, List[str]]] = None,
    branch_id_map: Optional[Dict[int, List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    branches = dict(branch_map or {})
    branch_ids = dict(branch_id_map or {})
    topology_tokens: List[str] = []
    node_id_tokens: List[str] = []
    for index, (label, node_id) in enumerate(
        zip(list(labels or []), list(node_ids or []))
    ):
        topology_token = str(label)
        node_id_token = "?" if node_id is None else str(int(node_id))
        for branch_topology, branch_node_id in zip(
            list(branches.get(int(index), []) or []),
            list(branch_ids.get(int(index), []) or []),
        ):
            topology_token += f"({branch_topology})"
            node_id_token += f"({branch_node_id})"
        topology_tokens.append(topology_token)
        node_id_tokens.append(node_id_token)
    return topology_tokens, node_id_tokens


def strip_open_markers(text: str) -> str:
    result = str(text or "")
    while result.startswith("-"):
        result = result[1:]
    while result.endswith("-"):
        result = result[:-1]
    return result


def clean_ring_segment_text(text: str) -> str:
    result = strip_open_markers(str(text or ""))
    while "()" in result:
        result = result.replace("()", "")
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-")


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

    def _endpoint_role(self, *, is_open: bool, label: str) -> str:
        if is_open:
            roles = {str(value) for value in list(self.flex_reconcile_roles or [])}
            if "b_like_open" in roles or str(self.class_type) in B_LIKE_OPEN_CLASSES:
                return "b_like_open_end"
            if str(self.class_type) in E_LIKE_OPEN_CLASSES:
                return "e_like_open_end"
            if "d_like_open" in roles or str(self.class_type) in D_LIKE_OPEN_CLASSES:
                return "double_open_end"
            if str(self.family) in {"like24", "like24_secondary", "like24_bridge"}:
                if str(self.preferred_product_type) == "side_chain":
                    return "e_like_open_end"
                if self.open_left and self.open_right:
                    return "double_open_end"
                return "b_like_open_end"
            return "closed_internal_end"
        if is_aromatic_label(label):
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
            bridge_label = label
        return AssemblyEndpoint(
            side=side_name,
            node_id=node_id,
            label=label,
            is_open=is_open,
            role=self._endpoint_role(is_open=is_open, label=label),
            bridge_node_id=None if bridge_node_id < 0 else bridge_node_id,
            bridge_label=bridge_label,
        )

    def oriented_endpoint(self, side: str, *, reversed_view: bool) -> AssemblyEndpoint:
        actual_side = str(side)
        if reversed_view:
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

    def oriented_mainline(
        self, *, reversed_view: bool,
    ) -> Tuple[List[str], List[Optional[int]]]:
        if reversed_view:
            return list(reversed(self.main_labels)), list(reversed(self.main_node_ids))
        return list(self.main_labels), list(self.main_node_ids)

    def branch_stubs_map(self) -> Dict[int, List[BranchStub]]:
        if self.branch_stubs_by_anchor:
            return {
                int(anchor): list(values or [])
                for anchor, values in self.branch_stubs_by_anchor.items()
            }
        return {int(self.branch_anchor_index): list(self.actual_branch_stubs)}

    def branch_kinds_by_anchor(self) -> Dict[int, List[str]]:
        if self.unresolved_branch_kinds_by_anchor:
            return {
                int(anchor): [str(value) for value in list(values or [])]
                for anchor, values in self.unresolved_branch_kinds_by_anchor.items()
            }
        return {
            int(self.branch_anchor_index): [
                str(value) for value in self.unresolved_branch_kinds
            ]
        }

    def fixed_branch_contracts_by_anchor(self) -> Dict[int, List[Dict[str, Any]]]:
        return {
            int(anchor): [
                dict(item) for item in list(values or []) if dict(item or {})
            ]
            for anchor, values in self.branch_closure_contracts_by_anchor.items()
        }

    def real_branch_stubs_map(self) -> Dict[int, List[BranchStub]]:
        if self.real_branch_stubs_by_anchor:
            return {
                int(anchor): list(values or [])
                for anchor, values in self.real_branch_stubs_by_anchor.items()
            }
        return self.branch_stubs_map()

    def pending_branch_stubs_map(self) -> Dict[int, List[BranchStub]]:
        result = self.real_branch_stubs_map()
        for anchor, contracts in self.fixed_branch_contracts_by_anchor().items():
            result.setdefault(int(anchor), [])
            rendered_ids = {
                int(node_id)
                for stub in result[int(anchor)]
                for node_id in list(stub.source_node_ids or [])
            }
            for contract in contracts:
                labels = [
                    str(value)
                    for value in list(contract.get("base_path_labels", []) or [])
                    if str(value)
                ]
                node_ids = [
                    int(value)
                    for value in list(contract.get("base_path_node_ids", []) or [])
                    if _safe_int(value, -1) >= 0
                ]
                if not labels or not node_ids or set(node_ids).issubset(rendered_ids):
                    continue
                result[int(anchor)].append(BranchStub(
                    topology_notation="-" + "-".join(labels),
                    node_id_notation="-" + "-".join(str(value) for value in node_ids),
                    source_node_ids=node_ids,
                    branch_kind="actual_fixed_connection_contract",
                    mode="fixed_contract_base",
                ))
                rendered_ids.update(node_ids)
        return result

    def real_branch_kinds_by_anchor(self) -> Dict[int, List[str]]:
        if self.real_unresolved_branch_kinds_by_anchor:
            return {
                int(anchor): [str(value) for value in list(values or [])]
                for anchor, values in self.real_unresolved_branch_kinds_by_anchor.items()
            }
        return self.branch_kinds_by_anchor()

    def partial_render(self) -> TopologyPathRender:
        return self._render_from_maps(self.branch_stubs_map(), self.branch_kinds_by_anchor())

    def real_render(self) -> TopologyPathRender:
        return self._render_from_maps(
            self.real_branch_stubs_map(), self.real_branch_kinds_by_anchor()
        )

    def pending_render(self) -> TopologyPathRender:
        return self._render_from_maps(
            self.pending_branch_stubs_map(), self.real_branch_kinds_by_anchor()
        )

    def _render_from_maps(
        self,
        branch_stubs: Dict[int, List[BranchStub]],
        branch_kinds: Dict[int, List[str]],
    ) -> TopologyPathRender:
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        for anchor in sorted(set(branch_stubs) | set(branch_kinds)):
            entries = [
                (str(stub.topology_notation), str(stub.node_id_notation))
                for stub in list(branch_stubs.get(int(anchor), []) or [])
            ]
            for kind in list(branch_kinds.get(int(anchor), []) or []):
                name = str(kind or "")
                if name in {"H", "H1"}:
                    continue
                entries.append(("-22", "-22") if name == "FIXED22" else (name, name))
            branch_map[int(anchor)] = [topology for topology, _ in entries]
            branch_id_map[int(anchor)] = [node_ids for _, node_ids in entries]
        return render_linear_path(
            self.main_labels,
            self.main_node_ids,
            open_left=self.open_left,
            open_right=self.open_right,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=self.suppressed_support_node_ids,
        )
