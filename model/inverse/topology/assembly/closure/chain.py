"""Flex-chain and side-chain materialization."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...fixed_edges import nodes_have_fixed_edge as _nodes_have_fixed_edge
from ...classification.classification_tokens import AROMATIC_SU_TYPES
from ...fixed_routes import fragment_structure_signature, fragment_structure_snapshot
from ...resources.profiles import (
    alternative_closure_required_counts,
    closure_sequences,
    minimum_closure_required_counts,
)
from ...stage3_models import (
    RingTopologyContract,
    TopologyAssemblyProduct,
    TopologyClosureSlot,
    TopologyPathRender,
)
from . import branch as branch_closure_module
from ..state import ownership as ownership_module
from ..state import pending as pending_module
from ..ring import topology as ring_topology_module
from ..ring import policy as ring_policy_module
from ..core.fragment import (
    AssemblyFragmentView,
    render_linear_path as _render_linear_path,
)
from ..core.models import (
    AssemblyCompositionCandidate,
    BranchClosureResult,
    ClosureOperatorSpec,
)
from ..projection.engine import (
    _collect_source_metadata,
    _dedupe_rendered_route_with_maps,
    _direct_join_bridge_tokens,
    _fixed_source_ids_from_views,
    _node_label,
    _normalize_ring_backbone_labels,
    _normalize_ring_backbone_node_ids,
    _safe_int,
    _slot_required_counts,
    _should_materialize_bridge_literal,
    _source_fragment_classes,
    _strip_node_ids_from_path,
)
from ..core.rules import (
    MAINLINE_CLOSURE_OPERATOR_SPECS,
    RING_PRODUCT_KIND_TO_TYPE,
    TERMINAL_ENDPOINT_LABELS,
    external_phase_for_role as _external_phase_for_role,
)
from ..core.validation import (
    mainline_contract_reason,
    branch_route_shape_reason as _branch_route_shape_reason,
    nested_path_edges as _nested_path_edges,
    validate_branch_capacity as _validate_branch_capacity,
    validate_chain_product_shape as _validate_chain_product_shape,
)

from ..core.rules import (
    CHAIN_MATERIALIZABLE_LIKE24_CLASSES,
    CHAIN_MATERIALIZABLE_OPEN_CLASSES,
    EXPLICIT_FLEX_CLASSES,
    EXPLICIT_SIDE_CLASSES,
)


def is_like24_family(view: Any) -> bool:
    return str(getattr(view, "family", "")) in {
        "like24",
        "like24_secondary",
        "like24_bridge",
    }


def is_chain_materializable_view(view: Any) -> bool:
    if (
        is_like24_family(view)
        and bool(
            dict(getattr(view, "evidence", {}) or {}).get(
                "edge_semantics_finalized", False
            )
        )
        and not bool(
            getattr(view, "open_left", False)
            or getattr(view, "open_right", False)
        )
    ):
        return False
    class_type = str(getattr(view, "class_type", ""))
    if class_type in set(EXPLICIT_FLEX_CLASSES) | set(EXPLICIT_SIDE_CLASSES):
        return True
    if class_type in set(CHAIN_MATERIALIZABLE_OPEN_CLASSES):
        return True
    if is_like24_family(view):
        return class_type in (
            set(CHAIN_MATERIALIZABLE_LIKE24_CLASSES)
            | {
                "24_A", "24_B", "24_C", "24_D", "24_AB_N", "24_CD_M",
                "24_CD_N", "24_AB_N_24_CD", "24_CD_N_24_CD",
            }
        )
    return True


def is_generic_chain_join_view(view: Any) -> bool:
    if not is_chain_materializable_view(view):
        return False
    if is_like24_family(view):
        return str(getattr(view, "class_type", "")) in {"24_AB_N", "24_CD_M"}
    return True


def remove_materialized_bridge_from_payload(
    payload: Tuple[List[str], List[int]],
    *,
    bridge: Optional[Tuple[str, int]],
    materialized_node_ids: Iterable[int],
) -> Tuple[List[str], List[int]]:
    """Remove one already-rendered endpoint bridge from a closure payload."""

    labels, node_ids = payload
    if bridge is None:
        return list(labels), list(node_ids)
    bridge_id = int(bridge[1])
    owned = {int(value) for value in list(materialized_node_ids or [])}
    if bridge_id not in owned:
        return list(labels), list(node_ids)
    kept_labels: List[str] = []
    kept_node_ids: List[int] = []
    removed = False
    for label, node_id in zip(list(labels or []), list(node_ids or [])):
        if not removed and int(node_id) == bridge_id:
            removed = True
            continue
        kept_labels.append(str(label))
        kept_node_ids.append(int(node_id))
    return kept_labels, kept_node_ids


__all__ = [
    "is_chain_materializable_view",
    "is_generic_chain_join_view",
    "is_like24_family",
    "remove_materialized_bridge_from_payload",
]


def resolve_mainline_closure(
    assembler: Any,
    closure_mode: str,
) -> Optional[BranchClosureResult]:
    operator = MAINLINE_CLOSURE_OPERATOR_SPECS.get(str(closure_mode), ClosureOperatorSpec(name=str(closure_mode), scope="mainline"))
    if operator.name in {"F1", "F2"}:
        for index, sequence in enumerate(closure_sequences(operator.name)):
            result = branch_closure_module.claim_pool_sequence(
                assembler,
                closure_kind=str(operator.name),
                sequence=sequence,
                branch=False,
                mode_suffix="short" if index == 0 else "long",
            )
            if result is not None:
                return result
    return None


def render_product(
    assembler,
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


def closure_sequence(
    assembler,
    closure: BranchClosureResult,
    orientation: str,
) -> Tuple[List[str], List[int]]:
    labels = list(closure.main_labels or str(closure.branch_topology).split("-"))
    node_ids = list(closure.main_node_ids or [int(x) for x in str(closure.branch_node_ids).split("-") if str(x)])
    if str(orientation) == "left":
        return list(reversed(labels)), [int(x) for x in reversed(node_ids)]
    return labels, [int(x) for x in node_ids]


def main_closure_canonical_labels(
    closure_mode: str,
    side: str,
) -> List[str]:
    sequences = closure_sequences(str(closure_mode))
    if not sequences:
        return []
    labels = [str(value) for value in list(sequences[0] or [])]
    return list(reversed(labels)) if str(side) == "left" else labels


def invalid_product_reason(
    assembler,
    product_kind: str,
    labels: Sequence[str],
    source_views: Sequence[AssemblyFragmentView],
    node_ids: Optional[Sequence[Optional[int]]] = None,
) -> Optional[str]:
    normalized_labels, normalized_node_ids = _dedupe_rendered_route_with_maps(
        labels,
        node_ids or [],
        {},
        {},
    )[:2]
    return _validate_chain_product_shape(
        str(product_kind),
        normalized_labels,
        source_views,
        node_ids=normalized_node_ids,
    )


def physical_product_reason(
    assembler,
    *,
    product_kind: str,
    source_views: Sequence[AssemblyFragmentView],
    node_ids: Sequence[Optional[int]],
    branch_id_map: Dict[int, List[str]],
    suppressed_support_node_ids: Optional[Sequence[int]] = None,
    ring_contract: Optional[RingTopologyContract] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if str(product_kind).startswith("fatty_ring_"):
        if ring_contract is None:
            return "ring_contract_missing"
        return ring_topology_module.audit_ring_topology_contract(
            ring_contract,
            nodes=assembler.nodes,
        )
    # A terminal is an endpoint unit, never a middle node.  Apply this to
    # the concrete IDs rather than numeric labels because global IDs and
    # SU labels are intentionally independent.
    mainline_ids = [
        int(value)
        for value in list(node_ids or [])
        if value is not None and int(value) >= 0
    ]
    for index, node_id in enumerate(mainline_ids):
        node = assembler.nodes_by_id.get(int(node_id))
        if node is None or index in {0, len(mainline_ids) - 1}:
            continue
        if _node_label(node) in set(TERMINAL_ENDPOINT_LABELS):
            return f"terminal_in_mainline_middle:{int(node_id)}"
    edges: Set[Tuple[int, int]] = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(mainline_ids, mainline_ids[1:])
        if int(left) != int(right)
    }
    for raw_anchor, branch_paths in dict(branch_id_map or {}).items():
        anchor_index = _safe_int(raw_anchor, -1)
        if anchor_index < 0 or anchor_index >= len(mainline_ids):
            return f"branch_anchor_out_of_range:{anchor_index}/{len(mainline_ids)}"
        anchor_id = int(mainline_ids[anchor_index])
        for raw_path in list(branch_paths or []):
            raw_path_text = str(raw_path or "")
            path_shape_reason = _branch_route_shape_reason(
                "",
                raw_path_text,
                assembler.nodes_by_id,
            )
            if path_shape_reason is not None:
                return str(path_shape_reason)
            path_edges, path_ids = _nested_path_edges(raw_path_text)
            first_match = re.search(r"\d+", raw_path_text)
            first_id = int(first_match.group(0)) if first_match else -1
            if first_id >= 0 and int(anchor_id) not in path_ids:
                if int(anchor_id) == int(first_id):
                    return f"branch_self_edge:{first_id}"
                path_edges.add(tuple(sorted((int(anchor_id), int(first_id)))))
            if int(anchor_id) in path_ids:
                path_edges = {
                    tuple(sorted((int(left), int(right))))
                    for left, right in path_edges
                    if int(left) != int(right)
                }
            edges.update(path_edges)

    # ``source_views`` can contain a donor's complete classification view,
    # while only a route subset of that view is rendered in this product.
    # Audit fixed edges only for centres that are actually rendered.  When
    # both ends of a fixed hop are rendered, retain that physical edge even
    # if the branch renderer did not emit a separate branch token for it.
    # This keeps the physical audit aligned with the canonical route
    # identity and avoids reporting an unrendered donor context as a
    # missing edge.
    rendered_node_ids = set(mainline_ids)
    for branch_paths in dict(branch_id_map or {}).values():
        for raw_path in list(branch_paths or []):
            rendered_node_ids.update(
                int(value)
                for value in re.findall(r"\d+", str(raw_path or ""))
            )
    for view in list(source_views or []):
        if str(view.family) not in {"like24", "like24_secondary", "like24_bridge", "su25"}:
            continue
        evidence = dict(view.evidence or {})
        center_ids = [
            int(value)
            for value in list(evidence.get("center_node_ids", []) or [])
        ] or [int(view.fragment_id)]
        view_source_ids = {
            int(value) for value in list(view.source_node_ids or [])
        }
        for center_id in center_ids:
            if int(center_id) not in rendered_node_ids:
                continue
            center = assembler.nodes_by_id.get(int(center_id))
            if center is None:
                continue
            for raw_neighbor_id in list(getattr(center, "hop1_ids", []) or []):
                neighbor_id = _safe_int(raw_neighbor_id, -1)
                if neighbor_id < 0 or neighbor_id not in rendered_node_ids:
                    continue
                if neighbor_id not in view_source_ids:
                    continue
                neighbor = assembler.nodes_by_id.get(int(neighbor_id))
                if neighbor is None or not _nodes_have_fixed_edge(center, neighbor):
                    continue
                edges.add(tuple(sorted((int(center_id), int(neighbor_id)))))

    degree = Counter()
    for left, right in edges:
        degree[int(left)] += 1
        degree[int(right)] += 1

    suppressed_ids = {
        int(value)
        for value in list(suppressed_support_node_ids or [])
    }
    center_degree_audit: List[Dict[str, Any]] = []
    audited_center_ids: Set[int] = set()
    for view in list(source_views or []):
        if str(view.family) not in {"like24", "like24_secondary", "like24_bridge", "su25"}:
            continue
        evidence = dict(view.evidence or {})
        center_ids = [
            int(value)
            for value in list(evidence.get("center_node_ids", []) or [])
        ] or [int(view.fragment_id)]
        view_source_ids = {
            int(value) for value in list(view.source_node_ids or [])
        }
        for center_id in center_ids:
            # A donor view may carry additional classification centres
            # that are not part of the branch route selected for this
            # product.  Those centres remain owned by their fragment but
            # are not part of this product's physical audit.
            if int(center_id) not in rendered_node_ids:
                continue
            center = assembler.nodes_by_id.get(int(center_id))
            if center is None:
                return f"physical_center_missing:{center_id}"
            center_hop1_ids = {
                _safe_int(value, -1)
                for value in list(getattr(center, "hop1_ids", []) or [])
                if _safe_int(value, -1) >= 0
            }
            suppressed_neighbor_ids = sorted(
                int(neighbor_id)
                for neighbor_id in center_hop1_ids & suppressed_ids
                if tuple(sorted((int(center_id), int(neighbor_id)))) not in edges
            )
            target_degree = _safe_int(
                getattr(center, "target_hop1_degree", 0),
                0,
            )
            rendered_degree = int(degree.get(int(center_id), 0))
            degree_deficit = max(0, int(target_degree) - int(rendered_degree))
            credited_suppressed_neighbor_ids = suppressed_neighbor_ids[:degree_deficit]
            effective_degree = int(
                rendered_degree + len(credited_suppressed_neighbor_ids)
            )
            if int(center_id) not in audited_center_ids:
                center_degree_audit.append({
                    "center_id": int(center_id),
                    "hop1_ids": sorted(center_hop1_ids),
                    "rendered_degree": int(rendered_degree),
                    "suppressed_neighbor_ids": list(suppressed_neighbor_ids),
                    "credited_suppressed_neighbor_ids": list(
                        credited_suppressed_neighbor_ids
                    ),
                    "effective_degree": int(effective_degree),
                    "target_degree": int(target_degree),
                })
                audited_center_ids.add(int(center_id))
            if target_degree > 0 and int(effective_degree) != int(target_degree):
                if diagnostics is not None:
                    diagnostics["attempted_edges"] = [list(edge) for edge in sorted(edges)]
                    diagnostics["suppressed_support_node_ids"] = sorted(suppressed_ids)
                    diagnostics["center_degree_audit"] = list(center_degree_audit)
                return (
                    f"center_degree_mismatch:{center_id}:"
                    f"rendered={int(rendered_degree)}:target={int(target_degree)}"
                )
            for neighbor_id_i in sorted(center_hop1_ids):
                if neighbor_id_i not in view_source_ids:
                    continue
                edge = tuple(sorted((int(center_id), int(neighbor_id_i))))
                if neighbor_id_i in suppressed_ids and edge not in edges:
                    continue
                if edge not in edges:
                    return f"real_center_edge_missing:{center_id}-{neighbor_id_i}"
    if diagnostics is not None:
        diagnostics["attempted_edges"] = [list(edge) for edge in sorted(edges)]
        diagnostics["suppressed_support_node_ids"] = sorted(suppressed_ids)
        diagnostics["center_degree_audit"] = list(center_degree_audit)
    return None


def merge_pool_ids(
    chunks: Iterable[Dict[str, List[int]]],
) -> Dict[str, List[int]]:
    merged: Dict[str, List[int]] = {}
    for chunk in list(chunks or []):
        for key, values in dict(chunk or {}).items():
            merged.setdefault(str(key), [])
            merged[str(key)].extend(int(x) for x in list(values or []))
    for key in list(merged.keys()):
        merged[key] = sorted(list(merged[key]))
    return merged


def required_counts_from_pool_ids(
    assembler,
    pool_node_ids_by_su: Optional[Dict[str, List[int]]],
) -> Dict[str, int]:
    return {
        str(key): len(list(values or []))
        for key, values in dict(pool_node_ids_by_su or {}).items()
        if list(values or [])
    }


def make_product(
    assembler,
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
    ring_contract: Optional[RingTopologyContract] = None,
    branch_results: Optional[Sequence[BranchClosureResult]] = None,
) -> Optional[TopologyAssemblyProduct]:
    is_ring = str(product_kind).startswith("fatty_ring_")
    if not is_ring:
        labels, node_ids, branch_map, branch_id_map = _dedupe_rendered_route_with_maps(
            labels,
            node_ids,
            branch_map,
            branch_id_map,
        )
    if is_ring:
        if ring_contract is None:
            validation_reason = "ring_contract_missing"
            record_invalid_topology_attempt(assembler, 
                product_kind=str(product_kind),
                actual_topology_su=str(dict(diagnostics or {}).get("actual_topology_su", "")),
                actual_topology_gid=str(dict(diagnostics or {}).get("actual_topology_gid", "")),
                topology_notation=str(dict(diagnostics or {}).get("topology_notation", "")),
                validation_reason=validation_reason,
                diagnostics=dict(diagnostics or {}),
            )
            return None
        validation_reason = ring_topology_module.audit_ring_topology_contract(
            ring_contract,
            nodes=assembler.nodes,
        )
        if validation_reason is not None:
            ring_contract.complete = False
            ring_contract.audit_reason = str(validation_reason)
            record_invalid_topology_attempt(assembler, 
                product_kind=str(product_kind),
                actual_topology_su=str(ring_contract.topology_notation),
                actual_topology_gid=str(ring_contract.node_id_notation),
                topology_notation=str(ring_contract.topology_notation),
                validation_reason=str(validation_reason),
                diagnostics={
                    "ring_template_kind": str(ring_contract.ring_kind),
                    "ring_like24_count": len(list(ring_contract.like24_node_ids or [])),
                    **dict(diagnostics or {}),
                },
            )
            return None
        ring_contract.complete = True
        ring_contract.audit_reason = ""
    physical_diagnostics: Dict[str, Any] = {}
    physical_validation_reason = (
        (None if is_ring else _validate_branch_capacity(labels, branch_map))
        or physical_product_reason(assembler, 
            product_kind=str(product_kind),
            source_views=source_views,
            node_ids=node_ids,
            branch_id_map=branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
            ring_contract=ring_contract,
            diagnostics=physical_diagnostics,
        )
    )
    if physical_validation_reason is not None:
        attempted_render = render_product(assembler, 
            labels,
            node_ids,
            branch_map,
            branch_id_map,
            suppressed_support_node_ids=suppressed_support_node_ids,
        )
        record_invalid_topology_attempt(assembler, 
            product_kind=str(product_kind),
            actual_topology_su=str(attempted_render.topology_notation),
            actual_topology_gid=str(attempted_render.node_id_notation),
            topology_notation=str(attempted_render.topology_notation),
            validation_reason=str(physical_validation_reason),
            diagnostics={**dict(diagnostics or {}), **physical_diagnostics},
        )
        return None
    render = render_product(assembler, 
        labels,
        node_ids,
        branch_map,
        branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
    )
    source_fragment_ids, source_fragment_classes, _ = _collect_source_metadata(source_views)
    rendered_node_ids = {
        int(value)
        for value in list(node_ids or [])
        if value is not None and int(value) >= 0
    }
    for values in dict(branch_id_map or {}).values():
        for token in list(values or []):
            for raw_value in str(token).replace("(", "-").replace(")", "-").split("-"):
                if str(raw_value).isdigit():
                    rendered_node_ids.add(int(raw_value))
    fixed_source_node_ids = sorted(
        int(value)
        for value in _fixed_source_ids_from_views(
            source_views,
            assembler=assembler,
        )
        if int(value) in rendered_node_ids
    )
    # Branches can render fixed nodes that are intentionally absent from
    # the projected view's source_node_ids.  Their canonical identity is
    # authoritative once branch resolution has succeeded.
    for branch in list(branch_results or []):
        fixed_source_node_ids = sorted(set(fixed_source_node_ids) | {
            int(value)
            for value in list(getattr(branch, "fixed_source_node_ids", []) or [])
        })
        source_fragment_ids = sorted(set(source_fragment_ids) | {
            int(value)
            for value in list(getattr(branch, "source_fragment_ids", []) or [])
        })
    identity_rendered_node_ids = (
        set(ring_contract.declared_node_ids())
        if ring_contract is not None
        else rendered_node_ids
    )
    identity_source_fragment_ids = (
        list(ring_contract.source_fragment_ids or [])
        if ring_contract is not None
        else source_fragment_ids
    )
    identity_conflict = ownership_module.candidate_identity_conflict(assembler, 
        rendered_node_ids=identity_rendered_node_ids,
        source_fragment_ids=identity_source_fragment_ids,
    )
    if identity_conflict is not None:
        record_invalid_topology_attempt(assembler, 
            product_kind=str(product_kind),
            actual_topology_su=str(render.topology_notation),
            actual_topology_gid=str(render.node_id_notation),
            topology_notation=str(render.topology_notation),
            validation_reason=str(identity_conflict.get("reason", "candidate_identity_invalid")),
            diagnostics={"candidate_identity": dict(identity_conflict)},
        )
        return None
    diagnostics_payload = dict(diagnostics or {})
    product_type = str(
        RING_PRODUCT_KIND_TO_TYPE.get(
            str(product_kind),
            str(product_kind),
        )
    )
    if is_ring and ring_contract is not None:
        source_fragment_ids = [int(value) for value in list(ring_contract.source_fragment_ids or [])]
        fixed_source_node_ids = sorted({
            int(value) for value in list(ring_contract.fixed_source_node_ids or [])
        })
        rendered_node_ids = set(ring_contract.declared_node_ids())
    actual_topology_su = str(
        ring_contract.topology_notation
        if ring_contract is not None
        else diagnostics_payload.get("actual_topology_su") or render.topology_notation
    )
    actual_topology_gid = str(
        ring_contract.node_id_notation
        if ring_contract is not None
        else diagnostics_payload.get("actual_topology_gid") or render.node_id_notation
    )
    product = TopologyAssemblyProduct(
        product_kind=str(product_kind),
        product_type=product_type,
        actual_topology_su=actual_topology_su,
        actual_topology_gid=actual_topology_gid,
        source_fragment_ids=source_fragment_ids,
        source_fragment_classes=source_fragment_classes,
        fixed_source_node_ids=fixed_source_node_ids,
        mainline_node_ids=(
            sorted(rendered_node_ids)
            if ring_contract is not None
            else [
                int(value)
                for value in list(node_ids or [])
                if value is not None and int(value) >= 0
            ]
        ),
        branch_paths=(
            []
            if ring_contract is not None
            else [
                {
                    "anchor_node_id": int(node_ids[int(anchor_index)]),
                    "node_ids": [
                        int(value)
                        for value in re.findall(r"\d+", str(token or ""))
                    ],
                }
                for anchor_index, values in dict(branch_id_map or {}).items()
                if 0 <= int(anchor_index) < len(list(node_ids or []))
                and list(node_ids or [])[int(anchor_index)] is not None
                for token in list(values or [])
                if re.findall(r"\d+", str(token or ""))
            ]
        ),
        ring_contract=ring_contract,
        closed=bool(ring_contract.complete) if ring_contract is not None else True,
        reason=str(reason),
    )
    if ring_contract is not None:
        product.claimed_resource_ids_by_su = {
            str(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(ring_contract.claimed_resource_ids_by_su or {}).items()
            if list(values or [])
        }
    expected_ids = set(product.rendered_node_ids()) | set(product.owned_node_ids())
    nodes_by_id = dict(getattr(assembler, "nodes_by_id", {}) or {})
    product.expected_su_type_by_node_id = {
        int(node_id): int(
            dict(ring_contract.node_su_by_id or {}).get(
                int(node_id),
                getattr(nodes_by_id.get(int(node_id)), "su_type", -1),
            )
            if ring_contract is not None
            else getattr(nodes_by_id.get(int(node_id)), "su_type", -1)
        )
        for node_id in sorted(expected_ids)
        if nodes_by_id.get(int(node_id)) is not None
    }
    return product


def record_invalid_topology_attempt(
    assembler,
    *,
    product_kind: str,
    actual_topology_su: str,
    actual_topology_gid: str,
    topology_notation: str,
    validation_reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    if str(product_kind).startswith("fatty_ring_"):
        assembler.last_ring_materialization_reason = str(validation_reason)
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
    for key in (
        "attempted_edges",
        "suppressed_support_node_ids",
        "center_degree_audit",
    ):
        if key in diag:
            entry[key] = deepcopy(diag[key])
    assembler.invalid_topology_count += 1
    assembler.last_invalid_topology = entry


def collect_suppressed_support_node_ids(
    assembler,
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


def claim_endcap(
    assembler: Any,
    su_type: int,
    label: str,
    mode: str,
) -> Optional[BranchClosureResult]:
    claimed = assembler.allocator.claim(int(su_type), 1)
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


def open_bridge_token(
    assembler: Any,
    view: AssemblyFragmentView,
    side: str,
) -> Optional[Tuple[str, int]]:
    endpoint = view.endpoint(str(side))
    if not endpoint.is_open or endpoint.bridge_node_id is None:
        return None
    return str(endpoint.bridge_label or endpoint.label), int(endpoint.bridge_node_id)


def _d2_endpoint_requires_full_f1(
    assembler: Any,
    view: AssemblyFragmentView,
    *,
    side: str,
    closure_mode: str,
) -> bool:
    """Whether a d2 fixed-path endpoint still exposes an SU23 boundary.

    A d2 bridge that is already terminal only needs the SU11/SU22 endcap
    shortcut.  A d2 node at the end of an open mainline route instead needs
    the complete ``[23, 11]`` sequence recorded by classification.
    """
    if str(closure_mode) != "F1":
        return False
    bridge = open_bridge_token(assembler, view, str(side))
    # A fixed d2 tail can be projected with only the opposite endpoint marked
    # open because its boundary is carried by an explicit classification
    # contract.  Resolve the concrete endpoint so that contract remains
    # addressable after a fresh Layer1/2 rebuild.
    if bridge is None:
        endpoint = view.endpoint(str(side))
        endpoint_label = str(endpoint.label or "")
        if (
            endpoint.node_id is not None
            and endpoint_label in {
                "single19d2",
                "single20d2",
                "double19d2",
                "double20d2",
                "21d2",
            }
        ):
            bridge = (endpoint_label, int(endpoint.node_id))
    if bridge is None or str(bridge[0]) not in {
        "single19d2",
        "single20d2",
        "double19d2",
        "double20d2",
        "21d2",
    }:
        return False
    bridge_id = int(bridge[1])
    for raw_profile in list(
        dict(getattr(view, "evidence", {}) or {}).get(
            "basic_open_endpoints", []
        )
        or []
    ):
        profile = dict(raw_profile or {})
        if (
            _safe_int(profile.get("node_id"), -1) == bridge_id
            and bool(profile.get("requires_full_f1", False))
        ):
            return True
    for raw_contract in list(getattr(view, "fixed_connection_contracts", []) or []):
        contract = dict(raw_contract or {})
        base_ids = [
            _safe_int(value, -1)
            for value in list(contract.get("base_path_node_ids", []) or [])
            if _safe_int(value, -1) >= 0
        ]
        roles = {str(value) for value in list(contract.get("role_candidates", []) or [])}
        fixed_role = str(contract.get("fixed_connection_role", ""))
        boundary_sus = {
            _safe_int(value, -1)
            for value in list(contract.get("tail_open_boundary_su_types", []) or [])
        }
        if (
            base_ids
            and int(base_ids[-1]) == bridge_id
            and ("mainline_only" in roles or "mainline_or_branch" in roles or fixed_role == "mainline")
            and (
                23 in boundary_sus
                or bool(
                    list(
                        contract.get("tail_open_boundary_node_ids", [])
                        or []
                    )
                )
                or str(contract.get("tail_open_reason", "") or "")
                == "fixed_branch_d2_requires_H1"
            )
        ):
            return True
    return False


def _side_closure_modes_for_view(
    assembler: Any,
    view: AssemblyFragmentView,
) -> Tuple[str, str]:
    """Return side-chain modes for a dual-open view in its physical orientation."""
    if _d2_endpoint_requires_full_f1(
        assembler,
        view,
        side="right",
        closure_mode="F1",
    ):
        # F2 renders [22, 23] on the left; the open d2 tail needs F1 [23, 11]
        # on the right, yielding an aromatic-to-terminal side chain.
        return "F2", "F1"
    if _d2_endpoint_requires_full_f1(
        assembler,
        view,
        side="left",
        closure_mode="F1",
    ):
        return "F1", "F2"
    return "F1", "F2"


def build_main_closure(
    assembler,
    view: AssemblyFragmentView,
    *,
    side: str,
    closure_mode: str,
) -> Tuple[Optional[Tuple[List[str], List[int]]], Dict[str, int]]:
    bridge = open_bridge_token(assembler, view, side)
    if bridge is not None:
        bridge_label, bridge_id = bridge
        if _d2_endpoint_requires_full_f1(
            assembler,
            view,
            side=str(side),
            closure_mode=str(closure_mode),
        ):
            closure = resolve_mainline_closure(assembler, str(closure_mode))
            if closure is None:
                return None, {
                    str(key): int(value)
                    for key, value in minimum_closure_required_counts(str(closure_mode)).items()
                }
            labels, node_ids = closure_sequence(assembler, closure, side)
            return (labels, node_ids), required_counts_from_pool_ids(
                assembler,
                closure.pool_node_ids_by_su,
            )
        if str(bridge_label) in {
            "23",
            "single19d2",
            "single20d2",
            "double19d2",
            "double20d2",
            "21d2",
        }:
            endcap = (
                claim_endcap(assembler, 11, "11", "pool:11")
                if str(closure_mode) == "F1"
                else claim_endcap(assembler, 22, "22", "pool:22")
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
            return (labels, node_ids), required_counts_from_pool_ids(assembler, 
                endcap.pool_node_ids_by_su
            )
        closure = resolve_mainline_closure(assembler, str(closure_mode))
        if closure is None:
            return None, {
                str(key): int(value)
                for key, value in minimum_closure_required_counts(str(closure_mode)).items()
            }
        labels, node_ids = closure_sequence(assembler, closure, side)
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
        return (labels, node_ids), required_counts_from_pool_ids(assembler, 
            closure.pool_node_ids_by_su
        )
    closure = resolve_mainline_closure(assembler, str(closure_mode))
    if closure is None:
        return None, {
            str(key): int(value)
            for key, value in minimum_closure_required_counts(str(closure_mode)).items()
        }
    labels, node_ids = closure_sequence(assembler, closure, side)
    return (labels, node_ids), required_counts_from_pool_ids(assembler, 
        closure.pool_node_ids_by_su
    )


def attempt_mainline_slots(
    assembler,
    view: AssemblyFragmentView,
    attempts: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for index, (side, closure_mode) in enumerate(list(attempts or [])):
        endpoint = view.endpoint(str(side))
        anchor_node_id = (
            int(endpoint.node_id)
            if endpoint.node_id is not None
            else int(view.fragment_id)
        )
        sequences = closure_sequences(str(closure_mode))
        required_sequence = [
            int(value) for value in list(sequences[0] if sequences else [])
        ]
        bridge = open_bridge_token(assembler, view, str(side))
        if bridge is not None and str(bridge[0]) in {
            "23",
            "single19d2",
            "single20d2",
            "double19d2",
            "double20d2",
            "21d2",
        }:
            if _d2_endpoint_requires_full_f1(
                assembler,
                view,
                side=str(side),
                closure_mode=str(closure_mode),
            ):
                required_sequence = [23, 11]
            elif str(closure_mode) == "F1":
                required_sequence = [11]
            elif str(closure_mode) == "F2":
                required_sequence = [22]
        slots.append(TopologyClosureSlot(
            slot_id=f"fragment:{int(view.fragment_id)}:attempt:{int(index)}:{str(side)}:{str(closure_mode)}",
            kind=str(closure_mode),
            anchor_node_id=int(anchor_node_id),
            anchor_role=f"{str(side)}_mainline",
            required_su_sequence=required_sequence,
        ).to_dict())
    return slots


def build_branch_maps(
    assembler,
    view: AssemblyFragmentView,
    branch_results: Optional[Sequence[BranchClosureResult]],
    *,
    reverse: bool = False,
    anchor_offset: int = 0,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    branch_map: Dict[int, List[str]] = {}
    branch_id_map: Dict[int, List[str]] = {}
    label_count = len(list(view.main_labels or []))
    mainline_node_ids = {
        int(value)
        for value in list(view.main_node_ids or [])
        if value is not None
    }
    for item in list(branch_results or []):
        anchor_index = int(getattr(item, "anchor_index", int(view.branch_anchor_index)))
        if reverse:
            anchor_index = label_count - 1 - int(anchor_index)
        anchor_index += int(anchor_offset)
        # Branch donor routes can include the anchor and a fixed mainline
        # tail in their serialized node notation.  Those nodes are
        # already rendered by this product's mainline and must not be
        # emitted a second time as branch nodes.
        branch_node_ids = _strip_node_ids_from_path(
            str(item.branch_node_ids),
            mainline_node_ids,
        )
        if not re.search(r"\d+", branch_node_ids):
            continue
        # Keep the already validated topology payload.  Rebuilding it from
        # numeric IDs flattened parentheses and produced a different tree
        # from ``branch_node_ids`` (for example ``single19d3-28`` versus
        # ``378(528)``), which made terminal nodes appear connected at both
        # ends and hid branch-role errors.
        branch_topology = str(item.branch_topology)
        branch_map.setdefault(int(anchor_index), [])
        branch_map[int(anchor_index)].append(str(branch_topology))
        branch_id_map.setdefault(int(anchor_index), [])
        branch_id_map[int(anchor_index)].append(str(branch_node_ids))
    return branch_map, branch_id_map


def collect_source_views_from_branches(
    assembler,
    base_views: Sequence[AssemblyFragmentView],
    branch_results: Optional[Sequence[BranchClosureResult]],
) -> List[AssemblyFragmentView]:
    # A base view may be a transaction-local copy with rebound context
    # resources. Preserve it when a branch result refers to the same
    # fragment instead of restoring the stale classification view.
    collected_by_id = {
        int(view.fragment_id): view for view in list(base_views or [])
    }
    for item in list(branch_results or []):
        for fragment_id in list(item.source_fragment_ids or []):
            fragment_id_i = int(fragment_id)
            if fragment_id_i in collected_by_id:
                continue
            source_view = assembler.view_by_id.get(fragment_id_i)
            if source_view is not None:
                collected_by_id[fragment_id_i] = source_view
    return list(collected_by_id.values())


def collect_claimed_fragment_ids(
    assembler,
    base_views: Sequence[AssemblyFragmentView],
    branch_results: Optional[Sequence[BranchClosureResult]],
) -> List[int]:
    return sorted({
        int(view.fragment_id) for view in list(base_views or [])
    } | {
        int(x)
        for item in list(branch_results or [])
        for x in list(item.source_fragment_ids or [])
    })


def record_invalid_materialization_failure(
    assembler,
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
    rolled_back_resource_ids = assembler.allocator.claimed_ids_since(snapshot)
    attempted_render = render_product(assembler, 
        labels,
        node_ids,
        branch_map,
        branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
    )
    assembler.allocator.rollback(snapshot)
    branch_donors[:] = list(donor_snapshot or [])
    failure = {
        "attempted_product_kind": str(product_kind),
        "source_fragment_ids": sorted({
            int(view.fragment_id) for view in list(source_views or [])
        }),
        "left_endpoint": {"node_id": node_ids[0] if node_ids else None, "label": labels[0] if labels else ""},
        "right_endpoint": {"node_id": node_ids[-1] if node_ids else None, "label": labels[-1] if labels else ""},
        "failure_domain": "topology_materialization",
        "failure_scope": "candidate",
        "validation_reason": str(validation_reason),
        "attempted_topology_su": str(attempted_render.topology_notation),
        "attempted_topology_gid": str(attempted_render.node_id_notation),
        "branch_id_map": {
            str(anchor): [str(path) for path in list(paths or [])]
            for anchor, paths in dict(branch_id_map or {}).items()
        },
        "suppressed_support_node_ids": sorted({
            int(value)
            for value in list(suppressed_support_node_ids or [])
        }),
        "attempted_resource_counts": {
            str(key): int(len(list(values or [])))
            for key, values in dict(rolled_back_resource_ids or {}).items()
        },
        "rolled_back_resource_ids": {
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(rolled_back_resource_ids or {}).items()
        },
    }
    physical_diagnostics: Dict[str, Any] = {}
    physical_product_reason(assembler, 
        product_kind=str(product_kind),
        source_views=source_views,
        node_ids=node_ids,
        branch_id_map=branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
        diagnostics=physical_diagnostics,
    )
    failure.update(physical_diagnostics)
    ownership_module.record_composition_failure_entry(assembler, failure)
    return False


def record_failed_product_result(
    assembler,
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
) -> bool:
    validation_reason = (
        str(dict(assembler.last_invalid_topology or {}).get("reason", ""))
        if assembler.last_invalid_topology
        else "product_materialization_failed"
    )
    return record_invalid_materialization_failure(assembler, 
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


def validate_external_branch_closure_state(
    assembler,
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


def branch_pending_reason(
    assembler,
    diagnostics: Optional[Dict[str, Any]],
    *,
    default: str,
) -> str:
    return str(validate_external_branch_closure_state(assembler, diagnostics) or str(default))


def has_pending_contracts_for_subphase(
    assembler,
    subphase: str,
) -> bool:
    subphase_name = str(subphase or "")
    return any(
        str(dict(item or {}).get("subphase", "") or "") == str(subphase_name)
        for item in list(assembler.pending_contracts or [])
    )


def external_subphase_status_snapshot(
    assembler: Any,
) -> Dict[str, Dict[str, Any]]:
    return {
        "explicit_branch_closure": {
            "ok": not has_pending_contracts_for_subphase(assembler, "explicit_branch_closure"),
        },
        "real_like24_ring_assembly": {
            "ok": not has_pending_contracts_for_subphase(assembler, "real_like24_ring_assembly"),
        },
        "remaining_external_halfclosed_closure": {
            "ok": not has_pending_contracts_for_subphase(assembler, "remaining_external_halfclosed_closure"),
        },
    }


def contracts_by_subphase(
    assembler: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in list(assembler.pending_contracts or []):
        contract = dict(item or {})
        grouped[
            str(contract.get("subphase", "") or _external_phase_for_role(contract.get("target_role", "")))
        ].append(contract)
    return {str(key): list(value or []) for key, value in grouped.items()}


def view_flex_reconcile_roles(
    assembler,
    view: Optional[AssemblyFragmentView],
) -> Set[str]:
    if view is None:
        return set()
    return {
        str(role)
        for role in list(getattr(view, "flex_reconcile_roles", []) or [])
        if str(role)
    }


def view_has_flex_reconcile_role(
    assembler,
    view: Optional[AssemblyFragmentView],
    *roles: str,
) -> bool:
    available = view_flex_reconcile_roles(assembler, view)
    return any(str(role) in available for role in list(roles or []))


def view_flex_reconcile_priority(
    assembler,
    view: Optional[AssemblyFragmentView],
) -> int:
    if view is None:
        return 99
    return _safe_int(getattr(view, "flex_reconcile_priority", 99), 99)


def pending_kind_for_view(
    assembler,
    view: Optional[AssemblyFragmentView],
) -> str:
    if view is None:
        return "other"
    profile = pending_module.view_external_pending_profile(view)
    pending_kind = str(profile.get("external_pending_kind", "") or "")
    if pending_kind:
        return str(pending_kind)
    if view_has_flex_reconcile_role(assembler, view, "d_like_open"):
        return "d_like_open"
    if view_has_flex_reconcile_role(assembler, view, "b_like_open"):
        return "b_like_open"
    if is_side_pending_view(assembler, view):
        return "explicit_side"
    if bool(getattr(view, "can_flex", False)):
        return "explicit_flex"
    if bool(getattr(view, "can_ring", False)):
        return "ring_only_like24"
    return "other"


def is_side_pending_view(
    assembler: Any,
    view: Optional[AssemblyFragmentView],
) -> bool:
    del assembler
    return bool(view is not None and getattr(view, "can_side", False))


def available_remaining_side_views(
    assembler,
    *,
    require_generic_join: bool,
) -> List[AssemblyFragmentView]:
    return ownership_module.available_views(assembler, 
        lambda item: (
            not ring_policy_module.ring_candidate_pending_for_ring_stage(
                assembler, item
            )
            and is_side_pending_view(assembler, item)
            and (
                not require_generic_join
                or is_generic_chain_join_view(item)
            )
        )
    )


def available_remaining_b_open_views(
    assembler,
    *,
    require_generic_join: bool,
) -> List[AssemblyFragmentView]:
    return ownership_module.available_views(assembler, 
        lambda item: (
            not ring_policy_module.ring_candidate_pending_for_ring_stage(
                assembler, item
            )
            and bool(getattr(item, "can_side", False))
            and view_has_flex_reconcile_role(assembler, item, "b_like_open")
            and (
                not require_generic_join
                or is_generic_chain_join_view(item)
            )
        )
    )


def available_branch_donors(assembler: Any) -> List[AssemblyFragmentView]:
    return ownership_module.available_views(assembler, 
        lambda item: str(getattr(item, "class_type", "")) in {"E", "E+", "G"}
    )


def available_flex_reconcile_views(
    assembler,
    *,
    require_roles: Sequence[str],
    require_materializable: bool = True,
    require_generic_join: bool = False,
) -> List[AssemblyFragmentView]:
    roles = {str(role) for role in list(require_roles or []) if str(role)}
    candidates = ownership_module.available_views(assembler, 
        lambda item: (
            (
                str(assembler.flex_audit.get("scenario", "") or "") != "below_range"
                or not assembler.supplemental_flex_candidate_ids
                or int(item.fragment_id) in assembler.supplemental_flex_candidate_ids
            )
            and
            (not roles or any(str(role) in view_flex_reconcile_roles(assembler, item) for role in roles))
            and (not bool(require_materializable) or is_chain_materializable_view(item))
            and (not bool(require_generic_join) or is_generic_chain_join_view(item))
        )
    )
    return sorted(
        list(candidates or []),
        key=lambda item: (
            view_flex_reconcile_priority(assembler, item),
            int(item.fragment_id),
        ),
    )


def view_context_required_counts(
    assembler,
    view: AssemblyFragmentView,
) -> Dict[str, int]:
    """Return the complete atomic pool demand rendered by one view."""
    owned_ids = {int(value) for value in list(view.source_node_ids or [])}
    rendered_context_ids: Set[int] = {
        int(value)
        for value in list(view.main_node_ids or [])
        if value is not None and int(value) not in owned_ids
    }
    for contracts in dict(view.branch_closure_contracts_by_anchor or {}).values():
        for contract in list(contracts or []):
            rendered_context_ids.update(
                int(value)
                for value in list(dict(contract or {}).get("base_path_node_ids", []) or [])
                if int(value) not in owned_ids
            )
    for stub_map in (
        view.branch_stubs_by_anchor,
        view.real_branch_stubs_by_anchor,
    ):
        for stubs in dict(stub_map or {}).values():
            rendered_context_ids.update(
                int(value)
                for stub in list(stubs or [])
                for value in list(getattr(stub, "source_node_ids", []) or [])
                if int(value) not in owned_ids
            )
    required = Counter()
    for node_id in sorted(rendered_context_ids):
        node = assembler.nodes_by_id.get(int(node_id))
        su_type = int(getattr(node, "su_type", -1))
        if su_type in {11, 22, 23}:
            required[str(su_type)] += 1
    return {str(key): int(value) for key, value in required.items()}


def claim_view_context_resources(
    assembler,
    view: AssemblyFragmentView,
) -> Tuple[Optional[AssemblyFragmentView], Dict[str, int]]:
    """Claim rendered pool context and rebind unavailable IDs by SU type."""
    local_view = deepcopy(view)
    owned_ids = {
        int(value) for value in list(local_view.source_node_ids or [])
    }
    rendered_context_ids: Set[int] = {
        int(value)
        for value in list(local_view.main_node_ids or [])
        if value is not None and int(value) not in owned_ids
    }
    for contracts in dict(local_view.branch_closure_contracts_by_anchor or {}).values():
        for contract in list(contracts or []):
            rendered_context_ids.update(
                int(value)
                for value in list(dict(contract or {}).get("base_path_node_ids", []) or [])
                if int(value) not in owned_ids
            )
    for stub_map in (
        local_view.branch_stubs_by_anchor,
        local_view.real_branch_stubs_by_anchor,
    ):
        for stubs in dict(stub_map or {}).values():
            rendered_context_ids.update(
                int(value)
                for stub in list(stubs or [])
                for value in list(getattr(stub, "source_node_ids", []) or [])
                if int(value) not in owned_ids
            )

    replacements: Dict[int, int] = {}
    required = view_context_required_counts(assembler, local_view)
    for node_id in sorted(rendered_context_ids):
        node = assembler.nodes_by_id.get(int(node_id))
        su_type = int(getattr(node, "su_type", -1))
        if su_type not in {11, 22, 23}:
            continue
        bound_owners = ownership_module.su24_support_owners(assembler, int(node_id))
        if bound_owners and int(local_view.fragment_id) not in bound_owners:
            return None, dict(required)
        with assembler.allocator.claim_trace_scope(
            "context",
            f"fragment:{int(local_view.fragment_id)}:node:{int(node_id)}",
        ):
            claimed = assembler.allocator.claim_specific(int(su_type), int(node_id))
            if claimed is None:
                claimed = assembler.allocator.claim(int(su_type), 1)
        if claimed is None:
            return None, dict(required)
        replacements[int(node_id)] = int(claimed[0])

    if not replacements:
        return local_view, {}

    def _replace_ids(values: Sequence[Any]) -> List[int]:
        return [
            int(replacements.get(int(value), int(value)))
            for value in list(values or [])
        ]

    local_view.main_node_ids = [
        None if value is None else int(replacements.get(int(value), int(value)))
        for value in list(local_view.main_node_ids or [])
    ]
    local_view.context_node_ids = _replace_ids(local_view.context_node_ids)
    for attr_name in ("fixed_connection_contracts",):
        contracts = []
        for raw_contract in list(getattr(local_view, attr_name, []) or []):
            contract = dict(raw_contract or {})
            contract["base_path_node_ids"] = _replace_ids(
                contract.get("base_path_node_ids", [])
            )
            connector_id = _safe_int(contract.get("connector_node_id"), -1)
            if connector_id in replacements:
                contract["connector_node_id"] = int(replacements[connector_id])
            contracts.append(contract)
        setattr(local_view, attr_name, contracts)
    rebound_contracts: Dict[int, List[Dict[str, Any]]] = {}
    for anchor, raw_contracts in dict(
        local_view.branch_closure_contracts_by_anchor or {}
    ).items():
        rebound_contracts[int(anchor)] = []
        for raw_contract in list(raw_contracts or []):
            contract = dict(raw_contract or {})
            contract["base_path_node_ids"] = _replace_ids(
                contract.get("base_path_node_ids", [])
            )
            connector_id = _safe_int(contract.get("connector_node_id"), -1)
            if connector_id in replacements:
                contract["connector_node_id"] = int(replacements[connector_id])
            rebound_contracts[int(anchor)].append(contract)
    local_view.branch_closure_contracts_by_anchor = rebound_contracts
    for stub_map_name in (
        "branch_stubs_by_anchor",
        "real_branch_stubs_by_anchor",
    ):
        rebound_stubs: Dict[int, List[BranchStub]] = {}
        for anchor, raw_stubs in dict(
            getattr(local_view, stub_map_name, {}) or {}
        ).items():
            rebound_stubs[int(anchor)] = []
            for raw_stub in list(raw_stubs or []):
                stub = deepcopy(raw_stub)
                stub.source_node_ids = _replace_ids(stub.source_node_ids)
                stub.node_id_notation = re.sub(
                    r"\d+",
                    lambda match: str(
                        replacements.get(int(match.group(0)), int(match.group(0)))
                    ),
                    str(stub.node_id_notation or ""),
                )
                rebound_stubs[int(anchor)].append(stub)
        setattr(local_view, stub_map_name, rebound_stubs)
    return local_view, dict(required)


def reject_invalid_mainline_contract(
    assembler,
    view: AssemblyFragmentView,
    *,
    product_kind: str,
    closing_mainline: bool = False,
) -> bool:
    """Record intrinsic route failures before branch or resource mutation."""
    reason = mainline_contract_reason(view)
    if not reason and not closing_mainline and not (view.open_left or view.open_right):
        reason = _validate_chain_product_shape(
            product_kind, view.main_labels, [view], view.main_node_ids,
        )
    if not reason:
        return False
    pending_module.record_pending(
        assembler,
        unit_id=int(view.fragment_id),
        pending_kind=pending_kind_for_view(assembler, view),
        views=[view],
        required_counts={},
        available_counts=assembler.allocator.available_counts(),
        reason=reason,
        partial_render=view.partial_render(),
        actual_render=view.pending_render(),
        diagnostics={
            "failure_domain": "topology_materialization",
            "failure_scope": "current_contract",
            "generation_id": int(getattr(assembler, "generation_id", 0)),
            "left_endpoint": view.endpoint("left").to_dict(),
            "right_endpoint": view.endpoint("right").to_dict(),
            "mainline_labels": list(view.main_labels),
            "mainline_node_ids": list(view.main_node_ids),
        },
        target_role_override=product_kind,
        mainline_slots_override=[],
    )
    return True


def finalize_with_two_end_closures(
    assembler,
    view: AssemblyFragmentView,
    *,
    product_kind: str,
    left_mode: str,
    right_mode: str,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    # This operator is only valid for a genuinely dual-open view.  A closed
    # endpoint may be treated as a second operator only for the explicit d2
    # contract whose tail boundary still requires a full F1 sequence.  In
    # particular, a basic SU3 carrying a fixed SU17-SU18 path has one
    # remaining external port and must never receive two independent closure
    # chains (which would create a spurious three-edge center).
    if reject_invalid_mainline_contract(
        assembler, view, product_kind=product_kind, closing_mainline=True,
    ):
        return False
    for side, mode in (("left", left_mode), ("right", right_mode)):
        endpoint = view.endpoint(side)
        if endpoint.label in TERMINAL_ENDPOINT_LABELS:
            return False
        if not endpoint.is_open and not _d2_endpoint_requires_full_f1(
            assembler, view, side=side, closure_mode=str(mode),
        ):
            return False
    snapshot = assembler.allocator.snapshot()
    trace_checkpoint = assembler.allocator.trace_checkpoint()
    donor_snapshot = list(branch_donors)
    original_view = view
    attempt_slots = attempt_mainline_slots(assembler, 
        original_view,
        [("left", str(left_mode)), ("right", str(right_mode))],
    )
    view, context_required = claim_view_context_resources(assembler, view)
    if view is None:
        assembler.allocator.rollback(snapshot)
        required_counts = Counter(context_required)
        required_counts.update(_slot_required_counts(attempt_slots))
        pending_module.record_pending(assembler, 
            unit_id=int(original_view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, original_view),
            views=[original_view],
            required_counts={str(key): int(value) for key, value in required_counts.items()},
            available_counts=assembler.allocator.available_counts(),
            reason="context_resource_shortage",
            partial_render=original_view.partial_render(),
            actual_render=original_view.pending_render(),
            diagnostics={
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    with assembler.allocator.claim_trace_scope("branch", f"fragment:{int(view.fragment_id)}"):
        branch_resolved, branch_required, branch_diag = branch_closure_module.resolve_branches(assembler, view, branch_donors)
    if branch_resolved is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        required_counts = Counter(context_required)
        required_counts.update(Counter(branch_required))
        required_counts.update(_slot_required_counts(attempt_slots))
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts={str(key): int(value) for key, value in required_counts.items()},
            available_counts=assembler.allocator.available_counts(),
            reason=branch_pending_reason(assembler, branch_diag, default="branch_pool_shortage"),
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics={
                **dict(branch_diag or {}),
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    validation_reason = validate_external_branch_closure_state(assembler, branch_diag)
    if validation_reason is not None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts=branch_required,
            available_counts=assembler.allocator.available_counts(),
            reason=str(validation_reason),
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics=branch_diag,
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    with assembler.allocator.claim_trace_scope("mainline", str(attempt_slots[0].get("slot_id", "left"))):
        left_payload, left_required = build_main_closure(assembler, view, side="left", closure_mode=str(left_mode))
    if left_payload is None:
        right_payload = None
        right_required = _slot_required_counts([attempt_slots[1]])
    else:
        with assembler.allocator.claim_trace_scope("mainline", str(attempt_slots[1].get("slot_id", "right"))):
            right_payload, right_required = build_main_closure(assembler, view, side="right", closure_mode=str(right_mode))
    if left_payload is None or right_payload is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        required_counts = dict(context_required or {})
        for key, value in dict(branch_required or {}).items():
            required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
        for chunk in (left_required, right_required):
            for key, value in dict(chunk or {}).items():
                required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts=required_counts,
            available_counts=assembler.allocator.available_counts(),
            reason="main_pool_shortage",
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics={
                **dict(branch_diag or {}),
                "mainline_alternative_required_counts": {
                    str(left_mode): [
                        {str(k): int(v) for k, v in option.items()}
                        for option in alternative_closure_required_counts(str(left_mode))
                    ],
                    str(right_mode): [
                        {str(k): int(v) for k, v in option.items()}
                        for option in alternative_closure_required_counts(str(right_mode))
                    ],
                },
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    left_labels, left_ids = left_payload
    right_labels, right_ids = right_payload
    # Both open endpoints may expose one shared physical bridge.  The
    # left closure is rendered first, so strip only that bridge from the
    # right payload while retaining the right endcap claimed by the
    # transaction.  Do this before composing the product route; the
    # strict duplicate/ownership audit below remains authoritative for
    # every other repeated node.
    right_labels, right_ids = remove_materialized_bridge_from_payload(
        (right_labels, right_ids),
        bridge=open_bridge_token(assembler, view, "right"),
        materialized_node_ids=(
            list(left_ids) + list(view.main_node_ids or [])
        ),
    )
    branch_map, branch_id_map = build_branch_maps(assembler, 
        view,
        branch_resolved,
        anchor_offset=len(left_labels),
    )
    # A fixed bridge can already be present in the projected view while
    # both endpoint closure operators still return it as their bridge
    # token.  Keep one physical occurrence in the product route; the
    # second occurrence is only a redundant rendering of the same edge.
    labels = list(left_labels) + list(view.main_labels or []) + list(right_labels)
    node_ids = list(left_ids) + list(view.main_node_ids or []) + list(right_ids)
    labels, node_ids, branch_map, branch_id_map = _dedupe_rendered_route_with_maps(
        labels,
        node_ids,
        branch_map,
        branch_id_map,
    )
    canonical_labels = list(view.canonical_labels or [])
    canonical_labels = (
        main_closure_canonical_labels(str(left_mode), "left")
        + canonical_labels
        + main_closure_canonical_labels(str(right_mode), "right")
    )
    source_views = collect_source_views_from_branches(assembler, [view], branch_resolved)
    validation_reason = invalid_product_reason(assembler, str(product_kind), labels, source_views, node_ids)
    suppressed_support_node_ids = collect_suppressed_support_node_ids(assembler, 
        source_views,
        branch_resolved,
    )
    if validation_reason is not None:
        return record_invalid_materialization_failure(assembler, 
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
    product = make_product(assembler, 
        product_kind=str(product_kind),
        reason=f"main_close:{left_mode}+{right_mode}",
        source_views=source_views,
        labels=labels,
        node_ids=node_ids,
        branch_map=branch_map,
        branch_id_map=branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
        diagnostics={"origin_type": str(view.class_type), "canonical_labels": canonical_labels, **dict(branch_diag or {})},
        branch_results=branch_resolved,
    )
    if product is not None:
        product.mainline_slots = [dict(item or {}) for item in attempt_slots]
    if product is None:
        return record_failed_product_result(assembler, 
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
        )
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, collect_claimed_fragment_ids(assembler, [view], branch_resolved))
    return True


def finalize_single_view_product(
    assembler,
    view: AssemblyFragmentView,
    *,
    product_kind: str,
    reason: str,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    if reject_invalid_mainline_contract(assembler, view, product_kind=product_kind):
        return False
    snapshot = assembler.allocator.snapshot()
    trace_checkpoint = assembler.allocator.trace_checkpoint()
    donor_snapshot = list(branch_donors)
    original_view = view
    view, context_required = claim_view_context_resources(assembler, view)
    if view is None:
        assembler.allocator.rollback(snapshot)
        pending_module.record_pending(assembler, 
            unit_id=int(original_view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, original_view),
            views=[original_view],
            required_counts=context_required,
            available_counts=assembler.allocator.available_counts(),
            reason="context_resource_shortage",
            partial_render=original_view.partial_render(),
            actual_render=original_view.pending_render(),
            diagnostics={
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
        )
        return False
    branch_resolved, required_counts, branch_diag = branch_closure_module.resolve_branches(assembler, view, branch_donors)
    if branch_resolved is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        total_required = Counter(context_required)
        total_required.update(Counter(required_counts))
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts={str(key): int(value) for key, value in total_required.items()},
            available_counts=assembler.allocator.available_counts(),
            reason=branch_pending_reason(assembler, branch_diag, default="branch_pool_shortage"),
            partial_render=view.partial_render(),
            actual_render=view.real_render(),
            diagnostics={
                **dict(branch_diag or {}),
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
        )
        return False
    validation_reason = validate_external_branch_closure_state(assembler, branch_diag)
    if validation_reason is not None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts=required_counts,
            available_counts=assembler.allocator.available_counts(),
            reason=str(validation_reason),
            partial_render=view.partial_render(),
            actual_render=view.real_render(),
            diagnostics=branch_diag,
        )
        return False
    normalized_labels, normalized_node_ids = _dedupe_rendered_route_with_maps(
        list(view.main_labels or []),
        list(view.main_node_ids or []),
        {},
        {},
    )[:2]
    view.main_labels = normalized_labels
    view.main_node_ids = normalized_node_ids
    branch_map, branch_id_map = build_branch_maps(assembler, view, branch_resolved)
    source_views = collect_source_views_from_branches(assembler, [view], branch_resolved)
    validation_reason = invalid_product_reason(assembler, 
        str(product_kind),
        list(view.main_labels or []),
        source_views,
        list(view.main_node_ids or []),
    )
    suppressed_support_node_ids = collect_suppressed_support_node_ids(assembler, 
        source_views,
        branch_resolved,
    )
    if validation_reason is not None:
        return record_invalid_materialization_failure(assembler, 
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
    product = make_product(assembler, 
        product_kind=str(product_kind),
        reason=str(reason),
        source_views=source_views,
        labels=list(view.main_labels or []),
        node_ids=list(view.main_node_ids or []),
        branch_map=branch_map,
        branch_id_map=branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
        diagnostics={"origin_type": str(view.class_type), "canonical_labels": list(view.canonical_labels or []), **dict(branch_diag or {})},
        branch_results=branch_resolved,
    )
    if product is None:
        return record_failed_product_result(assembler, 
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
        )
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, collect_claimed_fragment_ids(assembler, [view], branch_resolved))
    return True


def finalize_with_closure(
    assembler,
    view: AssemblyFragmentView,
    *,
    product_kind: str,
    closure_mode: str,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    if reject_invalid_mainline_contract(assembler, view, product_kind=product_kind):
        return False
    open_sides = [side for side in ("left", "right") if view.endpoint(side).is_open]
    if not open_sides:
        return finalize_single_view_product(
            assembler, view, product_kind=product_kind,
            reason="closed_mainline", branch_donors=branch_donors,
        )
    if len(open_sides) != 1:
        # This is a single-port operator. A dual-open view must be handled
        # by the caller's two-end strategy with its explicit pair of modes.
        return False
    side = open_sides[0]
    snapshot = assembler.allocator.snapshot()
    trace_checkpoint = assembler.allocator.trace_checkpoint()
    donor_snapshot = list(branch_donors)
    original_view = view
    attempt_slots = attempt_mainline_slots(assembler,
        original_view,
        [(str(side), str(closure_mode))],
    )
    view, context_required = claim_view_context_resources(assembler, view)
    if view is None:
        assembler.allocator.rollback(snapshot)
        required_counts = Counter(context_required)
        required_counts.update(_slot_required_counts(attempt_slots))
        pending_module.record_pending(assembler, 
            unit_id=int(original_view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, original_view),
            views=[original_view],
            required_counts={str(key): int(value) for key, value in required_counts.items()},
            available_counts=assembler.allocator.available_counts(),
            reason="context_resource_shortage",
            partial_render=original_view.partial_render(),
            actual_render=original_view.pending_render(),
            diagnostics={
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    branch_resolved, branch_required, branch_diag = branch_closure_module.resolve_branches(assembler, view, branch_donors)
    if branch_resolved is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        required_counts = Counter(context_required)
        required_counts.update(Counter(branch_required))
        required_counts.update(_slot_required_counts(attempt_slots))
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts={str(key): int(value) for key, value in required_counts.items()},
            available_counts=assembler.allocator.available_counts(),
            reason=branch_pending_reason(assembler, branch_diag, default="branch_pool_shortage"),
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics={
                **dict(branch_diag or {}),
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    validation_reason = validate_external_branch_closure_state(assembler, branch_diag)
    if validation_reason is not None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts=branch_required,
            available_counts=assembler.allocator.available_counts(),
            reason=str(validation_reason),
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics=branch_diag,
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    normalized_labels, normalized_node_ids = _dedupe_rendered_route_with_maps(
        list(view.main_labels or []),
        list(view.main_node_ids or []),
        {},
        {},
    )[:2]
    view.main_labels = normalized_labels
    view.main_node_ids = normalized_node_ids
    closure_payload, closure_required = build_main_closure(assembler, 
        view,
        side=side,
        closure_mode=str(closure_mode),
    )
    if closure_payload is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        required_counts = dict(context_required or {})
        for key, value in dict(branch_required or {}).items():
            required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
        for key, value in dict(closure_required or {}).items():
            required_counts[str(key)] = int(required_counts.get(str(key), 0)) + int(value)
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=pending_kind_for_view(assembler, view),
            views=[view],
            required_counts=required_counts,
            available_counts=assembler.allocator.available_counts(),
            reason=f"{closure_mode}_pool_shortage",
            partial_render=view.partial_render(),
            actual_render=view.pending_render(),
            diagnostics={
                **dict(branch_diag or {}),
                "mainline_alternative_required_counts": {
                    str(closure_mode): [
                        {str(k): int(v) for k, v in option.items()}
                        for option in alternative_closure_required_counts(str(closure_mode))
                    ],
                },
                "claim_trace": assembler.allocator.claim_trace_since(trace_checkpoint),
            },
            target_role_override=str(product_kind),
            mainline_slots_override=attempt_slots,
        )
        return False
    labels = list(view.main_labels or [])
    node_ids = list(view.main_node_ids or [])
    closure_labels, closure_ids = closure_payload
    branch_map, branch_id_map = build_branch_maps(assembler, 
        view,
        branch_resolved,
        anchor_offset=len(closure_labels) if bool(view.open_left) else 0,
    )
    if bool(view.open_left):
        labels = list(closure_labels) + labels
        node_ids = list(closure_ids) + node_ids
    else:
        labels = labels + list(closure_labels)
        node_ids = node_ids + list(closure_ids)
    # The closure operator can return a bridge already present in the
    # fixed fragment route.  Treat it as the same physical node and keep
    # a single occurrence before topology validation/commit.
    labels, node_ids, branch_map, branch_id_map = _dedupe_rendered_route_with_maps(
        labels,
        node_ids,
        branch_map,
        branch_id_map,
    )
    canonical_labels = list(view.canonical_labels or [])
    if bool(view.open_left):
        canonical_labels = main_closure_canonical_labels(str(closure_mode), side) + canonical_labels
    else:
        canonical_labels = canonical_labels + main_closure_canonical_labels(str(closure_mode), side)
    source_views = collect_source_views_from_branches(assembler, [view], branch_resolved)
    validation_reason = invalid_product_reason(assembler, str(product_kind), labels, source_views, node_ids)
    suppressed_support_node_ids = collect_suppressed_support_node_ids(assembler, 
        source_views,
        branch_resolved,
    )
    if validation_reason is not None:
        return record_invalid_materialization_failure(assembler, 
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
    product = make_product(assembler, 
        product_kind=str(product_kind),
        reason=f"main_close:{closure_mode}",
        source_views=source_views,
        labels=labels,
        node_ids=node_ids,
        branch_map=branch_map,
        branch_id_map=branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
        diagnostics={"origin_type": str(view.class_type), "canonical_labels": canonical_labels, **dict(branch_diag or {})},
        branch_results=branch_resolved,
    )
    if product is not None:
        product.mainline_slots = [dict(item or {}) for item in attempt_slots]
    if product is None:
        return record_failed_product_result(assembler, 
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
        )
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, collect_claimed_fragment_ids(assembler, [view], branch_resolved))
    return True


def composition_path(
    assembler,
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


def composition_candidate_score(
    assembler,
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


def build_composition_candidates(
    assembler,
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
                    labels, node_ids, _, _ = composition_path(assembler, provisional)
                    if invalid_product_reason(assembler, str(product_kind), labels, [left, right], node_ids) is not None:
                        continue
                    candidates.append(AssemblyCompositionCandidate(
                        left=left,
                        right=right,
                        left_reversed=bool(left_reversed),
                        right_reversed=bool(right_reversed),
                        product_kind=str(product_kind),
                        reason=str(reason),
                        score=composition_candidate_score(assembler, provisional, labels),
                    ))
    return sorted(candidates, key=lambda item: tuple(item.score))


def record_composition_failure(
    assembler,
    candidate: AssemblyCompositionCandidate,
    *,
    validation_reason: str,
    required_counts: Optional[Dict[str, int]] = None,
    rolled_back_resource_ids: Optional[Dict[str, List[int]]] = None,
    claim_trace: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    left_endpoint = candidate.left.oriented_endpoint(
        "right",
        reversed_view=bool(candidate.left_reversed),
    )
    right_endpoint = candidate.right.oriented_endpoint(
        "left",
        reversed_view=bool(candidate.right_reversed),
    )
    ownership_module.record_composition_failure_entry(assembler, {
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
        "claim_trace": [
            dict(item or {}) for item in list(claim_trace or [])
        ],
    })


def compose_two_views(
    assembler,
    candidate: AssemblyCompositionCandidate,
    *,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    left = candidate.left
    right = candidate.right
    reserved_fragment_ids = [int(left.fragment_id), int(right.fragment_id)]
    if not ownership_module.reserve_fragments(assembler, reserved_fragment_ids):
        return False
    snapshot = assembler.allocator.snapshot()
    trace_checkpoint = assembler.allocator.trace_checkpoint()
    donor_snapshot = list(branch_donors)
    prepared_left, left_context_required = claim_view_context_resources(assembler, left)
    if prepared_left is None:
        # Do not mutate the allocator further after the first failed claim.
        # The pure requirement projection still completes the atomic contract.
        prepared_right = None
        right_context_required = view_context_required_counts(assembler, right)
    else:
        prepared_right, right_context_required = claim_view_context_resources(assembler, right)
    if prepared_left is None or prepared_right is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        required = Counter(left_context_required)
        required.update(Counter(right_context_required))
        record_composition_failure(assembler, 
            candidate,
            validation_reason="context_resource_shortage",
            required_counts={str(key): int(value) for key, value in required.items()},
            claim_trace=assembler.allocator.claim_trace_since(trace_checkpoint),
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    left = prepared_left
    right = prepared_right
    candidate = AssemblyCompositionCandidate(
        left=left,
        right=right,
        left_reversed=bool(candidate.left_reversed),
        right_reversed=bool(candidate.right_reversed),
        product_kind=str(candidate.product_kind),
        reason=str(candidate.reason),
        score=tuple(candidate.score),
    )
    with assembler.allocator.claim_trace_scope("branch", f"fragment:{int(left.fragment_id)}"):
        left_branch, left_required, left_diag = branch_closure_module.resolve_branches(assembler, left, branch_donors)
    if left_branch is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        record_composition_failure(assembler, 
            candidate,
            validation_reason=branch_pending_reason(assembler, left_diag, default="left_branch_pool_shortage"),
            required_counts={
                str(key): int(value)
                for key, value in (
                    Counter(left_context_required) + Counter(right_context_required) + Counter(left_required)
                ).items()
            },
            claim_trace=assembler.allocator.claim_trace_since(trace_checkpoint),
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    left_validation_reason = validate_external_branch_closure_state(assembler, left_diag)
    if left_validation_reason is not None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        record_composition_failure(assembler, 
            candidate,
            validation_reason=str(left_validation_reason),
            required_counts=left_required,
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    with assembler.allocator.claim_trace_scope("branch", f"fragment:{int(right.fragment_id)}"):
        right_branch, right_required, right_diag = branch_closure_module.resolve_branches(assembler, right, branch_donors)
    if right_branch is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        req = Counter(left_context_required)
        req.update(Counter(right_context_required))
        req.update(Counter(left_required))
        req.update(Counter(right_required))
        record_composition_failure(assembler, 
            candidate,
            validation_reason=branch_pending_reason(assembler, right_diag, default="right_branch_pool_shortage"),
            required_counts={str(k): int(v) for k, v in req.items()},
            claim_trace=assembler.allocator.claim_trace_since(trace_checkpoint),
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    right_validation_reason = validate_external_branch_closure_state(assembler, right_diag)
    if right_validation_reason is not None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        req = Counter(left_required)
        req.update(Counter(right_required))
        record_composition_failure(assembler, 
            candidate,
            validation_reason=str(right_validation_reason),
            required_counts={str(k): int(v) for k, v in req.items()},
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    labels, node_ids, bridge_labels, _ = composition_path(assembler, candidate)
    left_labels, _ = left.oriented_mainline(reversed_view=bool(candidate.left_reversed))
    canonical_labels = ["11", "23", "11"] if str(candidate.product_kind) == "flex_chain" else ["11", "23", "22"]
    branch_map, branch_id_map = build_branch_maps(assembler, 
        left,
        left_branch,
        reverse=bool(candidate.left_reversed),
    )
    right_branch_map, right_branch_id_map = build_branch_maps(assembler, 
        right,
        right_branch,
        reverse=bool(candidate.right_reversed),
        anchor_offset=len(left_labels) + len(bridge_labels),
    )
    branch_map.update(right_branch_map)
    branch_id_map.update(right_branch_id_map)
    all_branch_results = list(left_branch or []) + list(right_branch or [])
    source_views = collect_source_views_from_branches(assembler, [left, right], all_branch_results)
    validation_reason = invalid_product_reason(assembler, 
        str(candidate.product_kind),
        labels,
        source_views,
        node_ids,
    )
    suppressed_support_node_ids = collect_suppressed_support_node_ids(assembler, 
        source_views,
        all_branch_results,
    )
    if validation_reason is not None:
        claimed_ids = assembler.allocator.claimed_ids_since(snapshot)
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        record_composition_failure(assembler, 
            candidate,
            validation_reason=str(validation_reason),
            rolled_back_resource_ids=claimed_ids,
        )
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    product = make_product(assembler, 
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
        branch_results=all_branch_results,
    )
    if product is None:
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        ownership_module.release_reserved_fragments(assembler, reserved_fragment_ids)
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, collect_claimed_fragment_ids(assembler, [left, right], all_branch_results))
    return True


def assemble_remaining_external_b_plus_b_flex(
    assembler,
    branch_donors: List[AssemblyFragmentView],
    *,
    target_count: int,
) -> bool:
    progress_made = False
    while int(current_flex_product_count(assembler, )) < int(target_count) and can_create_more_flex(assembler, ):
        b_views = available_flex_reconcile_views(assembler, 
            require_roles=["b_like_open"],
            require_materializable=True,
            require_generic_join=True,
        )
        if len(b_views) < 2:
            break
        candidates = build_composition_candidates(assembler, 
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
            if compose_two_views(assembler, candidate, branch_donors=branch_donors):
                created = True
                break
        if not created:
            break
        progress_made = True
    return progress_made


def assemble_remaining_external_b_plus_e_side(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    progress_made = False
    while True:
        b_views = available_remaining_b_open_views(assembler, require_generic_join=True)
        e_views = available_remaining_side_views(assembler, require_generic_join=True)
        if not b_views or not e_views:
            break
        candidates = build_composition_candidates(assembler, 
            [(b_view, e_view) for b_view in b_views for e_view in e_views],
            product_kind="side_chain",
            reason="B_plus_E_or_G",
            allowed_join_roles={"b_like_open_end", "e_like_open_end"},
        )
        created = False
        for candidate in candidates:
            if compose_two_views(assembler, candidate, branch_donors=branch_donors):
                created = True
                break
        if not created:
            break
        progress_made = True
    return progress_made


def assemble_remaining_external_d_double_flex(
    assembler,
    branch_donors: List[AssemblyFragmentView],
    *,
    target_count: int,
) -> bool:
    progress_made = False
    while int(current_flex_product_count(assembler, )) < int(target_count) and can_create_more_flex(assembler, ):
        cycle_progress = False
        for view in available_flex_reconcile_views(assembler, 
            require_roles=["d_like_open", "double_end_flex_source"],
            require_materializable=True,
            require_generic_join=False,
        ):
            if finalize_with_two_end_closures(assembler, 
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


def single_b_flex_closure_modes(
    assembler: Any, view: AssemblyFragmentView,
) -> Optional[List[Tuple[str, str]]]:
    if any(
        _d2_endpoint_requires_full_f1(assembler, view, side=side, closure_mode="F1")
        for side in ("left", "right")
    ):
        return [("left", "F1"), ("right", "F1")]
    open_sides = [side for side in ("left", "right") if view.endpoint(side).is_open]
    aromatic_labels = {str(value) for value in AROMATIC_SU_TYPES} | {"11"}
    if len(open_sides) != 1 or any(
        not view.endpoint(side).is_open and str(view.endpoint(side).label) not in aromatic_labels
        for side in ("left", "right")
    ):
        return None
    return [(open_sides[0], "F1")]


def close_single_b_as_flex(
    assembler: Any, view: AssemblyFragmentView, branch_donors: List[AssemblyFragmentView],
) -> bool:
    modes = single_b_flex_closure_modes(assembler, view)
    if modes is None:
        return False
    if len(modes) == 2:
        return finalize_with_two_end_closures(
            assembler, view, product_kind="flex_chain", left_mode="F1", right_mode="F1",
            branch_donors=branch_donors,
        )
    return finalize_with_closure(
        assembler, view, product_kind="flex_chain", closure_mode="F1", branch_donors=branch_donors,
    )


def assemble_remaining_external_single_b_closures_for_flex(
    assembler,
    branch_donors: List[AssemblyFragmentView],
    *,
    target_count: int,
) -> bool:
    progress_made = False
    while int(current_flex_product_count(assembler, )) < int(target_count) and can_create_more_flex(assembler, ):
        cycle_progress = False
        for view in available_flex_reconcile_views(assembler, 
            require_roles=["b_like_open", "single_end_flex_source"],
            require_materializable=True,
            require_generic_join=False,
        ):
            assembled = close_single_b_as_flex(assembler, view, branch_donors)
            if assembled:
                cycle_progress = True
                break
        if not cycle_progress:
            break
        progress_made = True
    return progress_made


def assemble_remaining_external_single_e_closures_for_side(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    progress_made = False
    for view in available_remaining_side_views(assembler, require_generic_join=False):
        if view_has_flex_reconcile_role(
            assembler, view, "b_like_open", "d_like_open"
        ):
            continue
        if finalize_with_closure(assembler, 
            view,
            product_kind="side_chain",
            closure_mode="F1",
            branch_donors=branch_donors,
        ):
            progress_made = True
    return progress_made


def assemble_remaining_b_d_as_side(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    """Close B/D fragments left after flex and ring materialization."""
    progress_made = False

    branch_only_views = sorted(
        ownership_module.available_views(assembler, 
            lambda item: (
                is_like24_family(item)
                and bool(getattr(item, "can_side", False))
                and bool(
                    dict(item.evidence or {}).get(
                        "edge_semantics_finalized", False
                    )
                )
                and not bool(item.open_left or item.open_right)
                and not ring_policy_module.ring_candidate_pending_for_ring_stage(
                    assembler, item
                )
            )
        ),
        key=lambda item: int(item.fragment_id),
    )
    for view in branch_only_views:
        if finalize_single_view_product(assembler, 
            view,
            product_kind="side_chain",
            reason="canonical_branch_only_closure",
            branch_donors=branch_donors,
        ):
            progress_made = True

    def _eligible(item: AssemblyFragmentView, role: str) -> bool:
        return bool(
            str(item.class_type) not in {"25_AB", "25_CD"}
            and bool(getattr(item, "can_side", False))
            and not ring_policy_module.ring_candidate_pending_for_ring_stage(
                assembler, item
            )
            and view_has_flex_reconcile_role(assembler, item, role)
            and is_chain_materializable_view(item)
        )

    d_views = sorted(
        ownership_module.available_views(assembler, lambda item: _eligible(item, "d_like_open")),
        key=lambda item: (view_flex_reconcile_priority(assembler, item), int(item.fragment_id)),
    )
    for view in d_views:
        left_mode, right_mode = _side_closure_modes_for_view(assembler, view)
        if finalize_with_two_end_closures(assembler, 
            view,
            product_kind="side_chain",
            left_mode=left_mode,
            right_mode=right_mode,
            branch_donors=branch_donors,
        ):
            progress_made = True

    b_views = sorted(
        ownership_module.available_views(assembler, lambda item: _eligible(item, "b_like_open")),
        key=lambda item: (view_flex_reconcile_priority(assembler, item), int(item.fragment_id)),
    )
    for view in b_views:
        if (
            _d2_endpoint_requires_full_f1(
                assembler, view, side="right", closure_mode="F1"
            )
            or _d2_endpoint_requires_full_f1(
                assembler, view, side="left", closure_mode="F1"
            )
        ):
            left_mode, right_mode = _side_closure_modes_for_view(assembler, view)
            assembled = finalize_with_two_end_closures(
                assembler,
                view,
                product_kind="side_chain",
                left_mode=left_mode,
                right_mode=right_mode,
                branch_donors=branch_donors,
            )
        else:
            # B-like fragments already have an aromatic fixed end.  Closing
            # the other side as a side chain uses F2 = [23, 22] so the
            # materialized product ends in a terminal unit.
            assembled = finalize_with_closure(
                assembler,
                view,
                product_kind="side_chain",
                closure_mode="F2",
                branch_donors=branch_donors,
            )
        if assembled:
            progress_made = True
    return progress_made


def _pending_mainline_closure_mode(
    contract: Dict[str, Any],
    view: AssemblyFragmentView,
    *,
    default: str = "F1",
) -> str:
    """Resolve the operator for a single-end pending contract.

    The pending contract may be a residual record copied from the fragment's
    flex-oriented open slot.  B-like fragments are dual-use: F1 closes them as
    flex chains, while side-chain fallback must use F2 to create a terminal
    endpoint.  Therefore target-role semantics override a stale F1 slot for
    B-like side-chain retries.
    """

    target_role = str(dict(contract or {}).get("target_role", "") or "")
    pending_kind = str(
        dict(contract or {}).get("diagnostics", {}).get(
            "public_pending_kind", ""
        )
        or ""
    )
    if target_role == "side_chain" and (
        pending_kind == "b_like_open"
        or view_has_flex_reconcile_role(None, view, "b_like_open")
    ):
        return "F2"
    for raw_slot in list(dict(contract or {}).get("mainline_slots", []) or []):
        kind = str(dict(raw_slot or {}).get("kind", "") or "")
        if kind in {"F1", "F2"}:
            return kind
    if pending_kind == "b_like_open" or view_has_flex_reconcile_role(
        None, view, "b_like_open"
    ):
        return "F1"
    if str(getattr(view, "class_type", "")) in {"25_AB"}:
        return "F2"
    return str(default) if str(default) in {"F1", "F2"} else "F1"


def flex_bounds_enforced(assembler: Any) -> bool:
    return bool(dict(assembler.flex_bounds or {}).get("enforced", False))


def flex_limit(assembler: Any, name: str) -> int:
    return max(0, _safe_int(dict(assembler.flex_bounds or {}).get(str(name), 0), 0))


def current_flex_product_count(assembler: Any) -> int:
    return len({
        tuple(item.rendered_node_ids())
        for item in list(assembler.products or [])
        if str(getattr(item, "product_kind", "")) in {"flex_chain", "residual_flex_chain"}
        and bool(getattr(item, "closed", False))
        and not bool(getattr(item, "provisional", False))
        and item.rendered_node_ids()
    })


def can_create_more_flex(assembler: Any) -> bool:
    if not flex_bounds_enforced(assembler, ):
        return True
    return int(current_flex_product_count(assembler, )) < int(flex_limit(assembler, "max"))


def release_satisfied_flex_selection(assembler: Any) -> List[int]:
    """Release unused selected fragments once the planned flex target is met."""
    target_count = int(
        dict(assembler.flex_bounds or {}).get("target", flex_limit(assembler, "min")) or 0
    )
    if int(current_flex_product_count(assembler, )) < int(target_count):
        return []
    # Backfill can fail on one optional source before a different source
    # reaches the target. Its unused flex contract must not keep the phase
    # blocked or reserve resources needed by the subsequent ring/side pass.
    unused_flex_ids = {
        int(fragment_id)
        for contract in list(assembler.pending_contracts or [])
        if str(contract.get("target_role", "")) == "flex_chain"
        for fragment_id in list(contract.get("source_fragment_ids", []) or [])
        if int(fragment_id) in assembler.supplemental_flex_candidate_ids
    }
    released_ids = sorted(
        int(fragment_id)
        for fragment_id in set(assembler.selected_flex_candidate_ids) | unused_flex_ids
        if ownership_module.fragment_available(assembler, int(fragment_id))
    )
    if not released_ids:
        return []

    released_set = set(released_ids)
    assembler.selected_flex_candidate_ids.difference_update(released_set)
    assembler.selected_flex_candidate_groups = [
        dict(group or {})
        for group in list(assembler.selected_flex_candidate_groups or [])
        if not (
            {
                int(value)
                for value in list(dict(group or {}).get("fragment_ids", []) or [])
            }
            & released_set
        )
    ]
    released_node_ids = {
        int(node_id)
        for fragment_id in released_ids
        for node_id in list(
            getattr(assembler.view_by_id.get(int(fragment_id)), "source_node_ids", [])
            if assembler.view_by_id.get(int(fragment_id)) is not None
            else []
        )
    }
    ownership_module.invalidate_pending_identity(assembler, 
        fragment_ids=released_ids,
        claimed_node_ids=released_node_ids,
    )
    assembler.flex_audit["runtime_released_selected_candidate_fragment_ids"] = list(
        released_ids
    )
    return released_ids


def flex_exceeds_max(assembler: Any) -> bool:
    if not flex_bounds_enforced(assembler, ):
        return False
    return int(current_flex_product_count(assembler, )) > int(flex_limit(assembler, "max"))


def flex_below_min(assembler: Any) -> bool:
    if not flex_bounds_enforced(assembler, ):
        return False
    return int(current_flex_product_count(assembler, )) < int(flex_limit(assembler, "min"))


def close_explicit_flex_branches_if_needed(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    progress_made = False
    views = sorted(
        ownership_module.available_views(assembler, 
            lambda item: str(pending_kind_for_view(assembler, item)) == "explicit_flex"
        ),
        key=lambda item: (
            int(item.fragment_id),
        ),
    )
    for view in views:
        if finalize_single_view_product(assembler, 
            view,
            product_kind="flex_chain",
            reason="explicit_branch_closure",
            branch_donors=branch_donors,
        ):
            progress_made = True
            continue
        break
    return progress_made


def assemble_selected_supplemental_flex(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    if not assembler.selected_flex_candidate_ids:
        return False
    target_count = int(dict(assembler.flex_bounds or {}).get("target", flex_limit(assembler, "min")) or 0)
    progress_made = False

    # The phase evaluator owns supplemental selection and ordering. Execute
    # those canonical groups first; the wider candidate pool is only a
    # deterministic backfill when a selected group cannot materialize.
    for raw_group in list(assembler.selected_flex_candidate_groups or []):
        if (int(current_flex_product_count(assembler, )) >= target_count
                or not can_create_more_flex(assembler, )):
            break
        group = dict(raw_group or {})
        fragment_ids = [
            int(value)
            for value in list(group.get("fragment_ids", []) or [])
        ]
        views = [
            assembler.view_by_id[int(fragment_id)]
            for fragment_id in fragment_ids
            if int(fragment_id) in assembler.view_by_id
            and ownership_module.fragment_available(assembler, int(fragment_id))
        ]
        if len(views) != len(fragment_ids) or not views:
            continue

        expected_signatures = {
            int(fragment_id): str(signature or "")
            for fragment_id, signature in dict(
                group.get("fragment_structure_signatures", {}) or {}
            ).items()
            if str(fragment_id).lstrip("-").isdigit() and str(signature or "")
        }
        signature_mismatches: Dict[str, Dict[str, Any]] = {}
        for fragment_id, expected_signature in sorted(
            expected_signatures.items()
        ):
            fragment = assembler.fragment_by_id.get(int(fragment_id))
            actual_signature = (
                fragment_structure_signature(fragment)
                if fragment is not None
                else ""
            )
            if str(actual_signature) == str(expected_signature):
                continue
            signature_mismatches[str(fragment_id)] = {
                "expected_signature": str(expected_signature),
                "actual_signature": str(actual_signature),
                "expected_snapshot": dict(
                    dict(
                        group.get("fragment_structure_snapshots", {}) or {}
                    ).get(str(fragment_id), {})
                    or {}
                ),
                "actual_snapshot": (
                    fragment_structure_snapshot(fragment)
                    if fragment is not None
                    else {}
                ),
            }
        if signature_mismatches:
            reason = "selected_flex_structure_signature_mismatch"
            detail = {
                "failure_domain": "topology_materialization",
                "reason": reason,
                "structure_signature_mismatches": signature_mismatches,
            }
            diagnostics = {
                "branch_contract_blocker_details": {"0": [detail]},
                "selected_flex_group_id": str(group.get("group_id", "") or ""),
                "structure_signature_mismatches": signature_mismatches,
            }
            pending_module.record_pending(assembler, 
                unit_id=min(fragment_ids),
                pending_kind=pending_kind_for_view(assembler, views[0]),
                views=views,
                required_counts={},
                available_counts=assembler.allocator.available_counts(),
                reason=reason,
                partial_render=views[0].partial_render(),
                actual_render=views[0].pending_render(),
                diagnostics=diagnostics,
                target_role_override="flex_chain",
            )
            for fragment_id in fragment_ids:
                assembler.fragment_state_by_id[int(fragment_id)] = "blocked"
                assembler.materialization_failure_by_fragment_id[int(fragment_id)] = {
                    "attempted_product_kind": "flex_chain",
                    "validation_reason": reason,
                    **detail,
                }
            continue

        strategy = str(group.get("strategy", "") or "")
        assembled = False
        if strategy == "b_plus_b" and len(views) == 2:
            candidates = build_composition_candidates(assembler, 
                [(views[0], views[1])],
                product_kind="flex_chain",
                reason="B_plus_B",
                allowed_join_roles={"b_like_open_end"},
            )
            assembled = any(
                compose_two_views(assembler, candidate, branch_donors=branch_donors)
                for candidate in candidates
            )
        elif strategy == "d_double_closure" and len(views) == 1:
            assembled = finalize_with_two_end_closures(assembler, 
                views[0],
                product_kind="flex_chain",
                left_mode="F1",
                right_mode="F1",
                branch_donors=branch_donors,
            )
        elif strategy == "b_single_closure" and len(views) == 1:
            assembled = close_single_b_as_flex(assembler, views[0], branch_donors)
        progress_made = bool(assembled or progress_made)

    if int(current_flex_product_count(assembler, )) >= int(target_count):
        return progress_made

    progress_made = bool(
        assemble_remaining_external_b_plus_b_flex(assembler, 
            branch_donors,
            target_count=target_count,
        )
        or progress_made
    )
    progress_made = bool(
        assemble_remaining_external_d_double_flex(assembler, 
            branch_donors,
            target_count=target_count,
        ) or progress_made
    )
    progress_made = bool(
        assemble_remaining_external_single_b_closures_for_flex(assembler, 
            branch_donors,
            target_count=target_count,
        ) or progress_made
    )
    return progress_made


def flex_range_violation_reason(assembler: Any) -> Optional[str]:
    if not bool(dict(assembler.flex_audit or {}).get("enforced", False)):
        return None
    blocked_reason = str(dict(assembler.flex_audit or {}).get("blocked_reason", "") or "")
    if str(blocked_reason) == "flex_window_invalid":
        return "flex_window_invalid"
    if flex_exceeds_max(assembler, ):
        return "flex_range_exceeded"
    if flex_below_min(assembler, ):
        return "flex_range_below_minimum"
    return None


def assemble_remaining_flex(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    # This fallback is only used when no enforceable flex window exists.
    # It still belongs to the external-flex phase and therefore runs
    # before any ring or side-chain allocation.
    target_count = int(
        dict(assembler.flex_bounds or {}).get("target", flex_limit(assembler, "max"))
        if flex_bounds_enforced(assembler, )
        else 10**9
    )
    progress_made = assemble_remaining_external_b_plus_b_flex(assembler, 
        branch_donors,
        target_count=target_count,
    )
    progress_made = bool(
        assemble_remaining_external_d_double_flex(assembler, 
            branch_donors,
            target_count=target_count,
        ) or progress_made
    )
    progress_made = bool(
        assemble_remaining_external_single_b_closures_for_flex(assembler, 
            branch_donors,
            target_count=target_count,
        ) or progress_made
    )
    return progress_made


def assemble_remaining_su25(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    progress_made = False
    # Do not stop after the first successful SU25.  SU25 fragments are
    # independent four-port products and all of them must be attempted in
    # this materialization pass.  Returning from inside the loop used to
    # leave later SU25 fragments in the generic pending registry, even
    # when their resources were available.
    for view in ownership_module.available_views(assembler, lambda item: (
        str(item.class_type) in {"25_AB", "25_CD"}
    )):
        if str(view.class_type) == "25_AB":
            assembled_now = finalize_with_closure(assembler, 
                view,
                product_kind="side_chain",
                closure_mode="F2",
                branch_donors=branch_donors,
            )
        else:
            assembled_now = finalize_with_two_end_closures(assembler, 
                view,
                product_kind="side_chain",
                left_mode="F1",
                right_mode="F2",
                branch_donors=branch_donors,
            )
        progress_made = bool(assembled_now) or progress_made
    return bool(progress_made)


def _assemble_remaining_side_products_once(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    progress_made = False
    for view in sorted(
        ownership_module.available_views(assembler, 
            lambda item: (
                not ring_policy_module.ring_candidate_pending_for_ring_stage(
                    assembler, item
                )
                and bool(getattr(item, "can_side", False))
                and not view_has_flex_reconcile_role(
                    assembler, item, "b_like_open", "d_like_open", "e_like_open"
                )
            )
        ),
        key=lambda item: int(item.fragment_id),
    ):
        progress_made = bool(
            finalize_single_view_product(assembler, 
                view,
                product_kind="side_chain",
                reason="explicit_side_closure",
                branch_donors=branch_donors,
            ) or progress_made
        )
    progress_made = bool(assemble_remaining_external_b_plus_e_side(assembler, branch_donors) or progress_made)
    progress_made = bool(assemble_remaining_su25(assembler, branch_donors) or progress_made)
    progress_made = bool(assemble_remaining_b_d_as_side(assembler, branch_donors) or progress_made)
    progress_made = bool(assemble_remaining_external_single_e_closures_for_side(assembler, branch_donors) or progress_made)
    return progress_made


def _recyclable_ring_candidates(assembler: Any) -> List[TopologyAssemblyProduct]:
    """Return completed pure resource rings in the required recovery order."""
    order = {
        "fatty_ring_side": 0,
        "fatty_ring_upper_lower": 1,
        "fatty_ring_fused": 2,
    }
    candidates = [
        product
        for product in list(getattr(assembler, "products", []) or [])
        if str(getattr(product, "product_kind", "") or "") in order
        and ring_topology_module.is_recyclable_pure_resource_ring(
            product,
            assembler=assembler,
        )
    ]
    return sorted(
        candidates,
        key=lambda product: (
            order[str(getattr(product, "product_kind", "") or "")],
            str(getattr(getattr(product, "ring_contract", None), "contract_id", "") or ""),
        ),
    )


def ring_recovery_candidate_costs(assembler: Any) -> List[Dict[str, Any]]:
    """Return ordered atomic recovery costs without mutating the assembler."""
    rows: List[Dict[str, Any]] = []
    for product in _recyclable_ring_candidates(assembler):
        contract = getattr(product, "ring_contract", None)
        node_su = dict(getattr(contract, "node_su_by_id", {}) or {})
        rows.append({
            "product_kind": str(getattr(product, "product_kind", "") or ""),
            "contract_id": str(getattr(contract, "contract_id", "") or ""),
            "su24_cost": sum(
                1 for su_type in node_su.values() if int(su_type) == 24
            ),
        })
    return rows


def assemble_remaining_side_products(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    """Retry side products against the allocator's current resources.

    Ring recovery is authorized and executed by the Stage3 coordinator. This
    materializer performs no independent capacity inference.
    """
    return bool(_assemble_remaining_side_products_once(assembler, branch_donors))


def retry_ready_pending_contracts(
    assembler,
    branch_donors: List[AssemblyFragmentView],
    *,
    target_role: str = "",
    subphase: str = "",
) -> bool:
    """Materialize contracts made claimable by an accepted conversion.

    A planner conversion changes the concrete allocator, not the already
    rendered product.  The owning live assembler therefore retries the exact
    pending fragment before the phase coordinator applies any broad phase
    heuristics.  This is deliberately one pass; a failed retry remains
    pending and must obtain a new resource proof before another recovery.
    """
    progress_made = False
    ready = [
        dict(contract or {})
        for contract in list(getattr(assembler, "pending_contracts", []) or [])
        if isinstance(contract, dict) and bool(contract.get("continuation_ready", False))
        and (
            not str(target_role)
            or str(contract.get("target_role", "") or "") == str(target_role)
        )
        and (
            not str(subphase)
            or str(contract.get("subphase", "") or "") == str(subphase)
        )
    ]
    for contract in sorted(
        ready,
        key=lambda item: (
            int(item.get("contract_id", 0) or 0),
            str(item.get("target_role", "") or ""),
        ),
    ):
        fragment_ids = [
            int(value)
            for value in list(contract.get("source_fragment_ids", []) or [])
        ]
        views = [
            assembler.view_by_id[int(fragment_id)]
            for fragment_id in fragment_ids
            if int(fragment_id) in getattr(assembler, "view_by_id", {})
            and ownership_module.fragment_available(assembler, int(fragment_id))
        ]
        if not views:
            continue
        target_role = str(contract.get("target_role", "") or "")
        pending_kind = str(
            contract.get("diagnostics", {}).get("public_pending_kind", "")
            or ""
        )
        assembled = False
        if len(views) == 2 and target_role in {"flex_chain", "side_chain"}:
            allowed_join_roles = (
                {"b_like_open_end"}
                if target_role == "flex_chain"
                else {"b_like_open_end", "e_like_open_end"}
            )
            candidates = build_composition_candidates(
                assembler,
                [(views[0], views[1])],
                product_kind=target_role,
                reason=f"pending_{target_role}_continuation",
                allowed_join_roles=allowed_join_roles,
            )
            assembled = any(
                compose_two_views(
                    assembler, candidate, branch_donors=branch_donors
                )
                for candidate in candidates
            )
        elif target_role == "flex_chain":
            candidate = views[0]
            if (
                pending_kind in {"d_like_open"}
                or view_has_flex_reconcile_role(
                    assembler, candidate, "d_like_open", "double_end_flex_source"
                )
            ):
                assembled = finalize_with_two_end_closures(
                    assembler,
                    candidate,
                    product_kind="flex_chain",
                    left_mode="F1",
                    right_mode="F1",
                    branch_donors=branch_donors,
                )
            elif (
                (
                    _d2_endpoint_requires_full_f1(
                        assembler, candidate, side="right", closure_mode="F1"
                    )
                    or _d2_endpoint_requires_full_f1(
                        assembler, candidate, side="left", closure_mode="F1"
                    )
                )
                and bool(
                    getattr(candidate, "open_left", False)
                    or getattr(candidate, "open_right", False)
                )
            ):
                # Fresh projections may expose only the left B-like opening
                # while the fixed single19d2 tail remains externally open.
                # Treat it as a double-ended flex source rather than closing
                # only the aromatic left side.
                assembled = finalize_with_two_end_closures(
                    assembler,
                    candidate,
                    product_kind="flex_chain",
                    left_mode="F1",
                    right_mode="F1",
                    branch_donors=branch_donors,
                )
            elif pending_kind == "b_like_open" or view_has_flex_reconcile_role(
                assembler, candidate, "b_like_open", "single_end_flex_source"
            ):
                assembled = finalize_with_closure(
                    assembler,
                    candidate,
                    product_kind="flex_chain",
                    closure_mode="F1",
                    branch_donors=branch_donors,
                )
            else:
                assembled = finalize_single_view_product(
                    assembler,
                    candidate,
                    product_kind="flex_chain",
                    reason="pending_flex_continuation",
                    branch_donors=branch_donors,
                )
        elif target_role == "side_chain":
            candidate = views[0]
            if pending_kind == "d_like_open" or view_has_flex_reconcile_role(
                assembler, candidate, "d_like_open"
            ):
                left_mode, right_mode = _side_closure_modes_for_view(assembler, candidate)
                assembled = finalize_with_two_end_closures(
                    assembler,
                    candidate,
                    product_kind="side_chain",
                    left_mode=left_mode,
                    right_mode=right_mode,
                    branch_donors=branch_donors,
                )
            elif (
                (
                    _d2_endpoint_requires_full_f1(
                        assembler, candidate, side="right", closure_mode="F1"
                    )
                    or _d2_endpoint_requires_full_f1(
                        assembler, candidate, side="left", closure_mode="F1"
                    )
                )
                and bool(
                    getattr(candidate, "open_left", False)
                    or getattr(candidate, "open_right", False)
                )
            ):
                # B-like side fallback with a fixed d2 tail requires the
                # orientation-aware F2/F1 pair (or its reversed F1/F2 form).
                left_mode, right_mode = _side_closure_modes_for_view(
                    assembler, candidate
                )
                assembled = finalize_with_two_end_closures(
                    assembler,
                    candidate,
                    product_kind="side_chain",
                    left_mode=left_mode,
                    right_mode=right_mode,
                    branch_donors=branch_donors,
                )
            elif pending_kind == "b_like_open" or view_has_flex_reconcile_role(
                assembler, candidate, "b_like_open"
            ):
                assembled = finalize_with_closure(
                    assembler,
                    candidate,
                    product_kind="side_chain",
                    closure_mode=_pending_mainline_closure_mode(
                        contract,
                        candidate,
                        default="F1",
                    ),
                    branch_donors=branch_donors,
                )
            else:
                assembled = finalize_single_view_product(
                    assembler,
                    candidate,
                    product_kind="side_chain",
                    reason="pending_side_continuation",
                    branch_donors=branch_donors,
                )
        progress_made = bool(assembled or progress_made)
    return progress_made


def recover_one_pure_resource_ring(assembler: Any) -> bool:
    """Dismantle at most one completed pure-resource ring.

    The caller must already have a resource-capacity proof.  This helper is
    shared by the fatty-ring and side-chain phases; it only performs the
    atomic ownership/allocator transition and never triggers a rebuild.
    """
    max_su24_cost = getattr(assembler, "ring_recovery_max_su24_cost", None)
    candidates = [
        product
        for product in _recyclable_ring_candidates(assembler)
        if max_su24_cost is None
        or sum(
            1
            for su_type in dict(
                getattr(getattr(product, "ring_contract", None), "node_su_by_id", {})
                or {}
            ).values()
            if int(su_type) == 24
        ) <= int(max_su24_cost)
    ]
    if not candidates:
        return False
    ring = candidates[0]
    released_ids = ownership_module.recycle_resource_ring(assembler, ring)
    if not released_ids:
        return False
    recovery_record = dict(getattr(assembler, "last_recycled_ring", {}) or {})
    recovery_record["released_ids"] = sorted(int(value) for value in released_ids)
    attempts = getattr(assembler, "ring_recovery_attempts", None)
    if attempts is not None:
        attempts.append(recovery_record)
    # The proof is one-shot. A second ring requires a new ordinary-conversion
    # search and a new authorization from Stage3.
    assembler.allow_ring_recovery = False
    assembler.ring_recovery_max_su24_cost = None
    return True


def remaining_pending_kind_and_reason(
    assembler,
    view: AssemblyFragmentView,
) -> Tuple[str, str]:
    profile = pending_module.view_external_pending_profile(view)
    profile_pending_kind = str(profile.get("external_pending_kind", "") or "")
    profile_reason = str(profile.get("external_pending_reason", "") or "")
    if profile_pending_kind and profile_reason:
        return str(profile_pending_kind), str(profile_reason)
    if str(view.class_type) in {"25_AB", "25_CD"}:
        return "pending_su25", "su25_closure_not_materialized"
    if view_has_flex_reconcile_role(assembler, view, "b_like_open", "d_like_open"):
        if view_has_flex_reconcile_role(assembler, view, "d_like_open"):
            return "d_like_open", "remaining_external_halfclosed_open"
        return "b_like_open", "remaining_external_halfclosed_open"
    if is_side_pending_view(assembler, view):
        return "explicit_side", "remaining_external_side_open"
    if bool(getattr(view, "can_flex", False)):
        return "explicit_flex", "remaining_external_halfclosed_open"
    if bool(getattr(view, "can_ring", False)):
        return "ring_only_like24", "remaining_external_halfclosed_open"
    return "other", "remaining_external_halfclosed_open"


def record_remaining_external_pending(
    assembler,
) -> None:
    for view in ownership_module.available_views(assembler, lambda item: True):
        fragment_id = int(view.fragment_id)
        ring_contract = dict(
            assembler.ring_infeasible_contracts.get(fragment_id, {}) or {}
        )
        ring_required_counts = {
            str(key): int(value)
            for key, value in dict(
                ring_contract.get("required_counts", {}) or {}
            ).items()
            if int(value) > 0
        }
        ring_missing_counts = {
            str(key): int(value)
            for key, value in dict(
                ring_contract.get("missing_counts", {}) or {}
            ).items()
            if int(value) > 0
        }
        # A dual-use 24_D-like view may eventually fall back to side/flex, but
        # a concrete resource-backed ring plan remains authoritative until the
        # Stage3 conversion pass has had an opportunity to satisfy it.
        resource_backed_ring_pending = bool(
            bool(getattr(view, "can_ring", False))
            and ring_required_counts
            and ring_missing_counts
            and ring_policy_module.view_requires_ring_resource_retry(view)
        )
        mandatory_ring_pending = bool(
            bool(getattr(view, "can_ring", False))
            and ring_policy_module.view_requires_ring(view)
        )
        materialization_failure = dict(
            assembler.materialization_failure_by_fragment_id.get(fragment_id, {}) or {}
        )
        if resource_backed_ring_pending or mandatory_ring_pending:
            if not str(
                materialization_failure.get("attempted_product_kind", "")
                or ""
            ).startswith("fatty_ring_"):
                materialization_failure = {}
            pending_kind = "pending_ring_candidate"
            ring_attempt_state = dict(
                getattr(assembler, "ring_pending_attempts", {}).get(
                    fragment_id, {}
                )
                or {}
            )
            untried_skeletons = [
                str(value)
                for value in list(
                    ring_attempt_state.get("untried_skeletons", []) or []
                )
                if str(value)
            ]
            ring_reason = str(
                assembler.ring_pending_reasons.get(fragment_id, "")
                or ""
            )
            pending_reason = str(
                ring_reason
                if ring_reason
                else (
                    "ring_skeletons_remaining:"
                    + ",".join(sorted(untried_skeletons))
                )
                if untried_skeletons
                else "mandatory_ring_candidates_exhausted"
            )
        else:
            if str(
                materialization_failure.get("attempted_product_kind", "")
                or ""
            ).startswith("fatty_ring_"):
                materialization_failure = {}
            pending_kind, pending_reason = remaining_pending_kind_and_reason(assembler, view)
        if materialization_failure:
            pending_reason = str(
                materialization_failure.get("validation_reason", "")
                or pending_reason
            )
        existing_pending = any(
            fragment_id
            in {
                int(value)
                for value in list(getattr(existing, "source_fragment_ids", []) or [])
            }
            for existing in list(assembler.pending_units or [])
        )
        # A ring materialization pass may select a different concrete plan
        # after resources are converted.  Refresh the existing pending row
        # with that plan's required counts and candidate diagnostics; the old
        # blanket skip left the first ring contract frozen forever and made
        # Stage3 stop even when the live allocator could satisfy a retry.
        ring_pending_refresh = bool(
            resource_backed_ring_pending
            or mandatory_ring_pending
            or str(
                materialization_failure.get("attempted_product_kind", "") or ""
            ).startswith("fatty_ring_")
        )
        if existing_pending and not ring_pending_refresh:
            continue
        partial = view.partial_render()
        attempted_required_counts = {
            str(key): int(value)
            for key, value in dict(
                materialization_failure.get("attempted_resource_counts", {}) or {}
            ).items()
            if int(value) > 0
        }
        if (
            not (resource_backed_ring_pending or mandatory_ring_pending)
            and materialization_failure.get("attempted_mainline_slots")
        ):
            mainline_slots_override = [
                dict(item or {})
                for item in list(
                    materialization_failure.get("attempted_mainline_slots", [])
                    or []
                )
            ]
        elif (
            str(pending_kind) == "b_like_open"
            and pending_module.pending_target_role(
                assembler,
                str(pending_kind),
                [view],
            ) == "side_chain"
        ):
            side = "left" if bool(getattr(view, "open_left", False)) else "right"
            mainline_slots_override = attempt_mainline_slots(
                assembler,
                view,
                [(str(side), "F2")],
            )
        elif (
            str(pending_kind) == "d_like_open"
            and pending_module.pending_target_role(
                assembler,
                str(pending_kind),
                [view],
            ) == "side_chain"
        ):
            attempts: List[Tuple[str, str]] = []
            left_mode, right_mode = _side_closure_modes_for_view(assembler, view)
            if bool(getattr(view, "open_left", False)):
                attempts.append(("left", str(left_mode)))
            if bool(getattr(view, "open_right", False)):
                attempts.append(("right", str(right_mode)))
            mainline_slots_override = attempt_mainline_slots(
                assembler,
                view,
                attempts,
            ) if attempts else None
        else:
            mainline_slots_override = None
        pending_module.record_pending(assembler, 
            unit_id=int(view.fragment_id),
            pending_kind=str(pending_kind),
            views=[view],
            required_counts={
                str(key): int(value)
                for key, value in (
                    ring_required_counts
                    if resource_backed_ring_pending
                    else attempted_required_counts
                ).items()
                if int(value) > 0
            },
            available_counts=assembler.allocator.available_counts(),
            reason=str(pending_reason),
            partial_render=partial,
            actual_render=view.pending_render(),
            diagnostics=(
                {
                    "materialization_failure": materialization_failure,
                    "failure_domain": materialization_failure.get("failure_domain", ""),
                    "failure_scope": materialization_failure.get("failure_scope", "candidate"),
                    "claim_trace": [
                        dict(item or {})
                        for item in list(
                            materialization_failure.get("claim_trace", []) or []
                        )
                    ],
                }
                if materialization_failure
                else None
            ),
            mainline_slots_override=mainline_slots_override,
        )
