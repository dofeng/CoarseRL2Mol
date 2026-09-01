"""Ring-view projection and concrete segment construction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ...classification.classification_tokens import AROMATIC_SU_TYPES
from ...resources.profiles import (
    build_raw_su24_resource_profiles,
    raw_su24_segment_support,
)
from ...resources.types import PHYSICAL_TOPOLOGY_RESOURCE_TYPES
from ...stage3_models import TopologyPathRender
from ..closure import branch as branch_closure_module
from ..closure import chain as chain_materializer_module
from ..state import ownership as ownership_module
from ..core.fragment import (
    AssemblyFragmentView,
    render_linear_path as _render_linear_path,
)
from ..core.models import BranchClosureResult
from ..projection.engine import _normalize_ring_backbone_node_ids, _safe_int
from . import planner as ring_planner_module
from .limits import MAX_RAW_SU24_PAIR_CANDIDATES as _MAX_RAW_SU24_PAIR_CANDIDATES

from .models import RingCandidatePlan, ResolvedRingSegment, ResolvedRingTopology

__all__ = ["RingCandidatePlan", "ResolvedRingSegment", "ResolvedRingTopology"]


def view_masked_ring_open_sides(assembler, view: AssemblyFragmentView) -> Set[str]:
    return {
        str(item)
        for item in list(dict(view.evidence or {}).get("masked_ring_open_sides", []) or [])
    }


def view_is_structurally_explicit(assembler, view: AssemblyFragmentView, *, max_suppressed: int) -> bool:
    suppressed_count = int(len(list(getattr(view, "suppressed_support_node_ids", []) or [])))
    masked_open_count = int(len(view_masked_ring_open_sides(assembler, view)))
    effective_suppressed = max(0, int(suppressed_count - masked_open_count))
    return int(effective_suppressed) <= int(max_suppressed)


def view_is_ring_core_explicit(assembler, view: AssemblyFragmentView) -> bool:
    if ring_planner_module.view_has_slot_profile(assembler, view, "fused_pair_inner_segment", "upper_lower_pair_core_segment", "side_pair_core_segment"):
        return True
    bridgehead_role = view_bridgehead_role(assembler, view)
    if bridgehead_role in {"fused_bridgehead_substitute_14", "fused_bridgehead_substitute_pair"}:
        return True
    return view_is_structurally_explicit(assembler, view, max_suppressed=0)


def view_required_branch_kinds_for_skeleton(
    assembler,
    view: AssemblyFragmentView,
    skeleton_kind: str,
) -> Dict[int, List[str]]:
    if view is None:
        return {}
    raw = dict(
        dict(dict(view.evidence or {}).get("required_branch_closures_by_anchor_in_ring", {}) or {}).get(str(skeleton_kind), {}) or {}
    )
    if not raw:
        return view_required_branch_kinds(assembler, view)
    out: Dict[int, List[str]] = {}
    for anchor, values in raw.items():
        if str(anchor) == "center":
            anchor_idx = int(view.branch_anchor_index)
        else:
            anchor_idx = _safe_int(anchor, int(view.branch_anchor_index))
        out[int(anchor_idx)] = [str(value) for value in list(values or [])]
    return out


def view_ring_contract(
    assembler,
    view: AssemblyFragmentView,
    skeleton_kind: str,
) -> Dict[str, Any]:
    contracts = dict(dict(view.evidence or {}).get("ring_contracts", {}) or {})
    return dict(contracts.get(str(skeleton_kind), {}) or {})


def view_has_ring_contract_role(
    assembler,
    view: AssemblyFragmentView,
    skeleton_kind: str,
    *roles: str,
) -> bool:
    contract = view_ring_contract(assembler, view, str(skeleton_kind))
    slot_roles = {str(item) for item in list(contract.get("slot_roles", []) or [])}
    return any(str(role) in slot_roles for role in list(roles or []))


def view_ring_contract_priority(
    assembler,
    view: AssemblyFragmentView,
    skeleton_kind: str,
) -> int:
    contract = view_ring_contract(assembler, view, str(skeleton_kind))
    return _safe_int(contract.get("priority"), 99)


def view_ring_path_capabilities(
    assembler,
    view: Optional[AssemblyFragmentView],
    skeleton_kind: str,
) -> List[str]:
    if view is None:
        return []
    raw = dict(dict(view.evidence or {}).get("ring_path_capabilities", {}) or {})
    return [str(item) for item in list(raw.get(str(skeleton_kind), []) or []) if str(item)]


def view_ring_backbone_labels(
    assembler,
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


def view_ring_backbone_node_ids(
    assembler,
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


def view_ring_backbone_span(
    assembler,
    view: Optional[AssemblyFragmentView],
    *,
    skeleton_kind: str,
) -> int:
    if view is None:
        return 0
    span = _safe_int(dict(getattr(view, "ring_backbone_span", {}) or {}).get(str(skeleton_kind)), -1)
    if span >= 0:
        return int(span)
    return int(len(view_ring_backbone_labels(assembler, view, skeleton_kind=str(skeleton_kind), reverse=False)))


def claim_ring_pool_segment(
    assembler,
    su_type: int,
    *,
    mode: str,
    candidate_id: str = "",
    node_id: Optional[int] = None,
    skeleton_kind: str = "",
    role: str = "pool",
    profile_role: str = "",
    profile_branch_h_count: int = 0,
    support_required_counts: Optional[Dict[int, int]] = None,
    support_specific_ids_by_su: Optional[Dict[int, List[int]]] = None,
    existing_branch_stubs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[ResolvedRingSegment]:
    snapshot = assembler.allocator.snapshot()
    claimed_support_ids: Dict[int, List[int]] = {}
    fixed_support_ids: Dict[int, List[int]] = {}

    def _record_missing(su_type: int, count: int = 1) -> None:
        if int(count) <= 0:
            return
        assembler.last_ring_missing_counts[str(int(su_type))] = max(
            int(assembler.last_ring_missing_counts.get(str(int(su_type)), 0) or 0),
            int(count),
        )

    for support_su, support_ids in dict(support_specific_ids_by_su or {}).items():
        for support_id in list(support_ids or []):
            # Classification-bound supports are fixed physical members of
            # their SU24 center. Keep them out of generic resource claims,
            # even when a caller reconstructed the allocator manually.
            if ownership_module.is_bound_su24_support(
                assembler,
                node_id,
                int(support_su),
                int(support_id),
            ):
                if int(support_id) in getattr(assembler, "occupied_node_ids", set()):
                    assembler.last_ring_materialization_reason = (
                        "ring_pool_fixed_support_owned_by_completed_product:"
                        f"{int(support_id)}"
                    )
                    assembler.allocator.rollback(snapshot)
                    return None
                fixed_support_ids.setdefault(int(support_su), []).append(
                    int(support_id)
                )
                continue
            claimed_support = assembler.allocator.claim_specific(
                int(support_su),
                int(support_id),
            )
            if claimed_support is None:
                if (
                    ownership_module.is_bound_su24_support(assembler,
                        node_id,
                        int(support_su),
                        int(support_id),
                    )
                    and assembler.allocator.su_key_for_node_id(int(support_id)) is None
                    and int(support_id) not in getattr(
                        assembler, "occupied_node_ids", set()
                    )
                ):
                    fixed_support_ids.setdefault(int(support_su), []).append(
                        int(support_id)
                    )
                    continue
                _record_missing(int(support_su))
                assembler.last_ring_materialization_reason = (
                    "ring_pool_specific_support_unavailable:"
                    f"{str(candidate_id or role)}:su{int(support_su)}:"
                    f"node{int(support_id)}"
                )
                assembler.allocator.rollback(snapshot)
                return None
            claimed_support_ids.setdefault(int(support_su), [])
            claimed_support_ids[int(support_su)].extend(
                int(node_id) for node_id in list(claimed_support or [])
            )
    occupied_stub_ids = {
        int(value)
        for stub in list(existing_branch_stubs or [])
        for value in list(dict(stub or {}).get("node_ids", []) or [])
    } & {
        int(value)
        for value in getattr(assembler, "occupied_node_ids", set())
    }
    if occupied_stub_ids:
        assembler.last_ring_materialization_reason = (
            "ring_pool_fixed_support_owned_by_completed_product:"
            + ",".join(str(value) for value in sorted(occupied_stub_ids))
        )
        assembler.allocator.rollback(snapshot)
        return None
    for support_su, support_count in dict(support_required_counts or {}).items():
        claimed_support = assembler.allocator.claim(int(support_su), int(support_count))
        if claimed_support is None:
            _record_missing(
                int(support_su),
                max(
                    1,
                    int(support_count)
                    - int(
                        assembler.allocator.available_counts().get(
                            str(int(support_su)), 0
                        )
                    ),
                ),
            )
            assembler.last_ring_materialization_reason = (
                "ring_pool_generic_support_unavailable:"
                f"{str(candidate_id or role)}:su{int(support_su)}:"
                f"count{int(support_count)}"
            )
            assembler.allocator.rollback(snapshot)
            return None
        claimed_support_ids[int(support_su)] = [int(node_id) for node_id in list(claimed_support or [])]
    claimed = (
        assembler.allocator.claim_specific(int(su_type), int(node_id))
        if node_id is not None
        else assembler.allocator.claim(int(su_type), 1)
    )
    if claimed is None:
        if node_id is not None:
            _record_missing(int(su_type))
        else:
            _record_missing(
                int(su_type),
                max(
                    1,
                    1
                    - int(
                        assembler.allocator.available_counts().get(
                            str(int(su_type)), 0
                        )
                    ),
                ),
            )
        assembler.last_ring_materialization_reason = (
            "ring_pool_node_unavailable:"
            f"{str(candidate_id or role)}:su{int(su_type)}:"
            + (
                f"node{int(node_id)}"
                if node_id is not None
                else "generic"
            )
        )
        assembler.allocator.rollback(snapshot)
        return None
    label = str(int(su_type))
    labels = [label]
    node_ids = [int(claimed[0])]
    branch_map: Dict[int, List[str]] = {}
    branch_id_map: Dict[int, List[str]] = {}
    branch_paths: List[Dict[str, Any]] = []
    external_attachments: List[List[int]] = []
    replaced_branch_requirements: List[Dict[str, Any]] = []
    materialized_support_ids: Set[int] = set()
    fixed_source_node_ids: Set[int] = set()
    for support_ids in dict(fixed_support_ids or {}).values():
        materialized_support_ids.update(int(value) for value in support_ids)
        fixed_source_node_ids.update(int(value) for value in support_ids)
    if int(su_type) == 24:
        cap_ids = [
            int(value)
            for value in (
                list(claimed_support_ids.get(11, []) or [])
                + list(fixed_support_ids.get(11, []) or [])
            )
        ]
        if str(role) in {"left_outer_anchor", "right_outer_anchor", "left_anchor", "right_anchor", "fixed_anchor"} and cap_ids:
            if str(role) in {"left_outer_anchor", "left_anchor", "fixed_anchor"}:
                labels = ["11", "24"]
                node_ids = [int(cap_ids[0]), int(claimed[0])]
                branch_anchor = 1
            else:
                labels = ["24", "11"]
                node_ids = [int(claimed[0]), int(cap_ids[0])]
                branch_anchor = 0
            materialized_support_ids.add(int(cap_ids[0]))
            if str(skeleton_kind) == "upper_lower" and str(role) == "fixed_anchor":
                external_attachments.append([int(cap_ids[0]), int(claimed[0])])
        else:
            branch_anchor = 0
        h23_ids = list(claimed_support_ids.get(23, []) or [])
        h22_ids = list(claimed_support_ids.get(22, []) or [])
        h_replaced_by_ring = bool(
            (
                str(skeleton_kind) == "upper_lower"
                and str(role) == "fixed_anchor"
            )
            or (
                str(skeleton_kind) == "fused"
                and str(role) in {"left_bridgehead", "right_bridgehead"}
            )
        )
        if h_replaced_by_ring:
            replaced_branch_requirements.extend({
                "anchor_node_id": int(claimed[0]),
                "kind": "H",
                "state": "REPLACED_BY_RING_BONDS",
                "ring_role": str(role),
            } for _ in range(max(0, int(profile_branch_h_count))))
        elif h23_ids and h22_ids:
            branch_map[int(branch_anchor)] = ["-23-22"]
            branch_id_map[int(branch_anchor)] = [f"-{int(h23_ids[0])}-{int(h22_ids[0])}"]
            branch_paths.append({
                "anchor_node_id": int(claimed[0]),
                "node_ids": [int(h23_ids[0]), int(h22_ids[0])],
                "kind": "H",
                "state": "FILLED",
            })
            materialized_support_ids.update({int(h23_ids[0]), int(h22_ids[0])})
        for stub in list(existing_branch_stubs or []):
            topology = str(dict(stub or {}).get("topology_notation", "") or "")
            node_text = str(dict(stub or {}).get("node_id_notation", "") or "")
            if not topology or not node_text:
                continue
            branch_map.setdefault(int(branch_anchor), [])
            branch_map[int(branch_anchor)].append(topology)
            branch_id_map.setdefault(int(branch_anchor), [])
            branch_id_map[int(branch_anchor)].append(node_text)
            stub_node_ids = [
                int(stub_node_id)
                for stub_node_id in list(dict(stub or {}).get("node_ids", []) or [])
                if _safe_int(stub_node_id, -1) >= 0
            ]
            if not stub_node_ids and _safe_int(dict(stub or {}).get("node_id"), -1) >= 0:
                stub_node_ids = [int(dict(stub or {})["node_id"])]
            if stub_node_ids:
                branch_paths.append({
                    "anchor_node_id": int(claimed[0]),
                    "node_ids": list(stub_node_ids),
                    "kind": str(dict(stub or {}).get("branch_kind", "FIXED_TERMINAL")),
                    "state": "FILLED",
                })
            for raw in re.findall(r"\d+", node_text):
                raw_id = int(raw)
                materialized_support_ids.add(raw_id)
                raw_node = assembler.nodes_by_id.get(raw_id)
                raw_su = _safe_int(getattr(raw_node, "su_type", -1), -1)
                if raw_su not in PHYSICAL_TOPOLOGY_RESOURCE_TYPES:
                    fixed_source_node_ids.add(raw_id)
    pool_node_ids_by_su = {str(int(su_type)): [int(claimed[0])]}
    for support_su, support_ids in dict(claimed_support_ids or {}).items():
        pool_node_ids_by_su.setdefault(str(int(support_su)), [])
        pool_node_ids_by_su[str(int(support_su))].extend(int(node_id) for node_id in list(support_ids or []))
    ring_labels = list(labels)
    ring_node_ids = list(node_ids)
    if (
        int(su_type) == 24
        and str(skeleton_kind) == "upper_lower"
        and str(role) == "fixed_anchor"
    ):
        ring_labels = ["24"]
        ring_node_ids = [int(claimed[0])]
    return ResolvedRingSegment(
        view=None,
        labels=labels,
        node_ids=node_ids,
        ring_backbone_labels=ring_labels,
        ring_backbone_node_ids=ring_node_ids,
        branch_map=branch_map,
        branch_id_map=branch_id_map,
        branch_modes=[str(mode)],
        branch_paths=branch_paths,
        external_attachments=external_attachments,
        like24_node_ids=[int(claimed[0])] if int(su_type) == 24 else [],
        node_profiles_by_id=(
            {int(claimed[0]): str(profile_role).upper()}
            if int(su_type) == 24 and str(profile_role)
            else {}
        ),
        replaced_branch_requirements=replaced_branch_requirements,
        pool_node_ids_by_su=pool_node_ids_by_su,
        fixed_source_node_ids=sorted(fixed_source_node_ids),
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
            "raw_su24_role": str(profile_role),
            "ring_slot_role": str(role),
            "skeleton_kind": str(skeleton_kind),
            "support_required_counts": {
                str(int(key)): int(value)
                for key, value in dict(support_required_counts or {}).items()
                if int(value) > 0
            },
        },
    )


def view_bridgehead_role(assembler, view: AssemblyFragmentView) -> str:
    return str(dict(view.evidence or {}).get("bridgehead_role", ""))


def view_ring_like24_units(assembler, view: Optional[AssemblyFragmentView]) -> int:
    if view is None:
        return 0
    return max(0, _safe_int(getattr(view, "ring_like24_units", 0), 0))


def view_fixed_branch_contracts_for_skeleton(
    assembler,
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
            for node_id in view_ring_backbone_node_ids(assembler,
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


def view_ring_branch_anchor_index(
    assembler,
    view: AssemblyFragmentView,
    skeleton_kind: str,
) -> int:
    center_ids = [
        int(node_id)
        for node_id in list(dict(view.evidence or {}).get("center_node_ids", []) or [])
    ]
    backbone_ids = view_ring_backbone_node_ids(assembler, view, skeleton_kind=str(skeleton_kind), reverse=False)
    for center_id in center_ids:
        if int(center_id) in [int(node_id) for node_id in list(backbone_ids or []) if node_id is not None]:
            return list(backbone_ids).index(int(center_id))
    return min(int(view.branch_anchor_index), max(0, len(backbone_ids) - 1))


def view_required_branch_kinds(assembler, view: AssemblyFragmentView) -> Dict[int, List[str]]:
    if view is None:
        return {}
    projected = {
        int(anchor): [str(value) for value in list(values or []) if str(value)]
        for anchor, values in dict(view.real_branch_kinds_by_anchor() or {}).items()
        if list(values or [])
    }
    if projected:
        return projected
    out: Dict[int, List[str]] = {}
    for anchor, values in dict(dict(view.evidence or {}).get("required_branch_kinds_by_anchor", {}) or {}).items():
        if str(anchor) == "center":
            anchor_idx = int(view.branch_anchor_index)
        else:
            anchor_idx = _safe_int(anchor, int(view.branch_anchor_index))
        out[int(anchor_idx)] = [str(value) for value in list(values or [])]
    return out


def render_resolved_view(
    assembler,
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
    suppressed_support_node_ids = chain_materializer_module.collect_suppressed_support_node_ids(assembler, [view], branch_resolved)
    return chain_materializer_module.render_product(assembler,
        labels,
        node_ids,
        branch_map,
        branch_id_map,
        suppressed_support_node_ids=suppressed_support_node_ids,
    )


def view_is_upper_fixed_anchor(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        (
            view_has_ring_contract_role(assembler, view, "upper_lower_ring_plan", "upper_lower_fixed_anchor")
            or (
                str(view.class_type) == "24_A"
                and bool(dict(view.evidence or {}).get("fixed_anchor_eligible", False))
                and ring_planner_module.view_has_slot_profile(assembler, view, "upper_lower_fixed_anchor")
            )
        )
    )


def side_span_priority(
    assembler,
    view: AssemblyFragmentView,
) -> Tuple[int, int, int, int]:
    class_type = str(getattr(view, "class_type", ""))
    span = view_ring_backbone_span(assembler, view, skeleton_kind="side")
    if class_type in {"24_A", "24_B", "24_AB_N"} and int(span) <= 2:
        class_rank = 0
    elif class_type in {"24_C", "24_D", "24_CD_N"} and int(span) <= 1:
        class_rank = 1
    elif class_type in {"24_CD_M", "24_AB_N_24_CD", "24_AB_N_24_AB", "24_CD_N_24_CD"}:
        class_rank = 2
    elif class_type in {"24_AB_N", "24_CD_N"}:
        class_rank = 3
    else:
        class_rank = 50
    return (
        int(class_rank),
        int(span),
        view_ring_contract_priority(assembler, view, "side_ring_plan"),
        int(view.fragment_id),
    )


def upper_path_priority(
    assembler,
    view: AssemblyFragmentView,
) -> Tuple[int, int, int, int]:
    class_type = str(getattr(view, "class_type", ""))
    span = view_ring_backbone_span(assembler, view, skeleton_kind="upper_lower")
    if class_type == "24_CD_N_24_CD":
        class_rank = 0
    elif class_type == "24_CD_N" and int(span) <= 1:
        class_rank = 1
    elif class_type in {"24_C", "24_D"} and int(span) <= 1:
        class_rank = 1
    elif class_type == "24_CD_N":
        class_rank = 2
    else:
        class_rank = 50
    return (
        int(class_rank),
        int(span),
        view_ring_contract_priority(assembler, view, "upper_lower_ring_plan"),
        int(view.fragment_id),
    )


def build_ring_template_segment(
    assembler,
    *,
    skeleton_kind: str,
    role: str,
    candidate_id: str,
    view: Optional[AssemblyFragmentView] = None,
    reverse: bool = False,
    pool_su: Optional[int] = None,
    priority: Tuple[Any, ...] = (),
) -> Optional[Dict[str, Any]]:
    if str(candidate_id) in set(
        getattr(assembler, "_ring_plan_excluded_candidate_ids", set()) or set()
    ):
        return None
    if view is not None:
        backbone_labels = view_ring_backbone_labels(assembler,
            view,
            skeleton_kind=str(skeleton_kind),
            reverse=bool(reverse),
        )
        backbone_node_ids = view_ring_backbone_node_ids(assembler,
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
            "like24_units": view_ring_like24_units(assembler, view),
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


def cached_raw_su24_profiles(
    assembler,
    remaining_ids: Sequence[int],
) -> List[Any]:
    requested_ids = {
        int(value) for value in list(remaining_ids or [])
    }
    cache = dict(getattr(assembler, "_raw_su24_profiles_by_id", {}) or {})
    scanned_ids = set(
        getattr(assembler, "_raw_su24_profile_scanned_ids", set()) or set()
    )
    unscanned_ids = sorted(requested_ids - scanned_ids)
    if unscanned_ids:
        cache.update({
            int(profile.node_id): profile
            for profile in build_raw_su24_resource_profiles(
                assembler.nodes,
                {24: unscanned_ids},
            )
        })
        scanned_ids.update(unscanned_ids)
        assembler._raw_su24_profiles_by_id = cache
        assembler._raw_su24_profile_scanned_ids = scanned_ids
    excluded = set(getattr(assembler, "_ring_plan_excluded_pool_node_ids", set()) or set())
    return [
        cache[int(node_id)]
        for node_id in sorted(requested_ids)
        if int(node_id) not in excluded
        and int(node_id) in cache
    ]


def build_pool_ring_template_segments(
    assembler,
    *,
    skeleton_kind: str,
    role: str,
    allowed_su: Sequence[int],
    priority_by_su: Optional[Dict[int, int]] = None,
    include_unavailable: bool = False,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    counts = assembler.allocator.available_counts()
    for su_type in list(allowed_su or []):
        # Pool-backed ring segments must be backed by a real Stage3
        # resource.  Structural SU labels (for example SU3) may appear in
        # a classified fragment/view, but they are not allocator-owned
        # resources and must never become generic ring requirements.
        if int(su_type) not in set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES):
            continue
        if int(su_type) == 24:
            released_su24_ids = {
                int(value)
                for value in list(getattr(assembler, "released_su24_ids", set()) or set())
            }
            remaining_ids = [
                int(value)
                for value in list(assembler.allocator.remaining_ids_by_su.get("24", []) or [])
                if int(value) not in assembler._ring_plan_excluded_pool_node_ids
                and int(value) not in released_su24_ids
            ]
            profiles = cached_raw_su24_profiles(assembler, remaining_ids)
            compatible_profiles = [
                profile
                for profile in profiles
                if f"{str(skeleton_kind)}:{str(role)}" in set(profile.allowed_ring_slots or [])
            ]
            individual_24_candidates: List[Dict[str, Any]] = []
            for profile in compatible_profiles:
                raw_role = str(profile.role or next(iter(profile.available_roles or profile.roles or []), ""))
                support = raw_su24_segment_support(
                    profile,
                    skeleton_kind=str(skeleton_kind),
                    slot_role=str(role),
                )
                candidate = build_ring_template_segment(assembler,
                    skeleton_kind=str(skeleton_kind),
                    role=str(role),
                    candidate_id=f"pool:{str(skeleton_kind)}:{str(role)}:24:{int(profile.node_id)}",
                    pool_su=24,
                    priority=(
                        int(dict(priority_by_su or {}).get(24, 99)),
                        {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(raw_role), 9),
                        int(profile.branch_h_count),
                        int(profile.node_id),
                    ),
                )
                if candidate is not None:
                    candidate["pool_node_id"] = int(profile.node_id)
                    candidate["raw_su24_role"] = str(raw_role)
                    candidate["raw_su24_profile"] = profile.to_dict()
                    candidate["support_required_counts"] = {
                        int(key): int(value)
                        for key, value in dict(support.get("support_required_counts", {}) or {}).items()
                    }
                    candidate["support_specific_ids_by_su"] = {
                        int(key): [int(value) for value in list(values or [])]
                        for key, values in dict(support.get("support_specific_ids_by_su", {}) or {}).items()
                    }
                    candidate["existing_branch_stubs"] = [
                        dict(item or {})
                        for item in list(support.get("existing_branch_stubs", []) or [])
                    ]
                    if str(role) in {"left_outer_anchor", "right_outer_anchor", "left_anchor", "right_anchor", "fixed_anchor"} and str(raw_role) in {"A", "B"}:
                        is_upper_fixed = bool(
                            str(skeleton_kind) == "upper_lower"
                            and str(role) == "fixed_anchor"
                        )
                        candidate["backbone_span"] = 1 if is_upper_fixed else 2
                        if str(role) in {"left_outer_anchor", "left_anchor", "fixed_anchor"}:
                            cap_ids = list(getattr(profile, "cap_node_ids", []) or [])
                            candidate["backbone_labels"] = (
                                ["24"] if is_upper_fixed else ["11", "24"]
                            )
                            candidate["backbone_node_ids"] = (
                                [int(profile.node_id)]
                                if is_upper_fixed
                                else [
                                    int(cap_ids[0]) if cap_ids else None,
                                    int(profile.node_id),
                                ]
                            )
                        else:
                            cap_ids = list(getattr(profile, "cap_node_ids", []) or [])
                            candidate["backbone_labels"] = ["24", "11"]
                            candidate["backbone_node_ids"] = [
                                int(profile.node_id),
                                int(cap_ids[0]) if cap_ids else None,
                            ]
                    individual_24_candidates.append(candidate)
            individual_24_candidates = ring_planner_module.bound_ring_role_candidates(assembler,
                individual_24_candidates,
                include_unavailable=bool(include_unavailable),
            )
            candidates.extend(individual_24_candidates)
            if (
                str(skeleton_kind) == "fused"
                and str(role) == "inner_core"
                and len(individual_24_candidates) >= 2
            ):
                # Pair identities are only needed when a concrete plan is
                # built.  Keep a deterministic prefix of distinct pairs;
                # the exclusion BFS exposes later IDs on retry.
                pair_count = 0
                for right_index in range(len(individual_24_candidates)):
                    for left_index in range(right_index):
                        if pair_count >= int(_MAX_RAW_SU24_PAIR_CANDIDATES):
                            break
                        left = dict(individual_24_candidates[left_index])
                        right = dict(individual_24_candidates[right_index])
                        left_id = int(left.get("pool_node_id", -1))
                        right_id = int(right.get("pool_node_id", -1))
                        if left_id < 0 or right_id < 0 or left_id == right_id:
                            continue
                        group_candidate_id = (
                            f"pool:{str(skeleton_kind)}:{str(role)}:24pair:"
                            f"{left_id}:{right_id}"
                        )
                        if group_candidate_id in set(
                            getattr(
                                assembler,
                                "_ring_plan_excluded_candidate_ids",
                                set(),
                            )
                            or set()
                        ):
                            continue
                        candidates.append({
                            "kind": "pool_group",
                            "role": str(role),
                            "candidate_id": group_candidate_id,
                            "fragment_id": None,
                            "view": None,
                            "reverse": False,
                            "pool_su": 24,
                            "pool_node_id": None,
                            "pool_node_ids": [left_id, right_id],
                            "group_segments": [left, right],
                            "backbone_span": int(left.get("backbone_span", 1))
                            + int(right.get("backbone_span", 1)),
                            "backbone_labels": list(left.get("backbone_labels", []) or [])
                            + list(right.get("backbone_labels", []) or []),
                            "backbone_node_ids": list(left.get("backbone_node_ids", []) or [])
                            + list(right.get("backbone_node_ids", []) or []),
                            "like24_units": 2,
                            "priority": (
                                int(dict(priority_by_su or {}).get(24, 99)),
                                -1,
                                left_id,
                                right_id,
                            ),
                        })
                        pair_count += 1
                    if pair_count >= int(_MAX_RAW_SU24_PAIR_CANDIDATES):
                        break
            continue
        # Generic resources are logical template slots, not distinct
        # structural alternatives.  A missing slot is still emitted so
        # the caller can produce a shortage contract.
        logical_slot_limits = {
            ("side", "middle"): 4,
            ("upper_lower", "path"): 5,
            ("fused", "inner_core"): 1,
        }
        # Do not clamp by the current pool count.  The planner needs to
        # retain a structurally valid blocked plan when, for example,
        # two SU23 middle slots are required but only one real SU23 is
        # available.  _pool_ring_attempt() then reports the exact
        # shortage and Stage3 can plan the conversion transaction.
        count = int(logical_slot_limits.get((str(skeleton_kind), str(role)), 1))
        for idx in range(count):
            candidate = build_ring_template_segment(assembler,
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


def build_pool_ring_endpoint_segment(
    assembler,
    *,
    skeleton_kind: str,
    role: str,
    su_sequence: Sequence[int],
    priority: Tuple[Any, ...],
) -> Dict[str, Any]:
    """Build one logical endpoint whose real IDs are claimed atomically."""
    children = [
        build_ring_template_segment(assembler,
            skeleton_kind=str(skeleton_kind),
            role=str(role),
            candidate_id=(
                f"pool:{str(skeleton_kind)}:{str(role)}:endpoint:"
                f"{index}:{int(su_type)}"
            ),
            pool_su=int(su_type),
            priority=tuple(priority) + (int(index),),
        )
        for index, su_type in enumerate(list(su_sequence or []))
    ]
    resolved_children = [dict(child) for child in children if child is not None]
    return {
        "kind": "pool_group",
        "role": str(role),
        "candidate_id": (
            f"pool:{str(skeleton_kind)}:{str(role)}:endpoint:"
            + "-".join(str(int(value)) for value in list(su_sequence or []))
        ),
        "fragment_id": None,
        "view": None,
        "reverse": False,
        "pool_su": None,
        "pool_node_id": None,
        "pool_node_ids": [],
        "group_segments": resolved_children,
        "backbone_span": len(resolved_children),
        "backbone_labels": [
            str(value)
            for child in resolved_children
            for value in list(child.get("backbone_labels", []) or [])
        ],
        "backbone_node_ids": [None] * len(resolved_children),
        "like24_units": 0,
        "priority": tuple(priority),
    }


def resolve_planned_ring_segment(
    assembler,
    segment_spec: Dict[str, Any],
    branch_donors: List[AssemblyFragmentView],
    *,
    skeleton_kind: str,
) -> Optional[ResolvedRingSegment]:
    if str(segment_spec.get("kind", "")) == "pool_group":
        snapshot = assembler.allocator.snapshot()
        donor_snapshot = list(branch_donors)
        children = [
            dict(child or {})
            for child in list(segment_spec.get("group_segments", []) or [])
        ]
        resolved_by_index: Dict[int, ResolvedRingSegment] = {}
        for child_index, child in sorted(
            enumerate(children),
            key=lambda item: (
                ring_planner_module.ring_segment_claim_priority(dict(item[1] or {})),
                int(item[0]),
            ),
        ):
            segment = resolve_planned_ring_segment(assembler,
                dict(child or {}),
                branch_donors,
                skeleton_kind=str(skeleton_kind),
            )
            if segment is None:
                assembler.allocator.rollback(snapshot)
                branch_donors[:] = donor_snapshot
                return None
            resolved_by_index[int(child_index)] = segment
        resolved = [resolved_by_index[index] for index in range(len(children))]
        labels: List[str] = []
        node_ids: List[Optional[int]] = []
        branch_map: Dict[int, List[str]] = {}
        branch_id_map: Dict[int, List[str]] = {}
        branch_paths: List[Dict[str, Any]] = []
        external_attachments: List[List[int]] = []
        like24_node_ids: List[int] = []
        node_profiles_by_id: Dict[int, str] = {}
        replaced_branch_requirements: List[Dict[str, Any]] = []
        source_views: List[AssemblyFragmentView] = []
        for segment in resolved:
            offset = len(labels)
            labels.extend(str(value) for value in list(segment.labels or []))
            node_ids.extend(list(segment.node_ids or []))
            for anchor, values in dict(segment.branch_map or {}).items():
                branch_map.setdefault(int(anchor) + int(offset), []).extend(
                    str(value) for value in list(values or [])
                )
            for anchor, values in dict(segment.branch_id_map or {}).items():
                branch_id_map.setdefault(int(anchor) + int(offset), []).extend(
                    str(value) for value in list(values or [])
                )
            branch_paths.extend(dict(value or {}) for value in list(segment.branch_paths or []))
            external_attachments.extend(
                [int(value) for value in list(edge or [])]
                for edge in list(segment.external_attachments or [])
            )
            like24_node_ids.extend(int(value) for value in list(segment.like24_node_ids or []))
            node_profiles_by_id.update({
                int(key): str(value)
                for key, value in dict(segment.node_profiles_by_id or {}).items()
            })
            replaced_branch_requirements.extend(
                dict(value or {})
                for value in list(segment.replaced_branch_requirements or [])
            )
            source_views.extend(
                view
                for view in list(segment.source_views or [])
                if view is not None
            )
        render = _render_linear_path(
            labels=labels,
            node_ids=node_ids,
            branch_map=branch_map,
            branch_id_map=branch_id_map,
        )
        return ResolvedRingSegment(
            view=None,
            labels=list(labels),
            node_ids=list(node_ids),
            ring_backbone_labels=list(labels),
            ring_backbone_node_ids=list(node_ids),
            branch_map=branch_map,
            branch_id_map=branch_id_map,
            branch_modes=[
                str(mode)
                for segment in resolved
                for mode in list(segment.branch_modes or [])
            ],
            branch_paths=branch_paths,
            external_attachments=external_attachments,
            like24_node_ids=sorted(set(like24_node_ids)),
            node_profiles_by_id=node_profiles_by_id,
            replaced_branch_requirements=replaced_branch_requirements,
            pool_node_ids_by_su=chain_materializer_module.merge_pool_ids(segment.pool_node_ids_by_su for segment in resolved
            ),
            fixed_source_node_ids=sorted({
                int(node_id)
                for segment in resolved
                for node_id in list(segment.fixed_source_node_ids or [])
            }),
            source_views=list({
                int(view.fragment_id): view
                for view in source_views
            }.values()),
            suppressed_support_node_ids=sorted({
                int(node_id)
                for segment in resolved
                for node_id in list(segment.suppressed_support_node_ids or [])
            }),
            ring_template_kind="pool_group",
            ring_template_roles=[str(segment_spec.get("role", "pool"))],
            diagnostics={
                "branch_render": str(render.topology_notation),
                "branch_render_gid": str(render.node_id_notation),
                "pool_group_node_ids": [
                    int(value)
                    for value in list(segment_spec.get("pool_node_ids", []) or [])
                ],
            },
        )
    if str(segment_spec.get("kind", "")) == "pool":
        segment = claim_ring_pool_segment(assembler,
            int(segment_spec.get("pool_su", -1)),
            mode=f"ring:{str(skeleton_kind)}:{str(segment_spec.get('role', 'pool'))}",
            candidate_id=str(segment_spec.get("candidate_id", "") or ""),
            node_id=(
                int(segment_spec.get("pool_node_id"))
                if segment_spec.get("pool_node_id") is not None
                else None
            ),
            skeleton_kind=str(skeleton_kind),
            role=str(segment_spec.get("role", "pool")),
            profile_role=str(segment_spec.get("raw_su24_role", "")),
            profile_branch_h_count=int(
                dict(segment_spec.get("raw_su24_profile", {}) or {}).get(
                    "branch_h_count", 0
                ) or 0
            ),
            support_required_counts={
                int(key): int(value)
                for key, value in dict(segment_spec.get("support_required_counts", {}) or {}).items()
                if int(value) > 0
            },
            support_specific_ids_by_su={
                int(key): [int(value) for value in list(values or [])]
                for key, values in dict(segment_spec.get("support_specific_ids_by_su", {}) or {}).items()
                if list(values or [])
            },
            existing_branch_stubs=[
                dict(item or {})
                for item in list(segment_spec.get("existing_branch_stubs", []) or [])
                if dict(item or {})
            ],
        )
        if segment is None and not str(assembler.last_ring_materialization_reason):
            candidate_id = str(
                segment_spec.get("candidate_id", "")
                or segment_spec.get("role", "pool")
            )
            assembler.last_ring_materialization_reason = (
                f"ring_pool_segment_claim_failed:{candidate_id}"
            )
        return segment
    view = segment_spec.get("view")
    if view is None and segment_spec.get("fragment_id") is not None:
        view = assembler.view_by_id.get(int(segment_spec.get("fragment_id")))
    if view is None:
        candidate_id = str(
            segment_spec.get("candidate_id", "")
            or segment_spec.get("fragment_id", "unknown")
        )
        assembler.last_ring_materialization_reason = (
            f"ring_segment_view_missing:{candidate_id}"
        )
        return None
    segment, segment_required_counts, diag = resolve_view_segment(assembler,
        view,
        branch_donors,
        reverse=bool(segment_spec.get("reverse", False)),
        skeleton_kind=str(skeleton_kind),
        ring_role=str(segment_spec.get("role", "")),
    )
    if segment is None:
        assembler.last_ring_missing_counts = {
            str(key): int(value)
            for key, value in dict(segment_required_counts or {}).items()
            if int(value) > 0
        }
    if segment is None and not str(assembler.last_ring_materialization_reason):
        reason = str(chain_materializer_module.validate_external_branch_closure_state(assembler, diag) or "")
        if not reason:
            reason = str(dict(diag or {}).get("external_branch_closure_reason", "") or "")
        assembler.last_ring_materialization_reason = str(reason or "external_branch_closure_pending")
    return segment


def resolve_view_segment(
    assembler,
    view: AssemblyFragmentView,
    branch_donors: List[AssemblyFragmentView],
    *,
    reverse: bool = False,
    skeleton_kind: Optional[str] = None,
    ring_role: str = "",
) -> Tuple[Optional[ResolvedRingSegment], Dict[str, int], Dict[str, Any]]:
    branch_override = (
        view_required_branch_kinds_for_skeleton(assembler, view, str(skeleton_kind))
        if skeleton_kind
        else None
    )
    branch_contract_override = (
        view_fixed_branch_contracts_for_skeleton(assembler, view, str(skeleton_kind))
        if skeleton_kind
        else None
    )
    replaced_branch_requirements: List[Dict[str, Any]] = []
    replace_h_with_ring_bonds = bool(
        (str(skeleton_kind or "") == "upper_lower" and str(ring_role) == "fixed_anchor")
        or (
            str(skeleton_kind or "") == "fused"
            and str(ring_role) in {"left_bridgehead", "right_bridgehead"}
        )
    )
    if replace_h_with_ring_bonds and branch_override is not None:
        filtered_override: Dict[int, List[str]] = {}
        for anchor, kinds in dict(branch_override or {}).items():
            kept: List[str] = []
            for kind in list(kinds or []):
                if str(kind) in {"H", "H1"}:
                    replaced_branch_requirements.append({
                        "anchor_index": int(anchor),
                        "kind": str(kind),
                        "state": "REPLACED_BY_RING_BONDS",
                        "ring_role": str(ring_role),
                    })
                else:
                    kept.append(str(kind))
            filtered_override[int(anchor)] = kept
        branch_override = filtered_override
    branch_resolved, required_counts, branch_diag = branch_closure_module.resolve_branches(assembler,
        view,
        branch_donors,
        branch_kind_map_override=branch_override,
        branch_contract_map_override=branch_contract_override,
    )
    if branch_resolved is None:
        return None, dict(required_counts or {}), dict(branch_diag or {})
    validation_reason = chain_materializer_module.validate_external_branch_closure_state(assembler, branch_diag)
    if validation_reason is not None:
        branch_diag = dict(branch_diag or {})
        branch_diag["external_branch_closure_reason"] = str(validation_reason)
        return None, dict(required_counts or {}), branch_diag
    if skeleton_kind:
        labels = view_ring_backbone_labels(assembler, view, skeleton_kind=str(skeleton_kind), reverse=False)
        node_ids = view_ring_backbone_node_ids(assembler, view, skeleton_kind=str(skeleton_kind), reverse=False)
        center_anchor_override = view_ring_branch_anchor_index(assembler, view, str(skeleton_kind))
        render_labels = list(labels)
        render_node_ids = list(node_ids)
        if (
            str(skeleton_kind) == "upper_lower"
            and view_is_upper_fixed_anchor(assembler, view)
            and list(view.main_labels or [])
            and list(view.main_node_ids or [])
        ):
            render_labels = [str(value) for value in list(view.main_labels or []) if str(value)]
            render_node_ids = [
                None if value is None else int(value)
                for value in list(view.main_node_ids or [])
            ]

        # Ring-only open endpoints are represented by ``None`` in the
        # projected backbone.  Claim their real pool nodes here, inside
        # the caller's candidate transaction, so the same physical SU11
        # cannot be reused by another ring or later side closure.
        open_pool_positions = dict(
            dict(getattr(view, "diagnostics", {}) or {}).get(
                "ring_open_pool_positions", {}
            )
            or {}
        ).get(str(skeleton_kind), {})
        ring_pool_ids_by_su: Dict[str, List[int]] = {}
        for raw_index, raw_su in dict(open_pool_positions or {}).items():
            index = _safe_int(raw_index, -1)
            su_type = _safe_int(raw_su, -1)
            if index < 0 or index >= len(render_node_ids) or su_type < 0:
                continue
            if render_node_ids[index] is not None:
                continue
            claimed = assembler.allocator.claim(int(su_type), 1)
            if claimed is None:
                assembler.last_ring_missing_counts[str(su_type)] = int(
                    assembler.last_ring_missing_counts.get(str(su_type), 0)
                ) + 1
                assembler.last_ring_materialization_reason = (
                    f"ring_open_endpoint_resource_missing:{int(su_type)}"
                )
                return None, {str(su_type): 1}, {
                    "ring_open_endpoint_resource_missing": int(su_type)
                }
            render_node_ids[index] = int(claimed[0])
            node_ids[index] = int(claimed[0])
            ring_pool_ids_by_su.setdefault(str(su_type), []).append(int(claimed[0]))
    else:
        labels = list(view.main_labels or [])
        node_ids = list(view.main_node_ids or [])
        center_anchor_override = None
        render_labels = list(labels)
        render_node_ids = list(node_ids)
        ring_pool_ids_by_su = {}
    render = render_resolved_view(assembler,
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
    branch_paths: List[Dict[str, Any]] = []
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
        anchor_node_id = (
            int(ring_backbone_node_ids[int(anchor_index)])
            if 0 <= int(anchor_index) < len(ring_backbone_node_ids)
            and ring_backbone_node_ids[int(anchor_index)] is not None
            else -1
        )
        branch_paths.append({
            "anchor_node_id": int(anchor_node_id),
            "node_ids": [
                int(node_id)
                for node_id in list(item.main_node_ids or [])
                if node_id is not None
            ],
            "kind": str(item.branch_kind),
            "state": "FILLED",
        })
    source_views = [view] + [
        assembler.view_by_id[x]
        for item in list(branch_resolved or [])
        for x in list(item.source_fragment_ids or [])
        if x in assembler.view_by_id
    ]
    pool_ids = chain_materializer_module.merge_pool_ids([item.pool_node_ids_by_su for item in list(branch_resolved or [])]
        + [ring_pool_ids_by_su]
    )
    external_attachments: List[List[int]] = []
    if str(skeleton_kind or "") == "upper_lower" and view_is_upper_fixed_anchor(assembler, view):
        ring_ids = {
            int(value) for value in list(ring_backbone_node_ids or [])
            if value is not None
        }
        external_ids = [
            int(value)
            for value in list(render_node_ids or [])
            if value is not None
            and int(value) not in ring_ids
            and int(getattr(assembler.nodes_by_id.get(int(value)), "su_type", -1)) in set(AROMATIC_SU_TYPES)
        ]
        if ring_ids and external_ids:
            external_attachments = [[int(external_ids[0]), int(ring_backbone_node_ids[0])]]
    like24_units = view_ring_like24_units(assembler, view)
    like24_node_ids = (
        [
            int(value)
            for value in list(
                dict(view.evidence or {}).get("center_node_ids", []) or []
            )
            if _safe_int(value, -1) >= 0
        ]
        if int(like24_units) > 0
        else []
    )
    if not like24_node_ids and int(like24_units) > 0:
        like24_node_ids = [int(view.fragment_id)]
    raw_profile = str(dict(view.evidence or {}).get("raw_su24_role", "") or "").upper()
    if not raw_profile and str(view.class_type) in {"24_A", "24_B", "24_C", "24_D"}:
        raw_profile = str(view.class_type)[-1]
    if not raw_profile and str(view.class_type).startswith("24_"):
        raw_profile = str(view.class_type).split("_", 2)[1].upper()
    bridgehead_role = str(dict(view.evidence or {}).get("bridgehead_role", "") or "")
    if bridgehead_role in {
        "fused_bridgehead_substitute_14",
        "fused_bridgehead_substitute_pair",
    }:
        raw_profile = "14_SUBSTITUTE"
    elif bridgehead_role == "fused_bridgehead_candidate":
        raw_profile = "C"
    segment = ResolvedRingSegment(
        view=view,
        labels=labels,
        node_ids=node_ids,
        ring_backbone_labels=ring_backbone_labels,
        ring_backbone_node_ids=ring_backbone_node_ids,
        branch_map=branch_map,
        branch_id_map=branch_id_map,
        branch_modes=[str(item.mode) for item in list(branch_resolved or [])],
        branch_paths=branch_paths,
        external_attachments=external_attachments,
        like24_node_ids=sorted(set(like24_node_ids)),
        node_profiles_by_id={
            int(node_id): str(raw_profile)
            for node_id in list(like24_node_ids or [])
            if str(raw_profile)
        },
        replaced_branch_requirements=[
            {
                **dict(requirement or {}),
                "anchor_node_id": (
                    int(ring_backbone_node_ids[int(dict(requirement or {}).get("anchor_index", -1))])
                    if 0 <= int(dict(requirement or {}).get("anchor_index", -1)) < len(ring_backbone_node_ids)
                    and ring_backbone_node_ids[int(dict(requirement or {}).get("anchor_index", -1))] is not None
                    else -1
                ),
            }
            for requirement in replaced_branch_requirements
        ],
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
