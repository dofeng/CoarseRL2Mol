"""Bounded search helpers for fused-ring middle segments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core.fragment import AssemblyFragmentView, strip_open_markers as _strip_open_markers
from .models import ResolvedRingSegment
from . import planner as ring_planner_module
from . import segments as ring_segments_module
from .limits import (
    MAX_EQUIVALENT_RING_IDENTITIES,
    MAX_FUSED_COMBINATION_EVALUATIONS,
    MAX_MIDDLE_SEARCH_STATES,
    MAX_RAW_SU24_PAIR_CANDIDATES,
    MAX_RING_ROLE_CANDIDATES,
)

__all__ = [
    "MAX_EQUIVALENT_RING_IDENTITIES",
    "MAX_FUSED_COMBINATION_EVALUATIONS",
    "MAX_MIDDLE_SEARCH_STATES",
    "MAX_RAW_SU24_PAIR_CANDIDATES",
    "MAX_RING_ROLE_CANDIDATES",
]


def view_supports_fused_inner(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        "fused" in ring_planner_module.view_ring_allowed_skeletons(assembler, view)
        and (
            ring_segments_module.view_has_ring_contract_role(assembler, view, "fused_ring_plan", "fused_inner_core")
            or ring_planner_module.view_has_slot_profile(assembler, view, "fused_inner_segment", "fused_pair_inner_segment")
        )
    )


def view_is_fused_pair_bridgehead(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        ring_segments_module.view_has_ring_contract_role(assembler, view, "fused_ring_plan", "fused_bridgehead_pair")
        or str(ring_segments_module.view_bridgehead_role(assembler, view)) == "fused_bridgehead_substitute_pair"
    )


def view_supports_fused_pair_bridgeheads(
    assembler,
    view: AssemblyFragmentView,
) -> bool:
    """Return whether one real adjacent 14-14 view can fill both bridgeheads."""
    if not view_is_fused_pair_bridgehead(assembler, view):
        return False
    if "fused" not in ring_planner_module.view_ring_allowed_skeletons(assembler, view):
        return False
    node_ids = ring_segments_module.view_ring_backbone_node_ids(
        assembler,
        view,
        skeleton_kind="fused",
        reverse=False,
    )
    if len(node_ids) != 2 or any(node_id is None for node_id in node_ids):
        return False
    left_id, right_id = (int(node_ids[0]), int(node_ids[1]))
    if left_id == right_id:
        return False
    left = assembler.nodes_by_id.get(left_id)
    right = assembler.nodes_by_id.get(right_id)
    if left is None or right is None:
        return False
    if int(getattr(left, "su_type", -1)) != 14 or int(getattr(right, "su_type", -1)) != 14:
        return False
    left_neighbors = {int(value) for value in list(getattr(left, "hop1_ids", []) or [])}
    right_neighbors = {int(value) for value in list(getattr(right, "hop1_ids", []) or [])}
    return bool(right_id in left_neighbors or left_id in right_neighbors)


def view_supports_fused_bridgehead_single(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        "fused" in ring_planner_module.view_ring_allowed_skeletons(assembler, view)
        and (
            ring_segments_module.view_has_ring_contract_role(assembler, view, "fused_ring_plan", "fused_bridgehead")
            or ring_planner_module.view_has_slot_profile(assembler, view, "fused_bridgehead")
        )
        and not view_is_fused_pair_bridgehead(assembler, view)
        and ring_segments_module.view_ring_backbone_span(assembler, view, skeleton_kind="fused") == 1
    )


def view_supports_real_su24_c_bridgehead(
    assembler,
    view: AssemblyFragmentView,
) -> bool:
    if not view_supports_fused_bridgehead_single(assembler, view):
        return False
    evidence = dict(view.evidence or {})
    raw_role = str(evidence.get("raw_su24_role", "") or "").upper()
    if not raw_role and str(view.class_type) == "24_C":
        raw_role = "C"
    center_ids = [
        int(value)
        for value in list(evidence.get("center_node_ids", []) or [])
        if int(value) in assembler.nodes_by_id
    ]
    if not center_ids and int(view.fragment_id) in assembler.nodes_by_id:
        center_ids = [int(view.fragment_id)]
    return bool(
        raw_role == "C"
        and len(center_ids) == 1
        and int(getattr(assembler.nodes_by_id[center_ids[0]], "su_type", -1)) == 24
    )


def fused_inner_priority(
    assembler,
    view: AssemblyFragmentView,
    *,
    pair_bridgehead_mode: bool,
) -> Tuple[int, int, int]:
    class_type = str(getattr(view, "class_type", ""))
    family = str(getattr(view, "family", ""))
    backbone_labels = ring_segments_module.view_ring_backbone_labels(assembler, 
        view,
        skeleton_kind="fused",
        reverse=False,
    )
    subtype = str((backbone_labels or [None])[0] or "")
    bridgehead_capable = bool(
        ring_planner_module.view_has_slot_profile(assembler, view, "fused_bridgehead")
        or ring_segments_module.view_bridgehead_role(assembler, view) in {
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
            ring_segments_module.view_ring_contract_priority(assembler, view, "fused_ring_plan"),
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
        ring_segments_module.view_ring_contract_priority(assembler, view, "fused_ring_plan"),
        int(view.fragment_id),
    )


def fused_inner_pool23_distribution(
    assembler,
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


def search_ring_middle_segments(
    assembler,
    candidates: Sequence[Dict[str, Any]],
    *,
    remaining_span: int,
    claimed_fragment_ids: Set[int],
    used_candidate_ids: Set[str],
    current_like24: int,
    min_like24: int,
    max_like24: int,
    claimed_pool_node_ids: Optional[Set[int]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if int(remaining_span) == 0:
        if int(current_like24) < int(min_like24) or int(current_like24) > int(max_like24):
            return None
        return []
    ordered = sorted(
        [dict(item or {}) for item in list(candidates or [])],
        key=lambda item: tuple(item.get("priority", (99, 99, 99))),
    )
    failed_states: Set[Tuple[Any, ...]] = set()
    search_state_count = 0

    def _search(
        span_left: int,
        used_fragments_local: Set[int],
        used_pool_nodes_local: Set[int],
        used_candidates_local: Set[str],
        like24_local: int,
        start_index: int,
    ) -> Optional[List[Dict[str, Any]]]:
        nonlocal search_state_count
        search_state_count += 1
        if search_state_count > int(MAX_MIDDLE_SEARCH_STATES):
            return None
        if int(span_left) == 0:
            if int(like24_local) < int(min_like24) or int(like24_local) > int(max_like24):
                return None
            return []
        state_key = (
            int(start_index),
            int(span_left),
            int(like24_local),
            tuple(sorted(int(value) for value in used_fragments_local)),
            tuple(sorted(int(value) for value in used_pool_nodes_local)),
            tuple(sorted(str(value) for value in used_candidates_local)),
        )
        if state_key in failed_states:
            return None
        for candidate_index in range(int(start_index), len(ordered)):
            candidate = ordered[candidate_index]
            candidate_id = str(candidate.get("candidate_id", ""))
            fragment_id = candidate.get("fragment_id")
            span = int(candidate.get("backbone_span", 0) or 0)
            if span <= 0 or span > int(span_left):
                continue
            if candidate_id and candidate_id in used_candidates_local:
                continue
            if fragment_id is not None and int(fragment_id) in used_fragments_local:
                continue
            candidate_pool_node_ids = ring_planner_module.ring_segment_specific_claim_ids(
                dict(candidate or {})
            )
            if candidate_pool_node_ids & used_pool_nodes_local:
                continue
            next_like24 = int(like24_local) + int(candidate.get("like24_units", 0) or 0)
            if next_like24 > int(max_like24):
                continue
            next_fragments = set(used_fragments_local)
            next_pool_nodes = set(used_pool_nodes_local)
            next_candidates = set(used_candidates_local)
            if fragment_id is not None:
                next_fragments.add(int(fragment_id))
            next_pool_nodes.update(candidate_pool_node_ids)
            if candidate_id:
                next_candidates.add(candidate_id)
            tail = _search(
                int(span_left) - int(span),
                next_fragments,
                next_pool_nodes,
                next_candidates,
                next_like24,
                int(candidate_index) + 1,
            )
            if tail is not None:
                return [dict(candidate)] + list(tail or [])
        failed_states.add(state_key)
        return None

    return _search(
        int(remaining_span),
        set(claimed_fragment_ids or set()),
        set(claimed_pool_node_ids or set()),
        set(used_candidate_ids or set()),
        int(current_like24),
        0,
    )


def segment_has_forbidden_ring_branch_mix(
    assembler,
    segment: ResolvedRingSegment,
) -> bool:
    branch_topologies: List[str] = []
    for values in dict(segment.branch_map or {}).values():
        branch_topologies.extend(str(value) for value in list(values or []))
    normalized = [_strip_open_markers(value) for value in branch_topologies]
    has_h_closure = any(str(value) in {"23-22", "23-23-22"} for value in normalized)
    has_fixed_terminal = any(str(value).startswith("22") or str(value) in {"1", "4"} for value in normalized)
    return bool(has_h_closure and has_fixed_terminal)
