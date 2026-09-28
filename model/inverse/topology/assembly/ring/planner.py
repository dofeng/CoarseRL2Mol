"""Pure ring-plan identity helpers used by deterministic enumeration."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..state import ownership as ownership_module
from ..core.fragment import AssemblyFragmentView, is_aromatic_label as _is_aromatic_label
from ..projection.engine import (
    _normalize_ring_backbone_node_ids,
    _safe_int,
    _skeleton_allowed_name,
)
from . import materializer as ring_materializer_module
from . import policy as ring_policy_module
from . import search as ring_search_module
from . import segments as ring_segments_module
from .limits import (
    MAX_EQUIVALENT_RING_IDENTITIES as _MAX_EQUIVALENT_RING_IDENTITIES,
    MAX_FUSED_COMBINATION_EVALUATIONS as _MAX_FUSED_COMBINATION_EVALUATIONS,
    MAX_RING_ROLE_CANDIDATES as _MAX_RING_ROLE_CANDIDATES,
)
from .policy import view_requires_ring
from .segments import RingCandidatePlan


def ring_plan_signature(plan: RingCandidatePlan) -> str:
    candidate_ids = [
        str(dict(segment or {}).get("candidate_id", ""))
        for segment in list(plan.ordered_segments or [])
    ]
    return f"{str(plan.skeleton_kind)}:" + "|".join(candidate_ids)


def ring_plan_pool_su24_ids(plan: RingCandidatePlan) -> Set[int]:
    selected: Set[int] = set()

    def _collect(segment: Dict[str, Any]) -> None:
        if str(segment.get("kind", "")) == "pool_group":
            for child in list(segment.get("group_segments", []) or []):
                _collect(dict(child or {}))
            return
        if (
            str(segment.get("kind", "")) == "pool"
            and int(segment.get("pool_su", -1)) == 24
            and segment.get("pool_node_id") is not None
        ):
            selected.add(int(segment["pool_node_id"]))

    for raw_segment in list(plan.ordered_segments or []):
        _collect(dict(raw_segment or {}))
    return selected


def ring_plan_candidate_ids(plan: RingCandidatePlan) -> Set[str]:
    return {
        str(dict(segment or {}).get("candidate_id", ""))
        for segment in list(plan.ordered_segments or [])
        if str(dict(segment or {}).get("candidate_id", ""))
    }


__all__ = [
    "ring_plan_candidate_ids",
    "ring_plan_pool_su24_ids",
    "ring_plan_signature",
]


def ring_candidate_shape(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return the structural shape used to collapse interchangeable IDs."""
    profile = dict(candidate.get("raw_su24_profile", {}) or {})
    support = tuple(sorted(
        (int(key), int(value))
        for key, value in dict(candidate.get("support_required_counts", {}) or {}).items()
    ))
    fixed = tuple(sorted(
        int(value)
        for value in dict(profile.get("terminal_su_by_node_id", {}) or {}).values()
    ))
    # Classified fragment views are not interchangeable resources.  Two
    # ``24_D`` views can have different fixed neighbours, branch contracts,
    # and ownership state even when their rendered labels are identical.  The
    # old shape omitted the fragment identity, so
    # ``MAX_EQUIVALENT_RING_IDENTITIES`` silently discarded later views (for
    # example 414 after 402-413 had been consumed).  Pool IDs remain
    # interchangeable and continue to use the compact structural shape.
    fragment_identity = (
        int(candidate.get("fragment_id"))
        if str(candidate.get("kind", "")) == "view"
        and candidate.get("fragment_id") is not None
        else None
    )
    return (
        str(candidate.get("kind", "")),
        fragment_identity,
        str(candidate.get("role", "")),
        int(candidate.get("pool_su", -1) or -1),
        int(candidate.get("backbone_span", 0) or 0),
        int(candidate.get("like24_units", 0) or 0),
        str(candidate.get("raw_su24_role", "") or ""),
        support,
        fixed,
        tuple(str(value) for value in list(candidate.get("backbone_labels", []) or [])),
    )


def ring_candidate_specific_claims_available(
    assembler,
    candidate: Dict[str, Any],
) -> bool:
    """Return whether every candidate-bound pool ID is claimable now."""
    if str(candidate.get("kind", "")) == "pool_group":
        return all(
            ring_candidate_specific_claims_available(assembler, dict(child or {}))
            for child in list(candidate.get("group_segments", []) or [])
        )
    if str(candidate.get("kind", "")) != "pool":
        return True
    pool_su = candidate.get("pool_su")
    pool_node_id = candidate.get("pool_node_id")
    if pool_node_id is not None:
        if int(pool_su) == 24 and int(pool_node_id) in {
            int(value)
            for value in list(getattr(assembler, "released_su24_ids", set()) or set())
        }:
            return False
        if int(pool_node_id) in getattr(assembler, "occupied_node_ids", set()):
            return False
        if int(pool_node_id) not in {
            int(value)
            for value in list(
                assembler.allocator.remaining_ids_by_su.get(str(int(pool_su)), [])
                or []
            )
        }:
            return False
    for raw_su, values in dict(
        candidate.get("support_specific_ids_by_su", {}) or {}
    ).items():
        support_su = int(raw_su)
        bound_support_ids = {
            int(node_id)
            for node_id in list(values or [])
            if ownership_module.is_bound_su24_support(assembler, 
                pool_node_id,
                int(support_su),
                int(node_id),
            )
        }
        if bound_support_ids & {
            int(node_id)
            for node_id in getattr(assembler, "occupied_node_ids", set())
        }:
            return False
        if any(
            int(node_id) in getattr(assembler, "occupied_node_ids", set())
            for node_id in list(values or [])
            if int(node_id) not in bound_support_ids
        ):
            return False
        available = {
            int(value)
            for value in list(
                assembler.allocator.remaining_ids_by_su.get(str(int(support_su)), [])
                or []
            )
        }
        if any(
            int(node_id) not in available
            for node_id in list(values or [])
            if int(node_id) not in bound_support_ids
        ):
            return False
    profile = dict(candidate.get("raw_su24_profile", {}) or {})
    for field in ("cap_node_ids", "terminal_node_ids"):
        if any(
            int(node_id) in getattr(assembler, "occupied_node_ids", set())
            for node_id in list(profile.get(field, []) or [])
        ):
            return False
    return True


def bound_ring_role_candidates(
    assembler,
    candidates: Sequence[Dict[str, Any]],
    *,
    max_total: int = _MAX_RING_ROLE_CANDIDATES,
    include_unavailable: bool = False,
) -> List[Dict[str, Any]]:
    """Keep representative identities while preserving exclusion retries."""
    selected: List[Dict[str, Any]] = []
    shape_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for candidate in sorted(
        [
            dict(item or {})
            for item in list(candidates or [])
            if dict(item or {})
            and (
                bool(include_unavailable)
                or ring_candidate_specific_claims_available(assembler, 
                    dict(item or {})
                )
            )
        ],
        key=lambda item: tuple(item.get("priority", (99, 99, 99))),
    ):
        shape = ring_candidate_shape(candidate)
        if shape_counts[shape] >= int(_MAX_EQUIVALENT_RING_IDENTITIES):
            continue
        selected.append(candidate)
        shape_counts[shape] += 1
        if len(selected) >= int(max_total):
            break
    return selected


def ring_segment_claim_priority(segment_spec: Dict[str, Any]) -> int:
    """Claim candidate-bound IDs before generic resources and view branches."""
    kind = str(segment_spec.get("kind", ""))
    if kind == "pool_group":
        children = [
            dict(child or {})
            for child in list(segment_spec.get("group_segments", []) or [])
        ]
        if any(ring_segment_claim_priority(child) == 0 for child in children):
            return 0
        return 1
    if kind == "pool":
        has_specific_support = any(
            list(values or [])
            for values in dict(
                segment_spec.get("support_specific_ids_by_su", {}) or {}
            ).values()
        )
        return 0 if segment_spec.get("pool_node_id") is not None or has_specific_support else 1
    return 2


def ring_segment_specific_claim_id_list(
    segment_spec: Dict[str, Any],
) -> List[int]:
    if str(segment_spec.get("kind", "")) == "pool_group":
        return [
            int(node_id)
            for child in list(segment_spec.get("group_segments", []) or [])
            for node_id in ring_segment_specific_claim_id_list(
                dict(child or {})
            )
        ]
    if str(segment_spec.get("kind", "")) != "pool":
        return []
    selected = [
        int(node_id)
        for values in dict(
            segment_spec.get("support_specific_ids_by_su", {}) or {}
        ).values()
        for node_id in list(values or [])
    ]
    if segment_spec.get("pool_node_id") is not None:
        selected.append(int(segment_spec["pool_node_id"]))
    return selected


def ring_segment_specific_claim_ids(
    segment_spec: Dict[str, Any],
) -> Set[int]:
    return set(ring_segment_specific_claim_id_list(segment_spec))


def planned_ring_source_views(
    assembler,
    ordered_segments: Sequence[Dict[str, Any]],
) -> List[AssemblyFragmentView]:
    source: Dict[int, AssemblyFragmentView] = {}
    for segment in list(ordered_segments or []):
        view = segment.get("view")
        if view is None and segment.get("fragment_id") is not None:
            view = assembler.view_by_id.get(int(segment.get("fragment_id")))
        if view is None:
            continue
        source[int(view.fragment_id)] = view
    return list(source.values())


def ring_plan_contains_candidate_view(
    assembler,
    plan: RingCandidatePlan,
    candidate_views: Sequence[AssemblyFragmentView],
) -> bool:
    candidate_ids = {
        int(view.fragment_id) for view in list(candidate_views or [])
    }
    return bool(candidate_ids & {
        int(view.fragment_id)
        for view in planned_ring_source_views(assembler, plan.ordered_segments)
    })


def planned_ring_like24_count(
    assembler,
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


def build_side_ring_template_candidates(
    assembler,
    views: Sequence[AssemblyFragmentView],
    *,
    include_unavailable: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    left_anchors: List[Dict[str, Any]] = []
    right_anchors: List[Dict[str, Any]] = []
    middle_segments: List[Dict[str, Any]] = []
    for view in list(views or []):
        capabilities = set(ring_segments_module.view_ring_path_capabilities(assembler, view, "side"))
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
                backbone_labels = ring_segments_module.view_ring_backbone_labels(assembler, 
                    view,
                    skeleton_kind="side",
                    reverse=bool(reverse),
                )
                if not backbone_labels:
                    continue
                if _is_aromatic_label(str(backbone_labels[0])):
                    candidate = ring_segments_module.build_ring_template_segment(assembler, 
                        skeleton_kind="side",
                        role="left_anchor",
                        candidate_id=f"view:side:left:{int(view.fragment_id)}:{1 if reverse else 0}",
                        view=view,
                        reverse=bool(reverse),
                        priority=(int(terminal_rank), *ring_segments_module.side_span_priority(assembler, view), 0 if not reverse else 1),
                    )
                    if candidate is not None:
                        left_anchors.append(candidate)
                if _is_aromatic_label(str(backbone_labels[-1])):
                    candidate = ring_segments_module.build_ring_template_segment(assembler, 
                        skeleton_kind="side",
                        role="right_anchor",
                        candidate_id=f"view:side:right:{int(view.fragment_id)}:{1 if reverse else 0}",
                        view=view,
                        reverse=bool(reverse),
                        priority=(int(terminal_rank), *ring_segments_module.side_span_priority(assembler, view), 0 if not reverse else 1),
                    )
                    if candidate is not None:
                        right_anchors.append(candidate)
        if capabilities & {"inner_single", "inner_pair"}:
            inner_role = "inner_pair" if "inner_pair" in capabilities else "inner_single"
            inner_rank = 0 if inner_role == "inner_pair" else 1
            candidate = ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="side",
                role="middle",
                candidate_id=f"view:side:middle:{int(view.fragment_id)}",
                view=view,
                reverse=False,
                priority=(int(inner_rank), *ring_segments_module.side_span_priority(assembler, view)),
            )
            if candidate is not None:
                middle_segments.append(candidate)
    left_anchors.extend(
        ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="side",
            role="left_anchor",
            allowed_su=[24],
            priority_by_su={24: 30},
            include_unavailable=bool(include_unavailable),
        )
    )
    right_anchors.extend(
        ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="side",
            role="right_anchor",
            allowed_su=[24],
            priority_by_su={24: 30},
            include_unavailable=bool(include_unavailable),
        )
    )
    # External side rings use the same real resource pool as the other
    # external ring templates. Keep ordinary 11-23 endpoints available in
    # every mode.
    left_anchors.append(
        ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
            skeleton_kind="side",
            role="left_anchor",
            su_sequence=(11, 23),
            priority=(40, 11, 23),
        )
    )
    right_anchors.append(
        ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
            skeleton_kind="side",
            role="right_anchor",
            su_sequence=(23, 11),
            priority=(40, 23, 11),
        )
    )
    # Pair-core fragments already contribute three consecutive ring
    # nodes (24-23-24).  A six-member side ring therefore needs one F1
    # endpoint and one direct aromatic endpoint; requiring F1 on both
    # sides would produce a seven-node path and can never pass ring audit.
    left_anchors.append(
        ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
            skeleton_kind="side",
            role="left_anchor",
            su_sequence=(11,),
            priority=(41, 11),
        )
    )
    right_anchors.append(
        ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
            skeleton_kind="side",
            role="right_anchor",
            su_sequence=(11,),
            priority=(41, 11),
        )
    )
    middle_segments.extend(
        ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="side",
            role="middle",
            allowed_su=[23, 24],
            priority_by_su={23: 20, 24: 21},
            include_unavailable=bool(include_unavailable),
        )
    )
    return (
        sorted(
            bound_ring_role_candidates(assembler, 
                left_anchors,
                include_unavailable=bool(include_unavailable),
            ),
            key=lambda item: tuple(item.get("priority", (99, 99, 99))),
        ),
        sorted(
            bound_ring_role_candidates(assembler, 
                right_anchors,
                include_unavailable=bool(include_unavailable),
            ),
            key=lambda item: tuple(item.get("priority", (99, 99, 99))),
        ),
        sorted(
            bound_ring_role_candidates(assembler, 
                middle_segments,
                include_unavailable=bool(include_unavailable),
            ),
            key=lambda item: tuple(item.get("priority", (99, 99, 99))),
        ),
    )


def build_upper_ring_template_candidates(
    assembler,
    views: Sequence[AssemblyFragmentView],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    anchors: List[Dict[str, Any]] = []
    path_segments: List[Dict[str, Any]] = []
    for view in list(views or []):
        capabilities = set(ring_segments_module.view_ring_path_capabilities(assembler, view, "upper_lower"))
        if ring_segments_module.view_is_upper_fixed_anchor(assembler, view):
            candidate = ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="upper_lower",
                role="fixed_anchor",
                candidate_id=f"view:upper:anchor:{int(view.fragment_id)}",
                view=view,
                reverse=False,
                priority=(0, ring_segments_module.view_ring_contract_priority(assembler, view, "upper_lower_ring_plan"), int(view.fragment_id)),
            )
            if candidate is not None:
                anchors.append(candidate)
        if capabilities & {"path_single", "path_pair"}:
            path_role = "path_pair" if "path_pair" in capabilities else "path_single"
            path_rank = 0 if path_role == "path_pair" else 1
            candidate = ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="upper_lower",
                role="path",
                candidate_id=f"view:upper:path:{int(view.fragment_id)}",
                view=view,
                reverse=False,
                priority=(int(path_rank), *ring_segments_module.upper_path_priority(assembler, view)),
            )
            if candidate is not None:
                path_segments.append(candidate)
    anchors.extend(
        ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="upper_lower",
            role="fixed_anchor",
            allowed_su=[24],
            priority_by_su={24: 30},
        )
    )
    path_segments.extend(
        ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="upper_lower",
            role="path",
            allowed_su=[23, 24],
            priority_by_su={23: 20, 24: 21},
        )
    )
    return (
        sorted(anchors, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
        sorted(path_segments, key=lambda item: tuple(item.get("priority", (99, 99, 99)))),
    )


def plan_side_ring_templates(
    assembler,
    preferred_views: Sequence[AssemblyFragmentView],
) -> RingCandidatePlan:
    available = [
        view for view in list(preferred_views or [])
        if ownership_module.fragment_available(assembler, int(view.fragment_id))
    ]
    left_anchors, right_anchors, middle_segments = build_side_ring_template_candidates(assembler, available)
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
                left_pool_ids = ring_segment_specific_claim_ids(
                    dict(left_anchor or {})
                )
                right_pool_ids = ring_segment_specific_claim_ids(
                    dict(right_anchor or {})
                )
                if left_pool_ids & right_pool_ids:
                    continue
                base_like24 = int(left_anchor.get("like24_units", 0) or 0) + int(right_anchor.get("like24_units", 0) or 0)
                if base_like24 > 4:
                    continue
                remaining_span = 6 - int(left_anchor.get("backbone_span", 0) or 0) - int(right_anchor.get("backbone_span", 0) or 0)
                if remaining_span < 0:
                    continue
                middle = ring_search_module.search_ring_middle_segments(assembler, 
                    middle_segments,
                    remaining_span=int(remaining_span),
                    claimed_fragment_ids={
                        int(fragment_id)
                        for fragment_id in [left_fragment_id, right_fragment_id]
                        if fragment_id is not None
                    },
                    claimed_pool_node_ids=set(left_pool_ids) | set(right_pool_ids),
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
                like24_count = planned_ring_like24_count(assembler, candidate_segments)
                if like24_count < 2 or like24_count > 4:
                    continue
                ordered_segments = candidate_segments
                break
            if ordered_segments:
                break
    if not ordered_segments and not missing_slots:
            if not middle_segments:
                missing_slots.append("side_inner_slot_missing")
            else:
                missing_slots.append("ring_backbone_nodes_insufficient")
    source_views = planned_ring_source_views(assembler, ordered_segments)
    return RingCandidatePlan(
        ring_kind="fatty_ring_side",
        skeleton_kind="side_ring_plan",
        source_views=source_views,
        ordered_segments=list(ordered_segments or []),
        missing_slots=sorted({str(item) for item in list(missing_slots or []) if str(item)}),
        like24_count_min=2,
        like24_count_max=4,
        template_complete=bool(ordered_segments) and not bool(missing_slots),
    )


def plan_upper_ring_templates(
    assembler,
    preferred_views: Sequence[AssemblyFragmentView],
) -> RingCandidatePlan:
    available = [
        view for view in list(preferred_views or [])
        if ownership_module.fragment_available(assembler, int(view.fragment_id))
    ]
    anchors, path_segments = build_upper_ring_template_candidates(assembler, available)
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
            path = ring_search_module.search_ring_middle_segments(assembler, 
                path_segments,
                remaining_span=int(remaining_span),
                claimed_fragment_ids={
                    int(anchor.get("fragment_id"))
                    for _ in [0]
                    if anchor.get("fragment_id") is not None
                },
                claimed_pool_node_ids={
                    int(anchor.get("pool_node_id"))
                    for _ in [0]
                    if anchor.get("pool_node_id") is not None
                },
                used_candidate_ids={str(anchor.get("candidate_id", ""))},
                current_like24=int(anchor.get("like24_units", 0) or 0),
                min_like24=2,
                max_like24=3,
            )
            if path is None:
                continue
            candidate_segments = [dict(anchor)] + list(path or [])
            like24_count = planned_ring_like24_count(assembler, candidate_segments)
            if like24_count < 2 or like24_count > 3:
                continue
            ordered_segments = candidate_segments
            if isinstance(anchor_view, AssemblyFragmentView):
                closure_target_fragment_id = int(anchor_view.fragment_id)
                anchor_node_ids = ring_segments_module.view_ring_backbone_node_ids(assembler, 
                    anchor_view,
                    skeleton_kind="upper_lower",
                    reverse=False,
                )
                if anchor_node_ids:
                    closure_target_node_id = None if anchor_node_ids[-1] is None else int(anchor_node_ids[-1])
            else:
                pool_anchor_node_ids = _normalize_ring_backbone_node_ids(
                    anchor.get("backbone_node_ids", [])
                )
                closure_target_node_id = next(
                    (
                        int(node_id)
                        for node_id in reversed(pool_anchor_node_ids)
                        if node_id is not None
                    ),
                    None,
                )
            break
    if not ordered_segments:
        if not missing_slots:
            if not path_segments:
                missing_slots.append("upper_path_slot_missing")
            else:
                missing_slots.append("upper_ring_path_span_unresolved")
    source_views = planned_ring_source_views(assembler, ordered_segments)
    return RingCandidatePlan(
        ring_kind="fatty_ring_upper_lower",
        skeleton_kind="upper_lower_ring_plan",
        source_views=source_views,
        ordered_segments=list(ordered_segments or []),
        closure_target_fragment_id=closure_target_fragment_id,
        closure_target_node_id=closure_target_node_id,
        missing_slots=sorted({str(item) for item in list(missing_slots or []) if str(item)}),
        like24_count_min=2,
        like24_count_max=3,
        template_complete=bool(ordered_segments) and not bool(missing_slots),
    )


def plan_fused_ring_templates(
    assembler,
    preferred_views: Sequence[AssemblyFragmentView],
) -> RingCandidatePlan:
    available = [
        view for view in list(preferred_views or [])
        if ownership_module.fragment_available(assembler, int(view.fragment_id))
    ]
    anchors = [
        view for view in available
        if "fused" in view_ring_allowed_skeletons(assembler, view)
        and (
            ring_segments_module.view_has_ring_contract_role(assembler, view, "fused_ring_plan", "fused_outer_anchor")
            or view_has_slot_profile(assembler, view, "fused_outer_anchor")
        )
        and bool(dict(view.evidence or {}).get("touches_aromatic_anchor", False))
        and ring_segments_module.view_is_structurally_explicit(assembler, view, max_suppressed=1)
    ]
    bridgeheads = [
        view for view in available
        if "fused" in view_ring_allowed_skeletons(assembler, view)
        and ring_search_module.view_supports_real_su24_c_bridgehead(assembler, view)
        and ring_segments_module.view_is_ring_core_explicit(assembler, view)
    ]
    pair_bridgeheads = [
        view for view in available
        if ring_search_module.view_supports_fused_pair_bridgeheads(assembler, view)
        and ring_segments_module.view_is_ring_core_explicit(assembler, view)
    ]
    inner_cores = [
        view for view in available
        if ring_search_module.view_supports_fused_inner(assembler, view)
        and not ring_search_module.view_is_fused_pair_bridgehead(assembler, view)
        and ring_segments_module.view_is_ring_core_explicit(assembler, view)
    ]
    missing_slots: List[str] = []
    if len(anchors) < 2:
        missing_slots.append("fused_outer_anchor_missing")
    if not bridgeheads and not pair_bridgeheads:
        missing_slots.append("fused_bridgehead_missing")
    if not inner_cores:
        missing_slots.append("fused_inner_core_missing")
    anchors.sort(key=lambda item: (ring_segments_module.view_ring_contract_priority(assembler, item, "fused_ring_plan"), int(item.fragment_id)))
    bridgeheads.sort(key=lambda item: (
        ring_segments_module.view_ring_contract_priority(assembler, item, "fused_ring_plan"),
        int(item.fragment_id),
    ))
    pair_bridgeheads.sort(key=lambda item: (
        ring_segments_module.view_ring_contract_priority(assembler, item, "fused_ring_plan"),
        int(item.fragment_id),
    ))
    inner_cores.sort(
        key=lambda item: ring_search_module.fused_inner_priority(assembler, 
            item,
            pair_bridgehead_mode=False,
        )
    )
    left_anchor = anchors[0] if anchors else None
    right_anchor = next((item for item in anchors[1:] if left_anchor is not None and int(item.fragment_id) != int(left_anchor.fragment_id)), None)
    # Pair bridgeheads have priority and require the bounded combination search
    # below because one physical fragment fills two logical bridgehead slots.
    bridge = bridgeheads[0] if bridgeheads and not pair_bridgeheads else None
    right_bridge = (
        next(
            (
                item for item in bridgeheads[1:]
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
            and ring_search_module.fused_inner_pool23_distribution(assembler, 
                ring_segments_module.view_ring_backbone_span(assembler, item, skeleton_kind="fused")
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
        left_anchor_spec = ring_segments_module.build_ring_template_segment(assembler, 
            skeleton_kind="fused",
            role="left_outer_anchor",
            candidate_id=f"view:fused:left_anchor:{int(left_anchor.fragment_id)}",
            view=left_anchor,
            reverse=False,
            priority=(ring_segments_module.view_ring_contract_priority(assembler, left_anchor, "fused_ring_plan"), int(left_anchor.fragment_id)),
        )
        left_bridge_spec = ring_segments_module.build_ring_template_segment(assembler, 
            skeleton_kind="fused",
            role="left_bridgehead",
            candidate_id=f"view:fused:left_bridge:{int(bridge.fragment_id)}",
            view=bridge,
            reverse=False,
            priority=(ring_segments_module.view_ring_contract_priority(assembler, bridge, "fused_ring_plan"), int(bridge.fragment_id)),
        )
        inner_spec = ring_segments_module.build_ring_template_segment(assembler, 
            skeleton_kind="fused",
            role="inner_core",
            candidate_id=f"view:fused:inner:{int(core.fragment_id)}",
            view=core,
            reverse=False,
            priority=ring_search_module.fused_inner_priority(assembler, core, pair_bridgehead_mode=False),
        )
        right_bridge_spec = ring_segments_module.build_ring_template_segment(assembler, 
            skeleton_kind="fused",
            role="right_bridgehead",
            candidate_id=f"view:fused:right_bridge:{int(right_bridge.fragment_id)}",
            view=right_bridge,
            reverse=False,
            priority=(ring_segments_module.view_ring_contract_priority(assembler, right_bridge, "fused_ring_plan"), int(right_bridge.fragment_id)),
        )
        right_anchor_spec = ring_segments_module.build_ring_template_segment(assembler, 
            skeleton_kind="fused",
            role="right_outer_anchor",
            candidate_id=f"view:fused:right_anchor:{int(right_anchor.fragment_id)}",
            view=right_anchor,
            reverse=True,
            priority=(ring_segments_module.view_ring_contract_priority(assembler, right_anchor, "fused_ring_plan"), int(right_anchor.fragment_id)),
        )
        if left_anchor_spec is not None and left_bridge_spec is not None and inner_spec is not None and right_bridge_spec is not None and right_anchor_spec is not None:
            ordered_segments = [
                dict(left_anchor_spec),
                dict(left_bridge_spec),
                dict(inner_spec),
                dict(right_bridge_spec),
                dict(right_anchor_spec),
            ]
    if not ordered_segments:
        left_anchor_specs = [
            ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="fused",
                role="left_outer_anchor",
                candidate_id=f"view:fused:left_anchor:{int(view.fragment_id)}",
                view=view,
                reverse=False,
                priority=(ring_segments_module.view_ring_contract_priority(assembler, view, "fused_ring_plan"), int(view.fragment_id)),
            )
            for view in anchors
        ]
        right_anchor_specs = [
            ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="fused",
                role="right_outer_anchor",
                candidate_id=f"view:fused:right_anchor:{int(view.fragment_id)}",
                view=view,
                reverse=True,
                priority=(ring_segments_module.view_ring_contract_priority(assembler, view, "fused_ring_plan"), int(view.fragment_id)),
            )
            for view in anchors
        ]
        left_anchor_specs = [item for item in left_anchor_specs if item is not None]
        right_anchor_specs = [item for item in right_anchor_specs if item is not None]
        left_anchor_specs.extend(ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="fused",
            role="left_outer_anchor",
            allowed_su=[24],
            priority_by_su={24: 30},
        ))
        right_anchor_specs.extend(ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="fused",
            role="right_outer_anchor",
            allowed_su=[24],
            priority_by_su={24: 30},
        ))
        left_anchor_specs.append(
            ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
                skeleton_kind="fused",
                role="left_outer_anchor",
                su_sequence=(11, 23),
                priority=(40, 11, 23),
            )
        )
        right_anchor_specs.append(
            ring_segments_module.build_pool_ring_endpoint_segment(assembler, 
                skeleton_kind="fused",
                role="right_outer_anchor",
                su_sequence=(23, 11),
                priority=(40, 23, 11),
            )
        )

        bridge_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for pair_view in pair_bridgeheads:
            candidate_id = f"view:fused:bridge_pair:{int(pair_view.fragment_id)}"
            pair_spec = ring_segments_module.build_ring_template_segment(assembler,
                skeleton_kind="fused",
                role="left_bridgehead",
                candidate_id=candidate_id,
                view=pair_view,
                reverse=False,
                priority=(
                    ring_segments_module.view_ring_contract_priority(
                        assembler, pair_view, "fused_ring_plan"
                    ),
                    int(pair_view.fragment_id),
                ),
            )
            if pair_spec is None:
                continue
            if len(list(pair_spec.get("backbone_node_ids", []) or [])) != 2:
                continue
            # The source view resolves atomically as a two-node segment.  The
            # planner accounts for one logical node in each bridgehead slot.
            left_spec = dict(pair_spec)
            left_spec["backbone_span"] = 1
            right_spec = dict(pair_spec)
            right_spec["role"] = "right_bridgehead"
            right_spec["backbone_span"] = 1
            right_spec["shared_candidate_id"] = str(candidate_id)
            bridge_pairs.append((left_spec, right_spec))
        for left_view in bridgeheads:
            left_spec = ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="fused",
                role="left_bridgehead",
                candidate_id=f"view:fused:left_bridge:{int(left_view.fragment_id)}",
                view=left_view,
                reverse=False,
                priority=(ring_segments_module.view_ring_contract_priority(assembler, left_view, "fused_ring_plan"), int(left_view.fragment_id)),
            )
            if left_spec is None:
                continue
            for right_view in bridgeheads:
                if int(right_view.fragment_id) == int(left_view.fragment_id):
                    continue
                right_spec = ring_segments_module.build_ring_template_segment(assembler, 
                    skeleton_kind="fused",
                    role="right_bridgehead",
                    candidate_id=f"view:fused:right_bridge:{int(right_view.fragment_id)}",
                    view=right_view,
                    reverse=False,
                    priority=(ring_segments_module.view_ring_contract_priority(assembler, right_view, "fused_ring_plan"), int(right_view.fragment_id)),
                )
                if right_spec is not None:
                    bridge_pairs.append((left_spec, right_spec))
        pool_left_bridges = ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="fused",
            role="left_bridgehead",
            allowed_su=[24],
            priority_by_su={24: 30},
        )
        pool_right_bridges = ring_segments_module.build_pool_ring_template_segments(assembler, 
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
            ring_segments_module.build_ring_template_segment(assembler, 
                skeleton_kind="fused",
                role="inner_core",
                candidate_id=f"view:fused:inner:{int(view.fragment_id)}",
                view=view,
                reverse=False,
                priority=ring_search_module.fused_inner_priority(assembler, view, pair_bridgehead_mode=False),
            )
            for view in inner_cores
        ]
        inner_specs = [item for item in inner_specs if item is not None]
        inner_specs.extend(ring_segments_module.build_pool_ring_template_segments(assembler, 
            skeleton_kind="fused",
            role="inner_core",
            allowed_su=[23, 24],
            priority_by_su={23: 20, 24: 30},
        ))
        left_anchor_specs = bound_ring_role_candidates(assembler, left_anchor_specs)
        right_anchor_specs = bound_ring_role_candidates(assembler, right_anchor_specs)
        inner_specs = bound_ring_role_candidates(assembler, 
            inner_specs,
            max_total=_MAX_RING_ROLE_CANDIDATES,
        )
        bridge_pairs = sorted(
            bridge_pairs,
            key=lambda pair: (
                tuple(pair[0].get("priority", (99,))),
                tuple(pair[1].get("priority", (99,))),
            ),
        )[: int(_MAX_RING_ROLE_CANDIDATES * 4)]

        def _candidate_owner_ids(spec: Dict[str, Any]) -> Set[Tuple[str, int]]:
            out: Set[Tuple[str, int]] = set()
            if spec.get("fragment_id") is not None:
                out.add(("fragment", int(spec.get("fragment_id"))))
            out.update(
                ("pool", int(value))
                for value in ring_segment_specific_claim_ids(
                    dict(spec or {})
                )
            )
            return out

        combination_evaluations = 0
        search_budget_exhausted = False
        for left_spec in sorted(left_anchor_specs, key=lambda item: tuple(item.get("priority", (99,)))):
            if search_budget_exhausted:
                break
            for right_spec in sorted(right_anchor_specs, key=lambda item: tuple(item.get("priority", (99,)))):
                if search_budget_exhausted:
                    break
                if _candidate_owner_ids(left_spec) & _candidate_owner_ids(right_spec):
                    continue
                for left_bridge_spec, right_bridge_spec in bridge_pairs:
                    if search_budget_exhausted:
                        break
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
                        combination_evaluations += 1
                        if combination_evaluations > int(_MAX_FUSED_COMBINATION_EVALUATIONS):
                            search_budget_exhausted = True
                            break
                        if any(_candidate_owner_ids(inner_spec) & owners for owners in owner_sets):
                            continue
                        if ring_search_module.fused_inner_pool23_distribution(assembler, int(inner_spec.get("backbone_span", 0) or 0)) is None:
                            continue
                        candidate_segments = [
                            dict(left_spec),
                            dict(left_bridge_spec),
                            dict(inner_spec),
                            dict(right_bridge_spec),
                            dict(right_spec),
                        ]
                        like24_count = planned_ring_like24_count(assembler, candidate_segments)
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
        if search_budget_exhausted and not ordered_segments:
            missing_slots.append("fused_candidate_search_budget_exhausted")
    source_views = planned_ring_source_views(assembler, ordered_segments)
    plan = RingCandidatePlan(
        ring_kind="fatty_ring_fused",
        skeleton_kind="fused_ring_plan",
        source_views=source_views,
        ordered_segments=list(ordered_segments or []),
        missing_slots=sorted({str(item) for item in missing_slots}),
        like24_count_min=4,
        like24_count_max=6,
        template_complete=bool(ordered_segments) and not bool(missing_slots),
    )
    return plan


def pool_ring_spec_requirements(
    assembler,
    spec: Dict[str, Any],
) -> Tuple[Counter, Counter, Dict[int, Set[int]], Set[int]]:
    total: Counter = Counter()
    generic: Counter = Counter()
    source_ids_by_su: Dict[int, Set[int]] = defaultdict(set)
    fixed_source_ids: Set[int] = set()
    if str(spec.get("kind", "")) == "pool_group":
        for child in list(spec.get("group_segments", []) or []):
            child_total, child_generic, child_sources, child_fixed = (
                pool_ring_spec_requirements(assembler, dict(child or {}))
            )
            total.update(child_total)
            generic.update(child_generic)
            fixed_source_ids.update(child_fixed)
            for su_type, values in child_sources.items():
                source_ids_by_su[int(su_type)].update(int(value) for value in values)
        return total, generic, source_ids_by_su, fixed_source_ids
    if str(spec.get("kind", "")) != "pool":
        return total, generic, source_ids_by_su, fixed_source_ids
    pool_su = int(spec.get("pool_su", -1))
    pool_node_id = spec.get("pool_node_id")
    if pool_node_id is None:
        if pool_su >= 0:
            total[int(pool_su)] += 1
            generic[int(pool_su)] += 1
    else:
        node_id = int(pool_node_id)
        total[int(pool_su)] += 1
        source_ids_by_su[int(pool_su)].add(node_id)
    for raw_su, values in dict(spec.get("support_specific_ids_by_su", {}) or {}).items():
        su_type = int(raw_su)
        for value in list(values or []):
            node_id = int(value)
            # A support is a fixed source only when classification bound
            # this exact ID to the selected SU24 center.  Do not infer
            # ownership merely because an ID happens to be absent from
            # the allocator; that hid genuine ownership conflicts.
            if ownership_module.is_bound_su24_support(assembler, 
                pool_node_id,
                int(su_type),
                int(node_id),
            ):
                fixed_source_ids.add(node_id)
                continue
            total[su_type] += 1
            source_ids_by_su[su_type].add(node_id)
    for raw_su, count in dict(spec.get("support_required_counts", {}) or {}).items():
        su_type = int(raw_su)
        total[su_type] += int(count)
        generic[su_type] += int(count)
    profile = dict(spec.get("raw_su24_profile", {}) or {})
    fixed_source_ids.update(
        int(value)
        for value in list(profile.get("terminal_node_ids", []) or [])
        if int(value) >= 0
    )
    fixed_source_ids.update(
        int(value)
        for value in list(profile.get("cap_node_ids", []) or [])
        if int(value) >= 0
    )
    return total, generic, source_ids_by_su, fixed_source_ids


def pool_ring_attempt(
    assembler,
    plan: RingCandidatePlan,
    *,
    product_kind: str,
    skeleton_kind: str,
) -> Dict[str, Any]:
    required: Counter = Counter()
    generic: Counter = Counter()
    source_ids_by_su: Dict[int, Set[int]] = defaultdict(set)
    fixed_source_ids: Set[int] = set()
    selected_su24_ids: Set[int] = set()
    for spec in list(plan.ordered_segments or []):
        spec_total, spec_generic, spec_sources, spec_fixed = (
            pool_ring_spec_requirements(assembler, dict(spec or {}))
        )
        required.update(spec_total)
        generic.update(spec_generic)
        fixed_source_ids.update(spec_fixed)
        for su_type, values in spec_sources.items():
            source_ids_by_su[int(su_type)].update(int(value) for value in values)
            if int(su_type) == 24:
                selected_su24_ids.update(int(value) for value in values)
    if str(skeleton_kind) == "fused":
        inner_spec = next(
            (
                dict(spec or {})
                for spec in list(plan.ordered_segments or [])
                if str(dict(spec or {}).get("role", "")) == "inner_core"
            ),
            {},
        )
        filler = ring_search_module.fused_inner_pool23_distribution(assembler, 
            int(inner_spec.get("backbone_span", 0) or 0)
        )
        if filler is not None:
            filler_count = int(sum(filler))
            required[23] += filler_count
            generic[23] += filler_count
    available = {
        int(key): int(value)
        for key, value in (
            (int(raw_key), raw_value)
            for raw_key, raw_value in assembler.allocator.available_counts().items()
            if str(raw_key).lstrip("-").isdigit()
        )
    }
    selected_specific_ids = {
        int(key): set(int(value) for value in list(values or []))
        for key, values in source_ids_by_su.items()
    }
    missing = {}
    for su_type, count in required.items():
        shortage = max(
            0,
            int(count) - int(available.get(int(su_type), 0)),
        )
        available_specific_ids = {
            int(value)
            for value in list(
                assembler.allocator.remaining_ids_by_su.get(str(int(su_type)), [])
                or []
            )
        }
        shortage = max(
            int(shortage),
            sum(
                1
                for node_id in selected_specific_ids.get(int(su_type), set())
                if int(node_id) not in available_specific_ids
            ),
        )
        if shortage > 0:
            missing[int(su_type)] = int(shortage)
    source_ids = sorted(
        fixed_source_ids
        | {
            int(value)
            for values in source_ids_by_su.values()
            for value in values
        }
    )
    protected_source_ids_by_su: Dict[int, List[int]] = defaultdict(list)
    for node_id in source_ids:
        node = assembler.nodes_by_id.get(int(node_id))
        if node is None:
            continue
        protected_source_ids_by_su[int(getattr(node, "su_type", -1))].append(
            int(node_id)
        )
    generic_sequence = [
        int(su_type)
        for su_type, count in sorted(generic.items())
        for _ in range(max(0, int(count)))
    ]
    protected_node_ids = sorted({
        int(value)
        for values in protected_source_ids_by_su.values()
        for value in values
    })
    plan.required_counts = {
        int(key): int(value) for key, value in required.items() if int(value) > 0
    }
    plan.available_counts = dict(available)
    plan.missing_counts = dict(missing)
    plan.selected_su24_ids = sorted(selected_su24_ids)
    plan.protected_node_ids = list(protected_node_ids)
    return {
        "product_kind": str(product_kind),
        "skeleton_kind": str(skeleton_kind),
        "ring_template": str(plan.skeleton_kind),
        "selected_su24_ids": sorted(selected_su24_ids),
        "selected_source_node_ids": source_ids,
        "source_ids_by_su": {
            int(key): sorted(values) for key, values in source_ids_by_su.items()
        },
        "protected_source_ids_by_su": {
            int(key): sorted(set(values))
            for key, values in protected_source_ids_by_su.items()
            if int(key) >= 0 and values
        },
        "protected_node_ids": list(protected_node_ids),
        "template_complete": bool(plan.template_complete),
        "required_counts": dict(plan.required_counts),
        "generic_required_counts": {
            int(key): int(value) for key, value in generic.items() if int(value) > 0
        },
        "generic_required_sequence": generic_sequence,
        "available_counts": dict(plan.available_counts),
        "missing_counts": dict(plan.missing_counts),
        "like24_count": int(planned_ring_like24_count(assembler, plan.ordered_segments)),
        "ordered_segments": [dict(item or {}) for item in list(plan.ordered_segments or [])],
    }


def enumerate_ring_plans(
    assembler,
    builder: Any,
    *,
    max_plans: int = 8,
    max_evaluations: int = 24,
    excluded_plan_signatures: Optional[Iterable[str]] = None,
) -> List[RingCandidatePlan]:
    """Enumerate deterministic structural alternatives without claiming nodes."""

    original_pool_exclusions = set(
        getattr(assembler, "_ring_plan_excluded_pool_node_ids", set()) or set()
    )
    original_candidate_exclusions = set(
        getattr(assembler, "_ring_plan_excluded_candidate_ids", set()) or set()
    )
    queued: deque[Tuple[frozenset[int], frozenset[str]]] = deque(
        [(frozenset(), frozenset())]
    )
    visited: Set[Tuple[frozenset[int], frozenset[str]]] = set()
    signatures: Set[str] = set()
    excluded_signatures = {
        str(value)
        for value in list(excluded_plan_signatures or [])
        if str(value)
    }
    plans: List[RingCandidatePlan] = []
    evaluation_count = 0
    try:
        while (
            queued
            and len(plans) < max(1, int(max_plans))
            and evaluation_count < max(1, int(max_evaluations))
        ):
            pool_exclusions, candidate_exclusions = queued.popleft()
            exclusion_key = (pool_exclusions, candidate_exclusions)
            if exclusion_key in visited:
                continue
            visited.add(exclusion_key)
            assembler._ring_plan_excluded_pool_node_ids = (
                set(original_pool_exclusions) | set(pool_exclusions)
            )
            assembler._ring_plan_excluded_candidate_ids = (
                set(original_candidate_exclusions) | set(candidate_exclusions)
            )
            evaluation_count += 1
            plan = builder()
            if not bool(plan.template_complete) or list(plan.missing_slots or []):
                continue
            signature = ring_plan_signature(plan)
            if signature not in signatures and signature not in excluded_signatures:
                signatures.add(signature)
                plans.append(plan)
            for candidate_id in sorted(ring_plan_candidate_ids(plan)):
                next_candidates = frozenset(
                    set(candidate_exclusions) | {str(candidate_id)}
                )
                next_key = (pool_exclusions, next_candidates)
                if next_key not in visited:
                    queued.append(next_key)
            for node_id in sorted(ring_plan_pool_su24_ids(plan)):
                next_pool = frozenset(set(pool_exclusions) | {int(node_id)})
                next_key = (next_pool, candidate_exclusions)
                if next_key not in visited:
                    queued.append(next_key)
    finally:
        assembler._ring_plan_excluded_pool_node_ids = original_pool_exclusions
        assembler._ring_plan_excluded_candidate_ids = original_candidate_exclusions
    return plans


def view_ring_allowed_skeletons(assembler, view: AssemblyFragmentView) -> Set[str]:
    contracts = {
        str(key): dict(value or {})
        for key, value in dict(dict(view.evidence or {}).get("ring_contracts", {}) or {}).items()
    }
    if contracts:
        out: Set[str] = set()
        for skeleton_kind, contract in contracts.items():
            if bool(contract.get("eligible", False)):
                out.add(_skeleton_allowed_name(str(skeleton_kind)))
        # Explicit contract data is authoritative, including an all-ineligible
        # set.  Falling back to a legacy flat field would resurrect a forbidden
        # ring skeleton.
        return out
    allowed = {
        str(item)
        for item in list(dict(view.evidence or {}).get("ring_allowed_skeletons", []) or [])
    }
    if allowed:
        return allowed
    return view_ring_eligibility(assembler, view)


def view_ring_eligibility(assembler, view: AssemblyFragmentView) -> Set[str]:
    contracts = {
        str(key): dict(value or {})
        for key, value in dict(dict(view.evidence or {}).get("ring_contracts", {}) or {}).items()
    }
    if contracts:
        out: Set[str] = set()
        for skeleton_kind, contract in contracts.items():
            if bool(contract.get("eligible", False)):
                out.add(_skeleton_allowed_name(str(skeleton_kind)))
        return out
    return {
        str(item)
        for item in list(dict(view.evidence or {}).get("ring_eligibility", []) or [])
    }


def is_like24_ring_fill_view(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        bool(getattr(view, "can_ring", False))
        and str(getattr(view, "family", "")) in {"like24", "like24_secondary", "like24_bridge"}
        and str(getattr(view, "preferred_product_type", "")) == "ring_candidate"
    )


def is_external_ring_view(assembler, view: AssemblyFragmentView) -> bool:
    return bool(
        bool(getattr(view, "can_ring", False))
        and not is_like24_ring_fill_view(assembler, view)
    )


def view_ring_slot_profiles(assembler, view: AssemblyFragmentView) -> Set[str]:
    return {
        str(item)
        for item in list(dict(view.evidence or {}).get("ring_slot_profiles", []) or [])
    }


def view_has_slot_profile(assembler, view: AssemblyFragmentView, *profiles: str) -> bool:
    if view is None:
        return False
    available = view_ring_slot_profiles(assembler, view)
    return any(str(profile) in available for profile in list(profiles or []))


def ring_resource_blockers(assembler) -> Dict[int, Dict[str, Any]]:
    available = {
        str(key): int(value)
        for key, value in dict(
            assembler.allocator.available_counts() or {}
        ).items()
    }
    blockers: Dict[int, Dict[str, Any]] = {}
    for fragment_id, raw_contract in dict(assembler.ring_infeasible_contracts or {}).items():
        view = getattr(assembler, "view_by_id", {}).get(int(fragment_id))
        # Mandatory rings and ring-first dual-use views both retain a concrete
        # resource blocker.  The latter may fall back to side/flex only after
        # the ring planner has no resource-backed candidate left to retry.
        if (
            view is not None
            and not ring_policy_module.view_requires_ring_resource_retry(view)
        ):
            continue
        raw_required = dict(dict(raw_contract or {}).get("required_counts", {}) or {})
        required = {
            str(key): int(value)
            for key, value in dict(raw_required or {}).items()
            if int(value) > 0
        }
        missing = {
            str(key): max(0, int(value) - int(available.get(str(key), 0)))
            for key, value in required.items()
            if int(value) > int(available.get(str(key), 0))
        }
        if not missing:
            continue
        attempt_state = dict(
            getattr(assembler, "ring_pending_attempts", {}).get(
                int(fragment_id), {}
            )
            or {}
        )
        attempt_reasons = [
            f"{str(key)}:{str(dict(value or {}).get('reason', '') or '')}"
            for key, value in sorted(
                dict(attempt_state.get("by_skeleton", {}) or {}).items()
            )
            if str(dict(value or {}).get("reason", "") or "")
        ]
        blockers[int(fragment_id)] = {
            "required_counts": required,
            "available_counts": dict(available),
            "missing_counts": missing,
            "reasons": [
                *attempt_reasons,
                *([
                    str(assembler.ring_pending_reasons.get(int(fragment_id), ""))
                ] if str(assembler.ring_pending_reasons.get(int(fragment_id), ""))
                and not attempt_reasons else []),
            ],
            "candidate_signature": str(dict(raw_contract or {}).get("candidate_signature", "")),
        }
    return blockers


def ring_materialization_blockers(assembler) -> Dict[int, Dict[str, Any]]:
    blockers: Dict[int, Dict[str, Any]] = {}
    for view in ownership_module.available_views(assembler, 
        lambda item: (
            bool(getattr(item, "can_ring", False))
            and view_requires_ring(item)
        )
    ):
        failure = dict(
            assembler.materialization_failure_by_fragment_id.get(
                int(view.fragment_id), {}
            )
            or {}
        )
        if not str(
            failure.get("attempted_product_kind", "") or ""
        ).startswith("fatty_ring_"):
            continue
        attempts = dict(
            getattr(assembler, "ring_pending_attempts", {}).get(
                int(view.fragment_id), {}
            )
            or {}
        )
        by_skeleton = {
            str(key)
            for key in dict(attempts.get("by_skeleton", {}) or {})
        }
        allowed_skeletons = {
            str(value)
            for value in view_ring_allowed_skeletons(assembler, view)
        }
        # A structural failure from one skeleton is not a terminal blocker
        # while another contract-defined skeleton has not been attempted.
        # In particular, fused/upper-lower diagnostics must not suppress a
        # legal side-ring route for the same 24_D fragment.
        if allowed_skeletons and not allowed_skeletons.issubset(by_skeleton):
            continue
        blockers[int(view.fragment_id)] = failure
    return blockers


def release_ring_view_to_chain_fallback(assembler, fragment_id: int) -> None:
    """Discard exhausted ring retry state without consuming the fragment."""
    fragment_id = int(fragment_id)
    if hasattr(assembler, "pending_units") and hasattr(
        assembler, "pending_contracts"
    ):
        ownership_module.invalidate_pending_identity(
            assembler,
            fragment_ids=[fragment_id],
            claimed_node_ids=[],
        )
    assembler.ring_exhausted_fragment_ids.add(fragment_id)
    assembler.ring_pending_reasons.pop(fragment_id, None)
    assembler.ring_pending_attempts.pop(fragment_id, None)
    assembler.ring_infeasible_contracts.pop(fragment_id, None)
    assembler.materialization_failure_by_fragment_id.pop(fragment_id, None)


def assemble_priority_rings(
    assembler,
    branch_donors: List[AssemblyFragmentView],
) -> bool:
    ring_views = ownership_module.available_views(assembler, lambda item: bool(getattr(item, "can_ring", False)))
    has_pool_su24_pair = len(
        list(assembler.allocator.remaining_ids_by_su.get("24", []) or [])
    ) >= 2
    if not ring_views and not has_pool_su24_pair:
        return False
    like24_fill_views = ownership_module.available_views(assembler, lambda item: is_like24_ring_fill_view(assembler, item))
    progress_made = assemble_all_rings(assembler, 
        branch_donors,
        ring_views=ring_views,
        like24_fill_views=like24_fill_views,
        candidate_predicate=lambda item: bool(getattr(item, "can_ring", False)),
    )
    pool_only_blocked = False
    available_ring_views = ownership_module.available_views(
        assembler,
        lambda item: bool(getattr(item, "can_ring", False)),
    )
    for view in available_ring_views:
        fragment_id = int(view.fragment_id)
        attempts = dict(
            getattr(assembler, "ring_pending_attempts", {}).get(
                fragment_id, {}
            )
            or {}
        )
        attempted_skeletons = {
            str(key)
            for key in dict(attempts.get("by_skeleton", {}) or {})
        }
        allowed_skeletons = {
            str(value)
            for value in view_ring_allowed_skeletons(assembler, view)
        }
        all_allowed_skeletons_attempted = bool(
            allowed_skeletons
            and allowed_skeletons.issubset(attempted_skeletons)
        )
        resource_contract = dict(
            assembler.ring_infeasible_contracts.get(fragment_id, {}) or {}
        )
        missing_counts = {
            int(key): int(value)
            for key, value in dict(
                resource_contract.get("missing_counts", {}) or {}
            ).items()
            if str(key).lstrip("-").isdigit() and int(value) > 0
        }
        has_resource_deficit = bool(
            dict(resource_contract.get("required_counts", {}) or {})
            and missing_counts
        )
        if (
            has_resource_deficit
            and ring_policy_module.view_requires_ring_resource_retry(view)
        ):
            available_counts = {
                str(key): int(value)
                for key, value in dict(
                    assembler.allocator.available_counts() or {}
                ).items()
            }
            unavailable_non_convertible = {
                int(su_type): int(missing_counts.get(int(su_type), 0))
                for su_type in (
                    ring_policy_module.NON_CONVERTIBLE_RING_RESOURCE_SU_TYPES
                )
                if int(missing_counts.get(int(su_type), 0)) > 0
                and int(available_counts.get(str(int(su_type)), 0)) <= 0
            }
            if (
                unavailable_non_convertible
                and all_allowed_skeletons_attempted
                and ring_policy_module.view_allows_non_ring_fallback(view)
            ):
                # No later Stage3 conversion can create the missing SU24.
                # Release only dual-use views; mandatory ring-only views
                # remain explicit resource blockers.
                release_ring_view_to_chain_fallback(assembler, fragment_id)
                continue
            pool_only_blocked = True
            continue

        # Do not consume a view merely because another candidate represented
        # the current skeleton.  It is exhausted only after every skeleton
        # allowed by its own contract has had a concrete attempt.
        if allowed_skeletons and not allowed_skeletons.issubset(attempted_skeletons):
            continue

        # A plan is allowed to use only a bounded representative prefix of
        # equivalent middle/anchor candidates.  The per-view resource check
        # above is authoritative: a shortage belonging to another fragment
        # must not keep this fully-probed view in ring-pending state.  That
        # global coupling previously left valid 24_D side fallbacks blocked
        # whenever any unrelated ring needed a conversion.
        release_ring_view_to_chain_fallback(assembler, fragment_id)

    # A non-mandatory classified ring view can fail to materialize while a
    # complete ring made only from pool resources is still feasible.  Run an
    # explicit pool-only pass after those views have been exhausted; otherwise
    # the candidate-view requirement in the materializer shadows every pure
    # resource-ring plan for the rest of this assembly round.
    has_pool_su24_pair = len(
        list(assembler.allocator.remaining_ids_by_su.get("24", []) or [])
    ) >= 2
    if has_pool_su24_pair and not pool_only_blocked:
        progress_made = bool(
            assemble_all_rings(
                assembler,
                branch_donors,
                ring_views=[],
                like24_fill_views=[],
                candidate_predicate=lambda _item: False,
                ring_order=[
                    "fatty_ring_fused",
                    "fatty_ring_side",
                    "fatty_ring_upper_lower",
                ],
            )
            or progress_made
        )
    return bool(progress_made)


def assemble_all_rings(
    assembler,
    branch_donors: List[AssemblyFragmentView],
    *,
    ring_views: Optional[Sequence[AssemblyFragmentView]] = None,
    like24_fill_views: Optional[Sequence[AssemblyFragmentView]] = None,
    candidate_predicate: Optional[Any] = None,
    ring_order: Optional[Sequence[str]] = None,
) -> bool:
    resolved_ring_order: List[str] = list(ring_order or [
        "fatty_ring_fused",
        "fatty_ring_side",
        "fatty_ring_upper_lower",
    ])
    predicate = candidate_predicate or (
        lambda view: is_external_ring_view(assembler, view)
    )
    progress_made = False
    while True:
        cycle_progress = False
        for product_kind in list(resolved_ring_order):
            source_views = (
                ownership_module.available_views(assembler, lambda item: True)
                if ring_views is None
                else list(ring_views)
            )
            candidate_ring_views = [
                view for view in source_views
                if ownership_module.fragment_available(assembler, int(view.fragment_id))
                and bool(predicate(view))
            ]
            has_pool_su24_pair = len(
                list(assembler.allocator.remaining_ids_by_su.get("24", []) or [])
            ) >= 2
            if not candidate_ring_views and not has_pool_su24_pair:
                continue
            product_count_before = len(assembler.products)
            created = ring_materializer_module.assemble_ring(assembler, 
                product_kind=str(product_kind),
                preferred_views=candidate_ring_views,
                branch_donors=branch_donors,
                like24_fill_views=like24_fill_views,
            )
            if created and len(assembler.products) > product_count_before:
                cycle_progress = True
                progress_made = True
                break
        if not cycle_progress:
            break
    return progress_made
