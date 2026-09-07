"""Transactional materialization and commit of selected ring plans."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...stage3_models import RingTopologyContract, TopologyAssemblyProduct
from ..closure import chain as chain_materializer_module
from ..state import ownership as ownership_module
from . import topology as ring_topology_module
from ..core.fragment import (
    AssemblyFragmentView,
    clean_ring_segment_text as _clean_ring_segment_text,
    render_token_sequence as _render_token_sequence,
)
from ..projection.engine import (
    _node_label,
    _normalize_ring_backbone_node_ids,
    _skeleton_allowed_name,
)
from .models import RingCandidatePlan, ResolvedRingSegment, ResolvedRingTopology
from . import planner as ring_planner_module
from . import search as ring_search_module
from . import segments as ring_segments_module

def resolve_ring_candidate_topology(
    assembler,
    plan: RingCandidatePlan,
    *,
    skeleton_kind: str,
    branch_donors: List[AssemblyFragmentView],
) -> ResolvedRingTopology:
    assembler.last_ring_materialization_reason = ""
    resolved_skeleton_kind = _skeleton_allowed_name(str(skeleton_kind))
    assembler.last_ring_missing_counts = {}
    # Callers may replace the allocator when retrying a candidate. Reapply
    # ownership filters before the transaction snapshot so filtered IDs are
    # not misreported as claims by claimed_ids_since().
    ownership_module.exclude_occupied_nodes_from_allocator(assembler)
    ownership_module.exclude_bound_su24_supports_from_allocator(assembler)
    ordered_specs = [
        dict(segment_spec or {})
        for segment_spec in list(plan.ordered_segments or [])
    ]
    resolved_by_index: Dict[int, ResolvedRingSegment] = {}
    candidate_specific_id_list = [
        int(node_id)
        for segment_spec in ordered_specs
        for node_id in ring_planner_module.ring_segment_specific_claim_id_list(segment_spec)
    ]
    duplicate_specific_ids = sorted({
        int(node_id)
        for node_id in candidate_specific_id_list
        if candidate_specific_id_list.count(int(node_id)) > 1
    })
    if duplicate_specific_ids:
        assembler.last_ring_materialization_reason = (
            "ring_candidate_specific_id_collision:"
            + ",".join(str(node_id) for node_id in duplicate_specific_ids)
        )
        raise ValueError(assembler.last_ring_materialization_reason)
    candidate_specific_ids = set(candidate_specific_id_list)
    with assembler.allocator.candidate_specific_claim_scope(candidate_specific_ids):
        for segment_index, segment_spec in sorted(
            enumerate(ordered_specs),
            key=lambda item: (
                ring_planner_module.ring_segment_claim_priority(dict(item[1] or {})),
                int(item[0]),
            ),
        ):
            segment = ring_segments_module.resolve_planned_ring_segment(assembler, 
                segment_spec,
                branch_donors,
                skeleton_kind=str(resolved_skeleton_kind),
            )
            if segment is None:
                reason = str(
                    assembler.last_ring_materialization_reason
                    or f"{str(skeleton_kind)}_ring_segment_resolution_failed"
                )
                raise ValueError(reason)
            if ring_search_module.segment_has_forbidden_ring_branch_mix(assembler, segment):
                assembler.last_ring_materialization_reason = "ring_forbidden_branch_mix"
                raise ValueError("ring_forbidden_branch_mix")
            resolved_by_index[int(segment_index)] = segment
    resolved_segments = [
        resolved_by_index[index]
        for index in range(len(ordered_specs))
    ]
    source_views = list({
        int(view.fragment_id): view
        for segment in list(resolved_segments)
        for view in list(segment.source_views or [])
    }.values())
    backbone_labels: List[str] = []
    backbone_node_ids: List[Optional[int]] = []
    topology_parts: List[str] = []
    node_id_parts: List[str] = []
    for segment in list(resolved_segments):
        backbone_labels.extend([str(value) for value in list(segment.ring_backbone_labels or []) if str(value)])
        backbone_node_ids.extend(_normalize_ring_backbone_node_ids(segment.ring_backbone_node_ids or []))
        segment_topology = _clean_ring_segment_text(segment.diagnostics.get("branch_render", ""))
        segment_node_ids = _clean_ring_segment_text(segment.diagnostics.get("branch_render_gid", ""))
        if not str(segment_topology) or not str(segment_node_ids):
            raise ValueError("resolved ring segment missing rendered topology")
        topology_parts.append(str(segment_topology))
        node_id_parts.append(str(segment_node_ids))
    return ResolvedRingTopology(
        source_views=source_views,
        labels=list(backbone_labels),
        node_ids=list(backbone_node_ids),
        backbone_labels=backbone_labels,
        backbone_node_ids=backbone_node_ids,
        topology_notation="-".join(str(value) for value in topology_parts if str(value)),
        node_id_notation="-".join(str(value) for value in node_id_parts if str(value)),
        support_node_ids=resolved_ring_support_node_ids(assembler, resolved_segments),
        pool_node_ids_by_su=chain_materializer_module.merge_pool_ids(segment.pool_node_ids_by_su for segment in resolved_segments
        ),
        fixed_source_node_ids=sorted({
            int(node_id)
            for segment in list(resolved_segments or [])
            for node_id in list(getattr(segment, "fixed_source_node_ids", []) or [])
        }),
        branch_paths=[
            dict(path or {})
            for segment in list(resolved_segments or [])
            for path in list(segment.branch_paths or [])
        ],
        external_attachments=[
            [int(value) for value in list(edge or [])]
            for segment in list(resolved_segments or [])
            for edge in list(segment.external_attachments or [])
        ],
        like24_node_ids=sorted({
            int(node_id)
            for segment in list(resolved_segments or [])
            for node_id in list(segment.like24_node_ids or [])
        }),
        node_profiles_by_id={
            int(key): str(value)
            for segment in list(resolved_segments or [])
            for key, value in dict(segment.node_profiles_by_id or {}).items()
        },
        replaced_branch_requirements=[
            dict(requirement or {})
            for segment in list(resolved_segments or [])
            for requirement in list(segment.replaced_branch_requirements or [])
        ],
    )


def resolved_ring_support_node_ids(
    assembler,
    resolved_segments: Sequence[ResolvedRingSegment],
) -> List[int]:
    physical_support_ids: Set[int] = set()
    for segment in list(resolved_segments or []):
        suppressed_ids = {
            int(node_id)
            for node_id in list(
                getattr(segment, "suppressed_support_node_ids", []) or []
            )
        }
        # Suppressed IDs also contain non-fixed hop context used only by the
        # degree audit.  Such neighbours are not physical members of the new
        # ring and must never enter its ownership contract.  A suppressed node
        # is physical here only when this candidate claimed it from the pool or
        # explicitly resolved it as a fixed source route.
        claimed_ids = {
            int(node_id)
            for values in dict(
                getattr(segment, "pool_node_ids_by_su", {}) or {}
            ).values()
            for node_id in list(values or [])
        }
        fixed_ids = {
            int(node_id)
            for node_id in list(
                getattr(segment, "fixed_source_node_ids", []) or []
            )
        }
        view = getattr(segment, "view", None)
        if view is not None:
            # A classified SU24 may carry a real fixed cap/terminal that is
            # intentionally absent from the rendered ring backbone.  Keep
            # only IDs explicitly owned by the view or marked fixed by its
            # classification evidence; ordinary context_node_ids are never
            # promoted to physical support by this fallback.
            view_evidence = dict(getattr(view, "evidence", {}) or {})
            fixed_ids.update(
                int(node_id)
                for node_id in list(getattr(view, "source_node_ids", []) or [])
            )
            for key in ("fixed_source_node_ids", "fixed_hop1_ids"):
                fixed_ids.update(
                    int(node_id)
                    for node_id in list(view_evidence.get(key, []) or [])
                )
        physical_support_ids.update(
            suppressed_ids & (claimed_ids | fixed_ids)
        )
    return sorted(physical_support_ids)


def build_ring_topology_contract(
    assembler,
    *,
    ring_kind: str,
    materialization: ResolvedRingTopology,
    cycles: Sequence[Sequence[int]],
    topology_notation: str,
    node_id_notation: str,
    role_node_ids: Optional[Dict[str, List[int]]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> RingTopologyContract:
    normalized_cycles = [
        [int(value) for value in list(cycle or [])]
        for cycle in list(cycles or [])
    ]
    branch_paths: List[Dict[str, Any]] = []
    branch_path_keys: Set[Tuple[int, Tuple[int, ...], str]] = set()
    for raw_path in list(materialization.branch_paths or []):
        path = {
            "anchor_node_id": int(dict(raw_path or {}).get("anchor_node_id", -1)),
            "node_ids": [
                int(value)
                for value in list(dict(raw_path or {}).get("node_ids", []) or [])
            ],
            "kind": str(dict(raw_path or {}).get("kind", "")),
            "state": str(dict(raw_path or {}).get("state", "FILLED")),
        }
        key = (
            int(path["anchor_node_id"]),
            tuple(int(value) for value in list(path["node_ids"] or [])),
            str(path["kind"]),
        )
        if key in branch_path_keys:
            continue
        branch_path_keys.add(key)
        branch_paths.append(path)
    attachments = [
        [int(value) for value in list(edge or [])]
        for edge in list(materialization.external_attachments or [])
    ]
    declared_node_ids: Set[int] = {
        int(value)
        for cycle in normalized_cycles
        for value in list(cycle or [])
    }
    for edge in attachments:
        declared_node_ids.update(int(value) for value in edge)
    for path in branch_paths:
        declared_node_ids.add(int(path["anchor_node_id"]))
        declared_node_ids.update(int(value) for value in list(path["node_ids"] or []))
    # Some classified SU24 supports are deliberately suppressed from the
    # rendered notation (for example a fixed 11 cap or 22 terminal).  They
    # are still physical members of this ring and must be owned by the ring
    # contract so that dismantling releases them as resources.  Include fixed
    # source IDs as well; a non-resource ID here makes the ring non-recyclable
    # through the physical-SU predicate below.
    support_node_ids = sorted({
        int(value)
        for value in list(materialization.support_node_ids or [])
        + list(materialization.fixed_source_node_ids or [])
        if int(value) >= 0
    })
    declared_node_ids.update(support_node_ids)
    claimed_resource_ids = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(materialization.pool_node_ids_by_su or {}).items()
        if list(values or [])
    }
    claimed_node_ids = {
        int(value)
        for values in claimed_resource_ids.values()
        for value in list(values or [])
    }
    source_fragment_ids = sorted({
        int(view.fragment_id)
        for view in list(materialization.source_views or [])
    })
    node_labels_by_id = {
        int(node_id): _node_label(assembler.nodes_by_id[int(node_id)])
        for node_id in sorted(declared_node_ids)
        if int(node_id) in assembler.nodes_by_id
    }
    node_su_by_id = {
        int(node_id): int(getattr(assembler.nodes_by_id[int(node_id)], "su_type", -1))
        for node_id in sorted(declared_node_ids)
        if int(node_id) in assembler.nodes_by_id
    }
    # A recyclable ring is identified from the actual physical node types,
    # rather than from whether it happened to use a classifier fragment.  A
    # fixed 11 cap/support inside an otherwise resource-only ring is therefore
    # released together with its 22/23/24 neighbours.
    recyclable_resource_ids_by_su: Dict[str, List[int]] = {}
    recyclable_su_types = {11, 22, 23, 24}
    for node_id, su_type in node_su_by_id.items():
        if int(su_type) in recyclable_su_types:
            recyclable_resource_ids_by_su.setdefault(str(int(su_type)), []).append(
                int(node_id)
            )
    recyclable_resource_ids_by_su = {
        str(key): sorted({int(value) for value in values})
        for key, values in recyclable_resource_ids_by_su.items()
        if values
    }
    requirements = [
        {
            "anchor_node_id": int(path["anchor_node_id"]),
            "kind": str(path["kind"]),
            "state": str(path["state"]),
            "node_ids": list(path["node_ids"]),
        }
        for path in branch_paths
    ]
    requirements.extend(
        dict(value or {})
        for value in list(materialization.replaced_branch_requirements or [])
    )
    signature_cycles = ":".join(
        ",".join(str(value) for value in cycle)
        for cycle in normalized_cycles
    )
    contract = RingTopologyContract(
        contract_id=f"ring:{str(ring_kind)}:{signature_cycles}",
        candidate_signature=f"{str(ring_kind)}:{signature_cycles}",
        ring_kind=str(ring_kind),
        cycles=normalized_cycles,
        node_labels_by_id=node_labels_by_id,
        node_su_by_id=node_su_by_id,
        node_profiles_by_id={
            int(key): str(value)
            for key, value in dict(materialization.node_profiles_by_id or {}).items()
            if int(key) in declared_node_ids
        },
        role_node_ids={
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(role_node_ids or {}).items()
        },
        external_attachments=attachments,
        branch_paths=branch_paths,
        branch_requirements=requirements,
        source_fragment_ids=source_fragment_ids,
        fixed_source_node_ids=sorted(declared_node_ids - claimed_node_ids),
        claimed_resource_ids_by_su=claimed_resource_ids,
        support_node_ids=support_node_ids,
        recyclable_resource_ids_by_su=recyclable_resource_ids_by_su,
        required_resource_counts={
            str(key): len(list(values or []))
            for key, values in claimed_resource_ids.items()
        },
        protected_node_ids=sorted(declared_node_ids),
        topology_notation=str(topology_notation),
        node_id_notation=str(node_id_notation),
        like24_node_ids=sorted({
            int(value) for value in list(materialization.like24_node_ids or [])
        }),
        bridgehead_kind=str(dict(diagnostics or {}).get("ring_bridgehead_kind", "")),
        diagnostics=dict(diagnostics or {}),
    )
    reason = ring_topology_module.audit_ring_topology_contract(
        contract,
        nodes=assembler.nodes,
    )
    contract.complete = reason is None
    contract.audit_reason = str(reason or "")
    return contract


def product_from_ring_topology_contract(
    assembler,
    *,
    product_kind: str,
    reason: str,
    materialization: ResolvedRingTopology,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[TopologyAssemblyProduct]:
    diagnostics_payload = {
        "generation_id": int(assembler.generation_id),
        "topology_notation": str(materialization.topology_notation),
        "node_id_notation": str(materialization.node_id_notation),
        "actual_topology_su": str(materialization.topology_notation),
        "actual_topology_gid": str(materialization.node_id_notation),
        "ring_backbone_labels": list(materialization.backbone_labels or []),
        "ring_backbone_node_ids": list(materialization.backbone_node_ids or []),
        **dict(diagnostics or {}),
    }
    ring_kind = str(diagnostics_payload.get("ring_template_kind", ""))
    if ring_kind == "fused":
        cycles = [
            [int(value) for value in list(diagnostics_payload.get("ring_outer_backbone_node_ids", []) or [])],
            [int(value) for value in list(diagnostics_payload.get("ring_inner_backbone_node_ids", []) or [])],
        ]
        role_node_ids = {
            "bridgeheads": [
                int(value)
                for value in list(diagnostics_payload.get("ring_bridgehead_ids", []) or [])
                if value is not None
            ]
        }
        topology_notation = str(materialization.topology_notation)
        node_id_notation = str(materialization.node_id_notation)
    else:
        cycle = [
            int(value)
            for value in list(materialization.backbone_node_ids or [])
            if value is not None
        ]
        cycles = [cycle]
        role_node_ids = (
            {"fixed_anchor": [int(cycle[0])]}
            if ring_kind == "upper_lower" and cycle
            else {}
        )
        topology_notation = str(materialization.topology_notation)
        node_id_notation = str(materialization.node_id_notation)
        if ring_kind == "upper_lower" and cycle:
            fixed_label = str(materialization.backbone_labels[0])
            topology_notation = f"{topology_notation}-@({fixed_label})"
            node_id_notation = f"{node_id_notation}-@({int(cycle[0])})"
    ring_contract = build_ring_topology_contract(assembler, 
        ring_kind=ring_kind,
        materialization=materialization,
        cycles=cycles,
        topology_notation=topology_notation,
        node_id_notation=node_id_notation,
        role_node_ids=role_node_ids,
        diagnostics=diagnostics_payload,
    )
    product = chain_materializer_module.make_product(assembler, 
        product_kind=str(product_kind),
        reason=str(reason),
        source_views=list(materialization.source_views or []),
        labels=list(materialization.labels or []),
        node_ids=list(materialization.node_ids or []),
        branch_map={},
        branch_id_map={},
        suppressed_support_node_ids=chain_materializer_module.collect_suppressed_support_node_ids(assembler, 
            materialization.source_views,
            extra_ids=materialization.support_node_ids,
        ),
        diagnostics=diagnostics_payload,
        ring_contract=ring_contract,
    )
    return product


def commit_single_cycle_ring_candidate(
    assembler,
    plan: RingCandidatePlan,
    branch_donors: List[AssemblyFragmentView],
    *,
    skeleton_kind: str,
    product_kind: str,
    reason: str,
    ring_template_kind: str,
    ring_has_fixed_anchor: bool,
    extra_diagnostics: Optional[Dict[str, Any]] = None,
) -> bool:
    if not bool(getattr(plan, "template_complete", False)) or list(plan.missing_slots or []):
        return False
    ownership_module.exclude_occupied_nodes_from_allocator(assembler)
    ownership_module.exclude_bound_su24_supports_from_allocator(assembler)
    snapshot = assembler.allocator.snapshot()
    assembler.last_ring_candidate_required_counts = {}
    assembler.last_ring_missing_counts = {}
    donor_snapshot = list(branch_donors)
    try:
        materialization = resolve_ring_candidate_topology(assembler, 
            plan,
            skeleton_kind=str(skeleton_kind),
            branch_donors=branch_donors,
        )
    except Exception as exc:
        claimed_counts = {
            str(key): len(list(values or []))
            for key, values in assembler.allocator.claimed_ids_since(snapshot).items()
            if list(values or [])
        }
        required = Counter(claimed_counts)
        required.update(Counter(assembler.last_ring_missing_counts))
        assembler.last_ring_candidate_required_counts = {
            str(key): int(value) for key, value in required.items() if int(value) > 0
        }
        assembler.last_ring_materialization_reason = str(exc or assembler.last_ring_materialization_reason or "")
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        return False
    like24_count = ring_planner_module.planned_ring_like24_count(assembler, plan.ordered_segments)
    if like24_count < int(plan.like24_count_min) or like24_count > int(plan.like24_count_max):
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        return False
    diagnostics = {
        "ring_like24_count": int(like24_count),
        "ring_template_kind": str(ring_template_kind),
        "ring_has_fixed_anchor": bool(ring_has_fixed_anchor),
        **dict(extra_diagnostics or {}),
    }
    product = product_from_ring_topology_contract(assembler, 
        product_kind=str(product_kind),
        reason=str(reason),
        materialization=materialization,
        diagnostics=diagnostics,
    )
    if product is None:
        assembler.last_ring_candidate_required_counts = {
            str(key): len(list(values or []))
            for key, values in assembler.allocator.claimed_ids_since(snapshot).items()
            if list(values or [])
        }
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        return False
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, [int(view.fragment_id) for view in list(materialization.source_views or [])])
    return True


def commit_fused_ring_candidate(assembler, plan: RingCandidatePlan, branch_donors: List[AssemblyFragmentView]) -> bool:
    if not bool(getattr(plan, "template_complete", False)):
        return False
    if list(plan.missing_slots or []):
        return False
    ownership_module.exclude_occupied_nodes_from_allocator(assembler)
    ownership_module.exclude_bound_su24_supports_from_allocator(assembler)
    spec_by_role = {
        str(item.get("role", "")): dict(item or {})
        for item in list(plan.ordered_segments or [])
    }
    left_anchor_spec = dict(spec_by_role.get("left_outer_anchor", {}) or {})
    left_bridge_spec = dict(spec_by_role.get("left_bridgehead", {}) or {})
    inner_spec = dict(spec_by_role.get("inner_core", {}) or {})
    right_bridge_spec = dict(spec_by_role.get("right_bridgehead", {}) or {})
    right_anchor_spec = dict(spec_by_role.get("right_outer_anchor", {}) or {})
    if not all([left_anchor_spec, left_bridge_spec, inner_spec, right_bridge_spec, right_anchor_spec]):
        return False
    snapshot = assembler.allocator.snapshot()
    donor_snapshot = list(branch_donors)
    assembler.last_ring_candidate_required_counts = {}
    assembler.last_ring_missing_counts = {}
    planned_required_counts = {
        str(key): int(value)
        for key, value in dict(ring_planner_module.pool_ring_attempt(assembler, 
            plan,
            product_kind="fatty_ring_fused",
            skeleton_kind="fused",
        ).get("required_counts", {}) or {}).items()
        if int(value) > 0
    }

    def _remember_required_counts() -> None:
        claimed_counts = {
            str(key): len(list(values or []))
            for key, values in assembler.allocator.claimed_ids_since(snapshot).items()
            if list(values or [])
        }
        missing_counts = {
            str(key): int(value)
            for key, value in dict(assembler.last_ring_missing_counts or {}).items()
            if int(value) > 0
        }
        required = dict(planned_required_counts)
        for key in set(claimed_counts) | set(missing_counts):
            required[str(key)] = max(
                int(required.get(str(key), 0)),
                int(claimed_counts.get(str(key), 0))
                + int(missing_counts.get(str(key), 0)),
            )
        assembler.last_ring_candidate_required_counts = {
            str(key): int(value)
            for key, value in required.items()
            if int(value) > 0
        }

    def _rollback_fused(reason: str) -> bool:
        _remember_required_counts()
        assembler.last_ring_materialization_reason = str(
            reason or assembler.last_ring_materialization_reason or "fused_ring_materialization_failed"
        )
        assembler.allocator.rollback(snapshot)
        branch_donors[:] = donor_snapshot
        return False

    left_anchor_segment = ring_segments_module.resolve_planned_ring_segment(assembler, 
        left_anchor_spec,
        branch_donors,
        skeleton_kind="fused",
    )
    left_bridge_segment = ring_segments_module.resolve_planned_ring_segment(assembler, 
        left_bridge_spec,
        branch_donors,
        skeleton_kind="fused",
    )
    inner_segment = ring_segments_module.resolve_planned_ring_segment(assembler, 
        inner_spec,
        branch_donors,
        skeleton_kind="fused",
    )
    pair_bridgehead_mode = bool(
        str(right_bridge_spec.get("shared_candidate_id", ""))
        and str(right_bridge_spec.get("shared_candidate_id", "")) == str(left_bridge_spec.get("candidate_id", ""))
    )
    right_bridge_segment = left_bridge_segment if pair_bridgehead_mode else ring_segments_module.resolve_planned_ring_segment(assembler, 
        right_bridge_spec,
        branch_donors,
        skeleton_kind="fused",
    )
    right_anchor_segment = ring_segments_module.resolve_planned_ring_segment(assembler, 
        right_anchor_spec,
        branch_donors,
        skeleton_kind="fused",
    )
    if left_anchor_segment is None or left_bridge_segment is None or inner_segment is None or right_bridge_segment is None or right_anchor_segment is None:
        return _rollback_fused("fused_ring_segment_resolution_failed")
    inner_core_span = int(len(list(inner_segment.ring_backbone_labels or [])))
    filler_distribution = ring_search_module.fused_inner_pool23_distribution(assembler, inner_core_span)
    if filler_distribution is None:
        return _rollback_fused("fused_ring_inner_span_invalid")
    left_filler_count, right_filler_count = filler_distribution
    filler_total = int(left_filler_count + right_filler_count)
    filler_ids = assembler.allocator.claim(23, filler_total) if filler_total > 0 else []
    if filler_total > 0 and filler_ids is None:
        available_23 = int(assembler.allocator.available_counts().get("23", 0) or 0)
        assembler.last_ring_missing_counts = {
            "23": max(1, int(filler_total) - int(available_23))
        }
        return _rollback_fused("fused_ring_inner_pool23_missing")
    filler_ids = [int(node_id) for node_id in list(filler_ids or [])]
    left_filler_ids = list(filler_ids[:left_filler_count])
    right_filler_ids = list(filler_ids[left_filler_count:])
    like24_count = ring_planner_module.planned_ring_like24_count(assembler, plan.ordered_segments)
    if like24_count < int(plan.like24_count_min) or like24_count > int(plan.like24_count_max):
        return _rollback_fused("fused_ring_like24_count_invalid")
    outer_left = _clean_ring_segment_text(left_anchor_segment.diagnostics.get("branch_render", ""))
    inner_text = _clean_ring_segment_text(inner_segment.diagnostics.get("branch_render", ""))
    outer_right = _clean_ring_segment_text(right_anchor_segment.diagnostics.get("branch_render", ""))
    outer_left_gid = _clean_ring_segment_text(left_anchor_segment.diagnostics.get("branch_render_gid", ""))
    inner_gid = _clean_ring_segment_text(inner_segment.diagnostics.get("branch_render_gid", ""))
    outer_right_gid = _clean_ring_segment_text(right_anchor_segment.diagnostics.get("branch_render_gid", ""))
    if pair_bridgehead_mode:
        bridge_tokens, bridge_gid_tokens = _render_token_sequence(
            left_bridge_segment.labels,
            left_bridge_segment.node_ids,
            branch_map=left_bridge_segment.branch_map,
            branch_id_map=left_bridge_segment.branch_id_map,
        )
        if len(bridge_tokens) < 2 or len(bridge_gid_tokens) < 2:
            return _rollback_fused("fused_ring_bridgehead_render_invalid")
        left_bridge_text = _clean_ring_segment_text(bridge_tokens[0])
        right_bridge_text = _clean_ring_segment_text(bridge_tokens[-1])
        left_bridge_gid = _clean_ring_segment_text(bridge_gid_tokens[0])
        right_bridge_gid = _clean_ring_segment_text(bridge_gid_tokens[-1])
        pair_bridge_labels = [str(value) for value in list(left_bridge_segment.ring_backbone_labels or []) if str(value)]
        pair_bridge_node_ids = _normalize_ring_backbone_node_ids(left_bridge_segment.ring_backbone_node_ids or [])
        if len(pair_bridge_labels) < 2 or len(pair_bridge_node_ids) < 2:
            return _rollback_fused("fused_ring_bridgehead_pair_invalid")
        left_bridge_labels = [str(pair_bridge_labels[0])]
        right_bridge_labels = [str(pair_bridge_labels[-1])]
        left_bridge_node_ids = [pair_bridge_node_ids[0]]
        right_bridge_node_ids = [pair_bridge_node_ids[-1]]
    else:
        left_bridge_text = _clean_ring_segment_text(left_bridge_segment.diagnostics.get("branch_render", ""))
        right_bridge_text = _clean_ring_segment_text(right_bridge_segment.diagnostics.get("branch_render", ""))
        left_bridge_gid = _clean_ring_segment_text(left_bridge_segment.diagnostics.get("branch_render_gid", ""))
        right_bridge_gid = _clean_ring_segment_text(right_bridge_segment.diagnostics.get("branch_render_gid", ""))
        left_bridge_labels = [str(value) for value in list(left_bridge_segment.ring_backbone_labels or []) if str(value)]
        right_bridge_labels = [str(value) for value in list(right_bridge_segment.ring_backbone_labels or []) if str(value)]
        left_bridge_node_ids = _normalize_ring_backbone_node_ids(left_bridge_segment.ring_backbone_node_ids or [])
        right_bridge_node_ids = _normalize_ring_backbone_node_ids(right_bridge_segment.ring_backbone_node_ids or [])
    if not left_bridge_text or not right_bridge_text:
        return _rollback_fused("fused_ring_bridgehead_render_missing")
    left_filler_labels = ["23"] * int(left_filler_count)
    right_filler_labels = ["23"] * int(right_filler_count)
    inner_loop_tokens = list(left_filler_labels)
    if inner_text:
        inner_loop_tokens.append(inner_text)
    inner_loop_tokens.extend(list(right_filler_labels))
    inner_loop_gid_tokens = [str(node_id) for node_id in list(left_filler_ids or [])]
    if inner_gid:
        inner_loop_gid_tokens.append(inner_gid)
    inner_loop_gid_tokens.extend(str(node_id) for node_id in list(right_filler_ids or []))
    if not inner_loop_tokens or not inner_loop_gid_tokens:
        return _rollback_fused("fused_ring_inner_path_missing")
    inner_loop_text = "-".join(str(token) for token in list(inner_loop_tokens or []) if str(token))
    inner_loop_gid = "-".join(str(token) for token in list(inner_loop_gid_tokens or []) if str(token))
    topology_notation = f"{outer_left}-{left_bridge_text}-[-{inner_loop_text}-]-{right_bridge_text}-{outer_right}"
    node_id_notation = f"{outer_left_gid}-{left_bridge_gid}-[-{inner_loop_gid}-]-{right_bridge_gid}-{outer_right_gid}"
    outer_backbone_labels = (
        list(left_anchor_segment.ring_backbone_labels)
        + list(left_bridge_labels)
        + [str(item) for item in list(right_bridge_labels or [])]
        + list(right_anchor_segment.ring_backbone_labels)
    )
    outer_backbone_node_ids = (
        list(left_anchor_segment.ring_backbone_node_ids)
        + _normalize_ring_backbone_node_ids(left_bridge_node_ids)
        + _normalize_ring_backbone_node_ids(right_bridge_node_ids)
        + list(right_anchor_segment.ring_backbone_node_ids)
    )
    inner_backbone_labels = (
        [str(left_bridge_labels[-1])] if list(left_bridge_labels or []) else []
    ) + list(left_filler_labels) + list(inner_segment.ring_backbone_labels) + list(right_filler_labels) + (
        [str(right_bridge_labels[0])] if list(right_bridge_labels or []) else []
    )
    inner_backbone_node_ids = (
        [left_bridge_node_ids[-1]] if list(left_bridge_node_ids or []) else []
    ) + list(left_filler_ids) + list(inner_segment.ring_backbone_node_ids) + list(right_filler_ids) + (
        [right_bridge_node_ids[0]] if list(right_bridge_node_ids or []) else []
    )
    source_views = list({
        int(view.fragment_id): view
        for view in list(left_anchor_segment.source_views)
        + list(left_bridge_segment.source_views)
        + list(inner_segment.source_views)
        + list(right_bridge_segment.source_views)
        + list(right_anchor_segment.source_views)
    }.values())
    materialization = ResolvedRingTopology(
        source_views=source_views,
        labels=list(outer_backbone_labels),
        node_ids=list(outer_backbone_node_ids),
        backbone_labels=outer_backbone_labels,
        backbone_node_ids=outer_backbone_node_ids,
        topology_notation=topology_notation,
        node_id_notation=node_id_notation,
        support_node_ids=resolved_ring_support_node_ids(assembler, [
            left_anchor_segment,
            left_bridge_segment,
            inner_segment,
            right_bridge_segment,
            right_anchor_segment,
        ]),
        pool_node_ids_by_su=chain_materializer_module.merge_pool_ids([
            left_anchor_segment.pool_node_ids_by_su,
            left_bridge_segment.pool_node_ids_by_su,
            inner_segment.pool_node_ids_by_su,
            right_bridge_segment.pool_node_ids_by_su,
            right_anchor_segment.pool_node_ids_by_su,
            {"23": list(filler_ids)},
        ]),
        fixed_source_node_ids=sorted({
            int(node_id)
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for node_id in list(getattr(segment, "fixed_source_node_ids", []) or [])
        }),
        branch_paths=[
            dict(path or {})
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for path in list(segment.branch_paths or [])
        ],
        external_attachments=[
            [int(value) for value in list(edge or [])]
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for edge in list(segment.external_attachments or [])
        ],
        like24_node_ids=sorted({
            int(node_id)
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for node_id in list(segment.like24_node_ids or [])
        }),
        node_profiles_by_id={
            int(key): str(value)
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for key, value in dict(segment.node_profiles_by_id or {}).items()
        },
        replaced_branch_requirements=[
            dict(requirement or {})
            for segment in [
                left_anchor_segment,
                left_bridge_segment,
                inner_segment,
                right_bridge_segment,
                right_anchor_segment,
            ]
            for requirement in list(segment.replaced_branch_requirements or [])
        ],
    )
    product = product_from_ring_topology_contract(assembler, 
        product_kind="fatty_ring_fused",
        reason="ring:fused_template",
        materialization=materialization,
        diagnostics={
            "ring_outer_backbone_labels": list(outer_backbone_labels or []),
            "ring_outer_backbone_node_ids": list(outer_backbone_node_ids or []),
            "ring_inner_backbone_labels": list(inner_backbone_labels or []),
            "ring_inner_backbone_node_ids": list(inner_backbone_node_ids or []),
            "ring_like24_count": int(like24_count),
            "ring_template_kind": "fused",
            "ring_inner_pool23_count": int(filler_total),
            "ring_inner_pool23_distribution": [int(left_filler_count), int(right_filler_count)],
            "ring_bridgehead_ids": [
                int(left_bridge_node_ids[-1]) if list(left_bridge_node_ids or []) and left_bridge_node_ids[-1] is not None else None,
                int(right_bridge_node_ids[0]) if list(right_bridge_node_ids or []) and right_bridge_node_ids[0] is not None else None,
            ],
            "ring_bridgehead_kind": (
                "substitute_14_pair" if pair_bridgehead_mode else "real_24c"
            ),
        },
    )
    if product is None:
        return _rollback_fused(
            assembler.last_ring_materialization_reason or "fused_ring_contract_audit_failed"
        )
    _remember_required_counts()
    if not ownership_module.commit_product(assembler, 
        product,
        allocator_snapshot=snapshot,
        branch_donors=branch_donors,
        donor_snapshot=donor_snapshot,
    ):
        return False
    ownership_module.drop_pending_overlapping_product(assembler, product)
    ownership_module.mark_fragments_completed(assembler, [int(view.fragment_id) for view in list(materialization.source_views or [])])
    return True


def assemble_ring(
    assembler,
    *,
    product_kind: str,
    preferred_views: Sequence[AssemblyFragmentView],
    branch_donors: List[AssemblyFragmentView],
    like24_fill_views: Optional[Sequence[AssemblyFragmentView]] = None,
) -> bool:
    plan: Optional[RingCandidatePlan] = None
    reason = "ring_slot_selection_failed"
    selected_required_counts: Dict[str, int] = {}
    selected_missing_counts: Dict[str, int] = {}
    skeleton_kind_for_attempt = {
        "fatty_ring_fused": "fused",
        "fatty_ring_upper_lower": "upper_lower",
        "fatty_ring_side": "side",
    }.get(str(product_kind), "")
    attempt_failures_by_fragment: Dict[int, Dict[str, Any]] = {}

    def _build_and_materialize(
        candidate_views: Sequence[AssemblyFragmentView],
        *,
        expanded: bool = False,
        excluded_plan_signatures: Optional[Iterable[str]] = None,
    ) -> Tuple[
        bool,
        Optional[RingCandidatePlan],
        str,
        Dict[str, int],
        Dict[str, int],
    ]:
        skeleton_kind = ""
        builder: Any
        if str(product_kind) == "fatty_ring_fused":
            skeleton_kind = "fused"
            builder = lambda: ring_planner_module.plan_fused_ring_templates(assembler, candidate_views)
        elif str(product_kind) == "fatty_ring_upper_lower":
            skeleton_kind = "upper_lower"
            builder = lambda: ring_planner_module.plan_upper_ring_templates(assembler, candidate_views)
        elif str(product_kind) == "fatty_ring_side":
            skeleton_kind = "side"
            builder = lambda: ring_planner_module.plan_side_ring_templates(assembler, candidate_views)
        else:
            return False, None, "unknown_ring_product_kind", {}, {}

        structural_probe = builder()
        identity_capacity = max(
            1,
            len(list(candidate_views or []))
            + len(
                list(
                    assembler.allocator.remaining_ids_by_su.get("24", [])
                    or []
                )
            ),
        )
        enumeration_plans = (
            max(32, min(256, int(identity_capacity) * 2))
            if expanded
            else 8
        )
        enumeration_evaluations = (
            max(96, min(2048, int(identity_capacity) * 8))
            if expanded
            else 24
        )
        enumerated_plans = ring_planner_module.enumerate_ring_plans(assembler, 
            builder,
            max_plans=int(enumeration_plans),
            max_evaluations=int(enumeration_evaluations),
            excluded_plan_signatures=excluded_plan_signatures,
        )
        complete_plans = [
            candidate
            for candidate in enumerated_plans
            if (
                ring_planner_module.ring_plan_contains_candidate_view(assembler, 
                    candidate, candidate_views
                )
                or (
                    not list(candidate_views or [])
                    and len(ring_planner_module.ring_plan_pool_su24_ids(candidate)) >= 2
                )
            )
        ]
        ranked_plans: List[Tuple[Tuple[Any, ...], RingCandidatePlan, Dict[str, Any]]] = []
        for candidate in complete_plans:
            attempt = ring_planner_module.pool_ring_attempt(assembler, 
                candidate,
                product_kind=str(product_kind),
                skeleton_kind=str(skeleton_kind),
            )
            missing = {
                str(key): int(value)
                for key, value in dict(attempt.get("missing_counts", {}) or {}).items()
                if int(value) > 0
            }
            ranked_plans.append((
                (
                    1 if missing else 0,
                    sum(missing.values()),
                    ring_planner_module.ring_plan_signature(candidate),
                ),
                candidate,
                attempt,
            ))
        ranked_plans.sort(key=lambda item: item[0])

        failures: List[Tuple[RingCandidatePlan, str, Dict[str, int], Dict[str, int]]] = []
        for _, candidate, attempt in ranked_plans:
            assembler.last_ring_candidate_required_counts = {}
            assembler.last_ring_missing_counts = {}
            assembler.last_ring_materialization_reason = ""
            if str(product_kind) == "fatty_ring_fused":
                created = commit_fused_ring_candidate(assembler, candidate, branch_donors)
            else:
                created = commit_single_cycle_ring_candidate(assembler, 
                    candidate,
                    branch_donors,
                    skeleton_kind=str(skeleton_kind),
                    product_kind=str(product_kind),
                    reason=f"ring:{str(skeleton_kind)}_template",
                    ring_template_kind=str(skeleton_kind),
                    ring_has_fixed_anchor=str(skeleton_kind) == "upper_lower",
                    extra_diagnostics=(
                        {
                            "closure_target_fragment_id": candidate.closure_target_fragment_id,
                            "closure_target_node_id": candidate.closure_target_node_id,
                        }
                        if str(skeleton_kind) == "upper_lower"
                        else None
                    ),
                )
            if created:
                return True, candidate, "", {}, {}

            required = {
                str(key): int(value)
                for key, value in dict(
                    assembler.last_ring_candidate_required_counts
                    or attempt.get("required_counts", {})
                    or {}
                ).items()
                if int(value) > 0
            }
            available = {
                str(key): int(value)
                for key, value in dict(
                    assembler.allocator.available_counts() or {}
                ).items()
            }
            missing = {
                str(key): int(value)
                for key, value in dict(
                    assembler.last_ring_missing_counts
                    or attempt.get("missing_counts", {})
                    or {}
                ).items()
                if int(value) > 0
            }
            for key, value in required.items():
                deficit = max(0, int(value) - int(available.get(str(key), 0)))
                if deficit > 0:
                    missing[str(key)] = max(
                        int(missing.get(str(key), 0)),
                        int(deficit),
                    )
            failure_reason = str(
                assembler.last_ring_materialization_reason
                or "ring_candidate_materialization_rejected"
            )
            candidate_signature = ring_planner_module.ring_plan_signature(candidate)
            candidate_failure = {
                "reason": str(failure_reason),
                "required_counts": dict(required),
                "missing_counts": dict(missing),
                "plan_complete": True,
                "candidate_signature": str(candidate_signature),
                "source_fragment_ids": sorted({
                    int(source_view.fragment_id)
                    for source_view in list(
                        getattr(candidate, "source_views", []) or []
                    )
                }),
            }
            for source_view in list(getattr(candidate, "source_views", []) or []):
                attempt_failures_by_fragment[int(source_view.fragment_id)] = dict(
                    candidate_failure
                )
            failures.append((candidate, failure_reason, required, missing))

        if failures:
            if not expanded:
                failed_signatures = {
                    ring_planner_module.ring_plan_signature(item[0])
                    for item in failures
                }
                return _build_and_materialize(
                    candidate_views,
                    expanded=True,
                    excluded_plan_signatures=failed_signatures,
                )
            selected = min(
                failures,
                key=lambda item: (
                    1 if dict(item[3] or {}) else 0,
                    sum(int(value) for value in dict(item[3] or {}).values()),
                    ring_planner_module.ring_plan_signature(item[0]),
                ),
            )
            return False, selected[0], selected[1], selected[2], selected[3]
        structural_reason = str(
            (list(structural_probe.missing_slots or []) or [
                "external_ring_requires_external_source"
                if bool(structural_probe.template_complete)
                else "ring_slot_selection_failed"
            ])[0]
        )
        for candidate_view in list(candidate_views or []):
            attempt_failures_by_fragment.setdefault(int(candidate_view.fragment_id), {
                "reason": str(structural_reason),
                "required_counts": {},
                "missing_counts": {},
                "plan_complete": False,
                "candidate_signature": "",
            })
        return False, structural_probe, structural_reason, {}, {}

    # Classified like24 fragments may supplement an external ring
    # candidate, but are never created from resource nodes.  Enumerate the
    # combined set once so a failed preferred-only plan is not retried
    # before its fill alternatives.
    fill_views = [
        view
        for view in list(like24_fill_views or [])
        if ownership_module.fragment_available(assembler, int(view.fragment_id))
        and str(getattr(view, "class_type", "")) in {"24_A", "24_B", "24_C", "24_D"}
    ]
    preferred_ids = {
        int(view.fragment_id) for view in list(preferred_views or [])
    }
    candidate_views = list(preferred_views or []) + [
        view
        for view in fill_views
        if int(view.fragment_id) not in preferred_ids
    ]
    (
        created,
        plan,
        reason,
        selected_required_counts,
        selected_missing_counts,
    ) = _build_and_materialize(candidate_views)
    if created:
        return True

    skeleton_name = {
        "fatty_ring_fused": "fused_ring_plan",
        "fatty_ring_upper_lower": "upper_lower_ring_plan",
        "fatty_ring_side": "side_ring_plan",
    }.get(str(product_kind), "")
    attempted_view_ids = {
        int(fragment_id) for fragment_id in attempt_failures_by_fragment
    }
    if attempted_view_ids:
        # The planner may test many alternatives before returning one
        # representative failure.  Preserve every fragment that participated
        # in a real attempt so unselected alternatives are not exhausted.
        relevant_views = [
            view for view in list(candidate_views or [])
            if int(view.fragment_id) in attempted_view_ids
        ]
    elif plan is not None and list(plan.source_views or []):
        relevant_views = list(plan.source_views or [])
    else:
        allowed_name = _skeleton_allowed_name(str(skeleton_name))
        relevant_views = [
            view for view in list(preferred_views or [])
            if str(allowed_name) in ring_planner_module.view_ring_allowed_skeletons(assembler, view)
        ]
    for view in list(relevant_views or []):
        if not ownership_module.fragment_available(assembler, int(view.fragment_id)):
            continue
        view_failure = dict(
            attempt_failures_by_fragment.get(int(view.fragment_id), {}) or {}
        )
        view_reason = str(view_failure.get("reason", reason) or reason)
        view_required_counts = {
            str(key): int(value)
            for key, value in dict(
                view_failure.get("required_counts", selected_required_counts)
                or {}
            ).items()
            if int(value) > 0
        }
        view_missing_counts = {
            str(key): int(value)
            for key, value in dict(
                view_failure.get("missing_counts", selected_missing_counts)
                or {}
            ).items()
            if int(value) > 0
        }
        view_plan_complete = bool(
            view_failure.get(
                "plan_complete",
                plan is not None
                and getattr(plan, "template_complete", False)
                and not list(getattr(plan, "missing_slots", []) or []),
            )
        )
        view_candidate_signature = str(
            view_failure.get("candidate_signature", "") or ""
        )
        view_source_fragment_ids = [
            int(value)
            for value in list(view_failure.get("source_fragment_ids", []) or [])
        ]
        missing_slots = sorted({
            str(item)
            for item in ([str(view_reason)] if str(view_reason) else [])
            + (list(plan.missing_slots or []) if plan is not None else [])
        })
        plan_complete = bool(view_plan_complete)
        selected_required_counts = dict(view_required_counts)
        selected_missing_counts = dict(view_missing_counts)
        required_counts_payload = (
            dict(selected_required_counts or {}) if plan_complete else {}
        )
        if plan_complete and not required_counts_payload and plan is not None:
            for segment in list(plan.ordered_segments or []):
                for key, value in dict(segment.get("support_required_counts", {}) or {}).items():
                    key_s = str(key)
                    required_counts_payload[key_s] = int(required_counts_payload.get(key_s, 0)) + int(value)
        # A missing anchor/slot only rejects this skeleton.  It is not a
        # materialization failure and must not create a resource-backed
        # contract from the fragment's generic F1/H slots.
        if plan_complete and selected_missing_counts:
            assembler.ring_infeasible_contracts[int(view.fragment_id)] = {
                "candidate_signature": str(
                    view_candidate_signature
                    or ring_planner_module.ring_plan_signature(plan)
                ),
                "product_kind": str(product_kind),
                "source_fragment_ids": sorted({
                    int(value)
                    for value in (
                        view_source_fragment_ids
                        or [
                            int(source.fragment_id)
                            for source in list(
                                getattr(plan, "source_views", []) or []
                            )
                        ]
                    )
                }),
                "required_counts": {
                    str(key): int(value)
                    for key, value in required_counts_payload.items()
                    if int(value) > 0
                },
                "missing_counts": {
                    str(key): int(value)
                    for key, value in selected_missing_counts.items()
                    if int(value) > 0
                },
                "failure_reason": str(view_reason),
            }

        fragment_id = int(view.fragment_id)
        # A fragment can advertise more than one legal ring skeleton.  Keep
        # the outcome of each skeleton independently; a later probe (for
        # example upper/lower) must not overwrite the fact that side/fused
        # still needs a retry or is a valid fallback.  The flat fields remain
        # for old consumers and intentionally expose the latest attempt only.
        attempt_state = dict(
            assembler.ring_pending_attempts.get(fragment_id, {}) or {}
        )
        by_skeleton = {
            str(key): dict(value or {})
            for key, value in dict(attempt_state.get("by_skeleton", {}) or {}).items()
        }
        skeleton_attempt = {
            "skeleton": str(skeleton_kind_for_attempt),
            "product_kind": str(product_kind),
            "reason": str(view_reason),
            "missing_slots": list(missing_slots),
            "plan_complete": bool(plan_complete),
            "required_counts": {
                str(key): int(value)
                for key, value in dict(required_counts_payload or {}).items()
                if int(value) > 0
            },
            "missing_counts": {
                str(key): int(value)
                for key, value in dict(selected_missing_counts or {}).items()
                if int(value) > 0
            },
            "candidate_signature": (
                str(view_candidate_signature)
                or ring_planner_module.ring_plan_signature(plan)
                if plan is not None and hasattr(plan, "ordered_segments")
                else str(view_candidate_signature)
            ),
        }
        by_skeleton[str(skeleton_kind_for_attempt)] = skeleton_attempt
        all_missing_slots = sorted({
            str(value)
            for item in by_skeleton.values()
            for value in list(item.get("missing_slots", []) or [])
            if str(value)
        })
        attempt_state.update({
            "missing_slots": all_missing_slots,
            "by_skeleton": by_skeleton,
            "last_skeleton": str(skeleton_kind_for_attempt),
        })
        assembler.ring_pending_attempts[fragment_id] = attempt_state
        allowed_skeletons = {
            str(value)
            for value in ring_planner_module.view_ring_allowed_skeletons(
                assembler, view
            )
        }
        attempted_skeletons = set(by_skeleton)
        if allowed_skeletons and not allowed_skeletons.issubset(attempted_skeletons):
            # No aggregate structural blocker exists yet.  Keep the fragment
            # visible to the next allowed skeleton, and retain the latest
            # concrete reason only inside ``by_skeleton``.
            assembler.ring_pending_reasons.pop(fragment_id, None)
            attempt_state["untried_skeletons"] = sorted(
                allowed_skeletons - attempted_skeletons
            )
            attempt_state["all_allowed_skeletons_attempted"] = False
        else:
            attempt_state["untried_skeletons"] = []
            attempt_state["all_allowed_skeletons_attempted"] = True
            assembler.ring_pending_reasons[fragment_id] = str(view_reason)

        existing_failure = dict(
            assembler.materialization_failure_by_fragment_id.get(fragment_id, {})
            or {}
        )
        historical_attempts = {
            str(key): dict(value or {})
            for key, value in dict(existing_failure.get("attempts_by_skeleton", {}) or {}).items()
        }
        historical_attempts[str(skeleton_kind_for_attempt)] = dict(skeleton_attempt)
        # Keep one compatibility payload at the top level, while making the
        # complete per-skeleton failure map available to diagnostics and
        # pending-contract replay.
        assembler.materialization_failure_by_fragment_id[fragment_id] = {
            **existing_failure,
            "attempted_product_kind": str(product_kind),
            "attempted_skeleton": str(skeleton_kind_for_attempt),
            "validation_reason": str(view_reason),
            "attempted_resource_counts": {
                str(key): int(value)
                for key, value in dict(required_counts_payload or {}).items()
                if int(value) > 0
            },
            "attempted_missing_counts": {
                str(key): int(value)
                for key, value in dict(selected_missing_counts or {}).items()
                if int(value) > 0
            },
            "attempts_by_skeleton": historical_attempts,
        }
    return False
