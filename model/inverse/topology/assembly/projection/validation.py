"""Coverage and connectivity checks for projected topology fragments."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .....shared.inverse_common import _NodeV3
from ...classification.classification_models import TopologyFragment
from ...classification.classification_tokens import normalize_subtype
from ...fixed_edges import nodes_have_fixed_edge as _nodes_have_fixed_edge
from ...resources.types import PHYSICAL_TOPOLOGY_RESOURCE_TYPES
from ..core.fragment import AssemblyFragmentView
from ..state.iteration_ledger import normalize_slot
from ..core.models import BranchStub


def fragment_atomic_groups(fragment: TopologyFragment) -> List[Dict[str, Any]]:
    return [
        dict(group or {})
        for group in list(
            dict(fragment.evidence or {}).get("atomic_groups", []) or []
        )
        if dict(group or {}).get("node_ids")
    ]


def projection_contract_node_ids(contract: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    for key in (
        "connector_node_id",
        "center_node_id",
        "anchor_node_id",
        "fixed_source_node_id",
    ):
        value = dict(contract or {}).get(key)
        if (
            value is not None
            and str(value).lstrip("-").isdigit()
            and int(value) >= 0
        ):
            out.add(int(value))
    for key in (
        "base_path_node_ids",
        "remaining_node_ids",
        "existing_node_ids",
        "fixed_source_node_ids",
        "source_node_ids",
        "node_ids",
    ):
        out.update(
            int(value)
            for value in list(dict(contract or {}).get(key, []) or [])
            if str(value).lstrip("-").isdigit() and int(value) >= 0
        )
    return out


def view_render_route_node_ids(view: AssemblyFragmentView) -> Set[int]:
    routed = {
        int(value)
        for value in list(view.main_node_ids or [])
        if value is not None
    }
    stub_groups = [list(view.actual_branch_stubs or [])]
    for stub_map in (
        view.branch_stubs_by_anchor,
        view.real_branch_stubs_by_anchor,
    ):
        stub_groups.extend(list(dict(stub_map or {}).values()))
    for stubs in stub_groups:
        for stub in list(stubs or []):
            routed.update(
                int(value) for value in list(stub.source_node_ids or [])
            )
    for contract in list(view.fixed_connection_contracts or []):
        routed.update(projection_contract_node_ids(dict(contract or {})))
    for contracts in dict(
        view.branch_closure_contracts_by_anchor or {}
    ).values():
        for contract in list(contracts or []):
            routed.update(projection_contract_node_ids(dict(contract or {})))
    for node_ids in dict(view.ring_backbone_node_ids or {}).values():
        routed.update(
            int(value) for value in list(node_ids or []) if value is not None
        )
    if view.resolved_fixed_mainline_connector_id is not None:
        routed.add(int(view.resolved_fixed_mainline_connector_id))
    routed.update(
        int(value)
        for value in list(
            dict(view.diagnostics or {}).get(
                "real_mainline_support_node_ids", []
            )
            or []
        )
    )
    return routed


def foreign_fixed_node_ids_for_projection(
    fragment: TopologyFragment,
    view: AssemblyFragmentView,
    *,
    fixed_node_owner_by_id: Optional[Dict[int, int]] = None,
    nodes_by_id: Optional[Dict[int, _NodeV3]] = None,
) -> Dict[int, int]:
    """Return fixed nodes rendered by ``view`` but owned by another fragment.

    Generic SU11/SU22/SU23/SU24 nodes without a classification owner are
    allocator resources and are intentionally allowed as projection context.
    Once classification assigns an owner, however, the node is fixed physical
    topology and cannot be silently pulled into a different fragment's route.
    This check belongs at projection time; waiting for product commit leaves a
    malformed pending contract behind and obscures the real source fragment.
    """

    owner_map = {
        int(node_id): int(owner)
        for node_id, owner in dict(fixed_node_owner_by_id or {}).items()
    }
    if not owner_map:
        return {}
    fragment_id = int(fragment.fragment_id)
    routed = view_render_route_node_ids(view)
    claimed = {
        int(value)
        for value in list(
            fragment.claimed_node_ids or fragment.source_node_ids or []
        )
    }
    context = {
        int(value) for value in list(fragment.context_node_ids or [])
    }
    foreign: Dict[int, int] = {}
    for node_id in sorted(routed | claimed):
        owner = owner_map.get(int(node_id))
        if owner is None or int(owner) == fragment_id:
            continue
        # A resource node may appear as unowned context and is rebound by the
        # allocator later.  An owned resource node is still fixed and must not
        # cross fragment boundaries.
        node = dict(nodes_by_id or {}).get(int(node_id))
        su_type = int(getattr(node, "su_type", -1)) if node is not None else -1
        if int(node_id) in context and su_type in set(
            PHYSICAL_TOPOLOGY_RESOURCE_TYPES
        ):
            # The owner map is authoritative: this branch is only reachable
            # for legacy snapshots that retained a stale context owner.
            # Keep it visible as a conflict rather than silently rebinding it.
            pass
        foreign[int(node_id)] = int(owner)
    return foreign


def canonical_view_render_node_ids(view: AssemblyFragmentView) -> Set[int]:
    """Return physical nodes present in the view's actual render."""

    rendered: Set[int] = set()
    try:
        render = view.real_render()
    except Exception:
        render = None
    if render is not None:
        rendered.update(
            int(value)
            for value in re.findall(
                r"\d+", str(getattr(render, "node_id_notation", "") or "")
            )
        )
        rendered.update(
            int(value)
            for value in re.findall(
                r"\d+",
                str(getattr(render, "main_node_id_notation", "") or ""),
            )
        )
        for text in list(
            getattr(render, "branch_node_id_notations", []) or []
        ):
            rendered.update(
                int(value) for value in re.findall(r"\d+", str(text or ""))
            )
    rendered.update(
        int(value)
        for value in list(getattr(view, "main_node_ids", []) or [])
        if value is not None and str(value).lstrip("-").isdigit()
        and int(value) >= 0
    )
    for stubs in list(
        getattr(view, "real_branch_stubs_map", lambda: {})().values()
    ):
        for stub in list(stubs or []):
            rendered.update(
                int(value)
                for value in list(getattr(stub, "source_node_ids", []) or [])
            )
    return rendered


def _node_label(node: _NodeV3) -> str:
    subtype = str(normalize_subtype(node))
    try:
        su_type = int(getattr(node, "su_type", -1))
    except (TypeError, ValueError):
        su_type = -1
    return subtype if subtype not in {str(su_type), ""} else str(su_type)


def route_unrendered_claimed_fixed_components(
    fragment: TopologyFragment,
    view: AssemblyFragmentView,
    nodes_by_id: Dict[int, _NodeV3],
) -> None:
    """Route only a complete, unambiguous linear fixed side component."""

    claimed = {
        int(value)
        for value in list(
            fragment.claimed_node_ids or fragment.source_node_ids or []
        )
    }
    suppressed = {
        int(value) for value in list(view.suppressed_support_node_ids or [])
    }
    missing = claimed - view_render_route_node_ids(view) - suppressed
    if not missing:
        return
    anchor_by_node_id = {
        int(node_id): int(index)
        for index, node_id in enumerate(list(view.main_node_ids or []))
        if node_id is not None and int(node_id) in nodes_by_id
    }
    if not anchor_by_node_id:
        return

    missing_adjacency: Dict[int, Set[int]] = {
        int(node_id): set() for node_id in missing
    }
    roots_by_node_id: Dict[int, List[int]] = {}
    for node_id in sorted(missing):
        node = nodes_by_id.get(int(node_id))
        if node is None:
            continue
        for other_id in sorted(missing):
            if int(other_id) <= int(node_id):
                continue
            other = nodes_by_id.get(int(other_id))
            if other is not None and _nodes_have_fixed_edge(node, other):
                missing_adjacency[int(node_id)].add(int(other_id))
                missing_adjacency[int(other_id)].add(int(node_id))
        roots_by_node_id[int(node_id)] = sorted(
            int(anchor)
            for anchor in anchor_by_node_id
            if _nodes_have_fixed_edge(node, nodes_by_id[int(anchor)])
        )

    visited: Set[int] = set()
    routed_components: List[Dict[str, Any]] = []
    for seed_id in sorted(missing):
        if int(seed_id) in visited:
            continue
        component: Set[int] = set()
        pending = [int(seed_id)]
        while pending:
            current = int(pending.pop())
            if current in component:
                continue
            component.add(current)
            pending.extend(
                sorted(missing_adjacency.get(current, set()) - component)
            )
        visited.update(component)
        root_candidates = sorted(
            (
                int(anchor_by_node_id[int(anchor_id)]),
                int(node_id),
                int(anchor_id),
            )
            for node_id in component
            for anchor_id in roots_by_node_id.get(int(node_id), [])
        )
        if len(root_candidates) != 1:
            continue
        anchor_index, root_id, anchor_id = root_candidates[0]

        ordered_ids: List[int] = []
        previous = None
        current = int(root_id)
        linear = True
        while True:
            ordered_ids.append(int(current))
            next_ids = sorted(
                int(value)
                for value in missing_adjacency.get(int(current), set())
                if int(value) != previous and int(value) not in ordered_ids
            )
            if not next_ids:
                break
            if len(next_ids) != 1:
                linear = False
                break
            previous, current = int(current), int(next_ids[0])
        if not linear or set(ordered_ids) != component:
            continue

        stub = BranchStub(
            topology_notation="-"
            + "-".join(
                _node_label(nodes_by_id[int(node_id)])
                for node_id in ordered_ids
            ),
            node_id_notation="-"
            + "-".join(str(int(node_id)) for node_id in ordered_ids),
            source_node_ids=[int(node_id) for node_id in ordered_ids],
            branch_kind="actual_claimed_fixed_component",
            mode="claimed_fixed_component_projection",
        )
        view.branch_stubs_by_anchor.setdefault(int(anchor_index), []).append(stub)
        view.real_branch_stubs_by_anchor.setdefault(int(anchor_index), []).append(
            stub
        )
        if int(anchor_index) == int(view.branch_anchor_index):
            view.actual_branch_stubs.append(stub)
        routed_components.append({
            "anchor_node_id": int(anchor_id),
            "anchor_index": int(anchor_index),
            "node_ids": [int(node_id) for node_id in ordered_ids],
        })
    if routed_components:
        view.diagnostics[
            "auto_routed_claimed_fixed_components"
        ] = routed_components


def validate_fragment_view_projection(
    fragment: TopologyFragment,
    view: AssemblyFragmentView,
    *,
    fixed_node_owner_by_id: Optional[Dict[int, int]] = None,
    nodes_by_id: Optional[Dict[int, _NodeV3]] = None,
) -> None:
    claimed = {
        int(value)
        for value in list(
            fragment.claimed_node_ids or fragment.source_node_ids or []
        )
    }
    context = {int(value) for value in list(fragment.context_node_ids or [])}
    suppressed = {
        int(value) for value in list(view.suppressed_support_node_ids or [])
    }
    routed = view_render_route_node_ids(view)
    errors: List[str] = []
    foreign_fixed = foreign_fixed_node_ids_for_projection(
        fragment,
        view,
        fixed_node_owner_by_id=fixed_node_owner_by_id,
        nodes_by_id=nodes_by_id,
    )
    if foreign_fixed:
        errors.append(
            "foreign_fixed_node_owner:"
            + str(
                [
                    [int(node_id), int(owner)]
                    for node_id, owner in sorted(foreign_fixed.items())
                ]
            )
        )
    overlap = sorted(claimed & context)
    if overlap:
        errors.append(f"claimed_context_overlap:{overlap}")
    for group in fragment_atomic_groups(fragment):
        group_ids = {
            int(value) for value in list(group.get("node_ids", []) or [])
        }
        owned_ids = group_ids & claimed
        if owned_ids and owned_ids != group_ids:
            errors.append(
                f"atomic_group_split:{int(group.get('group_id', min(group_ids)))}:"
                f"owned={sorted(owned_ids)}:group={sorted(group_ids)}"
            )
        suppressed_atomic = sorted(group_ids & suppressed)
        if suppressed_atomic:
            errors.append(f"atomic_group_suppressed:{suppressed_atomic}")
    missing_claimed = sorted(claimed - routed - suppressed)
    if missing_claimed:
        errors.append(f"claimed_node_without_render_route:{missing_claimed}")
    for raw_slot in list(fragment.open_mainline_slots or []) + list(
        fragment.branch_contracts or []
    ):
        anchor_id = int(normalize_slot(raw_slot).anchor_node_id)
        if anchor_id in claimed and anchor_id not in routed:
            errors.append(f"slot_anchor_without_render_route:{anchor_id}")
    if not errors:
        return
    route_details = {
        "family": str(fragment.family),
        "class_type": str(fragment.class_type),
        "claimed": sorted(claimed),
        "routed": sorted(routed),
        "suppressed": sorted(suppressed),
        "main": [
            None if value is None else int(value)
            for value in list(view.main_node_ids or [])
        ],
        "foreign_fixed_node_owner": {
            str(int(node_id)): int(owner)
            for node_id, owner in sorted(foreign_fixed.items())
        },
        "raw_su24_role": str(
            dict(fragment.evidence or {}).get("raw_su24_role", "") or ""
        ),
        "raw_su24_profile": dict(
            dict(fragment.evidence or {}).get("raw_su24_profile", {}) or {}
        ),
    }
    raise ValueError(
        f"fragment_projection_invalid:{int(fragment.fragment_id)}:"
        + "|".join(errors)
        + f":details={route_details}"
    )
