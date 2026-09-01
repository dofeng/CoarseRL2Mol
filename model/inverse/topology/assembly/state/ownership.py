"""Authoritative fixed-node ownership for topology assembly."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...resources.types import (
    EXTERNAL_ADJUSTMENT_RESOURCE_TYPES,
    PHYSICAL_TOPOLOGY_RESOURCE_TYPES,
)
from ...stage3_models import (
    TopologyAssemblyProduct,
    TopologyPendingUnit,
    TopologyPoolLedger,
)
from ..ring import topology as ring_topology_module
from ..core.fragment import AssemblyFragmentView
from ..core.models import BranchClosureResult
from ..projection.engine import _normalize_su_key, _safe_int
from ..projection.validation import (
    canonical_view_render_node_ids as _canonical_view_render_node_ids,
)
from ..core.validation import branch_route_shape_reason as _branch_route_shape_reason


RECYCLABLE_PURE_RESOURCE_SU_TYPES = frozenset({11, 22, 23, 24})


def _record_invalid_topology_attempt(assembler: Any, **kwargs: Any) -> None:
    # Imported lazily because chain materialization commits through this module.
    from ..closure.chain import record_invalid_topology_attempt

    record_invalid_topology_attempt(assembler, **kwargs)


@dataclass
class OwnershipRegistry:
    """One node-to-fragment map shared by every assembly subsystem."""

    node_owner_by_id: Dict[int, int] = field(default_factory=dict)
    classification_conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_classification(cls, classification: Any) -> "OwnershipRegistry":
        raw_owners = getattr(classification, "fixed_node_owner_by_id", None)
        if raw_owners is None and isinstance(classification, dict):
            raw_owners = classification.get("fixed_node_owner_by_id", {})
        owners = {
            int(node_id): int(fragment_id)
            for node_id, fragment_id in dict(raw_owners or {}).items()
        }
        # Resource-only classifications legitimately have no fixed fragment
        # owners (for example a pool consisting solely of SU24 nodes).  An
        # empty registry is a valid ownership state; the allocator still
        # tracks every concrete resource node independently.
        return cls(node_owner_by_id=owners)


def exclude_occupied_nodes_from_allocator(assembler: Any) -> None:
    """Remove product-owned physical IDs from this assembler's pool.

    A fixed SU24 neighbour can live outside ``remaining_ids_by_su``.  The
    allocator filter alone therefore cannot protect it; the same
    ownership set is also consulted by fixed-support materialization.
    """
    occupied_node_ids = set(
        int(value)
        for value in getattr(assembler, "occupied_node_ids", set()) or set()
    )
    if not occupied_node_ids:
        return
    for mapping_name in ("initial_ids_by_su", "remaining_ids_by_su"):
        mapping = getattr(assembler.allocator, mapping_name, {})
        for su_key, values in list(mapping.items()):
            mapping[str(su_key)] = [
                int(value)
                for value in list(values or [])
                if int(value) not in occupied_node_ids
            ]
    assembler.allocator.su_key_by_node_id = {
        int(node_id): str(su_key)
        for su_key, values in assembler.allocator.initial_ids_by_su.items()
        for node_id in list(values or [])
    }


def exclude_bound_su24_supports_from_allocator(assembler: Any) -> None:
    """Remove exact SU24 supports from every generic allocator view."""
    bound_support_ids = {
        int(node_id)
        for support_by_su in dict(
            getattr(assembler, "su24_support_bindings_by_node_id", {}) or {}
        ).values()
        for values in dict(support_by_su or {}).values()
        for node_id in set(values or [])
    }
    if not bound_support_ids:
        return
    for mapping_name in ("initial_ids_by_su", "remaining_ids_by_su"):
        mapping = getattr(assembler.allocator, mapping_name, {})
        for su_key, values in list(mapping.items()):
            mapping[str(su_key)] = [
                int(value)
                for value in list(values or [])
                if int(value) not in bound_support_ids
            ]
    assembler.allocator.su_key_by_node_id = {
        int(node_id): str(su_key)
        for su_key, values in assembler.allocator.initial_ids_by_su.items()
        for node_id in list(values or [])
    }


def is_bound_su24_support(
    assembler,
    su24_node_id: Optional[int],
    support_su: int,
    support_node_id: int,
) -> bool:
    """Return whether an exact support belongs to this raw SU24 center."""
    if su24_node_id is None:
        return False
    if len(su24_support_owners(assembler, int(support_node_id))) > 1:
        return False
    bound = dict(
        getattr(assembler, "su24_support_bindings_by_node_id", {}).get(
            int(su24_node_id), {}
        )
        or {}
    )
    return int(support_node_id) in set(
        bound.get(int(support_su), set()) or set()
    )


def su24_support_owners(assembler: Any, support_node_id: int) -> Set[int]:
    owners: Set[int] = set()
    for su24_node_id, support_by_su in dict(
        getattr(assembler, "su24_support_bindings_by_node_id", {}) or {}
    ).items():
        if any(
            int(support_node_id) in set(values or [])
            for values in dict(support_by_su or {}).values()
        ):
            owners.add(int(su24_node_id))
    return owners


def remember_commit_failure(
    assembler,
    product: TopologyAssemblyProduct,
    *,
    reason: str,
    allocator_snapshot: Optional[Dict[str, Any]],
) -> None:
    if str(product.product_kind).startswith("fatty_ring_"):
        assembler.last_ring_materialization_reason = str(reason)
    attempted_claims = (
        assembler.allocator.claimed_ids_since(allocator_snapshot)
        if allocator_snapshot is not None
        else dict(product.claimed_resource_ids_by_su or {})
    )
    failure = {
        "attempted_product_kind": str(product.product_kind),
        "source_fragment_ids": sorted({
            int(value) for value in list(product.source_fragment_ids or [])
        }),
        "validation_reason": str(reason),
        "attempted_resource_counts": {
            str(key): len(list(values or []))
            for key, values in dict(attempted_claims or {}).items()
            if list(values or [])
        },
        "rolled_back_resource_ids": {
            str(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(attempted_claims or {}).items()
            if list(values or [])
        },
        "attempted_mainline_slots": [
            dict(item or {})
            for item in list(product.mainline_slots or [])
            if dict(item or {})
        ],
    }
    if str(reason) == "duplicate_node_within_product":
        failure["attempted_mainline_node_ids"] = [
            int(value) for value in list(product.mainline_node_ids or [])
        ]
        failure["attempted_branch_paths"] = [
            dict(path or {}) for path in list(product.branch_paths or [])
        ]
    record_composition_failure_entry(assembler, failure)


def record_composition_failure_entry(
    assembler,
    failure: Dict[str, Any],
) -> None:
    payload = dict(failure or {})
    assembler.composition_attempt_failure_count += 1
    assembler.last_composition_attempt_failure = payload
    for fragment_id in list(payload.get("source_fragment_ids", []) or []):
        fragment_id_i = int(fragment_id)
        previous = dict(
            assembler.materialization_failure_by_fragment_id.get(fragment_id_i, {})
            or {}
        )
        previous_is_ring = str(
            previous.get("attempted_product_kind", "") or ""
        ).startswith("fatty_ring_")
        current_is_ring = str(
            payload.get("attempted_product_kind", "") or ""
        ).startswith("fatty_ring_")
        if previous_is_ring and not current_is_ring:
            continue
        assembler.materialization_failure_by_fragment_id[fragment_id_i] = payload


def record_product_ownership_conflict(
    assembler,
    conflict: Dict[str, Any],
) -> None:
    assembler.product_ownership_conflict_count += 1
    assembler.last_product_ownership_conflict = dict(conflict or {})


def candidate_identity_conflict(
    assembler,
    *,
    rendered_node_ids: Iterable[int],
    source_fragment_ids: Iterable[int],
) -> Optional[Dict[str, Any]]:
    """Validate one candidate's physical identity before submission.

    A rendered route can contain nodes introduced by a branch donor.  The
    route is valid only when every fixed node belongs to one of the
    candidate's source fragments and no submitted product owns it already.
    Resource nodes are checked by the allocator transaction itassembler.
    """
    rendered = {int(value) for value in list(rendered_node_ids or [])}
    source_fragments = {int(value) for value in list(source_fragment_ids or [])}
    fixed_owner_by_id = dict(
        getattr(assembler, "fixed_node_owner_by_id", {}) or {}
    )
    product_owner_by_node_id = dict(
        getattr(assembler, "product_owner_by_node_id", {}) or {}
    )
    foreign_fixed = sorted(
        int(node_id)
        for node_id in rendered
        if int(node_id) in fixed_owner_by_id
        and int(node_id) not in set(
            getattr(assembler, "recycled_node_ids", set()) or set()
        )
        and int(fixed_owner_by_id[int(node_id)]) not in source_fragments
    )
    submitted = sorted(
        int(node_id)
        for node_id in rendered
        if int(node_id) in product_owner_by_node_id
    )
    if foreign_fixed:
        return {
            "reason": "candidate_fixed_node_foreign_owner",
            "foreign_fixed_node_ids": foreign_fixed,
            "foreign_fragment_ids": {
                str(node_id): int(fixed_owner_by_id[int(node_id)])
                for node_id in foreign_fixed
            },
            "source_fragment_ids": sorted(source_fragments),
        }
    if submitted:
        return {
            "reason": "candidate_node_already_owned",
            "submitted_node_ids": submitted,
            "existing_product_indices": sorted({
                int(product_owner_by_node_id[int(node_id)])
                for node_id in submitted
            }),
            "source_fragment_ids": sorted(source_fragments),
        }
    return None


def build_candidate_identity(
    assembler,
    product: TopologyAssemblyProduct,
    *,
    transaction_claims: Optional[Dict[str, Iterable[int]]] = None,
) -> Dict[str, Any]:
    """Return the canonical physical identity of a materialization candidate."""
    claimed_by_su: Dict[str, Set[int]] = {
        str(key): {int(value) for value in list(values or [])}
        for key, values in dict(product.claimed_resource_ids_by_su or {}).items()
    }
    for key, values in dict(transaction_claims or {}).items():
        claimed_by_su.setdefault(str(key), set()).update(
            int(value) for value in list(values or [])
        )
    rendered_node_ids = {
        int(value) for value in list(product.rendered_node_ids() or [])
    }
    fixed_node_ids = {
        int(value) for value in list(product.fixed_source_node_ids or [])
    }
    resource_node_ids = {
        int(value)
        for values in claimed_by_su.values()
        for value in values
    }
    return {
        "fragment_ids": sorted({
            int(value) for value in list(product.source_fragment_ids or [])
        }),
        "fixed_node_ids": sorted(fixed_node_ids),
        "resource_node_ids": sorted(resource_node_ids),
        "claimed_resource_ids_by_su": {
            str(key): sorted(int(value) for value in values)
            for key, values in sorted(claimed_by_su.items())
            if values
        },
        "rendered_node_ids": sorted(rendered_node_ids),
    }


def preflight_branch_donor(assembler: Any, view: AssemblyFragmentView) -> bool:
    """Reject a donor whose real route is no longer physically available."""
    fragment_id = int(view.fragment_id)
    route_ids = _canonical_view_render_node_ids(view)
    owner_map = dict(getattr(assembler, "fixed_node_owner_by_id", {}) or {})
    source_fragment_ids = {
        int(fragment_id),
        *(
            int(owner_map[int(node_id)])
            for node_id in route_ids
            if int(node_id) in owner_map
        ),
    }
    conflict = candidate_identity_conflict(assembler,
        rendered_node_ids=route_ids,
        source_fragment_ids=source_fragment_ids,
    )
    if conflict is not None:
        record_composition_failure_entry(assembler, {
            "attempted_product_kind": "branch_donor",
            "source_fragment_ids": sorted(source_fragment_ids),
            "validation_reason": "branch_donor_route_unavailable",
            "submitted_node_ids": list(conflict.get("submitted_node_ids", []) or []),
            "foreign_fixed_node_ids": list(conflict.get("foreign_fixed_node_ids", []) or []),
        })
        return False
    return True


def branch_result_route_ids(assembler: Any, result: BranchClosureResult) -> Set[int]:
    route_ids = {
        int(value)
        for value in list(getattr(result, "rendered_node_ids", []) or [])
        if _safe_int(value, -1) >= 0
    }
    route_ids.update(
        int(value)
        for value in list(getattr(result, "main_node_ids", []) or [])
        if _safe_int(value, -1) >= 0
    )
    route_ids.update(
        int(value)
        for value in re.findall(r"\d+", str(getattr(result, "branch_node_ids", "") or ""))
    )
    return route_ids


def fixed_branch_route_ids(assembler: Any, route_ids: Iterable[int]) -> Set[int]:
    owner_map = dict(getattr(assembler, "fixed_node_owner_by_id", {}) or {})
    recycled_node_ids = {
        int(value)
        for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
    }
    fixed: Set[int] = set()
    for raw_node_id in list(route_ids or []):
        node_id = int(raw_node_id)
        su_key = assembler.allocator.su_key_for_node_id(node_id)
        if (
            (node_id in owner_map and node_id not in recycled_node_ids)
            or su_key is None
            or int(su_key) not in set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES)
        ):
            fixed.add(node_id)
    return fixed


def preflight_branch_result(
    assembler: Any,
    result: BranchClosureResult,
) -> Optional[Dict[str, Any]]:
    """Validate a resolved branch before it can enter a product route."""
    shape_reason = _branch_route_shape_reason(
        str(getattr(result, "branch_topology", "") or ""),
        str(getattr(result, "branch_node_ids", "") or ""),
        assembler.nodes_by_id,
    )
    if shape_reason is not None:
        return {
            "reason": str(shape_reason),
            "branch_topology": str(getattr(result, "branch_topology", "") or ""),
            "branch_node_ids": str(getattr(result, "branch_node_ids", "") or ""),
        }
    route_ids = branch_result_route_ids(assembler, result)
    owner_map = dict(getattr(assembler, "fixed_node_owner_by_id", {}) or {})
    source_fragment_ids = {
        int(value) for value in list(getattr(result, "source_fragment_ids", []) or [])
    }
    source_fragment_ids.update(
        int(owner_map[int(node_id)])
        for node_id in route_ids
        if int(node_id) in owner_map
        and int(node_id) not in {
            int(value)
            for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
        }
    )
    conflict = candidate_identity_conflict(assembler,
        rendered_node_ids=route_ids,
        source_fragment_ids=source_fragment_ids,
    )
    if conflict is None:
        result.rendered_node_ids = sorted(route_ids)
        result.source_fragment_ids = sorted(source_fragment_ids)
        result.fixed_source_node_ids = sorted(fixed_branch_route_ids(assembler, route_ids))
        result.source_node_ids = sorted({
            int(value) for value in list(result.source_node_ids or [])
        } | route_ids)
        return None
    return {
        "reason": "branch_route_unavailable",
        "route_node_ids": sorted(route_ids),
        "source_fragment_ids": sorted(source_fragment_ids),
        "identity_conflict": dict(conflict),
    }


def invalidate_branch_donors_for_product(
    assembler,
    branch_donors: Optional[List[AssemblyFragmentView]],
    product: TopologyAssemblyProduct,
) -> None:
    """Remove donors whose canonical route is now owned by this product."""
    if branch_donors is None:
        return
    product_nodes = {
        int(value) for value in list(product.rendered_node_ids() or [])
    }
    product_fragments = {
        int(value) for value in list(product.source_fragment_ids or [])
    }
    branch_donors[:] = [
        view for view in list(branch_donors or [])
        if int(view.fragment_id) not in product_fragments
        and not (_canonical_view_render_node_ids(view) & product_nodes)
    ]


def commit_product(
    assembler,
    product: Optional[TopologyAssemblyProduct],
    *,
    allocator_snapshot: Optional[Dict[str, Any]] = None,
    branch_donors: Optional[List[AssemblyFragmentView]] = None,
    donor_snapshot: Optional[List[AssemblyFragmentView]] = None,
) -> bool:
    if product is None:
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    ring_contract = product.ring_contract
    is_ring = ring_contract is not None
    # Mainline closure operators may expose the same fixed bridge at the
    # boundary of a symbolic segment and the classified fragment.  The
    # product route contains one physical node in that case.  Normalize
    # only adjacent repeats here, leaving non-local repeats to the strict
    # ownership audit below.
    if not is_ring:
        normalized_mainline: List[int] = []
        for raw_node_id in list(product.mainline_node_ids or []):
            node_id = int(raw_node_id)
            if normalized_mainline and normalized_mainline[-1] == node_id:
                continue
            normalized_mainline.append(node_id)
        product.mainline_node_ids = normalized_mainline
    rendered_node_ids = list(product.rendered_node_ids())
    declared_source_ids = {
        int(value) for value in list(product.fixed_source_node_ids or [])
    }
    rendered_source_ids = set(rendered_node_ids)
    identity_conflict = candidate_identity_conflict(assembler,
        rendered_node_ids=rendered_source_ids,
        source_fragment_ids=product.source_fragment_ids,
    )
    if identity_conflict is not None:
        failure_reason = str(identity_conflict.get("reason", "candidate_identity_invalid"))
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=failure_reason,
            diagnostics={"candidate_identity": dict(identity_conflict)},
        )
        remember_commit_failure(assembler,
            product,
            reason=failure_reason,
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    raw_pool_su24_ids = {
        int(value)
        for value in list(assembler.allocator.initial_ids_by_su.get("24", []) or [])
    }
    released_su24_ids = {
        int(value)
        for value in list(getattr(assembler, "released_su24_ids", set()) or set())
    }
    captured_raw_su24_ids = sorted(
        (rendered_source_ids & raw_pool_su24_ids) - released_su24_ids
    )
    if captured_raw_su24_ids and not str(product.product_kind).startswith("fatty_ring_"):
        failure_reason = (
            "raw_su24_requires_ring_materialization:"
            f"{captured_raw_su24_ids}"
        )
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=str(failure_reason),
            diagnostics={"raw_su24_node_ids": captured_raw_su24_ids},
        )
        remember_commit_failure(assembler,
            product,
            reason=str(failure_reason),
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    transaction_claims = (
        assembler.allocator.claimed_ids_since(allocator_snapshot)
        if allocator_snapshot is not None
        else {}
    )
    candidate_identity = build_candidate_identity(assembler,
        product,
        transaction_claims=transaction_claims,
    )
    product.fixed_source_node_ids = list(candidate_identity["fixed_node_ids"])
    authorized_rendered_ids = {
        int(value) for value in candidate_identity["fixed_node_ids"]
    } | {
        int(value) for value in candidate_identity["resource_node_ids"]
    } | {
        int(value)
        for values in dict(transaction_claims or {}).values()
        for value in list(values or [])
    }
    unauthorized_rendered_ids = sorted(
        rendered_source_ids - authorized_rendered_ids
    )
    if unauthorized_rendered_ids:
        failure_reason = (
            "rendered_node_without_transaction_ownership:"
            f"{unauthorized_rendered_ids}"
        )
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=str(failure_reason),
            diagnostics={
                "fixed_source_node_ids": sorted({
                    int(value)
                    for value in candidate_identity["fixed_node_ids"]
                }),
                "transaction_claimed_ids": {
                    str(key): [int(value) for value in list(values or [])]
                    for key, values in dict(transaction_claims or {}).items()
                },
            },
        )
        remember_commit_failure(assembler,
            product,
            reason=str(failure_reason),
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    source_fragment_id_set = {
        int(value) for value in list(product.source_fragment_ids or [])
    }
    recycled_node_ids = {
        int(value)
        for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
    }
    cross_fragment_ids = sorted(
        int(node_id)
        for node_id in rendered_source_ids
        if int(node_id) in assembler.fragment_owner_by_node_id
        and int(node_id) not in recycled_node_ids
        and int(assembler.fragment_owner_by_node_id[int(node_id)]) not in source_fragment_id_set
    )
    if cross_fragment_ids:
        failure_reason = f"cross_fragment_node_capture:{cross_fragment_ids}"
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=str(failure_reason),
            diagnostics={
                "source_fragment_ids": sorted(source_fragment_id_set),
                "foreign_fragment_owners": {
                    str(node_id): int(assembler.fragment_owner_by_node_id[int(node_id)])
                    for node_id in cross_fragment_ids
                },
            },
        )
        remember_commit_failure(assembler,
            product,
            reason=str(failure_reason),
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    missing_fixed_ids = sorted(declared_source_ids - rendered_source_ids)
    duplicate_rendered_ids = [] if is_ring else sorted({
        int(value)
        for value in rendered_node_ids
        if rendered_node_ids.count(int(value)) > 1
    })
    if missing_fixed_ids or duplicate_rendered_ids:
        failure_reason = (
            f"fixed_source_not_in_rendered_product:{missing_fixed_ids}"
            if missing_fixed_ids
            else f"duplicate_rendered_node_ids:{duplicate_rendered_ids}"
        )
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=str(failure_reason),
            diagnostics={
                "fixed_source_node_ids": sorted(declared_source_ids),
                "rendered_node_ids": sorted(rendered_source_ids),
            },
        )
        remember_commit_failure(assembler,
            product,
            reason=str(failure_reason),
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    mainline_ids = [int(value) for value in list(product.mainline_node_ids or [])]
    branch_paths = [dict(path or {}) for path in list(product.branch_paths or [])]
    fixed_source_ids = {
        int(value) for value in list(product.fixed_source_node_ids or [])
    }
    for node_id in sorted(set(rendered_node_ids)):
        # Fixed classification/support nodes are already owned by the
        # product contract. They are intentionally absent from the allocator
        # and must not be re-claimed during rendered-resource reconciliation.
        if int(node_id) in fixed_source_ids:
            continue
        su_key = assembler.allocator.su_key_for_node_id(int(node_id))
        if (
            su_key is None
            or int(su_key) not in set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES)
        ):
            continue
        if int(node_id) not in set(assembler.allocator.initial_ids_by_su.get(su_key, []) or []):
            continue
        if int(node_id) in set(assembler.allocator.consumed_ids_by_su.get(su_key, []) or []):
            continue
        claim_specific = (
            assembler.allocator.claim_specific
            if is_ring
            else assembler.allocator.claim_specific
        )
        if claim_specific(int(su_key), int(node_id)) is not None:
            continue
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=f"rendered_resource_claim_failed:{int(node_id)}",
            diagnostics={"node_id": int(node_id), "su_type": int(su_key)},
        )
        remember_commit_failure(assembler,
            product,
            reason=f"rendered_resource_claim_failed:{int(node_id)}",
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    existing_claims = {
        str(key): {int(value) for value in list(values or [])}
        for key, values in dict(product.claimed_resource_ids_by_su or {}).items()
    }
    fragment_source_ids = {
        int(value)
        for fragment_id in list(product.source_fragment_ids or [])
        for value in list(
            getattr(
                next((
                    fragment
                    for fragment in list(getattr(assembler.classification, "fragments", []) or [])
                    if int(getattr(fragment, "fragment_id", -1)) == int(fragment_id)
                ), None),
                "claimed_node_ids",
                [],
            )
            or []
        )
    }
    if allocator_snapshot is not None:
        for key, values in assembler.allocator.claimed_ids_since(allocator_snapshot).items():
            existing_claims.setdefault(str(key), set()).update(
                int(value) for value in list(values or [])
            )
    symbolic_resource_ids = {
        int(value) for values in existing_claims.values() for value in values
    }
    product.fixed_source_node_ids = sorted({
        int(value) for value in list(product.fixed_source_node_ids or [])
    })

    def _connection_slot(
        *,
        left: int,
        right: int,
        slot_id: str,
        anchor_role: str,
    ) -> Dict[str, Any]:
        left_node = assembler.nodes_by_id.get(int(left))
        right_node = assembler.nodes_by_id.get(int(right))
        fixed = bool(
            int(right) in set(getattr(left_node, "fixed_hop1_ids", set()) or set())
            or int(left) in set(getattr(right_node, "fixed_hop1_ids", set()) or set())
        )
        assigned_resource_ids = sorted(
            {int(left), int(right)} & symbolic_resource_ids
        )
        connection_semantics = (
            "fixed"
            if fixed
            else "symbolic"
            if assigned_resource_ids
            else "open"
        )
        return {
            "slot_id": str(slot_id),
            "kind": f"{connection_semantics.upper()}_CONNECTION",
            "anchor_node_id": int(left),
            "anchor_role": str(anchor_role),
            "required_su_sequence": [],
            "existing_node_ids": [int(left), int(right)] if fixed else [],
            "assigned_resource_ids": assigned_resource_ids,
            "connection_semantics": str(connection_semantics),
            "state": "OPEN" if connection_semantics == "open" else "FILLED",
        }

    if ring_contract is not None:
        product.mainline_slots = []
        product.branch_slots = []
        ring_reason = ring_topology_module.audit_ring_topology_contract(
            ring_contract,
            nodes=assembler.nodes,
            owned_node_ids=rendered_source_ids,
        )
        if ring_reason is not None:
            _record_invalid_topology_attempt(assembler,
                product_kind=str(product.product_kind),
                actual_topology_su=str(product.actual_topology_su),
                actual_topology_gid=str(product.actual_topology_gid),
                topology_notation=str(product.actual_topology_su),
                validation_reason=str(ring_reason),
                diagnostics={"ring_template_kind": str(ring_contract.ring_kind)},
            )
            remember_commit_failure(assembler,
                product,
                reason=str(ring_reason),
                allocator_snapshot=allocator_snapshot,
            )
            if allocator_snapshot is not None:
                assembler.allocator.rollback(allocator_snapshot)
            if branch_donors is not None and donor_snapshot is not None:
                branch_donors[:] = donor_snapshot
            return False
    else:
        product.mainline_slots = [
            _connection_slot(
                left=int(left),
                right=int(right),
                slot_id=f"product:{len(assembler.products)}:mainline:{index}",
                anchor_role="mainline",
            )
            for index, (left, right) in enumerate(zip(mainline_ids, mainline_ids[1:]))
        ]
        product.branch_slots = []
        for branch_index, path in enumerate(branch_paths):
            anchor_id = int(path.get("anchor_node_id", -1))
            path_ids = [
                int(value)
                for value in list(path.get("node_ids", []) or [])
            ]
            product.branch_slots.extend([
                _connection_slot(
                    left=int(left),
                    right=int(right),
                    slot_id=f"product:{len(assembler.products)}:branch:{branch_index}:{edge_index}",
                    anchor_role="branch",
                )
                for edge_index, (left, right) in enumerate(zip([anchor_id] + path_ids, path_ids))
            ])
    flattened_rendered_node_ids = list(mainline_ids) + [
        int(value)
        for path in list(branch_paths or [])
        for value in list(dict(path or {}).get("node_ids", []) or [])
    ]
    flattened_unique_route: List[int] = []
    for node_id in flattened_rendered_node_ids:
        if flattened_unique_route and flattened_unique_route[-1] == int(node_id):
            continue
        flattened_unique_route.append(int(node_id))
    if len(flattened_unique_route) != len(set(flattened_unique_route)):
        conflict = {
            "reason": "duplicate_node_within_product",
            "product_kind": str(product.product_kind),
            "node_ids": list(flattened_rendered_node_ids),
        }
        record_product_ownership_conflict(assembler, conflict)
        remember_commit_failure(assembler,
            product,
            reason="duplicate_node_within_product",
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    owned_node_ids = sorted(rendered_source_ids)
    overlaps = sorted(
        int(node_id)
        for node_id in owned_node_ids
        if int(node_id) in assembler.product_owner_by_node_id
    )
    if overlaps:
        record_product_ownership_conflict(assembler, {
            "reason": "duplicate_topology_ownership",
            "product_kind": str(product.product_kind),
            "actual_topology_gid": str(product.actual_topology_gid),
            "node_ids": overlaps,
            "existing_product_indices": sorted({
                int(assembler.product_owner_by_node_id[node_id]) for node_id in overlaps
            }),
        })
        remember_commit_failure(assembler,
            product,
            reason=f"duplicate_topology_ownership:{overlaps}",
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False
    committed_resource_ids: Dict[str, Set[int]] = defaultdict(set)
    for node_id in owned_node_ids:
        su_key = assembler.allocator.su_key_for_node_id(int(node_id))
        if (
            su_key is not None
            and int(su_key) in set(PHYSICAL_TOPOLOGY_RESOURCE_TYPES)
        ):
            committed_resource_ids[str(su_key)].add(int(node_id))
            if int(node_id) not in fragment_source_ids:
                existing_claims.setdefault(str(su_key), set()).add(int(node_id))

    claim_key_by_node_id: Dict[int, str] = {}
    claim_type_conflicts: List[Dict[str, Any]] = []
    for raw_key, values in existing_claims.items():
        claim_key = _normalize_su_key(raw_key)
        for node_id in sorted({int(value) for value in values}):
            allocator_key = assembler.allocator.su_key_for_node_id(int(node_id))
            if allocator_key is not None and str(allocator_key) != str(claim_key):
                claim_type_conflicts.append({
                    "node_id": int(node_id),
                    "claimed_su": str(claim_key),
                    "allocator_su": str(allocator_key),
                })
            prior_key = claim_key_by_node_id.get(int(node_id))
            if prior_key is not None and str(prior_key) != str(claim_key):
                claim_type_conflicts.append({
                    "node_id": int(node_id),
                    "claimed_su": str(claim_key),
                    "prior_claimed_su": str(prior_key),
                })
            claim_key_by_node_id[int(node_id)] = str(claim_key)
    current_transaction_claims = (
        assembler.allocator.claimed_ids_since(allocator_snapshot)
        if allocator_snapshot is not None
        else {}
    )
    product_pool_claims = {
        str(key): sorted(
            int(node_id)
            for node_id in values
            if assembler.allocator.su_key_for_node_id(int(node_id)) is not None
        )
        for key, values in existing_claims.items()
        if any(
            assembler.allocator.su_key_for_node_id(int(node_id)) is not None
            for node_id in values
        )
    }
    normalized_transaction_claims = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(current_transaction_claims or {}).items()
        if list(values or [])
    }
    transaction_claim_mismatch = bool(
        allocator_snapshot is not None
        and product_pool_claims != normalized_transaction_claims
    )
    fixed_ids = {
        int(value) for value in list(product.fixed_source_node_ids or [])
    }
    claimed_ids = {
        int(value)
        for values in dict(existing_claims or {}).values()
        for value in list(values or [])
    }
    identity_overlap = sorted(fixed_ids & claimed_ids)
    identity_missing = sorted(
        rendered_source_ids - (fixed_ids | claimed_ids)
    )
    identity_unrendered = sorted(
        (fixed_ids | claimed_ids) - rendered_source_ids
    )
    identity_mismatch = bool(
        identity_overlap or identity_missing or identity_unrendered
    )
    if claim_type_conflicts or transaction_claim_mismatch or identity_mismatch:
        failure_reason = (
            "resource_claim_type_conflict"
            if claim_type_conflicts
            else "transaction_product_claim_mismatch"
            if transaction_claim_mismatch
            else "rendered_product_ownership_mismatch"
        )
        _record_invalid_topology_attempt(assembler,
            product_kind=str(product.product_kind),
            actual_topology_su=str(product.actual_topology_su),
            actual_topology_gid=str(product.actual_topology_gid),
            topology_notation=str(product.actual_topology_su),
            validation_reason=str(failure_reason),
            diagnostics={
                "claim_type_conflicts": list(claim_type_conflicts),
                "product_pool_claims": dict(product_pool_claims),
                "transaction_claims": dict(normalized_transaction_claims),
                "fixed_source_node_ids": sorted(fixed_ids),
                "rendered_node_ids": sorted(rendered_source_ids),
                "identity_overlap": list(identity_overlap),
                "rendered_without_owner": list(identity_missing),
                "owned_without_render": list(identity_unrendered),
            },
        )
        remember_commit_failure(assembler,
            product,
            reason=str(failure_reason),
            allocator_snapshot=allocator_snapshot,
        )
        if allocator_snapshot is not None:
            assembler.allocator.rollback(allocator_snapshot)
        if branch_donors is not None and donor_snapshot is not None:
            branch_donors[:] = donor_snapshot
        return False

    product.claimed_resource_ids_by_su = {
        str(key): sorted(int(value) for value in values)
        for key, values in existing_claims.items()
        if values
    }
    product_index = int(len(assembler.products))
    assembler.products.append(product)
    for node_id in owned_node_ids:
        assembler.product_owner_by_node_id[int(node_id)] = product_index
    invalidate_branch_donors_for_product(assembler, branch_donors, product)
    assembler.allocator.consume_existing_ids({
        str(key): sorted(int(value) for value in values)
        for key, values in committed_resource_ids.items()
        if values
    })
    return True


def reconcile_resource_ownership(
    assembler,
    ledger: TopologyPoolLedger,
) -> TopologyPoolLedger:
    resource_types = sorted({10, 11, 12, 13, 22, 23, 24})
    global_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
    for node in list(assembler.nodes or []):
        su_type = int(getattr(node, "su_type", -1))
        if su_type in resource_types:
            global_ids_by_su[str(su_type)].append(int(getattr(node, "global_id", -1)))
    global_ids_by_su = {
        str(key): sorted(set(int(value) for value in list(values or []) if int(value) >= 0))
        for key, values in global_ids_by_su.items()
    }

    fixed_by_product: Dict[str, Dict[str, Any]] = {}
    fixed_node_ids: Set[int] = set()
    for product_index, product in enumerate(list(assembler.products or [])):
        product_dict = product.to_dict() if hasattr(product, "to_dict") else dict(product or {})
        source_ids = sorted({
            int(value)
            for value in list(product_dict.get("fixed_source_node_ids", []) or [])
            + [
                int(value)
                for values in dict(product_dict.get("claimed_resource_ids_by_su", {}) or {}).values()
                for value in list(values or [])
            ]
        })
        counts = Counter(
            int(getattr(assembler.nodes_by_id.get(int(node_id)), "su_type", -1))
            for node_id in source_ids
            if int(getattr(assembler.nodes_by_id.get(int(node_id)), "su_type", -1)) in resource_types
        )
        product_key = f"{str(product_dict.get('product_kind', 'product'))}:{int(product_index)}"
        fixed_by_product[product_key] = {
            "source_node_ids": source_ids,
            "counts_by_su": {str(key): int(value) for key, value in sorted(counts.items())},
        }
        fixed_node_ids.update(source_ids)

    fragment_owned_node_ids: Set[int] = set()
    recycled_node_ids = {
        int(value)
        for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
    }
    recycled_fragment_ids = {
        int(value)
        for value in list(getattr(assembler, "recycled_fragment_ids", set()) or set())
    }
    fixed_fragment_ids = {
        int(fragment_id)
        for product in list(assembler.products or [])
        for fragment_id in list(getattr(product, "source_fragment_ids", []) or [])
    }
    for fragment in list(getattr(assembler.classification, "fragments", []) or []):
        if int(getattr(fragment, "fragment_id", -1)) in (
            fixed_fragment_ids | recycled_fragment_ids
        ):
            continue
        fragment_owned_node_ids.update(
            int(value)
            for value in list(getattr(fragment, "claimed_node_ids", []) or [])
        )
    fragment_owned_node_ids.difference_update(fixed_node_ids)
    fragment_owned_node_ids.difference_update(recycled_node_ids)

    fixed_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
    fragment_owned_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
    remaining_ids_by_su: Dict[str, List[int]] = {str(su): [] for su in resource_types}
    for su_type in resource_types:
        for node_id in list(global_ids_by_su.get(str(su_type), []) or []):
            if node_id in fixed_node_ids:
                fixed_ids_by_su[str(su_type)].append(node_id)
            elif node_id in fragment_owned_node_ids:
                fragment_owned_ids_by_su[str(su_type)].append(node_id)
            else:
                remaining_ids_by_su[str(su_type)].append(node_id)

    allocation_state_by_node: Dict[str, str] = {}
    for node_id in fixed_node_ids:
        allocation_state_by_node[str(node_id)] = "fixed"
    for node_id in fragment_owned_node_ids:
        allocation_state_by_node[str(node_id)] = "fragment_owned"
    for values in remaining_ids_by_su.values():
        for node_id in values:
            allocation_state_by_node[str(node_id)] = "remaining"

    allocator_source_ids_by_su = dict(getattr(ledger, "remaining_ids_by_su", {}) or {})
    effective_remaining_ids_by_su = {
        str(key): sorted(
            set(allocator_source_ids_by_su.get(str(key), []))
            & set(remaining_ids_by_su.get(str(key), []))
        )
        for key in global_ids_by_su
    }
    allocator = getattr(assembler, "allocator", None)
    protected_counts = {
        str(key): max(0, int(getattr(allocator, "protected_counts_by_su", {}).get(str(key), 0) or 0))
        for key in global_ids_by_su
    }
    convertible_ids_by_su = {
        str(su_type): list(
            effective_remaining_ids_by_su.get(str(su_type), [])[
                : max(
                    0,
                    len(
                        effective_remaining_ids_by_su.get(
                            str(su_type), []
                        )
                    )
                    - int(protected_counts.get(str(su_type), 0)),
                )
            ]
        )
        for su_type in EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
    }
    # SU24 remains a ring-only resource. Recycled SU24 has already been
    # reclassified to SU23 before this reconciliation step.
    released_su24_ids = {
        int(value)
        for value in list(getattr(assembler, "released_su24_ids", set()) or set())
    }
    invariant_errors: List[str] = []
    for key, global_ids in global_ids_by_su.items():
        buckets = [
            set(fixed_ids_by_su.get(str(key), [])),
            set(fragment_owned_ids_by_su.get(str(key), [])),
            set(remaining_ids_by_su.get(str(key), [])),
        ]
        if any(buckets[left] & buckets[right] for left in range(3) for right in range(left + 1, 3)):
            invariant_errors.append(f"su{key}:ownership_overlap")
        if set(global_ids) - set().union(*buckets):
            invariant_errors.append(f"su{key}:unaccounted")

    ledger.global_ids_by_su = global_ids_by_su
    ledger.fixed_ids_by_su = fixed_ids_by_su
    ledger.fragment_owned_ids_by_su = fragment_owned_ids_by_su
    ledger.remaining_ids_by_su = deepcopy(effective_remaining_ids_by_su)
    ledger.convertible_ids_by_su = deepcopy(convertible_ids_by_su)
    allocator_initial_source = dict(
        getattr(
            getattr(assembler, "allocator", None), "initial_ids_by_su", {}
        )
        or {}
    )
    if not allocator_initial_source:
        allocator_initial_source = dict(ledger.initial_ids_by_su or ledger.remaining_ids_by_su or {})
    allocator_initial_ids_by_su = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in allocator_initial_source.items()
    }
    ledger.initial_ids_by_su = deepcopy(allocator_initial_ids_by_su)
    ledger.consumed_ids_by_su = {
        str(key): sorted(
            set(allocator_initial_ids_by_su.get(str(key), []) or [])
            - set(effective_remaining_ids_by_su.get(str(key), []) or [])
        )
        for key in allocator_initial_ids_by_su
    }
    ledger.global_counts = {str(key): int(len(values)) for key, values in global_ids_by_su.items()}
    ledger.fixed_counts_by_su = {str(key): int(len(values)) for key, values in fixed_ids_by_su.items()}
    ledger.fragment_owned_counts_by_su = {str(key): int(len(values)) for key, values in fragment_owned_ids_by_su.items()}
    ledger.remaining_counts = {
        str(key): int(len(values))
        for key, values in effective_remaining_ids_by_su.items()
    }
    ledger.convertible_counts = {
        str(key): int(len(values))
        for key, values in ledger.convertible_ids_by_su.items()
    }
    ledger.protected_counts = dict(protected_counts)
    ledger.invariant_errors = list(invariant_errors)
    ledger.initial_counts = {str(key): int(len(values)) for key, values in ledger.initial_ids_by_su.items()}
    ledger.consumed_counts = {
        str(key): int(len(values))
        for key, values in ledger.consumed_ids_by_su.items()
    }
    ledger.snapshot_stage = "post_assembly_reconciled"
    ledger.fixed_node_ids = sorted(fixed_node_ids)
    ledger.fragment_owned_node_ids = sorted(fragment_owned_node_ids)
    ledger.fixed_by_product = fixed_by_product
    ledger.allocation_state_by_node = allocation_state_by_node
    ledger.released_su24_ids = sorted(released_su24_ids)
    ledger.released_resource_ids_by_su = {
        str(key): sorted(int(value) for value in values)
        for key, values in dict(
            getattr(assembler, "released_resource_ids_by_su", {}) or {}
        ).items()
        if list(values or [])
    }
    return ledger


def recycle_resource_ring(assembler: Any, product: Any) -> List[int]:
    """Dismantle one pure resource ring and return every released node ID.

    The operation is deliberately atomic at the ownership boundary: the ring
    product, its completed fragment state, SU24 support bindings, pending
    contracts, and allocator are updated together.  A ring containing any
    physical SU outside ``11/22/23/24`` is rejected by the shared topology
    predicate and is never partially dismantled.
    """
    if not ring_topology_module.is_recyclable_pure_resource_ring(
        product,
        assembler=assembler,
    ):
        return []
    contract = getattr(product, "ring_contract", None)
    node_su = {
        int(node_id): int(su_type)
        for node_id, su_type in dict(
            getattr(contract, "node_su_by_id", {}) or {}
        ).items()
    }
    released_ids_by_su: Dict[str, List[int]] = defaultdict(list)
    for node_id, su_type in sorted(node_su.items()):
        if int(su_type) in RECYCLABLE_PURE_RESOURCE_SU_TYPES:
            released_ids_by_su[str(int(su_type))].append(int(node_id))
    released_ids = sorted({
        int(value)
        for values in released_ids_by_su.values()
        for value in values
    })
    if not released_ids:
        return []

    products = list(getattr(assembler, "products", []) or [])
    product_index = next(
        (
            index
            for index, existing in enumerate(products)
            if existing is product or existing == product
        ),
        None,
    )
    if product_index is None:
        return []
    allocator_snapshot = assembler.allocator.snapshot()
    allocator_su_map = deepcopy(
        getattr(assembler.allocator, "su_key_by_node_id", {}) or {}
    )
    products.pop(int(product_index))
    try:
        # Re-register fixed classifier nodes as concrete allocator resources
        # before continuing with the new product list. ``release_ids`` is
        # idempotent for pool nodes that were already claimed by this ring.
        assembler.allocator.release_ids(released_ids_by_su)
        su24_ids = [
            int(node_id)
            for node_id, su_type in node_su.items()
            if int(su_type) == 24
        ]
        converted_su24_ids = assembler.allocator.reclassify_ids(
            su24_ids,
            from_su=24,
            to_su=23,
        )
        if len(converted_su24_ids) != len(su24_ids):
            raise ValueError("released SU24 reclassification failed")
    except Exception:
        assembler.allocator.rollback(allocator_snapshot)
        assembler.allocator.su_key_by_node_id = allocator_su_map
        return []
    assembler.products = products

    # The external assembler owns a transaction-local node copy.  Updating
    # these types makes subsequent side-chain retries see the released nodes
    # as real SU23 resources, while the outer Stage3 commit remains atomic.
    for node_id in converted_su24_ids:
        node = getattr(assembler, "nodes_by_id", {}).get(int(node_id))
        if node is not None:
            setattr(node, "su_type", 23)

    # Product indices are part of the ownership map; rebuild them after the
    # removal instead of leaving indices shifted by one ring.
    assembler.product_owner_by_node_id = {}
    for product_index, existing in enumerate(list(assembler.products or [])):
        for node_id in existing.owned_node_ids():
            assembler.product_owner_by_node_id[int(node_id)] = int(product_index)

    fragment_ids = {
        int(value) for value in list(getattr(product, "source_fragment_ids", []) or [])
    }
    assembler.recycled_fragment_ids = set(
        getattr(assembler, "recycled_fragment_ids", set()) or set()
    ) | fragment_ids
    assembler.recycled_node_ids = set(
        getattr(assembler, "recycled_node_ids", set()) or set()
    ) | set(released_ids)
    assembler.occupied_node_ids.difference_update(released_ids)
    ownership_registry = getattr(assembler, "ownership_registry", None)
    if ownership_registry is not None:
        for node_id in released_ids:
            ownership_registry.node_owner_by_id.pop(int(node_id), None)
    released_registry = {
        str(key): {
            int(value) for value in list(values or [])
        }
        for key, values in dict(
            getattr(assembler, "released_resource_ids_by_su", {}) or {}
        ).items()
    }
    for key, values in released_ids_by_su.items():
        released_registry.setdefault(str(key), set()).update(
            int(value) for value in list(values or [])
        )
    # ``released_resource_ids_by_su`` describes the current resource type,
    # not the original provenance.  SU24 released by this ring is therefore
    # moved to the ordinary SU23 registry.
    converted_set = {int(value) for value in converted_su24_ids}
    released_registry.setdefault("24", set()).difference_update(converted_set)
    released_registry.setdefault("23", set()).update(converted_set)
    assembler.released_resource_ids_by_su = released_registry
    for fragment_id in fragment_ids:
        assembler.fragment_state_by_id[int(fragment_id)] = "recycled"
        assembler.ring_pending_reasons.pop(int(fragment_id), None)
        assembler.ring_pending_attempts.pop(int(fragment_id), None)
        assembler.ring_infeasible_contracts.pop(int(fragment_id), None)
        assembler.materialization_failure_by_fragment_id.pop(int(fragment_id), None)

    # A released SU24 no longer owns the fixed support nodes discovered by
    # classification.  Remove only bindings touching this ring; unrelated
    # SU24 centers remain protected.
    bindings: Dict[int, Dict[int, Set[int]]] = {}
    for center_id, support_by_su in dict(
        getattr(assembler, "su24_support_bindings_by_node_id", {}) or {}
    ).items():
        center = int(center_id)
        if center in set(released_ids):
            continue
        cleaned: Dict[int, Set[int]] = {}
        for support_su, support_ids in dict(support_by_su or {}).items():
            kept = {
                int(value)
                for value in set(support_ids or set())
                if int(value) not in set(released_ids)
            }
            if kept:
                cleaned[int(support_su)] = kept
        if cleaned:
            bindings[center] = cleaned
    assembler.su24_support_bindings_by_node_id = bindings
    assembler.released_su24_ids = set(
        getattr(assembler, "released_su24_ids", set()) or set()
    ) | {
        int(node_id) for node_id, su_type in node_su.items() if int(su_type) == 24
    }
    assembler.recycled_su24_to_23_ids = set(
        getattr(assembler, "recycled_su24_to_23_ids", set()) or set()
    ) | converted_set
    conversion_ops = list(
        getattr(assembler, "recycled_resource_conversion_ops", []) or []
    )
    conversion_ops.extend(
        {
            "from_su": 24,
            "to_su": 23,
            "node_id": int(node_id),
            "count": 1,
            "reason": "recycle_pure_resource_ring",
        }
        for node_id in sorted(converted_set)
    )
    assembler.recycled_resource_conversion_ops = conversion_ops
    pending_conversion_ops = list(
        getattr(assembler, "pending_recycled_resource_conversion_ops", []) or []
    )
    pending_conversion_ops.extend(
        {
            "from_su": 24,
            "to_su": 23,
            "node_id": int(node_id),
            "count": 1,
            "reason": "recycle_pure_resource_ring",
        }
        for node_id in sorted(converted_set)
    )
    assembler.pending_recycled_resource_conversion_ops = pending_conversion_ops

    invalidate_pending_identity(
        assembler,
        fragment_ids=fragment_ids,
        claimed_node_ids=released_ids,
    )
    assembler.last_recycled_ring = {
        "product_kind": str(getattr(product, "product_kind", "") or ""),
        "ring_kind": str(getattr(contract, "ring_kind", "") or ""),
        "source_fragment_ids": sorted(fragment_ids),
        "released_ids_by_su": {
            str(key): sorted(int(value) for value in values)
            for key, values in released_ids_by_su.items()
        },
        "reclassified_su24_to_23_ids": sorted(converted_set),
    }
    return released_ids


def fragment_available(assembler: Any, fragment_id: int) -> bool:
    return str(assembler.fragment_state_by_id.get(int(fragment_id), "available")) == "available"


def reserve_fragments(assembler: Any, fragment_ids: Iterable[int]) -> bool:
    selected = sorted({int(value) for value in list(fragment_ids or [])})
    if not selected or any(not fragment_available(assembler, fragment_id) for fragment_id in selected):
        return False
    for fragment_id in selected:
        assembler.fragment_state_by_id[int(fragment_id)] = "reserved"
    return True


def release_reserved_fragments(assembler: Any, fragment_ids: Iterable[int]) -> None:
    for fragment_id in sorted({int(value) for value in list(fragment_ids or [])}):
        if str(assembler.fragment_state_by_id.get(int(fragment_id), "")) == "reserved":
            assembler.fragment_state_by_id[int(fragment_id)] = "available"


def pending_identity_overlaps(
    pending: TopologyPendingUnit,
    fragment_ids: Set[int],
    claimed_node_ids: Set[int],
) -> bool:
    pending_fragment_ids = {
        int(value) for value in list(getattr(pending, "source_fragment_ids", []) or [])
    }
    pending_claimed_ids = {
        int(value)
        for value in list(pending.claimed_source_node_ids or [])
    }
    return bool(pending_fragment_ids & fragment_ids or pending_claimed_ids & claimed_node_ids)


def contract_identity_overlaps(
    contract: Dict[str, Any],
    fragment_ids: Set[int],
    claimed_node_ids: Set[int],
) -> bool:
    data = dict(contract or {})
    contract_fragment_ids = {
        int(value) for value in list(data.get("source_fragment_ids", []) or [])
    }
    contract_claimed_ids = {
        int(value)
        for value in list(data.get("claimed_source_node_ids", []) or [])
    }
    return bool(contract_fragment_ids & fragment_ids or contract_claimed_ids & claimed_node_ids)


def invalidate_pending_identity(
    assembler,
    *,
    fragment_ids: Iterable[int],
    claimed_node_ids: Iterable[int],
) -> None:
    fragment_id_set = {int(value) for value in list(fragment_ids or [])}
    claimed_node_id_set = {int(value) for value in list(claimed_node_ids or [])}
    assembler.pending_units = [
        pending
        for pending in list(assembler.pending_units or [])
        if not pending_identity_overlaps(pending, fragment_id_set, claimed_node_id_set)
    ]
    assembler.pending_contracts = [
        contract
        for contract in list(assembler.pending_contracts or [])
        if not contract_identity_overlaps(contract, fragment_id_set, claimed_node_id_set)
    ]


def mark_fragments_completed(assembler: Any, fragment_ids: Iterable[int]) -> None:
    completed_fragment_ids = {int(value) for value in list(fragment_ids or [])}
    for fragment_id in completed_fragment_ids:
        frag_id = int(fragment_id)
        assembler.fragment_state_by_id[frag_id] = "completed"
        assembler.ring_pending_reasons.pop(frag_id, None)
        assembler.ring_pending_attempts.pop(frag_id, None)
        assembler.ring_infeasible_contracts.pop(frag_id, None)
        assembler.materialization_failure_by_fragment_id.pop(frag_id, None)
    used_node_ids = {
        int(node_id)
        for fragment_id in list(fragment_ids or [])
        for node_id in list(
            getattr(assembler.view_by_id.get(int(fragment_id)), "source_node_ids", [])
            if assembler.view_by_id.get(int(fragment_id)) is not None
            else []
        )
    }
    invalidate_pending_identity(assembler,
        fragment_ids=completed_fragment_ids,
        claimed_node_ids=used_node_ids,
    )


def drop_pending_overlapping_product(
    assembler: Any,
    product: TopologyAssemblyProduct,
) -> None:
    owned_node_ids = {
        int(node_id)
        for node_id in list(getattr(product, "fixed_source_node_ids", []) or [])
    } | {
        int(node_id)
        for values in dict(getattr(product, "claimed_resource_ids_by_su", {}) or {}).values()
        for node_id in list(values or [])
    }
    if not owned_node_ids:
        return
    invalidate_pending_identity(assembler,
        fragment_ids=list(getattr(product, "source_fragment_ids", []) or []),
        claimed_node_ids=owned_node_ids,
    )


def available_views(
    assembler: Any,
    predicate: Any,
) -> List[AssemblyFragmentView]:
    return [
        view for view in assembler.views
        if fragment_available(assembler, int(view.fragment_id)) and bool(predicate(view))
    ]
