"""Canonical pending-contract diagnostics for topology assembly."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...stage3_models import (
    TopologyPathRender,
    TopologyPendingUnit,
    stage3_contract_owner_key,
)
from ..core.fragment import AssemblyFragmentView
from ..projection.engine import (
    _collect_source_metadata,
    _fixed_source_ids_from_views,
    _slot_required_counts,
)
from ..core.rules import external_phase_for_role


_STRUCTURAL_DETAIL_KEYS = (
    "contract_base_path_labels",
    "contract_base_path_node_ids",
    "unexpected_fixed_edges",
    "offending_fixed_edges",
    "terminal_node_ids",
    "missing_node_ids",
    "mainline_node_ids",
    "structural_boundary",
    "structure_signature_mismatches",
)


def structural_blocker_details(
    diagnostics: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Return the first concrete topology-materialization failure.

    Branch diagnostics are grouped by anchor because SU25 may own multiple
    sibling branches.  A pending contract is emitted at the owning fragment
    level, so its proof uses the first failed branch in deterministic anchor
    order while retaining the complete per-anchor map in ``diagnostics``.
    """

    payload = dict(diagnostics or {})
    details_by_anchor = dict(
        payload.get("branch_contract_blocker_details", {}) or {}
    )
    for anchor in sorted(
        details_by_anchor,
        key=lambda value: (
            (0, int(value))
            if str(value).lstrip("-").isdigit()
            else (1, str(value))
        ),
    ):
        for raw_detail in list(details_by_anchor.get(anchor, []) or []):
            detail = dict(raw_detail or {})
            if str(detail.get("failure_domain", "") or "") != "topology_materialization":
                continue
            result = {
                "failure_domain": "topology_materialization",
                "branch_anchor_index": int(anchor),
            }
            reason = str(detail.get("reason", "") or "")
            if reason:
                result["assembler_reason"] = reason
            for key in _STRUCTURAL_DETAIL_KEYS:
                if key in detail:
                    result[key] = detail[key]
            return result
    return {}


def pending_owner_key(
    unit_id: int,
    source_fragment_ids: Sequence[int],
    claimed_source_node_ids: Sequence[int],
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    """Return the canonical identity shared by pending rows and contracts."""

    return (
        int(unit_id),
        tuple(sorted({int(value) for value in list(source_fragment_ids or [])})),
        tuple(sorted({int(value) for value in list(claimed_source_node_ids or [])})),
    )


def build_pending_contract(
    *,
    unit_id: int,
    pending_kind: str,
    target_role: str,
    contract_kind: str,
    subphase: str,
    source_fragment_ids: Sequence[int],
    source_fragment_classes: Sequence[str],
    claimed_source_node_ids: Sequence[int],
    context_node_ids: Sequence[int],
    mainline_slots: Sequence[Mapping[str, Any]],
    branch_slots: Sequence[Mapping[str, Any]],
    required_counts: Mapping[str, int],
    expected_topology: str,
    reason: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one canonical pending contract without touching assembler state."""

    if not str(target_role):
        raise ValueError("canonical topology contract requires target_role")
    if not str(contract_kind):
        raise ValueError("canonical topology contract requires contract_kind")
    diagnostic_payload = dict(diagnostics or {})
    structural_details = structural_blocker_details(diagnostic_payload)
    blocker_kind = str(reason) if str(contract_kind) == "structural_blocker" else ""
    return {
        "contract_id": int(unit_id),
        "contract_kind": str(contract_kind),
        "target_role": str(target_role),
        "phase": external_phase_for_role(str(target_role)),
        "subphase": str(subphase),
        "source_fragment_ids": [int(value) for value in source_fragment_ids],
        "source_fragment_classes": [str(value) for value in source_fragment_classes],
        "claimed_source_node_ids": [int(value) for value in claimed_source_node_ids],
        "context_node_ids": [int(value) for value in context_node_ids],
        "mainline_slots": [dict(item or {}) for item in mainline_slots],
        "branch_slots": [dict(item or {}) for item in branch_slots],
        "required_counts": {
            str(key): int(value)
            for key, value in dict(required_counts or {}).items()
            if int(value) > 0
        },
        "raw_available_counts": {
            str(key): int(value)
            for key, value in dict(
                diagnostic_payload.get("raw_available_counts", {}) or {}
            ).items()
        },
        "claimable_available_counts": {
            str(key): int(value)
            for key, value in dict(
                diagnostic_payload.get("claimable_available_counts", {}) or {}
            ).items()
        },
        "protected_counts": {
            str(key): int(value)
            for key, value in dict(
                diagnostic_payload.get("protected_counts", {}) or {}
            ).items()
        },
        "excluded_node_ids": [
            int(value)
            for value in list(diagnostic_payload.get("excluded_node_ids", []) or [])
        ],
        "claim_trace": [
            dict(item or {})
            for item in list(diagnostic_payload.get("claim_trace", []) or [])
        ],
        "missing_counts": {
            str(key): int(value)
            for key, value in dict(
                diagnostic_payload.get("missing_counts", {}) or {}
            ).items()
            if int(value) > 0
        },
        "expected_topology": str(expected_topology or ""),
        "reason": str(reason or ""),
        "blocker_kind": str(blocker_kind),
        "failure_domain": (
            str(structural_details.get("failure_domain", ""))
            or str(diagnostic_payload.get("failure_domain", "") or "")
            or (
                "topology_materialization"
                if str(contract_kind) == "structural_blocker"
                else ""
            )
        ),
        **dict(structural_details),
        "diagnostics": {
            "public_pending_kind": str(pending_kind or ""),
            **diagnostic_payload,
        },
    }


def upsert_pending_contract(
    contracts: List[Dict[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    """Replace one owner-equivalent contract while preserving attempt history."""

    incoming = dict(contract or {})
    contract_key = stage3_contract_owner_key(incoming)
    for index, raw_existing in enumerate(list(contracts or [])):
        existing = dict(raw_existing or {})
        if stage3_contract_owner_key(existing) != contract_key:
            continue
        if existing != incoming:
            prior_attempts = list(
                dict(existing.get("diagnostics", {}) or {}).get(
                    "attempt_history", []
                )
                or []
            )
            prior_attempts.append({
                "reason": str(existing.get("reason", "") or ""),
                "target_role": str(existing.get("target_role", "") or ""),
                "required_counts": dict(existing.get("required_counts", {}) or {}),
                "mainline_slots": [
                    dict(slot or {})
                    for slot in list(existing.get("mainline_slots", []) or [])
                ],
            })
            incoming["diagnostics"] = {
                **dict(incoming.get("diagnostics", {}) or {}),
                "attempt_history": prior_attempts,
            }
            contracts[index] = incoming
        return
    contracts.append(incoming)


def pending_target_role(
    assembler,
    pending_kind: str,
    views: Sequence[AssemblyFragmentView],
) -> str:
    pending_kind_name = str(pending_kind or "")
    if pending_kind_name in {"b_like_open", "d_like_open"}:
        fragment_ids = {
            int(view.fragment_id)
            for view in list(views or [])
        }
        selected_for_flex = bool(
            fragment_ids
            and fragment_ids.issubset(
                set(
                    getattr(
                        assembler, "selected_flex_candidate_ids", set()
                    )
                    or set()
                )
            )
            and str(
                dict(getattr(assembler, "flex_audit", {}) or {}).get(
                    "scenario", ""
                )
                or ""
            ) == "below_range"
        )
        # B/D open fragments are dual-use.  The flex-range planner owns
        # the flex selection; every unselected fragment is closed later
        # as an ordinary side chain.
        return "flex_chain" if selected_for_flex else "side_chain"
    # These kinds are emitted by a concrete phase dispatcher.  A view may
    # advertise both capabilities (for example 24_AB_M is flex-preferred
    # but can fall back to a side chain), so do not infer the target from the
    # full capability set and report an artificial ambiguity.
    if pending_kind_name == "explicit_side":
        return "side_chain"
    if pending_kind_name == "explicit_flex":
        return "flex_chain"
    if pending_kind_name == "ring_only_like24":
        available_roles = {
            str(role)
            for view in list(views or [])
            for role in list(getattr(view, "target_roles", []) or [])
            if str(role)
        }
        # Once the ring planner has exhausted this candidate, a legal side
        # capability is the intended fallback.  During ring planning itself,
        # keep the contract owned by the fatty-ring phase.
        ring_pending = any(
            bool(getattr(view, "can_ring", False))
            and int(view.fragment_id)
            not in set(
                getattr(assembler, "ring_exhausted_fragment_ids", set()) or set()
            )
            for view in list(views or [])
        )
        if "side_chain" in available_roles and not ring_pending:
            return "side_chain"
        if "fatty_ring" in available_roles:
            return "fatty_ring"
    available_roles = {
        str(role)
        for view in list(views or [])
        for role in list(getattr(view, "target_roles", []) or [])
        if str(role)
    }
    required_role = ""
    if pending_kind_name == "pending_ring_candidate":
        required_role = "fatty_ring"
    elif pending_kind_name == "pending_su25":
        required_role = "side_chain"
    else:
        chain_roles = available_roles & {"flex_chain", "side_chain"}
        if len(chain_roles) == 1:
            required_role = next(iter(chain_roles))
    if required_role and required_role in available_roles:
        return str(required_role)
    raise ValueError(
        f"pending target role is undefined or ambiguous: kind={pending_kind} roles={sorted(available_roles)}"
    )


def canonical_contract_kind(
    pending_kind: str,
    reason: str,
    *,
    required_counts: Dict[str, int],
    has_branch_contracts: bool,
) -> str:
    reason_name = str(reason or "")
    if reason_name in {"fixed_branch_partner_missing", "fused_bridgehead_missing"}:
        return "ring_partner" if reason_name == "fixed_branch_partner_missing" else "structural_blocker"
    if str(pending_kind) == "pending_ring_candidate":
        return "mainline_closure" if dict(required_counts or {}) else "structural_blocker"
    if str(pending_kind) == "pending_su25":
        return "fixed_su25"
    if bool(has_branch_contracts) or "branch" in reason_name:
        return "branch_closure" if dict(required_counts or {}) else "structural_blocker"
    if dict(required_counts or {}):
        return "mainline_closure"
    return "structural_blocker"


def record_pending(
    assembler,
    *,
    unit_id: int,
    pending_kind: str,
    views: Sequence[AssemblyFragmentView],
    required_counts: Dict[str, int],
    available_counts: Dict[str, int],
    reason: str,
    partial_render: TopologyPathRender,
    actual_render: Optional[TopologyPathRender] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    target_role_override: Optional[str] = None,
    mainline_slots_override: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    actual = actual_render or partial_render
    diagnostic_payload = dict(diagnostics or {})
    allocator = getattr(assembler, "allocator", None)
    claimability = (
        dict(allocator.claimability_snapshot() or {})
        if allocator is not None
        and callable(getattr(allocator, "claimability_snapshot", None))
        else {}
    )
    raw_available_counts = {
        str(key): int(value)
        for key, value in dict(
            claimability.get("raw_available_counts", available_counts) or {}
        ).items()
    }
    claimable_available_counts = {
        str(key): int(value)
        for key, value in dict(
            claimability.get("claimable_available_counts", available_counts) or {}
        ).items()
    }
    protected_counts = {
        str(key): int(value)
        for key, value in dict(claimability.get("protected_counts", {}) or {}).items()
    }
    claim_trace = [
        dict(item or {})
        for item in list(diagnostic_payload.get("claim_trace", []) or [])
    ]
    trace_excluded_by_su: Dict[str, set[int]] = {}
    for row in claim_trace:
        if str(row.get("mode", "") or "") != "generic":
            continue
        su_key = str(row.get("su_type", ""))
        if not su_key.lstrip("-").isdigit():
            continue
        trace_excluded_by_su.setdefault(su_key, set()).update(
            int(value)
            for value in list(row.get("excluded_node_ids", []) or [])
        )
    excluded_node_ids = sorted({
        int(value)
        for value in list(claimability.get("excluded_node_ids", []) or [])
    } | {
        int(value)
        for values in trace_excluded_by_su.values()
        for value in values
    })
    # Candidate-specific exclusions exist only inside the materializer claim
    # scope, so the post-rollback allocator snapshot cannot see them.  Fold the
    # failed request's SU-local exclusion set into the contract projection.
    for su_key, values in trace_excluded_by_su.items():
        raw_count = int(raw_available_counts.get(str(su_key), 0) or 0)
        protected_count = int(protected_counts.get(str(su_key), 0) or 0)
        request_claimable = max(
            0,
            raw_count - protected_count - len(set(values)),
        )
        claimable_available_counts[str(su_key)] = min(
            int(claimable_available_counts.get(str(su_key), 0) or 0),
            int(request_claimable),
        )
    diagnostic_payload.update({
        "raw_available_counts": raw_available_counts,
        "claimable_available_counts": claimable_available_counts,
        "protected_counts": protected_counts,
        "excluded_node_ids": excluded_node_ids,
        "claim_trace": claim_trace,
    })
    # The contract's ``available_counts`` means resources the same generic
    # materializer request can actually claim. Raw inventory remains diagnostic.
    available_counts = dict(claimable_available_counts)
    structural_details = structural_blocker_details(diagnostic_payload)
    source_fragment_ids, source_fragment_classes, source_node_ids = _collect_source_metadata(list(views or []))
    fragment_by_id = {
        int(getattr(fragment, "fragment_id", -1)): fragment
        for fragment in list(getattr(assembler.classification, "fragments", []) or [])
    }
    rendered_branch_source_ids = {
        int(node_id)
        for view in list(views or [])
        for stub_map in (
            getattr(view, "branch_stubs_by_anchor", {}) or {},
            getattr(view, "real_branch_stubs_by_anchor", {}) or {},
        )
        for stubs in dict(stub_map or {}).values()
        for stub in list(stubs or [])
        for node_id in list(getattr(stub, "source_node_ids", []) or [])
    }
    claimed_source_node_ids = sorted(
        set(
            _fixed_source_ids_from_views(
                list(views or []),
                assembler=assembler,
            )
        )
        | rendered_branch_source_ids
    )
    if not claimed_source_node_ids:
        claimed_source_node_ids = sorted(
            set(int(value) for value in list(source_node_ids or []))
        )
    context_node_ids = sorted({
        int(node_id)
        for view in list(views or [])
        for node_id in list(getattr(view, "context_node_ids", []) or [])
    })
    public_pending_node_notation = re.sub(
        r"\d+",
        lambda match: (
            match.group(0)
            if int(match.group(0)) in set(claimed_source_node_ids)
            else "OPEN"
        ),
        str(actual.node_id_notation or ""),
    )
    rendered_pending_ids = {
        int(value)
        for value in re.findall(r"\d+", str(public_pending_node_notation or ""))
    }
    render_validation_errors: List[str] = []
    suppressed_support_node_ids = sorted(({
        int(value)
        for view in list(views or [])
        for value in list(getattr(view, "suppressed_support_node_ids", []) or [])
    } | {
        int(value)
        for value in list(getattr(actual, "suppressed_support_node_ids", []) or [])
    }) & set(claimed_source_node_ids))
    expected_render_ids = (
        set(claimed_source_node_ids) - set(suppressed_support_node_ids)
    )
    if expected_render_ids != rendered_pending_ids:
        missing_render_ids = sorted(expected_render_ids - rendered_pending_ids)
        extra_render_ids = sorted(
            rendered_pending_ids - set(claimed_source_node_ids)
        )
        render_validation_errors.append(
            f"render_ownership_mismatch:missing={missing_render_ids}:extra={extra_render_ids}"
        )
    completed_node_ids = {
        int(value)
        for product in list(assembler.products or [])
        for value in list(product.rendered_node_ids())
    }
    if set(claimed_source_node_ids) & completed_node_ids:
        return
    open_mainline_slots = (
        [dict(slot or {}) for slot in list(mainline_slots_override or [])]
        if mainline_slots_override is not None
        else [
            slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
            for fragment_id in source_fragment_ids
            for slot in list(getattr(fragment_by_id.get(int(fragment_id)), "open_mainline_slots", []) or [])
        ]
    )
    branch_slots = [
        slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
        for fragment_id in source_fragment_ids
        for slot in list(getattr(fragment_by_id.get(int(fragment_id)), "branch_contracts", []) or [])
    ]
    intrinsic_failure = (
        diagnostic_payload.get("failure_domain") == "topology_materialization"
        and diagnostic_payload.get("failure_scope") == "current_contract"
    )
    if intrinsic_failure:
        # These slots cannot be executed. Do not resurrect their resource
        # demand merely because the fragment still carries an older slot.
        required_counts = {}
        open_mainline_slots = []
        branch_slots = []
    slot_counts = _slot_required_counts(open_mainline_slots + branch_slots)
    if str(reason) in {"fixed_branch_partner_missing", "fused_bridgehead_missing"}:
        required_counts = {}
    elif (
        slot_counts
        and not dict(required_counts or {})
        # Generic F1/H slots describe fragment closure context.  They are
        # not a fatty-ring resource contract until a complete six-member
        # ring template has actually been selected.
        and str(pending_kind) != "pending_ring_candidate"
    ):
        required_counts = dict(slot_counts)
    target_role = (
        str(target_role_override)
        if str(target_role_override or "")
        else pending_target_role(assembler, str(pending_kind), list(views or []))
    )
    contract_kind = canonical_contract_kind(
        str(pending_kind),
        str(reason),
        required_counts=dict(required_counts or {}),
        has_branch_contracts=bool(branch_slots),
    )
    if intrinsic_failure:
        contract_kind = "structural_blocker"
    missing_counts = {
        str(key): max(
            0,
            int(required_counts.get(str(key), 0))
            - int(available_counts.get(str(key), 0)),
        )
        for key in {str(k) for k in dict(required_counts or {}).keys()}
    }
    explicit_failure_domain = str(
        diagnostic_payload.get("failure_domain", "") or ""
    )
    shortage_reason = any(
        token in str(reason or "").lower()
        for token in ("shortage", "resource", "pool", "missing")
    )
    trace_missing_counts: Dict[str, int] = {}
    for row in claim_trace:
        if bool(row.get("success", False)) or str(row.get("mode", "")) != "generic":
            continue
        su_key = str(row.get("su_type", ""))
        if not su_key.lstrip("-").isdigit():
            continue
        shortage = max(
            0,
            int(row.get("requested_count", 0) or 0)
            - int(row.get("claimable_available_count", 0) or 0),
        )
        if shortage > 0:
            trace_missing_counts[su_key] = max(
                int(trace_missing_counts.get(su_key, 0)), int(shortage)
            )
    diagnostic_payload["claim_failure_missing_counts"] = dict(trace_missing_counts)
    if shortage_reason and trace_missing_counts and not any(missing_counts.values()):
        for key, shortage in trace_missing_counts.items():
            if int(protected_counts.get(str(key), 0) or 0) > 0:
                continue
            required_counts[str(key)] = max(
                int(required_counts.get(str(key), 0)),
                int(claimable_available_counts.get(str(key), 0)) + int(shortage),
            )
        missing_counts = {
            str(key): max(
                0,
                int(required_counts.get(str(key), 0))
                - int(available_counts.get(str(key), 0)),
            )
            for key in {str(k) for k in dict(required_counts or {}).keys()}
        }
    diagnostic_payload["missing_counts"] = {
        str(key): int(value)
        for key, value in missing_counts.items()
        if int(value) > 0
    }
    if not explicit_failure_domain and shortage_reason and trace_missing_counts:
        if any(
            int(protected_counts.get(str(key), 0) or 0) > 0
            for key in trace_missing_counts
        ):
            explicit_failure_domain = "protected_resource"
        elif any(
            list(row.get("excluded_node_ids", []) or [])
            for row in claim_trace
            if not bool(row.get("success", False))
            and str(row.get("mode", "")) == "generic"
        ):
            explicit_failure_domain = "ownership_constraint"
    if not explicit_failure_domain and any(
        int(value) > 0 for value in missing_counts.values()
    ) and shortage_reason:
        raw_could_cover = all(
            int(raw_available_counts.get(str(key), 0)) >= int(value)
            for key, value in dict(required_counts or {}).items()
            if int(value) > 0
        )
        if raw_could_cover and protected_counts:
            explicit_failure_domain = "protected_resource"
        elif raw_could_cover and excluded_node_ids:
            explicit_failure_domain = "ownership_constraint"
        else:
            explicit_failure_domain = "resource_capacity"
    if not explicit_failure_domain and contract_kind == "structural_blocker":
        explicit_failure_domain = "topology_materialization"
    if not explicit_failure_domain and shortage_reason and required_counts:
        explicit_failure_domain = "resource_capacity"
    if explicit_failure_domain:
        diagnostic_payload["failure_domain"] = str(explicit_failure_domain)
    pending_key = pending_owner_key(
        unit_id,
        source_fragment_ids,
        claimed_source_node_ids,
    )
    for existing in list(assembler.pending_units or []):
        existing_key = pending_owner_key(
            int(getattr(existing, "unit_id", 0)),
            list(getattr(existing, "source_fragment_ids", []) or []),
            list(existing.claimed_source_node_ids or []),
        )
        if existing_key == pending_key:
            existing.attempt_reasons = sorted({
                str(value)
                for value in list(getattr(existing, "attempt_reasons", []) or [])
                + [str(getattr(existing, "reason", "") or ""), str(reason or "")]
                if str(value)
            })
            existing.validation_errors = sorted({
                str(value)
                for value in list(getattr(existing, "validation_errors", []) or [])
                + list(render_validation_errors or [])
                if str(value)
            })
            existing.required_counts = {
                str(key): int(value) for key, value in dict(required_counts or {}).items()
            }
            existing.available_counts = {
                str(key): int(value) for key, value in dict(available_counts or {}).items()
            }
            existing.raw_available_counts = dict(raw_available_counts)
            existing.claimable_available_counts = dict(claimable_available_counts)
            existing.protected_counts = dict(protected_counts)
            existing.excluded_node_ids = list(excluded_node_ids)
            existing.claim_trace = list(claim_trace)
            existing.contract_kind = str(contract_kind)
            existing.target_role = str(target_role)
            existing.phase = external_phase_for_role(str(target_role))
            existing.subphase = str(pending_subphase(assembler, pending_kind, reason))
            existing.open_mainline_slots = [dict(slot or {}) for slot in open_mainline_slots]
            existing.branch_slots = [dict(slot or {}) for slot in branch_slots]
            existing.reason = str(reason)
            existing.infeasibility_proof = {
                **dict(getattr(existing, "infeasibility_proof", {}) or {}),
                "reason": str(reason),
                "required_counts": dict(existing.required_counts),
                "available_counts": dict(existing.available_counts),
                "raw_available_counts": dict(raw_available_counts),
                "claimable_available_counts": dict(claimable_available_counts),
                "protected_counts": dict(protected_counts),
                "excluded_node_ids": list(excluded_node_ids),
                "claim_trace": list(claim_trace),
                "missing_counts": {
                    str(key): int(value)
                    for key, value in missing_counts.items()
                    if int(value) > 0
                },
                "suppressed_support_node_ids": list(suppressed_support_node_ids),
                "validation_errors": list(existing.validation_errors),
                "failure_domain": (
                    str(structural_details.get("failure_domain", ""))
                    or str(explicit_failure_domain)
                    or (
                        "topology_materialization"
                        if str(contract_kind) == "structural_blocker"
                        else ""
                    )
                ),
                **dict(structural_details),
                **({"diagnostics": diagnostic_payload} if diagnostic_payload else {}),
            }
            record_pending_contract(assembler, 
                unit_id=int(unit_id),
                pending_kind=str(pending_kind),
                views=list(views or []),
                required_counts=dict(required_counts or {}),
                reason=str(reason),
                target_role=str(target_role),
                contract_kind=str(contract_kind),
                expected_topology=str(actual.topology_notation or ""),
                mainline_slots=open_mainline_slots,
                branch_slots=branch_slots,
                claimed_source_node_ids=claimed_source_node_ids,
                context_node_ids=context_node_ids,
                diagnostics=diagnostic_payload,
            )
            return
    record_pending_contract(assembler, 
        unit_id=int(unit_id),
        pending_kind=str(pending_kind),
        views=list(views or []),
        required_counts=dict(required_counts or {}),
        reason=str(reason),
        target_role=str(target_role),
        contract_kind=str(contract_kind),
        expected_topology=str(actual.topology_notation or ""),
        mainline_slots=open_mainline_slots,
        branch_slots=branch_slots,
        claimed_source_node_ids=claimed_source_node_ids,
        context_node_ids=context_node_ids,
        diagnostics=diagnostic_payload,
    )
    assembler.pending_units.append(TopologyPendingUnit(
        unit_id=int(unit_id),
        pending_kind=str(pending_kind),
        contract_kind=str(contract_kind),
        target_role=str(target_role),
        real_topology_partial=str(actual.topology_notation),
        real_node_partial=str(public_pending_node_notation),
        source_fragment_ids=source_fragment_ids,
        source_fragment_classes=source_fragment_classes,
        claimed_source_node_ids=claimed_source_node_ids,
        phase=external_phase_for_role(target_role),
        subphase=str(pending_subphase(assembler, pending_kind, reason)),
        required_counts={str(key): int(value) for key, value in dict(required_counts or {}).items()},
        available_counts={str(key): int(value) for key, value in dict(available_counts or {}).items()},
        raw_available_counts=dict(raw_available_counts),
        claimable_available_counts=dict(claimable_available_counts),
        protected_counts=dict(protected_counts),
        excluded_node_ids=list(excluded_node_ids),
        claim_trace=list(claim_trace),
        reason=str(reason),
        context_node_ids=context_node_ids,
        open_mainline_slots=open_mainline_slots,
        branch_slots=branch_slots,
        infeasibility_proof={
            "reason": str(reason),
            "required_counts": {str(key): int(value) for key, value in dict(required_counts or {}).items()},
            "available_counts": {str(key): int(value) for key, value in dict(available_counts or {}).items()},
            "raw_available_counts": dict(raw_available_counts),
            "claimable_available_counts": dict(claimable_available_counts),
            "protected_counts": dict(protected_counts),
            "excluded_node_ids": list(excluded_node_ids),
            "claim_trace": list(claim_trace),
            "missing_counts": {str(key): int(value) for key, value in missing_counts.items() if int(value) > 0},
            "suppressed_support_node_ids": list(suppressed_support_node_ids),
            "validation_errors": list(render_validation_errors),
            "failure_domain": (
                str(structural_details.get("failure_domain", ""))
                or str(explicit_failure_domain)
                or (
                    "topology_materialization"
                    if str(contract_kind) == "structural_blocker"
                    else ""
                )
            ),
            **dict(structural_details),
            **({"diagnostics": diagnostic_payload} if diagnostic_payload else {}),
        },
        attempt_reasons=[str(reason)] if str(reason) else [],
        validation_errors=list(render_validation_errors),
    ))


def consolidate_pending_owners(assembler: Any) -> None:
    canonical: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], TopologyPendingUnit] = {}
    for pending in list(assembler.pending_units or []):
        key = pending_owner_key(
            int(getattr(pending, "unit_id", 0)),
            list(getattr(pending, "source_fragment_ids", []) or []),
            list(pending.claimed_source_node_ids or []),
        )
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = pending
        elif existing.to_dict() != pending.to_dict():
            assembler.contract_definition_conflict_count += 1
            assembler.last_contract_definition_conflict = {
                "reason": "duplicate_pending_definition",
                "owner_key": list(key),
                "existing": existing.to_dict(),
                "incoming": pending.to_dict(),
            }
    assembler.pending_units = list(canonical.values())


def pending_subphase(
    assembler,
    pending_kind: str,
    reason: str,
) -> str:
    kind = str(pending_kind or "")
    reason_name = str(reason or "")
    if kind == "pending_ring_candidate":
        return "real_like24_ring_assembly"
    if kind == "explicit_flex":
        return "explicit_branch_closure"
    if kind == "explicit_side":
        return "remaining_external_halfclosed_closure"
    if reason_name in {"fixed_branch_H1_missing", "fixed_branch_role_conflict"}:
        return "explicit_branch_closure"
    if reason_name == "fixed_branch_partner_missing":
        return "real_like24_ring_assembly"
    return "remaining_external_halfclosed_closure"


def view_external_pending_profile(
    view: Optional[AssemblyFragmentView],
) -> Dict[str, Any]:
    if view is None:
        return {}
    evidence = dict(getattr(view, "evidence", {}) or {})
    return dict(evidence.get("external_pending_profile", {}) or {})


def record_pending_contract(
    assembler,
    *,
    unit_id: int,
    pending_kind: str,
    views: Sequence[AssemblyFragmentView],
    required_counts: Dict[str, int],
    reason: str,
    target_role: str,
    contract_kind: str,
    expected_topology: str,
    mainline_slots: Sequence[Dict[str, Any]],
    branch_slots: Sequence[Dict[str, Any]],
    claimed_source_node_ids: Sequence[int],
    context_node_ids: Sequence[int],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    source_fragment_ids, source_fragment_classes, _ = _collect_source_metadata(list(views or []))
    contract = build_pending_contract(
        unit_id=int(unit_id),
        pending_kind=str(pending_kind),
        target_role=str(target_role),
        contract_kind=str(contract_kind),
        subphase=pending_subphase(assembler, str(pending_kind), str(reason)),
        source_fragment_ids=source_fragment_ids,
        source_fragment_classes=source_fragment_classes,
        claimed_source_node_ids=claimed_source_node_ids,
        context_node_ids=context_node_ids,
        mainline_slots=mainline_slots,
        branch_slots=branch_slots,
        required_counts=required_counts,
        expected_topology=str(expected_topology or ""),
        reason=str(reason or ""),
        diagnostics=diagnostics,
    )
    upsert_pending_contract(assembler.pending_contracts, contract)


__all__ = [
    "build_pending_contract",
    "canonical_contract_kind",
    "consolidate_pending_owners",
    "pending_owner_key",
    "pending_subphase",
    "pending_target_role",
    "record_pending",
    "record_pending_contract",
    "structural_blocker_details",
    "upsert_pending_contract",
    "view_external_pending_profile",
]
