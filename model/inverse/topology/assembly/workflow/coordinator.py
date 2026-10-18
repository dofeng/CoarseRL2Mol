"""Authoritative external topology assembly phase ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .....shared.aromatic_plan import MAX_FLEX_WINDOW_WIDTH

from ..closure import chain as chain_materializer_module
from ..closure.flex import run_external_flex_phase
from ..state import ownership as ownership_module
from ..ring import planner as ring_planner_module
from ..state.pending import consolidate_pending_owners


@dataclass(frozen=True)
class ExternalAssemblyPhaseState:
    external_subphases: List[str] = field(default_factory=list)
    flex_violation_reason: Optional[str] = None
    ring_resource_blockers: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    ring_materialization_blockers: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def run_external_assembly_phases(assembler: Any) -> ExternalAssemblyPhaseState:
    """Run or continue flex, ring, and side phases in their only order.

    A generation keeps one mutable assembler.  Once a phase has completed, a
    later conversion/recovery retry starts at the first pending phase instead
    of rebuilding products from the classifier snapshot.
    """

    flex_enforced = bool(dict(assembler.flex_audit or {}).get("enforced", False))
    wide_window = bool(
        flex_enforced
        and int(assembler.flex_audit.get("flex_window_width", 0)) > MAX_FLEX_WINDOW_WIDTH
    )
    flex_state = run_external_flex_phase(
        assembler,
        complete_quantity=not wide_window or bool(assembler.flex_audit.get("stop_after_flex", False)),
    )
    external_subphases = list(flex_state.subphases)
    completed = set(assembler.completed_phases)
    flex_stage_ok = flex_state.ready
    flex_violation_reason = chain_materializer_module.flex_range_violation_reason(assembler)
    if assembler.flex_audit.get("stop_after_flex", False) or wide_window or not flex_stage_ok:
        assembler.current_phase = "flex_range_reconciliation" if wide_window or assembler.flex_audit.get("stop_after_flex", False) else "external_flex_closure"
        consolidate_pending_owners(assembler)
        return ExternalAssemblyPhaseState(
            external_subphases=external_subphases,
            flex_violation_reason=flex_violation_reason,
        )
    branch_donors = chain_materializer_module.available_branch_donors(assembler)

    ring_stage_attempted = False
    recovery_applied = False
    ring_pending = bool(
        ring_planner_module.ring_resource_blockers(assembler)
        or ring_planner_module.ring_materialization_blockers(assembler)
        or chain_materializer_module.has_pending_contracts_for_subphase(
            assembler, "real_like24_ring_assembly"
        )
    )
    if flex_stage_ok and flex_violation_reason is None and (
        ring_pending or "real_like24_ring_assembly" not in completed
    ):
        external_subphases.append("real_like24_ring_assembly")
        ring_stage_attempted = True
        ring_planner_module.assemble_priority_rings(assembler, branch_donors)
        # Recovery is a controlled second attempt for both fatty/SU24 rings
        # and side chains.  It is only requested by Stage3 after the ordinary
        # conversion planner has proved the resource path infeasible.
        ring_resource_blockers = ring_planner_module.ring_resource_blockers(assembler)
        if (
            ring_resource_blockers
            and bool(getattr(assembler, "allow_ring_recovery", False))
        ):
            # Stage3 has already supplied the capacity proof. Execute that
            # authorization exactly once; the assembler must not infer a
            # second recovery from an approximate inventory test.
            recovery_applied = bool(
                chain_materializer_module.recover_one_pure_resource_ring(assembler)
            )
            if recovery_applied:
                ring_planner_module.assemble_priority_rings(
                    assembler, branch_donors
                )
    ring_resource_blockers = ring_planner_module.ring_resource_blockers(assembler)
    ring_materialization_blockers = (
        ring_planner_module.ring_materialization_blockers(assembler)
    )
    blocking_ring_contracts = [
        dict(contract or {})
        for contract in list(assembler.pending_contracts or [])
        if str(dict(contract or {}).get("subphase", "") or "")
        == "real_like24_ring_assembly"
        and any(
            int(value) > 0
            for value in dict(
                dict(contract or {}).get("required_counts", {}) or {}
            ).values()
        )
    ]
    # A continuation may legitimately skip the ring planner because this
    # generation already completed the ring phase.  The old ``attempted``
    # guard made such a continuation report ``ring_stage_ok=False`` and,
    # consequently, never reach the pending side-chain contract.  Treat an
    # already-completed ring phase as successful as long as no blocker is
    # currently present; a fresh generation still enters the planner because
    # the phase is absent from ``completed`` above.
    ring_stage_ok = bool(
        (ring_stage_attempted or "real_like24_ring_assembly" in completed)
        and not blocking_ring_contracts
        and not ring_resource_blockers
        and not ring_materialization_blockers
    )
    if ring_stage_ok:
        completed.add("real_like24_ring_assembly")
        # Side-chain fallback predicates read ``assembler.completed_phases`` to
        # decide whether a dual-use ring candidate has been released from the
        # ring-first stage.  Keep the runtime state synchronized before
        # entering ``remaining_external_halfclosed_closure``; otherwise a
        # valid B/D/24_D side fallback can be skipped and later recorded as a
        # spurious half-open pending contract.
        assembler.completed_phases = set(completed)
    if flex_stage_ok and ring_stage_ok:
        side_pending = bool(
            chain_materializer_module.has_pending_contracts_for_subphase(
                assembler, "remaining_external_halfclosed_closure"
            )
            or chain_materializer_module.available_remaining_side_views(
                assembler, require_generic_join=False
            )
        )
        if side_pending or "remaining_external_halfclosed_closure" not in completed:
            external_subphases.append("remaining_external_halfclosed_closure")
            chain_materializer_module.retry_ready_pending_contracts(
                assembler,
                branch_donors,
                target_role="side_chain",
            )
            if (
                not recovery_applied
                and side_pending
                and bool(
                    getattr(assembler, "allow_ring_recovery", False)
                )
            ):
                recovery_applied = bool(
                    chain_materializer_module.recover_one_pure_resource_ring(
                        assembler
                    )
                )
            chain_materializer_module.assemble_remaining_side_products(
                assembler,
                branch_donors,
            )
        if not chain_materializer_module.has_pending_contracts_for_subphase(
            assembler, "remaining_external_halfclosed_closure"
        ):
            completed.add("remaining_external_halfclosed_closure")

    if ownership_module.available_views(assembler, lambda item: True):
        chain_materializer_module.record_remaining_external_pending(assembler)
    consolidate_pending_owners(assembler)
    assembler.completed_phases = completed
    if assembler.pending_contracts:
        assembler.current_phase = str(
            dict(assembler.pending_contracts[0] or {}).get(
                "subphase", "external_topology"
            )
            or "external_topology"
        )
    else:
        assembler.current_phase = "complete"
    return ExternalAssemblyPhaseState(
        external_subphases=external_subphases,
        flex_violation_reason=flex_violation_reason,
        ring_resource_blockers=ring_resource_blockers,
        ring_materialization_blockers=ring_materialization_blockers,
    )


__all__ = ["ExternalAssemblyPhaseState", "run_external_assembly_phases"]
