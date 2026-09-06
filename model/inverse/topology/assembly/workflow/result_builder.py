"""Build the immutable public result from completed assembly phases."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from ...resources.types import EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
from ...stage3_models import (
    Stage3OwnershipState,
    TopologyAssemblyResult,
    merge_stage3_contracts_by_owner,
)
from ..closure import chain as chain_materializer_module
from ..state import ownership as ownership_module
from .coordinator import ExternalAssemblyPhaseState
from ..projection.engine import _slot_required_counts
from ..state.round_context import (
    build_stage3_round_context,
    claim_round_product,
    mark_round_pending_contracts,
    normalize_stage3_product,
    ownership_public_view,
    round_context_invariants,
)
from ..core.rules import external_phase_for_role as _external_phase_for_role


def build_external_assembly_result(
    self: Any,
    phase_state: ExternalAssemblyPhaseState,
) -> TopologyAssemblyResult:
    external_subphases = list(phase_state.external_subphases)
    deferred_flex_shortfall = bool(phase_state.deferred_flex_shortfall)
    flex_violation_reason = phase_state.flex_violation_reason
    ring_resource_blockers = dict(phase_state.ring_resource_blockers)
    ring_materialization_blockers = dict(
        phase_state.ring_materialization_blockers
    )

    completed_product_node_ids = {
        int(node_id)
        for product in list(self.products or [])
        for node_id in (
            list(getattr(product, "fixed_source_node_ids", []) or [])
            + [
                int(value)
                for values in dict(getattr(product, "claimed_resource_ids_by_su", {}) or {}).values()
                for value in list(values or [])
            ]
        )
    }
    completed_product_fragment_ids = {
        int(fragment_id)
        for product in list(self.products or [])
        for fragment_id in list(getattr(product, "source_fragment_ids", []) or [])
    }
    ownership_module.invalidate_pending_identity(self, 
        fragment_ids=completed_product_fragment_ids,
        claimed_node_ids=completed_product_node_ids,
    )
    ledger = self.allocator.build_ledger({str(k): int(v) for k, v in self.shortage_counts.items()})
    explicit_flex_count = int(self.explicit_flex_count)
    final_flex_count = int(chain_materializer_module.current_flex_product_count(self, ))
    flex_target_plan = {
        "flex_lower": int(chain_materializer_module.flex_limit(self, "min")),
        "flex_upper": int(chain_materializer_module.flex_limit(self, "max")),
        "flex_target": int(dict(self.flex_bounds or {}).get("target", chain_materializer_module.flex_limit(self, "min")) or 0),
        "explicit_flex_count": int(explicit_flex_count),
        "minimum_gap": max(
            0,
            int(chain_materializer_module.flex_limit(self, "min"))
            - int(explicit_flex_count)
        ),
        "preferred_gap": max(
            0,
            int(dict(self.flex_bounds or {}).get("target", 0) or 0)
            - int(explicit_flex_count)
        ),
        "maximum_capacity": max(
            0,
            int(chain_materializer_module.flex_limit(self, "max"))
            - int(explicit_flex_count)
        ),
        "external_candidate_capacity": int(
            dict(self.flex_audit or {}).get("external_candidate_capacity", 0) or 0
        ),
        "external_planned_addition": int(
            dict(self.flex_audit or {}).get("external_planned_addition", 0) or 0
        ),
        "external_symbolic_addition": max(0, int(final_flex_count) - int(explicit_flex_count)),
        "external_capacity_exhausted": bool(
            dict(self.flex_audit or {}).get("external_capacity_exhausted", False)
            or deferred_flex_shortfall
        ),
        "external_shortfall_deferred_to_residual": bool(deferred_flex_shortfall),
        "remaining_minimum_gap": max(0, int(chain_materializer_module.flex_limit(self, "min")) - int(final_flex_count)),
        "remaining_preferred_gap": max(0, int(dict(self.flex_bounds or {}).get("target", 0) or 0) - int(final_flex_count)),
    }
    minimum_flex_gap = int(flex_target_plan["remaining_minimum_gap"])
    protected_counter: Counter = Counter()
    canonical_pending_contracts = merge_stage3_contracts_by_owner([
        dict(contract or {})
        for contract in list(self.pending_contracts or [])
        if dict(contract or {})
    ])
    for contract in canonical_pending_contracts:
        required_counts = {
            str(key): max(0, int(value))
            for key, value in dict(
                contract.get("required_counts", {}) or {}
            ).items()
            if str(key).lstrip("-").isdigit() and int(value) > 0
        }
        if not required_counts:
            required_counts = _slot_required_counts(
                list(contract.get("mainline_slots", []) or [])
                + list(contract.get("branch_slots", []) or [])
            )
        protected_counter.update(required_counts)
    if minimum_flex_gap > 0:
        protected_counter["11"] += 2 * int(minimum_flex_gap)
        protected_counter["23"] += 2 * int(minimum_flex_gap)
    protected_resource_counts = {
        str(key): int(value)
        for key, value in protected_counter.items()
        if int(value) > 0
    }
    flex_above_max = bool(chain_materializer_module.flex_exceeds_max(self, ))
    pending_contracts = [
        dict(item or {})
        for item in list(self.pending_contracts or [])
        if dict(item or {})
    ]
    has_external_pending = bool(self.pending_units)
    has_shortage_contract = any(
        int(value) > 0
        for contract in list(pending_contracts or [])
        for value in dict(contract.get("required_counts", {}) or {}).values()
    ) and any(int(value) > 0 for value in dict(self.shortage_counts or {}).values())
    ledger = ownership_module.reconcile_resource_ownership(self, ledger)
    ledger.residual_consumed_counts = {str(key): 0 for key in ledger.consumed_ids_by_su}
    ledger.topology_consumed_counts = {
        str(key): max(
            0,
            int(ledger.consumed_counts.get(str(key), 0) or 0)
            - int(ledger.residual_consumed_counts.get(str(key), 0) or 0),
        )
        for key in set(ledger.initial_counts) | set(ledger.consumed_counts)
    }
    ledger.snapshot_stage = "post_external_final"
    role_priority = {"flex_chain": 0, "fatty_ring": 1, "side_chain": 2}
    active_pending_contract = min(
        list(pending_contracts or []),
        key=lambda item: (
            int(role_priority.get(str(dict(item or {}).get("target_role", "")), 3)),
            int(dict(item or {}).get("contract_id", 10**12) or 10**12),
        ),
        default={},
    )
    active_external_phase = _external_phase_for_role(
        str(dict(active_pending_contract or {}).get("target_role", "") or "")
    )
    if str(flex_violation_reason) == "flex_range_below_minimum":
        result_reason = "external_flex_range_below_minimum"
    elif flex_violation_reason is not None or flex_above_max:
        result_reason = "external_flex_range_exceeded"
    elif not has_external_pending:
        result_reason = "remaining_pool_spectrum_rebalance_ready"
    elif has_shortage_contract:
        result_reason = "external_pool_shortage_pending"
    elif any(str(unit.pending_kind) == "pending_ring_candidate" for unit in list(self.pending_units or [])):
        result_reason = "external_ring_candidates_pending"
    else:
        result_reason = f"{active_external_phase}_pending"
    if flex_violation_reason is not None or flex_above_max:
        stage3_phase = "flex_range_reconciliation"
    elif has_external_pending:
        stage3_phase = str(active_external_phase)
    elif not has_external_pending:
        stage3_phase = "remaining_pool_spectrum_rebalance"
    else:
        stage3_phase = "complete"
    completed_phases: List[str] = [
        str(value)
        for value in list(getattr(self, "completed_phases", set()) or set())
    ]
    if str(stage3_phase) != "flex_range_reconciliation":
        if "flex_range_reconciliation" not in completed_phases:
            completed_phases.append("flex_range_reconciliation")
        for phase_name in (
            "external_flex_closure",
            "external_fatty_ring_closure",
            "external_side_chain_closure",
        ):
            if has_external_pending and str(phase_name) == str(stage3_phase):
                break
            if str(phase_name) not in completed_phases:
                completed_phases.append(str(phase_name))
    completed_phases = sorted(set(completed_phases))
    diagnostics = {
        "explicit_flex_count": int(explicit_flex_count),
        "total_shortage": {str(k): int(v) for k, v in self.shortage_counts.items()},
        "stage3_phase": stage3_phase,
        "external_subphases": list(external_subphases),
        "active_external_subphase": (
            str(stage3_phase)
            if stage3_phase != "remaining_pool_spectrum_rebalance"
            else "remaining_pool_spectrum_rebalance"
        ),
        "flex_audit": dict(self.flex_audit or {}),
        "pending_contracts": list(pending_contracts),
        "external_subphase_status": chain_materializer_module.external_subphase_status_snapshot(self, ),
        "completed_phases": list(completed_phases),
        "protected_resource_counts": dict(protected_resource_counts),
        "ledger_invariant_errors": list(ledger.invariant_errors or []),
        "remaining_minimum_flex_gap": int(flex_target_plan["remaining_minimum_gap"]),
        "remaining_preferred_flex_gap": int(flex_target_plan["remaining_preferred_gap"]),
        "product_ownership_conflict_count": int(
            self.product_ownership_conflict_count
        ),
        "last_product_ownership_conflict": dict(
            self.last_product_ownership_conflict or {}
        ),
        "composition_attempt_failure_count": int(
            self.composition_attempt_failure_count
        ),
        "last_composition_attempt_failure": dict(
            self.last_composition_attempt_failure or {}
        ),
        "last_materialization_failure_by_fragment_id": {
            str(fragment_id): dict(failure or {})
            for fragment_id, failure in dict(
                self.materialization_failure_by_fragment_id or {}
            ).items()
        },
        "contract_definition_conflict_count": int(
            self.contract_definition_conflict_count
        ),
        "last_contract_definition_conflict": dict(
            self.last_contract_definition_conflict or {}
        ),
        "ring_resource_blockers": {
            str(key): dict(value or {})
            for key, value in dict(ring_resource_blockers or {}).items()
        },
        "ring_materialization_blockers": {
            str(key): dict(value or {})
            for key, value in dict(
                ring_materialization_blockers or {}
            ).items()
        },
    }
    diagnostics["invalid_topology_count"] = int(self.invalid_topology_count)
    if self.last_invalid_topology:
        diagnostics["last_invalid_topology"] = dict(
            self.last_invalid_topology
        )
    released_resource_ids_by_su = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(
            getattr(self, "released_resource_ids_by_su", {}) or {}
        ).items()
        if list(values or [])
    }
    recycled_su24_to_23_ids = sorted({
        int(value)
        for value in list(
            getattr(self, "recycled_su24_to_23_ids", set()) or set()
        )
    })
    recycled_resource_conversion_ops = [
        dict(item or {})
        for item in list(
            getattr(self, "recycled_resource_conversion_ops", []) or []
        )
        if dict(item or {})
    ]
    diagnostics["recycled_su24_to_23_ids"] = list(recycled_su24_to_23_ids)
    diagnostics["recycled_resource_conversion_ops"] = [
        dict(item or {}) for item in recycled_resource_conversion_ops
    ]
    # The classifier is a fresh-generation snapshot. During continuation the
    # live allocator is authoritative for concrete resources after ordinary
    # SU conversions; overlay its current buckets before rebuilding context.
    context_classification = deepcopy(self.classification)
    live_allocator = getattr(self, "allocator", None)
    live_remaining = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(
            getattr(live_allocator, "remaining_ids_by_su", {}) or {}
        ).items()
    }
    if live_remaining:
        context_pool = deepcopy(
            dict(getattr(context_classification, "resource_pool", {}) or {})
        )
        context_pool["remaining_ids_by_su"] = dict(live_remaining)
        context_pool["su11_donor_ids_by_su"] = {
            str(su_type): list(live_remaining.get(str(su_type), []) or [])
            for su_type in (12, 13)
        }
        context_classification.resource_pool = context_pool
    round_context = build_stage3_round_context(
        nodes=self.nodes,
        classification=context_classification,
        flex_audit=self.flex_audit,
        generation_id=int(self.generation_id),
        released_resource_ids_by_su=released_resource_ids_by_su,
        recycled_su24_to_23_ids=recycled_su24_to_23_ids,
    )
    for product_index, raw_product in enumerate(list(self.products or [])):
        symbolic_product = normalize_stage3_product(
            raw_product,
            generation_id=int(self.generation_id),
            index=product_index,
        )
        claimed_context, claim_proof = claim_round_product(round_context, symbolic_product)
        if claimed_context is None:
            round_context.conflicts.append({
                "reason": "external_product_ownership_conflict",
                "product_id": str(symbolic_product.product_id),
                "proof": dict(claim_proof or {}),
            })
            continue
        round_context = claimed_context
    round_context = mark_round_pending_contracts(
        round_context,
        [
            pending.to_ownership_dict()
            if hasattr(pending, "to_ownership_dict")
            else (pending.to_dict() if hasattr(pending, "to_dict") else dict(pending or {}))
            for pending in list(self.pending_units or [])
        ],
    )
    for atomic_group in list(getattr(self.classification, "atomic_groups", []) or []):
        group = dict(atomic_group or {})
        group_node_ids = [int(value) for value in list(group.get("node_ids", []) or [])]
        group_owners = [round_context.ownership_by_node_id.get(int(node_id)) for node_id in group_node_ids]
        if not group_node_ids or any(owner is None for owner in group_owners):
            round_context.conflicts.append({
                "reason": "atomic_group_node_missing",
                "atomic_group_id": int(group.get("group_id", -1)),
                "node_ids": group_node_ids,
            })
            continue
        identities = {
            (
                str(owner.state),
                str(owner.owner_product_id or ""),
                owner.owner_contract_id,
                owner.owner_fragment_id,
            )
            for owner in group_owners
            if owner is not None
        }
        if len(identities) != 1:
            round_context.conflicts.append({
                "reason": "atomic_group_identity_split",
                "atomic_group_id": int(group.get("group_id", -1)),
                "kind": str(group.get("kind", "")),
                "node_ids": group_node_ids,
                "identities": [
                    {
                        "node_id": int(owner.node_id),
                        "state": str(owner.state),
                        "owner_product_id": str(owner.owner_product_id or ""),
                        "owner_contract_id": owner.owner_contract_id,
                        "owner_fragment_id": owner.owner_fragment_id,
                    }
                    for owner in group_owners
                    if owner is not None
                ],
            })
    ownership_view = ownership_public_view(round_context)
    ownership_states = dict(ownership_view.get("states", {}) or {})
    ownership_remaining_ids = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in {
            **dict(
                ownership_view.get("generic_closure_pool_ids_by_su", {})
                or {}
            ),
            "24": list(
                ownership_view.get("su24_ring_resource_ids", []) or []
            ),
            **dict(ownership_view.get("aromatic_donor_ids_by_su", {}) or {}),
        }.items()
    }
    ledger.remaining_ids_by_su = deepcopy(ownership_remaining_ids)
    ledger.convertible_ids_by_su = {
        str(su_type): list(
            ownership_remaining_ids.get(str(su_type), []) or []
        )
        for su_type in EXTERNAL_ADJUSTMENT_RESOURCE_TYPES
    }
    # ``released_su24_ids`` remains provenance only.  Recycled SU24 nodes are
    # already represented in the ordinary SU23 bucket above and must never be
    # reintroduced as convertible SU24 resources.
    ledger.released_su24_ids = sorted({
        int(value)
        for value in list(getattr(self, "released_su24_ids", set()) or set())
    })
    ledger.released_resource_ids_by_su = deepcopy(
        released_resource_ids_by_su
    )
    ledger.recycled_su24_to_23_ids = recycled_su24_to_23_ids
    ledger.recycled_resource_conversion_ops = recycled_resource_conversion_ops
    ledger.remaining_counts = {
        str(key): int(len(values)) for key, values in ownership_remaining_ids.items()
    }
    ledger.convertible_counts = {
        str(key): int(len(values))
        for key, values in ledger.convertible_ids_by_su.items()
    }
    ledger.protected_counts = {
        str(key): max(0, int(protected_resource_counts.get(str(key), 0) or 0))
        for key in ownership_remaining_ids
    }
    closure_initial_ids = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(getattr(self.allocator, "initial_ids_by_su", {}) or {}).items()
    }
    classification_pool = dict(
        getattr(self.classification, "resource_pool", {}) or {}
    )
    donor_initial_ids = {
        str(key): sorted(set(int(value) for value in list(values or [])))
        for key, values in dict(
            classification_pool.get("su11_donor_ids_by_su", {}) or {}
        ).items()
    }
    ledger.initial_ids_by_su = {
        str(key): sorted({
            int(value)
            for value in (
                list(closure_initial_ids.get(str(key), []) or [])
                + list(donor_initial_ids.get(str(key), []) or [])
            )
        })
        for key in sorted(
            set(closure_initial_ids)
            | set(donor_initial_ids)
            | set(ownership_remaining_ids),
            key=int,
        )
    }
    ledger.consumed_ids_by_su = {
        str(key): sorted(
            set(ledger.initial_ids_by_su.get(str(key), []) or [])
            - set(ownership_remaining_ids.get(str(key), []) or [])
        )
        for key in ledger.initial_ids_by_su
    }
    ledger.initial_counts = {
        str(key): int(len(values)) for key, values in ledger.initial_ids_by_su.items()
    }
    ledger.consumed_counts = {
        str(key): int(len(values)) for key, values in ledger.consumed_ids_by_su.items()
    }
    ledger.residual_consumed_counts = {str(key): 0 for key in ledger.consumed_ids_by_su}
    ledger.topology_consumed_counts = {
        str(key): max(
            0,
            int(ledger.consumed_counts.get(str(key), 0) or 0)
            - int(ledger.residual_consumed_counts.get(str(key), 0) or 0),
        )
        for key in set(ledger.initial_counts) | set(ledger.consumed_counts)
    }
    ledger.snapshot_stage = "final_ownership_reconciled"
    diagnostics["ownership_ledger"] = dict(ownership_view)
    diagnostics["ring_recovery_attempts"] = [
        dict(item or {})
        for item in list(
            getattr(self, "ring_recovery_attempts", []) or []
        )
        if dict(item or {})
    ]
    completed_node_ids = set(int(x) for x in ownership_states.get(Stage3OwnershipState.COMPLETED_TOPOLOGY.value, []))
    completed_node_ids.update(
        int(x) for x in ownership_states.get(Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value, [])
    )
    completed_node_ids.update(
        int(x)
        for x in ownership_states.get(
            Stage3OwnershipState.CLAIMED_SU24_RING_RESOURCE.value,
            [],
        )
    )
    pending_node_ids = set(int(x) for x in ownership_states.get(Stage3OwnershipState.PENDING_TOPOLOGY.value, []))
    fixed_node_ids = set(int(x) for x in ownership_states.get(Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value, []))
    support_node_ids = set(int(x) for x in ownership_states.get(Stage3OwnershipState.SUPPORT_CONTEXT.value, []))
    resource_node_ids = {
        int(node_id)
        for values in ownership_remaining_ids.values()
        for node_id in list(values or [])
    }
    conflict_node_ids = set(
        int(x) for x in ownership_states.get(Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value, [])
    )
    diagnostics["node_coverage"] = {
        "total_node_count": int(len(self.nodes_by_id)),
        "completed_node_count": int(len(completed_node_ids)),
        "pending_node_count": int(len(pending_node_ids)),
        "fixed_topology_core_node_count": int(len(fixed_node_ids)),
        "remaining_pool_node_count": int(len(resource_node_ids)),
        "conflict_node_count": int(len(conflict_node_ids)),
        "support_context_node_count": int(len(support_node_ids)),
        "covered_node_count": int(len(round_context.ownership_by_node_id)),
        "uncovered_node_ids": sorted(
            set(self.nodes_by_id) - set(round_context.ownership_by_node_id)
        ),
        "conflict_node_ids": sorted(conflict_node_ids),
        "duplicate_node_ids": [],
    }
    generic_closure_remaining_total = sum(
        int(ledger.remaining_counts.get(str(su_type), 0) or 0)
        for su_type in (11, 22, 23)
    )
    su24_ring_resource_remaining_total = int(
        ledger.remaining_counts.get("24", 0) or 0
    )
    classification_valid = bool(
        len(self.classification.unclassified_nodes or []) == 0
        and len(getattr(self.classification, "claim_conflicts", []) or []) == 0
        and len(self.classification.classification_rejects or []) == 0
    )
    ownership_valid = bool(
        not list(round_context_invariants(round_context) or [])
        and not list(diagnostics.get("node_coverage", {}).get("uncovered_node_ids", []) or [])
    )
    ledger_valid = bool(not list(ledger.invariant_errors or []))
    # This helper owns only external symbolic assembly. Stage3 completion is
    # decided later by the authoritative state machine.
    result = TopologyAssemblyResult(
        ok=False,
        reason=result_reason,
        feasible=bool(classification_valid and ownership_valid and ledger_valid),
        products=list(self.products or []),
        pending_units=list(self.pending_units or []),
        flex_bounds=dict(self.flex_bounds or {}),
        flex_counts={
            "explicit_total": int(explicit_flex_count),
            "final": int(final_flex_count),
        },
        resource_ledger=ledger,
        diagnostics=diagnostics,
        planned_conversions=recycled_resource_conversion_ops,
        active_phase=str(stage3_phase),
        completed_phases=list(completed_phases),
        phase_metrics={
            "generic_closure_remaining_total": int(
                generic_closure_remaining_total
            ),
            "su24_ring_resource_remaining_total": int(
                su24_ring_resource_remaining_total
            ),
        },
        flex_target_plan=dict(flex_target_plan),
        resource_demand={
            "hard_required_counts": {
                str(key): int(value)
                for key, value in dict(self.shortage_counts or {}).items()
                if int(value) > 0
            },
            "preferred_required_counts": {},
            "available_counts": self.allocator.available_counts(),
            "shortage_counts": {
                str(key): int(value)
                for key, value in dict(self.shortage_counts or {}).items()
                if int(value) > 0
            },
            "contracts_by_subphase": chain_materializer_module.contracts_by_subphase(self, ),
        },
        ownership_ledger=dict(ownership_view),
        protected_resource_counts=dict(protected_resource_counts),
        referenced_resource_counts={
            str(key): int(value)
            for key, value in dict(getattr(ledger, "fragment_owned_counts_by_su", {}) or {}).items()
        },
        external_candidate_capacity_exhausted=bool(flex_target_plan["external_capacity_exhausted"]),
    )
    return result
