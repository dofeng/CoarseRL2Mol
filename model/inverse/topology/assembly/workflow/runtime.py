"""Mutable runtime state for external topology assembly."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .....shared.inverse_common import _NodeV3
from ...classification.classification_models import TopologyClassificationResult
from ...stage3.flex_audit import _compute_flex_audit
from ...stage3_models import (
    TopologyAssemblyProduct,
    TopologyAssemblyResult,
    TopologyPendingUnit,
)
from ..core.pool import PoolAllocator
from ..closure import chain as chain_materializer_module
from ..state import ownership as ownership_module
from ..state.ownership import OwnershipRegistry
from ...resources.profiles import build_raw_su24_resource_profiles
from ..projection.engine import (
    _build_node_lookup,
    _compute_flex_bounds,
    _flex_preassembly_gate_result,
    _safe_int,
)


class ExternalTopologyAssembler:
    @property
    def fixed_node_owner_by_id(self) -> Dict[int, int]:
        registry = getattr(self, "ownership_registry", None)
        return registry.node_owner_by_id if registry is not None else {}

    @fixed_node_owner_by_id.setter
    def fixed_node_owner_by_id(self, value: Dict[int, int]) -> None:
        normalized = {
            int(node_id): int(fragment_id)
            for node_id, fragment_id in dict(value or {}).items()
        }
        registry = getattr(self, "ownership_registry", None)
        if registry is None:
            self.ownership_registry = OwnershipRegistry(
                node_owner_by_id=normalized
            )
        else:
            registry.node_owner_by_id.clear()
            registry.node_owner_by_id.update(normalized)

    @property
    def fragment_owner_by_node_id(self) -> Dict[int, int]:
        return self.fixed_node_owner_by_id

    @fragment_owner_by_node_id.setter
    def fragment_owner_by_node_id(self, value: Dict[int, int]) -> None:
        self.fixed_node_owner_by_id = value

    def __init__(
        self,
        nodes: Sequence[_NodeV3],
        classification: TopologyClassificationResult,
        *,
        flex_audit: Optional[Dict[str, Any]] = None,
        generation_id: int = 0,
        occupied_node_ids: Optional[Iterable[int]] = None,
        ownership_registry: Optional[OwnershipRegistry] = None,
        allow_ring_recovery: bool = False,
    ) -> None:
        self.nodes = list(nodes or [])
        self.nodes_by_id = _build_node_lookup(self.nodes)
        self.classification = classification
        self.fragment_by_id = {
            int(getattr(fragment, "fragment_id", -1)): fragment
            for fragment in list(getattr(classification, "fragments", []) or [])
        }
        self.generation_id = int(generation_id)
        # Stage3 enables this only after ordinary SU conversion planning has
        # proved a fatty-ring or side resource contract infeasible.
        self.allow_ring_recovery = bool(allow_ring_recovery)
        self.ring_recovery_max_su24_cost: Optional[int] = None
        # Mutable generation-local cursor.  This object is intentionally kept
        # out of public result serialization and is reused after a resource
        # conversion/recovery within the same Layer1/2 generation.
        self.completed_phases: Set[str] = set()
        self.current_phase: str = "explicit_branch_closure"
        self.current_H = None
        # Nodes owned by products from the current Stage3 generation are not
        # available to a new SU24 pool candidate.  This includes fixed
        # neighbours that are not represented in the allocator pool.
        self.occupied_node_ids: Set[int] = {
            int(value) for value in list(occupied_node_ids or [])
        }
        self.su24_support_bindings_by_node_id: Dict[int, Dict[int, Set[int]]] = {}
        for raw_binding in list(
            getattr(classification, "su24_support_bindings", []) or []
        ):
            binding = dict(raw_binding or {})
            center_id = _safe_int(binding.get("su24_node_id"), -1)
            if center_id < 0:
                continue
            support_by_su: Dict[int, Set[int]] = {}
            for raw_su, values in dict(
                binding.get("fixed_support_ids_by_su", {}) or {}
            ).items():
                su_type = _safe_int(raw_su, -1)
                if su_type < 0:
                    continue
                support_by_su[int(su_type)] = {
                    int(value) for value in list(values or [])
                }
            self.su24_support_bindings_by_node_id[int(center_id)] = support_by_su
        self._ring_plan_excluded_pool_node_ids: Set[int] = set()
        self._ring_plan_excluded_candidate_ids: Set[str] = set()
        initial_resource_ids = dict(
            dict(classification.resource_pool or {}).get("remaining_ids_by_su", {})
            or {}
        )
        initial_su24_ids = [
            int(value)
            for value in list(
                initial_resource_ids.get("24", [])
                or initial_resource_ids.get(24, [])
                or []
            )
        ]
        self._raw_su24_profiles_by_id = {
            int(profile.node_id): profile
            for profile in build_raw_su24_resource_profiles(
                self.nodes,
                {24: initial_su24_ids},
            )
        }
        self._raw_su24_profile_scanned_ids = set(initial_su24_ids)
        self.ring_exhausted_fragment_ids: Set[int] = set()
        self.product_ownership_conflict_count = 0
        self.last_product_ownership_conflict: Dict[str, Any] = {}
        self.flex_audit = dict(flex_audit or {})
        if not self.flex_audit:
            self.flex_audit = dict(_compute_flex_audit(self.nodes, classification, enforce=True) or {})
        self.selected_flex_candidate_ids = {
            int(value)
            for value in list(self.flex_audit.get("selected_candidate_fragment_ids", []) or [])
        }
        self.selected_flex_candidate_groups = [
            dict(item or {})
            for item in list(self.flex_audit.get("selected_candidate_groups", []) or [])
            if dict(item or {})
        ]
        self.supplemental_flex_candidate_ids = {
            int(value)
            for value in list(self.flex_audit.get("supplemental_flex_candidate_ids", []) or [])
        } | set(self.selected_flex_candidate_ids)
        pool = dict(classification.resource_pool or {})
        allocator_ids = {
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(pool.get("remaining_ids_by_su", {}) or {}).items()
        }
        # Aromatic SU12/SU13 donors are part of the same concrete allocator
        # namespace.  Keeping them only in a side diagnostic map made a
        # 12/13->11 conversion impossible to mirror on continuation.
        for raw_su, values in dict(pool.get("su11_donor_ids_by_su", {}) or {}).items():
            su_key = str(raw_su)
            if su_key not in {"12", "13"}:
                continue
            allocator_ids.setdefault(su_key, [])
            allocator_ids[su_key] = sorted({
                int(value)
                for value in list(allocator_ids[su_key] or [])
                + list(values or [])
            })
        self.allocator = PoolAllocator(allocator_ids)
        ownership_module.exclude_occupied_nodes_from_allocator(self, )
        ownership_module.exclude_bound_su24_supports_from_allocator(self, )
        from ..projection.engine import project_fragment_views

        self.views = project_fragment_views(classification, self.nodes_by_id)
        self.view_by_id = {int(view.fragment_id): view for view in self.views}
        # Classification is the only normal ownership authority.  Both public
        # attribute names reference the registry's same mapping.
        self.ownership_registry = (
            ownership_registry
            if ownership_registry is not None
            else OwnershipRegistry.from_classification(classification)
        )
        self.last_classification_owner_conflicts = list(
            self.ownership_registry.classification_conflicts
        )
        for view in self.views:
            foreign_context_owners = {
                int(node_id): int(self.fragment_owner_by_node_id[int(node_id)])
                for node_id in list(getattr(view, "context_node_ids", []) or [])
                if int(node_id) in self.fragment_owner_by_node_id
                and int(self.fragment_owner_by_node_id[int(node_id)]) != int(view.fragment_id)
            }
            if foreign_context_owners:
                view.diagnostics["foreign_context_fragment_owners"] = {
                    str(key): int(value)
                    for key, value in foreign_context_owners.items()
                }
        self.products: List[TopologyAssemblyProduct] = []
        self.product_owner_by_node_id: Dict[int, int] = {}
        # Nodes/products released by ring recovery are no longer available as
        # classifier fragments, but their physical SU identities remain valid
        # allocator resources.
        self.recycled_fragment_ids: Set[int] = set()
        self.recycled_node_ids: Set[int] = set()
        self.released_su24_ids: Set[int] = set()
        # Provenance only: these IDs are ordinary SU23 resources after a
        # pure-resource ring is dismantled and must never re-enter a ring.
        self.recycled_su24_to_23_ids: Set[int] = set()
        self.recycled_resource_conversion_ops: List[Dict[str, Any]] = []
        self.pending_recycled_resource_conversion_ops: List[Dict[str, Any]] = []
        self.released_resource_ids_by_su: Dict[str, Set[int]] = {}
        self.last_recycled_ring: Dict[str, Any] = {}
        self.ring_recovery_attempts: List[Dict[str, Any]] = []
        self.pending_units: List[TopologyPendingUnit] = []
        self.pending_contracts: List[Dict[str, Any]] = []
        self.shortage_counts: Counter = Counter()
        self.fragment_state_by_id: Dict[int, str] = {
            int(view.fragment_id): "available"
            for view in self.views
        }
        self.ring_pending_reasons: Dict[int, str] = {}
        self.ring_pending_attempts: Dict[int, Dict[str, Any]] = {}
        self.last_ring_materialization_reason: str = ""
        self.last_ring_missing_counts: Dict[str, int] = {}
        self.last_ring_candidate_required_counts: Dict[str, int] = {}
        self.ring_infeasible_contracts: Dict[int, Dict[str, Any]] = {}
        self.invalid_topology_count = 0
        self.last_invalid_topology: Dict[str, Any] = {}
        self.composition_attempt_failure_count = 0
        self.last_composition_attempt_failure: Dict[str, Any] = {}
        self.materialization_failure_by_fragment_id: Dict[int, Dict[str, Any]] = {}
        self.contract_definition_conflict_count = 0
        self.last_contract_definition_conflict: Dict[str, Any] = {}
        self.explicit_flex_count = int(sum(
            1
            for view in ownership_module.available_views(self, lambda item: True)
            if str(chain_materializer_module.pending_kind_for_view(self, view)) == "explicit_flex"
        ))
        self.flex_bounds = _compute_flex_bounds(
            explicit_flex_count=int(self.explicit_flex_count),
            flex_audit=self.flex_audit,
        )


    def assemble(self) -> TopologyAssemblyResult:
        if bool(dict(self.flex_audit or {}).get("enforced", False)):
            scenario = str(dict(self.flex_audit or {}).get("scenario", "") or "")
            blocked_reason = str(
                dict(self.flex_audit or {}).get("blocked_reason", "") or ""
            )
            if scenario == "above_range" or blocked_reason == "flex_window_invalid":
                return _flex_preassembly_gate_result(
                    classification=self.classification,
                    flex_audit=dict(self.flex_audit or {}),
                    flex_bounds=dict(self.flex_bounds or {}),
                )

        return self.continue_assembly()

    def sync_after_resource_adjustments(
        self,
        nodes: Sequence[_NodeV3],
        H: Any = None,
        applied_ops: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Synchronize concrete SU identities after an accepted Stage3 plan.

        Planner transactions operate on a copied graph.  Before continuing the
        live assembler, mirror every concrete operation into its node map and
        allocator.  This keeps pending contracts and resource buckets in the
        same identity space as the caller's accepted graph.
        """
        incoming = list(nodes or [])
        incoming_by_id = {
            int(getattr(node, "global_id", -1)): node
            for node in incoming
            if int(getattr(node, "global_id", -1)) >= 0
        }
        for node_id, incoming_node in incoming_by_id.items():
            local = self.nodes_by_id.get(int(node_id))
            if local is None:
                continue
            setattr(local, "su_type", int(getattr(incoming_node, "su_type", -1)))
        # Keep the assembler's transaction-local node copies. Ring recovery
        # mutates those copies before Stage3 accepts the corresponding physical
        # conversion; aliasing caller nodes here would leak a provisional
        # SU24->SU23 change and make the pending commit disappear.
        applied_recovery_ids: Set[int] = set()
        applied_resource_change = False
        for raw_group in list(applied_ops or []):
            group = dict(raw_group or {})
            operations: List[Dict[str, Any]] = []
            if group.get("from_su") is not None and group.get("to_su") is not None:
                operations.append(group)
            primary = group.get("primary_op")
            if isinstance(primary, dict):
                operations.append(primary)
            operations.extend(
                dict(item or {})
                for item in list(group.get("h_compensation_ops", []) or [])
                if isinstance(item, dict)
            )
            for field in ("parity_balance_op", "port_role_balance_op"):
                balance = group.get(field)
                if isinstance(balance, dict):
                    operations.append(balance)
            for operation in operations:
                source = operation.get("from_su")
                target = operation.get("to_su")
                if source is None or target is None:
                    continue
                if int(source) != int(target):
                    # Accepted conversions change the concrete pool used by
                    # ring planning.  Even if this operation was already
                    # mirrored by an idempotent continuation call, any ring
                    # candidate enumerated before it is now historical.
                    applied_resource_change = True
                node_ids = [
                    int(value)
                    for value in list(operation.get("node_ids", []) or [])
                    if value is not None
                ]
                if not node_ids and operation.get("node_id") is not None:
                    node_ids = [int(operation["node_id"])]
                if not node_ids or int(source) == int(target):
                    continue
                if int(source) == 24 and int(target) == 23:
                    applied_recovery_ids.update(node_ids)
                # The planner has already validated the concrete path.  The
                # allocator is authoritative for availability; tolerate an
                # already synchronized node so continuation remains idempotent.
                moved = self.allocator.reclassify_ids(
                    node_ids,
                    from_su=int(source),
                    to_su=int(target),
                )
                if not moved:
                    # Donors may be absent from the live allocator (aromatic
                    # SU12/SU13 pools, or a planner transaction that already
                    # consumed the source bucket). Re-register the concrete
                    # node exactly once in its new target bucket.
                    for node_id in node_ids:
                        for bucket in list(self.allocator.remaining_ids_by_su):
                            self.allocator.remaining_ids_by_su[bucket] = [
                                int(value)
                                for value in list(
                                    self.allocator.remaining_ids_by_su.get(bucket, [])
                                    or []
                                )
                                if int(value) != int(node_id)
                            ]
                        for bucket in list(self.allocator.initial_ids_by_su):
                            self.allocator.initial_ids_by_su[bucket] = [
                                int(value)
                                for value in list(
                                    self.allocator.initial_ids_by_su.get(bucket, [])
                                    or []
                                )
                                if int(value) != int(node_id)
                            ]
                        for bucket in list(self.allocator.consumed_ids_by_su):
                            self.allocator.consumed_ids_by_su[bucket] = [
                                int(value)
                                for value in list(
                                    self.allocator.consumed_ids_by_su.get(bucket, [])
                                    or []
                                )
                                if int(value) != int(node_id)
                            ]
                        self.allocator.su_key_by_node_id.pop(int(node_id), None)
                    self.allocator.release_ids({int(target): node_ids})
        if applied_recovery_ids:
            # Durable recovery rows remain available for final diagnostics;
            # only the continuation queue is consumed.
            self.pending_recycled_resource_conversion_ops = [
                dict(item or {})
                for item in list(
                    self.pending_recycled_resource_conversion_ops or []
                )
                if int(dict(item or {}).get("node_id", -1))
                not in applied_recovery_ids
            ]

        # Ring plans carry concrete SU24/support identities.  Invalidate
        # their retry state after an accepted conversion so the next pass
        # enumerates against the live allocator instead of reusing a plan
        # whose anchor may already belong to another product.  The old state
        # is retained locally and copied into the pending contract history
        # below for auditability; it is never used as the current candidate.
        stale_ring_attempts: Dict[int, Dict[str, Any]] = {}
        stale_ring_failures: Dict[int, Dict[str, Any]] = {}
        if applied_resource_change:
            available_ring_fragment_ids = {
                int(view.fragment_id)
                for view in list(getattr(self, "views", []) or [])
                if bool(getattr(view, "can_ring", False))
                and str(
                    self.fragment_state_by_id.get(int(view.fragment_id), "available")
                ) == "available"
            }
            ring_state_ids = (
                available_ring_fragment_ids
                | {int(value) for value in dict(self.ring_infeasible_contracts or {})}
                | {int(value) for value in dict(self.ring_pending_attempts or {})}
                | {int(value) for value in dict(self.ring_pending_reasons or {})}
                | {
                    int(value)
                    for value in dict(
                        self.materialization_failure_by_fragment_id or {}
                    )
                    if str(
                        dict(
                            self.materialization_failure_by_fragment_id.get(
                                int(value), {}
                            )
                            or {}
                        ).get("attempted_product_kind", "")
                        or ""
                    ).startswith("fatty_ring_")
                }
            )
            for fragment_id in sorted(ring_state_ids):
                prior_attempt = dict(
                    self.ring_pending_attempts.get(int(fragment_id), {}) or {}
                )
                prior_failure = dict(
                    self.materialization_failure_by_fragment_id.get(
                        int(fragment_id), {}
                    )
                    or {}
                )
                if prior_attempt:
                    stale_ring_attempts[int(fragment_id)] = deepcopy(prior_attempt)
                if prior_failure:
                    stale_ring_failures[int(fragment_id)] = deepcopy(prior_failure)
                self.ring_infeasible_contracts.pop(int(fragment_id), None)
                self.ring_pending_attempts.pop(int(fragment_id), None)
                self.ring_pending_reasons.pop(int(fragment_id), None)
                if str(prior_failure.get("attempted_product_kind", "") or "").startswith(
                    "fatty_ring_"
                ):
                    self.materialization_failure_by_fragment_id.pop(
                        int(fragment_id), None
                    )
            self.ring_exhausted_fragment_ids.difference_update(
                available_ring_fragment_ids
            )
            if ring_state_ids:
                self.completed_phases.discard("real_like24_ring_assembly")
        # Pending records are mutable continuation state, not immutable
        # diagnostics. Refresh their resource view after every accepted
        # conversion so the next recovery gate sees the allocator's concrete
        # inventory rather than the snapshot from the previous attempt.
        raw_available_counts = {
            str(key): int(value)
            for key, value in dict(self.allocator.available_counts() or {}).items()
        }
        available_counts = {
            str(key): int(value)
            for key, value in dict(self.allocator.claimable_counts() or {}).items()
        }

        def _missing(required: Any) -> Dict[str, int]:
            return {
                str(key): max(
                    0,
                    int(value) - int(available_counts.get(str(key), 0)),
                )
                for key, value in dict(required or {}).items()
                if int(value) > int(available_counts.get(str(key), 0))
            }

        for contract in list(self.pending_contracts or []):
            if not isinstance(contract, dict):
                continue
            had_accepted_operations = bool(list(applied_ops or []))
            diagnostics = dict(contract.get("diagnostics", {}) or {})
            stale_missing = dict(
                diagnostics.get("claim_failure_missing_counts", {}) or {}
            )
            prior_failure_domains = {
                str(value)
                for value in (
                    contract.get("failure_domain", ""),
                    diagnostics.get("failure_domain", ""),
                    dict(contract.get("proof", {}) or {}).get(
                        "failure_domain", ""
                    ),
                )
                if str(value)
            }
            resource_retry_candidate = bool(
                stale_missing
                or "resource_capacity" in prior_failure_domains
                or any(
                    token in str(contract.get("reason", "") or "").lower()
                    for token in ("shortage", "resource", "pool")
                )
            )
            if had_accepted_operations:
                # Preserve the old attempt as history, but never let its
                # failed claim be interpreted as the current inventory after
                # concrete SU conversion has been accepted.  A contract has
                # two layers of diagnostics (the public row and the nested
                # materialization proof); clearing only claim_trace used to
                # leave ``branch_pool_shortage`` in the latter and the Stage3
                # planner consequently treated a now-satisfied contract as a
                # resource-accounting contradiction.
                stale_trace = list(diagnostics.get("claim_trace", []) or [])
                stale_materialization = dict(
                    diagnostics.get("materialization_failure", {}) or {}
                )
                stale_proof = dict(contract.get("proof", {}) or {})
                if stale_trace or stale_missing:
                    attempt_history = list(
                        diagnostics.get("attempt_history", []) or []
                    )
                    attempt_history.append({
                        "kind": "pre_adjustment_claim_attempt",
                        "claim_trace": stale_trace,
                        "claim_failure_missing_counts": stale_missing,
                    })
                    diagnostics["attempt_history"] = attempt_history[-16:]
                if stale_materialization or stale_proof.get("reason"):
                    attempt_history = list(
                        diagnostics.get("attempt_history", []) or []
                    )
                    attempt_history.append({
                        "kind": "pre_adjustment_failure_proof",
                        "reason": str(contract.get("reason", "") or ""),
                        "failure_domain": str(
                            contract.get("failure_domain", "") or ""
                        ),
                        "proof": stale_proof,
                        "materialization_failure": stale_materialization,
                    })
                    diagnostics["attempt_history"] = attempt_history[-16:]
                diagnostics["claim_trace"] = []
                diagnostics["claim_failure_missing_counts"] = {}
                # The old materialization attempt is historical evidence only.
                # It must not win over a fresh retry after the allocator has
                # been changed by an accepted conversion.
                diagnostics.pop("materialization_failure", None)
                if stale_ring_attempts or stale_ring_failures:
                    fragment_ids = {
                        int(value)
                        for value in list(
                            contract.get("source_fragment_ids", []) or []
                        )
                    }
                    for fragment_id in sorted(fragment_ids):
                        prior_attempt = stale_ring_attempts.get(int(fragment_id))
                        prior_failure = stale_ring_failures.get(int(fragment_id))
                        if not prior_attempt and not prior_failure:
                            continue
                        attempt_history = list(
                            diagnostics.get("attempt_history", []) or []
                        )
                        attempt_history.append({
                            "kind": "pre_adjustment_ring_candidate_state",
                            "ring_pending_attempts": dict(prior_attempt or {}),
                            "materialization_failure": dict(prior_failure or {}),
                        })
                        diagnostics["attempt_history"] = attempt_history[-16:]
            required = dict(contract.get("required_counts", {}) or {})
            missing = _missing(required)
            contract["available_counts"] = dict(available_counts)
            contract["raw_available_counts"] = dict(raw_available_counts)
            contract["claimable_available_counts"] = dict(available_counts)
            contract["missing_counts"] = dict(missing)
            proof = dict(contract.get("proof", {}) or {})
            proof["available_counts"] = dict(available_counts)
            proof["raw_available_counts"] = dict(raw_available_counts)
            proof["claimable_available_counts"] = dict(available_counts)
            proof["missing_counts"] = dict(missing)
            contract["proof"] = proof
            diagnostics["raw_available_counts"] = dict(raw_available_counts)
            diagnostics["claimable_available_counts"] = dict(available_counts)
            diagnostics["missing_counts"] = dict(missing)
            contract["diagnostics"] = diagnostics
            if (
                not missing
                and had_accepted_operations
                and resource_retry_candidate
            ):
                # The previous shortage has been removed by an accepted
                # concrete conversion.  The materializer retry below is now
                # authoritative; do not let the stale failure domain turn a
                # ready contract back into a resource blocker.
                contract["reason"] = "continuation_ready"
                contract["failure_domain"] = ""
                diagnostics["failure_domain"] = ""
                diagnostics["missing_counts"] = {}
                diagnostics["claim_failure_missing_counts"] = {}
                proof["reason"] = "continuation_ready"
                proof["failure_domain"] = ""
                proof["missing_counts"] = {}
                contract["proof"] = proof
                contract["continuation_ready"] = True
                # Fragment-level materialization failures are consulted by
                # ``record_remaining_external_pending`` when the phase pass
                # is serialized.  Retain them only in the contract's bounded
                # attempt history; otherwise an old failure can overwrite the
                # fresh continuation state after the retry.
                for fragment_id in list(
                    contract.get("source_fragment_ids", []) or []
                ):
                    self.materialization_failure_by_fragment_id.pop(
                        int(fragment_id), None
                    )
            elif missing:
                contract.pop("continuation_ready", None)
        for pending in list(self.pending_units or []):
            required = dict(getattr(pending, "required_counts", {}) or {})
            missing = _missing(required)
            old_reason = str(getattr(pending, "reason", "") or "")
            old_proof = dict(getattr(pending, "infeasibility_proof", {}) or {})
            pending.available_counts = dict(available_counts)
            pending.raw_available_counts = dict(raw_available_counts)
            pending.claimable_available_counts = dict(available_counts)
            pending.infeasibility_proof = {
                **dict(getattr(pending, "infeasibility_proof", {}) or {}),
                "available_counts": dict(available_counts),
                "raw_available_counts": dict(raw_available_counts),
                "claimable_available_counts": dict(available_counts),
                "missing_counts": dict(missing),
            }
            if not missing and list(applied_ops or []):
                pending.claim_trace = []
                pending.infeasibility_proof["claim_trace"] = []
                if any(
                    token in old_reason.lower()
                    for token in ("shortage", "resource", "pool")
                ):
                    pending.infeasibility_proof.setdefault(
                        "attempt_history", []
                    ).append({
                        "kind": "pre_adjustment_failure_proof",
                        "reason": old_reason,
                        "proof": old_proof,
                    })
                    pending.reason = "continuation_ready"
                    pending.infeasibility_proof["failure_domain"] = ""
                    pending.infeasibility_proof["reason"] = "continuation_ready"
                    pending.infeasibility_proof["missing_counts"] = {}
                    proof_diagnostics = dict(
                        pending.infeasibility_proof.get("diagnostics", {}) or {}
                    )
                    if proof_diagnostics.get("materialization_failure"):
                        pending.infeasibility_proof.setdefault(
                            "attempt_history", []
                        ).append({
                            "kind": "pre_adjustment_materialization_failure",
                            "materialization_failure": dict(
                                proof_diagnostics.get(
                                    "materialization_failure", {}
                                )
                                or {}
                            ),
                        })
                        proof_diagnostics.pop("materialization_failure", None)
                        pending.infeasibility_proof["diagnostics"] = (
                            proof_diagnostics
                        )
        self.current_H = H.detach().clone() if hasattr(H, "detach") else H

    def continue_assembly(self) -> TopologyAssemblyResult:
        """Continue the current generation from its phase/pending contracts."""
        from .coordinator import run_external_assembly_phases
        from .result_builder import build_external_assembly_result

        phase_state = run_external_assembly_phases(self)
        # Ring/side materialization may consume concrete allocator resources
        # while creating a product.  The contracts emitted during that same
        # pass must describe the post-materialization inventory; otherwise a
        # later Stage3 continuation can see a stale ``claimable=1`` snapshot
        # even though the live bucket is already empty and report an
        # accounting inconsistency instead of a real resource deficit.
        self.sync_after_resource_adjustments(
            list(self.nodes or []),
            H=self.current_H,
        )
        return build_external_assembly_result(self, phase_state)


def assemble_external_topologies(
    nodes: Sequence[_NodeV3],
    classification: TopologyClassificationResult,
    flex_audit: Optional[Dict[str, Any]] = None,
    generation_id: int = 0,
    allow_ring_recovery: bool = False,
) -> TopologyAssemblyResult:
    resolved_flex_audit = dict(flex_audit or {})
    if not resolved_flex_audit:
        resolved_flex_audit = dict(_compute_flex_audit(
            list(nodes or []),
            classification,
            enforce=True,
        ) or {})
    assembler = ExternalTopologyAssembler(
        # Recycling may reclassify a SU24 node to SU23.  Keep that mutation
        # transaction-local until Stage3 accepts the corresponding physical
        # conversion, so a rejected candidate cannot corrupt the caller graph.
        nodes=deepcopy(list(nodes or [])),
        classification=classification,
        flex_audit=resolved_flex_audit,
        generation_id=int(generation_id),
        ownership_registry=OwnershipRegistry.from_classification(classification),
        allow_ring_recovery=bool(allow_ring_recovery),
    )
    if bool(dict(assembler.flex_audit or {}).get("enforced", False)):
        scenario = str(dict(assembler.flex_audit or {}).get("scenario", "") or "")
        blocked_reason = str(dict(assembler.flex_audit or {}).get("blocked_reason", "") or "")
        if scenario == "above_range" or blocked_reason == "flex_window_invalid":
            return _flex_preassembly_gate_result(
                classification=classification,
                flex_audit=dict(assembler.flex_audit or {}),
                flex_bounds=dict(assembler.flex_bounds or {}),
            )
    result = assembler.assemble()
    # Runtime-only handle used by Stage3 continuation.  ``TopologyAssemblyResult``
    # deliberately serializes only declared dataclass fields, so this never
    # leaks into final diagnostics.
    result._runtime_assembler = assembler
    return result
