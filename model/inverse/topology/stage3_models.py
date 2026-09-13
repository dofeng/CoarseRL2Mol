from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional

import torch


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return [_serialize_value(v) for v in sorted(list(value), key=str)]
    if hasattr(value, "global_id") and hasattr(value, "su_type"):
        try:
            return {
                "global_id": int(getattr(value, "global_id")),
                "su_type": int(getattr(value, "su_type")),
            }
        except Exception:
            return str(value)
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {
            str(key): _serialize_value(getattr(value, key))
            for key in getattr(value, "__dataclass_fields__", {}).keys()
        }
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


class Stage3Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return {
            str(key): _serialize_value(getattr(self, key))
            for key in getattr(self, "__dataclass_fields__", {}).keys()
        }


class Stage3OwnershipState(str, Enum):
    FIXED_TOPOLOGY_CORE = "fixed_topology_core"
    PENDING_TOPOLOGY = "pending_topology"
    COMPLETED_TOPOLOGY = "completed_topology"
    CLAIMED_CLOSURE_RESOURCE = "claimed_closure_resource"
    REMAINING_CLOSURE_RESOURCE = "remaining_closure_resource"
    CLAIMED_SU24_RING_RESOURCE = "claimed_su24_ring_resource"
    REMAINING_SU24_RING_RESOURCE = "remaining_su24_ring_resource"
    REMAINING_AROMATIC_DONOR = "remaining_aromatic_donor"
    SUPPORT_CONTEXT = "support_context"
    UNCLASSIFIED_OR_CONFLICT = "unclassified_or_conflict"


STAGE3_OWNERSHIP_STATES = tuple(state.value for state in Stage3OwnershipState)


class Stage3ProgressPhase(str, Enum):
    """Generation-local Stage3 progress over one Layer1/2 graph."""

    FLEX_RECONCILIATION = "flex_range_reconciliation"
    EXTERNAL_TOPOLOGY = "external_topology"
    SPECTRUM_REBALANCE = "remaining_pool_spectrum_rebalance"
    INTERNAL_TOPOLOGY = "internal_topology"
    COMPLETE = "complete"


@dataclass
class Stage3NodeOwnership(Stage3Serializable):
    node_id: int = -1
    su_type: int = -1
    state: str = Stage3OwnershipState.UNCLASSIFIED_OR_CONFLICT.value
    owner_fragment_id: Optional[int] = None
    owner_contract_id: Optional[int] = None
    owner_product_id: str = ""
    owner_slot_id: str = ""
    role: str = ""
    reason: str = ""


@dataclass
class Stage3TopologyProduct(Stage3Serializable):
    product_id: str = ""
    generation_id: int = 0
    product_type: str = ""
    phase: str = ""
    provisional: bool = False
    source_fragment_ids: List[int] = field(default_factory=list)
    fixed_source_node_ids: List[int] = field(default_factory=list)
    claimed_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    expected_su_type_by_node_id: Dict[int, int] = field(default_factory=dict)
    mainline_slots: List[Dict[str, Any]] = field(default_factory=list)
    branch_slots: List[Dict[str, Any]] = field(default_factory=list)
    topology_notation: str = ""
    node_id_notation: str = ""
    ring_audit: Dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    reason: str = ""


@dataclass
class Stage3SpectrumCandidate(Stage3Serializable):
    candidate_id: str = ""
    histogram_signature: str = ""
    inverse_signature: str = ""
    operations: List[Dict[str, Any]] = field(default_factory=list)
    # Optional first-class metadata for spectrum-only paired conversions.  The
    # individual operations remain in ``operations`` for backwards
    # compatibility with the existing apply/rebuild path; each pair group
    # identifies the two operations that must be committed atomically.
    pair_groups: List[Dict[str, Any]] = field(default_factory=list)
    target_su_types: List[int] = field(default_factory=list)
    predicted_window_gain: float = 0.0
    primary_adjustment_count: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3RoundContext(Stage3Serializable):
    generation_id: int = 0
    classification: Dict[str, Any] = field(default_factory=dict)
    ownership_by_node_id: Dict[int, Stage3NodeOwnership] = field(default_factory=dict)
    generic_closure_pool_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)
    su24_ring_resource_ids: List[int] = field(default_factory=list)
    # SU24 IDs reclassified to ordinary SU23 resources after ring recovery.
    recycled_su24_to_23_ids: List[int] = field(default_factory=list)
    aromatic_donor_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)
    flex_audit: Dict[str, Any] = field(default_factory=dict)
    products: List[Stage3TopologyProduct] = field(default_factory=list)
    pending_contracts: List[Dict[str, Any]] = field(default_factory=list)
    accepted_histogram_signatures: List[str] = field(default_factory=list)
    rejected_histogram_signatures: List[str] = field(default_factory=list)
    primary_conversion_count: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


class TopologySlotKind(str, Enum):
    F1 = "F1"
    F2 = "F2"
    H = "H"
    H1 = "H1"
    FIXED22 = "FIXED22"
    CLOSED_TERMINAL = "CLOSED_TERMINAL"


@dataclass
class TopologyClosureSlot(Stage3Serializable):
    slot_id: str = ""
    kind: str = ""
    anchor_node_id: int = -1
    anchor_role: str = ""
    required_su_sequence: List[int] = field(default_factory=list)
    existing_node_ids: List[int] = field(default_factory=list)
    assigned_resource_ids: List[int] = field(default_factory=list)
    state: str = "OPEN"

    def is_complete(self) -> bool:
        return str(self.state) in {"FILLED", TopologySlotKind.CLOSED_TERMINAL.value}


@dataclass
class Stage3OwnershipLedger(Stage3Serializable):
    generation_id: int = 0
    ownership_by_node_id: Dict[int, Stage3NodeOwnership] = field(default_factory=dict)
    completed_products: List[Dict[str, Any]] = field(default_factory=list)
    pending_contracts: List[Dict[str, Any]] = field(default_factory=list)
    available_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)
    support_context_node_ids: List[int] = field(default_factory=list)
    conflict_node_ids: List[int] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Stage3FlexDecision(Stage3Serializable):
    scenario: str = "in_range"
    cluster_count: int = 0
    su10_pair_count: int = 0
    effective_flex_count: int = 0
    lower_bound: int = 0
    upper_bound: int = 0
    target_count: int = 0
    selected_candidate_fragment_ids: List[int] = field(default_factory=list)
    selected_candidate_node_ids: List[int] = field(default_factory=list)
    required_resource_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_resource_counts: Dict[str, int] = field(default_factory=dict)
    blocked_reason: str = ""


@dataclass
class TopologyPathRender(Stage3Serializable):
    topology_notation: str = ""
    node_id_notation: str = ""
    main_topology_notation: str = ""
    main_node_id_notation: str = ""
    branch_topology_notations: List[str] = field(default_factory=list)
    branch_node_id_notations: List[str] = field(default_factory=list)
    suppressed_support_node_ids: List[int] = field(default_factory=list)


@dataclass
class TopologyPoolLedger(Stage3Serializable):
    global_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    fixed_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    fragment_owned_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    convertible_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    initial_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    remaining_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    consumed_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    global_counts: Dict[str, int] = field(default_factory=dict)
    fixed_counts_by_su: Dict[str, int] = field(default_factory=dict)
    fragment_owned_counts_by_su: Dict[str, int] = field(default_factory=dict)
    convertible_counts: Dict[str, int] = field(default_factory=dict)
    protected_counts: Dict[str, int] = field(default_factory=dict)
    invariant_errors: List[str] = field(default_factory=list)
    initial_counts: Dict[str, int] = field(default_factory=dict)
    consumed_counts: Dict[str, int] = field(default_factory=dict)
    topology_consumed_counts: Dict[str, int] = field(default_factory=dict)
    residual_consumed_counts: Dict[str, int] = field(default_factory=dict)
    remaining_counts: Dict[str, int] = field(default_factory=dict)
    shortage_counts: Dict[str, int] = field(default_factory=dict)
    snapshot_stage: str = "assembly_input"
    fixed_node_ids: List[int] = field(default_factory=list)
    fragment_owned_node_ids: List[int] = field(default_factory=list)
    fixed_by_product: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allocation_state_by_node: Dict[str, str] = field(default_factory=dict)
    released_su24_ids: List[int] = field(default_factory=list)
    released_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    recycled_su24_to_23_ids: List[int] = field(default_factory=list)
    recycled_resource_conversion_ops: List[Dict[str, Any]] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "initial_counts": {str(k): int(v) for k, v in dict(self.initial_counts or {}).items()},
            "consumed_counts": {str(k): int(v) for k, v in dict(self.consumed_counts or {}).items()},
            "topology_consumed_counts": {
                str(k): int(v) for k, v in dict(self.topology_consumed_counts or {}).items()
            },
            "residual_consumed_counts": {
                str(k): int(v) for k, v in dict(self.residual_consumed_counts or {}).items()
            },
            "remaining_counts": {str(k): int(v) for k, v in dict(self.remaining_counts or {}).items()},
            "shortage_counts": {str(k): int(v) for k, v in dict(self.shortage_counts or {}).items()},
            "released_su24_ids": [int(value) for value in list(self.released_su24_ids or [])],
            "released_resource_ids_by_su": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in sorted(
                    dict(self.released_resource_ids_by_su or {}).items(),
                    key=lambda item: str(item[0]),
                )
            },
            "recycled_su24_to_23_ids": [
                int(value) for value in list(self.recycled_su24_to_23_ids or [])
            ],
            "recycled_resource_conversion_ops": [
                dict(item or {})
                for item in list(self.recycled_resource_conversion_ops or [])
            ],
            "snapshot_stage": str(self.snapshot_stage or "assembly_input"),
        }


@dataclass
class RingTopologyContract(Stage3Serializable):
    """Internal, node-bound source of truth for one materialized aliphatic ring."""

    contract_id: str = ""
    candidate_signature: str = ""
    ring_kind: str = ""
    cycles: List[List[int]] = field(default_factory=list)
    node_labels_by_id: Dict[int, str] = field(default_factory=dict)
    node_su_by_id: Dict[int, int] = field(default_factory=dict)
    node_profiles_by_id: Dict[int, str] = field(default_factory=dict)
    role_node_ids: Dict[str, List[int]] = field(default_factory=dict)
    external_attachments: List[List[int]] = field(default_factory=list)
    branch_paths: List[Dict[str, Any]] = field(default_factory=list)
    branch_requirements: List[Dict[str, Any]] = field(default_factory=list)
    source_fragment_ids: List[int] = field(default_factory=list)
    fixed_source_node_ids: List[int] = field(default_factory=list)
    claimed_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    # Fixed support nodes which are physically part of the materialized ring
    # but are intentionally omitted from its rendered backbone/branch text.
    # They remain part of ownership and are released when a pure resource ring
    # is dismantled.
    support_node_ids: List[int] = field(default_factory=list)
    # Complete physical resource ownership of a ring.  This deliberately
    # includes fixed classifier nodes (for example the 11 cap of a 24
    # fragment), because a pure 11/22/23/24 ring can be dismantled as a unit.
    recyclable_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    required_resource_counts: Dict[str, int] = field(default_factory=dict)
    protected_node_ids: List[int] = field(default_factory=list)
    conversion_ops: List[Dict[str, Any]] = field(default_factory=list)
    topology_notation: str = ""
    node_id_notation: str = ""
    like24_node_ids: List[int] = field(default_factory=list)
    bridgehead_kind: str = ""
    complete: bool = False
    audit_reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def declared_node_ids(self) -> List[int]:
        return sorted(
            {
                int(node_id)
                for node_id in dict(self.node_su_by_id or {})
            }
            | {
                int(node_id)
                for node_id in list(self.support_node_ids or [])
            }
        )

    def recyclable_node_ids(self) -> List[int]:
        return sorted({
            int(value)
            for values in dict(self.recyclable_resource_ids_by_su or {}).values()
            for value in list(values or [])
        })

    def to_audit_dict(self) -> Dict[str, Any]:
        """Return the read-only ring graph needed by final diagnostics."""
        generation_id = int(
            dict(self.diagnostics or {}).get("generation_id", 0) or 0
        )
        return {
            "generation_id": generation_id,
            "ring_kind": str(self.ring_kind),
            "candidate_signature": str(self.candidate_signature),
            "cycles": [
                [int(value) for value in list(cycle or [])]
                for cycle in list(self.cycles or [])
            ],
            "like24_node_ids": [
                int(value) for value in list(self.like24_node_ids or [])
            ],
            "node_profiles_by_id": {
                str(key): str(value)
                for key, value in dict(self.node_profiles_by_id or {}).items()
            },
            "external_attachments": [
                [int(value) for value in list(edge or [])]
                for edge in list(self.external_attachments or [])
            ],
            "branch_paths": [
                dict(path or {}) for path in list(self.branch_paths or [])
            ],
            "claimed_resource_ids_by_su": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(
                    self.claimed_resource_ids_by_su or {}
                ).items()
            },
            "support_node_ids": [
                int(value) for value in list(self.support_node_ids or [])
            ],
            "recyclable_resource_ids_by_su": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(
                    self.recyclable_resource_ids_by_su or {}
                ).items()
            },
            "complete": bool(self.complete),
            "audit_reason": str(self.audit_reason or ""),
        }


@dataclass
class TopologyAssemblyProduct(Stage3Serializable):
    product_kind: str = ""
    product_type: str = ""
    actual_topology_su: str = ""
    actual_topology_gid: str = ""
    source_fragment_ids: List[int] = field(default_factory=list)
    source_fragment_classes: List[str] = field(default_factory=list)
    fixed_source_node_ids: List[int] = field(default_factory=list)
    mainline_node_ids: List[int] = field(default_factory=list)
    branch_paths: List[Dict[str, Any]] = field(default_factory=list)
    claimed_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    expected_su_type_by_node_id: Dict[int, int] = field(default_factory=dict)
    mainline_slots: List[Dict[str, Any]] = field(default_factory=list)
    branch_slots: List[Dict[str, Any]] = field(default_factory=list)
    ring_contract: Optional[RingTopologyContract] = None
    closed: bool = True
    reason: str = ""

    def rendered_node_ids(self) -> List[int]:
        return sorted({
            int(value)
            for value in list(self.mainline_node_ids or [])
        } | {
            int(value)
            for path in list(self.branch_paths or [])
            for value in list(dict(path or {}).get("node_ids", []) or [])
        })

    def claimed_resource_node_ids(self) -> List[int]:
        return sorted({
            int(value)
            for values in dict(self.claimed_resource_ids_by_su or {}).values()
            for value in list(values or [])
        })

    def owned_node_ids(self) -> List[int]:
        return sorted(
            set(int(value) for value in list(self.fixed_source_node_ids or []))
            | set(self.claimed_resource_node_ids())
        )

    def to_dict(self) -> Dict[str, Any]:
        # The mutable contract remains internal; only its read-only graph audit
        # crosses the assembly boundary for final diagnostics.
        payload = {
            str(key): _serialize_value(getattr(self, key))
            for key in getattr(self, "__dataclass_fields__", {}).keys()
            if str(key) != "ring_contract"
        }
        if self.ring_contract is not None:
            payload["ring_audit"] = self.ring_contract.to_audit_dict()
        return payload

    def to_public_dict(self) -> Dict[str, Any]:
        claimed_ids = {
            int(value)
            for values in dict(self.claimed_resource_ids_by_su or {}).values()
            for value in list(values or [])
        }
        fixed_ids = {int(value) for value in list(self.fixed_source_node_ids or [])}
        owned_ids = fixed_ids | claimed_ids
        is_ring = bool(
            str(self.product_kind or "").startswith("fatty_ring_")
            or str(self.product_type or "") == "aliphatic_ring"
        )
        node_id_notation = str(self.actual_topology_gid or "") if is_ring else re.sub(
            r"\d+",
            lambda match: match.group(0) if int(match.group(0)) in owned_ids else "OPEN",
            str(self.actual_topology_gid or ""),
        )
        payload = {
            "product_type": str(self.product_type or self.product_kind or ""),
            "source_fragment_ids": [int(x) for x in list(self.source_fragment_ids or [])],
            "fixed_source_node_ids": sorted(fixed_ids),
            "claimed_resource_ids_by_su": {
                str(key): sorted({int(value) for value in list(values or [])})
                for key, values in dict(self.claimed_resource_ids_by_su or {}).items()
                if list(values or [])
            },
            "expected_su_type_by_node_id": {
                str(node_id): int(su_type)
                for node_id, su_type in sorted(
                    dict(self.expected_su_type_by_node_id or {}).items()
                )
            },
            "topology_notation": str(self.actual_topology_su or ""),
            "node_id_notation": str(node_id_notation),
            "closed": bool(self.closed),
        }
        if is_ring:
            payload["node_ids"] = self.rendered_node_ids()
        else:
            payload["mainline_slots"] = [
                dict(item or {}) for item in list(self.mainline_slots or [])
            ]
            payload["branch_slots"] = [
                dict(item or {}) for item in list(self.branch_slots or [])
            ]
        return payload


@dataclass
class TopologyPendingUnit(Stage3Serializable):
    unit_id: int = 0
    pending_kind: str = ""
    contract_kind: str = ""
    target_role: str = ""
    real_topology_partial: str = ""
    real_node_partial: str = ""
    source_fragment_ids: List[int] = field(default_factory=list)
    source_fragment_classes: List[str] = field(default_factory=list)
    claimed_source_node_ids: List[int] = field(default_factory=list)
    claimed_resource_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    phase: str = ""
    subphase: str = ""
    required_counts: Dict[str, int] = field(default_factory=dict)
    available_counts: Dict[str, int] = field(default_factory=dict)
    raw_available_counts: Dict[str, int] = field(default_factory=dict)
    claimable_available_counts: Dict[str, int] = field(default_factory=dict)
    protected_counts: Dict[str, int] = field(default_factory=dict)
    excluded_node_ids: List[int] = field(default_factory=list)
    claim_trace: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    context_node_ids: List[int] = field(default_factory=list)
    open_mainline_slots: List[Dict[str, Any]] = field(default_factory=list)
    branch_slots: List[Dict[str, Any]] = field(default_factory=list)
    infeasibility_proof: Dict[str, Any] = field(default_factory=dict)
    attempt_reasons: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": int(self.unit_id),
            "pending_kind": str(self.pending_kind or ""),
            "contract_kind": str(self.contract_kind or ""),
            "target_role": str(self.target_role or ""),
            "real_topology_partial": str(self.real_topology_partial or ""),
            "real_node_partial": str(self.real_node_partial or ""),
            "source_fragment_ids": [int(value) for value in list(self.source_fragment_ids or [])],
            "source_fragment_classes": [str(value) for value in list(self.source_fragment_classes or [])],
            "claimed_source_node_ids": [
                int(value) for value in list(self.claimed_source_node_ids or [])
            ],
            "phase": str(self.phase or ""),
            "subphase": str(self.subphase or ""),
            "required_counts": {str(key): int(value) for key, value in dict(self.required_counts or {}).items()},
            "available_counts": {str(key): int(value) for key, value in dict(self.available_counts or {}).items()},
            "raw_available_counts": {str(key): int(value) for key, value in dict(self.raw_available_counts or {}).items()},
            "claimable_available_counts": {str(key): int(value) for key, value in dict(self.claimable_available_counts or {}).items()},
            "protected_counts": {str(key): int(value) for key, value in dict(self.protected_counts or {}).items()},
            "excluded_node_ids": [int(value) for value in list(self.excluded_node_ids or [])],
            "claim_trace": [dict(item or {}) for item in list(self.claim_trace or [])],
            "reason": str(self.reason or ""),
            "context_node_ids": [int(value) for value in list(self.context_node_ids or [])],
            "open_mainline_slots": [dict(item or {}) for item in list(self.open_mainline_slots or [])],
            "branch_slots": [dict(item or {}) for item in list(self.branch_slots or [])],
            "infeasibility_proof": dict(self.infeasibility_proof or {}),
            "attempt_reasons": [str(value) for value in list(self.attempt_reasons or [])],
            "validation_errors": [str(value) for value in list(self.validation_errors or [])],
        }

    def to_ownership_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        data["claimed_resource_ids_by_su"] = {
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(self.claimed_resource_ids_by_su or {}).items()
        }
        return data

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": int(self.unit_id),
            "real": str(self.real_topology_partial or ""),
            "real_gid": str(self.real_node_partial or ""),
            "pending_kind": str(self.pending_kind or ""),
            "phase": str(self.phase or ""),
            "subphase": str(self.subphase or ""),
            "source_fragment_ids": [int(value) for value in list(self.source_fragment_ids or [])],
            "source_fragment_classes": [str(value) for value in list(self.source_fragment_classes or [])],
            "claimed_source_node_ids": [
                int(value) for value in list(self.claimed_source_node_ids or [])
            ],
            "required_counts": {str(k): int(v) for k, v in dict(self.required_counts or {}).items()},
            "available_counts": {str(k): int(v) for k, v in dict(self.available_counts or {}).items()},
            "reason": str(self.reason or ""),
            "context_node_ids": [int(value) for value in list(self.context_node_ids or [])],
            "open_mainline_slots": [dict(item or {}) for item in list(self.open_mainline_slots or [])],
            "branch_slots": [dict(item or {}) for item in list(self.branch_slots or [])],
            "infeasibility_proof": dict(self.infeasibility_proof or {}),
            "attempt_reasons": [str(value) for value in list(self.attempt_reasons or [])],
            "validation_errors": [str(value) for value in list(self.validation_errors or [])],
        }


@dataclass
class TopologyAssemblyResult(Stage3Serializable):
    ok: bool = False
    reason: str = ""
    feasible: bool = False
    products: List[Dict[str, Any]] = field(default_factory=list)
    pending_units: List[Dict[str, Any]] = field(default_factory=list)
    flex_bounds: Dict[str, Any] = field(default_factory=dict)
    flex_counts: Dict[str, Any] = field(default_factory=dict)
    resource_ledger: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    planned_conversions: List[Dict[str, Any]] = field(default_factory=list)
    flex_decision: Dict[str, Any] = field(default_factory=dict)
    topology_metrics: Dict[str, Any] = field(default_factory=dict)
    topology_signature: str = ""
    requested_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_counts: Dict[str, int] = field(default_factory=dict)
    blocked_reason: str = ""
    active_subphase: str = ""
    active_phase: str = ""
    completed_phases: List[str] = field(default_factory=list)
    phase_metrics: Dict[str, Any] = field(default_factory=dict)
    phase_signature: str = ""
    flex_target_plan: Dict[str, Any] = field(default_factory=dict)
    resource_demand: Dict[str, Any] = field(default_factory=dict)
    protected_resource_counts: Dict[str, int] = field(default_factory=dict)
    referenced_resource_counts: Dict[str, int] = field(default_factory=dict)
    external_candidate_capacity_exhausted: bool = False
    ownership_ledger: Dict[str, Any] = field(default_factory=dict)

    def _public_products(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in list(self.products or []):
            if hasattr(item, "to_public_dict"):
                out.append(item.to_public_dict())
            else:
                product = dict(item or {})
                claimed = {
                    str(key): [int(value) for value in list(values or [])]
                    for key, values in dict(
                        product.get("claimed_resource_ids_by_su", {}) or {}
                    ).items()
                }
                claimed_ids = {
                    int(value) for values in claimed.values() for value in values
                }
                source_ids = {
                    int(value)
                    for value in list(product.get("fixed_source_node_ids", []) or [])
                }
                out.append({
                    "product_type": str(product.get("product_type", product.get("product_kind", "")) or ""),
                    "source_fragment_ids": [int(x) for x in list(product.get("source_fragment_ids", []) or [])],
                    "fixed_source_node_ids": sorted(source_ids - claimed_ids),
                    "claimed_resource_ids_by_su": claimed,
                    "expected_su_type_by_node_id": {
                        str(node_id): int(su_type)
                        for node_id, su_type in dict(
                            product.get("expected_su_type_by_node_id", {}) or {}
                        ).items()
                    },
                    "mainline_slots": [dict(value or {}) for value in list(product.get("mainline_slots", []) or [])],
                    "branch_slots": [dict(value or {}) for value in list(product.get("branch_slots", []) or [])],
                    "topology_notation": str(product.get("topology_notation", product.get("actual_topology_su", "")) or ""),
                    "node_id_notation": str(product.get("node_id_notation", product.get("actual_topology_gid", "")) or ""),
                    "closed": bool(product.get("closed", True)),
                })
        return out

    def _public_pending_units(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in list(self.pending_units or []):
            if hasattr(item, "to_public_dict"):
                out.append(item.to_public_dict())
            else:
                pending = dict(item or {})
                out.append({
                    "real": str(pending.get("real_topology_partial", "")),
                    "real_gid": str(pending.get("real_node_partial", "")),
                    "pending_kind": str(pending.get("pending_kind", "")),
                    "phase": str(pending.get("phase", "")),
                    "subphase": str(pending.get("subphase", "")),
                    "source_fragment_ids": [int(value) for value in list(pending.get("source_fragment_ids", []) or [])],
                    "claimed_source_node_ids": [
                        int(value)
                        for value in list(
                            pending.get("claimed_source_node_ids", []) or []
                        )
                    ],
                    "required_counts": {str(k): int(v) for k, v in dict(pending.get("required_counts", {}) or {}).items()},
                    "available_counts": {str(k): int(v) for k, v in dict(pending.get("available_counts", {}) or {}).items()},
                    "reason": str(pending.get("reason", "")),
                })
        return out

    def _public_resource_ledger(self) -> Dict[str, Any]:
        if hasattr(self.resource_ledger, "to_public_dict"):
            return self.resource_ledger.to_public_dict()
        ledger = dict(self.resource_ledger or {})
        return {
            "consumed_counts": {str(k): int(v) for k, v in dict(ledger.get("consumed_counts", {}) or {}).items()},
            "remaining_counts": {str(k): int(v) for k, v in dict(ledger.get("remaining_counts", {}) or {}).items()},
            "shortage_counts": {str(k): int(v) for k, v in dict(ledger.get("shortage_counts", {}) or {}).items()},
            "released_su24_ids": [
                int(value) for value in list(ledger.get("released_su24_ids", []) or [])
            ],
            "released_resource_ids_by_su": {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(
                    ledger.get("released_resource_ids_by_su", {}) or {}
                ).items()
            },
            "recycled_su24_to_23_ids": [
                int(value)
                for value in list(ledger.get("recycled_su24_to_23_ids", []) or [])
            ],
            "recycled_resource_conversion_ops": [
                dict(item or {})
                for item in list(
                    ledger.get("recycled_resource_conversion_ops", []) or []
                )
            ],
        }

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": str(self.reason or ""),
            "feasible": bool(self.feasible),
            "products": self._public_products(),
            "pending_units": self._public_pending_units(),
            "resource_ledger": self._public_resource_ledger(),
            "flex_decision": _serialize_value(self.flex_decision),
            "topology_metrics": _serialize_value(self.topology_metrics),
            "topology_signature": str(self.topology_signature or ""),
            "requested_counts": {str(k): int(v) for k, v in dict(self.requested_counts or {}).items()},
            "unresolved_counts": {str(k): int(v) for k, v in dict(self.unresolved_counts or {}).items()},
            "blocked_reason": str(self.blocked_reason or ""),
            "active_subphase": str(self.active_subphase or ""),
            "active_phase": str(self.active_phase or ""),
            "completed_phases": [str(value) for value in list(self.completed_phases or [])],
            "phase_metrics": _serialize_value(self.phase_metrics),
            "phase_signature": str(self.phase_signature or ""),
            "flex_target_plan": _serialize_value(self.flex_target_plan),
            "resource_demand": _serialize_value(self.resource_demand),
            "protected_resource_counts": {str(k): int(v) for k, v in dict(self.protected_resource_counts or {}).items()},
            "referenced_resource_counts": {str(k): int(v) for k, v in dict(self.referenced_resource_counts or {}).items()},
            "external_candidate_capacity_exhausted": bool(self.external_candidate_capacity_exhausted),
            "ownership_ledger": _serialize_value(self.ownership_ledger),
        }


@dataclass
class Stage3PhaseEvaluation(Stage3Serializable):
    generation_id: int = 0
    phase: str = "flex_range_reconciliation"
    done: bool = False
    classification_ready: bool = False
    assembly_attempted: bool = False
    flex_range_reconciliation_ok: bool = False
    external_closure_ok: bool = False
    flex_reconciliation_required: bool = False
    flex_reconciliation_scenario: str = ""
    flex_reconciliation_block_reason: str = ""
    flex_bounds: Dict[str, Any] = field(default_factory=dict)
    flex_counts: Dict[str, Any] = field(default_factory=dict)
    flex_audit: Dict[str, Any] = field(default_factory=dict)
    flex_range_case: str = ""
    flex_target_count: int = 0
    explicit_flex_count: int = 0
    explicit_flex_fragment_ids: List[int] = field(default_factory=list)
    supplemental_flex_candidate_ids: List[int] = field(default_factory=list)
    supplemental_flex_candidate_profiles: List[Dict[str, Any]] = field(default_factory=list)
    flex_reconciliation_contracts: List[Dict[str, Any]] = field(default_factory=list)
    flex_reconciliation_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)
    canonical_contracts: List[Dict[str, Any]] = field(default_factory=list)
    flex_range_action_class: str = ""
    active_external_subphase: str = ""
    external_subphase_status: Dict[str, Any] = field(default_factory=dict)
    assembly_view: Dict[str, Any] = field(default_factory=dict)
    classification_view: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    flex_decision: Stage3FlexDecision = field(default_factory=Stage3FlexDecision)
    topology_metrics: Dict[str, Any] = field(default_factory=dict)
    topology_signature: str = ""
    requested_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_counts: Dict[str, int] = field(default_factory=dict)
    blocked_reason: str = ""
    active_phase: str = ""
    completed_phases: List[str] = field(default_factory=list)
    phase_metrics: Dict[str, Any] = field(default_factory=dict)
    phase_signature: str = ""
    round_context: Stage3RoundContext = field(default_factory=Stage3RoundContext)
    ownership_initialized: bool = False
    ownership_ledger: Dict[str, Any] = field(default_factory=dict)
    generic_closure_pool_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)
    su24_ring_resource_ids: List[int] = field(default_factory=list)
    aromatic_donor_ids_by_su: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class Stage3AdjustmentOp(Stage3Serializable):
    op: str = ""
    from_su: Optional[int] = None
    to_su: Optional[int] = None
    node_id: Optional[int] = None
    via_su: Optional[int] = None
    delta: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Stage3AdjustmentBundle(Stage3Serializable):
    bundle_id: str = ""
    phase: str = ""
    bundle_ops: List[Dict[str, Any]] = field(default_factory=list)
    resource_delta: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyClosureContract(Stage3Serializable):
    contract_id: int = -1
    contract_kind: str = ""
    phase: str = ""
    priority: int = 0
    target_role: str = ""
    source_fragment_ids: List[int] = field(default_factory=list)
    source_fragment_classes: List[str] = field(default_factory=list)
    claimed_source_node_ids: List[int] = field(default_factory=list)
    context_node_ids: List[int] = field(default_factory=list)
    mainline_slots: List[TopologyClosureSlot] = field(default_factory=list)
    branch_slots: List[TopologyClosureSlot] = field(default_factory=list)
    slot_requirements: List[Dict[str, Any]] = field(default_factory=list)
    required_counts: Dict[int, int] = field(default_factory=dict)
    required_roles: List[str] = field(default_factory=list)
    expected_topology: str = ""
    reason: str = ""
    blocker_kind: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyResourceTransaction(Stage3Serializable):
    outer_cycle: int = 0
    iteration: int = 0
    phase: str = ""
    contract_id: int = -1
    target_role: str = ""
    source_node_ids: List[int] = field(default_factory=list)
    converted_node_ids: List[int] = field(default_factory=list)
    conversion_path: List[Dict[str, Any]] = field(default_factory=list)
    compensation_node_ids: List[int] = field(default_factory=list)
    claimed_resource_ids: Dict[int, List[int]] = field(default_factory=dict)
    product_node_ids: List[int] = field(default_factory=list)
    slot_assignments: List[Dict[str, Any]] = field(default_factory=list)
    h_before: float = 0.0
    h_after_primary: float = 0.0
    h_after: float = 0.0
    commit_scope: str = "iteration_ledger"
    transaction_origin: str = ""
    committed: bool = False
    rollback_reason: str = ""
    infeasibility_proof: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3PhaseResult(Stage3Serializable):
    active_phase: str = ""
    phase_completed: bool = False
    rebuild_required: bool = False
    primary_adjustment_count: int = 0
    # Internal topology has two independent physical-conversion budgets:
    # ordinary residual assembly and the post-assembly 11-22 reclassification
    # batch.  Keep the aggregate count above for generic callers, but expose
    # the two accounting domains explicitly so they are never rendered as a
    # single ``ordinary + reclassification / one_budget`` value.
    ordinary_internal_primary_adjustment_count: int = 0
    residual_1122_reclassification_primary_adjustment_count: int = 0
    h_compensation_count: int = 0
    parity_balance_count: int = 0
    symbolic_topology_count: int = 0
    transactions: List[TopologyResourceTransaction] = field(default_factory=list)
    stop_reason: str = ""
    infeasibility_proof: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SU24ResourceRoleProfile(Stage3Serializable):
    node_id: int = -1
    role: str = ""
    roles: List[str] = field(default_factory=list)
    available_roles: List[str] = field(default_factory=list)
    hop1_su: Dict[int, int] = field(default_factory=dict)
    hop1_node_ids: List[int] = field(default_factory=list)
    fixed_neighbor_ids: List[int] = field(default_factory=list)
    cap_node_ids: List[int] = field(default_factory=list)
    terminal_node_ids: List[int] = field(default_factory=list)
    terminal_su_by_node_id: Dict[str, int] = field(default_factory=dict)
    open_neighbor_ids: List[int] = field(default_factory=list)
    fixed_neighbor_count: int = 0
    remaining_degree: int = 0
    branch_h_count: int = 0
    requires_su11_cap: bool = False
    allowed_ring_slots: List[str] = field(default_factory=list)
    forbidden_ring_slots: List[str] = field(default_factory=list)
    rejected_reason: str = ""


@dataclass
class HistogramConversion(Stage3Serializable):
    from_su: int = 0
    to_su: int = 0
    count: int = 1
    reason: str = ""
    node_ids: List[int] = field(default_factory=list)
    via_su: Optional[int] = None


@dataclass
class AtomicAdjustmentGroup(Stage3Serializable):
    primary_op: HistogramConversion = field(default_factory=HistogramConversion)
    h_compensation_ops: List[HistogramConversion] = field(default_factory=list)
    combined_delta: Dict[int, int] = field(default_factory=dict)
    demand_before: Dict[int, int] = field(default_factory=dict)
    demand_after: Dict[int, int] = field(default_factory=dict)
    h_before: float = 0.0
    h_after_primary: float = 0.0
    h_after_final: float = 0.0
    scoring_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistogramAdjustmentPlan(Stage3Serializable):
    phase: str = ""
    groups: List[AtomicAdjustmentGroup] = field(default_factory=list)
    primary_adjustment_count: int = 0
    ordinary_internal_primary_adjustment_count: int = 0
    residual_1122_reclassification_primary_adjustment_count: int = 0
    physical_conversion_count: int = 0
    parity_balance_op: Optional[HistogramConversion] = None
    port_role_balance_op: Optional[HistogramConversion] = None
    H_before: Optional[torch.Tensor] = None
    H_after: Optional[torch.Tensor] = None
    demand_before: Dict[int, int] = field(default_factory=dict)
    demand_after: Dict[int, int] = field(default_factory=dict)
    inventory_before: Dict[int, int] = field(default_factory=dict)
    inventory_after: Dict[int, int] = field(default_factory=dict)
    aggregate_inventory_before: Dict[int, int] = field(default_factory=dict)
    aggregate_inventory_after: Dict[int, int] = field(default_factory=dict)
    aggregate_required_counts: Dict[int, int] = field(default_factory=dict)
    protected_resource_counts: Dict[int, int] = field(default_factory=dict)
    accounting_diagnostics: Dict[str, Any] = field(default_factory=dict)
    unresolved_counts: Dict[int, int] = field(default_factory=dict)
    blocked_reason: str = ""
    su11_rotation_index_before: int = 0
    su11_rotation_index_after: int = 0


@dataclass
class Stage3AdjustmentPlan(Stage3Serializable):
    ok: bool = False
    reason: str = ""
    phase: str = ""
    plan_kind: str = ""
    active_subphase: str = ""
    resource_deficit: Dict[int, int] = field(default_factory=dict)
    requirement_contracts: List[Dict[str, Any]] = field(default_factory=list)
    pending_contracts: List[Dict[str, Any]] = field(default_factory=list)
    resource_request_breakdown: Dict[str, Any] = field(default_factory=dict)
    minimum_resource_request: Dict[str, int] = field(default_factory=dict)
    hard_required_counts: Dict[str, int] = field(default_factory=dict)
    preferred_required_counts: Dict[str, int] = field(default_factory=dict)
    protected_resource_counts: Dict[str, int] = field(default_factory=dict)
    planned_conversions: List[Stage3AdjustmentOp] = field(default_factory=list)
    adjustment_bundles: List[Stage3AdjustmentBundle] = field(default_factory=list)
    histogram_adjustment_plan: Optional[HistogramAdjustmentPlan] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    requested_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_counts: Dict[str, int] = field(default_factory=dict)
    blocked_reason: str = ""
    phase_result: Stage3PhaseResult = field(default_factory=Stage3PhaseResult)
    spectrum_candidates: List[Stage3SpectrumCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the durable plan contract, never its live graph runtime."""
        return {
            str(key): _serialize_value(getattr(self, key))
            for key in getattr(self, "__dataclass_fields__", {}).keys()
            if str(key) != "runtime"
        }


@dataclass
class Stage3AdjustmentApplyResult(Stage3Serializable):
    nodes: List[Any] = field(default_factory=list)
    H: Optional[torch.Tensor] = None
    applied_ops: List[Dict[str, Any]] = field(default_factory=list)
    request_ops: List[Dict[str, Any]] = field(default_factory=list)
    rejected_ops: List[Dict[str, Any]] = field(default_factory=list)
    conservation_before: Dict[str, Any] = field(default_factory=dict)
    conservation_after: Dict[str, Any] = field(default_factory=dict)
    changed: bool = False
    requested_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class Stage3WorkflowCycleRecord(Stage3Serializable):
    cycle: int = 0
    attempt: int = 0
    phase: str = ""
    applied: bool = False
    stop_reason: str = ""
    accepted_state_retained: bool = True
    rebuild_valid: bool = False
    layer2_r2: Optional[float] = None
    blocked_reason: str = ""
    accepted: bool = False
    acceptance_reason: str = ""
    primary_adjustment_count: int = 0
    ordinary_internal_primary_adjustment_count: int = 0
    residual_1122_reclassification_primary_adjustment_count: int = 0
    physical_conversion_count: int = 0
    h_compensation_count: int = 0
    parity_balance_count: int = 0
    fresh_rebuild: bool = False
    # Spectrum trial accounting is generation-local.  A candidate (including
    # a two-node pair) consumes one trial, while primary_adjustment_count
    # continues to report the number of physical node conversions.
    spectrum_trial_count: int = 0
    spectrum_max_candidate_trials: int = 0
    spectrum_candidate_batches: int = 0
    spectrum_cap_reached: bool = False
    spectrum_stop_reason: str = ""


@dataclass
class Stage3WorkflowResult(Stage3Serializable):
    nodes: List[Any] = field(default_factory=list)
    H: Optional[torch.Tensor] = None
    diff: Dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    stop_reason: str = ""
    accepted_cycles: int = 0
    cycles: List[Stage3WorkflowCycleRecord] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Nodes, H and difference spectra are returned through the inference
        # interface.  Re-serializing them into metadata retained a second copy
        # of every rebuilt graph across outer cycles.
        diagnostics = {
            str(key): value
            for key, value in dict(self.diagnostics or {}).items()
            if str(key) not in {"completion_snapshot", "terminal_snapshot"}
        }
        return {
            "ok": bool(self.ok),
            "stop_reason": str(self.stop_reason),
            "accepted_cycles": int(self.accepted_cycles),
            "cycles": [_serialize_value(item) for item in list(self.cycles or [])],
            "diagnostics": _serialize_value(diagnostics),
        }
def stage3_contract_owner_key(contract: Dict[str, Any]) -> tuple:
    item = dict(contract or {})
    required_fields = {"contract_id", "contract_kind", "target_role", "claimed_source_node_ids"}
    missing_fields = sorted(required_fields - set(item))
    if missing_fields or not str(item.get("contract_kind", "") or "") or not str(item.get("target_role", "") or ""):
        raise ValueError(f"non-canonical topology contract: missing={missing_fields} contract={item}")
    fragment_ids = tuple(sorted({int(value) for value in list(item.get("source_fragment_ids", []) or [])}))
    node_ids = tuple(sorted({
        int(value) for value in list(item.get("claimed_source_node_ids", []) or [])
    }))
    if fragment_ids:
        return ("fragments", fragment_ids)
    if node_ids:
        return ("real_sources", node_ids)
    return (
        "synthetic",
        str(item.get("phase", "") or ""),
        str(item.get("contract_kind", "") or ""),
        int(item.get("contract_id", 0) or 0),
        str(item.get("reason", "") or ""),
    )


def merge_stage3_contracts_by_owner(contracts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    canonical: Dict[tuple, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    for raw_contract in list(contracts or []):
        contract = dict(raw_contract or {})
        if not contract:
            continue
        owner_key = stage3_contract_owner_key(contract)
        existing = canonical.get(owner_key)
        if existing is None:
            canonical[owner_key] = contract
            continue
        if existing != contract:
            conflicts.append({
                "reason": "duplicate_contract_definition",
                "owner_key": list(owner_key),
                "existing": existing,
                "incoming": contract,
            })
    if conflicts:
        raise ValueError(f"duplicate topology contract definitions: {conflicts}")
    return list(canonical.values())
