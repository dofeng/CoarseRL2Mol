"""Authoritative Layer4 Block-C Stage3 workflow loop.

Detailed topology rules live under ``model.inverse.topology``.
"""

from copy import deepcopy
from collections import Counter
import hashlib
import math
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...shared.inverse_common import (
    SPECIAL_DEGREE_PRIORS,
    normalize_special_degree_meta,
)
from ..topology import Stage3WorkflowCycleRecord, Stage3WorkflowResult
from ..topology.resources.histogram import (
    _h_total,
    _stateful_conversion_bundle_signature,
)
from ..topology.resources.accounting import (
    apply_count_delta,
    build_resource_accounting_snapshot,
    combined_group_delta,
)
from ..topology.resources.conversion_scoring import evaluate_flex_impact
from ..topology.resources.profiles import (
    DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    MIN_STAGE3_AROMATIC_RESOURCE_COUNT,
    h_soft_bounds,
)
from ..topology.resources.types import SU24_RING_RESOURCE_TYPE
from ..topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    Stage3ProgressPhase,
)
from ..topology.resources.spectrum_rebalance import (
    apply_spectrum_candidate,
    histogram_signature_after_operations,
    inverse_bundle_signature,
    spectrum_profile_loss,
)
from ..topology.assembly.closure.chain import ring_recovery_candidate_costs
from ..layer1.port_role_balance import (
    audit_layer1_port_roles,
    select_layer1_port_balance,
)


_MAX_RETAINED_STAGE3_CYCLES = 8
_MAX_RETAINED_GENERATION_SIGNATURES = 64
_DEFAULT_STAGE3_SPECTRUM_ACTIVE_THROUGH_CYCLE = 20


def _console_conversion_count(op: Any) -> int:
    node_ids = [int(value) for value in list(getattr(op, "node_ids", []) or [])]
    return len(node_ids) if node_ids else max(0, int(getattr(op, "count", 0) or 0))


def _console_conversion_path(op: Any) -> str:
    source = int(getattr(op, "from_su", -1))
    target = int(getattr(op, "to_su", -1))
    via = getattr(op, "via_su", None)
    middle = f"->SU{int(via)}" if via is not None else ""
    return f"SU{source}{middle}->SU{target}"


class _Stage3CycleConsoleReporter:
    """Aggregate concise console output without affecting workflow state."""

    def __init__(
        self,
        *,
        verbose: bool,
        flex_limit: int,
        external_limit: int,
        internal_limit: int,
    ) -> None:
        self.verbose = bool(verbose)
        self.flex_limit = max(0, int(flex_limit))
        self.external_limit = max(0, int(external_limit))
        self.internal_limit = max(0, int(internal_limit))
        self.cycle: Optional[int] = None
        self.phases: List[str] = []
        self.attempts = 0
        self.primary_count = 0
        self.compensation_count = 0
        self.recovery_count = 0
        self.primary_paths: Counter[str] = Counter()
        self.balance_paths: Counter[str] = Counter()
        self.external_budget_used = 0
        self.spectrum_trials = 0
        self.spectrum_eligible = 0
        self.summary_emitted = False

    def begin(
        self,
        *,
        cycle: int,
        attempt: int,
        phase: str,
        external_used: int,
    ) -> None:
        if self.cycle != int(cycle):
            self.cycle = int(cycle)
            self.phases = []
            self.attempts = 0
            self.primary_count = 0
            self.compensation_count = 0
            self.recovery_count = 0
            self.primary_paths = Counter()
            self.balance_paths = Counter()
            self.external_budget_used = max(0, int(external_used))
            self.spectrum_trials = 0
            self.spectrum_eligible = 0
            self.summary_emitted = False
            budget = (
                f" external_budget={self.external_budget_used}/{self.external_limit}"
                if str(phase).startswith("external")
                else ""
            )
            print(
                f"\n[Stage3 Cycle {int(cycle)}] phase={str(phase or 'unknown')}{budget}",
                flush=True,
            )
        if str(phase or "") and str(phase) not in self.phases:
            self.phases.append(str(phase))
        self.attempts = max(int(self.attempts), int(attempt))
        self.external_budget_used = max(
            int(self.external_budget_used), max(0, int(external_used))
        )

    def record_histogram(
        self,
        histogram_plan: Any,
        *,
        external_budget_after: Optional[int] = None,
        recovery: bool = False,
    ) -> None:
        if histogram_plan is None:
            return
        for group in list(getattr(histogram_plan, "groups", []) or []):
            op = getattr(group, "primary_op", None)
            if op is not None:
                count = _console_conversion_count(op)
                self.primary_count += int(count)
                self.primary_paths[_console_conversion_path(op)] += int(count)
                if recovery or (
                    int(getattr(op, "from_su", -1)) == 24
                    and int(getattr(op, "to_su", -1)) == 23
                ):
                    self.recovery_count += int(count)
            self.compensation_count += len(
                list(getattr(group, "h_compensation_ops", []) or [])
            )
        for field in ("parity_balance_op", "port_role_balance_op"):
            op = getattr(histogram_plan, field, None)
            if op is not None:
                self.balance_paths[_console_conversion_path(op)] += (
                    _console_conversion_count(op)
                )
        if external_budget_after is not None:
            self.external_budget_used = max(
                int(self.external_budget_used), int(external_budget_after)
            )

    def record_balance_operations(self, operations: Any) -> None:
        for raw in list(operations or []):
            op = raw if not isinstance(raw, dict) else SimpleNamespace(
                from_su=int(raw.get("from_su", -1)),
                to_su=int(raw.get("to_su", -1)),
                via_su=raw.get("via_su"),
                count=int(raw.get("count", 1) or 1),
                node_ids=list(raw.get("node_ids", []) or []),
            )
            self.balance_paths[_console_conversion_path(op)] += (
                _console_conversion_count(op)
            )

    def record_spectrum_trials(self, records: Any) -> None:
        rows = [dict(item or {}) for item in list(records or [])]
        self.spectrum_trials += len(rows)
        self.spectrum_eligible += sum(
            1 for row in rows if bool(row.get("significant", False))
        )

    @staticmethod
    def _format_paths(paths: Counter[str]) -> str:
        return "; ".join(
            f"{path} x{int(count)}" for path, count in paths.items()
        ) or "none"

    def finish(
        self,
        *,
        boundary: str,
        complete: bool = False,
    ) -> None:
        if self.summary_emitted or self.cycle is None:
            return
        phase = " -> ".join(self.phases) or "unknown"
        if phase.startswith("external") or any(
            item.startswith("external") for item in self.phases
        ):
            budget = f"{self.external_budget_used}/{self.external_limit}"
        elif "internal_topology" in self.phases:
            budget = f"{self.primary_count}/{self.internal_limit}"
        elif "flex_range_reconciliation" in self.phases:
            budget = f"{self.primary_count}/{self.flex_limit}"
        else:
            budget = "n/a"
        print(
            f"  conversions: phase={phase} attempts={self.attempts} "
            f"primary={budget} compensation={self.compensation_count}",
            flush=True,
        )
        print(
            f"  primary: {self._format_paths(self.primary_paths)}",
            flush=True,
        )
        if self.recovery_count:
            print(
                f"  recovery: SU24->SU23 x{self.recovery_count}",
                flush=True,
            )
        if self.balance_paths:
            print(
                f"  balance: {self._format_paths(self.balance_paths)}",
                flush=True,
            )
        if self.spectrum_trials:
            print(
                f"  spectrum: trials={self.spectrum_trials} "
                f"eligible={self.spectrum_eligible}",
                flush=True,
            )
        print(
            f"  boundary: {str(boundary)} complete={bool(complete)}",
            flush=True,
        )
        if bool(complete) and self.primary_count == 0 and self.compensation_count == 0:
            print("  complete: no conversions; Stage3 complete", flush=True)
        self.summary_emitted = True

    def rebuild_start(self, reason: str) -> float:
        print(f"  rebuild: start reason={str(reason)}", flush=True)
        return time.perf_counter()

    def rebuild_done(
        self,
        *,
        elapsed: float,
        r2: float,
        difference: Optional[Dict[str, Any]],
    ) -> None:
        layer12 = dict(dict(difference or {}).get("layer12_summary", {}) or {})
        selected = str(layer12.get("selected_variant_id", "") or "unknown")
        matches = (
            f"exact={int(layer12.get('exact_match', 0) or 0)} "
            f"approx={int(layer12.get('approx_match', 0) or 0)} "
            f"missing={int(layer12.get('missing', 0) or 0)}"
        )
        print(
            f"  rebuild: done r2={float(r2):.4f} elapsed={float(elapsed):.2f}s "
            f"selected={selected} {matches}",
            flush=True,
        )

    def spectrum_commit(
        self,
        *,
        records: Any,
        candidate_id: str,
        r2_before: float,
        r2_after: float,
        operations: Any = None,
    ) -> None:
        self.record_spectrum_trials(records)
        operation_text, operation_delta = _spectrum_trial_operations(
            SimpleNamespace(operations=list(operations or []))
        )
        print(
            f"  spectrum: trials={self.spectrum_trials} "
            f"eligible={self.spectrum_eligible} selected={candidate_id} "
            f"r2={float(r2_before):.4f}->{float(r2_after):.4f} "
            f"gain={float(r2_after) - float(r2_before):.4f}",
            flush=True,
        )
        print(
            f"  spectrum_adjustments: {operation_text} delta={operation_delta}",
            flush=True,
        )
        print("  rebuild: committed spectrum candidate", flush=True)
        self.summary_emitted = True


def _append_stage3_cycle(
    cycles: List[Stage3WorkflowCycleRecord],
    record: Stage3WorkflowCycleRecord,
) -> None:
    if int(getattr(record, "attempt", 0) or 0) <= 0:
        record.attempt = 1 + max(
            [
                int(getattr(previous, "attempt", 0) or 0)
                for previous in cycles
                if int(getattr(previous, "cycle", -1)) == int(record.cycle)
            ],
            default=0,
        )
    cycles.append(record)
    overflow = len(cycles) - int(_MAX_RETAINED_STAGE3_CYCLES)
    if overflow > 0:
        del cycles[:overflow]


def _clear_stage3_node_runtime_state(nodes: List[Any]) -> None:
    for node in list(nodes or []):
        if hasattr(node, "stage3_topology_owner_id"):
            node.stage3_topology_owner_id = None
        if hasattr(node, "stage3_topology_role"):
            node.stage3_topology_role = None
        if hasattr(node, "stage3_locked_neighbor_ids"):
            node.stage3_locked_neighbor_ids = set()
        if hasattr(node, "stage3_resource_claimed"):
            node.stage3_resource_claimed = False


def _compact_stage3_plan_dict(plan: Any) -> Dict[str, Any]:
    """Return only fields needed by the bounded cycle summary."""
    phase_result_obj = getattr(plan, "phase_result", None)
    phase_result = (
        phase_result_obj.to_dict()
        if phase_result_obj is not None and hasattr(phase_result_obj, "to_dict")
        else {}
    )
    transactions = list(phase_result.pop("transactions", []) or [])
    phase_result["transaction_count"] = len(transactions)
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostic_keys = {
        "stage3_phase",
        "completed_phases",
        "external_contracts_seen",
        "external_directly_completed",
        "external_completed_after_conversion",
        "external_blocked",
        "external_remaining_pending",
        "external_primary_count",
        "external_remaining_budget",
        "external_stage_completed",
        "stage_rebuild_reason",
        "residual_infeasibility_proof",
        "h_constraint",
        "h_before",
        "h_after",
        "h_delta",
    }
    compact_diagnostics = {
        key: diagnostics[key]
        for key in diagnostic_keys
        if key in diagnostics
    }
    planned = [
        item.to_dict() if hasattr(item, "to_dict") else dict(item or {})
        for item in list(getattr(plan, "planned_conversions", []) or [])
    ]
    return {
        "ok": bool(getattr(plan, "ok", False)),
        "reason": str(getattr(plan, "reason", "") or ""),
        "phase": str(getattr(plan, "phase", "") or ""),
        "plan_kind": str(getattr(plan, "plan_kind", "") or ""),
        "active_subphase": str(getattr(plan, "active_subphase", "") or ""),
        "resource_deficit": dict(getattr(plan, "resource_deficit", {}) or {}),
        "requested_counts": dict(getattr(plan, "requested_counts", {}) or {}),
        "unresolved_counts": dict(getattr(plan, "unresolved_counts", {}) or {}),
        "blocked_reason": str(getattr(plan, "blocked_reason", "") or ""),
        "planned_conversions": planned,
        "phase_result": phase_result,
        "diagnostics": compact_diagnostics,
    }


def _stage3_runtime_snapshot(plan: Any) -> Dict[str, Any]:
    """Capture the single completion contract needed by the final audit."""
    runtime = dict(getattr(plan, "runtime", {}) or {})

    def payload(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            converted = value.to_dict()
            return dict(converted) if isinstance(converted, dict) else {}
        return dict(value) if isinstance(value, dict) else {}

    phase_eval = runtime.get("phase_evaluation")
    live_assembler = runtime.get("external_assembler") or getattr(
        phase_eval, "external_assembler", None
    )
    raw_ledger = payload(runtime.get("iteration_topology_ledger"))
    # Keep the complete serialized ledger.  The selected fields below are
    # normalized for consumers, while release/recycle provenance and claim
    # diagnostics must remain available to terminal snapshot exporters.
    ledger = {
        **raw_ledger,
        "generation_id": int(raw_ledger.get("generation_id", 0) or 0),
        "ownership_by_node_id": dict(raw_ledger.get("ownership_by_node_id", {}) or {}),
        "completed_products": list(raw_ledger.get("completed_products", []) or []),
        "pending_contracts": list(raw_ledger.get("pending_contracts", []) or []),
        "available_ids_by_su": dict(raw_ledger.get("available_ids_by_su", {}) or {}),
        "conflicts": list(raw_ledger.get("conflicts", []) or []),
    }
    # The live assembler is the authoritative source at a terminal internal
    # failure.  Its products/pending contracts must survive even when the
    # state-machine plan itself contains no newly committed transaction.
    live_products = [
        product.to_dict() if hasattr(product, "to_dict") else dict(product or {})
        for product in list(getattr(live_assembler, "products", []) or [])
        if dict(product.to_dict() if hasattr(product, "to_dict") else product or {})
    ]
    # Merge every available runtime representation.  A continuation plan may
    # carry a ledger containing only the newly committed residual product while
    # ``assembly_view`` still holds the earlier external products; either one
    # alone is therefore incomplete.
    assembly_products = (
        dict(getattr(phase_eval, "assembly_view", {}) or {}).get("products", [])
        if phase_eval is not None
        else []
    )
    live_products = _merge_stage3_committed_products(
        list(ledger.get("completed_products", []) or []),
        live_products,
        assembly_products,
    )
    live_pending = [
        pending.to_ownership_dict()
        if hasattr(pending, "to_ownership_dict")
        else pending.to_dict()
        if hasattr(pending, "to_dict")
        else dict(pending or {})
        for pending in list(getattr(live_assembler, "pending_units", []) or [])
    ]
    live_pending = [
        dict(item or {}) for item in list(ledger.get("pending_contracts", []) or [])
    ] + live_pending
    if not live_pending and phase_eval is not None:
        live_pending.extend(
            dict(item or {})
            for item in list(
                dict(getattr(phase_eval, "assembly_view", {}) or {}).get(
                    "pending_units", []
                )
                or []
            )
        )
    if not live_pending:
        live_pending.extend(
            dict(item or {})
            for item in list(getattr(plan, "pending_contracts", []) or [])
        )
    live_contracts = [
        dict(contract or {})
        for contract in list(getattr(live_assembler, "pending_contracts", []) or [])
        if dict(contract or {})
    ]
    if live_contracts:
        live_pending.extend(live_contracts)
    if live_assembler is not None:
        allocator = getattr(live_assembler, "allocator", None)
        if allocator is not None:
            ledger["available_ids_by_su"] = {
                int(key): [int(value) for value in list(values or [])]
                for key, values in dict(
                    getattr(allocator, "remaining_ids_by_su", {}) or {}
                ).items()
            }
        ledger["completed_products"] = list(live_products)
        ledger["pending_contracts"] = list(live_pending)
    # A lone physical SU24 has no fragment contract, but it is still an
    # unfinished internal resource action. Keep it visible in the pending
    # export without changing its ownership state from resource_pool.
    available_ids_by_su = dict(ledger.get("available_ids_by_su", {}) or {})
    remaining_su24 = [
        int(value)
        for value in list(
            available_ids_by_su.get("24", [])
            or available_ids_by_su.get(24, [])
            or []
        )
    ]
    snapshot_phases = {
        str(getattr(plan, "phase", "") or ""),
        str(getattr(plan, "active_subphase", "") or ""),
    }
    # SU24 is never a valid terminal residual.  Export a synthetic internal
    # conversion contract whenever one remains in the allocator, regardless
    # of which phase label a terminal plan happened to carry.  This keeps an
    # isolated SU24 visible to the subsequent internal assembly and prevents
    # a misleading "complete" snapshot.
    internal_su24_pending = bool(snapshot_phases & {
        "internal_topology",
        "residual_resource_self_closure",
        "complete",
        "remaining_pool_spectrum_rebalance",
        "external_topology",
        "external_topology_assembly",
    })
    if remaining_su24 and internal_su24_pending:
        available_counts = {
            str(key): len(list(values or []))
            for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
        }
        live_pending.extend({
            "contract_id": -900000 - int(node_id),
            "pending_kind": "internal_su24_resource_conversion",
            "contract_kind": "internal_resource_conversion",
            "target_role": "internal_su24_aromatic_donor",
            "phase": "internal_topology",
            "subphase": "residual_resource_self_closure",
            "real_topology_partial": "SU24",
            "real_node_partial": str(node_id),
            # Exporters consume the canonical compact pending names.  Keep
            # the *_partial aliases above for diagnostics, but do not force
            # downstream writers to know about the synthetic contract.
            "real": "SU24",
            "real_gid": str(node_id),
            "source_fragment_ids": [],
            "claimed_source_node_ids": [],
            "required_counts": {"12": 1, "13": 1},
            "required_alternatives": [{"12": 1}, {"13": 1}],
            "available_counts": available_counts,
            "reason": "isolated_su24_requires_internal_12_or_13_conversion",
        } for node_id in remaining_su24)
        ledger["pending_contracts"] = list(live_pending)
    pending_by_key: Dict[str, Dict[str, Any]] = {}
    for item in list(live_pending):
        payload_item = dict(item or {})
        if not payload_item:
            continue
        raw_contract_id = payload_item.get("contract_id")
        if raw_contract_id not in (None, "", -1, "-1"):
            pending_key = f"contract:{raw_contract_id}"
        else:
            pending_key = (
                f"{payload_item.get('pending_kind', '')}:"
                f"{payload_item.get('real_gid', payload_item.get('real_node_partial', ''))}:"
                f"{','.join(str(value) for value in list(payload_item.get('claimed_source_node_ids', []) or []))}"
            )
        pending_by_key[pending_key] = payload_item
    live_pending = list(pending_by_key.values())
    ledger["pending_contracts"] = list(live_pending)
    ownership_states: Dict[str, List[int]] = {}
    for raw_owner in dict(ledger.get("ownership_by_node_id", {}) or {}).values():
        owner = dict(raw_owner or {})
        state = str(owner.get("state", "unclassified_or_conflict") or "unclassified_or_conflict")
        ownership_states.setdefault(state, []).append(int(owner.get("node_id", -1)))
    ownership_states = {
        str(key): sorted({int(value) for value in values if int(value) >= 0})
        for key, values in ownership_states.items()
    }
    available = {
        str(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
    }
    ownership_ledger = {
        "states": ownership_states,
        "generic_closure_pool_ids_by_su": {
            str(su): list(available.get(str(su), [])) for su in (11, 22, 23)
        },
        "su24_ring_resource_ids": list(available.get("24", [])),
        "aromatic_donor_ids_by_su": {
            str(su): list(available.get(str(su), [])) for su in (10, 12, 13)
        },
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
        "invariant_errors": list(ledger.get("conflicts", []) or []),
    }
    raw_classification = payload(runtime.get("classification_before_adjustment"))
    if not raw_classification and phase_eval is not None:
        raw_classification = payload(getattr(phase_eval, "classification_view", {}))
    classification = {
        "summary": dict(raw_classification.get("summary", {}) or {}),
        "resource_pool": dict(raw_classification.get("resource_pool", {}) or {}),
    }
    raw_assembly = dict(
        runtime.get("assembly_before_adjustment", {})
        or getattr(phase_eval, "assembly_view", {})
        or {}
    )
    assembly = {
        key: raw_assembly.get(key, {})
        for key in ("resource_ledger", "flex_bounds", "flex_counts", "flex_target_plan")
    }
    resource_ledger = dict(assembly.get("resource_ledger", {}) or {})
    resource_ledger["remaining_ids_by_su"] = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
    }
    resource_ledger["remaining_counts"] = {
        str(key): len(list(values or []))
        for key, values in dict(ledger.get("available_ids_by_su", {}) or {}).items()
    }
    assembly["resource_ledger"] = resource_ledger
    if not ownership_ledger["released_resource_ids_by_su"]:
        ownership_ledger["released_resource_ids_by_su"] = {
            str(key): [int(value) for value in list(values or [])]
            for key, values in dict(
                resource_ledger.get("released_resource_ids_by_su", {}) or {}
            ).items()
        }
    if not ownership_ledger["recycled_su24_to_23_ids"]:
        ownership_ledger["recycled_su24_to_23_ids"] = [
            int(value)
            for value in list(
                resource_ledger.get("recycled_su24_to_23_ids", []) or []
            )
        ]
    if not ownership_ledger["released_su24_ids"]:
        ownership_ledger["released_su24_ids"] = [
            int(value)
            for value in list(resource_ledger.get("released_su24_ids", []) or [])
        ]
    return {
        "generation_id": int(
            ledger.get(
                "generation_id",
                getattr(phase_eval, "generation_id", 0) if phase_eval is not None else 0,
            )
            or 0
        ),
        "classification": classification,
        "external_assembly": assembly,
        "resource_ledger": resource_ledger,
        "iteration_topology_ledger": ledger,
        "products": list(live_products),
        "pending_contracts": list(live_pending),
        "ownership_ledger": ownership_ledger,
    }


def _stage3_runtime_generation_id(plan: Any) -> int:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    ledger = runtime.get("iteration_topology_ledger")
    if hasattr(ledger, "generation_id"):
        return int(getattr(ledger, "generation_id", 0) or 0)
    if isinstance(ledger, dict):
        return int(ledger.get("generation_id", 0) or 0)
    phase_eval = runtime.get("phase_evaluation")
    return int(getattr(phase_eval, "generation_id", 0) or 0)


def _progress_phase_after_plan(
    plan: Any,
    current: Stage3ProgressPhase,
) -> Stage3ProgressPhase:
    phase = str(getattr(plan, "phase", "") or "")
    return {
        "flex_range_reconciliation": Stage3ProgressPhase.FLEX_RECONCILIATION,
        "external_topology": Stage3ProgressPhase.EXTERNAL_TOPOLOGY,
        "external_topology_assembly": Stage3ProgressPhase.EXTERNAL_TOPOLOGY,
        "remaining_pool_spectrum_rebalance": Stage3ProgressPhase.SPECTRUM_REBALANCE,
        "residual_resource_self_closure": Stage3ProgressPhase.INTERNAL_TOPOLOGY,
        "internal_topology": Stage3ProgressPhase.INTERNAL_TOPOLOGY,
        "complete": Stage3ProgressPhase.COMPLETE,
    }.get(phase, current)


def _freeze_stage3_signature_value(value: Any) -> Any:
    """Convert Stage3 context values into a deterministic hashable payload."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return (
            "tensor",
            str(tensor.dtype),
            tuple(int(item) for item in tensor.shape),
            tuple(tensor.reshape(-1).tolist()),
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return (
            "array",
            str(array.dtype),
            tuple(int(item) for item in array.shape),
            tuple(array.reshape(-1).tolist()),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    str(key),
                    _freeze_stage3_signature_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_stage3_signature_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        frozen = [_freeze_stage3_signature_value(item) for item in value]
        return tuple(sorted(frozen, key=repr))
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _freeze_stage3_signature_value(value.to_dict())
    return repr(value)


def _stage3_physical_state_signature(
    nodes: List[Any],
    H: torch.Tensor,
    context: Any = None,
) -> str:
    """Hash only the physical state that survives a Layer1/2 generation."""
    histogram = tuple(
        int(value)
        for value in torch.clamp(H.detach().cpu(), min=0).long().tolist()
    )
    node_rows = []
    edge_rows = set()
    for node in sorted(
        list(nodes or []),
        key=lambda item: int(getattr(item, "global_id", -1)),
    ):
        node_id = int(getattr(node, "global_id", -1))
        node_rows.append((node_id, int(getattr(node, "su_type", -1))))
        for neighbor_id in list(getattr(node, "hop1_ids", []) or []):
            edge_rows.add(tuple(sorted((node_id, int(neighbor_id)))))
    payload = (
        histogram,
        tuple(node_rows),
        tuple(sorted(edge_rows)),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _stage3_chemical_state_signature(
    nodes: List[Any],
    H: torch.Tensor,
) -> str:
    """Hash the chemical assignment independently of rebuilt Layer1 edges.

    Layer1/2 reconstruction can choose a different valid edge realization for
    the same real-node SU assignment.  Such a graph is still the same physical
    resource state for Stage3 convergence purposes, so edge-free detection is
    kept alongside the stricter full graph signature.
    """
    histogram = tuple(
        int(value)
        for value in torch.clamp(H.detach().cpu(), min=0).long().tolist()
    )
    node_rows = tuple(
        (
            int(getattr(node, "global_id", -1)),
            int(getattr(node, "su_type", -1)),
        )
        for node in sorted(
            list(nodes or []),
            key=lambda item: int(getattr(item, "global_id", -1)),
        )
    )
    return hashlib.sha1(repr((histogram, node_rows)).encode("utf-8")).hexdigest()


def _spectrum_search_baseline_key(
    nodes: List[Any],
    H: torch.Tensor,
    difference_spectrum: Dict[str, Any],
) -> str:
    digest = hashlib.sha1()
    histogram = np.asarray(
        torch.clamp(H.detach().cpu(), min=0).long().tolist(),
        dtype=np.int64,
    )
    digest.update(histogram.tobytes())
    digest.update(_stage3_physical_state_signature(nodes, H).encode("ascii"))
    diff = np.ascontiguousarray(
        np.asarray(dict(difference_spectrum or {}).get("diff", []), dtype=np.float64).reshape(-1)
    )
    digest.update(diff.tobytes())
    return digest.hexdigest()


def _spectrum_search_memory_entry(adjuster: Any, baseline_key: str) -> Dict[str, set[str]]:
    memory = getattr(adjuster, "_spectrum_search_memory", None)
    if not isinstance(memory, dict):
        memory = {}
        adjuster._spectrum_search_memory = memory
    raw_entry = memory.setdefault(str(baseline_key), {})
    fields = (
        "primary_histogram_signatures",
        "final_histogram_signatures",
        "bundle_signatures",
        "rejected_inverse_signatures",
    )
    entry: Dict[str, set[str]] = {}
    for field in fields:
        values = raw_entry.get(field, set())
        if not isinstance(values, set):
            values = {str(value) for value in list(values or []) if str(value)}
        raw_entry[field] = values
        entry[field] = values
    return entry


def _update_generation_metrics(
    metrics: List[Dict[str, int]],
    *,
    generation_id: int,
    plan: Any,
) -> None:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    phase = str(getattr(plan, "phase", "") or "")
    if phase not in {"internal_topology", "complete"}:
        return
    if not metrics or int(metrics[-1].get("generation_id", -1)) != int(generation_id):
        metrics.append({
            "generation_id": int(generation_id),
            "internal_physical_conversion_count": 0,
            "internal_ring_count": 0,
        })
    metrics[-1]["internal_physical_conversion_count"] = int(
        diagnostics.get("internal_physical_conversion_count", 0) or 0
    )
    metrics[-1]["internal_ring_count"] = int(
        diagnostics.get("internal_ring_count", 0) or 0
    )
    if len(metrics) > _MAX_RETAINED_GENERATION_SIGNATURES:
        del metrics[:-_MAX_RETAINED_GENERATION_SIGNATURES]


def _stage3_r2(diff: Optional[Dict[str, Any]], fallback: Any = float("nan")) -> float:
    try:
        value = float(dict(diff or {}).get("r2", fallback))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _stage3_flex_audit_from_plan(plan: Any) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    return dict(getattr(phase_eval, "flex_audit", {}) or {})


def _histogram_dict(H: torch.Tensor) -> Dict[str, int]:
    values = torch.clamp(H.detach().cpu(), min=0).long().tolist()
    return {str(index): int(value) for index, value in enumerate(values) if int(value) != 0}


def _resource_gap(inventory: Dict[int, int], demand: Dict[int, int]) -> Dict[str, int]:
    keys = set(int(key) for key in inventory) | set(int(key) for key in demand)
    return {
        str(key): int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0))
        for key in sorted(keys)
        if int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0)) != 0
    }


def _mapping_payload(value: Any, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return dict(converted)
    raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")


def _stage3_product_identity(product: Dict[str, Any]) -> str:
    payload = dict(product or {})
    node_ids = sorted({
        int(value)
        for value in list(payload.get("fixed_source_node_ids", []) or [])
        + [
            int(value)
            for values in dict(
                payload.get("claimed_resource_ids_by_su", {}) or {}
            ).values()
            for value in list(values or [])
        ]
    })
    # A physical product is identified by the concrete nodes it owns.  The
    # assembler and ledger may assign different product IDs while exporting
    # the same materialized product; using those IDs as the primary key
    # duplicates rings (and double-counts their SU24 resources) in the final
    # snapshot.  Product IDs remain a fallback only for node-less diagnostics.
    if node_ids:
        return f"nodes:{','.join(str(value) for value in node_ids)}"
    product_id = str(payload.get("product_id", "") or "")
    if product_id:
        return f"id:{product_id}"
    return (
        f"phase:{str(payload.get('phase', '') or '')}|"
        f"contract:{int(payload.get('contract_id', -1) or -1)}|"
        "nodes:"
    )


def _merge_stage3_committed_products(
    current: List[Dict[str, Any]],
    *collections: Any,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {
        _stage3_product_identity(dict(product or {})): dict(product or {})
        for product in list(current or [])
        if dict(product or {})
    }
    for collection in collections:
        for product in list(collection or []):
            payload = dict(product or {})
            if payload:
                merged[_stage3_product_identity(payload)] = payload
    return [merged[key] for key in sorted(merged)]


def _target_h(E_target: Optional[torch.Tensor]) -> Optional[float]:
    try:
        value = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        return None
    return value if value > 0.0 else None


def _h_soft_diagnostics(
    *,
    H: torch.Tensor,
    E_target: Optional[torch.Tensor],
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target_h = _target_h(E_target)
    current_h = float(_h_total(H, special_degree_meta))
    lower, upper = h_soft_bounds(target_h, h_tolerance)
    final_outside = bool(
        lower is not None
        and upper is not None
        and not float(lower) <= float(current_h) <= float(upper)
    )
    return {
        "h_soft_constraint": {
            "topology_gate": False,
            "target_h": float(target_h) if target_h is not None else None,
            "current_h": float(current_h),
            "soft_lower": float(lower) if lower is not None else None,
            "soft_upper": float(upper) if upper is not None else None,
            "outside_soft_window": bool(final_outside),
        },
    }


def _selected_histogram_counts(H: Optional[torch.Tensor]) -> str:
    values = (
        torch.clamp(H.detach().cpu(), min=0).long()
        if isinstance(H, torch.Tensor)
        else torch.zeros(25, dtype=torch.long)
    )
    return " ".join(
        f"SU{su}={int(values[su].item()) if su < int(values.numel()) else 0}"
        for su in (11, 12, 13, 22, 23, 24)
    )


def _format_conversion(op: Any) -> str:
    via = (
        f"->SU{int(getattr(op, 'via_su'))}"
        if getattr(op, "via_su", None) is not None
        else ""
    )
    return (
        f"SU{int(getattr(op, 'from_su', -1))}{via}"
        f"->SU{int(getattr(op, 'to_su', -1))}"
        f" x{int(getattr(op, 'count', 1) or 1)}"
        f" nodes={list(getattr(op, 'node_ids', []) or [])}"
        f" reason={str(getattr(op, 'reason', '') or '')}"
    )


def _format_conversion_compact(op: Any) -> str:
    via = (
        f"->SU{int(getattr(op, 'via_su'))}"
        if getattr(op, "via_su", None) is not None
        else ""
    )
    return (
        f"SU{int(getattr(op, 'from_su', -1))}{via}"
        f"->SU{int(getattr(op, 'to_su', -1))}"
        f" x{int(getattr(op, 'count', 1) or 1)}"
    )


def _spectrum_trial_operations(candidate: Any) -> Tuple[str, Dict[int, int]]:
    descriptions: List[str] = []
    delta: Dict[int, int] = {}
    for raw in list(getattr(candidate, "operations", []) or []):
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        node_id = int(op.get("node_id", -1))
        descriptions.append(f"node={node_id}:SU{source}->SU{target}")
        delta[source] = int(delta.get(source, 0)) - 1
        delta[target] = int(delta.get(target, 0)) + 1
    return ", ".join(descriptions) or "none", {
        int(key): int(value) for key, value in sorted(delta.items()) if int(value) != 0
    }


def _short_candidate_id(candidate: Any, limit: int = 52) -> str:
    candidate_id = str(getattr(candidate, "candidate_id", "") or "")
    if len(candidate_id) <= int(limit):
        return candidate_id
    return f"{candidate_id[: max(1, int(limit) - 9)]}...{candidate_id[-6:]}"


def _print_stage3_plan_summary(
    cycle_number: int,
    plan: Any,
    H_current: torch.Tensor,
    E_target: Optional[torch.Tensor],
    special_degree_meta: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    attempt_number: int = 1,
    external_used_before: int = 0,
    external_limit: int = 0,
) -> None:
    histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
    groups = list(getattr(histogram_plan, "groups", []) or [])
    H_after = getattr(histogram_plan, "H_after", None)
    if not isinstance(H_after, torch.Tensor):
        H_after = H_current
    h_before = (
        float(getattr(groups[0], "h_before", 0.0))
        if groups
        else float(_h_total(H_current, special_degree_meta))
    )
    h_after = float(_h_total(H_after, special_degree_meta))
    target_h = _target_h(E_target)
    plan_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    h_constraint = dict(plan_diagnostics.get("h_constraint", {}) or {})
    h_tolerance = float(
        h_constraint.get(
            "effective_h_tolerance",
            DEFAULT_STAGE3_H_SOFT_TOLERANCE,
        )
    )
    lower, upper = h_soft_bounds(target_h, h_tolerance)
    compensation_count = sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in groups
    )
    phase = str(
        getattr(plan, "phase", "")
        or dict(getattr(plan, "diagnostics", {}) or {}).get("stage3_phase", "")
    )
    primary_count = int(
        getattr(histogram_plan, "primary_adjustment_count", len(groups))
        if histogram_plan is not None
        else 0
    )
    if primary_count <= 0 and groups:
        primary_count = sum(
            max(
                1,
                len(list(getattr(getattr(group, "primary_op", None), "node_ids", []) or [])),
            )
            for group in groups
        )
    external_status = ""
    if phase.startswith("external"):
        external_used_after = min(
            max(0, int(external_limit)),
            max(0, int(external_used_before)) + max(0, int(primary_count)),
        )
        phase_result = getattr(plan, "phase_result", None)
        external_status = (
            f" external_budget={external_used_after}/{max(0, int(external_limit))}"
            f" rebuild_boundary={bool(getattr(phase_result, 'rebuild_required', False))}"
        )
    primary_ops = [
        _format_conversion_compact(getattr(group, "primary_op", None))
        for group in groups
    ]
    conversion_status = (
        f" conversions=[{'; '.join(primary_ops)}]" if primary_ops else " conversions=[]"
    )
    print(
        f"[Stage3 Plan] cycle={int(cycle_number)} "
        f"attempt={int(attempt_number)} phase={phase or 'unknown'} "
        f"kind={str(getattr(plan, 'plan_kind', '') or '')} "
        f"reason={str(getattr(plan, 'reason', '') or '')} "
        f"primary={primary_count} compensation={int(compensation_count)}"
        f"{external_status}{conversion_status}",
        flush=True,
    )
    if not verbose:
        return
    print(
        "  [Stage3 H] "
        f"current={h_before:.1f} target={target_h if target_h is not None else 'none'} "
        f"soft=[{lower if lower is not None else 'none'},"
        f"{upper if upper is not None else 'none'}] planned={h_after:.1f}",
        flush=True,
    )
    print(
        f"  [Stage3 Histogram] before: {_selected_histogram_counts(H_current)}",
        flush=True,
    )
    print(
        f"  [Stage3 Histogram] after:  {_selected_histogram_counts(H_after)}",
        flush=True,
    )
    if str(getattr(plan, "plan_kind", "") or "") == "stage3_completion_infeasible":
        phase_result = getattr(plan, "phase_result", None)
        proof = dict(
            getattr(phase_result, "infeasibility_proof", {}) or {}
        )
        print(
            "  [Stage3 Completion Invariants] "
            f"failed={dict(proof.get('failed_invariants', {}) or {})} "
            f"remaining_su24={list(proof.get('remaining_su24_node_ids', []) or [])} "
            f"flex_final={proof.get('flex_final')} flex_bounds={dict(proof.get('flex_bounds', {}) or {})} "
            f"fragment_coverage={dict(proof.get('fragment_coverage', {}) or {})}",
            flush=True,
        )


def _print_stage3_histogram_adjustments(
    cycle_number: int,
    histogram_plan: Any,
    *,
    verbose: bool = False,
    attempt_number: int = 1,
    emit_header: bool = True,
) -> None:
    groups = list(getattr(histogram_plan, "groups", []) or [])
    parity_op = getattr(histogram_plan, "parity_balance_op", None)
    role_op = getattr(histogram_plan, "port_role_balance_op", None)
    if not groups and parity_op is None and role_op is None:
        if verbose and emit_header:
            print(
                f"[Block C:resource #{cycle_number} attempt={attempt_number}] "
                "no physical histogram adjustment",
                flush=True,
            )
        return
    if not verbose:
        primary_ops = [
            _format_conversion_compact(getattr(group, "primary_op", None))
            for group in groups
        ]
        balance_ops = [
            _format_conversion_compact(op)
            for op in (parity_op, role_op)
            if op is not None
        ]
        details = f" primary=[{'; '.join(primary_ops)}]" if primary_ops else ""
        if balance_ops:
            details += f" balance=[{'; '.join(balance_ops)}]"
        print(
            f"[Block C:resource #{cycle_number} attempt={attempt_number}] "
            f"conversions={len(groups)}{details}",
            flush=True,
        )
        return
    if groups and emit_header:
        print(
            f"[Block C:resource #{cycle_number} attempt={attempt_number}] "
            f"structure-unit adjustments: {len(groups)} primary bundle(s)",
            flush=True,
        )
    for group_index, group in enumerate(groups, start=1):
        primary = getattr(group, "primary_op", None)
        compensation = list(getattr(group, "h_compensation_ops", []) or [])
        scoring = dict(getattr(group, "scoring_diagnostics", {}) or {})
        h_scoring = dict(scoring.get("h", {}) or {})
        flex_scoring = dict(scoring.get("flex", {}) or {})
        print(
            f"  [Adjustment {group_index}] primary={_format_conversion(primary)} "
            f"delta={dict(getattr(group, 'combined_delta', {}) or {})} "
            f"H={float(getattr(group, 'h_before', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_primary', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_final', 0.0)):.1f}",
            flush=True,
        )
        for compensation_index, op in enumerate(compensation, start=1):
            print(
                f"    [H Compensation {compensation_index}] {_format_conversion(op)}",
                flush=True,
            )
        if not compensation:
            print("    [H Compensation] none", flush=True)
        print(
            "    [Adjustment Score] "
            f"resource_gap={dict(scoring.get('resource_gap_before', getattr(group, 'demand_before', {})) or {})}"
            f"->{dict(scoring.get('resource_gap_after', getattr(group, 'demand_after', {})) or {})} "
            f"H_soft_window_penalty={float(h_scoring.get('soft_window_penalty', 0.0) or 0.0):.4f} "
            f"H_drift_penalty={float(h_scoring.get('drift_penalty', 0.0) or 0.0):.4f} "
            f"flex_penalty={float(flex_scoring.get('penalty', 0.0) or 0.0):.4f} "
            f"total={float(scoring.get('total_score', 0.0) or 0.0):.4f}",
            flush=True,
        )
        if flex_scoring:
            flex_before = dict(flex_scoring.get("before", {}) or {})
            flex_after = dict(flex_scoring.get("after", {}) or {})
            print(
                "    [Flex Impact] "
                f"X/Y/Z={flex_before.get('X', 0)}/{flex_before.get('Y', 0)}/{flex_before.get('Z', 0)}"
                f"->{flex_after.get('X', 0)}/{flex_after.get('Y', 0)}/{flex_after.get('Z', 0)} "
                f"range={flex_before.get('flex_lower', 0)}-{flex_before.get('flex_upper', 0)}"
                f"->{flex_after.get('flex_lower', 0)}-{flex_after.get('flex_upper', 0)} "
                f"gap={flex_before.get('minimum_flex_gap', 0)}"
                f"->{flex_after.get('minimum_flex_gap', 0)}",
                flush=True,
            )
    if parity_op is not None:
        print(
            f"  [Port Parity Balance] {_format_conversion(parity_op)}",
            flush=True,
        )
    if role_op is not None:
        print(
            f"  [Port Role Balance] {_format_conversion(role_op)}",
            flush=True,
        )


def _histogram_h_compensation_count(histogram_plan: Any) -> int:
    return sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in list(getattr(histogram_plan, "groups", []) or [])
    )


def _histogram_parity_balance_count(histogram_plan: Any) -> int:
    return sum(
        getattr(histogram_plan, field, None) is not None
        for field in ("parity_balance_op", "port_role_balance_op")
    )


def _primary_conversion_node_ids(histogram_plan: Any) -> List[int]:
    """Concrete nodes changed by primary SU operations (not H/port balance)."""
    node_ids: List[int] = []
    for group in list(getattr(histogram_plan, "groups", []) or []):
        op = getattr(group, "primary_op", None)
        if op is None:
            continue
        raw_ids = list(getattr(op, "node_ids", []) or [])
        expected = max(0, int(getattr(op, "count", 0) or 0))
        if expected <= 0 or len(raw_ids) != expected:
            raise ValueError(
                "primary_conversion_requires_concrete_node_ids: "
                f"expected={expected} provided={len(raw_ids)}"
            )
        if any(int(value) < 0 for value in raw_ids):
            raise ValueError("primary_conversion_contains_invalid_node_id")
        node_ids.extend(int(value) for value in raw_ids)
    return node_ids


def _external_primary_conversion_count(histogram_plan: Any) -> int:
    ids = _primary_conversion_node_ids(histogram_plan)
    if len(ids) != len(set(ids)):
        raise ValueError("primary_conversion_reuses_concrete_node_id")
    return len(ids)


def _last_resource_step_signature(histogram_plan: Any) -> str:
    groups = list(getattr(histogram_plan, "groups", []) or [])
    if not groups:
        return ""
    H_state = getattr(histogram_plan, "H_before", None)
    if not isinstance(H_state, torch.Tensor):
        return ""
    H_state = torch.clamp(H_state.detach().clone(), min=0).long()
    for group in groups[:-1]:
        for su_type, delta in dict(getattr(group, "combined_delta", {}) or {}).items():
            H_state[int(su_type)] += int(delta)
    final_group = groups[-1]
    operations = [getattr(final_group, "primary_op", None)] + list(
        getattr(final_group, "h_compensation_ops", []) or []
    )
    return _stateful_conversion_bundle_signature(
        H_state,
        [op for op in operations if op is not None],
    )


def _register_rejected_resource_candidate(
    plan: Any,
    histogram_plan: Any,
    rejected_signatures: set[str],
) -> str:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    if not bool(diagnostics.get("candidate_alternatives_enabled", False)):
        return ""
    signature = ""
    if str(getattr(plan, "plan_kind", "") or "") == "fixed_flex_capacity_expansion":
        H_before = getattr(histogram_plan, "H_before", None)
        conversions = [
            HistogramConversion(
                from_su=int(getattr(op, "from_su")),
                to_su=int(getattr(op, "to_su")),
                count=1,
                node_ids=(
                    [int(getattr(op, "node_id"))]
                    if getattr(op, "node_id", None) is not None
                    else []
                ),
                via_su=(
                    int(getattr(op, "via_su"))
                    if getattr(op, "via_su", None) is not None
                    else None
                ),
            )
            for op in list(getattr(plan, "planned_conversions", []) or [])
            if getattr(op, "from_su", None) is not None
            and getattr(op, "to_su", None) is not None
        ]
        if isinstance(H_before, torch.Tensor) and conversions:
            signature = _stateful_conversion_bundle_signature(
                H_before, conversions
            )
    if not signature:
        signature = _last_resource_step_signature(histogram_plan)
    if not signature or signature in rejected_signatures:
        return ""
    rejected_signatures.add(signature)
    return signature


def _rollback_stage3_plan_transactions(plan: Any, reason: str) -> List[int]:
    phase_result = getattr(plan, "phase_result", None)
    rolled_back_contract_ids: List[int] = []
    for transaction in list(getattr(phase_result, "transactions", []) or []):
        is_current_candidate = (
            str(getattr(transaction, "transaction_origin", "") or "")
            == "current_plan_candidate"
        )
        has_rebuild_effect = bool(
            list(getattr(transaction, "converted_node_ids", []) or [])
            or list(getattr(transaction, "conversion_path", []) or [])
            or list(getattr(transaction, "compensation_node_ids", []) or [])
        )
        if not has_rebuild_effect and not is_current_candidate:
            continue
        attempted = {
            "converted_node_ids": list(getattr(transaction, "converted_node_ids", []) or []),
            "conversion_path": list(getattr(transaction, "conversion_path", []) or []),
            "compensation_node_ids": list(getattr(transaction, "compensation_node_ids", []) or []),
            "claimed_resource_ids": dict(getattr(transaction, "claimed_resource_ids", {}) or {}),
            "product_node_ids": list(getattr(transaction, "product_node_ids", []) or []),
            "slot_assignments": list(getattr(transaction, "slot_assignments", []) or []),
        }
        transaction.committed = False
        transaction.rollback_reason = str(reason)
        transaction.infeasibility_proof = {
            **dict(getattr(transaction, "infeasibility_proof", {}) or {}),
            "reason": str(reason),
            "rolled_back": True,
            "attempted_transaction": attempted,
        }
        transaction.converted_node_ids = []
        transaction.conversion_path = []
        transaction.compensation_node_ids = []
        transaction.claimed_resource_ids = {}
        transaction.product_node_ids = []
        transaction.slot_assignments = []
        rolled_back_contract_ids.append(int(getattr(transaction, "contract_id", -1)))
    return rolled_back_contract_ids


def _restore_stage3_rotation_cursor(adjuster: Any, cursor_before: Optional[int]) -> None:
    if cursor_before is not None and hasattr(adjuster, "_stage3_su11_rotation_index"):
        adjuster._stage3_su11_rotation_index = int(cursor_before)


def _restore_stage3_parity_cursor(adjuster: Any, cursor_before: int) -> None:
    if hasattr(adjuster, "_stage3_parity_balance_cursor"):
        adjuster._stage3_parity_balance_cursor = int(cursor_before)


def _reset_stage3_generation_cursors(adjuster: Any) -> None:
    """Reset cursors that belong exclusively to one Layer1/2 generation."""
    for name in (
        "_stage3_su11_rotation_index",
        "_stage3_parity_balance_cursor",
        "_stage3_su24_nonfused_cursor",
    ):
        if hasattr(adjuster, name):
            setattr(adjuster, name, 0)


def _stage3_port_capacity_diagnostics(
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    histogram = torch.clamp(H.detach().cpu(), min=0).long()
    special_counts = {
        int(su_type): int(histogram[int(su_type)].item())
        for su_type in SPECIAL_DEGREE_PRIORS
        if int(su_type) < int(histogram.numel())
    }
    normalized_special = normalize_special_degree_meta(
        special_counts,
        special_degree_meta=special_degree_meta,
    )

    def count(su_type: int) -> int:
        return (
            int(histogram[int(su_type)].item())
            if int(su_type) < int(histogram.numel())
            else 0
        )

    direct_external = {
        int(su_type): count(int(su_type))
        for su_type in (15, 17, 22, 23, 24, 25)
    }
    special_external = {
        int(su_type): sum(
            int(value)
            for degree, value in dict(normalized_special.get(int(su_type), {}) or {}).items()
            if int(degree) in (2, 3)
        )
        for su_type in (19, 20, 21)
    }
    su11_count = count(11)
    aromatic_unit_count = sum(count(su_type) for su_type in range(5, 14))
    role_audit = audit_layer1_port_roles(histogram, special_degree_meta)
    total_degree = int(role_audit["total_ports"])
    return {
        "total_port_degree": int(total_degree),
        "degree_is_even": bool(int(total_degree) % 2 == 0),
        "su11_external_ports_required": int(su11_count),
        "eligible_external_counts": {
            **{str(key): int(value) for key, value in direct_external.items()},
            **{str(key): int(value) for key, value in special_external.items()},
        },
        "eligible_external_node_capacity": int(
            sum(direct_external.values()) + sum(special_external.values())
        ),
        "su11_aromatic_ports_required": int(2 * su11_count),
        "coarse_aromatic_unit_count": int(aromatic_unit_count),
        "aromatic_internal_ports": int(role_audit["aromatic_internal_ports"]),
        "non_aromatic_internal_ports": int(
            role_audit["non_aromatic_internal_ports"]
        ),
        "aromatic_internal_even": bool(role_audit["aromatic_internal_even"]),
        "non_aromatic_internal_even": bool(
            role_audit["non_aromatic_internal_even"]
        ),
        "port_roles_ok": bool(role_audit["ok"]),
        "layer1_port_role_audit": dict(role_audit),
        "scope": "necessary_coarse_preflight_not_exact_template_proof",
    }


def _ids_by_su(values: Any) -> Dict[int, List[int]]:
    return {
        int(key): sorted({int(node_id) for node_id in list(node_ids or [])})
        for key, node_ids in dict(values or {}).items()
        if str(key).lstrip("-").isdigit()
    }


def _ledger_value(ledger: Any, name: str) -> Any:
    if isinstance(ledger, dict):
        return dict(ledger).get(str(name))
    return getattr(ledger, str(name), None)


def _stage3_balance_resource_view(
    plan: Any,
    histogram_plan: Any,
) -> Tuple[Dict[int, List[int]], Dict[int, str], str]:
    accounting = dict(getattr(histogram_plan, "accounting_diagnostics", {}) or {})
    final_remaining = _ids_by_su(accounting.get("final_remaining_ids_by_su", {}))
    if "final_remaining_ids_by_su" in accounting:
        return final_remaining, {}, "resource_accounting_final_remaining"

    runtime = dict(getattr(plan, "runtime", {}) or {})
    ledger = runtime.get("iteration_topology_ledger")
    remaining = _ids_by_su(_ledger_value(ledger, "available_ids_by_su") or {})
    ownership: Dict[int, str] = {}
    for node_id, raw_owner in dict(
        _ledger_value(ledger, "ownership_by_node_id") or {}
    ).items():
        owner = dict(raw_owner or {}) if isinstance(raw_owner, dict) else raw_owner
        state = (
            str(dict(owner).get("state", "") or "")
            if isinstance(owner, dict)
            else str(getattr(owner, "state", "") or "")
        )
        ownership[int(node_id)] = state
    if remaining:
        return remaining, ownership, "iteration_topology_ledger"

    assembly = dict(runtime.get("assembly_before_adjustment", {}) or {})
    resource_ledger = dict(assembly.get("resource_ledger", {}) or {})
    for field in ("remaining_ids_by_su", "su11_donor_ids_by_su"):
        for su_type, node_ids in _ids_by_su(resource_ledger.get(field, {})).items():
            remaining.setdefault(int(su_type), []).extend(int(value) for value in node_ids)
    remaining = {
        int(key): sorted(set(values)) for key, values in remaining.items()
    }
    return remaining, {}, "assembly_resource_ledger" if remaining else "unavailable"


def _stage3_used_conversion_ids(histogram_plan: Any) -> set[int]:
    used = {
        int(node_id)
        for group in list(getattr(histogram_plan, "groups", []) or [])
        for op in [getattr(group, "primary_op", None)]
        + list(getattr(group, "h_compensation_ops", []) or [])
        if op is not None
        for node_id in list(getattr(op, "node_ids", []) or [])
    }
    for field in ("parity_balance_op", "port_role_balance_op"):
        balance_op = getattr(histogram_plan, field, None)
        if balance_op is not None:
            used.update(
                int(node_id)
                for node_id in list(getattr(balance_op, "node_ids", []) or [])
            )
    return used


def _append_stage3_parity_balance(
    *,
    plan: Any,
    histogram_plan: Any,
    nodes: List[Any],
    special_degree_meta: Optional[Dict[str, Any]],
    balance_cursor: int,
    E_target: Optional[torch.Tensor] = None,
    h_tolerance: float = DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    force: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Apply total-port and Layer1 port-role balance before a fresh rebuild."""
    H_after = getattr(histogram_plan, "H_after", None)
    groups = list(getattr(histogram_plan, "groups", []) or [])
    if not isinstance(H_after, torch.Tensor) or (not groups and not force):
        return True, {
            "enforced": False,
            "reason": "no_physical_histogram_batch",
        }

    audit_before = _stage3_port_capacity_diagnostics(H_after, special_degree_meta)
    if bool(audit_before["port_roles_ok"]):
        return True, {
            "enforced": True,
            "reason": "pre_rebuild_ports_balanced",
            "balance_applied": False,
            "before": audit_before,
            "after": audit_before,
        }

    remaining, ownership, pool_source = _stage3_balance_resource_view(
        plan,
        histogram_plan,
    )
    node_types = {
        int(getattr(node, "global_id", -1)): int(getattr(node, "su_type", -1))
        for node in list(nodes or [])
    }
    # SU12/SU13 are legal balance donors even when the external allocator has
    # consumed every closure resource and therefore reports an explicitly empty
    # remaining ledger. Add them before the empty-ledger failure check.
    for node_id, su_type in node_types.items():
        if int(su_type) in {12, 13}:
            remaining.setdefault(int(su_type), []).append(int(node_id))
    remaining = {
        int(key): sorted(set(values)) for key, values in remaining.items()
    }
    if not remaining:
        accounting = dict(
            getattr(histogram_plan, "accounting_diagnostics", {}) or {}
        )
        runtime = dict(getattr(plan, "runtime", {}) or {})
        ledger = runtime.get("iteration_topology_ledger")
        snapshot_declared = bool(
            "final_remaining_ids_by_su" in accounting
            or _ledger_value(ledger, "available_ids_by_su") is not None
            or dict(runtime.get("assembly_before_adjustment", {}) or {}).get(
                "resource_ledger"
            )
        )
        if snapshot_declared:
            missing_reason = (
                "odd_port_degree_no_unused_safe_balance_donor"
                if not bool(audit_before["degree_is_even"])
                else "port_role_balance_no_unused_safe_donor"
            )
            return False, {
                "enforced": True,
                "reason": missing_reason,
                "balance_applied": False,
                "before": audit_before,
                "pool_source": pool_source,
            }
        remaining = {}
        for node in list(nodes or []):
            node_id = int(getattr(node, "global_id", -1))
            su_type = int(getattr(node, "su_type", -1))
            if node_id >= 0 and su_type >= 0:
                remaining.setdefault(int(su_type), []).append(int(node_id))
        remaining = {
            int(key): sorted(set(values)) for key, values in remaining.items()
        }
        pool_source = "fresh_rebuild_node_pool"
        if not remaining:
            return False, {
                "enforced": True,
                "reason": "pre_rebuild_balance_node_identity_unavailable",
                "balance_applied": False,
                "before": audit_before,
                "pool_source": pool_source,
            }

    accounting_before = dict(
        getattr(histogram_plan, "accounting_diagnostics", {}) or {}
    )
    authorized_before = _ids_by_su(
        accounting_before.get("authorized_ids_before_by_su", {})
    )
    aggregate_before = _ids_by_su(
        accounting_before.get("aggregate_ids_before_by_su", {})
    )
    # Internal isolated-SU24 reclassification is a legal conversion of a
    # residual node that is deliberately outside the external aggregate
    # resource partition.  Its source is added below as a concrete donor;
    # applying the aggregate partition audit again after port balancing would
    # count that same node twice and reject an otherwise valid internal step.
    # The internal ledger invariant and concrete-node application still audit
    # this transaction, so only the duplicate aggregate check is suppressed.
    internal_su24_reclassification = bool(
        dict(accounting_before).get(
            "mode", ""
        ) == "internal_isolated_su24_reclassification"
        or dict(accounting_before).get(
            "allow_unreleased_su24_conversion", False
        )
        or dict(getattr(plan, "diagnostics", {}) or {}).get(
            "internal_only_su24_conversion", False
        )
        or any(
            int(getattr(op, "from_su", -1)) == SU24_RING_RESOURCE_TYPE
            and int(getattr(op, "to_su", -1)) in {12, 13}
            for group in list(getattr(histogram_plan, "groups", []) or [])
            for op in [getattr(group, "primary_op", None)]
            + list(getattr(group, "h_compensation_ops", []) or [])
            if op is not None
        )
    )
    strict_resource_accounting = bool(
        authorized_before and aggregate_before and not internal_su24_reclassification
    )
    for node_id, su_type in node_types.items():
        if int(su_type) not in {12, 13}:
            continue
        authorized_before.setdefault(int(su_type), []).append(int(node_id))
        aggregate_before.setdefault(int(su_type), []).append(int(node_id))

    # Internal isolated-SU24 reclassification is an authorized physical
    # conversion even though the pre-rebuild accounting view may only expose
    # the post-conversion aromatic bucket.  Add each concrete conversion
    # source to the before-inventory so strict accounting can reconcile the
    # source removal and target insertion atomically.
    accounting_allows_unreleased_su24 = bool(
        dict(accounting_before).get("allow_unreleased_su24_conversion", False)
    )
    if accounting_allows_unreleased_su24:
        for group in groups:
            for op in [getattr(group, "primary_op", None)] + list(
                getattr(group, "h_compensation_ops", []) or []
            ):
                if op is None:
                    continue
                source = int(getattr(op, "from_su", -1))
                if source != SU24_RING_RESOURCE_TYPE:
                    continue
                for node_id in list(getattr(op, "node_ids", []) or []):
                    authorized_before.setdefault(source, []).append(int(node_id))
                    aggregate_before.setdefault(source, []).append(int(node_id))
    authorized_before = {
        int(key): sorted(set(values)) for key, values in authorized_before.items()
    }
    aggregate_before = {
        int(key): sorted(set(values)) for key, values in aggregate_before.items()
    }
    used_ids = _stage3_used_conversion_ids(histogram_plan)
    for group in groups:
        for op in [getattr(group, "primary_op", None)] + list(
            getattr(group, "h_compensation_ops", []) or []
        ):
            if op is None:
                continue
            for node_id in list(getattr(op, "node_ids", []) or []):
                node_types[int(node_id)] = int(getattr(op, "to_su", -1))
    # A classification-authorized accounting view may omit aromatic donors;
    # they were added to ``remaining`` before the empty-ledger check above.
    for su_type in (12, 13):
        remaining[int(su_type)] = [
            int(node_id)
            for node_id in list(remaining.get(int(su_type), []) or [])
            if int(node_id) not in used_ids
        ]
    remaining = {
        int(key): sorted(set(values)) for key, values in remaining.items()
    }
    allowed_states = {
        "remaining_closure_resource",
        "remaining_su24_ring_resource",
        "remaining_aromatic_donor",
    }
    working_histogram = deepcopy(histogram_plan)
    working_groups = list(getattr(working_histogram, "groups", []) or [])
    H_balanced = H_after.detach().clone()
    attempted_directions: List[Dict[str, Any]] = []
    balance_ops: List[HistogramConversion] = []
    phase_eval = dict(getattr(plan, "runtime", {}) or {}).get(
        "phase_evaluation"
    )
    flex_audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
    fixed_flex_expansion = bool(
        str(getattr(plan, "plan_kind", "") or "")
        == "fixed_flex_capacity_expansion"
        and flex_audit
    )
    fixed_flex_selection = dict(
        dict(getattr(plan, "diagnostics", {}) or {}).get(
            "fixed_flex_capacity_expansion", {}
        )
        or {}
    )
    preferred_balance_ids = {
        "total_port_parity": {
            int(value)
            for value in list(
                fixed_flex_selection.get(
                    "predicted_parity_balance_node_ids", []
                )
                or []
            )
        },
        "layer1_port_roles": {
            int(value)
            for value in list(
                fixed_flex_selection.get(
                    "predicted_role_balance_node_ids", []
                )
                or []
            )
        },
    }

    def _fixed_flex_progress(candidate_H: torch.Tensor) -> Dict[str, Any]:
        if not fixed_flex_expansion:
            return {"enforced": False, "ok": True}
        H_before = getattr(histogram_plan, "H_before", None)
        if not isinstance(H_before, torch.Tensor):
            return {
                "enforced": True,
                "ok": False,
                "reason": "fixed_flex_balance_missing_baseline_histogram",
            }
        impact = evaluate_flex_impact(H_before, candidate_H, flex_audit)
        before = dict(impact.before or {})
        after = dict(impact.after or {})
        fixed_count = int(flex_audit.get("explicit_flex_count", 0) or 0)
        before_excess = max(
            0, fixed_count - int(before.get("flex_upper", 0) or 0)
        )
        after_excess = max(
            0, fixed_count - int(after.get("flex_upper", 0) or 0)
        )
        ok = bool(
            str(after.get("scenario", "") or "") != "invalid"
            and int(after.get("flex_lower", 0) or 0) <= fixed_count
            and int(after.get("flex_upper", 0) or 0)
            > int(before.get("flex_upper", 0) or 0)
            and int(after_excess) < int(before_excess)
        )
        return {
            "enforced": True,
            "ok": bool(ok),
            "reason": (
                "fixed_flex_progress_preserved"
                if ok
                else "port_balance_would_cancel_fixed_flex_progress"
            ),
            "before": before,
            "after": after,
            "excess_before": int(before_excess),
            "excess_after": int(after_excess),
        }

    def _direction_preserves_fixed_flex(source: int, target: int) -> bool:
        candidate = H_balanced.detach().clone()
        if int(candidate[int(source)].item()) <= 0:
            return False
        candidate[int(source)] -= 1
        candidate[int(target)] += 1
        return bool(_fixed_flex_progress(candidate).get("ok", False))

    def _select_unused(
        directions: Tuple[Tuple[int, int], ...],
        kind: str,
    ):
        for source, target in directions:
            if (
                int(source) in {12, 13}
                and int(H_balanced[int(source)].item())
                <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
            ):
                attempted_directions.append({
                    "kind": str(kind),
                    "from_su": int(source),
                    "to_su": int(target),
                    "candidate_node_ids": [],
                    "reason": "aromatic_minimum_preserved",
                })
                continue
            candidates: List[int] = []
            for node_id in list(remaining.get(int(source), []) or []):
                if int(node_id) in used_ids:
                    continue
                if int(node_types.get(int(node_id), -1)) != int(source):
                    continue
                aromatic_donor = int(source) in {12, 13}
                if not aromatic_donor:
                    owner_state = str(ownership.get(int(node_id), "") or "")
                    if owner_state and owner_state not in allowed_states:
                        continue
                    if strict_resource_accounting and int(node_id) not in set(
                        authorized_before.get(int(source), [])
                    ):
                        continue
                    if strict_resource_accounting and int(node_id) not in set(
                        aggregate_before.get(int(source), [])
                    ):
                        continue
                candidates.append(int(node_id))
            attempted_directions.append({
                "kind": str(kind),
                "from_su": int(source),
                "to_su": int(target),
                "candidate_node_ids": sorted(candidates),
            })
            if candidates:
                preferred = sorted(
                    int(value)
                    for value in candidates
                    if int(value) in preferred_balance_ids.get(str(kind), set())
                )
                return int(source), int(target), (
                    int(preferred[0]) if preferred else min(candidates)
                )
        return None

    def _append_balance(
        selected: Tuple[int, int, int],
        *,
        field: str,
        reason: str,
    ) -> None:
        source, target, node_id = selected
        op = HistogramConversion(
            from_su=int(source),
            to_su=int(target),
            count=1,
            reason=str(reason),
            node_ids=[int(node_id)],
        )
        setattr(working_histogram, field, op)
        H_balanced[int(source)] -= 1
        H_balanced[int(target)] += 1
        remaining[int(source)] = [
            int(value)
            for value in list(remaining.get(int(source), []) or [])
            if int(value) != int(node_id)
        ]
        remaining.setdefault(int(target), []).append(int(node_id))
        remaining[int(target)] = sorted(set(remaining[int(target)]))
        node_types[int(node_id)] = int(target)
        used_ids.add(int(node_id))
        balance_ops.append(op)

    # Either 13->12 or 12->13 flips the total degree parity. The cursor only
    # defines which direction is preferred; it must not make the other valid
    # direction unavailable when the preferred donor pool is exhausted.
    if fixed_flex_expansion:
        total_preferred = ((12, 13), (13, 12))
    elif int(balance_cursor) % 2 == 0:
        total_preferred = ((13, 12), (12, 13))
    else:
        total_preferred = ((12, 13), (13, 12))
    if not bool(audit_before["degree_is_even"]):
        total_directions = tuple(
            direction
            for direction in total_preferred
            if not fixed_flex_expansion
            or _direction_preserves_fixed_flex(*direction)
        )
        selected = _select_unused(total_directions, "total_port_parity")
        if selected is None:
            return False, {
                "enforced": True,
                "reason": (
                    "port_balance_would_cancel_fixed_flex_progress"
                    if fixed_flex_expansion and not total_directions
                    else "odd_port_degree_no_unused_safe_balance_donor"
                ),
                "balance_applied": False,
                "before": audit_before,
                "pool_source": pool_source,
                "required_direction": list(total_preferred[0]),
                "attempted_directions": attempted_directions,
            }
        _append_balance(
            selected,
            field="parity_balance_op",
            reason="pre_rebuild_port_parity_balance",
        )

    role_audit = audit_layer1_port_roles(H_balanced, special_degree_meta)
    role_selection = select_layer1_port_balance(H_balanced, role_audit)
    if bool(role_selection.get("required", False)):
        if not bool(role_selection.get("possible", False)):
            return False, {
                "enforced": True,
                "reason": str(role_selection.get(
                    "reason", "layer1_port_role_balance_impossible"
                )),
                "balance_applied": bool(balance_ops),
                "before": audit_before,
                "intermediate": _stage3_port_capacity_diagnostics(
                    H_balanced, special_degree_meta
                ),
                "attempted_directions": attempted_directions,
            }
        role_preferred = (
            ((11, 12), (12, 11))
            if int(balance_cursor) % 2 == 0
            else ((12, 11), (11, 12))
        )
        role_directions = tuple(
            (int(source), int(target))
            for source, target in role_preferred
            if int(H_balanced[int(source)].item()) > 0
            and (
                not fixed_flex_expansion
                or _direction_preserves_fixed_flex(int(source), int(target))
            )
        )
        selected = _select_unused(role_directions, "layer1_port_roles")
        if selected is None:
            return False, {
                "enforced": True,
                "reason": "port_role_balance_no_unused_safe_donor",
                "balance_applied": bool(balance_ops),
                "before": audit_before,
                "intermediate": _stage3_port_capacity_diagnostics(
                    H_balanced, special_degree_meta
                ),
                "pool_source": pool_source,
                "attempted_directions": attempted_directions,
            }
        _append_balance(
            selected,
            field="port_role_balance_op",
            reason="pre_rebuild_layer1_port_role_balance",
        )

    if not balance_ops:
        return True, {
            "enforced": True,
            "reason": "pre_rebuild_ports_balanced",
            "balance_applied": False,
            "before": audit_before,
            "after": audit_before,
        }

    working_histogram.H_after = H_balanced
    working_histogram.physical_conversion_count = int(
        getattr(working_histogram, "physical_conversion_count", 0) or 0
    ) + len(balance_ops)

    updated_accounting: Dict[str, Any] = {}
    if strict_resource_accounting:
        final_remaining = {
            int(key): sorted(set(int(value) for value in values))
            for key, values in remaining.items()
        }
        updated_accounting = build_resource_accounting_snapshot(
            aggregate_ids_before_by_su=aggregate_before,
            authorized_ids_before_by_su=authorized_before,
            final_remaining_ids_by_su=final_remaining,
            committed_claimed_ids_by_su=_ids_by_su(
                accounting_before.get("committed_claimed_ids_by_su", {})
            ),
            groups=working_groups,
            extra_ops=balance_ops,
        )
        if not bool(updated_accounting.get("ok", False)):
            return False, {
                "enforced": True,
                "reason": "parity_balance_resource_accounting_failed",
                "balance_applied": False,
                "before": audit_before,
                "resource_accounting": updated_accounting,
            }
        working_histogram.inventory_before = {
            int(key): int(value)
            for key, value in dict(
                updated_accounting.get("authorized_inventory_before", {}) or {}
            ).items()
        }
        working_histogram.inventory_after = {
            int(key): int(value)
            for key, value in dict(
                updated_accounting.get("authorized_inventory_after", {}) or {}
            ).items()
        }
        working_histogram.aggregate_inventory_before = {
            int(key): int(value)
            for key, value in dict(
                updated_accounting.get("aggregate_inventory_before", {}) or {}
            ).items()
        }
        working_histogram.aggregate_inventory_after = {
            int(key): int(value)
            for key, value in dict(
                updated_accounting.get("aggregate_inventory_after", {}) or {}
            ).items()
        }
        working_histogram.aggregate_required_counts = {
            int(key): int(value)
            for key, value in dict(
                updated_accounting.get("aggregate_required_counts", {}) or {}
            ).items()
        }
        # The accounting snapshot is rebuilt here, but the release registry
        # is durable plan metadata.  Preserve it so an internal follow-up can
        # still authorize the concrete SU24 IDs released from a pure ring.
        release_metadata = {
            str(key): accounting_before[key]
            for key in (
                "released_su24_ids",
                "released_resource_ids_by_su",
            )
            if key in accounting_before
        }
        working_histogram.accounting_diagnostics = {
            **updated_accounting,
            **release_metadata,
        }
    else:
        inventory_after, inventory_errors = apply_count_delta(
            getattr(working_histogram, "inventory_before", {}) or {},
            combined_group_delta(working_groups, extra_ops=balance_ops),
        )
        if inventory_errors:
            return False, {
                "enforced": True,
                "reason": "parity_balance_inventory_update_failed",
                "balance_applied": False,
                "before": audit_before,
                "inventory_errors": inventory_errors,
            }
        working_histogram.inventory_after = inventory_after

    if internal_su24_reclassification:
        # ``_finalize_phase_plan`` may have copied aggregate external
        # partition fields onto this histogram.  They describe the completed
        # external products, not the isolated SU24 residual conversion, and
        # would make the generic apply preflight reject the same concrete node
        # a second time.  Keep the authorized inventory and node-level ops;
        # remove only the external aggregate view for this internal plan.
        working_histogram.aggregate_inventory_before = {}
        working_histogram.aggregate_inventory_after = {}
        working_histogram.aggregate_required_counts = {}

    objective_audit = _fixed_flex_progress(H_balanced)
    if not bool(objective_audit.get("ok", False)):
        return False, {
            "enforced": True,
            "reason": str(objective_audit.get(
                "reason", "port_balance_would_cancel_fixed_flex_progress"
            )),
            "balance_applied": bool(balance_ops),
            "before": audit_before,
            "after": _stage3_port_capacity_diagnostics(
                H_balanced, special_degree_meta
            ),
            "fixed_flex_objective": objective_audit,
            "attempted_directions": attempted_directions,
        }

    target_h: Optional[float]
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    current_h = float(_h_total(H_balanced, special_degree_meta))
    h_lower, h_upper = h_soft_bounds(target_h, h_tolerance)

    histogram_plan.__dict__.clear()
    histogram_plan.__dict__.update(deepcopy(working_histogram.__dict__))

    phase_result = getattr(plan, "phase_result", None)
    if phase_result is not None:
        phase_result.parity_balance_count = int(
            getattr(phase_result, "parity_balance_count", 0) or 0
        ) + len(balance_ops)

    audit_after = _stage3_port_capacity_diagnostics(H_balanced, special_degree_meta)
    diagnostics = {
        "enforced": True,
        "reason": "pre_rebuild_ports_balanced",
        "balance_applied": True,
        "before": audit_before,
        "after": audit_after,
        "pool_source": pool_source,
        "applied_operations": [op.to_dict() for op in balance_ops],
        "attempted_directions": attempted_directions,
        "fixed_flex_objective": objective_audit,
        "h_after": float(current_h),
        "h_soft_bounds": (
            [float(h_lower), float(h_upper)]
            if h_lower is not None and h_upper is not None
            else []
        ),
        "next_cursor": int(balance_cursor) + (1 if balance_ops else 0),
    }
    parity_op = getattr(working_histogram, "parity_balance_op", None)
    if parity_op is not None:
        diagnostics["required_direction"] = list(total_preferred[0])
        diagnostics["applied_direction"] = [
            int(parity_op.from_su), int(parity_op.to_su)
        ]
        diagnostics["node_id"] = int(parity_op.node_ids[0])
    plan_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    plan_diagnostics["pre_rebuild_port_feasibility"] = diagnostics
    plan.diagnostics = plan_diagnostics
    return bool(audit_after["port_roles_ok"]), diagnostics


def _apply_deferred_port_balance(
    *,
    plan: Any,
    nodes: List[Any],
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    balance_cursor: int,
    E_target: Optional[torch.Tensor],
    h_tolerance: float,
) -> Tuple[List[Any], torch.Tensor, Dict[str, Any]]:
    """Balance ports at the external generation boundary.

    Continuation plans intentionally omit balance operations.  At completion
    or budget exhaustion, run the same audited selector against the current
    graph and apply only the resulting port operations before rebuilding.
    """
    boundary_inventory = {
        int(su_type): int(H[int(su_type)].item())
        for su_type in range(int(H.numel()))
        if int(H[int(su_type)].item()) > 0
    }
    balance_histogram = HistogramAdjustmentPlan(
        phase="external_topology",
        groups=[],
        primary_adjustment_count=0,
        H_before=H.detach().clone(),
        H_after=H.detach().clone(),
        inventory_before=dict(boundary_inventory),
        inventory_after=dict(boundary_inventory),
    )
    balance_plan = deepcopy(plan)
    ok, diagnostics = _append_stage3_parity_balance(
        plan=balance_plan,
        histogram_plan=balance_histogram,
        nodes=list(nodes or []),
        special_degree_meta=special_degree_meta,
        balance_cursor=int(balance_cursor),
        E_target=E_target,
        h_tolerance=h_tolerance,
        force=True,
    )
    if not ok:
        return list(nodes or []), H.detach().clone(), dict(diagnostics or {})
    H_out = torch.clamp(
        getattr(balance_histogram, "H_after", H).detach().clone(), min=0
    ).long()
    nodes_out = deepcopy(list(nodes or []))
    nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in nodes_out
        if int(getattr(node, "global_id", -1)) >= 0
    }
    for field in ("parity_balance_op", "port_role_balance_op"):
        op = getattr(balance_histogram, field, None)
        if op is None:
            continue
        for node_id in list(getattr(op, "node_ids", []) or []):
            node = nodes_by_id.get(int(node_id))
            if node is None or int(getattr(node, "su_type", -1)) != int(op.from_su):
                return list(nodes or []), H.detach().clone(), {
                    **dict(diagnostics or {}),
                    "reason": "pre_rebuild_balance_node_identity_unavailable",
                    "balance_applied": False,
                }
            setattr(node, "su_type", int(op.to_su))
    diagnostics = dict(diagnostics or {})
    diagnostics["balance_applied"] = bool(
        getattr(balance_histogram, "parity_balance_op", None) is not None
        or getattr(balance_histogram, "port_role_balance_op", None) is not None
    )
    diagnostics["operations"] = [
        op.to_dict()
        for op in (
            getattr(balance_histogram, "parity_balance_op", None),
            getattr(balance_histogram, "port_role_balance_op", None),
        )
        if op is not None
    ]
    return nodes_out, H_out, diagnostics


def _prepare_spectrum_candidate_for_rebuild(
    *,
    plan: Any,
    candidate: Any,
    nodes: List[Any],
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
    balance_cursor: int,
) -> Tuple[Any, Optional[List[Any]], Optional[torch.Tensor], Dict[str, Any]]:
    """Apply one spectrum bundle and make its fixed-port parity rebuild-safe."""
    effective_candidate = deepcopy(candidate)
    candidate_diagnostics = dict(
        getattr(effective_candidate, "diagnostics", {}) or {}
    )
    candidate_diagnostics.setdefault(
        "primary_inverse_signature",
        str(getattr(effective_candidate, "inverse_signature", "") or ""),
    )
    effective_candidate.diagnostics = candidate_diagnostics
    candidate_nodes, candidate_H, apply_proof = apply_spectrum_candidate(
        nodes=nodes,
        H=H,
        candidate=effective_candidate,
    )
    if candidate_nodes is None or candidate_H is None:
        return effective_candidate, None, None, {
            "ok": False,
            "reason": str(dict(apply_proof or {}).get(
                "reason", "spectrum_candidate_apply_failed"
            )),
            "apply_proof": dict(apply_proof or {}),
        }

    audit_before = _stage3_port_capacity_diagnostics(
        candidate_H,
        special_degree_meta,
    )
    if bool(audit_before["port_roles_ok"]):
        candidate_diagnostics = dict(
            getattr(effective_candidate, "diagnostics", {}) or {}
        )
        candidate_diagnostics["final_balance_operations"] = []
        candidate_diagnostics["total_operation_count"] = len(
            list(getattr(effective_candidate, "operations", []) or [])
        )
        effective_candidate.diagnostics = candidate_diagnostics
        return effective_candidate, candidate_nodes, candidate_H, {
            "ok": True,
            "reason": "spectrum_candidate_pre_rebuild_ports_valid",
            "balance_applied": False,
            "before": audit_before,
            "after": audit_before,
        }

    groups: List[AtomicAdjustmentGroup] = []
    for raw in list(getattr(effective_candidate, "operations", []) or []):
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        node_id = int(op.get("node_id", -1))
        groups.append(AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=int(source),
                to_su=int(target),
                count=1,
                reason=str(op.get("reason", "remaining_pool_difference_spectrum_rebalance")),
                node_ids=[int(node_id)],
            ),
            combined_delta={int(source): -1, int(target): 1},
        ))
    trial_histogram_plan = HistogramAdjustmentPlan(
        phase="remaining_pool_spectrum_rebalance",
        groups=groups,
        primary_adjustment_count=int(
            getattr(effective_candidate, "primary_adjustment_count", 0) or 0
        ),
        physical_conversion_count=len(groups),
        H_before=H.detach().clone(),
        H_after=candidate_H.detach().clone(),
        inventory_before={
            int(index): int(value)
            for index, value in enumerate(
                torch.clamp(H.detach().cpu(), min=0).long().tolist()
            )
            if int(value) != 0
        },
    )
    trial_plan = deepcopy(plan)
    parity_ok, parity_diagnostics = _append_stage3_parity_balance(
        plan=trial_plan,
        histogram_plan=trial_histogram_plan,
        nodes=nodes,
        special_degree_meta=special_degree_meta,
        balance_cursor=int(balance_cursor),
    )
    balance_fields = (
        ("parity_balance_op", "parity_balance"),
        ("port_role_balance_op", "port_role_balance"),
    )
    balance_ops = [
        (field, kind, getattr(trial_histogram_plan, field, None))
        for field, kind in balance_fields
        if getattr(trial_histogram_plan, field, None) is not None
    ]
    if not bool(parity_ok) or not balance_ops:
        return effective_candidate, None, None, {
            **dict(parity_diagnostics or {}),
            "ok": False,
            "reason": str(dict(parity_diagnostics or {}).get(
                "reason", "spectrum_candidate_pre_rebuild_port_infeasible"
            )),
        }

    balance_operations: List[Dict[str, Any]] = []
    for _field, kind, balance_op in balance_ops:
        balance_node_ids = [
            int(node_id)
            for node_id in list(getattr(balance_op, "node_ids", []) or [])
        ]
        if len(balance_node_ids) != 1:
            return effective_candidate, None, None, {
                **dict(parity_diagnostics or {}),
                "ok": False,
                "reason": f"spectrum_candidate_{kind}_id_invalid",
                "balance_node_ids": balance_node_ids,
            }
        balance_operations.append({
            "op": "convert",
            "node_id": int(balance_node_ids[0]),
            "from_su": int(getattr(balance_op, "from_su")),
            "to_su": int(getattr(balance_op, "to_su")),
            "reason": str(getattr(balance_op, "reason", kind)),
            "operation_kind": str(kind),
        })
    effective_candidate.operations = list(
        getattr(effective_candidate, "operations", []) or []
    ) + balance_operations
    effective_candidate.histogram_signature = histogram_signature_after_operations(
        H,
        effective_candidate.operations,
    )
    effective_candidate.inverse_signature = inverse_bundle_signature(
        effective_candidate.operations
    )
    balance_suffix = ":".join(
        f"{op['operation_kind']}:{int(op['from_su'])}-{int(op['to_su'])}"
        for op in balance_operations
    )
    effective_candidate.candidate_id = (
        f"{str(getattr(effective_candidate, 'candidate_id', '') or '')}:"
        f"{balance_suffix}"
    )
    candidate_diagnostics = dict(
        getattr(effective_candidate, "diagnostics", {}) or {}
    )
    candidate_diagnostics["parity_balance"] = dict(parity_diagnostics or {})
    candidate_diagnostics["final_balance_operations"] = [
        dict(operation) for operation in balance_operations
    ]
    candidate_diagnostics["total_operation_count"] = len(
        effective_candidate.operations
    )
    effective_candidate.diagnostics = candidate_diagnostics

    candidate_nodes, candidate_H, apply_proof = apply_spectrum_candidate(
        nodes=nodes,
        H=H,
        candidate=effective_candidate,
    )
    if candidate_nodes is None or candidate_H is None:
        return effective_candidate, None, None, {
            **dict(parity_diagnostics or {}),
            "ok": False,
            "reason": str(dict(apply_proof or {}).get(
                "reason", "spectrum_candidate_parity_apply_failed"
            )),
            "apply_proof": dict(apply_proof or {}),
        }
    audit_after = _stage3_port_capacity_diagnostics(
        candidate_H,
        special_degree_meta,
    )
    final_ok = bool(audit_after["port_roles_ok"])
    return effective_candidate, candidate_nodes, candidate_H, {
        **dict(parity_diagnostics or {}),
        "ok": bool(final_ok),
        "reason": (
            "spectrum_candidate_pre_rebuild_ports_balanced"
            if final_ok
            else "spectrum_candidate_pre_rebuild_port_infeasible"
        ),
        "after": audit_after,
        "balance_operations": balance_operations,
    }


def _stage3_live_assembler_summary(live_assembler: Any) -> Dict[str, Any]:
    """Serialize bounded continuation state without serializing the assembler."""
    if live_assembler is None:
        return {}
    allocator = getattr(live_assembler, "allocator", None)
    claimability = (
        dict(allocator.claimability_snapshot() or {})
        if allocator is not None
        and callable(getattr(allocator, "claimability_snapshot", None))
        else {}
    )
    pending_contracts = [
        dict(item or {})
        for item in list(getattr(live_assembler, "pending_contracts", []) or [])
        if dict(item or {})
    ]
    selected_groups = []
    for raw_group in list(
        getattr(live_assembler, "selected_flex_candidate_groups", []) or []
    ):
        group = dict(raw_group or {})
        selected_groups.append({
            key: deepcopy(group[key])
            for key in ("group_id", "strategy", "fragment_ids", "resource_template")
            if key in group
        })
    return {
        "generation_id": int(
            getattr(live_assembler, "generation_id", 0) or 0
        ),
        "current_phase": str(
            getattr(live_assembler, "current_phase", "") or ""
        ),
        "completed_phases": sorted({
            str(value)
            for value in list(
                getattr(live_assembler, "completed_phases", set()) or set()
            )
            if str(value)
        }),
        "fragment_state_by_id": {
            str(key): str(value)
            for key, value in sorted(
                dict(
                    getattr(live_assembler, "fragment_state_by_id", {}) or {}
                ).items(),
                key=lambda item: int(item[0]),
            )
        },
        "selected_flex_candidate_ids": sorted({
            int(value)
            for value in list(
                getattr(live_assembler, "selected_flex_candidate_ids", set())
                or set()
            )
        }),
        "selected_flex_candidate_groups": selected_groups,
        "allocator": {
            "raw_available_ids_by_su": deepcopy(
                claimability.get("raw_available_ids_by_su", {}) or {}
            ),
            "claimable_ids_by_su": deepcopy(
                claimability.get("claimable_ids_by_su", {}) or {}
            ),
            "protected_counts": deepcopy(
                claimability.get("protected_counts", {}) or {}
            ),
            "excluded_node_ids": [
                int(value)
                for value in list(
                    claimability.get("excluded_node_ids", []) or []
                )
            ],
        },
        "continuation_ready_contract_ids": sorted({
            int(contract.get("contract_id", -1))
            for contract in pending_contracts
            if bool(contract.get("continuation_ready", False))
        }),
        "ring_recovery_attempts": [
            deepcopy(dict(item or {}))
            for item in list(
                getattr(live_assembler, "ring_recovery_attempts", []) or []
            )[-8:]
            if dict(item or {})
        ],
    }


def _stage3_terminal_infeasibility_proof(
    plan: Any,
    *,
    default_reason: str = "current_stage3_plan_infeasible",
) -> Dict[str, Any]:
    histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
    phase_result = getattr(plan, "phase_result", None)
    prior_proof = _mapping_payload(
        getattr(phase_result, "infeasibility_proof", {}) or {},
        field_name="infeasibility_proof",
    )
    slot_proof = _mapping_payload(
        prior_proof.get("slot_assignment_proof", {}) or {},
        field_name="slot_assignment_proof",
    )
    failed_contract = _mapping_payload(
        prior_proof.get("failed_contract", {}) or {},
        field_name="failed_contract",
    )
    has_explicit_pending_requirement = bool(
        "required_counts" in prior_proof
        or "required_counts" in failed_contract
    )
    required_source = (
        prior_proof.get("required_counts", {})
        if "required_counts" in prior_proof
        else failed_contract.get("required_counts", {})
        if "required_counts" in failed_contract
        else getattr(plan, "hard_required_counts", {})
        or getattr(histogram_plan, "demand_before", {})
        or {}
    )
    required_counts = {
        str(key): int(value)
        for key, value in dict(required_source or {}).items()
        if int(value) > 0
    }
    available_counts = {
        str(key): int(value)
        for key, value in dict(
            prior_proof.get("available_counts", {})
            or getattr(histogram_plan, "inventory_before", {})
            or {}
        ).items()
    }
    runtime = dict(getattr(plan, "runtime", {}) or {})
    ledger = runtime.get("iteration_topology_ledger")
    if not available_counts and ledger is not None:
        pools = (
            getattr(ledger, "available_ids_by_su", None)
            if not isinstance(ledger, dict)
            else dict(ledger).get("available_ids_by_su", {})
        )
        available_counts = {
            str(key): len(list(values or []))
            for key, values in dict(pools or {}).items()
        }
    missing_source: Any = {}
    for source, key in (
        (prior_proof, "missing_counts"),
        (prior_proof, "unresolved_counts"),
        (dict(getattr(plan, "unresolved_counts", {}) or {}), "__self__"),
    ):
        if key == "__self__":
            missing_source = source
            break
        if key in source:
            missing_source = source.get(key) or {}
            break
    missing_counts = {
        str(key): int(value)
        for key, value in dict(missing_source or {}).items()
        if int(value) > 0
    }
    if not missing_counts and not (
        has_explicit_pending_requirement and not required_counts
    ):
        missing_counts = {
            str(key): max(0, int(required) - int(available_counts.get(str(key), 0)))
            for key, required in required_counts.items()
            if int(required) > int(available_counts.get(str(key), 0))
        }
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    live_assembler = runtime.get("external_assembler") or getattr(
        phase_eval,
        "external_assembler",
        None,
    )
    live_pending_contracts = [
        deepcopy(dict(item or {}))
        for item in list(
            getattr(live_assembler, "pending_contracts", []) or []
        )
        if dict(item or {})
    ]
    live_assembler_summary = _stage3_live_assembler_summary(live_assembler)
    pending_contracts = list(getattr(plan, "pending_contracts", []) or [])
    infeasible_contracts = list(diagnostics.get("infeasible_contracts", []) or [])
    selected_contracts = list(
        dict(diagnostics.get("resource_request_breakdown", {}) or {}).get(
            "selected_active_contracts", []
        )
        or []
    )
    current_contract = _mapping_payload(
        diagnostics.get("current_contract", {}) or {},
        field_name="current_contract",
    )
    if not failed_contract and current_contract:
        failed_contract = current_contract
    for candidates in (pending_contracts, infeasible_contracts, selected_contracts):
        if candidates:
            failed_contract = _mapping_payload(
                candidates[0],
                field_name="failed_contract_candidate",
            )
            break
    if not failed_contract:
        failed_contract = {
            "phase": str(getattr(plan, "phase", "") or "unknown"),
            "plan_kind": str(getattr(plan, "plan_kind", "") or "unknown"),
        }
    planned_conversions = list(prior_proof.get("attempted_conversions", []) or [])
    if not planned_conversions:
        planned_conversions = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item or {})
            for item in list(getattr(plan, "planned_conversions", []) or [])
        ]
    rejection_reasons = [
        value
        for value in (
            str(prior_proof.get("reason", "") or ""),
            str(slot_proof.get("reason", "") or ""),
            str(getattr(plan, "blocked_reason", "") or ""),
            str(getattr(histogram_plan, "blocked_reason", "") or ""),
        )
        if value
    ]
    recovery_proof = dict(
        diagnostics.get("ring_recovery_capacity_proof", {}) or {}
    )
    claim_trace = list(prior_proof.get("claim_trace", []) or [])
    if not claim_trace and live_pending_contracts:
        first_live = dict(live_pending_contracts[0] or {})
        claim_trace = list(
            first_live.get("claim_trace", [])
            or dict(first_live.get("diagnostics", {}) or {}).get(
                "claim_trace", []
            )
            or []
        )
    proof = {
        **prior_proof,
        "phase": str(getattr(plan, "phase", "") or "unknown"),
        "reason": str(
            prior_proof.get("reason")
            or slot_proof.get("reason")
            or getattr(plan, "blocked_reason", "")
            or getattr(histogram_plan, "blocked_reason", "")
            or default_reason
        ),
        "failed_contract": failed_contract,
        "required_counts": required_counts,
        "available_counts": available_counts,
        "missing_counts": missing_counts,
        "live_pending_contracts": live_pending_contracts,
        "live_assembler_summary": live_assembler_summary,
        "claim_trace": [dict(item or {}) for item in claim_trace],
        "recovery_proof": recovery_proof,
        "attempted_conversions": planned_conversions,
        "h_or_element_rejections": list(
            prior_proof.get("h_or_element_rejections", []) or rejection_reasons
        ),
        "template_checks": {
            **_mapping_payload(
                prior_proof.get("template_checks", {}) or {},
                field_name="template_checks",
            ),
            "next_plan_reason": str(getattr(plan, "reason", "") or ""),
            "next_plan_kind": str(getattr(plan, "plan_kind", "") or ""),
            "rebuild_required": bool(getattr(phase_result, "rebuild_required", False)),
            "spectrum_candidate_count": len(
                list(getattr(plan, "spectrum_candidates", []) or [])
            ),
        },
    }
    return proof


def _external_ring_recovery_authorized(plan: Any) -> bool:
    """Identify a fatty-ring/side resource blocker eligible for recovery.

    A failed side contract is not enough by itself.  The recovery gate is only
    valid when the ordinary 11/22/23 pool, plus the one-way 12/13 -> 11 donor
    path, is mathematically unable to cover the reported deficit.  This keeps
    H/scoring/protection rejections from dismantling an otherwise reusable
    pure-resource ring.
    """
    phase = str(getattr(plan, "phase", "") or "")
    if phase not in {
        "external_topology_assembly",
        "external_side_chain_closure",
        "external_fatty_ring_closure",
        "external_topology",
    }:
        return False
    if str(getattr(plan, "reason", "") or "") != "phase_resource_infeasible":
        return False
    candidates: List[Dict[str, Any]] = []
    candidates.extend(
        dict(item or {})
        for item in list(getattr(plan, "pending_contracts", []) or [])
        if dict(item or {})
    )
    proof = _stage3_terminal_infeasibility_proof(plan)
    failed = proof.get("failed_contract", {})
    if isinstance(failed, dict) and failed:
        candidates.append(dict(failed))
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    candidates.extend(
        dict(item or {})
        for item in list(diagnostics.get("infeasible_contracts", []) or [])
        if dict(item or {})
    )
    if not candidates:
        candidates.append({"target_role": ""})
    recovery_candidate = None
    for candidate in candidates:
        role = str(candidate.get("target_role", "") or "")
        candidate_proof = dict(candidate.get("proof", {}) or {})
        reason = str(
            candidate_proof.get("reason", "")
            or candidate.get("reason", "")
            or proof.get("reason", "")
        )
        failure_domain = str(
            candidate_proof.get("failure_domain", "")
            or candidate.get("failure_domain", "")
            or proof.get("failure_domain", "")
        )
        capacity_proof = dict(
            candidate_proof.get("conversion_capacity_proof", {}) or {}
        )
        if role in {"side_chain", "fatty_ring"}:
            if failure_domain in {
                "topology_materialization",
                "h_constraint",
                "h_compensation",
                "protected_resource",
                "future_topology_reservation",
                "conversion_scoring",
            } or reason in {
                "topology_materialization",
                "h_constraint",
                "h_compensation",
                "protected_resource",
                "future_topology_reservation",
                "conversion_scoring_no_feasible_candidate",
            }:
                continue
            if not bool(capacity_proof.get("search_exhausted", False)):
                continue
            if bool(capacity_proof.get("ordinary_conversion_feasible", False)):
                continue
            capacity_domain = str(
                capacity_proof.get("failure_domain", "")
                or failure_domain
                or ""
            )
            if capacity_domain not in {"resource_capacity", "resource_accounting"}:
                continue
            recovery_candidate = candidate
            break
    if recovery_candidate is None:
        return False

    # Prefer the authoritative merged proof, while accepting the candidate
    # payload used by lightweight callers/tests.
    candidate_proof = dict(recovery_candidate.get("proof", {}) or {})
    capacity_proof = dict(
        candidate_proof.get("conversion_capacity_proof", {}) or {}
    )
    required_source = (
        candidate_proof.get("required_counts", {})
        or recovery_candidate.get("required_counts", {})
        or proof.get("required_counts", {})
    )
    available_source = (
        candidate_proof.get("available_counts", {})
        or recovery_candidate.get("available_counts", {})
        or proof.get("available_counts", {})
    )
    required = {
        int(key): max(0, int(value))
        for key, value in dict(required_source or {}).items()
        if str(key).lstrip("-").isdigit() and int(value) > 0
    }
    available = {
        int(key): max(0, int(value))
        for key, value in dict(available_source or {}).items()
        if str(key).lstrip("-").isdigit() and int(value) > 0
    }
    capacity_missing = dict(capacity_proof.get("missing_counts", {}) or {})
    missing = {
        su_type: max(0, int(required.get(su_type, 0)) - int(available.get(su_type, 0)))
        for su_type in (11, 22, 23, 24)
    }
    for raw_key, raw_value in capacity_missing.items():
        if str(raw_key).lstrip("-").isdigit() and int(raw_key) in missing:
            missing[int(raw_key)] = max(
                int(missing[int(raw_key)]), max(0, int(raw_value))
            )
    if int(missing.get(24, 0)) > 0:
        diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
        diagnostics["ring_recovery_capacity_proof"] = {
            "required_counts": {str(key): int(value) for key, value in required.items()},
            "available_counts": {str(key): int(value) for key, value in available.items()},
            "missing_counts": {"24": int(missing[24])},
            "generic_surplus": 0,
            "aromatic_su11_donors": int(available.get(12, 0) + available.get(13, 0)),
            "generic_required": int(missing[24]),
        }
        plan.diagnostics = diagnostics
        diagnostics["ring_recovery_capacity_proof"]["recovery_allowed"] = False
        diagnostics["ring_recovery_capacity_proof"]["blocked_reason"] = (
            "su24_deficit_cannot_be_filled_by_recovery"
        )
        plan.diagnostics = diagnostics
        return False
    if not any(int(value) > 0 for value in missing.values()):
        return False

    # The conversion planner's exhausted-search proof is authoritative. Do not
    # reproduce its donor graph with a second approximate capacity formula.
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostics["ring_recovery_capacity_proof"] = {
        "required_counts": {str(key): int(value) for key, value in required.items()},
        "available_counts": {str(key): int(value) for key, value in available.items()},
        "missing_counts": {str(key): int(value) for key, value in missing.items() if int(value) > 0},
        "conversion_capacity_proof": dict(capacity_proof),
        "recovery_allowed": True,
    }
    plan.diagnostics = diagnostics
    return True


def run_block_c_resource_adjustment_loop_impl(
    adjuster: Any,
    H: torch.Tensor,
    nodes: Optional[List[Any]],
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    rebuild_fn: Any = None,
    validate_fn: Any = None,
    output_dir: Optional[str] = None,
    flex_max_primary_adjustments: int = 10,
    external_max_primary_adjustments: int = 10,
    internal_max_primary_adjustments: int = 30,
    difference_spectrum: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Stage3WorkflowResult:
    flex_primary_limit = max(0, int(flex_max_primary_adjustments))
    external_primary_limit = max(0, int(external_max_primary_adjustments))
    internal_primary_limit = max(0, int(internal_max_primary_adjustments))
    compensation_limit = int(kwargs.pop("max_h_compensation_ops_per_adjustment", 8))
    residual_flex_preferred_23 = int(kwargs.pop("residual_flex_preferred_23", 3))
    residual_flex_min_23 = int(kwargs.pop("residual_flex_min_23", 2))
    residual_side_preferred_23 = int(kwargs.pop("residual_side_preferred_23", 2))
    residual_side_min_23 = int(kwargs.pop("residual_side_min_23", 1))
    h_tolerance = float(
        kwargs.pop("h_tolerance", DEFAULT_STAGE3_H_SOFT_TOLERANCE)
    )
    adjuster._h_tolerance = h_tolerance
    outer_cycle = int(kwargs.pop("outer_cycle", 0))
    iteration_offset = max(0, int(kwargs.pop("iteration_offset", 0)))
    enable_post_external_spectrum = bool(kwargs.pop("enable_post_external_spectrum", True))
    spectrum_active_through_cycle = int(
        kwargs.pop(
            "spectrum_active_through_cycle",
            _DEFAULT_STAGE3_SPECTRUM_ACTIVE_THROUGH_CYCLE,
        )
    )
    if spectrum_active_through_cycle < 0:
        raise ValueError(
            "block_c_resource.spectrum_active_through_cycle must be >= 0"
        )
    spectrum_candidate_limit = int(kwargs.pop("spectrum_candidate_limit", 8))
    spectrum_max_primary_adjustments = int(kwargs.pop("spectrum_max_primary_adjustments", 10))
    spectrum_min_r2_gain = float(kwargs.pop("spectrum_min_r2_gain", 0.0005))
    verbose = bool(kwargs.pop("verbose", False))
    context_snapshot_fn = kwargs.pop("context_snapshot_fn", None)
    context_restore_fn = kwargs.pop("context_restore_fn", None)
    if kwargs:
        raise ValueError(
            "unsupported block_c_resource options: "
            + ", ".join(sorted(str(key) for key in kwargs))
        )
    _reset_stage3_generation_cursors(adjuster)
    # One live assembler belongs to one Layer1/2 generation.  It is discarded
    # only after a fresh rebuild, never after an ordinary SU conversion.
    adjuster._stage3_external_assembler = None
    nodes_current = list(nodes or [])
    _clear_stage3_node_runtime_state(nodes_current)
    H_current = torch.clamp(H.detach().clone(), min=0).long()
    diff_current: Dict[str, Any] = dict(difference_spectrum or {})
    cycles: List[Stage3WorkflowCycleRecord] = []
    accepted_cycles = 0
    stop_reason = "stage3_not_completed"
    progress_phase = Stage3ProgressPhase.FLEX_RECONCILIATION
    # The primary-conversion budget is shared by all external roles in one
    # Layer1/2 generation (flex, fatty ring, and side chain).  Recovery is
    # retried in this same generation, so its forced 24->23 conversion must
    # consume the remaining budget instead of silently resetting it.
    external_primary_adjustments_used = 0
    external_primary_adjustments_total = 0
    external_generation_rebuild_pending = False
    rejected_spectrum_histogram_signatures: set[str] = set()
    rejected_spectrum_bundle_signatures: set[str] = set()
    forbidden_spectrum_inverse_signatures: set[str] = set()
    rejected_resource_conversion_signatures: set[str] = set()
    console_reporter = _Stage3CycleConsoleReporter(
        verbose=verbose,
        flex_limit=flex_primary_limit,
        external_limit=external_primary_limit,
        internal_limit=internal_primary_limit,
    )
    # Ring recovery is a second-stage fallback. It is enabled only after the
    # current fatty-ring or side contract has failed the ordinary conversion
    # planner, then reset so the released pool is retried before another ring.
    allow_ring_recovery = False
    initial_context = (
        deepcopy(context_snapshot_fn())
        if callable(context_snapshot_fn)
        else None
    )
    initial_state_signature = _stage3_physical_state_signature(
        nodes_current,
        H_current,
        initial_context,
    )
    initial_chemical_state_signature = _stage3_chemical_state_signature(
        nodes_current,
        H_current,
    )
    visited_state_signatures = {initial_state_signature}
    visited_chemical_state_signatures = {initial_chemical_state_signature}
    graph_generation = int(max(0, outer_cycle)) * 1_000_000
    generation_metrics: List[Dict[str, int]] = [{
        "generation_id": int(graph_generation),
        "internal_physical_conversion_count": 0,
        "internal_ring_count": 0,
    }]
    spectrum_rebalance_finalized = False
    spectrum_finalized_on_cycle: Optional[int] = None

    # ``cycle_idx`` counts complete Stage3 generations.  A generation may
    # require several planner/materialization attempts while its live
    # assembler is continued; those attempts must not consume a new cycle or
    # a new external budget.  ``attempt_idx`` is diagnostic only.
    cycle_idx = 0
    attempt_idx = 0
    generation_cycle_number: Optional[int] = None
    generation_attempt_number = 0
    while True:
        attempt_idx += 1
        if generation_cycle_number is None:
            cycle_idx += 1
            generation_cycle_number = int(iteration_offset) + int(cycle_idx)
            generation_attempt_number = 0
        generation_attempt_number += 1
        cycle_number = int(generation_cycle_number)
        attempt_number = int(generation_attempt_number)
        spectrum_enabled_this_cycle = bool(
            enable_post_external_spectrum
            and cycle_number <= spectrum_active_through_cycle
        )
        if (
            enable_post_external_spectrum
            and not spectrum_enabled_this_cycle
            and not spectrum_rebalance_finalized
        ):
            spectrum_rebalance_finalized = True
            spectrum_finalized_on_cycle = int(cycle_number)
            print(
                "[Spectrum Trial Finalized] "
                f"active_through_cycle={spectrum_active_through_cycle} "
                f"finalized_on_cycle={cycle_number}; "
                "retaining the latest accepted Layer2 spectral state and "
                "continuing topology assembly only",
                flush=True,
            )
        generation_id = int(graph_generation)
        cycle_context_before = (
            context_snapshot_fn()
            if callable(context_snapshot_fn)
            else None
        )
        rotation_cursor_before = (
            int(getattr(adjuster, "_stage3_su11_rotation_index", 0) or 0)
            if hasattr(adjuster, "_stage3_su11_rotation_index")
            else None
        )
        parity_cursor_before = int(
            getattr(adjuster, "_stage3_parity_balance_cursor", 0) or 0
        )
        special_degree_meta = (
            adjuster._get_special_degree_meta(H_current)
            if callable(getattr(adjuster, "_get_special_degree_meta", None))
            else None
        )
        spectrum_baseline_key = _spectrum_search_baseline_key(
            nodes_current,
            H_current,
            diff_current,
        )
        spectrum_memory_entry = _spectrum_search_memory_entry(
            adjuster,
            spectrum_baseline_key,
        )
        persistent_histogram_signatures = (
            set(spectrum_memory_entry["primary_histogram_signatures"])
            | set(spectrum_memory_entry["final_histogram_signatures"])
        )
        plan_started = time.perf_counter()
        if verbose:
            print(f"[Block C:resource #{cycle_number}] topology plan start", flush=True)
        plan = adjuster.plan_topology_resource_adjustments(
            H=H_current.detach().clone(),
            nodes=list(nodes_current or []),
            S_target=S_target,
            E_target=E_target,
            flex_max_primary_adjustments=flex_primary_limit,
            external_max_primary_adjustments=external_primary_limit,
            external_primary_adjustments_used=int(
                external_primary_adjustments_used
            ),
            internal_max_primary_adjustments=internal_primary_limit,
            allow_ring_recovery=bool(allow_ring_recovery),
            max_h_compensation_ops_per_adjustment=compensation_limit,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            h_tolerance=h_tolerance,
            generation_id=int(generation_id),
            difference_spectrum=dict(diff_current or {}),
            enable_post_external_spectrum=bool(spectrum_enabled_this_cycle),
            progress_phase=progress_phase,
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            rejected_spectrum_histogram_signatures=sorted(
                set(rejected_spectrum_histogram_signatures)
                | persistent_histogram_signatures
            ),
            rejected_spectrum_bundle_signatures=sorted(
                set(rejected_spectrum_bundle_signatures)
                | set(spectrum_memory_entry["bundle_signatures"])
            ),
            forbidden_spectrum_inverse_signatures=sorted(
                set(forbidden_spectrum_inverse_signatures)
                | set(spectrum_memory_entry["rejected_inverse_signatures"])
            ),
            forbidden_resource_conversion_signatures=sorted(
                rejected_resource_conversion_signatures
            ),
            external_assembler=getattr(adjuster, "_stage3_external_assembler", None),
        )
        runtime_handle = dict(getattr(plan, "runtime", {}) or {}).get(
            "external_assembler"
        )
        if runtime_handle is not None:
            adjuster._stage3_external_assembler = runtime_handle
        initial_plan_phase = str(
            getattr(plan, "phase", "")
            or dict(getattr(plan, "diagnostics", {}) or {}).get(
                "stage3_phase", ""
            )
        )
        console_reporter.begin(
            cycle=cycle_number,
            attempt=attempt_number,
            phase=initial_plan_phase,
            external_used=external_primary_adjustments_used,
        )
        progress_phase = _progress_phase_after_plan(plan, progress_phase)
        _update_generation_metrics(
            generation_metrics,
            generation_id=generation_id,
            plan=plan,
        )
        if verbose:
            _print_stage3_plan_summary(
                cycle_number,
                plan,
                H_current,
                E_target,
                special_degree_meta=special_degree_meta,
                verbose=verbose,
                attempt_number=attempt_number,
                external_used_before=external_primary_adjustments_used,
                external_limit=external_primary_limit,
            )
        if str(getattr(plan, "reason", "") or "") == "stage3_spectrum_candidates_ready":
            # The one-generation inverse guard has now been consumed by the
            # candidate generator.  A newly accepted bundle installs the next
            # guard below; rejected/no-op generations carry no stale filter.
            forbidden_spectrum_inverse_signatures.clear()
            spectrum_H_before = H_current.detach().clone()
            spectrum_nodes_before = deepcopy(nodes_current)
            baseline_r2 = float(dict(diff_current or {}).get("r2", float("nan")))
            baseline_diff = dict(diff_current or {})
            baseline_source = str(baseline_diff.get("source_stage", "") or "")
            baseline_contract = str(baseline_diff.get("spectral_contract", "") or "")
            spectrum_context_before = (
                deepcopy(cycle_context_before)
            )
            baseline_valid = bool(
                np.isfinite(float(baseline_r2))
                and baseline_source == "layer2"
                and baseline_contract == "template_center_mu"
                and rebuild_fn is not None
                and validate_fn is not None
            )
            baseline_reason = (
                "ok"
                if baseline_valid
                else "spectrum_baseline_is_not_layer2"
                if baseline_source != "layer2"
                else "spectrum_baseline_contract_mismatch"
                if baseline_contract != "template_center_mu"
                else "spectrum_baseline_r2_invalid"
                if not np.isfinite(float(baseline_r2))
                else "candidate_rebuild_unavailable"
            )
            if verbose:
                print(
                    f"[Spectrum Layer2 Baseline] cycle={cycle_number} "
                    f"valid={baseline_valid} reason={baseline_reason} "
                    f"r2={baseline_r2:.4f} "
                    f"histogram={_selected_histogram_counts(spectrum_H_before)}",
                    flush=True,
                )
            evaluated_candidates: List[Dict[str, Any]] = []
            best_accepted_option: Optional[Tuple[Any, ...]] = None
            for candidate in list(getattr(plan, "spectrum_candidates", []) or []):
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(spectrum_context_before))
                candidate_diagnostics = dict(getattr(candidate, "diagnostics", {}) or {})
                primary_histogram_signature = str(
                    candidate_diagnostics.get(
                        "primary_histogram_signature",
                        getattr(candidate, "histogram_signature", ""),
                    )
                    or ""
                )
                primary_bundle_signature = str(
                    candidate_diagnostics.get("bundle_signature", "") or ""
                )
                if primary_histogram_signature:
                    spectrum_memory_entry["primary_histogram_signatures"].add(
                        primary_histogram_signature
                    )
                if primary_bundle_signature:
                    spectrum_memory_entry["bundle_signatures"].add(
                        primary_bundle_signature
                    )
                if not baseline_valid:
                    evaluated_candidates.append({
                        "candidate": (
                            candidate.to_dict()
                            if hasattr(candidate, "to_dict")
                            else dict(candidate or {})
                        ),
                        "accepted": False,
                        "valid": False,
                        "valid_reason": str(baseline_reason),
                        "reason": "saved_layer2_baseline_unavailable",
                        "significant": False,
                    })
                    continue
                (
                    candidate,
                    candidate_nodes,
                    candidate_H,
                    parity_diagnostics,
                ) = _prepare_spectrum_candidate_for_rebuild(
                    plan=plan,
                    candidate=candidate,
                    nodes=spectrum_nodes_before,
                    H=spectrum_H_before,
                    special_degree_meta=special_degree_meta,
                    balance_cursor=int(parity_cursor_before),
                )
                apply_proof = dict(parity_diagnostics.get("apply_proof", {}) or {})
                candidate_diagnostics = dict(
                    getattr(candidate, "diagnostics", {}) or {}
                )
                final_histogram_signature = str(
                    getattr(candidate, "histogram_signature", "") or ""
                )
                if final_histogram_signature:
                    spectrum_memory_entry["final_histogram_signatures"].add(
                        final_histogram_signature
                    )
                trial_operations, trial_delta = _spectrum_trial_operations(candidate)
                candidate_h_value = (
                    float(_h_total(candidate_H, special_degree_meta))
                    if isinstance(candidate_H, torch.Tensor)
                    else float("nan")
                )
                if verbose:
                    print(
                        f"[Spectrum Trial Start] cycle={cycle_number} "
                        f"candidate_id={getattr(candidate, 'candidate_id', '')} "
                        f"spectral_rank={candidate_diagnostics.get('spectral_priority_rank', 'n/a')} "
                        f"bundle={candidate_diagnostics.get('bundle_kind', 'unknown')} "
                        f"batch={int(candidate_diagnostics.get('batch_size', len(list(getattr(candidate, 'operations', []) or []))) or 0)} "
                        f"conversions={dict(candidate_diagnostics.get('conversion_counts', {}) or {})} "
                        f"h_compensation={dict(candidate_diagnostics.get('h_compensation_conversion_counts', {}) or {})} "
                        f"parity_balance={bool(parity_diagnostics.get('balance_applied', False))} "
                        f"operations=[{trial_operations}] delta={trial_delta} "
                        f"H={float(_h_total(H_current, special_degree_meta)):.1f}->{candidate_h_value:.1f}",
                        flush=True,
                    )
                record: Dict[str, Any] = {
                    "candidate": candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate or {}),
                    "accepted": False,
                    "apply_proof": dict(apply_proof or {}),
                    "pre_rebuild_port_feasibility": dict(parity_diagnostics or {}),
                }
                if candidate_nodes is None or candidate_H is None or rebuild_fn is None or validate_fn is None:
                    record["reason"] = str(
                        dict(parity_diagnostics or {}).get(
                            "reason",
                            dict(apply_proof or {}).get(
                                "reason", "candidate_apply_or_rebuild_unavailable"
                            ),
                        )
                    )
                    rejected_spectrum_histogram_signatures.add(str(getattr(candidate, "histogram_signature", "") or ""))
                    if primary_bundle_signature:
                        rejected_spectrum_bundle_signatures.add(primary_bundle_signature)
                    rejected_inverse_signature = str(
                        dict(getattr(candidate, "diagnostics", {}) or {}).get(
                            "primary_inverse_signature",
                            getattr(candidate, "inverse_signature", ""),
                        )
                        or ""
                    )
                    if rejected_inverse_signature:
                        spectrum_memory_entry["rejected_inverse_signatures"].add(
                            rejected_inverse_signature
                        )
                    record.update({
                        "valid": False,
                        "valid_reason": str(record["reason"]),
                        "significant": False,
                        "reason": str(record["reason"]),
                    })
                    evaluated_candidates.append(record)
                    if verbose:
                        print(
                            f"[Spectrum Trial Result] candidate_id={_short_candidate_id(candidate)} "
                            f"graph_valid=False valid_reason={record['reason']} "
                            "rebuild_started=False "
                            "accepted=False state_retained=False",
                            flush=True,
                        )
                    continue
                candidate_context = None
                try:
                    cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                        H_seed=candidate_H,
                        phase="remaining_pool_spectrum_rebalance",
                        cycle_idx=cycle_number - 1,
                        nodes_seed=candidate_nodes,
                    )
                    _clear_stage3_node_runtime_state(cand_nodes)
                    valid, valid_reason = validate_fn(
                        nodes=cand_nodes,
                        H=cand_H,
                        tag=f"BlockC:spectrum #{cycle_number}/{getattr(candidate, 'candidate_id', '')}",
                    )
                    candidate_source = str(dict(cand_diff or {}).get("source_stage", "") or "")
                    candidate_contract = str(
                        dict(cand_diff or {}).get("spectral_contract", "") or ""
                    )
                    if candidate_source != "layer2":
                        valid = False
                        valid_reason = "spectrum_candidate_is_not_layer2"
                    elif candidate_contract != "template_center_mu":
                        valid = False
                        valid_reason = "spectrum_candidate_contract_mismatch"
                    candidate_context = (
                        deepcopy(context_snapshot_fn())
                        if callable(context_snapshot_fn)
                        else None
                    )
                except Exception as exc:
                    valid = False
                    valid_reason = f"spectrum_candidate_rebuild_failed:{exc}"
                    cand_nodes = cand_H = cand_diff = None
                    cand_r2 = float("nan")
                profile_stats = (
                    adjuster._get_su_common_stats()
                    if callable(getattr(adjuster, "_get_su_common_stats", None))
                    else None
                )
                before_loss = spectrum_profile_loss(
                    baseline_diff,
                    profile_stats,
                )
                after_loss = (
                    spectrum_profile_loss(cand_diff, profile_stats)
                    if cand_diff is not None
                    else float("inf")
                )
                r2_gain = float(cand_r2) - float(baseline_r2) if np.isfinite(cand_r2) and np.isfinite(baseline_r2) else float("-inf")
                window_gain = float(before_loss) - float(after_loss)
                candidate_state_signature = (
                    _stage3_physical_state_signature(
                        list(cand_nodes or []),
                        cand_H,
                        candidate_context,
                    )
                    if isinstance(cand_H, torch.Tensor)
                    else ""
                )
                candidate_chemical_state_signature = (
                    _stage3_chemical_state_signature(
                        list(cand_nodes or []),
                        cand_H,
                    )
                    if isinstance(cand_H, torch.Tensor)
                    else ""
                )
                repeated_state = bool(
                    candidate_state_signature
                    and candidate_state_signature in visited_state_signatures
                ) or bool(
                    candidate_chemical_state_signature
                    and candidate_chemical_state_signature
                    in visited_chemical_state_signatures
                )
                predicted_sse_gain = float(
                    candidate_diagnostics.get("predicted_global_sse_gain", 0.0)
                    or 0.0
                )
                required_r2_gain = float(spectrum_min_r2_gain)
                significant = bool(
                    valid
                    and (
                        float(r2_gain) >= float(required_r2_gain)
                        or math.isclose(
                            float(r2_gain),
                            float(required_r2_gain),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                )
                record.update({
                    "valid": bool(valid),
                    "valid_reason": str(valid_reason),
                    "r2_before": float(baseline_r2),
                    "r2_after": float(cand_r2),
                    "r2_gain": float(r2_gain),
                    "window_loss_before": float(before_loss),
                    "window_loss_after": float(after_loss),
                    "window_loss_gain": float(window_gain),
                    "predicted_global_sse_gain": float(predicted_sse_gain),
                    "required_r2_gain": float(required_r2_gain),
                    "repeated_state": bool(repeated_state),
                    "state_signature": str(candidate_state_signature),
                    "chemical_state_signature": str(candidate_chemical_state_signature),
                    "significant": bool(significant),
                })
                if verbose:
                    print(
                        f"[Spectrum Trial Result] candidate_id={getattr(candidate, 'candidate_id', '')} "
                        f"graph_valid={bool(valid)} valid_reason={str(valid_reason)} "
                        f"r2={baseline_r2:.4f}->{float(cand_r2):.4f} "
                        f"r2_gain={float(r2_gain):.4f} window_gain={float(window_gain):.4f} "
                        f"predicted_sse_gain={float(predicted_sse_gain):.4f} "
                        f"required_r2_gain={float(required_r2_gain):.4f} "
                        f"selection_eligible={bool(significant)} accepted=False state_retained=False",
                        flush=True,
                    )
                if significant:
                    option = (
                        float(r2_gain),
                        float(window_gain),
                        float(predicted_sse_gain),
                        str(getattr(candidate, "candidate_id", "") or ""),
                        cand_nodes,
                        cand_H,
                        cand_diff,
                        float(cand_r2),
                        candidate,
                        candidate_context,
                        str(candidate_state_signature),
                    )
                    option_key = (
                        -float(option[0]),
                        str(option[3]),
                    )
                    best_key = None
                    if best_accepted_option is not None:
                        best_key = (
                            -float(best_accepted_option[0]),
                            str(best_accepted_option[3]),
                        )
                    if best_key is None or option_key < best_key:
                        best_accepted_option = option
                else:
                    rejected_spectrum_histogram_signatures.add(str(getattr(candidate, "histogram_signature", "") or ""))
                    if primary_bundle_signature:
                        rejected_spectrum_bundle_signatures.add(primary_bundle_signature)
                    rejected_inverse_signature = str(
                        candidate_diagnostics.get(
                            "primary_inverse_signature",
                            getattr(candidate, "inverse_signature", ""),
                        )
                        or ""
                    )
                    if rejected_inverse_signature:
                        spectrum_memory_entry["rejected_inverse_signatures"].add(
                            rejected_inverse_signature
                        )
                evaluated_candidates.append(record)
                # Only the current best candidate keeps its rebuilt node graph.
                # Rejected and superseded graphs become collectible here.
                cand_nodes = cand_H = cand_diff = candidate_context = None
            if best_accepted_option is not None:
                selected = best_accepted_option
                (
                    r2_gain,
                    window_gain,
                    predicted_sse_gain,
                    _,
                    cand_nodes,
                    cand_H,
                    cand_diff,
                    cand_r2,
                    candidate,
                    candidate_context,
                    selected_state_signature,
                ) = selected
                if callable(context_restore_fn):
                    context_restore_fn(deepcopy(candidate_context))
                selected_parity = dict(
                    dict(getattr(candidate, "diagnostics", {}) or {}).get(
                        "parity_balance", {}
                    ) or {}
                )
                nodes_current = list(cand_nodes or [])
                H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
                diff_current = dict(cand_diff or {})
                progress_phase = Stage3ProgressPhase.FLEX_RECONCILIATION
                external_primary_adjustments_used = 0
                external_generation_rebuild_pending = False
                adjuster._stage3_external_assembler = None
                _reset_stage3_generation_cursors(adjuster)
                rejected_spectrum_histogram_signatures.clear()
                rejected_spectrum_bundle_signatures.clear()
                rejected_resource_conversion_signatures.clear()
                selected_inverse_signature = str(
                    dict(getattr(candidate, "diagnostics", {}) or {}).get(
                        "primary_inverse_signature",
                        getattr(candidate, "inverse_signature", ""),
                    )
                    or ""
                )
                forbidden_spectrum_inverse_signatures = (
                    {selected_inverse_signature}
                    if selected_inverse_signature
                    else set()
                )
                accepted_cycles += 1
                # A Layer1/2 rebuild starts a new Stage3 generation. Do not
                # carry candidate-repeat blacklists from the old graph.
                visited_state_signatures.add(str(selected_state_signature))
                visited_chemical_state_signatures.add(
                    _stage3_chemical_state_signature(nodes_current, H_current)
                )
                if len(visited_state_signatures) > _MAX_RETAINED_GENERATION_SIGNATURES:
                    visited_state_signatures.pop()
                if len(visited_chemical_state_signatures) > _MAX_RETAINED_GENERATION_SIGNATURES:
                    visited_chemical_state_signatures.pop()
                for record in evaluated_candidates:
                    candidate_data = dict(record.get("candidate", {}) or {})
                    if str(candidate_data.get("candidate_id", "")) == str(getattr(candidate, "candidate_id", "")):
                        record["accepted"] = True
                        record["reason"] = "layer2_r2_gain_threshold_met"
                    else:
                        rejected_inverse_signature = str(
                            dict(candidate_data.get("diagnostics", {}) or {}).get(
                                "primary_inverse_signature",
                                candidate_data.get("inverse_signature", ""),
                            )
                            or ""
                        )
                        if rejected_inverse_signature:
                            spectrum_memory_entry[
                                "rejected_inverse_signatures"
                            ].add(rejected_inverse_signature)
                _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                    cycle=cycle_number,
                    phase="remaining_pool_spectrum_rebalance",
                    applied=True,
                    stop_reason="accepted",
                    accepted_state_retained=True,
                    rebuild_valid=True,
                    layer2_r2=float(cand_r2),
                    accepted=True,
                    acceptance_reason="layer2_r2_gain_threshold_met",
                    primary_adjustment_count=int(getattr(candidate, "primary_adjustment_count", 0) or 0),
                    parity_balance_count=len(
                        list(selected_parity.get("applied_operations", []) or [])
                    ),
                    fresh_rebuild=True,
                ))
                if verbose:
                    print(
                        f"[Spectrum Trial Commit] cycle={cycle_number} "
                        f"candidate_id={_short_candidate_id(candidate)} "
                        f"r2={baseline_r2:.4f}->{float(cand_r2):.4f} "
                        f"r2_gain={float(r2_gain):.4f} "
                        f"predicted_sse_gain={float(predicted_sse_gain):.4f} "
                        "accepted=True state_retained=True",
                        flush=True,
                    )
                console_reporter.spectrum_commit(
                    records=evaluated_candidates,
                    candidate_id=_short_candidate_id(candidate),
                    r2_before=baseline_r2,
                    r2_after=float(cand_r2),
                    operations=list(getattr(candidate, "operations", []) or []),
                )
                graph_generation += 1
                generation_metrics.append({
                    "generation_id": int(graph_generation),
                    "internal_physical_conversion_count": 0,
                    "internal_ring_count": 0,
                })
                if len(generation_metrics) > _MAX_RETAINED_GENERATION_SIGNATURES:
                    del generation_metrics[:-_MAX_RETAINED_GENERATION_SIGNATURES]
                generation_cycle_number = None
                continue
            console_reporter.record_spectrum_trials(evaluated_candidates)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(spectrum_context_before))
            plan = adjuster.plan_topology_resource_adjustments(
                H=H_current.detach().clone(),
                nodes=list(nodes_current or []),
                S_target=S_target,
                E_target=E_target,
                    flex_max_primary_adjustments=flex_primary_limit,
                    external_max_primary_adjustments=external_primary_limit,
                    external_primary_adjustments_used=int(
                        external_primary_adjustments_used
                    ),
                    internal_max_primary_adjustments=internal_primary_limit,
                max_h_compensation_ops_per_adjustment=compensation_limit,
                residual_flex_preferred_23=residual_flex_preferred_23,
                residual_flex_min_23=residual_flex_min_23,
                residual_side_preferred_23=residual_side_preferred_23,
                residual_side_min_23=residual_side_min_23,
                h_tolerance=h_tolerance,
                generation_id=int(generation_id),
                difference_spectrum=dict(diff_current or {}),
                enable_post_external_spectrum=bool(spectrum_enabled_this_cycle),
                progress_phase=Stage3ProgressPhase.SPECTRUM_REBALANCE,
                spectrum_candidate_limit=int(spectrum_candidate_limit),
                spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
                rejected_spectrum_histogram_signatures=sorted(
                    set(rejected_spectrum_histogram_signatures)
                    | persistent_histogram_signatures
                ),
                rejected_spectrum_bundle_signatures=sorted(
                    set(rejected_spectrum_bundle_signatures)
                    | set(spectrum_memory_entry["bundle_signatures"])
                ),
                forbidden_spectrum_inverse_signatures=sorted(
                    set(forbidden_spectrum_inverse_signatures)
                    | set(spectrum_memory_entry["rejected_inverse_signatures"])
                ),
                forbidden_resource_conversion_signatures=sorted(
                    rejected_resource_conversion_signatures
                ),
            )
            progress_phase = _progress_phase_after_plan(plan, progress_phase)
            _update_generation_metrics(
                generation_metrics,
                generation_id=generation_id,
                plan=plan,
            )
            plan.diagnostics["rejected_spectrum_candidates"] = evaluated_candidates
            plan.diagnostics["spectrum_rebalance_rolled_back"] = True
            plan.diagnostics["spectrum_layer2_baseline"] = {
                "valid": bool(baseline_valid),
                "reason": str(baseline_reason),
                "source_stage": str(baseline_source),
                "spectral_contract": str(baseline_contract),
                "r2": float(baseline_r2),
            }
            replanned_phase = str(
                getattr(plan, "phase", "")
                or dict(getattr(plan, "diagnostics", {}) or {}).get(
                    "stage3_phase", ""
                )
            )
            console_reporter.begin(
                cycle=cycle_number,
                attempt=attempt_number,
                phase=replanned_phase,
                external_used=external_primary_adjustments_used,
            )
            if verbose:
                _print_stage3_plan_summary(
                    cycle_number,
                    plan,
                    H_current,
                    E_target,
                    special_degree_meta=special_degree_meta,
                    verbose=True,
                    attempt_number=attempt_number,
                    external_used_before=external_primary_adjustments_used,
                    external_limit=external_primary_limit,
                )
            if str(getattr(plan, "reason", "") or "") == "stage3_spectrum_candidates_ready":
                # The replanner produced another candidate batch. Re-enter the
                # normal spectrum-evaluation path instead of treating this
                # intermediate plan as a histogram-application stop condition.
                continue
        elif bool(
            dict(getattr(plan, "diagnostics", {}) or {}).get(
                "spectrum_stage_evaluated", False
            )
        ):
            forbidden_spectrum_inverse_signatures.clear()
        phase_result = getattr(plan, "phase_result", None)
        for transaction in list(getattr(phase_result, "transactions", []) or []):
            transaction.outer_cycle = int(outer_cycle)
            transaction.iteration = int(cycle_number)
        if verbose:
            print(
                f"[Block C:resource #{cycle_number}] topology plan done "
                f"({time.perf_counter() - plan_started:.2f}s)",
                flush=True,
            )
        phase = str(
            getattr(plan, "phase", "")
            or dict(getattr(plan, "diagnostics", {}) or {}).get(
                "stage3_phase", ""
            )
        )
        histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
        plan_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
        forced_recovery_plan = bool(
            plan_diagnostics.get("external_recovery_continuation", False)
        )
        recovery_request_was_active = bool(allow_ring_recovery)
        recovery_gate_terminal = False
        # A recovery request is one-shot.  If the recovery-enabled assembly
        # pass found no recyclable ring, fall through to the normal terminal
        # handling instead of retrying the same request forever.
        if allow_ring_recovery and not forced_recovery_plan:
            allow_ring_recovery = False

        # The first ordinary fatty-ring/side resource failure is a gate, not a
        # terminal result. Ask the assembler for exactly one pure-resource ring
        # only after the regular conversion planner has produced no candidate.
        if (
            not bool(getattr(plan, "ok", False))
            and histogram_plan is not None
            and not list(getattr(histogram_plan, "groups", []) or [])
            and not forced_recovery_plan
            and not recovery_request_was_active
            and external_primary_adjustments_used < external_primary_limit
            and _external_ring_recovery_authorized(plan)
        ):
            live_assembler = getattr(adjuster, "_stage3_external_assembler", None)
            candidates = ring_recovery_candidate_costs(live_assembler)
            remaining_budget = max(
                0,
                int(external_primary_limit)
                - int(external_primary_adjustments_used),
            )
            affordable = [
                dict(item or {})
                for item in candidates
                if int(dict(item or {}).get("su24_cost", 0) or 0)
                <= int(remaining_budget)
            ]
            full_budget_candidates = [
                dict(item or {})
                for item in candidates
                if int(dict(item or {}).get("su24_cost", 0) or 0)
                <= int(external_primary_limit)
            ]
            if affordable and live_assembler is not None:
                live_assembler.ring_recovery_max_su24_cost = int(
                    remaining_budget
                )
                allow_ring_recovery = True
                print(
                    f"[Block C:resource #{cycle_number}] ordinary conversion "
                    "capacity exhausted; enabling one-ring recovery retry",
                    flush=True,
                )
                continue
            if full_budget_candidates:
                # Do not dismantle a ring that cannot be committed atomically
                # in this generation. Cross the normal external boundary with
                # the graph untouched, then retry under a fresh full budget.
                plan.ok = True
                plan.reason = "stage3_complete"
                plan.plan_kind = "external_recovery_budget_boundary"
                plan_diagnostics.update({
                    "external_stage_completed": False,
                    "external_generation_rebuild_required": True,
                    "external_conversion_budget_exhausted": True,
                    "external_generation_rebuild_reason": (
                        "external_budget_insufficient_for_atomic_recovery"
                    ),
                    "ring_recovery_candidates": list(candidates),
                    "external_remaining_budget": int(remaining_budget),
                })
                plan.diagnostics = plan_diagnostics
                phase_result = getattr(plan, "phase_result", None)
                if phase_result is not None:
                    phase_result.rebuild_required = True
                    phase_result.stop_reason = (
                        "external_conversion_budget_exhausted"
                    )
            else:
                terminal_reason = (
                    "recovery_exceeds_external_budget"
                    if candidates
                    else "no_recyclable_resource_ring"
                )
                terminal_proof = _stage3_terminal_infeasibility_proof(plan)
                terminal_proof.update({
                    "reason": terminal_reason,
                    "recovery_is_atomic": True,
                    "graph_state_unchanged": True,
                    "ring_recovery_candidates": list(candidates),
                    "external_primary_adjustments_used": int(
                        external_primary_adjustments_used
                    ),
                    "external_max_primary_adjustments": int(
                        external_primary_limit
                    ),
                })
                phase_result = getattr(plan, "phase_result", None)
                if phase_result is not None:
                    phase_result.infeasibility_proof = terminal_proof
                    phase_result.stop_reason = terminal_reason
                plan.blocked_reason = terminal_reason
                plan.plan_kind = terminal_reason
                recovery_gate_terminal = True
        if (
            not bool(getattr(plan, "ok", False))
            and str(getattr(plan, "reason", "") or "") == "phase_resource_infeasible"
            and histogram_plan is not None
            and not list(getattr(histogram_plan, "groups", []) or [])
            and not forced_recovery_plan
            and not recovery_request_was_active
            and (
                recovery_gate_terminal
                or not _external_ring_recovery_authorized(plan)
            )
        ):
            # An empty-group resource failure without the explicit ordinary
            # conversion proof is terminal for this contract. It must not be
            # guessed as topology materialization and must not reopen recovery.
            terminal_proof = _stage3_terminal_infeasibility_proof(plan)
            stop_reason = "phase_resource_infeasible"
            _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=False,
                blocked_reason=str(
                    terminal_proof.get("reason", "ordinary_conversion_not_proven")
                    or "ordinary_conversion_not_proven"
                ),
                accepted=False,
                acceptance_reason="resource_failure_without_recovery_proof",
            ))
            break
        if (
            histogram_plan is not None
            and list(getattr(histogram_plan, "groups", []) or [])
            and not bool(getattr(plan, "ok", False))
            and str(getattr(plan, "reason", "") or "")
            == "phase_resource_infeasible"
        ):
            rejection_reason = str(
                getattr(plan, "blocked_reason", "")
                or "phase_resource_infeasible"
            )
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(plan, rejection_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason="phase_resource_infeasible",
                accepted_state_retained=True,
                rebuild_valid=False,
                blocked_reason=rejection_reason,
                accepted=False,
                acceptance_reason="planner_resource_accounting_failed",
                primary_adjustment_count=int(
                    getattr(histogram_plan, "primary_adjustment_count", 0) or 0
                ),
                physical_conversion_count=int(
                    getattr(histogram_plan, "physical_conversion_count", 0) or 0
                ),
                h_compensation_count=_histogram_h_compensation_count(
                    histogram_plan
                ),
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion "
                    f"variant after planner accounting failure: "
                    f"rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "phase_resource_infeasible"
            break
        if (
            not bool(getattr(plan, "ok", False))
            and str(getattr(plan, "reason", "") or "")
            == "phase_topology_materialization_invalid"
        ):
            stop_reason = "phase_topology_materialization_invalid"
            terminal_proof = _stage3_terminal_infeasibility_proof(plan)
            print(
                f"[Block C:resource #{cycle_number}] topology invalid "
                f"phase={str(terminal_proof.get('phase', phase) or phase)} "
                f"reason={str(terminal_proof.get('reason', stop_reason) or stop_reason)} "
                f"failed_contract={terminal_proof.get('failed_contract', {})}",
                flush=True,
            )
            _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=False,
                blocked_reason=str(
                    terminal_proof.get("reason", "")
                    or getattr(plan, "blocked_reason", "")
                    or stop_reason
                ),
                accepted=False,
                acceptance_reason="topology_materialization_invalid",
                primary_adjustment_count=0,
                physical_conversion_count=0,
                h_compensation_count=0,
            ))
            break
        external_plan_phase = str(phase).startswith("external")
        planned_external_primary_count = _external_primary_conversion_count(
            histogram_plan
        ) if histogram_plan is not None else 0
        external_budget_will_exhaust = bool(
            external_plan_phase
            and int(external_primary_adjustments_used)
            + int(planned_external_primary_count)
            >= int(external_primary_limit)
            and int(planned_external_primary_count) > 0
        )
        if external_plan_phase and not external_budget_will_exhaust:
            # Keep the live generation intact while the current contract is
            # retried. This applies equally to ordinary conversion and ring
            # recovery; port balance belongs to the eventual rebuild boundary.
            parity_ok, parity_diagnostics = True, {
                "enforced": False,
                "reason": "external_continuation_deferred_port_balance",
                "balance_applied": False,
            }
        elif histogram_plan is not None and list(
            getattr(histogram_plan, "groups", []) or []
        ):
            parity_ok, parity_diagnostics = _append_stage3_parity_balance(
                plan=plan,
                histogram_plan=histogram_plan,
                nodes=nodes_current,
                special_degree_meta=special_degree_meta,
                balance_cursor=int(parity_cursor_before),
                E_target=E_target,
                h_tolerance=h_tolerance,
            )
        else:
            parity_ok, parity_diagnostics = True, {
                "enforced": False,
                "reason": "no_physical_histogram_batch",
            }
        plan_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
        plan_diagnostics["pre_rebuild_port_feasibility"] = dict(parity_diagnostics)
        plan.diagnostics = plan_diagnostics
        if verbose and bool(parity_diagnostics.get("enforced", False)):
            before_degree = int(
                dict(parity_diagnostics.get("before", {}) or {}).get(
                    "total_port_degree",
                    0,
                )
                or 0
            )
            after_degree = int(
                dict(parity_diagnostics.get("after", {}) or {}).get(
                    "total_port_degree",
                    before_degree,
                )
                or before_degree
            )
            print(
                f"[Block C:resource #{cycle_number}] pre-rebuild port audit "
                f"degree={before_degree}->{after_degree} "
                f"balance={bool(parity_diagnostics.get('balance_applied', False))} "
                f"reason={str(parity_diagnostics.get('reason', ''))}",
                flush=True,
            )
        if not parity_ok:
            rejection_reason = str(
                parity_diagnostics.get("reason", "pre_rebuild_port_feasibility_failed")
            )
            setattr(plan, "blocked_reason", rejection_reason)
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(
                plan,
                rejection_reason,
            )
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason="pre_rebuild_port_feasibility_failed",
                accepted_state_retained=True,
                rebuild_valid=False,
                blocked_reason=rejection_reason,
                accepted=False,
                acceptance_reason="pre_rebuild_port_feasibility_failed",
                primary_adjustment_count=int(
                    getattr(histogram_plan, "primary_adjustment_count", 0) or 0
                ),
                physical_conversion_count=int(
                    getattr(histogram_plan, "physical_conversion_count", 0) or 0
                ),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                parity_balance_count=_histogram_parity_balance_count(histogram_plan),
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after port preflight failure: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "pre_rebuild_port_feasibility_failed"
            break

        phase_inventory = dict(getattr(histogram_plan, "inventory_before", {}) or {})
        phase_demand = dict(getattr(histogram_plan, "demand_before", {}) or {})
        aggregate_inventory_before = dict(
            getattr(histogram_plan, "aggregate_inventory_before", {}) or {}
        )
        aggregate_inventory_after = dict(
            getattr(histogram_plan, "aggregate_inventory_after", {}) or {}
        )
        aggregate_required_counts = dict(
            getattr(histogram_plan, "aggregate_required_counts", {}) or {}
        )
        state_signature = _stage3_physical_state_signature(
            nodes_current,
            H_current,
            cycle_context_before,
        )
        phase_eval = dict(getattr(plan, "runtime", {}) or {}).get("phase_evaluation")
        flex_audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
        assembly_view = dict(getattr(phase_eval, "assembly_view", {}) or {})
        flex_target_plan = dict(assembly_view.get("flex_target_plan", {}) or {})
        if verbose:
            print(
                "[Flex Target] "
                f"M={int(flex_audit.get('flex_lower', 0) or 0)} "
                f"N={int(flex_audit.get('flex_upper', 0) or 0)} "
                f"T={int(flex_audit.get('flex_target', flex_audit.get('target_flex_count', 0)) or 0)} "
                f"current={int(dict(getattr(phase_eval, 'flex_counts', {}) or {}).get('final', 0) or 0)} "
                f"external_capacity={int(flex_target_plan.get('external_candidate_capacity', 0) or 0)}"
                f" X/Y/Z={int(flex_audit.get('X', 0) or 0)}/{int(flex_audit.get('Y', 0) or 0)}/{int(flex_audit.get('Z', 0) or 0)}"
            )
        if verbose and phase in {
            "external_topology_assembly",
            "external_flex_closure",
            "external_fatty_ring_closure",
            "external_side_chain_closure",
        }:
            preferred_demand = dict(
                getattr(plan, "preferred_required_counts", {}) or {}
            )
            print(
                f"[External Demand] hard={phase_demand} preferred={preferred_demand} "
                f"available={phase_inventory} gap={_resource_gap(phase_inventory, phase_demand)}"
            )
            if aggregate_inventory_before or aggregate_required_counts:
                print(
                    f"[Aggregate Resources] required={aggregate_required_counts} "
                    f"before={aggregate_inventory_before} after={aggregate_inventory_after} "
                    f"gap={_resource_gap(aggregate_inventory_after, aggregate_required_counts)}"
                )
            resource_breakdown = dict(
                getattr(plan, "resource_request_breakdown", {}) or {}
            )
            selected_contracts = list(resource_breakdown.get("selected_active_contracts", []) or [])
            external_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
            print(
                "[External Contracts] "
                f"seen={int(external_diagnostics.get('external_contracts_seen', len(selected_contracts)) or 0)} "
                f"direct={int(external_diagnostics.get('external_directly_completed', 0) or 0)} "
                f"converted={int(external_diagnostics.get('external_completed_after_conversion', 0) or 0)} "
                f"blocked={int(external_diagnostics.get('external_blocked', 0) or 0)} "
                f"remaining={int(external_diagnostics.get('external_remaining_pending', 0) or 0)} "
                f"primary={int(external_diagnostics.get('external_primary_count', 0) or 0)}/"
                    f"{int(external_primary_limit)} "
                f"complete={bool(external_diagnostics.get('external_stage_completed', False))}"
            )
            for contract in selected_contracts:
                print(
                    "  [External Contract] "
                    f"contract={int(dict(contract or {}).get('contract_id', 0) or 0)} "
                    f"kind={str(dict(contract or {}).get('contract_kind', '') or '')} "
                    f"fragments={list(dict(contract or {}).get('source_fragment_ids', []) or [])} "
                    f"nodes={list(dict(contract or {}).get('claimed_source_node_ids', []) or [])} "
                    f"required={dict(dict(contract or {}).get('required_counts', {}) or {})}"
                )
        elif verbose and phase == "residual_resource_self_closure":
            print(
                "[Residual Flex] "
                f"minimum_gap={int(flex_target_plan.get('remaining_minimum_gap', 0) or 0)} "
                f"preferred_gap={int(flex_target_plan.get('remaining_preferred_gap', 0) or 0)}"
            )
        if str(getattr(plan, "reason", "")) == "stage3_complete":
            plan_external_budget_exhausted = bool(
                dict(getattr(plan, "diagnostics", {}) or {}).get(
                    "external_conversion_budget_exhausted", False
                )
            )
            plan_external_rebuild_required = bool(
                dict(getattr(plan, "diagnostics", {}) or {}).get(
                    "external_generation_rebuild_required", False
                )
            )
            if (
                external_generation_rebuild_pending
                or plan_external_budget_exhausted
                or plan_external_rebuild_required
            ):
                # Any accepted external primary conversion makes the current
                # Layer1/2 graph stale. Balance ports once at this generation
                # boundary, then rebuild from the balanced histogram.
                balanced_nodes, balanced_H, balance_diagnostics = (
                    _apply_deferred_port_balance(
                        plan=plan,
                        nodes=list(nodes_current or []),
                        H=H_current.detach().clone(),
                        special_degree_meta=special_degree_meta,
                        balance_cursor=int(parity_cursor_before),
                        E_target=E_target,
                        h_tolerance=h_tolerance,
                    )
                )
                balance_reason = str(balance_diagnostics.get("reason", ""))
                if balance_reason not in {"pre_rebuild_ports_balanced", ""}:
                    stop_reason = "pre_rebuild_port_feasibility_failed"
                    _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                        cycle=cycle_number,
                        phase="external_topology",
                        applied=False,
                        stop_reason=stop_reason,
                        accepted_state_retained=True,
                        blocked_reason=balance_reason,
                        accepted=False,
                        acceptance_reason=stop_reason,
                    ))
                    break
                nodes_current = list(balanced_nodes or nodes_current)
                H_current = torch.clamp(balanced_H.detach().clone(), min=0).long()
                console_reporter.record_balance_operations(
                    balance_diagnostics.get("operations", [])
                )
                console_reporter.finish(
                    boundary=(
                        "external_budget_exhausted"
                        if plan_external_budget_exhausted
                        else "external_generation_complete"
                    ),
                )
                if rebuild_fn is None or validate_fn is None:
                    stop_reason = "layer12_rebuild_invalid"
                    external_generation_rebuild_pending = False
                    _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                        cycle=cycle_number,
                        phase="external_topology",
                        applied=True,
                        stop_reason=stop_reason,
                        accepted_state_retained=True,
                        accepted=False,
                        acceptance_reason=stop_reason,
                        fresh_rebuild=False,
                    ))
                    break
                try:
                    external_rebuild_started = time.perf_counter()
                    console_reporter.rebuild_start(
                        "external_generation_boundary"
                    )
                    cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                        H_seed=H_current.detach().clone(),
                        phase="external_topology",
                        cycle_idx=cycle_number - 1,
                        nodes_seed=list(nodes_current or []),
                    )
                    _clear_stage3_node_runtime_state(cand_nodes)
                    candidate_histogram = torch.clamp(
                        cand_H.detach().cpu(), min=0
                    ).long()
                    if not torch.equal(candidate_histogram, H_current):
                        raise ValueError("fresh_rebuild_histogram_mismatch")
                    valid, valid_reason = validate_fn(
                        nodes=cand_nodes,
                        H=cand_H,
                        tag=f"BlockC:external-generation-rebuild #{cycle_number}",
                    )
                    if not valid:
                        raise ValueError(str(valid_reason))
                    external_rebuild_context = (
                        deepcopy(context_snapshot_fn())
                        if callable(context_snapshot_fn)
                        else None
                    )
                except Exception as exc:
                    stop_reason = "layer12_rebuild_invalid"
                    external_generation_rebuild_pending = False
                    print(
                        f"[Block C:resource #{cycle_number}] external generation rebuild failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                        cycle=cycle_number,
                        phase="external_topology",
                        applied=True,
                        stop_reason=stop_reason,
                        accepted_state_retained=True,
                        rebuild_valid=False,
                        blocked_reason=str(exc),
                        accepted=False,
                        acceptance_reason=stop_reason,
                        fresh_rebuild=True,
                    ))
                    break
                nodes_current = list(cand_nodes or [])
                H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
                diff_current = dict(cand_diff or {})
                graph_generation += 1
                generation_metrics.append({
                    "generation_id": int(graph_generation),
                    "internal_physical_conversion_count": 0,
                    "internal_ring_count": 0,
                })
                if len(generation_metrics) > _MAX_RETAINED_GENERATION_SIGNATURES:
                    del generation_metrics[:-_MAX_RETAINED_GENERATION_SIGNATURES]
                progress_phase = Stage3ProgressPhase.FLEX_RECONCILIATION
                external_primary_adjustments_used = 0
                external_generation_rebuild_pending = False
                adjuster._stage3_external_assembler = None
                _reset_stage3_generation_cursors(adjuster)
                # All continuation cursors and rejection blacklists belong to
                # the old Layer1/2 generation.  The rebuilt graph gets a
                # clean external search state; persistent spectrum memory is
                # keyed separately by its new baseline.
                rejected_spectrum_histogram_signatures.clear()
                rejected_spectrum_bundle_signatures.clear()
                forbidden_spectrum_inverse_signatures.clear()
                rejected_resource_conversion_signatures.clear()
                visited_state_signatures.add(
                    _stage3_physical_state_signature(
                        nodes_current,
                        H_current,
                        external_rebuild_context,
                    )
                )
                visited_chemical_state_signatures.add(
                    _stage3_chemical_state_signature(nodes_current, H_current)
                )
                accepted_cycles += 1
                _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                    cycle=cycle_number,
                    phase="external_topology",
                    applied=True,
                    stop_reason="external_generation_fresh_rebuild_accepted",
                    accepted_state_retained=True,
                    rebuild_valid=True,
                    layer2_r2=float(cand_r2),
                    accepted=True,
                    acceptance_reason="external_generation_fresh_rebuild_accepted",
                    fresh_rebuild=True,
                ))
                console_reporter.rebuild_done(
                    elapsed=time.perf_counter() - external_rebuild_started,
                    r2=float(cand_r2),
                    difference=diff_current,
                )
                generation_cycle_number = None
                continue
            complete_diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
            stop_reason = "stage3_complete"
            console_reporter.finish(
                boundary="stage3_complete",
                complete=True,
            )
            completion_snapshot = _stage3_runtime_snapshot(plan)
            # The runtime snapshot already merges the live assembler and the
            # iteration ledger.  Merge only genuinely new residual products;
            # no parallel product cache is allowed to replace this authority.
            snapshot_products = _merge_stage3_committed_products(
                list(completion_snapshot.get("products", []) or []),
                complete_diagnostics.get("committed_residual_products", []),
            )
            completion_snapshot.update({
                "stage3_complete": True,
                "products": list(snapshot_products),
                "generation_count": int(graph_generation - int(max(0, outer_cycle)) * 1_000_000 + 1),
                "internal_physical_conversion_count": int(
                    complete_diagnostics.get(
                        "internal_physical_conversion_count", 0
                    )
                    or 0
                ),
                "internal_ring_count": int(
                    complete_diagnostics.get("internal_ring_count", 0) or 0
                ),
                "remaining_resource_counts": {
                    str(key): len(list(values or []))
                    for key, values in dict(
                        complete_diagnostics.get("remaining_ids_by_su", {}) or {}
                    ).items()
                },
                "generation_metrics": list(generation_metrics),
                "external_primary_adjustments_total": int(
                    external_primary_adjustments_total
                ),
                "external_primary_adjustment_limit": int(
                    external_primary_limit
                ),
                "spectrum_rebalance_finalization": {
                    "enabled_by_config": bool(enable_post_external_spectrum),
                    "active_through_cycle": int(spectrum_active_through_cycle),
                    "finalized": bool(spectrum_rebalance_finalized),
                    "finalized_on_cycle": spectrum_finalized_on_cycle,
                    "reason": (
                        "cycle_window_completed_topology_only"
                        if spectrum_rebalance_finalized
                        else "stage3_completed_within_spectrum_window"
                        if enable_post_external_spectrum
                        else "disabled_by_config"
                    ),
                },
                "phase_result": dict(
                    _compact_stage3_plan_dict(plan).get("phase_result", {}) or {}
                ),
            })
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase="complete",
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                accepted=True,
                acceptance_reason=stop_reason,
            ))
            return Stage3WorkflowResult(
                nodes=nodes_current,
                H=H_current,
                diff=diff_current,
                ok=True,
                stop_reason=stop_reason,
                accepted_cycles=accepted_cycles,
                cycles=cycles,
                diagnostics={
                    "stage": "topology_resource_adjustment",
                    "terminal_state_signature": str(state_signature),
                    "visited_state_count": len(visited_state_signatures),
                    "retained_cycle_limit": int(_MAX_RETAINED_STAGE3_CYCLES),
                    "total_cycle_count": int(cycle_idx),
                    "total_planner_attempt_count": int(attempt_idx),
                    "discarded_cycle_count": max(
                        0,
                        int(cycle_idx)
                        - len({int(record.cycle) for record in cycles}),
                    ),
                    "discarded_planner_attempt_count": max(
                        0, int(attempt_idx) - len(cycles)
                    ),
                    "spectrum_rebalance_finalized": bool(
                        spectrum_rebalance_finalized
                    ),
                    "spectrum_finalized_on_cycle": spectrum_finalized_on_cycle,
                    "spectrum_active_through_cycle": int(
                        spectrum_active_through_cycle
                    ),
                    "completion_snapshot": completion_snapshot,
                    "tolerated_remaining_ids_by_su": dict(
                        complete_diagnostics.get(
                            "tolerated_remaining_ids_by_su", {}
                        )
                        or {}
                    ),
                    "tolerated_remaining_count": int(
                        complete_diagnostics.get(
                            "tolerated_remaining_count", 0
                        )
                        or 0
                    ),
                    **_h_soft_diagnostics(
                        H=H_current,
                        E_target=E_target,
                        h_tolerance=h_tolerance,
                        special_degree_meta=special_degree_meta,
                    ),
                },
            )
        if verbose:
            _print_stage3_histogram_adjustments(
                cycle_number,
                histogram_plan,
                verbose=True,
                attempt_number=attempt_number,
                emit_header=False,
            )
        apply_started = time.perf_counter()
        if verbose:
            print(f"[Block C:resource #{cycle_number}] histogram apply start", flush=True)
        apply_result = adjuster.apply_topology_resource_adjustments(
            nodes=list(nodes_current or []),
            H=H_current.detach().clone(),
            plan=plan,
        )
        if verbose:
            print(
                f"[Block C:resource #{cycle_number}] histogram apply done "
                f"({time.perf_counter() - apply_started:.2f}s)",
                flush=True,
            )
        requested_counts = dict(getattr(apply_result, "requested_counts", {}) or {})
        unresolved_counts = dict(getattr(apply_result, "unresolved_counts", {}) or {})
        applied_ops = list(getattr(apply_result, "applied_ops", []) or [])
        request_ops = list(getattr(apply_result, "request_ops", []) or [])
        rejected_ops = list(getattr(apply_result, "rejected_ops", []) or [])
        histogram_changed = not torch.equal(
            torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long(),
            torch.clamp(H_current.detach().cpu(), min=0).long(),
        )
        if not bool(getattr(apply_result, "changed", False)):
            rejection_reason = str(
                dict(rejected_ops[0] or {}).get("reason", "")
                if rejected_ops
                else ""
            )
            plan_blocked_reason = str(
                getattr(plan, "blocked_reason", "")
                or getattr(histogram_plan, "blocked_reason", "")
                or getattr(plan, "reason", "")
                or ""
            )
            if rejection_reason:
                stop_reason = f"histogram_apply_rejected:{rejection_reason}"
            elif list(getattr(histogram_plan, "groups", []) or []):
                stop_reason = "histogram_apply_no_effect"
            elif request_ops or unresolved_counts or str(getattr(plan, "reason", "")) == "phase_resource_infeasible":
                stop_reason = "phase_resource_infeasible"
            else:
                stop_reason = plan_blocked_reason or "histogram_adjustment_impossible"
            terminal_proof: Dict[str, Any] = {}
            if stop_reason == "phase_resource_infeasible":
                terminal_proof = _stage3_terminal_infeasibility_proof(plan)
                print(
                    f"[Block C:resource #{cycle_number}] infeasible "
                    f"phase={str(terminal_proof.get('phase', phase) or phase)} "
                    f"reason={str(terminal_proof.get('reason', stop_reason) or stop_reason)} "
                    f"missing={dict(terminal_proof.get('missing_counts', {}) or {})} "
                    f"failed_contract={terminal_proof.get('failed_contract', {})}",
                    flush=True,
                )
                _rollback_stage3_plan_transactions(
                    plan,
                    stop_reason,
                )
            elif rejection_reason or stop_reason == "histogram_apply_no_effect":
                _rollback_stage3_plan_transactions(plan, stop_reason)
            retry_signature = ""
            if rejection_reason or (
                stop_reason == "histogram_apply_no_effect"
                and list(getattr(histogram_plan, "groups", []) or [])
            ):
                retry_signature = _register_rejected_resource_candidate(
                    plan,
                    histogram_plan,
                    rejected_resource_conversion_signatures,
                )
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                blocked_reason=str(
                    terminal_proof.get("reason", "")
                    or getattr(plan, "blocked_reason", "")
                    or stop_reason
                ),
                accepted=False,
                acceptance_reason=stop_reason,
                primary_adjustment_count=int(
                    getattr(histogram_plan, "primary_adjustment_count", 0) or 0
                ),
                physical_conversion_count=int(
                    getattr(histogram_plan, "physical_conversion_count", 0) or 0
                ),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                parity_balance_count=_histogram_parity_balance_count(histogram_plan),
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion "
                    f"variant after histogram apply rejection: "
                    f"rejected={retry_signature}",
                    flush=True,
                )
                continue
            break

        next_histogram_signature = tuple(
            int(value)
            for value in torch.clamp(
                getattr(apply_result, "H").detach().cpu(), min=0
            ).long().tolist()
        )
        planned_primary_for_console = (
            _external_primary_conversion_count(histogram_plan)
            if external_plan_phase
            else int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0)
        )
        console_reporter.record_histogram(
            histogram_plan,
            external_budget_after=(
                min(
                    int(external_primary_limit),
                    int(external_primary_adjustments_used)
                    + int(planned_primary_for_console),
                )
                if external_plan_phase
                else None
            ),
            recovery=forced_recovery_plan,
        )

        if external_plan_phase and not external_budget_will_exhaust:
            # Ordinary conversions and ring recovery both continue the same
            # live assembler.  The pending contract is retried against the
            # synchronized allocator; completed products remain fixed.
            nodes_current = list(getattr(apply_result, "nodes", []) or nodes_current)
            H_current = torch.clamp(
                getattr(apply_result, "H").detach().clone(), min=0
            ).long()
            live_assembler = getattr(adjuster, "_stage3_external_assembler", None)
            if live_assembler is not None and callable(
                getattr(live_assembler, "sync_after_resource_adjustments", None)
            ):
                live_assembler.sync_after_resource_adjustments(
                    nodes_current,
                    H=H_current,
                    applied_ops=applied_ops,
                )
            applied_primary_count = _external_primary_conversion_count(
                histogram_plan
            )
            external_primary_adjustments_used += int(
                applied_primary_count
            )
            external_primary_adjustments_total += int(
                applied_primary_count
            )
            if external_primary_adjustments_used > int(external_primary_limit):
                external_primary_adjustments_used = int(external_primary_limit)
            # Any accepted external primary conversion marks this generation
            # for its single boundary rebuild.  Recovery is just one source of
            # such a conversion; it no longer has a separate immediate path.
            external_generation_rebuild_pending = bool(
                external_generation_rebuild_pending
                or int(applied_primary_count) > 0
            )
            progress_phase = Stage3ProgressPhase.EXTERNAL_TOPOLOGY
            allow_ring_recovery = False
            if bool(parity_diagnostics.get("balance_applied", False)):
                adjuster._stage3_parity_balance_cursor = int(
                    parity_diagnostics.get(
                        "next_cursor", int(parity_cursor_before) + 1
                    )
                )
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=(
                    "recovery_applied_continuation"
                    if forced_recovery_plan
                    else "external_conversion_applied_continuation"
                ),
                accepted_state_retained=True,
                rebuild_valid=False,
                accepted=True,
                acceptance_reason=(
                    "pure_resource_ring_recovered"
                    if forced_recovery_plan
                    else "external_conversion_applied"
                ),
                primary_adjustment_count=int(applied_primary_count),
                physical_conversion_count=int(
                    getattr(histogram_plan, "physical_conversion_count", 0) or 0
                ),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                parity_balance_count=_histogram_parity_balance_count(histogram_plan),
                fresh_rebuild=False,
            ))
            continue

        if rebuild_fn is None or validate_fn is None:
            stop_reason = "layer12_rebuild_invalid"
            _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                blocked_reason=stop_reason,
                accepted=False,
                acceptance_reason=stop_reason,
            ))
            break

        try:
            rebuild_started = time.perf_counter()
            plan_kind = str(getattr(plan, "plan_kind", "") or "")
            plan_reason = str(getattr(plan, "reason", "") or "")
            boundary = (
                "external_budget_exhausted"
                if "budget" in plan_kind
                else plan_reason or plan_kind or "rebuild_required"
            )
            console_reporter.finish(boundary=boundary)
            console_reporter.rebuild_start(boundary)
            H_seed = torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long()
            cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                H_seed=getattr(apply_result, "H"),
                phase=phase,
                cycle_idx=cycle_number - 1,
                nodes_seed=list(getattr(apply_result, "nodes", []) or []),
            )
            _clear_stage3_node_runtime_state(cand_nodes)
            rebuild_elapsed = time.perf_counter() - rebuild_started
            console_reporter.rebuild_done(
                elapsed=rebuild_elapsed,
                r2=float(cand_r2),
                difference=dict(cand_diff or {}),
            )
        except Exception as exc:
            stop_reason = "layer12_rebuild_invalid"
            rollback_reason = f"{stop_reason}:{exc}"
            print(
                f"[Block C:resource #{cycle_number}] fresh Layer1/2 rebuild failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(plan, rollback_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=False,
                blocked_reason=str(exc),
                accepted=False,
                acceptance_reason=stop_reason,
                primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
                physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                parity_balance_count=_histogram_parity_balance_count(histogram_plan),
                fresh_rebuild=True,
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after rebuild failure: rejected={retry_signature}",
                    flush=True,
                )
                continue
            break
        candidate_histogram = torch.clamp(cand_H.detach().cpu(), min=0).long()
        if not torch.equal(candidate_histogram, H_seed):
            valid_reason = "fresh_rebuild_histogram_mismatch"
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(plan, valid_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                layer2_r2=float(cand_r2),
                blocked_reason=valid_reason,
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                fresh_rebuild=True,
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after histogram mismatch: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break
        validate_started = time.perf_counter()
        if verbose:
            print(f"[Block C:resource #{cycle_number}] rebuilt state validation start", flush=True)
        valid, valid_reason = validate_fn(nodes=cand_nodes, H=cand_H, tag=f"BlockC:resource #{cycle_number}")
        if verbose:
            print(
                f"[Block C:resource #{cycle_number}] rebuilt state validation done "
                f"({time.perf_counter() - validate_started:.2f}s, valid={bool(valid)})",
                flush=True,
            )
        if not valid:
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(plan, str(valid_reason))
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                layer2_r2=float(cand_r2),
                blocked_reason=str(valid_reason),
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                fresh_rebuild=True,
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after validation failure: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break

        candidate_context = (
            deepcopy(context_snapshot_fn())
            if callable(context_snapshot_fn)
            else None
        )
        candidate_state_signature = _stage3_physical_state_signature(
            list(cand_nodes or []),
            cand_H,
            candidate_context,
        )
        candidate_chemical_state_signature = _stage3_chemical_state_signature(
            list(cand_nodes or []),
            cand_H,
        )
        if (
            candidate_state_signature in visited_state_signatures
            or candidate_chemical_state_signature
            in visited_chemical_state_signatures
        ):
            cycle_reason = "stage3_generation_cycle_detected"
            retry_signature = _register_rejected_resource_candidate(
                plan,
                histogram_plan,
                rejected_resource_conversion_signatures,
            )
            _rollback_stage3_plan_transactions(
                plan,
                cycle_reason,
            )
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            _restore_stage3_parity_cursor(adjuster, parity_cursor_before)
            if callable(context_restore_fn):
                context_restore_fn(deepcopy(cycle_context_before))
            if verbose:
                print(
                    f"[Block C:resource #{cycle_number}] full-state cycle rejected "
                    f"state={candidate_state_signature} "
                    f"chemical_state={candidate_chemical_state_signature} "
                    f"histogram={next_histogram_signature}",
                    flush=True,
                )
            _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=cycle_reason,
                accepted_state_retained=True,
                rebuild_valid=True,
                layer2_r2=float(cand_r2),
                accepted=False,
                acceptance_reason=cycle_reason,
                blocked_reason=cycle_reason,
                fresh_rebuild=True,
            ))
            if retry_signature and verbose:
                print(
                    f"[Block C:resource #{cycle_number}] retry next conversion variant "
                    f"after full-state cycle: rejected={retry_signature}",
                    flush=True,
                )
                continue
            stop_reason = cycle_reason
            break

        acceptance_reason = "valid_topology_histogram_fresh_rebuild"
        nodes_current = list(cand_nodes or [])
        H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
        diff_current = dict(cand_diff or {})
        if str(phase).startswith("external"):
            applied_external_primary_count = _external_primary_conversion_count(
                histogram_plan
            )
            external_primary_adjustments_used += int(
                applied_external_primary_count
            )
            external_primary_adjustments_total += int(
                applied_external_primary_count
            )
            if external_primary_adjustments_used > int(external_primary_limit):
                external_primary_adjustments_used = int(external_primary_limit)
        graph_generation += 1
        generation_metrics.append({
            "generation_id": int(graph_generation),
            "internal_physical_conversion_count": 0,
            "internal_ring_count": 0,
        })
        if len(generation_metrics) > _MAX_RETAINED_GENERATION_SIGNATURES:
            del generation_metrics[:-_MAX_RETAINED_GENERATION_SIGNATURES]
        # A rebuilt Layer1/2 graph invalidates every symbolic Stage3 product,
        # reservation and ring-choice cursor from the previous graph.
        progress_phase = Stage3ProgressPhase.FLEX_RECONCILIATION
        external_primary_adjustments_used = 0
        external_generation_rebuild_pending = False
        adjuster._stage3_external_assembler = None
        _reset_stage3_generation_cursors(adjuster)
        rejected_spectrum_histogram_signatures.clear()
        rejected_spectrum_bundle_signatures.clear()
        forbidden_spectrum_inverse_signatures.clear()
        rejected_resource_conversion_signatures.clear()
        # Candidate signatures are generation-local; a fresh graph must be
        # audited and planned without any blacklist from the previous graph.
        visited_state_signatures.add(candidate_state_signature)
        if len(visited_state_signatures) > _MAX_RETAINED_GENERATION_SIGNATURES:
            visited_state_signatures.pop()
        visited_chemical_state_signatures.add(candidate_chemical_state_signature)
        if len(visited_chemical_state_signatures) > _MAX_RETAINED_GENERATION_SIGNATURES:
            visited_chemical_state_signatures.pop()
        accepted_cycles += 1
        _append_stage3_cycle(cycles, Stage3WorkflowCycleRecord(
            cycle=cycle_number,
            phase=phase,
            applied=True,
            stop_reason="accepted",
            accepted_state_retained=True,
            rebuild_valid=True,
            layer2_r2=float(cand_r2),
            accepted=True,
            acceptance_reason=acceptance_reason,
            primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
            physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
            h_compensation_count=_histogram_h_compensation_count(histogram_plan),
            parity_balance_count=_histogram_parity_balance_count(histogram_plan),
            fresh_rebuild=True,
        ))
        generation_cycle_number = None

    console_reporter.finish(boundary=str(stop_reason), complete=False)
    terminal_context = (
        deepcopy(context_snapshot_fn())
        if callable(context_snapshot_fn)
        else None
    )
    terminal_state_signature = _stage3_physical_state_signature(
        nodes_current,
        H_current,
        terminal_context,
    )
    terminal_special_degree_meta = (
        adjuster._get_special_degree_meta(H_current)
        if callable(getattr(adjuster, "_get_special_degree_meta", None))
        else None
    )
    terminal_proof = _stage3_terminal_infeasibility_proof(plan)
    terminal_proof["external_budget"] = {
        "used": int(external_primary_adjustments_used),
        "limit": int(external_primary_limit),
        "remaining": max(
            0,
            int(external_primary_limit)
            - int(external_primary_adjustments_used),
        ),
    }
    terminal_runtime = _stage3_runtime_snapshot(plan)
    terminal_products = list(
        terminal_runtime.get("products", [])
        or []
    )
    terminal_pending = list(
        terminal_runtime.get("pending_contracts", [])
        or getattr(plan, "pending_contracts", [])
        or []
    )
    terminal_snapshot = {
        **terminal_runtime,
        "stage3_complete": False,
        "node_ids": sorted({
            int(getattr(node, "global_id", -1))
            for node in list(nodes_current or [])
            if int(getattr(node, "global_id", -1)) >= 0
        }),
        "generation_id": _stage3_runtime_generation_id(plan),
        "stop_reason": str(stop_reason),
        "phase": str(getattr(plan, "phase", "") or "unknown"),
        "infeasibility_proof": terminal_proof,
        "products": [dict(item or {}) for item in terminal_products],
        "pending_contracts": [dict(item or {}) for item in terminal_pending],
        "plan": _compact_stage3_plan_dict(plan),
    }
    return Stage3WorkflowResult(
        nodes=nodes_current,
        H=H_current,
        diff=diff_current,
        ok=stop_reason == "stage3_complete",
        stop_reason=stop_reason,
        accepted_cycles=accepted_cycles,
        cycles=cycles,
        diagnostics={
            "stage": "topology_resource_adjustment",
            "output_dir": output_dir,
            "terminal_state_signature": str(terminal_state_signature),
            "visited_state_count": len(visited_state_signatures),
            "visited_chemical_state_count": len(visited_chemical_state_signatures),
            "retained_cycle_limit": int(_MAX_RETAINED_STAGE3_CYCLES),
            "total_cycle_count": int(cycle_idx),
            "total_planner_attempt_count": int(attempt_idx),
            "discarded_cycle_count": max(
                0,
                int(cycle_idx) - len({int(record.cycle) for record in cycles}),
            ),
            "discarded_planner_attempt_count": max(
                0, int(attempt_idx) - len(cycles)
            ),
            "spectrum_rebalance_finalized": bool(spectrum_rebalance_finalized),
            "spectrum_finalized_on_cycle": spectrum_finalized_on_cycle,
            "spectrum_active_through_cycle": int(spectrum_active_through_cycle),
            "external_primary_adjustments_total": int(
                external_primary_adjustments_total
            ),
            "external_primary_adjustment_limit": int(external_primary_limit),
            "terminal_snapshot": terminal_snapshot,
            **_h_soft_diagnostics(
                H=H_current,
                E_target=E_target,
                h_tolerance=h_tolerance,
                special_degree_meta=terminal_special_degree_meta,
            ),
        },
    )
