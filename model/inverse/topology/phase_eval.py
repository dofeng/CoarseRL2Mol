from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...shared.inverse_common import _NodeV3
from .classification.classifier import classify_topology_units
from .classification.classification_models import TopologyClassificationResult
from .flex_audit import (
    _compute_flex_audit,
    _flex_bounds_from_audit,
    _flex_precheck_required,
)
from .round_context import build_stage3_round_context, ownership_public_view
from .stage3_models import (
    Stage3FlexDecision,
    Stage3PhaseEvaluation,
    TopologyAssemblyResult,
)


PHASE_SEQUENCE = (
    "flex_range_reconciliation",
    "external_flex_closure",
    "external_fatty_ring_closure",
    "external_side_chain_closure",
    "remaining_pool_spectrum_rebalance",
    "resource_su24_ring_closure",
    "residual_su24_aromatization",
    "residual_resource_self_closure",
    "complete",
)

EXTERNAL_SUBPHASE_ORDER = {
    "flex_range_reconciliation": 0,
    "external_flex_closure": 1,
    "external_fatty_ring_closure": 2,
    "external_side_chain_closure": 3,
    "remaining_pool_spectrum_rebalance": 4,
    "resource_su24_ring_closure": 5,
    "residual_su24_aromatization": 6,
    "residual_resource_self_closure": 7,
    "complete": 8,
}


def build_stage3_topology_metrics(
    classification: Any,
    assembly: Any,
    flex_decision: Any,
) -> Dict[str, Any]:
    classification_dict = classification.to_dict() if hasattr(classification, "to_dict") else dict(classification or {})
    assembly_dict = assembly.to_dict() if hasattr(assembly, "to_dict") else dict(assembly or {})
    decision_dict = flex_decision.to_dict() if hasattr(flex_decision, "to_dict") else dict(flex_decision or {})
    diagnostics = dict(assembly_dict.get("diagnostics", {}) or {})
    ledger = dict(assembly_dict.get("resource_ledger", {}) or {})
    shortage = {
        str(key): int(value)
        for key, value in dict(ledger.get("shortage_counts", {}) or {}).items()
        if int(value) > 0
    }
    unresolved = {
        str(key): int(value)
        for key, value in dict(assembly_dict.get("unresolved_counts", shortage) or {}).items()
        if int(value) > 0
    }
    flex_counts = dict(assembly_dict.get("flex_counts", {}) or {})
    flex_bounds = dict(assembly_dict.get("flex_bounds", {}) or {})
    effective = int(
        flex_counts.get("final", decision_dict.get("effective_flex_count", 0)) or 0
    )
    lower = int(flex_bounds.get("min", decision_dict.get("lower_bound", 0)) or 0)
    upper = int(flex_bounds.get("max", decision_dict.get("upper_bound", 0)) or 0)
    flex_gap = (
        max(0, int(lower) - int(effective), int(effective) - int(upper))
        if bool(flex_bounds.get("enforced", False))
        else 0
    )
    active_subphase = str(
        assembly_dict.get("active_phase")
        or assembly_dict.get("active_subphase")
        or diagnostics.get("active_external_subphase")
        or diagnostics.get("stage3_phase")
        or "flex_range_reconciliation"
    )
    products = list(assembly_dict.get("products", []) or [])
    return {
        "active_subphase": active_subphase,
        "subphase_rank": int(EXTERNAL_SUBPHASE_ORDER.get(active_subphase, 0)),
        "product_count": int(len(products)),
        "product_types": sorted(str(dict(item or {}).get("product_type", "")) for item in products),
        "pending_count": int(len(list(assembly_dict.get("pending_units", []) or []))),
        "shortage_total": int(sum(shortage.values())),
        "unresolved_total": int(sum(unresolved.values())),
        "flex_gap": int(flex_gap),
        "unclassified_count": int(len(list(classification_dict.get("unclassified_nodes", []) or []))),
        "reject_count": int(len(list(classification_dict.get("classification_rejects", []) or []))),
        "consumed_counts": dict(ledger.get("consumed_counts", {}) or {}),
        "remaining_counts": dict(ledger.get("remaining_counts", {}) or {}),
    }


def build_stage3_topology_signature(
    classification: Any,
    assembly: Any,
    flex_decision: Any,
    H: Any = None,
) -> str:
    classification_dict = classification.to_dict() if hasattr(classification, "to_dict") else dict(classification or {})
    assembly_dict = assembly.to_dict() if hasattr(assembly, "to_dict") else dict(assembly or {})
    decision_dict = flex_decision.to_dict() if hasattr(flex_decision, "to_dict") else dict(flex_decision or {})
    histogram = []
    if H is not None:
        histogram = [int(value) for value in H.detach().cpu().long().tolist()]
    payload = {
        "fragments": sorted(
            (
                str(dict(item or {}).get("class_type", "")),
                tuple(sorted(int(value) for value in list(dict(item or {}).get("claimed_node_ids", []) or []))),
            )
            for item in list(classification_dict.get("fragments", []) or [])
        ),
        "flex_decision": decision_dict,
        "active_phase": str(assembly_dict.get("active_phase", "") or dict(assembly_dict.get("diagnostics", {}) or {}).get("stage3_phase", "")),
        "active_subphase": str(assembly_dict.get("active_subphase", "") or dict(assembly_dict.get("diagnostics", {}) or {}).get("active_external_subphase", "")),
        "products": list(assembly_dict.get("products", []) or []),
        "pending_units": list(assembly_dict.get("pending_units", []) or []),
        "resource_ledger": dict(assembly_dict.get("resource_ledger", {}) or {}),
        "requested_counts": dict(assembly_dict.get("requested_counts", {}) or {}),
        "unresolved_counts": dict(assembly_dict.get("unresolved_counts", {}) or {}),
        "H": histogram,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _minimal_resource_ledger_from_classification(classification_dict: Dict[str, Any]) -> Dict[str, Any]:
    resource_pool = dict(dict(classification_dict or {}).get("resource_pool", {}) or {})
    remaining_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("remaining_ids_by_su", {}) or {}).items()
    }
    initial_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("initial_ids_by_su", remaining_ids_by_su) or {}).items()
    }
    all_keys = sorted({
        str(key)
        for key in set(initial_ids_by_su.keys()) | set(remaining_ids_by_su.keys())
    }, key=int)
    initial_ids_by_su = {
        str(key): [int(value) for value in list(initial_ids_by_su.get(str(key), []) or [])]
        for key in all_keys
    }
    remaining_ids_by_su = {
        str(key): [int(value) for value in list(remaining_ids_by_su.get(str(key), []) or [])]
        for key in all_keys
    }
    consumed_ids_by_su = {str(key): [] for key in all_keys}
    initial_counts = {str(key): int(len(list(values or []))) for key, values in initial_ids_by_su.items()}
    remaining_counts = {str(key): int(len(list(values or []))) for key, values in remaining_ids_by_su.items()}
    consumed_counts = {str(key): 0 for key in all_keys}
    su11_donor_ids_by_su = {
        str(key): [int(value) for value in list(values or [])]
        for key, values in dict(resource_pool.get("su11_donor_ids_by_su", {}) or {}).items()
    }
    return {
        "initial_ids_by_su": initial_ids_by_su,
        "remaining_ids_by_su": remaining_ids_by_su,
        "consumed_ids_by_su": consumed_ids_by_su,
        "initial_counts": initial_counts,
        "remaining_counts": remaining_counts,
        "consumed_counts": consumed_counts,
        "shortage_counts": {},
        "su11_donor_ids_by_su": su11_donor_ids_by_su,
        "su11_donor_counts": {
            str(key): int(len(values or []))
            for key, values in su11_donor_ids_by_su.items()
        },
    }


def _preassembly_pending_units(
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    reason = str(blocked_reason or ("flex_range_reconciliation_pending" if scenario else "flex_range_reconciliation"))
    hard_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
    }
    supplemental_ids = {
        int(value)
        for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
    }
    pending_ids: List[int] = []
    pending_kind_by_id: Dict[int, str] = {}
    if str(blocked_reason) == "flex_window_invalid":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
        for fragment_id in sorted(supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = "supplemental_flex_candidate"
            pending_ids.append(int(fragment_id))
    elif str(scenario) == "below_range":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
        for fragment_id in sorted(supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = "supplemental_flex_candidate"
            pending_ids.append(int(fragment_id))
    elif str(scenario) == "above_range":
        for fragment_id in sorted(hard_ids):
            pending_kind_by_id[int(fragment_id)] = "explicit_flex"
            pending_ids.append(int(fragment_id))
    else:
        for fragment_id in sorted(hard_ids | supplemental_ids):
            pending_kind_by_id[int(fragment_id)] = (
                "explicit_flex" if int(fragment_id) in hard_ids else "supplemental_flex_candidate"
            )
            pending_ids.append(int(fragment_id))
    fragments_by_id = {
        int(getattr(fragment, "fragment_id", -1)): fragment
        for fragment in list(getattr(classification, "fragments", []) or [])
    }
    pending_units: List[Dict[str, Any]] = []
    for fragment_id in pending_ids:
        fragment = fragments_by_id.get(int(fragment_id))
        if fragment is None:
            continue
        class_type = str(getattr(fragment, "class_type", "") or "")
        pending_kind = str(pending_kind_by_id.get(int(fragment_id), "supplemental_flex_candidate"))
        pending_units.append({
            "unit_id": int(fragment_id),
            "pending_kind": str(pending_kind),
            "real_topology_partial": str(getattr(fragment, "topology_template", "") or class_type),
            "real_node_partial": "",
            "source_fragment_ids": [int(fragment_id)],
            "source_fragment_classes": [str(class_type)],
            "source_node_ids": [int(value) for value in list(getattr(fragment, "source_node_ids", []) or [])],
            "reason": str(reason),
        })
    return pending_units


def _flex_reconciliation_contracts(
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    explicit_count = int(
        dict(flex_audit or {}).get("explicit_flex_count", 0) or 0
    )
    contracts: List[Dict[str, Any]] = []
    if str(blocked_reason) == "flex_window_invalid":
        contracts.append({
            "contract_id": -1000,
            "contract_kind": "structural_blocker",
            "target_role": "flex_chain",
            "source_fragment_ids": [],
            "source_fragment_classes": [],
            "claimed_source_node_ids": [],
            "context_node_ids": [],
            "mainline_slots": [],
            "branch_slots": [],
            "required_counts": {},
            "expected_topology": "",
            "reason": "flex_window_invalid",
            "blocker_kind": "flex_window_invalid",
            "diagnostics": {
                "scenario": str(scenario or "in_range"),
                "X": int(dict(flex_audit or {}).get("X", 0) or 0),
                "Y": int(dict(flex_audit or {}).get("Y", 0) or 0),
                "Z": int(dict(flex_audit or {}).get("Z", 0) or 0),
                "flex_lower": int(dict(flex_audit or {}).get("flex_lower", 0) or 0),
                "flex_upper": int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
            },
        })
    if scenario == "below_range":
        fragments_by_id = {
            int(getattr(fragment, "fragment_id", -1)): fragment
            for fragment in list(getattr(classification, "fragments", []) or [])
        }
        candidate_profiles = [
            dict(item or {})
            for item in list(dict(flex_audit or {}).get("supplemental_flex_candidate_profiles", []) or [])
            if dict(item or {})
        ]
        selected_groups = [
            dict(item or {})
            for item in list(dict(flex_audit or {}).get("selected_candidate_groups", []) or [])
            if dict(item or {})
        ]
        selected_group_by_fragment = {
            int(fragment_id): group
            for group in selected_groups
            for fragment_id in list(group.get("fragment_ids", []) or [])
        }
        emitted_group_ids: set[str] = set()
        for item in list(candidate_profiles or []):
            fragment_id = int(item.get("fragment_id", -1))
            if fragment_id < 0:
                continue
            fragment = fragments_by_id.get(int(fragment_id))
            selected_group = selected_group_by_fragment.get(int(fragment_id))
            selected_for_target = selected_group is not None
            if not selected_for_target:
                continue
            if selected_for_target:
                group_id = str(selected_group.get("group_id", "") or "")
                if group_id in emitted_group_ids:
                    continue
                emitted_group_ids.add(group_id)
                group_fragment_ids = [
                    int(value) for value in list(selected_group.get("fragment_ids", []) or [])
                ]
                group_fragments = [
                    fragments_by_id[value]
                    for value in group_fragment_ids
                    if value in fragments_by_id
                ]
                required_counts = {
                    str(key): int(value)
                    for key, value in dict(selected_group.get("resource_template", {}) or {}).items()
                    if int(value) > 0
                }
                is_composition = str(selected_group.get("strategy", "")) == "b_plus_b"
                source_fragment_ids = list(group_fragment_ids)
                source_fragment_classes = [
                    str(getattr(value, "class_type", "")) for value in group_fragments
                ]
                claimed_source_node_ids = sorted({
                    int(value)
                    for group_fragment in group_fragments
                    for value in list(getattr(group_fragment, "claimed_node_ids", []) or [])
                })
                context_node_ids = sorted({
                    int(value)
                    for group_fragment in group_fragments
                    for value in list(getattr(group_fragment, "context_node_ids", []) or [])
                })
                mainline_slots = [] if is_composition else [
                    slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
                    for group_fragment in group_fragments
                    for slot in list(getattr(group_fragment, "open_mainline_slots", []) or [])
                ]
                branch_slots = [
                    slot.to_dict() if hasattr(slot, "to_dict") else dict(slot or {})
                    for group_fragment in group_fragments
                    for slot in list(getattr(group_fragment, "branch_contracts", []) or [])
                ]
                contract_kind = "fragment_composition" if is_composition else "mainline_closure"
                expected_topology = "+".join(
                    str(getattr(value, "topology_template", "") or "")
                    for value in group_fragments
                )
                reason = "supplemental_flex_candidate"
                blocker_kind = ""
            contracts.append({
                "contract_id": min(source_fragment_ids or [int(fragment_id)]),
                "contract_kind": str(contract_kind),
                "target_role": "flex_chain",
                "source_fragment_ids": list(source_fragment_ids),
                "source_fragment_classes": list(source_fragment_classes),
                "claimed_source_node_ids": list(claimed_source_node_ids),
                "context_node_ids": list(context_node_ids),
                "mainline_slots": list(mainline_slots),
                "branch_slots": list(branch_slots),
                "required_counts": dict(required_counts),
                "expected_topology": str(expected_topology),
                "reason": str(reason),
                "blocker_kind": str(blocker_kind),
                "diagnostics": {
                    "scenario": str(scenario),
                    "candidate_main_kind": str(item.get("main_kind", "")),
                    "candidate_count_role": str(item.get("count_role", "")),
                    "candidate_kind": str(item.get("candidate_kind", "")),
                    "candidate_priority": int(item.get("priority", 99) or 99),
                    "selected_for_target": bool(selected_for_target),
                    "selection_group": dict(selected_group or {}),
                },
            })
        return contracts
    if scenario == "above_range":
        contracts.append({
            "contract_id": -1002,
            "contract_kind": "structural_blocker",
            "target_role": "flex_chain",
            "source_fragment_ids": [],
            "source_fragment_classes": [],
            "claimed_source_node_ids": [],
            "context_node_ids": [],
            "mainline_slots": [],
            "branch_slots": [],
            "required_counts": {},
            "expected_topology": "",
            "reason": "explicit_flex_reduction",
            "blocker_kind": "explicit_flex_reduction",
            "diagnostics": {
                "scenario": str(scenario),
                "explicit_count": int(explicit_count),
                "flex_upper": int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
                "flex_excess": int(max(0, int(explicit_count) - int(dict(flex_audit or {}).get("flex_upper", 0) or 0))),
                "scored_conversion_candidates": ["12_to_11", "10_pair_to_11_pair"],
            },
        })
        return contracts
    return contracts


def _preassembly_gate_assembly_dict(
    classification: TopologyClassificationResult,
    classification_dict: Dict[str, Any],
    flex_audit: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = str(dict(flex_audit or {}).get("scenario", "") or "")
    blocked_reason = str(dict(flex_audit or {}).get("blocked_reason", "") or "")
    if scenario == "above_range" and str(blocked_reason) != "flex_window_invalid":
        result_reason = "external_flex_range_exceeded"
    else:
        result_reason = "external_flex_range_pending"
    pending_units = _preassembly_pending_units(classification, flex_audit)
    flex_contracts = _flex_reconciliation_contracts(classification, flex_audit)
    contract_counts: Dict[str, int] = {}
    for contract in list(flex_contracts or []):
        required_counts = dict(contract.get("required_counts", {}) or {})
        for key, value in required_counts.items():
            contract_counts[str(key)] = int(contract_counts.get(str(key), 0) or 0) + int(value or 0)
    return {
        "ok": False,
        "reason": str(result_reason),
        "feasible": bool(
            len(list(getattr(classification, "unclassified_nodes", []) or [])) == 0
            and len(list(getattr(classification, "claim_conflicts", []) or [])) == 0
            and len(list(getattr(classification, "classification_rejects", []) or [])) == 0
        ),
        "products": [],
        "pending_units": list(pending_units),
        "flex_bounds": _flex_bounds_from_audit(flex_audit),
        "flex_counts": {
            "explicit_total": int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
            "final": 0,
        },
        "resource_ledger": _minimal_resource_ledger_from_classification(classification_dict),
        "diagnostics": {
            "stage3_phase": "flex_range_reconciliation",
            "external_subphases": [],
            "active_external_subphase": "flex_range_reconciliation",
            "external_subphase_status": {
                "explicit_branch_closure": {"ok": True},
                "real_like24_ring_assembly": {"ok": True},
                "remaining_external_halfclosed_closure": {"ok": True},
            },
            "explicit_flex_count": int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
            "flex_audit": dict(flex_audit or {}),
            "pending_contracts": list(flex_contracts),
            "total_shortage": dict(contract_counts or {}),
        },
        "planned_conversions": [],
    }


def _assemble_external_only(
    *,
    nodes: List[_NodeV3],
    classification: TopologyClassificationResult,
    flex_audit: Dict[str, Any],
    S_target: Any = None,
    E_target: Any = None,
    generation_id: int = 0,
) -> TopologyAssemblyResult:
    from .assembly_external import assemble_external_topologies

    return assemble_external_topologies(
        nodes=list(nodes or []),
        classification=classification,
        S_target=S_target,
        E_target=E_target,
        flex_audit=flex_audit,
        generation_id=int(generation_id),
    )


def assemble_topology_units(
    nodes: List[_NodeV3],
    classification: TopologyClassificationResult,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    flex_audit: Optional[Dict[str, Any]] = None,
    generation_id: int = 0,
) -> TopologyAssemblyResult:
    del fixed_partition_meta
    classification_dict = classification.to_dict()
    flex_audit = dict(flex_audit or {}) or _compute_flex_audit(
        list(nodes or []), classification, enforce=True
    )
    decision = Stage3FlexDecision(
        scenario=str(flex_audit.get("scenario", "in_range") or "in_range"),
        cluster_count=int(flex_audit.get("X", 0) or 0),
        su10_pair_count=int(flex_audit.get("Y", 0) or 0),
        effective_flex_count=int(flex_audit.get("explicit_flex_count", 0) or 0),
        lower_bound=int(flex_audit.get("flex_lower", 0) or 0),
        upper_bound=int(flex_audit.get("flex_upper", 0) or 0),
        target_count=int(flex_audit.get("target_flex_count", 0) or 0),
        selected_candidate_fragment_ids=list(flex_audit.get("selected_candidate_fragment_ids", []) or []),
        selected_candidate_node_ids=list(flex_audit.get("selected_candidate_node_ids", []) or []),
        required_resource_counts=dict(flex_audit.get("required_resource_counts", {}) or {}),
        unresolved_resource_counts=dict(flex_audit.get("unresolved_resource_counts", {}) or {}),
        blocked_reason=str(flex_audit.get("blocked_reason", "") or ""),
    )
    if _flex_precheck_required(flex_audit):
        result = TopologyAssemblyResult(**_preassembly_gate_assembly_dict(
            classification,
            classification_dict,
            flex_audit,
        ))
    else:
        result = _assemble_external_only(
            nodes=list(nodes or []),
            classification=classification,
            S_target=S_target,
            E_target=E_target,
            flex_audit=flex_audit,
            generation_id=int(generation_id),
        )
    assembled_flex_counts = dict(getattr(result, "flex_counts", {}) or {})
    assembled_flex_bounds = dict(getattr(result, "flex_bounds", {}) or {})
    effective_flex_count = int(
        assembled_flex_counts.get(
            "final", flex_audit.get("explicit_flex_count", 0)
        )
        or 0
    )
    lower_bound = int(
        assembled_flex_bounds.get("min", flex_audit.get("flex_lower", 0)) or 0
    )
    upper_bound = int(
        assembled_flex_bounds.get("max", flex_audit.get("flex_upper", 0)) or 0
    )
    if not bool(flex_audit.get("enforced", False)):
        assembled_scenario = "in_range"
    elif int(effective_flex_count) < int(lower_bound):
        assembled_scenario = "below_range"
    elif int(effective_flex_count) > int(upper_bound):
        assembled_scenario = "above_range"
    else:
        assembled_scenario = "in_range"
    decision.scenario = str(assembled_scenario)
    decision.effective_flex_count = int(effective_flex_count)
    decision.lower_bound = int(lower_bound)
    decision.upper_bound = int(upper_bound)
    result.flex_decision = decision.to_dict()
    resource_demand = dict(getattr(result, "resource_demand", {}) or {})
    result.requested_counts = {
        str(key): int(value)
        for key, value in dict(
            resource_demand.get("hard_required_counts", {})
            or decision.required_resource_counts
            or {}
        ).items()
        if int(value) > 0
    }
    ledger = result.resource_ledger.to_dict() if hasattr(result.resource_ledger, "to_dict") else dict(result.resource_ledger or {})
    shortages = {
        str(key): int(value)
        for key, value in dict(ledger.get("shortage_counts", {}) or {}).items()
        if int(value) > 0
    }
    result.unresolved_counts = dict(shortages or decision.unresolved_resource_counts)
    result.blocked_reason = str(decision.blocked_reason or ("external_resource_shortage" if shortages else ""))
    result.active_subphase = str(
        dict(result.diagnostics or {}).get("active_external_subphase", "")
        or dict(result.diagnostics or {}).get("stage3_phase", "")
        or ("complete" if result.ok else "flex_range_reconciliation")
    )
    result.active_phase = str(
        getattr(result, "active_phase", "")
        or dict(result.diagnostics or {}).get("stage3_phase", "")
        or result.active_subphase
    )
    result.topology_metrics = build_stage3_topology_metrics(classification, result, decision)
    result.topology_signature = build_stage3_topology_signature(classification, result, decision)
    result.phase_signature = str(result.topology_signature)
    return result


def build_stage3_phase_evaluation(
    nodes: List[_NodeV3],
    classification: Optional[TopologyClassificationResult] = None,
    fixed_partition_meta: Optional[Dict[str, Any]] = None,
    S_target: Any = None,
    E_target: Any = None,
    generation_id: int = 0,
) -> Stage3PhaseEvaluation:
    del S_target, E_target
    classification_obj = classification or classify_topology_units(
        list(nodes or []),
        fixed_partition_meta=fixed_partition_meta,
    )
    flex_audit = _compute_flex_audit(
        list(nodes or []),
        classification_obj,
        enforce=True,
    )
    classification_dict = classification_obj.to_dict()
    round_context = build_stage3_round_context(
        nodes=list(nodes or []),
        classification=classification_obj,
        flex_audit=flex_audit,
        generation_id=int(generation_id),
    )
    ownership_view = ownership_public_view(round_context)
    flex_bounds = _flex_bounds_from_audit(flex_audit)
    explicit_count = int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0)
    flex_counts = {
        "explicit_total": int(explicit_count),
        "final": int(explicit_count),
    }
    flex_enforced = bool(flex_audit.get("enforced", False))
    flex_reconciliation_required = bool(_flex_precheck_required(flex_audit))
    flex_contracts = _flex_reconciliation_contracts(classification_obj, flex_audit)
    phase = "flex_range_reconciliation" if flex_reconciliation_required else "external_flex_closure"
    minimal_ledger = _minimal_resource_ledger_from_classification(classification_dict)
    minimal_ledger["ownership"] = ownership_view
    assembly_dict = {
        "ok": False,
        "reason": "stage3_round_not_executed",
        "products": [],
        "pending_units": [],
        "resource_ledger": minimal_ledger,
        "flex_bounds": dict(flex_bounds),
        "flex_counts": dict(flex_counts),
        "active_phase": str(phase),
        "active_subphase": str(phase),
        "completed_phases": [],
        "diagnostics": {
            "stage3_phase": str(phase),
            "evaluation_only": True,
            "pending_contracts": list(flex_contracts or []),
        },
    }

    diagnostics = {
        "phase_sequence": list(PHASE_SEQUENCE),
        "phase_status": {
            "classification": True,
            "ownership_partition": not bool(ownership_view.get("invariant_errors", [])),
            "flex_range_reconciliation": not bool(flex_reconciliation_required),
        },
        "resource_pool_remaining_counts": {
            str(key): len(values)
            for key, values in dict(round_context.closure_pool_ids_by_su or {}).items()
        },
        "unclassified_non_resource_count": int(
            dict(classification_dict.get("summary", {}) or {}).get("unclassified_non_resource_count", 0) or 0
        ),
        "claim_conflict_count": int(dict(classification_dict.get("summary", {}) or {}).get("claim_conflict_count", 0) or 0),
        "support_units_only_count": int(dict(classification_dict.get("summary", {}) or {}).get("support_units_only_count", 0) or 0),
        "pending_unit_count": int(len(flex_contracts)),
        "reject_count": int(dict(classification_dict.get("summary", {}) or {}).get("reject_count", 0) or 0),
        "open_fragment_count": int(dict(classification_dict.get("summary", {}) or {}).get("open_fragment_count", 0) or 0),
        "flex_audit": dict(flex_audit or {}),
        "classification_ready": True,
        "assembly_attempted": False,
        "evaluation_only": True,
        "flex_enforced": bool(flex_enforced),
        "flex_reconciliation_required": bool(flex_reconciliation_required),
        "flex_reconciliation_scenario": str(dict(flex_audit or {}).get("scenario", "") or ""),
        "flex_reconciliation_block_reason": str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        "flex_reconciliation_contracts": list(flex_contracts or []),
        "ownership_ledger": ownership_view,
        "generation_id": int(generation_id),
    }
    flex_decision = Stage3FlexDecision(
        scenario=str(dict(flex_audit or {}).get("scenario", "in_range") or "in_range"),
        cluster_count=int(dict(flex_audit or {}).get("X", 0) or 0),
        su10_pair_count=int(dict(flex_audit or {}).get("Y", 0) or 0),
        effective_flex_count=int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
        lower_bound=int(dict(flex_audit or {}).get("flex_lower", 0) or 0),
        upper_bound=int(dict(flex_audit or {}).get("flex_upper", 0) or 0),
        target_count=int(dict(flex_audit or {}).get("target_flex_count", 0) or 0),
        selected_candidate_fragment_ids=[
            int(value) for value in list(dict(flex_audit or {}).get("selected_candidate_fragment_ids", []) or [])
        ],
        selected_candidate_node_ids=[
            int(value) for value in list(dict(flex_audit or {}).get("selected_candidate_node_ids", []) or [])
        ],
        required_resource_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("required_resource_counts", {}) or {}).items()
        },
        unresolved_resource_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("unresolved_resource_counts", {}) or {}).items()
        },
        blocked_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
    )
    topology_metrics = build_stage3_topology_metrics(classification_obj, assembly_dict, flex_decision)
    topology_signature = build_stage3_topology_signature(classification_obj, assembly_dict, flex_decision)
    return Stage3PhaseEvaluation(
        generation_id=int(generation_id),
        phase=str(phase),
        done=False,
        classification_ready=True,
        assembly_attempted=False,
        flex_range_reconciliation_ok=not bool(flex_reconciliation_required),
        external_closure_ok=False,
        flex_reconciliation_required=bool(flex_reconciliation_required),
        flex_reconciliation_scenario=str(dict(flex_audit or {}).get("scenario", "") or ""),
        flex_reconciliation_block_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        flex_bounds=dict(flex_bounds or {}),
        flex_counts=dict(flex_counts or {}),
        flex_audit=dict(flex_audit or {}),
        flex_range_case=str(dict(flex_audit or {}).get("scenario", "") or ""),
        flex_target_count=int(dict(flex_audit or {}).get("target_flex_count", 0) or 0),
        explicit_flex_count=int(dict(flex_audit or {}).get("explicit_flex_count", 0) or 0),
        explicit_flex_fragment_ids=[
            int(value)
            for value in list(dict(flex_audit or {}).get("hard_explicit_flex_fragment_ids", []) or [])
        ],
        supplemental_flex_candidate_ids=[
            int(value)
            for value in list(dict(flex_audit or {}).get("supplemental_flex_candidate_ids", []) or [])
        ],
        supplemental_flex_candidate_profiles=[
            dict(item or {})
            for item in list(dict(flex_audit or {}).get("supplemental_flex_candidate_profiles", []) or [])
            if dict(item or {})
        ],
        flex_reconciliation_contracts=list(flex_contracts or []),
        canonical_contracts=list(flex_contracts or []),
        flex_range_action_class=str(dict(flex_audit or {}).get("scenario", "") or "in_range"),
        active_external_subphase="external_flex_closure" if not flex_reconciliation_required else "",
        assembly_view=assembly_dict,
        classification_view=classification_dict,
        diagnostics=diagnostics,
        flex_decision=flex_decision,
        topology_metrics=topology_metrics,
        topology_signature=topology_signature,
        requested_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("required_resource_counts", {}) or {}).items()
        },
        unresolved_counts={
            str(key): int(value) for key, value in dict(dict(flex_audit or {}).get("unresolved_resource_counts", {}) or {}).items()
        },
        blocked_reason=str(dict(flex_audit or {}).get("blocked_reason", "") or ""),
        active_phase=str(phase),
        completed_phases=[],
        phase_signature=str(topology_signature),
        round_context=round_context,
        ownership_ledger=ownership_view,
        closure_pool_ids_by_su=deepcopy(dict(round_context.closure_pool_ids_by_su or {})),
        aromatic_donor_ids_by_su=deepcopy(dict(round_context.aromatic_donor_ids_by_su or {})),
    )
