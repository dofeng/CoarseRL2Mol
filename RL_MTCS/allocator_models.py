from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pandas is only required for csv_path mode.
    pd = None

MAX_23_PER_CHAIN = 6

RESOURCE_KEYS = ("11", "22", "23", "24", "25")
CONVERSION_RESOURCE_KEYS = ("11", "22", "23", "24", "25", "12", "13")

BRANCH_CLOSURE_INTENTS = {
    "H": [[23, 22], [23, 23, 22]],
    "H1": [[22], [23, 22]],
    "F1": [[23, 11], [23, 23, 11]],
    "F2": [[23, 22], [23, 23, 22]],
    "F1-": [[11]],
    "F2-": [[22]],
}


@dataclass
class SUNode:
    global_id: int
    su_type: int
    hop1: Tuple[int, ...] = field(default_factory=tuple)
    hop2: Tuple[int, ...] = field(default_factory=tuple)
    target_degree: Optional[int] = None
    target_fixed_anchor_count: Optional[int] = None
    anchor_partition: Optional[str] = None
    anchor_mode: Optional[str] = None
    hop1_ids: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ChainSpec:
    chain_type: str
    composition: List[int]
    origin_type: str
    source_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_11: int = 0
    n_23: int = 0
    n_22: int = 0
    n_24: int = 0
    n_25: int = 0
    trace_required_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.composition = [int(x) for x in list(self.composition or [])]
        self.source_ids = [int(x) for x in list(self.source_ids or [])]
        self.metadata = dict(self.metadata or {})
        body = self.placement_su_sequence()
        self.n_11 = int(sum(1 for su in self.composition if int(su) == 11))
        self.n_23 = int(len(body))
        self.n_22 = int(sum(1 for su in body if int(su) == 22))
        self.n_24 = int(sum(1 for su in body if int(su) in {14, 24}))
        self.n_25 = int(sum(1 for su in body if int(su) == 25))
        if not self.trace_required_counts:
            self.trace_required_counts = {"raw_nodes": int(len(body))}

    def placement_su_sequence(self) -> List[int]:
        main_body = self.metadata.get("main_body_su")
        if isinstance(main_body, (list, tuple)) and main_body:
            return [int(x) for x in main_body]
        if self.chain_type in {"bridge", "branch_bridge"}:
            return list(self.composition[1:-1])
        if self.chain_type in {"side", "branch_side"}:
            return list(self.composition[1:])
        return list(self.composition)


@dataclass
class ResourceVector:
    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.counts = {
            str(k): max(0, int(dict(self.counts or {}).get(str(k), dict(self.counts or {}).get(int(k), 0)) or 0))
            for k in CONVERSION_RESOURCE_KEYS
        }

    def copy(self) -> "ResourceVector":
        return ResourceVector(dict(self.counts))

    def get(self, key: Any, default: int = 0) -> int:
        return int(self.counts.get(str(key), int(default)))

    def set(self, key: Any, value: int) -> None:
        self.counts[str(key)] = max(0, int(value))

    def add(self, key: Any, delta: int) -> None:
        self.set(str(key), int(self.get(str(key))) + int(delta))

    def to_dict(self, keys: Sequence[str] = CONVERSION_RESOURCE_KEYS) -> Dict[str, int]:
        return {str(k): int(self.counts.get(str(k), 0)) for k in keys}


@dataclass(frozen=True)
class ConversionRule:
    op: str
    consume: Dict[str, int]
    produce: Dict[str, int]
    phase: str = "any"
    cost: int = 1


@dataclass
class PlannedConversion:
    op: str
    count: int = 1
    phase: str = "closure"
    reason: str = ""
    source: str = "solver"

    def to_row(self) -> Dict[str, Any]:
        return {
            "op": str(self.op),
            "count": max(0, int(self.count)),
            "phase": str(self.phase),
            "reason": str(self.reason),
            "source": str(self.source),
        }


@dataclass
class ResidualGrowthAction:
    chain_index: int
    add_23: int
    endpoint: int
    before_topology: List[int] = field(default_factory=list)
    after_topology: List[int] = field(default_factory=list)
    reason: str = "residual_growth"

    def to_row(self) -> Dict[str, Any]:
        return {
            "chain_index": int(self.chain_index),
            "add_23": max(0, int(self.add_23)),
            "endpoint": int(self.endpoint),
            "before_topology": [int(x) for x in list(self.before_topology or [])],
            "after_topology": [int(x) for x in list(self.after_topology or [])],
            "reason": str(self.reason),
        }


@dataclass
class ResidualChainAction:
    composition: List[int]
    chain_type: str = "side"
    reason: str = "residual_chain"

    def to_row(self) -> Dict[str, Any]:
        return {
            "composition": [int(x) for x in list(self.composition or [])],
            "chain_type": str(self.chain_type),
            "reason": str(self.reason),
        }


@dataclass
class TopologyAssemblyPlan:
    feasible: bool = False
    materialized: bool = False
    materialization_required: bool = False
    selected_option_ids: List[str] = field(default_factory=list)
    planned_conversions: List[PlannedConversion] = field(default_factory=list)
    residual_growth_actions: List[ResidualGrowthAction] = field(default_factory=list)
    residual_chain_actions: List[ResidualChainAction] = field(default_factory=list)
    residual_remaining: Dict[str, int] = field(default_factory=dict)
    chain_23_limit_violations: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "materialized": bool(self.materialized),
            "materialization_required": bool(self.materialization_required),
            "selected_option_ids": [str(x) for x in list(self.selected_option_ids or [])],
            "planned_conversions": [conv.to_row() for conv in list(self.planned_conversions or [])],
            "residual_growth_actions": [act.to_row() for act in list(self.residual_growth_actions or [])],
            "residual_chain_actions": [act.to_row() for act in list(self.residual_chain_actions or [])],
            "residual_remaining": {str(k): int(v) for k, v in dict(self.residual_remaining or {}).items()},
            "chain_23_limit_violations": [dict(row) for row in list(self.chain_23_limit_violations or [])],
            "reason": str(self.reason),
        }


@dataclass
class AllocationResult:
    bridge_chains: List[ChainSpec] = field(default_factory=list)
    side_chains: List[ChainSpec] = field(default_factory=list)
    branch_chains: List[ChainSpec] = field(default_factory=list)
    pending_closure_topologies: List[ChainSpec] = field(default_factory=list)
    total_11: int = 0
    total_23: int = 0
    total_22: int = 0
    total_24: int = 0
    total_25: int = 0
    consumed_11: int = 0
    consumed_23: int = 0
    consumed_22: int = 0
    consumed_24: int = 0
    consumed_25: int = 0
    remaining_11: int = 0
    remaining_23: int = 0
    remaining_22: int = 0
    remaining_24: int = 0
    remaining_25: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    unallocated_bridge: int = 0
    unallocated_branch: int = 0
    required_extra_22: int = 0
    required_extra_11: int = 0
    required_extra_23: int = 0
    required_extra_24: int = 0
    required_extra_25: int = 0
    unsupported_special_count: int = 0
    unsupported_special_blocked_count: int = 0
    unsupported_special_nodes: List[int] = field(default_factory=list)
    unsupported_special_details: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_special_reasons: Dict[str, int] = field(default_factory=dict)
    extra_11_23_11_count: int = 0
    extra_long_23_chains: int = 0
    extra_11_22_count: int = 0
    extra_short_bridge_count: int = 0
    extra_side_to_22_count: int = 0
    extra_bridge_avg_23: float = 0.0
    post_flex_23_requested: int = 0
    post_flex_23_applied: int = 0
    post_flex_23_to_side: int = 0
    post_flex_23_to_branch: int = 0
    native_total_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_total_by_kind: Dict[str, int] = field(default_factory=dict)
    native_consumed_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_consumed_by_kind: Dict[str, int] = field(default_factory=dict)
    native_remaining_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_remaining_by_kind: Dict[str, int] = field(default_factory=dict)
    source_su_total_counts: Dict[int, int] = field(default_factory=dict)
    source_su_consumed_counts: Dict[int, int] = field(default_factory=dict)
    source_su_remaining_counts: Dict[int, int] = field(default_factory=dict)
    section_consumed_by_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)
    origin_consumed_by_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)
    abstract_gap_by_kind: Dict[str, int] = field(default_factory=dict)
    abstract_gap_audit_by_kind: Dict[str, int] = field(default_factory=dict)
    trace_consumed_by_section_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)
    trace_consumed_by_origin_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)
    trace_exact_chain_count: int = 0
    trace_inexact_chain_count: int = 0
    pending_closure_count: int = 0
    pending_closure_total_count: int = 0
    pending_closure_blocking_count: int = 0
    pending_closure_deferred_count: int = 0
    fragment_rows: List[Dict[str, Any]] = field(default_factory=list)
    pending_closure_rows: List[Dict[str, Any]] = field(default_factory=list)
    catalog_fragment_rows: List[Dict[str, Any]] = field(default_factory=list)
    catalog_summary: Dict[str, int] = field(default_factory=dict)
    closed_topology_specs: List[ChainSpec] = field(default_factory=list)
    topology_assembly_diagnostics: Dict[str, Any] = field(default_factory=dict)
    topology_assembly_options: List[Dict[str, Any]] = field(default_factory=list)
    topology_selected_options: List[Dict[str, Any]] = field(default_factory=list)
    topology_unclosed_candidates: List[Dict[str, Any]] = field(default_factory=list)
    topology_plan_feasible: bool = False
    topology_materialized_ok: bool = False
    materialization_required: bool = False
    planned_conversions: List[Dict[str, Any]] = field(default_factory=list)
    residual_topology_specs: List[ChainSpec] = field(default_factory=list)
    residual_growth_actions: List[Dict[str, Any]] = field(default_factory=list)
    residual_remaining_by_kind: Dict[str, int] = field(default_factory=dict)
    chain_23_limit_violations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _DoubleHalf:
    kind: str
    states: List[Any]
    connector_id: int
    outer_open: str
    origin_rule: str
    actual_states: List[Any] = field(default_factory=list)
    abstracted_neighbors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _FixedConnectorExpansion:
    kind: str
    connector: Any
    path_states: List[Any]
    path_su: List[int]
    path_labels: List[str]
    node_ids: List[int]
    open_state: str
    display_topology: str
    origin_rule: str
    actual_path_su: List[int] = field(default_factory=list)
    actual_path_labels: List[str] = field(default_factory=list)
    actual_node_ids: List[int] = field(default_factory=list)
    abstracted_neighbors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _BranchTopologyCandidate:
    group_id: str
    abcd_type: str
    center: Any
    source_kind: str
    fixed_role: str
    main_body_su: List[int]
    main_labels: List[str]
    branch_su: List[int]
    branch_labels: List[str]
    source_ids: List[int]
    open_topology: str
    open_ends: List[str]
    native24_class: str = ""
    analog24_class: str = ""
    branch_display_class: str = ""
    actual_main_path_su: List[int] = field(default_factory=list)
    actual_main_path_labels: List[str] = field(default_factory=list)
    actual_used_su: List[int] = field(default_factory=list)
    actual_used_source_ids: List[int] = field(default_factory=list)
    chain_port_signature: List[int] = field(default_factory=list)
    connector_expansions: List[_FixedConnectorExpansion] = field(default_factory=list)
    origin_rule: str = ""
    analog_template: str = ""
    closure_intents: Dict[str, List[str]] = field(default_factory=dict)
    pattern_rule: str = ""
    main_open: bool = False
    branch_open: bool = False
    assembly_family: str = ""
    assembly_shape: str = ""
    usage_modes: List[str] = field(default_factory=list)
    ring_eligible: bool = False
    ring_scope: str = ""
    ring_position_role: str = ""
    selected_fixed_role: str = ""
    main_connector_kind: str = ""
    branch_connector_kind: str = ""
    main_connector_ids: List[int] = field(default_factory=list)
    branch_connector_ids: List[int] = field(default_factory=list)
    exclusive_group_key: str = ""
    disallow_ring_reason: str = ""
    bridge_connector_id: int = -1
    bridge_connector_su: int = -1
    bridge_left_gid: int = -1
    bridge_right_gid: int = -1
    bridge_left_abcd_type: str = ""
    bridge_right_abcd_type: str = ""
    bridgehead_replace_24c_eligible: bool = False
    suppresses_local_candidate_keys: List[str] = field(default_factory=list)
    priority_rank: int = 0
    branch_slots_su: Dict[str, List[int]] = field(default_factory=dict)
    branch_slot_labels: Dict[str, str] = field(default_factory=dict)
    fixed_22_is_abstract: bool = False
    evidence_ids: List[int] = field(default_factory=list)
    abstracted_neighbors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def center_label(self) -> str:
        return str(self.center.label)

    def profile(self, allocator_role: str) -> Dict[str, Any]:
        connector_ids: List[int] = []
        connector_types: List[int] = []
        for exp in self.connector_expansions:
            connector_ids.extend(int(x) for x in exp.node_ids if int(x) >= 0)
            if exp.connector is not None:
                connector_types.append(int(exp.connector.su))
        return {
            "node_id": int(self.center.gid),
            "su_type": int(self.center.su),
            "abcd_type": str(self.abcd_type),
            "native24_class": str(self.native24_class),
            "analog24_class": str(self.analog24_class),
            "branch_display_class": str(self.branch_display_class),
            "source_kind": str(self.source_kind),
            "allocator_role": str(allocator_role),
            "fixed_usage": str(self.fixed_role),
            "fixed_label": str(self.center.label),
            "fixed_connector_ids": sorted(set(connector_ids)),
            "fixed_connector_types": list(connector_types),
            "partner_special_ids": [],
            "partner_special_degrees": [],
            "fixed_path_kind": str(self.origin_rule),
            "is_double_special": str(self.source_kind).startswith("double"),
            "allowed_tail_modes": ["pending"],
            "branch_su": [int(x) for x in self.branch_su],
            "branch_labels": list(self.branch_labels),
            "branch_slots_su": {
                str(k): [int(x) for x in list(v or [])]
                for k, v in dict(self.branch_slots_su or {}).items()
            },
            "branch_slot_labels": {
                str(k): str(v)
                for k, v in dict(self.branch_slot_labels or {}).items()
            },
            "fixed_22_is_abstract": bool(self.fixed_22_is_abstract),
            "evidence_ids": [int(x) for x in list(self.evidence_ids or []) if int(x) >= 0],
            "abstracted_neighbors": [
                dict(row) for row in list(self.abstracted_neighbors or [])
                if isinstance(row, dict)
            ],
            "actual_main_path_su": [int(x) for x in self.actual_main_path_su],
            "actual_main_path_labels": list(self.actual_main_path_labels),
            "actual_used_su": [int(x) for x in self.actual_used_su],
            "actual_used_source_ids": [int(x) for x in self.actual_used_source_ids],
            "chain_port_signature": [int(x) for x in self.chain_port_signature],
            "branch_node_ids": [
                int(x)
                for x in self.source_ids
                if int(x) >= 0 and int(x) != int(self.center.gid)
            ],
            "main_body_su": [int(x) for x in self.main_body_su],
            "main_labels": list(self.main_labels),
            "open_topology": str(self.open_topology),
            "analog_template": str(self.analog_template),
            "pattern_rule": str(self.pattern_rule or self.origin_rule),
            "closure_intents": {
                str(k): list(v) for k, v in dict(self.closure_intents or {}).items()
            },
            "main_open": bool(self.main_open),
            "branch_open": bool(self.branch_open),
            "assembly_family": str(self.assembly_family),
            "assembly_shape": str(self.assembly_shape),
            "usage_modes": list(self.usage_modes),
            "ring_eligible": bool(self.ring_eligible),
            "ring_scope": str(self.ring_scope),
            "ring_position_role": str(self.ring_position_role),
            "selected_fixed_role": str(self.selected_fixed_role or self.fixed_role),
            "main_connector_kind": str(self.main_connector_kind),
            "branch_connector_kind": str(self.branch_connector_kind),
            "main_connector_ids": [int(x) for x in list(self.main_connector_ids or [])],
            "branch_connector_ids": [int(x) for x in list(self.branch_connector_ids or [])],
            "exclusive_group_key": str(self.exclusive_group_key or self.group_id),
            "disallow_ring_reason": str(self.disallow_ring_reason),
            "bridge_connector_id": int(self.bridge_connector_id),
            "bridge_connector_su": int(self.bridge_connector_su),
            "bridge_left_gid": int(self.bridge_left_gid),
            "bridge_right_gid": int(self.bridge_right_gid),
            "bridge_left_abcd_type": str(self.bridge_left_abcd_type),
            "bridge_right_abcd_type": str(self.bridge_right_abcd_type),
            "bridgehead_replace_24c_eligible": bool(self.bridgehead_replace_24c_eligible),
            "suppresses_local_candidate_keys": list(self.suppresses_local_candidate_keys),
            "priority_rank": int(self.priority_rank),
            "connector_expansions": [
                {
                    "kind": str(exp.kind),
                    "path_su": [int(x) for x in exp.path_su],
                    "path_labels": list(exp.path_labels),
                    "actual_path_su": [
                        int(x) for x in list(exp.actual_path_su or exp.path_su)
                    ],
                    "actual_path_labels": list(exp.actual_path_labels or exp.path_labels),
                    "node_ids": [int(x) for x in exp.node_ids],
                    "open_state": str(exp.open_state),
                    "display_topology": str(exp.display_topology),
                    "origin_rule": str(exp.origin_rule),
                    "actual_node_ids": [
                        int(x) for x in list(exp.actual_node_ids or exp.node_ids)
                    ],
                    "abstracted_neighbors": [
                        dict(row) for row in list(exp.abstracted_neighbors or [])
                        if isinstance(row, dict)
                    ],
                }
                for exp in self.connector_expansions
            ],
        }


def _counts_for_sequence(values: Iterable[int]) -> Dict[str, int]:
    vals = [int(x) for x in list(values or [])]
    return {
        "11": int(sum(1 for x in vals if x == 11)),
        "22": int(sum(1 for x in vals if x == 22)),
        "23": int(sum(1 for x in vals if x == 23)),
        "24": int(sum(1 for x in vals if x in {14, 24})),
        "25": int(sum(1 for x in vals if x == 25)),
    }


def chain_actual_topology_su(chain: ChainSpec) -> List[int]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    for key in ("actual_topology_su", "resolved_topology_su"):
        vals = meta.get(str(key))
        if isinstance(vals, (list, tuple)) and vals:
            return [int(x) for x in list(vals or [])]
    branch = meta.get("branch_su")
    vals = [int(x) for x in list(getattr(chain, "composition", []) or [])]
    if isinstance(branch, (list, tuple)) and branch:
        vals.extend(int(x) for x in list(branch or []))
    return vals


def sequence_23_count(values: Iterable[int]) -> int:
    return int(sum(1 for su in list(values or []) if int(su) == 23))


def chain_23_count(chain: ChainSpec) -> int:
    return sequence_23_count(chain_actual_topology_su(chain))


def chain_23_limit_ok(chain: ChainSpec, max_23: int = MAX_23_PER_CHAIN) -> bool:
    if str(getattr(chain, "chain_type", "")) not in {
        "bridge",
        "side",
        "branch_bridge",
        "branch_side",
        "side_ring",
        "vertical_ring",
        "fused_side_ring",
    }:
        return True
    return bool(int(chain_23_count(chain)) <= int(max_23))


def _is_missing_value(value: Any) -> bool:
    if pd is None:
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _chain_required_resource_counts(chain: ChainSpec) -> Dict[str, int]:
    override = dict((getattr(chain, "metadata", {}) or {}).get("resource_requirements_override", {}) or {})
    if override:
        return {
            "11": int(override.get("11", 0)),
            "22": int(override.get("22", 0)),
            "23": int(override.get("23", 0)),
            "24": int(override.get("24", 0)),
            "25": int(override.get("25", 0)),
        }
    if chain.chain_type in {"bridge", "branch_bridge"} and len(chain.composition) >= 2:
        body = list(chain.composition[1:-1])
    elif chain.chain_type in {"side", "branch_side"} and len(chain.composition) >= 1:
        body = list(chain.composition[1:])
    else:
        body = list(chain.composition)
    counts = _counts_for_sequence(body)
    counts["11"] = int(chain.n_11)
    return counts


def _sum_chain_required_resource_counts(chains: Sequence[ChainSpec]) -> Dict[str, int]:
    total = {"11": 0, "22": 0, "23": 0, "24": 0, "25": 0}
    for chain in list(chains or []):
        counts = _chain_required_resource_counts(chain)
        for key in total:
            total[key] += int(counts.get(key, 0))
    return total


def chain_spec_counts_match(chain_spec: ChainSpec, candidate_sus: List[int]) -> bool:
    target = [int(x) for x in chain_spec.placement_su_sequence()]
    if chain_spec.chain_type in {"branch_bridge", "branch_side"}:
        meta = dict(getattr(chain_spec, "metadata", {}) or {})
        branch_su = meta.get("branch_su")
        if isinstance(branch_su, (list, tuple)) and branch_su:
            target.extend(int(x) for x in list(branch_su or []))
    got = [int(x) for x in list(candidate_sus or [])]
    return got == target or Counter(got) == Counter(target)
