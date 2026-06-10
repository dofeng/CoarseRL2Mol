from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .allocator_patterns import AROMATIC_SU, TERMINAL_BASE_SU
from .allocator_models import (
    BRANCH_CLOSURE_INTENTS,
    CONVERSION_RESOURCE_KEYS,
    MAX_23_PER_CHAIN,
    ChainSpec,
    _chain_required_resource_counts,
    _sum_chain_required_resource_counts,
    chain_23_count,
    chain_23_limit_ok,
    chain_actual_topology_su,
)
from .topology_solver import TopologyAssemblySolver


RESOURCE_KEYS = ("11", "22", "23", "24", "25")


@dataclass
class TopologyAssemblyOption:
    option_id: str
    source_group_id: str
    source_ids: List[int]
    assembly_family: str
    usage_mode: str
    topology_template: str
    closure_plan: Dict[str, List[int]]
    resource_cost: Dict[str, int]
    base_resource_cost: Dict[str, int]
    closure_resource_cost: Dict[str, int]
    flex_count_delta: int = 0
    side_count_delta: int = 0
    ring_count_delta: int = 0
    priority: int = 0
    constraints: Dict[str, Any] = field(default_factory=dict)
    candidate_row: Dict[str, Any] = field(default_factory=dict)
    source_chain: Optional[ChainSpec] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "option_id": str(self.option_id),
            "source_group_id": str(self.source_group_id),
            "source_ids": [int(x) for x in self.source_ids],
            "assembly_family": str(self.assembly_family),
            "usage_mode": str(self.usage_mode),
            "topology_template": str(self.topology_template),
            "closure_plan": {str(k): [int(x) for x in v] for k, v in self.closure_plan.items()},
            "resource_cost": {str(k): int(self.resource_cost.get(k, 0)) for k in RESOURCE_KEYS},
            "base_resource_cost": {str(k): int(self.base_resource_cost.get(k, 0)) for k in RESOURCE_KEYS},
            "closure_resource_cost": {str(k): int(self.closure_resource_cost.get(k, 0)) for k in RESOURCE_KEYS},
            "flex_count_delta": int(self.flex_count_delta),
            "side_count_delta": int(self.side_count_delta),
            "ring_count_delta": int(self.ring_count_delta),
            "priority": int(self.priority),
            "constraints": dict(self.constraints or {}),
        }


class ResourceLedger:
    def __init__(self, totals: Dict[str, int], pre_consumed: Optional[Dict[str, int]] = None):
        self.total = {str(k): int(totals.get(k, 0)) for k in RESOURCE_KEYS}
        self.pre_consumed = {str(k): int((pre_consumed or {}).get(k, 0)) for k in RESOURCE_KEYS}
        self.consumed = {str(k): int(self.pre_consumed.get(k, 0)) for k in RESOURCE_KEYS}
        self.shortage = {str(k): 0 for k in RESOURCE_KEYS}
        self.consumed_by_option: List[Dict[str, Any]] = []

    def remaining(self) -> Dict[str, int]:
        return {
            str(k): max(0, int(self.total.get(k, 0)) - int(self.consumed.get(k, 0)))
            for k in RESOURCE_KEYS
        }

    def can_consume(self, cost: Dict[str, int]) -> bool:
        rem = self.remaining()
        return all(int(cost.get(k, 0)) <= int(rem.get(k, 0)) for k in RESOURCE_KEYS)

    def consume(self, option: TopologyAssemblyOption) -> bool:
        cost = {str(k): int(option.resource_cost.get(k, 0)) for k in RESOURCE_KEYS}
        if not self.can_consume(cost):
            rem = self.remaining()
            for key in RESOURCE_KEYS:
                gap = max(0, int(cost.get(key, 0)) - int(rem.get(key, 0)))
                if gap:
                    self.shortage[key] = max(int(self.shortage.get(key, 0)), int(gap))
            return False
        for key in RESOURCE_KEYS:
            self.consumed[key] = int(self.consumed.get(key, 0)) + int(cost.get(key, 0))
        self.consumed_by_option.append({
            "option_id": str(option.option_id),
            "assembly_family": str(option.assembly_family),
            "usage_mode": str(option.usage_mode),
            "cost": dict(cost),
            "source_ids": [int(x) for x in option.source_ids],
        })
        return True

    def row(self) -> Dict[str, Any]:
        return {
            "total": dict(self.total),
            "pre_consumed": dict(self.pre_consumed),
            "consumed": dict(self.consumed),
            "remaining": self.remaining(),
            "shortage": dict(self.shortage),
            "consumed_by_option": list(self.consumed_by_option),
        }


@dataclass
class BranchAssemblyResult:
    ordinary_chains: List[ChainSpec] = field(default_factory=list)
    closed_topology_specs: List[ChainSpec] = field(default_factory=list)
    unclosed_pending_specs: List[ChainSpec] = field(default_factory=list)
    residual_topology_specs: List[ChainSpec] = field(default_factory=list)
    selected_options: List[TopologyAssemblyOption] = field(default_factory=list)
    all_options: List[TopologyAssemblyOption] = field(default_factory=list)
    unclosed_candidates: List[Dict[str, Any]] = field(default_factory=list)
    resource_ledger: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    recommended_adjustments: List[Dict[str, Any]] = field(default_factory=list)


class BranchTopologyAssembler:
    def __init__(
        self,
        pending_branch_chains: Sequence[ChainSpec],
        ordinary_chains: Sequence[ChainSpec],
        resource_totals: Dict[str, int],
        constraints: Optional[Dict[str, Any]] = None,
        pending_ordinary_chains: Optional[Sequence[ChainSpec]] = None,
    ):
        self.pending_branch_chains = list(pending_branch_chains or [])
        self.ordinary_chains = list(ordinary_chains or [])
        self.pending_ordinary_chains = list(pending_ordinary_chains or [])
        self.resource_totals = {
            str(k): int(dict(resource_totals or {}).get(str(k), dict(resource_totals or {}).get(int(k), 0)) or 0)
            for k in tuple(RESOURCE_KEYS) + ("12", "13")
        }
        self.constraints = dict(constraints or {})
        self.flex_lower_raw = max(0, int(
            self.constraints.get(
                "flex_lower_raw",
                self.constraints.get("flex_lower", self.constraints.get("flexible_bridge_min", 0)),
            ) or 0
        ))
        self.flex_upper_raw = int(
            self.constraints.get(
                "flex_upper_raw",
                self.constraints.get("flex_upper", self.constraints.get("flexible_bridge_limit", 10**9)),
            ) or 10**9
        )
        self.flex_window_ok = bool(self.constraints.get("flex_window_ok", int(self.flex_lower_raw) <= int(self.flex_upper_raw)))
        self.flex_lower = int(self.flex_lower_raw)
        self.flex_upper = int(self.flex_upper_raw)
        self.flex_upper_adjusted_to_lower = False
        if self.flex_upper < self.flex_lower:
            self.flex_upper = self.flex_lower
            self.flex_upper_adjusted_to_lower = True
        self.allowed_backbone_residual = max(0, int(self.constraints.get("allowed_backbone_residual", 0) or 0))

    def assemble(self) -> BranchAssemblyResult:
        return self._assemble_with_solver()

    def _assemble_with_solver(self) -> BranchAssemblyResult:
        pre_consumed = _sum_chain_required_resource_counts(self.ordinary_chains)
        base_options = self.expand_options()
        ring_options = self._build_ring_options(base_options, set())
        all_options = list(base_options) + list(ring_options)
        pending_specs, pending_group_ids = self._pending_specs_with_group_ids()
        solver = TopologyAssemblySolver(
            resource_totals={
                **{str(k): int(self.resource_totals.get(str(k), 0)) for k in RESOURCE_KEYS},
                "12": int(self.constraints.get("total_12", self.resource_totals.get("12", 0)) or 0),
                "13": int(self.constraints.get("total_13", self.resource_totals.get("13", 0)) or 0),
            },
            flex_lower=int(self.flex_lower),
            flex_upper=int(self.flex_upper),
            ordinary_flex_count=self._ordinary_flex_count(self.ordinary_chains),
            ordinary_chains=self.ordinary_chains,
            pending_group_ids=pending_group_ids,
            option_to_chain=self._chain_from_option,
            used_source_ids=self._source_ids_for_chains(self.ordinary_chains),
            used_exclusive_keys=self._exclusive_keys_for_chains(self.ordinary_chains),
            max_23_per_chain=MAX_23_PER_CHAIN,
        )
        solved = solver.solve(all_options, pre_consumed)
        plan = solved.plan

        selected_options = list(solved.selected_options or [])
        selected_specs = [self._chain_from_option(opt) for opt in selected_options]
        materialized_base, residual_specs = TopologyAssemblySolver._apply_growth_specs(
            list(self.ordinary_chains) + list(selected_specs),
            list(solved.residual_specs or []),
        )
        ordinary_chains = list(materialized_base[:len(self.ordinary_chains)])
        selected_specs = list(materialized_base[len(self.ordinary_chains):])
        closed_specs = list(selected_specs) + list(residual_specs)
        unclosed_specs: List[ChainSpec] = []
        unclosed_rows: List[Dict[str, Any]] = []
        if not bool(plan.feasible):
            for chain, group_id in zip(pending_specs, pending_group_ids):
                row = self._unclosed_row_for_chain(chain, str(group_id), all_options, {})
                row["reason"] = str(plan.reason or solved.reason or row.get("reason", "solver_infeasible"))
                row["deferred_nonblocking"] = False
                unclosed_rows.append(row)
                pending = copy.deepcopy(chain)
                pending.metadata = dict(pending.metadata or {})
                pending.metadata["unclosed_topology_candidate"] = True
                pending.metadata["unclosed_reason"] = str(row["reason"])
                pending.metadata["deferred_nonblocking"] = False
                unclosed_specs.append(pending)

        flex_count = int(self._ordinary_flex_count(ordinary_chains))
        side_count = int(self._ordinary_side_count(ordinary_chains))
        ring_count = int(self._ordinary_ring_count(ordinary_chains))
        for opt in selected_options:
            flex_count += int(getattr(opt, "flex_count_delta", 0) or 0)
            side_count += int(getattr(opt, "side_count_delta", 0) or 0)
            ring_count += int(getattr(opt, "ring_count_delta", 0) or 0)
        for chain in residual_specs:
            if str(getattr(chain, "chain_type", "")) in {"bridge", "branch_bridge"}:
                flex_count += 1
            elif str(getattr(chain, "chain_type", "")) in {"side", "branch_side"}:
                side_count += 1

        residual_remaining = {
            str(k): max(0, int(dict(plan.residual_remaining or {}).get(str(k), 0)))
            for k in RESOURCE_KEYS
        }
        chain_23_violations = list(plan.chain_23_limit_violations or [])
        for idx, chain in enumerate(list(ordinary_chains) + list(closed_specs)):
            if not chain_23_limit_ok(chain, MAX_23_PER_CHAIN):
                chain_23_violations.append({
                    "chain_index": int(idx),
                    "chain_type": str(getattr(chain, "chain_type", "")),
                    "origin_type": str(getattr(chain, "origin_type", "")),
                    "count_23": int(chain_23_count(chain)),
                    "max_23": int(MAX_23_PER_CHAIN),
                    "composition": [int(x) for x in chain_actual_topology_su(chain)],
                })

        resource_shortage = {str(k): 0 for k in RESOURCE_KEYS}
        pending_demand_summary = self._pending_closure_demand_summary(
            unclosed_rows,
            all_options,
            [],
            {str(k): max(0, int(self.resource_totals.get(str(k), 0)) - int(pre_consumed.get(str(k), 0))) for k in RESOURCE_KEYS},
        )
        if not bool(plan.feasible):
            for key, value in dict(pending_demand_summary.get("resource_deficit", {}) or {}).items():
                resource_shortage[str(key)] = max(0, int(value))

        flex_short = max(0, int(self.flex_lower) - int(flex_count))
        flex_excess = max(0, int(flex_count) - int(self.flex_upper))
        materialization_required = bool(plan.materialization_required)
        materialized_ok = bool(
            bool(plan.feasible)
            and not bool(materialization_required)
            and not unclosed_specs
            and not any(int(v) > 0 for v in resource_shortage.values())
            and not any(int(v) > 0 for v in residual_remaining.values())
            and not chain_23_violations
            and int(flex_short) == 0
            and int(flex_excess) == 0
        )
        recommendations = self._recommended_adjustments(
            resource_shortage,
            residual_remaining,
            flex_count,
            self.flex_lower,
            self.flex_upper,
            len(unclosed_specs),
            backbone_residual=sum(int(v) for v in residual_remaining.values()),
            unclosed_rows=unclosed_rows,
            pending_demand_summary=pending_demand_summary,
        )
        diag = {
            "ok": bool(materialized_ok),
            "topology_plan_feasible": bool(plan.feasible),
            "topology_materialized_ok": bool(materialized_ok),
            "materialization_required": bool(materialization_required),
            "solver_reason": str(plan.reason or solved.reason or ""),
            "solver_state_count": int(getattr(solver, "states_seen", 0)),
            "planned_conversions": [conv.to_row() for conv in list(plan.planned_conversions or [])],
            "planned_conversion_count": int(sum(int(conv.count) for conv in list(plan.planned_conversions or []))),
            "residual_growth_actions": [act.to_row() for act in list(plan.residual_growth_actions or [])],
            "residual_chain_actions": [act.to_row() for act in list(plan.residual_chain_actions or [])],
            "residual_topology_count": int(len(residual_specs)),
            "residual_remaining": dict(residual_remaining),
            "chain_23_limit_violations": list(chain_23_violations),
            "closed_topology_count": int(len(closed_specs)),
            "unclosed_like24_count": int(len(unclosed_specs)),
            "deferred_ring_pending_count": 0,
            "selected_option_count": int(len(selected_options)),
            "available_option_count": int(len(all_options)),
            "flex_count": int(flex_count),
            "fixed_flex_count": int(self._ordinary_flex_count(self.ordinary_chains)),
            "flex_scenario": "solver",
            "explicit_flex_count": int(flex_count),
            "ordinary_explicit_flex_count": int(self._ordinary_flex_count(ordinary_chains)),
            "mandatory_branch_flex_count": 0,
            "generated_flex_count": max(0, int(flex_count) - int(self._ordinary_flex_count(self.ordinary_chains))),
            "prefill_flex_count": 0,
            "final_repair_flex_count": 0,
            "extra_backbone_flex_count": int(sum(1 for ch in residual_specs if str(getattr(ch, "chain_type", "")) == "bridge")),
            "flex_prefill_target": int(self._target_flex_count(flex_count)),
            "flex_lower": int(self.flex_lower),
            "flex_upper": int(self.flex_upper),
            "flex_lower_raw": int(self.flex_lower_raw),
            "flex_upper_raw": int(self.flex_upper_raw),
            "flex_window_ok": bool(self.flex_window_ok),
            "flex_upper_adjusted_to_lower": bool(self.flex_upper_adjusted_to_lower),
            "X": int(self.constraints.get("X", 0) or 0),
            "Y": int(self.constraints.get("Y", 0) or 0),
            "Z": int(self.constraints.get("Z", 0) or 0),
            "flex_window_source": str(self.constraints.get("flex_window_source") or "solver"),
            "flex_window_error": str(self.constraints.get("flex_window_error") or ""),
            "bridgehead_info": tuple(int(x) for x in list(self.constraints.get("bridgehead_info", []) or [])),
            "flex_short": int(flex_short),
            "flex_excess": int(flex_excess),
            "side_count": int(side_count),
            "ring_count": int(ring_count),
            "extra_backbone_topology_count": int(len(residual_specs)),
            "resource_shortage": dict(resource_shortage),
            "resource_shortage_ledger_max_gap": dict(resource_shortage),
            "resource_surplus": dict(residual_remaining),
            "pending_closure_demand_summary": dict(pending_demand_summary),
            "pending_closure_intent_demand": dict(pending_demand_summary.get("intent_demand", {}) or {}),
            "pending_closure_resource_demand": dict(pending_demand_summary.get("resource_demand", {}) or {}),
            "pending_closure_resource_available": dict(pending_demand_summary.get("resource_available", {}) or {}),
            "pending_closure_resource_deficit": dict(pending_demand_summary.get("resource_deficit", {}) or {}),
            "minimum_conversions": list(pending_demand_summary.get("minimum_conversions", []) or []),
            "abstract_gap_by_kind": {"pending_closure": int(len(unclosed_specs))} if unclosed_specs else {},
            "abstract_gap_total": int(len(unclosed_specs)),
            "backbone_residual": int(sum(int(v) for v in residual_remaining.values())),
            "backbone_residual_ok": bool(not any(int(v) > 0 for v in residual_remaining.values())),
            "allowed_backbone_residual": 0,
            "resource_ledger": {
                "total": {str(k): int(self.resource_totals.get(str(k), 0)) for k in RESOURCE_KEYS},
                "pre_consumed": dict(pre_consumed),
                "consumed": _sum_chain_required_resource_counts(list(ordinary_chains) + list(closed_specs)),
                "remaining": dict(residual_remaining),
                "shortage": dict(resource_shortage),
                "planned_conversions": [conv.to_row() for conv in list(plan.planned_conversions or [])],
            },
            "ring_assembly_enabled": True,
            "ring_option_count": int(sum(1 for opt in all_options if str(getattr(opt, "usage_mode", "")) == "ring_candidate")),
            "ring_pair_option_count": int(len(ring_options)),
            "selected_ring_plan_counts_by_type": {},
            "selected_ring_planner_counts": {},
            "bridgehead_14_replacement_count": 0,
            "like24_used_by_ring": 0,
            "raw_su24_filler_used": 0,
            "disabled_ring_option_count": 0,
            "disabled_ring_options": [],
            "rejected_option_count": 0,
            "rejected_options": [],
            "ordinary_source_ids": sorted(int(x) for x in self._source_ids_for_chains(ordinary_chains)),
            "recommended_adjustments": list(recommendations),
            "closure_engine_removed": False,
        }
        return BranchAssemblyResult(
            ordinary_chains=ordinary_chains,
            closed_topology_specs=closed_specs,
            unclosed_pending_specs=unclosed_specs,
            residual_topology_specs=residual_specs,
            selected_options=selected_options,
            all_options=all_options,
            unclosed_candidates=unclosed_rows,
            resource_ledger=dict(diag["resource_ledger"]),
            diagnostics=diag,
            recommended_adjustments=list(recommendations),
        )

    def _assemble_greedy_legacy(self) -> BranchAssemblyResult:
        pre_consumed = _sum_chain_required_resource_counts(self.ordinary_chains)
        ledger = ResourceLedger(self.resource_totals, pre_consumed=pre_consumed)
        all_options = self.expand_options()
        selected_options: List[TopologyAssemblyOption] = []
        closed_specs: List[ChainSpec] = []
        rejected_options: List[Dict[str, Any]] = []
        group_reject_reasons: Dict[str, List[str]] = {}
        used_groups: set[str] = set()
        used_source_ids = self._source_ids_for_chains(self.ordinary_chains)
        used_exclusive_keys = self._exclusive_keys_for_chains(self.ordinary_chains)
        flex_count = self._ordinary_flex_count(self.ordinary_chains)
        side_count = self._ordinary_side_count(self.ordinary_chains)
        ring_count = self._ordinary_ring_count(self.ordinary_chains)
        ordinary_explicit_flex_count = int(flex_count)
        mandatory_branch_flex_count = 0
        prefill_flex_count = 0
        final_repair_flex_count = 0
        extra_backbone_flex_count = 0

        def reject(option: TopologyAssemblyOption, reason: str) -> None:
            row = option.to_row()
            row["reason"] = str(reason)
            rejected_options.append(row)
            group_reject_reasons.setdefault(str(option.source_group_id), []).append(str(reason))
            if str(option.usage_mode) == "ring":
                for slot_group_id in list((option.constraints or {}).get("slot_group_ids", []) or []):
                    if str(slot_group_id):
                        group_reject_reasons.setdefault(str(slot_group_id), []).append(str(reason))

        def try_select(option: TopologyAssemblyOption) -> bool:
            nonlocal flex_count, side_count, ring_count
            group_id = str(option.source_group_id)
            if group_id in used_groups:
                reject(option, "candidate_group_already_selected")
                return False
            src = {int(x) for x in list(option.source_ids or []) if int(x) >= 0}
            if src & used_source_ids:
                reason = (
                    "source_id_conflict_with_ordinary_chain"
                    if not selected_options
                    else "source_id_conflict_with_selected_option"
                )
                reject(option, reason)
                return False
            exclusive = self._exclusive_keys_for_option(option)
            if exclusive & used_exclusive_keys:
                reject(option, "exclusive_group_conflict")
                return False
            if not ledger.consume(option):
                reject(option, "resource_shortage")
                return False
            chain = self._chain_from_option(option)
            closed_specs.append(chain)
            selected_options.append(option)
            used_groups.add(group_id)
            used_source_ids.update(src)
            used_exclusive_keys.update(exclusive)
            flex_count += int(option.flex_count_delta)
            side_count += int(option.side_count_delta)
            ring_count += int(option.ring_count_delta)
            return True

        selectable = [
            opt for opt in all_options
            if str(opt.usage_mode) in {
                "ordinary_bridge",
                "ordinary_side",
                "ordinary_pair_bridge",
                "ordinary_pair_side",
                "flex_branch",
                "side_branch",
            }
        ]
        mandatory_flex_families = {
            "24_AB_M",
            "24_CD_M_BRIDGE",
            "24_AB_N_AB_BRIDGE",
        }
        for opt in sorted(
            [
                o for o in selectable
                if str(o.usage_mode) == "flex_branch"
                and str(o.assembly_family) in mandatory_flex_families
            ],
            key=self._option_sort_key,
        ):
            before = int(flex_count)
            if try_select(opt) and int(flex_count) > before:
                mandatory_branch_flex_count += int(flex_count) - int(before)

        flex_target = self._target_flex_count(flex_count)
        explicit_flex_count = int(flex_count)
        fixed_flex_count = int(explicit_flex_count)
        if not bool(self.flex_window_ok):
            flex_scenario = "flex_window_invalid"
        elif int(explicit_flex_count) < int(self.flex_lower):
            flex_scenario = "flex_short"
        elif int(explicit_flex_count) > int(self.flex_upper):
            flex_scenario = "explicit_flex_excess"
        else:
            flex_scenario = "flex_in_range"
        flex_prefill_target = (
            int(flex_target)
            if str(flex_scenario) == "flex_short"
            else int(flex_count)
        )
        if str(flex_scenario) == "flex_short" and int(flex_count) < int(flex_prefill_target):
            flex_candidates = [
                o for o in selectable
                if str(o.source_group_id) not in used_groups
                and str(o.usage_mode) in {"ordinary_bridge", "ordinary_pair_bridge", "flex_branch"}
            ]
            for opt in sorted(flex_candidates, key=self._option_sort_key):
                if flex_count >= flex_prefill_target:
                    break
                before = int(flex_count)
                if try_select(opt) and int(flex_count) > before:
                    prefill_flex_count += int(flex_count) - int(before)

        ring_options = self._build_ring_options(all_options, used_groups)
        for opt in sorted(ring_options, key=self._ring_option_sort_key):
            try_select(opt)

        side_candidates = [
            o for o in selectable
            if str(o.source_group_id) not in used_groups
            and str(o.usage_mode) in {"ordinary_side", "ordinary_pair_side", "side_branch"}
        ]
        for opt in sorted(side_candidates, key=self._side_option_sort_key):
            try_select(opt)

        if bool(self.flex_window_ok):
            pending_flex_candidates = [
                o for o in selectable
                if str(o.source_group_id) not in used_groups
                and str(o.usage_mode) in {"flex_branch", "ordinary_bridge", "ordinary_pair_bridge"}
            ]
            for opt in sorted(pending_flex_candidates, key=self._option_sort_key):
                if int(flex_count) >= int(self.flex_upper):
                    break
                if int(opt.flex_count_delta) <= 0:
                    continue
                before = int(flex_count)
                if try_select(opt) and int(flex_count) > before:
                    final_repair_flex_count += int(flex_count) - int(before)

        if bool(self.flex_window_ok) and flex_count < self.flex_lower:
            remaining_flex = [
                o for o in selectable
                if str(o.source_group_id) not in used_groups
                and str(o.usage_mode) in {"ordinary_bridge", "ordinary_pair_bridge", "flex_branch"}
            ]
            for opt in sorted(remaining_flex, key=self._option_sort_key):
                if flex_count >= self.flex_lower:
                    break
                before = int(flex_count)
                if try_select(opt) and int(flex_count) > before:
                    final_repair_flex_count += int(flex_count) - int(before)

        pending_specs, pending_group_ids = self._pending_specs_with_group_ids()
        unclosed_specs: List[ChainSpec] = []
        unclosed_rows: List[Dict[str, Any]] = []
        disabled_ring_count = 0
        blocking_unclosed_count = 0
        blocking_unclosed_rows: List[Dict[str, Any]] = []
        for chain, group_id in zip(pending_specs, pending_group_ids):
            if str(group_id) in used_groups:
                continue
            chain_src = {int(x) for x in list(getattr(chain, "source_ids", []) or []) if int(x) >= 0}
            if chain_src and chain_src.issubset(used_source_ids):
                continue
            row = self._unclosed_row_for_chain(
                chain,
                str(group_id),
                all_options,
                group_reject_reasons,
            )
            if row["reason"] == "ring_option_disabled_pending_ring_builder":
                disabled_ring_count += 1
                row["deferred_nonblocking"] = True
            else:
                blocking_unclosed_count += 1
                row["deferred_nonblocking"] = False
                blocking_unclosed_rows.append(row)
            unclosed_rows.append(row)
            pending = copy.deepcopy(chain)
            pending.metadata = dict(pending.metadata or {})
            pending.metadata["unclosed_topology_candidate"] = True
            pending.metadata["unclosed_reason"] = str(row["reason"])
            pending.metadata["deferred_nonblocking"] = bool(row["deferred_nonblocking"])
            pending.metadata["ring_builder_deferred"] = bool(row["deferred_nonblocking"])
            unclosed_specs.append(pending)

        shortage_before_extra = dict(ledger.shortage)
        closure_blocked_before_extra = bool(
            int(blocking_unclosed_count) > 0
            or any(int(v) > 0 for v in dict(shortage_before_extra or {}).values())
        )
        if not closure_blocked_before_extra:
            extra_specs = self._build_extra_backbone_specs(
                ledger,
                start_index=len(closed_specs),
                flex_count=flex_count,
            )
            for chain in extra_specs:
                option = self._option_from_extra_chain(chain)
                if not ledger.consume(option):
                    continue
                selected_options.append(option)
                closed_specs.append(chain)
                if str(chain.chain_type) == "bridge":
                    flex_count += 1
                    extra_backbone_flex_count += 1
                elif str(chain.chain_type) == "side":
                    side_count += 1

        remaining = ledger.remaining()
        backbone_residual = int(sum(int(remaining.get(k, 0)) for k in ("11", "22", "23")))
        pending_demand_summary = self._pending_closure_demand_summary(
            blocking_unclosed_rows,
            all_options,
            rejected_options,
            remaining,
        )
        pending_deficit = dict(pending_demand_summary.get("resource_deficit", {}) or {})
        ledger_shortage = dict(ledger.shortage)
        shortage = {
            str(key): max(
                0,
                int(ledger_shortage.get(str(key), 0) or 0),
                int(pending_deficit.get(str(key), 0) or 0),
            )
            for key in RESOURCE_KEYS
        }
        flex_short = max(0, int(self.flex_lower) - int(flex_count))
        flex_excess = max(0, int(flex_count) - int(self.flex_upper))
        recommendations = self._recommended_adjustments(
            shortage,
            remaining,
            flex_count,
            self.flex_lower,
            self.flex_upper,
            blocking_unclosed_count,
            disabled_ring_count=disabled_ring_count,
            backbone_residual=backbone_residual,
            unclosed_rows=blocking_unclosed_rows,
            pending_demand_summary=pending_demand_summary,
        )
        if not bool(self.flex_window_ok):
            recommendations.insert(0, {
                "op": "adjust_10_or_12_for_flex_window",
                "count": max(0, int(self.flex_lower_raw) - int(self.flex_upper_raw)),
                "reason": "flex_window_lower_exceeds_upper",
                "X": int(self.constraints.get("X", 0) or 0),
                "Y": int(self.constraints.get("Y", 0) or 0),
                "Z": int(self.constraints.get("Z", 0) or 0),
                "flex_lower_raw": int(self.flex_lower_raw),
                "flex_upper_raw": int(self.flex_upper_raw),
                "actionable": True,
            })
        generated_flex_count = int(prefill_flex_count) + int(final_repair_flex_count) + int(extra_backbone_flex_count)
        selected_ring_plan_counts: Dict[str, int] = {}
        selected_ring_planner_counts: Dict[str, int] = {}
        bridgehead_14_replacement_count = 0
        like24_used_by_ring = 0
        raw_su24_filler_used = 0
        for chain in list(closed_specs or []):
            if str(getattr(chain, "chain_type", "")) not in {"side_ring", "vertical_ring", "fused_side_ring"}:
                continue
            meta = dict(getattr(chain, "metadata", {}) or {})
            ring_type = str(meta.get("ring_type") or getattr(chain, "chain_type", ""))
            planner = str(meta.get("ring_planner") or "generic_ring_pair")
            selected_ring_plan_counts[ring_type] = int(selected_ring_plan_counts.get(ring_type, 0)) + 1
            selected_ring_planner_counts[planner] = int(selected_ring_planner_counts.get(planner, 0)) + 1
            if bool(meta.get("bridgehead_replace_24c", False)):
                bridgehead_14_replacement_count += 1
            like24_used_by_ring += int(
                meta.get("side_ring_total_like24_count")
                or meta.get("vertical_ring_like24_count")
                or meta.get("fused_ring_like24_count")
                or 0
            )
            raw_su24_filler_used += int(
                meta.get("side_ring_filler_24_count")
                or meta.get("vertical_ring_filler_24_count")
                or meta.get("fused_ring_filler_24_count")
                or 0
            )
        diag = {
            "ok": bool(
                bool(self.flex_window_ok)
                and
                int(blocking_unclosed_count) == 0
                and all(int(v) == 0 for v in shortage.values())
                and int(flex_short) == 0
                and int(flex_excess) == 0
            ),
            "closed_topology_count": int(len(closed_specs)),
            "unclosed_like24_count": int(blocking_unclosed_count),
            "deferred_ring_pending_count": int(disabled_ring_count),
            "selected_option_count": int(len(selected_options)),
            "available_option_count": int(len(all_options)),
            "flex_count": int(flex_count),
            "fixed_flex_count": int(fixed_flex_count),
            "flex_scenario": str(flex_scenario),
            "explicit_flex_count": int(explicit_flex_count),
            "ordinary_explicit_flex_count": int(ordinary_explicit_flex_count),
            "mandatory_branch_flex_count": int(mandatory_branch_flex_count),
            "generated_flex_count": int(generated_flex_count),
            "prefill_flex_count": int(prefill_flex_count),
            "final_repair_flex_count": int(final_repair_flex_count),
            "extra_backbone_flex_count": int(extra_backbone_flex_count),
            "flex_prefill_target": int(flex_prefill_target),
            "flex_lower": int(self.flex_lower),
            "flex_upper": int(self.flex_upper),
            "flex_lower_raw": int(self.flex_lower_raw),
            "flex_upper_raw": int(self.flex_upper_raw),
            "flex_window_ok": bool(self.flex_window_ok),
            "flex_upper_adjusted_to_lower": bool(self.flex_upper_adjusted_to_lower),
            "X": int(self.constraints.get("X", 0) or 0),
            "Y": int(self.constraints.get("Y", 0) or 0),
            "Z": int(self.constraints.get("Z", 0) or 0),
            "flex_window_source": str(self.constraints.get("flex_window_source") or ""),
            "flex_window_error": str(self.constraints.get("flex_window_error") or ""),
            "bridgehead_info": tuple(int(x) for x in list(self.constraints.get("bridgehead_info", []) or [])),
            "flex_short": int(flex_short),
            "flex_excess": int(flex_excess),
            "side_count": int(side_count),
            "ring_count": int(ring_count),
            "extra_backbone_topology_count": int(sum(
                1 for ch in closed_specs
                if str(getattr(ch, "origin_type", "")).startswith("Extra-backbone")
            )),
            "resource_shortage": dict(shortage),
            "resource_shortage_ledger_max_gap": dict(ledger_shortage),
            "resource_surplus": dict(remaining),
            "pending_closure_demand_summary": dict(pending_demand_summary),
            "pending_closure_intent_demand": dict(pending_demand_summary.get("intent_demand", {}) or {}),
            "pending_closure_resource_demand": dict(pending_demand_summary.get("resource_demand", {}) or {}),
            "pending_closure_resource_available": dict(pending_demand_summary.get("resource_available", {}) or {}),
            "pending_closure_resource_deficit": dict(pending_demand_summary.get("resource_deficit", {}) or {}),
            "minimum_conversions": list(pending_demand_summary.get("minimum_conversions", []) or []),
            "abstract_gap_by_kind": (
                {"pending_closure": int(blocking_unclosed_count)}
                if int(blocking_unclosed_count) > 0 else
                {}
            ),
            "abstract_gap_total": int(blocking_unclosed_count),
            "backbone_residual": int(backbone_residual),
            "backbone_residual_ok": bool(int(backbone_residual) <= int(self.allowed_backbone_residual)),
            "allowed_backbone_residual": int(self.allowed_backbone_residual),
            "resource_ledger": dict(ledger.row()),
            "ring_assembly_enabled": True,
            "ring_option_count": int(sum(1 for opt in all_options if str(opt.usage_mode) == "ring_candidate")),
            "ring_pair_option_count": int(len(ring_options)),
            "fused_ring_planner_option_count": int(sum(
                1 for opt in ring_options
                if str((opt.constraints or {}).get("ring_planner") or "") == "fused_aliphatic_ring_v1"
            )),
            "side_ring_planner_option_count": int(sum(
                1 for opt in ring_options
                if str((opt.constraints or {}).get("ring_planner") or "") == "side_aliphatic_ring_v1"
            )),
            "vertical_ring_planner_option_count": int(sum(
                1 for opt in ring_options
                if str((opt.constraints or {}).get("ring_planner") or "") == "vertical_aliphatic_ring_v1"
            )),
            "selected_ring_plan_counts_by_type": dict(sorted(selected_ring_plan_counts.items())),
            "selected_ring_planner_counts": dict(sorted(selected_ring_planner_counts.items())),
            "bridgehead_14_replacement_count": int(bridgehead_14_replacement_count),
            "like24_used_by_ring": int(like24_used_by_ring),
            "raw_su24_filler_used": int(raw_su24_filler_used),
            "disabled_ring_option_count": int(disabled_ring_count),
            "disabled_ring_options": [
                row for row in unclosed_rows
                if row["reason"] == "ring_option_disabled_pending_ring_builder"
            ],
            "rejected_option_count": int(len(rejected_options)),
            "rejected_options": list(rejected_options),
            "ordinary_source_ids": sorted(int(x) for x in self._source_ids_for_chains(self.ordinary_chains)),
            "recommended_adjustments": list(recommendations),
            "closure_engine_removed": False,
        }
        return BranchAssemblyResult(
            ordinary_chains=list(self.ordinary_chains),
            closed_topology_specs=closed_specs,
            unclosed_pending_specs=unclosed_specs,
            selected_options=selected_options,
            all_options=all_options,
            unclosed_candidates=unclosed_rows,
            resource_ledger=ledger.row(),
            diagnostics=diag,
            recommended_adjustments=list(recommendations),
        )

    def _build_extra_backbone_specs(
        self,
        ledger: ResourceLedger,
        start_index: int,
        flex_count: int,
    ) -> List[ChainSpec]:
        specs: List[ChainSpec] = []
        remaining = ledger.remaining()
        idx = int(start_index)
        local_flex = int(flex_count)

        while (
            int(local_flex) < int(self.flex_lower)
            and int(local_flex) < int(self.flex_upper)
            and int(remaining.get("11", 0)) >= 2
            and int(remaining.get("23", 0)) >= 1
        ):
            specs.append(self._extra_backbone_chain("bridge", [11, 23, 11], idx))
            idx += 1
            local_flex += 1
            remaining["11"] = int(remaining.get("11", 0)) - 2
            remaining["23"] = int(remaining.get("23", 0)) - 1

        return specs

    @staticmethod
    def _extra_backbone_chain(chain_type: str, composition: Sequence[int], idx: int) -> ChainSpec:
        ctype = str(chain_type)
        comp = [int(x) for x in list(composition or [])]
        return ChainSpec(
            chain_type=ctype,
            composition=list(comp),
            origin_type=f"Extra-backbone-{ctype}:{int(idx)}",
            source_ids=[],
            metadata={
                "allocator_mode": "extra_backbone_v1",
                "topology_state": "closed_extra_backbone_topology",
                "generated_from_remaining_backbone": True,
                "resource_requirements_override": _chain_required_resource_counts(
                    ChainSpec(chain_type=ctype, composition=list(comp), origin_type="extra")
                ),
                "main_body_su": [int(x) for x in comp[1:-1]] if ctype == "bridge" else [int(x) for x in comp[1:]],
            },
        )

    @staticmethod
    def _option_from_extra_chain(chain: ChainSpec) -> TopologyAssemblyOption:
        cost = _chain_required_resource_counts(chain)
        usage = "extra_bridge" if str(chain.chain_type) == "bridge" else "extra_side"
        return TopologyAssemblyOption(
            option_id=f"{usage}:{chain.origin_type}",
            source_group_id=str(chain.origin_type),
            source_ids=[],
            assembly_family=str(usage),
            usage_mode=str(usage),
            topology_template="-".join(str(x) for x in list(chain.composition or [])),
            closure_plan={},
            resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            base_resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: 0 for key in RESOURCE_KEYS},
            flex_count_delta=1 if str(chain.chain_type) == "bridge" else 0,
            side_count_delta=1 if str(chain.chain_type) == "side" else 0,
            priority=0,
            source_chain=chain,
        )

    def _pending_specs_with_group_ids(self) -> Tuple[List[ChainSpec], List[str]]:
        specs: List[ChainSpec] = []
        group_ids: List[str] = []
        for chain in self.pending_branch_chains:
            meta = dict(getattr(chain, "metadata", {}) or {})
            specs.append(chain)
            group_ids.append(str(meta.get("candidate_group_id") or f"pending:{id(chain)}"))
        for idx, chain in enumerate(self.pending_ordinary_chains):
            meta = dict(getattr(chain, "metadata", {}) or {})
            specs.append(chain)
            group_ids.append(str(meta.get("candidate_group_id") or f"ordinary:{idx}"))
        return specs, group_ids

    def _target_flex_count(self, current_flex_count: int) -> int:
        if int(self.flex_upper) >= 10**8:
            return max(int(self.flex_lower), int(current_flex_count))
        mid = (int(self.flex_lower) + int(self.flex_upper)) // 2
        return max(int(self.flex_lower), min(int(self.flex_upper), int(mid)))

    @staticmethod
    def _ordinary_flex_count(chains: Sequence[ChainSpec]) -> int:
        return int(sum(1 for ch in list(chains or []) if str(ch.chain_type) in {"bridge", "branch_bridge"}))

    @staticmethod
    def _ordinary_side_count(chains: Sequence[ChainSpec]) -> int:
        return int(sum(1 for ch in list(chains or []) if str(ch.chain_type) in {"side", "branch_side"}))

    @staticmethod
    def _ordinary_ring_count(chains: Sequence[ChainSpec]) -> int:
        return int(sum(1 for ch in list(chains or []) if str(ch.chain_type) in {"side_ring", "vertical_ring", "fused_side_ring"}))

    def _unclosed_row_for_chain(
        self,
        chain: ChainSpec,
        group_id: str,
        all_options: Sequence[TopologyAssemblyOption],
        group_reject_reasons: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        meta = dict(getattr(chain, "metadata", {}) or {})
        reason = self._unclosed_reason_for_group(group_id, all_options, group_reject_reasons)
        return {
            "candidate_group_id": str(group_id),
            "origin_type": str(chain.origin_type),
            "source_ids": [int(x) for x in list(chain.source_ids or [])],
            "assembly_families": list(meta.get("assembly_families", []) or []),
            "closure_intents": dict(meta.get("closure_intents", {}) or {}),
            "reason": str(reason),
        }

    def expand_options(self) -> List[TopologyAssemblyOption]:
        out: List[TopologyAssemblyOption] = []
        for chain in self.pending_branch_chains:
            meta = dict(chain.metadata or {})
            group_id = str(meta.get("candidate_group_id") or f"pending:{id(chain)}")
            alternatives = list(meta.get("branch_candidate_alternatives", []) or [])
            for idx, alt in enumerate(alternatives):
                out.extend(self._options_for_alt(chain, group_id, idx, dict(alt or {})))
        for idx, chain in enumerate(self.pending_ordinary_chains):
            out.extend(self._ordinary_options_for_chain(chain, idx))
        out.extend(self._ordinary_pair_options())
        return out

    def _options_for_alt(
        self,
        chain: ChainSpec,
        group_id: str,
        alt_idx: int,
        alt: Dict[str, Any],
    ) -> List[TopologyAssemblyOption]:
        family = str(alt.get("assembly_family") or "")
        if not family:
            return []
        usage_modes = [str(x) for x in list(alt.get("usage_modes", []) or [])]
        if not usage_modes:
            usage_modes = ["flex_branch"] if family.endswith("_M") else ["side_branch"]
        out: List[TopologyAssemblyOption] = []
        for usage in usage_modes:
            if usage in {"ring_only", "aliphatic_side_ring", "upper_aliphatic_ring", "fused_aliphatic_ring", "side_or_fused_aliphatic_ring"}:
                out.append(self._make_option(chain, group_id, alt_idx, alt, usage_mode="ring_candidate"))
                continue
            if usage not in {"flex_branch", "side_branch"}:
                continue
            out.extend(self._make_closure_options(chain, group_id, alt_idx, alt, usage))
        return out

    def _make_closure_options(
        self,
        chain: ChainSpec,
        group_id: str,
        alt_idx: int,
        alt: Dict[str, Any],
        usage_mode: str,
    ) -> List[TopologyAssemblyOption]:
        family = str(alt.get("assembly_family") or "")
        slot_intents = self._branch_slot_intents_for_alt(alt)
        branch_intent = self._branch_intent_for_alt(alt)
        main_intents = self._main_intents_for_family(family, usage_mode)
        branch_variants = self._intent_variants(branch_intent) if branch_intent else [()]
        slot_variant_groups = [
            (slot, self._intent_variants(intent))
            for slot, intent in slot_intents
        ]
        main_variants = [self._intent_variants(intent) for intent in main_intents]
        combos: List[Tuple[Tuple[int, ...], ...]] = [tuple()]
        for variants in main_variants:
            combos = [combo + (tuple(var),) for combo in combos for var in variants]
        slot_combos: List[Tuple[Tuple[str, Tuple[int, ...]], ...]] = [tuple()]
        for slot, variants in slot_variant_groups:
            slot_combos = [
                combo + ((str(slot), tuple(var)),)
                for combo in slot_combos
                for var in variants
            ]
        out: List[TopologyAssemblyOption] = []
        for branch_seq in branch_variants:
            for slot_combo in slot_combos:
                for main_combo in combos:
                    closure_plan: Dict[str, List[int]] = {}
                    if branch_seq:
                        closure_plan["branch"] = [int(x) for x in branch_seq]
                    for slot, seq in slot_combo:
                        if seq:
                            closure_plan[f"branch:{slot}"] = [int(x) for x in seq]
                    for idx, seq in enumerate(main_combo):
                        closure_plan[f"main{idx + 1}"] = [int(x) for x in seq]
                    out.append(self._make_option(
                        chain,
                        group_id,
                        alt_idx,
                        alt,
                        usage_mode=usage_mode,
                        closure_plan=closure_plan,
                    ))
        native24_h_attachment_allowed = bool(
            self.constraints.get("allow_native24_branch_h_attachment", False)
        )
        is_native24_alt = str(alt.get("source_kind") or "") == "native24"
        if str(branch_intent) == "H" and (not is_native24_alt or native24_h_attachment_allowed):
            for attach_idx, attach_chain in self._branch_h_attachment_chains():
                attached_alt = self._alt_with_branch_h_attachment(alt, attach_chain, attach_idx)
                for main_combo in combos:
                    closure_plan = {}
                    for idx, seq in enumerate(main_combo):
                        closure_plan[f"main{idx + 1}"] = [int(x) for x in seq]
                    out.append(self._make_option(
                        chain,
                        group_id,
                        alt_idx,
                        attached_alt,
                        usage_mode=usage_mode,
                        closure_plan=closure_plan,
                    ))
        return out

    def _branch_h_attachment_chains(self) -> List[Tuple[int, ChainSpec]]:
        out: List[Tuple[int, ChainSpec]] = []
        for idx, chain in enumerate(list(self.pending_ordinary_chains or [])):
            meta = dict(getattr(chain, "metadata", {}) or {})
            kind = str(meta.get("fragment_kind") or "").strip()
            if kind not in {"E", "G"}:
                continue
            left = str(meta.get("open_left") or "")
            right = str(meta.get("open_right") or "")
            if {left, right} != {"chain", "terminal"}:
                continue
            out.append((int(idx), chain))
        return out

    def _alt_with_branch_h_attachment(
        self,
        alt: Dict[str, Any],
        attach_chain: ChainSpec,
        attach_idx: int,
    ) -> Dict[str, Any]:
        out = dict(alt or {})
        attach_meta = dict(getattr(attach_chain, "metadata", {}) or {})
        attach_su = [
            int(x)
            for x in list(attach_meta.get("actual_used_su", []) or getattr(attach_chain, "composition", []) or [])
        ]
        attach_ids = [
            int(x)
            for x in list(attach_meta.get("actual_used_source_ids", []) or getattr(attach_chain, "source_ids", []) or [])
            if int(x) >= 0
        ]
        attach_display = str(
            attach_meta.get("open_topology")
            or attach_meta.get("display_topology")
            or "-".join(str(x) for x in attach_su)
        )
        out["branch_su"] = [
            int(x) for x in list(out.get("branch_su", []) or [])
        ] + list(attach_su)
        branch_slots = {
            str(k): [int(x) for x in list(v or [])]
            for k, v in dict(out.get("branch_slots_su", {}) or {}).items()
        }
        if branch_slots:
            branch_slots["H"] = list(branch_slots.get("H", []) or []) + list(attach_su)
            out["branch_slots_su"] = branch_slots
        out["branch_labels"] = [
            str(x) for x in list(out.get("branch_labels", []) or [])
        ] + [str(x) for x in list(attach_meta.get("labels", []) or attach_su)]
        out["actual_used_su"] = [
            int(x) for x in list(out.get("actual_used_su", []) or [])
        ] + list(attach_su)
        out["actual_used_source_ids"] = sorted(set(
            int(x)
            for x in list(out.get("actual_used_source_ids", []) or []) + list(attach_ids)
            if int(x) >= 0
        ))
        out["source_ids"] = sorted(set(
            int(x)
            for x in list(out.get("source_ids", []) or []) + list(getattr(attach_chain, "source_ids", []) or [])
            if int(x) >= 0
        ))
        out["open_topology"] = str(out.get("open_topology") or "").replace(
            "(-H)",
            f"(-{attach_display})",
            1,
        )
        if str(out.get("open_topology") or "") == str(alt.get("open_topology") or ""):
            out["open_topology"] = f"{out.get('open_topology', '')} branchH[{attach_display}]"
        intents = {
            str(k): list(v)
            for k, v in dict(out.get("closure_intents", {}) or {}).items()
        }
        intents.pop("branch", None)
        out["closure_intents"] = intents
        out["branch_open"] = False
        out["origin_rule"] = (
            f"{out.get('origin_rule', '')}+branchH:{attach_meta.get('origin_rule', attach_chain.origin_type)}"
        )
        out["pattern_rule"] = (
            f"{out.get('pattern_rule', '')}+branchH:{attach_meta.get('fragment_kind', '')}"
        )
        out["branch_h_attachment"] = True
        out["branch_h_attachment_index"] = int(attach_idx)
        out["branch_h_attachment_origin"] = str(attach_meta.get("origin_rule", attach_chain.origin_type))
        out["branch_h_attachment_source_ids"] = [
            int(x) for x in list(getattr(attach_chain, "source_ids", []) or []) if int(x) >= 0
        ]
        out["branch_closure_need"] = "none"
        out["priority_rank"] = max(int(out.get("priority_rank", 0) or 0), 90)
        return out

    def _make_option(
        self,
        chain: ChainSpec,
        group_id: str,
        alt_idx: int,
        alt: Dict[str, Any],
        usage_mode: str,
        closure_plan: Optional[Dict[str, List[int]]] = None,
    ) -> TopologyAssemblyOption:
        closure_plan = {str(k): [int(x) for x in v] for k, v in dict(closure_plan or {}).items()}
        base_cost = self._branch_base_cost(chain, alt)
        if str(usage_mode) == "ring_candidate":
            # Ring candidates are combined later.  The base candidate is paid by
            # the ring option, not by this placeholder.
            base_cost = {key: 0 for key in RESOURCE_KEYS}
        closure_cost = self._cost_for_sequences(closure_plan.values())
        resource_cost = {
            key: int(base_cost.get(key, 0)) + int(closure_cost.get(key, 0))
            for key in RESOURCE_KEYS
        }
        family = str(alt.get("assembly_family") or "")
        priority = int(alt.get("priority_rank") or 0)
        if family in {"24_AB_M", "24_CD_M_BRIDGE", "24_AB_N_AB_BRIDGE"}:
            priority = max(priority, 100)
        option_id = (
            f"{group_id}:alt{alt_idx}:{family}:{usage_mode}:"
            f"{self._closure_key(closure_plan)}"
        )
        return TopologyAssemblyOption(
            option_id=option_id,
            source_group_id=str(group_id),
            source_ids=self._source_ids_for_option(chain, alt),
            assembly_family=family,
            usage_mode=str(usage_mode),
            topology_template=str(alt.get("open_topology") or chain.metadata.get("open_topology") or ""),
            closure_plan=closure_plan,
            resource_cost=resource_cost,
            base_resource_cost={key: int(base_cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: int(closure_cost.get(key, 0)) for key in RESOURCE_KEYS},
            flex_count_delta=1 if str(usage_mode) == "flex_branch" else 0,
            side_count_delta=1 if str(usage_mode) == "side_branch" else 0,
            ring_count_delta=0,
            priority=int(priority),
            constraints={
                "ring_eligible": bool(alt.get("ring_eligible", False)),
                "ring_scope": str(alt.get("ring_scope") or ""),
                "disallow_ring_reason": str(alt.get("disallow_ring_reason") or ""),
                "exclusive_group_key": str(alt.get("exclusive_group_key") or group_id),
                "exclusive_group_keys": [
                    str(x) for x in list(alt.get("exclusive_group_keys", []) or []) if str(x)
                ],
                "candidate_group_id": str(group_id),
                "alt_source_ids": [
                    int(x) for x in list(alt.get("source_ids", []) or []) if int(x) >= 0
                ],
                "main_connector_kind": str(alt.get("main_connector_kind") or ""),
                "branch_connector_kind": str(alt.get("branch_connector_kind") or ""),
                "selected_fixed_role": str(alt.get("selected_fixed_role") or ""),
                "topology_role": str(alt.get("topology_role") or ""),
                "main_open_class": str(alt.get("main_open_class") or ""),
                "branch_closure_need": str(alt.get("branch_closure_need") or ""),
                "ring_usable": str(alt.get("ring_usable") or ""),
            },
            candidate_row=dict(alt),
            source_chain=chain,
        )

    def _branch_base_cost(self, chain: ChainSpec, alt: Dict[str, Any]) -> Dict[str, int]:
        actual_used = alt.get("actual_used_su")
        if isinstance(actual_used, (list, tuple)) and actual_used:
            cost = self._base_cost_for_sequence([int(x) for x in list(actual_used or [])])
            return self._apply_abstract_fixed22_cost(cost, alt)
        main = alt.get("main_body_su")
        branch = alt.get("branch_su")
        if not isinstance(main, (list, tuple)) and not isinstance(branch, (list, tuple)):
            cost = _chain_required_resource_counts(chain)
            return self._apply_abstract_fixed22_cost(cost, alt)
        vals: List[int] = []
        vals.extend(int(x) for x in list(main or []) if str(x) != "")
        vals.extend(int(x) for x in list(branch or []) if str(x) != "")
        cost = self._base_cost_for_sequence(vals)
        return self._apply_abstract_fixed22_cost(cost, alt)

    @staticmethod
    def _apply_abstract_fixed22_cost(cost: Dict[str, int], alt: Dict[str, Any]) -> Dict[str, int]:
        out = {key: int(dict(cost or {}).get(key, 0)) for key in RESOURCE_KEYS}
        if bool(dict(alt or {}).get("fixed_22_is_abstract", False)):
            branch_slots = dict(dict(alt or {}).get("branch_slots_su", {}) or {})
            fixed_slot = [int(x) for x in list(branch_slots.get("fixed22", []) or [])]
            fixed_count = int(sum(1 for x in fixed_slot if int(x) == 22)) or 1
            if int(out.get("22", 0)) < fixed_count:
                out["22"] = int(out.get("22", 0)) + (fixed_count - int(out.get("22", 0)))
        return out

    @staticmethod
    def _base_cost_for_sequence(values: Sequence[int]) -> Dict[str, int]:
        vals = [int(x) for x in list(values or [])]
        return {
            "11": int(sum(1 for x in vals if int(x) == 11)),
            "22": int(sum(1 for x in vals if int(x) == 22)),
            "23": int(sum(1 for x in vals if int(x) == 23)),
            "24": int(sum(1 for x in vals if int(x) in {14, 24})),
            "25": int(sum(1 for x in vals if int(x) == 25)),
        }

    @staticmethod
    def _planner_cost_for_sequence(values: Sequence[int]) -> Dict[str, int]:
        vals = [int(x) for x in list(values or [])]
        cost = BranchTopologyAssembler._base_cost_for_sequence(vals)
        # SU14 can be used as an explicit bridgehead substitute.  It consumes a
        # structural body slot but must not debit the raw SU24 filler pool.
        cost["24"] = int(sum(1 for x in vals if int(x) == 24))
        return cost

    def _ordinary_options_for_chain(self, chain: ChainSpec, idx: int) -> List[TopologyAssemblyOption]:
        meta = dict(getattr(chain, "metadata", {}) or {})
        group_id = str(meta.get("candidate_group_id") or f"ordinary:{idx}")
        intents = dict(meta.get("closure_intents", {}) or {})
        main_intents = [str(x) for x in list(intents.get("main", []) or [])]
        out: List[TopologyAssemblyOption] = []
        open_left = str(meta.get("open_left") or "")
        open_right = str(meta.get("open_right") or "")
        f1_intent = next((x for x in main_intents if x.startswith("F1")), "")
        f2_intent = next((x for x in main_intents if x.startswith("F2")), "")

        def add_option(usage: str, plan: Dict[str, List[int]], intent_label: str) -> None:
            closure_plan = {str(k): [int(x) for x in v] for k, v in dict(plan or {}).items()}
            closure_cost = self._cost_for_sequences(closure_plan.values())
            resource_cost = self._merge_costs(_chain_required_resource_counts(chain), closure_cost)
            priority = 75 if usage == "ordinary_bridge" else 65
            out.append(TopologyAssemblyOption(
                option_id=f"{group_id}:{usage}:{intent_label}:{len(out)}:{self._closure_key(closure_plan)}",
                source_group_id=group_id,
                source_ids=[int(x) for x in list(chain.source_ids or [])],
                assembly_family=f"ordinary_{intent_label}",
                usage_mode=usage,
                topology_template=str(meta.get("open_topology") or meta.get("raw_topology") or chain.origin_type),
                closure_plan=closure_plan,
                resource_cost={key: int(resource_cost.get(key, 0)) for key in RESOURCE_KEYS},
                base_resource_cost=_chain_required_resource_counts(chain),
                closure_resource_cost={key: int(closure_cost.get(key, 0)) for key in RESOURCE_KEYS},
                flex_count_delta=1 if usage == "ordinary_bridge" else 0,
                side_count_delta=1 if usage == "ordinary_side" else 0,
                priority=priority,
                constraints={
                    "candidate_group_id": group_id,
                    "ordinary_pending": True,
                    "closure_intent": str(intent_label),
                    "open_left": str(open_left),
                    "open_right": str(open_right),
                },
                candidate_row={"origin_rule": str(meta.get("origin_rule", "")), "intent": str(intent_label)},
                source_chain=chain,
            ))

        def variants(intent: str) -> List[Tuple[int, ...]]:
            return self._intent_variants(intent) if intent else []

        if open_left == "chain" and open_right == "chain":
            for left_seq in variants(f1_intent):
                for right_seq in variants(f1_intent):
                    add_option(
                        "ordinary_bridge",
                        {"left": list(left_seq), "right": list(right_seq)},
                        f"{f1_intent}+{f1_intent}",
                    )
            for left_seq in variants(f1_intent):
                for right_seq in variants(f2_intent):
                    add_option(
                        "ordinary_side",
                        {"left": list(left_seq), "right": list(right_seq)},
                        f"{f1_intent}+{f2_intent}",
                    )
        elif open_left == "chain" and open_right == "closed":
            for seq in variants(f1_intent):
                add_option("ordinary_bridge", {"left": list(seq)}, f1_intent)
            for seq in variants(f2_intent):
                add_option("ordinary_side", {"left": list(seq)}, f2_intent)
        elif open_left == "closed" and open_right == "chain":
            for seq in variants(f1_intent):
                add_option("ordinary_bridge", {"right": list(seq)}, f1_intent)
            for seq in variants(f2_intent):
                add_option("ordinary_side", {"right": list(seq)}, f2_intent)
        elif open_left == "chain" and open_right == "terminal":
            for seq in variants(f1_intent):
                add_option("ordinary_side", {"left": list(seq)}, f1_intent)
        elif open_left == "terminal" and open_right == "chain":
            for seq in variants(f1_intent):
                add_option("ordinary_side", {"right": list(seq)}, f1_intent)
        else:
            for intent in main_intents:
                usage = "ordinary_bridge" if str(intent).startswith("F1") else "ordinary_side"
                for seq in variants(intent):
                    add_option(usage, {"right": list(seq)}, intent)
        return out

    def _ordinary_pair_options(self) -> List[TopologyAssemblyOption]:
        chains = list(self.pending_ordinary_chains or [])
        out: List[TopologyAssemblyOption] = []
        for left_idx, left in enumerate(chains):
            for right_idx, right in enumerate(chains[left_idx + 1:], start=left_idx + 1):
                out.extend(self._ordinary_pair_options_for_pair(left_idx, left, right_idx, right))
        return out

    def _ordinary_pair_options_for_pair(
        self,
        left_idx: int,
        left: ChainSpec,
        right_idx: int,
        right: ChainSpec,
    ) -> List[TopologyAssemblyOption]:
        out: List[TopologyAssemblyOption] = []
        for left_oriented in self._ordinary_pair_left_orientations(left):
            for right_oriented in self._ordinary_pair_right_orientations(right):
                left_outer = str(left_oriented[1])
                right_outer = str(right_oriented[1])
                if set(left.source_ids) & set(right.source_ids):
                    continue
                left_seq = list(left_oriented[0])
                right_seq = list(right_oriented[0])
                if left_outer == "closed" and right_outer == "closed":
                    self._add_ordinary_pair_option(
                        out,
                        left_idx,
                        left,
                        right_idx,
                        right,
                        usage="ordinary_pair_bridge",
                        composition=list(left_seq) + list(right_seq),
                        pair_kind="B+B",
                    )
                elif {left_outer, right_outer} == {"closed", "terminal"}:
                    self._add_ordinary_pair_option(
                        out,
                        left_idx,
                        left,
                        right_idx,
                        right,
                        usage="ordinary_pair_side",
                        composition=list(left_seq) + list(right_seq),
                        pair_kind="B+E",
                    )
                elif left_outer == "chain" and right_outer == "closed":
                    self._add_pair_with_single_open_outer(
                        out, left_idx, left, right_idx, right,
                        left_seq, right_seq, open_side="left", closed_outer="closed",
                    )
                elif left_outer == "closed" and right_outer == "chain":
                    self._add_pair_with_single_open_outer(
                        out, left_idx, left, right_idx, right,
                        left_seq, right_seq, open_side="right", closed_outer="closed",
                    )
                elif left_outer == "chain" and right_outer == "terminal":
                    self._add_pair_with_single_open_outer(
                        out, left_idx, left, right_idx, right,
                        left_seq, right_seq, open_side="left", closed_outer="terminal",
                    )
                elif left_outer == "terminal" and right_outer == "chain":
                    self._add_pair_with_single_open_outer(
                        out, left_idx, left, right_idx, right,
                        left_seq, right_seq, open_side="right", closed_outer="terminal",
                    )
                elif left_outer == "chain" and right_outer == "chain":
                    self._add_pair_with_two_open_outers(
                        out, left_idx, left, right_idx, right, left_seq, right_seq,
                    )
        return out

    def _add_pair_with_single_open_outer(
        self,
        out: List[TopologyAssemblyOption],
        left_idx: int,
        left: ChainSpec,
        right_idx: int,
        right: ChainSpec,
        left_seq: Sequence[int],
        right_seq: Sequence[int],
        *,
        open_side: str,
        closed_outer: str,
    ) -> None:
        base = [int(x) for x in list(left_seq or [])] + [int(x) for x in list(right_seq or [])]
        if str(closed_outer) == "closed":
            for seq in self._intent_variants("F1"):
                comp = self._apply_outer_closure(base, str(open_side), seq)
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_bridge",
                    composition=comp,
                    pair_kind="open+closed:F1",
                )
            for seq in self._intent_variants("F2"):
                comp = self._apply_outer_closure(base, str(open_side), seq)
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_side",
                    composition=comp,
                    pair_kind="open+closed:F2",
                )
        elif str(closed_outer) == "terminal":
            for seq in self._intent_variants("F1"):
                comp = self._apply_outer_closure(base, str(open_side), seq)
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_side",
                    composition=comp,
                    pair_kind="open+terminal:F1",
                )

    def _add_pair_with_two_open_outers(
        self,
        out: List[TopologyAssemblyOption],
        left_idx: int,
        left: ChainSpec,
        right_idx: int,
        right: ChainSpec,
        left_seq: Sequence[int],
        right_seq: Sequence[int],
    ) -> None:
        base = [int(x) for x in list(left_seq or [])] + [int(x) for x in list(right_seq or [])]
        for left_close in self._intent_variants("F1"):
            for right_close in self._intent_variants("F1"):
                comp = self._apply_outer_closure(
                    self._apply_outer_closure(base, "left", left_close),
                    "right",
                    right_close,
                )
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_bridge",
                    composition=comp,
                    pair_kind="open+open:F1+F1",
                )
        for left_close in self._intent_variants("F1"):
            for right_close in self._intent_variants("F2"):
                comp = self._apply_outer_closure(
                    self._apply_outer_closure(base, "left", left_close),
                    "right",
                    right_close,
                )
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_side",
                    composition=comp,
                    pair_kind="open+open:F1+F2",
                )
        for left_close in self._intent_variants("F2"):
            for right_close in self._intent_variants("F1"):
                comp = self._apply_outer_closure(
                    self._apply_outer_closure(base, "left", left_close),
                    "right",
                    right_close,
                )
                self._add_ordinary_pair_option(
                    out,
                    left_idx,
                    left,
                    right_idx,
                    right,
                    usage="ordinary_pair_side",
                    composition=comp,
                    pair_kind="open+open:F2+F1",
                )

    @staticmethod
    def _apply_outer_closure(
        composition: Sequence[int],
        side: str,
        closure: Sequence[int],
    ) -> List[int]:
        comp = [int(x) for x in list(composition or [])]
        seq = [int(x) for x in list(closure or [])]
        if str(side) == "left":
            return list(reversed(seq)) + list(comp)
        return list(comp) + list(seq)

    def _add_ordinary_pair_option(
        self,
        out: List[TopologyAssemblyOption],
        left_idx: int,
        left: ChainSpec,
        right_idx: int,
        right: ChainSpec,
        usage: str,
        composition: Sequence[int],
        pair_kind: str,
    ) -> None:
        ctype = "bridge" if str(usage) == "ordinary_pair_bridge" else "side"
        comp = [int(x) for x in list(composition or [])]
        if ctype == "side":
            comp = self._orient_side_composition(comp)
        left_meta = dict(getattr(left, "metadata", {}) or {})
        right_meta = dict(getattr(right, "metadata", {}) or {})
        group_id = f"ordinary_pair:{left_idx}+{right_idx}"
        source_ids = sorted(set(int(x) for x in list(left.source_ids) + list(right.source_ids) if int(x) >= 0))
        source = ChainSpec(
            chain_type=ctype,
            composition=list(comp),
            origin_type=f"PendingPair:{pair_kind}",
            source_ids=source_ids,
            metadata={
                "allocator_mode": "ordinary_pair_closure_v1",
                "pending_closure": True,
                "pair_kind": str(pair_kind),
                "left_origin_rule": str(left_meta.get("origin_rule", "")),
                "right_origin_rule": str(right_meta.get("origin_rule", "")),
                "left_open_topology": str(left_meta.get("open_topology", "")),
                "right_open_topology": str(right_meta.get("open_topology", "")),
                "main_body_su": [int(x) for x in comp[1:-1]] if ctype == "bridge" and len(comp) >= 2 else [int(x) for x in comp[1:]],
            },
        )
        resource_cost = _chain_required_resource_counts(source)
        out.append(TopologyAssemblyOption(
            option_id=f"{group_id}:{usage}:{pair_kind}:{len(out)}",
            source_group_id=group_id,
            source_ids=source_ids,
            assembly_family=f"ordinary_pair_{pair_kind}",
            usage_mode=str(usage),
            topology_template=f"{left_meta.get('open_topology', left.origin_type)} + {right_meta.get('open_topology', right.origin_type)}",
            closure_plan={},
            resource_cost={key: int(resource_cost.get(key, 0)) for key in RESOURCE_KEYS},
            base_resource_cost={key: int(resource_cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: 0 for key in RESOURCE_KEYS},
            flex_count_delta=1 if ctype == "bridge" else 0,
            side_count_delta=1 if ctype == "side" else 0,
            priority=85 if ctype == "bridge" else 80,
            constraints={
                "candidate_group_id": group_id,
                "ordinary_pair": True,
                "pair_kind": str(pair_kind),
                "exclusive_group_key": group_id,
            },
            candidate_row={
                "origin_rule": f"{left_meta.get('origin_rule', '')}+{right_meta.get('origin_rule', '')}",
                "pair_kind": str(pair_kind),
            },
            source_chain=source,
        ))

    @staticmethod
    def _ordinary_pair_left_orientations(chain: ChainSpec) -> List[Tuple[List[int], str]]:
        meta = dict(getattr(chain, "metadata", {}) or {})
        comp = [int(x) for x in list(getattr(chain, "composition", []) or [])]
        left_open = str(meta.get("open_left") or "")
        right_open = str(meta.get("open_right") or "")
        out: List[Tuple[List[int], str]] = []
        if right_open == "chain":
            out.append((list(comp), left_open))
        if left_open == "chain":
            out.append((list(reversed(comp)), right_open))
        return out

    @staticmethod
    def _ordinary_pair_right_orientations(chain: ChainSpec) -> List[Tuple[List[int], str]]:
        meta = dict(getattr(chain, "metadata", {}) or {})
        comp = [int(x) for x in list(getattr(chain, "composition", []) or [])]
        left_open = str(meta.get("open_left") or "")
        right_open = str(meta.get("open_right") or "")
        out: List[Tuple[List[int], str]] = []
        if left_open == "chain":
            out.append((list(comp), right_open))
        if right_open == "chain":
            out.append((list(reversed(comp)), left_open))
        return out

    def _build_ring_options(
        self,
        all_options: Sequence[TopologyAssemblyOption],
        used_groups: set[str],
    ) -> List[TopologyAssemblyOption]:
        ring_candidates = self._dedupe_ring_candidates([
            opt for opt in all_options
            if str(opt.usage_mode) == "ring_candidate"
            and str(opt.source_group_id) not in used_groups
            and bool(opt.constraints.get("ring_eligible", False))
            and not str(opt.constraints.get("disallow_ring_reason") or "")
        ])
        out: List[TopologyAssemblyOption] = []
        out.extend(self._build_fused_ring_options(ring_candidates))
        out.extend(self._build_side_ring_options(ring_candidates))
        out.extend(self._build_vertical_ring_options(ring_candidates))
        return out

    @staticmethod
    def _dedupe_ring_candidates(candidates: Sequence[TopologyAssemblyOption]) -> List[TopologyAssemblyOption]:
        out: List[TopologyAssemblyOption] = []
        seen: set[Tuple[str, str, str, Tuple[int, ...]]] = set()
        for opt in list(candidates or []):
            alt = dict(opt.candidate_row or {})
            key = (
                str(opt.source_group_id),
                str(opt.assembly_family),
                str(alt.get("origin_rule") or ""),
                tuple(int(x) for x in list(opt.source_ids or [])),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(opt)
        return out

    def _build_fused_ring_options(
        self,
        ring_candidates: Sequence[TopologyAssemblyOption],
    ) -> List[TopologyAssemblyOption]:
        units = [
            unit for unit in (
                self._ring_unit_for_candidate(opt, planner="fused")
                for opt in list(ring_candidates or [])
            )
            if unit is not None
        ]
        if not units:
            return []
        units.sort(key=self._planner_unit_sort_key)
        pool_limit = max(1, int(self.constraints.get("fused_ring_candidate_pool", 12) or 12))
        pool = units[:pool_limit]
        out: List[TopologyAssemblyOption] = []
        seen: set[Tuple[Any, ...]] = set()
        max_units = min(4, len(pool))
        for size in range(max_units, 0, -1):
            for combo in itertools.combinations(pool, size):
                if self._side_ring_unit_source_conflict(combo):
                    continue
                option = self._make_fused_ring_option(combo)
                if option is None:
                    continue
                key = (
                    tuple(sorted(str(unit["group_id"]) for unit in combo)),
                    tuple(option.constraints.get("base_ring_su", []) or []),
                    tuple(option.constraints.get("outer_ring_su", []) or []),
                    tuple(
                        tuple(int(x) for x in list((option.constraints.get("slot_branch_su", {}) or {}).get(slot, []) or []))
                        for slot in ("base_upper", "base_lower", "outer_upper", "outer_lower")
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(option)
        return out

    def _build_vertical_ring_options(
        self,
        ring_candidates: Sequence[TopologyAssemblyOption],
    ) -> List[TopologyAssemblyOption]:
        units = [
            unit for unit in (
                self._ring_unit_for_candidate(opt, planner="vertical")
                for opt in list(ring_candidates or [])
            )
            if unit is not None
        ]
        if not units:
            return []
        units.sort(key=self._planner_unit_sort_key)
        pool_limit = max(1, int(self.constraints.get("vertical_ring_candidate_pool", 12) or 12))
        pool = units[:pool_limit]
        out: List[TopologyAssemblyOption] = []
        seen: set[Tuple[Any, ...]] = set()
        max_units = min(2, len(pool))
        for size in range(max_units, 0, -1):
            for combo in itertools.combinations(pool, size):
                if self._side_ring_unit_source_conflict(combo):
                    continue
                option = self._make_vertical_ring_option(combo)
                if option is None:
                    continue
                key = (
                    tuple(sorted(str(unit["group_id"]) for unit in combo)),
                    tuple(option.constraints.get("vertical_ring_su", []) or []),
                    tuple(
                        tuple(int(x) for x in list((option.constraints.get("slot_branch_su", {}) or {}).get(slot, []) or []))
                        for slot in ("right", "left")
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(option)
        return out

    def _ring_unit_for_candidate(self, option: TopologyAssemblyOption, *, planner: str) -> Optional[Dict[str, Any]]:
        planner_name = str(planner)
        if planner_name == "fused":
            if not self._candidate_can_build_fused_ring(option):
                return None
            rank = self._fused_ring_family_rank(str(option.assembly_family))
        elif planner_name == "vertical":
            if not self._candidate_can_build_vertical_ring(option):
                return None
            rank = self._vertical_ring_family_rank(str(option.assembly_family))
        else:
            return None

        family = str(option.assembly_family)
        alt = dict(option.candidate_row or {})
        source_ids = sorted(set(int(x) for x in list(option.source_ids or []) if int(x) >= 0))
        entries: List[Dict[str, Any]] = []
        if family in {"24_AB_N_CD_BRIDGE", "24_CD_N_CD_BRIDGE"}:
            main = self._alt_main_body(option)
            left_type = str(alt.get("bridge_left_abcd_type") or "").strip()
            right_type = str(alt.get("bridge_right_abcd_type") or "").strip()
            left_gid = int(alt.get("bridge_left_gid", -1) or -1)
            right_gid = int(alt.get("bridge_right_gid", -1) or -1)
            if left_type in {"24_A", "24_B", "24_C", "24_D"}:
                entries.append(self._side_ring_entry_for_candidate(
                    option,
                    abcd_type=left_type,
                    su_type=int(main[0]) if main else self._ring_slot_su_for_candidate(option),
                    source_ids=[left_gid] if left_gid >= 0 else source_ids,
                    entry_index=0,
                ))
            if right_type in {"24_A", "24_B", "24_C", "24_D"}:
                entries.append(self._side_ring_entry_for_candidate(
                    option,
                    abcd_type=right_type,
                    su_type=int(main[-1]) if main else self._ring_slot_su_for_candidate(option),
                    source_ids=[right_gid] if right_gid >= 0 else source_ids,
                    entry_index=1,
                ))
        else:
            abcd_type = self._primary_abcd_type(alt)
            if abcd_type in {"24_A", "24_B", "24_C", "24_D"}:
                entries.append(self._side_ring_entry_for_candidate(
                    option,
                    abcd_type=abcd_type,
                    su_type=self._ring_slot_su_for_candidate(option),
                    source_ids=source_ids,
                    entry_index=0,
                ))
        entries = [entry for entry in entries if entry]
        if not entries and not bool(alt.get("bridgehead_replace_24c_eligible", False)):
            return None
        return {
            "option": option,
            "group_id": str(option.source_group_id),
            "source_ids": source_ids,
            "assembly_family": family,
            "rank": int(rank),
            "entries": entries,
            "bridgehead_replace_24c_eligible": bool(alt.get("bridgehead_replace_24c_eligible", False)),
        }

    def _candidate_can_build_fused_ring(self, option: TopologyAssemblyOption) -> bool:
        alt = dict(option.candidate_row or {})
        usage_modes = {str(x) for x in list(alt.get("usage_modes", []) or [])}
        text = " ".join([
            str(option.constraints.get("ring_scope") or ""),
            str(alt.get("ring_scope") or ""),
            str(option.constraints.get("ring_usable") or ""),
            str(alt.get("ring_usable") or ""),
            " ".join(sorted(usage_modes)),
        ]).lower()
        bridgehead = bool(alt.get("bridgehead_replace_24c_eligible", False))
        if "fused" not in text and not bridgehead:
            return False
        family = str(option.assembly_family)
        return family in {
            "24_AB_N_BRANCH",
            "24_CD_N_BRANCH",
            "24_AB_N_CD_BRIDGE",
            "24_CD_N_CD_BRIDGE",
            "24_CD_N_14_MAIN",
            "24_CD_N_MAIN",
        }

    def _candidate_can_build_vertical_ring(self, option: TopologyAssemblyOption) -> bool:
        alt = dict(option.candidate_row or {})
        usage_modes = {str(x) for x in list(alt.get("usage_modes", []) or [])}
        text = " ".join([
            str(option.constraints.get("ring_scope") or ""),
            str(alt.get("ring_scope") or ""),
            str(option.constraints.get("ring_usable") or ""),
            str(alt.get("ring_usable") or ""),
            " ".join(sorted(usage_modes)),
        ]).lower()
        if "upper" not in text and "vertical" not in text:
            return False
        family = str(option.assembly_family)
        return family in {
            "24_CD_N_CD_BRIDGE",
            "24_CD_N_BRANCH",
            "24_CD_N_MAIN",
        }

    @staticmethod
    def _fused_ring_family_rank(family: str) -> int:
        family = str(family)
        if family in {"24_AB_N_BRANCH", "24_CD_N_BRANCH", "24_CD_N_CD_BRIDGE"}:
            return 0
        if family in {"24_AB_N_CD_BRIDGE", "24_CD_N_14_MAIN"}:
            return 1
        if family == "24_CD_N_MAIN":
            return 2
        return 3

    @staticmethod
    def _vertical_ring_family_rank(family: str) -> int:
        family = str(family)
        if family == "24_CD_N_CD_BRIDGE":
            return 0
        if family == "24_CD_N_BRANCH":
            return 1
        if family == "24_CD_N_MAIN":
            return 2
        return 3

    @staticmethod
    def _planner_unit_sort_key(unit: Dict[str, Any]) -> Tuple[int, int, int, str]:
        return (
            int(unit.get("rank", 9)),
            0 if bool(unit.get("bridgehead_replace_24c_eligible", False)) else 1,
            -int(len(unit.get("entries", []) or [])),
            str(unit.get("group_id") or ""),
        )

    def _make_fused_ring_option(self, units: Sequence[Dict[str, Any]]) -> Optional[TopologyAssemblyOption]:
        units_list = sorted(list(units or []), key=self._planner_unit_sort_key)
        if not units_list:
            return None
        bridge_unit = next((unit for unit in units_list if bool(unit.get("bridgehead_replace_24c_eligible", False))), None)
        entries: List[Dict[str, Any]] = []
        for unit in units_list:
            if bridge_unit is not None and str(unit.get("group_id")) == str(bridge_unit.get("group_id")):
                continue
            entries.extend(list(unit.get("entries", []) or []))
        entries.sort(key=lambda entry: (
            0 if self._is_ab_type(str(entry.get("abcd_type") or "")) else 1,
            int(entry.get("rank", 9)),
            str(entry.get("unit_group_id") or ""),
            int(entry.get("entry_index", 0)),
        ))
        ab_entries = [entry for entry in entries if self._is_ab_type(str(entry.get("abcd_type") or ""))]
        cd_entries = [entry for entry in entries if self._is_cd_type(str(entry.get("abcd_type") or ""))]
        if len(entries) != 4 or len(ab_entries) != 2 or len(cd_entries) != 2:
            return None
        base_upper = ab_entries[0]
        base_lower = ab_entries[1]
        outer_upper = cd_entries[0]
        outer_lower = cd_entries[1]

        bridge_su = [14, 14] if bridge_unit is not None else [24, 24]
        base_ring_su = [
            int(base_upper.get("su_type", 24)) if base_upper else 24,
            int(bridge_su[0]),
            int(bridge_su[1]),
            int(base_lower.get("su_type", 24)) if base_lower else 24,
        ]
        outer_ring_su: List[int] = [
            23,
            int(outer_upper.get("su_type", 24)),
            int(outer_lower.get("su_type", 24)),
            23,
        ]
        total_like24 = int(sum(1 for su in list(base_ring_su) + list(outer_ring_su) if self._is_like24_ring_su(int(su))))
        if not (4 <= int(total_like24) <= 6):
            return None

        slot_entries = {
            "base_upper": base_upper,
            "base_lower": base_lower,
            "outer_upper": outer_upper,
            "outer_lower": outer_lower,
        }
        slot_branch_su = {
            slot: [int(x) for x in list(entry.get("branch_su", []) or [])]
            for slot, entry in slot_entries.items()
            if entry is not None and entry.get("branch_su")
        }
        base_upper_branch = list(slot_branch_su.get("base_upper", []) or [])
        base_lower_branch = list(slot_branch_su.get("base_lower", []) or [])
        outer_upper_branch = list(slot_branch_su.get("outer_upper", []) or [])
        outer_lower_branch = list(slot_branch_su.get("outer_lower", []) or [])
        fused_filler_24_count = 0 if bridge_unit is not None else 2
        composition = (
            list(base_ring_su)
            + list(outer_ring_su)
            + list(base_upper_branch)
            + list(base_lower_branch)
            + list(outer_upper_branch)
            + list(outer_lower_branch)
        )
        cost = self._planner_cost_for_sequence(composition)
        cost["11"] = int(cost.get("11", 0)) + 2

        source_ids = sorted(set(
            int(src)
            for unit in units_list
            for src in list(unit.get("source_ids", []) or [])
            if int(src) >= 0
        ))
        slot_group_ids = [str(unit.get("group_id") or "") for unit in units_list if str(unit.get("group_id") or "")]
        families = [str(unit.get("assembly_family") or "") for unit in units_list if str(unit.get("assembly_family") or "")]
        family_text = "+".join(families) or "fused_ring"
        group_id = f"fused_ring:{'+'.join(slot_group_ids)}"

        profiles: List[Dict[str, Any]] = []
        tail_sources: List[Optional[str]] = []
        branch_tail_lengths: Dict[str, int] = {}

        def profile_for(entry: Optional[Dict[str, Any]], role: str, slot: str, su_type: int) -> Dict[str, Any]:
            profile = dict((entry or {}).get("profile", {}) or {})
            profile.update({
                "allocator_role": str(role),
                "fused_slot_name": str(slot),
                "abcd_type": str((entry or {}).get("abcd_type") or ""),
                "su_type": int(su_type),
                "branch_su": [int(x) for x in list(slot_branch_su.get(slot, []) or [])],
                "main_body_su": [int(su_type)],
                "source_ids": [int(x) for x in list((entry or {}).get("source_ids", []) or []) if int(x) >= 0],
                "assembly_family": str((entry or {}).get("assembly_family") or "filler"),
                "usage_mode": "ring",
                "origin_rule": str((entry or {}).get("origin_rule") or ""),
            })
            return profile

        profiles.append(profile_for(base_upper, "fused_base", "base_upper", base_ring_su[0]))
        profiles.append(profile_for(base_lower, "fused_base", "base_lower", base_ring_su[3]))
        bridge_source_ids = [int(x) for x in list((bridge_unit or {}).get("source_ids", []) or []) if int(x) >= 0]
        for idx, su in enumerate(bridge_su):
            profiles.append({
                "allocator_role": "fused_fixed_c",
                "fused_slot_name": "bridge_upper" if idx == 0 else "bridge_lower",
                "abcd_type": "24_C",
                "su_type": int(su),
                "branch_su": [],
                "main_body_su": [int(su)],
                "source_ids": list(bridge_source_ids),
                "assembly_family": str((bridge_unit or {}).get("assembly_family") or "raw_24_bridgehead"),
                "usage_mode": "ring",
                "origin_rule": "14-14_bridgehead" if bridge_unit is not None else "raw_su24_bridgehead",
            })
        if outer_ring_su:
            profiles.append(profile_for(outer_upper, "fused_outer", "outer_upper", outer_ring_su[1]))
            profiles.append(profile_for(outer_lower, "fused_outer", "outer_lower", outer_ring_su[2]))
        for slot in ("base_upper", "base_lower", "outer_upper", "outer_lower"):
            vals = list(slot_branch_su.get(slot, []) or [])
            tail_sources.append(str((slot_entries.get(slot) or {}).get("origin_rule") or "") or None)
            if vals:
                branch_tail_lengths[slot] = int(len(vals))

        meta = {
            "allocator_mode": "fused_ring_planner_v1",
            "pending_closure": False,
            "closure_stage_required": False,
            "topology_state": "closed_ring_topology",
            "usage_mode": "ring",
            "ring_type": "fused_side_ring",
            "ring_planner": "fused_aliphatic_ring_v1",
            "assembly_family": f"fused_ring({family_text})",
            "source_group_id": group_id,
            "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
            "resolved_topology": f"fused_ring[base={','.join(str(x) for x in base_ring_su)};outer={','.join(str(x) for x in outer_ring_su)}]",
            "resource_requirements_override": {key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            "main_body_su": [int(x) for x in list(base_ring_su) + list(outer_ring_su)],
            "base_ring_su": [int(x) for x in base_ring_su],
            "outer_ring_su": [int(x) for x in outer_ring_su],
            "branch_tail_lengths": dict(branch_tail_lengths),
            "tail_sources": list(tail_sources),
            "branch24_profiles": profiles,
            "slot_branch_su": dict(slot_branch_su),
            "fused_ring_like24_count": int(total_like24),
            "fused_ring_like24_min": 4,
            "fused_ring_like24_max": 6,
            "fused_ring_filler_24_count": int(fused_filler_24_count),
            "bridgehead_replace_24c": bool(bridge_unit is not None),
            "ring_substitute_types": [14, 14] if bridge_unit is not None else [],
            "source_candidate_group_ids": list(slot_group_ids),
        }
        chain = ChainSpec(
            chain_type="fused_side_ring",
            composition=[int(x) for x in composition],
            origin_type=f"Closed-fused_ring({family_text})",
            source_ids=[int(x) for x in source_ids],
            metadata=meta,
        )
        priority = 125 - int(sum(int(unit.get("rank", 0)) for unit in units_list)) + int(len(entries)) * 4
        if bridge_unit is not None:
            priority += 5
        option = TopologyAssemblyOption(
            option_id=f"{group_id}:base_outer",
            source_group_id=group_id,
            source_ids=[int(x) for x in source_ids],
            assembly_family=f"fused_ring({family_text})",
            usage_mode="ring",
            topology_template=str(meta["resolved_topology"]),
            closure_plan={"fused_ring": [int(x) for x in list(base_ring_su) + list(outer_ring_su)]},
            resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            base_resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: 0 for key in RESOURCE_KEYS},
            ring_count_delta=1,
            priority=int(priority),
            constraints={
                "ring_type": "fused_side_ring",
                "ring_planner": "fused_aliphatic_ring_v1",
                "slot_group_ids": list(slot_group_ids),
                "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
                "base_ring_su": [int(x) for x in base_ring_su],
                "outer_ring_su": [int(x) for x in outer_ring_su],
                "slot_branch_su": dict(slot_branch_su),
                "fused_ring_like24_count": int(total_like24),
                "fused_ring_filler_24_count": int(fused_filler_24_count),
                "min_like24_ok": bool(4 <= int(total_like24) <= 6),
                "bridgehead_replace_24c": bool(bridge_unit is not None),
            },
            candidate_row={
                "ring_planner": "fused_aliphatic_ring_v1",
                "source_candidate_group_ids": list(slot_group_ids),
                "assembly_families": list(families),
            },
            source_chain=chain,
        )
        chain.metadata["assembly_option"] = option.to_row()
        return option

    def _make_vertical_ring_option(self, units: Sequence[Dict[str, Any]]) -> Optional[TopologyAssemblyOption]:
        units_list = sorted(list(units or []), key=self._planner_unit_sort_key)
        entries: List[Dict[str, Any]] = []
        for unit in units_list:
            entries.extend(list(unit.get("entries", []) or []))
        entries = [entry for entry in entries if str(entry.get("abcd_type") or "") in {"24_C", "24_D"}]
        entries.sort(key=lambda entry: (
            int(entry.get("rank", 9)),
            str(entry.get("unit_group_id") or ""),
            int(entry.get("entry_index", 0)),
        ))
        if not entries:
            return None
        right = entries[0]
        left = entries[1] if len(entries) > 1 else None
        vertical_ring_su = [
            24,
            23,
            int(right.get("su_type", 24)),
            23,
            int(left.get("su_type", 23)) if left is not None else 23,
            23,
        ]
        vertical_filler_24_count = 1
        total_like24 = int(sum(1 for su in vertical_ring_su if self._is_like24_ring_su(int(su))))
        if not (2 <= int(total_like24) <= 3):
            return None
        slot_entries = {"right": right, "left": left}
        slot_branch_su = {
            slot: [int(x) for x in list(entry.get("branch_su", []) or [])]
            for slot, entry in slot_entries.items()
            if entry is not None and entry.get("branch_su")
        }
        right_branch = list(slot_branch_su.get("right", []) or [])
        left_branch = list(slot_branch_su.get("left", []) or [])
        composition = list(vertical_ring_su) + list(right_branch) + list(left_branch)
        cost = self._planner_cost_for_sequence(composition)
        cost["11"] = int(cost.get("11", 0)) + 1
        source_ids = sorted(set(
            int(src)
            for unit in units_list
            for src in list(unit.get("source_ids", []) or [])
            if int(src) >= 0
        ))
        slot_group_ids = [str(unit.get("group_id") or "") for unit in units_list if str(unit.get("group_id") or "")]
        families = [str(unit.get("assembly_family") or "") for unit in units_list if str(unit.get("assembly_family") or "")]
        family_text = "+".join(families) or "vertical_ring"
        group_id = f"vertical_ring:{'+'.join(slot_group_ids)}"
        profiles: List[Dict[str, Any]] = []
        tail_sources: List[Optional[str]] = []
        branch_tail_lengths: Dict[str, int] = {}

        def profile_for(entry: Optional[Dict[str, Any]], slot: str, su_type: int) -> Dict[str, Any]:
            profile = dict((entry or {}).get("profile", {}) or {})
            profile.update({
                "allocator_role": "vertical_inter",
                "vertical_slot_name": str(slot),
                "abcd_type": str((entry or {}).get("abcd_type") or ""),
                "su_type": int(su_type),
                "branch_su": [int(x) for x in list(slot_branch_su.get(slot, []) or [])],
                "main_body_su": [int(su_type)],
                "source_ids": [int(x) for x in list((entry or {}).get("source_ids", []) or []) if int(x) >= 0],
                "assembly_family": str((entry or {}).get("assembly_family") or "filler"),
                "usage_mode": "ring",
                "origin_rule": str((entry or {}).get("origin_rule") or ""),
            })
            return profile

        profiles.append(profile_for(right, "right", vertical_ring_su[2]))
        profiles.append(profile_for(left, "left", vertical_ring_su[4]))
        for slot in ("right", "left"):
            vals = list(slot_branch_su.get(slot, []) or [])
            tail_sources.append(str((slot_entries.get(slot) or {}).get("origin_rule") or "") or None)
            if vals:
                branch_tail_lengths[slot] = int(len(vals))

        meta = {
            "allocator_mode": "vertical_ring_planner_v1",
            "pending_closure": False,
            "closure_stage_required": False,
            "topology_state": "closed_ring_topology",
            "usage_mode": "ring",
            "ring_type": "vertical_ring",
            "ring_planner": "vertical_aliphatic_ring_v1",
            "assembly_family": f"vertical_ring({family_text})",
            "source_group_id": group_id,
            "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
            "resolved_topology": f"vertical_ring[{','.join(str(x) for x in vertical_ring_su)}]",
            "resource_requirements_override": {key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            "main_body_su": [int(x) for x in vertical_ring_su],
            "vertical_ring_su": [int(x) for x in vertical_ring_su],
            "branch_tail_lengths": dict(branch_tail_lengths),
            "tail_sources": list(tail_sources),
            "branch24_profiles": profiles,
            "slot_branch_su": dict(slot_branch_su),
            "vertical_ring_like24_count": int(total_like24),
            "vertical_ring_like24_min": 2,
            "vertical_ring_like24_max": 3,
            "vertical_ring_filler_24_count": int(vertical_filler_24_count),
            "source_candidate_group_ids": list(slot_group_ids),
        }
        chain = ChainSpec(
            chain_type="vertical_ring",
            composition=[int(x) for x in composition],
            origin_type=f"Closed-vertical_ring({family_text})",
            source_ids=[int(x) for x in source_ids],
            metadata=meta,
        )
        option = TopologyAssemblyOption(
            option_id=f"{group_id}:vertical",
            source_group_id=group_id,
            source_ids=[int(x) for x in source_ids],
            assembly_family=f"vertical_ring({family_text})",
            usage_mode="ring",
            topology_template=str(meta["resolved_topology"]),
            closure_plan={"vertical_ring": [int(x) for x in vertical_ring_su]},
            resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            base_resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: 0 for key in RESOURCE_KEYS},
            ring_count_delta=1,
            priority=85 - int(sum(int(unit.get("rank", 0)) for unit in units_list)) + int(len(entries)) * 4,
            constraints={
                "ring_type": "vertical_ring",
                "ring_planner": "vertical_aliphatic_ring_v1",
                "slot_group_ids": list(slot_group_ids),
                "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
                "vertical_ring_su": [int(x) for x in vertical_ring_su],
                "slot_branch_su": dict(slot_branch_su),
                "vertical_ring_like24_count": int(total_like24),
                "vertical_ring_filler_24_count": int(vertical_filler_24_count),
                "min_like24_ok": bool(2 <= int(total_like24) <= 3),
            },
            candidate_row={
                "ring_planner": "vertical_aliphatic_ring_v1",
                "source_candidate_group_ids": list(slot_group_ids),
                "assembly_families": list(families),
            },
            source_chain=chain,
        )
        chain.metadata["assembly_option"] = option.to_row()
        return option

    def _build_side_ring_options(
        self,
        ring_candidates: Sequence[TopologyAssemblyOption],
    ) -> List[TopologyAssemblyOption]:
        units = [
            unit for unit in (
                self._side_ring_unit_for_candidate(opt)
                for opt in list(ring_candidates or [])
            )
            if unit is not None
        ]
        if not units:
            return []
        units.sort(key=self._side_ring_unit_sort_key)
        configured_pool = self.constraints.get("side_ring_candidate_pool")
        if configured_pool is None:
            pool_limit = len(units) if len(units) <= 24 else 18
        else:
            pool_limit = max(1, int(configured_pool or 18))
        pool = units[:pool_limit]
        out: List[TopologyAssemblyOption] = []
        seen: set[Tuple[Any, ...]] = set()
        max_units = min(4, len(pool))
        for size in range(max_units, 0, -1):
            for combo in itertools.combinations(pool, size):
                entry_count = int(sum(
                    self._side_ring_entry_span(entry)
                    for unit in combo
                    for entry in list(unit.get("entries", []) or [])
                ))
                if entry_count <= 0 or entry_count > 4:
                    continue
                if self._side_ring_unit_source_conflict(combo):
                    continue
                for variant in (0, 1):
                    slot_entries = self._side_ring_slot_assignment(combo, variant)
                    if not slot_entries:
                        continue
                    option = self._make_side_ring_option(combo, slot_entries, variant)
                    if not bool(option.constraints.get("min_like24_ok", False)):
                        continue
                    key = (
                        tuple(sorted(str(unit["group_id"]) for unit in combo)),
                        tuple(option.constraints.get("side_ring_su", []) or []),
                        tuple(str(x) for x in list(option.constraints.get("side_ring_node_types", []) or [])),
                        tuple(
                            tuple(int(x) for x in list((option.constraints.get("slot_branch_su", {}) or {}).get(slot, []) or []))
                            for slot in ("pos1", "pos3", "pos4", "pos2")
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(option)
        return out

    @staticmethod
    def _side_ring_unit_source_conflict(units: Sequence[Dict[str, Any]]) -> bool:
        groups: set[str] = set()
        source_ids: set[int] = set()
        for unit in list(units or []):
            group_id = str(unit.get("group_id") or "")
            if group_id in groups:
                return True
            groups.add(group_id)
            src = {int(x) for x in list(unit.get("source_ids", []) or []) if int(x) >= 0}
            if src & source_ids:
                return True
            source_ids.update(src)
        return False

    def _side_ring_unit_for_candidate(self, option: TopologyAssemblyOption) -> Optional[Dict[str, Any]]:
        if not self._candidate_can_build_side_ring(option):
            return None
        family = str(option.assembly_family)
        alt = dict(option.candidate_row or {})
        source_ids = sorted(set(int(x) for x in list(option.source_ids or []) if int(x) >= 0))
        entries: List[Dict[str, Any]] = []

        if family in {"24_AB_N_CD_BRIDGE", "24_CD_N_CD_BRIDGE"}:
            main = self._alt_main_body(option)
            left_type = str(alt.get("bridge_left_abcd_type") or "").strip()
            right_type = str(alt.get("bridge_right_abcd_type") or "").strip()
            left_gid = int(alt.get("bridge_left_gid", -1) or -1)
            right_gid = int(alt.get("bridge_right_gid", -1) or -1)
            if left_type in {"24_A", "24_B", "24_C", "24_D"}:
                entries.append(self._side_ring_entry_for_candidate(
                    option,
                    abcd_type=left_type,
                    su_type=int(main[0]) if main else self._ring_slot_su_for_candidate(option),
                    source_ids=[left_gid] if left_gid >= 0 else source_ids,
                    entry_index=0,
                ))
            if right_type in {"24_A", "24_B", "24_C", "24_D"}:
                entries.append(self._side_ring_entry_for_candidate(
                    option,
                    abcd_type=right_type,
                    su_type=int(main[-1]) if main else self._ring_slot_su_for_candidate(option),
                    source_ids=[right_gid] if right_gid >= 0 else source_ids,
                    entry_index=1,
                ))
        else:
            abcd_type = self._primary_abcd_type(alt)
            if abcd_type not in {"24_A", "24_B", "24_C", "24_D"}:
                return None
            entries.append(self._side_ring_entry_for_candidate(
                option,
                abcd_type=abcd_type,
                su_type=self._ring_slot_su_for_candidate(option),
                source_ids=source_ids,
                entry_index=0,
            ))

        entries = [entry for entry in entries if entry]
        if not entries:
            return None
        return {
            "option": option,
            "group_id": str(option.source_group_id),
            "source_ids": source_ids,
            "assembly_family": family,
            "rank": int(self._side_ring_family_rank(family)),
            "entries": entries,
        }

    def _side_ring_entry_for_candidate(
        self,
        option: TopologyAssemblyOption,
        *,
        abcd_type: str,
        su_type: int,
        source_ids: Sequence[int],
        entry_index: int,
    ) -> Dict[str, Any]:
        abcd = str(abcd_type)
        tail_su = self._ring_tail_su_for_candidate(option, abcd)
        profile = self._ring_profile_template_for_option(option, abcd)
        alt = dict(option.candidate_row or {})
        main_path_su = self._side_ring_entry_main_path_su(option, int(su_type), int(entry_index))
        main_path_labels = self._side_ring_entry_main_path_labels(option, main_path_su)
        span = max(1, len(main_path_su))
        profile.update({
            "allocator_role": "side_slot",
            "abcd_type": abcd,
            "su_type": int(su_type),
            "branch_su": [int(x) for x in list(tail_su or [])],
            "main_body_su": [int(x) for x in main_path_su],
            "main_labels": list(main_path_labels),
            "source_ids": [int(x) for x in list(source_ids or []) if int(x) >= 0],
            "assembly_family": str(option.assembly_family),
            "usage_mode": "ring",
            "origin_rule": str(alt.get("origin_rule") or profile.get("origin_rule") or ""),
            "side_ring_entry_index": int(entry_index),
            "side_ring_slot_span": int(span),
            "side_ring_path_su": [int(x) for x in main_path_su],
            "source_group_id": str(option.source_group_id),
        })
        return {
            "option": option,
            "unit_group_id": str(option.source_group_id),
            "assembly_family": str(option.assembly_family),
            "abcd_type": abcd,
            "su_type": int(su_type),
            "path_su": [int(x) for x in main_path_su],
            "slot_span": int(span),
            "branch_su": [int(x) for x in list(tail_su or [])],
            "profile": profile,
            "source_ids": [int(x) for x in list(source_ids or []) if int(x) >= 0],
            "origin_rule": str(alt.get("origin_rule") or ""),
            "rank": int(self._side_ring_family_rank(str(option.assembly_family))),
            "entry_index": int(entry_index),
        }

    def _side_ring_entry_main_path_su(
        self,
        option: TopologyAssemblyOption,
        su_type: int,
        entry_index: int,
    ) -> List[int]:
        family = str(option.assembly_family)
        main = self._alt_main_body(option)
        if family in {"24_AB_N_CD_BRIDGE", "24_CD_N_CD_BRIDGE"}:
            if main:
                return [int(main[0] if int(entry_index) == 0 else main[-1])]
            return [int(su_type)]
        return [int(x) for x in main] if main else [int(su_type)]

    @staticmethod
    def _side_ring_entry_main_path_labels(
        option: TopologyAssemblyOption,
        main_path_su: Sequence[int],
    ) -> List[str]:
        alt = dict(option.candidate_row or {})
        labels = list(alt.get("main_labels", []) or [])
        vals = [int(x) for x in list(main_path_su or [])]
        if labels and len(labels) == len(vals):
            return [str(x) for x in labels]
        return [str(x) for x in vals]

    def _candidate_can_build_side_ring(self, option: TopologyAssemblyOption) -> bool:
        alt = dict(option.candidate_row or {})
        usage_modes = {str(x) for x in list(alt.get("usage_modes", []) or [])}
        text = " ".join([
            str(option.constraints.get("ring_scope") or ""),
            str(alt.get("ring_scope") or ""),
            str(option.constraints.get("ring_usable") or ""),
            str(alt.get("ring_usable") or ""),
            " ".join(sorted(usage_modes)),
        ]).lower()
        if "side" not in text:
            return False
        family = str(option.assembly_family)
        if family in {"24_AB_N_MAIN", "24_CD_N_MAIN"} and not bool(self.constraints.get("allow_long_side_ring_main", False)):
            return False
        return family in {
            "24_AB_N_BRANCH",
            "24_CD_N_BRANCH",
            "24_CD_M_MAIN",
            "24_CD_N_14_MAIN",
            "24_AB_N_CD_BRIDGE",
            "24_CD_N_CD_BRIDGE",
            "24_AB_N_MAIN",
            "24_CD_N_MAIN",
        }

    @staticmethod
    def _side_ring_family_rank(family: str) -> int:
        family = str(family)
        if family in {"24_AB_N_BRANCH", "24_CD_N_BRANCH"}:
            return 0
        if family in {"24_CD_M_MAIN", "24_CD_N_14_MAIN", "24_AB_N_CD_BRIDGE", "24_CD_N_CD_BRIDGE"}:
            return 1
        if family in {"24_AB_N_MAIN", "24_CD_N_MAIN"}:
            return 2
        return 3

    @staticmethod
    def _side_ring_unit_sort_key(unit: Dict[str, Any]) -> Tuple[int, int, str]:
        return (
            int(unit.get("rank", 9)),
            -int(len(unit.get("entries", []) or [])),
            str(unit.get("group_id") or ""),
        )

    @staticmethod
    def _side_ring_entry_span(entry: Dict[str, Any]) -> int:
        raw = int(entry.get("slot_span", 1) or 1)
        return max(1, min(2, raw))

    @staticmethod
    def _side_ring_slot_entry_for_path_index(
        entry: Dict[str, Any],
        path_index: int,
    ) -> Dict[str, Any]:
        out = dict(entry)
        path = [int(x) for x in list(entry.get("path_su", []) or [])]
        if not path:
            path = [int(entry.get("su_type", 24))]
        idx = max(0, min(int(path_index), len(path) - 1))
        out["su_type"] = int(path[idx])
        out["side_ring_path_index"] = int(idx)
        if idx > 0:
            out["branch_su"] = []
            out["side_ring_continuation"] = True
        profile = dict(out.get("profile", {}) or {})
        profile.update({
            "su_type": int(path[idx]),
            "main_body_su": [int(path[idx])],
            "side_ring_path_index": int(idx),
            "side_ring_continuation": bool(idx > 0),
            "branch_su": [int(x) for x in list(out.get("branch_su", []) or [])],
        })
        out["profile"] = profile
        return out

    def _side_ring_slot_assignment(
        self,
        units: Sequence[Dict[str, Any]],
        variant: int,
    ) -> Dict[str, Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for unit in sorted(list(units or []), key=self._side_ring_unit_sort_key):
            entries.extend(list(unit.get("entries", []) or []))
        if not entries or len(entries) > 4:
            return {}
        entries.sort(key=lambda entry: (
            0 if self._is_ab_type(str(entry.get("abcd_type") or "")) else 1,
            int(entry.get("rank", 9)),
            str(entry.get("unit_group_id") or ""),
            int(entry.get("entry_index", 0)),
        ))
        if int(variant) == 0:
            ab_slots = ["pos1", "pos2"]
            cd_slots = ["pos3", "pos4"]
        else:
            ab_slots = ["pos2", "pos1"]
            cd_slots = ["pos4", "pos3"]
        assigned: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            is_ab = self._is_ab_type(str(entry.get("abcd_type") or ""))
            span = self._side_ring_entry_span(entry)
            if is_ab and span != 1:
                return {}
            if is_ab:
                slot = next((name for name in ab_slots if name not in assigned), "")
                if not slot:
                    return {}
                assigned[slot] = self._side_ring_slot_entry_for_path_index(entry, 0)
                continue
            if span == 2:
                ordered = ["pos3", "pos4"] if int(variant) == 0 else ["pos4", "pos3"]
                if any(name in assigned for name in ordered):
                    return {}
                for idx, slot in enumerate(ordered):
                    assigned[slot] = self._side_ring_slot_entry_for_path_index(entry, idx)
                continue
            slot = next((name for name in cd_slots if name not in assigned), "")
            if not slot:
                return {}
            assigned[slot] = self._side_ring_slot_entry_for_path_index(entry, 0)
        return assigned

    def _make_side_ring_option(
        self,
        units: Sequence[Dict[str, Any]],
        slot_entries: Dict[str, Dict[str, Any]],
        variant: int,
    ) -> TopologyAssemblyOption:
        slot_names = ["pos1", "pos3", "pos4", "pos2"]
        default_su = {"pos1": 23, "pos3": 23, "pos4": 23, "pos2": 23}
        ring_su = [
            int(slot_entries[slot]["su_type"]) if slot in slot_entries else int(default_su[slot])
            for slot in slot_names
        ]
        node_types = [
            str(slot_entries[slot]["abcd_type"]) if slot in slot_entries else None
            for slot in slot_names
        ]
        slot_branch_su = {
            slot: [int(x) for x in list(slot_entries[slot].get("branch_su", []) or [])]
            for slot in slot_names
            if slot in slot_entries
        }
        composition = [int(x) for x in ring_su]
        for slot in slot_names:
            composition.extend(int(x) for x in list(slot_branch_su.get(slot, []) or []))
        cost = self._planner_cost_for_sequence(composition)
        cost["11"] = int(cost.get("11", 0)) + 2
        total_like24_count = int(sum(1 for su in list(ring_su or []) if self._is_like24_ring_su(int(su))))
        source_ids = sorted(set(
            int(src)
            for unit in list(units or [])
            for src in list(unit.get("source_ids", []) or [])
            if int(src) >= 0
        ))
        slot_group_ids = [str(unit.get("group_id") or "") for unit in list(units or []) if str(unit.get("group_id") or "")]
        families = [str(unit.get("assembly_family") or "") for unit in list(units or []) if str(unit.get("assembly_family") or "")]
        family_text = "+".join(families) or "side_ring"
        group_id = f"side_ring:{'+'.join(slot_group_ids)}:v{int(variant)}"
        profiles: List[Dict[str, Any]] = []
        tail_sources: List[Optional[str]] = []
        branch_tail_lengths: Dict[str, int] = {}
        profile_idx = 0
        for slot in slot_names:
            entry = slot_entries.get(slot)
            if entry is None:
                tail_sources.append(None)
                continue
            profile = dict(entry.get("profile", {}) or {})
            profile.update({
                "allocator_role": "side_slot",
                "side_slot_name": str(slot),
                "side_slot_index": int(profile_idx),
                "branch_su": [int(x) for x in list(slot_branch_su.get(slot, []) or [])],
                "main_body_su": [int(entry.get("su_type", 24))],
            })
            profiles.append(profile)
            tail_sources.append(str(entry.get("origin_rule") or profile.get("origin_rule") or "") or None)
            if slot_branch_su.get(slot):
                branch_tail_lengths[slot] = int(len(slot_branch_su.get(slot, []) or []))
            profile_idx += 1
        filler_24_count = int(sum(
            1 for slot in slot_names
            if slot not in slot_entries and int(default_su[slot]) == 24
        ))
        meta = {
            "allocator_mode": "side_ring_planner_v1",
            "pending_closure": False,
            "closure_stage_required": False,
            "topology_state": "closed_ring_topology",
            "usage_mode": "ring",
            "ring_type": "side_ring",
            "ring_planner": "side_aliphatic_ring_v1",
            "assembly_family": f"side_ring({family_text})",
            "source_group_id": group_id,
            "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
            "resolved_topology": f"side_ring[{','.join(str(x) for x in ring_su)}]",
            "resource_requirements_override": {key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            "main_body_su": [int(x) for x in ring_su],
            "side_ring_su": [int(x) for x in ring_su],
            "side_ring_node_types": list(node_types),
            "side_ring_slot_names": list(slot_names),
            "branch_tail_lengths": dict(branch_tail_lengths),
            "tail_sources": list(tail_sources),
            "branch24_profiles": profiles,
            "slot_branch_su": dict(slot_branch_su),
            "side_ring_like24_count": int(len(slot_entries)),
            "side_ring_total_like24_count": int(total_like24_count),
            "side_ring_filler_24_count": int(filler_24_count),
            "side_ring_like24_min": 2,
            "side_ring_like24_max": 4,
            "min_like24_ok": bool(2 <= int(total_like24_count) <= 4),
            "source_candidate_group_ids": list(slot_group_ids),
        }
        chain = ChainSpec(
            chain_type="side_ring",
            composition=[int(x) for x in composition],
            origin_type=f"Closed-side_ring({family_text})",
            source_ids=[int(x) for x in source_ids],
            metadata=meta,
        )
        option = TopologyAssemblyOption(
            option_id=f"{group_id}:{','.join(slot_names)}",
            source_group_id=group_id,
            source_ids=[int(x) for x in source_ids],
            assembly_family=f"side_ring({family_text})",
            usage_mode="ring",
            topology_template=str(meta["resolved_topology"]),
            closure_plan={"side_ring": [int(x) for x in ring_su]},
            resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            base_resource_cost={key: int(cost.get(key, 0)) for key in RESOURCE_KEYS},
            closure_resource_cost={key: 0 for key in RESOURCE_KEYS},
            ring_count_delta=1,
            priority=95 - int(sum(int(unit.get("rank", 0)) for unit in list(units or []))) + int(len(slot_entries)) * 4,
            constraints={
                "ring_type": "side_ring",
                "ring_planner": "side_aliphatic_ring_v1",
                "slot_group_ids": list(slot_group_ids),
                "exclusive_group_keys": sorted(set([group_id] + slot_group_ids)),
                "side_ring_su": [int(x) for x in ring_su],
                "side_ring_node_types": list(node_types),
                "side_ring_slot_names": list(slot_names),
                "slot_branch_su": dict(slot_branch_su),
                "side_ring_like24_count": int(len(slot_entries)),
                "side_ring_total_like24_count": int(total_like24_count),
                "side_ring_filler_24_count": int(filler_24_count),
                "side_ring_like24_min": 2,
                "side_ring_like24_max": 4,
                "min_like24_ok": bool(2 <= int(total_like24_count) <= 4),
            },
            candidate_row={
                "ring_planner": "side_aliphatic_ring_v1",
                "source_candidate_group_ids": list(slot_group_ids),
                "assembly_families": list(families),
            },
            source_chain=chain,
        )
        chain.metadata["assembly_option"] = option.to_row()
        return option

    @staticmethod
    def _primary_abcd_type(alt: Dict[str, Any]) -> str:
        raw = str((alt or {}).get("abcd_type") or "")
        for part in raw.replace("/", "+").split("+"):
            text = str(part).strip()
            if text in {"24_A", "24_B", "24_C", "24_D"}:
                return text
        return raw

    @staticmethod
    def _is_ab_type(abcd_type: str) -> bool:
        return str(abcd_type) in {"24_A", "24_B"}

    @staticmethod
    def _is_cd_type(abcd_type: str) -> bool:
        return str(abcd_type) in {"24_C", "24_D"}

    @staticmethod
    def _is_like24_ring_su(su: int) -> bool:
        return int(su) in {14, 19, 20, 21, 24, 25}

    def _ring_slot_su_for_candidate(self, option: TopologyAssemblyOption) -> int:
        profile = self._ring_profile_template_for_option(option, self._primary_abcd_type(dict(option.candidate_row or {})))
        raw = profile.get("su_type")
        if raw is not None:
            try:
                return int(raw)
            except Exception:
                pass
        body = self._alt_main_body(option)
        if body:
            return int(body[0])
        return 24

    def _ring_profile_template_for_option(
        self,
        option: TopologyAssemblyOption,
        abcd_type: str,
    ) -> Dict[str, Any]:
        alt = dict(option.candidate_row or {})
        profiles = [
            dict(p)
            for p in list((getattr(option.source_chain, "metadata", {}) or {}).get("branch24_profiles", []) or [])
            if isinstance(p, dict)
        ]
        origin = str(alt.get("origin_rule") or "")
        profile = next(
            (
                dict(p) for p in profiles
                if str(p.get("origin_rule") or p.get("fixed_path_kind") or "") == origin
            ),
            {},
        )
        if not profile and str(abcd_type):
            profile = next(
                (dict(p) for p in profiles if str(p.get("abcd_type") or "") == str(abcd_type)),
                {},
            )
        if not profile and profiles:
            profile = dict(profiles[0])
        if alt.get("origin_rule") is not None:
            profile.setdefault("origin_rule", str(alt.get("origin_rule") or ""))
            profile.setdefault("fixed_path_kind", str(alt.get("origin_rule") or ""))
        if alt.get("source_kind") is not None:
            profile.setdefault("source_kind", str(alt.get("source_kind") or ""))
        return profile

    def _ring_tail_su_for_candidate(
        self,
        option: TopologyAssemblyOption,
        abcd_type: str,
    ) -> List[int]:
        alt = dict(option.candidate_row or {})
        tail = [int(x) for x in list(alt.get("branch_su", []) or [])]
        need = str(
            option.constraints.get("branch_closure_need")
            or alt.get("branch_closure_need")
            or ""
        )
        if not need:
            text = str(alt.get("open_topology") or "")
            if "H1" in text:
                need = "H1"
            elif "H" in text:
                need = "H"
        if need in {"H", "H1"}:
            tail.extend(self._shortest_intent_sequence(need))
        if not tail:
            if str(abcd_type) in {"24_A", "24_C"}:
                tail = [23, 22]
            elif str(abcd_type) in {"24_B", "24_D"}:
                tail = [22]
        return [int(x) for x in tail]

    @staticmethod
    def _shortest_intent_sequence(intent: str) -> List[int]:
        variants = BRANCH_CLOSURE_INTENTS.get(str(intent), [])
        if not variants:
            return []
        best = min(
            ([int(x) for x in list(seq or [])] for seq in variants),
            key=lambda seq: (len(seq), sum(1 for x in seq if int(x) == 23), seq),
        )
        return [int(x) for x in best]

    def _chain_from_option(self, option: TopologyAssemblyOption) -> ChainSpec:
        if str(option.usage_mode) in {"ordinary_bridge", "ordinary_side", "ordinary_pair_bridge", "ordinary_pair_side"}:
            return self._ordinary_chain_from_option(option)
        if str(option.usage_mode) == "flex_branch":
            return self._flex_chain_from_option(option)
        if str(option.usage_mode) == "side_branch":
            return self._side_chain_from_option(option)
        if option.source_chain is not None and str(getattr(option.source_chain, "chain_type", "")) in {
            "side_ring",
            "vertical_ring",
            "fused_side_ring",
        }:
            return copy.deepcopy(option.source_chain)
        if str(option.usage_mode) == "ring":
            raise ValueError(f"ring option {option.option_id} is missing planner-generated ring source_chain")
        raise ValueError(f"unsupported topology assembly usage mode: {option.usage_mode}")

    def _ordinary_chain_from_option(self, option: TopologyAssemblyOption) -> ChainSpec:
        source = option.source_chain
        base = list(getattr(source, "composition", []) or [])
        plan = dict(option.closure_plan or {})
        left = [int(x) for x in list(plan.get("left", []) or [])]
        right = [int(x) for x in list(plan.get("right", []) or [])]
        composition = list(reversed(left)) + list(base) + list(right)
        ctype = "bridge" if str(option.usage_mode) in {"ordinary_bridge", "ordinary_pair_bridge"} else "side"
        if ctype == "side":
            composition = self._orient_side_composition(composition)
        meta = dict(getattr(source, "metadata", {}) or {})
        base_actual = [int(x) for x in list(meta.get("actual_used_su", []) or base)]
        base_actual_ids = [
            int(x) for x in list(meta.get("actual_used_source_ids", []) or getattr(source, "source_ids", []) or [])
        ]
        closure_resource_su = list(reversed(left)) + list(right)
        meta.update({
            "allocator_mode": "ordinary_closure_v1",
            "pending_closure": False,
            "closure_stage_required": False,
            "topology_state": "closed_ordinary_topology",
            "assembly_option": option.to_row(),
            "resolved_topology": self._resolved_topology_text(option),
            "resource_requirements_override": dict(option.resource_cost),
            "main_body_su": [int(x) for x in composition[1:-1]] if ctype == "bridge" and len(composition) >= 2 else [int(x) for x in composition[1:]],
            "actual_used_su": list(base_actual),
            "actual_used_source_ids": list(base_actual_ids),
            "actual_topology_su": [int(x) for x in composition],
            "closure_resource_su": [int(x) for x in closure_resource_su],
            "chain_port_signature": [int(x) for x in list(meta.get("chain_port_signature", []) or base)],
            "assembly_family": str(option.assembly_family),
            "usage_mode": str(option.usage_mode),
            "source_group_id": str(option.source_group_id),
        })
        return ChainSpec(
            chain_type=ctype,
            composition=[int(x) for x in composition],
            origin_type=f"Closed-{option.usage_mode}({getattr(source, 'origin_type', '')})",
            source_ids=[int(x) for x in list(option.source_ids or [])],
            metadata=meta,
        )

    @staticmethod
    def _orient_side_composition(composition: Sequence[int]) -> List[int]:
        vals = [int(x) for x in list(composition or [])]
        if len(vals) < 2:
            return vals
        if BranchTopologyAssembler._is_terminal_endpoint(vals[0]) and BranchTopologyAssembler._is_aromatic_endpoint(vals[-1]):
            return list(reversed(vals))
        return vals

    @staticmethod
    def _is_terminal_endpoint(su: int) -> bool:
        return int(su) in TERMINAL_BASE_SU or int(su) in {19, 20}

    @staticmethod
    def _is_aromatic_endpoint(su: int) -> bool:
        return int(su) in AROMATIC_SU

    def _flex_chain_from_option(self, option: TopologyAssemblyOption) -> ChainSpec:
        main_body = self._main_body_for_option(option, default_closure="main1")
        branch_su = self._branch_su_for_option(option)
        composition = self._main_path_composition_for_option(
            option,
            main_body,
            chain_type="branch_bridge",
        )
        meta = self._closed_option_metadata(option, main_body, branch_su, composition)
        return ChainSpec(
            chain_type="branch_bridge",
            composition=[int(x) for x in composition],
            origin_type=f"Closed-flex_branch({option.assembly_family})",
            source_ids=[int(x) for x in option.source_ids],
            metadata=meta,
        )

    def _side_chain_from_option(self, option: TopologyAssemblyOption) -> ChainSpec:
        main_body = self._main_body_for_option(option, default_closure="main1")
        branch_su = self._branch_su_for_option(option)
        composition = self._main_path_composition_for_option(
            option,
            main_body,
            chain_type="branch_side",
        )
        meta = self._closed_option_metadata(option, main_body, branch_su, composition)
        return ChainSpec(
            chain_type="branch_side",
            composition=[int(x) for x in composition],
            origin_type=f"Closed-side_branch({option.assembly_family})",
            source_ids=[int(x) for x in option.source_ids],
            metadata=meta,
        )

    def _main_path_composition_for_option(
        self,
        option: TopologyAssemblyOption,
        main_body: Sequence[int],
        *,
        chain_type: str,
    ) -> List[int]:
        prefix, suffix = self._actual_main_path_prefix_suffix(option)
        comp = [int(x) for x in list(prefix) + list(main_body or []) + list(suffix)]
        ctype = str(chain_type)
        if ctype == "branch_bridge":
            if not comp or not self._is_aromatic_endpoint(comp[0]):
                comp = [11] + list(comp)
            if not comp or not self._is_aromatic_endpoint(comp[-1]):
                comp = list(comp) + [11]
        elif ctype == "branch_side":
            if (
                not comp
                or not self._is_aromatic_endpoint(comp[0])
            ) and self._closure_plan_has_endpoint(option, 11):
                comp = [11] + list(comp)
        return [int(x) for x in comp]

    def _actual_main_path_prefix_suffix(self, option: TopologyAssemblyOption) -> Tuple[List[int], List[int]]:
        alt = dict(option.candidate_row or {})
        actual = [int(x) for x in list(alt.get("actual_used_su", []) or [])]
        main = [int(x) for x in list(alt.get("actual_main_path_su", []) or [])]
        if not actual or not main:
            return [], []
        start = self._find_subsequence(actual, main)
        if start < 0:
            return [], []
        end = int(start) + len(main)
        prefix = list(actual[:start])
        suffix = list(actual[end:])
        branch = [int(x) for x in list(alt.get("branch_su", []) or [])]
        if branch and self._endswith_sequence(suffix, branch):
            suffix = suffix[: -len(branch)]
        return [int(x) for x in prefix], [int(x) for x in suffix]

    @staticmethod
    def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int:
        vals = [int(x) for x in list(values or [])]
        ndl = [int(x) for x in list(needle or [])]
        if not ndl or len(ndl) > len(vals):
            return -1
        for idx in range(0, len(vals) - len(ndl) + 1):
            if vals[idx:idx + len(ndl)] == ndl:
                return int(idx)
        return -1

    @staticmethod
    def _endswith_sequence(values: Sequence[int], suffix: Sequence[int]) -> bool:
        vals = [int(x) for x in list(values or [])]
        suf = [int(x) for x in list(suffix or [])]
        return bool(suf) and len(vals) >= len(suf) and vals[-len(suf):] == suf

    @staticmethod
    def _closure_plan_has_endpoint(option: TopologyAssemblyOption, endpoint_su: int) -> bool:
        for seq in dict(option.closure_plan or {}).values():
            vals = [int(x) for x in list(seq or [])]
            if vals and int(vals[-1]) == int(endpoint_su):
                return True
        return False

    def _closed_option_metadata(
        self,
        option: TopologyAssemblyOption,
        main_body: Sequence[int],
        branch_su: Sequence[int],
        composition: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        meta = dict(getattr(option.source_chain, "metadata", {}) or {})
        alt = dict(option.candidate_row or {})
        branch_profiles = self._branch_profiles_for_option(option, main_body, branch_su)
        actual_used_su = [int(x) for x in list(alt.get("actual_used_su", []) or [])]
        actual_used_source_ids = [int(x) for x in list(alt.get("actual_used_source_ids", []) or [])]
        actual_main_path_su = [int(x) for x in list(alt.get("actual_main_path_su", []) or [])]
        actual_main_path_labels = [str(x) for x in list(alt.get("actual_main_path_labels", []) or [])]
        chain_port_signature = [int(x) for x in list(alt.get("chain_port_signature", []) or [])]
        closure_resource_su = self._closure_resource_su_for_option(option)
        main_topology_su = [int(x) for x in list(composition or [])]
        if main_topology_su:
            actual_topology_su = list(main_topology_su) + [int(x) for x in list(branch_su or [])]
        else:
            actual_topology_su = list(actual_used_su) + list(closure_resource_su)
        meta.update({
            "allocator_mode": "branch_assembly_v1",
            "pending_closure": False,
            "closure_stage_required": False,
            "topology_state": "closed_branch_topology",
            "assembly_option": option.to_row(),
            "resolved_topology": self._resolved_topology_text(option),
            "resource_requirements_override": dict(option.resource_cost),
            "main_body_su": [int(x) for x in list(main_body or [])],
            "actual_main_path_su": list(actual_main_path_su),
            "actual_main_path_labels": list(actual_main_path_labels),
            "actual_used_su": list(actual_used_su),
            "actual_used_source_ids": list(actual_used_source_ids),
            "actual_topology_su": list(actual_topology_su),
            "chain_port_signature": list(chain_port_signature),
            "closure_resource_su": list(closure_resource_su),
            "assembly_family": str(option.assembly_family),
            "usage_mode": str(option.usage_mode),
            "source_group_id": str(option.source_group_id),
            "exclusive_group_keys": sorted(self._exclusive_keys_for_option(option)),
        })
        if branch_profiles:
            meta["branch24_profiles"] = branch_profiles
            meta["branch_type"] = str(branch_profiles[0].get("abcd_type") or meta.get("branch_type") or "24_A")
            meta["branch_su"] = [int(x) for x in list(branch_su or [])]
            branch_slots = self._branch_slots_for_option(option, branch_su)
            if branch_slots:
                meta["branch_slots_su"] = branch_slots
                meta["branch_slot_labels"] = {
                    str(k): str(v)
                    for k, v in dict(alt.get("branch_slot_labels", {}) or {}).items()
                }
            meta["fixed_22_is_abstract"] = bool(alt.get("fixed_22_is_abstract", False))
            if alt.get("abstracted_neighbors"):
                meta["abstracted_neighbors"] = [
                    dict(row) for row in list(alt.get("abstracted_neighbors", []) or [])
                    if isinstance(row, dict)
                ]
            meta["branch_23_count"] = int(sum(1 for x in list(branch_su or []) if int(x) == 23))
            meta["branch_22_count"] = int(sum(1 for x in list(branch_su or []) if int(x) == 22))
            meta["extra_22_count"] = 1 if any(int(x) == 25 for x in list(main_body or [])) else 0
            if alt.get("origin_rule") is not None:
                meta["tail_source"] = str(alt.get("origin_rule") or "")
        return meta

    @staticmethod
    def _branch_slots_for_option(
        option: TopologyAssemblyOption,
        branch_su: Sequence[int],
    ) -> Dict[str, List[int]]:
        alt = dict(option.candidate_row or {})
        slots = {
            str(k): [int(x) for x in list(v or [])]
            for k, v in dict(alt.get("branch_slots_su", {}) or {}).items()
        }
        if not slots:
            return {}
        for key, vals in dict(option.closure_plan or {}).items():
            text = str(key)
            if not text.startswith("branch:"):
                continue
            slot = text.split(":", 1)[1]
            slots[str(slot)] = [int(x) for x in list(slots.get(str(slot), []) or [])] + [
                int(x) for x in list(vals or [])
            ]
        declared = [int(x) for vals in slots.values() for x in list(vals or [])]
        branch_vals = [int(x) for x in list(branch_su or [])]
        extra = list(branch_vals)
        for val in declared:
            try:
                extra.remove(int(val))
            except ValueError:
                pass
        if extra:
            slots["H"] = [int(x) for x in list(slots.get("H", []) or [])] + list(extra)
        return slots

    @staticmethod
    def _closure_resource_su_for_option(option: TopologyAssemblyOption) -> List[int]:
        out: List[int] = []
        for _site, seq in sorted(dict(option.closure_plan or {}).items()):
            out.extend(int(x) for x in list(seq or []))
        return out

    @staticmethod
    def _source_ids_for_option(chain: ChainSpec, alt: Dict[str, Any]) -> List[int]:
        alt_ids = [int(x) for x in list((alt or {}).get("source_ids", []) or []) if int(x) >= 0]
        if alt_ids:
            return sorted(set(alt_ids))
        return sorted(set(int(x) for x in list(getattr(chain, "source_ids", []) or []) if int(x) >= 0))

    @staticmethod
    def _source_ids_for_chains(chains: Sequence[ChainSpec]) -> set[int]:
        out: set[int] = set()
        for chain in list(chains or []):
            for gid in list(getattr(chain, "source_ids", []) or []):
                if int(gid) >= 0:
                    out.add(int(gid))
        return out

    @staticmethod
    def _exclusive_keys_for_chains(chains: Sequence[ChainSpec]) -> set[str]:
        out: set[str] = set()
        for chain in list(chains or []):
            meta = dict(getattr(chain, "metadata", {}) or {})
            for key in list(meta.get("exclusive_group_keys", []) or []):
                if str(key):
                    out.add(str(key))
            key = meta.get("exclusive_group_key")
            if key:
                out.add(str(key))
        return out

    @staticmethod
    def _exclusive_keys_for_option(option: TopologyAssemblyOption) -> set[str]:
        out = {
            str(option.source_group_id),
            str(option.constraints.get("candidate_group_id") or ""),
            str(option.constraints.get("exclusive_group_key") or ""),
        }
        for key in list(option.constraints.get("exclusive_group_keys", []) or []):
            out.add(str(key))
        alt = dict(option.candidate_row or {})
        out.add(str(alt.get("exclusive_group_key") or ""))
        for key in list(alt.get("exclusive_group_keys", []) or []):
            out.add(str(key))
        return {key for key in out if key}

    @staticmethod
    def _unclosed_reason_for_group(
        group_id: str,
        all_options: Sequence[TopologyAssemblyOption],
        group_reject_reasons: Dict[str, List[str]],
    ) -> str:
        reasons = list(group_reject_reasons.get(str(group_id), []) or [])
        if reasons:
            priority = [
                "source_id_conflict_with_ordinary_chain",
                "source_id_conflict_with_selected_option",
                "exclusive_group_conflict",
                "resource_shortage",
                "candidate_group_already_selected",
            ]
            for reason in priority:
                if reason in reasons:
                    return reason
            return str(reasons[0])
        has_ring_only = any(
            str(opt.source_group_id) == str(group_id) and str(opt.usage_mode) == "ring_candidate"
            for opt in list(all_options or [])
        )
        has_selectable = any(
            str(opt.source_group_id) == str(group_id) and str(opt.usage_mode) in {"flex_branch", "side_branch"}
            for opt in list(all_options or [])
        )
        if has_ring_only and not has_selectable:
            return "ring_option_disabled_pending_ring_builder"
        return "not_selected_by_resource_scheduler"

    @staticmethod
    def _alt_main_body(option: TopologyAssemblyOption) -> List[int]:
        alt = dict(option.candidate_row or {})
        vals = alt.get("main_body_su")
        if isinstance(vals, (list, tuple)):
            return [int(x) for x in vals]
        return []

    @staticmethod
    def _alt_branch_su(option: TopologyAssemblyOption) -> List[int]:
        alt = dict(option.candidate_row or {})
        vals = alt.get("branch_su")
        if isinstance(vals, (list, tuple)):
            return [int(x) for x in vals]
        return []

    def _main_body_for_option(self, option: TopologyAssemblyOption, default_closure: str) -> List[int]:
        body = self._alt_main_body(option)
        alt = dict(option.candidate_row or {})
        open_ends = {str(x) for x in list(alt.get("open_ends", []) or [])}
        if not open_ends:
            text = str(alt.get("open_topology") or option.topology_template or "")
            if text.startswith("-"):
                open_ends.add("left")
            if text.endswith("-"):
                open_ends.add("right")
        main_keys = sorted(
            [str(k) for k in dict(option.closure_plan or {}) if str(k).startswith("main")],
            key=lambda val: (len(val), val),
        )
        if not main_keys and default_closure in option.closure_plan:
            main_keys = [str(default_closure)]
        if str(option.usage_mode) == "side_branch" and main_keys:
            prefix: List[int] = []
            for key in main_keys[:-1]:
                seq = [int(x) for x in list(option.closure_plan.get(key, []) or [])]
                prefix.extend(seq[:-1] if seq and int(seq[-1]) == 11 else seq)
            tail = [int(x) for x in list(option.closure_plan.get(main_keys[-1], []) or [])]
            if tail and int(tail[-1]) == 11:
                prefix.extend(tail[:-1])
                tail = []
            body = list(prefix) + list(body) + list(tail)
        elif str(option.usage_mode) == "flex_branch":
            stripped: List[List[int]] = []
            for key in main_keys:
                seq = [int(x) for x in list(option.closure_plan.get(key, []) or [])]
                stripped.append(seq[:-1] if seq and int(seq[-1]) == 11 else seq)
            if len(stripped) >= 2:
                body = list(stripped[0]) + list(body)
                for seq in stripped[1:]:
                    body.extend(seq)
            elif stripped:
                if "left" in open_ends and "right" not in open_ends:
                    body = list(stripped[0]) + list(body)
                else:
                    body = list(body) + list(stripped[0])
        return [int(x) for x in body]

    def _branch_su_for_option(self, option: TopologyAssemblyOption) -> List[int]:
        branch = self._alt_branch_su(option)
        closure_branch = [int(x) for x in list((option.closure_plan or {}).get("branch", []) or [])]
        if closure_branch:
            branch = list(branch) + list(closure_branch)
        for key in sorted(str(k) for k in dict(option.closure_plan or {}) if str(k).startswith("branch:")):
            branch.extend(int(x) for x in list((option.closure_plan or {}).get(key, []) or []))
        return [int(x) for x in branch]

    def _branch_profiles_for_option(
        self,
        option: TopologyAssemblyOption,
        main_body: Sequence[int],
        branch_su: Sequence[int],
    ) -> List[Dict[str, Any]]:
        branch_anchor_values = {14, 24, 25}
        if not branch_su and not any(int(x) in branch_anchor_values for x in list(main_body or [])):
            return []
        alt = dict(option.candidate_row or {})
        profiles = [
            dict(p) for p in list((getattr(option.source_chain, "metadata", {}) or {}).get("branch24_profiles", []) or [])
            if isinstance(p, dict)
        ]
        profile = next(
            (
                dict(p) for p in profiles
                if str(p.get("origin_rule") or p.get("fixed_path_kind") or "") == str(alt.get("origin_rule") or "")
            ),
            dict(profiles[0]) if profiles else {},
        )
        su_type = profile.get("su_type")
        if su_type is None:
            su_type = next((int(x) for x in list(main_body or []) if int(x) in branch_anchor_values), 24)
        profile.update({
            "allocator_role": "branch_single",
            "abcd_type": str(alt.get("abcd_type") or profile.get("abcd_type") or "24_A"),
            "su_type": int(su_type),
            "branch_su": [int(x) for x in list(branch_su or [])],
            "branch_slots_su": self._branch_slots_for_option(option, branch_su),
            "branch_slot_labels": {
                str(k): str(v)
                for k, v in dict(alt.get("branch_slot_labels", {}) or {}).items()
            },
            "fixed_22_is_abstract": bool(alt.get("fixed_22_is_abstract", False)),
            "main_body_su": [int(x) for x in list(main_body or [])],
            "source_ids": [int(x) for x in list(option.source_ids or [])],
            "assembly_family": str(option.assembly_family),
            "usage_mode": str(option.usage_mode),
            "origin_rule": str(alt.get("origin_rule") or profile.get("origin_rule") or ""),
        })
        return [profile]

    @staticmethod
    def _resolved_topology_text(option: TopologyAssemblyOption) -> str:
        parts = [str(option.topology_template)]
        if option.closure_plan:
            closure = ",".join(
                f"{site}:{'-'.join(str(x) for x in seq)}"
                for site, seq in option.closure_plan.items()
            )
            parts.append(f"closures[{closure}]")
        return " ".join(part for part in parts if part)

    @staticmethod
    def _branch_intent_for_alt(alt: Dict[str, Any]) -> Optional[str]:
        family = str(alt.get("assembly_family") or "")
        if family in {"24_AB_P", "24_CD_P"}:
            return None
        intents = list((alt.get("closure_intents", {}) or {}).get("branch", []) or [])
        if "H1" in intents:
            return "H1"
        if "H" in intents:
            return "H"
        if str(alt.get("branch_connector_kind") or "") == "N":
            return "H1"
        if str(alt.get("selected_fixed_role") or "") == "main":
            return "H"
        return None

    @staticmethod
    def _branch_slot_intents_for_alt(alt: Dict[str, Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        intents = list((dict(alt.get("closure_intents", {}) or {})).get("branch_slots", []) or [])
        for raw in intents:
            text = str(raw)
            if ":" in text:
                intent, slot = text.split(":", 1)
            else:
                intent, slot = text, text
            intent = str(intent).strip()
            slot = str(slot).strip()
            if intent and slot:
                out.append((slot, intent))
        return out

    @staticmethod
    def _main_intents_for_family(family: str, usage_mode: str) -> List[str]:
        family = str(family)
        if str(usage_mode) == "flex_branch":
            if family in {"24_AB_M", "24_CD_M_BRIDGE", "24_AB_N_AB_BRIDGE"}:
                return []
            if family == "24_CD_M_MAIN":
                return ["F1"]
            if family.startswith("24_CD_") or family == "24_CD_N_CD_BRIDGE":
                return ["F1", "F1"]
            return ["F1"]
        if str(usage_mode) == "side_branch":
            if family in {"24_AB_P"}:
                return []
            if family in {"24_CD_P", "24_CD_M_MAIN"}:
                return ["F1"]
            if family.startswith("24_CD_") or family == "24_CD_N_CD_BRIDGE":
                return ["F1", "F2"]
            return ["F2"]
        return []

    @staticmethod
    def _intent_variants(intent: Optional[str]) -> List[Tuple[int, ...]]:
        if not intent:
            return [tuple()]
        vals = BRANCH_CLOSURE_INTENTS.get(str(intent), [])
        return [tuple(int(x) for x in seq) for seq in vals] or [tuple()]

    @staticmethod
    def _cost_for_sequences(sequences: Sequence[Sequence[int]]) -> Dict[str, int]:
        out = {key: 0 for key in RESOURCE_KEYS}
        for seq in list(sequences or []):
            for su in list(seq or []):
                val = int(su)
                if val == 11:
                    out["11"] += 1
                elif val == 22:
                    out["22"] += 1
                elif val == 23:
                    out["23"] += 1
                elif val == 24:
                    out["24"] += 1
                elif val == 25:
                    out["25"] += 1
        return out

    @staticmethod
    def _merge_costs(*costs: Dict[str, int]) -> Dict[str, int]:
        out = {key: 0 for key in RESOURCE_KEYS}
        for cost in costs:
            for key in RESOURCE_KEYS:
                out[key] += int((cost or {}).get(key, 0))
        return out

    @staticmethod
    def _closure_key(plan: Dict[str, List[int]]) -> str:
        if not plan:
            return "none"
        return "+".join(f"{k}{'-'.join(str(x) for x in v)}" for k, v in sorted(plan.items()))

    @staticmethod
    def _option_sort_key(option: TopologyAssemblyOption) -> Tuple[int, int, int, int, str]:
        cost = option.resource_cost
        closure = option.closure_resource_cost
        closure_units = int(closure.get("11", 0)) + int(closure.get("22", 0)) + int(closure.get("23", 0))
        consumes_backbone = int(cost.get("11", 0)) + int(cost.get("22", 0)) + int(cost.get("23", 0))
        return (-int(option.priority), -int(option.flex_count_delta), closure_units, consumes_backbone, str(option.option_id))

    @staticmethod
    def _side_option_sort_key(option: TopologyAssemblyOption) -> Tuple[int, int, int, str]:
        cost = option.resource_cost
        closure = option.closure_resource_cost
        closure_units = int(closure.get("11", 0)) + int(closure.get("22", 0)) + int(closure.get("23", 0))
        consumes_tail = int(cost.get("22", 0)) + int(cost.get("23", 0))
        return (-int(option.priority), closure_units, consumes_tail, str(option.option_id))

    @staticmethod
    def _ring_option_sort_key(option: TopologyAssemblyOption) -> Tuple[int, int, int, str]:
        cost = option.resource_cost
        closure = option.closure_resource_cost
        closure_units = int(closure.get("11", 0)) + int(closure.get("22", 0)) + int(closure.get("23", 0))
        consumes_backbone = int(cost.get("11", 0)) + int(cost.get("22", 0)) + int(cost.get("23", 0))
        return (-int(option.priority), closure_units, consumes_backbone, str(option.option_id))

    @staticmethod
    def _pending_option_sort_key(option: TopologyAssemblyOption) -> Tuple[int, int, int, int, str]:
        resource_cost = {str(k): int((option.resource_cost or {}).get(str(k), 0)) for k in RESOURCE_KEYS}
        closure_cost = {str(k): int((option.closure_resource_cost or {}).get(str(k), 0)) for k in RESOURCE_KEYS}
        resource_total = int(sum(int(v) for v in resource_cost.values()))
        closure_total = int(sum(int(v) for v in closure_cost.values()))
        backbone_total = int(sum(int(resource_cost.get(str(k), 0)) for k in ("11", "22", "23")))
        return (
            int(resource_total),
            int(closure_total),
            int(backbone_total),
            -int(option.priority),
            str(option.option_id),
        )

    @classmethod
    def _pending_closure_demand_summary(
        cls,
        pending_rows: Sequence[Dict[str, Any]],
        all_options: Sequence[TopologyAssemblyOption],
        rejected_options: Sequence[Dict[str, Any]],
        remaining: Dict[str, int],
    ) -> Dict[str, Any]:
        rows = [dict(row or {}) for row in list(pending_rows or [])]
        available = {str(k): max(0, int((remaining or {}).get(str(k), 0) or 0)) for k in RESOURCE_KEYS}
        demand = {str(k): 0 for k in RESOURCE_KEYS}
        base_demand = {str(k): 0 for k in RESOURCE_KEYS}
        closure_demand = {str(k): 0 for k in RESOURCE_KEYS}
        intent_demand: Dict[str, int] = {}
        usage_mode_demand: Dict[str, int] = {}
        assembly_family_demand: Dict[str, int] = {}
        selected_rows: List[Dict[str, Any]] = []
        missing_option_groups: List[str] = []

        options_by_group: Dict[str, List[TopologyAssemblyOption]] = {}
        for opt in list(all_options or []):
            group_id = str(getattr(opt, "source_group_id", "") or "")
            if not group_id:
                continue
            options_by_group.setdefault(group_id, []).append(opt)

        rejected_by_group: Dict[str, List[str]] = {}
        for row in list(rejected_options or []):
            group_id = str(dict(row or {}).get("source_group_id", "") or "")
            if not group_id:
                continue
            reason = str(dict(row or {}).get("reason", "") or "")
            if reason:
                rejected_by_group.setdefault(group_id, []).append(reason)

        def _add_counter(counter: Dict[str, int], key: str, count: int = 1) -> None:
            key_s = str(key)
            if not key_s:
                return
            counter[key_s] = int(counter.get(key_s, 0)) + max(0, int(count))

        def _add_cost(target: Dict[str, int], cost: Dict[str, int]) -> None:
            for key in RESOURCE_KEYS:
                target[str(key)] = int(target.get(str(key), 0)) + max(0, int((cost or {}).get(str(key), 0) or 0))

        def _fallback_closure_from_row(row: Dict[str, Any]) -> Dict[str, int]:
            cost = {str(k): 0 for k in RESOURCE_KEYS}
            intents_obj = row.get("closure_intents", {}) or {}
            intent_values: List[str] = []

            def _collect(obj: Any) -> None:
                if isinstance(obj, dict):
                    for val in obj.values():
                        _collect(val)
                elif isinstance(obj, (list, tuple, set)):
                    for val in obj:
                        _collect(val)
                else:
                    text = str(obj or "")
                    if text:
                        intent_values.append(text)

            _collect(intents_obj)
            for intent in intent_values:
                seq = cls._shortest_intent_sequence(str(intent))
                if not seq:
                    continue
                _add_counter(intent_demand, f"fallback:{intent}:{'-'.join(str(x) for x in seq)}")
                _add_cost(cost, cls._cost_for_sequences([seq]))
            return cost

        for row in rows:
            group_id = str(row.get("candidate_group_id", "") or "")
            if not group_id:
                continue
            options = [
                opt for opt in list(options_by_group.get(group_id, []) or [])
                if str(getattr(opt, "usage_mode", "") or "") != "ring_candidate"
            ]
            if not options:
                options = list(options_by_group.get(group_id, []) or [])
            if not options:
                fallback_cost = _fallback_closure_from_row(row)
                if any(int(v) > 0 for v in fallback_cost.values()):
                    _add_cost(demand, fallback_cost)
                    _add_cost(closure_demand, fallback_cost)
                missing_option_groups.append(str(group_id))
                selected_rows.append({
                    "candidate_group_id": str(group_id),
                    "origin_type": str(row.get("origin_type", "")),
                    "reason": str(row.get("reason", "")),
                    "best_option_id": "",
                    "usage_mode": "",
                    "assembly_family": "",
                    "resource_cost": dict(fallback_cost),
                    "base_resource_cost": {str(k): 0 for k in RESOURCE_KEYS},
                    "closure_resource_cost": dict(fallback_cost),
                    "rejected_reasons": list(rejected_by_group.get(group_id, []) or []),
                    "fallback_from_intents": True,
                })
                continue
            best = min(options, key=cls._pending_option_sort_key)
            resource_cost = {str(k): max(0, int((best.resource_cost or {}).get(str(k), 0) or 0)) for k in RESOURCE_KEYS}
            base_cost = {str(k): max(0, int((best.base_resource_cost or {}).get(str(k), 0) or 0)) for k in RESOURCE_KEYS}
            closure_cost = {str(k): max(0, int((best.closure_resource_cost or {}).get(str(k), 0) or 0)) for k in RESOURCE_KEYS}
            _add_cost(demand, resource_cost)
            _add_cost(base_demand, base_cost)
            _add_cost(closure_demand, closure_cost)
            _add_counter(usage_mode_demand, str(best.usage_mode))
            _add_counter(assembly_family_demand, str(best.assembly_family))
            for site, seq in sorted(dict(best.closure_plan or {}).items()):
                seq_vals = [int(x) for x in list(seq or [])]
                label = "-".join(str(x) for x in seq_vals) if seq_vals else "none"
                _add_counter(intent_demand, f"{site}:{label}")
            selected_rows.append({
                "candidate_group_id": str(group_id),
                "origin_type": str(row.get("origin_type", "")),
                "reason": str(row.get("reason", "")),
                "best_option_id": str(best.option_id),
                "usage_mode": str(best.usage_mode),
                "assembly_family": str(best.assembly_family),
                "resource_cost": dict(resource_cost),
                "base_resource_cost": dict(base_cost),
                "closure_resource_cost": dict(closure_cost),
                "closure_plan": {
                    str(k): [int(x) for x in list(v or [])]
                    for k, v in dict(best.closure_plan or {}).items()
                },
                "rejected_reasons": list(rejected_by_group.get(group_id, []) or []),
                "fallback_from_intents": False,
            })

        deficit = {
            str(key): max(0, int(demand.get(str(key), 0)) - int(available.get(str(key), 0)))
            for key in RESOURCE_KEYS
        }
        minimum_conversions = [
            {
                "resource": str(key),
                "shortage_key": str(key),
                "shortage_type": f"{key}_shortage",
                "count": int(deficit.get(str(key), 0)),
            }
            for key in RESOURCE_KEYS
            if int(deficit.get(str(key), 0)) > 0
        ]
        return {
            "pending_group_count": int(len(rows)),
            "intent_demand": dict(sorted(intent_demand.items())),
            "usage_mode_demand": dict(sorted(usage_mode_demand.items())),
            "assembly_family_demand": dict(sorted(assembly_family_demand.items())),
            "resource_demand": dict(demand),
            "base_resource_demand": dict(base_demand),
            "closure_resource_demand": dict(closure_demand),
            "resource_available": dict(available),
            "resource_deficit": dict(deficit),
            "minimum_conversions": list(minimum_conversions),
            "minimum_conversion_count": int(sum(int(v) for v in deficit.values())),
            "candidate_demands": list(selected_rows),
            "missing_option_groups": list(missing_option_groups),
        }

    @staticmethod
    def _recommended_adjustments(
        shortage: Dict[str, int],
        remaining: Dict[str, int],
        flex_count: int,
        flex_lower: int,
        flex_upper: int,
        unclosed_count: int,
        disabled_ring_count: int = 0,
        backbone_residual: int = 0,
        unclosed_rows: Optional[Sequence[Dict[str, Any]]] = None,
        pending_demand_summary: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        shortage_i = {str(k): max(0, int((shortage or {}).get(str(k), 0))) for k in RESOURCE_KEYS}
        remaining_i = {str(k): max(0, int((remaining or {}).get(str(k), 0))) for k in RESOURCE_KEYS}
        pending_summary_i = dict(pending_demand_summary or {})

        def _append(
            op: str,
            count: int,
            reason: str,
            shortage_key: Optional[str] = None,
            preferred_ops: Optional[List[str]] = None,
            actionable: bool = True,
        ) -> None:
            count_i = max(0, int(count))
            if count_i <= 0 and bool(actionable):
                return
            row: Dict[str, Any] = {
                "op": str(op),
                "count": int(count_i),
                "reason": str(reason),
                "actionable": bool(actionable),
                "resource_shortage": dict(shortage_i),
                "resource_surplus": dict(remaining_i),
                "resource_deficit_total": int(sum(int(v) for v in shortage_i.values())),
                "resource_surplus_total": int(sum(int(v) for v in remaining_i.values())),
            }
            if pending_summary_i:
                row["pending_resource_demand"] = dict(pending_summary_i.get("resource_demand", {}) or {})
                row["pending_resource_deficit"] = dict(pending_summary_i.get("resource_deficit", {}) or {})
                row["pending_intent_demand"] = dict(pending_summary_i.get("intent_demand", {}) or {})
                row["minimum_conversions"] = list(pending_summary_i.get("minimum_conversions", []) or [])
            if shortage_key is not None:
                row["shortage_key"] = str(shortage_key)
                row["shortage_type"] = f"{str(shortage_key)}_shortage"
            if preferred_ops is not None:
                row["preferred_ops"] = [str(x) for x in list(preferred_ops)]
            out.append(row)

        def _plan_shortage(key: str, amount: int) -> None:
            amount_i = max(0, int(amount))
            if amount_i <= 0:
                return
            rem11 = int(remaining_i.get("11", 0))
            rem22 = int(remaining_i.get("22", 0))
            rem23 = int(remaining_i.get("23", 0))
            rem24 = int(remaining_i.get("24", 0))
            rem25 = int(remaining_i.get("25", 0))
            short11 = int(shortage_i.get("11", 0))
            short22 = int(shortage_i.get("22", 0))
            short23 = int(shortage_i.get("23", 0))
            short24 = int(shortage_i.get("24", 0))

            if str(key) == "23":
                left = amount_i
                take = min(left, rem22)
                _append("22->23", take, "branch_assembly_23_shortage_from_22", "23")
                left -= take
                take = min(left, rem24 if short24 <= 0 else 0)
                _append("24->23", take, "branch_assembly_23_shortage_from_24", "23")
                left -= take
                take = min(left, rem11 if short11 <= 0 else 0)
                _append("11->23", take, "branch_assembly_23_shortage_from_11", "23")
                left -= take
                _append("12/13->11->23", left, "branch_assembly_23_shortage_from_aromatic_fallback", "23")
            elif str(key) == "24":
                left = amount_i
                take = min(left, rem23 if short23 <= 0 else 0)
                _append("23->24", take, "branch_assembly_24_shortage_from_23", "24")
                left -= take
                take = min(left, rem25)
                _append("25->24", take, "branch_assembly_24_shortage_from_25", "24")
                left -= take
                take = min(left, rem22 if short22 <= 0 else 0)
                _append("22->23->24", take, "branch_assembly_24_shortage_from_22_chain", "24")
                left -= take
                take = min(left, rem11 if short11 <= 0 else 0)
                _append("11->23->24", take, "branch_assembly_24_shortage_from_11_chain", "24")
                left -= take
                _append("12/13->11->23->24", left, "branch_assembly_24_shortage_from_aromatic_fallback", "24")
            elif str(key) == "22":
                left = amount_i
                take = min(left, rem23 if short23 <= 0 else 0)
                _append("23->22", take, "branch_assembly_22_shortage_from_23", "22")
                left -= take
                take = min(left, rem24 if short24 <= 0 and short23 <= 0 else 0)
                _append("24->23->22", take, "branch_assembly_22_shortage_from_24_chain", "22")
                left -= take
                take = min(left, rem11 if short11 <= 0 and short23 <= 0 else 0)
                _append("11->23->22", take, "branch_assembly_22_shortage_from_11_chain", "22")
                left -= take
                _append("12/13->11->23->22", left, "branch_assembly_22_shortage_from_aromatic_fallback", "22")
            elif str(key) == "11":
                left = amount_i
                take = min(left, rem23 if short23 <= 0 else 0)
                _append("23->11", take, "branch_assembly_11_shortage_from_23", "11")
                left -= take
                take = min(left, rem22 if short22 <= 0 and short23 <= 0 else 0)
                _append("22->23->11", take, "branch_assembly_11_shortage_from_22_chain", "11")
                left -= take
                take = min(left, rem24 if short24 <= 0 and short23 <= 0 else 0)
                _append("24->23->11", take, "branch_assembly_11_shortage_from_24_chain", "11")
                left -= take
                _append("12/13->11", left, "branch_assembly_11_shortage_from_aromatic_fallback", "11")
            elif str(key) == "25":
                _append(
                    "increase_25_specific_motif",
                    amount_i,
                    "branch_assembly_25_shortage_requires_specific_quaternary_motif",
                    "25",
                    actionable=False,
                )

        flex_short = max(0, int(flex_lower) - int(flex_count))
        flex_excess = max(0, int(flex_count) - int(flex_upper))
        residual_cleanup_allowed = bool(
            int(unclosed_count) == 0
            and int(flex_short) == 0
            and int(flex_excess) == 0
            and not any(int(v) > 0 for v in dict(shortage or {}).values())
        )
        shortage_priority = {"23": 0, "24": 1, "22": 2, "11": 3, "25": 4}
        for key, value in sorted(
            (shortage or {}).items(),
            key=lambda kv: (-max(0, int(kv[1])), int(shortage_priority.get(str(kv[0]), 99)), str(kv[0])),
        ):
            if int(value) <= 0:
                continue
            _plan_shortage(str(key), int(value))
        if int(flex_short) > 0:
            _append(
                "flex_short_make_F1",
                int(flex_short),
                "flex_count_below_lower_bound",
                preferred_ops=["12/13->11", "22->23", "11->23"],
            )
            out[-1]["flex_count"] = int(flex_count)
            out[-1]["flex_lower"] = int(flex_lower)
            _append("12/13->11", int(flex_short), "flex_short_F1_endpoint_11", "11")
            _append("22->23", int(flex_short), "flex_short_F1_internal_23_from_22", "23")
        if int(flex_excess) > 0:
            _append(
                "reduce_flex_prefer_side",
                int(flex_excess),
                "flex_count_above_upper_bound",
                preferred_ops=["23->22", "24->23->22"],
            )
            out[-1]["flex_count"] = int(flex_count)
            out[-1]["flex_upper"] = int(flex_upper)
            _append("23->22", int(flex_excess), "flex_excess_shift_F1_to_F2_side", "22")
        surplus_24 = max(0, int((remaining or {}).get("24", 0)))
        surplus_25 = max(0, int((remaining or {}).get("25", 0)))
        if int(unclosed_count) > 0:
            ordinary_f1 = 0
            ordinary_f2 = 0
            like24_unclosed = 0
            for row in list(unclosed_rows or []):
                intents = list(dict(row.get("closure_intents", {}) or {}).get("main", []) or [])
                if any(str(x).startswith("F1") for x in intents):
                    ordinary_f1 += 1
                if any(str(x).startswith("F2") for x in intents):
                    ordinary_f2 += 1
                if str(row.get("assembly_families", "")) not in {"", "[]"} and row.get("assembly_families"):
                    like24_unclosed += 1
            if ordinary_f1 > 0:
                _append(
                    "ordinary_pending_F1_resources",
                    int(ordinary_f1),
                    "ordinary_pending_F1_shortage",
                    preferred_ops=["12/13->11", "22->23", "11->23"],
                )
            if ordinary_f2 > 0:
                _append(
                    "ordinary_pending_F2_resources",
                    int(ordinary_f2),
                    "ordinary_pending_F2_shortage",
                    preferred_ops=["22->23", "23->22"],
                )
            if like24_unclosed > 0:
                _append(
                    "like24_unclosed_rebalance",
                    int(like24_unclosed),
                    "like24_unclosed_after_scheduler",
                    preferred_ops=["23->24", "25->24", "22->23->24"],
                )
            if surplus_24 > 0:
                _append("24->23", min(int(unclosed_count), int(surplus_24)), "unclosed_like24_candidates_with_24_surplus")
            if surplus_25 > 0:
                _append("25->24", min(int(unclosed_count), int(surplus_25)), "unclosed_like24_candidates_with_25_surplus")
            _append("increase_branch_resources_or_reduce_like24", int(unclosed_count), "unclosed_like24_candidates", actionable=False)
        rem11 = max(0, int((remaining or {}).get("11", 0)))
        rem22 = max(0, int((remaining or {}).get("22", 0)))
        rem23 = max(0, int((remaining or {}).get("23", 0)))
        if int(backbone_residual) > 0 and bool(residual_cleanup_allowed):
            if rem11 > 0 and rem23 > 0:
                _append("23->22", min(rem11, rem23), "backbone_resource_residual")
            if rem22 > 0:
                _append("22->23", rem22, "backbone_resource_residual_22")
        if int(disabled_ring_count) > 0:
            _append("enable_ring_builder_or_defer_ring", int(disabled_ring_count), "ring_options_temporarily_disabled", actionable=False)
        for idx, row in enumerate(out):
            row["_order"] = int(idx)
            reason = str(row.get("reason", "") or "")
            op_name = str(row.get("op", "") or "")
            if reason.startswith("backbone_resource_residual"):
                row["phase"] = "residual_cleanup"
                row["defer_until_pending_zero"] = True
            elif reason.startswith("flex_") or op_name in {"flex_short_make_F1", "reduce_flex_prefer_side"}:
                row["phase"] = "flex_repair"
                row["defer_until_pending_zero"] = False
            elif op_name == "enable_ring_builder_or_defer_ring":
                row["phase"] = "planner_deferred"
                row["defer_until_pending_zero"] = True
            else:
                row["phase"] = "closure_repair"
                row["defer_until_pending_zero"] = False
        phase_priority = {
            "closure_repair": 0,
            "flex_repair": 1,
            "residual_cleanup": 2,
            "planner_deferred": 3,
        }

        def _recommendation_rank(row: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
            shortage_key = str(row.get("shortage_key", "") or "")
            shortage_amount = int(shortage_i.get(shortage_key, 0)) if shortage_key else 0
            return (
                0 if bool(row.get("actionable", True)) else 1,
                0 if int(shortage_amount) > 0 else 1,
                -int(shortage_amount),
                int(shortage_priority.get(shortage_key, 99)),
                int(phase_priority.get(str(row.get("phase", "") or ""), 99)),
                int(row.get("_order", 0)),
            )

        out.sort(key=_recommendation_rank)
        for row in out:
            row.pop("_order", None)
        return out
