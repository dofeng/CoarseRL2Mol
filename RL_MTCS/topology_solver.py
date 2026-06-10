from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .allocator_models import (
    CONVERSION_RESOURCE_KEYS,
    MAX_23_PER_CHAIN,
    RESOURCE_KEYS,
    ChainSpec,
    PlannedConversion,
    ResidualChainAction,
    ResidualGrowthAction,
    TopologyAssemblyPlan,
    _chain_required_resource_counts,
    chain_23_count,
    chain_23_limit_ok,
    chain_actual_topology_su,
)


def _norm_counts(raw: Optional[Dict[Any, Any]], keys: Sequence[str] = CONVERSION_RESOURCE_KEYS) -> Dict[str, int]:
    src = dict(raw or {})
    return {
        str(k): max(0, int(src.get(str(k), src.get(int(k), 0)) or 0))
        for k in keys
    }


def _add_counts(dst: Dict[str, int], src: Dict[str, int], sign: int = 1) -> None:
    for key in RESOURCE_KEYS:
        dst[str(key)] = int(dst.get(str(key), 0)) + int(sign) * int((src or {}).get(str(key), 0))


def _sequence_cost(values: Iterable[int]) -> Dict[str, int]:
    vals = [int(x) for x in list(values or [])]
    return {
        "11": int(sum(1 for x in vals if x == 11)),
        "22": int(sum(1 for x in vals if x == 22)),
        "23": int(sum(1 for x in vals if x == 23)),
        "24": int(sum(1 for x in vals if x in {14, 24})),
        "25": int(sum(1 for x in vals if x == 25)),
    }


@dataclass
class ConversionPlanResult:
    feasible: bool
    inventory: Dict[str, int]
    conversions: List[PlannedConversion] = field(default_factory=list)
    reason: str = ""


class ConversionPlanner:
    """Small deterministic resource conversion planner.

    It is intentionally dependency-free and only plans count-level conversions.
    Block C later materializes these conversion records on concrete nodes.
    """

    _DELTAS: Dict[str, Tuple[str, str]] = {
        "11->23": ("11", "23"),
        "22->23": ("22", "23"),
        "24->23": ("24", "23"),
        "25->24": ("25", "24"),
        "23->24": ("23", "24"),
        "23->22": ("23", "22"),
        "23->11": ("23", "11"),
        "12->11": ("12", "11"),
        "13->11": ("13", "11"),
    }

    def _apply(self, inv: Dict[str, int], conversions: List[PlannedConversion], op: str, *, phase: str, reason: str) -> bool:
        if str(op) not in self._DELTAS:
            return False
        src, dst = self._DELTAS[str(op)]
        if int(inv.get(src, 0)) <= 0:
            return False
        inv[src] = int(inv.get(src, 0)) - 1
        inv[dst] = int(inv.get(dst, 0)) + 1
        conversions.append(PlannedConversion(op=str(op), count=1, phase=str(phase), reason=str(reason)))
        return True

    @staticmethod
    def _merge_conversions(conversions: Sequence[PlannedConversion]) -> List[PlannedConversion]:
        merged: Dict[Tuple[str, str, str, str], PlannedConversion] = {}
        for conv in list(conversions or []):
            key = (str(conv.op), str(conv.phase), str(conv.reason), str(conv.source))
            if key not in merged:
                merged[key] = PlannedConversion(
                    op=str(conv.op),
                    count=0,
                    phase=str(conv.phase),
                    reason=str(conv.reason),
                    source=str(conv.source),
                )
            merged[key].count = int(merged[key].count) + max(0, int(conv.count))
        return [conv for conv in merged.values() if int(conv.count) > 0]

    @staticmethod
    def _deficit(inv: Dict[str, int], demand: Dict[str, int], key: str) -> int:
        return max(0, int(demand.get(str(key), 0)) - int(inv.get(str(key), 0)))

    @staticmethod
    def _surplus(inv: Dict[str, int], demand: Dict[str, int], key: str) -> int:
        return max(0, int(inv.get(str(key), 0)) - int(demand.get(str(key), 0)))

    def _try_path(
        self,
        inv: Dict[str, int],
        demand: Dict[str, int],
        conversions: List[PlannedConversion],
        ops: Sequence[str],
        *,
        phase: str,
        reason: str,
    ) -> bool:
        local = dict(inv)
        local_conversions: List[PlannedConversion] = []
        for op in list(ops or []):
            src, _dst = self._DELTAS[str(op)]
            if src in RESOURCE_KEYS and self._surplus(local, demand, src) <= 0:
                return False
            if not self._apply(local, local_conversions, str(op), phase=phase, reason=reason):
                return False
        inv.update(local)
        conversions.extend(local_conversions)
        return True

    def plan_to_cover(
        self,
        inventory: Dict[str, int],
        demand: Dict[str, int],
        *,
        phase: str = "closure",
        reason: str = "resource_deficit",
        max_steps: int = 256,
    ) -> ConversionPlanResult:
        inv = _norm_counts(inventory)
        demand_i = {str(k): max(0, int((demand or {}).get(str(k), 0) or 0)) for k in RESOURCE_KEYS}
        conversions: List[PlannedConversion] = []

        for _ in range(max(0, int(max_steps))):
            deficits = {key: self._deficit(inv, demand_i, key) for key in RESOURCE_KEYS}
            if not any(int(v) > 0 for v in deficits.values()):
                return ConversionPlanResult(True, inv, self._merge_conversions(conversions), "")

            progressed = False
            if deficits["23"] > 0:
                paths = (("22->23",), ("24->23",), ("11->23",), ("12->11", "11->23"), ("13->11", "11->23"))
            elif deficits["11"] > 0:
                paths = (("23->11",), ("22->23", "23->11"), ("24->23", "23->11"), ("12->11",), ("13->11",))
            elif deficits["22"] > 0:
                paths = (("23->22",), ("24->23", "23->22"), ("11->23", "23->22"))
            elif deficits["24"] > 0:
                paths = (("23->24",), ("25->24",), ("22->23", "23->24"), ("11->23", "23->24"))
            else:
                return ConversionPlanResult(False, inv, self._merge_conversions(conversions), "25_shortage")

            for path in paths:
                if self._try_path(inv, demand_i, conversions, path, phase=phase, reason=reason):
                    progressed = True
                    break
            if not progressed:
                missing = ",".join(f"{k}:{v}" for k, v in deficits.items() if int(v) > 0)
                return ConversionPlanResult(False, inv, self._merge_conversions(conversions), f"cannot_cover:{missing}")

        return ConversionPlanResult(False, inv, self._merge_conversions(conversions), "conversion_step_limit")


@dataclass
class ResidualPlanResult:
    feasible: bool
    specs: List[ChainSpec] = field(default_factory=list)
    conversions: List[PlannedConversion] = field(default_factory=list)
    growth_actions: List[ResidualGrowthAction] = field(default_factory=list)
    chain_actions: List[ResidualChainAction] = field(default_factory=list)
    remaining: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


class ResidualPlanner:
    def __init__(self, max_23_per_chain: int = MAX_23_PER_CHAIN, prefer_bridge: bool = False):
        self.max_23_per_chain = max(1, int(max_23_per_chain))
        self.prefer_bridge = bool(prefer_bridge)
        self.converter = ConversionPlanner()

    def plan(self, remaining: Dict[str, int], closed_specs: Sequence[ChainSpec]) -> ResidualPlanResult:
        for allow_growth in (True, False):
            result = self._plan_once(remaining, closed_specs, allow_growth=allow_growth)
            if bool(result.feasible):
                return result
        return self._plan_once(remaining, closed_specs, allow_growth=False)

    def _plan_once(
        self,
        remaining: Dict[str, int],
        closed_specs: Sequence[ChainSpec],
        *,
        allow_growth: bool,
    ) -> ResidualPlanResult:
        inv = _norm_counts(remaining)
        if bool(allow_growth):
            planned = self._search_growth_then_residual(inv, closed_specs)
            if planned is not None:
                return planned

        search = self._search_residual_chains(inv)
        if not bool(search.feasible):
            return ResidualPlanResult(
                feasible=False,
                specs=[],
                conversions=list(search.conversions),
                growth_actions=[],
                chain_actions=list(search.chain_actions),
                remaining={str(k): int(search.remaining.get(str(k), 0)) for k in RESOURCE_KEYS},
                reason=str(search.reason or "residual_unclosed"),
            )
        return ResidualPlanResult(
            feasible=True,
            specs=list(search.specs),
            conversions=list(search.conversions),
            growth_actions=[],
            chain_actions=list(search.chain_actions),
            remaining={str(k): 0 for k in RESOURCE_KEYS},
            reason="ok",
        )

    def _search_growth_then_residual(
        self,
        inv: Dict[str, int],
        closed_specs: Sequence[ChainSpec],
    ) -> Optional[ResidualPlanResult]:
        start = _norm_counts(inv)
        candidates: List[Tuple[int, ChainSpec, List[int], int]] = []
        for idx, chain in enumerate(list(closed_specs or [])):
            vals = chain_actual_topology_su(chain)
            if len(vals) < 2 or int(vals[-1]) not in {11, 22}:
                continue
            current_23 = chain_23_count(chain)
            cap = max(0, int(self.max_23_per_chain) - int(current_23))
            if int(cap) <= 0:
                continue
            candidates.append((int(idx), chain, [int(x) for x in vals], int(cap)))
        if not candidates:
            return None

        memo: set[Tuple[int, Tuple[int, ...]]] = set()

        def state_key(state: Dict[str, int]) -> Tuple[int, ...]:
            return tuple(int(state.get(str(k), 0)) for k in CONVERSION_RESOURCE_KEYS)

        def dfs(
            pos: int,
            rem: Dict[str, int],
            grown_specs: List[ChainSpec],
            actions: List[ResidualGrowthAction],
        ) -> Optional[ResidualPlanResult]:
            key = (int(pos), state_key(rem))
            if key in memo:
                return None
            memo.add(key)
            if int(pos) >= len(candidates) or int(rem.get("23", 0)) <= 0:
                search = self._search_residual_chains(rem)
                if not bool(search.feasible):
                    return None
                return ResidualPlanResult(
                    feasible=True,
                    specs=list(grown_specs) + list(search.specs),
                    conversions=list(search.conversions),
                    growth_actions=list(actions),
                    chain_actions=list(search.chain_actions),
                    remaining={str(k): 0 for k in RESOURCE_KEYS},
                    reason="ok",
                )

            chain_index, chain, vals, cap = candidates[int(pos)]
            max_add = min(int(cap), int(rem.get("23", 0)))
            for add in range(int(max_add), -1, -1):
                next_rem = dict(rem)
                next_specs = list(grown_specs)
                next_actions = list(actions)
                if int(add) > 0:
                    grown, action = self._growth_spec(chain_index, chain, vals, int(add))
                    if grown is None or action is None:
                        continue
                    next_rem["23"] = int(next_rem.get("23", 0)) - int(add)
                    next_specs.append(grown)
                    next_actions.append(action)
                found = dfs(int(pos) + 1, next_rem, next_specs, next_actions)
                if found is not None and bool(found.feasible):
                    return found
            return None

        return dfs(0, start, [], [])

    def _growth_spec(
        self,
        chain_index: int,
        chain: ChainSpec,
        vals: Sequence[int],
        add: int,
    ) -> Tuple[Optional[ChainSpec], Optional[ResidualGrowthAction]]:
        after = [int(x) for x in list(vals[:-1])] + [23] * int(add) + [int(vals[-1])]
        grown = copy.deepcopy(chain)
        grown.composition = [int(x) for x in after]
        grown.origin_type = f"Residual-growth({getattr(chain, 'origin_type', '')})"
        base_cost = _chain_required_resource_counts(chain)
        cost = {str(k): int(base_cost.get(str(k), 0)) for k in RESOURCE_KEYS}
        cost["23"] = int(cost.get("23", 0)) + int(add)
        meta = dict(getattr(grown, "metadata", {}) or {})
        meta.update({
            "allocator_mode": "residual_growth_v1",
            "generated_from_remaining_backbone": True,
            "growth_target_chain_index": int(chain_index),
            "residual_growth_added_23": int(add),
            "actual_topology_su": [int(x) for x in after],
            "resource_requirements_override": dict(cost),
        })
        grown.metadata = meta
        if not chain_23_limit_ok(grown, self.max_23_per_chain):
            return None, None
        action = ResidualGrowthAction(
            chain_index=int(chain_index),
            add_23=int(add),
            endpoint=int(vals[-1]),
            before_topology=[int(x) for x in vals],
            after_topology=[int(x) for x in after],
        )
        return grown, action

    def _search_residual_chains(self, inv: Dict[str, int]) -> ResidualPlanResult:
        start = _norm_counts(inv)
        memo: set[Tuple[int, ...]] = set()
        max_depth = max(1, int(sum(start.get(k, 0) for k in RESOURCE_KEYS)) + 5)

        def state_key(state: Dict[str, int]) -> Tuple[int, ...]:
            return tuple(int(state.get(str(k), 0)) for k in CONVERSION_RESOURCE_KEYS)

        def done(state: Dict[str, int]) -> bool:
            return all(int(state.get(str(k), 0)) == 0 for k in RESOURCE_KEYS)

        def dfs(
            state: Dict[str, int],
            depth: int,
            specs: List[ChainSpec],
            actions: List[ResidualChainAction],
            conversions: List[PlannedConversion],
        ) -> Optional[ResidualPlanResult]:
            if done(state):
                return ResidualPlanResult(True, list(specs), list(conversions), [], list(actions), {str(k): 0 for k in RESOURCE_KEYS}, "ok")
            if int(depth) >= int(max_depth):
                return None
            key = state_key(state)
            if key in memo:
                return None
            memo.add(key)

            templates = self._candidate_residual_templates(state)
            for comp in templates:
                demand = _sequence_cost(comp)
                cover = self.converter.plan_to_cover(
                    state,
                    demand,
                    phase="residual",
                    reason="residual_chain_endpoint_or_body",
                    max_steps=64,
                )
                if not bool(cover.feasible):
                    continue
                next_state = dict(cover.inventory)
                if any(int(next_state.get(k, 0)) < int(demand.get(k, 0)) for k in RESOURCE_KEYS):
                    continue
                for rk in RESOURCE_KEYS:
                    next_state[str(rk)] = int(next_state.get(str(rk), 0)) - int(demand.get(str(rk), 0))
                if state_key(next_state) == key:
                    continue
                action = ResidualChainAction(
                    composition=[int(x) for x in comp],
                    chain_type="bridge" if int(comp[-1]) == 11 else "side",
                )
                chain = self._residual_chain(action, len(specs))
                found = dfs(
                    next_state,
                    int(depth) + 1,
                    list(specs) + [chain],
                    list(actions) + [action],
                    list(conversions) + list(cover.conversions),
                )
                if found is not None and bool(found.feasible):
                    found.conversions = ConversionPlanner._merge_conversions(found.conversions)
                    return found
            return None

        found = dfs(start, 0, [], [], [])
        if found is not None:
            return found
        return ResidualPlanResult(False, [], [], [], [], {str(k): int(start.get(str(k), 0)) for k in RESOURCE_KEYS}, "no_residual_template_solution")

    def _candidate_residual_templates(self, state: Dict[str, int]) -> List[List[int]]:
        out: List[List[int]] = []
        target_sum = int(sum(int(state.get(str(k), 0)) for k in RESOURCE_KEYS))
        endpoints = (11, 22) if bool(self.prefer_bridge) else (22, 11)
        for k in range(int(self.max_23_per_chain), 0, -1):
            for endpoint in endpoints:
                comp = [11] + [23] * int(k) + [int(endpoint)]
                cost = sum(_sequence_cost(comp).values())
                if int(cost) <= int(target_sum) + int(state.get("12", 0)) + int(state.get("13", 0)):
                    out.append(comp)
        return out

    @staticmethod
    def _residual_chain(action: ResidualChainAction, idx: int) -> ChainSpec:
        comp = [int(x) for x in list(action.composition or [])]
        ctype = str(action.chain_type)
        cost = _sequence_cost(comp)
        return ChainSpec(
            chain_type=ctype,
            composition=list(comp),
            origin_type=f"Residual-{ctype}:{int(idx)}",
            source_ids=[],
            metadata={
                "allocator_mode": "residual_chain_v1",
                "topology_state": "closed_residual_topology",
                "generated_from_remaining_backbone": True,
                "actual_topology_su": list(comp),
                "resource_requirements_override": dict(cost),
                "main_body_su": [int(x) for x in comp[1:-1]] if ctype == "bridge" else [int(x) for x in comp[1:]],
            },
        )


@dataclass
class SolverResult:
    plan: TopologyAssemblyPlan
    selected_options: List[Any] = field(default_factory=list)
    residual_specs: List[ChainSpec] = field(default_factory=list)
    unclosed_group_ids: List[str] = field(default_factory=list)
    reason: str = ""


class TopologyAssemblySolver:
    def __init__(
        self,
        *,
        resource_totals: Dict[str, int],
        flex_lower: int,
        flex_upper: int,
        ordinary_flex_count: int,
        ordinary_chains: Sequence[ChainSpec],
        pending_group_ids: Sequence[str],
        option_to_chain: Callable[[Any], ChainSpec],
        used_source_ids: Iterable[int],
        used_exclusive_keys: Iterable[str],
        max_23_per_chain: int = MAX_23_PER_CHAIN,
        max_states: int = 50000,
    ):
        self.resource_totals = _norm_counts(resource_totals)
        self.flex_lower = int(flex_lower)
        self.flex_upper = int(flex_upper)
        self.ordinary_flex_count = int(ordinary_flex_count)
        self.ordinary_chains = list(ordinary_chains or [])
        self.pending_group_ids = [str(x) for x in list(pending_group_ids or []) if str(x)]
        self.option_to_chain = option_to_chain
        self.used_source_ids = {int(x) for x in set(used_source_ids or set()) if int(x) >= 0}
        self.used_exclusive_keys = {str(x) for x in set(used_exclusive_keys or set()) if str(x)}
        self.max_23_per_chain = max(1, int(max_23_per_chain))
        self.max_states = max(1, int(max_states))
        self.converter = ConversionPlanner()
        self.residual = ResidualPlanner(max_23_per_chain=self.max_23_per_chain)
        self.states_seen = 0

    def solve(self, options: Sequence[Any], pre_consumed: Dict[str, int]) -> SolverResult:
        usable_options = [
            opt for opt in list(options or [])
            if str(getattr(opt, "usage_mode", "") or "") != "ring_candidate"
        ]
        options_by_group: Dict[str, List[Any]] = {gid: [] for gid in self.pending_group_ids}
        for opt in usable_options:
            covers = self._covered_groups(opt)
            for gid in covers:
                if gid in options_by_group:
                    options_by_group.setdefault(gid, []).append(opt)

        for gid in self.pending_group_ids:
            opts = options_by_group.get(str(gid), [])
            if not opts:
                return SolverResult(
                    TopologyAssemblyPlan(False, False, False, reason=f"no_options_for_group:{gid}"),
                    [],
                    [],
                    [str(gid)],
                    f"no_options_for_group:{gid}",
                )
            options_by_group[gid] = sorted(opts, key=self._option_rank)

        best_failure = "search_exhausted"

        def choose_group(covered: set[str]) -> Optional[str]:
            remaining = [gid for gid in self.pending_group_ids if gid not in covered]
            if not remaining:
                return None
            return min(remaining, key=lambda gid: len([
                opt for opt in options_by_group.get(gid, [])
                if not (self._covered_groups(opt) & covered)
            ]) or 10**9)

        def dfs(
            covered: set[str],
            selected: List[Any],
            used_src: set[int],
            used_excl: set[str],
            demand: Dict[str, int],
            flex_count: int,
        ) -> Optional[SolverResult]:
            nonlocal best_failure
            self.states_seen += 1
            if int(self.states_seen) > int(self.max_states):
                best_failure = "solver_state_limit"
                return None
            group_id = choose_group(covered)
            if group_id is None:
                return self._finalize_plan(selected, demand, flex_count, pre_consumed)

            for opt in options_by_group.get(str(group_id), []):
                covers = self._covered_groups(opt)
                if covers & covered:
                    continue
                src = {int(x) for x in list(getattr(opt, "source_ids", []) or []) if int(x) >= 0}
                if src & used_src:
                    best_failure = "source_id_conflict"
                    continue
                excl = self._exclusive_keys(opt)
                if excl & used_excl:
                    best_failure = "exclusive_group_conflict"
                    continue
                try:
                    chain = self.option_to_chain(opt)
                    if not chain_23_limit_ok(chain, self.max_23_per_chain):
                        best_failure = "chain_23_limit"
                        continue
                except Exception:
                    best_failure = "option_chain_materialization_failed"
                    continue
                next_demand = dict(demand)
                _add_counts(next_demand, {str(k): int((getattr(opt, "resource_cost", {}) or {}).get(str(k), 0)) for k in RESOURCE_KEYS})
                found = dfs(
                    covered | covers,
                    list(selected) + [opt],
                    set(used_src) | src,
                    set(used_excl) | excl,
                    next_demand,
                    int(flex_count) + int(getattr(opt, "flex_count_delta", 0) or 0),
                )
                if found is not None and bool(found.plan.feasible):
                    return found
            return None

        initial_demand = {str(k): int((pre_consumed or {}).get(str(k), 0)) for k in RESOURCE_KEYS}
        result = dfs(set(), [], set(self.used_source_ids), set(self.used_exclusive_keys), initial_demand, self.ordinary_flex_count)
        if result is not None:
            return result
        return SolverResult(
            TopologyAssemblyPlan(False, False, False, reason=best_failure),
            [],
            [],
            [gid for gid in self.pending_group_ids],
            str(best_failure),
        )

    def _finalize_plan(
        self,
        selected: Sequence[Any],
        demand: Dict[str, int],
        flex_count: int,
        pre_consumed: Dict[str, int],
    ) -> Optional[SolverResult]:
        cover = self.converter.plan_to_cover(
            self.resource_totals,
            demand,
            phase="closure",
            reason="pending_closure",
            max_steps=512,
        )
        if not bool(cover.feasible):
            return None
        remaining = dict(cover.inventory)
        for key in RESOURCE_KEYS:
            remaining[str(key)] = int(remaining.get(str(key), 0)) - int((demand or {}).get(str(key), 0))
        selected_chains: List[ChainSpec] = []
        try:
            selected_chains = [self.option_to_chain(opt) for opt in list(selected or [])]
        except Exception:
            return None
        residual_planner = ResidualPlanner(
            max_23_per_chain=self.max_23_per_chain,
            prefer_bridge=bool(int(flex_count) < int(self.flex_lower)),
        )
        residual_plan = residual_planner.plan(remaining, list(self.ordinary_chains) + list(selected_chains))
        if not bool(residual_plan.feasible):
            return None
        materialized_base, residual_additions = self._apply_growth_specs(
            list(self.ordinary_chains) + list(selected_chains),
            residual_plan.specs,
        )
        residual_flex = int(sum(
            1 for chain in list(residual_plan.specs or [])
            if str(getattr(chain, "chain_type", "")) in {"bridge", "branch_bridge"}
            and not self._is_growth_spec(chain)
        ))
        final_flex_count = int(flex_count) + int(residual_flex)
        if int(final_flex_count) < int(self.flex_lower) or int(final_flex_count) > int(self.flex_upper):
            return None
        all_specs = list(materialized_base) + list(residual_additions)
        violations = self._chain_limit_violations(all_specs)
        if violations:
            return None
        conversions = ConversionPlanner._merge_conversions(list(cover.conversions) + list(residual_plan.conversions))
        materialization_required = bool(conversions)
        plan = TopologyAssemblyPlan(
            feasible=True,
            materialized=not bool(materialization_required),
            materialization_required=bool(materialization_required),
            selected_option_ids=[str(getattr(opt, "option_id", "")) for opt in list(selected or [])],
            planned_conversions=list(conversions),
            residual_growth_actions=list(residual_plan.growth_actions),
            residual_chain_actions=list(residual_plan.chain_actions),
            residual_remaining={str(k): int(residual_plan.remaining.get(str(k), 0)) for k in RESOURCE_KEYS},
            chain_23_limit_violations=list(violations),
            reason="ok",
        )
        return SolverResult(plan, list(selected), list(residual_plan.specs), [], "ok")

    @staticmethod
    def _is_growth_spec(chain: ChainSpec) -> bool:
        meta = dict(getattr(chain, "metadata", {}) or {})
        return bool(str(meta.get("allocator_mode") or "") == "residual_growth_v1")

    @classmethod
    def _apply_growth_specs(
        cls,
        base_chains: Sequence[ChainSpec],
        residual_specs: Sequence[ChainSpec],
    ) -> Tuple[List[ChainSpec], List[ChainSpec]]:
        materialized = [copy.deepcopy(ch) for ch in list(base_chains or [])]
        additions: List[ChainSpec] = []
        for spec in list(residual_specs or []):
            if cls._is_growth_spec(spec):
                meta = dict(getattr(spec, "metadata", {}) or {})
                try:
                    idx = int(meta.get("growth_target_chain_index", -1))
                except Exception:
                    idx = -1
                if 0 <= int(idx) < len(materialized):
                    materialized[int(idx)] = copy.deepcopy(spec)
                else:
                    additions.append(copy.deepcopy(spec))
                continue
            additions.append(copy.deepcopy(spec))
        return materialized, additions

    @staticmethod
    def _covered_groups(option: Any) -> set[str]:
        out = {str(getattr(option, "source_group_id", "") or "")}
        constraints = dict(getattr(option, "constraints", {}) or {})
        for key in list(constraints.get("slot_group_ids", []) or []):
            if str(key):
                out.add(str(key))
        return {x for x in out if x}

    @staticmethod
    def _exclusive_keys(option: Any) -> set[str]:
        constraints = dict(getattr(option, "constraints", {}) or {})
        alt = dict(getattr(option, "candidate_row", {}) or {})
        out = {
            str(getattr(option, "source_group_id", "") or ""),
            str(constraints.get("candidate_group_id") or ""),
            str(constraints.get("exclusive_group_key") or ""),
            str(alt.get("exclusive_group_key") or ""),
        }
        for key in list(constraints.get("exclusive_group_keys", []) or []):
            out.add(str(key))
        for key in list(alt.get("exclusive_group_keys", []) or []):
            out.add(str(key))
        return {key for key in out if key}

    @staticmethod
    def _option_rank(option: Any) -> Tuple[int, int, int, str]:
        cost = dict(getattr(option, "resource_cost", {}) or {})
        closure = dict(getattr(option, "closure_resource_cost", {}) or {})
        closure_total = sum(int(closure.get(str(k), 0)) for k in RESOURCE_KEYS)
        total = sum(int(cost.get(str(k), 0)) for k in RESOURCE_KEYS)
        return (-int(getattr(option, "priority", 0) or 0), int(closure_total), int(total), str(getattr(option, "option_id", "")))

    def _chain_limit_violations(self, chains: Sequence[ChainSpec]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, chain in enumerate(list(chains or [])):
            count = chain_23_count(chain)
            if int(count) > int(self.max_23_per_chain):
                rows.append({
                    "chain_index": int(idx),
                    "chain_type": str(getattr(chain, "chain_type", "")),
                    "origin_type": str(getattr(chain, "origin_type", "")),
                    "count_23": int(count),
                    "max_23": int(self.max_23_per_chain),
                    "composition": [int(x) for x in chain_actual_topology_su(chain)],
                })
        return rows
