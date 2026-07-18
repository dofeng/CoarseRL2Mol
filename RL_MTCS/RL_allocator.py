from __future__ import annotations

import ast
import csv
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pandas is only required for csv_path mode.
    pd = None

from .allocator_patterns import (
    AROMATIC_SU,
    TopologyFragment,
    fragment_signature,
    has_forbidden_fixed_connector_special_combo,
    is_abstract_chain_port_state,
    is_chain_state,
    is_double_special_d2,
    is_single_special_d2,
    is_single_special_d3,
    is_special_d2,
    is_terminal_state,
    node_id,
    node_su,
    state_for_node,
)
from .allocator_branch import BranchTopologyMixin
from .branch_assembler import BranchTopologyAssembler
from .allocator_models import (
    AllocationResult,
    ChainSpec,
    SUNode,
    _DoubleHalf,
    _chain_required_resource_counts,
    _is_missing_value,
    _sum_chain_required_resource_counts,
)


SHARED_BACKBONE_RESOURCE_SU = {11, 22, 23, 24, 25}
BRANCH_RESERVED_TERMINAL_SU = {1, 4, 16, 18, 28, 32}
STAGE_C_RESOURCE_SU = {11, 22, 23, 24, 25}
STAGE_C_RESOURCE_KEYS = ("11", "22", "23", "24", "25")
STAGE_C_RESOURCE_BY_SU = {11: "11", 22: "22", 23: "23", 24: "24", 25: "25"}
STAGE_C_SU_BY_RESOURCE = {"11": 11, "22": 22, "23": 23, "24": 24, "25": 25}
STAGE_C_IGNORED_TOPOLOGY_SU = {10, 12, 13, 26, 30}
STAGE_C_TOPOLOGY_SU = (
    set(range(0, 33)) - set(STAGE_C_IGNORED_TOPOLOGY_SU)
) | set(STAGE_C_RESOURCE_SU)
STAGE_C_BRANCH_CENTER_SU = {24, 25}
STAGE_C_TERMINAL_SU = {1, 4, 16, 18, 22, 28, 32}
STAGE_C_CHAIN_SU = {14, 15, 17, 19, 20, 21, 22, 23, 24, 25}
STAGE_C_ANCHOR_SU = set(AROMATIC_SU) | {0, 1, 2, 3, 4, 27, 28, 29, 31, 32}
STAGE_C_MAX_23_PER_CHAIN = 6


class FlexAllocator(BranchTopologyMixin):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        su_counts: Optional[Dict[int, int]] = None,
        nodes: Optional[List[Any]] = None,
        topology_constraints: Optional[Dict[str, Any]] = None,
    ):
        self.csv_path = csv_path
        self._input_nodes = nodes
        self._su_counts = dict(su_counts or {})
        self._topology_constraints = dict(topology_constraints or {})
        self._nodes: List[SUNode] = []
        self._node_lookup: Dict[int, SUNode] = {}
        self._result = AllocationResult()

    @staticmethod
    def _branch_cost(branch_type: Optional[str]) -> Tuple[int, int]:
        if branch_type in {"24_A", "24_C", "25_aro", "25_ali"}:
            return 1, 1
        return 0, 1

    def redistribute_remaining_flex_23(self, excess_23: int) -> Dict[str, int]:
        summary = {
            "requested_23": int(max(0, excess_23)),
            "applied_23": 0,
            "to_side": 0,
            "to_branch": 0,
            "remaining_23": int(max(0, excess_23)),
        }
        self._result.post_flex_23_requested += int(summary["requested_23"])
        return summary

    def _topology_constraints_with_flex_window(self) -> Dict[str, Any]:
        constraints = dict(self._topology_constraints or {})
        constraints.setdefault("defer_residual_to_extra", True)
        has_external_window = any(
            key in constraints
            for key in ("flex_lower", "flex_upper", "flexible_bridge_min", "flexible_bridge_limit")
        )
        if has_external_window:
            lower_raw = constraints.get("flex_lower", constraints.get("flexible_bridge_min", 0))
            upper_raw = constraints.get("flex_upper", constraints.get("flexible_bridge_limit", None))
            lower = int(lower_raw or 0)
            upper = int(lower if upper_raw is None else upper_raw)
            bounds_ok = int(lower) <= int(upper)
            constraints.setdefault("flex_lower_raw", int(lower))
            constraints.setdefault("flex_upper_raw", int(upper))
            constraints["flex_window_ok"] = bool(bounds_ok and bool(constraints.get("flex_window_ok", bounds_ok)))
            constraints.setdefault("flex_window_source", "external")
            return constraints

        window = self._compute_internal_flex_window()
        constraints.update(window)
        constraints.setdefault("flex_lower", int(window.get("flex_lower_raw", 0)))
        constraints.setdefault("flex_upper", int(window.get("flex_upper_raw", 10**9)))
        constraints.setdefault("flexible_bridge_min", int(constraints["flex_lower"]))
        constraints.setdefault("flexible_bridge_limit", int(constraints["flex_upper"]))
        return constraints

    def _compute_internal_flex_window(self) -> Dict[str, Any]:
        if not self._nodes:
            return {
                "X": 0,
                "Y": 0,
                "Z": 0,
                "flex_lower_raw": 0,
                "flex_upper_raw": 10**9,
                "flex_window_ok": True,
                "flex_window_source": "no_nodes_default",
                "bridgehead_info": tuple(),
            }
        counts = Counter(int(n.su_type) for n in self._nodes)
        cluster_count = 0
        bridgehead_info: Tuple[int, ...] = tuple()
        source = "internal_cluster_generator"
        error = ""
        try:
            from .RL_init import ClusterGenerator

            bridgehead_info = self._derive_bridgehead_info_for_flex_window()
            gen = ClusterGenerator(dict(counts), bridgehead_info=bridgehead_info)
            cluster_count = int(len(gen.generate()))
        except Exception as exc:
            source = "internal_aromatic_count_fallback"
            error = f"{type(exc).__name__}: {exc}"
            aromatic_count = int(sum(counts.get(int(su), 0) for su in AROMATIC_SU | {10, 12, 13}))
            cluster_count = int(max(0, round(float(aromatic_count) / 6.0)))
        y = max(0, int(counts.get(10, 0)) // 2)
        z = max(0, int(cluster_count) - int(y))
        lower = int(z + 1)
        upper = int(0.8 * float(max(0, int(cluster_count))))
        return {
            "X": int(cluster_count),
            "Y": int(y),
            "Z": int(z),
            "flex_lower_raw": int(lower),
            "flex_upper_raw": int(upper),
            "flex_window_ok": bool(int(lower) <= int(upper)),
            "flex_window_source": str(source),
            "flex_window_error": str(error),
            "bridgehead_info": tuple(int(x) for x in bridgehead_info),
        }

    def _derive_bridgehead_info_for_flex_window(self) -> Tuple[int, int, int]:
        m = n = p = 0
        any_hop2 = False
        for node in self._nodes:
            if int(node.su_type) != 12:
                continue
            hop2 = list(node.hop2 or [])
            if hop2:
                any_hop2 = True
            count_12 = int(sum(1 for su in hop2 if int(su) == 12))
            if count_12 >= 2:
                m += 1
            elif count_12 == 1:
                n += 1
            else:
                p += 1
        if not any_hop2:
            p = int(sum(1 for node in self._nodes if int(node.su_type) == 12))
            m = n = 0
        return int(m), int(n), int(p)

    def allocate(self) -> AllocationResult:
        print("=" * 60)
        print("[FlexAllocator] Starting Stage-C graph topology allocation")
        print("=" * 60)
        self._result = AllocationResult()
        self._parse_input()
        self._populate_totals()
        self._allocate_stage_c_graph_topology()
        self._print_summary()
        return self._result

    # ---------- Stage-C graph topology allocator ----------

    def _allocate_stage_c_graph_topology(self) -> None:
        if not self._nodes and self._su_counts:
            gid = 0
            for su, count in sorted((int(k), int(v)) for k, v in dict(self._su_counts or {}).items()):
                for _ in range(max(0, int(count))):
                    self._add_node(SUNode(global_id=int(gid), su_type=int(su)))
                    gid += 1
            self._populate_totals()

        node_by_id = {int(n.global_id): n for n in self._nodes if int(n.global_id) >= 0}
        adj = self._stage_c_adjacency()
        relevant_ids = {
            int(n.global_id)
            for n in self._nodes
            if int(n.global_id) >= 0 and self._stage_c_is_relevant_su(int(n.su_type))
        }
        components = self._stage_c_components(relevant_ids, adj)
        bridge_chains: List[ChainSpec] = []
        side_chains: List[ChainSpec] = []
        branch_chains: List[ChainSpec] = []
        pending_chains: List[ChainSpec] = []
        component_rows: List[Dict[str, Any]] = []
        issue_rows: List[Dict[str, Any]] = []
        shortage = {key: 0 for key in STAGE_C_RESOURCE_KEYS}
        chain_23_limit_violations: List[Dict[str, Any]] = []

        for comp_idx, comp in enumerate(components):
            nodes = [node_by_id[int(gid)] for gid in sorted(comp) if int(gid) in node_by_id]
            if not nodes:
                continue
            ordered = self._stage_c_order_component(nodes, comp, adj)
            comp_issues = self._stage_c_component_issues(ordered, comp, adj)
            for row in comp_issues:
                key = str(row.get("shortage_key", ""))
                if key in shortage:
                    shortage[key] += 1
            issue_rows.extend(comp_issues)

            chain_type = self._stage_c_component_type(ordered, comp, adj)
            has_branch = any(
                int(n.su_type) in STAGE_C_BRANCH_CENTER_SU or len(adj.get(int(n.global_id), set()) & set(comp)) >= 3
                for n in ordered
            )
            pending = bool(comp_issues)
            spec = self._stage_c_chain_spec(
                chain_type="pending_closure" if pending else chain_type,
                nodes=ordered,
                origin_type=("Pending-graph" if pending else f"Graph-{chain_type}"),
                metadata={
                    "allocator_mode": "stage_c_graph_v2",
                    "component_index": int(comp_idx),
                    "component_size": int(len(nodes)),
                    "has_branch_center": bool(has_branch),
                    "pending_closure": bool(pending),
                    "topology_state": "pending" if pending else "closed",
                    "component_issues": [dict(row) for row in comp_issues],
                    "display_topology": "-".join(str(int(n.su_type)) for n in ordered),
                    "open_topology": "-".join(str(int(n.su_type)) for n in ordered),
                    "chain_port_signature": [int(n.su_type) for n in ordered],
                },
            )
            row = self._chain_row(spec, int(comp_idx))
            row["component_issues"] = [dict(x) for x in comp_issues]
            component_rows.append(row)
            n23 = sum(1 for su in list(spec.composition or []) if int(su) == 23)
            if int(n23) > STAGE_C_MAX_23_PER_CHAIN:
                chain_23_limit_violations.append({
                    "component_index": int(comp_idx),
                    "chain_type": str(chain_type),
                    "n_23": int(n23),
                    "limit": int(STAGE_C_MAX_23_PER_CHAIN),
                    "composition": [int(x) for x in list(spec.composition or [])],
                    "source_ids": [int(x) for x in list(spec.source_ids or [])],
                })
            if pending:
                pending_chains.append(spec)
            elif chain_type == "bridge":
                bridge_chains.append(spec)
            elif chain_type == "branch":
                branch_chains.append(spec)
            else:
                side_chains.append(spec)

        source_pool = self._stage_c_source_pool(shortage)
        recommendations = self._stage_c_recommendations(shortage, source_pool)
        flex_diag = self._stage_c_flex_diagnostics(bridge_chains, branch_chains)
        pending_count = int(len(pending_chains))
        shortage_total = int(sum(max(0, int(v)) for v in shortage.values()))
        flex_window_conflict = bool(flex_diag.get("flex_window_conflict", False))
        flex_short = max(0, int(flex_diag.get("flex_short", 0)))
        flex_excess = max(0, int(flex_diag.get("flex_excess", 0)))
        topology_ok = bool(
            shortage_total == 0
            and pending_count == 0
            and not chain_23_limit_violations
            and not flex_window_conflict
            and int(flex_short) == 0
            and int(flex_excess) == 0
        )

        demand_summary = {
            "pending_group_count": int(pending_count),
            "resource_demand": {str(k): int(shortage.get(str(k), 0)) for k in STAGE_C_RESOURCE_KEYS},
            "resource_available": {str(k): int(source_pool.get(str(k), 0)) for k in STAGE_C_RESOURCE_KEYS},
            "resource_deficit": {str(k): int(shortage.get(str(k), 0)) for k in STAGE_C_RESOURCE_KEYS},
            "minimum_conversions": [
                {
                    "resource": str(k),
                    "shortage_key": str(k),
                    "count": int(v),
                    "reason": "graph_valence_target",
                }
                for k, v in shortage.items()
                if int(v) > 0
            ],
            "candidate_demands": list(component_rows),
        }
        planned_conversions = [
            {
                "op": str(row.get("op", "")),
                "count": int(row.get("count", 1)),
                "phase": str(row.get("phase", "closure_repair")),
                "reason": str(row.get("reason", "graph_valence_repair")),
                "shortage_key": str(row.get("shortage_key", "")),
                "shortage_type": str(row.get("shortage_type", "")),
                "source": "stage_c_graph_v2",
            }
            for row in recommendations
            if str(row.get("op", ""))
        ]
        abstract_gap_by_kind = (
            {"pending_closure": int(pending_count)}
            if int(pending_count) > 0 and int(shortage_total) == 0
            else {}
        )
        unallocated_branch = int(sum(
            1
            for ch in pending_chains
            if bool(dict(getattr(ch, "metadata", {}) or {}).get("has_branch_center", False))
        ))
        diag = {
            "allocator_mode": "stage_c_graph_v2",
            "ok": bool(topology_ok),
            "topology_assembly_ok": bool(topology_ok),
            "topology_materialized_ok": bool(topology_ok),
            "topology_plan_feasible": bool(topology_ok or bool(recommendations)),
            "topology_ready_for_extra": bool(topology_ok),
            "defer_residual_to_extra": True,
            "materialization_required": bool(shortage_total > 0),
            "planned_conversions": list(planned_conversions),
            "planned_conversion_count": int(sum(int(row.get("count", 0)) for row in planned_conversions)),
            "recommended_adjustments": list(recommendations),
            "resource_shortage": {str(k): int(shortage.get(str(k), 0)) for k in STAGE_C_RESOURCE_KEYS},
            "pending_closure_demand_summary": dict(demand_summary),
            "pending_closure_intent_demand": {},
            "pending_closure_resource_demand": dict(demand_summary["resource_demand"]),
            "pending_closure_resource_available": dict(demand_summary["resource_available"]),
            "pending_closure_resource_deficit": dict(demand_summary["resource_deficit"]),
            "minimum_conversions": list(demand_summary["minimum_conversions"]),
            "component_rows": list(component_rows),
            "component_issue_rows": list(issue_rows),
            "abstract_gap_by_kind": dict(abstract_gap_by_kind),
            "abstract_gap_audit_by_kind": dict(abstract_gap_by_kind),
            "unclosed_like24_count": int(unallocated_branch),
            "closed_topology_count": int(len(bridge_chains) + len(side_chains) + len(branch_chains)),
            "chain_23_limit_violations": list(chain_23_limit_violations),
            "residual_remaining": {str(k): 0 for k in STAGE_C_RESOURCE_KEYS},
            **dict(flex_diag),
        }
        self._commit_stage_c_graph_result(
            bridge_chains=bridge_chains,
            side_chains=side_chains,
            branch_chains=branch_chains,
            pending_chains=pending_chains,
            source_pool=source_pool if shortage_total > 0 else {str(k): 0 for k in STAGE_C_RESOURCE_KEYS},
            diagnostics=diag,
            component_rows=component_rows,
        )

    @staticmethod
    def _stage_c_is_relevant_su(su: int) -> bool:
        return int(su) in STAGE_C_TOPOLOGY_SU

    def _stage_c_adjacency(self) -> Dict[int, set[int]]:
        lookup = {int(n.global_id): n for n in self._nodes if int(n.global_id) >= 0}
        adj: Dict[int, set[int]] = {int(gid): set() for gid in lookup}
        for node in self._nodes:
            gid = int(node.global_id)
            if gid < 0:
                continue
            for nb_gid in list(node.hop1_ids or []):
                nb_i = int(nb_gid)
                if nb_i not in lookup or nb_i == gid:
                    continue
                adj.setdefault(gid, set()).add(nb_i)
                adj.setdefault(nb_i, set()).add(gid)
        return adj

    @staticmethod
    def _stage_c_components(ids: set[int], adj: Dict[int, set[int]]) -> List[set[int]]:
        left = set(int(x) for x in ids)
        comps: List[set[int]] = []
        while left:
            start = min(left)
            stack = [int(start)]
            comp: set[int] = set()
            left.remove(start)
            while stack:
                gid = int(stack.pop())
                comp.add(gid)
                for nb in sorted(adj.get(gid, set()) & ids):
                    if nb not in left:
                        continue
                    left.remove(nb)
                    stack.append(int(nb))
            comps.append(comp)
        return comps

    def _stage_c_order_component(
        self,
        nodes: List[SUNode],
        comp: set[int],
        adj: Dict[int, set[int]],
    ) -> List[SUNode]:
        node_by_id = {int(n.global_id): n for n in nodes}
        deg = {int(gid): len(adj.get(int(gid), set()) & set(comp)) for gid in comp}
        endpoints = [int(gid) for gid in comp if int(deg.get(int(gid), 0)) <= 1]

        def _rank(gid: int) -> Tuple[int, int, int]:
            su = int(node_by_id[int(gid)].su_type)
            if su in STAGE_C_ANCHOR_SU:
                cls = 0
            elif su in STAGE_C_TERMINAL_SU:
                cls = 2
            elif su in STAGE_C_BRANCH_CENTER_SU:
                cls = 3
            else:
                cls = 1
            return int(cls), int(su), int(gid)

        if endpoints:
            start = sorted(endpoints, key=_rank)[0]
            ordered_ids: List[int] = []
            prev = -1
            cur = int(start)
            visited: set[int] = set()
            while cur not in visited:
                visited.add(cur)
                ordered_ids.append(cur)
                nxts = [
                    int(x)
                    for x in sorted(adj.get(cur, set()) & set(comp))
                    if int(x) != int(prev) and int(x) not in visited
                ]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
            if len(ordered_ids) == len(comp):
                return [node_by_id[int(gid)] for gid in ordered_ids]
        return [node_by_id[int(gid)] for gid in sorted(comp, key=_rank)]

    def _stage_c_component_type(
        self,
        nodes: List[SUNode],
        comp: set[int],
        adj: Dict[int, set[int]],
    ) -> str:
        has_branch = any(
            int(n.su_type) in STAGE_C_BRANCH_CENTER_SU
            or len(adj.get(int(n.global_id), set()) & set(comp)) >= 3
            for n in nodes
        )
        if has_branch:
            return "branch"
        anchor_endpoints = 0
        for node in nodes:
            gid = int(node.global_id)
            su = int(node.su_type)
            deg = len(adj.get(gid, set()) & set(comp))
            if deg <= 1 and (su in STAGE_C_ANCHOR_SU or su == 11):
                anchor_endpoints += 1
        if int(anchor_endpoints) >= 2:
            return "bridge"
        return "side"

    def _stage_c_component_issues(
        self,
        nodes: List[SUNode],
        comp: set[int],
        adj: Dict[int, set[int]],
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for node in nodes:
            gid = int(node.global_id)
            su = int(node.su_type)
            deg = len(adj.get(gid, set()) & set(comp))
            target = self._stage_c_target_su_for_degree(su, deg)
            if target is None or int(target) == int(su):
                continue
            key = STAGE_C_RESOURCE_BY_SU.get(int(target))
            if key is None:
                continue
            issues.append({
                "global_id": int(gid),
                "su_type": int(su),
                "observed_degree": int(deg),
                "target_su": int(target),
                "shortage_key": str(key),
                "shortage_type": f"{key}_shortage",
                "reason": "stage_c_degree_role_mismatch",
            })
        return issues

    @staticmethod
    def _stage_c_target_su_for_degree(su: int, degree: int) -> Optional[int]:
        su_i = int(su)
        deg = max(0, int(degree))
        if su_i == 22:
            if deg <= 1:
                return 22
            if deg == 2:
                return 23
            return 24 if deg == 3 else 25
        if su_i == 23:
            if deg <= 1:
                return 22
            if deg == 2:
                return 23
            return 24 if deg == 3 else 25
        if su_i == 24:
            if deg <= 1:
                return 22
            if deg == 2:
                return 23
            if deg == 3:
                return 24
            return 25
        if su_i == 25:
            if deg <= 1:
                return 22
            if deg == 2:
                return 23
            if deg == 3:
                return 24
            return 25
        if su_i == 11:
            return 11 if deg <= 1 else (23 if deg == 2 else 24)
        return None

    def _stage_c_chain_spec(
        self,
        chain_type: str,
        nodes: List[SUNode],
        origin_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChainSpec:
        composition = [int(n.su_type) for n in list(nodes or [])]
        source_ids = [int(n.global_id) for n in list(nodes or []) if int(n.global_id) >= 0]
        resource_counts = {
            str(key): int(sum(1 for su in composition if int(su) == STAGE_C_SU_BY_RESOURCE[str(key)]))
            for key in STAGE_C_RESOURCE_KEYS
        }
        meta = dict(metadata or {})
        meta.update({
            "actual_topology_su": list(composition),
            "actual_used_su": list(composition),
            "actual_used_source_ids": list(source_ids),
            "resource_requirements_override": dict(resource_counts),
            "trace_required_counts": dict(resource_counts),
        })
        return ChainSpec(
            chain_type=str(chain_type),
            composition=list(composition),
            origin_type=str(origin_type),
            source_ids=list(source_ids),
            metadata=meta,
            trace_required_counts=dict(resource_counts),
        )

    def _stage_c_source_pool(self, shortage: Dict[str, int]) -> Dict[str, int]:
        totals = {
            "11": int(self._result.total_11),
            "22": int(self._result.total_22),
            "23": int(self._result.total_23),
            "24": int(self._result.total_24),
            "25": int(self._result.total_25),
        }
        return {
            str(key): (
                max(0, int(totals.get(str(key), 0)) - int(shortage.get(str(key), 0)))
                if int(shortage.get(str(key), 0)) > 0
                else max(0, int(totals.get(str(key), 0)))
            )
            for key in STAGE_C_RESOURCE_KEYS
        }

    @staticmethod
    def _stage_c_recommendations(shortage: Dict[str, int], source_pool: Dict[str, int]) -> List[Dict[str, Any]]:
        preferred = {
            "23": ["22->23", "24->23", "11->23", "12/13->11->23"],
            "24": ["23->24", "25->24", "22->23->24", "11->23->24", "12/13->11->23->24"],
            "22": ["23->22", "24->23->22", "11->23->22", "12/13->11->23->22"],
            "11": ["23->11", "22->23->11", "24->23->11", "12/13->11"],
            "25": [],
        }
        out: List[Dict[str, Any]] = []
        for key in ("23", "24", "22", "11", "25"):
            count = max(0, int(dict(shortage or {}).get(str(key), 0)))
            if count <= 0:
                continue
            ops = list(preferred.get(str(key), []) or [])
            out.append({
                "op": ops[0] if ops else "",
                "preferred_ops": list(ops),
                "count": int(count),
                "phase": "closure_repair",
                "shortage_key": str(key),
                "shortage_type": f"{key}_shortage",
                "reason": "stage_c_graph_degree_role_mismatch",
                "resource_surplus": {str(k): int(v) for k, v in dict(source_pool or {}).items()},
                "actionable": bool(ops),
            })
        return out

    def _stage_c_flex_diagnostics(
        self,
        bridge_chains: List[ChainSpec],
        branch_chains: List[ChainSpec],
    ) -> Dict[str, Any]:
        constraints = self._topology_constraints_with_flex_window()
        flex_lower = int(constraints.get("flex_lower", constraints.get("flexible_bridge_min", 0)) or 0)
        flex_upper = int(constraints.get("flex_upper", constraints.get("flexible_bridge_limit", flex_lower)) or flex_lower)
        flex_count = 0
        for chain in list(bridge_chains or []) + list(branch_chains or []):
            comp = [int(x) for x in list(getattr(chain, "composition", []) or [])]
            if len(comp) >= 3 and int(comp[0]) == 11 and int(comp[-1]) == 11 and 23 in set(comp[1:-1]):
                flex_count += 1
        flex_window_conflict = bool(int(flex_lower) > int(flex_upper)) or not bool(constraints.get("flex_window_ok", True))
        return {
            "flex_lower": int(flex_lower),
            "flex_upper": int(flex_upper),
            "flex_lower_raw": int(constraints.get("flex_lower_raw", flex_lower)),
            "flex_upper_raw": int(constraints.get("flex_upper_raw", flex_upper)),
            "flexible_bridge_min": int(flex_lower),
            "flexible_bridge_limit": int(flex_upper),
            "flex_count": int(flex_count),
            "flex_short": 0 if flex_window_conflict else max(0, int(flex_lower) - int(flex_count)),
            "flex_excess": 0 if flex_window_conflict else max(0, int(flex_count) - int(flex_upper)),
            "flex_window_ok": bool(not flex_window_conflict),
            "flex_window_conflict": bool(flex_window_conflict),
            "flex_window_source": str(constraints.get("flex_window_source", "")),
        }

    def _commit_stage_c_graph_result(
        self,
        bridge_chains: List[ChainSpec],
        side_chains: List[ChainSpec],
        branch_chains: List[ChainSpec],
        pending_chains: List[ChainSpec],
        source_pool: Dict[str, int],
        diagnostics: Dict[str, Any],
        component_rows: List[Dict[str, Any]],
    ) -> None:
        r = self._result
        r.bridge_chains = list(bridge_chains or [])
        r.side_chains = list(side_chains or [])
        r.branch_chains = list(branch_chains or [])
        r.pending_closure_topologies = list(pending_chains or [])
        closed = list(r.bridge_chains) + list(r.side_chains) + list(r.branch_chains)
        r.closed_topology_specs = list(closed)
        r.final_topology_specs = list(closed)
        r.residual_topology_specs = []
        r.topology_assembly_diagnostics = dict(diagnostics or {})
        r.topology_selected_options = []
        r.topology_assembly_options = []
        r.topology_unclosed_candidates = list(component_rows or [])
        r.topology_classification_rejects = list((diagnostics or {}).get("component_issue_rows", []) or [])
        shortage = dict((diagnostics or {}).get("resource_shortage", {}) or {})
        r.required_extra_11 = int(shortage.get("11", 0))
        r.required_extra_22 = int(shortage.get("22", 0))
        r.required_extra_23 = int(shortage.get("23", 0))
        r.required_extra_24 = int(shortage.get("24", 0))
        r.required_extra_25 = int(shortage.get("25", 0))
        r.unallocated_branch = int((diagnostics or {}).get("unclosed_like24_count", 0))
        r.unallocated_bridge = 0
        r.topology_plan_feasible = bool((diagnostics or {}).get("topology_plan_feasible", False))
        r.topology_materialized_ok = bool((diagnostics or {}).get("topology_materialized_ok", False))
        r.materialization_required = bool((diagnostics or {}).get("materialization_required", False))
        r.planned_conversions = list((diagnostics or {}).get("planned_conversions", []) or [])
        r.residual_growth_actions = list((diagnostics or {}).get("residual_growth_actions", []) or [])
        r.residual_remaining_by_kind = dict((diagnostics or {}).get("residual_remaining", {}) or {})
        r.chain_23_limit_violations = list((diagnostics or {}).get("chain_23_limit_violations", []) or [])
        r.pending_closure_count = int(len(r.pending_closure_topologies))
        r.pending_closure_blocking_count = int(len(r.pending_closure_topologies))
        r.pending_closure_total_count = int(len(r.pending_closure_topologies))
        r.pending_closure_deferred_count = 0
        r.trace_exact_chain_count = int(len(closed))
        r.trace_inexact_chain_count = int(len(r.pending_closure_topologies))
        r.abstract_gap_by_kind = dict((diagnostics or {}).get("abstract_gap_by_kind", {}) or {})
        r.abstract_gap_audit_by_kind = dict((diagnostics or {}).get("abstract_gap_audit_by_kind", r.abstract_gap_by_kind) or {})
        r.type_counts = dict(Counter(str(getattr(ch, "origin_type", "")) for ch in closed + list(r.pending_closure_topologies)))
        r.catalog_summary = dict(r.type_counts)
        r.fragment_rows = [self._chain_row(ch, idx) for idx, ch in enumerate(closed)]
        r.pending_closure_rows = [self._chain_row(ch, idx) for idx, ch in enumerate(r.pending_closure_topologies)]
        r.catalog_fragment_rows = list(r.fragment_rows) + list(r.pending_closure_rows)

        required = _sum_chain_required_resource_counts(closed)
        r.consumed_11 = int(required.get("11", 0))
        r.consumed_22 = int(required.get("22", 0))
        r.consumed_23 = int(required.get("23", 0))
        r.consumed_24 = int(required.get("24", 0))
        r.consumed_25 = int(required.get("25", 0))
        r.remaining_11 = int(source_pool.get("11", 0))
        r.remaining_22 = int(source_pool.get("22", 0))
        r.remaining_23 = int(source_pool.get("23", 0))
        r.remaining_24 = int(source_pool.get("24", 0))
        r.remaining_25 = int(source_pool.get("25", 0))
        r.native_total_by_kind = {
            "11": int(r.total_11), "22": int(r.total_22), "23": int(r.total_23),
            "24": int(r.total_24), "25": int(r.total_25),
        }
        r.native_remaining_by_kind = {
            "11": int(r.remaining_11), "22": int(r.remaining_22), "23": int(r.remaining_23),
            "24": int(r.remaining_24), "25": int(r.remaining_25),
        }
        r.native_consumed_by_kind = {
            key: max(0, int(r.native_total_by_kind.get(key, 0)) - int(r.native_remaining_by_kind.get(key, 0)))
            for key in STAGE_C_RESOURCE_KEYS
        }
        r.proxy_total_by_kind = {key: 0 for key in STAGE_C_RESOURCE_KEYS}
        r.proxy_consumed_by_kind = {key: 0 for key in STAGE_C_RESOURCE_KEYS}
        r.proxy_remaining_by_kind = {key: 0 for key in STAGE_C_RESOURCE_KEYS}
        r.section_consumed_by_kind = {
            "bridge": _sum_chain_required_resource_counts(r.bridge_chains),
            "side": _sum_chain_required_resource_counts(r.side_chains),
            "branch": _sum_chain_required_resource_counts(r.branch_chains),
            "pending_closure": {key: 0 for key in STAGE_C_RESOURCE_KEYS},
        }
        r.origin_consumed_by_kind = {}
        r.trace_consumed_by_section_kind = dict(r.section_consumed_by_kind)
        r.trace_consumed_by_origin_kind = {}
        source_counts = Counter(int(n.su_type) for n in self._nodes)
        r.source_su_total_counts = dict(sorted((int(k), int(v)) for k, v in source_counts.items()))
        r.source_su_consumed_counts = {}
        r.source_su_remaining_counts = dict(r.source_su_total_counts)

    # ---------- parsing ----------

    def _parse_input(self) -> None:
        self._nodes = []
        self._node_lookup = {}
        if self._input_nodes is not None:
            self._parse_nodes(self._input_nodes)
        elif self.csv_path:
            self._parse_csv(self.csv_path)
        elif self._su_counts:
            print("  [FlexAllocator] su_counts-only mode: no 1-hop motif extraction")
        else:
            raise ValueError("Must provide csv_path, nodes, or su_counts")
        self._refresh_hop1_from_ids()

    def _parse_nodes(self, nodes: Sequence[Any]) -> None:
        raw_lookup: Dict[int, Any] = {}
        for node in list(nodes or []):
            gid = node_id(node)
            if gid >= 0:
                raw_lookup[int(gid)] = node
        self._su_counts = {}
        for raw in sorted(raw_lookup.values(), key=node_id):
            hop1_ids = tuple(int(x) for x in list(getattr(raw, "hop1_ids", ()) or ()))
            hop1 = [
                int(node_su(raw_lookup[nid]))
                for nid in hop1_ids
                if int(nid) in raw_lookup
            ]
            node = SUNode(
                global_id=node_id(raw),
                su_type=node_su(raw),
                hop1=tuple(hop1),
                hop2=tuple(),
                target_degree=self._first_int(raw, "target_degree", "target_hop1_degree", "actual_hop1_degree"),
                target_fixed_anchor_count=self._first_int(raw, "target_fixed_anchor_count"),
                anchor_partition=self._first_str(raw, "anchor_partition", "special_anchor_partition"),
                anchor_mode=self._first_str(raw, "anchor_mode", "special_anchor_mode"),
                hop1_ids=hop1_ids,
            )
            self._add_node(node)

    def _parse_csv(self, csv_path: str) -> None:
        if pd is None:
            self._parse_csv_without_pandas(csv_path)
            return
        df = pd.read_csv(csv_path)
        su_col = next((c for c in ("center_su_idx", "su_type", "su_idx", "type") if c in df.columns), None)
        if su_col is None:
            raise ValueError(f"Cannot find SU type column in {csv_path}")
        self._su_counts = dict(Counter(int(x) for x in df[su_col].values))
        for idx, row in df.iterrows():
            gid = int(row.get("global_id", idx))
            hop1: List[int] = []
            hop2: List[int] = []
            if "template_key" in df.columns and not pd.isna(row.get("template_key", None)):
                from .RL_init import _parse_template_key

                _, hop1, hop2 = _parse_template_key(str(row["template_key"]))
            if not hop1 and "hop1_ms" in df.columns:
                hop1 = self._parse_multiset(row.get("hop1_ms", None))
            if not hop2 and "hop2_ms" in df.columns:
                hop2 = self._parse_multiset(row.get("hop2_ms", None))
            node = SUNode(
                global_id=gid,
                su_type=int(row[su_col]),
                hop1=tuple(hop1),
                hop2=tuple(hop2),
                target_degree=self._parse_optional_int(row.get("target_hop1_degree", row.get("target_degree", None))),
                target_fixed_anchor_count=self._parse_optional_int(row.get("target_fixed_anchor_count", None)),
                anchor_partition=self._parse_optional_str(row.get("special_anchor_partition", row.get("anchor_partition", None))),
                anchor_mode=self._parse_optional_str(row.get("special_anchor_mode", row.get("anchor_mode", None))),
                hop1_ids=self._parse_int_tuple(row.get("hop1_ids", None)),
            )
            self._add_node(node)

    def _parse_csv_without_pandas(self, csv_path: str) -> None:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            su_col = next((c for c in ("center_su_idx", "su_type", "su_idx", "type") if c in fieldnames), None)
            if su_col is None:
                raise ValueError(f"Cannot find SU type column in {csv_path}")
            rows = list(reader)
        self._su_counts = dict(Counter(int(float(row[su_col])) for row in rows if str(row.get(su_col, "")).strip()))
        for idx, row in enumerate(rows):
            gid_raw = row.get("global_id", idx)
            gid = int(float(gid_raw)) if str(gid_raw).strip() else int(idx)
            hop1: List[int] = []
            hop2: List[int] = []
            template_key = row.get("template_key", None)
            if template_key not in {None, ""}:
                try:
                    from .RL_init import _parse_template_key

                    _, hop1, hop2 = _parse_template_key(str(template_key))
                except Exception:
                    hop1, hop2 = [], []
            if not hop1:
                hop1 = self._parse_multiset(row.get("hop1_ms", None))
            if not hop2:
                hop2 = self._parse_multiset(row.get("hop2_ms", None))
            node = SUNode(
                global_id=gid,
                su_type=int(float(row[su_col])),
                hop1=tuple(hop1),
                hop2=tuple(hop2),
                target_degree=self._parse_optional_int(row.get("target_hop1_degree", row.get("target_degree", None))),
                target_fixed_anchor_count=self._parse_optional_int(row.get("target_fixed_anchor_count", None)),
                anchor_partition=self._parse_optional_str(row.get("special_anchor_partition", row.get("anchor_partition", None))),
                anchor_mode=self._parse_optional_str(row.get("special_anchor_mode", row.get("anchor_mode", None))),
                hop1_ids=self._parse_int_tuple(row.get("hop1_ids", None)),
            )
            self._add_node(node)

    def _add_node(self, node: SUNode) -> None:
        self._nodes.append(node)
        self._node_lookup[int(node.global_id)] = node

    def _refresh_hop1_from_ids(self) -> None:
        for node in self._nodes:
            if not node.hop1_ids:
                continue
            vals = [
                int(self._node_lookup[int(nid)].su_type)
                for nid in node.hop1_ids
                if int(nid) in self._node_lookup
            ]
            if vals:
                node.hop1 = tuple(vals)
            if node.target_degree is None and vals:
                node.target_degree = len(vals)

    @staticmethod
    def _first_int(obj: Any, *names: str) -> Optional[int]:
        for name in names:
            raw = getattr(obj, name, None)
            if raw is None:
                continue
            try:
                return int(raw)
            except Exception:
                continue
        return None

    @staticmethod
    def _first_str(obj: Any, *names: str) -> Optional[str]:
        for name in names:
            raw = getattr(obj, name, None)
            if raw is None:
                continue
            text = str(raw).strip()
            if text and text.lower() != "none":
                return text
        return None

    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if _is_missing_value(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "none":
            return None
        try:
            return int(float(text))
        except Exception:
            return None

    @staticmethod
    def _parse_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if _is_missing_value(value):
            return None
        text = str(value).strip()
        return text if text and text.lower() != "none" else None

    @staticmethod
    def _parse_int_tuple(value: Any) -> Tuple[int, ...]:
        if value is None:
            return tuple()
        if _is_missing_value(value):
            return tuple()
        if isinstance(value, (list, tuple, set)):
            return tuple(int(x) for x in value)
        text = str(value).strip()
        if not text or text.lower() == "none":
            return tuple()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return tuple(int(x) for x in parsed)
        return tuple(int(x.strip()) for x in text.split(",") if x.strip())

    @staticmethod
    def _parse_multiset(value: Any) -> List[int]:
        if value is None:
            return []
        if _is_missing_value(value):
            return []
        if isinstance(value, (list, tuple, set)):
            return [int(x) for x in value]
        text = str(value).strip()
        if not text or text.lower() == "none":
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [int(x) for x in parsed]
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    # ---------- motif extraction ----------

    def _neighbor_states(self, node: SUNode) -> List[Tuple[SUNode, Any]]:
        out: List[Tuple[SUNode, Any]] = []
        if node.hop1_ids:
            for nid in node.hop1_ids:
                nb = self._node_lookup.get(int(nid))
                if nb is not None:
                    out.append((nb, state_for_node(nb)))
        if out:
            return out
        for idx, su in enumerate(node.hop1):
            proxy = SUNode(global_id=-1000000 - idx, su_type=int(su))
            out.append((proxy, state_for_node(proxy)))
        return out

    def _make_fragment(
        self,
        kind: str,
        states: Sequence[Any],
        left_open: str,
        right_open: str,
        origin_rule: str,
        abstract_indices: Optional[Sequence[int]] = None,
        abstracted_neighbors: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> TopologyFragment:
        st = list(states)
        abstract_idx = {int(x) for x in list(abstract_indices or [])}
        su_path: List[int] = []
        node_ids: List[int] = []
        labels: List[str] = []
        abstracted: List[Dict[str, Any]] = [
            dict(row) for row in list(abstracted_neighbors or [])
            if isinstance(row, dict)
        ]
        for idx, item in enumerate(st):
            if int(idx) in abstract_idx:
                su_path.append(23)
                node_ids.append(-1)
                labels.append("23")
                abstracted.append(self._abstracted_neighbor_row(int(idx), item, "generic_chain_port_to_23"))
                continue
            su_path.append(int(item.su))
            node_ids.append(int(item.gid))
            labels.append(str(item.label))
        meta = {"origin_rule": str(origin_rule)}
        if abstracted:
            meta["abstracted_chain_port_neighbors"] = list(abstracted)
        return TopologyFragment(
            kind=str(kind),
            su_path=su_path,
            node_ids=node_ids,
            labels=labels,
            left_open=str(left_open),
            right_open=str(right_open),
            closed=left_open == "closed" and right_open == "closed",
            origin_rule=str(origin_rule),
            metadata=meta,
        )

    @staticmethod
    def _abstracted_neighbor_row(index: int, state: Any, reason: str) -> Dict[str, Any]:
        return {
            "path_index": int(index),
            "actual_gid": int(getattr(state, "gid", -1)),
            "actual_su": int(getattr(state, "su", -1)),
            "actual_label": str(getattr(state, "label", "")),
            "topology_su": 23,
            "reason": str(reason),
        }

    @staticmethod
    def _proxy_state(su: int) -> Any:
        return state_for_node(SUNode(global_id=-2300000 - int(su), su_type=int(su)))

    def _extract_double_d2_fragments(self) -> Tuple[List[TopologyFragment], set[int]]:
        fragments: List[TopologyFragment] = []
        consumed: set[int] = set()
        seen = set()
        for node in self._nodes:
            if int(node.global_id) in consumed:
                continue
            center = state_for_node(node)
            if not is_double_special_d2(center, 19) and not is_double_special_d2(center, 20):
                continue
            nbs = [
                (nb, st)
                for nb, st in self._neighbor_states(node)
                if int(nb.global_id) >= 0
            ]
            halves: List[_DoubleHalf] = []
            for nb, st in nbs:
                half = self._classify_double_half(center, nb, st)
                if half is not None:
                    halves.append(half)
            pair = self._choose_double_half_pair(halves)
            if pair is None:
                continue
            frag = self._combine_double_halves(center, pair[0], pair[1])
            if frag is None:
                continue
            if self._fragment_hits_skip(frag, consumed):
                continue
            sig = fragment_signature(frag)
            if sig in seen:
                continue
            seen.add(sig)
            fragments.append(frag)
            for gid in frag.node_ids:
                if int(gid) >= 0:
                    consumed.add(int(gid))
        return fragments, consumed

    def _branch_reserved_terminal_source_ids(self, branch_chains: Sequence[ChainSpec]) -> set[int]:
        reserved: set[int] = set()
        for chain in list(branch_chains or []):
            for gid in list(getattr(chain, "source_ids", []) or []):
                gid_i = int(gid)
                if gid_i < 0:
                    continue
                node = self._node_lookup.get(gid_i)
                if node is None:
                    continue
                if int(getattr(node, "su_type", -1)) in BRANCH_RESERVED_TERMINAL_SU:
                    reserved.add(gid_i)
        return reserved

    @staticmethod
    def _choose_double_half_pair(halves: Sequence[_DoubleHalf]) -> Optional[Tuple[_DoubleHalf, _DoubleHalf]]:
        candidates: List[Tuple[int, _DoubleHalf, _DoubleHalf]] = []
        priority = {
            ("M", "M"): 0,
            ("M", "P"): 1,
            ("M", "N"): 2,
            ("N", "P"): 3,
            ("N", "N"): 4,
        }
        values = list(halves or [])
        for i, left in enumerate(values):
            for j, right in enumerate(values):
                if i >= j:
                    continue
                if int(left.connector_id) == int(right.connector_id):
                    continue
                key = tuple(sorted((left.kind, right.kind)))
                if key not in priority:
                    continue
                candidates.append((priority[key], left, right))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], int(item[1].connector_id), int(item[2].connector_id)))
        return candidates[0][1], candidates[0][2]

    def _classify_double_half(self, double_center: Any, connector_node: SUNode, connector: Any) -> Optional[_DoubleHalf]:
        target_su = 19 if int(double_center.su) == 19 else 20
        connector_su = int(connector.su)
        connector_neighbors = [
            st for _nb, st in self._neighbor_states(connector_node)
        ]
        if has_forbidden_fixed_connector_special_combo(connector, connector_neighbors):
            return None
        nbs = [
            (nb, st)
            for nb, st in self._neighbor_states(connector_node)
            if int(st.gid) != int(double_center.gid)
        ]
        outside = [st for _nb, st in nbs]

        # P special case: 28-double19d2 has no additional outside neighbor.
        if target_su == 19 and connector_su == 28:
            return _DoubleHalf(
                kind="P",
                states=[connector],
                connector_id=int(connector.gid),
                outer_open="terminal",
                origin_rule="P:28-double19d2",
            )

        for outer in outside:
            if (kind := self._double_half_kind(target_su, connector_su, outer)) is None:
                continue
            outer_for_path, abstracted = self._double_half_outer_for_path(connector_su, kind, outer)
            states = [outer, connector]
            path_states = [outer_for_path, connector]
            outer_open = {"M": "closed", "N": "chain", "P": "terminal"}[kind]
            return _DoubleHalf(
                kind=kind,
                states=path_states,
                connector_id=int(connector.gid),
                outer_open=outer_open,
                origin_rule=f"{kind}:{outer.label}-{connector_su}-double{target_su}d2",
                actual_states=states,
                abstracted_neighbors=abstracted,
            )
        return None

    def _double_half_outer_for_path(
        self,
        connector_su: int,
        kind: str,
        outer: Any,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        if str(kind) == "N" and int(connector_su) in {0, 2} and is_abstract_chain_port_state(outer):
            return self._proxy_state(23), [self._abstracted_neighbor_row(0, outer, "double_d2_N_generic_chain_port_to_23")]
        return outer, []

    def _double_half_kind(self, target_su: int, connector_su: int, outer: Any) -> Optional[str]:
        if target_su == 20:
            if connector_su == 0:
                if int(outer.su) == 9:
                    return "M"
                if self._double_chain_for_0(outer):
                    return "N"
                if int(outer.su) == 22:
                    return "P"
            if connector_su == 27:
                if int(outer.su) == 6:
                    return "M"
                if is_single_special_d2(outer, 20):
                    return "N"
                if int(outer.su) == 20 and int(outer.degree or 0) == 1:
                    return "P"
            return None

        if target_su == 19:
            if connector_su == 2:
                if int(outer.su) == 9:
                    return "M"
                if self._double_chain_for_2(outer):
                    return "N"
                if int(outer.su) == 22:
                    return "P"
            if connector_su == 29:
                if int(outer.su) == 5:
                    return "M"
                if is_single_special_d2(outer, 19):
                    return "N"
                if int(outer.su) == 19 and int(outer.degree or 0) == 1:
                    return "P"
            if connector_su == 31:
                if int(outer.su) == 7:
                    return "M"
                if is_single_special_d2(outer, 19):
                    return "N"
                if int(outer.su) == 19 and int(outer.degree or 0) == 1:
                    return "P"
        return None

    def _double_chain_for_0(self, state: Any) -> bool:
        return self._generic_chain(state)

    def _double_chain_for_2(self, state: Any) -> bool:
        return self._generic_chain(state)

    def _combine_double_halves(self, center: Any, left: _DoubleHalf, right: _DoubleHalf) -> Optional[TopologyFragment]:
        a, b = self._orient_double_pair(left, right)
        combo_key = tuple(sorted((a.kind, b.kind)))
        combo = {
            ("M", "M"): ("A+", "closed", "closed"),
            ("M", "P"): ("C+", "closed", "closed"),
            ("M", "N"): ("B+", "closed", "chain"),
            ("N", "N"): ("D+", "chain", "chain"),
            ("N", "P"): ("E+", "chain", "terminal"),
        }.get(combo_key)
        if combo is None:
            return None
        kind, left_open, right_open = combo
        states = list(a.states) + [center] + list(reversed(b.states))
        abstracted: List[Dict[str, Any]] = []
        for row in list(getattr(a, "abstracted_neighbors", []) or []):
            data = dict(row)
            data["half"] = "left"
            abstracted.append(data)
        right_offset = len(list(a.states or [])) + 1
        for row in list(getattr(b, "abstracted_neighbors", []) or []):
            data = dict(row)
            data["half"] = "right"
            data["path_index"] = int(right_offset + (len(list(b.states or [])) - 1 - int(data.get("path_index", 0))))
            abstracted.append(data)
        frag = self._make_fragment(
            kind,
            states,
            left_open,
            right_open,
            f"{kind}:{a.origin_rule}+{b.origin_rule}",
            abstracted_neighbors=abstracted,
        )
        frag.metadata.update({
            "double_center_label": str(center.label),
            "double_center_id": int(center.gid),
            "half_kinds": [a.kind, b.kind],
            "half_rules": [a.origin_rule, b.origin_rule],
            "origin_rule": frag.origin_rule,
        })
        return frag

    @staticmethod
    def _orient_double_pair(left: _DoubleHalf, right: _DoubleHalf) -> Tuple[_DoubleHalf, _DoubleHalf]:
        # Place the more closed half on the left for stable output:
        # M+N -> M...N, N+P -> N...P, M+P -> M...P.
        order = {"M": 0, "N": 1, "P": 2}
        if order.get(left.kind, 99) <= order.get(right.kind, 99):
            return left, right
        return right, left

    def _extract_fragments(self, skip_node_ids: Optional[set[int]] = None) -> List[TopologyFragment]:
        skip = {int(x) for x in set(skip_node_ids or set()) if int(x) >= 0}
        fragments: List[TopologyFragment] = []
        seen = set()
        for node in self._nodes:
            if int(node.global_id) in skip:
                continue
            center = state_for_node(node)
            nbs = [(nb, st) for nb, st in self._neighbor_states(node)]
            for frag in self._match_node_fragments(center, nbs):
                if self._fragment_hits_skip(frag, skip):
                    continue
                sig = fragment_signature(frag)
                if sig in seen:
                    continue
                seen.add(sig)
                fragments.append(frag)
        for frag in self._extract_terminal_fragments(skip):
            if self._fragment_hits_skip(frag, skip):
                continue
            sig = fragment_signature(frag)
            if sig in seen:
                continue
            seen.add(sig)
            fragments.append(frag)
        return fragments

    @staticmethod
    def _fragment_hits_skip(frag: TopologyFragment, skip_node_ids: set[int]) -> bool:
        return any(int(gid) >= 0 and int(gid) in skip_node_ids for gid in frag.node_ids)

    def _match_node_fragments(self, center: Any, nbs: Sequence[Tuple[SUNode, Any]]) -> List[TopologyFragment]:
        out: List[TopologyFragment] = []
        if int(center.su) not in {0, 2, 3, 15, 27, 29, 31}:
            return out
        states = [st for _nb, st in list(nbs or [])]
        if int(center.su) == 3 and len(states) == 1 and int(states[0].su) == 22:
            proxy = self._proxy_state(23)
            out.append(
                self._make_fragment(
                    "E",
                    [states[0], center, proxy],
                    "terminal",
                    "chain",
                    "E:22-3-pending-chain",
                )
            )
            return out
        for i, left in enumerate(states):
            for j, right in enumerate(states):
                if i == j:
                    continue
                frag = self._classify_oriented(center, left, right)
                if frag is not None:
                    out.append(frag)
        return out

    def _classify_oriented(self, center: Any, left: Any, right: Any) -> Optional[TopologyFragment]:
        c = int(center.su)

        # A: closed bridge fragments.
        if c == 0 and left.su == 9 and right.su == 6:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:9-0-6")
        if c == 2 and left.su == 9 and right.su == 5:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:9-2-5")
        if c == 3 and left.su == 9 and right.su == 9:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:9-3-9")
        if c == 27 and left.su == 6 and right.su == 6:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:6-27-6")
        if c == 29 and left.su == 5 and right.su == 5:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:5-29-5")
        if c == 31 and left.su == 7 and right.su == 7:
            return self._make_fragment("A", [left, center, right], "closed", "closed", "A:7-31-7")

        # B: aromatic side closed on the left, chain-open on the right.
        if c == 0 and left.su == 6 and self._b_chain_for_0(right):
            return self._make_fragment("B", [left, center, right], "closed", "chain", "B:6-0-chain", abstract_indices=[2])
        if c == 0 and left.su == 9 and is_single_special_d2(right, 20):
            return self._make_fragment("B", [left, center, right], "closed", "chain", "B:9-0-single20d2")
        if c == 2 and left.su == 5 and self._b_chain_for_2(right):
            return self._make_fragment("B", [left, center, right], "closed", "chain", "B:5-2-chain", abstract_indices=[2])
        if c == 2 and left.su == 9 and is_single_special_d2(right, 19):
            return self._make_fragment("B", [left, center, right], "closed", "chain", "B:9-2-single19d2")
        if c == 3 and left.su == 9 and self._generic_chain(right):
            return self._make_fragment("B", [left, center, right], "closed", "chain", "B:9-3-chain", abstract_indices=[2])
        if c == 27 and left.su == 6 and self._single_special_chain(right, 20):
            return self._make_fragment("B", [left, center, right], "closed", "chain", f"B:6-27-{right.label}")
        if c == 29 and left.su == 5 and self._single_special_chain(right, 19):
            return self._make_fragment("B", [left, center, right], "closed", "chain", f"B:5-29-{right.label}")
        if c == 31 and left.su == 7 and self._single_special_chain(right, 19):
            return self._make_fragment("B", [left, center, right], "closed", "chain", f"B:7-31-{right.label}")

        # C: closed side fragments.
        if c == 0 and left.su == 6 and is_terminal_state(right):
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:6-0-terminal")
        if c == 0 and left.su == 9 and right.su == 20 and right.degree == 1:
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:9-0-20d1")
        if c == 2 and left.su == 5 and is_terminal_state(right):
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:5-2-terminal")
        if c == 2 and left.su == 9 and right.su == 19 and right.degree == 1:
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:9-2-19d1")
        if c == 3 and left.su == 9 and is_terminal_state(right):
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:9-3-terminal")
        if c == 27 and left.su == 6 and right.su == 20 and right.degree == 1:
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:6-27-20d1")
        if c == 29 and left.su == 5 and right.su == 19 and right.degree == 1:
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:5-29-19d1")
        if c == 31 and left.su == 7 and right.su == 19 and right.degree == 1:
            return self._make_fragment("C", [left, center, right], "closed", "closed", "C:7-31-19d1")

        # D: open-open chain fragments.
        if c == 0 and self._b_chain_for_0(left) and is_single_special_d2(right, 20):
            return self._make_fragment("D", [left, center, right], "chain", "chain", "D:chain-0-single20d2", abstract_indices=[0])
        if c == 2 and self._b_chain_for_2(left) and is_single_special_d2(right, 19):
            return self._make_fragment("D", [left, center, right], "chain", "chain", "D:chain-2-single19d2", abstract_indices=[0])
        if c == 3 and self._generic_chain(left) and self._generic_chain(right):
            return self._make_fragment("D", [left, center, right], "chain", "chain", "D:chain-3-chain", abstract_indices=[0, 2])
        if c == 27 and self._single_special_chain(left, 20) and self._single_special_chain(right, 20):
            return self._make_fragment("D", [left, center, right], "chain", "chain", f"D:{left.label}-27-{right.label}")
        if c == 29 and self._single_special_chain(left, 19) and self._single_special_chain(right, 19):
            return self._make_fragment("D", [left, center, right], "chain", "chain", f"D:{left.label}-29-{right.label}")
        if c == 31 and self._single_special_chain(left, 19) and self._single_special_chain(right, 19):
            return self._make_fragment("D", [left, center, right], "chain", "chain", f"D:{left.label}-31-{right.label}")
        if c == 15 and self._generic_chain(left) and right.su == 15:
            return self._make_fragment("D", [left, center, right], "chain", "chain", "D:chain-15-15", abstract_indices=[0])

        # E: terminal-open side-tail fragments.
        if c == 0 and is_terminal_state(left) and is_single_special_d2(right, 20):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", "E:terminal-0-single20d2")
        if c == 0 and self._b_chain_for_0(left) and right.su == 20 and right.degree == 1:
            return self._make_fragment("E", [left, center, right], "chain", "terminal", "E:chain-0-20d1", abstract_indices=[0])
        if c == 2 and is_terminal_state(left) and is_single_special_d2(right, 19):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", "E:terminal-2-single19d2")
        if c == 2 and self._b_chain_for_2(left) and right.su == 19 and right.degree == 1:
            return self._make_fragment("E", [left, center, right], "chain", "terminal", "E:chain-2-19d1", abstract_indices=[0])
        if c == 3 and self._generic_chain(right) and is_terminal_state(left):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", "E:terminal-3-chain", abstract_indices=[2])
        if c == 15 and left.su in {1, 4, 22} and right.su == 15:
            return self._make_fragment("E", [left, center, right], "terminal", "chain", "E:terminal-15-15")
        if c == 27 and left.su == 20 and left.degree == 1 and self._single_special_chain(right, 20):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", f"E:20d1-27-{right.label}")
        if c == 29 and left.su == 19 and left.degree == 1 and self._single_special_chain(right, 19):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", f"E:19d1-29-{right.label}")
        if c == 31 and left.su == 19 and left.degree == 1 and self._single_special_chain(right, 19):
            return self._make_fragment("E", [left, center, right], "terminal", "chain", f"E:19d1-31-{right.label}")

        p_half = self._classify_residual_p_double_half(center, left, right)
        if p_half is not None:
            return p_half

        return None

    def _generic_chain(self, state: Any) -> bool:
        return is_chain_state(state)

    @staticmethod
    def _single_special_chain(state: Any, su: int) -> bool:
        return bool(is_single_special_d2(state, su) or is_single_special_d3(state, su))

    def _classify_residual_p_double_half(self, center: Any, left: Any, right: Any) -> Optional[TopologyFragment]:
        c = int(center.su)
        target_su = 20 if c in {0, 27} else 19
        if not is_double_special_d2(right, target_su):
            return None

        terminal_ok = False
        if c in {0, 2} and int(left.su) == 22:
            terminal_ok = True
        elif c == 27 and int(left.su) == 20 and int(left.degree or 0) == 1:
            terminal_ok = True
        elif c in {29, 31} and int(left.su) == 19 and int(left.degree or 0) == 1:
            terminal_ok = True
        if not bool(terminal_ok):
            return None

        proxy = self._proxy_state(23)
        abstracted = [{
            "path_index": 2,
            "actual_gid": int(getattr(right, "gid", -1)),
            "actual_su": int(getattr(right, "su", -1)),
            "actual_label": str(getattr(right, "label", "")),
            "topology_su": 23,
            "reason": "residual_P_double_d2_half_to_chain_port",
        }]
        return self._make_fragment(
            "E",
            [left, center, proxy],
            "terminal",
            "chain",
            f"E:P_half:{left.label}-{c}-{right.label}",
            abstracted_neighbors=abstracted,
        )

    def _b_chain_for_0(self, state: Any) -> bool:
        return self._generic_chain(state)

    def _b_chain_for_2(self, state: Any) -> bool:
        return self._generic_chain(state)

    def _extract_terminal_fragments(self, skip_node_ids: Optional[set[int]] = None) -> List[TopologyFragment]:
        skip = {int(x) for x in set(skip_node_ids or set()) if int(x) >= 0}
        out: List[TopologyFragment] = []
        for node in self._nodes:
            if int(node.global_id) in skip:
                continue
            center = state_for_node(node)
            nbs = [(nb, st) for nb, st in self._neighbor_states(node)]
            for nb_node, nb in nbs:
                nb_skipped = int(getattr(nb_node, "global_id", -1)) in skip
                if bool(nb_skipped) and not (center.su == 28 and is_double_special_d2(nb, 19)):
                    continue
                if center.su == 1 and nb.su == 9:
                    out.append(self._make_fragment("F", [nb, center], "closed", "closed", "F:9-1"))
                elif center.su == 28 and nb.su == 5:
                    out.append(self._make_fragment("F", [nb, center], "closed", "closed", "F:5-28"))
                elif center.su == 32 and nb.su == 8:
                    out.append(self._make_fragment("F", [nb, center], "closed", "closed", "F:8-32"))
                elif center.su == 1 and self._generic_chain(nb):
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:chain-1", abstract_indices=[0]))
                elif center.su == 4 and nb.su in {23, 24, 25}:
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:chain-4", abstract_indices=[0]))
                elif center.su == 28 and is_single_special_d2(nb, 19):
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:single19d2-28"))
                elif center.su == 28 and is_double_special_d2(nb, 19):
                    proxy = self._proxy_state(23)
                    out.append(self._make_fragment(
                        "G",
                        [proxy, center],
                        "chain",
                        "terminal",
                        "G:P_half:double19d2-28",
                        abstracted_neighbors=[{
                            "path_index": 0,
                            "actual_gid": int(getattr(nb, "gid", -1)),
                            "actual_su": int(getattr(nb, "su", -1)),
                            "actual_label": str(getattr(nb, "label", "")),
                            "topology_su": 23,
                            "reason": "residual_P_double_d2_half_to_chain_port",
                        }],
                    ))
                elif center.su == 32 and is_special_d2(nb, 21):
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:21d2-32"))
                elif center.su == 16 and nb.su == 15:
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:15-16"))
                elif center.su == 18 and nb.su == 17:
                    out.append(self._make_fragment("G", [nb, center], "chain", "terminal", "G:17-18"))
        return out

    # ---------- assembly ----------

    def _assemble_chains(self, fragments: List[TopologyFragment]) -> List[ChainSpec]:
        chains: List[ChainSpec] = []
        used: set[int] = set()
        closed_primary = [f for f in fragments if f.closed and f.kind in {"A", "C", "A+", "C+"}]
        closed_f = [f for f in fragments if f.closed and f.kind == "F"]
        open_frags = [f for f in fragments if not f.closed]

        for frag in closed_primary:
            if self._fragment_conflicts(frag, used):
                continue
            if frag.kind in {"A", "A+"}:
                chains.append(self._chain_from_path("bridge", frag.su_path, frag, frag.kind))
            else:
                chains.append(self._chain_from_path("side", frag.su_path, frag, frag.kind))
            self._mark_used(frag, used)

        # Open fragments are topology candidates only.  They are not combined or
        # closed in this stage.
        for frag in open_frags:
            if self._fragment_conflicts(frag, used):
                frag = self._virtualize_shared_backbone_conflicts(frag, used)
                if frag is None:
                    continue
            pending = self._pending_closure_from_fragment(frag)
            if pending is not None:
                chains.append(pending)
                self._mark_used(frag, used)

        for frag in closed_f:
            if self._fragment_conflicts(frag, used):
                continue
            chains.append(self._chain_from_path("side", frag.su_path, frag, "F"))
            self._mark_used(frag, used)

        return chains

    def _pending_closure_from_fragment(self, frag: TopologyFragment) -> Optional[ChainSpec]:
        path = self._orient_for_closure(frag)
        intents = self._fragment_closure_intents(path)
        if not intents:
            return None
        open_ends = [
            side
            for side, kind in (("left", path.left_open), ("right", path.right_open))
            if str(kind) != "closed"
        ]
        open_topology = self._format_open_fragment(path)
        meta = {
            "allocator_mode": "raw_motif_v1",
            "pending_closure": True,
            "closure_stage_required": True,
            "topology_state": "semi_closed",
            "fragment_kind": str(path.kind),
            "origin_rule": str(path.origin_rule),
            "node_ids": list(path.node_ids),
            "labels": list(path.labels),
            "open_left": str(path.left_open),
            "open_right": str(path.right_open),
            "open_ends": open_ends,
            "display_topology": open_topology,
            "raw_topology": open_topology,
            "open_topology": open_topology,
            "closure_intents": dict(intents),
            "closure_intent_options": self._branch_intent_options(dict(intents)),
            "actual_used_su": [int(x) for x in list(path.su_path or [])],
            "actual_used_source_ids": [int(x) for x in list(path.node_ids or [])],
            "chain_port_signature": [int(x) for x in list(path.su_path or [])],
        }
        for key in (
            "abstracted_chain_port_neighbors",
            "virtualized_shared_source_ids",
            "virtualized_shared_source_su",
            "virtualized_shared_reason",
        ):
            if key in dict(path.metadata or {}):
                meta[key] = dict(path.metadata or {})[key]
        return ChainSpec(
            chain_type="pending_closure",
            composition=[int(x) for x in list(path.su_path or [])],
            origin_type=f"Pending:{path.kind}",
            source_ids=[int(x) for x in path.node_ids if int(x) >= 0],
            metadata=meta,
        )

    @staticmethod
    def _fragment_closure_intents(frag: TopologyFragment) -> Dict[str, List[str]]:
        left, right = str(frag.left_open), str(frag.right_open)
        kind = str(frag.kind)
        if left == "chain" and right == "chain":
            if kind in {"D+"}:
                return {"main": ["F1-", "F2-"]}
            return {"main": ["F1", "F2"]}
        if left == "closed" and right == "chain":
            if kind in {"B+"}:
                return {"main": ["F1-", "F2-"]}
            return {"main": ["F1", "F2"]}
        if left == "terminal" and right == "chain":
            if kind in {"E+"}:
                return {"main": ["F1-"]}
            return {"main": ["F1"]}
        if left == "chain" and right == "terminal":
            if kind in {"E+"}:
                return {"main": ["F1-"]}
            return {"main": ["F1"]}
        return {}

    @staticmethod
    def _format_open_fragment(frag: TopologyFragment) -> str:
        labels = list(frag.labels or [])
        values = labels if labels else [str(x) for x in list(frag.su_path or [])]
        text = "-".join(str(x) for x in values)
        if str(frag.left_open) == "chain":
            text = "-" + text
        elif str(frag.left_open) == "terminal":
            text = "T-" + text
        if str(frag.right_open) == "chain":
            text = text + "-"
        elif str(frag.right_open) == "terminal":
            text = text + "-T"
        return text

    @staticmethod
    def _orient_for_closure(frag: TopologyFragment) -> TopologyFragment:
        if frag.left_open == "closed":
            return frag
        if frag.right_open == "closed":
            return frag.reversed()
        if frag.left_open == "chain" and frag.right_open == "terminal":
            return frag
        if frag.left_open == "terminal" and frag.right_open == "chain":
            return frag
        return frag

    @staticmethod
    def _fragment_conflicts(frag: TopologyFragment, used: set[int]) -> bool:
        return any(int(gid) >= 0 and int(gid) in used for gid in frag.node_ids)

    @staticmethod
    def _virtualize_shared_backbone_conflicts(
        frag: TopologyFragment,
        used: set[int],
    ) -> Optional[TopologyFragment]:
        conflicts: List[Tuple[int, int, int]] = []
        for idx, gid in enumerate(list(frag.node_ids or [])):
            if int(gid) < 0 or int(gid) not in used:
                continue
            su = int(list(frag.su_path or [])[idx])
            conflicts.append((int(idx), int(gid), int(su)))
        if not conflicts:
            return frag
        if any(int(su) not in SHARED_BACKBONE_RESOURCE_SU for _idx, _gid, su in conflicts):
            return None
        node_ids = [int(x) for x in list(frag.node_ids or [])]
        for idx, _gid, _su in conflicts:
            node_ids[int(idx)] = -1
        meta = dict(frag.metadata or {})
        meta["virtualized_shared_source_ids"] = [int(gid) for _idx, gid, _su in conflicts]
        meta["virtualized_shared_source_su"] = [int(su) for _idx, _gid, su in conflicts]
        meta["virtualized_shared_reason"] = "shared_backbone_resource_conflict"
        return TopologyFragment(
            kind=str(frag.kind),
            su_path=[int(x) for x in list(frag.su_path or [])],
            node_ids=node_ids,
            labels=list(frag.labels or []),
            left_open=str(frag.left_open),
            right_open=str(frag.right_open),
            closed=bool(frag.closed),
            origin_rule=str(frag.origin_rule),
            metadata=meta,
        )

    @staticmethod
    def _mark_used(frag: TopologyFragment, used: set[int]) -> None:
        for gid in frag.node_ids:
            if int(gid) >= 0:
                used.add(int(gid))

    def _chain_from_path(
        self,
        chain_type: str,
        composition: Sequence[int],
        frag: TopologyFragment,
        origin: str,
    ) -> ChainSpec:
        meta = {
            "allocator_mode": "raw_motif_v1",
            "fragment_kind": frag.kind,
            "origin_rule": frag.origin_rule,
            "node_ids": list(frag.node_ids),
            "labels": list(frag.labels),
            "closure_type": str(origin),
        }
        return ChainSpec(
            chain_type=chain_type,
            composition=[int(x) for x in list(composition or [])],
            origin_type=str(origin),
            source_ids=[int(x) for x in frag.node_ids if int(x) >= 0],
            metadata=meta,
        )

    # ---------- accounting / diagnostics ----------

    def _populate_totals(self) -> None:
        r = self._result
        counts = Counter(int(n.su_type) for n in self._nodes)
        r.source_su_total_counts = dict(sorted((int(k), int(v)) for k, v in counts.items()))
        r.total_11 = int(counts.get(11, 0))
        r.total_22 = int(counts.get(22, 0))
        r.total_23 = int(counts.get(23, 0))
        r.total_24 = counts.get(24, 0)
        r.total_25 = counts.get(25, 0)

    def _fill_result(self, chains: List[ChainSpec], fragments: List[TopologyFragment]) -> None:
        r = self._result
        closed_chains: List[ChainSpec] = []
        consumed_su: Counter[int] = Counter()
        section_consumed: Dict[str, Dict[str, int]] = {}
        origin_consumed: Dict[str, Dict[str, int]] = {}
        trace_section_consumed: Dict[str, Dict[str, int]] = {}
        trace_origin_consumed: Dict[str, Dict[str, int]] = {}

        def _add_counts(dst: Dict[str, Dict[str, int]], key: str, counts: Dict[str, int]) -> None:
            bucket = dst.setdefault(str(key), {rk: 0 for rk in ("11", "22", "23", "24", "25")})
            for rk in ("11", "22", "23", "24", "25"):
                bucket[rk] = int(bucket.get(rk, 0)) + int((counts or {}).get(rk, 0))

        for chain in chains:
            ctype = str(chain.chain_type)
            counts = _chain_required_resource_counts(chain) if ctype != "pending_closure" else {rk: 0 for rk in ("11", "22", "23", "24", "25")}
            section = "pending_closure"
            if ctype in {"bridge", "branch_bridge"}:
                r.bridge_chains.append(chain)
                closed_chains.append(chain)
                section = "bridge"
            elif ctype in {"side", "branch_side"}:
                r.side_chains.append(chain)
                closed_chains.append(chain)
                section = "side"
            elif ctype == "pending_closure":
                r.pending_closure_topologies.append(chain)
                section = "pending_closure"
            elif ctype in {"side_ring", "vertical_ring", "fused_side_ring"}:
                r.branch_chains.append(chain)
                closed_chains.append(chain)
                section = "branch"
            else:
                r.branch_chains.append(chain)
                closed_chains.append(chain)
                section = "branch"
            if ctype != "pending_closure":
                consumed_su.update(int(x) for x in self._chain_consumed_su_sequence(chain))
            _add_counts(section_consumed, section, counts)
            _add_counts(origin_consumed, str(chain.origin_type), counts)
            _add_counts(trace_section_consumed, section, dict(getattr(chain, "trace_required_counts", {}) or counts))
            _add_counts(trace_origin_consumed, str(chain.origin_type), dict(getattr(chain, "trace_required_counts", {}) or counts))

        required = _sum_chain_required_resource_counts([
            ch for ch in list(chains or [])
            if str(getattr(ch, "chain_type", "")) != "pending_closure"
        ])
        r.consumed_11 = int(required["11"])
        r.consumed_23 = int(required["23"])
        r.consumed_22 = int(required["22"])
        r.consumed_24 = int(required["24"])
        r.consumed_25 = int(required["25"])
        r.remaining_11 = max(0, int(r.total_11) - int(r.consumed_11))
        r.remaining_23 = max(0, int(r.total_23) - int(r.consumed_23))
        r.remaining_22 = max(0, int(r.total_22) - int(r.consumed_22))
        r.remaining_24 = max(0, int(r.total_24) - int(r.consumed_24))
        r.remaining_25 = max(0, int(r.total_25) - int(r.consumed_25))
        r.source_su_consumed_counts = dict(sorted((int(k), int(v)) for k, v in consumed_su.items()))
        r.source_su_remaining_counts = {
            int(k): max(0, int(v) - int(consumed_su.get(int(k), 0)))
            for k, v in dict(r.source_su_total_counts or {}).items()
        }
        r.section_consumed_by_kind = section_consumed
        r.origin_consumed_by_kind = origin_consumed
        r.trace_consumed_by_section_kind = trace_section_consumed
        r.trace_consumed_by_origin_kind = trace_origin_consumed
        r.native_total_by_kind = {
            "11": int(r.total_11), "22": int(r.total_22), "23": int(r.total_23),
            "24": int(r.total_24), "25": int(r.total_25),
        }
        r.native_consumed_by_kind = {
            "11": int(r.consumed_11), "22": int(r.consumed_22), "23": int(r.consumed_23),
            "24": int(r.consumed_24), "25": int(r.consumed_25),
        }
        r.native_remaining_by_kind = {
            "11": int(r.remaining_11), "22": int(r.remaining_22), "23": int(r.remaining_23),
            "24": int(r.remaining_24), "25": int(r.remaining_25),
        }
        r.proxy_total_by_kind = {k: 0 for k in ("11", "22", "23", "24", "25")}
        r.proxy_consumed_by_kind = {k: 0 for k in ("11", "22", "23", "24", "25")}
        r.proxy_remaining_by_kind = {k: 0 for k in ("11", "22", "23", "24", "25")}
        r.type_counts = dict(Counter(str(f.kind) for f in fragments))
        blocking_pending = [
            ch for ch in list(r.pending_closure_topologies or [])
            if self._is_blocking_pending_closure(ch)
        ]
        deferred_pending_count = max(0, int(len(r.pending_closure_topologies)) - int(len(blocking_pending)))
        r.trace_exact_chain_count = int(len(closed_chains))
        r.trace_inexact_chain_count = int(len(blocking_pending))
        r.abstract_gap_by_kind = (
            {"pending_closure": int(len(blocking_pending))}
            if blocking_pending else {}
        )
        r.abstract_gap_audit_by_kind = dict(r.abstract_gap_by_kind)
        r.pending_closure_count = int(len(blocking_pending))
        r.pending_closure_blocking_count = int(len(blocking_pending))
        r.pending_closure_total_count = int(len(r.pending_closure_topologies))
        r.pending_closure_deferred_count = int(deferred_pending_count)
        r.fragment_rows = [self._chain_row(ch, idx) for idx, ch in enumerate(closed_chains)]
        r.pending_closure_rows = [
            self._chain_row(ch, idx)
            for idx, ch in enumerate(r.pending_closure_topologies)
        ]
        r.catalog_fragment_rows = [self._fragment_row(frag, idx) for idx, frag in enumerate(fragments)]
        r.catalog_summary = dict(Counter(str(frag.kind) for frag in fragments))

    @staticmethod
    def _chain_consumed_su_sequence(chain: ChainSpec) -> List[int]:
        meta = dict(getattr(chain, "metadata", {}) or {})
        actual = meta.get("actual_topology_su")
        if isinstance(actual, (list, tuple)) and actual:
            return [int(x) for x in list(actual or [])]
        actual_used = meta.get("actual_used_su")
        closure = meta.get("closure_resource_su")
        if isinstance(actual_used, (list, tuple)) and actual_used:
            vals = [int(x) for x in list(actual_used or [])]
            if isinstance(closure, (list, tuple)):
                vals.extend(int(x) for x in list(closure or []))
            return vals
        return [int(x) for x in list(chain.placement_su_sequence() or [])]

    @staticmethod
    def _is_blocking_pending_closure(chain: ChainSpec) -> bool:
        meta = dict(getattr(chain, "metadata", {}) or {})
        def _flag_true(key: str) -> bool:
            value = meta.get(str(key))
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)

        if _flag_true("deferred_nonblocking") or _flag_true("ring_builder_deferred"):
            return False
        reason = str(meta.get("unclosed_reason") or "")
        if reason == "ring_option_disabled_pending_ring_builder":
            return False
        return True

    @staticmethod
    def _chain_row(chain: ChainSpec, idx: int) -> Dict[str, Any]:
        return {
            "chain_index": int(idx),
            "chain_type": str(chain.chain_type),
            "origin_type": str(chain.origin_type),
            "composition": [int(x) for x in chain.composition],
            "source_ids": [int(x) for x in chain.source_ids],
            "metadata": dict(chain.metadata or {}),
        }

    @staticmethod
    def _fragment_row(frag: TopologyFragment, idx: int) -> Dict[str, Any]:
        return {
            "fragment_index": int(idx),
            "kind": str(frag.kind),
            "su_path": [int(x) for x in frag.su_path],
            "labels": list(frag.labels),
            "node_ids": [int(x) for x in frag.node_ids],
            "left_open": str(frag.left_open),
            "right_open": str(frag.right_open),
            "closed": bool(frag.closed),
            "origin_rule": str(frag.origin_rule),
        }

    def _print_summary(self) -> None:
        r = self._result
        print("\n[FlexAllocator] Raw motif allocation summary")
        print(f"  fragments: {dict(r.catalog_summary)}")
        print(f"  bridge_chains: {len(r.bridge_chains)}")
        for i, ch in enumerate(r.bridge_chains):
            print(f"    [{i}] {'-'.join(str(x) for x in ch.composition)} ({ch.origin_type})")
        print(f"  side_chains: {len(r.side_chains)}")
        for i, ch in enumerate(r.side_chains):
            print(f"    [{i}] {'-'.join(str(x) for x in ch.composition)} ({ch.origin_type})")
        print(f"  pending_closure_topologies: {len(r.pending_closure_topologies)}")
        for i, ch in enumerate(r.pending_closure_topologies):
            print(f"    [{i}] {'-'.join(str(x) for x in ch.composition)} ({ch.origin_type})")
        print("=" * 60)

    # ---------- diagnostics API used by Layer4 ----------

    def evaluate_su25_only(self, nodes: List[Any]) -> Dict[str, Any]:
        allocator = FlexAllocator(nodes=nodes)
        with redirect_stdout(io.StringIO()):
            result = allocator.allocate()
        return {
            "ok": int(result.remaining_25) == 0,
            "unallocated_25": int(result.remaining_25),
            "shortage_type": "none" if int(result.remaining_25) == 0 else "raw_motif_unplaced_25",
            "resource_shortage_hint": None,
            "avail_after_25": {
                "11": result.remaining_11,
                "22": result.remaining_22,
                "23": result.remaining_23,
            },
            "pre_branch_available": {},
            "branch_chains": list(result.branch_chains),
            "pending_closure_topologies": list(result.pending_closure_topologies),
            "alloc_result": result,
            "abstract_gap_by_kind": {},
            "abstract_gap_audit_by_kind": {},
        }

    def evaluate_su24_branches(self, nodes: List[Any], quiet: bool = False) -> Dict[str, Any]:
        allocator = FlexAllocator(nodes=nodes)
        if bool(quiet):
            with redirect_stdout(io.StringIO()):
                result = allocator.allocate()
        else:
            result = allocator.allocate()
        assembly_diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
        shortage = dict(assembly_diag.get("resource_shortage", {}) or {})
        flex_short = max(0, int(assembly_diag.get("flex_short", 0) or 0))
        flex_excess = max(0, int(assembly_diag.get("flex_excess", 0) or 0))
        flex_window_conflict = bool(assembly_diag.get("flex_window_conflict", not bool(assembly_diag.get("flex_window_ok", True))))
        topology_ok = bool(assembly_diag.get("topology_materialized_ok", assembly_diag.get("ok", False)))
        topology_ready_for_extra = bool(
            bool(assembly_diag.get("defer_residual_to_extra", False))
            and bool(assembly_diag.get("topology_plan_feasible", False))
            and int(getattr(result, "pending_closure_total_count", 0) or 0) == 0
            and int(result.unallocated_branch) == 0
            and not bool(assembly_diag.get("materialization_required", False))
            and not list(assembly_diag.get("chain_23_limit_violations", []) or [])
            and not any(int(v) > 0 for v in dict(assembly_diag.get("resource_shortage", {}) or {}).values())
            and int(flex_short) == 0
            and int(flex_excess) == 0
        )
        materialization_required = bool(assembly_diag.get("materialization_required", False))
        pending_blocking = int(getattr(result, "pending_closure_count", 0) or 0)
        pending_total = int(
            getattr(
                result,
                "pending_closure_total_count",
                len(getattr(result, "pending_closure_topologies", []) or []),
            )
            or 0
        )
        ok = bool(
            (bool(topology_ok) or bool(topology_ready_for_extra))
            and int(pending_total) == 0
            and int(result.unallocated_branch) == 0
            and not bool(materialization_required)
            and not bool(flex_window_conflict)
            and int(shortage.get("11", 0)) == 0
            and int(shortage.get("22", 0)) == 0
            and int(shortage.get("23", 0)) == 0
            and int(shortage.get("24", 0)) == 0
            and int(shortage.get("25", 0)) == 0
            and int(flex_short) == 0
            and int(flex_excess) == 0
        )
        shortage_type = "none"
        if not ok:
            if bool(flex_window_conflict):
                shortage_type = "flex_window_conflict"
            elif bool(materialization_required):
                shortage_type = "planned_conversion_required"
            elif int(result.unallocated_branch) > 0:
                shortage_type = "abstract_topology_gap"
            elif int(shortage.get("23", 0)) > 0:
                shortage_type = "23_shortage"
            elif int(shortage.get("22", 0)) > 0:
                shortage_type = "22_shortage"
            elif int(shortage.get("11", 0)) > 0:
                shortage_type = "11_shortage"
            elif int(shortage.get("24", 0)) > 0:
                shortage_type = "24_shortage"
            elif int(shortage.get("25", 0)) > 0:
                shortage_type = "25_shortage"
            elif int(pending_total) > 0:
                shortage_type = "pending_closure"
            elif not bool(topology_ok):
                shortage_type = "topology_assembly_failed"
            elif int(flex_short) > 0:
                shortage_type = "flex_shortage"
            elif int(flex_excess) > 0:
                shortage_type = "flex_excess"
        return {
            "ok": bool(ok),
            "unallocated_branch": int(result.unallocated_branch),
            "unallocated_bridge": int(result.unallocated_bridge),
            "shortage_type": str(shortage_type),
            "flex_window_conflict": bool(flex_window_conflict),
            "flex_window_ok": bool(not bool(flex_window_conflict)),
            "resource_shortage_hint": None,
            "req_22": int(shortage.get("22", 0)),
            "req_11": int(shortage.get("11", 0)),
            "req_23": int(shortage.get("23", 0)),
            "req_24": int(shortage.get("24", 0)),
            "req_25": int(shortage.get("25", 0)),
            "remaining": {
                "11": result.remaining_11,
                "22": result.remaining_22,
                "23": result.remaining_23,
                "24": result.remaining_24,
                "25": result.remaining_25,
            },
            "type_counts": dict(result.type_counts),
            "branch_chains": list(result.branch_chains),
            "pending_closure_topologies": list(result.pending_closure_topologies),
            "alloc_result": result,
            "closed_topology_specs": list(result.closed_topology_specs),
            "final_topology_specs": list(result.final_topology_specs),
            "topology_classification_rejects": list(result.topology_classification_rejects),
            "topology_assembly_diagnostics": dict(assembly_diag),
            "topology_selected_options": list(result.topology_selected_options),
            "topology_unclosed_candidates": list(result.topology_unclosed_candidates),
            "recommended_adjustments": list(assembly_diag.get("recommended_adjustments", []) or []),
            "topology_plan_feasible": bool(assembly_diag.get("topology_plan_feasible", False)),
            "topology_materialized_ok": bool(topology_ok),
            "topology_ready_for_extra": bool(topology_ready_for_extra),
            "materialization_required": bool(materialization_required),
            "planned_conversions": list(assembly_diag.get("planned_conversions", []) or []),
            "planned_conversion_count": int(assembly_diag.get("planned_conversion_count", 0) or 0),
            "residual_growth_actions": list(assembly_diag.get("residual_growth_actions", []) or []),
            "residual_chain_actions": list(assembly_diag.get("residual_chain_actions", []) or []),
            "residual_remaining": dict(assembly_diag.get("residual_remaining", {}) or {}),
            "chain_23_limit_violations": list(assembly_diag.get("chain_23_limit_violations", []) or []),
            "abstract_gap_by_kind": dict(
                assembly_diag.get(
                    "abstract_gap_by_kind",
                    {"pending_closure": int(result.unallocated_branch)} if int(result.unallocated_branch) > 0 else {},
                )
                or {}
            ),
            "abstract_gap_audit_by_kind": dict(
                assembly_diag.get(
                    "abstract_gap_audit_by_kind",
                    assembly_diag.get(
                        "abstract_gap_by_kind",
                        {"pending_closure": int(result.unallocated_branch)} if int(result.unallocated_branch) > 0 else {},
                    ),
                )
                or {}
            ),
            "trace_exact_chain_count": int(result.trace_exact_chain_count),
            "trace_inexact_chain_count": int(result.trace_inexact_chain_count),
            "topology_assembly_ok": bool(topology_ok),
            "pending_closure_count": int(pending_blocking),
            "pending_closure_total_count": int(pending_total),
            "pending_closure_blocking_count": int(result.pending_closure_blocking_count),
            "pending_closure_deferred_count": int(result.pending_closure_deferred_count),
        }

    def evaluate_extra_allocation(
        self,
        nodes: List[Any],
        short_bridge_threshold: int = 8,
        min_side_to_22: int = 5,
    ) -> Dict[str, Any]:
        allocator = FlexAllocator(nodes=nodes)
        result = allocator.allocate()
        assembly_diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
        return {
            "ok": bool(assembly_diag.get("ok", True)),
            "reason": "raw_motif_no_extra_fabrication",
            "bridge_chains": list(result.bridge_chains),
            "side_chains": list(result.side_chains),
            "branch_chains": list(result.branch_chains),
            "pending_closure_topologies": list(result.pending_closure_topologies),
            "closed_topology_specs": list(result.closed_topology_specs),
            "final_topology_specs": list(result.final_topology_specs),
            "topology_classification_rejects": list(result.topology_classification_rejects),
            "topology_assembly_diagnostics": dict(assembly_diag),
            "topology_selected_options": list(result.topology_selected_options),
            "topology_unclosed_candidates": list(result.topology_unclosed_candidates),
            "recommended_adjustments": list(assembly_diag.get("recommended_adjustments", []) or []),
            "remaining": {
                "11": result.remaining_11,
                "22": result.remaining_22,
                "23": result.remaining_23,
                "24": result.remaining_24,
                "25": result.remaining_25,
            },
            "alloc_result": result,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m RL_MTCS.RL_allocator <nodes.csv>")
    FlexAllocator(sys.argv[1]).allocate()
