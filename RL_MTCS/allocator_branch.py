from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .allocator_patterns import (
    TERMINAL_BASE_SU,
    has_forbidden_fixed_connector_special_combo,
    is_abstract_chain_port_state,
    is_double_special_d3,
    is_single_special_d2,
    is_special_d3,
    state_for_node,
)
from .allocator_models import (
    BRANCH_CLOSURE_INTENTS,
    ChainSpec,
    SUNode,
    _BranchTopologyCandidate,
    _FixedConnectorExpansion,
)


class BranchTopologyMixin:
    # ---------- branch topology candidates ----------

    def _extract_branch_topology_candidates(
        self,
        skip_node_ids: Optional[set[int]] = None,
    ) -> List[_BranchTopologyCandidate]:
        skip = {int(x) for x in set(skip_node_ids or set()) if int(x) >= 0}
        out: List[_BranchTopologyCandidate] = []
        for node in self._nodes:
            gid = int(node.global_id)
            if gid in skip:
                continue
            center = state_for_node(node)
            if int(center.su) == 24:
                out.extend(self._classify_native24_branch_candidates(node, center, skip))
            elif int(center.su) == 25:
                out.extend(self._classify_su25_branch_candidates(node, center, skip))
            elif int(center.su) == 14:
                out.extend(self._classify_su14_branch_candidates(node, center, skip))
            elif is_special_d3(center):
                out.extend(self._classify_special_d3_branch_candidates(node, center, skip))
        out.extend(self._classify_su14_pair_candidates(skip))
        out.extend(self._classify_like24_n_bridge_candidates(skip))
        return out

    def _assemble_branch_topology_candidates(
        self,
        candidates: Sequence[_BranchTopologyCandidate],
    ) -> Tuple[List[ChainSpec], set[int]]:
        chains: List[ChainSpec] = []
        suppress_keys = {
            str(key)
            for cand in list(candidates or [])
            if int(getattr(cand, "priority_rank", 0) or 0) > 0
            for key in list(getattr(cand, "suppresses_local_candidate_keys", []) or [])
        }
        active_candidates = [
            cand
            for cand in list(candidates or [])
            if int(getattr(cand, "priority_rank", 0) or 0) > 0
            or str(cand.group_id) not in suppress_keys
        ]
        groups: Dict[str, List[_BranchTopologyCandidate]] = {}
        for cand in active_candidates:
            groups.setdefault(str(cand.group_id), []).append(cand)
        seen_pending = set()
        for group_id in sorted(groups):
            alts = groups[group_id]
            sig = (
                tuple(sorted(set(int(gid) for cand in alts for gid in cand.source_ids if int(gid) >= 0))),
                tuple(sorted(str(cand.abcd_type) for cand in alts)),
                tuple(sorted(str(cand.fixed_role) for cand in alts)),
                tuple(sorted(str(cand.open_topology) for cand in alts)),
                tuple(sorted(str(cand.pattern_rule) for cand in alts)),
                tuple(sorted(str(cand.origin_rule) for cand in alts)),
            )
            if sig in seen_pending:
                continue
            seen_pending.add(sig)
            chains.append(self._make_pending_branch_topology(group_id, alts))
        priority_consumed = {
            int(gid)
            for cand in active_candidates
            if int(getattr(cand, "priority_rank", 0) or 0) > 0
            for gid in list(getattr(cand, "source_ids", []) or [])
            if int(gid) >= 0
        }
        return chains, priority_consumed

    def _classify_native24_branch_candidates(
        self,
        node: SUNode,
        center: Any,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        nbs = [st for nb, st in self._neighbor_states(node) if int(nb.global_id) not in skip]
        if not nbs:
            return []
        aromatic = next((st for st in nbs if self._is_aromatic_11(st)), None)
        terminal = next((st for st in nbs if self._native24_terminal(st)), None)
        abstracted = [
            st
            for st in nbs
            if int(st.gid) not in {
                int(getattr(aromatic, "gid", -1)) if aromatic is not None else -1,
                int(getattr(terminal, "gid", -1)) if terminal is not None else -1,
            }
        ]
        evidence_ids = sorted(set(int(st.gid) for st in abstracted if int(st.gid) >= 0))
        abstracted_neighbors = [
            self._branch_abstracted_neighbor_row(st, "native24_classification_evidence")
            for st in abstracted
        ]
        if aromatic is not None and terminal is not None:
            return [
                self._make_branch_candidate(
                    center=center,
                    abcd_type="24_B",
                    source_kind="native24",
                    fixed_role="native_terminal_branch",
                    main_states=[center],
                    branch_states=[terminal],
                    source_ids=[center.gid, terminal.gid],
                    open_topology=f"11-{center.label}(-{terminal.label})-",
                    actual_used_su=[aromatic.su, center.su, terminal.su],
                    actual_used_source_ids=[aromatic.gid, center.gid, terminal.gid],
                    origin_rule="24_B:native24-terminal-presence",
                    analog_template="11-24B-",
                    closure_intents={"main": ["F1", "F2"]},
                    pattern_rule="native24:has11+has_terminal",
                    main_open=True,
                    evidence_ids=evidence_ids,
                    abstracted_neighbors=abstracted_neighbors,
                    **self._ab_no_connector_ring_kwargs(),
                )
            ]
        if aromatic is not None:
            return [
                self._make_branch_candidate(
                    center=center,
                    abcd_type="24_A",
                    source_kind="native24",
                    fixed_role="native_H_branch",
                    main_states=[center],
                    branch_states=[],
                    source_ids=[center.gid],
                    open_topology=f"11-{center.label}(-H)-",
                    actual_used_su=[aromatic.su, center.su],
                    actual_used_source_ids=[aromatic.gid, center.gid],
                    origin_rule="24_A:native24-has11-no-terminal",
                    analog_template="11-24A-",
                    closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                    pattern_rule="native24:has11+no_terminal",
                    main_open=True,
                    branch_open=True,
                    evidence_ids=evidence_ids,
                    abstracted_neighbors=abstracted_neighbors,
                    **self._ab_no_connector_ring_kwargs(),
                )
            ]
        if terminal is not None:
            return [
                self._make_branch_candidate(
                    center=center,
                    abcd_type="24_D",
                    source_kind="native24",
                    fixed_role="native_terminal_branch",
                    main_states=[center],
                    branch_states=[terminal],
                    source_ids=[center.gid, terminal.gid],
                    open_topology=f"-{center.label}(-{terminal.label})-",
                    origin_rule="24_D:native24-terminal-presence",
                    analog_template="-24D-",
                    closure_intents={"main": ["F1", "F2"]},
                    pattern_rule="native24:no11+has_terminal",
                    main_open=True,
                    evidence_ids=evidence_ids,
                    abstracted_neighbors=abstracted_neighbors,
                    **self._cd_branch_assembly_kwargs(),
                )
            ]
        return [
            self._make_branch_candidate(
                center=center,
                abcd_type="24_C",
                source_kind="native24",
                fixed_role="native_H_branch",
                main_states=[center],
                branch_states=[],
                source_ids=[center.gid],
                open_topology=f"-{center.label}(-H)-",
                origin_rule="24_C:native24-no11-no-terminal",
                analog_template="-24C-",
                closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                pattern_rule="native24:no11+no_terminal",
                main_open=True,
                branch_open=True,
                evidence_ids=evidence_ids,
                abstracted_neighbors=abstracted_neighbors,
                **self._cd_branch_assembly_kwargs(),
            )
        ]

    def _classify_su14_branch_candidates(
        self,
        node: SUNode,
        center: Any,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        nbs = [(nb, st) for nb, st in self._neighbor_states(node) if int(nb.global_id) not in skip]
        states = [st for _nb, st in nbs]
        cand: List[_BranchTopologyCandidate] = []
        group = self._branch_group_id(center)
        match = self._match_branch_roles(
            states,
            {"fixed": lambda st: int(st.su) == 15, "chain1": self._branch_chain_port, "chain2": self._branch_chain_port},
        )
        if match:
            st = match["fixed"]
            exp = self._direct_path_expansion("N", st, [st], "-14-15-")
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_C",
                source_kind="su14",
                fixed_role="branch",
                main_states=[center],
                branch_states=[st],
                source_ids=[center.gid, st.gid],
                open_topology="-14(-15-H1)-",
                origin_rule="24_C:14-15-branch",
                analog_template="-24C(-H1)-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"branch": ["H1"], "main": ["F1", "F2"]},
                pattern_rule="su14:[15,chain,chain]",
                main_open=True,
                branch_open=True,
                **self._cd_branch_assembly_kwargs(),
            ))
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_C",
                source_kind="su14",
                fixed_role="main",
                main_states=[center, st],
                branch_states=[],
                source_ids=[center.gid, st.gid],
                open_topology="-14(-H)-15-",
                origin_rule="24_C:14-15-main",
                analog_template="-24C-23-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                pattern_rule="su14:[15,chain,chain]",
                main_open=True,
                branch_open=True,
                **self._cd_14_main_assembly_kwargs(),
            ))
        match = self._match_branch_roles(
            states,
            {"fixed": lambda st: int(st.su) == 14, "chain1": self._branch_chain_port, "chain2": self._branch_chain_port},
        )
        if match:
            st = match["fixed"]
            exp = self._direct_path_expansion("N", st, [st], "-14-14-")
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_C",
                source_kind="su14",
                fixed_role="main",
                main_states=[center, st],
                branch_states=[],
                source_ids=[center.gid, st.gid],
                open_topology="-14(-H)-14(?)-",
                origin_rule=f"24_C:14-14-main:center{int(center.gid)}:fixed{int(st.gid)}",
                analog_template="-24C-24C-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                pattern_rule="su14:[14,chain,chain]",
                main_open=True,
                branch_open=True,
                **self._cd_14_main_assembly_kwargs(bridgehead_replace=True),
            ))
        match = self._match_branch_roles(
            states,
            {"fixed": lambda st: int(st.su) == 16, "chain1": self._branch_chain_port, "chain2": self._branch_chain_port},
        )
        if match:
            st = match["fixed"]
            exp = self._direct_path_expansion("P", st, [st], "-14-16")
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_D",
                source_kind="su14",
                fixed_role="branch",
                main_states=[center],
                branch_states=[st],
                source_ids=[center.gid, st.gid],
                open_topology="-14(-16)-",
                origin_rule="24_D:14-16-branch",
                analog_template="-24D-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"main": ["F1", "F2"]},
                pattern_rule="su14:[16,chain,chain]",
                main_open=True,
                **self._cd_branch_assembly_kwargs(),
            ))
        match = self._match_branch_roles(
            states,
            {
                "fixed": lambda st: int(st.su) == 15,
                "terminal": self._base_terminal_1_or_22,
                "chain": self._branch_chain_port,
            },
        )
        if match:
            st = match["fixed"]
            term = match["terminal"]
            exp = self._direct_path_expansion("N", st, [st], "-14-15-")
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_D",
                source_kind="su14",
                fixed_role="main",
                main_states=[center, st],
                branch_states=[term],
                source_ids=[center.gid, st.gid, term.gid],
                open_topology=f"-14(-{term.label})-15-",
                origin_rule="24_D:14-15-terminal-main",
                analog_template="-24D-23-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"main": ["F1", "F2"]},
                pattern_rule="su14:[15,terminal,chain]",
                main_open=True,
                **self._cd_14_main_assembly_kwargs(),
            ))
        match = self._match_branch_roles(
            states,
            {
                "fixed": lambda st: int(st.su) == 14,
                "terminal": self._base_terminal_1_or_22,
                "chain": self._branch_chain_port,
            },
        )
        if match:
            st = match["fixed"]
            term = match["terminal"]
            exp = self._direct_path_expansion("N", st, [st], "-14-14-")
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_D",
                source_kind="su14",
                fixed_role="main",
                main_states=[center, st],
                branch_states=[term],
                source_ids=[center.gid, st.gid, term.gid],
                open_topology=f"-14(-{term.label})-14(?)-",
                origin_rule=(
                    f"24_D:14-14-terminal-main:center{int(center.gid)}:"
                    f"fixed{int(st.gid)}:terminal{int(term.gid)}"
                ),
                analog_template="-24D-24D-",
                group_id=group,
                connector_expansions=[exp],
                closure_intents={"main": ["F1", "F2"]},
                pattern_rule="su14:[14,terminal,chain]",
                main_open=True,
                **self._cd_14_main_assembly_kwargs(),
            ))
        return self._dedupe_branch_candidates(cand)

    def _classify_su14_pair_candidates(
        self,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        cand: List[_BranchTopologyCandidate] = []
        lookup = {
            int(getattr(node, "global_id", -1)): node
            for node in list(getattr(self, "_nodes", []) or [])
            if int(getattr(node, "global_id", -1)) >= 0
        }
        for node in list(getattr(self, "_nodes", []) or []):
            gid = int(node.global_id)
            if gid in skip or int(node.su_type) != 14:
                continue
            center = state_for_node(node)
            for nb, st in self._neighbor_states(node):
                other_gid = int(nb.global_id)
                if other_gid in skip or other_gid <= gid or int(st.su) != 14:
                    continue
                other_node = lookup.get(other_gid)
                if other_node is None:
                    continue
                other = state_for_node(other_node)
                left = self._su14_pair_side_role(node, int(other.gid))
                right = self._su14_pair_side_role(other_node, int(center.gid))
                if left is None or right is None:
                    continue
                cand.append(self._make_su14_pair_candidate(center, other, left, right))
        return self._dedupe_branch_candidates(cand)

    def _su14_pair_side_role(self, node: SUNode, partner_gid: int) -> Optional[Dict[str, Any]]:
        center = state_for_node(node)
        states = [
            st for nb, st in self._neighbor_states(node)
            if int(nb.global_id) != int(partner_gid)
        ]
        terminals = [st for st in states if self._base_terminal_1_or_22(st)]
        chain_ports = [st for st in states if self._branch_chain_port(st)]
        if terminals:
            term = min(terminals, key=lambda st: int(st.gid))
            evidence = [
                st for st in states
                if int(st.gid) != int(term.gid) and self._branch_chain_port(st)
            ]
            return {
                "kind": "terminal",
                "branch_su": [int(term.su)],
                "label": str(term.label),
                "source_ids": [int(term.gid)] if int(term.gid) >= 0 else [],
                "actual_used_su": [int(center.su), int(term.su)],
                "actual_used_source_ids": [int(center.gid), int(term.gid)] if int(term.gid) >= 0 else [int(center.gid)],
                "evidence": list(evidence),
            }
        if len(chain_ports) >= 2:
            return {
                "kind": "H",
                "branch_su": [],
                "label": "H",
                "source_ids": [],
                "actual_used_su": [int(center.su)],
                "actual_used_source_ids": [int(center.gid)],
                "evidence": list(chain_ports),
            }
        return None

    def _make_su14_pair_candidate(
        self,
        left: Any,
        right: Any,
        left_role: Dict[str, Any],
        right_role: Dict[str, Any],
    ) -> _BranchTopologyCandidate:
        left_branch = [int(x) for x in list(left_role.get("branch_su", []) or [])]
        right_branch = [int(x) for x in list(right_role.get("branch_su", []) or [])]
        branch_slots = {
            "left": list(left_branch),
            "right": list(right_branch),
        }
        branch_slot_labels = {
            "left": str(left_role.get("label") or ""),
            "right": str(right_role.get("label") or ""),
        }
        branch_states = [self._proxy_state(su) for su in list(left_branch) + list(right_branch)]
        branch_intents: List[str] = []
        if str(left_role.get("kind")) == "H":
            branch_intents.append("H:left")
        if str(right_role.get("kind")) == "H":
            branch_intents.append("H:right")
        source_ids = sorted(set(
            [int(left.gid), int(right.gid)]
            + [int(x) for x in list(left_role.get("source_ids", []) or [])]
            + [int(x) for x in list(right_role.get("source_ids", []) or [])]
        ))
        evidence_states = list(left_role.get("evidence", []) or []) + list(right_role.get("evidence", []) or [])
        evidence_ids = sorted(set(int(st.gid) for st in evidence_states if int(st.gid) >= 0))
        abstracted_neighbors = [
            {
                "actual_gid": int(st.gid),
                "actual_su": int(st.su),
                "actual_label": str(st.label),
                "topology_su": 23,
                "reason": "su14_pair_classification_evidence",
            }
            for st in evidence_states
        ]
        left_display = (
            f"-{left_role.get('label')}"
            if str(left_role.get("kind")) == "terminal"
            else "-H"
        )
        right_display = (
            f"-{right_role.get('label')}"
            if str(right_role.get("kind")) == "terminal"
            else "-H"
        )
        actual_used_su = [
            int(left.su),
            int(right.su),
            *[int(x) for x in list(left_branch)],
            *[int(x) for x in list(right_branch)],
        ]
        actual_used_source_ids = sorted(set(
            [int(left.gid), int(right.gid)]
            + [int(x) for x in list(left_role.get("actual_used_source_ids", []) or []) if int(x) >= 0]
            + [int(x) for x in list(right_role.get("actual_used_source_ids", []) or []) if int(x) >= 0]
        ))
        group = f"branch14pair:{min(int(left.gid), int(right.gid))}:{max(int(left.gid), int(right.gid))}"
        return self._make_branch_candidate(
            center=left,
            abcd_type="24_D",
            source_kind="su14_pair",
            fixed_role="pair_main",
            main_states=[left, right],
            branch_states=branch_states,
            source_ids=source_ids,
            open_topology=f"-14({left_display})-14({right_display})-",
            actual_used_su=actual_used_su,
            actual_used_source_ids=actual_used_source_ids,
            actual_main_path_su=[14, 14],
            actual_main_path_labels=["14", "14"],
            chain_port_signature=[14, 14],
            origin_rule=(
                f"24_D:14-14-pair:left{int(left.gid)}:"
                f"{left_role.get('kind')}:right{int(right.gid)}:{right_role.get('kind')}"
            ),
            analog_template="-24D-24D-",
            group_id=group,
            connector_expansions=[
                self._direct_path_expansion("N", right, [right], "-14-14-")
            ],
            closure_intents={"branch_slots": branch_intents, "main": ["F1", "F2"]},
            pattern_rule="su14_pair:[14,terminal_or_H,terminal_or_H]",
            main_open=True,
            branch_open=bool(branch_intents),
            assembly_family="24_CD_N_14_PAIR",
            assembly_shape="-14(-slot)-14(-slot)-",
            usage_modes=["flex_branch", "side_branch"],
            ring_eligible=False,
            disallow_ring_reason="su14_pair_closed_as_flex_or_side",
            exclusive_group_key=group,
            suppresses_local_candidate_keys=[
                self._branch_group_id(left),
                self._branch_group_id(right),
            ],
            priority_rank=110,
            branch_slots_su=branch_slots,
            branch_slot_labels=branch_slot_labels,
            evidence_ids=evidence_ids,
            abstracted_neighbors=abstracted_neighbors,
        )

    def _classify_su25_branch_candidates(
        self,
        node: SUNode,
        center: Any,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        nbs = [(nb, st) for nb, st in self._neighbor_states(node) if int(nb.global_id) not in skip]
        states = [st for _nb, st in nbs]
        cand: List[_BranchTopologyCandidate] = []
        group = self._branch_group_id(center)

        aromatic_candidates = [st for st in states if self._is_aromatic_11(st)]
        aromatic = min(aromatic_candidates, key=lambda st: int(st.gid)) if aromatic_candidates else None
        port_states = [
            st for st in states
            if aromatic is None or int(st.gid) != int(aromatic.gid)
        ]
        if not port_states:
            return []
        if not all(self._su25_abstract_port(st) for st in port_states):
            return []

        fixed22 = self._proxy_state(22)
        branch_slots = {"fixed22": [22], "H": []}
        branch_slot_labels = {"fixed22": "22", "H": "H"}
        abstracted_neighbors = [
            {
                "actual_gid": int(st.gid),
                "actual_su": int(st.su),
                "actual_label": str(st.label),
                "topology_su": 23 if int(st.su) != 22 else 22,
                "reason": "su25_abstract_port",
            }
            for st in port_states
        ]

        if aromatic is not None:
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_B",
                source_kind="su25",
                fixed_role="fixed22_branch+H",
                main_states=[center],
                branch_states=[fixed22],
                source_ids=[center.gid, aromatic.gid],
                open_topology=f"11-{center.label}(-22)(-H)-",
                actual_used_su=[aromatic.su, center.su, fixed22.su],
                actual_used_source_ids=[aromatic.gid, center.gid],
                chain_port_signature=[center.su],
                origin_rule="25_B:11-25-abstract22-H",
                analog_template="11-25(-22)(-H)-",
                group_id=group,
                closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                pattern_rule="su25:[11,abstract22,abstract_port,abstract_port]",
                main_open=True,
                branch_open=True,
                assembly_family="24_AB_N_BRANCH",
                assembly_shape="11-25(-22)(-H)-",
                usage_modes=["flex_branch", "side_branch"],
                ring_eligible=False,
                disallow_ring_reason="su25_fixed22_branch_flex_side_only",
                branch_slots_su=branch_slots,
                branch_slot_labels=branch_slot_labels,
                fixed_22_is_abstract=True,
                evidence_ids=[st.gid for st in port_states],
                abstracted_neighbors=abstracted_neighbors,
            ))
        else:
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_D",
                source_kind="su25",
                fixed_role="fixed22_branch+H",
                main_states=[center],
                branch_states=[fixed22],
                source_ids=[center.gid],
                open_topology=f"-{center.label}(-22)(-H)-",
                actual_used_su=[center.su, fixed22.su],
                actual_used_source_ids=[center.gid],
                chain_port_signature=[center.su],
                origin_rule="25_D:25-abstract22-H",
                analog_template="-25(-22)(-H)-",
                group_id=group,
                closure_intents={"branch": ["H"], "main": ["F1", "F2"]},
                pattern_rule="su25:[abstract22,abstract_port,abstract_port,abstract_port]",
                main_open=True,
                branch_open=True,
                assembly_family="24_CD_N_BRANCH",
                assembly_shape="-25(-22)(-H)-",
                usage_modes=["flex_branch", "side_branch"],
                ring_eligible=False,
                disallow_ring_reason="su25_fixed22_branch_flex_side_only",
                branch_slots_su=branch_slots,
                branch_slot_labels=branch_slot_labels,
                fixed_22_is_abstract=True,
                evidence_ids=[st.gid for st in port_states],
                abstracted_neighbors=abstracted_neighbors,
            ))

        return self._dedupe_branch_candidates(cand)

    def _classify_special_d3_branch_candidates(
        self,
        node: SUNode,
        center: Any,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        su = int(center.su)
        nbs = [(nb, st) for nb, st in self._neighbor_states(node) if int(nb.global_id) not in skip]
        states = [st for _nb, st in nbs]
        node_by_gid = {int(st.gid): nb for nb, st in nbs}
        source_kind = self._special_d3_source_kind(center)
        group = self._branch_group_id(center)
        cand: List[_BranchTopologyCandidate] = []

        def connector_expansions(connector_state: Any) -> List[_FixedConnectorExpansion]:
            connector_node = node_by_gid.get(int(connector_state.gid))
            if connector_node is None:
                return []
            return self._expand_d3_fixed_connector(center, connector_node, connector_state, skip)

        if su == 21:
            match = self._match_branch_roles(
                states,
                {
                    "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                    "chain": self._branch_24ab_chain_port,
                    "aromatic": self._is_aromatic_11,
                },
            )
            if match:
                terminal = match["terminal"]
                aromatic = match["aromatic"]
                cand.append(self._make_branch_candidate(
                    center=center,
                    abcd_type="24_B",
                    source_kind="21d3",
                    fixed_role="branch",
                    main_states=[center],
                    branch_states=[terminal],
                    source_ids=[center.gid, terminal.gid],
                    open_topology=f"11-{center.label}(-{terminal.label})-",
                    actual_used_su=[aromatic.su, center.su, terminal.su],
                    actual_used_source_ids=[aromatic.gid, center.gid, terminal.gid],
                    origin_rule="24_B:21d3-32",
                    analog_template="11-24B-",
                    group_id=group,
                    closure_intents={"main": ["F1", "F2"]},
                    pattern_rule="21d3:[32,chain,11]",
                    main_open=True,
                    **self._ab_no_connector_ring_kwargs(),
                ))
            match = self._match_branch_roles(
                states,
                {
                    "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                    "chain1": self._branch_chain_port,
                    "chain2": self._branch_chain_port,
                },
            )
            if match:
                terminal = match["terminal"]
                cand.append(self._make_branch_candidate(
                    center=center,
                    abcd_type="24_D",
                    source_kind="21d3",
                    fixed_role="branch",
                    main_states=[center],
                    branch_states=[terminal],
                    source_ids=[center.gid, terminal.gid],
                    open_topology=f"-{center.label}(-{terminal.label})-",
                    origin_rule="24_D:21d3-32",
                    analog_template="-24D-",
                    group_id=group,
                    closure_intents={"main": ["F1", "F2"]},
                    pattern_rule="21d3:[32,chain,chain]",
                    main_open=True,
                    **self._cd_branch_assembly_kwargs(),
                ))
            return self._dedupe_branch_candidates(cand)

        if self._is_double_d3_center(center, su):
            match = self._match_branch_roles(
                states,
                {
                    "connector1": lambda st: self._special_d3_connector(su, st),
                    "connector2": lambda st: self._special_d3_connector(su, st),
                    "aromatic": self._is_aromatic_11,
                },
            )
            if match:
                for main_conn, branch_conn in ((match["connector1"], match["connector2"]), (match["connector2"], match["connector1"])):
                    for main_exp in connector_expansions(main_conn):
                        for branch_exp in connector_expansions(branch_conn):
                            if not self._allowed_24ab_double_pair(main_exp, branch_exp):
                                continue
                            cand.append(self._candidate_from_main_connector(
                                center, source_kind, "24_A", main_exp, [branch_exp], "11",
                                group,
                                (
                                    f"24_A:double-d3-two-connectors+{main_exp.kind}{branch_exp.kind}:"
                                    f"main{int(main_exp.connector.gid)}:branch{int(branch_exp.connector.gid)}"
                                ),
                                "11-24A-23-23-",
                                left_endpoint_state=match["aromatic"],
                                pattern_rule=f"{source_kind}:[connector,connector,11]+{main_exp.kind}{branch_exp.kind}",
                                branch_connector_kind=str(branch_exp.kind),
                                branch_connector_ids=self._connector_expansion_ids(branch_exp),
                                **self._ab_main_assembly_kwargs(main_exp),
                            ))
            match = self._match_branch_roles(
                states,
                {
                    "connector": lambda st: self._special_d3_connector(su, st),
                    "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                    "aromatic": self._is_aromatic_11,
                },
            )
            if match:
                term = match["terminal"]
                for exp in connector_expansions(match["connector"]):
                    cand.append(self._candidate_from_main_connector(
                        center, source_kind, "24_B", exp, [], "11",
                        group, f"24_B:double-d3-terminal-main+{exp.kind}", "11-24B-23-",
                        left_endpoint_state=match["aromatic"],
                        branch_states=[term],
                        source_ids_extra=[term.gid],
                        pattern_rule=f"{source_kind}:[connector,direct_terminal,11]+{exp.kind}-main",
                        **self._ab_main_assembly_kwargs(exp),
                    ))
            match = self._match_branch_roles(
                states,
                {
                    "connector1": lambda st: self._special_d3_connector(su, st),
                    "connector2": lambda st: self._special_d3_connector(su, st),
                    "chain": self._double_d3_abstract_port,
                },
            )
            if match:
                abstract_state = match["chain"]
                for main_conn, branch_conn in ((match["connector1"], match["connector2"]), (match["connector2"], match["connector1"])):
                    for main_exp in connector_expansions(main_conn):
                        for branch_exp in connector_expansions(branch_conn):
                            if not self._allowed_24cd_double_pair(main_exp, branch_exp):
                                continue
                            cand.append(self._candidate_from_double_d3_fixed_connectors(
                                center=center,
                                source_kind=source_kind,
                                abcd_type="24_C",
                                main_exp=main_exp,
                                branch_exp=branch_exp,
                                abstract_state=abstract_state,
                                group_id=group,
                                origin_rule=(
                                    f"24_C:double-d3-two-fixed-connectors+{main_exp.kind}{branch_exp.kind}:"
                                    f"main{int(main_exp.connector.gid)}:branch{int(branch_exp.connector.gid)}"
                                ),
                                analog_template="-24C-23-23-",
                                pattern_rule=f"{source_kind}:[connector,connector,abstract]+{main_exp.kind}{branch_exp.kind}",
                                **self._cd_main_assembly_kwargs(main_exp, branch_exp),
                            ))
            match = self._match_branch_roles(
                states,
                {
                    "connector": lambda st: self._special_d3_connector(su, st),
                    "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                    "chain": self._double_d3_abstract_port,
                },
            )
            if match:
                term = match["terminal"]
                for exp in connector_expansions(match["connector"]):
                    cand.append(self._candidate_from_main_connector(
                        center, source_kind, "24_D", exp, [], "",
                        group, f"24_D:double-d3-terminal-main+{exp.kind}", "-24D-23-",
                        branch_states=[term],
                        source_ids_extra=[term.gid],
                        pattern_rule=f"{source_kind}:[connector,direct_terminal,chain]+{exp.kind}-main",
                        **self._cd_main_assembly_kwargs(exp),
                    ))
            return self._dedupe_branch_candidates(cand)

        match = self._match_branch_roles(
            states,
            {
                "connector": lambda st: self._special_d3_connector(su, st),
                "chain": self._branch_24ab_chain_port,
                "aromatic": self._is_aromatic_11,
            },
        )
        if match:
            for exp in connector_expansions(match["connector"]):
                if str(exp.kind) == "N":
                    cand.append(self._candidate_from_branch_connector(
                        center, source_kind, "24_A", exp, "11", group,
                        "24_A:single-d3-fixed-as-branch+N", "11-24A-",
                        left_endpoint_state=match["aromatic"],
                        pattern_rule=f"{source_kind}:[connector,chain,11]+N-branch",
                        **self._ab_branch_assembly_kwargs(),
                    ))
                cand.append(self._candidate_from_main_connector(
                    center, source_kind, "24_A", exp, [], "11", group,
                    f"24_A:single-d3-fixed-as-main+{exp.kind}", "11-24A-23-",
                    left_endpoint_state=match["aromatic"],
                    pattern_rule=f"{source_kind}:[connector,chain,11]+{exp.kind}-main",
                    **self._ab_main_assembly_kwargs(exp),
                ))

        match = self._match_branch_roles(
            states,
            {
                "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                "chain": self._branch_24ab_chain_port,
                "aromatic": self._is_aromatic_11,
            },
        )
        if match:
            term = match["terminal"]
            aromatic = match["aromatic"]
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_B",
                source_kind=source_kind,
                fixed_role="branch",
                main_states=[center],
                branch_states=[term],
                source_ids=[center.gid, term.gid],
                open_topology=f"11-{center.label}(-{term.label})-",
                actual_used_su=[aromatic.su, center.su, term.su],
                actual_used_source_ids=[aromatic.gid, center.gid, term.gid],
                origin_rule=f"24_B:{source_kind}-direct-terminal",
                analog_template="11-24B-",
                group_id=group,
                closure_intents={"main": ["F1", "F2"]},
                pattern_rule=f"{source_kind}:[direct_terminal,chain,11]",
                main_open=True,
                **self._ab_no_connector_ring_kwargs(),
            ))

        match = self._match_branch_roles(
            states,
            {
                "connector": lambda st: self._special_d3_connector(su, st),
                "terminal": self._base_terminal_1_or_22,
                "aromatic": self._is_aromatic_11,
            },
        )
        if match:
            term = match["terminal"]
            for exp in connector_expansions(match["connector"]):
                cand.append(self._candidate_from_main_connector(
                    center, source_kind, "24_B", exp, [], "11", group,
                    f"24_B:single-d3-terminal-main+{exp.kind}", "11-24B-23-",
                    left_endpoint_state=match["aromatic"],
                    branch_states=[term],
                    source_ids_extra=[term.gid],
                    pattern_rule=f"{source_kind}:[connector,terminal,11]+{exp.kind}-main",
                    **self._ab_main_assembly_kwargs(exp),
                ))

        match = self._match_branch_roles(
            states,
            {
                "connector": lambda st: self._special_d3_connector(su, st),
                "chain1": self._branch_chain_port,
                "chain2": self._branch_chain_port,
            },
        )
        if match:
            for exp in connector_expansions(match["connector"]):
                if str(exp.kind) == "N":
                    cand.append(self._candidate_from_branch_connector(
                        center, source_kind, "24_C", exp, "", group,
                        "24_C:single-d3-fixed-as-branch+N", "-24C-",
                        pattern_rule=f"{source_kind}:[connector,chain,chain]+N-branch",
                        **self._cd_branch_assembly_kwargs(exp),
                    ))
                cand.append(self._candidate_from_main_connector(
                    center, source_kind, "24_C", exp, [], "", group,
                    f"24_C:single-d3-fixed-as-main+{exp.kind}", "-24C-23-",
                    pattern_rule=f"{source_kind}:[connector,chain,chain]+{exp.kind}-main",
                    **self._cd_main_assembly_kwargs(exp),
                ))

        match = self._match_branch_roles(
            states,
            {
                "terminal": lambda st: self._special_d3_direct_terminal(center, st),
                "chain1": self._branch_chain_port,
                "chain2": self._branch_chain_port,
            },
        )
        if match:
            term = match["terminal"]
            cand.append(self._make_branch_candidate(
                center=center,
                abcd_type="24_D",
                source_kind=source_kind,
                fixed_role="branch",
                main_states=[center],
                branch_states=[term],
                source_ids=[center.gid, term.gid],
                open_topology=f"-{center.label}(-{term.label})-",
                origin_rule=f"24_D:{source_kind}-direct-terminal",
                analog_template="-24D-",
                group_id=group,
                closure_intents={"main": ["F1", "F2"]},
                pattern_rule=f"{source_kind}:[direct_terminal,chain,chain]",
                main_open=True,
                **self._cd_branch_assembly_kwargs(),
            ))

        match = self._match_branch_roles(
            states,
            {
                "connector": lambda st: self._special_d3_connector(su, st),
                "terminal": self._base_terminal_1_or_22,
                "chain": self._branch_chain_port,
            },
        )
        if match:
            term = match["terminal"]
            for exp in connector_expansions(match["connector"]):
                cand.append(self._candidate_from_main_connector(
                    center, source_kind, "24_D", exp, [], "", group,
                    f"24_D:single-d3-terminal-main+{exp.kind}", "-24D-23-",
                    branch_states=[term],
                    source_ids_extra=[term.gid],
                    pattern_rule=f"{source_kind}:[connector,terminal,chain]+{exp.kind}-main",
                    **self._cd_main_assembly_kwargs(exp),
                ))
        return self._dedupe_branch_candidates(cand)

    def _classify_like24_n_bridge_candidates(
        self,
        skip: set[int],
    ) -> List[_BranchTopologyCandidate]:
        out: List[_BranchTopologyCandidate] = []
        node_by_gid = {int(node.global_id): node for node in self._nodes}
        for connector_node in self._nodes:
            if int(connector_node.global_id) in skip:
                continue
            connector = state_for_node(connector_node)
            c_su = int(connector.su)
            expected_su = {27: 20, 29: 19, 31: 19}.get(c_su)
            if expected_su is None:
                continue
            connector_neighbors = [
                st for _nb, st in self._neighbor_states(connector_node)
            ]
            if has_forbidden_fixed_connector_special_combo(connector, connector_neighbors):
                continue
            neighbors = [
                (nb, st)
                for nb, st in self._neighbor_states(connector_node)
                if int(nb.global_id) not in skip
                and int(st.su) == int(expected_su)
                and is_special_d3(st)
            ]
            if len(neighbors) != 2:
                continue
            (left_node, left_st), (right_node, right_st) = sorted(
                neighbors,
                key=lambda item: int(item[1].gid),
            )
            if self._has_parallel_double_d3_connector(connector, left_st, right_st):
                continue
            left_types = self._like24_bridge_side_types(left_node, left_st, connector)
            right_types = self._like24_bridge_side_types(right_node, right_st, connector)
            if not left_types or not right_types:
                continue
            for left_type in left_types:
                for right_type in right_types:
                    bridge_kwargs = self._like24_bridge_assembly_kwargs(left_type, right_type, connector)
                    if not bridge_kwargs:
                        continue
                    out.append(self._make_like24_bridge_candidate(
                        connector=connector,
                        left=left_st,
                        right=right_st,
                        left_type=left_type,
                        right_type=right_type,
                        **bridge_kwargs,
                    ))
        return self._dedupe_branch_candidates(out)

    def _has_parallel_double_d3_connector(self, connector: Any, left: Any, right: Any) -> bool:
        if not (
            is_double_special_d3(left, int(left.su))
            and is_double_special_d3(right, int(right.su))
        ):
            return False
        connector_su = int(getattr(connector, "su", -1))
        if connector_su not in {27, 29, 31}:
            return False
        left_gid = int(getattr(left, "gid", -1))
        right_gid = int(getattr(right, "gid", -1))
        connector_gid = int(getattr(connector, "gid", -1))
        for node in list(self._nodes or []):
            try:
                if int(node.global_id) == connector_gid or int(node.su_type) != connector_su:
                    continue
                neighbor_ids = {
                    int(st.gid)
                    for _nb, st in self._neighbor_states(node)
                    if int(st.gid) >= 0
                }
            except Exception:
                continue
            if left_gid in neighbor_ids and right_gid in neighbor_ids:
                return True
        return False

    def _like24_bridge_side_types(self, node: SUNode, center: Any, connector: Any) -> List[str]:
        states = [
            st
            for _nb, st in self._neighbor_states(node)
            if int(st.gid) != int(connector.gid)
        ]
        out: List[str] = []
        su = int(center.su)
        if self._match_branch_roles(
            states,
            {"chain": self._branch_24ab_chain_port, "aromatic": self._is_aromatic_11},
        ):
            out.append("24_A")
        if self._match_branch_roles(
            states,
            {"terminal": self._base_terminal_1_or_22, "aromatic": self._is_aromatic_11},
        ):
            out.append("24_B")
        if self._is_double_d3_center(center, su) and self._match_branch_roles(
            states,
            {
                "connector": lambda st: self._special_d3_connector(su, st),
                "chain": self._double_d3_abstract_port,
            },
        ):
            out.append("24_C")
        if self._match_branch_roles(
            states,
            {"chain1": self._branch_chain_port, "chain2": self._branch_chain_port},
        ):
            out.append("24_C")
        if self._match_branch_roles(
            states,
            {"terminal": self._base_terminal_1_or_22, "chain": self._branch_chain_port},
        ):
            out.append("24_D")
        if self._match_branch_roles(
            states,
            {"direct_terminal": lambda st: self._special_d3_direct_terminal(center, st), "chain1": self._branch_chain_port},
        ):
            out.append("24_D")
        return sorted(set(out))

    def _like24_bridge_assembly_kwargs(
        self,
        left_type: str,
        right_type: str,
        connector: Any,
    ) -> Dict[str, Any]:
        left_ab = str(left_type) in {"24_A", "24_B"}
        right_ab = str(right_type) in {"24_A", "24_B"}
        left_cd = str(left_type) in {"24_C", "24_D"}
        right_cd = str(right_type) in {"24_C", "24_D"}
        connector_uses_02 = int(connector.su) in {0, 2}
        if left_ab and right_ab:
            return {
                "assembly_family": "24_AB_N_AB_BRIDGE",
                "assembly_shape": "11-24(?)-23-24(?)-11",
                "usage_modes": ["flex_branch"],
                "ring_eligible": False,
                "ring_scope": "",
                "ring_position_role": "endpoint_24AB_bridge",
                "disallow_ring_reason": "AB_AB_bridge_direct_flex_only",
            }
        if (left_ab and right_cd) or (left_cd and right_ab):
            ring_eligible = not connector_uses_02
            usage = ["flex_branch", "side_branch"]
            if ring_eligible:
                usage.append("aliphatic_side_ring")
            return {
                "assembly_family": "24_AB_N_CD_BRIDGE",
                "assembly_shape": "11-24(?)-23-24(?)-",
                "usage_modes": usage,
                "ring_eligible": ring_eligible,
                "ring_scope": "side_aliphatic_ring" if ring_eligible else "",
                "ring_position_role": "endpoint_24AB_to_internal_24CD" if ring_eligible else "",
                "disallow_ring_reason": "" if ring_eligible else "main_connector_0_or_2",
            }
        if left_cd and right_cd:
            ring_eligible = not connector_uses_02
            usage = ["flex_branch", "side_branch"]
            if ring_eligible:
                usage.extend(["side_aliphatic_ring", "fused_aliphatic_ring"])
            return {
                "assembly_family": "24_CD_N_CD_BRIDGE",
                "assembly_shape": "-24(?)-23-24(?)-",
                "usage_modes": usage,
                "ring_eligible": ring_eligible,
                "ring_scope": "side_or_fused_aliphatic_ring" if ring_eligible else "",
                "ring_position_role": "internal_24CD_bridge" if ring_eligible else "",
                "disallow_ring_reason": "" if ring_eligible else "main_connector_0_or_2",
            }
        return {}

    def _make_like24_bridge_candidate(
        self,
        *,
        connector: Any,
        left: Any,
        right: Any,
        left_type: str,
        right_type: str,
        assembly_family: str,
        assembly_shape: str,
        usage_modes: Sequence[str],
        ring_eligible: bool = False,
        ring_scope: str = "",
        ring_position_role: str = "",
        disallow_ring_reason: str = "",
    ) -> _BranchTopologyCandidate:
        left_prefix = "11-" if str(left_type) in {"24_A", "24_B"} else "-"
        right_suffix = "-11" if str(right_type) in {"24_A", "24_B"} else "-"
        left_terminal = self._closed_terminal_branch_for_bridge(left, exclude_ids={connector.gid, right.gid})
        right_terminal = self._closed_terminal_branch_for_bridge(right, exclude_ids={connector.gid, left.gid})
        left_aromatic = self._closed_aromatic_for_bridge(left, exclude_ids={connector.gid, right.gid})
        right_aromatic = self._closed_aromatic_for_bridge(right, exclude_ids={connector.gid, left.gid})
        left_branch = (
            "H" if str(left_type) in {"24_A", "24_C"}
            else str(left_terminal.label) if left_terminal is not None
            else "?"
        )
        right_branch = (
            "H" if str(right_type) in {"24_A", "24_C"}
            else str(right_terminal.label) if right_terminal is not None
            else "?"
        )
        closed_branch_states = [
            st for st in (left_terminal, right_terminal)
            if st is not None
        ]
        actual_prefix_states = [
            st for st, typ in ((left_aromatic, left_type),)
            if st is not None and str(typ) in {"24_A", "24_B"}
        ]
        actual_suffix_states = [
            st for st, typ in ((right_aromatic, right_type),)
            if st is not None and str(typ) in {"24_A", "24_B"}
        ]
        actual_used_states = (
            list(actual_prefix_states)
            + [left, connector, right]
            + list(actual_suffix_states)
            + list(closed_branch_states)
        )
        topology = (
            f"{left_prefix}{left.label}(-{left_branch})-"
            f"{connector.label}-{right.label}(-{right_branch}){right_suffix}"
        )
        if not topology.endswith("-") and str(right_type) in {"24_C", "24_D"}:
            topology += "-"
        group = f"like24_bridge:{int(connector.gid)}:{int(left.gid)}:{int(right.gid)}"
        closure_intents: Dict[str, List[str]] = {}
        if "H" in {left_branch, right_branch}:
            closure_intents["branch"] = ["H"]
        if str(assembly_family) != "24_AB_N_AB_BRIDGE":
            closure_intents["main"] = ["F1", "F2"]
        return self._make_branch_candidate(
            center=left,
            abcd_type=f"{left_type}+{right_type}",
            source_kind="like24_n_bridge",
            fixed_role="bridge",
            main_states=[left, connector, right],
            branch_states=closed_branch_states,
            source_ids=[
                int(x) for x in [
                    left.gid,
                    connector.gid,
                    right.gid,
                    *[int(st.gid) for st in closed_branch_states],
                ]
                if int(x) >= 0
            ],
            open_topology=topology,
            actual_main_path_su=[left.su, connector.su, right.su],
            actual_main_path_labels=[left.label, connector.label, right.label],
            actual_used_su=[int(st.su) for st in actual_used_states],
            actual_used_source_ids=[int(st.gid) for st in actual_used_states if int(st.gid) >= 0],
            chain_port_signature=[left.su, connector.su, right.su],
            origin_rule=(
                f"{assembly_family}:connector{int(connector.gid)}:"
                f"{left_type}{int(left.gid)}-{int(connector.su)}-{right_type}{int(right.gid)}"
            ),
            analog_template=str(assembly_shape),
            group_id=group,
            connector_expansions=[
                self._direct_path_expansion("N", connector, [connector, right], str(assembly_shape))
            ],
            closure_intents=closure_intents,
            pattern_rule=f"like24-N-like24:{left_type}+N+{right_type}",
            main_open=bool(closure_intents.get("main")),
            branch_open=bool(closure_intents.get("branch")),
            assembly_family=assembly_family,
            assembly_shape=assembly_shape,
            usage_modes=usage_modes,
            ring_eligible=ring_eligible,
            ring_scope=ring_scope,
            ring_position_role=ring_position_role,
            selected_fixed_role="bridge",
            main_connector_kind="N",
            main_connector_ids=[int(connector.gid)],
            exclusive_group_key=group,
            disallow_ring_reason=disallow_ring_reason,
            bridge_connector_id=int(connector.gid),
            bridge_connector_su=int(connector.su),
            bridge_left_gid=int(left.gid),
            bridge_right_gid=int(right.gid),
            bridge_left_abcd_type=str(left_type),
            bridge_right_abcd_type=str(right_type),
            suppresses_local_candidate_keys=[
                self._branch_group_id(left),
                self._branch_group_id(right),
            ],
            priority_rank=100,
        )

    def _closed_terminal_branch_for_bridge(
        self,
        center: Any,
        *,
        exclude_ids: set[int],
    ) -> Optional[Any]:
        node = self._node_lookup.get(int(getattr(center, "gid", -1)))
        if node is None:
            return None
        excluded = {int(x) for x in set(exclude_ids or set()) if int(x) >= 0}
        terminals: List[Any] = []
        for nb, st in self._neighbor_states(node):
            if int(nb.global_id) in excluded:
                continue
            if int(st.su) in TERMINAL_BASE_SU:
                terminals.append(st)
        terminals.sort(key=lambda st: (int(st.su), str(st.label), int(st.gid)))
        return terminals[0] if terminals else None

    def _closed_aromatic_for_bridge(
        self,
        center: Any,
        *,
        exclude_ids: set[int],
    ) -> Optional[Any]:
        node = self._node_lookup.get(int(getattr(center, "gid", -1)))
        if node is None:
            return None
        excluded = {int(x) for x in set(exclude_ids or set()) if int(x) >= 0}
        aromatics: List[Any] = []
        for nb, st in self._neighbor_states(node):
            if int(nb.global_id) in excluded:
                continue
            if int(st.su) == 11:
                aromatics.append(st)
        aromatics.sort(key=lambda st: (int(st.su), str(st.label), int(st.gid)))
        return aromatics[0] if aromatics else None

    def _candidate_from_branch_connector(
        self,
        center: Any,
        source_kind: str,
        abcd_type: str,
        exp: _FixedConnectorExpansion,
        left_prefix: str,
        group_id: str,
        origin_rule: str,
        analog_template: str,
        pattern_rule: str = "",
        assembly_family: str = "",
        assembly_shape: str = "",
        usage_modes: Optional[Sequence[str]] = None,
        ring_eligible: bool = False,
        ring_scope: str = "",
        ring_position_role: str = "",
        disallow_ring_reason: str = "",
        left_endpoint_state: Optional[Any] = None,
    ) -> _BranchTopologyCandidate:
        prefix = f"{left_prefix}-" if left_prefix else "-"
        topology = f"{prefix}{center.label}(-{'-'.join(exp.path_labels)})-"
        closure_intents = {"main": ["F1", "F2"]}
        if exp.open_state == "chain":
            closure_intents["branch"] = ["H1"]
        endpoint_su = [int(left_endpoint_state.su)] if left_endpoint_state is not None else []
        endpoint_ids = [int(left_endpoint_state.gid)] if left_endpoint_state is not None and int(left_endpoint_state.gid) >= 0 else []
        branch_path_su = [int(x) for x in list(exp.path_su or [])]
        branch_path_ids = [int(x) for x in list(exp.node_ids or []) if int(x) >= 0]
        return self._make_branch_candidate(
            center=center,
            abcd_type=abcd_type,
            source_kind=source_kind,
            fixed_role="branch",
            main_states=[center],
            branch_states=self._states_from_expansion(exp),
            source_ids=[center.gid] + list(exp.node_ids),
            open_topology=topology,
            origin_rule=origin_rule,
            analog_template=analog_template,
            group_id=group_id,
            connector_expansions=[exp],
            closure_intents=closure_intents,
            actual_main_path_su=[int(center.su)],
            actual_main_path_labels=[str(center.label)],
            actual_used_su=list(endpoint_su) + [int(center.su)] + list(branch_path_su),
            actual_used_source_ids=list(endpoint_ids) + [int(center.gid)] + list(branch_path_ids),
            chain_port_signature=[int(center.su)] + list(branch_path_su),
            pattern_rule=pattern_rule or origin_rule,
            main_open=True,
            branch_open=bool(exp.open_state == "chain"),
            assembly_family=assembly_family,
            assembly_shape=assembly_shape,
            usage_modes=usage_modes,
            ring_eligible=ring_eligible,
            ring_scope=ring_scope,
            ring_position_role=ring_position_role,
            selected_fixed_role="branch",
            branch_connector_kind=str(exp.kind) if assembly_family else "",
            branch_connector_ids=self._connector_expansion_ids(exp) if assembly_family else [],
            exclusive_group_key=group_id,
            disallow_ring_reason=disallow_ring_reason,
        )

    def _candidate_from_main_connector(
        self,
        center: Any,
        source_kind: str,
        abcd_type: str,
        main_exp: _FixedConnectorExpansion,
        branch_exps: Sequence[_FixedConnectorExpansion],
        left_prefix: str,
        group_id: str,
        origin_rule: str,
        analog_template: str,
        branch_states: Optional[Sequence[Any]] = None,
        source_ids_extra: Optional[Sequence[int]] = None,
        pattern_rule: str = "",
        assembly_family: str = "",
        assembly_shape: str = "",
        usage_modes: Optional[Sequence[str]] = None,
        ring_eligible: bool = False,
        ring_scope: str = "",
        ring_position_role: str = "",
        branch_connector_kind: str = "",
        branch_connector_ids: Optional[Sequence[int]] = None,
        disallow_ring_reason: str = "",
        left_endpoint_state: Optional[Any] = None,
    ) -> _BranchTopologyCandidate:
        branch_states_all: List[Any] = list(branch_states or [])
        if not branch_states_all:
            for exp in branch_exps:
                branch_states_all.extend(self._states_from_expansion(exp))
        main_states = [center] + self._states_from_expansion(main_exp)
        prefix = f"{left_prefix}-" if left_prefix else "-"
        branch_text = "-".join(str(st.label) for st in branch_states_all) if branch_states_all else "H"
        topology = f"{prefix}{center.label}(-{branch_text})-{ '-'.join(main_exp.path_labels) }"
        if main_exp.open_state == "chain":
            topology += "-"
        source_ids = [center.gid] + list(main_exp.node_ids) + list(source_ids_extra or [])
        for exp in branch_exps:
            source_ids.extend(exp.node_ids)
        endpoint_su = [int(left_endpoint_state.su)] if left_endpoint_state is not None else []
        endpoint_ids = [int(left_endpoint_state.gid)] if left_endpoint_state is not None and int(left_endpoint_state.gid) >= 0 else []
        main_path_su = [int(center.su)] + [int(x) for x in list(main_exp.path_su or [])]
        main_path_labels = [str(center.label)] + [str(x) for x in list(main_exp.path_labels or [])]
        main_path_ids = [int(center.gid)] + [int(x) for x in list(main_exp.node_ids or []) if int(x) >= 0]
        branch_path_su: List[int] = []
        branch_path_ids: List[int] = []
        if branch_states:
            branch_path_su.extend(int(st.su) for st in list(branch_states or []))
            branch_path_ids.extend(int(st.gid) for st in list(branch_states or []) if int(st.gid) >= 0)
        else:
            for exp in branch_exps:
                branch_path_su.extend(int(x) for x in list(exp.path_su or []))
                branch_path_ids.extend(int(x) for x in list(exp.node_ids or []) if int(x) >= 0)
        branch_intents: List[str] = []
        if not branch_states_all:
            branch_intents.append("H")
        if any(str(exp.open_state) == "chain" for exp in list(branch_exps or [])):
            branch_intents.append("H1")
        main_intents: List[str] = []
        if str(abcd_type) in {"24_C", "24_D"} or str(main_exp.open_state) == "chain":
            main_intents.extend(["F1", "F2"])
        closure_intents: Dict[str, List[str]] = {}
        if branch_intents:
            closure_intents["branch"] = branch_intents
        if main_intents:
            closure_intents["main"] = main_intents
        return self._make_branch_candidate(
            center=center,
            abcd_type=abcd_type,
            source_kind=source_kind,
            fixed_role="main",
            main_states=main_states,
            branch_states=branch_states_all,
            source_ids=source_ids,
            open_topology=topology,
            origin_rule=origin_rule,
            analog_template=analog_template,
            group_id=group_id,
            connector_expansions=[main_exp] + list(branch_exps or []),
            closure_intents=closure_intents,
            actual_main_path_su=list(main_path_su),
            actual_main_path_labels=list(main_path_labels),
            actual_used_su=list(endpoint_su) + list(main_path_su) + list(branch_path_su),
            actual_used_source_ids=list(endpoint_ids) + list(main_path_ids) + list(branch_path_ids),
            chain_port_signature=list(main_path_su),
            pattern_rule=pattern_rule or origin_rule,
            main_open=bool(main_intents),
            branch_open=bool(branch_intents),
            assembly_family=assembly_family,
            assembly_shape=assembly_shape,
            usage_modes=usage_modes,
            ring_eligible=ring_eligible,
            ring_scope=ring_scope,
            ring_position_role=ring_position_role,
            selected_fixed_role="main",
            main_connector_kind=str(main_exp.kind) if assembly_family else "",
            branch_connector_kind=str(branch_connector_kind),
            main_connector_ids=self._connector_expansion_ids(main_exp) if assembly_family else [],
            branch_connector_ids=branch_connector_ids,
            exclusive_group_key=group_id,
            disallow_ring_reason=disallow_ring_reason,
        )

    def _candidate_from_double_d3_fixed_connectors(
        self,
        *,
        center: Any,
        source_kind: str,
        abcd_type: str,
        main_exp: _FixedConnectorExpansion,
        branch_exp: _FixedConnectorExpansion,
        abstract_state: Any,
        group_id: str,
        origin_rule: str,
        analog_template: str,
        pattern_rule: str = "",
        assembly_family: str = "",
        assembly_shape: str = "",
        usage_modes: Optional[Sequence[str]] = None,
        ring_eligible: bool = False,
        ring_scope: str = "",
        ring_position_role: str = "",
        disallow_ring_reason: str = "",
    ) -> _BranchTopologyCandidate:
        main_states = [center] + self._states_from_expansion(main_exp)
        branch_states = self._states_from_expansion(branch_exp)
        main_text = "-".join(str(x) for x in list(main_exp.path_labels or []))
        branch_text = "-".join(str(x) for x in list(branch_exp.path_labels or []))
        topology = f"-{center.label}(-{branch_text})-{main_text}"
        if str(main_exp.open_state) == "chain":
            topology += "-"

        actual_su: List[int] = []
        actual_ids: List[int] = []
        seen_ids: set[int] = set()

        def add_actual(su_val: int, gid_val: int) -> None:
            gid_int = int(gid_val)
            if gid_int >= 0:
                if gid_int in seen_ids:
                    return
                seen_ids.add(gid_int)
                actual_ids.append(gid_int)
            actual_su.append(int(su_val))

        add_actual(int(center.su), int(center.gid))
        for su_val, gid_val in zip(
            [int(x) for x in list(main_exp.actual_path_su or main_exp.path_su or [])],
            [int(x) for x in list(main_exp.actual_node_ids or main_exp.node_ids or [])],
        ):
            add_actual(su_val, gid_val)
        for su_val, gid_val in zip(
            [int(x) for x in list(branch_exp.actual_path_su or branch_exp.path_su or [])],
            [int(x) for x in list(branch_exp.actual_node_ids or branch_exp.node_ids or [])],
        ):
            add_actual(su_val, gid_val)

        source_ids = sorted(set(
            [int(center.gid)]
            + [int(x) for x in list(main_exp.node_ids or []) if int(x) >= 0]
            + [int(x) for x in list(branch_exp.node_ids or []) if int(x) >= 0]
            + list(actual_ids)
        ))
        abstracted_neighbors = [
            self._branch_abstracted_neighbor_row(abstract_state, "double_d3_classification_evidence"),
            *[
                dict(row)
                for row in list(main_exp.abstracted_neighbors or []) + list(branch_exp.abstracted_neighbors or [])
                if isinstance(row, dict)
            ],
        ]
        evidence_ids = sorted(set(
            [int(abstract_state.gid)]
            + [
                int(row.get("actual_gid", -1))
                for row in abstracted_neighbors
                if isinstance(row, dict) and int(row.get("actual_gid", -1)) >= 0
            ]
        ))
        main_path_su = [int(center.su)] + [int(x) for x in list(main_exp.path_su or [])]
        main_path_labels = [str(center.label)] + [str(x) for x in list(main_exp.path_labels or [])]
        closure_intents: Dict[str, List[str]] = {}
        if str(abcd_type) in {"24_C", "24_D"} or str(main_exp.open_state) == "chain":
            closure_intents["main"] = ["F1", "F2"]

        return self._make_branch_candidate(
            center=center,
            abcd_type=abcd_type,
            source_kind=source_kind,
            fixed_role="double_fixed",
            main_states=main_states,
            branch_states=branch_states,
            source_ids=source_ids,
            open_topology=topology,
            origin_rule=origin_rule,
            analog_template=analog_template,
            group_id=group_id,
            connector_expansions=[main_exp, branch_exp],
            closure_intents=closure_intents,
            actual_main_path_su=main_path_su,
            actual_main_path_labels=main_path_labels,
            actual_used_su=actual_su,
            actual_used_source_ids=actual_ids,
            chain_port_signature=main_path_su,
            pattern_rule=pattern_rule or origin_rule,
            main_open=bool(closure_intents.get("main")),
            branch_open=False,
            assembly_family=assembly_family,
            assembly_shape=assembly_shape,
            usage_modes=usage_modes,
            ring_eligible=ring_eligible,
            ring_scope=ring_scope,
            ring_position_role=ring_position_role,
            selected_fixed_role="double_fixed",
            main_connector_kind=str(main_exp.kind) if assembly_family else "",
            branch_connector_kind=f"fixed_{branch_exp.kind}" if assembly_family else "",
            main_connector_ids=self._connector_expansion_ids(main_exp) if assembly_family else [],
            branch_connector_ids=self._connector_expansion_ids(branch_exp) if assembly_family else [],
            exclusive_group_key=group_id,
            disallow_ring_reason=disallow_ring_reason,
            evidence_ids=evidence_ids,
            abstracted_neighbors=abstracted_neighbors,
        )

    @staticmethod
    def _connector_expansion_ids(exp: _FixedConnectorExpansion) -> List[int]:
        return [int(x) for x in list(getattr(exp, "node_ids", []) or []) if int(x) >= 0]

    def _ab_main_assembly_kwargs(self, main_exp: _FixedConnectorExpansion) -> Dict[str, Any]:
        kind = str(main_exp.kind)
        if kind == "M":
            return {
                "assembly_family": "24_AB_M",
                "assembly_shape": "11-24(?)-23-11",
                "usage_modes": ["flex_branch"],
            }
        if kind == "P":
            return {
                "assembly_family": "24_AB_P",
                "assembly_shape": "11-24(?)-23-22",
                "usage_modes": ["side_branch"],
            }
        ring_eligible = not self._connector_expansion_uses_connector(main_exp, {0, 2})
        usage = ["flex_branch", "side_branch"]
        if ring_eligible:
            usage.append("aliphatic_side_ring")
        return {
            "assembly_family": "24_AB_N_MAIN",
            "assembly_shape": "11-24(?)-23-23/24-",
            "usage_modes": usage,
            "ring_eligible": ring_eligible,
            "ring_scope": "side_aliphatic_ring" if ring_eligible else "",
            "ring_position_role": "endpoint_24AB" if ring_eligible else "",
            "disallow_ring_reason": "" if ring_eligible else "main_connector_0_or_2",
        }

    @staticmethod
    def _ab_branch_assembly_kwargs() -> Dict[str, Any]:
        return {
            "assembly_family": "24_AB_N_BRANCH",
            "assembly_shape": "11-24(?)-",
            "usage_modes": [
                "ring_only",
                "aliphatic_side_ring",
                "fused_aliphatic_ring",
                "side_branch",
                "flex_branch",
            ],
            "ring_eligible": True,
            "ring_scope": "side_or_fused_aliphatic_ring",
            "ring_position_role": "endpoint_24AB",
        }

    def _ab_no_connector_ring_kwargs(self) -> Dict[str, Any]:
        return self._ab_branch_assembly_kwargs()

    def _cd_main_assembly_kwargs(
        self,
        main_exp: _FixedConnectorExpansion,
        branch_exp: Optional[_FixedConnectorExpansion] = None,
    ) -> Dict[str, Any]:
        kind = str(main_exp.kind)
        main_uses_02 = self._connector_expansion_uses_connector(main_exp, {0, 2})
        if kind == "M":
            if branch_exp is not None and str(branch_exp.kind) == "M":
                return {
                    "assembly_family": "24_CD_M_BRIDGE",
                    "assembly_shape": "11-23-24(?)-23-11",
                    "usage_modes": ["flex_branch"],
                    "ring_eligible": False,
                    "ring_scope": "",
                    "ring_position_role": "internal_24CD_bridge",
                }
            usage = ["flex_branch", "side_branch"]
            ring_eligible = not main_uses_02
            if ring_eligible:
                usage.append("aliphatic_side_ring")
            return {
                "assembly_family": "24_CD_M_MAIN",
                "assembly_shape": "-24(?)-23-11",
                "usage_modes": usage,
                "ring_eligible": ring_eligible,
                "ring_scope": "side_aliphatic_ring" if ring_eligible else "",
                "ring_position_role": "internal_24CD" if ring_eligible else "",
                "disallow_ring_reason": "" if ring_eligible else "main_connector_0_or_2",
            }
        if kind == "P":
            return {
                "assembly_family": "24_CD_P",
                "assembly_shape": "-24(?)-23-22",
                "usage_modes": ["side_branch"],
                "ring_eligible": False,
                "disallow_ring_reason": "terminal_P_side_branch_only",
            }
        ring_eligible = not main_uses_02
        usage = ["flex_branch", "side_branch"]
        if ring_eligible:
            usage.extend(["upper_aliphatic_ring", "fused_aliphatic_ring"])
        return {
            "assembly_family": "24_CD_N_MAIN",
            "assembly_shape": "-24(?)-23-23/24-",
            "usage_modes": usage,
            "ring_eligible": ring_eligible,
            "ring_scope": "upper_or_fused_aliphatic_ring" if ring_eligible else "",
            "ring_position_role": "internal_24CD" if ring_eligible else "",
            "disallow_ring_reason": "" if ring_eligible else "main_connector_0_or_2",
        }

    def _cd_14_main_assembly_kwargs(self, bridgehead_replace: bool = False) -> Dict[str, Any]:
        return {
            "assembly_family": "24_CD_N_14_MAIN",
            "assembly_shape": "-24(?)-23/24(?)-",
            "usage_modes": [
                "flex_branch",
                "side_branch",
                "side_aliphatic_ring",
                "fused_aliphatic_ring",
            ],
            "ring_eligible": True,
            "ring_scope": "side_or_fused_aliphatic_ring",
            "ring_position_role": "internal_24CD",
            "bridgehead_replace_24c_eligible": bool(bridgehead_replace),
        }

    def _cd_branch_assembly_kwargs(
        self,
        exp: Optional[_FixedConnectorExpansion] = None,
    ) -> Dict[str, Any]:
        return {
            "assembly_family": "24_CD_N_BRANCH",
            "assembly_shape": "-24(?)-",
            "usage_modes": [
                "ring_only",
                "side_aliphatic_ring",
                "upper_aliphatic_ring",
                "fused_aliphatic_ring",
            ],
            "ring_eligible": True,
            "ring_scope": "side_or_upper_or_fused_aliphatic_ring",
            "ring_position_role": "internal_24CD",
        }

    @staticmethod
    def _connector_expansion_uses_connector(exp: _FixedConnectorExpansion, connector_su: set[int]) -> bool:
        conn = getattr(exp, "connector", None)
        return conn is not None and int(conn.su) in {int(x) for x in set(connector_su or set())}

    @staticmethod
    def _allowed_24ab_double_pair(main_exp: _FixedConnectorExpansion, branch_exp: _FixedConnectorExpansion) -> bool:
        main_kind = str(main_exp.kind)
        branch_kind = str(branch_exp.kind)
        if main_kind == "M":
            return branch_kind in {"M", "N", "P"}
        if main_kind == "N":
            return branch_kind in {"N", "P"}
        if main_kind == "P":
            return branch_kind == "P"
        return False

    @staticmethod
    def _allowed_24cd_double_pair(main_exp: _FixedConnectorExpansion, branch_exp: _FixedConnectorExpansion) -> bool:
        main_kind = str(main_exp.kind)
        branch_kind = str(branch_exp.kind)
        if main_kind == "M":
            return branch_kind in {"M", "N", "P"}
        if main_kind == "N":
            return branch_kind in {"N", "P"}
        if main_kind == "P":
            return branch_kind == "P"
        return False

    def _make_branch_candidate(
        self,
        *,
        center: Any,
        abcd_type: str,
        source_kind: str,
        fixed_role: str,
        main_states: Sequence[Any],
        branch_states: Sequence[Any],
        source_ids: Sequence[int],
        open_topology: str,
        origin_rule: str,
        analog_template: str,
        group_id: Optional[str] = None,
        connector_expansions: Optional[Sequence[_FixedConnectorExpansion]] = None,
        closure_intents: Optional[Dict[str, List[str]]] = None,
        pattern_rule: str = "",
        main_open: bool = False,
        branch_open: bool = False,
        assembly_family: str = "",
        assembly_shape: str = "",
        usage_modes: Optional[Sequence[str]] = None,
        ring_eligible: bool = False,
        ring_scope: str = "",
        ring_position_role: str = "",
        selected_fixed_role: str = "",
        main_connector_kind: str = "",
        branch_connector_kind: str = "",
        main_connector_ids: Optional[Sequence[int]] = None,
        branch_connector_ids: Optional[Sequence[int]] = None,
        exclusive_group_key: str = "",
        disallow_ring_reason: str = "",
        bridge_connector_id: int = -1,
        bridge_connector_su: int = -1,
        bridge_left_gid: int = -1,
        bridge_right_gid: int = -1,
        bridge_left_abcd_type: str = "",
        bridge_right_abcd_type: str = "",
        bridgehead_replace_24c_eligible: bool = False,
        suppresses_local_candidate_keys: Optional[Sequence[str]] = None,
        priority_rank: int = 0,
        actual_main_path_su: Optional[Sequence[int]] = None,
        actual_main_path_labels: Optional[Sequence[str]] = None,
        actual_used_su: Optional[Sequence[int]] = None,
        actual_used_source_ids: Optional[Sequence[int]] = None,
        chain_port_signature: Optional[Sequence[int]] = None,
        branch_slots_su: Optional[Dict[str, Sequence[int]]] = None,
        branch_slot_labels: Optional[Dict[str, str]] = None,
        fixed_22_is_abstract: bool = False,
        evidence_ids: Optional[Sequence[int]] = None,
        abstracted_neighbors: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> _BranchTopologyCandidate:
        main = list(main_states or [])
        branch = list(branch_states or [])
        main_su = [int(st.su) for st in main]
        main_labels = [str(st.label) for st in main]
        branch_su = [int(st.su) for st in branch]
        branch_labels = [str(st.label) for st in branch]
        actual_main_su = (
            [int(x) for x in list(actual_main_path_su or [])]
            if actual_main_path_su is not None
            else list(main_su)
        )
        actual_main_labels = (
            [str(x) for x in list(actual_main_path_labels or [])]
            if actual_main_path_labels is not None
            else list(main_labels)
        )
        actual_su = (
            [int(x) for x in list(actual_used_su or [])]
            if actual_used_su is not None
            else list(actual_main_su) + list(branch_su)
        )
        actual_ids = [int(x) for x in list(actual_used_source_ids or [])]
        src_ids = sorted(set(
            int(x)
            for x in list(source_ids or []) + list(actual_ids)
            if int(x) >= 0
        ))
        signature = (
            [int(x) for x in list(chain_port_signature or [])]
            if chain_port_signature is not None
            else list(main_su)
        )
        open_ends = self._branch_candidate_open_ends(str(abcd_type), str(open_topology))
        native24_class = str(abcd_type) if str(source_kind) == "native24" else ""
        analog24_class = "" if str(source_kind) == "native24" else str(abcd_type)
        branch_display_class = (
            f"native24:{native24_class}"
            if native24_class
            else f"{str(source_kind)}:{analog24_class}"
        )
        return _BranchTopologyCandidate(
            group_id=str(group_id or self._branch_group_id(center)),
            abcd_type=str(abcd_type),
            center=center,
            source_kind=str(source_kind),
            fixed_role=str(fixed_role),
            main_body_su=list(main_su),
            main_labels=list(main_labels),
            branch_su=list(branch_su),
            branch_labels=list(branch_labels),
            source_ids=src_ids,
            open_topology=str(open_topology),
            open_ends=list(open_ends),
            native24_class=native24_class,
            analog24_class=analog24_class,
            branch_display_class=branch_display_class,
            actual_main_path_su=list(actual_main_su),
            actual_main_path_labels=list(actual_main_labels),
            actual_used_su=list(actual_su),
            actual_used_source_ids=list(actual_ids or src_ids),
            chain_port_signature=list(signature),
            connector_expansions=list(connector_expansions or []),
            origin_rule=str(origin_rule),
            analog_template=str(analog_template),
            closure_intents=dict(closure_intents or {}),
            pattern_rule=str(pattern_rule or origin_rule),
            main_open=bool(main_open),
            branch_open=bool(branch_open),
            assembly_family=str(assembly_family),
            assembly_shape=str(assembly_shape),
            usage_modes=[str(x) for x in list(usage_modes or [])],
            ring_eligible=bool(ring_eligible),
            ring_scope=str(ring_scope),
            ring_position_role=str(ring_position_role),
            selected_fixed_role=str(selected_fixed_role or fixed_role),
            main_connector_kind=str(main_connector_kind),
            branch_connector_kind=str(branch_connector_kind),
            main_connector_ids=[int(x) for x in list(main_connector_ids or []) if int(x) >= 0],
            branch_connector_ids=[int(x) for x in list(branch_connector_ids or []) if int(x) >= 0],
            exclusive_group_key=str(exclusive_group_key or group_id or self._branch_group_id(center)),
            disallow_ring_reason=str(disallow_ring_reason),
            bridge_connector_id=int(bridge_connector_id),
            bridge_connector_su=int(bridge_connector_su),
            bridge_left_gid=int(bridge_left_gid),
            bridge_right_gid=int(bridge_right_gid),
            bridge_left_abcd_type=str(bridge_left_abcd_type),
            bridge_right_abcd_type=str(bridge_right_abcd_type),
            bridgehead_replace_24c_eligible=bool(bridgehead_replace_24c_eligible),
            suppresses_local_candidate_keys=[str(x) for x in list(suppresses_local_candidate_keys or [])],
            priority_rank=int(priority_rank),
            branch_slots_su={
                str(k): [int(x) for x in list(v or [])]
                for k, v in dict(branch_slots_su or {}).items()
            },
            branch_slot_labels={
                str(k): str(v)
                for k, v in dict(branch_slot_labels or {}).items()
            },
            fixed_22_is_abstract=bool(fixed_22_is_abstract),
            evidence_ids=[
                int(x) for x in list(evidence_ids or [])
                if int(x) >= 0
            ],
            abstracted_neighbors=[
                dict(row) for row in list(abstracted_neighbors or [])
                if isinstance(row, dict)
            ],
        )

    def _make_pending_branch_topology(
        self,
        group_id: str,
        candidates: Sequence[_BranchTopologyCandidate],
    ) -> ChainSpec:
        alts = list(candidates or [])
        primary = alts[0]
        source_ids = sorted(set(int(gid) for cand in alts for gid in cand.source_ids if int(gid) >= 0))
        profiles = [cand.profile("branch_candidate") for cand in alts]

        def _candidate_semantics(cand: _BranchTopologyCandidate) -> Dict[str, str]:
            family = str(cand.assembly_family or "")
            usage_modes = {str(x) for x in list(cand.usage_modes or [])}
            closure_intents = dict(cand.closure_intents or {})
            main_intents = [str(x) for x in list(closure_intents.get("main", []) or [])]
            branch_intents = [str(x) for x in list(closure_intents.get("branch", []) or [])]

            if family in {"24_AB_M", "24_CD_M_BRIDGE", "24_AB_N_AB_BRIDGE"}:
                topology_role = "definite_flex"
            elif family in {"24_AB_P", "24_CD_P"}:
                topology_role = "definite_side"
            elif "ring_only" in usage_modes and not (usage_modes & {"flex_branch", "side_branch"}):
                topology_role = "ring_slot_only"
            elif "ring_only" in usage_modes or bool(cand.ring_eligible):
                topology_role = "ring_slot_open"
            elif "F1" in main_intents and "F2" in main_intents:
                topology_role = "d_like_open"
            elif "F1" in main_intents:
                topology_role = "b_like_open"
            elif "F2" in main_intents:
                topology_role = "e_like_open"
            else:
                topology_role = "branch_candidate"

            if not main_intents:
                main_open_class = "closed"
            elif len(main_intents) >= 2 and all(str(x).startswith("F1") for x in main_intents):
                main_open_class = "two_chain_open"
            elif len(main_intents) >= 2:
                main_open_class = "chain_terminal_open"
            else:
                main_open_class = "one_chain_open"

            if "H1" in branch_intents:
                branch_closure_need = "H1"
            elif "H" in branch_intents:
                branch_closure_need = "H"
            else:
                branch_closure_need = "none"

            scope = str(cand.ring_scope or "")
            if not cand.ring_eligible:
                ring_usable = "none"
            elif "fused" in scope:
                ring_usable = "fused_ring"
            elif "upper" in scope:
                ring_usable = "upper_ring"
            elif "side" in scope:
                ring_usable = "side_ring"
            else:
                ring_usable = "ring"

            return {
                "topology_role": topology_role,
                "main_open_class": main_open_class,
                "branch_closure_need": branch_closure_need,
                "ring_usable": ring_usable,
            }

        candidate_rows = [
            {
                "abcd_type": str(cand.abcd_type),
                "native24_class": str(cand.native24_class),
                "analog24_class": str(cand.analog24_class),
                "branch_display_class": str(cand.branch_display_class),
                "source_kind": str(cand.source_kind),
                "fixed_role": str(cand.fixed_role),
                "pattern_rule": str(cand.pattern_rule),
                "main_body_su": [int(x) for x in cand.main_body_su],
                "main_labels": list(cand.main_labels),
                "branch_su": [int(x) for x in cand.branch_su],
                "branch_labels": list(cand.branch_labels),
                "branch_slots_su": {
                    str(k): [int(x) for x in list(v or [])]
                    for k, v in dict(cand.branch_slots_su or {}).items()
                },
                "branch_slot_labels": {
                    str(k): str(v)
                    for k, v in dict(cand.branch_slot_labels or {}).items()
                },
                "fixed_22_is_abstract": bool(cand.fixed_22_is_abstract),
                "evidence_ids": [int(x) for x in list(cand.evidence_ids or []) if int(x) >= 0],
                "abstracted_neighbors": [
                    dict(row) for row in list(cand.abstracted_neighbors or [])
                    if isinstance(row, dict)
                ],
                "actual_main_path_su": [int(x) for x in cand.actual_main_path_su],
                "actual_main_path_labels": list(cand.actual_main_path_labels),
                "actual_used_su": [int(x) for x in cand.actual_used_su],
                "actual_used_source_ids": [int(x) for x in cand.actual_used_source_ids],
                "chain_port_signature": [int(x) for x in cand.chain_port_signature],
                "source_ids": [int(x) for x in list(cand.source_ids or []) if int(x) >= 0],
                "open_topology": str(cand.open_topology),
                "open_ends": list(cand.open_ends),
                "main_open": bool(cand.main_open),
                "branch_open": bool(cand.branch_open),
                "closure_intents": {
                    str(k): list(v) for k, v in dict(cand.closure_intents or {}).items()
                },
                "closure_intent_options": self._branch_intent_options(dict(cand.closure_intents or {})),
                "origin_rule": str(cand.origin_rule),
                "analog_template": str(cand.analog_template),
                "assembly_family": str(cand.assembly_family),
                "assembly_shape": str(cand.assembly_shape),
                "usage_modes": list(cand.usage_modes),
                "ring_eligible": bool(cand.ring_eligible),
                "ring_scope": str(cand.ring_scope),
                "ring_position_role": str(cand.ring_position_role),
                "selected_fixed_role": str(cand.selected_fixed_role or cand.fixed_role),
                "main_connector_kind": str(cand.main_connector_kind),
                "branch_connector_kind": str(cand.branch_connector_kind),
                "main_connector_ids": [int(x) for x in list(cand.main_connector_ids or [])],
                "branch_connector_ids": [int(x) for x in list(cand.branch_connector_ids or [])],
                "exclusive_group_key": str(cand.exclusive_group_key or cand.group_id),
                "exclusive_group_keys": [
                    str(cand.exclusive_group_key or cand.group_id),
                    *[str(x) for x in list(cand.suppresses_local_candidate_keys or []) if str(x)],
                ],
                "disallow_ring_reason": str(cand.disallow_ring_reason),
                "bridge_connector_id": int(cand.bridge_connector_id),
                "bridge_connector_su": int(cand.bridge_connector_su),
                "bridge_left_gid": int(cand.bridge_left_gid),
                "bridge_right_gid": int(cand.bridge_right_gid),
                "bridge_left_abcd_type": str(cand.bridge_left_abcd_type),
                "bridge_right_abcd_type": str(cand.bridge_right_abcd_type),
                "bridgehead_replace_24c_eligible": bool(cand.bridgehead_replace_24c_eligible),
                "suppresses_local_candidate_keys": list(cand.suppresses_local_candidate_keys),
                "priority_rank": int(cand.priority_rank),
                **_candidate_semantics(cand),
                "connector_expansions": [
                    self._connector_expansion_row(exp)
                    for exp in list(cand.connector_expansions or [])
                ],
            }
            for cand in alts
        ]
        meta = {
            "allocator_mode": "raw_motif_v2",
            "pending_closure": True,
            "closure_stage_required": True,
            "topology_state": "branch_candidate_group",
            "template": "branch24_candidate_group",
            "candidate_group_id": str(group_id),
            "branch_type": str(primary.abcd_type),
            "native24_class": str(primary.native24_class),
            "analog24_class": str(primary.analog24_class),
            "branch_display_class": str(primary.branch_display_class),
            "branch24_profiles": profiles,
            "branch_candidate_alternatives": candidate_rows,
            "like24_types": sorted(set(str(c.abcd_type) for c in alts)),
            "like24_labels": sorted(set(str(c.center_label) for c in alts)),
            "pattern_rules": sorted(set(str(c.pattern_rule) for c in alts if str(c.pattern_rule))),
            "assembly_families": sorted(set(str(c.assembly_family) for c in alts if str(c.assembly_family))),
            "assembly_shapes": sorted(set(str(c.assembly_shape) for c in alts if str(c.assembly_shape))),
            "usage_modes": sorted(set(str(mode) for c in alts for mode in list(c.usage_modes or []))),
            "ring_eligible": any(bool(c.ring_eligible) for c in alts),
            "ring_scopes": sorted(set(str(c.ring_scope) for c in alts if str(c.ring_scope))),
            "exclusive_group_keys": sorted(set(str(c.exclusive_group_key or c.group_id) for c in alts)),
            "bridge_connector_ids": sorted(set(int(c.bridge_connector_id) for c in alts if int(c.bridge_connector_id) >= 0)),
            "bridge_connector_su": sorted(set(int(c.bridge_connector_su) for c in alts if int(c.bridge_connector_su) >= 0)),
            "bridge_abcd_pairs": sorted(set(
                f"{c.bridge_left_abcd_type}-{c.bridge_right_abcd_type}"
                for c in alts
                if str(c.bridge_left_abcd_type) and str(c.bridge_right_abcd_type)
            )),
            "suppresses_local_candidate_keys": sorted(set(
                str(key) for c in alts for key in list(c.suppresses_local_candidate_keys or [])
            )),
            "priority_rank": max([int(c.priority_rank) for c in alts] or [0]),
            "node_ids": source_ids,
            "source_kinds": sorted(set(str(c.source_kind) for c in alts)),
            "tail_sources": [str(c.origin_rule) for c in alts],
            "main_body_su": [int(x) for x in primary.main_body_su],
            "branch_slots_su": {
                str(k): [int(x) for x in list(v or [])]
                for k, v in dict(primary.branch_slots_su or {}).items()
            },
            "branch_slot_labels": {
                str(k): str(v)
                for k, v in dict(primary.branch_slot_labels or {}).items()
            },
            "fixed_22_is_abstract": bool(primary.fixed_22_is_abstract),
            "evidence_ids": [int(x) for x in list(primary.evidence_ids or []) if int(x) >= 0],
            "abstracted_neighbors": [
                dict(row) for row in list(primary.abstracted_neighbors or [])
                if isinstance(row, dict)
            ],
            "actual_main_path_su": [int(x) for x in primary.actual_main_path_su],
            "actual_main_path_labels": list(primary.actual_main_path_labels),
            "actual_used_su": [int(x) for x in primary.actual_used_su],
            "actual_used_source_ids": [int(x) for x in primary.actual_used_source_ids],
            "chain_port_signature": [int(x) for x in primary.chain_port_signature],
            "display_topology": str(primary.open_topology),
            "raw_topology": str(primary.open_topology),
            "open_topology": str(primary.open_topology),
            "open_ends": list(primary.open_ends),
            "closure_intents": {
                str(cand.origin_rule): {
                    str(k): list(v) for k, v in dict(cand.closure_intents or {}).items()
                }
                for cand in alts
            },
        }
        comp = []
        if primary.abcd_type in {"24_A", "24_B"}:
            comp.append(11)
        comp.extend(int(x) for x in primary.main_body_su)
        comp.extend(int(x) for x in primary.branch_su)
        if str(primary.source_kind) == "native24":
            origin_type = f"Pending-native24({primary.abcd_type})"
        else:
            origin_type = f"Pending-like24({primary.source_kind}:{primary.abcd_type})"
        return ChainSpec(
            chain_type="pending_closure",
            composition=comp,
            origin_type=origin_type,
            source_ids=source_ids,
            metadata=meta,
        )

    @staticmethod
    def _connector_expansion_row(exp: _FixedConnectorExpansion) -> Dict[str, Any]:
        return {
            "kind": str(exp.kind),
            "connector_su": int(exp.connector.su) if exp.connector is not None else None,
            "connector_gid": int(exp.connector.gid) if exp.connector is not None else None,
            "path_su": [int(x) for x in list(exp.path_su or [])],
            "path_labels": list(exp.path_labels or []),
            "actual_path_su": [int(x) for x in list(exp.actual_path_su or exp.path_su or [])],
            "actual_path_labels": list(exp.actual_path_labels or exp.path_labels or []),
            "actual_node_ids": [int(x) for x in list(exp.actual_node_ids or exp.node_ids or [])],
            "abstracted_neighbors": [
                dict(row) for row in list(exp.abstracted_neighbors or [])
                if isinstance(row, dict)
            ],
            "node_ids": [int(x) for x in list(exp.node_ids or [])],
            "open_state": str(exp.open_state),
            "display_topology": str(exp.display_topology),
            "origin_rule": str(exp.origin_rule),
        }

    @staticmethod
    def _branch_intent_options(closure_intents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        opts: List[Dict[str, Any]] = []
        for site, intents in dict(closure_intents or {}).items():
            for intent in list(intents or []):
                opts.append({
                    "kind": "closure_intent",
                    "site": str(site),
                    "intent": str(intent),
                    "allowed_tails": [
                        [int(x) for x in tail]
                        for tail in BRANCH_CLOSURE_INTENTS.get(str(intent), [])
                    ],
                    "deferred": True,
                })
        return opts

    @staticmethod
    def _branch_candidate_open_ends(abcd_type: str, open_topology: str) -> List[str]:
        text = str(open_topology)
        if str(abcd_type) in {"24_A", "24_B"}:
            return ["right"] if text.endswith("-") else []
        ends: List[str] = []
        if text.startswith("-") or text.startswith("?-"):
            ends.append("left")
        if text.endswith("-"):
            ends.append("right")
        return ends or ["left", "right"]

    @staticmethod
    def _match_branch_roles(
        states: Sequence[Any],
        role_predicates: Dict[str, Callable[[Any], bool]],
    ) -> Optional[Dict[str, Any]]:
        vals = list(states or [])
        roles = list(dict(role_predicates or {}).items())
        if len(vals) != len(roles):
            return None

        used: set[int] = set()
        assigned: Dict[str, Any] = {}

        def backtrack(role_idx: int) -> bool:
            if role_idx >= len(roles):
                return True
            role, pred = roles[role_idx]
            for idx, st in enumerate(vals):
                if idx in used:
                    continue
                try:
                    ok = bool(pred(st))
                except Exception:
                    ok = False
                if not ok:
                    continue
                used.add(idx)
                assigned[str(role)] = st
                if backtrack(role_idx + 1):
                    return True
                used.remove(idx)
                assigned.pop(str(role), None)
            return False

        return dict(assigned) if backtrack(0) else None

    @staticmethod
    def _is_aromatic_11(state: Any) -> bool:
        return int(state.su) == 11

    @staticmethod
    def _native24_terminal(state: Any) -> bool:
        return int(state.su) in {1, 4, 22}

    @staticmethod
    def _base_terminal_1_or_22(state: Any) -> bool:
        return int(state.su) in {1, 22}

    def _expand_d3_fixed_connector(
        self,
        center: Any,
        connector_node: SUNode,
        connector: Any,
        skip: set[int],
    ) -> List[_FixedConnectorExpansion]:
        su = int(center.su)
        c = int(connector.su)
        connector_neighbors = [
            st for _nb, st in self._neighbor_states(connector_node)
        ]
        if has_forbidden_fixed_connector_special_combo(connector, connector_neighbors):
            return []
        others = [
            st
            for nb, st in self._neighbor_states(connector_node)
            if int(st.gid) != int(center.gid) and int(nb.global_id) not in skip
        ]
        if not others:
            return []

        expansions: List[_FixedConnectorExpansion] = []
        for other in sorted(list(others or []), key=lambda st: (self._fixed_connector_kind(su, c, st), int(st.su), str(st.label), int(st.gid))):
            if not self._valid_fixed_connector_other(su, c, other):
                continue
            kind = self._fixed_connector_kind(su, c, other)
            open_state = {"M": "closed", "N": "chain", "P": "terminal"}.get(kind, "chain")
            other_for_path, abstracted = self._connector_other_for_topology_path(c, kind, other)
            display = f"-{center.label}-{connector.label}-{other_for_path.label}"
            if open_state == "chain":
                display += "-"
            node_ids = [int(connector.gid)]
            if not abstracted or any(
                str(row.get("reason", "")) == "double_d3_fixed_connector_terminal22_to_23_like_chain_port"
                for row in list(abstracted or [])
                if isinstance(row, dict)
            ):
                node_ids.append(int(other.gid))
            expansions.append(_FixedConnectorExpansion(
                kind=kind,
                connector=connector,
                path_states=[connector, other_for_path],
                path_su=[c, int(other_for_path.su)],
                path_labels=[str(connector.label), str(other_for_path.label)],
                node_ids=node_ids,
                open_state=open_state,
                display_topology=display,
                origin_rule=f"{kind}:{center.label}-{connector.label}-{other.label}",
                actual_path_su=[c, int(other.su)],
                actual_path_labels=[str(connector.label), str(other.label)],
                actual_node_ids=[int(connector.gid), int(other.gid)],
                abstracted_neighbors=abstracted,
            ))
        return expansions

    def _connector_other_for_topology_path(
        self,
        connector_su: int,
        kind: str,
        other: Any,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        if str(kind) != "N":
            return other, []
        if int(connector_su) == 27 and is_single_special_d2(other, 20):
            return other, []
        if int(connector_su) in {29, 31} and is_single_special_d2(other, 19):
            return other, []
        if int(other.su) in {19, 20, 21} and int(other.degree or 0) in {2, 3}:
            return other, []
        if int(other.su) == 22 and int(connector_su) in {27, 29, 31}:
            row = {
                "path_index": 1,
                "actual_gid": int(getattr(other, "gid", -1)),
                "actual_su": int(getattr(other, "su", -1)),
                "actual_label": str(getattr(other, "label", "")),
                "topology_su": 23,
                "reason": "double_d3_fixed_connector_terminal22_to_23_like_chain_port",
            }
            return self._proxy_state(23), [row]
        if is_abstract_chain_port_state(other):
            row = {
                "path_index": 1,
                "actual_gid": int(getattr(other, "gid", -1)),
                "actual_su": int(getattr(other, "su", -1)),
                "actual_label": str(getattr(other, "label", "")),
                "topology_su": 23,
                "reason": "fixed_connector_N_generic_chain_port_to_23",
            }
            return self._proxy_state(23), [row]
        return other, []

    def _valid_fixed_connector_other(self, su: int, connector_su: int, other: Any) -> bool:
        o = int(other.su)
        if int(su) == 20 and int(connector_su) == 0:
            return o in {9, 22} or self._d3_connector_chain_port(other)
        if int(su) == 19 and int(connector_su) == 2:
            return o in {9, 22} or self._d3_connector_chain_port(other)
        if int(su) == 20 and int(connector_su) == 27:
            return (
                o == 6
                or (o == 20 and int(other.degree or 0) == 1)
                or self._d3_connector_chain_port(other)
            )
        if int(su) == 19 and int(connector_su) == 29:
            return (
                o == 5
                or (o == 19 and int(other.degree or 0) == 1)
                or self._d3_connector_chain_port(other)
            )
        if int(su) == 19 and int(connector_su) == 31:
            return (
                o == 7
                or (o == 19 and int(other.degree or 0) == 1)
                or self._d3_connector_chain_port(other)
            )
        return True

    def _d3_connector_chain_port(self, state: Any) -> bool:
        if int(state.su) in {0, 2, 3, 11, 14, 15, 17, 22, 23, 24, 25}:
            return True
        return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}

    def _fixed_connector_kind(self, su: int, connector_su: int, other: Any) -> str:
        o = int(other.su)
        if (connector_su, o) in {(0, 9), (2, 9), (27, 6), (29, 5), (31, 7)}:
            return "M"
        if su == 20 and connector_su == 27 and o == 20 and int(other.degree or 0) == 1:
            return "P"
        if su == 19 and connector_su in {29, 31} and o == 19 and int(other.degree or 0) == 1:
            return "P"
        if connector_su in {0, 2} and o == 22:
            return "P"
        return "N"

    @staticmethod
    def _states_from_expansion(exp: _FixedConnectorExpansion) -> List[Any]:
        return [
            st
            for st in list(getattr(exp, "path_states", []) or [])
        ]

    @staticmethod
    def _direct_path_expansion(kind: str, connector: Any, states: Sequence[Any], display: str) -> _FixedConnectorExpansion:
        vals = list(states or [])
        return _FixedConnectorExpansion(
            kind=str(kind),
            connector=connector,
            path_states=list(vals),
            path_su=[int(st.su) for st in vals],
            path_labels=[str(st.label) for st in vals],
            node_ids=[int(st.gid) for st in vals],
            open_state="chain" if str(kind) == "N" else ("terminal" if str(kind) == "P" else "closed"),
            display_topology=str(display),
            origin_rule=f"{kind}:{display}",
        )

    def _dedupe_branch_candidates(
        self,
        candidates: Sequence[_BranchTopologyCandidate],
    ) -> List[_BranchTopologyCandidate]:
        seen = set()
        out: List[_BranchTopologyCandidate] = []
        for cand in list(candidates or []):
            sig = (
                str(cand.group_id),
                str(cand.abcd_type),
                str(cand.fixed_role),
                tuple(int(x) for x in cand.main_body_su),
                tuple(int(x) for x in cand.branch_su),
                str(cand.open_topology),
                tuple(int(x) for x in cand.source_ids),
                str(cand.assembly_family),
                str(cand.assembly_shape),
                str(cand.main_connector_kind),
                str(cand.branch_connector_kind),
                tuple(int(x) for x in list(cand.main_connector_ids or [])),
                tuple(int(x) for x in list(cand.branch_connector_ids or [])),
            )
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cand)
        return out

    @staticmethod
    def _branch_group_id(center: Any) -> str:
        return f"branch24:{int(center.gid)}"

    @staticmethod
    def _special_d3_source_kind(center: Any) -> str:
        su = int(center.su)
        if su == 21:
            return "21d3"
        if is_double_special_d3(center, su):
            return f"double{su}d3"
        return f"single{su}d3"

    @staticmethod
    def _is_double_d3_center(center: Any, su: int) -> bool:
        return int(su) in {19, 20} and is_double_special_d3(center, su)

    @staticmethod
    def _special_d3_connector(su: int, state: Any) -> bool:
        if int(su) == 19:
            return int(state.su) in {2, 29, 31}
        if int(su) == 20:
            return int(state.su) in {0, 27}
        return False

    @staticmethod
    def _special_d3_direct_terminal(center: Any, state: Any) -> bool:
        if int(center.su) == 19:
            return int(state.su) == 28
        if int(center.su) == 21:
            return int(state.su) == 32
        return False

    @staticmethod
    def _proxy_state(su: int) -> Any:
        return state_for_node(SUNode(global_id=-2000000 - int(su), su_type=int(su)))

    @staticmethod
    def _branch_abstracted_neighbor_row(state: Any, reason: str, topology_su: int = 23) -> Dict[str, Any]:
        return {
            "actual_gid": int(getattr(state, "gid", -1)),
            "actual_su": int(getattr(state, "su", -1)),
            "actual_label": str(getattr(state, "label", "")),
            "topology_su": int(topology_su),
            "reason": str(reason),
        }

    def _branch_chain_port(self, state: Any) -> bool:
        if int(state.su) in {0, 2, 3, 14, 15, 17, 23, 24, 25}:
            return True
        return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}

    def _double_d3_abstract_port(self, state: Any) -> bool:
        if int(state.su) in {0, 1, 2, 3, 14, 15, 17, 22, 23, 24, 25}:
            return True
        return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}

    def _branch_24ab_chain_port(self, state: Any) -> bool:
        return int(state.su) == 11 or self._branch_chain_port(state)

    def _su25_abstract_port(self, state: Any) -> bool:
        if int(state.su) in {0, 1, 2, 3, 11, 14, 15, 17, 22, 23, 24, 25}:
            return True
        return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}
