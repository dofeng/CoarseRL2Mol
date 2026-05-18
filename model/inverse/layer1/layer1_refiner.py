import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict

from ...paths import Z_LIBRARY_DIR
from ..hop1_adjuster import Hop1Adjuster
from ...shared.inverse_common import HOP1_PORT_COMBINATIONS, _NodeV3, check_external_connection_requirement, validate_connection
from .layer1_eval import Layer1NmrEvaluator


class Layer1Refiner:
    def __init__(self,
                 device: str | torch.device,
                 evaluator: Layer1NmrEvaluator):
        self.device = device
        self.evaluator = evaluator
        self._su_common_range_path = str(Z_LIBRARY_DIR / 'su_nmr_common_range_filtered.csv')
        self._su_common_ranges_cache: Optional[Dict[int, Dict[str, float]]] = None

    @staticmethod
    def _capture_hop1_state(nodes: List[_NodeV3]) -> List[Tuple[Counter, List[int], set]]:
        return [
            (
                Counter(getattr(node, 'hop1_su', {}) or {}),
                list(getattr(node, 'hop1_ids', []) or []),
                set(getattr(node, 'fixed_hop1_ids', set()) or set()),
            )
            for node in nodes
        ]

    @staticmethod
    def _restore_hop1_state(nodes: List[_NodeV3], state: List[Tuple[Counter, List[int], set]]) -> None:
        for node, (hop1_su, hop1_ids, fixed_ids) in zip(nodes, state):
            node.hop1_su = Counter(hop1_su)
            node.hop1_ids = list(hop1_ids)
            try:
                node.fixed_hop1_ids = set(fixed_ids)
            except Exception:
                pass

    @staticmethod
    def _node_target_degree(node: _NodeV3) -> Optional[int]:
        try:
            val = getattr(node, 'target_hop1_degree', None)
            return int(val) if val is not None else None
        except Exception:
            return None

    @staticmethod
    def _node_anchor_partition(node: _NodeV3) -> Optional[str]:
        try:
            val = getattr(node, 'special_anchor_partition', None)
            return str(val) if val is not None else None
        except Exception:
            return None

    def _required_external_candidates_for_node(self, node: _NodeV3) -> Optional[List[int]]:
        su_i = int(getattr(node, 'su_type', -1))
        if int(su_i) == 19:
            part = self._node_anchor_partition(node)
            deg = self._node_target_degree(node)
            if str(part) == 'thio':
                return [31]
            if str(part) == 'ether':
                if int(deg or 0) == 1:
                    return [2, 29]
                return [2, 28, 29]
            return None
        try:
            return [int(x) for x in list({
                5: [2, 28, 29],
                6: [0, 27],
                7: [31],
                8: [32],
                9: [0, 1, 2, 3],
                10: [4, 10],
                11: [23, 24, 25, 22, 19, 20, 21, 15, 17],
                20: [0, 27],
                21: [32],
            }.get(int(su_i), []) or [])]
        except Exception:
            return []

    def _required_external_priority(self, center: _NodeV3, required_external: List[int]) -> List[int]:
        center_su = int(getattr(center, 'su_type', -1))
        if int(center_su) == 11:
            preferred = [23, 22, 24, 25, 15, 17, 19, 20, 21]
            return [int(x) for x in preferred if int(x) in set(int(v) for v in required_external)]
        return [int(x) for x in list(required_external or [])]

    @staticmethod
    def _should_lock_required_external_edge(center: _NodeV3) -> bool:
        return int(getattr(center, 'su_type', -1)) in {5, 6, 7, 8, 9, 19, 20, 21}

    def _required_external_gap(self,
                               assigner: Any,
                               node: _NodeV3,
                               nodes: List[_NodeV3]) -> int:
        required_external = assigner._required_external_candidates_for_node(node)
        if not required_external:
            return 0
        try:
            target_count = max(1, int(assigner._node_fixed_anchor_target(node)))
        except Exception:
            target_count = 1
        try:
            current_count = int(assigner._required_external_count_for_node(node, nodes))
        except Exception:
            current_count = 0
        gap = max(0, int(target_count) - int(current_count))
        return min(int(gap), int(node.remaining_hop1_slots()))

    def _required_external_candidate_nodes(self,
                                           assigner: Any,
                                           nodes: List[_NodeV3],
                                           center: _NodeV3,
                                           allow_pending_special: bool) -> List[_NodeV3]:
        required_external = assigner._required_external_candidates_for_node(center)
        if not required_external:
            return []
        priority = self._required_external_priority(center, required_external)
        candidate_pool: List[_NodeV3] = []
        seen_ids: Set[int] = set()
        for ext_su in priority:
            for cand in assigner._get_nodes_by_su_type(nodes, int(ext_su)):
                gid = int(getattr(cand, 'global_id', -1))
                if int(gid) in seen_ids:
                    continue
                seen_ids.add(int(gid))
                if int(gid) == int(getattr(center, 'global_id', -1)):
                    continue
                if int(gid) in set(getattr(center, 'hop1_ids', []) or []):
                    continue
                if int(cand.remaining_hop1_slots()) <= 0:
                    continue
                if (not bool(allow_pending_special)) and assigner._has_pending_required_external(cand, nodes):
                    continue
                candidate_pool.append(cand)
        return list(candidate_pool)

    def _pick_required_external_candidate(self,
                                          assigner: Any,
                                          nodes: List[_NodeV3],
                                          center: _NodeV3,
                                          allow_pending_special: bool) -> Optional[_NodeV3]:
        required_external = assigner._required_external_candidates_for_node(center)
        if not required_external:
            return None
        priority = self._required_external_priority(center, required_external)
        candidate_pool = self._required_external_candidate_nodes(
            assigner=assigner,
            nodes=nodes,
            center=center,
            allow_pending_special=bool(allow_pending_special),
        )
        return assigner._pick_fixed_target(nodes, center, candidate_pool, priority)

    def _complete_required_external_exact(self,
                                          assigner: Any,
                                          nodes: List[_NodeV3],
                                          su_type: int,
                                          allow_pending_special: bool) -> int:
        centers = [
            n for n in list(nodes or [])
            if int(getattr(n, 'su_type', -1)) == int(su_type)
            and int(n.remaining_hop1_slots()) > 0
            and int(self._required_external_gap(assigner, n, nodes)) > 0
        ]
        if not centers:
            return 0

        demands: List[Dict[str, Any]] = []
        owner_types: Set[int] = set()
        for center in centers:
            required_external = assigner._required_external_candidates_for_node(center)
            if not required_external:
                continue
            priority = self._required_external_priority(center, required_external)
            gap = int(self._required_external_gap(assigner, center, nodes))
            for demand_idx in range(max(0, gap)):
                demands.append({
                    'target_id': int(getattr(center, 'global_id', -1)),
                    'demand_index': int(demand_idx),
                    'required': set(int(x) for x in list(required_external or [])),
                    'priority': list(priority),
                    'priority_rank': {int(su): int(idx) for idx, su in enumerate(priority)},
                    'lock_edge': bool(self._should_lock_required_external_edge(center)),
                })
            owner_types.update(int(x) for x in list(priority or []))
        if not demands:
            return 0

        owner_slots: List[Dict[str, int]] = []
        for owner_su in sorted(int(x) for x in owner_types):
            for owner in assigner._get_nodes_by_su_type(nodes, int(owner_su)):
                if int(owner.remaining_hop1_slots()) <= 0:
                    continue
                if (not bool(allow_pending_special)) and assigner._has_pending_required_external(owner, nodes):
                    continue
                for slot_idx in range(max(0, int(owner.remaining_hop1_slots()))):
                    owner_slots.append({
                        'owner_id': int(getattr(owner, 'global_id', -1)),
                        'slot_index': int(slot_idx),
                        'su_type': int(owner_su),
                    })
        if not owner_slots:
            return 0

        work_nodes = copy.deepcopy(nodes)
        used_slots = [False] * len(owner_slots)
        owner_d1_usage: Dict[int, int] = defaultdict(int)
        chosen_edges: List[Tuple[int, int, bool]] = []

        def _candidate_slots(demand_idx: int) -> List[int]:
            demand = dict(demands[int(demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            target_deg = int(assigner._node_target_degree(target) or 0)
            out: List[int] = []
            for slot_idx, slot in enumerate(owner_slots):
                if bool(used_slots[int(slot_idx)]):
                    continue
                owner = work_nodes[int(slot['owner_id'])]
                owner_su = int(getattr(owner, 'su_type', -1))
                if int(owner_su) not in set(int(x) for x in demand.get('required', set()) or set()):
                    continue
                if int(owner.global_id) == int(target.global_id):
                    continue
                if int(owner.global_id) in set(getattr(target, 'hop1_ids', []) or []):
                    continue
                if int(owner.remaining_hop1_slots()) <= 0 or int(target.remaining_hop1_slots()) <= 0:
                    continue
                if (not bool(allow_pending_special)) and assigner._has_pending_required_external(owner, work_nodes):
                    continue
                if int(owner_su) == 29 and int(getattr(target, 'su_type', -1)) == 19 and int(target_deg) == 1:
                    if int(owner_d1_usage.get(int(owner.global_id), 0)) >= 1:
                        continue
                if not assigner._can_add_hop1_connection(work_nodes, owner, target):
                    continue
                out.append(int(slot_idx))
            out.sort(
                key=lambda idx: (
                    int(demand.get('priority_rank', {}).get(int(owner_slots[int(idx)]['su_type']), len(demand.get('priority', [])))),
                    -int(work_nodes[int(owner_slots[int(idx)]['owner_id'])].remaining_hop1_slots()),
                    int(owner_slots[int(idx)]['owner_id']),
                    int(owner_slots[int(idx)]['slot_index']),
                )
            )
            return out

        def _demand_key(demand_idx: int, options: List[int]) -> Tuple[int, int, int, int, int, int]:
            demand = dict(demands[int(demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            su_i = int(getattr(target, 'su_type', -1))
            target_deg = int(assigner._node_target_degree(target) or 0)
            fixed_target = int(assigner._node_fixed_anchor_target(target))
            return (
                int(len(options)),
                0 if int(su_i) in {19, 20, 21, 5, 6, 7, 8, 9} else 1,
                -int(fixed_target),
                int(target_deg),
                int(getattr(target, 'global_id', 0)),
                int(demand.get('demand_index', 0)),
            )

        def _dfs(remaining: List[int]) -> bool:
            if not remaining:
                return True

            best_idx: Optional[int] = None
            best_opts: Optional[List[int]] = None
            best_key: Optional[Tuple[int, int, int, int, int, int]] = None
            for demand_idx in list(remaining):
                opts = _candidate_slots(int(demand_idx))
                if not opts:
                    return False
                key = _demand_key(int(demand_idx), opts)
                if best_key is None or tuple(key) < tuple(best_key):
                    best_idx = int(demand_idx)
                    best_opts = list(opts)
                    best_key = tuple(key)
                    if int(len(best_opts)) <= 1:
                        break

            if best_idx is None or best_opts is None:
                return False

            next_remaining = [int(x) for x in remaining if int(x) != int(best_idx)]
            demand = dict(demands[int(best_idx)] or {})
            target_id = int(demand['target_id'])
            target = work_nodes[int(target_id)]
            target_deg = int(assigner._node_target_degree(target) or 0)

            for slot_idx in list(best_opts):
                owner_id = int(owner_slots[int(slot_idx)]['owner_id'])
                owner = work_nodes[int(owner_id)]
                if not assigner._add_bidirectional_hop1(work_nodes, int(owner_id), int(target_id), lock=False):
                    continue
                used_slots[int(slot_idx)] = True
                inc_d1 = (
                    int(getattr(owner, 'su_type', -1)) == 29 and
                    int(getattr(target, 'su_type', -1)) == 19 and
                    int(target_deg) == 1
                )
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] += 1
                chosen_edges.append((int(owner_id), int(target_id), bool(demand.get('lock_edge', False))))
                if _dfs(next_remaining):
                    return True
                chosen_edges.pop()
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] -= 1
                    if int(owner_d1_usage[int(owner_id)]) <= 0:
                        owner_d1_usage.pop(int(owner_id), None)
                assigner._remove_bidirectional_hop1_with_force(
                    work_nodes,
                    int(owner_id),
                    int(target_id),
                    force=True,
                )
                used_slots[int(slot_idx)] = False
            return False

        matched = 0
        if _dfs([int(i) for i in range(len(demands))]):
            for owner_id, target_id, lock_flag in list(chosen_edges):
                if int(owner_id) in set(getattr(nodes[int(target_id)], 'hop1_ids', []) or []):
                    continue
                if not assigner._add_bidirectional_hop1(
                    nodes,
                    int(owner_id),
                    int(target_id),
                    lock=bool(lock_flag),
                ):
                    continue
                matched += 1
            return int(matched)
        return 0

    def _complete_required_external_greedy_mrv(self,
                                               assigner: Any,
                                               nodes: List[_NodeV3],
                                               su_type: int,
                                               allow_pending_special: bool,
                                               max_steps: Optional[int] = None) -> int:
        applied = 0
        limit = int(max_steps) if max_steps is not None else max(8, len(nodes) * 2)
        steps = 0
        while steps < int(limit):
            steps += 1
            centers = [
                n for n in list(nodes or [])
                if int(getattr(n, 'su_type', -1)) == int(su_type)
                and int(n.remaining_hop1_slots()) > 0
                and int(self._required_external_gap(assigner, n, nodes)) > 0
            ]
            if not centers:
                break

            center_choices: List[Tuple[int, int, int, _NodeV3, Optional[_NodeV3]]] = []
            for center in centers:
                candidate_pool = self._required_external_candidate_nodes(
                    assigner=assigner,
                    nodes=nodes,
                    center=center,
                    allow_pending_special=bool(allow_pending_special),
                )
                candidate = self._pick_required_external_candidate(
                    assigner=assigner,
                    nodes=nodes,
                    center=center,
                    allow_pending_special=bool(allow_pending_special),
                )
                if candidate is None:
                    continue
                center_choices.append((
                    int(len(candidate_pool)),
                    -int(self._required_external_gap(assigner, center, nodes)),
                    int(getattr(center, 'global_id', 0)),
                    center,
                    candidate,
                ))
            if not center_choices:
                break

            center_choices.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))
            _, _, _, center, candidate = center_choices[0]
            if not assigner._add_bidirectional_hop1(
                nodes,
                int(getattr(center, 'global_id', -1)),
                int(getattr(candidate, 'global_id', -1)),
                lock=bool(self._should_lock_required_external_edge(center)),
            ):
                continue
            applied += 1
        return int(applied)

    def complete_required_external_anchors(self,
                                           assigner: Any,
                                           nodes: List[_NodeV3],
                                           su_types: List[int],
                                           max_rounds: int = 4) -> None:
        """
        Stronger version of required-anchor filling.

        We retry in multiple rounds and, for SU11, prefer cheaper external
        aliphatic/unsaturated donors before consuming scarcer 19/20/21 nodes.
        """
        target_types = [int(x) for x in list(su_types or [])]
        special_exact_types = {5, 6, 7, 8, 19, 20, 21}
        for round_idx in range(max(1, int(max_rounds))):
            progressed = False
            ordered_types = sorted(
                set(int(x) for x in target_types),
                key=lambda su: (0 if int(su) in special_exact_types else 1, int(su)),
            )
            for allow_pending_special in (False, True):
                for su_type in ordered_types:
                    delta = 0
                    if int(su_type) in special_exact_types:
                        delta = self._complete_required_external_exact(
                            assigner=assigner,
                            nodes=nodes,
                            su_type=int(su_type),
                            allow_pending_special=bool(allow_pending_special),
                        )
                    if int(delta) <= 0:
                        delta = self._complete_required_external_greedy_mrv(
                            assigner=assigner,
                            nodes=nodes,
                            su_type=int(su_type),
                            allow_pending_special=bool(allow_pending_special),
                            max_steps=max(4, len(target_types) * 2),
                        )
                    if int(delta) > 0:
                        progressed = True
            if not bool(progressed):
                break

    def repair_remaining_hop1_slots(self,
                                    assigner: Any,
                                    nodes: List[_NodeV3]) -> None:
        """
        Final structural repair pass for Layer1.

        Compared with the original version, this pass:
        - prioritizes nodes missing required external anchors,
        - retries other unfinished nodes instead of stopping at the first hard case,
        - re-runs required-anchor filling between repair rounds.
        """
        remaining = [n for n in nodes if n.remaining_hop1_slots() > 0]
        if not remaining:
            return

        max_iters = len(remaining) * 20
        iters = 0

        def _can_direct_connect(u: _NodeV3, v: _NodeV3) -> bool:
            if u.global_id == v.global_id:
                return False
            if v.global_id in u.hop1_ids:
                return False
            if u.remaining_hop1_slots() <= 0 or v.remaining_hop1_slots() <= 0:
                return False
            return assigner._can_add_hop1_connection(nodes, u, v)

        def _is_locked(n: _NodeV3) -> bool:
            return int(getattr(n, 'su_type', -1)) in {27, 28, 29, 31, 32}

        def _has_pending_external(n: _NodeV3, graph_nodes: List[_NodeV3]) -> bool:
            return bool(assigner._has_pending_required_external(n, graph_nodes))

        def _make_local_work_nodes(src_nodes: List[_NodeV3], affected_ids: List[int]) -> List[_NodeV3]:
            work_nodes = list(src_nodes)
            for nid in sorted(set(int(x) for x in affected_ids)):
                if 0 <= int(nid) < len(src_nodes):
                    work_nodes[int(nid)] = copy.deepcopy(src_nodes[int(nid)])
            return work_nodes

        def _repair_priority(node: _NodeV3) -> Tuple[int, int, int, int, int]:
            current_deg = int(node.get_hop1_degree())
            target_deg = assigner._node_target_degree(node)
            deficit = int(node.remaining_hop1_slots())
            if target_deg is not None:
                deficit = max(int(deficit), max(0, int(target_deg) - int(current_deg)))
            missing_required = 1 if assigner._has_pending_required_external(node, nodes) else 0
            su_i = int(getattr(node, 'su_type', -1))
            su11_missing_external = int(su_i == 11 and missing_required > 0)
            special_missing = int(su_i in {19, 20, 21} and missing_required > 0)
            return (
                -int(su11_missing_external),
                -int(special_missing),
                -int(missing_required),
                -int(deficit),
                int(getattr(node, 'global_id', 0)),
            )

        def _node_degree_gap(node: _NodeV3, graph_nodes: List[_NodeV3]) -> int:
            _ = graph_nodes
            current_deg = int(node.get_hop1_degree())
            target_deg = assigner._node_target_degree(node)
            if target_deg is not None:
                return max(0, int(target_deg) - int(current_deg))
            return max(0, int(node.remaining_hop1_slots()))

        def _node_penalty(node: _NodeV3, graph_nodes: List[_NodeV3]) -> int:
            su_i = int(getattr(node, 'su_type', -1))
            required_gap = int(self._required_external_gap(assigner, node, graph_nodes))
            degree_gap = int(_node_degree_gap(node, graph_nodes))
            rem_slots = int(node.remaining_hop1_slots())
            special_weight = 120 if int(su_i) in {19, 20, 21, 5, 6, 7, 8, 9} else 80 if int(su_i) == 11 else 18
            degree_weight = 14 if int(su_i) in {19, 20, 21, 11, 5, 6, 7, 8, 9} else 4
            return (
                int(required_gap) * int(special_weight) +
                int(degree_gap) * int(degree_weight) +
                int(rem_slots)
            )

        def _score_nodes(graph_nodes: List[_NodeV3], node_ids: List[int]) -> int:
            uniq_ids = sorted(set(int(x) for x in list(node_ids or [])))
            total = 0
            for nid in uniq_ids:
                if 0 <= int(nid) < len(graph_nodes):
                    total += int(_node_penalty(graph_nodes[int(nid)], graph_nodes))
            return int(total)

        def _try_fill_pending_external(center: _NodeV3) -> bool:
            if int(center.remaining_hop1_slots()) <= 0:
                return False
            if not _has_pending_external(center, nodes):
                return False

            for allow_pending_special in (False, True):
                target = self._pick_required_external_candidate(
                    assigner=assigner,
                    nodes=nodes,
                    center=center,
                    allow_pending_special=bool(allow_pending_special),
                )
                if target is not None:
                    if not assigner._add_bidirectional_hop1(
                        nodes,
                        int(center.global_id),
                        int(target.global_id),
                        lock=bool(self._should_lock_required_external_edge(center)),
                    ):
                        continue
                    return True

            if int(getattr(center, 'su_type', -1)) != 11:
                return False

            required_external = assigner._required_external_candidates_for_node(center) or []
            priority = self._required_external_priority(center, list(required_external))
            best_move: Optional[Tuple[int, int, int]] = None
            best_gain = 0
            ordinary_rewire_types = {15, 17, 22, 23, 24, 25}

            seen_owner_ids: Set[int] = set()
            for ext_su in priority:
                for owner in assigner._get_nodes_by_su_type(nodes, int(ext_su)):
                    owner_id = int(getattr(owner, 'global_id', -1))
                    if int(owner_id) in seen_owner_ids:
                        continue
                    seen_owner_ids.add(int(owner_id))
                    if int(owner_id) == int(center.global_id):
                        continue
                    if int(owner_id) in set(getattr(center, 'hop1_ids', []) or []):
                        continue
                    if int(getattr(owner, 'su_type', -1)) not in ordinary_rewire_types:
                        continue
                    if int(owner.remaining_hop1_slots()) > 0:
                        continue

                    for old_nb_id in list(getattr(owner, 'hop1_ids', []) or []):
                        old_nb_i = int(old_nb_id)
                        if old_nb_i < 0 or old_nb_i >= len(nodes):
                            continue
                        old_nb = nodes[int(old_nb_i)]
                        if _is_locked(old_nb) or _has_pending_external(old_nb, nodes):
                            continue
                        if assigner._edge_is_fixed(owner, old_nb):
                            continue

                        affected_ids = [int(center.global_id), int(owner_id), int(old_nb_i)]
                        before_score = _score_nodes(nodes, affected_ids)
                        work_nodes = _make_local_work_nodes(nodes, affected_ids)
                        if not assigner._remove_bidirectional_hop1(work_nodes, int(owner_id), int(old_nb_i)):
                            continue
                        center_w = work_nodes[int(center.global_id)]
                        owner_w = work_nodes[int(owner_id)]
                        if not assigner._can_add_hop1_connection(work_nodes, owner_w, center_w):
                            continue
                        if not assigner._add_bidirectional_hop1(work_nodes, int(owner_id), int(center.global_id), lock=False):
                            continue
                        after_score = _score_nodes(work_nodes, affected_ids)
                        gain = int(before_score - after_score)
                        if int(gain) <= 0:
                            continue
                        if int(self._required_external_gap(assigner, center_w, work_nodes)) >= int(self._required_external_gap(assigner, center, nodes)):
                            continue
                        if best_move is None or int(gain) > int(best_gain):
                            best_move = (int(owner_id), int(old_nb_i), int(center.global_id))
                            best_gain = int(gain)

            if best_move is None:
                return False

            owner_id, old_nb_id, center_id = best_move
            if not assigner._remove_bidirectional_hop1(nodes, int(owner_id), int(old_nb_id)):
                return False
            if not assigner._add_bidirectional_hop1(
                nodes,
                int(owner_id),
                int(center_id),
                lock=False,
            ):
                assigner._add_bidirectional_hop1(nodes, int(owner_id), int(old_nb_id), lock=False)
                return False
            return True

        def _try_rewire_for_special_degree_gap(center: _NodeV3) -> bool:
            """Borrow one ordinary non-fixed edge to close a special-anchor degree gap."""
            if int(getattr(center, 'su_type', -1)) not in {19, 20, 21}:
                return False
            if int(center.remaining_hop1_slots()) <= 0:
                return False
            if _has_pending_external(center, nodes):
                return False
            if int(_node_degree_gap(center, nodes)) <= 0:
                return False

            protected_su = {5, 6, 7, 8, 9, 19, 20, 21, 27, 28, 29, 31, 32}
            terminal_su = {1, 4, 16, 18, 22, 28, 32}
            best_move: Optional[Tuple[int, int, int]] = None
            best_gain: Optional[int] = None

            for owner in list(nodes or []):
                owner_id = int(getattr(owner, 'global_id', -1))
                if int(owner_id) == int(center.global_id):
                    continue
                if int(owner_id) in set(getattr(center, 'hop1_ids', []) or []):
                    continue
                if int(getattr(owner, 'su_type', -1)) in protected_su:
                    continue
                if int(owner.remaining_hop1_slots()) > 0:
                    continue
                if _has_pending_external(owner, nodes):
                    continue

                for old_nb_id in list(getattr(owner, 'hop1_ids', []) or []):
                    old_nb_i = int(old_nb_id)
                    if old_nb_i < 0 or old_nb_i >= len(nodes):
                        continue
                    old_nb = nodes[int(old_nb_i)]
                    if int(getattr(old_nb, 'su_type', -1)) in protected_su:
                        continue
                    if int(getattr(old_nb, 'su_type', -1)) in terminal_su:
                        continue
                    if _has_pending_external(old_nb, nodes):
                        continue
                    if assigner._edge_is_fixed(owner, old_nb):
                        continue

                    affected_ids = [int(center.global_id), int(owner_id), int(old_nb_i)]
                    before_score = _score_nodes(nodes, affected_ids)
                    work_nodes = _make_local_work_nodes(nodes, affected_ids)
                    if not assigner._remove_bidirectional_hop1(work_nodes, int(owner_id), int(old_nb_i)):
                        continue
                    owner_w = work_nodes[int(owner_id)]
                    center_w = work_nodes[int(center.global_id)]
                    old_nb_w = work_nodes[int(old_nb_i)]
                    if not assigner._can_add_hop1_connection(work_nodes, owner_w, center_w):
                        continue
                    if not assigner._add_bidirectional_hop1(work_nodes, int(owner_id), int(center.global_id), lock=False):
                        continue
                    if int(_node_degree_gap(center_w, work_nodes)) >= int(_node_degree_gap(center, nodes)):
                        continue
                    after_score = _score_nodes(work_nodes, affected_ids)
                    # Do not trade a special-anchor hard gap for another special
                    # or fixed-anchor gap; owner/old_nb are filtered above, so a
                    # positive local score gain is enough to make this safe.
                    gain = int(before_score - after_score)
                    if int(gain) <= 0:
                        continue
                    if best_gain is None or int(gain) > int(best_gain):
                        _ = old_nb_w
                        best_move = (int(owner_id), int(old_nb_i), int(center.global_id))
                        best_gain = int(gain)

            if best_move is None:
                return False

            owner_id, old_nb_id, center_id = best_move
            if not assigner._remove_bidirectional_hop1(nodes, int(owner_id), int(old_nb_id)):
                return False
            if not assigner._add_bidirectional_hop1(nodes, int(owner_id), int(center_id), lock=False):
                assigner._add_bidirectional_hop1(nodes, int(owner_id), int(old_nb_id), lock=False)
                return False
            return True

        while iters < max_iters:
            iters += 1
            self.complete_required_external_anchors(assigner, nodes, [11, 19, 20, 21, 5, 6, 7, 8, 9], max_rounds=1)
            remaining = [n for n in nodes if n.remaining_hop1_slots() > 0]
            if not remaining:
                break

            remaining.sort(key=_repair_priority)
            progressed = False

            for u in remaining:
                if _try_fill_pending_external(u):
                    progressed = True
                    break
                if _try_rewire_for_special_degree_gap(u):
                    progressed = True
                    break

                # 1) direct connect
                direct_candidates = [v for v in remaining if int(v.global_id) != int(u.global_id) and _can_direct_connect(u, v)]
                if direct_candidates:
                    best_direct: Optional[Tuple[int, int]] = None
                    best_gain = None
                    for v in list(direct_candidates):
                        affected_ids = [int(u.global_id), int(v.global_id)]
                        before_score = _score_nodes(nodes, affected_ids)
                        work_nodes = _make_local_work_nodes(nodes, affected_ids)
                        if not assigner._add_bidirectional_hop1(work_nodes, int(u.global_id), int(v.global_id), lock=False):
                            continue
                        after_score = _score_nodes(work_nodes, affected_ids)
                        gain = int(before_score - after_score)
                        if best_gain is None or int(gain) > int(best_gain):
                            best_gain = int(gain)
                            best_direct = (int(u.global_id), int(v.global_id))
                    if best_direct is not None and int(best_gain or 0) >= 0:
                        if not assigner._add_bidirectional_hop1(nodes, int(best_direct[0]), int(best_direct[1])):
                            continue
                        progressed = True
                        break

                # 2) split one edge into two edges centered on u
                if int(u.remaining_hop1_slots()) >= 2:
                    preferred_edge_types = {10, 11, 12, 13}
                    edge_endpoints = []
                    for a in nodes:
                        if _is_locked(a) or _has_pending_external(a, nodes):
                            continue
                        for b_id in list(getattr(a, 'hop1_ids', []) or []):
                            if int(b_id) <= int(a.global_id):
                                continue
                            b = nodes[int(b_id)]
                            if _is_locked(b) or _has_pending_external(b, nodes):
                                continue
                            edge_endpoints.append((a, b))

                    found_split = False
                    for only_preferred in (True, False):
                        for a, b in edge_endpoints:
                            if only_preferred and (int(a.su_type) not in preferred_edge_types or int(b.su_type) not in preferred_edge_types):
                                continue
                            if int(a.global_id) in set(getattr(u, 'hop1_ids', []) or []):
                                continue
                            if int(b.global_id) in set(getattr(u, 'hop1_ids', []) or []):
                                continue

                            work_nodes = _make_local_work_nodes(nodes, [int(u.global_id), int(a.global_id), int(b.global_id)])
                            before_score = _score_nodes(nodes, [int(u.global_id), int(a.global_id), int(b.global_id)])
                            ok = assigner._remove_bidirectional_hop1(work_nodes, int(a.global_id), int(b.global_id))
                            if not ok:
                                continue
                            u_w = work_nodes[int(u.global_id)]
                            a_w = work_nodes[int(a.global_id)]
                            b_w = work_nodes[int(b.global_id)]
                            if not assigner._can_add_hop1_connection(work_nodes, u_w, a_w):
                                continue
                            if not assigner._add_bidirectional_hop1(work_nodes, int(u.global_id), int(a.global_id)):
                                continue
                            u_w = work_nodes[int(u.global_id)]
                            if not assigner._can_add_hop1_connection(work_nodes, u_w, b_w):
                                continue
                            if not assigner._add_bidirectional_hop1(work_nodes, int(u.global_id), int(b.global_id), lock=False):
                                continue
                            after_score = _score_nodes(work_nodes, [int(u.global_id), int(a.global_id), int(b.global_id)])
                            if int(after_score) >= int(before_score):
                                continue

                            ok = assigner._remove_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                            if not ok:
                                continue
                            if not assigner._add_bidirectional_hop1(nodes, int(u.global_id), int(a.global_id)):
                                assigner._add_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                                continue
                            if not assigner._add_bidirectional_hop1(nodes, int(u.global_id), int(b.global_id)):
                                assigner._remove_bidirectional_hop1_with_force(nodes, int(u.global_id), int(a.global_id), force=True)
                                assigner._add_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                                continue
                            found_split = True
                            progressed = True
                            break
                        if found_split:
                            break
                    if found_split:
                        break

                # 3) two-edge swap
                found_swap = False
                for v in remaining:
                    if int(u.global_id) == int(v.global_id):
                        continue
                    if int(u.remaining_hop1_slots()) <= 0 or int(v.remaining_hop1_slots()) <= 0:
                        continue
                    for a in nodes:
                        if int(a.remaining_hop1_slots()) != 0:
                            continue
                        if _is_locked(a) or _has_pending_external(a, nodes):
                            continue
                        if int(a.global_id) in {int(u.global_id), int(v.global_id)}:
                            continue
                        if int(a.global_id) in set(getattr(u, 'hop1_ids', []) or []):
                            continue

                        for b_id in list(getattr(a, 'hop1_ids', []) or []):
                            b = nodes[int(b_id)]
                            if int(b.remaining_hop1_slots()) != 0:
                                continue
                            if _is_locked(b) or _has_pending_external(b, nodes):
                                continue
                            if int(b.global_id) in {int(u.global_id), int(v.global_id)}:
                                continue
                            if int(b.global_id) in set(getattr(v, 'hop1_ids', []) or []):
                                continue

                            work_nodes = _make_local_work_nodes(nodes, [int(u.global_id), int(v.global_id), int(a.global_id), int(b.global_id)])
                            before_score = _score_nodes(nodes, [int(u.global_id), int(v.global_id), int(a.global_id), int(b.global_id)])
                            ok = assigner._remove_bidirectional_hop1(work_nodes, int(a.global_id), int(b.global_id))
                            if not ok:
                                continue
                            u_w = work_nodes[int(u.global_id)]
                            v_w = work_nodes[int(v.global_id)]
                            a_w = work_nodes[int(a.global_id)]
                            b_w = work_nodes[int(b.global_id)]
                            if not assigner._can_add_hop1_connection(work_nodes, u_w, a_w):
                                continue
                            if not assigner._add_bidirectional_hop1(work_nodes, int(u.global_id), int(a.global_id)):
                                continue
                            if not assigner._can_add_hop1_connection(work_nodes, v_w, b_w):
                                continue
                            if not assigner._add_bidirectional_hop1(work_nodes, int(v.global_id), int(b.global_id), lock=False):
                                continue
                            after_score = _score_nodes(work_nodes, [int(u.global_id), int(v.global_id), int(a.global_id), int(b.global_id)])
                            if int(after_score) >= int(before_score):
                                continue

                            ok = assigner._remove_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                            if not ok:
                                continue
                            if not assigner._add_bidirectional_hop1(nodes, int(u.global_id), int(a.global_id)):
                                assigner._add_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                                continue
                            if not assigner._add_bidirectional_hop1(nodes, int(v.global_id), int(b.global_id)):
                                assigner._remove_bidirectional_hop1_with_force(nodes, int(u.global_id), int(a.global_id), force=True)
                                assigner._add_bidirectional_hop1(nodes, int(a.global_id), int(b.global_id))
                                continue
                            found_swap = True
                            progressed = True
                            break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if not bool(progressed):
                break

    def _partition_aware_external_requirement(self,
                                              node: _NodeV3,
                                              hop1_counter: Counter) -> Tuple[bool, str]:
        required = self._required_external_candidates_for_node(node)
        center_su = int(getattr(node, 'su_type', -1))
        if required is None:
            ext_present = sorted(int(x) for x in hop1_counter.keys() if int(x) in {2, 28, 29, 31})
            if ext_present:
                return False, f"SU19 missing partition but uses fixed anchors {ext_present}"
            return False, "SU19 missing partition"
        if not required:
            return check_external_connection_requirement(center_su, hop1_counter)

        target_count = 1
        try:
            val = getattr(node, 'target_fixed_anchor_count', None)
            if val is not None:
                target_count = max(1, int(val))
        except Exception:
            target_count = 1

        ext_cnt = int(sum(int(v) for k, v in hop1_counter.items() if int(k) in set(int(x) for x in required)))
        if int(ext_cnt) < int(target_count):
            return False, (
                f"SU {center_su} requires external connection to "
                f"{sorted(int(x) for x in required)} count={int(ext_cnt)}/{int(target_count)}"
            )
        if int(ext_cnt) > int(target_count):
            return False, (
                f"SU {center_su} has too many external anchors: "
                f"required={sorted(int(x) for x in required)} count={int(ext_cnt)}/{int(target_count)}"
            )
        return True, ""

    def _get_su_common_ranges(self) -> Dict[int, Dict[str, float]]:
        if isinstance(self._su_common_ranges_cache, dict):
            return self._su_common_ranges_cache
        out: Dict[int, Dict[str, float]] = {}
        try:
            df = pd.read_csv(self._su_common_range_path)
            for _, row in df.iterrows():
                try:
                    su = int(row['center_su_idx'])
                    out[su] = {
                        'mu_median': float(row.get('mu_median', 0.0) or 0.0),
                        'mu_min': float(row.get('mu_common_min', 0.0) or 0.0),
                        'mu_max': float(row.get('mu_common_max', 0.0) or 0.0),
                    }
                except Exception:
                    continue
        except Exception:
            out = {}
        self._su_common_ranges_cache = out
        return out

    @staticmethod
    def _region_specs() -> List[Tuple[str, float, float, List[int]]]:
        return [
            ('aliphatic', 0.0, 90.0, [22, 23, 24, 25, 19, 20, 21]),
            ('aromatic', 90.0, 160.0, [13, 12, 11, 10, 9, 5, 6, 7, 8, 14, 15, 16, 17, 18, 4]),
            ('carbonyl', 160.0, 240.0, [1, 2, 3, 0]),
        ]

    @staticmethod
    def _ranges_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
        return not (float(hi1) < float(lo2) or float(hi2) < float(lo1))

    def _rank_su_types_for_positive_peaks(self,
                                          priority_list: List[int],
                                          positive_peaks: List,
                                          fallback_window: Optional[Tuple[float, float]] = None) -> List[int]:
        common_ranges = self._get_su_common_ranges()
        ranked: List[int] = []
        for su in priority_list:
            meta = common_ranges.get(int(su), None)
            if not isinstance(meta, dict):
                continue
            mu_lo = float(meta.get('mu_min', meta.get('mu_median', 0.0)))
            mu_hi = float(meta.get('mu_max', meta.get('mu_median', 0.0)))
            ok = False
            for pk in positive_peaks:
                if self._ranges_overlap(mu_lo, mu_hi, float(pk.ppm_min), float(pk.ppm_max)):
                    ok = True
                    break
            if not ok and fallback_window is not None:
                ok = self._ranges_overlap(mu_lo, mu_hi, float(fallback_window[0]), float(fallback_window[1]))
            if ok:
                ranked.append(int(su))
        return ranked if ranked else [int(x) for x in priority_list]

    def _pick_positive_peak_for_node(self,
                                     center_su: int,
                                     positive_peaks: List,
                                     neg_peak,
                                     target_window_ppm: float = 24.0) -> Optional[object]:
        common_ranges = self._get_su_common_ranges()
        meta = common_ranges.get(int(center_su), None)
        mu_lo = float(meta.get('mu_min', 0.0)) if isinstance(meta, dict) else -1e9
        mu_hi = float(meta.get('mu_max', 0.0)) if isinstance(meta, dict) else 1e9
        chosen = None
        best_key = None
        neg_center = float(getattr(neg_peak, 'center_ppm', 0.0))
        for pk in positive_peaks:
            pos_lo = float(getattr(pk, 'ppm_min', neg_center))
            pos_hi = float(getattr(pk, 'ppm_max', neg_center))
            pos_center = float(getattr(pk, 'center_ppm', 0.5 * (pos_lo + pos_hi)))
            if abs(float(pos_center) - float(neg_center)) > float(target_window_ppm):
                continue
            if not self._ranges_overlap(mu_lo, mu_hi, pos_lo, pos_hi):
                continue
            key = (
                -float(getattr(pk, 'intensity', 0.0)),
                abs(float(pos_center) - float(neg_center)),
                abs(0.5 * (pos_lo + pos_hi) - float((mu_lo + mu_hi) * 0.5)),
            )
            if best_key is None or key < best_key:
                best_key = key
                chosen = pk
        return chosen

    @staticmethod
    def _region_abs_loss(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> float:
        mask = (ppm >= float(lo)) & (ppm <= float(hi))
        if not np.any(mask):
            return float('inf')
        return float(np.sum(np.abs(diff[mask])))

    def _get_carbonyl_anchor_ids(self, nodes: List[_NodeV3], node: _NodeV3) -> List[int]:
        center_su = int(node.su_type)
        hop1_ids = [int(nid) for nid in list(getattr(node, 'hop1_ids', []) or []) if 0 <= int(nid) < len(nodes)]
        if center_su == 1:
            return hop1_ids[:1]
        if center_su == 2:
            return [nid for nid in hop1_ids if int(nodes[nid].su_type) not in {5, 19}]
        return []

    @staticmethod
    def _joint_window_score(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
        mask = (ppm >= float(lo)) & (ppm <= float(hi))
        if not mask.any():
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0}
        seg = diff[mask]
        pos = float(np.sum(seg[seg > 0])) if np.any(seg > 0) else 0.0
        neg = float(-np.sum(seg[seg < 0])) if np.any(seg < 0) else 0.0
        return {
            'pos': pos,
            'neg': neg,
            'net': float(pos - neg),
            'abs': float(np.sum(np.abs(seg))),
        }

    def _decide_carbonyl_joint_direction(self,
                                         ppm: np.ndarray,
                                         diff: np.ndarray,
                                         pos_rel_threshold: float = 0.08,
                                         neg_rel_threshold: float = 0.08) -> Dict[str, object]:
        low = self._joint_window_score(ppm, diff, 160.0, 170.0)
        mid = self._joint_window_score(ppm, diff, 172.0, 180.0)
        ket = self._joint_window_score(ppm, diff, 186.0, 205.0)

        carbonyl_mask = (ppm >= 160.0) & (ppm <= 205.0)
        carbonyl_abs = float(np.sum(np.abs(diff[carbonyl_mask]))) if np.any(carbonyl_mask) else float(np.sum(np.abs(diff)))
        pos_thr = float(pos_rel_threshold) * max(1e-8, carbonyl_abs)
        neg_thr = float(neg_rel_threshold) * max(1e-8, carbonyl_abs)

        direction = None
        if float(low['neg']) > float(neg_thr) and float(mid['pos']) > float(pos_thr):
            direction = 'to_aliphatic'
        elif float(low['pos']) > float(pos_thr) and float(mid['neg']) > float(neg_thr):
            direction = 'to_aryl9'
        return {
            'direction': direction,
            'thresholds': {'pos': float(pos_thr), 'neg': float(neg_thr), 'carbonyl_abs': float(carbonyl_abs)},
            'windows': {
                '160_170': low,
                '172_180': mid,
                '186_205': ket,
            },
        }

    def _rank_joint_target_anchor_types(self,
                                        ppm: np.ndarray,
                                        diff: np.ndarray,
                                        direction: str) -> List[int]:
        if direction == 'to_aryl9':
            return [9]
        scores = {
            23: self._joint_window_score(ppm, diff, 18.0, 35.0),
            24: self._joint_window_score(ppm, diff, 32.0, 50.0),
            25: self._joint_window_score(ppm, diff, 40.0, 60.0),
        }
        ranked = sorted(
            scores.keys(),
            key=lambda su: (float(scores[su]['net']), float(scores[su]['pos']), -float(scores[su]['neg'])),
            reverse=True,
        )
        return [int(su) for su in ranked]

    def _try_joint_carbonyl_swap(self,
                                 nodes: List[_NodeV3],
                                 center_id: int,
                                 old_anchor_id: int,
                                 target_anchor_types: List[int],
                                 E_target: torch.Tensor,
                                 swap_helper: Hop1Adjuster) -> Optional[Dict[str, object]]:
        center_node = nodes[int(center_id)]
        old_anchor = nodes[int(old_anchor_id)]
        for target_type in target_anchor_types:
            candidates = [
                n for n in nodes
                if int(n.su_type) == int(target_type)
                and int(n.global_id) not in center_node.hop1_ids
                and int(n.global_id) != int(center_id)
            ]
            candidates.sort(key=lambda n: (int(n.remaining_hop1_slots()) > 0, -int(n.remaining_hop1_slots()), -int(n.global_id)), reverse=True)
            for target in candidates:
                success, swap_tail_id, affected = swap_helper._try_two_edge_swap(
                    nodes,
                    t=int(center_id),
                    u=int(old_anchor_id),
                    v=int(target.global_id),
                    E_target=E_target,
                )
                if not success:
                    continue
                swap_helper._remove_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                swap_helper._remove_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                if not swap_helper._add_hop1_edge(nodes, int(center_id), int(target.global_id)):
                    swap_helper._add_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                    swap_helper._add_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                    continue
                if not swap_helper._add_hop1_edge(nodes, int(old_anchor_id), int(swap_tail_id)):
                    swap_helper._remove_hop1_edge(nodes, int(center_id), int(target.global_id))
                    swap_helper._add_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                    swap_helper._add_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                    continue
                return {
                    'center_id': int(center_id),
                    'center_su': int(center_node.su_type),
                    'old_anchor_id': int(old_anchor_id),
                    'old_anchor_su': int(old_anchor.su_type),
                    'new_anchor_id': int(target.global_id),
                    'new_anchor_su': int(target.su_type),
                    'swap_tail_id': int(swap_tail_id),
                    'affected_nodes': list(sorted(set(int(x) for x in affected))),
                }
        return None

    def adjust_carbonyl_anchor_jointly(self,
                                       nodes: List[_NodeV3],
                                       S_target: torch.Tensor,
                                       E_target: torch.Tensor,
                                       lib_path: Optional[str] = None,
                                       hwhm: float = 1.0,
                                       allow_approx: bool = True,
                                       max_iterations: int = 3,
                                       max_adjustments_per_iter: int = 3,
                                       pos_rel_threshold: float = 0.08,
                                       neg_rel_threshold: float = 0.08) -> Tuple[List[_NodeV3], Dict[str, object]]:
        if lib_path is None:
            return nodes, {'adjustments': 0, 'iterations': 0, 'details': [], 'reason': 'missing_lib'}

        swap_helper = Hop1Adjuster(
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            external_requirement_fn=check_external_connection_requirement,
            external_requirement_node_fn=self._partition_aware_external_requirement,
        )

        all_moves: List[Dict[str, object]] = []
        diagnostics: List[Dict[str, object]] = []
        for iter_idx in range(max(1, int(max_iterations))):
            diff_info = self.evaluator.compute_difference_spectrum(
                nodes=nodes,
                S_target=S_target,
                lib_path=lib_path,
                hwhm=hwhm,
                allow_approx=allow_approx,
            )
            ppm = np.asarray(diff_info.get('ppm', []), dtype=np.float64)
            diff = np.asarray(diff_info.get('diff', []), dtype=np.float64)
            if ppm.size == 0 or diff.size == 0:
                break

            decision = self._decide_carbonyl_joint_direction(
                ppm,
                diff,
                pos_rel_threshold=float(pos_rel_threshold),
                neg_rel_threshold=float(neg_rel_threshold),
            )
            direction = decision.get('direction')
            diagnostics.append({
                'iteration': int(iter_idx + 1),
                'direction': direction,
                'decision': decision,
                'r2_before': float(diff_info.get('r2', 0.0)),
            })
            if direction is None:
                break

            if direction == 'to_aliphatic':
                source_anchor_types = {9}
                target_anchor_types = self._rank_joint_target_anchor_types(ppm, diff, direction)
            else:
                source_anchor_types = {23, 24, 25}
                target_anchor_types = [9]

            grouped_assignments = self.evaluator.build_grouped_assignments(nodes, lib_path, allow_approx)
            candidates: List[Dict[str, object]] = []
            for node in nodes:
                center_su = int(node.su_type)
                if center_su not in {1, 2}:
                    continue
                anchor_ids = self._get_carbonyl_anchor_ids(nodes, node)
                if not anchor_ids:
                    continue
                mu_pred = grouped_assignments.get(int(node.global_id), {}).get('mu', 0.0)
                for anchor_id in anchor_ids:
                    anchor_su = int(nodes[int(anchor_id)].su_type)
                    if anchor_su not in source_anchor_types:
                        continue
                    candidates.append({
                        'center_id': int(node.global_id),
                        'center_su': center_su,
                        'mu_pred': float(mu_pred or 0.0),
                        'old_anchor_id': int(anchor_id),
                        'old_anchor_su': int(anchor_su),
                    })

            if direction == 'to_aliphatic':
                candidates.sort(key=lambda c: (0 if int(c['center_su']) == 1 else 1, abs(float(c['mu_pred']) - 165.0), int(c['center_id'])))
            else:
                candidates.sort(key=lambda c: (0 if int(c['center_su']) == 2 else 1, abs(float(c['mu_pred']) - 176.0), int(c['center_id'])))

            iter_moves: List[Dict[str, object]] = []
            used_centers = set()
            for cand in candidates:
                if len(iter_moves) >= int(max_adjustments_per_iter):
                    break
                if int(cand['center_id']) in used_centers:
                    continue
                move = self._try_joint_carbonyl_swap(
                    nodes=nodes,
                    center_id=int(cand['center_id']),
                    old_anchor_id=int(cand['old_anchor_id']),
                    target_anchor_types=target_anchor_types,
                    E_target=E_target,
                    swap_helper=swap_helper,
                )
                if move is None:
                    continue
                move['iteration'] = int(iter_idx + 1)
                move['direction'] = str(direction)
                iter_moves.append(move)
                used_centers.add(int(cand['center_id']))

            if not iter_moves:
                break
            all_moves.extend(iter_moves)

        return nodes, {
            'adjustments': int(len(all_moves)),
            'iterations': int(len(diagnostics)),
            'details': all_moves,
            'diagnostics': diagnostics,
        }

    def adjust_hop1_based_on_spectrum(self,
                                      nodes: List[_NodeV3],
                                      S_target: torch.Tensor,
                                      E_target: torch.Tensor,
                                      lib_path: Optional[str] = None,
                                      output_dir: Optional[str] = None,
                                      hwhm: float = 1.0,
                                      allow_approx: bool = True,
                                      neg_threshold: float = -0.5,
                                      pos_threshold: float = 0.5,
                                      max_iterations: int = 3,
                                      adjustment_groups: Optional[List[str]] = None) -> Tuple[List[_NodeV3], Dict[str, object]]:
        device = self.device
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        if adjustment_groups is None:
            adjustment_groups = ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic']

        adjuster = Hop1Adjuster(
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            external_requirement_fn=check_external_connection_requirement,
            external_requirement_node_fn=self._partition_aware_external_requirement,
        )

        E_target = E_target.to(device).flatten()
        S_target, _ = self.evaluator.resolve_inputs(S_target)

        total_adjustments = 0
        iteration_summaries: List[Dict[str, object]] = []
        best_r2 = -1e9
        best_nodes = copy.deepcopy(nodes)
        region_summaries: List[Dict[str, object]] = []

        for region_name, region_lo, region_hi, priority_list in self._region_specs():
            region_adjustments = 0
            region_details: List[Dict[str, object]] = []
            stop_reason = 'max_iterations'
            for region_iter in range(max(1, int(max_iterations))):
                snapshot = self.evaluator.build_eval_snapshot(
                    nodes=nodes,
                    S_target=S_target,
                    lib_path=lib_path,
                    hwhm=hwhm,
                    allow_approx=bool(allow_approx),
                )
                if int(snapshot.get('n_peaks', 0)) <= 0:
                    stop_reason = 'no_valid_peaks'
                    break

                r2 = float(snapshot.get('r2', 0.0))
                diff = np.asarray(snapshot.get('diff', []), dtype=np.float64)
                ppm_np = np.asarray(snapshot.get('ppm', []), dtype=np.float64)
                node_peak_rows = list(snapshot.get('rows', []))
                loss_before = self._region_abs_loss(ppm_np, diff, float(region_lo), float(region_hi))
                if float(r2) > float(best_r2):
                    best_r2 = float(r2)
                    best_nodes = copy.deepcopy(nodes)

                region_mask = (ppm_np >= float(region_lo)) & (ppm_np <= float(region_hi))
                neg_peaks, pos_peaks = adjuster.analyze_difference_spectrum(
                    diff=diff[region_mask],
                    ppm=ppm_np[region_mask],
                    neg_threshold=float(neg_threshold),
                    pos_threshold=float(pos_threshold),
                )
                if not neg_peaks:
                    stop_reason = 'no_negative_peaks'
                    break
                if not pos_peaks:
                    stop_reason = 'no_positive_peaks'
                    break

                neg_peak = neg_peaks[0]
                su_priority = self._rank_su_types_for_positive_peaks(
                    priority_list=priority_list,
                    positive_peaks=pos_peaks,
                    fallback_window=(float(region_lo), float(region_hi)),
                )

                accepted_move: Optional[Dict[str, object]] = None
                neg_nodes_all = adjuster.find_nodes_in_peak_region(nodes, node_peak_rows, neg_peak)
                for su_type in su_priority:
                    cand_nodes = [x for x in neg_nodes_all if int(x['center_su']) == int(su_type)]
                    cand_nodes.sort(key=lambda x: int(x['global_id']))
                    for node_info in cand_nodes:
                        pos_peak = self._pick_positive_peak_for_node(
                            center_su=int(node_info['center_su']),
                            positive_peaks=pos_peaks,
                            neg_peak=neg_peak,
                        )
                        if pos_peak is None:
                            continue
                        alternatives = adjuster.find_alternative_hop1(
                            int(node_info['center_su']),
                            tuple(node_info['hop1_ms']),
                            (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                        )
                        if not alternatives:
                            continue
                        for alt in alternatives[: min(8, len(alternatives))]:
                            prev_state = self._capture_hop1_state(nodes)
                            result = adjuster.execute_hop1_replacement(
                                nodes=nodes,
                                target_node_id=int(node_info['global_id']),
                                new_hop1_tuple=tuple(alt['hop1_tuple']),
                                E_target=E_target,
                                dry_run=False,
                            )
                            if not bool(result.get('success', False)):
                                self._restore_hop1_state(nodes, prev_state)
                                continue
                            snapshot_try = self.evaluator.build_eval_snapshot(
                                nodes=nodes,
                                S_target=S_target,
                                lib_path=lib_path,
                                hwhm=hwhm,
                                allow_approx=bool(allow_approx),
                            )
                            diff_try = np.asarray(snapshot_try.get('diff', []), dtype=np.float64)
                            ppm_try = np.asarray(snapshot_try.get('ppm', []), dtype=np.float64)
                            loss_after = self._region_abs_loss(ppm_try, diff_try, float(region_lo), float(region_hi))
                            if float(loss_after) < float(loss_before) - 1e-9:
                                accepted_move = {
                                    'region': str(region_name),
                                    'iteration': int(region_iter + 1),
                                    'global_id': int(node_info['global_id']),
                                    'center_su': int(node_info['center_su']),
                                    'old_hop1': tuple(node_info['hop1_ms']),
                                    'new_hop1': tuple(alt['hop1_tuple']),
                                    'old_mu': float(node_info['mu']),
                                    'target_pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                                    'target_pos_center': float(pos_peak.center_ppm),
                                    'target_mu_median': float(alt.get('mu_median', 0.0)),
                                    'loss_before': float(loss_before),
                                    'loss_after': float(loss_after),
                                    'affected_nodes': list(result.get('affected_nodes', []) or []),
                                }
                                break
                            self._restore_hop1_state(nodes, prev_state)
                        if accepted_move is not None:
                            break
                    if accepted_move is not None:
                        break

                if accepted_move is None:
                    stop_reason = 'no_valid_structural_swap'
                    break

                region_adjustments += 1
                total_adjustments += 1
                region_details.append(dict(accepted_move))
                iteration_summaries.append(dict(accepted_move))

            region_summaries.append({
                'region': str(region_name),
                'window': (float(region_lo), float(region_hi)),
                'adjustments': int(region_adjustments),
                'stop_reason': str(stop_reason),
                'details': region_details,
            })

        final_metrics = self.evaluator.evaluate_with_library(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path,
            output_dir=output_dir,
            hwhm=hwhm,
            allow_approx=bool(allow_approx),
        )
        final_attempt_r2 = float(final_metrics.get('r2', 0.0))
        selected_nodes = nodes
        selected_metrics = final_metrics
        best_source = 'final_attempt'

        if float(final_attempt_r2) >= float(best_r2):
            best_r2 = float(final_attempt_r2)
            best_nodes = copy.deepcopy(nodes)
        else:
            selected_nodes = copy.deepcopy(best_nodes)
            selected_metrics = self.evaluator.evaluate_with_library(
                nodes=selected_nodes,
                S_target=S_target,
                lib_path=lib_path,
                output_dir=output_dir,
                hwhm=hwhm,
                allow_approx=bool(allow_approx),
            )
            best_r2 = float(selected_metrics.get('r2', best_r2))
            best_source = 'best_iteration'

        summary = {
            'total_adjustments': int(total_adjustments),
            'iterations': int(len(iteration_summaries)),
            'iteration_details': iteration_summaries,
            'region_summaries': region_summaries,
            'best_r2': float(best_r2),
            'final_r2': float(selected_metrics.get('r2', 0.0)),
            'final_attempt_r2': float(final_attempt_r2),
            'selected_source': str(best_source),
            'adjuster_stats': dict(adjuster.stats),
        }
        return selected_nodes, summary
