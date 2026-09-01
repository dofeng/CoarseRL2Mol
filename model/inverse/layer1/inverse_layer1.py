import copy
import hashlib
import time
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple, Set
from collections import Counter, defaultdict

from ...shared.coarse_graph import E_SU, NUM_SU_TYPES
from ...shared.inverse_common import (
    _NodeV3, HOP1_PORT_COMBINATIONS,
    SU_CONNECTION_DEGREE,
    SPECIAL_DEGREE_PRIORS,
    allocate_special_degree_counts,
    rebuild_su19_partition_meta,
    SU_EXTERNAL_CONNECTIONS,
    format_port_patterns_debug,
    has_forbidden_fixed_connector_special_combo,
    is_mode_specific_neighbor_forbidden,
    is_terminal_port_node,
    select_port_patterns_for_degree,
    validate_connection,
    violates_special_d3_terminal_limit,
)
from .hop1_adjuster import (
    Hop1Adjuster,
    can_match_ports_exact,
    can_match_ports_partial,
    hop1_counter_to_multiset,
)
from .layer1_validator import Layer1GraphValidator
from .port_role_balance import (
    LAYER1_AROMATIC_RING_TYPES,
    audit_layer1_port_roles,
)


class Layer1GreedyCompleter:
    """Deterministic diversity-first completion with bounded local repair."""

    MAX_REPAIR_DEPTH = 2
    MAX_REPAIR_ATTEMPTS = 64

    def __init__(
        self,
        assigner: Any,
        nodes: List[_NodeV3],
        phase: str = 'all',
        verbose: bool = False,
        forced_decision: Optional[Dict[str, Any]] = None,
    ) -> None:
        if str(phase) not in {'all', 'non_aromatic', 'aromatic'}:
            raise ValueError(f"unsupported Layer1 completion phase: {phase}")
        self.assigner = assigner
        self.nodes = nodes
        self.phase = str(phase)
        self.verbose = bool(verbose)
        self.forced_decision = dict(forced_decision or {})
        self.started_at = time.perf_counter()
        self.decisions = 0
        self.repair_attempts = 0
        self.repair_rewinds = 0
        self.domains: Dict[int, Set[int]] = {}
        self._exact_options_cache: Dict[
            Tuple[int, int, Tuple[int, ...], Optional[int]],
            Tuple[Tuple[int, ...], ...],
        ] = {}
        self._soft_motif_contribution_cache: Dict[
            Tuple[int, int, Tuple[int, ...], Optional[int]],
            Optional[Dict[Tuple[int, int, Tuple[int, ...]], float]],
        ] = {}
        self._decision_stack: List[Dict[str, Any]] = []
        self._failed_edge_states: Set[Tuple[Tuple[int, int], ...]] = set()
        self._repair_branch_keys: Set[Tuple[Any, ...]] = set()
        self._rebuild_domains()

    def _su11_external_complete(self, node: _NodeV3) -> bool:
        return bool(self.assigner._has_required_external_for_node(node, self.nodes))

    def _node_is_in_phase(self, node: _NodeV3) -> bool:
        su_type = int(node.su_type)
        if self.phase == 'all':
            return True
        if self.phase == 'aromatic':
            return su_type in LAYER1_AROMATIC_RING_TYPES
        if su_type == 11:
            return not self._su11_external_complete(node)
        return su_type not in LAYER1_AROMATIC_RING_TYPES

    def _active_ids(self) -> List[int]:
        return sorted(
            int(node.global_id)
            for node in self.nodes
            if int(node.remaining_hop1_slots()) > 0
            and self._node_is_in_phase(node)
        )

    def _phase_open_ports(self) -> int:
        if self.phase != 'non_aromatic':
            return int(sum(
                int(node.remaining_hop1_slots())
                for node in self.nodes
                if self._node_is_in_phase(node)
            ))
        non_aromatic = int(sum(
            int(node.remaining_hop1_slots())
            for node in self.nodes
            if int(node.su_type) not in LAYER1_AROMATIC_RING_TYPES
        ))
        pending_su11 = int(sum(
            max(
                0,
                int(self.assigner._node_fixed_anchor_target(node))
                - int(self.assigner._required_external_count_for_node(node, self.nodes)),
            )
            for node in self.nodes
            if int(node.su_type) == 11
        ))
        return int(non_aromatic + pending_su11)

    @staticmethod
    def _is_subcounter(current: Counter, target: Counter) -> bool:
        return not bool(current - target)

    def _exact_options(
        self,
        node: _NodeV3,
        proposed_su: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...], ...]:
        if not self.assigner._exact_hop1_shift_constraints_enabled:
            return ((),)
        center_su = int(node.su_type)
        if center_su not in self.assigner._carbon_su_set:
            return ((),)
        degree = self.assigner._node_target_degree(node)
        if degree is None:
            return ()
        current = Counter(node.hop1_su)
        cache_key = (
            center_su,
            int(degree),
            tuple(sorted(int(x) for x in current.elements())),
            None if proposed_su is None else int(proposed_su),
        )
        cached = self._exact_options_cache.get(cache_key)
        if cached is not None:
            return cached
        if proposed_su is not None:
            current[int(proposed_su)] += 1
        options = tuple(
            signature
            for signature in self.assigner._exact_hop1_signatures_by_center_degree.get(
                (center_su, int(degree)),
                (),
            )
            if self._is_subcounter(current, Counter(signature))
        )
        self._exact_options_cache[cache_key] = options
        return options

    def _candidate_ids(self, node: _NodeV3) -> List[int]:
        options = self._exact_options(node)
        if not options or options == ((),):
            return self._active_ids()
        current = Counter(node.hop1_su)
        missing_types: Set[int] = set()
        for signature in options:
            missing = Counter(signature) - current
            missing_types.update(
                int(su_type)
                for su_type, count in missing.items()
                if int(count) > 0
            )
        available = self.assigner._available_ids_by_su
        return sorted({
            int(node_id)
            for su_type in missing_types
            for node_id in available.get(int(su_type), set())
            if int(self.nodes[int(node_id)].remaining_hop1_slots()) > 0
        })

    def _rebuild_domains(self) -> None:
        active = self._active_ids()
        self.domains = {node_id: set() for node_id in active}
        seen: Set[Tuple[int, int]] = set()
        for left_id in active:
            left = self.nodes[left_id]
            for right_id in self._candidate_ids(left):
                if left_id == right_id or right_id not in self.domains:
                    continue
                edge = tuple(sorted((left_id, right_id)))
                if edge in seen:
                    continue
                seen.add(edge)
                right = self.nodes[right_id]
                if self.assigner._can_add_hop1_connection(self.nodes, left, right):
                    self.domains[left_id].add(right_id)
                    self.domains[right_id].add(left_id)

    def _refresh_endpoint_domains(self, endpoint_ids: Set[int]) -> None:
        endpoint_ids = set(int(x) for x in endpoint_ids)
        for candidates in self.domains.values():
            candidates.difference_update(endpoint_ids)
        for endpoint_id in endpoint_ids:
            self.domains.pop(endpoint_id, None)

        active = set(self._active_ids())
        for endpoint_id in sorted(endpoint_ids & active):
            self.domains[endpoint_id] = set()
            endpoint = self.nodes[endpoint_id]
            for candidate_id in self._candidate_ids(endpoint):
                candidate_id = int(candidate_id)
                if candidate_id == endpoint_id or candidate_id not in active:
                    continue
                candidate = self.nodes[candidate_id]
                if self.assigner._can_add_hop1_connection(
                    self.nodes,
                    endpoint,
                    candidate,
                ):
                    self.domains[endpoint_id].add(candidate_id)
                    self.domains.setdefault(candidate_id, set()).add(endpoint_id)

    def _add_edge(self, left_id: int, right_id: int) -> bool:
        if not self.assigner._add_bidirectional_hop1(
            self.nodes,
            int(left_id),
            int(right_id),
        ):
            return False
        self._refresh_endpoint_domains({int(left_id), int(right_id)})
        return True

    def _remove_edge(self, left_id: int, right_id: int) -> bool:
        if not self.assigner._remove_bidirectional_hop1_with_force(
            self.nodes,
            int(left_id),
            int(right_id),
            force=False,
        ):
            return False
        self._refresh_endpoint_domains({int(left_id), int(right_id)})
        return True

    def _select_center(self) -> Tuple[Optional[_NodeV3], List[int]]:
        best: Optional[Tuple[Tuple[int, int, int], _NodeV3, List[int]]] = None
        for node_id in self._active_ids():
            node = self.nodes[node_id]
            partners = list(self.domains.get(node_id, set()))
            key = (
                len(partners),
                int(node.remaining_hop1_slots()),
                int(node_id),
            )
            if best is None or key < best[0]:
                best = (key, node, partners)
        if best is None:
            return None, []
        return best[1], best[2]

    @staticmethod
    def _su12_role_is_complete(node: _NodeV3) -> bool:
        return bool(
            int(getattr(node, 'su_type', -1)) != 12
            or int(Counter(getattr(node, 'hop1_su', {}) or {}).get(12, 0)) >= 1
        )

    def _uncovered_su12_ids(self) -> List[int]:
        if self.phase != 'aromatic':
            return []
        return sorted(
            int(node.global_id)
            for node in self.nodes
            if int(node.su_type) == 12
            and not self._su12_role_is_complete(node)
        )

    def _select_su12_role_center(self) -> Tuple[Optional[_NodeV3], List[int]]:
        """Select an uncovered SU12 and a legal SU12 partner.

        SU12-SU12 is a required aromatic role, not a fixed edge. Prefer pairing
        two uncovered nodes so one node cannot consume several SU12 ports while
        another node is still uncovered. An odd final node may attach to an
        already covered SU12 with remaining capacity.
        """
        uncovered = set(self._uncovered_su12_ids())
        if not uncovered:
            return None, []

        best: Optional[Tuple[Tuple[int, int, int], _NodeV3, List[int]]] = None
        for node_id in sorted(uncovered):
            node = self.nodes[int(node_id)]
            all_partners = [
                int(partner_id)
                for partner_id in self.domains.get(int(node_id), set())
                if int(self.nodes[int(partner_id)].su_type) == 12
                and int(self.nodes[int(partner_id)].remaining_hop1_slots()) > 0
            ]
            uncovered_partners = [
                int(partner_id)
                for partner_id in all_partners
                if int(partner_id) in uncovered
            ]
            partners = uncovered_partners or all_partners
            key = (
                len(partners),
                int(node.remaining_hop1_slots()),
                int(node_id),
            )
            if best is None or key < best[0]:
                best = (key, node, partners)
        if best is None:
            return None, []

        center, partners = best[1], best[2]
        base_order = self._order_partners(center, partners)
        base_rank = {
            int(partner_id): int(rank)
            for rank, partner_id in enumerate(base_order)
        }
        ordered = sorted(
            (int(partner_id) for partner_id in partners),
            key=lambda partner_id: (
                int(partner_id not in uncovered),
                int(Counter(self.nodes[partner_id].hop1_su).get(12, 0)),
                int(base_rank.get(int(partner_id), len(base_rank))),
                int(partner_id),
            ),
        )
        return center, ordered

    def _node_soft_motif_contribution(
        self,
        node: _NodeV3,
        proposed_su: Optional[int] = None,
    ) -> Optional[Dict[Tuple[int, int, Tuple[int, ...]], float]]:
        """Distribute one node over every exact template it can still reach."""
        if not bool(self.assigner._exact_hop1_shift_constraints_enabled):
            return {}
        center_su = int(node.su_type)
        if center_su not in self.assigner._carbon_su_set:
            return {}
        degree = self.assigner._node_target_degree(node)
        if degree is None:
            return {}
        cache_key = (
            int(center_su),
            int(degree),
            tuple(sorted(int(x) for x in Counter(node.hop1_su).elements())),
            None if proposed_su is None else int(proposed_su),
        )
        cache = getattr(self, "_soft_motif_contribution_cache", None)
        if cache is None:
            cache = {}
            self._soft_motif_contribution_cache = cache
        if cache_key in cache:
            return cache[cache_key]
        options = self._exact_options(node, proposed_su)
        if not options:
            cache[cache_key] = None
            return None
        weight = 1.0 / float(len(options))
        contribution: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
        for option in options:
            key = (
                int(center_su),
                int(degree),
                tuple(int(value) for value in option),
            )
            contribution[key] = float(contribution.get(key, 0.0)) + float(weight)
        cache[cache_key] = contribution
        return contribution

    def _soft_motif_usage(
        self,
    ) -> Tuple[
        Counter,
        Dict[int, Dict[Tuple[int, int, Tuple[int, ...]], float]],
    ]:
        usage: Counter = Counter()
        contributions: Dict[
            int,
            Dict[Tuple[int, int, Tuple[int, ...]], float],
        ] = {}
        for node in self.nodes:
            contribution = self._node_soft_motif_contribution(node)
            if contribution is None:
                continue
            contributions[int(node.global_id)] = dict(contribution)
            for key, value in contribution.items():
                usage[key] += float(value)
        return usage, contributions

    def _candidate_template_concentration_costs(
        self,
        center: _NodeV3,
        partner_ids: List[int],
    ) -> Dict[int, float]:
        """Return the concentration increase for every exact-template edge."""
        usage, contributions = self._soft_motif_usage()
        center_before = contributions.get(int(center.global_id), {})
        costs: Dict[int, float] = {}
        for partner_id in (int(value) for value in partner_ids):
            partner = self.nodes[int(partner_id)]
            center_after = self._node_soft_motif_contribution(
                center,
                int(partner.su_type),
            )
            partner_after = self._node_soft_motif_contribution(
                partner,
                int(center.su_type),
            )
            if center_after is None or partner_after is None:
                continue
            partner_before = contributions.get(int(partner_id), {})
            affected_keys = (
                set(center_before)
                | set(partner_before)
                | set(center_after)
                | set(partner_after)
            )
            delta = 0.0
            for key in affected_keys:
                before = float(usage.get(key, 0.0))
                after = (
                    before
                    - float(center_before.get(key, 0.0))
                    - float(partner_before.get(key, 0.0))
                    + float(center_after.get(key, 0.0))
                    + float(partner_after.get(key, 0.0))
                )
                delta += float(after * after - before * before)
            costs[int(partner_id)] = round(float(delta), 12)
        return costs

    def _component(self, start_id: int) -> Set[int]:
        visited: Set[int] = set()
        pending = [int(start_id)]
        while pending:
            node_id = int(pending.pop())
            if node_id in visited:
                continue
            visited.add(node_id)
            pending.extend(
                int(x)
                for x in list(self.nodes[node_id].hop1_ids or [])
                if int(x) not in visited
            )
        return visited

    def _joint_primary_scores(
        self,
        center: _NodeV3,
        partner_ids: List[int],
    ) -> Dict[int, float]:
        partner_ids = [int(value) for value in partner_ids]
        if not partner_ids:
            return {}

        template_costs = self._candidate_template_concentration_costs(
            center,
            partner_ids,
        )
        partner_ids = [
            int(partner_id)
            for partner_id in partner_ids
            if int(partner_id) in template_costs
        ]
        if not partner_ids:
            return {}

        local_su_usage = {
            partner_id: int(Counter(center.hop1_su).get(
                int(self.nodes[partner_id].su_type),
                0,
            ))
            for partner_id in partner_ids
        }
        remaining_ports = {
            partner_id: int(self.nodes[partner_id].remaining_hop1_slots())
            for partner_id in partner_ids
        }

        def _normalized_dense_ranks(
            values: Dict[int, Any],
            *,
            reverse: bool,
        ) -> Dict[int, float]:
            levels = sorted(set(values.values()), reverse=bool(reverse))
            denominator = max(1, len(levels) - 1)
            rank_by_value = {
                value: float(rank) / float(denominator)
                for rank, value in enumerate(levels)
            }
            return {
                int(partner_id): float(rank_by_value[value])
                for partner_id, value in values.items()
            }

        # Template concentration is one half of the sole primary score; the
        # two resource heuristics share the other half.
        su_diversity_rank = _normalized_dense_ranks(
            local_su_usage,
            reverse=False,
        )
        port_capacity_rank = _normalized_dense_ranks(
            remaining_ports,
            reverse=True,
        )
        template_diversity_rank = _normalized_dense_ranks(
            {
                int(partner_id): float(template_costs[int(partner_id)])
                for partner_id in partner_ids
            },
            reverse=False,
        )
        return {
            int(partner_id): (
                0.50 * float(template_diversity_rank[partner_id])
                + 0.25 * float(su_diversity_rank[partner_id])
                + 0.25 * float(port_capacity_rank[partner_id])
            )
            for partner_id in partner_ids
        }

    def _joint_primary_frontier(
        self,
        center: _NodeV3,
        partner_ids: List[int],
    ) -> List[int]:
        scores = self._joint_primary_scores(center, partner_ids)
        if not scores:
            return []
        best = min(float(value) for value in scores.values())
        return [
            int(partner_id)
            for partner_id in partner_ids
            if int(partner_id) in scores
            if abs(float(scores[int(partner_id)]) - float(best)) <= 1e-12
        ]

    def _order_partners(self, center: _NodeV3, partner_ids: List[int]) -> List[int]:
        component = self._component(int(center.global_id))
        partner_ids = [int(value) for value in partner_ids]
        primary_scores = self._joint_primary_scores(center, partner_ids)
        partner_ids = [
            int(partner_id)
            for partner_id in partner_ids
            if int(partner_id) in primary_scores
        ]

        return sorted(
            partner_ids,
            key=lambda partner_id: (
                float(primary_scores[partner_id]),
                int(partner_id in component),
                int(self.nodes[partner_id].su_type),
                int(partner_id),
            ),
        )

    def _all_complete_and_exact(self) -> bool:
        if self.phase == 'non_aromatic':
            non_aromatic_complete = all(
                int(node.remaining_hop1_slots()) == 0
                and self.assigner._node_exact_hop1_shift_state_ok(
                    node,
                    require_full=True,
                )
                for node in self.nodes
                if int(node.su_type) not in LAYER1_AROMATIC_RING_TYPES
            )
            su11_external_complete = all(
                self._su11_external_complete(node)
                for node in self.nodes
                if int(node.su_type) == 11
            )
            return bool(non_aromatic_complete and su11_external_complete)
        return all(
            int(node.remaining_hop1_slots()) == 0
            and self.assigner._node_exact_hop1_shift_state_ok(
                node,
                require_full=True,
            )
            for node in self.nodes
            if self._node_is_in_phase(node)
        )

    def _edge_state_signature(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (int(node.global_id), int(neighbor_id))
            for node in self.nodes
            for neighbor_id in sorted(int(x) for x in list(node.hop1_ids or []))
            if int(node.global_id) < int(neighbor_id)
        )

    def _restore_rewound_decisions(
        self,
        rewound_decisions: List[Dict[str, Any]],
    ) -> None:
        for decision in reversed(rewound_decisions):
            center_id = int(decision['center_id'])
            chosen_id = int(decision['chosen_id'])
            if not self._add_edge(center_id, chosen_id):
                raise RuntimeError(
                    "layer1_rule_graph_repair_restore_failed: "
                    f"edge=({center_id}, {chosen_id})"
                )
            self._decision_stack.append(decision)
        # A failed repair is exceptional, so prefer a complete, consistent
        # domain rebuild over leaving diagnostics based on incremental state.
        self._rebuild_domains()

    def _repair_branch_key(
        self,
        center_id: int,
        partner_id: int,
        decision_kind: str,
    ) -> Tuple[Any, ...]:
        center = self.nodes[int(center_id)]
        partner = self.nodes[int(partner_id)]
        partition_fn = getattr(
            self.assigner,
            '_node_anchor_partition',
            lambda _node: None,
        )
        anchor_target_fn = getattr(
            self.assigner,
            '_node_fixed_anchor_target',
            lambda _node: 0,
        )
        anchor_mode_fn = getattr(
            self.assigner,
            '_node_anchor_mode',
            lambda _node: None,
        )

        def _node_key(node: _NodeV3) -> Tuple[Any, ...]:
            return (
                int(node.su_type),
                int(self.assigner._node_target_degree(node) or -1),
                tuple(sorted(int(x) for x in Counter(node.hop1_su).elements())),
                int(node.remaining_hop1_slots()),
                partition_fn(node),
                int(anchor_target_fn(node)),
                anchor_mode_fn(node),
                bool(self._su12_role_is_complete(node)),
            )

        partner_component_profile = tuple(sorted(
            (_node_key(self.nodes[int(node_id)]) for node_id in self._component(partner_id)),
            key=repr,
        ))
        open_state_profile = tuple(sorted(
            (
                _node_key(node)
                for node in self.nodes
                if int(node.remaining_hop1_slots()) > 0
                and self._node_is_in_phase(node)
            ),
            key=repr,
        ))

        return (
            str(decision_kind),
            _node_key(center),
            _node_key(partner),
            bool(int(partner_id) in self._component(int(center_id))),
            partner_component_profile,
            open_state_profile,
        )

    def _selection_has_immediate_dead_end(self, decision_kind: str) -> bool:
        if str(decision_kind) == 'su12_role' and self._uncovered_su12_ids():
            center, partner_ids = self._select_su12_role_center()
        else:
            center, partner_ids = self._select_center()
        return bool(center is not None and not partner_ids)

    def _try_local_repair(self, decision_kind: str = 'ordinary') -> bool:
        """Backtrack the saved decision branches, preserving the fixed limits."""
        self._failed_edge_states.add(self._edge_state_signature())
        rewound = 0
        rewound_decisions: List[Dict[str, Any]] = []
        while (
            self._decision_stack
            and rewound < int(self.MAX_REPAIR_DEPTH)
            and self.repair_attempts < int(self.MAX_REPAIR_ATTEMPTS)
        ):
            decision = self._decision_stack.pop()
            if not self._remove_edge(decision['center_id'], decision['chosen_id']):
                self._decision_stack.append(decision)
                self._restore_rewound_decisions(rewound_decisions)
                return False
            rewound_decisions.append(decision)
            rewound += 1
            self.repair_rewinds += 1

            center_id = int(decision['center_id'])
            saved_order = [
                int(x) for x in list(decision.get('alternatives', []) or [])
            ]
            next_index = max(0, int(decision.get('next_index', 0) or 0))
            frame_kind = str(decision.get('decision_kind', decision_kind) or decision_kind)
            saved_untried = saved_order[next_index:]
            center = self.nodes[int(center_id)]
            refreshed = self._order_partners(
                center,
                list(self.domains.get(int(center_id), set())),
            )
            ordered = list(saved_untried)
            ordered.extend(
                int(partner_id)
                for partner_id in refreshed
                if int(partner_id) != int(decision['chosen_id'])
                and int(partner_id) not in set(ordered)
            )
            for alternative_index, partner_id in enumerate(ordered):
                partner_id = int(partner_id)
                if int(partner_id) not in self.domains.get(int(center_id), set()):
                    continue
                branch_key = self._repair_branch_key(
                    center_id,
                    partner_id,
                    frame_kind,
                )
                if branch_key in self._repair_branch_keys:
                    continue
                if self.repair_attempts >= int(self.MAX_REPAIR_ATTEMPTS):
                    break
                self._repair_branch_keys.add(branch_key)
                self.repair_attempts += 1
                if not self._add_edge(center_id, partner_id):
                    continue

                state_signature = self._edge_state_signature()
                immediate_dead_end = self._selection_has_immediate_dead_end(
                    frame_kind
                )
                if (
                    state_signature in self._failed_edge_states
                    or immediate_dead_end
                ):
                    if immediate_dead_end:
                        self._failed_edge_states.add(state_signature)
                    if not self._remove_edge(center_id, partner_id):
                        raise RuntimeError(
                            "layer1_rule_graph_repair_trial_restore_failed: "
                            f"edge=({center_id}, {partner_id})"
                        )
                    continue

                self._decision_stack.append({
                    'center_id': center_id,
                    'chosen_id': partner_id,
                    'alternatives': ordered,
                    'next_index': int(alternative_index) + 1,
                    'decision_kind': frame_kind,
                    'stack_index': int(len(self._decision_stack)),
                    'primary_frontier': [int(partner_id)],
                    'forced_decision': False,
                })
                return True

        self._restore_rewound_decisions(rewound_decisions)
        return False

    def _commit_first_candidate(
        self,
        center: _NodeV3,
        ordered: List[int],
        decision_kind: str,
    ) -> bool:
        if not ordered:
            return False
        center_id = int(center.global_id)
        stack_index = int(len(self._decision_stack))
        prefix_edge_signature = self._edge_state_signature()
        primary_frontier = (
            self._joint_primary_frontier(center, ordered)
            if str(decision_kind) == 'ordinary'
            else []
        )
        trial_order = [int(value) for value in ordered]
        forced = dict(self.forced_decision or {})
        forced_here = bool(
            forced
            and str(forced.get('phase', '')) == str(self.phase)
            and int(forced.get('stack_index', -1)) == int(stack_index)
        )
        if forced_here:
            forced_center = int(forced.get('center_id', -1))
            forced_partner = int(forced.get('chosen_partner_id', -1))
            if forced_center != int(center_id):
                raise RuntimeError(
                    'layer1_multistart_replay_center_mismatch: '
                    f"phase={self.phase} stack_index={stack_index} "
                    f"expected={forced_center} actual={center_id}"
                )
            if forced_partner not in set(primary_frontier):
                raise RuntimeError(
                    'layer1_multistart_replay_partner_not_in_primary_frontier: '
                    f"phase={self.phase} stack_index={stack_index} "
                    f"partner={forced_partner} frontier={primary_frontier}"
                )
            expected_prefix = tuple(
                (int(edge[0]), int(edge[1]))
                for edge in list(forced.get('prefix_edge_signature', []) or [])
            )
            if (
                'prefix_edge_signature' in forced
                and expected_prefix != prefix_edge_signature
            ):
                raise RuntimeError(
                    'layer1_multistart_replay_prefix_mismatch: '
                    f"phase={self.phase} stack_index={stack_index}"
                )
            trial_order = [forced_partner] + [
                int(value) for value in trial_order
                if int(value) != int(forced_partner)
            ]

        for candidate_index, chosen_id in enumerate(trial_order):
            chosen_id = int(chosen_id)
            if not self._add_edge(center_id, chosen_id):
                continue
            self.decisions += 1
            self._decision_stack.append({
                'center_id': center_id,
                'chosen_id': chosen_id,
                'alternatives': [int(x) for x in trial_order],
                'next_index': int(candidate_index) + 1,
                'decision_kind': str(decision_kind),
                'stack_index': int(stack_index),
                'prefix_edge_signature': tuple(prefix_edge_signature),
                'primary_frontier': [int(x) for x in primary_frontier],
                'forced_decision': bool(forced_here and chosen_id == int(
                    forced.get('chosen_partner_id', -1)
                )),
            })
            return True
        return False

    def _complete_required_su12_roles(self) -> None:
        if self.phase != 'aromatic':
            return
        su12_count = int(sum(
            int(node.su_type) == 12 for node in self.nodes
        ))
        if int(su12_count) == 1:
            raise RuntimeError(
                "layer1_su12_required_role_infeasible: "
                "a single SU12 cannot have a distinct SU12 neighbor"
            )

        while self._uncovered_su12_ids():
            center, partner_ids = self._select_su12_role_center()
            if center is None or not partner_ids:
                if self._try_local_repair(decision_kind='su12_role'):
                    continue
                uncovered = self._uncovered_su12_ids()
                raise RuntimeError(
                    "layer1_su12_required_role_infeasible: "
                    f"uncovered_ids={uncovered[:16]} "
                    f"repair_attempts={self.repair_attempts}"
                )
            if not self._commit_first_candidate(
                center,
                [int(x) for x in partner_ids],
                decision_kind='su12_role',
            ):
                if self._try_local_repair(decision_kind='su12_role'):
                    continue
                raise RuntimeError(
                    "layer1_su12_required_role_infeasible: "
                    f"center_id={int(center.global_id)}"
                )

    def _failure_diagnostics(
        self,
        center: Optional[_NodeV3],
        partner_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        open_nodes = [
            node for node in self.nodes
            if int(node.remaining_hop1_slots()) > 0
            and self._node_is_in_phase(node)
        ]
        open_ports_by_su: Counter = Counter()
        for node in open_nodes:
            open_ports_by_su[int(node.su_type)] += int(
                node.remaining_hop1_slots()
            )
        su12_nodes = [
            node for node in self.nodes if int(node.su_type) == 12
        ]
        uncovered_su12 = [
            int(node.global_id)
            for node in su12_nodes
            if not self._su12_role_is_complete(node)
        ]
        multi_covered_su12 = [
            int(node.global_id)
            for node in su12_nodes
            if int(Counter(node.hop1_su).get(12, 0)) > 1
        ]
        blocked: Optional[Dict[str, Any]] = None
        if center is not None:
            partner_ids = [int(x) for x in list(partner_ids or [])]
            blocked = {
                'node_id': int(center.global_id),
                'su_type': int(center.su_type),
                'remaining_ports': int(center.remaining_hop1_slots()),
                'current_hop1': sorted(
                    int(x) for x in Counter(center.hop1_su).elements()
                ),
                'legal_partner_types': sorted({
                    int(self.nodes[x].su_type) for x in partner_ids
                }),
                'exact_options': [
                    list(int(x) for x in option)
                    for option in self._exact_options(center)[:12]
                ],
            }
        return {
            'blocked': blocked,
            'open_node_count': int(len(open_nodes)),
            'open_ports_by_su': dict(sorted(open_ports_by_su.items())),
            'open_nodes': [
                {
                    'node_id': int(node.global_id),
                    'su_type': int(node.su_type),
                    'remaining_ports': int(node.remaining_hop1_slots()),
                    'current_hop1': sorted(
                        int(x) for x in Counter(node.hop1_su).elements()
                    ),
                }
                for node in open_nodes[:16]
            ],
            'su12': {
                'total': int(len(su12_nodes)),
                'covered': int(len(su12_nodes) - len(uncovered_su12)),
                'uncovered_ids': uncovered_su12[:16],
                'multi_covered_ids': multi_covered_su12[:16],
            },
            'last_decisions': [
                {
                    'center_id': int(row.get('center_id', -1)),
                    'chosen_id': int(row.get('chosen_id', -1)),
                    'next_index': int(row.get('next_index', 0) or 0),
                    'decision_kind': str(row.get('decision_kind', 'ordinary')),
                }
                for row in self._decision_stack[-2:]
            ],
        }

    def _branch_points(self) -> List[Dict[str, Any]]:
        return [
            {
                'phase': str(self.phase),
                'stack_index': int(row.get('stack_index', index)),
                'decision_kind': str(row.get('decision_kind', 'ordinary')),
                'center_id': int(row.get('center_id', -1)),
                'chosen_partner_id': int(row.get('chosen_id', -1)),
                'prefix_edge_signature': [
                    (int(edge[0]), int(edge[1]))
                    for edge in list(row.get('prefix_edge_signature', []) or [])
                ],
                'primary_frontier': [
                    int(value)
                    for value in list(row.get('primary_frontier', []) or [])
                ],
                'alternative_partner_ids': [
                    int(value)
                    for value in list(row.get('primary_frontier', []) or [])
                    if int(value) != int(row.get('chosen_id', -1))
                ],
            }
            for index, row in enumerate(self._decision_stack)
            if len(list(row.get('primary_frontier', []) or [])) > 1
        ]

    def solve(self) -> Dict[str, Any]:
        initial_open_ports = int(self._phase_open_ports())

        self._complete_required_su12_roles()

        while not self._all_complete_and_exact():
            center, partner_ids = self._select_center()
            if center is None or not partner_ids:
                if self._try_local_repair(decision_kind='ordinary'):
                    continue
                center, partner_ids = self._select_center()
                if center is not None and partner_ids:
                    continue
                diagnostics = self._failure_diagnostics(center, partner_ids)
                raise RuntimeError(
                    "layer1_rule_graph_search_exhausted: bounded_greedy_repair_exhausted "
                    f"phase={self.phase} diagnostics={diagnostics} "
                    f"repair_attempts={self.repair_attempts}"
                )

            ordered = self._order_partners(center, partner_ids)
            if not self._commit_first_candidate(
                center,
                ordered,
                decision_kind='ordinary',
            ):
                continue
            if self.verbose and self.decisions % 50 == 0:
                open_ports = sum(
                    int(node.remaining_hop1_slots()) for node in self.nodes
                )
                print(
                    f"  [Layer1 {self.phase} Completion Progress] "
                    f"decisions={self.decisions}, repairs={self.repair_attempts}, "
                    f"open_ports={open_ports}",
                    flush=True,
                )

        return {
            'solver': 'diversity_first_greedy_bounded_repair',
            'phase': str(self.phase),
            'initial_open_ports': int(initial_open_ports),
            'final_open_ports': int(self._phase_open_ports()),
            'decisions': int(self.decisions),
            'backtracks': int(self.repair_rewinds),
            'repair_attempts': int(self.repair_attempts),
            'branch_points': self._branch_points(),
            'forced_decision_requested': dict(self.forced_decision or {}),
            'forced_decision_applied': any(
                bool(row.get('forced_decision', False))
                for row in self._decision_stack
            ),
            'elapsed_s': float(time.perf_counter() - self.started_at),
            'stop_reason': 'rule_valid_graph_found',
        }


class Layer1Assigner:
    """Layer1的1-hop分配器"""

    def __init__(self,
                 device: str = 'cpu',
                 E_SU_tensor: torch.Tensor = None,
                 layer0_estimator=None,
                 deterministic: bool = True):
        """
        初始化Layer1分配器
        """
        self.device = device
        self.E_SU = E_SU_tensor.to(device) if E_SU_tensor is not None else E_SU.to(device)
        self.deterministic = bool(deterministic)
        self._current_E_target: Optional[torch.Tensor] = None
        self.layer0_estimator = layer0_estimator
        self._hop1_adjuster: Optional[Hop1Adjuster] = None
        self._validator = Layer1GraphValidator(self)
        self._runtime_nodes_ref: Optional[List[_NodeV3]] = None
        self._node_ids_by_su: Dict[int, List[int]] = {}
        self._available_ids_by_su: Dict[int, Set[int]] = {}
        self._port_pattern_cache: Dict[Tuple[int, Optional[int]], Any] = {}
        self._edge_semantics_cache: Dict[Tuple[int, int, Optional[int]], bool] = {}
        self._can_add_hop1_cache: Dict[Tuple[Any, Any], bool] = {}
        self._exact_hop1_shift_constraints_enabled = False
        self._exact_hop1_constraint_lib_path: Optional[str] = None
        self._exact_hop1_signature_set: Set[Tuple[int, Tuple[int, ...]]] = set()
        self._exact_hop1_mu_by_signature: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        self._exact_hop1_signatures_by_center_degree: Dict[
            Tuple[int, int], Tuple[Tuple[int, ...], ...]
        ] = {}
        self._exact_hop1_partial_cache: Dict[Tuple[int, int, Tuple[int, ...]], bool] = {}
        self._last_hop1_adjust_diagnostics: Dict[str, Any] = {}
        self._carbon_su_set = {
            int(i)
            for i in range(int(self.E_SU.shape[0]))
            if float(self.E_SU[int(i), 0].detach().cpu().item()) > 0.0
        }
    def _reset_layer1_search_caches(self) -> None:
        self._edge_semantics_cache = {}
        self._can_add_hop1_cache = {}

    def _configure_exact_hop1_shift_constraints(
        self,
        lib_path: Optional[str],
    ) -> None:
        """Load the exact Layer1 shift keys without selecting any 2-hop template."""
        self._exact_hop1_shift_constraints_enabled = False
        self._exact_hop1_constraint_lib_path = None
        self._exact_hop1_signature_set = set()
        self._exact_hop1_mu_by_signature = {}
        self._exact_hop1_signatures_by_center_degree = {}
        self._exact_hop1_partial_cache = {}
        self._reset_layer1_search_caches()
        if not lib_path:
            return

        self._hop1_adjuster = Hop1Adjuster(
            template_lib_path=str(lib_path),
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            enforce_external_requirements=False,
            exact_hop1_required_centers=set(self._carbon_su_set),
            carbon_su_set=set(self._carbon_su_set),
            device=self.device,
        )
        median_index = self._hop1_adjuster.get_exact_hop1_median_index()
        signatures = {
            (
                int(center_su),
                tuple(sorted(int(x) for x in tuple(hop1_ms))),
            )
            for center_su, hop1_ms in median_index.keys()
        }
        if not signatures:
            raise RuntimeError(
                f"layer1_exact_hop1_shift_index_empty: lib_path={lib_path}"
            )

        grouped: Dict[Tuple[int, int], List[Tuple[int, ...]]] = defaultdict(list)
        for center_su, hop1_ms in sorted(signatures):
            grouped[(int(center_su), int(len(hop1_ms)))].append(tuple(hop1_ms))
        self._exact_hop1_signature_set = set(signatures)
        self._exact_hop1_mu_by_signature = {
            (
                int(center_su),
                tuple(sorted(int(x) for x in tuple(hop1_ms))),
            ): float(mu)
            for (center_su, hop1_ms), mu in median_index.items()
        }
        self._exact_hop1_signatures_by_center_degree = {
            key: tuple(values)
            for key, values in grouped.items()
        }
        self._exact_hop1_constraint_lib_path = str(lib_path)
        self._exact_hop1_shift_constraints_enabled = True

    def _node_exact_hop1_shift_state_ok(
        self,
        node: _NodeV3,
        *,
        proposed_neighbor_su: Optional[int] = None,
        require_full: bool = False,
    ) -> bool:
        """Check that a partial motif can still reach an exact Layer1 shift key."""
        if not bool(self._exact_hop1_shift_constraints_enabled):
            return True
        center_su = int(getattr(node, 'su_type', -1))
        if center_su not in self._carbon_su_set:
            return True

        target_degree = self._node_target_degree(node)
        if target_degree is None:
            try:
                target_degree = int(node.get_max_degree())
            except Exception:
                return False
        current = list(
            int(x)
            for x in Counter(getattr(node, 'hop1_su', {}) or {}).elements()
        )
        if proposed_neighbor_su is not None:
            current.append(int(proposed_neighbor_su))
        partial = tuple(sorted(current))
        degree = int(target_degree)
        if len(partial) > degree:
            return False
        if bool(require_full) or len(partial) == degree:
            return (int(center_su), tuple(partial)) in self._exact_hop1_signature_set

        cache_key = (int(center_su), int(degree), tuple(partial))
        cached = self._exact_hop1_partial_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        partial_counts = Counter(partial)
        options = self._exact_hop1_signatures_by_center_degree.get(
            (int(center_su), int(degree)),
            (),
        )
        feasible = any(
            not bool(partial_counts - Counter(signature))
            for signature in options
        )
        self._exact_hop1_partial_cache[cache_key] = bool(feasible)
        return bool(feasible)

    def _runtime_index_matches(self, nodes: List[_NodeV3]) -> bool:
        return nodes is self._runtime_nodes_ref

    def _refresh_runtime_node_entry(self, nodes: List[_NodeV3], node_id: int) -> None:
        if not self._runtime_index_matches(nodes):
            return
        if int(node_id) < 0 or int(node_id) >= len(nodes):
            return
        node = nodes[int(node_id)]
        su = int(getattr(node, 'su_type', -1))
        if su < 0:
            return

        available_bucket = self._available_ids_by_su.setdefault(int(su), set())
        available_bucket.discard(int(node_id))

        if int(node.remaining_hop1_slots()) > 0:
            available_bucket.add(int(node_id))

    def _init_runtime_node_index(self, nodes: List[_NodeV3]) -> None:
        self._reset_layer1_search_caches()
        self._runtime_nodes_ref = nodes
        ids_by_su: Dict[int, List[int]] = defaultdict(list)
        self._available_ids_by_su = defaultdict(set)
        for node in nodes:
            nid = int(getattr(node, 'global_id', -1))
            su = int(getattr(node, 'su_type', -1))
            if nid < 0 or su < 0:
                continue
            ids_by_su[int(su)].append(int(nid))
        self._node_ids_by_su = {
            int(su): sorted(int(x) for x in ids)
            for su, ids in ids_by_su.items()
        }
        for nid in range(len(nodes)):
            self._refresh_runtime_node_entry(nodes, int(nid))

    def _stable_node_order(self, nodes: List[_NodeV3]) -> List[_NodeV3]:
        ordered = list(nodes)
        ordered.sort(key=lambda n: (int(getattr(n, 'global_id', -1)), int(getattr(n, 'su_type', -1))))
        return ordered

    def _canonical_node_order(self, nodes: List[_NodeV3], salt: int = 0) -> List[_NodeV3]:
        ordered = list(nodes)
        if bool(self.deterministic):
            ordered = self._stable_node_order(ordered)
            if ordered:
                offset = int(salt) % len(ordered)
                if offset:
                    ordered = ordered[offset:] + ordered[:offset]
            return ordered
        random.shuffle(ordered)
        return ordered

    def _histogram_from_nodes(self, nodes: List[_NodeV3]) -> torch.Tensor:
        """从节点列表构建SU直方图"""
        H = torch.zeros(NUM_SU_TYPES, dtype=torch.float, device=self.device)
        for n in nodes:
            try:
                su = int(n.su_type)
                if 0 <= su < NUM_SU_TYPES:
                    H[su] += 1
            except Exception as e:
                import logging
                logging.warning(f"Failed to count SU for node {getattr(n, 'global_id', '?')}: {e}")
        return H

    def _get_special_degree_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[int, int]]:
        raw_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_meta = dict(layer0_meta.get('special_degree_meta', {}) or {})
        except Exception:
            raw_meta = {}

        H_cpu = H_init.detach().cpu().long()
        meta: Dict[int, Dict[int, int]] = {}
        for su_type, priors in SPECIAL_DEGREE_PRIORS.items():
            total = int(H_cpu[int(su_type)].item()) if int(H_cpu.numel()) > int(su_type) else 0
            src = raw_meta.get(int(su_type), raw_meta.get(str(int(su_type)), {}))
            degree_counts: Dict[int, int] = {}
            if isinstance(src, dict):
                for degree in priors.keys():
                    degree_counts[int(degree)] = max(
                        0,
                        int(src.get(int(degree), src.get(str(int(degree)), 0)) or 0),
                    )
            if sum(degree_counts.values()) != int(total):
                degree_counts = allocate_special_degree_counts(int(total), priors)
            meta[int(su_type)] = {
                int(degree): int(cnt)
                for degree, cnt in degree_counts.items()
            }
        return meta

    def _get_special_partition_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[str, Dict[int, int]]]:
        raw_meta = {}
        layer0_meta = {}
        raw_degree_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_meta = dict(layer0_meta.get('special_partition_meta', {}) or {})
            raw_degree_meta = dict(layer0_meta.get('special_degree_meta', {}) or {})
        except Exception:
            raw_meta = {}
            layer0_meta = {}
            raw_degree_meta = {}

        H_cpu = H_init.detach().cpu().long()
        total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
        if int(total_19) <= 0:
            return {}

        src_19 = dict(raw_meta.get(19, raw_meta.get('19', {})) or {})
        raw_degree_19 = dict(raw_degree_meta.get(19, raw_degree_meta.get('19', {})) or {})
        if int(sum(int(raw_degree_19.get(int(deg), raw_degree_19.get(str(int(deg)), 0)) or 0) for deg in [1, 2, 3])) != int(total_19):
            raw_degree_19 = allocate_special_degree_counts(int(total_19), SPECIAL_DEGREE_PRIORS[19])
        o_base_19 = max(0, min(int(layer0_meta.get('o_base_19', total_19)), int(total_19)))
        s_reserved_19 = max(0, min(int(layer0_meta.get('s_reserved_19', max(0, total_19 - o_base_19))), int(total_19 - o_base_19)))
        if int(o_base_19 + s_reserved_19) < int(total_19):
            o_base_19 += int(total_19 - (o_base_19 + s_reserved_19))
        rebuilt = rebuild_su19_partition_meta(
            total_19=int(total_19),
            o_base_19=int(o_base_19),
            s_reserved_19=int(s_reserved_19),
            special_degree_meta_19={
                int(deg): max(0, int(raw_degree_19.get(int(deg), raw_degree_19.get(str(int(deg)), 0)) or 0))
                for deg in [1, 2, 3]
            },
            existing_partition_meta_19=src_19,
        )
        return {
            19: {
                'ether': {int(deg): max(0, int(rebuilt.get('ether', {}).get(int(deg), 0) or 0)) for deg in [1, 2, 3]},
                'thio': {int(deg): max(0, int(rebuilt.get('thio', {}).get(int(deg), 0) or 0)) for deg in [1, 2, 3]},
            }
        }

    def _get_special_anchor_mode_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[str, Dict[int, int]]]:
        layer0_meta = {}
        raw_mode_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_mode_meta = dict(layer0_meta.get('special_anchor_mode_meta', {}) or {})
        except Exception:
            layer0_meta = {}
            raw_mode_meta = {}

        H_cpu = H_init.detach().cpu().long()
        degree_meta = self._get_special_degree_meta(H_init)
        partition_meta = self._get_special_partition_meta(H_init)
        out: Dict[int, Dict[str, Dict[int, int]]] = {}

        total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
        if int(total_19) > 0:
            ether_counts = dict((partition_meta.get(19, {}) or {}).get('ether', {}) or {})
            thio_counts = dict((partition_meta.get(19, {}) or {}).get('thio', {}) or {})
            raw_19 = dict(raw_mode_meta.get(19, raw_mode_meta.get('19', {})) or {})
            ether_double_src = dict(raw_19.get('ether_double', {}) or {})
            thio_double_src = dict(raw_19.get('thio_double', {}) or {})
            ether_double = {
                int(deg): max(0, int(ether_double_src.get(int(deg), ether_double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            thio_double = {
                int(deg): max(0, int(thio_double_src.get(int(deg), thio_double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            for deg in [2, 3]:
                ether_double[int(deg)] = min(int(ether_double[int(deg)]), int(ether_counts.get(int(deg), 0)))
                thio_double[int(deg)] = min(int(thio_double[int(deg)]), int(thio_counts.get(int(deg), 0)))
            ether_single = {
                int(deg): max(0, int(ether_counts.get(int(deg), 0)) - int(ether_double.get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            thio_single = {
                int(deg): max(0, int(thio_counts.get(int(deg), 0)) - int(thio_double.get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            out[19] = {
                'ether_single': {int(deg): int(ether_single.get(int(deg), 0)) for deg in [1, 2, 3]},
                'ether_double': {int(deg): int(ether_double.get(int(deg), 0)) for deg in [2, 3]},
                'thio_single': {int(deg): int(thio_single.get(int(deg), 0)) for deg in [1, 2, 3]},
                'thio_double': {int(deg): int(thio_double.get(int(deg), 0)) for deg in [2, 3]},
            }

        total_20 = int(H_cpu[20].item()) if int(H_cpu.numel()) > 20 else 0
        if int(total_20) > 0:
            raw_20 = dict(raw_mode_meta.get(20, raw_mode_meta.get('20', {})) or {})
            total_counts_20 = {
                int(deg): max(0, int(dict(degree_meta.get(20, {}) or {}).get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            double_src = dict(raw_20.get('double', {}) or {})
            double_counts = {
                int(deg): max(0, int(double_src.get(int(deg), double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            for deg in [2, 3]:
                double_counts[int(deg)] = min(int(double_counts[int(deg)]), int(total_counts_20.get(int(deg), 0)))
            required_20_edges = int(max(
                0,
                int(H_cpu[0].item()) + 2 * int(H_cpu[27].item()) - int(H_cpu[6].item()),
            )) if int(H_cpu.numel()) > 27 else int(total_20)
            total_nodes_20 = int(sum(int(v) for v in total_counts_20.values()))
            if int(total_nodes_20) <= 0:
                total_nodes_20 = int(total_20)
            target_double_total = int(max(0, int(required_20_edges) - int(total_nodes_20)))
            target_double_total = min(
                int(target_double_total),
                int(total_counts_20.get(2, 0)) + int(total_counts_20.get(3, 0)),
            )
            current_double_total = int(sum(int(v) for v in double_counts.values()))
            if int(current_double_total) < int(target_double_total):
                deficit = int(target_double_total) - int(current_double_total)
                for deg in [3, 2]:
                    room = max(0, int(total_counts_20.get(int(deg), 0)) - int(double_counts.get(int(deg), 0)))
                    take = min(int(deficit), int(room))
                    if int(take) > 0:
                        double_counts[int(deg)] = int(double_counts.get(int(deg), 0)) + int(take)
                        deficit -= int(take)
                    if int(deficit) <= 0:
                        break
            elif int(current_double_total) > int(target_double_total):
                excess = int(current_double_total) - int(target_double_total)
                for deg in [2, 3]:
                    take = min(int(excess), int(double_counts.get(int(deg), 0)))
                    if int(take) > 0:
                        double_counts[int(deg)] = int(double_counts.get(int(deg), 0)) - int(take)
                        excess -= int(take)
                    if int(excess) <= 0:
                        break
            single_counts = {
                int(deg): max(0, int(total_counts_20.get(int(deg), 0)) - int(double_counts.get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            out[20] = {
                'single': {int(deg): int(single_counts.get(int(deg), 0)) for deg in [1, 2, 3]},
                'double': {int(deg): int(double_counts.get(int(deg), 0)) for deg in [2, 3]},
            }
        total_21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
        if int(total_21) > 0:
            raw_21 = dict(raw_mode_meta.get(21, raw_mode_meta.get('21', {})) or {})
            total_counts_21 = dict(degree_meta.get(21, {}) or {})
            raw_single_21 = dict(raw_21.get('single', {}) or {})
            raw_double_21 = dict(raw_21.get('double', {}) or {})
            if int(sum(int(total_counts_21.get(int(deg), 0)) for deg in [2, 3])) > 0:
                single_counts = {
                    int(deg): max(0, int(total_counts_21.get(int(deg), 0)))
                    for deg in [2, 3]
                }
            else:
                single_counts = {
                    int(deg): (
                        max(0, int(raw_single_21.get(int(deg), raw_single_21.get(str(deg), 0)) or 0)) +
                        max(0, int(raw_double_21.get(int(deg), raw_double_21.get(str(deg), 0)) or 0))
                    )
                    for deg in [2, 3]
                }
                if int(sum(single_counts.values())) <= 0:
                    single_counts[2] = int(total_21)
            excess = max(0, int(sum(single_counts.values())) - int(total_21))
            for deg in [2, 3]:
                if int(excess) <= 0:
                    break
                take = min(int(excess), int(single_counts.get(int(deg), 0)))
                single_counts[int(deg)] = max(0, int(single_counts.get(int(deg), 0)) - int(take))
                excess -= int(take)
            deficit = max(0, int(total_21) - int(sum(single_counts.values())))
            if int(deficit) > 0:
                single_counts[2] = int(single_counts.get(2, 0)) + int(deficit)
            out[21] = {
                'single': {int(deg): int(single_counts.get(int(deg), 0)) for deg in [2, 3]},
                'double': {2: 0, 3: 0},
            }
        return out

    @staticmethod
    def _build_special_degree_sequence(su_type: int,
                                       count: int,
                                       special_degree_meta: Dict[int, Dict[int, int]]) -> List[int]:
        if int(su_type) not in SPECIAL_DEGREE_PRIORS:
            return []
        seq: List[int] = []
        degree_map = dict(special_degree_meta.get(int(su_type), {}) or {})
        for degree in sorted(degree_map.keys()):
            seq.extend([int(degree)] * max(0, int(degree_map.get(int(degree), 0))))
        count_i = max(0, int(count))
        if len(seq) < count_i:
            max_degree = max([int(k) for k in degree_map.keys()] or [0])
            if int(max_degree) <= 0:
                return []
            seq.extend([int(max_degree)] * max(0, count_i - len(seq)))
        return seq[:count_i]

    @staticmethod
    def _build_special_partition_sequence_19(count: int,
                                             partition_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, Optional[str]]]:
        seq: List[Tuple[int, Optional[str]]] = []
        part_meta = dict(partition_meta.get(19, {}) or {})
        for part_name in ('thio', 'ether'):
            degree_map = dict(part_meta.get(str(part_name), {}) or {})
            for degree in sorted(degree_map.keys()):
                seq.extend([(int(degree), str(part_name))] * max(0, int(degree_map.get(int(degree), 0))))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_19(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        seq: List[Tuple[int, Optional[str], int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(19, {}) or {})
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('thio_single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'thio', 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('thio_double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'thio', 2, 'double')] * max(0, cnt))
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('ether_single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'ether', 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('ether_double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'ether', 2, 'double')] * max(0, cnt))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_20(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, int, Optional[str]]]:
        seq: List[Tuple[int, int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(20, {}) or {})
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 2, 'double')] * max(0, cnt))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_21(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, int, Optional[str]]]:
        seq: List[Tuple[int, int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(21, {}) or {})
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 1, 'single')] * max(0, cnt))
        return seq[: max(0, int(count))]

    def _current_neighbor_types(self, nodes: List[_NodeV3], node: _NodeV3) -> List[int]:
        return [int(nodes[nid].su_type) for nid in node.hop1_ids if 0 <= int(nid) < len(nodes)]

    @staticmethod
    def _node_target_degree(node: _NodeV3) -> Optional[int]:
        try:
            val = getattr(node, 'target_hop1_degree', None)
            if val is not None:
                return int(val)
            su_type = int(getattr(node, 'su_type', -1))
            if su_type in {19, 20, 21}:
                return None
            degree = SU_CONNECTION_DEGREE.get(su_type)
            return int(degree) if degree is not None else None
        except Exception:
            return None

    def _port_sets_for_node(self, node: _NodeV3):
        su_i = int(getattr(node, 'su_type', -1))
        target_degree = self._node_target_degree(node)
        key = (int(su_i), target_degree)
        if key not in self._port_pattern_cache:
            port_sets = HOP1_PORT_COMBINATIONS.get(int(su_i))
            self._port_pattern_cache[key] = select_port_patterns_for_degree(port_sets, target_degree)
        return self._port_pattern_cache.get(key)

    def _can_add_state_signature(self, nodes: List[_NodeV3], node: _NodeV3) -> Tuple[Any, ...]:
        neighbor_ids = tuple(sorted(int(x) for x in list(getattr(node, 'hop1_ids', []) or [])))
        neighbor_states = []
        for nid in neighbor_ids:
            if 0 <= int(nid) < len(nodes):
                nb = nodes[int(nid)]
                neighbor_states.append((
                    int(getattr(nb, 'su_type', -1)),
                    self._node_target_degree(nb),
                    self._node_anchor_partition(nb),
                    self._node_fixed_anchor_target(nb),
                    self._node_anchor_mode(nb),
                ))
        identity_sensitive = bool(
            int(getattr(node, 'su_type', -1)) in {19, 20, 21, 27, 29, 31}
            or any(int(state[0]) in {19, 20, 21, 27, 29, 31} for state in neighbor_states)
        )
        return (
            int(getattr(node, 'global_id', -1)) if identity_sensitive else None,
            int(getattr(node, 'su_type', -1)),
            self._node_target_degree(node),
            self._node_anchor_partition(node),
            self._node_fixed_anchor_target(node),
            self._node_anchor_mode(node),
            int(node.remaining_hop1_slots()),
            neighbor_ids if identity_sensitive else None,
            tuple(sorted(neighbor_states, key=repr)),
        )

    @staticmethod
    def _node_anchor_partition(node: _NodeV3) -> Optional[str]:
        try:
            val = getattr(node, 'special_anchor_partition', None)
            return str(val) if val is not None else None
        except Exception:
            return None

    @staticmethod
    def _node_anchor_mode(node: _NodeV3) -> Optional[str]:
        try:
            val = getattr(node, 'special_anchor_mode', None)
            if val is None:
                return None
            mode = str(val)
            if str(mode).startswith('double'):
                return 'double'
            if str(mode).startswith('single'):
                return 'single'
            return mode
        except Exception:
            return None

    @staticmethod
    def _node_fixed_anchor_target(node: _NodeV3) -> int:
        try:
            val = getattr(node, 'target_fixed_anchor_count', None)
            if val is not None:
                return max(0, int(val))
        except Exception:
            pass
        try:
            mode = str(getattr(node, 'special_anchor_mode', None) or '')
            if str(mode).startswith('double'):
                return 2
            if str(mode) in {'single', 'thio'}:
                return 1
        except Exception:
            pass
        su_i = int(getattr(node, 'su_type', -1))
        if int(su_i) == 19:
            if Layer1Assigner._node_anchor_partition(node) in {'ether', 'thio'}:
                return 1
        if int(su_i) == 20:
            return 1
        if int(su_i) == 21:
            return 1
        if int(su_i) in set(int(x) for x in SU_EXTERNAL_CONNECTIONS.keys()):
            return 1
        return 0

    def _sort_special_degree_nodes(self, nodes: List[_NodeV3]) -> List[_NodeV3]:
        return sorted(
            list(nodes),
            key=lambda n: (
                0 if self._node_anchor_partition(n) == 'thio' else 1,
                self._node_target_degree(n) if self._node_target_degree(n) is not None else 99,
                int(getattr(n, 'global_id', 0)),
            ),
        )

    def _required_external_candidates_for_node(self, node: _NodeV3) -> List[int]:
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
            return []
        return [int(x) for x in list(SU_EXTERNAL_CONNECTIONS.get(int(su_i), []) or [])]

    def _has_required_external_for_node(self, node: _NodeV3, nodes: List[_NodeV3]) -> bool:
        required = set(int(x) for x in self._required_external_candidates_for_node(node))
        if not required:
            return True
        target_count = max(1, int(self._node_fixed_anchor_target(node)))
        current_count = int(sum(1 for x in self._current_neighbor_types(nodes, node) if int(x) in required))
        return int(current_count) >= int(target_count)

    def _has_pending_required_external(self, node: _NodeV3, nodes: List[_NodeV3]) -> bool:
        required = self._required_external_candidates_for_node(node)
        if not required:
            return False
        return not self._has_required_external_for_node(node, nodes)

    def _required_external_count_for_node(self, node: _NodeV3, nodes: List[_NodeV3]) -> int:
        required = set(int(x) for x in self._required_external_candidates_for_node(node))
        if not required:
            return 0
        return int(sum(1 for x in self._current_neighbor_types(nodes, node) if int(x) in required))

    def _fixed_target_rank(self,
                           center: _NodeV3,
                           candidate: _NodeV3,
                           priority_list: List[int]) -> Tuple[int, int, int, int]:
        center_su = int(getattr(center, 'su_type', -1))
        cand_su = int(getattr(candidate, 'su_type', -1))
        cand_deg = self._node_target_degree(candidate)
        try:
            pri_idx = int(priority_list.index(int(cand_su)))
        except ValueError:
            pri_idx = int(len(priority_list))

        fixed_pref = 50
        if center_su in {2, 29}:
            if cand_su == 19:
                if self._node_anchor_partition(candidate) != 'ether':
                    fixed_pref = 98
                else:
                    fixed_pref = {1: 0, 2: 2, 3: 3}.get(int(cand_deg or 0), 9)
            elif cand_su == 5:
                fixed_pref = 1
        elif center_su == 28:
            if cand_su == 5:
                fixed_pref = 0
            elif cand_su == 19:
                if self._node_anchor_partition(candidate) != 'ether':
                    fixed_pref = 98
                else:
                    fixed_pref = {2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su == 31 and cand_su == 19:
            if self._node_anchor_partition(candidate) != 'thio':
                fixed_pref = 98
            else:
                fixed_pref = {1: 0, 2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su in {0, 27} and cand_su == 20:
            fixed_pref = {1: 0, 2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su == 32 and cand_su == 21:
            fixed_pref = {2: 0, 3: 1}.get(int(cand_deg or 0), 9)

        return (
            int(fixed_pref),
            int(pri_idx),
            int(candidate.remaining_hop1_slots()),
            int(getattr(candidate, 'global_id', 0)),
        )

    def _choose_fixed_candidate(self,
                                center: _NodeV3,
                                candidates: List[_NodeV3],
                                priority_list: List[int],
                                nodes: Optional[List[_NodeV3]] = None) -> Optional[_NodeV3]:
        filtered = [
            n for n in candidates
            if n.global_id != center.global_id
            and n.remaining_hop1_slots() > 0
            and n.global_id not in center.hop1_ids
        ]
        if not filtered:
            return None
        if nodes is not None:
            filtered = [n for n in filtered if self._can_add_hop1_connection(nodes, center, n)]
            if not filtered:
                return None
        ranked = sorted(
            filtered,
            key=lambda n: self._fixed_target_rank(center, n, priority_list),
        )
        return ranked[0] if ranked else None

    def _current_neighbor_nodes(self, nodes: List[_NodeV3], node: _NodeV3) -> List[_NodeV3]:
        return [
            nodes[int(nid)]
            for nid in list(getattr(node, 'hop1_ids', []) or [])
            if 0 <= int(nid) < len(nodes)
        ]

    def _is_su19_partition_compatible(self, node: _NodeV3, neighbor_su: int) -> bool:
        if int(getattr(node, 'su_type', -1)) != 19:
            return True
        part = self._node_anchor_partition(node)
        if int(neighbor_su) == 31:
            return str(part) == 'thio'
        if int(neighbor_su) in {2, 28, 29}:
            return str(part) == 'ether'
        return True

    def _violates_mode_specific_rule(self, node: _NodeV3, neighbor_su: int) -> bool:
        return bool(is_mode_specific_neighbor_forbidden(
            int(getattr(node, 'su_type', -1)),
            self._node_target_degree(node),
            int(neighbor_su),
        ))

    def _violates_special_d3_terminal_rule(self,
                                           nodes: List[_NodeV3],
                                           node: _NodeV3,
                                           neighbor_su: int) -> bool:
        target_degree = self._node_target_degree(node)
        current_neighbors = self._current_neighbor_nodes(nodes, node)
        proposed_neighbors = list(current_neighbors) + [int(neighbor_su)]
        return bool(violates_special_d3_terminal_limit(
            int(getattr(node, 'su_type', -1)),
            target_degree,
            proposed_neighbors,
        ))

    @staticmethod
    def _is_double_special_d2_node(node: _NodeV3, su_type: Optional[int] = None) -> bool:
        try:
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            return False
        if int(su_i) not in {19, 20}:
            return False
        if su_type is not None and int(su_i) != int(su_type):
            return False
        try:
            if int(getattr(node, 'target_hop1_degree', 0) or 0) != 2:
                return False
        except Exception:
            return False
        try:
            fixed_i = int(getattr(node, 'target_fixed_anchor_count', 0) or 0)
        except Exception:
            fixed_i = 0
        try:
            mode = str(getattr(node, 'special_anchor_mode', '') or '')
        except Exception:
            mode = ''
        return bool(int(fixed_i) >= 2 or str(mode).startswith('double'))

    @staticmethod
    def _is_double_special_node(node: _NodeV3,
                                su_type: Optional[int] = None,
                                allowed_degrees: Optional[Set[int]] = None) -> bool:
        try:
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            return False
        if int(su_i) not in {19, 20}:
            return False
        if su_type is not None and int(su_i) != int(su_type):
            return False
        try:
            deg_i = int(getattr(node, 'target_hop1_degree', 0) or 0)
        except Exception:
            deg_i = 0
        if allowed_degrees is not None and int(deg_i) not in set(int(x) for x in allowed_degrees):
            return False
        try:
            fixed_i = int(getattr(node, 'target_fixed_anchor_count', 0) or 0)
        except Exception:
            fixed_i = 0
        try:
            mode = str(getattr(node, 'special_anchor_mode', '') or '')
        except Exception:
            mode = ''
        return bool(int(fixed_i) >= 2 or str(mode).startswith('double'))

    @staticmethod
    def _is_single_special_d1_node(node: _NodeV3, su_type: int) -> bool:
        try:
            return bool(
                int(getattr(node, 'su_type', -1)) == int(su_type)
                and int(getattr(node, 'target_hop1_degree', 0) or 0) == 1
            )
        except Exception:
            return False

    @staticmethod
    def _is_single_special_d3_node(node: _NodeV3, su_type: int) -> bool:
        try:
            return bool(
                int(getattr(node, 'su_type', -1)) == int(su_type)
                and int(getattr(node, 'target_hop1_degree', 0) or 0) == 3
                and int(getattr(node, 'target_fixed_anchor_count', 0) or 0) <= 1
                and not str(getattr(node, 'special_anchor_mode', '') or '').startswith('double')
            )
        except Exception:
            return False

    @staticmethod
    def _is_terminal_like_neighbor(node: _NodeV3) -> bool:
        return bool(is_terminal_port_node(node))

    def _node_has_terminal_neighbor_limit(self, node: _NodeV3) -> bool:
        return True

    def _terminal_neighbor_count(self,
                                 nodes: List[_NodeV3],
                                 node: _NodeV3,
                                 node1: Optional[_NodeV3] = None,
                                 node2: Optional[_NodeV3] = None) -> int:
        return int(sum(
            1 for nb in self._neighbors_with_proposed_edge(nodes, node, node1, node2)
            if self._is_terminal_like_neighbor(nb)
        ))

    def _violates_terminal_neighbor_limit(self,
                                          nodes: List[_NodeV3],
                                          node: _NodeV3,
                                          node1: Optional[_NodeV3] = None,
                                          node2: Optional[_NodeV3] = None) -> bool:
        if not self._node_has_terminal_neighbor_limit(node):
            return False
        return int(self._terminal_neighbor_count(nodes, node, node1, node2)) >= 2

    def _fixed_connector_special_target_su(self, connector: _NodeV3) -> Optional[int]:
        su_i = int(getattr(connector, 'su_type', -1))
        if int(su_i) == 27:
            return 20
        if int(su_i) in {29, 31}:
            return 19
        return None

    def _violates_fixed_connector_special_end_rules(self,
                                                    nodes: List[_NodeV3],
                                                    connector: _NodeV3,
                                                    node1: Optional[_NodeV3] = None,
                                                    node2: Optional[_NodeV3] = None) -> bool:
        target_su = self._fixed_connector_special_target_su(connector)
        if target_su is None:
            return False
        neighbors = self._neighbors_with_proposed_edge(nodes, connector, node1, node2)
        double_count = int(sum(
            1 for nb in neighbors
            if self._is_double_special_node(nb, int(target_su), allowed_degrees={2, 3})
        ))
        if int(double_count) >= 2:
            return True
        has_single_d3 = any(self._is_single_special_d3_node(nb, int(target_su)) for nb in neighbors)
        return bool(has_single_d3 and int(double_count) >= 1)

    @staticmethod
    def _is_p_connector_node(node: _NodeV3) -> bool:
        try:
            return int(getattr(node, 'su_type', -1)) in {0, 2, 27, 28, 29, 31}
        except Exception:
            return False

    def _neighbors_with_proposed_edge(self,
                                      nodes: List[_NodeV3],
                                      node: _NodeV3,
                                      node1: Optional[_NodeV3] = None,
                                      node2: Optional[_NodeV3] = None) -> List[_NodeV3]:
        try:
            gid = int(getattr(node, 'global_id', -1))
        except Exception:
            gid = -1
        ids = [
            int(nid)
            for nid in list(getattr(node, 'hop1_ids', []) or [])
            if 0 <= int(nid) < len(nodes)
        ]
        if node1 is not None and node2 is not None:
            try:
                id1 = int(getattr(node1, 'global_id', -1))
                id2 = int(getattr(node2, 'global_id', -1))
            except Exception:
                id1 = id2 = -1
            if gid == id1 and 0 <= id2 < len(nodes):
                ids.append(int(id2))
            elif gid == id2 and 0 <= id1 < len(nodes):
                ids.append(int(id1))
        out: List[_NodeV3] = []
        seen: Set[int] = set()
        for nid in ids:
            if int(nid) in seen:
                continue
            seen.add(int(nid))
            out.append(nodes[int(nid)])
        return out

    def _is_p_connector_topology(self,
                                 connector: _NodeV3,
                                 neighbors: List[_NodeV3]) -> bool:
        su_i = int(getattr(connector, 'su_type', -1))
        if int(su_i) == 0:
            return bool(
                any(int(getattr(nb, 'su_type', -1)) == 22 for nb in list(neighbors or []))
                and any(self._is_double_special_d2_node(nb, 20) for nb in list(neighbors or []))
            )
        if int(su_i) == 2:
            return bool(
                any(int(getattr(nb, 'su_type', -1)) == 22 for nb in list(neighbors or []))
                and any(self._is_double_special_d2_node(nb, 19) for nb in list(neighbors or []))
            )
        if int(su_i) == 27:
            return bool(
                any(self._is_single_special_d1_node(nb, 20) for nb in list(neighbors or []))
                and any(self._is_double_special_d2_node(nb, 20) for nb in list(neighbors or []))
            )
        if int(su_i) == 28:
            return bool(any(self._is_double_special_d2_node(nb, 19) for nb in list(neighbors or [])))
        if int(su_i) in {29, 31}:
            return bool(
                any(self._is_single_special_d1_node(nb, 19) for nb in list(neighbors or []))
                and any(self._is_double_special_d2_node(nb, 19) for nb in list(neighbors or []))
            )
        return False

    def _forbidden_p_pair_rows(self,
                               nodes: List[_NodeV3],
                               node1: Optional[_NodeV3] = None,
                               node2: Optional[_NodeV3] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_double_ids: Set[int] = set()
        for node in list(nodes or []):
            if not self._is_double_special_d2_node(node):
                continue
            gid = int(getattr(node, 'global_id', -1))
            if gid in seen_double_ids:
                continue
            seen_double_ids.add(int(gid))
            p_connectors: List[_NodeV3] = []
            for nb in self._neighbors_with_proposed_edge(nodes, node, node1, node2):
                if not self._is_p_connector_node(nb):
                    continue
                nb_neighbors = self._neighbors_with_proposed_edge(nodes, nb, node1, node2)
                if self._is_p_connector_topology(nb, nb_neighbors):
                    p_connectors.append(nb)
            p_ids = sorted(set(int(getattr(nb, 'global_id', -1)) for nb in p_connectors))
            if len(p_ids) >= 2:
                rows.append({
                    'double_id': int(gid),
                    'double_su': int(getattr(node, 'su_type', -1)),
                    'p_connector_ids': list(p_ids),
                    'p_connector_su': [
                        int(getattr(nodes[int(pid)], 'su_type', -1))
                        for pid in p_ids
                        if 0 <= int(pid) < len(nodes)
                    ],
                })
        return rows

    def _p_connector_ids_for_double(self,
                                    nodes: List[_NodeV3],
                                    double_node: _NodeV3,
                                    node1: Optional[_NodeV3] = None,
                                    node2: Optional[_NodeV3] = None) -> List[int]:
        p_ids: Set[int] = set()
        for nb in self._neighbors_with_proposed_edge(nodes, double_node, node1, node2):
            if not self._is_p_connector_node(nb):
                continue
            nb_neighbors = self._neighbors_with_proposed_edge(nodes, nb, node1, node2)
            if self._is_p_connector_topology(nb, nb_neighbors):
                p_ids.add(int(getattr(nb, 'global_id', -1)))
        return sorted(int(x) for x in p_ids if int(x) >= 0)

    def _violates_forbidden_p_pair_connection(self,
                                              nodes: List[_NodeV3],
                                              node1: _NodeV3,
                                              node2: _NodeV3) -> bool:
        affected_double_ids: Set[int] = set()
        for endpoint in (node1, node2):
            if self._is_double_special_d2_node(endpoint):
                affected_double_ids.add(int(getattr(endpoint, 'global_id', -1)))
            if self._is_p_connector_node(endpoint):
                for nb in self._neighbors_with_proposed_edge(nodes, endpoint, node1, node2):
                    if self._is_double_special_d2_node(nb):
                        affected_double_ids.add(int(getattr(nb, 'global_id', -1)))
        if not affected_double_ids:
            return False
        for double_id in sorted(int(x) for x in affected_double_ids if 0 <= int(x) < len(nodes)):
            double_node = nodes[int(double_id)]
            if len(self._p_connector_ids_for_double(nodes, double_node, node1, node2)) >= 2:
                return True
        return False

    def _fixed_connector_assignment_ok(self,
                                       nodes: List[_NodeV3],
                                       owner: _NodeV3,
                                       target: _NodeV3) -> bool:
        owner_su = int(getattr(owner, 'su_type', -1))
        target_su = int(getattr(target, 'su_type', -1))
        if not self._is_su19_partition_compatible(owner, target_su):
            return False
        if not self._is_su19_partition_compatible(target, owner_su):
            return False
        if self._violates_mode_specific_rule(owner, target_su):
            return False
        if self._violates_mode_specific_rule(target, owner_su):
            return False
        if self._violates_special_d3_terminal_rule(nodes, owner, target_su):
            return False
        if self._violates_special_d3_terminal_rule(nodes, target, owner_su):
            return False
        if self._violates_terminal_neighbor_limit(nodes, owner, owner, target):
            return False
        if self._violates_terminal_neighbor_limit(nodes, target, owner, target):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, owner, owner, target):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, target, owner, target):
            return False
        owner_neighbors = self._current_neighbor_nodes(nodes, owner) + [target]
        if has_forbidden_fixed_connector_special_combo(owner, owner_neighbors):
            return False
        target_neighbors = self._current_neighbor_nodes(nodes, target) + [owner]
        if has_forbidden_fixed_connector_special_combo(target, target_neighbors):
            return False
        if self._violates_forbidden_p_pair_connection(nodes, owner, target):
            return False
        return True

    def _collect_fixed_connection_markers(self, nodes: List[_NodeV3]) -> Dict[str, Set[int]]:
        markers: Dict[str, Set[Any]] = {
            'carbonyl_9_ids': set(),
            'ether_5_ids': set(),
            'ether_19_ids': set(),
            'ether_5_edges': set(),
            'ether_19_edges': set(),
            'sulfur_7_ids': set(),
            'sulfur_19_ids': set(),
            'sulfur_7_edges': set(),
            'sulfur_19_edges': set(),
            'amine_6_ids': set(),
            'amine_20_ids': set(),
            'amine_6_edges': set(),
            'amine_20_edges': set(),
            'halogen_8_ids': set(),
            'halogen_21_ids': set(),
            'halogen_8_edges': set(),
            'halogen_21_edges': set(),
            'invalid_su19_31_ids': set(),
            'invalid_su19_o_ids': set(),
            'unpartitioned_su19_external_ids': set(),
        }
        for node in nodes:
            gid = int(getattr(node, 'global_id', -1))
            su = int(getattr(node, 'su_type', -1))
            neighbor_ids = [int(nid) for nid in list(getattr(node, 'hop1_ids', []) or []) if 0 <= int(nid) < len(nodes)]
            neighbor_types = [int(nodes[int(nid)].su_type) for nid in neighbor_ids]
            neigh = set(int(x) for x in neighbor_types)
            if su == 9 and any(int(x) in {0, 1, 2, 3} for x in neigh):
                markers['carbonyl_9_ids'].add(gid)
            elif su == 5:
                used_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {2, 28, 29}]
                if used_nb_ids:
                    markers['ether_5_ids'].add(gid)
                for nb_id in used_nb_ids:
                    markers['ether_5_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 19:
                part = self._node_anchor_partition(node)
                sulfur_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 31]
                if sulfur_nb_ids:
                    if str(part) == 'thio':
                        markers['sulfur_19_ids'].add(gid)
                        for nb_id in sulfur_nb_ids:
                            markers['sulfur_19_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
                    elif str(part) == 'ether':
                        markers['invalid_su19_31_ids'].add(gid)
                    else:
                        markers['invalid_su19_31_ids'].add(gid)
                        markers['unpartitioned_su19_external_ids'].add(gid)
                oxygen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {2, 28, 29}]
                if oxygen_nb_ids:
                    if str(part) == 'ether':
                        markers['ether_19_ids'].add(gid)
                        for nb_id in oxygen_nb_ids:
                            markers['ether_19_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
                    elif str(part) == 'thio':
                        markers['invalid_su19_o_ids'].add(gid)
                    else:
                        markers['invalid_su19_o_ids'].add(gid)
                        markers['unpartitioned_su19_external_ids'].add(gid)
            elif su == 7:
                sulfur_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 31]
                if sulfur_nb_ids:
                    markers['sulfur_7_ids'].add(gid)
                for nb_id in sulfur_nb_ids:
                    markers['sulfur_7_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 6:
                amine_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {0, 27}]
                if amine_nb_ids:
                    markers['amine_6_ids'].add(gid)
                for nb_id in amine_nb_ids:
                    markers['amine_6_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 20:
                amine_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {0, 27}]
                if amine_nb_ids:
                    markers['amine_20_ids'].add(gid)
                for nb_id in amine_nb_ids:
                    markers['amine_20_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 8:
                halogen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 32]
                if halogen_nb_ids:
                    markers['halogen_8_ids'].add(gid)
                for nb_id in halogen_nb_ids:
                    markers['halogen_8_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 21:
                halogen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 32]
                if halogen_nb_ids:
                    markers['halogen_21_ids'].add(gid)
                for nb_id in halogen_nb_ids:
                    markers['halogen_21_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
        return markers

    def _fixed_anchor_invariant_errors(self,
                                       nodes: List[_NodeV3],
                                       check_thio: bool = True,
                                       check_ether: bool = True,
                                       check_amine: bool = True,
                                       check_halogen: bool = True) -> List[str]:
        errors: List[str] = []
        H_actual = self._histogram_from_nodes(nodes).detach().cpu().long()
        markers = self._collect_fixed_connection_markers(nodes)

        invalid_31 = sorted(int(x) for x in set(markers.get('invalid_su19_31_ids', set()) or set()))
        invalid_o = sorted(int(x) for x in set(markers.get('invalid_su19_o_ids', set()) or set()))
        unpartitioned = sorted(int(x) for x in set(markers.get('unpartitioned_su19_external_ids', set()) or set()))
        if invalid_31:
            errors.append(f"存在非thio的SU19连接到SU31: ids={invalid_31}")
        if invalid_o:
            errors.append(f"存在非ether的SU19连接到O锚点(2/28/29): ids={invalid_o}")
        if unpartitioned:
            errors.append(f"存在未分区的SU19参与固定外接锚点: ids={unpartitioned}")

        if bool(check_thio):
            n31 = int(H_actual[31].item()) if int(H_actual.numel()) > 31 else 0
            n7 = int(H_actual[7].item()) if int(H_actual.numel()) > 7 else 0
            expected_thio = int(2 * n31)
            used_7 = int(len(markers.get('sulfur_7_edges', set()) or set()))
            used_19 = int(len(markers.get('sulfur_19_edges', set()) or set()))
            target_19 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 19 and self._node_anchor_partition(node) == 'thio'
            ))
            if int(n7 + target_19) != int(expected_thio):
                errors.append(
                    f"thio固定池数量不匹配: SU7({n7}) + SU19(thio_edges)({target_19}) != 2*SU31({expected_thio})"
                )
            if int(used_7) != int(n7):
                errors.append(f"SU7外接31边数不匹配: used={used_7} vs total={n7}")
            if int(used_19) != int(target_19):
                errors.append(f"SU19(thio)外接31边数不匹配: used={used_19} vs target={target_19}")
            if int(used_7 + used_19) != int(expected_thio):
                errors.append(
                    f"固定连接总量不匹配: SU7(连31)={used_7}, SU19(thio)(连31)={used_19}, SU31需求={expected_thio}"
                )

        if bool(check_ether):
            n2 = int(H_actual[2].item()) if int(H_actual.numel()) > 2 else 0
            n28 = int(H_actual[28].item()) if int(H_actual.numel()) > 28 else 0
            n29 = int(H_actual[29].item()) if int(H_actual.numel()) > 29 else 0
            n5 = int(H_actual[5].item()) if int(H_actual.numel()) > 5 else 0
            expected_ether = int(n2 + n28 + 2 * n29)
            used_5 = int(len(markers.get('ether_5_edges', set()) or set()))
            used_19 = int(len(markers.get('ether_19_edges', set()) or set()))
            target_19 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 19 and self._node_anchor_partition(node) == 'ether'
            ))
            if int(used_5) != int(n5):
                errors.append(
                    f"SU5外接O锚点边数不匹配: used={used_5} vs total={n5}"
                )
            if int(used_19) != int(target_19):
                errors.append(f"SU19(ether)外接O锚点边数不匹配: used={used_19} vs target={target_19}")
            if int(used_5 + used_19) != int(expected_ether):
                errors.append(
                    f"固定连接总量不匹配: SU5(连O)={used_5}, SU19(ether)(连O)={used_19}, SU2+SU28+2*SU29={expected_ether}"
                )

        if bool(check_amine):
            n0 = int(H_actual[0].item()) if int(H_actual.numel()) > 0 else 0
            n27 = int(H_actual[27].item()) if int(H_actual.numel()) > 27 else 0
            n6 = int(H_actual[6].item()) if int(H_actual.numel()) > 6 else 0
            expected_amine = int(n0 + 2 * n27)
            used_6 = int(len(markers.get('amine_6_edges', set()) or set()))
            used_20 = int(len(markers.get('amine_20_edges', set()) or set()))
            target_20 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 20
            ))
            if int(used_6) != int(n6):
                errors.append(f"SU6外接N锚点边数不匹配: used={used_6} vs total={n6}")
            if int(used_20) != int(target_20):
                errors.append(f"SU20外接N锚点边数不匹配: used={used_20} vs target={target_20}")
            if int(used_6 + used_20) != int(expected_amine):
                errors.append(
                    f"固定连接总量不匹配: SU6(连N)={used_6}, SU20(连N)={used_20}, SU0+2*SU27={expected_amine}"
                )

        if bool(check_halogen):
            n32 = int(H_actual[32].item()) if int(H_actual.numel()) > 32 else 0
            n8 = int(H_actual[8].item()) if int(H_actual.numel()) > 8 else 0
            expected_halogen = int(n32)
            used_8 = int(len(markers.get('halogen_8_edges', set()) or set()))
            used_21 = int(len(markers.get('halogen_21_edges', set()) or set()))
            target_21 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 21
            ))
            if int(used_8) != int(n8):
                errors.append(f"SU8外接X锚点边数不匹配: used={used_8} vs total={n8}")
            if int(used_21) != int(target_21):
                errors.append(f"SU21外接X锚点边数不匹配: used={used_21} vs target={target_21}")
            if int(used_8 + used_21) != int(expected_halogen):
                errors.append(
                    f"固定连接总量不匹配: SU8(连X)={used_8}, SU21(连X)={used_21}, SU32={expected_halogen}"
                )

        return errors

    def _raise_if_fixed_anchor_invariants_fail(self,
                                               nodes: List[_NodeV3],
                                               stage: str,
                                               check_thio: bool = True,
                                               check_ether: bool = True,
                                               check_amine: bool = True,
                                               check_halogen: bool = True) -> None:
        errors = self._fixed_anchor_invariant_errors(
            nodes,
            check_thio=bool(check_thio),
            check_ether=bool(check_ether),
            check_amine=bool(check_amine),
            check_halogen=bool(check_halogen),
        )
        if errors:
            msg = "; ".join(str(x) for x in errors[:6])
            raise RuntimeError(f"[{stage}] 固定锚点约束失败: {msg}")

    def _clear_all_edges_incident_to_set(self,
                                         nodes: List[_NodeV3],
                                         node_ids: List[int]) -> None:
        node_set = set(int(x) for x in list(node_ids or []))
        if not node_set:
            return
        seen_edges: Set[Tuple[int, int]] = set()
        for nid in sorted(node_set):
            if int(nid) < 0 or int(nid) >= len(nodes):
                continue
            node = nodes[int(nid)]
            for nb_id in list(getattr(node, 'hop1_ids', []) or []):
                nb_i = int(nb_id)
                edge = tuple(sorted((int(nid), int(nb_i))))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                self._remove_bidirectional_hop1_with_force(
                    nodes,
                    int(edge[0]),
                    int(edge[1]),
                    force=True,
                )

    def _solve_exact_fixed_anchor_matching_demands(self,
                                                   nodes: List[_NodeV3],
                                                   owner_slots: List[Dict[str, int]],
                                                   target_demands: List[Dict[str, Any]],
                                                   compat_fn) -> Optional[List[Tuple[int, int]]]:
        base_nodes = copy.deepcopy(nodes)
        demand_indices = [int(i) for i in range(len(target_demands))]

        def _target_priority(work_nodes: List[_NodeV3], demand: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
            node = work_nodes[int(demand['target_id'])]
            su_i = int(getattr(node, 'su_type', -1))
            deg_i = int(self._node_target_degree(node) or 0)
            fixed_i = int(self._node_fixed_anchor_target(node))
            mode_i = self._node_anchor_mode(node)
            if int(su_i) == 19 and str(mode_i) == 'double':
                rank = 0
            elif int(su_i) == 20 and str(mode_i) == 'double':
                rank = 1
            elif int(su_i) == 19 and int(deg_i) == 1:
                rank = 2
            elif int(su_i) == 7:
                rank = 3
            elif int(su_i) == 19:
                rank = 4
            elif int(su_i) == 5:
                rank = 5
            elif int(su_i) == 20:
                rank = 6
            elif int(su_i) == 6:
                rank = 7
            else:
                rank = 8
            return (
                int(rank),
                -int(fixed_i),
                int(su_i),
                int(deg_i),
                int(demand.get('demand_index', 0)),
            )

        def _slot_priority(work_nodes: List[_NodeV3],
                           slot: Dict[str, int],
                           target: Optional[_NodeV3] = None) -> Tuple[int, int, int, int]:
            owner = work_nodes[int(slot['owner_id'])]
            owner_su = int(getattr(owner, 'su_type', -1))
            if int(owner_su) == 28:
                rank = 0
            elif int(owner_su) == 29:
                rank = 1
            elif int(owner_su) == 2:
                rank = 2
            elif int(owner_su) == 27:
                rank = 3
            elif int(owner_su) == 0:
                rank = 4
            else:
                rank = 5
            p_penalty = 0
            if target is not None and self._is_double_special_d2_node(target, 19):
                # SU28 is already a P-type connector once attached to double19d2.
                # Prefer non-terminal oxygen connectors for double19d2 to avoid
                # producing 28-double19d2-P motifs that later block closure.
                if int(owner_su) == 28:
                    p_penalty = 20
                elif int(owner_su) in {29, 2}:
                    p_penalty = 2
            if target is not None and self._is_double_special_d2_node(target, 20):
                if int(owner_su) == 27:
                    p_penalty = 10
                elif int(owner_su) == 0:
                    p_penalty = 2
            slot_rank = int(slot.get('priority_rank', rank))
            owner_order = int(slot.get('spread_rank', slot['owner_id']))
            return (
                int(p_penalty),
                int(slot_rank),
                int(owner_order),
                int(slot['owner_id']),
                int(slot.get('slot_index', 0)),
            )

        def _candidate_slots(work_nodes: List[_NodeV3],
                             used_slots: List[bool],
                             owner_d1_usage: Dict[int, int],
                             demand_idx: int) -> List[int]:
            demand = dict(target_demands[int(demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            out: List[int] = []
            for slot_idx, slot in enumerate(owner_slots):
                if bool(used_slots[slot_idx]):
                    continue
                owner = work_nodes[int(slot['owner_id'])]
                if not compat_fn(work_nodes, owner, target, slot, owner_d1_usage, demand):
                    continue
                out.append(int(slot_idx))
            out.sort(key=lambda idx: _slot_priority(work_nodes, owner_slots[int(idx)], target))
            return out

        def _extract_edges(work_nodes: List[_NodeV3]) -> List[Tuple[int, int]]:
            target_id_set = set(int(d.get('target_id', -1)) for d in list(target_demands or []))
            owner_id_set = set(int(slot['owner_id']) for slot in list(owner_slots or []))
            edges: Set[Tuple[int, int]] = set()
            for owner_id in sorted(owner_id_set):
                owner = work_nodes[int(owner_id)]
                for nb_id in list(getattr(owner, 'hop1_ids', []) or []):
                    nb_i = int(nb_id)
                    if nb_i not in target_id_set:
                        continue
                    edges.add(tuple(sorted((int(owner_id), int(nb_i)))))
            return list(sorted(edges))

        def _apply_slot(work_nodes: List[_NodeV3],
                        used_slots: List[bool],
                        owner_d1_usage: Dict[int, int],
                        demand_idx: int,
                        slot_idx: int) -> Tuple[bool, bool]:
            demand = dict(target_demands[int(demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            target_deg = int(self._node_target_degree(target) or 0)
            slot = owner_slots[int(slot_idx)]
            owner_id = int(slot['owner_id'])
            owner = work_nodes[int(owner_id)]
            if not self._can_add_hop1_connection(work_nodes, owner, target):
                return False, False
            self._add_bidirectional_hop1(work_nodes, int(owner_id), int(demand['target_id']), lock=True)
            if int(demand['target_id']) not in set(int(x) for x in list(getattr(work_nodes[int(owner_id)], 'hop1_ids', []) or [])):
                return False, False
            used_slots[int(slot_idx)] = True
            inc_d1 = (
                int(getattr(owner, 'su_type', -1)) == 29 and
                int(getattr(target, 'su_type', -1)) == 19 and
                int(target_deg) == 1
            )
            if bool(inc_d1):
                owner_d1_usage[int(owner_id)] += 1
            return True, bool(inc_d1)

        def _undo_slot(work_nodes: List[_NodeV3],
                       used_slots: List[bool],
                       owner_d1_usage: Dict[int, int],
                       demand_idx: int,
                       slot_idx: int,
                       inc_d1: bool) -> None:
            demand = dict(target_demands[int(demand_idx)] or {})
            slot = owner_slots[int(slot_idx)]
            owner_id = int(slot['owner_id'])
            if bool(inc_d1):
                owner_d1_usage[int(owner_id)] -= 1
                if int(owner_d1_usage[int(owner_id)]) <= 0:
                    owner_d1_usage.pop(int(owner_id), None)
            self._remove_bidirectional_hop1_with_force(
                work_nodes,
                int(owner_id),
                int(demand['target_id']),
                force=True,
            )
            used_slots[int(slot_idx)] = False

        def _solve_greedy() -> Optional[List[Tuple[int, int]]]:
            work_nodes = copy.deepcopy(base_nodes)
            used_slots = [False] * len(owner_slots)
            owner_d1_usage: Dict[int, int] = defaultdict(int)
            ordered = sorted(
                list(demand_indices),
                key=lambda x: _target_priority(work_nodes, target_demands[int(x)]),
            )
            for demand_idx in ordered:
                options = _candidate_slots(work_nodes, used_slots, owner_d1_usage, int(demand_idx))
                if not options:
                    return None
                applied = False
                for slot_idx in list(options):
                    ok, _inc_d1 = _apply_slot(
                        work_nodes,
                        used_slots,
                        owner_d1_usage,
                        int(demand_idx),
                        int(slot_idx),
                    )
                    if bool(ok):
                        applied = True
                        break
                if not bool(applied):
                    return None
            return _extract_edges(work_nodes)

        greedy_edges = _solve_greedy()
        if greedy_edges is not None:
            return list(greedy_edges)

        # Large fixed-anchor pools are intentionally generated so that the
        # deterministic greedy order is feasible. If greedy cannot complete one
        # of these pools, the mode metadata is inconsistent with the hard Layer1
        # rules; exhaustive DFS only turns that into a multi-minute stall.
        if int(len(demand_indices)) > 64 or int(len(owner_slots)) > 64:
            return None

        work_nodes = copy.deepcopy(base_nodes)
        used_slots = [False] * len(owner_slots)
        owner_d1_usage: Dict[int, int] = defaultdict(int)
        max_states = max(200, min(5000, int(len(demand_indices) * 40)))
        states_seen = 0

        def _dfs(remaining_demand_indices: List[int]) -> bool:
            nonlocal states_seen
            states_seen += 1
            if int(states_seen) > int(max_states):
                return False
            if not remaining_demand_indices:
                return True

            best_demand_idx: Optional[int] = None
            best_options: Optional[List[int]] = None
            for idx in sorted(remaining_demand_indices, key=lambda x: _target_priority(work_nodes, target_demands[int(x)])):
                options = _candidate_slots(work_nodes, used_slots, owner_d1_usage, int(idx))
                if not options:
                    return False
                if best_options is None or len(options) < len(best_options):
                    best_demand_idx = int(idx)
                    best_options = list(options)
                    if len(best_options) <= 1:
                        break

            if best_demand_idx is None or best_options is None:
                return False

            next_remaining = [int(idx) for idx in remaining_demand_indices if int(idx) != int(best_demand_idx)]
            for slot_idx in list(best_options):
                ok, inc_d1 = _apply_slot(
                    work_nodes,
                    used_slots,
                    owner_d1_usage,
                    int(best_demand_idx),
                    int(slot_idx),
                )
                if not bool(ok):
                    continue
                if _dfs(next_remaining):
                    return True
                _undo_slot(
                    work_nodes,
                    used_slots,
                    owner_d1_usage,
                    int(best_demand_idx),
                    int(slot_idx),
                    bool(inc_d1),
                )
            return False

        ok = _dfs(list(demand_indices))
        if not ok:
            return None
        return _extract_edges(work_nodes)

    def _bridge_candidate_bias(self, center_su: int, current_neighbors: List[int], cand_su: int) -> float:
        center_su = int(center_su)
        cand_su = int(cand_su)
        current = [int(x) for x in current_neighbors]
        cnt = Counter(current)

        if center_su == 27:
            if cnt[6] > 0 and cand_su == 20:
                return 4.0
            if cnt[20] > 0 and cand_su == 6:
                return 4.0
            if cnt[cand_su] > 0:
                return 0.2

        if center_su == 29:
            if cnt[5] > 0 and cand_su == 19:
                return 4.0
            if cnt[19] > 0 and cand_su == 5:
                return 4.0
            if cnt[cand_su] > 0:
                return 0.15

        if center_su == 31:
            if cand_su == 7:
                if cnt[7] == 0:
                    return 5.0
                return 3.5
            if cand_su == 19:
                if cnt[7] == 0:
                    return 0.25
                return 1.0

        if center_su == 3:
            if cand_su == 9 and cnt[9] > 0:
                return 0.05
            if cnt[9] > 0 and cand_su != 9:
                return 2.5
            if cand_su == 9 and cnt[9] == 0:
                return 1.5

        if center_su == 2:
            if cnt[9] > 0 and cand_su == 19:
                return 3.0
            if cnt[9] > 0 and cand_su == 5:
                return 0.35
            if cand_su == 19 and cnt[19] == 0:
                return 1.75
            if cand_su == 5 and cnt[19] == 0:
                return 0.85

        return 1.0

    def _filter_valid_fixed_targets(self,
                                    nodes: List[_NodeV3],
                                    center: _NodeV3,
                                    targets: List[_NodeV3]) -> List[_NodeV3]:
        valid = []
        for n in targets:
            if int(n.global_id) == int(center.global_id):
                continue
            if int(n.global_id) in center.hop1_ids:
                continue
            if int(n.remaining_hop1_slots()) <= 0:
                continue
            if not self._can_add_hop1_connection(nodes, center, n):
                continue
            valid.append(n)
        return valid

    def _pick_fixed_target(self,
                           nodes: List[_NodeV3],
                           center: _NodeV3,
                           target_pool: List[_NodeV3],
                           priority_list: List[int]) -> Optional[_NodeV3]:
        candidates = self._filter_valid_fixed_targets(nodes, center, target_pool)
        if not candidates:
            return None
        return self._choose_fixed_candidate(center, candidates, priority_list, nodes=nodes)

    @staticmethod
    def _print_consistency_errors(errors: List[str], prefix: str, limit: int = 8) -> None:
        if not errors:
            return
        max_show = max(1, int(limit))
        shown = list(errors[:max_show])
        for idx, msg in enumerate(shown, start=1):
            print(f"    [{prefix} #{idx}] {msg}")
        if len(errors) > len(shown):
            print(f"    [{prefix}] ... 其余 {len(errors) - len(shown)} 条省略")

    def validate_graph_consistency(self, nodes: List[_NodeV3],
                                    H: torch.Tensor,
                                    E_target: Optional[torch.Tensor] = None,
                                    verbose: bool = False) -> Tuple[bool, List[str]]:
        """Compatibility wrapper for the pure Layer1 graph validator."""
        return self._validator.validate(
            nodes=nodes,
            H=H,
            E_target=E_target,
            verbose=verbose,
        )
    # Layer1: 1-hop分配辅助方法
    # ========================================================================

    def _initialize_node_pool(
        self,
        H_init: torch.Tensor,
        identity_seed_nodes: Optional[List[_NodeV3]] = None,
    ) -> List[_NodeV3]:
        """初始化全局节点池，为每个SU实例创建节点对象
        """
        nodes = []
        global_id = 0
        special_degree_meta = self._get_special_degree_meta(H_init)
        special_partition_meta = self._get_special_partition_meta(H_init)
        special_anchor_mode_meta = self._get_special_anchor_mode_meta(H_init)

        for su_type in range(NUM_SU_TYPES):
            count = int(H_init[su_type].item())
            degree_sequence = self._build_special_degree_sequence(int(su_type), int(count), special_degree_meta)
            partition_sequence_19 = self._build_special_partition_sequence_19(int(count), special_partition_meta) if int(su_type) == 19 else []
            anchor_sequence_19 = self._build_special_anchor_sequence_19(int(count), special_anchor_mode_meta) if int(su_type) == 19 else []
            anchor_sequence_20 = self._build_special_anchor_sequence_20(int(count), special_anchor_mode_meta) if int(su_type) == 20 else []
            anchor_sequence_21 = self._build_special_anchor_sequence_21(int(count), special_anchor_mode_meta) if int(su_type) == 21 else []
            for idx in range(count):
                target_fixed_anchor_count = None
                anchor_mode = None
                if int(su_type) == 19 and idx < len(anchor_sequence_19):
                    target_degree, partition, target_fixed_anchor_count, anchor_mode = anchor_sequence_19[idx]
                elif int(su_type) == 19 and idx < len(partition_sequence_19):
                    target_degree, partition = partition_sequence_19[idx]
                elif int(su_type) == 20 and idx < len(anchor_sequence_20):
                    target_degree, target_fixed_anchor_count, anchor_mode = anchor_sequence_20[idx]
                    partition = None
                elif int(su_type) == 21 and idx < len(anchor_sequence_21):
                    target_degree, target_fixed_anchor_count, anchor_mode = anchor_sequence_21[idx]
                    partition = None
                else:
                    target_degree = degree_sequence[idx] if idx < len(degree_sequence) else None
                    partition = None
                    if int(su_type) in {19, 20, 21} and target_degree is not None:
                        target_fixed_anchor_count = 1
                if int(su_type) not in {19, 20, 21}:
                    target_degree = int(SU_CONNECTION_DEGREE.get(int(su_type), 4))
                if int(su_type) in {19, 20, 21}:
                    special_source = 'layer0_special_degree_meta' if target_degree is not None else None
                else:
                    special_source = 'fixed_su_connection_degree'
                node = _NodeV3(
                    global_id=global_id,
                    su_type=su_type,
                    target_hop1_degree=target_degree,
                    special_degree_source=special_source,
                    special_anchor_partition=partition,
                    target_fixed_anchor_count=target_fixed_anchor_count,
                    special_anchor_mode=anchor_mode,
                )
                nodes.append(node)
                global_id += 1

        if identity_seed_nodes is None:
            return nodes

        seed_nodes = list(identity_seed_nodes or [])
        seed_ids = [int(getattr(node, 'global_id', -1)) for node in seed_nodes]
        if sorted(seed_ids) != list(range(len(seed_nodes))):
            raise ValueError(
                'identity_seed_nodes must have unique contiguous global_id values'
            )
        seed_ids_by_su: Dict[int, List[int]] = defaultdict(list)
        for node in seed_nodes:
            seed_ids_by_su[int(node.su_type)].append(int(node.global_id))
        for values in seed_ids_by_su.values():
            values.sort()
        configured_by_su: Dict[int, List[_NodeV3]] = defaultdict(list)
        for node in nodes:
            configured_by_su[int(node.su_type)].append(node)
        expected_counts = {
            int(su_type): len(values)
            for su_type, values in configured_by_su.items()
            if values
        }
        seed_counts = {
            int(su_type): len(values)
            for su_type, values in seed_ids_by_su.items()
            if values
        }
        if seed_counts != expected_counts:
            raise ValueError('identity_seed_nodes histogram does not match H_init')
        remapped: List[_NodeV3] = []
        for su_type, configured_nodes in configured_by_su.items():
            for node, global_id_value in zip(
                configured_nodes,
                seed_ids_by_su[int(su_type)],
            ):
                node.global_id = int(global_id_value)
                remapped.append(node)
        remapped.sort(key=lambda node: int(node.global_id))
        return remapped

    @staticmethod
    def _edge_is_fixed(node1: _NodeV3, node2: _NodeV3) -> bool:
        return bool(
            int(node2.global_id) in set(getattr(node1, 'fixed_hop1_ids', set()) or set())
            or int(node1.global_id) in set(getattr(node2, 'fixed_hop1_ids', set()) or set())
        )

    def _add_bidirectional_hop1(self,
                                nodes: List[_NodeV3],
                                id1: int,
                                id2: int,
                                lock: bool = False) -> bool:
        """添加双向1-hop连接
        """
        node1 = nodes[id1]
        node2 = nodes[id2]

        if id1 == id2:
            return False

        if id2 in node1.hop1_ids or id1 in node2.hop1_ids:
            return False
        if not self._can_add_hop1_connection(nodes, node1, node2):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node1, int(node2.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node2, int(node1.su_type)):
            return False
        if self._violates_terminal_neighbor_limit(nodes, node1, node1, node2):
            return False
        if self._violates_terminal_neighbor_limit(nodes, node2, node1, node2):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, node1, node1, node2):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, node2, node1, node2):
            return False
        if self._violates_forbidden_p_pair_connection(nodes, node1, node2):
            return False
        # 添加互为1-hop（SU类型计数）
        node1.hop1_su[node2.su_type] += 1
        node2.hop1_su[node1.su_type] += 1

        # 记录全局ID（用于追踪具体连接）
        node1.hop1_ids.append(id2)
        node2.hop1_ids.append(id1)
        if bool(lock):
            node1.fixed_hop1_ids.add(int(id2))
            node2.fixed_hop1_ids.add(int(id1))
        self._refresh_runtime_node_entry(nodes, int(id1))
        self._refresh_runtime_node_entry(nodes, int(id2))
        # Cache keys include each endpoint's neighbor ids/types and remaining
        # slots, so successful edge edits naturally produce new signatures.
        # Avoid clearing the whole cache on every add; Layer1 calls this path
        # many thousands of times on large graphs.
        return True

    def _remove_bidirectional_hop1(self, nodes: List[_NodeV3], id1: int, id2: int) -> bool:
        """移除一条双向1-hop连接（仅移除一条，多重边会移除一次）"""
        return self._remove_bidirectional_hop1_with_force(nodes, id1, id2, force=False)

    def _remove_bidirectional_hop1_with_force(self,
                                              nodes: List[_NodeV3],
                                              id1: int,
                                              id2: int,
                                              force: bool = False) -> bool:
        """移除一条双向1-hop连接；force=True 时允许拆除 fixed 边。"""
        node1 = nodes[id1]
        node2 = nodes[id2]

        # 先确认边存在
        if id2 not in node1.hop1_ids or id1 not in node2.hop1_ids:
            return False
        if (not bool(force)) and self._edge_is_fixed(node1, node2):
            return False

        # 更新SU类型计数
        node1.hop1_su[node2.su_type] -= 1
        if node1.hop1_su[node2.su_type] <= 0:
            del node1.hop1_su[node2.su_type]

        node2.hop1_su[node1.su_type] -= 1
        if node2.hop1_su[node1.su_type] <= 0:
            del node2.hop1_su[node1.su_type]

        # 更新全局ID列表（移除一次）
        node1.hop1_ids.remove(id2)
        node2.hop1_ids.remove(id1)
        node1.fixed_hop1_ids.discard(int(id2))
        node2.fixed_hop1_ids.discard(int(id1))
        self._refresh_runtime_node_entry(nodes, int(id1))
        self._refresh_runtime_node_entry(nodes, int(id2))
        return True

    def _edge_semantics_ok(self,
                           su1: int,
                           su2: int,
                           E_target: Optional[torch.Tensor] = None) -> bool:
        E_eval = E_target if E_target is not None else getattr(self, '_current_E_target', None)
        cache_key = (int(su1), int(su2), (id(E_eval) if E_eval is not None else None))
        cached = self._edge_semantics_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        try:
            ok = bool(validate_connection(int(su1), int(su2), E_eval)) and bool(
                validate_connection(int(su2), int(su1), E_eval)
            )
            self._edge_semantics_cache[cache_key] = bool(ok)
            return bool(ok)
        except Exception:
            self._edge_semantics_cache[cache_key] = False
            return False

    def _node_edge_semantics_ok(self,
                                nodes: List[_NodeV3],
                                node1: _NodeV3,
                                node2: _NodeV3,
                                E_target: Optional[torch.Tensor] = None) -> bool:
        if not self._edge_semantics_ok(int(node1.su_type), int(node2.su_type), E_target=E_target):
            return False
        port_sets1 = self._port_sets_for_node(node1)
        if port_sets1:
            if not can_match_ports_partial(self._current_neighbor_nodes(nodes, node1), port_sets1, center_node=node1):
                return False
        port_sets2 = self._port_sets_for_node(node2)
        if port_sets2:
            if not can_match_ports_partial(self._current_neighbor_nodes(nodes, node2), port_sets2, center_node=node2):
                return False
        return True

    def _can_add_hop1_connection(self, nodes: List[_NodeV3], node1: _NodeV3, node2: _NodeV3) -> bool:
        if int(node1.global_id) == int(node2.global_id):
            return False
        if int(node2.global_id) in set(node1.hop1_ids or []):
            return False
        if int(node1.global_id) in set(node2.hop1_ids or []):
            return False
        if int(node1.remaining_hop1_slots()) <= 0 or int(node2.remaining_hop1_slots()) <= 0:
            return False
        key = (
            self._can_add_state_signature(nodes, node1),
            self._can_add_state_signature(nodes, node2),
        )
        cached = self._can_add_hop1_cache.get(key)
        if cached is not None:
            return bool(cached)
        ok = self._can_add_hop1_connection_uncached(nodes, node1, node2)
        if len(self._can_add_hop1_cache) > 200000:
            self._can_add_hop1_cache.clear()
        self._can_add_hop1_cache[key] = bool(ok)
        return bool(ok)

    def _can_add_hop1_connection_uncached(self, nodes: List[_NodeV3], node1: _NodeV3, node2: _NodeV3) -> bool:
        """检查在 node1 和 node2 之间添加1-hop连接是否满足双向 per-port 规则

        对两侧分别检查：将新邻居加入当前邻居列表后，是否仍可分配到端口中。
        """
        if node1.global_id == node2.global_id:
            return False

        # Fixed-anchor matching may reason over multiple owner slots per node.
        # A repeated edge between the same two nodes must never count as filling
        # an extra slot/demand, otherwise the matcher can "use up" supply
        # without creating a real second connection.
        if int(node2.global_id) in set(getattr(node1, 'hop1_ids', []) or []):
            return False
        if int(node1.global_id) in set(getattr(node2, 'hop1_ids', []) or []):
            return False

        if not self._edge_semantics_ok(int(node1.su_type), int(node2.su_type)):
            return False

        if int(node1.remaining_hop1_slots()) <= 0 or int(node2.remaining_hop1_slots()) <= 0:
            return False

        if not self._is_su19_partition_compatible(node1, int(node2.su_type)):
            return False
        if not self._is_su19_partition_compatible(node2, int(node1.su_type)):
            return False

        if self._violates_mode_specific_rule(node1, int(node2.su_type)):
            return False
        if self._violates_mode_specific_rule(node2, int(node1.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node1, int(node2.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node2, int(node1.su_type)):
            return False
        if self._violates_terminal_neighbor_limit(nodes, node1, node1, node2):
            return False
        if self._violates_terminal_neighbor_limit(nodes, node2, node1, node2):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, node1, node1, node2):
            return False
        if self._violates_fixed_connector_special_end_rules(nodes, node2, node1, node2):
            return False
        if self._violates_forbidden_p_pair_connection(nodes, node1, node2):
            return False

        required1 = set(int(x) for x in self._required_external_candidates_for_node(node1))
        target_req1 = int(self._node_fixed_anchor_target(node1))
        current_req1 = int(self._required_external_count_for_node(node1, nodes))
        if required1 and int(target_req1) > 0:
            if int(current_req1) < int(target_req1) and int(node2.su_type) not in required1:
                return False
            if int(node2.su_type) in required1 and int(current_req1) >= int(target_req1):
                return False
        required2 = set(int(x) for x in self._required_external_candidates_for_node(node2))
        target_req2 = int(self._node_fixed_anchor_target(node2))
        current_req2 = int(self._required_external_count_for_node(node2, nodes))
        if required2 and int(target_req2) > 0:
            if int(current_req2) < int(target_req2) and int(node1.su_type) not in required2:
                return False
            if int(node1.su_type) in required2 and int(current_req2) >= int(target_req2):
                return False

        # 检查 node1 侧
        port_sets1 = self._port_sets_for_node(node1)
        if port_sets1:
            current_neighbors1 = self._current_neighbor_nodes(nodes, node1)
            proposed1 = current_neighbors1 + [node2]
            if not can_match_ports_partial(proposed1, port_sets1, center_node=node1):
                return False

        # 检查 node2 侧
        port_sets2 = self._port_sets_for_node(node2)
        if port_sets2:
            current_neighbors2 = self._current_neighbor_nodes(nodes, node2)
            proposed2 = current_neighbors2 + [node1]
            if not can_match_ports_partial(proposed2, port_sets2, center_node=node2):
                return False

        if not self._node_exact_hop1_shift_state_ok(
            node1,
            proposed_neighbor_su=int(node2.su_type),
        ):
            return False
        if not self._node_exact_hop1_shift_state_ok(
            node2,
            proposed_neighbor_su=int(node1.su_type),
        ):
            return False

        return True

    def _get_nodes_by_su_type(self, nodes: List[_NodeV3], su_type: int) -> List[_NodeV3]:
        """获取指定SU类型的所有节点"""
        if self._runtime_index_matches(nodes):
            ids = list(self._node_ids_by_su.get(int(su_type), []))
            return [nodes[int(i)] for i in ids]
        return [n for n in nodes if n.su_type == su_type]

    def _choose_weighted_candidate(self,
                                   center: _NodeV3,
                                   candidates: List[_NodeV3],
                                   priority_list: List[int],
                                   nodes: Optional[List[_NodeV3]] = None,
                                   motif_usage: Optional[Counter] = None) -> Optional[_NodeV3]:
        filtered = [
            n for n in candidates
            if n.global_id != center.global_id
            and n.remaining_hop1_slots() > 0
            and n.global_id not in center.hop1_ids
        ]
        if not filtered:
            return None

        # per-port 双向验证：确保添加连接后两侧端口规则均满足
        if nodes is not None:
            filtered = [n for n in filtered if self._can_add_hop1_connection(nodes, center, n)]
            if not filtered:
                return None

        weights = []
        current_neighbors = self._current_neighbor_types(nodes, center) if nodes is not None else []
        priority_rank = {int(su): int(idx) for idx, su in enumerate(priority_list)}
        required_external_center = set(int(x) for x in self._required_external_candidates_for_node(center))
        center_missing_external = bool(required_external_center) and not self._has_required_external_for_node(center, nodes) if nodes is not None else False
        for n in filtered:
            idx = priority_rank.get(int(n.su_type), len(priority_list))
            base = 1.0 / (1 + idx)
            w = base
            w /= (1 + center.hop1_su.get(n.su_type, 0))
            w /= (1 + n.hop1_su.get(center.su_type, 0))
            w *= self._bridge_candidate_bias(int(center.su_type), current_neighbors, int(n.su_type))
            if nodes is not None and int(center.su_type) == 11:
                cand_su = int(n.su_type)
                if bool(center_missing_external):
                    if cand_su in {22, 23, 24, 25, 15, 17}:
                        w *= 2.50
                    elif cand_su in {19, 20, 21}:
                        w *= 0.35
                    if cand_su in {19, 20, 21} and self._has_pending_required_external(n, nodes):
                        w *= 0.10
                elif cand_su in {19, 20, 21} and self._has_pending_required_external(n, nodes):
                    w *= 0.10
            weights.append(w)

        if bool(self.deterministic):
            ranked = sorted(
                range(len(filtered)),
                key=lambda i: (
                    float(weights[i]),
                    int(filtered[i].remaining_hop1_slots()),
                    -int(filtered[i].global_id),
                ),
                reverse=True,
            )
            if not ranked:
                return None
            top_weight = float(weights[int(ranked[0])])
            if top_weight > 0.0:
                window = [
                    idx for idx in ranked
                    if float(weights[int(idx)]) >= 0.85 * float(top_weight)
                ]
            else:
                window = list(ranked)
            window = window[: max(1, min(4, len(window)))]
            choose_pos = int(center.global_id) % len(window)
            return filtered[int(window[int(choose_pos)])]
        return random.choices(filtered, weights=weights, k=1)[0]

    # ========================================================================
    # Layer1: 固定连接分配方法（a-f）
    # ========================================================================

    def _assign_fixed_halogen_X(self, nodes: List[_NodeV3]):
        """a) 32号X -> 8号/21号"""
        owner_ids, target_ids_all, edge_pairs = self._build_exact_halogen_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Halogen Fixed Assign',
            check_thio=False,
            check_ether=False,
            check_amine=False,
            check_halogen=True,
        )

    def _build_exact_halogen_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        x_nodes = self._get_nodes_by_su_type(work_nodes, 32)
        su8_nodes = self._get_nodes_by_su_type(work_nodes, 8)
        su21_nodes = self._sort_special_degree_nodes(self._get_nodes_by_su_type(work_nodes, 21))
        owner_ids = sorted(int(n.global_id) for n in list(x_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su8_nodes) + list(su21_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(x_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 32,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su21_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '21_halogen',
                })
        for node in list(su8_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '8_halogen',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 21 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Halogen Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) not in {8, 21}:
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Halogen Fixed Assign] 无法找到满足严格 single/double 约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _build_exact_thio_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        s_nodes = self._get_nodes_by_su_type(work_nodes, 31)
        su7_nodes = self._get_nodes_by_su_type(work_nodes, 7)
        su19_nodes = [
            n for n in self._get_nodes_by_su_type(work_nodes, 19)
            if self._node_anchor_partition(n) == 'thio'
        ]
        owner_ids = sorted(int(n.global_id) for n in list(s_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su7_nodes) + list(su19_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for s_node in self._stable_node_order(s_nodes):
            for slot_idx in range(max(0, int(s_node.get_max_degree()))):
                owner_slots.append({
                    'owner_id': int(s_node.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 31,
                })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su19_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '19_thio',
                })
        for node in list(su7_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '7_thio',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 19 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Thio Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            if int(getattr(target, 'su_type', -1)) == 19 and self._node_anchor_partition(target) != 'thio':
                return False
            if not self._fixed_connector_assignment_ok(work_local, owner, target):
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Thio Fixed Assign] 无法找到满足严格分区约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _build_exact_ether_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        su2_nodes = self._get_nodes_by_su_type(work_nodes, 2)
        su28_nodes = self._get_nodes_by_su_type(work_nodes, 28)
        su29_nodes = self._get_nodes_by_su_type(work_nodes, 29)
        su5_nodes = self._get_nodes_by_su_type(work_nodes, 5)
        su19_nodes = [
            n for n in self._get_nodes_by_su_type(work_nodes, 19)
            if self._node_anchor_partition(n) == 'ether'
        ]
        owner_ids = sorted(int(n.global_id) for n in list(su2_nodes) + list(su28_nodes) + list(su29_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su5_nodes) + list(su19_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(su28_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 28,
            })
        for owner in self._stable_node_order(su29_nodes):
            for slot_idx in range(max(0, int(owner.get_max_degree()))):
                owner_slots.append({
                    'owner_id': int(owner.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 29,
                })
        for owner in self._stable_node_order(su2_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 2,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su19_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '19_ether',
                })
        for node in list(su5_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '5_ether',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 19 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Ether Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            owner_su = int(getattr(owner, 'su_type', -1))
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) == 5:
                return bool(self._can_add_hop1_connection(work_local, owner, target))
            if int(target_su) != 19 or self._node_anchor_partition(target) != 'ether':
                return False
            if not self._fixed_connector_assignment_ok(work_local, owner, target):
                return False
            target_deg = int(self._node_target_degree(target) or 0)
            if int(target_deg) == 1:
                if int(owner_su) == 28:
                    return False
                if int(owner_su) == 29 and int(owner_d1_usage.get(int(owner.global_id), 0)) >= 1:
                    return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Ether Fixed Assign] 无法找到满足严格分区约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _assign_fixed_thioether_S(self, nodes: List[_NodeV3]):
        """b) 31号硫醚 -> 7号/19号"""
        owner_ids, target_ids_all, edge_pairs = self._build_exact_thio_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Thio Fixed Assign',
            check_thio=True,
            check_ether=False,
            check_amine=False,
            check_halogen=False,
        )

    def _assign_fixed_amine_N(self, nodes: List[_NodeV3]):
        """c) 0号氨基端、27号 -> 6号/20号"""
        work_nodes = copy.deepcopy(nodes)
        su0_nodes = self._get_nodes_by_su_type(work_nodes, 0)
        su27_nodes = self._get_nodes_by_su_type(work_nodes, 27)
        su6_nodes = self._get_nodes_by_su_type(work_nodes, 6)
        su20_nodes = self._sort_special_degree_nodes(self._get_nodes_by_su_type(work_nodes, 20))

        owner_ids = sorted(int(n.global_id) for n in list(su0_nodes) + list(su27_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su6_nodes) + list(su20_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(su27_nodes):
            for slot_idx in range(2):
                owner_slots.append({
                    'owner_id': int(owner.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 27,
                })
        for owner in self._stable_node_order(su0_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 0,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su20_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '20_amine',
                })
        for node in list(su6_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '6_amine',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 20 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Amine Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) not in {6, 20}:
                return False
            if not self._fixed_connector_assignment_ok(work_local, owner, target):
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        if edge_pairs is None:
            raise RuntimeError("[Amine Fixed Assign] 无法找到满足严格 single/double 约束的精确匹配")
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Amine Fixed Assign',
            check_thio=True,
            check_ether=False,
            check_amine=True,
            check_halogen=False,
        )

    def _assign_fixed_ether_O(self, nodes: List[_NodeV3]):
        """d) 2号醚端、28号、29号 -> 5号/19号
        """
        owner_ids, target_ids_all, edge_pairs = self._build_exact_ether_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Ether Fixed Assign',
            check_thio=True,
            check_ether=True,
            check_amine=False,
            check_halogen=False,
        )

    def _assign_fixed_carbonyl(self, nodes: List[_NodeV3]):
        """e) Assign only the mandatory carbonyl neighbor of every SU9.

        核心规则：每个SU9的port3={0,1,2,3}，必须恰好有一个羰基邻居。
        普通羰基端口由全局补全求解器统一处理，不能在固定阶段提前
        消耗 SU23/24/25 等资源。
        """
        su0_nodes = self._get_nodes_by_su_type(nodes, 0)
        su1_nodes = self._get_nodes_by_su_type(nodes, 1)
        su2_nodes = self._get_nodes_by_su_type(nodes, 2)
        su3_nodes = self._get_nodes_by_su_type(nodes, 3)
        su9_nodes = [n for n in self._get_nodes_by_su_type(nodes, 9) if n.remaining_hop1_slots() > 0]

        # ---- 阶段1：先给每个0/1/2/3尽量分一个9号 ----
        first_round = self._canonical_node_order(
            list(su0_nodes) + list(su1_nodes) + list(su2_nodes) + list(su3_nodes),
            salt=401,
        )
        for carb in first_round:
            if carb.remaining_hop1_slots() <= 0:
                continue
            su9_cands = self._filter_valid_fixed_targets(nodes, carb, su9_nodes)
            if not su9_cands:
                continue
            su9 = self._choose_weighted_candidate(carb, su9_cands, [9], nodes=nodes)
            if su9 is None:
                continue
            self._add_bidirectional_hop1(
                nodes,
                carb.global_id,
                su9.global_id,
                lock=True,
            )

        # ---- 阶段1.5：如果9号还有剩余，再继续给3号分配第二个9 ----
        for n3 in su3_nodes:
            while n3.remaining_hop1_slots() > 0:
                su9_cands = self._filter_valid_fixed_targets(nodes, n3, su9_nodes)
                if not su9_cands:
                    break
                su9 = self._choose_weighted_candidate(n3, su9_cands, [9], nodes=nodes)
                if su9 is None:
                    break
                self._add_bidirectional_hop1(
                    nodes,
                    n3.global_id,
                    su9.global_id,
                    lock=True,
                )

        invalid_su9 = []
        for su9 in su9_nodes:
            carbonyl_count = sum(
                1
                for neighbor_id in list(su9.hop1_ids or [])
                if int(nodes[int(neighbor_id)].su_type) in {0, 1, 2, 3}
            )
            if carbonyl_count != 1:
                invalid_su9.append((int(su9.global_id), int(carbonyl_count)))
        if invalid_su9:
            raise RuntimeError(
                "[Carbonyl Fixed Assign] SU9 fixed-role matching incomplete: "
                + str(invalid_su9[:8])
            )

    def _su11_external_role_errors(self,
                                   nodes: List[_NodeV3]) -> List[str]:
        """Audit the single external role required by every SU11."""
        errors: List[str] = []
        external_types = set(
            int(x) for x in list(SU_EXTERNAL_CONNECTIONS.get(11, []) or [])
        )
        for node in self._get_nodes_by_su_type(nodes, 11):
            external_ids = [
                int(neighbor_id)
                for neighbor_id in list(getattr(node, 'hop1_ids', []) or [])
                if 0 <= int(neighbor_id) < len(nodes)
                and int(nodes[int(neighbor_id)].su_type) in external_types
            ]
            if len(external_ids) != 1:
                errors.append(
                    f"SU11节点{int(node.global_id)}外接角色边数={len(external_ids)} != 1, "
                    f"external_ids={sorted(external_ids)}"
                )
        return errors

    def _reserve_alkyl_aromatic_11_external_roles(
        self,
        nodes: List[_NodeV3],
    ) -> None:
        """Reserve one external role on every SU11 without choosing a partner."""
        allowed_external = set(
            int(x) for x in list(SU_EXTERNAL_CONNECTIONS.get(11, []) or [])
        )
        if not allowed_external:
            raise RuntimeError("[SU11 External Role] 合法外接类型集合为空")
        invalid = []
        for node in self._get_nodes_by_su_type(nodes, 11):
            target_count = int(self._node_fixed_anchor_target(node))
            current_count = int(self._required_external_count_for_node(node, nodes))
            pending_count = max(0, int(target_count) - int(current_count))
            if (
                int(target_count) != 1
                or int(current_count) > int(target_count)
                or int(node.remaining_hop1_slots()) < int(pending_count)
            ):
                invalid.append({
                    'node_id': int(node.global_id),
                    'target': int(target_count),
                    'current': int(current_count),
                    'remaining_ports': int(node.remaining_hop1_slots()),
                })
        if invalid:
            raise RuntimeError(
                "[SU11 External Role] 角色预留失败: " + str(invalid[:8])
            )

    def _layer1_phase_port_diagnostics(
        self,
        nodes: List[_NodeV3],
    ) -> Dict[str, Any]:
        aromatic_open_by_su: Counter = Counter()
        non_aromatic_open_by_su: Counter = Counter()
        for node in nodes:
            remaining = int(node.remaining_hop1_slots())
            if remaining <= 0:
                continue
            if int(node.su_type) in LAYER1_AROMATIC_RING_TYPES:
                aromatic_open_by_su[int(node.su_type)] += int(remaining)
            else:
                non_aromatic_open_by_su[int(node.su_type)] += int(remaining)

        pending_su11_external = int(sum(
            max(
                0,
                int(self._node_fixed_anchor_target(node))
                - int(self._required_external_count_for_node(node, nodes)),
            )
            for node in self._get_nodes_by_su_type(nodes, 11)
        ))
        non_aromatic_open = int(sum(non_aromatic_open_by_su.values()))
        aromatic_open = int(sum(aromatic_open_by_su.values()))
        projected_non_aromatic_internal = int(
            non_aromatic_open - pending_su11_external
        )
        projected_aromatic_ring = int(
            aromatic_open - pending_su11_external
        )
        enough_external_ports = bool(
            int(projected_non_aromatic_internal) >= 0
            and int(projected_aromatic_ring) >= 0
        )
        return {
            'ok': bool(
                enough_external_ports
                and int(projected_non_aromatic_internal) % 2 == 0
                and int(projected_aromatic_ring) % 2 == 0
            ),
            'su11_external_demands': int(pending_su11_external),
            'non_aromatic_open_ports': int(non_aromatic_open),
            'projected_non_aromatic_internal_ports': int(
                projected_non_aromatic_internal
            ),
            'projected_non_aromatic_internal_is_even': bool(
                int(projected_non_aromatic_internal) >= 0
                and int(projected_non_aromatic_internal) % 2 == 0
            ),
            'aromatic_open_ports_including_su11_external': int(aromatic_open),
            'projected_aromatic_ring_ports': int(projected_aromatic_ring),
            'projected_aromatic_ring_is_even': bool(
                int(projected_aromatic_ring) >= 0
                and int(projected_aromatic_ring) % 2 == 0
            ),
            'non_aromatic_open_by_su': dict(sorted(non_aromatic_open_by_su.items())),
            'aromatic_open_by_su': dict(sorted(aromatic_open_by_su.items())),
        }

    def _raise_if_layer1_phase_ports_infeasible(
        self,
        nodes: List[_NodeV3],
    ) -> Dict[str, Any]:
        diagnostics = self._layer1_phase_port_diagnostics(nodes)
        if bool(diagnostics.get('ok', False)):
            return diagnostics
        raise RuntimeError(
            "layer1_phase_port_infeasible: "
            f"su11_external_demands={diagnostics['su11_external_demands']} "
            f"non_aromatic_open={diagnostics['non_aromatic_open_ports']} "
            f"non_aromatic_internal={diagnostics['projected_non_aromatic_internal_ports']} "
            f"aromatic_open={diagnostics['aromatic_open_ports_including_su11_external']} "
            f"aromatic_ring={diagnostics['projected_aromatic_ring_ports']} "
            f"non_aromatic_by_su={diagnostics['non_aromatic_open_by_su']} "
            f"aromatic_by_su={diagnostics['aromatic_open_by_su']}"
        )

    def _non_aromatic_completion_errors(
        self,
        nodes: List[_NodeV3],
    ) -> List[str]:
        errors = [
            f"node={int(node.global_id)} SU{int(node.su_type)} "
            f"remaining_ports={int(node.remaining_hop1_slots())}"
            for node in nodes
            if int(node.su_type) not in LAYER1_AROMATIC_RING_TYPES
            and int(node.remaining_hop1_slots()) != 0
        ]
        for node in self._get_nodes_by_su_type(nodes, 11):
            external_count = int(self._required_external_count_for_node(node, nodes))
            if external_count != int(self._node_fixed_anchor_target(node)):
                errors.append(
                    f"node={int(node.global_id)} SU11 external_count={external_count}"
                )
        return errors

    def _assign_unsaturated_pairs(self, nodes: List[_NodeV3]):
        """f) 14/15/16双键配对，17/18三键配对
        """
        # 收集所有需要配对的双键节点（不饱和端连接度=1）
        su14_nodes = self._get_nodes_by_su_type(nodes, 14)
        su15_nodes = self._get_nodes_by_su_type(nodes, 15)
        su16_nodes = self._get_nodes_by_su_type(nodes, 16)

        # SU16 是只能和 SU15 配对的端基，不能参与 SU16-SU16。
        # 先完成这类强约束配对，再处理其余双键节点，避免普通贪心
        # 先消耗掉 SU15 后遗留孤立 SU16。
        paired = set()

        def pair_from_pools(left_pool, right_pool) -> None:
            for left in self._canonical_node_order(
                list(left_pool), salt=503 + len(paired)
            ):
                if left.global_id in paired:
                    continue
                candidates = [
                    right for right in self._canonical_node_order(
                        list(right_pool), salt=509 + left.global_id
                    )
                    if right.global_id not in paired
                    and self._can_add_hop1_connection(nodes, left, right)
                ]
                if candidates:
                    right = candidates[0]
                    paired.update((left.global_id, right.global_id))
                    self._add_bidirectional_hop1(
                        nodes, left.global_id, right.global_id, lock=True
                    )

        pair_from_pools(su16_nodes, su15_nodes)
        pair_from_pools(su15_nodes, su16_nodes)

        remaining_double_bond_pool = [
            node for node in list(su14_nodes) + list(su15_nodes)
            if node.global_id not in paired
        ]
        remaining_double_bond_pool = self._canonical_node_order(
            remaining_double_bond_pool, salt=517
        )
        while True:
            node1 = next((node for node in remaining_double_bond_pool if node.global_id not in paired), None)
            if node1 is None:
                break
            partner = next(
                (
                    node for node in remaining_double_bond_pool
                    if node.global_id != node1.global_id
                    and node.global_id not in paired
                    and self._can_add_hop1_connection(nodes, node1, node)
                ),
                None,
            )
            if partner is None:
                paired.add(node1.global_id)
                continue
            paired.update((node1.global_id, partner.global_id))
            self._add_bidirectional_hop1(
                nodes, node1.global_id, partner.global_id, lock=True
            )

        # 三键配对 (17, 18) - 17的不饱和端连接度=1，18的连接度=1
        su17_nodes = self._get_nodes_by_su_type(nodes, 17)
        su18_nodes = list(self._get_nodes_by_su_type(nodes, 18))

        # 先17-18配对
        for n17 in su17_nodes:
            if su18_nodes and n17.remaining_hop1_slots() > 0:
                partner_idx = None
                for idx, n18 in enumerate(su18_nodes):
                    if self._can_add_hop1_connection(nodes, n17, n18):
                        partner_idx = idx
                        break
                if partner_idx is not None:
                    n18 = su18_nodes.pop(partner_idx)
                    self._add_bidirectional_hop1(
                        nodes, n17.global_id, n18.global_id, lock=True
                    )

        # 如果还有剩余的17且port2未填充（即没有17/18邻居），才做17-17配对
        # SU17端口规则: [{芳香}, {17, 18}]，port2只能放17或18
        # 已有SU18邻居的SU17的port2已满，剩余槽位是port1(芳香)，不能再放17
        remaining_17_no_triple = [
            n for n in su17_nodes
            if n.remaining_hop1_slots() > 0
            and 18 not in n.hop1_su and 17 not in n.hop1_su
        ]
        while len(remaining_17_no_triple) >= 2:
            n1 = remaining_17_no_triple.pop(0)
            partner_idx = None
            for idx, n2 in enumerate(remaining_17_no_triple):
                if self._can_add_hop1_connection(nodes, n1, n2):
                    partner_idx = idx
                    break
            if partner_idx is None:
                continue
            n2 = remaining_17_no_triple.pop(partner_idx)
            self._add_bidirectional_hop1(
                nodes, n1.global_id, n2.global_id, lock=True
            )

        invalid_unsaturated = []
        for node in list(su14_nodes) + list(su15_nodes) + list(su16_nodes):
            pair_count = sum(
                int(count) for su, count in node.hop1_su.items() if int(su) in {14, 15, 16}
            )
            if pair_count != 1:
                invalid_unsaturated.append((int(node.global_id), int(node.su_type), int(pair_count)))
        for node in list(su17_nodes) + list(self._get_nodes_by_su_type(nodes, 18)):
            pair_count = sum(
                int(count) for su, count in node.hop1_su.items() if int(su) in {17, 18}
            )
            if pair_count != 1:
                invalid_unsaturated.append((int(node.global_id), int(node.su_type), int(pair_count)))
        if invalid_unsaturated:
            raise RuntimeError(
                "[Unsaturated Fixed Assign] fixed-role matching incomplete: "
                + str(invalid_unsaturated[:8])
            )

    def _assign_fixed_su10_external_roles(self, nodes: List[_NodeV3]) -> None:
        """Assign the strict one-to-one SU10 self-pair role."""
        self_nodes = self._canonical_node_order(
            self._get_nodes_by_su_type(nodes, 10),
            salt=601,
        )
        need_fixed_port = [
            node
            for node in self_nodes
            if int(node.remaining_hop1_slots()) > 0
            and int(node.hop1_su.get(10, 0)) == 0
        ]
        if len(need_fixed_port) % 2 != 0:
            raise RuntimeError(
                "[SU10 Fixed Assign] 自连接角色节点数必须为偶数: "
                f"count={len(need_fixed_port)}"
            )
        while need_fixed_port:
            left = need_fixed_port.pop(0)
            right = need_fixed_port.pop(0)
            if not self._add_bidirectional_hop1(
                nodes,
                int(left.global_id),
                int(right.global_id),
                lock=True,
            ):
                raise RuntimeError(
                    "[SU10 Fixed Assign] 固定自连接提交失败: "
                    f"edge=({int(left.global_id)}, {int(right.global_id)})"
                )

        invalid = [
            (int(node.global_id), int(node.hop1_su.get(10, 0)))
            for node in self_nodes
            if int(node.hop1_su.get(10, 0)) != 1
        ]
        if invalid:
            raise RuntimeError(
                "[SU10 Fixed Assign] fixed-role matching incomplete: "
                + str(invalid[:8])
            )

    # ========================================================================
    # Layer1: 强制边、全局补全、结构验收和差谱调整
    # ========================================================================

    def audit_layer1_shift_coverage(
        self,
        nodes: List[_NodeV3],
        lib_path: str,
    ) -> Dict[str, Any]:
        """Audit the shift index after structure construction; never guide it."""
        if (
            self._hop1_adjuster is None
            or str(lib_path) != str(self._exact_hop1_constraint_lib_path)
        ):
            self._configure_exact_hop1_shift_constraints(str(lib_path))
        if self._hop1_adjuster is None:
            raise ValueError('Layer1 shift audit requires an active template library')
        median_index = self._hop1_adjuster.get_exact_hop1_median_index()
        missing: Counter = Counter()
        carbon_count = 0
        for node in nodes:
            center_su = int(getattr(node, 'su_type', -1))
            if center_su not in self._carbon_su_set:
                continue
            carbon_count += 1
            hop1 = tuple(sorted(
                int(x) for x in Counter(getattr(node, 'hop1_su', {}) or {}).elements()
            ))
            key = (center_su, hop1)
            if key not in median_index:
                missing[key] += 1
        rows = [
            {
                'center_su': int(key[0]),
                'hop1': list(key[1]),
                'node_count': int(count),
            }
            for key, count in sorted(missing.items())
        ]
        return {
            'ok': not bool(missing),
            'carbon_count': int(carbon_count),
            'matched_count': int(carbon_count - sum(missing.values())),
            'missing_node_count': int(sum(missing.values())),
            'missing_signatures': rows,
        }

    def build_rule_valid_graph(
        self,
        H_init: torch.Tensor,
        E_target: torch.Tensor,
        eval_lib_path: Optional[str] = None,
        identity_seed_nodes: Optional[List[_NodeV3]] = None,
        verbose: bool = False,
        forced_decision: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[_NodeV3], Dict[str, Any]]:
        """Build one graph from fixed roles and exact Layer1 shift signatures."""
        H_local = H_init.to(self.device)
        E_local = E_target.to(self.device)
        self._current_E_target = E_local
        self._configure_exact_hop1_shift_constraints(eval_lib_path)
        role_audit = audit_layer1_port_roles(
            H_local,
            self._get_special_degree_meta(H_local),
        )
        if not bool(role_audit.get('ok', False)):
            raise RuntimeError(
                "layer1_port_roles_unbalanced: "
                f"total={role_audit['total_ports']} "
                f"aromatic_internal={role_audit['aromatic_internal_ports']} "
                f"non_aromatic_internal={role_audit['non_aromatic_internal_ports']}; "
                "Layer0/Stage3 must apply pre-Layer1 port-role balance"
            )
        nodes = (
            self._initialize_node_pool(H_local)
            if identity_seed_nodes is None
            else self._initialize_node_pool(
                H_local,
                identity_seed_nodes=identity_seed_nodes,
            )
        )
        self._init_runtime_node_index(nodes)
        initial_open_ports = sum(
            int(node.remaining_hop1_slots()) for node in nodes
        )
        if initial_open_ports % 2:
            raise RuntimeError(
                'layer1_rule_graph_infeasible: odd entry open-port count '
                f'({initial_open_ports}); Layer0/Stage3 must apply one 12/13 balance '
                'before Layer1'
            )
        if verbose:
            print("\n" + "=" * 60)
            print(f"Layer1: 单次端口规则构图 ({len(nodes)}个节点)")
            print("=" * 60)
        build_t0 = time.perf_counter()
        fixed_timings: Dict[str, float] = {}

        def _timed_step(label: str, fn, rebuild_index: bool = True) -> None:
            step_t0 = time.perf_counter()
            fn()
            if bool(rebuild_index):
                self._init_runtime_node_index(nodes)
            elapsed = time.perf_counter() - step_t0
            fixed_timings[str(label)] = float(elapsed)
            if verbose and float(elapsed) >= 0.05:
                print(f"  [Layer1 Timing] {label}: {elapsed:.2f}s")

        _timed_step("a fixed_halogen_X", lambda: self._assign_fixed_halogen_X(nodes))
        _timed_step("b fixed_thioether_S", lambda: self._assign_fixed_thioether_S(nodes))
        _timed_step("c fixed_amine_N", lambda: self._assign_fixed_amine_N(nodes))
        _timed_step("d fixed_ether_O", lambda: self._assign_fixed_ether_O(nodes))
        _timed_step("e fixed_carbonyl", lambda: self._assign_fixed_carbonyl(nodes))
        _timed_step("f unsaturated_pairs", lambda: self._assign_unsaturated_pairs(nodes))
        _timed_step(
            "g reserve_alkyl_aromatic_11",
            lambda: self._reserve_alkyl_aromatic_11_external_roles(nodes),
        )
        _timed_step(
            "h fixed_su10_external",
            lambda: self._assign_fixed_su10_external_roles(nodes),
            rebuild_index=False,
        )
        self._init_runtime_node_index(nodes)
        fixed_edge_count = sum(len(node.hop1_ids) for node in nodes) // 2
        fixed_open_ports = sum(int(node.remaining_hop1_slots()) for node in nodes)
        if verbose:
            print(
                "  [Layer1 Fixed Phase] "
                f"edges={fixed_edge_count}, open_ports={fixed_open_ports}"
            )

        phase_ports = self._raise_if_layer1_phase_ports_infeasible(nodes)
        if verbose:
            print(
                "  [Layer1 Phase Ports] "
                f"SU11_external={int(phase_ports['su11_external_demands'])}, "
                f"non_aromatic={int(phase_ports['non_aromatic_open_ports'])}"
                f"->{int(phase_ports['projected_non_aromatic_internal_ports'])}, "
                f"aromatic={int(phase_ports['aromatic_open_ports_including_su11_external'])}"
                f"->{int(phase_ports['projected_aromatic_ring_ports'])}"
            )

        non_aromatic_completion = dict(
            Layer1GreedyCompleter(
                self,
                nodes,
                phase='non_aromatic',
                verbose=verbose,
                forced_decision=forced_decision,
            ).solve()
        )
        non_aromatic_errors = self._non_aromatic_completion_errors(nodes)
        if non_aromatic_errors:
            raise RuntimeError(
                "layer1_non_aromatic_completion_incomplete: "
                + " | ".join(non_aromatic_errors[:8])
            )
        self._init_runtime_node_index(nodes)
        if verbose:
            print(
                "  [Layer1 Non-Aromatic Completion] "
                f"decisions={int(non_aromatic_completion.get('decisions', 0))}, "
                f"backtracks={int(non_aromatic_completion.get('backtracks', 0))}, "
                f"elapsed={float(non_aromatic_completion.get('elapsed_s', 0.0)):.2f}s"
            )

        aromatic_open_ports = int(sum(
            int(node.remaining_hop1_slots())
            for node in nodes
            if int(node.su_type) in LAYER1_AROMATIC_RING_TYPES
        ))
        if int(aromatic_open_ports) % 2 != 0:
            raise RuntimeError(
                "layer1_aromatic_phase_infeasible: "
                f"open_ports={aromatic_open_ports}"
            )
        aromatic_completion = dict(
            Layer1GreedyCompleter(
                self,
                nodes,
                phase='aromatic',
                verbose=verbose,
                forced_decision=forced_decision,
            ).solve()
        )
        if verbose:
            print(
                "  [Layer1 Aromatic Completion] "
                f"decisions={int(aromatic_completion.get('decisions', 0))}, "
                f"backtracks={int(aromatic_completion.get('backtracks', 0))}, "
                f"elapsed={float(aromatic_completion.get('elapsed_s', 0.0)):.2f}s"
            )

        completion = {
            'solver': 'two_phase_diversity_first_greedy_bounded_repair',
            'phase_port_diagnostics': dict(phase_ports),
            'non_aromatic': dict(non_aromatic_completion),
            'aromatic': dict(aromatic_completion),
            'decisions': int(non_aromatic_completion.get('decisions', 0))
            + int(aromatic_completion.get('decisions', 0)),
            'backtracks': int(non_aromatic_completion.get('backtracks', 0))
            + int(aromatic_completion.get('backtracks', 0)),
            'repair_attempts': int(non_aromatic_completion.get('repair_attempts', 0))
            + int(aromatic_completion.get('repair_attempts', 0)),
            'elapsed_s': float(non_aromatic_completion.get('elapsed_s', 0.0))
            + float(aromatic_completion.get('elapsed_s', 0.0)),
            'fixed_edge_count': int(fixed_edge_count),
            'fixed_open_ports': int(fixed_open_ports),
            'fixed_timings_s': fixed_timings,
            'build_elapsed_s': float(time.perf_counter() - build_t0),
            'branch_points': list(
                non_aromatic_completion.get('branch_points', []) or []
            ) + list(aromatic_completion.get('branch_points', []) or []),
            'forced_decision_requested': dict(forced_decision or {}),
            'forced_decision_applied': bool(
                non_aromatic_completion.get('forced_decision_applied', False)
                or aromatic_completion.get('forced_decision_applied', False)
            ),
        }
        if forced_decision and not bool(completion['forced_decision_applied']):
            raise RuntimeError(
                'layer1_multistart_replay_decision_not_applied: '
                f"decision={dict(forced_decision)}"
            )
        valid, errors = self.validate_graph_consistency(
            nodes=nodes,
            H=H_local,
            E_target=E_local,
            verbose=False,
        )
        if not bool(valid) or errors:
            raise RuntimeError(
                "layer1_rule_graph_validation_failed: "
                + " | ".join(str(error) for error in list(errors or [])[:8])
            )
        self._last_completion_diagnostics = dict(completion)
        if verbose:
            print(
                "  [Layer1 Rule Completion] "
                f"decisions={int(completion.get('decisions', 0))}, "
                f"backtracks={int(completion.get('backtracks', 0))}, "
                f"elapsed={float(completion.get('elapsed_s', 0.0)):.2f}s"
            )
        return nodes, completion

    @staticmethod
    def _graph_signature(nodes: List[_NodeV3]) -> str:
        payload = tuple(
            (
                int(node.global_id),
                int(node.su_type),
                tuple(sorted(int(value) for value in list(node.hop1_ids or []))),
            )
            for node in sorted(nodes, key=lambda item: int(item.global_id))
        )
        return hashlib.sha1(repr(payload).encode('utf-8')).hexdigest()

    def build_rule_valid_graph_variants(
        self,
        H_init: torch.Tensor,
        E_target: torch.Tensor,
        eval_lib_path: str,
        identity_seed_nodes: Optional[List[_NodeV3]] = None,
        candidate_count: int = 6,
        branch_limit: int = 12,
        verbose: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build deterministic one-fork variants within the best primary tier."""
        requested_count = max(1, int(candidate_count))
        baseline_nodes, baseline_completion = self.build_rule_valid_graph(
            H_init=H_init,
            E_target=E_target,
            eval_lib_path=eval_lib_path,
            identity_seed_nodes=copy.deepcopy(identity_seed_nodes),
            verbose=verbose,
        )
        baseline_signature = self._graph_signature(baseline_nodes)
        variants: List[Dict[str, Any]] = [{
            'variant_id': 'layer1:baseline',
            'nodes': baseline_nodes,
            'construction': baseline_completion,
            'branch': None,
            'graph_signature': baseline_signature,
        }]
        seen_signatures = {str(baseline_signature)}
        failures: List[Dict[str, Any]] = []
        all_branch_points = list(
            baseline_completion.get('branch_points', []) or []
        )
        branch_point_limit = max(0, int(branch_limit))
        branch_points_by_phase: Dict[str, List[Dict[str, Any]]] = {}
        phase_order: List[str] = []
        for point in all_branch_points:
            phase = str(point.get('phase', '') or 'unknown')
            if phase not in branch_points_by_phase:
                branch_points_by_phase[phase] = []
                phase_order.append(phase)
            branch_points_by_phase[phase].append(point)
        branch_points: List[Dict[str, Any]] = []
        phase_offsets = {str(phase): 0 for phase in phase_order}
        while len(branch_points) < int(branch_point_limit):
            added = False
            for phase in phase_order:
                offset = int(phase_offsets[str(phase)])
                group = branch_points_by_phase[str(phase)]
                if offset >= len(group):
                    continue
                branch_points.append(group[offset])
                phase_offsets[str(phase)] = int(offset) + 1
                added = True
                if len(branch_points) >= int(branch_point_limit):
                    break
            if not added:
                break
        directives: List[Dict[str, Any]] = []
        max_alternatives = max(
            [
                len(list(point.get('alternative_partner_ids', []) or []))
                for point in branch_points
            ]
            or [0]
        )
        # First visit distinct decisions across both phases. Only after every
        # selected decision has contributed one fork do we use its next peer.
        for alternative_index in range(int(max_alternatives)):
            for point in branch_points:
                alternatives = list(
                    point.get('alternative_partner_ids', []) or []
                )
                if int(alternative_index) >= len(alternatives):
                    continue
                partner_id = int(alternatives[int(alternative_index)])
                directives.append({
                    'phase': str(point.get('phase', '')),
                    'stack_index': int(point.get('stack_index', -1)),
                    'center_id': int(point.get('center_id', -1)),
                    'baseline_partner_id': int(
                        point.get('chosen_partner_id', -1)
                    ),
                    'chosen_partner_id': int(partner_id),
                    'prefix_edge_signature': [
                        (int(edge[0]), int(edge[1]))
                        for edge in list(
                            point.get('prefix_edge_signature', []) or []
                        )
                    ],
                    'primary_frontier': [
                        int(value)
                        for value in list(point.get('primary_frontier', []) or [])
                    ],
                })

        for directive in directives:
            if len(variants) >= int(requested_count):
                break
            try:
                nodes, completion = self.build_rule_valid_graph(
                    H_init=H_init,
                    E_target=E_target,
                    eval_lib_path=eval_lib_path,
                    identity_seed_nodes=copy.deepcopy(identity_seed_nodes),
                    verbose=verbose,
                    forced_decision=directive,
                )
            except Exception as exc:
                failures.append({
                    'branch': dict(directive),
                    'reason': str(exc),
                })
                continue
            graph_signature = self._graph_signature(nodes)
            if str(graph_signature) in seen_signatures:
                continue
            seen_signatures.add(str(graph_signature))
            variants.append({
                'variant_id': f"layer1:fork:{len(variants)}",
                'nodes': nodes,
                'construction': completion,
                'branch': dict(directive),
                'graph_signature': str(graph_signature),
            })

        return variants, {
            'requested_candidate_count': int(requested_count),
            'generated_candidate_count': int(len(variants)),
            'branch_limit': int(max(0, int(branch_limit))),
            'baseline_branch_point_count': int(len(
                baseline_completion.get('branch_points', []) or []
            )),
            'considered_branch_point_count': int(len(branch_points)),
            'baseline_branch_point_count_by_phase': {
                str(phase): int(len(points))
                for phase, points in branch_points_by_phase.items()
            },
            'considered_branch_point_count_by_phase': dict(Counter(
                str(point.get('phase', '') or 'unknown')
                for point in branch_points
            )),
            'directive_count': int(len(directives)),
            'failed_branch_count': int(len(failures)),
            'failures': failures,
        }

    def layer1_assign(self, H_init: torch.Tensor, S_target: torch.Tensor,
                      E_target: torch.Tensor,
                      eval_nmr: bool = True,
                      eval_output_dir: Optional[str] = None,
                      eval_lib_path: Optional[str] = None,
                      neg_peak_threshold: float = -0.8,
                      pos_peak_threshold: float = 0.8,
                      max_candidate_trials_per_region: int = 40,
                      max_moves_per_region: int = 6,
                      region_adjustment_cycles: int = 2,
                      prebuilt_nodes: Optional[List[_NodeV3]] = None,
                      identity_seed_nodes: Optional[List[_NodeV3]] = None,
                      verbose: bool = False,
                      emit_summary: bool = True) -> List[_NodeV3]:
        """Build, validate, and regionally adjust one exact Layer1 graph."""
        layer1_t0 = time.perf_counter()
        H_init = H_init.to(self.device)
        S_target = S_target.to(self.device)
        E_target = E_target.to(self.device)
        self._current_E_target = E_target
        self._last_hop1_adjust_diagnostics = {}
        if prebuilt_nodes is None:
            nodes, _completion_summary = self.build_rule_valid_graph(
                H_init=H_init,
                E_target=E_target,
                eval_lib_path=eval_lib_path,
                identity_seed_nodes=identity_seed_nodes,
                verbose=verbose,
            )
        else:
            self._configure_exact_hop1_shift_constraints(eval_lib_path)
            nodes = copy.deepcopy(prebuilt_nodes)
            self._init_runtime_node_index(nodes)
            if verbose:
                print("\n" + "=" * 60)
                print(f"Layer1: 单图精炼 ({len(nodes)}个节点)")
                print("=" * 60)

        check_t0 = time.perf_counter()
        is_valid, errors = self.validate_graph_consistency(
            nodes=nodes,
            H=H_init,
            E_target=E_target,
            verbose=False
        )
        check_elapsed = time.perf_counter() - check_t0
        if verbose and float(check_elapsed) >= 0.05:
            print(f"  [Layer1 Timing] consistency_check: {check_elapsed:.2f}s")
        if not is_valid:
            print(f"  ⚠ 图结构存在{len(errors)}个不一致")
            self._print_consistency_errors(errors, prefix='Layer1')
        if errors:
            raise RuntimeError(
                "layer1_global_completion_invalid: "
                + " | ".join(str(error) for error in list(errors or [])[:8])
            )
        if not eval_lib_path or self._hop1_adjuster is None:
            raise ValueError('Layer1 requires an active exact template library')
        coverage = self.audit_layer1_shift_coverage(nodes, str(eval_lib_path))
        self._last_shift_coverage_audit = dict(coverage)
        if not bool(coverage.get('ok', False)):
            raise RuntimeError(
                "rule_valid_but_shift_template_missing: "
                f"matched={int(coverage.get('matched_count', 0))}/"
                f"{int(coverage.get('carbon_count', 0))}, "
                f"missing={list(coverage.get('missing_signatures', []) or [])[:8]}"
            )

        raw_nodes = copy.deepcopy(nodes)
        raw_snapshot = self._hop1_adjuster.build_eval_snapshot(raw_nodes, S_target)
        raw_r2 = float(raw_snapshot.get('r2', 0.0))
        try:
            nodes, adjust_summary = self._hop1_adjuster.adjust_regions(
                nodes=nodes,
                S_target=S_target,
                E_target=E_target,
                validator=lambda candidate: self.validate_graph_consistency(
                    nodes=candidate,
                    H=H_init,
                    E_target=E_target,
                    verbose=False,
                ),
                neg_peak_threshold=float(neg_peak_threshold),
                pos_peak_threshold=float(pos_peak_threshold),
                max_candidate_trials_per_region=int(max_candidate_trials_per_region),
                max_moves_per_region=int(max_moves_per_region),
                region_adjustment_cycles=int(region_adjustment_cycles),
            )
            self._last_hop1_adjust_diagnostics = dict(adjust_summary)
        except Exception as exc:
            nodes = raw_nodes
            self._last_hop1_adjust_diagnostics = {
                'error': str(exc),
                'total_adjustments': 0,
            }
            print(f"  [Layer1区域调整回退] {exc}")

        final_ok, final_errors = self.validate_graph_consistency(
            nodes=nodes,
            H=H_init,
            E_target=E_target,
            verbose=False,
        )
        final_snapshot = self._hop1_adjuster.build_eval_snapshot(nodes, S_target)
        final_r2 = float(final_snapshot.get('r2', 0.0))
        if not final_ok or final_errors or final_r2 + 1e-12 < raw_r2:
            nodes = raw_nodes
            final_snapshot = raw_snapshot
            final_r2 = raw_r2
            self._last_hop1_adjust_diagnostics['final_fallback_to_raw'] = True
            self._last_hop1_adjust_diagnostics['final_validation_errors'] = list(
                final_errors or []
            )
        self._hop1_adjuster.apply_assignments_to_nodes(
            nodes,
            final_snapshot.get('assignments', {}),
        )

        final_coverage = self.audit_layer1_shift_coverage(nodes, str(eval_lib_path))
        self._last_shift_coverage_audit = dict(final_coverage)
        if not bool(final_coverage.get('ok', False)):
            raise RuntimeError(
                "rule_valid_but_shift_template_missing_after_refinement: "
                f"missing={list(final_coverage.get('missing_signatures', []) or [])[:8]}"
            )
        if eval_output_dir:
            out = Path(eval_output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                list(self._last_hop1_adjust_diagnostics.get('moves', []) or [])
            ).to_csv(out / 'layer1_hop1_adjust_moves.csv', index=False)
        if eval_nmr:
            self._hop1_adjuster.evaluate_with_library(
                nodes=nodes,
                S_target=S_target,
                output_dir=eval_output_dir,
            )
        if verbose:
            print(
                "  [Layer1 fixed-scale] "
                f"raw_r2={raw_r2:.4f}, final_r2={final_r2:.4f}, "
                f"moves={int(self._last_hop1_adjust_diagnostics.get('total_adjustments', 0))}, "
                f"cycles={int(self._last_hop1_adjust_diagnostics.get('completed_cycles', 0))}/"
                f"{int(self._last_hop1_adjust_diagnostics.get('configured_cycles', region_adjustment_cycles))}"
            )
            print(f"Layer1 完成 ({time.perf_counter() - layer1_t0:.2f}s)\n")
        elif emit_summary:
            completion = dict(
                getattr(self, "_last_completion_diagnostics", {}) or {}
            )
            print(
                f"[Layer1] nodes={len(nodes)} "
                f"decisions={int(completion.get('decisions', 0) or 0)} "
                f"r2={raw_r2:.4f}->{final_r2:.4f} "
                f"moves={int(self._last_hop1_adjust_diagnostics.get('total_adjustments', 0) or 0)} "
                f"cycles={int(self._last_hop1_adjust_diagnostics.get('completed_cycles', 0) or 0)}/"
                f"{int(self._last_hop1_adjust_diagnostics.get('configured_cycles', region_adjustment_cycles) or region_adjustment_cycles)} "
                f"elapsed={time.perf_counter() - layer1_t0:.2f}s",
                flush=True,
            )
        return nodes
    def precheck_fixed_assignment_feasibility(self,
                                              H_init: torch.Tensor,
                                              E_target: torch.Tensor) -> Tuple[bool, str]:
        """Cheaply reject histogram candidates whose fixed anchors cannot be matched."""
        H_local = H_init.to(self.device)
        self._current_E_target = E_target.to(self.device)
        role_audit = audit_layer1_port_roles(
            H_local,
            self._get_special_degree_meta(H_local),
        )
        if not bool(role_audit.get('ok', False)):
            return False, (
                'fixed_assignment_infeasible:port_roles:'
                f"total={role_audit['total_ports']},"
                f"aromatic_internal={role_audit['aromatic_internal_ports']},"
                f"non_aromatic_internal={role_audit['non_aromatic_internal_ports']}"
            )
        nodes = self._initialize_node_pool(H_local)
        self._init_runtime_node_index(nodes)
        steps = (
            ('halogen', self._assign_fixed_halogen_X),
            ('thioether', self._assign_fixed_thioether_S),
            ('amine', self._assign_fixed_amine_N),
            ('ether', self._assign_fixed_ether_O),
            ('carbonyl', self._assign_fixed_carbonyl),
            ('unsaturated', self._assign_unsaturated_pairs),
            ('su11_role', self._reserve_alkyl_aromatic_11_external_roles),
            ('su10_external', self._assign_fixed_su10_external_roles),
        )
        for label, fn in steps:
            try:
                fn(nodes)
                self._init_runtime_node_index(nodes)
            except Exception as exc:
                return False, f'fixed_assignment_infeasible:{label}:{exc}'
        diagnostics = self._layer1_phase_port_diagnostics(nodes)
        if not bool(diagnostics.get('ok', False)):
            return False, (
                'fixed_assignment_infeasible:phase_ports:'
                f"su11_external={diagnostics['su11_external_demands']},"
                f"non_aromatic_internal="
                f"{diagnostics['projected_non_aromatic_internal_ports']},"
                f"aromatic_ring={diagnostics['projected_aromatic_ring_ports']}"
            )
        return True, 'ok'

    def evaluate_layer1_nmr_with_library(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        lib_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        if lib_path and str(lib_path) != self._exact_hop1_constraint_lib_path:
            self._configure_exact_hop1_shift_constraints(str(lib_path))
        if self._hop1_adjuster is None:
            raise ValueError('Layer1 evaluation requires an active template library')
        return self._hop1_adjuster.evaluate_with_library(
            nodes=nodes,
            S_target=S_target,
            output_dir=output_dir,
        )
