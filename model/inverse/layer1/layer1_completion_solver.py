from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class Layer1CompletionSolver:
    """Complete every residual Layer1 port under exact-template constraints."""

    def __init__(
        self,
        assigner: Any,
        nodes: List[Any],
        *,
        deadline: Optional[float],
        max_backtracks: Optional[int],
        branch_limit: Optional[int],
    ) -> None:
        self.assigner = assigner
        self.nodes = nodes
        self.deadline = float(deadline) if deadline is not None else None
        self.max_backtracks = (
            None if max_backtracks is None else max(0, int(max_backtracks))
        )
        self.branch_limit = (
            None if branch_limit is None else max(1, int(branch_limit))
        )
        self.backtracks = 0
        self.forward_rejections: Counter = Counter()
        self.timings_s: Counter = Counter()
        self.edges_added = 0
        self.max_depth = 0
        self.started_at = time.perf_counter()
        self._template_option_cache: Dict[Tuple[int, Tuple[int, ...], int], Tuple[Tuple[int, ...], ...]] = {}
        self._semantic_pair_cache: Dict[Tuple[Any, Any], bool] = {}
        self._feasible_option_cache: Dict[Tuple[Any, Tuple[Tuple[int, int], ...]], Tuple[Tuple[int, ...], ...]] = {}
        self._candidate_type_cache: Dict[Tuple[Any, Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, int], ...]], Tuple[int, ...]] = {}
        self._scarcity_cache: Dict[Tuple[Any, Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, int], ...]], float] = {}
        self._state_keys: Dict[int, Tuple[Any, ...]] = {}
        self._state_groups: Dict[Tuple[Any, ...], Set[int]] = {}
        self._base_state = self._capture_state()
        self._domains: Dict[int, Set[int]] = {}
        self._domain_type_counts: Dict[int, Counter] = {}
        self._availability_key: Tuple[Tuple[int, int], ...] = ()
        self._motif_usage: Counter = Counter()
        self._complete_motif_usage: Counter = Counter()
        self._build_variant = int(getattr(assigner, "_build_variant", 0) or 0)
        self._rebuild_semantic_groups()
        self._availability_key = self._compute_availability_signature()
        self._rebuild_domains()
        self._rebuild_motif_usage()

    def _capture_state(self) -> List[Tuple[Counter, List[int], Set[int]]]:
        return [
            (
                Counter(getattr(node, "hop1_su", {}) or {}),
                list(getattr(node, "hop1_ids", []) or []),
                set(getattr(node, "fixed_hop1_ids", set()) or set()),
            )
            for node in self.nodes
        ]

    def _restore_state(self, state: Sequence[Tuple[Counter, List[int], Set[int]]]) -> None:
        for node, (hop1_su, hop1_ids, fixed_ids) in zip(self.nodes, state):
            node.hop1_su = Counter(hop1_su)
            node.hop1_ids = list(hop1_ids)
            node.fixed_hop1_ids = set(fixed_ids)
        self.assigner._init_runtime_node_index(self.nodes)
        self._rebuild_motif_usage()

    @staticmethod
    def _motif_key(node: Any, neighbor_types: Optional[Sequence[int]] = None) -> Tuple[int, Tuple[int, ...]]:
        motif = (
            tuple(sorted(int(x) for x in neighbor_types))
            if neighbor_types is not None
            else tuple(sorted(int(x) for x in Counter(getattr(node, "hop1_su", {}) or {}).elements()))
        )
        return int(node.su_type), motif

    def _rebuild_motif_usage(self) -> None:
        self._motif_usage = Counter()
        self._complete_motif_usage = Counter()
        for node in self.nodes:
            key = self._motif_key(node)
            self._motif_usage[key] += 1
            if int(node.remaining_hop1_slots()) == 0:
                self._complete_motif_usage[key] += 1

    def _remove_motif_usage(self, node_ids: Set[int]) -> None:
        for node_id in node_ids:
            node = self.nodes[int(node_id)]
            key = self._motif_key(node)
            self._motif_usage[key] -= 1
            if self._motif_usage[key] <= 0:
                del self._motif_usage[key]
            if int(node.remaining_hop1_slots()) == 0:
                self._complete_motif_usage[key] -= 1
                if self._complete_motif_usage[key] <= 0:
                    del self._complete_motif_usage[key]

    def _add_motif_usage(self, node_ids: Set[int]) -> None:
        for node_id in node_ids:
            node = self.nodes[int(node_id)]
            key = self._motif_key(node)
            self._motif_usage[key] += 1
            if int(node.remaining_hop1_slots()) == 0:
                self._complete_motif_usage[key] += 1

    def _deadline_reached(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= float(self.deadline)

    def _backtrack_limit_reached(self) -> bool:
        return (
            self.max_backtracks is not None
            and self.backtracks >= int(self.max_backtracks)
        )

    def _semantic_state_key(self, node: Any) -> Tuple[Any, ...]:
        neighbor_states = []
        for neighbor_id in list(getattr(node, "hop1_ids", []) or []):
            neighbor = self.nodes[int(neighbor_id)]
            neighbor_states.append((
                int(neighbor.su_type),
                int(neighbor.get_max_degree()),
                int(neighbor.remaining_hop1_slots()),
                getattr(neighbor, "special_anchor_partition", None),
                int(getattr(neighbor, "target_fixed_anchor_count", 0) or 0),
                getattr(neighbor, "special_anchor_mode", None),
            ))
        return (
            int(node.su_type),
            int(node.get_max_degree()),
            int(node.remaining_hop1_slots()),
            tuple(sorted(int(x) for x in Counter(getattr(node, "hop1_su", {}) or {}).elements())),
            getattr(node, "special_anchor_partition", None),
            int(getattr(node, "target_fixed_anchor_count", 0) or 0),
            getattr(node, "special_anchor_mode", None),
            tuple(sorted(neighbor_states, key=repr)),
        )

    def _rebuild_semantic_groups(self) -> None:
        self._state_keys = {}
        self._state_groups = {}
        for node in self.nodes:
            node_id = int(node.global_id)
            key = self._semantic_state_key(node)
            self._state_keys[node_id] = key
            self._state_groups.setdefault(key, set()).add(node_id)

    def _refresh_semantic_groups(self, endpoint_ids: Set[int]) -> None:
        for node_id in endpoint_ids:
            node_i = int(node_id)
            old_key = self._state_keys.get(node_i)
            new_key = self._semantic_state_key(self.nodes[node_i])
            if old_key == new_key:
                continue
            if old_key in self._state_groups:
                self._state_groups[old_key].discard(node_i)
                if not self._state_groups[old_key]:
                    del self._state_groups[old_key]
            self._state_keys[node_i] = new_key
            self._state_groups.setdefault(new_key, set()).add(node_i)

    def _semantic_pair_compatible(self, left_key: Tuple[Any, ...], right_key: Tuple[Any, ...]) -> bool:
        cache_key = tuple(sorted((left_key, right_key), key=repr))
        cached = self._semantic_pair_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        left_ids = self._state_groups.get(left_key, set())
        right_ids = self._state_groups.get(right_key, set())
        for left_id in left_ids:
            left = self.nodes[int(left_id)]
            if int(left.remaining_hop1_slots()) <= 0:
                continue
            existing = set(int(x) for x in list(getattr(left, "hop1_ids", []) or []))
            for right_id in right_ids:
                if int(left_id) == int(right_id) or int(right_id) in existing:
                    continue
                right = self.nodes[int(right_id)]
                if int(right.remaining_hop1_slots()) <= 0:
                    continue
                compatible = bool(self.assigner._can_add_hop1_connection(self.nodes, left, right))
                self._semantic_pair_cache[cache_key] = compatible
                return compatible
        self._semantic_pair_cache[cache_key] = False
        return False

    @staticmethod
    def _is_subcounter(current: Counter, target: Counter) -> bool:
        return all(int(target.get(su, 0)) >= int(count) for su, count in current.items())

    def _compute_availability_signature(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(
            (int(candidate_su), int(len(ids)))
            for candidate_su, ids in getattr(self.assigner, "_available_ids_by_su", {}).items()
            if ids
        ))

    def _availability_signature(self) -> Tuple[Tuple[int, int], ...]:
        return self._availability_key

    def _structural_template_options(self, node: Any) -> Tuple[Tuple[int, ...], ...]:
        su_type = int(getattr(node, "su_type", -1))
        if su_type not in set(getattr(self.assigner, "_carbon_su_set", set()) or set()):
            return ((),)
        current = Counter(getattr(node, "hop1_su", {}) or {})
        current_ms = tuple(sorted(int(x) for x in current.elements()))
        target_degree = int(node.get_max_degree())
        cache_key = (su_type, current_ms, target_degree)
        cached = self._template_option_cache.get(cache_key)
        if cached is not None:
            return cached
        options = []
        for template in getattr(self.assigner, "_hop1_templates_by_su", {}).get(su_type, ()):
            template_tuple = tuple(int(x) for x in tuple(template))
            if len(template_tuple) != target_degree:
                continue
            if self._is_subcounter(current, Counter(template_tuple)):
                options.append(template_tuple)
        result = tuple(sorted(set(options)))
        self._template_option_cache[cache_key] = result
        return result

    def _available_type_counts(self, node: Any) -> Counter:
        counts: Counter = Counter({
            int(su_type): int(len(ids))
            for su_type, ids in getattr(self.assigner, "_available_ids_by_su", {}).items()
        })
        node_id = int(getattr(node, "global_id", -1))
        if int(node.remaining_hop1_slots()) > 0:
            counts[int(node.su_type)] -= 1
        for candidate_id in list(getattr(node, "hop1_ids", []) or []):
            candidate = self.nodes[int(candidate_id)]
            if int(candidate.remaining_hop1_slots()) > 0:
                counts[int(candidate.su_type)] -= 1
        return counts

    def _feasible_template_options(self, node: Any) -> Tuple[Tuple[int, ...], ...]:
        options = self._structural_template_options(node)
        su_type = int(getattr(node, "su_type", -1))
        if su_type not in set(getattr(self.assigner, "_carbon_su_set", set()) or set()):
            return options
        state_key = self._state_keys.get(int(node.global_id), self._semantic_state_key(node))
        availability_key = self._availability_signature()
        cache_key = (state_key, availability_key)
        cached = self._feasible_option_cache.get(cache_key)
        if cached is not None:
            return cached
        current = Counter(getattr(node, "hop1_su", {}) or {})
        available = self._available_type_counts(node)
        feasible = []
        for template in options:
            missing = Counter(template) - current
            if all(int(available.get(su, 0)) >= int(count) for su, count in missing.items()):
                feasible.append(template)
        result = tuple(feasible)
        self._feasible_option_cache[cache_key] = result
        return result

    def _candidate_types(self, node: Any, options: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        state_key = self._state_keys.get(int(node.global_id), self._semantic_state_key(node))
        options_key = tuple(tuple(int(x) for x in option) for option in options)
        cache_key = (state_key, options_key, self._availability_signature())
        cached = self._candidate_type_cache.get(cache_key)
        if cached is not None:
            return cached
        su_type = int(getattr(node, "su_type", -1))
        if su_type in set(getattr(self.assigner, "_carbon_su_set", set()) or set()):
            current = Counter(getattr(node, "hop1_su", {}) or {})
            missing_types: Set[int] = set()
            for template in options:
                missing_types.update(int(x) for x in (Counter(template) - current).keys())
            candidate_types = sorted(missing_types)
        else:
            candidate_types = sorted(
                int(su)
                for su, ids in getattr(self.assigner, "_available_ids_by_su", {}).items()
                if ids
            )

        if self.assigner._has_pending_required_external(node, self.nodes):
            required = set(int(x) for x in self.assigner._required_external_candidates_for_node(node))
            candidate_types = [su for su in candidate_types if su in required]
        result = tuple(candidate_types)
        self._candidate_type_cache[cache_key] = result
        return result

    def _resource_scarcity(self, node: Any, options: Sequence[Tuple[int, ...]]) -> float:
        state_key = self._state_keys.get(int(node.global_id), self._semantic_state_key(node))
        options_key = tuple(tuple(int(x) for x in option) for option in options)
        cache_key = (state_key, options_key, self._availability_signature())
        cached = self._scarcity_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        current = Counter(getattr(node, "hop1_su", {}) or {})
        missing_options = [Counter(template) - current for template in options if template]
        scarcity = 0.0
        if missing_options:
            available = self._available_type_counts(node)
            scarcity = max(
                sum(float(count) / max(1.0, float(available.get(su, 0))) for su, count in missing.items())
                for missing in missing_options
            )
        self._scarcity_cache[cache_key] = float(scarcity)
        return float(scarcity)

    def _compute_legal_partners(
        self,
        node: Any,
        options: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> List[Any]:
        if int(node.remaining_hop1_slots()) <= 0:
            return []
        options = tuple(options) if options is not None else self._feasible_template_options(node)
        node_id = int(getattr(node, "global_id", -1))
        existing = set(int(x) for x in list(getattr(node, "hop1_ids", []) or []))
        candidates: List[Any] = []
        seen: Set[int] = set()
        candidate_types = set(self._candidate_types(node, options))
        center_key = self._state_keys[int(node_id)]
        for candidate_key, candidate_group_ids in self._state_groups.items():
            if int(candidate_key[0]) not in candidate_types:
                continue
            if not self._semantic_pair_compatible(center_key, candidate_key):
                continue
            for candidate_id in sorted(int(x) for x in candidate_group_ids):
                if candidate_id == node_id or candidate_id in existing or candidate_id in seen:
                    continue
                candidate = self.nodes[candidate_id]
                if int(candidate.remaining_hop1_slots()) <= 0:
                    continue
                seen.add(candidate_id)
                candidates.append(candidate)
        return candidates

    def _rebuild_domains(self) -> None:
        self._domains = {}
        self._domain_type_counts = {}
        for node in self.nodes:
            node_id = int(node.global_id)
            if int(node.remaining_hop1_slots()) <= 0:
                self._domains[node_id] = set()
                self._domain_type_counts[node_id] = Counter()
                continue
            options = self._feasible_template_options(node)
            domain = {
                int(partner.global_id)
                for partner in self._compute_legal_partners(node, options)
            }
            self._domains[node_id] = domain
            self._domain_type_counts[node_id] = Counter(
                int(self.nodes[partner_id].su_type) for partner_id in domain
            )

    def _replace_domain(self, node_id: int, candidate_ids: Set[int]) -> None:
        node_i = int(node_id)
        domain = set(int(candidate_id) for candidate_id in candidate_ids)
        self._domains[node_i] = domain
        self._domain_type_counts[node_i] = Counter(
            int(self.nodes[candidate_id].su_type) for candidate_id in domain
        )

    def _discard_from_domain(self, node_id: int, candidate_id: int) -> None:
        node_i = int(node_id)
        candidate_i = int(candidate_id)
        domain = self._domains.setdefault(node_i, set())
        if candidate_i not in domain:
            return
        domain.remove(candidate_i)
        su_type = int(self.nodes[candidate_i].su_type)
        counts = self._domain_type_counts.setdefault(node_i, Counter())
        counts[su_type] -= 1
        if counts[su_type] <= 0:
            del counts[su_type]

    def _add_to_domain(self, node_id: int, candidate_id: int) -> None:
        node_i = int(node_id)
        candidate_i = int(candidate_id)
        domain = self._domains.setdefault(node_i, set())
        if candidate_i in domain:
            return
        domain.add(candidate_i)
        self._domain_type_counts.setdefault(node_i, Counter())[int(self.nodes[candidate_i].su_type)] += 1

    def _refresh_domains_for_endpoints(self, endpoint_ids: Set[int]) -> None:
        endpoint_ids = {int(node_id) for node_id in endpoint_ids}
        self._availability_key = self._compute_availability_signature()
        self._refresh_semantic_groups(endpoint_ids)
        for endpoint_id in endpoint_ids:
            endpoint = self.nodes[int(endpoint_id)]
            if int(endpoint.remaining_hop1_slots()) <= 0:
                self._replace_domain(int(endpoint_id), set())
            else:
                options = self._feasible_template_options(endpoint)
                self._replace_domain(int(endpoint_id), {
                    int(partner.global_id)
                    for partner in self._compute_legal_partners(endpoint, options)
                })

        for state_key, group_ids in list(self._state_groups.items()):
            active_ids = [
                int(node_id) for node_id in group_ids
                if int(node_id) not in endpoint_ids
                and int(self.nodes[int(node_id)].remaining_hop1_slots()) > 0
            ]
            if not active_ids:
                continue
            representative = self.nodes[active_ids[0]]
            options = self._feasible_template_options(representative)
            allowed_types = set(self._candidate_types(representative, options))
            endpoint_compatibility = {
                int(endpoint_id): bool(
                    int(self.nodes[int(endpoint_id)].remaining_hop1_slots()) > 0
                    and int(self.nodes[int(endpoint_id)].su_type) in allowed_types
                    and self._semantic_pair_compatible(
                        state_key,
                        self._state_keys[int(endpoint_id)],
                    )
                )
                for endpoint_id in endpoint_ids
            }
            for node_id in active_ids:
                existing_ids = set(int(x) for x in list(self.nodes[node_id].hop1_ids or []))
                for endpoint_id in endpoint_ids:
                    self._discard_from_domain(node_id, endpoint_id)
                    if endpoint_compatibility[int(endpoint_id)] and int(endpoint_id) not in existing_ids:
                        self._add_to_domain(node_id, endpoint_id)

        for node in self.nodes:
            node_id = int(node.global_id)
            if int(node.remaining_hop1_slots()) <= 0:
                self._replace_domain(node_id, set())

    def _domain_partner_count(
        self,
        node: Any,
        options: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> int:
        options = tuple(options) if options is not None else self._feasible_template_options(node)
        allowed_types = self._candidate_types(node, options)
        type_counts = self._domain_type_counts.get(int(node.global_id), Counter())
        return int(sum(int(type_counts.get(su_type, 0)) for su_type in allowed_types))

    def _domain_partners(
        self,
        node: Any,
        options: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> List[Any]:
        options = tuple(options) if options is not None else self._feasible_template_options(node)
        allowed_types = set(self._candidate_types(node, options))
        partners = []
        for candidate_id in self._domains.get(int(node.global_id), set()):
            candidate = self.nodes[int(candidate_id)]
            if int(candidate.remaining_hop1_slots()) <= 0:
                continue
            if int(candidate.su_type) not in allowed_types:
                continue
            partners.append(candidate)
        return partners

    def _component(self, start_id: int) -> Set[int]:
        visited: Set[int] = set()
        queue = deque([int(start_id)])
        while queue:
            node_id = int(queue.popleft())
            if node_id in visited:
                continue
            visited.add(node_id)
            for neighbor_id in list(getattr(self.nodes[node_id], "hop1_ids", []) or []):
                neighbor_i = int(neighbor_id)
                if neighbor_i not in visited:
                    queue.append(neighbor_i)
        return visited

    def _projected_template_count(self, node: Any, neighbor_su: int) -> int:
        su_type = int(getattr(node, "su_type", -1))
        if su_type not in set(getattr(self.assigner, "_carbon_su_set", set()) or set()):
            return 1
        current = Counter(getattr(node, "hop1_su", {}) or {})
        current[int(neighbor_su)] += 1
        return sum(
            1
            for template in self._structural_template_options(node)
            if self._is_subcounter(current, Counter(template))
        )

    def _projected_motif_usage(self, node: Any, neighbor_su: int) -> Tuple[int, int]:
        current = list(Counter(getattr(node, "hop1_su", {}) or {}).elements())
        projected = tuple(sorted(int(x) for x in current + [int(neighbor_su)]))
        prefix_usage = int(self._motif_usage.get((int(node.su_type), projected), 0))
        if int(node.su_type) not in set(getattr(self.assigner, "_carbon_su_set", set()) or set()):
            return prefix_usage, 0

        current_counter = Counter(projected)
        projected_options = [
            tuple(int(x) for x in template)
            for template in self._structural_template_options(node)
            if self._is_subcounter(current_counter, Counter(template))
        ]
        if not projected_options:
            return prefix_usage, 10**9
        complete_usage = min(
            int(self._complete_motif_usage.get((int(node.su_type), template), 0))
            for template in projected_options
        )
        return prefix_usage, complete_usage

    def _partner_rank(self, center: Any, partner: Any, component: Set[int]) -> Tuple[Any, ...]:
        center_options = self._feasible_template_options(center)
        partner_options = self._feasible_template_options(partner)
        projected_loss = (
            len(center_options) - self._projected_template_count(center, int(partner.su_type))
            + len(partner_options) - self._projected_template_count(partner, int(center.su_type))
        )
        current_neighbors = [
            int(self.nodes[int(node_id)].su_type)
            for node_id in list(getattr(center, "hop1_ids", []) or [])
        ]
        chemistry_bias = float(
            self.assigner._bridge_candidate_bias(
                int(center.su_type),
                current_neighbors,
                int(partner.su_type),
            )
        )
        center_prefix_usage, center_complete_usage = self._projected_motif_usage(
            center,
            int(partner.su_type),
        )
        partner_prefix_usage, partner_complete_usage = self._projected_motif_usage(
            partner,
            int(center.su_type),
        )
        variant_key = (
            int(partner.global_id) - int(self._build_variant)
        ) % max(1, int(len(self.nodes)))
        return (
            -int(partner.remaining_hop1_slots()),
            int(projected_loss),
            int(int(partner.global_id) in component),
            -chemistry_bias,
            int(partner.su_type),
            int(center_complete_usage + partner_complete_usage),
            int(center_prefix_usage + partner_prefix_usage),
            int(variant_key),
            int(partner.global_id),
        )

    def _minimum_type_resource_snapshot(self) -> Dict[str, Any]:
        minimum_demand: Counter = Counter()
        available_slots: Counter = Counter()
        infeasible_node_ids: List[int] = []
        for node in self.nodes:
            gap = int(node.remaining_hop1_slots())
            if gap > 0:
                available_slots[int(node.su_type)] += gap
        carbon_sus = set(getattr(self.assigner, "_carbon_su_set", set()) or set())
        for state_key, node_ids in self._state_groups.items():
            active_ids = [
                int(node_id) for node_id in node_ids
                if int(self.nodes[int(node_id)].remaining_hop1_slots()) > 0
            ]
            if not active_ids:
                continue
            node = self.nodes[active_ids[0]]
            if int(node.su_type) not in carbon_sus:
                continue
            options = self._feasible_template_options(node)
            if not options:
                infeasible_node_ids.extend(active_ids)
                continue
            current = Counter(getattr(node, "hop1_su", {}) or {})
            missing_options = [Counter(template) - current for template in options]
            all_types = set().union(*(set(missing.keys()) for missing in missing_options))
            for su_type in all_types:
                per_node = min(
                    int(missing.get(int(su_type), 0)) for missing in missing_options
                )
                minimum_demand[int(su_type)] += int(per_node) * int(len(active_ids))
        shortages = Counter({
            int(su_type): int(demand) - int(available_slots.get(int(su_type), 0))
            for su_type, demand in minimum_demand.items()
            if int(demand) > int(available_slots.get(int(su_type), 0))
        })
        return {
            "minimum_demand": minimum_demand,
            "available_slots": available_slots,
            "shortages": shortages,
            "infeasible_node_ids": sorted(set(infeasible_node_ids)),
        }

    def _minimum_type_demand_ok(self) -> bool:
        snapshot = self._minimum_type_resource_snapshot()
        return not bool(snapshot["shortages"]) and not bool(snapshot["infeasible_node_ids"])

    def _special_capacity_snapshot(self) -> Dict[str, int]:
        special_nodes = [
            node for node in self.nodes
            if int(getattr(node, "su_type", -1)) in {19, 20, 21}
            and int(node.remaining_hop1_slots()) > 0
        ]
        candidate_ids: Set[int] = set()
        total_gap = 0
        min_partner_margin = 0
        for node in special_nodes:
            options = self._feasible_template_options(node)
            partners = self._domain_partners(node, options)
            gap = int(node.remaining_hop1_slots())
            total_gap += gap
            min_partner_margin = min(min_partner_margin, int(len(partners)) - gap)
            candidate_ids.update(int(partner.global_id) for partner in partners)
        capacity = sum(int(self.nodes[node_id].remaining_hop1_slots()) for node_id in candidate_ids)
        return {
            "node_count": int(len(special_nodes)),
            "total_degree_gap": int(total_gap),
            "unique_candidate_count": int(len(candidate_ids)),
            "unique_candidate_capacity": int(capacity),
            "minimum_partner_margin": int(min_partner_margin),
        }

    def _special_capacity_ok(self) -> bool:
        if not any(
            int(getattr(node, "su_type", -1)) in {19, 20, 21}
            and int(node.remaining_hop1_slots()) > 0
            for node in self.nodes
        ):
            return True
        snapshot = self._special_capacity_snapshot()
        return bool(
            int(snapshot["minimum_partner_margin"]) >= 0
            and int(snapshot["unique_candidate_capacity"]) >= int(snapshot["total_degree_gap"])
        )

    def _select_center(self) -> Tuple[Optional[Any], List[Any], Optional[str]]:
        ranked: List[Tuple[Tuple[Any, ...], Any, List[Any]]] = []
        for _state_key, group_ids in self._state_groups.items():
            active_ids = [
                int(node_id) for node_id in group_ids
                if int(self.nodes[int(node_id)].remaining_hop1_slots()) > 0
            ]
            if not active_ids:
                continue
            representative = self.nodes[active_ids[0]]
            options = self._feasible_template_options(representative)
            if not options:
                return representative, [], "no_resource_feasible_template"
            allowed_types = self._candidate_types(representative, options)
            scarcity = self._resource_scarcity(representative, options)
            gap = int(representative.remaining_hop1_slots())
            for node_id in active_ids:
                type_counts = self._domain_type_counts.get(node_id, Counter())
                partner_count = int(sum(int(type_counts.get(su_type, 0)) for su_type in allowed_types))
                node = self.nodes[node_id]
                if partner_count < gap:
                    return node, [], "insufficient_distinct_partners"
                rank = (
                    int(partner_count),
                    len(options),
                    gap,
                    -float(scarcity),
                    int(node_id),
                )
                ranked.append((rank, node, []))
        if not ranked:
            return None, [], None
        ranked.sort(key=lambda item: item[0])
        _, center, _ = ranked[0]
        partners = self._domain_partners(center, self._feasible_template_options(center))
        component = self._component(int(center.global_id))
        partners.sort(key=lambda partner: self._partner_rank(center, partner, component))
        return center, partners, None

    def _forward_check_after_add(self, left: Any, right: Any) -> Tuple[bool, str]:
        remaining_total = sum(int(node.remaining_hop1_slots()) for node in self.nodes)
        if remaining_total % 2 != 0:
            return False, "odd_remaining_degree_sum"
        for node in (left, right):
            if not self._feasible_template_options(node):
                return False, "endpoint_template_or_resource_exhausted"
        for _state_key, group_ids in self._state_groups.items():
            active_ids = [
                int(node_id) for node_id in group_ids
                if int(self.nodes[int(node_id)].remaining_hop1_slots()) > 0
            ]
            if not active_ids:
                continue
            representative = self.nodes[active_ids[0]]
            options = self._feasible_template_options(representative)
            if not options:
                return False, "node_template_or_resource_exhausted"
            allowed_types = self._candidate_types(representative, options)
            gap = int(representative.remaining_hop1_slots())
            for node_id in active_ids:
                type_counts = self._domain_type_counts.get(node_id, Counter())
                partner_count = int(sum(int(type_counts.get(su_type, 0)) for su_type in allowed_types))
                if partner_count < gap:
                    return False, "insufficient_distinct_partners"
        if not self._minimum_type_demand_ok():
            return False, "minimum_type_demand_exceeds_capacity"
        if not self._special_capacity_ok():
            return False, "special_capacity_exhausted"
        return True, "ok"

    def _search(self, depth: int = 0) -> bool:
        self.max_depth = max(self.max_depth, int(depth))
        if self._deadline_reached():
            self.forward_rejections["deadline"] += 1
            return False
        if self._backtrack_limit_reached():
            self.forward_rejections["backtrack_budget"] += 1
            return False

        step_t0 = time.perf_counter()
        center, partners, failure = self._select_center()
        self.timings_s['select_center'] += time.perf_counter() - step_t0
        if center is None:
            return True
        if failure is not None:
            self.forward_rejections[str(failure)] += 1
            return False

        ranked_partners = partners if self.branch_limit is None else partners[: int(self.branch_limit)]
        for partner in ranked_partners:
            if self._deadline_reached() or self._backtrack_limit_reached():
                return False
            left_id = int(center.global_id)
            right_id = int(partner.global_id)
            endpoint_ids = {left_id, right_id}
            self._remove_motif_usage(endpoint_ids)
            if not self.assigner._add_bidirectional_hop1(self.nodes, left_id, right_id):
                self._add_motif_usage(endpoint_ids)
                self.forward_rejections["edge_commit_rejected"] += 1
                continue
            self._add_motif_usage(endpoint_ids)
            step_t0 = time.perf_counter()
            self._refresh_domains_for_endpoints({left_id, right_id})
            self.timings_s['refresh_domains'] += time.perf_counter() - step_t0
            step_t0 = time.perf_counter()
            ok, reason = self._forward_check_after_add(center, partner)
            self.timings_s['forward_check'] += time.perf_counter() - step_t0
            if ok and self._search(depth + 1):
                self.edges_added += 1
                return True
            if not ok:
                self.forward_rejections[str(reason)] += 1
            self._remove_motif_usage(endpoint_ids)
            self.assigner._remove_bidirectional_hop1_with_force(
                self.nodes,
                left_id,
                right_id,
                force=True,
            )
            self._add_motif_usage(endpoint_ids)
            step_t0 = time.perf_counter()
            self._refresh_domains_for_endpoints({left_id, right_id})
            self.timings_s['refresh_domains'] += time.perf_counter() - step_t0
            self.backtracks += 1
        return False

    def _remaining_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for node in self.nodes:
            gap = int(node.remaining_hop1_slots())
            if gap <= 0:
                continue
            options = self._feasible_template_options(node)
            rows.append({
                "node_id": int(node.global_id),
                "su_type": int(node.su_type),
                "degree": int(node.get_hop1_degree()),
                "target_degree": int(node.get_max_degree()),
                "degree_gap": gap,
                "template_candidates": int(len(options)),
                "legal_partners": int(self._domain_partner_count(node, options)) if options else 0,
                "hop1": sorted(int(x) for x in Counter(node.hop1_su).elements()),
            })
        return rows

    def _resource_shortage_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        carbon_sus = set(getattr(self.assigner, "_carbon_su_set", set()) or set())
        for node in self.nodes:
            if int(node.remaining_hop1_slots()) <= 0 or int(node.su_type) not in carbon_sus:
                continue
            structural_options = self._structural_template_options(node)
            if not structural_options:
                rows.append({
                    "node_id": int(node.global_id),
                    "su_type": int(node.su_type),
                    "reason": "no_exact_template_for_current_prefix",
                    "hop1": sorted(int(x) for x in Counter(node.hop1_su).elements()),
                })
                continue
            if self._feasible_template_options(node):
                continue
            current = Counter(getattr(node, "hop1_su", {}) or {})
            available = self._available_type_counts(node)
            ranked = []
            for template in structural_options:
                missing = Counter(template) - current
                deficits = {
                    int(su_type): int(count) - int(available.get(int(su_type), 0))
                    for su_type, count in missing.items()
                    if int(count) > int(available.get(int(su_type), 0))
                }
                ranked.append((sum(deficits.values()), len(deficits), tuple(template), deficits))
            _, _, template, deficits = min(ranked)
            rows.append({
                "node_id": int(node.global_id),
                "su_type": int(node.su_type),
                "reason": "template_resource_shortage",
                "target_template": list(template),
                "shortage_by_su": dict(sorted(deficits.items())),
            })
        return rows

    def solve(self) -> Dict[str, Any]:
        initial_open = sum(int(node.remaining_hop1_slots()) for node in self.nodes)
        if initial_open % 2 != 0:
            self._restore_state(self._base_state)
            raise RuntimeError(f"layer1_completion_odd_open_ports: {initial_open}")
        success = self._search(0)
        elapsed = time.perf_counter() - self.started_at
        diagnostics = {
            "ok": bool(success),
            "elapsed_s": float(elapsed),
            "initial_open_ports": int(initial_open),
            "remaining_open_ports": int(sum(int(node.remaining_hop1_slots()) for node in self.nodes)),
            "edges_added": int(initial_open // 2) if success else 0,
            "backtracks": int(self.backtracks),
            "max_depth": int(self.max_depth),
            "branch_limit": self.branch_limit,
            "max_backtracks": self.max_backtracks,
            "forward_rejections": dict(self.forward_rejections),
            "timings_s": {key: float(value) for key, value in self.timings_s.items()},
            "stop_reason": "complete" if success else (
                "deadline" if self._deadline_reached() else
                "backtrack_budget" if self._backtrack_limit_reached() else
                "no_feasible_completion"
            ),
        }
        if success:
            return diagnostics

        diagnostics["remaining"] = self._remaining_rows()[:16]
        minimum_type_snapshot = self._minimum_type_resource_snapshot()
        diagnostics["minimum_type_resource_shortages"] = dict(sorted(
            (int(su_type), int(count))
            for su_type, count in minimum_type_snapshot["shortages"].items()
        ))
        diagnostics["resource_shortages"] = self._resource_shortage_rows()[:16]
        diagnostics["special_capacity"] = self._special_capacity_snapshot()
        self._restore_state(self._base_state)
        raise RuntimeError(f"layer1_completion_failed: {diagnostics}")
