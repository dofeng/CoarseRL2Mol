"""Convergence guards for complete Layer1/2 generation boundaries.

Physical node IDs remain authoritative for allocation.  Only the convergence
comparison ignores a relabelling of otherwise identical chemical graphs.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import networkx as nx
import torch


def _histogram(H: torch.Tensor) -> tuple[int, ...]:
    return tuple(torch.clamp(H.detach().cpu(), min=0).long().tolist())


def _chemical_graph(nodes: list[Any]) -> nx.Graph:
    graph = nx.Graph()
    by_id = {int(node.global_id): node for node in nodes}
    for node_id, node in by_id.items():
        label = (
            int(node.su_type),
            getattr(node, 'target_hop1_degree', None),
            getattr(node, 'target_fixed_anchor_count', None),
            getattr(node, 'special_anchor_partition', None),
            getattr(node, 'special_anchor_mode', None),
            repr(getattr(node, 'template_key', None)),
            round(float(getattr(node, 'mu', 0.0)), 7),
            round(float(getattr(node, 'pi', 1.0)), 7),
        )
        graph.add_node(node_id, chemical=repr(label))
    for node_id, node in by_id.items():
        fixed = set(getattr(node, 'fixed_hop1_ids', ()) or ())
        for neighbor in getattr(node, 'hop1_ids', ()) or ():
            neighbor = int(neighbor)
            if neighbor not in by_id:
                # Invalid/incomplete edges must not disappear from the guard.
                graph.add_node(neighbor, chemical=repr(('missing_node', neighbor)))
            edge_fixed = bool(neighbor in fixed)
            if graph.has_edge(node_id, neighbor):
                edge_fixed |= graph.edges[node_id, neighbor]['fixed'] == '1'
            graph.add_edge(node_id, neighbor, fixed=str(int(edge_fixed)))
    return graph


class Stage3StateMemory:
    """Use a graph hash to shortlist states, then check exact isomorphism.

    A matching histogram or Weisfeiler--Lehman hash alone is insufficient:
    isomers and regular graphs can share both.  Retain only compact chemical
    graphs, not latent tensors, assemblers or transaction snapshots.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, list[tuple[str, nx.Graph]]] = defaultdict(list)
        self._count = 0
        self.rejected_candidates: dict[str, set[str]] = defaultdict(set)

    def identify(self, nodes: list[Any], H: torch.Tensor, *, remember: bool = False) -> tuple[str, bool]:
        graph = _chemical_graph(nodes)
        graph_hash = nx.weisfeiler_lehman_graph_hash(
            graph, node_attr='chemical', edge_attr='fixed', iterations=3,
        )
        bucket = hashlib.sha256(repr((_histogram(H), graph_hash)).encode()).hexdigest()
        for key, previous in self._buckets.get(bucket, ()):
            if nx.is_isomorphic(
                graph, previous,
                node_match=nx.algorithms.isomorphism.categorical_node_match('chemical', None),
                edge_match=nx.algorithms.isomorphism.categorical_edge_match('fixed', None),
            ):
                return key, True
        key = f'{bucket}:{len(self._buckets.get(bucket, ()))}'
        if remember:
            self._buckets[bucket].append((key, graph))
            self._count += 1
        return key, False

    def __len__(self) -> int:
        return self._count


class Stage3BudgetExceeded(RuntimeError):
    """A resource search stopped with its last valid state retained."""


class Stage3SearchBudget:
    def __init__(self, *, max_rebuilds: int, max_planner_attempts: int,
                 max_attempts_per_generation: int,
                 max_no_progress_rebuilds: int) -> None:
        self.limits = {
            'max_rebuilds': max_rebuilds,
            'max_planner_attempts': max_planner_attempts,
            'max_attempts_per_generation': max_attempts_per_generation,
            'max_no_progress_rebuilds': max_no_progress_rebuilds,
        }
        for name, value in self.limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'block_c_resource.{name} must be a positive integer')
        self.rebuilds = 0
        self.planner_attempts = 0
        self.no_progress_rebuilds = 0
        self.reason = ''

    def plan(self, generation_attempt: int) -> bool:
        if self.planner_attempts >= self.limits['max_planner_attempts']:
            self.reason = 'stage3_planner_attempt_limit'
        elif generation_attempt > self.limits['max_attempts_per_generation']:
            self.reason = 'stage3_generation_attempt_limit'
        elif self.no_progress_rebuilds >= self.limits['max_no_progress_rebuilds']:
            self.reason = 'stage3_no_progress_rebuild_limit'
        if self.reason:
            return False
        self.planner_attempts += 1
        return True

    def rebuild(self) -> None:
        if self.rebuilds >= self.limits['max_rebuilds']:
            self.reason = 'stage3_rebuild_limit'
            raise Stage3BudgetExceeded(self.reason)
        if self.no_progress_rebuilds >= self.limits['max_no_progress_rebuilds']:
            self.reason = 'stage3_no_progress_rebuild_limit'
            raise Stage3BudgetExceeded(self.reason)
        self.rebuilds += 1
        self.no_progress_rebuilds += 1

    def accepted(self) -> None:
        self.no_progress_rebuilds = 0

    def to_dict(self) -> dict[str, Any]:
        return {**self.limits, 'rebuild_count': self.rebuilds,
                'planner_attempt_count': self.planner_attempts,
                'no_progress_rebuild_count': self.no_progress_rebuilds,
                'stop_reason': self.reason}
