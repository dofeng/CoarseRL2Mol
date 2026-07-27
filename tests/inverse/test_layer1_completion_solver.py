from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from model.inverse.layer1.layer1_completion_solver import Layer1CompletionSolver
from model.shared.inverse_common import _NodeV3


class _SolverAssigner:
    def __init__(self, nodes: list[_NodeV3], templates=None, carbon_sus=None) -> None:
        self._hop1_templates_by_su = dict(templates or {})
        self._carbon_su_set = set(carbon_sus or set())
        self._available_ids_by_su = defaultdict(set)
        self._init_runtime_node_index(nodes)

    def _init_runtime_node_index(self, nodes: list[_NodeV3]) -> None:
        self._available_ids_by_su = defaultdict(set)
        for node in nodes:
            if node.remaining_hop1_slots() > 0:
                self._available_ids_by_su[int(node.su_type)].add(int(node.global_id))

    @staticmethod
    def _has_pending_required_external(_node, _nodes) -> bool:
        return False

    @staticmethod
    def _required_external_candidates_for_node(_node) -> list[int]:
        return []

    @staticmethod
    def _bridge_candidate_bias(_center_su, _current_neighbors, _candidate_su) -> float:
        return 1.0

    @staticmethod
    def _special_capacity_snapshot(_nodes):
        return {"nodes": [], "unique_candidate_capacity": 0, "total_degree_gap": 0}

    @staticmethod
    def _can_add_hop1_connection(_nodes, left, right) -> bool:
        return bool(
            int(left.global_id) != int(right.global_id)
            and int(right.global_id) not in left.hop1_ids
            and left.remaining_hop1_slots() > 0
            and right.remaining_hop1_slots() > 0
        )

    def _add_bidirectional_hop1(self, nodes, left_id, right_id, lock=False) -> bool:
        left = nodes[int(left_id)]
        right = nodes[int(right_id)]
        if not self._can_add_hop1_connection(nodes, left, right):
            return False
        left.hop1_ids.append(int(right_id))
        right.hop1_ids.append(int(left_id))
        left.hop1_su[int(right.su_type)] += 1
        right.hop1_su[int(left.su_type)] += 1
        if lock:
            left.fixed_hop1_ids.add(int(right_id))
            right.fixed_hop1_ids.add(int(left_id))
        self._init_runtime_node_index(nodes)
        return True

    def _remove_bidirectional_hop1_with_force(self, nodes, left_id, right_id, force=True) -> bool:
        left = nodes[int(left_id)]
        right = nodes[int(right_id)]
        if int(right_id) not in left.hop1_ids or int(left_id) not in right.hop1_ids:
            return False
        left.hop1_ids.remove(int(right_id))
        right.hop1_ids.remove(int(left_id))
        left.hop1_su[int(right.su_type)] -= 1
        right.hop1_su[int(left.su_type)] -= 1
        if left.hop1_su[int(right.su_type)] <= 0:
            del left.hop1_su[int(right.su_type)]
        if right.hop1_su[int(left.su_type)] <= 0:
            del right.hop1_su[int(left.su_type)]
        left.fixed_hop1_ids.discard(int(right_id))
        right.fixed_hop1_ids.discard(int(left_id))
        self._init_runtime_node_index(nodes)
        return True


def _connect(assigner: _SolverAssigner, nodes: list[_NodeV3], left: int, right: int) -> None:
    assert assigner._add_bidirectional_hop1(nodes, left, right)


def test_mrv_min_gap_center_prefers_max_gap_partner() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=22, target_hop1_degree=1),
        _NodeV3(global_id=1, su_type=23, target_hop1_degree=3),
        _NodeV3(global_id=2, su_type=24, target_hop1_degree=1),
        _NodeV3(global_id=3, su_type=25, target_hop1_degree=1),
    ]
    assigner = _SolverAssigner(nodes)
    solver = Layer1CompletionSolver(
        assigner,
        nodes,
        deadline=None,
        max_backtracks=32,
        branch_limit=8,
    )

    center, partners, failure = solver._select_center()

    assert failure is None
    assert center.global_id == 0
    assert partners[0].global_id == 1
    summary = solver.solve()
    assert summary["ok"] is True
    assert all(node.is_hop1_complete() for node in nodes)
    assert nodes[0].hop1_ids == [1]


def test_su25_resource_infeasible_self_template_is_filtered() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=25, target_hop1_degree=4),
        _NodeV3(global_id=1, su_type=1, target_hop1_degree=1),
        _NodeV3(global_id=2, su_type=3, target_hop1_degree=1),
        _NodeV3(global_id=3, su_type=22, target_hop1_degree=1),
        _NodeV3(global_id=4, su_type=23, target_hop1_degree=1),
    ]
    assigner = _SolverAssigner(
        nodes,
        templates={25: ((1, 3, 25, 25), (1, 3, 22, 23))},
        carbon_sus={25},
    )
    _connect(assigner, nodes, 0, 1)
    _connect(assigner, nodes, 0, 2)
    solver = Layer1CompletionSolver(
        assigner,
        nodes,
        deadline=None,
        max_backtracks=32,
        branch_limit=8,
    )

    assert solver._structural_template_options(nodes[0]) == (
        (1, 3, 22, 23),
        (1, 3, 25, 25),
    )
    assert solver._feasible_template_options(nodes[0]) == ((1, 3, 22, 23),)


def test_failed_completion_restores_fixed_phase_snapshot() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=22, target_hop1_degree=1),
        _NodeV3(global_id=1, su_type=23, target_hop1_degree=1),
    ]
    assigner = _SolverAssigner(
        nodes,
        templates={22: ((22,),), 23: ((23,),)},
        carbon_sus={22, 23},
    )
    before = [(Counter(node.hop1_su), list(node.hop1_ids)) for node in nodes]
    solver = Layer1CompletionSolver(
        assigner,
        nodes,
        deadline=None,
        max_backtracks=4,
        branch_limit=2,
    )

    with pytest.raises(RuntimeError, match="layer1_completion_failed"):
        solver.solve()

    assert [(Counter(node.hop1_su), list(node.hop1_ids)) for node in nodes] == before


def test_completion_prefers_unused_projected_hop1_motif() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=30, target_hop1_degree=1),
        _NodeV3(global_id=1, su_type=22, target_hop1_degree=2),
        _NodeV3(global_id=2, su_type=22, target_hop1_degree=2),
        _NodeV3(global_id=3, su_type=24, target_hop1_degree=1),
        _NodeV3(global_id=4, su_type=25, target_hop1_degree=1),
        _NodeV3(global_id=5, su_type=22, target_hop1_degree=2),
        _NodeV3(global_id=6, su_type=30, target_hop1_degree=1),
        _NodeV3(global_id=7, su_type=24, target_hop1_degree=1),
    ]
    assigner = _SolverAssigner(
        nodes,
        templates={30: ((22,),)},
        carbon_sus={30},
    )
    _connect(assigner, nodes, 1, 3)
    _connect(assigner, nodes, 2, 4)
    _connect(assigner, nodes, 5, 6)
    _connect(assigner, nodes, 5, 7)
    solver = Layer1CompletionSolver(
        assigner,
        nodes,
        deadline=None,
        max_backtracks=32,
        branch_limit=8,
    )

    center, partners, failure = solver._select_center()

    assert failure is None
    assert center.global_id == 0
    assert [partner.global_id for partner in partners[:2]] == [2, 1]
