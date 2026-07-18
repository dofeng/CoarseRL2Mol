from __future__ import annotations

from collections import Counter

class DummyNode:
    def __init__(self, global_id: int, su_type: int, **kwargs) -> None:
        self.global_id = int(global_id)
        self.su_type = int(su_type)
        self.hop1_su = Counter()
        self.hop1_ids = []
        self.fixed_hop1_ids = set()
        self.target_hop1_degree = kwargs.get("target_hop1_degree")
        self.special_anchor_mode = kwargs.get("special_anchor_mode")
        self.special_anchor_partition = kwargs.get("special_anchor_partition")
        self.target_fixed_anchor_count = kwargs.get("target_fixed_anchor_count")
        self.special_degree_source = kwargs.get("special_degree_source")


def make_node(node_id: int, su_type: int, **kwargs) -> DummyNode:
    return DummyNode(global_id=int(node_id), su_type=int(su_type), **kwargs)


def connect(left: DummyNode, right: DummyNode, *, fixed: bool = False) -> None:
    left_id = int(left.global_id)
    right_id = int(right.global_id)
    if right_id not in list(getattr(left, "hop1_ids", []) or []):
        left.hop1_ids.append(int(right_id))
        left.hop1_su[int(right.su_type)] += 1
    if left_id not in list(getattr(right, "hop1_ids", []) or []):
        right.hop1_ids.append(int(left_id))
        right.hop1_su[int(left.su_type)] += 1
    if fixed:
        left.fixed_hop1_ids.add(int(right_id))
        right.fixed_hop1_ids.add(int(left_id))
