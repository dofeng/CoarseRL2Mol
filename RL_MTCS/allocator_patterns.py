from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


AROMATIC_SU = {5, 6, 7, 8, 9, 11}
TERMINAL_BASE_SU = {1, 4, 16, 18, 22, 28, 32}
CHAIN_BASE_SU = {14, 15, 17, 23, 24, 25}
ABSTRACT_CHAIN_PORT_SU = {0, 2, 3, 14, 15, 17, 23, 24, 25}


@dataclass(frozen=True)
class NodeState:
    gid: int
    su: int
    degree: Optional[int] = None
    fixed_anchor_count: int = 0
    anchor_mode: Optional[str] = None
    anchor_partition: Optional[str] = None

    @property
    def label(self) -> str:
        if self.su in {19, 20, 21} and self.degree is not None:
            mode = str(self.anchor_mode or "").strip().lower()
            if self.degree == 2 and self.su in {19, 20}:
                if self.fixed_anchor_count == 1 or mode == "single":
                    return f"single{self.su}d2"
                if self.fixed_anchor_count >= 2 or mode == "double":
                    return f"double{self.su}d2"
            if self.degree == 3 and self.su in {19, 20}:
                if self.fixed_anchor_count >= 2 or mode == "double":
                    return f"double{self.su}d3"
                return f"single{self.su}d3"
            if self.degree == 3 and self.su == 21:
                return "21d3"
            return f"{self.su}d{self.degree}"
        return str(self.su)


@dataclass
class TopologyFragment:
    kind: str
    su_path: List[int]
    node_ids: List[int]
    labels: List[str]
    left_open: str = "closed"
    right_open: str = "closed"
    closed: bool = False
    origin_rule: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reversed(self) -> "TopologyFragment":
        return TopologyFragment(
            kind=self.kind,
            su_path=list(reversed(self.su_path)),
            node_ids=list(reversed(self.node_ids)),
            labels=list(reversed(self.labels)),
            left_open=self.right_open,
            right_open=self.left_open,
            closed=self.closed,
            origin_rule=self.origin_rule,
            metadata=dict(self.metadata),
        )


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def node_id(node: Any) -> int:
    return safe_int(getattr(node, "global_id", -1), -1)


def node_su(node: Any) -> int:
    return safe_int(getattr(node, "su_type", -1), -1)


def node_degree(node: Any) -> Optional[int]:
    for attr in ("target_degree", "target_hop1_degree", "actual_hop1_degree"):
        raw = getattr(node, attr, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    hop1_ids = list(getattr(node, "hop1_ids", ()) or ())
    if hop1_ids:
        return len(hop1_ids)
    hop1 = list(getattr(node, "hop1", ()) or ())
    if hop1:
        return len(hop1)
    return None


def node_fixed_anchor_count(node: Any) -> int:
    raw = getattr(node, "target_fixed_anchor_count", None)
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except Exception:
        return 0


def node_anchor_mode(node: Any) -> Optional[str]:
    raw = getattr(node, "anchor_mode", None)
    if raw is None:
        raw = getattr(node, "special_anchor_mode", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def node_anchor_partition(node: Any) -> Optional[str]:
    raw = getattr(node, "anchor_partition", None)
    if raw is None:
        raw = getattr(node, "special_anchor_partition", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def state_for_node(node: Any) -> NodeState:
    return NodeState(
        gid=node_id(node),
        su=node_su(node),
        degree=node_degree(node),
        fixed_anchor_count=node_fixed_anchor_count(node),
        anchor_mode=node_anchor_mode(node),
        anchor_partition=node_anchor_partition(node),
    )


def is_terminal_state(state: NodeState) -> bool:
    if int(state.su) in TERMINAL_BASE_SU:
        return True
    return int(state.su) in {19, 20} and int(state.degree or 0) == 1


def is_single_special_d2(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    mode = str(state.anchor_mode or "").strip().lower()
    return int(state.su) in {19, 20} and int(state.degree or 0) == 2 and (
        int(state.fixed_anchor_count) == 1 or mode == "single"
    )


def is_double_special_d2(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    mode = str(state.anchor_mode or "").strip().lower()
    return int(state.su) in {19, 20} and int(state.degree or 0) == 2 and (
        int(state.fixed_anchor_count) >= 2 or mode == "double"
    )


def is_single_special_d3(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    mode = str(state.anchor_mode or "").strip().lower()
    if int(state.su) not in {19, 20} or int(state.degree or 0) != 3:
        return False
    return not (int(state.fixed_anchor_count) >= 2 or mode == "double")


def is_double_special_d3(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    mode = str(state.anchor_mode or "").strip().lower()
    return int(state.su) in {19, 20} and int(state.degree or 0) == 3 and (
        int(state.fixed_anchor_count) >= 2 or mode == "double"
    )


def is_special_d3(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    return int(state.su) in {19, 20, 21} and int(state.degree or 0) == 3


def fixed_connector_special_pair_ok(center: NodeState, neighbors: Iterable[NodeState]) -> bool:
    center_su = int(center.su)
    if center_su not in {27, 29, 31}:
        return True
    special_su = 20 if center_su == 27 else 19
    relevant = [st for st in list(neighbors or []) if int(st.su) == int(special_su)]
    if len(relevant) < 2:
        return True
    double_d3_count = int(sum(1 for st in relevant if is_double_special_d3(st, special_su)))
    return bool(double_d3_count < 2)


def has_forbidden_fixed_connector_special_combo(center: NodeState, neighbors: Iterable[NodeState]) -> bool:
    return not fixed_connector_special_pair_ok(center, neighbors)


def is_special_d2(state: NodeState, su: Optional[int] = None) -> bool:
    if su is not None and int(state.su) != int(su):
        return False
    return int(state.su) in {19, 20, 21} and int(state.degree or 0) == 2


def is_chain_state(state: NodeState) -> bool:
    if int(state.su) in CHAIN_BASE_SU:
        return True
    return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}


def is_abstract_chain_port_state(state: NodeState) -> bool:
    if int(state.su) in ABSTRACT_CHAIN_PORT_SU:
        return True
    return int(state.su) in {19, 20, 21} and int(state.degree or 0) in {2, 3}


def fragment_signature(fragment: TopologyFragment) -> Tuple:
    return (
        str(fragment.kind),
        tuple(int(x) for x in fragment.node_ids),
        tuple(str(x) for x in fragment.labels),
        str(fragment.left_open),
        str(fragment.right_open),
    )
