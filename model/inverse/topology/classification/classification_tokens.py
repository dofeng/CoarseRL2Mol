from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ....shared.inverse_common import _NodeV3
else:
    _NodeV3 = Any


SPECIAL_SUBTYPE_BY_SU_AND_MODE: Dict[Tuple[int, str], str] = {
    (19, "single"): "single19d2",
    (19, "double"): "double19d2",
    (20, "single"): "single20d2",
    (20, "double"): "double20d2",
}


SPECIAL_SUBTYPE_BY_SU_AND_DEGREE: Dict[Tuple[int, int], str] = {
    (19, 1): "19d1",
    (19, 2): "single19d2",
    (19, 3): "single19d3",
    (20, 1): "20d1",
    (20, 2): "single20d2",
    (20, 3): "single20d3",
    (21, 2): "21d2",
    (21, 3): "21d3",
}


PAIR_BRIDGE_SUBTYPES = {
    "single19d3",
    "double19d3",
    "single20d3",
    "double20d3",
    "21d3",
    "24",
    "14",
}


CONNECTOR_SU_TYPES = {0, 2, 27, 28, 29, 31}
AROMATIC_SU_TYPES = {5, 6, 7, 8, 9, 11}
TERMINAL_BASE_SU_TYPES = {1, 4, 16, 18, 22, 28, 32}
TERMINAL_SUBTYPES = {"19d1", "20d1", "21d2"}
OPEN_ALIPHATIC_BASE_SU_TYPES = {14, 15, 17, 23, 24, 25}
VISIBLE_TERMINAL_BASE_SU_TYPES = set(TERMINAL_BASE_SU_TYPES)
VISIBLE_TERMINAL_SUBTYPES = set(TERMINAL_SUBTYPES)
ABSTRACT_OPEN_NONFIXED_BASE_SU_TYPES = {0, 2, 3, 14, 15, 17, 23, 24, 25}
ABSTRACT_OPEN_NONFIXED_SUBTYPES = {
    "14",
    "single19d2",
    "single20d2",
    "double19d2",
    "double20d2",
    "single19d3",
    "single20d3",
    "double19d3",
    "double20d3",
    "21d3",
}
OPEN_ALIPHATIC_SUBTYPES = {
    "single19d2",
    "single20d2",
    "double19d2",
    "double20d2",
    "single19d3",
    "single20d3",
    "double19d3",
    "double20d3",
}

SPECIAL_OPEN_BUCKET_BY_SU = {
    19: "special19_open_like",
    20: "special20_open_like",
    21: "special21_open_like",
}


@dataclass
class NeighborToken:
    node_id: int
    su_type: int
    subtype: str
    is_fixed: bool = False
    target_degree: int = 0
    anchor_mode: str = ""
    anchor_partition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "su_type": int(self.su_type),
            "subtype": str(self.subtype),
            "is_fixed": bool(self.is_fixed),
            "target_degree": int(self.target_degree),
            "anchor_mode": str(self.anchor_mode),
            "anchor_partition": str(self.anchor_partition),
        }


@dataclass
class ClassificationToken:
    node_id: int
    su_type: int
    base_su: int
    subtype: str
    anchor_mode: str = ""
    anchor_partition: str = ""
    target_degree: int = 0
    fixed_anchor_count: int = 0
    special_degree_source: str = ""
    hop1_su_signature: Tuple[int, ...] = field(default_factory=tuple)
    hop1_node_ids: Tuple[int, ...] = field(default_factory=tuple)
    hop1_counter: Dict[int, int] = field(default_factory=dict)
    neighbor_tokens: List[NeighborToken] = field(default_factory=list)
    fixed_neighbor_ids: Tuple[int, ...] = field(default_factory=tuple)
    connector_neighbor_ids: Tuple[int, ...] = field(default_factory=tuple)
    terminal_neighbor_ids: Tuple[int, ...] = field(default_factory=tuple)
    aromatic_neighbor_ids: Tuple[int, ...] = field(default_factory=tuple)
    open_aliphatic_neighbor_ids: Tuple[int, ...] = field(default_factory=tuple)
    flags: Dict[str, bool] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "su_type": int(self.su_type),
            "base_su": int(self.base_su),
            "subtype": str(self.subtype),
            "anchor_mode": str(self.anchor_mode),
            "anchor_partition": str(self.anchor_partition),
            "target_degree": int(self.target_degree),
            "fixed_anchor_count": int(self.fixed_anchor_count),
            "special_degree_source": str(self.special_degree_source),
            "hop1_su_signature": list(self.hop1_su_signature),
            "hop1_node_ids": list(self.hop1_node_ids),
            "hop1_counter": {str(k): int(v) for k, v in self.hop1_counter.items()},
            "neighbor_tokens": [neighbor.to_dict() for neighbor in list(self.neighbor_tokens or [])],
            "fixed_neighbor_ids": list(self.fixed_neighbor_ids),
            "connector_neighbor_ids": list(self.connector_neighbor_ids),
            "terminal_neighbor_ids": list(self.terminal_neighbor_ids),
            "aromatic_neighbor_ids": list(self.aromatic_neighbor_ids),
            "open_aliphatic_neighbor_ids": list(self.open_aliphatic_neighbor_ids),
            "flags": {str(k): bool(v) for k, v in self.flags.items()},
            "meta": dict(self.meta or {}),
        }


def _normalized_anchor_mode(node: _NodeV3) -> str:
    raw = str(getattr(node, "special_anchor_mode", "") or "").strip().lower()
    if raw.startswith("double"):
        return "double"
    if raw.startswith("single"):
        return "single"
    return raw


def _normalized_partition(node: _NodeV3) -> str:
    return str(getattr(node, "special_anchor_partition", "") or "").strip().lower()


def _target_degree(node: _NodeV3) -> int:
    try:
        return int(getattr(node, "target_hop1_degree", 0) or 0)
    except Exception:
        return 0


def _fixed_anchor_count(node: _NodeV3) -> int:
    try:
        return int(getattr(node, "target_fixed_anchor_count", 0) or 0)
    except Exception:
        return 0


def _hop1_counter(node: _NodeV3) -> Counter:
    counter = getattr(node, "hop1_su", None)
    if isinstance(counter, Counter):
        return Counter({int(k): int(v) for k, v in counter.items()})
    try:
        return Counter({int(k): int(v) for k, v in dict(counter or {}).items()})
    except Exception:
        return Counter()


def build_hop1_signature(counter: Counter) -> Tuple[int, ...]:
    out: List[int] = []
    for su_type in sorted(int(k) for k in counter.keys()):
        out.extend([int(su_type)] * int(counter.get(int(su_type), 0)))
    return tuple(out)


def normalize_subtype(node: _NodeV3) -> str:
    su_type = int(getattr(node, "su_type", -1))
    anchor_mode = _normalized_anchor_mode(node)
    target_degree = _target_degree(node)
    fixed_anchor_count = _fixed_anchor_count(node)
    special_degree_source = str(getattr(node, "special_degree_source", "") or "").strip()

    if su_type not in {14, 19, 20, 21, 24, 25}:
        return str(su_type)
    if su_type == 24:
        return "24"
    if su_type == 25:
        return "25"
    if su_type == 14:
        return "14"

    if fixed_anchor_count >= 2 or anchor_mode == "double":
        direct = SPECIAL_SUBTYPE_BY_SU_AND_MODE.get((su_type, "double"))
        if direct:
            return str(direct).replace("d2", "d3") if target_degree >= 3 else str(direct)

    if target_degree > 0:
        subtype = SPECIAL_SUBTYPE_BY_SU_AND_DEGREE.get((su_type, target_degree))
        if subtype:
            if target_degree == 2 and anchor_mode == "double":
                return subtype.replace("single", "double")
            if target_degree == 3 and anchor_mode == "double" and su_type in {19, 20}:
                return subtype.replace("single", "double")
            return subtype

    if anchor_mode:
        subtype = SPECIAL_SUBTYPE_BY_SU_AND_MODE.get((su_type, anchor_mode))
        if subtype:
            return subtype
    if (
        su_type in {19, 20, 21}
        and (
            special_degree_source.endswith("d2d3")
            or special_degree_source in {"19d2d3", "20d2d3", "21d2d3"}
        )
    ):
        if su_type == 21:
            return "21"
        return "single19d2" if su_type == 19 else "single20d2"
    if su_type == 21:
        return "21"
    return str(su_type)


def semantic_bucket_for_node(node: _NodeV3) -> str:
    su_type = int(getattr(node, "su_type", -1))
    special_degree_source = str(getattr(node, "special_degree_source", "") or "").strip()
    if not (
        special_degree_source.endswith("d2d3")
        or special_degree_source in {"19d2d3", "20d2d3", "21d2d3"}
    ):
        return ""
    return str(SPECIAL_OPEN_BUCKET_BY_SU.get(int(su_type), ""))


def _is_terminal_neighbor(neighbor: NeighborToken) -> bool:
    return int(neighbor.su_type) in TERMINAL_BASE_SU_TYPES or str(neighbor.subtype) in TERMINAL_SUBTYPES


def _is_open_aliphatic_neighbor(neighbor: NeighborToken) -> bool:
    return (
        int(neighbor.su_type) in OPEN_ALIPHATIC_BASE_SU_TYPES
        or str(neighbor.subtype) in OPEN_ALIPHATIC_SUBTYPES
    )


def _semantic_fields(value: Any) -> Tuple[int, str, bool]:
    if isinstance(value, dict):
        return (
            int(value.get("su_type", -1)),
            str(value.get("subtype", "")),
            bool(value.get("is_fixed", False)),
        )
    return (
        int(getattr(value, "su_type", -1)),
        str(getattr(value, "subtype", "")),
        bool(getattr(value, "is_fixed", False)),
    )


def is_visible_terminal_semantic(value: Any) -> bool:
    su_type, subtype, _ = _semantic_fields(value)
    return su_type in VISIBLE_TERMINAL_BASE_SU_TYPES or subtype in VISIBLE_TERMINAL_SUBTYPES


def is_visible_fixed_connector_semantic(value: Any) -> bool:
    su_type, _, is_fixed = _semantic_fields(value)
    return bool(is_fixed and su_type in CONNECTOR_SU_TYPES)


def is_abstract_open_nonfixed_semantic(value: Any) -> bool:
    su_type, subtype, is_fixed = _semantic_fields(value)
    return bool(
        not is_fixed and (
            su_type in ABSTRACT_OPEN_NONFIXED_BASE_SU_TYPES
            or subtype in ABSTRACT_OPEN_NONFIXED_SUBTYPES
        )
    )


def is_open_boundary_semantic(value: Any) -> bool:
    """Every non-fixed neighbor is an unresolved topology boundary."""
    _, _, is_fixed = _semantic_fields(value)
    return not bool(is_fixed)


def _build_neighbor_tokens(
    node: _NodeV3,
    nodes_by_id: Dict[int, _NodeV3],
) -> List[NeighborToken]:
    node_id = int(getattr(node, "global_id", -1))
    fixed_ids = set(int(x) for x in list(getattr(node, "fixed_hop1_ids", set()) or set()))
    out: List[NeighborToken] = []
    for neighbor_id in list(getattr(node, "hop1_ids", []) or []):
        try:
            nid = int(neighbor_id)
        except Exception:
            continue
        neighbor = nodes_by_id.get(nid)
        if neighbor is None:
            raise ValueError(
                f"node {int(getattr(node, 'global_id', -1))} references missing hop1 node {nid}"
            )
        neighbor_fixed_ids = {
            int(value)
            for value in list(getattr(neighbor, "fixed_hop1_ids", set()) or set())
        }
        is_fixed = bool(int(nid) in fixed_ids and int(node_id) in neighbor_fixed_ids)
        out.append(NeighborToken(
            node_id=int(nid),
            su_type=int(getattr(neighbor, "su_type", -1)),
            subtype=normalize_subtype(neighbor),
            is_fixed=is_fixed,
            target_degree=_target_degree(neighbor),
            anchor_mode=_normalized_anchor_mode(neighbor),
            anchor_partition=_normalized_partition(neighbor),
        ))
    out.sort(key=lambda item: int(item.node_id))
    return out


def build_node_token(node: _NodeV3, nodes_by_id: Optional[Dict[int, _NodeV3]] = None) -> ClassificationToken:
    counter = _hop1_counter(node)
    subtype = normalize_subtype(node)
    anchor_mode = _normalized_anchor_mode(node)
    partition = _normalized_partition(node)
    target_degree = _target_degree(node)
    fixed_anchor_count = _fixed_anchor_count(node)
    node_id = int(getattr(node, "global_id", -1))
    hop1_ids = tuple(int(x) for x in list(getattr(node, "hop1_ids", []) or []))
    nodes_lookup = dict(nodes_by_id or {})
    neighbor_tokens = _build_neighbor_tokens(node, nodes_lookup)
    fixed_edge_violations: List[Dict[str, Any]] = []
    local_fixed_ids = {
        int(value)
        for value in list(getattr(node, "fixed_hop1_ids", set()) or set())
    }
    for neighbor in list(neighbor_tokens or []):
        neighbor_node = nodes_lookup.get(int(neighbor.node_id))
        reverse_fixed = bool(
            neighbor_node is not None
            and int(node_id) in {
                int(value)
                for value in list(getattr(neighbor_node, "fixed_hop1_ids", set()) or set())
            }
        )
        local_fixed = int(neighbor.node_id) in local_fixed_ids
        if bool(local_fixed) != bool(reverse_fixed):
            fixed_edge_violations.append({
                "edge_key": sorted([int(node_id), int(neighbor.node_id)]),
                "local_fixed": bool(local_fixed),
                "reverse_fixed": bool(reverse_fixed),
                "reason": "asymmetric_fixed_edge",
            })
    fixed_neighbor_ids = tuple(sorted(int(neighbor.node_id) for neighbor in neighbor_tokens if bool(neighbor.is_fixed)))
    connector_neighbor_ids = tuple(sorted(
        int(neighbor.node_id) for neighbor in neighbor_tokens if int(neighbor.su_type) in CONNECTOR_SU_TYPES
    ))
    terminal_neighbor_ids = tuple(sorted(
        int(neighbor.node_id) for neighbor in neighbor_tokens if _is_terminal_neighbor(neighbor)
    ))
    aromatic_neighbor_ids = tuple(sorted(
        int(neighbor.node_id) for neighbor in neighbor_tokens if int(neighbor.su_type) in AROMATIC_SU_TYPES
    ))
    open_aliphatic_neighbor_ids = tuple(sorted(
        int(neighbor.node_id) for neighbor in neighbor_tokens if _is_open_aliphatic_neighbor(neighbor)
    ))
    flags = {
        "is_resource_pool_member": int(getattr(node, "su_type", -1)) in {11, 22, 23, 24},
        "is_like24_candidate": subtype in PAIR_BRIDGE_SUBTYPES,
        "is_double_anchor": bool(anchor_mode == "double" or fixed_anchor_count >= 2),
        "is_single_anchor": bool(anchor_mode == "single" or fixed_anchor_count == 1),
    }
    meta = {
        "fixed_hop1_ids": sorted(int(x) for x in list(getattr(node, "fixed_hop1_ids", set()) or set())),
        "fixed_connector_inferred": bool(
            dict(getattr(node, "score_components", {}) or {}).get("fixture_inferred_fixed_connectors", False)
        ),
        "semantic_bucket": semantic_bucket_for_node(node),
        "fixed_edge_violations": fixed_edge_violations,
    }
    return ClassificationToken(
        node_id=node_id,
        su_type=int(getattr(node, "su_type", -1)),
        base_su=int(getattr(node, "su_type", -1)),
        subtype=subtype,
        anchor_mode=anchor_mode,
        anchor_partition=partition,
        target_degree=target_degree,
        fixed_anchor_count=fixed_anchor_count,
        special_degree_source=str(getattr(node, "special_degree_source", "") or ""),
        hop1_su_signature=build_hop1_signature(counter),
        hop1_node_ids=hop1_ids,
        hop1_counter={int(k): int(v) for k, v in counter.items()},
        neighbor_tokens=neighbor_tokens,
        fixed_neighbor_ids=fixed_neighbor_ids,
        connector_neighbor_ids=connector_neighbor_ids,
        terminal_neighbor_ids=terminal_neighbor_ids,
        aromatic_neighbor_ids=aromatic_neighbor_ids,
        open_aliphatic_neighbor_ids=open_aliphatic_neighbor_ids,
        flags=flags,
        meta=meta,
    )


def build_node_tokens(nodes: Sequence[_NodeV3]) -> List[ClassificationToken]:
    nodes_local = list(nodes or [])
    nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in nodes_local
        if hasattr(node, "global_id")
    }
    return [build_node_token(node, nodes_by_id=nodes_by_id) for node in nodes_local]
