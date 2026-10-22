#!/usr/bin/env python3
"""
cg_to_allatom.py - 将简化的粗粒度分子转化为全原子分子

输入：简化后的粗粒度JSON（nodes + bonds）
输出：全原子JSON（atoms + bonds）

每个SU的全原子形式：
- 0 : C(=O)-NH  酰胺，C连接芳香侧，N连接脂肪侧(6,20)
- 1 : COOH      羧酸
- 2 : C(=O)-O   酯基，C连接芳香侧，O连接脂肪侧(5,19)
- 3 : C(=O)     羰基（醛酮）
- 4 : C≡N       氰基
- 5-12 : C      单个碳（芳香碳）
- 13 : CH       芳香CH
- 14 : >C=      双键碳（四取代）
- 15 : -HC=     双键碳（三取代）
- 16 : =CH2     双键末端
- 17 : -C≡      三键碳
- 18 : ≡CH      三键末端
- 19-21 : 按实际连接度确定氢数的杂原子邻接饱和碳
- 23 : CH2 亚甲基
- 22 : CH3      甲基
- 24 : CH       次甲基
- 25 : C        季碳
- 26 : 芳香N，五元环吡咯型N与六元环吡啶型N分别处理
- 27 : NH       仲胺
- 28 : OH       羟基
- 29 : O        醚氧
- 30 : S        芳杂环硫
- 31 : S        硫醚
- 32 : X        卤素(默认Cl)
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class AtomExpansion:
    """SU展开为全原子的定义"""
    atoms: List[Tuple[str, str]]  # [(元素符号, 原子类型描述), ...]
    internal_bonds: List[Tuple[int, int, str]]  # [(原子i, 原子j, 键类型), ...]
    # 连接点映射：邻居SU类型集合 -> 本SU中用于连接的原子索引
    # 默认连接点是索引0
    # 对于有方向性的SU（如酰胺），需要根据邻居类型选择连接点
    connect_point: int = 0  # 默认连接点
    # 特殊连接规则：{邻居SU类型: 本SU连接原子索引}
    directional_connects: Dict[int, int] = field(default_factory=dict)
    # 坐标偏移（相对于中心原子）
    coord_offsets: List[Tuple[float, float]] = field(default_factory=list)


# 键类型常量
SINGLE = "SINGLE"
DOUBLE = "DOUBLE"
TRIPLE = "TRIPLE"
AROMATIC = "AROMATIC"

# ============================================================
# SU全原子展开定义
# ============================================================

SU_EXPANSIONS: Dict[int, AtomExpansion] = {}

# SU-0: C(=O)-NH 酰胺
# 结构: C(=O)-N(-H)
# 原子: 0=C, 1=O(=O), 2=N, 3=H
# C连接芳香侧(5-13)，N连接脂肪侧(6,19,20,21,23等)
SU_EXPANSIONS[0] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("N", "N-H"), ("H", "H-N")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE), (2, 3, SINGLE)],
    connect_point=0,  # 默认C连接
    directional_connects={
        # N连接的邻居类型
        6: 2, 20: 2,
    },
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.5, 0), (-0.8, 0.3)]
)

# SU-1: COOH 羧酸
# 结构: C(=O)-O-H
# 原子: 0=C, 1=O(=O), 2=O(-H), 3=H
SU_EXPANSIONS[1] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("O", "O-H"), ("H", "H-O")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE), (2, 3, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.3, -0.5), (-0.6, -0.8)]
)

# SU-2: C(=O)-O 酯基
# 结构: C(=O)-O-
# 原子: 0=C, 1=O(=O), 2=O
# C连接芳香侧，O连接脂肪侧(5,19等)
SU_EXPANSIONS[2] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("O", "O-")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE)],
    connect_point=0,  # 默认C连接
    directional_connects={
        5: 2, 19: 2,
    },
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.5, 0)]
)

# SU-3: C(=O) 羰基
# 原子: 0=C, 1=O
SU_EXPANSIONS[3] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O")],
    internal_bonds=[(0, 1, DOUBLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0, 0.5)]
)

# SU-4: C≡N 氰基
# 原子: 0=C, 1=N
SU_EXPANSIONS[4] = AtomExpansion(
    atoms=[("C", "C≡N"), ("N", "≡N")],
    internal_bonds=[(0, 1, TRIPLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.5, 0)]
)

# SU-5 到 SU-12: 单个芳香碳 C
for su_id in range(5, 13):
    SU_EXPANSIONS[su_id] = AtomExpansion(
        atoms=[("C", "Ar-C")],
        internal_bonds=[],
        connect_point=0,
        coord_offsets=[(0, 0)]
    )

# SU-13: 芳香CH
# 原子: 0=C, 1=H
SU_EXPANSIONS[13] = AtomExpansion(
    atoms=[("C", "Ar-CH"), ("H", "H-Ar")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-14: >C= 双键碳（四取代）
# 原子: 0=C
SU_EXPANSIONS[14] = AtomExpansion(
    atoms=[("C", ">C=")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-15: -HC= 双键碳（三取代）
# 原子: 0=C, 1=H
SU_EXPANSIONS[15] = AtomExpansion(
    atoms=[("C", "-HC="), ("H", "H-C=")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-16: =CH2 双键末端
# 原子: 0=C, 1=H, 2=H
SU_EXPANSIONS[16] = AtomExpansion(
    atoms=[("C", "=CH2"), ("H", "H-C"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0.3), (0.3, -0.3)]
)

# SU-17: -C≡ 三键碳
# 原子: 0=C
SU_EXPANSIONS[17] = AtomExpansion(
    atoms=[("C", "-C≡")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-18: ≡CH 三键末端
# 原子: 0=C, 1=H
SU_EXPANSIONS[18] = AtomExpansion(
    atoms=[("C", "≡CH"), ("H", "H-C≡")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-19, 20, 21, 23: CH2 亚甲基
for su_id in [19, 20, 21, 23]:
    SU_EXPANSIONS[su_id] = AtomExpansion(
        atoms=[("C", "CH2"), ("H", "H-C"), ("H", "H-C")],
        internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE)],
        connect_point=0,
        coord_offsets=[(0, 0), (0.2, 0.35), (0.2, -0.35)]
    )

# SU-22: CH3 甲基
SU_EXPANSIONS[22] = AtomExpansion(
    atoms=[("C", "CH3"), ("H", "H-C"), ("H", "H-C"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE), (0, 3, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0), (0.15, 0.26), (0.15, -0.26)]
)

# SU-24: CH 次甲基
SU_EXPANSIONS[24] = AtomExpansion(
    atoms=[("C", "CH"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-25: C 季碳
SU_EXPANSIONS[25] = AtomExpansion(
    atoms=[("C", "C")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-26: N 氮
SU_EXPANSIONS[26] = AtomExpansion(
    atoms=[("N", "N")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-27: NH 仲胺
SU_EXPANSIONS[27] = AtomExpansion(
    atoms=[("N", "NH"), ("H", "H-N")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-28: OH 羟基
SU_EXPANSIONS[28] = AtomExpansion(
    atoms=[("O", "OH"), ("H", "H-O")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.35, 0)]
)

# SU-29: O 醚氧
SU_EXPANSIONS[29] = AtomExpansion(
    atoms=[("O", "O")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-30: S 硫醚
SU_EXPANSIONS[30] = AtomExpansion(
    atoms=[("S", "S")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-31: S 硫醚
SU_EXPANSIONS[31] = AtomExpansion(
    atoms=[("S", "S")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-32: X 卤素 (默认Cl)
SU_EXPANSIONS[32] = AtomExpansion(
    atoms=[("Cl", "X")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)


# ============================================================
# 芳香SU和双键/三键SU集合
# ============================================================
AROMATIC_SU = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
DOUBLE_BOND_SU = {14, 15, 16}
TRIPLE_BOND_SU = {17, 18}


class CG2MolConversionError(ValueError):
    """Raised when the coarse-grained graph cannot represent a valid molecule."""


def _bond_records(simplified_cg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize legacy bonds and RL_MTCS bond details to one representation."""
    details = simplified_cg.get("bond_details")
    records: List[Dict[str, Any]] = []
    if isinstance(details, list) and details:
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                raise CG2MolConversionError(f"bond_details[{index}] is not an object")
            try:
                source_value = detail.get("source_seq")
                target_value = detail.get("target_seq")
                if source_value is None:
                    source_value = detail["source"]
                if target_value is None:
                    target_value = detail["target"]
                source = int(source_value)
                target = int(target_value)
            except (KeyError, TypeError, ValueError) as exc:
                raise CG2MolConversionError(
                    f"bond_details[{index}] has invalid sequence ids"
                ) from exc
            records.append({
                "source": source,
                "target": target,
                "kind": str(detail.get("kind", "")),
                "source_node_id": detail.get("source_node_id"),
                "target_node_id": detail.get("target_node_id"),
            })
        raw_bonds = simplified_cg.get("bonds")
        if raw_bonds is not None:
            raw_pairs = sorted(tuple(sorted((int(bond[0]), int(bond[1])))) for bond in raw_bonds)
            detail_pairs = sorted(tuple(sorted((record["source"], record["target"]))) for record in records)
            if raw_pairs != detail_pairs:
                raise CG2MolConversionError("bonds and bond_details describe different connectivity")
        return records

    for index, bond in enumerate(simplified_cg.get("bonds", []) or []):
        if not isinstance(bond, (list, tuple)) or len(bond) < 2:
            raise CG2MolConversionError(f"bonds[{index}] is not a pair")
        try:
            source, target = int(bond[0]), int(bond[1])
        except (TypeError, ValueError) as exc:
            raise CG2MolConversionError(f"bonds[{index}] has invalid sequence ids") from exc
        kind = str(bond[2]) if len(bond) > 2 and isinstance(bond[2], str) else ""
        records.append({"source": source, "target": target, "kind": kind})
    return records


def _node_expansion(su_type: int, degree: int, ring_size: Optional[int] = None) -> AtomExpansion:
    """Expand explicit hydrogens without changing the assembled SU label."""
    if su_type not in SU_EXPANSIONS:
        raise CG2MolConversionError(f"unsupported SU type: {su_type}")
    expected_degrees = (
        2, 1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 1,
        2, 1, 3, 3, 3, 1, 2, 3, 4, 2, 2, 1, 2, 2, 2, 1,
    )
    allowed_degrees = {expected_degrees[su_type]}
    if su_type in {19, 20}:
        allowed_degrees = {1, 2, 3}
    elif su_type == 21:
        allowed_degrees = {2, 3}
    elif su_type == 3:
        allowed_degrees = {1, 2}
    if degree not in allowed_degrees:
        raise CG2MolConversionError(
            f"SU {su_type} degree={degree}, expected one of {sorted(allowed_degrees)}"
        )
    base = SU_EXPANSIONS[su_type]
    if su_type in {19, 20, 21}:
        hydrogen_count = 4 - degree
    elif su_type == 26 and ring_size == 5:
        hydrogen_count = 1
    elif su_type == 3 and degree == 1:
        return AtomExpansion(
            atoms=[*base.atoms, ("H", "H-carbonyl")],
            internal_bonds=[*base.internal_bonds, (0, 2, SINGLE)],
            coord_offsets=[*base.coord_offsets, (0.35, -0.35)],
        )
    else:
        return base

    core_atoms = [base.atoms[0]]
    atoms = list(core_atoms)
    internal_bonds: List[Tuple[int, int, str]] = []
    coord_offsets = [base.coord_offsets[0] if base.coord_offsets else (0.0, 0.0)]
    for hydrogen_index in range(hydrogen_count):
        atoms.append(("H", f"H-{base.atoms[0][0]}"))
        internal_bonds.append((0, hydrogen_index + 1, SINGLE))
        if hydrogen_count == 1:
            offset = (0.4, 0.0)
        else:
            angle = 2.0 * math.pi * hydrogen_index / hydrogen_count
            offset = (0.35 * math.cos(angle), 0.35 * math.sin(angle))
        coord_offsets.append(offset)
    return AtomExpansion(
        atoms=atoms,
        internal_bonds=internal_bonds,
        connect_point=base.connect_point,
        directional_connects=dict(base.directional_connects),
        coord_offsets=coord_offsets,
    )


def _matching_edges(
    node_ids: Iterable[int],
    candidate_edges: Iterable[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Find a deterministic perfect matching for a multiple-bond subgraph."""
    nodes = set(int(node_id) for node_id in node_ids)
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(sorted({
        tuple(sorted((int(left), int(right))))
        for left, right in candidate_edges
        if left != right and left in nodes and right in nodes
    }))
    solution = nx.max_weight_matching(graph, maxcardinality=True)
    matched = {node_id for pair in solution for node_id in pair}
    if matched != nodes:
        raise CG2MolConversionError(
            "multiple-bond SU graph has no valid perfect matching; "
            f"unpaired_nodes={sorted(nodes - matched)}"
        )
    return {tuple(sorted(pair)) for pair in solution}


def _select_multiple_bonds(
    nodes: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    su_types: Set[int],
) -> Set[Tuple[int, int]]:
    """Select one double/triple edge per unsaturated SU node."""
    seq_to_su = {int(node["seq"]): int(node["su_type"]) for node in nodes}
    unsaturated = {
        sequence for sequence, su_type in seq_to_su.items() if su_type in su_types
    }
    candidate_edges = []
    for record in records:
        left, right = int(record["source"]), int(record["target"])
        kind = str(record.get("kind", "")).lower()
        if left in unsaturated and right in unsaturated and kind not in {"aromatic", "aromatic_ring"}:
            candidate_edges.append(tuple(sorted((left, right))))
    if not unsaturated:
        return set()
    return _matching_edges(unsaturated, candidate_edges)


def _aromatic_nitrogen_ring_sizes(
    nodes: List[Dict[str, Any]], records: List[Dict[str, Any]],
) -> Dict[int, int]:
    """Use final substitution provenance or ring connectivity for SU26."""
    graph = nx.Graph()
    for record in records:
        if str(record.get("kind", "")).lower().startswith("aromatic"):
            graph.add_edge(int(record["source"]), int(record["target"]))
    sizes: Dict[int, int] = {}
    for node in nodes:
        if int(node["su_type"]) != 26:
            continue
        sequence = int(node["seq"])
        declared_size = node.get("aromatic_ring_size")
        if declared_size is not None:
            sizes[sequence] = int(declared_size)
            continue
        if sequence not in graph or graph.degree(sequence) != 2:
            continue
        neighbors = sorted(graph.neighbors(sequence))
        remaining = graph.subgraph(set(graph) - {sequence})
        try:
            sizes[sequence] = nx.shortest_path_length(remaining, neighbors[0], neighbors[1]) + 2
        except nx.NetworkXNoPath:
            continue
    return sizes


def _connection_points(
    su_type: int, neighbors: List[int], seq_to_su: Dict[int, int],
) -> Dict[int, int]:
    """Assign the carbonyl-body and hetero-side ports jointly."""
    if su_type not in {0, 2}:
        return {neighbor: 0 for neighbor in neighbors}
    hetero_side = set(SU_EXPANSIONS[su_type].directional_connects)
    body_side = {0, 1, 2, 3, 9, 11, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25}
    for hetero_neighbor in neighbors:
        body_neighbors = [neighbor for neighbor in neighbors if neighbor != hetero_neighbor]
        if (
            len(body_neighbors) == 1
            and seq_to_su[hetero_neighbor] in hetero_side
            and seq_to_su[body_neighbors[0]] in body_side
        ):
            return {hetero_neighbor: 2, body_neighbors[0]: 0}
    raise CG2MolConversionError(
        f"SU {su_type} cannot assign carbonyl/hetero ports to neighbors "
        f"{[seq_to_su[neighbor] for neighbor in neighbors]}"
    )


def hex_to_cartesian(q: int, r: int, scale: float = 1.0) -> Tuple[float, float]:
    """六角坐标转笛卡尔坐标"""
    x = scale * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = scale * (3 / 2 * r)
    return (x, y)


def _bond_type_for_edge(
    su_a: int,
    su_b: int,
    edge_kind: str,
    pair: Tuple[int, int],
    double_edges: Set[Tuple[int, int]],
    triple_edges: Set[Tuple[int, int]],
) -> str:
    """Resolve a CG edge without confusing inter-unit and aromatic bonds."""
    kind = str(edge_kind or "").strip().lower()
    if kind.startswith("aromatic"):
        if su_a not in AROMATIC_SU or su_b not in AROMATIC_SU:
            raise CG2MolConversionError(
                f"aromatic edge connects non-aromatic SU types: {su_a}-{su_b}"
            )
        return AROMATIC
    if pair in double_edges:
        return DOUBLE
    if pair in triple_edges:
        return TRIPLE
    if not kind:
        if su_a in AROMATIC_SU and su_b in AROMATIC_SU:
            return AROMATIC
    return SINGLE


def convert_cg_to_allatom(
    simplified_cg: Dict,
    *,
    allow_partial: bool = False,
) -> Dict:
    """Convert a simplified or RL_MTCS coarse-grained graph to all atoms.

    ``simplified_cg`` may be the legacy ``nodes``/``bonds`` representation or
    the adapter output containing ``bond_details``.  RL_MTCS edge kinds are
    authoritative: only ``aromatic`` edges become aromatic bonds; ``rigid``
    inter-unit connections remain single bonds.
    """
    if "molecule" in simplified_cg or "edges" in simplified_cg:
        from .cg_simplify import simplify_cg_molecule

        simplified_cg = simplify_cg_molecule(simplified_cg)
    source_metadata = dict(simplified_cg.get("metadata", {}) or {})
    if not allow_partial and (
        source_metadata.get("assembly_complete") is False
        or source_metadata.get("legal") is False
        or str(source_metadata.get("assembly_status", "")).lower() in {"partial", "failed", "incomplete"}
    ):
        raise CG2MolConversionError(
            "RL_MTCS assembly is not complete; "
            f"status={source_metadata.get('assembly_status')}, "
            f"reasons={source_metadata.get('completion_reasons', [])}"
        )
    cg_nodes = list(simplified_cg.get("nodes", []) or [])
    if not cg_nodes:
        raise CG2MolConversionError("coarse-grained graph contains no nodes")
    records = _bond_records(simplified_cg)

    node_by_seq: Dict[int, Dict[str, Any]] = {}
    for node in cg_nodes:
        if not isinstance(node, dict) or "seq" not in node or "su_type" not in node:
            raise CG2MolConversionError("each CG node requires seq and su_type")
        sequence = int(node["seq"])
        if sequence in node_by_seq:
            raise CG2MolConversionError(f"duplicate CG sequence: {sequence}")
        node_by_seq[sequence] = node

    neighbors: Dict[int, List[int]] = {sequence: [] for sequence in node_by_seq}
    normalized_records: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    for record in records:
        left, right = int(record["source"]), int(record["target"])
        if left == right or left not in node_by_seq or right not in node_by_seq:
            raise CG2MolConversionError(
                f"CG edge {left}-{right} references an invalid node"
            )
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            raise CG2MolConversionError(f"duplicate CG edge: {pair[0]}-{pair[1]}")
        seen_pairs.add(pair)
        normalized = dict(record)
        normalized["source"] = left
        normalized["target"] = right
        normalized["kind"] = str(record.get("kind", ""))
        normalized_records.append(normalized)
        neighbors[left].append(right)
        neighbors[right].append(left)

    for sequence in neighbors:
        neighbors[sequence].sort()
        declared_degree = node_by_seq[sequence].get("actual_degree")
        if declared_degree is not None and int(declared_degree) != len(neighbors[sequence]):
            raise CG2MolConversionError(
                f"CG node {node_by_seq[sequence].get('id', sequence)} actual_degree "
                "does not match the final edges"
            )

    working_nodes = [node_by_seq[sequence] for sequence in sorted(node_by_seq)]
    nitrogen_ring_sizes = _aromatic_nitrogen_ring_sizes(working_nodes, normalized_records)
    double_edges = _select_multiple_bonds(working_nodes, normalized_records, DOUBLE_BOND_SU)
    triple_edges = _select_multiple_bonds(working_nodes, normalized_records, TRIPLE_BOND_SU)
    seq_to_su = {
        sequence: int(node_by_seq[sequence]["su_type"])
        for sequence in node_by_seq
    }

    all_atoms: List[Dict[str, Any]] = []
    all_bonds: List[List[Any]] = []
    cg_bond_details: List[Dict[str, Any]] = []
    cg_to_atom_map: Dict[int, Dict[int, int]] = {}
    node_id_to_atom_map: Dict[str, Dict[int, int]] = {}
    cg_connect_atoms: Dict[int, Dict[int, int]] = {}
    atom_idx = 0
    atom_bond_pairs: Set[Tuple[int, int]] = set()

    def append_bond(
        atom_a: int,
        atom_b: int,
        bond_type: str,
        source: str,
    ) -> None:
        pair = tuple(sorted((int(atom_a), int(atom_b))))
        if pair in atom_bond_pairs:
            raise CG2MolConversionError(
                f"duplicate all-atom bond between atoms {pair[0]}-{pair[1]}"
            )
        atom_bond_pairs.add(pair)
        all_bonds.append([int(atom_a), int(atom_b), str(bond_type), str(source)])

    for sequence in sorted(node_by_seq):
        node = node_by_seq[sequence]
        su_type = int(node["su_type"])
        degree = len(neighbors[sequence])
        try:
            expansion = _node_expansion(su_type, degree, nitrogen_ring_sizes.get(sequence))
        except CG2MolConversionError as exc:
            raise CG2MolConversionError(f"CG node {node.get('id', sequence)}: {exc}") from exc
        cg_id = node.get("node_id", node.get("id", sequence))
        if str(cg_id) in node_id_to_atom_map:
            raise CG2MolConversionError(f"duplicate CG node id: {cg_id}")
        axial_coord = node.get("axial_coord", [0, 0])
        xy_coord = node.get("xy_coord")
        if isinstance(xy_coord, (list, tuple)) and len(xy_coord) >= 2:
            try:
                base_x, base_y = float(xy_coord[0]), float(xy_coord[1])
            except (TypeError, ValueError):
                base_x, base_y = hex_to_cartesian(axial_coord[0], axial_coord[1])
        else:
            base_x, base_y = hex_to_cartesian(axial_coord[0], axial_coord[1])

        local_to_global: Dict[int, int] = {}
        cg_to_atom_map[sequence] = local_to_global
        node_id_to_atom_map[str(cg_id)] = local_to_global
        cg_connect_atoms[sequence] = {}
        for local_idx, (element, atom_type) in enumerate(expansion.atoms):
            dx, dy = (
                expansion.coord_offsets[local_idx]
                if local_idx < len(expansion.coord_offsets)
                else (0.0, 0.0)
            )
            atom = {
                "idx": atom_idx,
                "element": element,
                "atom_type": atom_type,
                "cg_seq": sequence,
                "cg_id": cg_id,
                "su_type": su_type,
                "local_idx": local_idx,
                "coord": [base_x + dx, base_y + dy, 0.0],
                "axial_coord": list(axial_coord),
                "aromatic": su_type in AROMATIC_SU and local_idx == 0,
            }
            all_atoms.append(atom)
            local_to_global[local_idx] = atom_idx
            atom_idx += 1

        for local_i, local_j, bond_type in expansion.internal_bonds:
            if local_i not in local_to_global or local_j not in local_to_global:
                raise CG2MolConversionError(
                    f"invalid internal bond in SU {su_type}: {local_i}-{local_j}"
                )
            append_bond(
                local_to_global[local_i],
                local_to_global[local_j],
                bond_type,
                "internal",
            )

        connection_points = _connection_points(su_type, neighbors[sequence], seq_to_su)
        for neighbor_sequence, local_connect in connection_points.items():
            if local_connect not in local_to_global:
                raise CG2MolConversionError(
                    f"SU {su_type} has no connection atom {local_connect}"
                )
            cg_connect_atoms[sequence][neighbor_sequence] = local_to_global[local_connect]

    for record in normalized_records:
        cg_a, cg_b = int(record["source"]), int(record["target"])
        atom_a = cg_connect_atoms[cg_a].get(cg_b)
        atom_b = cg_connect_atoms[cg_b].get(cg_a)
        if atom_a is None or atom_b is None:
            raise CG2MolConversionError(
                f"cannot resolve connection atoms for CG edge {cg_a}-{cg_b}"
            )
        su_a, su_b = seq_to_su[cg_a], seq_to_su[cg_b]
        pair = tuple(sorted((cg_a, cg_b)))
        bond_type = _bond_type_for_edge(
            su_a,
            su_b,
            str(record.get("kind", "")),
            pair,
            double_edges,
            triple_edges,
        )
        append_bond(atom_a, atom_b, bond_type, "cg_bond")
        cg_bond_details.append({
            "source_seq": cg_a,
            "target_seq": cg_b,
            "source_node_id": record.get("source_node_id"),
            "target_node_id": record.get("target_node_id"),
            "kind": str(record.get("kind", "")),
            "bond_type": bond_type,
            "atom_indices": [atom_a, atom_b],
        })

    element_counts: Dict[str, int] = {}
    for atom in all_atoms:
        element = str(atom["element"])
        element_counts[element] = element_counts.get(element, 0) + 1
    edge_kind_counts: Dict[str, int] = {}
    for record in normalized_records:
        kind = str(record.get("kind", "")) or "unspecified"
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1

    metadata = {
        "cg_node_count": len(cg_nodes),
        "cg_bond_count": len(normalized_records),
        "atom_count": len(all_atoms),
        "bond_count": len(all_bonds),
        "element_counts": element_counts,
        "edge_kind_counts": edge_kind_counts,
        "multiple_bond_count": len(double_edges) + len(triple_edges),
        "final_su_counts": {
            str(su_type): sum(int(node["su_type"]) == su_type for node in working_nodes)
            for su_type in sorted(set(seq_to_su.values()))
        },
        "pyrrolic_nitrogen_node_ids": [
            node_by_seq[sequence].get("node_id", node_by_seq[sequence].get("id", sequence))
            for sequence, ring_size in nitrogen_ring_sizes.items() if ring_size == 5
        ],
        "created_at": datetime.now().isoformat(),
    }
    if isinstance(source_metadata, dict):
        for key in (
            "source_format", "assembly_status", "assembly_complete", "legal",
            "completion_reasons", "source_file", "unassembled_resource_count",
            "strict_all_nodes_assembled", "allowed_remaining_resources",
        ):
            if key in source_metadata:
                metadata[key] = source_metadata[key]

    return {
        "atoms": all_atoms,
        "bonds": [[bond[0], bond[1], bond[2]] for bond in all_bonds],
        "bond_details": all_bonds,
        "cg_bond_details": cg_bond_details,
        "metadata": metadata,
        "cg_to_atom_map": {str(key): value for key, value in cg_to_atom_map.items()},
        "node_id_to_atom_map": node_id_to_atom_map,
    }


def convert_simplified_to_allatom(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    allow_partial: bool = False,
) -> Dict:
    """Read legacy or RL_MTCS graph JSON and write an all-atom JSON file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    from .cg_simplify import simplify_cg_molecule

    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list) and isinstance(payload.get("bonds"), list):
        simplified_cg = payload
    else:
        simplified_cg = simplify_cg_molecule(payload)

    source_metadata = simplified_cg.get("metadata", {})
    if (
        isinstance(source_metadata, dict)
        and source_metadata.get("source_format") == "rl_mtcs"
        and not allow_partial
    ):
        status = str(source_metadata.get("assembly_status", "unknown")).lower()
        complete = source_metadata.get("assembly_complete")
        legal = source_metadata.get("legal")
        if status in {"partial", "failed", "incomplete"} or complete is False or legal is False:
            reasons = source_metadata.get("completion_reasons", [])
            raise CG2MolConversionError(
                "RL_MTCS assembly is not complete; refusing all-atom conversion: "
                f"status={status}, reasons={reasons}"
            )

    result = convert_cg_to_allatom(simplified_cg, allow_partial=allow_partial)
    from .allatom_to_rdkit import convert_allatom_to_rdkit

    convert_allatom_to_rdkit(result, verbose=False)
    result["metadata"]["source_file"] = input_path

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if base.endswith("_simplified"):
            base = base[:-11]
        output_path = f"{base}_allatom{ext or '.json'}"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"[CG->AllAtom] Input: {input_path}")
    print(f"[CG->AllAtom] Output: {output_path}")
    print(f"[CG->AllAtom] CG Nodes: {result['metadata']['cg_node_count']}")
    print(f"[CG->AllAtom] CG Bonds: {result['metadata']['cg_bond_count']}")
    print(f"[CG->AllAtom] Atoms: {result['metadata']['atom_count']}")
    print(f"[CG->AllAtom] Bonds: {result['metadata']['bond_count']}")
    return result


def print_allatom_summary(allatom: Dict):
    """打印全原子结果的摘要"""
    atoms = allatom.get('atoms', [])
    bonds = allatom.get('bonds', [])
    
    print("\n" + "=" * 60)
    print("全原子分子转化结果")
    print("=" * 60)
    
    # 统计元素分布
    elem_counts: Dict[str, int] = {}
    for atom in atoms:
        elem = atom.get('element', '?')
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
    
    print(f"\n原子总数: {len(atoms)}")
    print(f"键总数: {len(bonds)}")
    
    print("\n元素分布:")
    for elem, count in sorted(elem_counts.items()):
        print(f"  {elem:2s}: {count:3d} 个")
    
    # 统计键类型
    bond_type_counts: Dict[str, int] = {}
    for bond in bonds:
        bt = bond[2] if len(bond) > 2 else "SINGLE"
        bond_type_counts[bt] = bond_type_counts.get(bt, 0) + 1
    
    print("\n键类型分布:")
    for bt, count in sorted(bond_type_counts.items()):
        print(f"  {bt:10s}: {count:3d} 个")
    
    # 分子式
    formula_parts = []
    for elem in ['C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'I']:
        if elem in elem_counts:
            if elem_counts[elem] == 1:
                formula_parts.append(elem)
            else:
                formula_parts.append(f"{elem}{elem_counts[elem]}")
    print(f"\n分子式: {''.join(formula_parts)}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse
    if not __package__:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        __package__ = "cg2mol"
    
    parser = argparse.ArgumentParser(description='将CG或RL_MTCS组装图转化为全原子分子')
    parser.add_argument(
        'input',
        help='输入JSON文件路径（简化CG、RL_MTCS final.json或final_molecule.json）',
    )
    parser.add_argument('-o', '--output', help='输出JSON文件路径（可选）')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印详细信息')
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='允许转换未完成的RL_MTCS组装图',
    )
    
    args = parser.parse_args()
    
    result = convert_simplified_to_allatom(
        args.input,
        args.output,
        allow_partial=args.allow_partial,
    )
    
    if args.verbose:
        print_allatom_summary(result)
