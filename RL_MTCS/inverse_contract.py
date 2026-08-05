"""Immutable inverse-to-MCTS topology contracts.

The legacy RL_MTCS implementation rebuilt aromatic clusters from SU counts and
then allocated new chains.  This module defines the replacement boundary: the
inverse pipeline owns topology and node identity, while MCTS only places the
already-complete units on the hexagonal lattice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Axial = Tuple[int, int]

HEX_DIRECTIONS: Tuple[Axial, ...] = (
    (1, 1),
    (1, 0),
    (-1, 0),
    (-1, -1),
    (0, 1),
    (0, -1),
)

AROMATIC_KINDS = {
    "benzene", "naphthalene", "phenanthrene", "anthracene", "tetracene",
    "chrysene", "triphenylene", "pyrene", "benzo_pyrene", "perylene",
    "coronene", "aromatic", "aromatic_cluster", "aromatic_unit",
}


class InverseInputError(ValueError):
    """Raised when inverse output is not complete enough for placement."""


def _dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_public_dict"):
        try:
            return dict(value.to_public_dict() or {})
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict() or {})
        except Exception:
            pass
    if is_dataclass(value):
        return {
            str(name): getattr(value, name)
            for name in getattr(value, "__dataclass_fields__", {})
        }
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _axial(value: Any, default: Optional[Axial] = None) -> Optional[Axial]:
    if isinstance(value, Mapping):
        value = value.get("local_axial", value.get("axial", value.get("coord")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except Exception:
            return default
    return default


def _unique_ints(values: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(sorted({_int(value) for value in values if _int(value) >= 0}))


def _node_id_from_row(row: Any) -> int:
    data = _dict(row)
    return _int(data.get("node_id", data.get("global_id", getattr(row, "global_id", -1))))


@dataclass(frozen=True)
class UnitNode:
    node_id: int
    su_type: int
    local_axial: Axial
    role: str = ""
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class UnitEdge:
    source_node_id: int
    target_node_id: int
    kind: str = "internal"


@dataclass(frozen=True)
class UnitPort:
    port_id: str
    node_id: int
    role: str = "open"
    phase: str = ""
    direction: Optional[Axial] = None
    contract_id: str = ""
    required_su_types: Tuple[int, ...] = ()
    used: bool = False
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TopologyUnit:
    unit_id: str
    product_type: str
    nodes: Tuple[UnitNode, ...]
    internal_edges: Tuple[UnitEdge, ...] = ()
    ports: Tuple[UnitPort, ...] = ()
    source_fragment_ids: Tuple[int, ...] = ()
    source_node_ids: Tuple[int, ...] = ()
    topology_notation: str = ""
    node_id_notation: str = ""
    closed: bool = True
    local_rings: Tuple[Tuple[Axial, ...], ...] = ()
    metadata: Tuple[Tuple[str, str], ...] = ()

    @property
    def phase(self) -> str:
        kind = str(self.product_type).lower()
        if kind in AROMATIC_KINDS or kind.startswith("aromatic"):
            return "aromatic"
        if kind in {"flex_chain", "bridge", "aliphatic_ring", "fatty_ring", "fatty_ring_fused", "fatty_ring_side", "fatty_ring_upper_lower"}:
            return "flex"
        if "flex" in kind or "ring" in kind:
            return "flex"
        return "side"

    def node_map(self) -> Dict[int, UnitNode]:
        return {int(node.node_id): node for node in self.nodes}


@dataclass(frozen=True)
class ExternalContract:
    contract_id: str
    source_unit_id: str
    source_node_id: int
    target_unit_id: str
    target_node_id: int
    kind: str
    mandatory: bool = True
    fixed: bool = False
    source_port_id: str = ""
    target_port_id: str = ""
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class InverseAssemblyInput:
    units: Tuple[TopologyUnit, ...]
    contracts: Tuple[ExternalContract, ...]
    node_su_types: Tuple[Tuple[int, int], ...]
    ownership: Tuple[Tuple[int, str], ...]
    diagnostics: Tuple[Tuple[str, str], ...] = ()

    def node_map(self) -> Dict[int, int]:
        return {int(node_id): int(su) for node_id, su in self.node_su_types}

    def ownership_map(self) -> Dict[int, str]:
        return {int(node_id): str(unit_id) for node_id, unit_id in self.ownership}


def _extract_products(payload: Any) -> List[Any]:
    root = _dict(payload)
    for key in ("topology_products", "products"):
        values = root.get(key)
        if isinstance(values, (list, tuple)):
            return list(values)
    for key in ("external_assembly", "assembly_view", "assembly"):
        nested = root.get(key)
        if nested is not None:
            found = _extract_products(nested)
            if found:
                return found
    return []


def _extract_aromatic_units(payload: Any) -> List[Any]:
    root = _dict(payload)
    for key in ("aromatic_units", "aromatic_clusters", "clusters"):
        values = root.get(key)
        if isinstance(values, (list, tuple)):
            return list(values)
    for key in ("external_assembly", "assembly_view", "assembly"):
        nested = root.get(key)
        if nested is not None:
            found = _extract_aromatic_units(nested)
            if found:
                return found
    return []


def _extract_nodes(payload: Any) -> Dict[int, Dict[str, Any]]:
    root = _dict(payload)
    values = root.get("nodes", root.get("node_table", []))
    if isinstance(values, Mapping):
        values = list(values.values())
    result: Dict[int, Dict[str, Any]] = {}
    for row in list(values or []):
        data = _dict(row)
        node_id = _node_id_from_row(row)
        if node_id < 0:
            continue
        result[node_id] = {
            **data,
            "node_id": node_id,
            "su_type": _int(data.get("su_type", getattr(row, "su_type", -1))),
            "hop1_ids": list(data.get("hop1_ids", getattr(row, "hop1_ids", [])) or []),
            "fixed_hop1_ids": list(data.get("fixed_hop1_ids", getattr(row, "fixed_hop1_ids", [])) or []),
        }
    return result


def _parse_ids(text: Any) -> List[int]:
    return [_int(value) for value in re.findall(r"\d+", str(text or ""))]


def _parse_su_tokens(text: Any) -> List[int]:
    values: List[int] = []
    for token in re.findall(r"\d+", str(text or "")):
        value = _int(token)
        if 0 <= value <= 32:
            values.append(value)
    return values


def _node_rows_for_unit(raw: Any, global_nodes: Dict[int, Dict[str, Any]], unit_id: str) -> List[Dict[str, Any]]:
    data = _dict(raw)
    rows = data.get("nodes", data.get("sites", data.get("local_nodes", [])))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    rows = list(rows or [])
    ids = list(data.get("source_node_ids", data.get("node_ids", [])) or [])
    if not rows and ids:
        su_tokens = _parse_su_tokens(data.get("topology_notation", data.get("actual_topology_su", "")))
        rows = [
            {
                "node_id": _int(node_id),
                "su_type": global_nodes.get(_int(node_id), {}).get("su_type", su_tokens[index] if index < len(su_tokens) else -1),
            }
            for index, node_id in enumerate(ids)
        ]
    if not rows:
        gid_ids = _parse_ids(data.get("node_id_notation", data.get("actual_topology_gid", "")))
        su_tokens = _parse_su_tokens(data.get("topology_notation", data.get("actual_topology_su", "")))
        rows = [
            {"node_id": node_id, "su_type": global_nodes.get(node_id, {}).get("su_type", su_tokens[index] if index < len(su_tokens) else -1)}
            for index, node_id in enumerate(gid_ids)
        ]

    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = _dict(row)
        node_id = _node_id_from_row(row)
        if node_id < 0:
            continue
        global_row = global_nodes.get(node_id, {})
        item["node_id"] = node_id
        item["su_type"] = _int(item.get("su_type", global_row.get("su_type", -1)))
        item["role"] = str(item.get("role", ""))
        item["local_axial"] = _axial(item.get("local_axial", item.get("axial", item.get("coord"))))
        if item["local_axial"] is None:
            item["local_axial"] = (index, 0)
        normalized.append(item)
    return normalized


def _normalize_edges(raw: Any, node_ids: Sequence[int]) -> List[UnitEdge]:
    data = _dict(raw)
    values = data.get("internal_edges", data.get("edges", []))
    if isinstance(values, Mapping):
        values = list(values.values())
    edges: List[UnitEdge] = []
    node_set = {int(node_id) for node_id in node_ids}
    for edge in list(values or []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            left, right = _int(edge[0]), _int(edge[1])
            kind = "internal"
        else:
            row = _dict(edge)
            left = _int(row.get("source_node_id", row.get("u", row.get("source", -1))))
            right = _int(row.get("target_node_id", row.get("v", row.get("target", -1))))
            kind = str(row.get("kind", "internal"))
        # Legacy aromatic cluster exports used site indices in ``edges``;
        # convert those indices to the inverse node IDs at the boundary.
        if left not in node_set and 0 <= left < len(node_ids):
            left = int(node_ids[left])
        if right not in node_set and 0 <= right < len(node_ids):
            right = int(node_ids[right])
        if left >= 0 and right >= 0 and left != right:
            edges.append(UnitEdge(left, right, kind))
    if not edges and len(node_ids) > 1:
        edges = [UnitEdge(int(left), int(right)) for left, right in zip(node_ids, node_ids[1:])]
    return edges


def _normalize_ports(raw: Any, unit_id: str, node_ids: Sequence[int], phase: str, edges: Sequence[UnitEdge]) -> List[UnitPort]:
    data = _dict(raw)
    values = data.get("ports", data.get("open_ports", []))
    if isinstance(values, Mapping):
        values = list(values.values())
    ports: List[UnitPort] = []
    for index, value in enumerate(list(values or [])):
        row = _dict(value)
        node_id = _int(row.get("node_id", row.get("anchor_node_id", -1)))
        if node_id < 0:
            continue
        required = tuple(sorted({_int(v) for v in list(row.get("required_su_types", row.get("required_su_sequence", [])) or []) if _int(v) >= 0}))
        ports.append(UnitPort(
            port_id=str(row.get("port_id", row.get("slot_id", f"{unit_id}:port:{index}"))),
            node_id=node_id,
            role=str(row.get("role", row.get("anchor_role", "open"))),
            phase=str(row.get("phase", phase)),
            direction=_axial(row.get("direction", row.get("outward_direction"))),
            contract_id=str(row.get("contract_id", row.get("slot_id", ""))),
            required_su_types=required,
            metadata=tuple(sorted((str(k), str(v)) for k, v in row.items() if k not in {"node_id", "anchor_node_id", "port_id", "slot_id", "role", "anchor_role", "phase", "direction", "outward_direction", "contract_id", "required_su_types", "required_su_sequence"})),
        ))
    if ports:
        return ports
    degree: Dict[int, int] = {int(node_id): 0 for node_id in node_ids}
    for edge in edges:
        degree[int(edge.source_node_id)] = degree.get(int(edge.source_node_id), 0) + 1
        degree[int(edge.target_node_id)] = degree.get(int(edge.target_node_id), 0) + 1
    endpoint_ids = [int(node_id) for node_id in node_ids if degree.get(int(node_id), 0) <= 1]
    if len(endpoint_ids) == 0 and node_ids:
        endpoint_ids = [int(node_ids[0])]
    return [UnitPort(
        port_id=f"{unit_id}:port:{index}",
        node_id=node_id,
        role="open",
        phase=phase,
    ) for index, node_id in enumerate(endpoint_ids)]


def _normalize_unit(raw: Any, index: int, global_nodes: Dict[int, Dict[str, Any]], aromatic: bool = False) -> TopologyUnit:
    data = _dict(raw)
    kind = str(data.get("product_type", data.get("product_kind", data.get("kind", "aromatic_cluster" if aromatic else ""))) or "")
    unit_id = str(data.get("unit_id", data.get("cluster_id", data.get("id", f"{'aromatic' if aromatic else 'product'}:{index}"))))
    rows = _node_rows_for_unit(data, global_nodes, unit_id)
    if not rows:
        raise InverseInputError(f"inverse unit {unit_id!r} has no node geometry")
    if any(_int(row.get("su_type", -1)) < 0 for row in rows):
        raise InverseInputError(f"inverse unit {unit_id!r} has nodes without su_type")
    nodes = tuple(UnitNode(
        node_id=_int(row["node_id"]),
        su_type=_int(row["su_type"]),
        local_axial=_axial(row.get("local_axial"), (0, 0)) or (0, 0),
        role=str(row.get("role", "")),
        metadata=tuple(sorted((str(k), str(v)) for k, v in row.items() if k not in {"node_id", "su_type", "local_axial", "axial", "coord", "role"})),
    ) for row in rows)
    node_ids = tuple(node.node_id for node in nodes)
    edges = tuple(_normalize_edges(data, node_ids))
    ports = tuple(_normalize_ports(data, unit_id, node_ids, "aromatic" if aromatic else "side", edges))
    source_fragment_ids = _unique_ints(data.get("source_fragment_ids", []))
    source_node_ids = _unique_ints(data.get("source_node_ids", data.get("node_ids", node_ids)))
    rings_raw = data.get("rings", data.get("local_rings", []))
    rings: List[Tuple[Axial, ...]] = []
    for ring in list(rings_raw or []):
        points = [_axial(point) for point in list(ring or [])]
        points = [point for point in points if point is not None]
        if points:
            rings.append(tuple(points))
    return TopologyUnit(
        unit_id=unit_id,
        product_type=kind,
        nodes=nodes,
        internal_edges=edges,
        ports=ports,
        source_fragment_ids=source_fragment_ids,
        source_node_ids=source_node_ids,
        topology_notation=str(data.get("topology_notation", data.get("actual_topology_su", "")) or ""),
        node_id_notation=str(data.get("node_id_notation", data.get("actual_topology_gid", "")) or ""),
        closed=bool(data.get("closed", True)),
        local_rings=tuple(rings),
        metadata=tuple(sorted((str(k), str(v)) for k, v in data.items() if k not in {"unit_id", "cluster_id", "id", "product_type", "product_kind", "kind", "nodes", "sites", "local_nodes", "node_ids", "source_node_ids", "internal_edges", "edges", "ports", "open_ports", "rings", "local_rings", "topology_notation", "actual_topology_su", "node_id_notation", "actual_topology_gid", "closed", "source_fragment_ids"})),
    )


def _infer_contracts(payload: Any, units: Sequence[TopologyUnit], nodes: Dict[int, Dict[str, Any]]) -> List[ExternalContract]:
    root = _dict(payload)
    explicit = root.get("external_contracts", root.get("connection_contracts", []))
    if not explicit:
        explicit = dict(root.get("diagnostics", {}) or {}).get("external_contracts", [])
    if not explicit:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = _dict(root.get(key))
            explicit = nested.get("external_contracts", nested.get("connection_contracts", []))
            if explicit:
                break
    ownership = {int(node.node_id): unit.unit_id for unit in units for node in unit.nodes}
    ports_by_node = {
        int(port.node_id): (unit.unit_id, port.port_id)
        for unit in units
        for port in unit.ports
    }
    contracts: List[ExternalContract] = []
    for index, value in enumerate(list(explicit or [])):
        row = _dict(value)
        source_node = _int(row.get("source_node_id", row.get("u", row.get("left_node_id", -1))))
        target_node = _int(row.get("target_node_id", row.get("v", row.get("right_node_id", -1))))
        source_unit = str(row.get("source_unit_id", ownership.get(source_node, "")))
        target_unit = str(row.get("target_unit_id", ownership.get(target_node, "")))
        if source_node < 0 or target_node < 0 or not source_unit or not target_unit or source_unit == target_unit:
            continue
        kind = str(row.get("kind", row.get("connection_kind", row.get("phase", "side"))))
        if kind not in {"rigid", "flex", "side"}:
            kind = "side"
        contracts.append(ExternalContract(
            contract_id=str(row.get("contract_id", f"external:{index}")),
            source_unit_id=source_unit,
            source_node_id=source_node,
            target_unit_id=target_unit,
            target_node_id=target_node,
            kind=kind,
            mandatory=bool(row.get("mandatory", True)),
            fixed=bool(row.get("fixed", row.get("connection_semantics", "") == "fixed")),
            source_port_id=str(row.get("source_port_id", ports_by_node.get(source_node, ("", ""))[1])),
            target_port_id=str(row.get("target_port_id", ports_by_node.get(target_node, ("", ""))[1])),
        ))
    if contracts:
        return contracts
    seen: set[Tuple[int, int]] = set()
    for source_node, row in sorted(nodes.items()):
        for target_node in list(row.get("hop1_ids", []) or []):
            target_node = _int(target_node)
            pair = tuple(sorted((int(source_node), int(target_node))))
            if target_node < 0 or pair in seen or source_node not in ownership or target_node not in ownership:
                continue
            seen.add(pair)
            source_unit, target_unit = ownership[source_node], ownership[target_node]
            if source_unit == target_unit:
                continue
            source_phase = next(unit.phase for unit in units if unit.unit_id == source_unit)
            target_phase = next(unit.phase for unit in units if unit.unit_id == target_unit)
            if source_phase == target_phase == "aromatic":
                kind = "rigid"
            elif "flex" in {source_phase, target_phase}:
                kind = "flex"
            else:
                kind = "side"
            fixed_ids = {_int(value) for value in list(row.get("fixed_hop1_ids", []) or [])}
            target_row = nodes.get(target_node, {})
            fixed = target_node in fixed_ids or source_node in {_int(value) for value in list(target_row.get("fixed_hop1_ids", []) or [])}
            contracts.append(ExternalContract(
                contract_id=f"hop1:{pair[0]}-{pair[1]}",
                source_unit_id=source_unit,
                source_node_id=source_node,
                target_unit_id=target_unit,
                target_node_id=target_node,
                kind=kind,
                fixed=fixed,
                source_port_id=ports_by_node.get(source_node, ("", ""))[1],
                target_port_id=ports_by_node.get(target_node, ("", ""))[1],
            ))
    return contracts


def build_inverse_input(payload: Any) -> InverseAssemblyInput:
    """Normalize a complete inverse result for the new MCTS initializer."""
    root = _dict(payload)
    global_nodes = _extract_nodes(payload)
    raw_aromatic = _extract_aromatic_units(payload)
    raw_products = _extract_products(payload)
    if not raw_aromatic:
        raise InverseInputError(
            "inverse result has no aromatic_units; export complete aromatic geometry "
            "with node_id/local_axial before starting RL_MTCS"
        )
    units: List[TopologyUnit] = []
    for index, raw in enumerate(raw_aromatic):
        units.append(_normalize_unit(raw, index, global_nodes, aromatic=True))
    for index, raw in enumerate(raw_products):
        units.append(_normalize_unit(raw, index, global_nodes, aromatic=False))
    by_node: Dict[int, str] = {}
    for unit in units:
        for node in unit.nodes:
            if node.node_id in by_node and by_node[node.node_id] != unit.unit_id:
                raise InverseInputError(
                    f"node {node.node_id} is claimed by both {by_node[node.node_id]!r} and {unit.unit_id!r}"
                )
            by_node[node.node_id] = unit.unit_id
    if global_nodes:
        missing = sorted(set(global_nodes) - set(by_node))
        support = {
            _int(value)
            for value in list(dict(root.get("ownership_ledger", {}) or {}).get("support_context_node_ids", []) or [])
        }
        missing = [node_id for node_id in missing if node_id not in support]
        if missing:
            raise InverseInputError(f"inverse nodes are not covered by any complete unit: {missing[:12]}")
    contracts = _infer_contracts(payload, units, global_nodes)
    node_su = {
        int(node.node_id): int(node.su_type)
        for unit in units
        for node in unit.nodes
    }
    return InverseAssemblyInput(
        units=tuple(units),
        contracts=tuple(contracts),
        node_su_types=tuple(sorted(node_su.items())),
        ownership=tuple(sorted(by_node.items())),
        diagnostics=tuple(sorted((str(k), str(v)) for k, v in {
            "unit_count": len(units),
            "contract_count": len(contracts),
            "aromatic_unit_count": sum(1 for unit in units if unit.phase == "aromatic"),
        }.items())),
    )
