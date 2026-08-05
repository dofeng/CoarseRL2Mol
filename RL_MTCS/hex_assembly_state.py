"""Hexagonal placement state for immutable inverse topology units."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .inverse_contract import (
    Axial,
    ExternalContract,
    HEX_DIRECTIONS,
    InverseAssemblyInput,
    TopologyUnit,
    UnitPort,
)


def axial_add(left: Axial, right: Axial) -> Axial:
    return (int(left[0]) + int(right[0]), int(left[1]) + int(right[1]))


def axial_neg(value: Axial) -> Axial:
    return (-int(value[0]), -int(value[1]))


def axial_distance(left: Axial, right: Axial) -> int:
    q1, r1 = int(left[0]), int(left[1])
    q2, r2 = int(right[0]), int(right[1])
    return int((abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2)


def rotate_axial(value: Axial, rotation: int) -> Axial:
    """Rotate in the coordinate convention used by legacy ``HexGrid``."""
    q, r = int(value[0]), int(value[1])
    for _ in range(int(rotation) % 6):
        q, r = q - r, q
    return int(q), int(r)


def transform_direction(direction: Axial, rotation: int) -> Axial:
    return rotate_axial(direction, rotation)


@dataclass(frozen=True)
class UnitPlacement:
    unit_id: str
    rotation: int
    translation: Axial


@dataclass(frozen=True)
class AssemblyEdge:
    source_node_id: int
    target_node_id: int
    kind: str
    contract_id: str = ""
    direction: Optional[Axial] = None


@dataclass
class HexAssemblyState:
    """Mutable search snapshot; ``problem.units`` remain immutable."""

    problem: InverseAssemblyInput
    placements: Dict[str, UnitPlacement] = field(default_factory=dict)
    node_axial: Dict[int, Axial] = field(default_factory=dict)
    occupied_axial: Dict[Axial, int] = field(default_factory=dict)
    edges: List[AssemblyEdge] = field(default_factory=list)
    used_ports: Set[str] = field(default_factory=set)
    satisfied_contracts: Set[str] = field(default_factory=set)
    stage: str = "aromatic"
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._units = {unit.unit_id: unit for unit in self.problem.units}
        self._ports = {
            f"{unit.unit_id}:{port.port_id}": port
            for unit in self.problem.units
            for port in unit.ports
        }

    def clone(self) -> "HexAssemblyState":
        new = HexAssemblyState(problem=self.problem)
        new.placements = dict(self.placements)
        new.node_axial = dict(self.node_axial)
        new.occupied_axial = dict(self.occupied_axial)
        new.edges = list(self.edges)
        new.used_ports = set(self.used_ports)
        new.satisfied_contracts = set(self.satisfied_contracts)
        new.stage = str(self.stage)
        new.action_history = copy.deepcopy(self.action_history)
        return new

    def unit(self, unit_id: str) -> TopologyUnit:
        return self._units[str(unit_id)]

    def is_placed(self, unit_id: str) -> bool:
        return str(unit_id) in self.placements

    def placed_units(self, phase: Optional[str] = None) -> List[TopologyUnit]:
        values = [self.unit(unit_id) for unit_id in self.placements]
        if phase is not None:
            values = [unit for unit in values if unit.phase == str(phase)]
        return sorted(values, key=lambda unit: unit.unit_id)

    def unplaced_units(self, phase: Optional[str] = None) -> List[TopologyUnit]:
        values = [unit for unit in self.problem.units if not self.is_placed(unit.unit_id)]
        if phase is not None:
            values = [unit for unit in values if unit.phase == str(phase)]
        return sorted(values, key=lambda unit: unit.unit_id)

    def global_node_axial(self, unit_id: str, node_id: int, rotation: Optional[int] = None, translation: Optional[Axial] = None) -> Axial:
        unit = self.unit(unit_id)
        node = next((node for node in unit.nodes if int(node.node_id) == int(node_id)), None)
        if node is None:
            raise KeyError(f"unit {unit_id!r} has no node {node_id}")
        placement = self.placements.get(str(unit_id))
        rot = int(rotation if rotation is not None else placement.rotation if placement else 0)
        trans = translation if translation is not None else placement.translation if placement else (0, 0)
        return axial_add(rotate_axial(node.local_axial, rot), trans)

    def global_port_direction(self, unit_id: str, port: UnitPort, rotation: Optional[int] = None) -> Optional[Axial]:
        if port.direction is None:
            return None
        placement = self.placements.get(str(unit_id))
        rot = int(rotation if rotation is not None else placement.rotation if placement else 0)
        return transform_direction(port.direction, rot)

    def _placement_coordinates(self, unit: TopologyUnit, rotation: int, translation: Axial) -> Dict[int, Axial]:
        return {
            int(node.node_id): axial_add(rotate_axial(node.local_axial, int(rotation)), translation)
            for node in unit.nodes
        }

    def can_place(self, unit_id: str, rotation: int, translation: Axial) -> Tuple[bool, str]:
        unit = self.unit(unit_id)
        if self.is_placed(unit_id):
            return False, "unit_already_placed"
        coordinates = self._placement_coordinates(unit, rotation, translation)
        if len(set(coordinates.values())) != len(coordinates):
            return False, "unit_internal_coordinate_collision"
        for node_id, axial in coordinates.items():
            occupant = self.occupied_axial.get(axial)
            if occupant is not None:
                return False, f"axial_collision:{axial}:{occupant}:{node_id}"
        return True, ""

    def place_unit(self, unit_id: str, rotation: int, translation: Axial) -> bool:
        ok, _reason = self.can_place(unit_id, rotation, translation)
        if not ok:
            return False
        unit = self.unit(unit_id)
        placement = UnitPlacement(str(unit_id), int(rotation) % 6, (int(translation[0]), int(translation[1])))
        coordinates = self._placement_coordinates(unit, placement.rotation, placement.translation)
        self.placements[str(unit_id)] = placement
        for node_id, axial in coordinates.items():
            self.node_axial[int(node_id)] = axial
            self.occupied_axial[axial] = int(node_id)
        for edge in unit.internal_edges:
            if int(edge.source_node_id) not in coordinates or int(edge.target_node_id) not in coordinates:
                continue
            direction = self._edge_direction(coordinates[int(edge.source_node_id)], coordinates[int(edge.target_node_id)])
            self._append_edge(int(edge.source_node_id), int(edge.target_node_id), str(edge.kind), direction=direction)
        return True

    @staticmethod
    def _edge_direction(source: Axial, target: Axial) -> Optional[Axial]:
        delta = (int(target[0]) - int(source[0]), int(target[1]) - int(source[1]))
        return delta if delta in HEX_DIRECTIONS else None

    def _append_edge(self, source: int, target: int, kind: str, contract_id: str = "", direction: Optional[Axial] = None) -> None:
        key = {int(source), int(target), str(kind), str(contract_id)}
        if any({int(edge.source_node_id), int(edge.target_node_id), str(edge.kind), str(edge.contract_id)} == key for edge in self.edges):
            return
        self.edges.append(AssemblyEdge(int(source), int(target), str(kind), str(contract_id), direction))

    def connect(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        snapshot = self.clone()
        source_unit_id = str(action.get("source_unit_id", ""))
        target_unit_id = str(action.get("target_unit_id", ""))
        source_node_id = int(action.get("source_node_id", -1))
        target_node_id = int(action.get("target_node_id", -1))
        if source_unit_id not in self._units or target_unit_id not in self._units:
            return False, "unknown_unit"
        if source_unit_id == target_unit_id:
            return False, "same_unit_connection"
        source_port_id = str(action.get("source_port_id", ""))
        target_port_id = str(action.get("target_port_id", ""))
        source_key = f"{source_unit_id}:{source_port_id}"
        target_key = f"{target_unit_id}:{target_port_id}"
        source_port = self._ports.get(source_key)
        target_port = self._ports.get(target_key)
        if source_port is None or target_port is None:
            return False, "unknown_port"
        if int(source_port.node_id) != source_node_id or int(target_port.node_id) != target_node_id:
            return False, "port_node_mismatch"
        if source_key in self.used_ports or target_key in self.used_ports:
            return False, "port_already_used"
        if not self.is_placed(source_unit_id):
            if not self.place_unit(source_unit_id, int(action.get("source_rotation", action.get("rotation", 0))), tuple(action.get("source_translation", action.get("translation", (0, 0))))):
                self._restore(snapshot)
                return False, "source_placement_invalid"
        if not self.is_placed(target_unit_id):
            if not self.place_unit(target_unit_id, int(action.get("target_rotation", 0)), tuple(action.get("target_translation", (0, 0)))):
                self._restore(snapshot)
                return False, "target_placement_invalid"
        if source_node_id not in self.node_axial or target_node_id not in self.node_axial:
            self._restore(snapshot)
            return False, "endpoint_not_placed"
        direction = self._edge_direction(self.node_axial[source_node_id], self.node_axial[target_node_id])
        if direction is None:
            self._restore(snapshot)
            return False, "endpoints_not_hex_neighbors"
        contract_id = str(action.get("contract_id", ""))
        kind = str(action.get("kind", self.stage))
        self._append_edge(source_node_id, target_node_id, kind, contract_id, direction)
        self.used_ports.update({source_key, target_key})
        if contract_id:
            self.satisfied_contracts.add(contract_id)
        self.action_history.append(copy.deepcopy(action))
        return True, ""

    def _restore(self, snapshot: "HexAssemblyState") -> None:
        self.placements = dict(snapshot.placements)
        self.node_axial = dict(snapshot.node_axial)
        self.occupied_axial = dict(snapshot.occupied_axial)
        self.edges = list(snapshot.edges)
        self.used_ports = set(snapshot.used_ports)
        self.satisfied_contracts = set(snapshot.satisfied_contracts)
        self.stage = str(snapshot.stage)
        self.action_history = copy.deepcopy(snapshot.action_history)

    def component_count(self) -> int:
        parent: Dict[int, int] = {int(node_id): int(node_id) for node_id in self.node_axial}
        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value
        for edge in self.edges:
            if edge.source_node_id not in parent or edge.target_node_id not in parent:
                continue
            left, right = find(int(edge.source_node_id)), find(int(edge.target_node_id))
            if left != right:
                parent[right] = left
        return len({find(value) for value in parent}) if parent else 0

    def mandatory_contracts(self, phase: Optional[str] = None) -> List[ExternalContract]:
        values = [contract for contract in self.problem.contracts if contract.mandatory and contract.contract_id not in self.satisfied_contracts]
        if phase is not None:
            values = [contract for contract in values if contract.kind == str(phase)]
        return values

    def phase_done(self, phase: str) -> bool:
        unit_phase = "aromatic" if str(phase) in {"aromatic", "rigid"} else str(phase)
        if any(not self.is_placed(unit.unit_id) for unit in self.problem.units if unit.phase == unit_phase):
            return False
        return not self.mandatory_contracts(phase)

    def is_complete(self) -> bool:
        return (
            all(self.is_placed(unit.unit_id) for unit in self.problem.units)
            and not self.mandatory_contracts()
            and len(self.node_axial) == len(self.problem.ownership)
            and self.component_count() <= 1
        )

    def state_signature(self) -> Tuple[Any, ...]:
        return (
            str(self.stage),
            tuple(sorted((unit_id, placement.rotation, placement.translation) for unit_id, placement in self.placements.items())),
            tuple(sorted((node_id, axial) for node_id, axial in self.node_axial.items())),
            tuple(sorted((edge.source_node_id, edge.target_node_id, edge.kind, edge.contract_id) for edge in self.edges)),
            tuple(sorted(self.used_ports)),
            tuple(sorted(self.satisfied_contracts)),
        )

    def audit(self) -> Dict[str, Any]:
        expected_ids = {int(node_id) for node_id, _unit_id in self.problem.ownership}
        placed_ids = set(int(node_id) for node_id in self.node_axial)
        duplicate_axials = {
            str(axial): sorted(node_id for node_id, value in self.node_axial.items() if value == axial)
            for axial in set(self.node_axial.values())
            if sum(1 for value in self.node_axial.values() if value == axial) > 1
        }
        return {
            "complete": bool(self.is_complete()),
            "stage": str(self.stage),
            "placed_unit_count": len(self.placements),
            "unit_count": len(self.problem.units),
            "placed_node_count": len(placed_ids),
            "expected_node_count": len(expected_ids),
            "missing_node_ids": sorted(expected_ids - placed_ids),
            "unexpected_node_ids": sorted(placed_ids - expected_ids),
            "pending_contract_ids": sorted(contract.contract_id for contract in self.mandatory_contracts()),
            "duplicate_axials": duplicate_axials,
            "component_count": self.component_count(),
            "edge_count": len(self.edges),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": {
                unit_id: {
                    "rotation": int(placement.rotation),
                    "translation": [int(placement.translation[0]), int(placement.translation[1])],
                }
                for unit_id, placement in sorted(self.placements.items())
            },
            "nodes": [
                {"node_id": int(node_id), "axial": [int(axial[0]), int(axial[1])], "unit_id": self.problem.ownership_map().get(int(node_id), "")}
                for node_id, axial in sorted(self.node_axial.items())
            ],
            "edges": [
                {
                    "source_node_id": int(edge.source_node_id),
                    "target_node_id": int(edge.target_node_id),
                    "kind": str(edge.kind),
                    "contract_id": str(edge.contract_id),
                    "direction": list(edge.direction) if edge.direction is not None else None,
                }
                for edge in self.edges
            ],
            "audit": self.audit(),
        }
