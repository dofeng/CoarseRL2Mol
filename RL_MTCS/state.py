"""Hexagonal placement state for immutable inverse topology units."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .geometry import (
    Axial,
    HEX_DIRECTIONS,
    axial_add,
    axial_distance,
    axial_neg,
    reflect_axial,
    rotate_axial,
    transform_axial,
    transform_direction,
    aromatic_slot_direction,
    coordinate_aspect,
    state_layout_metrics,
    unit_sort_key,
    valid_hex_directions,
)
from .models import (
    ExternalContract,
    InverseAssemblyInput,
    InverseInputError,
    MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES,
    TopologyUnit,
    UnitEdge,
    UnitPort,
)


def _allowed_contact_pairs(
    allowed_contact: Any = None,
    allowed_contacts: Optional[Iterable[Tuple[int, int]]] = None,
) -> Set[frozenset[int]]:
    pairs: Set[frozenset[int]] = set()
    if allowed_contact is not None:
        values = tuple(allowed_contact)
        if len(values) == 2 and all(isinstance(value, int) for value in values):
            pairs.add(frozenset(int(value) for value in values))
        else:
            for pair in values:
                pairs.add(frozenset(int(value) for value in pair))
    for pair in allowed_contacts or ():
        pairs.add(frozenset(int(value) for value in pair))
    return pairs


@dataclass(frozen=True)
class UnitPlacement:
    unit_id: str
    rotation: int
    translation: Axial
    reflection: bool = False


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
    # Final aromatic substitutions may place a merged atom at a half-grid
    # coordinate.  Keep the integer staging coordinates above for placement
    # and store the final continuous coordinates separately.
    final_node_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    occupied_axial: Dict[Axial, int] = field(default_factory=dict)
    edges: List[AssemblyEdge] = field(default_factory=list)
    used_ports: Set[str] = field(default_factory=set)
    satisfied_contracts: Set[str] = field(default_factory=set)
    allowed_unbonded_contacts: Set[frozenset[int]] = field(default_factory=set)
    reembedded_units: Set[str] = field(default_factory=set)
    current_su_types: Dict[int, int] = field(default_factory=dict)
    stage: str = "aromatic"
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    substitution_relaxed_edges: Set[frozenset[int]] = field(default_factory=set)
    substitution_relaxed_contracts: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._units = {unit.unit_id: unit for unit in self.problem.units}
        self._ports = {
            f"{unit.unit_id}:{port.port_id}": port
            for unit in self.problem.units
            for port in unit.ports
        }
        if not self.current_su_types:
            self.current_su_types = {
                int(node.node_id): int(node.su_type)
                for unit in self.problem.units
                for node in unit.nodes
            }
        if not self.final_node_positions:
            self.final_node_positions = {
                int(node_id): (float(axial[0]), float(axial[1]))
                for node_id, axial in self.node_axial.items()
            }

    def clone(self) -> "HexAssemblyState":
        new = HexAssemblyState(problem=self.problem)
        new.placements = dict(self.placements)
        new.node_axial = dict(self.node_axial)
        new.final_node_positions = dict(self.final_node_positions)
        new.occupied_axial = dict(self.occupied_axial)
        new.edges = list(self.edges)
        new.used_ports = set(self.used_ports)
        new.satisfied_contracts = set(self.satisfied_contracts)
        new.allowed_unbonded_contacts = set(self.allowed_unbonded_contacts)
        new.reembedded_units = set(self.reembedded_units)
        new.current_su_types = dict(self.current_su_types)
        new.stage = str(self.stage)
        new.action_history = copy.deepcopy(self.action_history)
        new.substitution_relaxed_edges = set(self.substitution_relaxed_edges)
        new.substitution_relaxed_contracts = set(self.substitution_relaxed_contracts)
        return new

    def component_unit_ids(self, unit_id: str) -> Set[str]:
        """Return placed units connected to ``unit_id`` by assembly edges."""
        unit_id = str(unit_id)
        if unit_id not in self.placements:
            return set()
        ownership = self.problem.ownership_map()
        adjacency: Dict[str, Set[str]] = {
            placed_id: set() for placed_id in self.placements
        }
        for edge in self.edges:
            left = ownership.get(int(edge.source_node_id), "")
            right = ownership.get(int(edge.target_node_id), "")
            if left in adjacency and right in adjacency and left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
        seen = {unit_id}
        stack = [unit_id]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return seen

    def component_node_ids(self, unit_id: str) -> Set[int]:
        ownership = self.problem.ownership_map()
        unit_ids = self.component_unit_ids(unit_id)
        return {
            int(node_id)
            for node_id in self.node_axial
            if ownership.get(int(node_id), "") in unit_ids
        }

    def can_transform_component(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        allowed_contact: Optional[Tuple[int, int]] = None,
        allowed_contacts: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
        allow_unbonded_neighbors: bool = False,
    ) -> Tuple[bool, str]:
        component_nodes = self.component_node_ids(unit_id)
        if not component_nodes:
            return False, "component_not_placed"
        transformed = {
            node_id: axial_add(
                transform_axial(self.node_axial[node_id], rotation, reflection),
                translation,
            )
            for node_id in component_nodes
        }
        if len(set(transformed.values())) != len(transformed):
            return False, "component_internal_coordinate_collision"
        geometry_edges = [
            UnitEdge(int(edge.source_node_id), int(edge.target_node_id), str(edge.kind))
            for edge in self.edges
            if int(edge.source_node_id) in component_nodes
            and int(edge.target_node_id) in component_nodes
        ]
        if not self._coordinates_geometry_valid(transformed, geometry_edges):
            return False, "component_non_honeycomb_edge"
        stationary = {
            axial: node_id
            for axial, node_id in self.occupied_axial.items()
            if int(node_id) not in component_nodes
        }
        # CSV initialization places every aromatic unit in a temporary
        # staging pose before the rigid pass.  Those disconnected poses are
        # not physical obstacles: only the component on the other side of
        # the contract may constrain a proposed rigid transform.  Build that
        # stationary map from the authoritative node coordinates rather than
        # from ``occupied_axial`` (which may intentionally contain the
        # staging objects for visualization/backward compatibility).
        if str(self.stage) == "rigid" and allowed_contact is not None:
            contact_nodes = {int(value) for value in allowed_contact}
            stationary_contact_nodes = contact_nodes - component_nodes
            if stationary_contact_nodes:
                ownership = self.problem.ownership_map()
                stationary_units = {
                    ownership.get(int(node_id), "")
                    for node_id in stationary_contact_nodes
                    if ownership.get(int(node_id), "")
                }
                relevant_nodes: Set[int] = set()
                for unit_id in stationary_units:
                    relevant_nodes.update(self.component_node_ids(unit_id))
                stationary = {
                    self.node_axial[int(node_id)]: int(node_id)
                    for node_id in relevant_nodes
                    if int(node_id) not in component_nodes
                    and int(node_id) in self.node_axial
                }
        for node_id, axial in transformed.items():
            if axial in stationary:
                return False, f"component_axial_collision:{axial}:{stationary[axial]}:{node_id}"
        allowed_pairs = _allowed_contact_pairs(
            allowed_contact, allowed_contacts
        )
        if not allow_unbonded_neighbors:
            for node_id, axial in transformed.items():
                for direction in HEX_DIRECTIONS:
                    neighbor_axial = axial_add(axial, direction)
                    occupant = stationary.get(neighbor_axial)
                    if (
                        occupant is not None
                        and frozenset((int(node_id), int(occupant)))
                        not in allowed_pairs
                        and self._potential_unintended_contact(
                            int(node_id), axial, int(occupant), neighbor_axial
                        )
                    ):
                        return False, f"component_unintended_contact:{node_id}:{occupant}"
        return True, ""

    def transform_component(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        allowed_contact: Optional[Tuple[int, int]] = None,
        allowed_contacts: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
        allow_unbonded_neighbors: bool = False,
    ) -> bool:
        ok, _reason = self.can_transform_component(
            unit_id,
            rotation,
            translation,
            allowed_contact=allowed_contact,
            allowed_contacts=allowed_contacts,
            reflection=reflection,
            allow_unbonded_neighbors=allow_unbonded_neighbors,
        )
        if not ok:
            return False
        component_units = self.component_unit_ids(unit_id)
        component_nodes = self.component_node_ids(unit_id)
        rigid_relevant_nodes: Optional[Set[int]] = None
        if str(self.stage) == "rigid" and allowed_contact is not None:
            contact_nodes = {int(value) for value in allowed_contact}
            stationary_contact_nodes = contact_nodes - component_nodes
            if stationary_contact_nodes:
                ownership = self.problem.ownership_map()
                stationary_units = {
                    ownership.get(int(node_id), "")
                    for node_id in stationary_contact_nodes
                    if ownership.get(int(node_id), "")
                }
                rigid_relevant_nodes = set(component_nodes)
                for stationary_unit in stationary_units:
                    rigid_relevant_nodes.update(
                        self.component_node_ids(stationary_unit)
                    )
        self.allowed_unbonded_contacts = {
            pair for pair in self.allowed_unbonded_contacts
            if pair.isdisjoint(component_nodes) or pair <= component_nodes
        }
        rotation = int(rotation) % 6
        translation = (int(translation[0]), int(translation[1]))
        for node_id in component_nodes:
            self.occupied_axial.pop(self.node_axial[node_id], None)
        for placed_id in component_units:
            placement = self.placements[placed_id]
            new_translation = axial_add(
                transform_axial(
                    placement.translation, rotation, reflection
                ),
                translation,
            )
            if reflection:
                new_rotation = (rotation - int(placement.rotation)) % 6
                new_reflection = not bool(placement.reflection)
            else:
                new_rotation = (int(placement.rotation) + rotation) % 6
                new_reflection = bool(placement.reflection)
            self.placements[placed_id] = UnitPlacement(
                unit_id=placed_id,
                rotation=new_rotation,
                translation=new_translation,
                reflection=new_reflection,
            )
        for node_id in component_nodes:
            axial = axial_add(
                transform_axial(
                    self.node_axial[node_id], rotation, reflection
                ),
                translation,
            )
            self.node_axial[node_id] = axial
            self.occupied_axial[axial] = node_id
            self.final_node_positions[node_id] = (float(axial[0]), float(axial[1]))
        if rigid_relevant_nodes is not None:
            # Keep only the active contract neighbourhood in the occupancy
            # index.  Other aromatic components remain available through
            # ``node_axial`` but cannot block the next rigid search step.
            self.occupied_axial = {
                self.node_axial[int(node_id)]: int(node_id)
                for node_id in rigid_relevant_nodes
                if int(node_id) in self.node_axial
            }
        self.edges = [
            AssemblyEdge(
                edge.source_node_id,
                edge.target_node_id,
                edge.kind,
                edge.contract_id,
                self._edge_direction(
                    self.node_axial[edge.source_node_id],
                    self.node_axial[edge.target_node_id],
                ),
            )
            for edge in self.edges
        ]
        if allow_unbonded_neighbors:
            self._remember_allowed_unbonded_contacts(component_nodes)
        return True

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

    def global_node_axial(
        self,
        unit_id: str,
        node_id: int,
        rotation: Optional[int] = None,
        translation: Optional[Axial] = None,
        reflection: Optional[bool] = None,
    ) -> Axial:
        unit = self.unit(unit_id)
        node = next((node for node in unit.nodes if int(node.node_id) == int(node_id)), None)
        if node is None:
            raise KeyError(f"unit {unit_id!r} has no node {node_id}")
        placement = self.placements.get(str(unit_id))
        rot = int(rotation if rotation is not None else placement.rotation if placement else 0)
        trans = translation if translation is not None else placement.translation if placement else (0, 0)
        reflected = bool(
            reflection
            if reflection is not None
            else placement.reflection if placement else False
        )
        return axial_add(
            transform_axial(node.local_axial, rot, reflected), trans
        )

    def global_port_direction(
        self,
        unit_id: str,
        port: UnitPort,
        rotation: Optional[int] = None,
        reflection: Optional[bool] = None,
    ) -> Optional[Axial]:
        if port.direction is None:
            return None
        placement = self.placements.get(str(unit_id))
        rot = int(rotation if rotation is not None else placement.rotation if placement else 0)
        reflected = bool(
            reflection
            if reflection is not None
            else placement.reflection if placement else False
        )
        return transform_axial(port.direction, rot, reflected)

    def _placement_coordinates(
        self,
        unit: TopologyUnit,
        rotation: int,
        translation: Axial,
        reflection: bool = False,
    ) -> Dict[int, Axial]:
        return {
            int(node.node_id): axial_add(
                transform_axial(
                    node.local_axial, int(rotation), bool(reflection)
                ),
                translation,
            )
            for node in unit.nodes
        }

    def _node_su_type(self, node_id: int) -> int:
        return int(self.problem.node_map().get(int(node_id), -1))

    def current_node_map(self) -> Dict[int, int]:
        """Return mutable assembly-time SU labels for visualization/export."""
        return dict(self.current_su_types)

    def _activate_aromatic_endpoints(
        self,
        endpoints: Iterable[Tuple[str, int]],
    ) -> None:
        planned = self.problem.node_map()
        for unit_id, node_id in endpoints:
            unit = self._units.get(str(unit_id))
            node_id = int(node_id)
            if unit is not None and unit.phase == "aromatic" and node_id in planned:
                self.current_su_types[node_id] = int(planned[node_id])

    def _edge_geometry_valid(
        self,
        source_node_id: int,
        target_node_id: int,
        source: Axial,
        target: Axial,
    ) -> bool:
        if frozenset((int(source_node_id), int(target_node_id))) in self.substitution_relaxed_edges:
            return True
        delta = (int(target[0]) - int(source[0]), int(target[1]) - int(source[1]))
        if delta not in HEX_DIRECTIONS:
            return False
        # The legacy assembly stages use the six axial neighbour offsets for
        # rigid and inter-unit bonds.  Flexible-chain construction imposes the
        # stricter alternating honeycomb directions when it lays out a chain;
        # do not apply that local path rule to the old rigid matcher.
        return True

    def _potential_unintended_contact(
        self,
        source_node_id: int,
        source: Axial,
        target_node_id: int,
        target: Axial,
    ) -> bool:
        """Distinguish six lattice neighbours from valid honeycomb bonds."""
        delta = (
            int(target[0]) - int(source[0]),
            int(target[1]) - int(source[1]),
        )
        if delta not in HEX_DIRECTIONS:
            return False
        if 25 in {
            self._node_su_type(int(source_node_id)),
            self._node_su_type(int(target_node_id)),
        }:
            return True
        source_directions = valid_hex_directions(source)
        target_directions = valid_hex_directions(target)
        # Phase-0 coordinates belong to the surrounding triangular lattice,
        # not to either honeycomb vertex sublattice.  They may appear in
        # external JSON fixtures, but no ordinary carbon bond can be inferred
        # from adjacency alone at such a coordinate.
        if not source_directions or not target_directions:
            return False
        return (
            delta in source_directions
            and (-delta[0], -delta[1]) in target_directions
        )

    def _remember_allowed_unbonded_contacts(
        self,
        node_ids: Iterable[int],
    ) -> None:
        """Remember close cross-unit contacts accepted by flex placement."""
        ownership = self.problem.ownership_map()
        bonded = {
            frozenset((int(edge.source_node_id), int(edge.target_node_id)))
            for edge in self.edges
        }
        for node_id in {int(value) for value in node_ids}:
            axial = self.node_axial.get(node_id)
            if axial is None:
                continue
            for direction in HEX_DIRECTIONS:
                occupant = self.occupied_axial.get(axial_add(axial, direction))
                if occupant is None or int(occupant) == node_id:
                    continue
                pair = frozenset((node_id, int(occupant)))
                if (
                    pair not in bonded
                    and ownership.get(node_id) != ownership.get(int(occupant))
                    and self._potential_unintended_contact(
                        node_id,
                        axial,
                        int(occupant),
                        self.node_axial[int(occupant)],
                    )
                ):
                    self.allowed_unbonded_contacts.add(pair)

    def _coordinates_geometry_valid(
        self,
        coordinates: Dict[int, Axial],
        edges: Iterable[UnitEdge],
    ) -> bool:
        return all(
            self._edge_geometry_valid(
                int(edge.source_node_id),
                int(edge.target_node_id),
                coordinates[int(edge.source_node_id)],
                coordinates[int(edge.target_node_id)],
            )
            for edge in edges
            if int(edge.source_node_id) in coordinates
            and int(edge.target_node_id) in coordinates
        )

    def can_place(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        allowed_contact: Optional[Tuple[int, int]] = None,
        allowed_contacts: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
        allow_unbonded_neighbors: bool = False,
    ) -> Tuple[bool, str]:
        unit = self.unit(unit_id)
        if self.is_placed(unit_id):
            return False, "unit_already_placed"
        coordinates = self._placement_coordinates(
            unit, rotation, translation, reflection
        )
        if len(set(coordinates.values())) != len(coordinates):
            return False, "unit_internal_coordinate_collision"
        if not self._coordinates_geometry_valid(coordinates, unit.internal_edges):
            return False, "unit_non_honeycomb_edge"
        for node_id, axial in coordinates.items():
            occupant = self.occupied_axial.get(axial)
            if occupant is not None:
                return False, f"axial_collision:{axial}:{occupant}:{node_id}"
        allowed_pairs = _allowed_contact_pairs(
            allowed_contact, allowed_contacts
        )
        if not allow_unbonded_neighbors:
            for node_id, axial in coordinates.items():
                for direction in HEX_DIRECTIONS:
                    neighbor_axial = axial_add(axial, direction)
                    occupant = self.occupied_axial.get(neighbor_axial)
                    if (
                        occupant is not None
                        and frozenset((int(node_id), int(occupant)))
                        not in allowed_pairs
                        and self._potential_unintended_contact(
                            int(node_id), axial, int(occupant), neighbor_axial
                        )
                    ):
                        return False, f"unintended_contact:{node_id}:{occupant}"
        return True, ""

    def place_unit(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        allowed_contact: Optional[Tuple[int, int]] = None,
        allowed_contacts: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
        allow_unbonded_neighbors: bool = False,
    ) -> bool:
        ok, _reason = self.can_place(
            unit_id,
            rotation,
            translation,
            allowed_contact=allowed_contact,
            allowed_contacts=allowed_contacts,
            reflection=reflection,
            allow_unbonded_neighbors=allow_unbonded_neighbors,
        )
        if not ok:
            return False
        unit = self.unit(unit_id)
        placement = UnitPlacement(
            str(unit_id),
            int(rotation) % 6,
            (int(translation[0]), int(translation[1])),
            bool(reflection),
        )
        coordinates = self._placement_coordinates(
            unit,
            placement.rotation,
            placement.translation,
            placement.reflection,
        )
        self.placements[str(unit_id)] = placement
        for node_id, axial in coordinates.items():
            self.node_axial[int(node_id)] = axial
            self.occupied_axial[axial] = int(node_id)
            self.final_node_positions[int(node_id)] = (float(axial[0]), float(axial[1]))
        for edge in unit.internal_edges:
            if int(edge.source_node_id) not in coordinates or int(edge.target_node_id) not in coordinates:
                continue
            direction = self._edge_direction(coordinates[int(edge.source_node_id)], coordinates[int(edge.target_node_id)])
            self._append_edge(int(edge.source_node_id), int(edge.target_node_id), str(edge.kind), direction=direction)
        if allow_unbonded_neighbors:
            self._remember_allowed_unbonded_contacts(coordinates)
        return True

    @staticmethod
    def _edge_direction(source: Axial, target: Axial) -> Optional[Axial]:
        delta = (int(target[0]) - int(source[0]), int(target[1]) - int(source[1]))
        return delta if delta in HEX_DIRECTIONS else None

    def _append_edge(self, source: int, target: int, kind: str, contract_id: str = "", direction: Optional[Axial] = None) -> None:
        self.allowed_unbonded_contacts.discard(
            frozenset((int(source), int(target)))
        )
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
        move_component = str(action.get("move_component_unit_id", ""))
        if move_component:
            if move_component not in self.placements:
                return False, "move_component_not_placed"
            if not self.transform_component(
                move_component,
                int(action.get("component_rotation", 0)),
                tuple(action.get("component_translation", (0, 0))),
                allowed_contact=(source_node_id, target_node_id),
                reflection=bool(action.get("component_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "component_transform_invalid"
        if not self.is_placed(source_unit_id):
            if not self.place_unit(
                source_unit_id,
                int(action.get("source_rotation", action.get("rotation", 0))),
                tuple(action.get("source_translation", action.get("translation", (0, 0)))),
                allowed_contact=(source_node_id, target_node_id),
                reflection=bool(action.get("source_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "source_placement_invalid"
        if not self.is_placed(target_unit_id):
            if not self.place_unit(
                target_unit_id,
                int(action.get("target_rotation", 0)),
                tuple(action.get("target_translation", (0, 0))),
                allowed_contact=(source_node_id, target_node_id),
                reflection=bool(action.get("target_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "target_placement_invalid"
        if source_node_id not in self.node_axial or target_node_id not in self.node_axial:
            self._restore(snapshot)
            return False, "endpoint_not_placed"
        direction = self._edge_direction(self.node_axial[source_node_id], self.node_axial[target_node_id])
        if direction is None or not self._edge_geometry_valid(
            source_node_id,
            target_node_id,
            self.node_axial[source_node_id],
            self.node_axial[target_node_id],
        ):
            self._restore(snapshot)
            return False, "endpoints_not_honeycomb_neighbors"
        contract_id = str(action.get("contract_id", ""))
        kind = str(action.get("kind", self.stage))
        self._append_edge(source_node_id, target_node_id, kind, contract_id, direction)
        self.used_ports.update({source_key, target_key})
        if contract_id:
            self.satisfied_contracts.add(contract_id)
        self._activate_aromatic_endpoints((
            (source_unit_id, source_node_id),
            (target_unit_id, target_node_id),
        ))
        self.action_history.append(copy.deepcopy(action))
        return True, ""

    def force_connect_contract(
        self,
        contract: ExternalContract,
        reason: str,
        *,
        geometry_fallback: bool = True,
    ) -> bool:
        """Connect an already adjacent contract without inventing a long edge."""
        source_key = f"{contract.source_unit_id}:{contract.source_port_id}"
        target_key = f"{contract.target_unit_id}:{contract.target_port_id}"
        if source_key in self.used_ports or target_key in self.used_ports:
            return False
        source_axial = self.node_axial.get(int(contract.source_node_id))
        target_axial = self.node_axial.get(int(contract.target_node_id))
        direction = (
            self._edge_direction(source_axial, target_axial)
            if source_axial is not None and target_axial is not None
            else None
        )
        if (
            source_axial is None
            or target_axial is None
            or direction is None
            or not self._edge_geometry_valid(
                int(contract.source_node_id),
                int(contract.target_node_id),
                source_axial,
                target_axial,
            )
        ):
            return False
        self._append_edge(
            int(contract.source_node_id),
            int(contract.target_node_id),
            str(contract.kind),
            str(contract.contract_id),
            direction,
        )
        self.used_ports.update({source_key, target_key})
        self.satisfied_contracts.add(str(contract.contract_id))
        self._activate_aromatic_endpoints((
            (str(contract.source_unit_id), int(contract.source_node_id)),
            (str(contract.target_unit_id), int(contract.target_node_id)),
        ))
        self.action_history.append({
            "kind": str(contract.kind),
            "contract_id": str(contract.contract_id),
            "geometry_fallback": bool(geometry_fallback),
            "planned_internal_closure": not bool(geometry_fallback),
            "reason": str(reason),
        })
        return True

    def reembed_topology_for_contract(
        self,
        contract: ExternalContract,
        max_attempts: int = 50000,
        search_variant: int = 0,
        allow_unbonded_neighbors: bool = False,
        solution_index: int = 0,
    ) -> bool:
        """Find a collision-free conformer while keeping satisfied ports fixed.

        The inverse product is a complete tree.  Re-embedding changes only its
        axial conformer; node IDs, SU types, internal edges and port ownership
        remain immutable.
        """
        candidates = [
            unit_id
            for unit_id in (contract.source_unit_id, contract.target_unit_id)
            if self.unit(unit_id).phase != "aromatic" and self.is_placed(unit_id)
        ]
        if not candidates:
            return False
        unit_id = str(candidates[0])
        unit = self.unit(unit_id)
        node_ids = {int(node.node_id) for node in unit.nodes}
        adjacency: Dict[int, Set[int]] = {node_id: set() for node_id in node_ids}
        for edge in unit.internal_edges:
            left, right = int(edge.source_node_id), int(edge.target_node_id)
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        if len(unit.nodes) > 1 and sum(len(values) for values in adjacency.values()) // 2 != len(unit.nodes) - 1:
            return False

        anchored: Dict[int, Set[Axial]] = {}
        allowed_external: Dict[int, Set[int]] = {}
        relevant_ids = set(self.satisfied_contracts) | {str(contract.contract_id)}
        for item in self.problem.contracts:
            if str(item.contract_id) not in relevant_ids:
                continue
            if item.source_unit_id == unit_id:
                local_node = int(item.source_node_id)
                external_node = int(item.target_node_id)
            elif item.target_unit_id == unit_id:
                local_node = int(item.target_node_id)
                external_node = int(item.source_node_id)
            else:
                continue
            external_axial = self.node_axial.get(external_node)
            if local_node not in node_ids or external_axial is None:
                return False
            allowed = {
                axial_add(external_axial, direction)
                for direction in HEX_DIRECTIONS
            }
            anchored[local_node] = allowed if local_node not in anchored else anchored[local_node] & allowed
            allowed_external.setdefault(local_node, set()).add(external_node)
            if not anchored[local_node]:
                return False
        if not anchored:
            return False

        stationary = {
            axial: int(node_id)
            for axial, node_id in self.occupied_axial.items()
            if int(node_id) not in node_ids
        }
        root = min(anchored, key=lambda node_id: (len(anchored[node_id]), -len(adjacency[node_id]), node_id))

        # Tree distances provide a strong endpoint-closure bound during DFS.
        distances: Dict[Tuple[int, int], int] = {}
        for start in node_ids:
            queue = [(start, 0)]
            seen = {start}
            for current, depth in queue:
                distances[(start, current)] = depth
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, depth + 1))

        assigned: Dict[int, Axial] = {}
        used: Set[Axial] = set()
        attempts = 0
        solutions_seen = 0

        def ordered_coordinates(values: Iterable[Axial]) -> List[Axial]:
            ordered = sorted(
                values,
                key=lambda value: (
                    abs(value[0]) + abs(value[1]), value[0], value[1]
                ),
            )
            if not ordered:
                return ordered
            variant = max(0, int(search_variant))
            if (variant // len(ordered)) % 2:
                ordered.reverse()
            shift = variant % len(ordered)
            return ordered[shift:] + ordered[:shift]

        def closure_possible() -> bool:
            for placed_node, placed_axial in assigned.items():
                for anchor_node, allowed in anchored.items():
                    if anchor_node in assigned:
                        continue
                    steps = distances.get((placed_node, anchor_node), len(node_ids) + 1)
                    if min(axial_distance(placed_axial, value) for value in allowed) > steps:
                        return False
            return True

        def choose_frontier() -> Optional[int]:
            values = [
                node_id for node_id in node_ids - set(assigned)
                if any(neighbor in assigned for neighbor in adjacency[node_id])
            ]
            if not values:
                return None
            return min(values, key=lambda node_id: (node_id not in anchored, -len(adjacency[node_id]), node_id))

        def place_remaining() -> bool:
            nonlocal attempts, solutions_seen
            if len(assigned) == len(node_ids):
                if solutions_seen >= max(0, int(solution_index)):
                    return True
                solutions_seen += 1
                return False
            node_id = choose_frontier()
            if node_id is None:
                return False
            assigned_neighbors = [neighbor for neighbor in adjacency[node_id] if neighbor in assigned]
            possible: Optional[Set[Axial]] = None
            for neighbor in assigned_neighbors:
                around = {axial_add(assigned[neighbor], direction) for direction in HEX_DIRECTIONS}
                possible = around if possible is None else possible & around
            possible = possible or set()
            if node_id in anchored:
                possible &= anchored[node_id]
            for coord in ordered_coordinates(possible):
                attempts += 1
                if attempts > int(max_attempts):
                    return False
                if coord in used or coord in stationary:
                    continue
                if not allow_unbonded_neighbors:
                    contacts = set()
                    for direction in HEX_DIRECTIONS:
                        neighbor_coord = axial_add(coord, direction)
                        occupant = stationary.get(neighbor_coord)
                        if occupant is not None and self._potential_unintended_contact(
                            int(node_id), coord, int(occupant), neighbor_coord
                        ):
                            contacts.add(int(occupant))
                    if contacts - allowed_external.get(node_id, set()):
                        continue
                assigned[node_id] = coord
                used.add(coord)
                if closure_possible() and place_remaining():
                    return True
                used.remove(coord)
                assigned.pop(node_id, None)
            return False

        for root_coord in ordered_coordinates(anchored[root]):
            if root_coord in stationary:
                continue
            assigned.clear()
            used.clear()
            assigned[root] = root_coord
            used.add(root_coord)
            if not closure_possible() or not place_remaining():
                continue
            for node_id in node_ids:
                old_axial = self.node_axial[node_id]
                if self.occupied_axial.get(old_axial) == node_id:
                    self.occupied_axial.pop(old_axial, None)
            for node_id, coord in assigned.items():
                self.node_axial[node_id] = coord
                self.occupied_axial[coord] = node_id
                self.final_node_positions[node_id] = (float(coord[0]), float(coord[1]))
            self.edges = [
                AssemblyEdge(
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.kind,
                    edge.contract_id,
                    self._edge_direction(
                        self.node_axial[edge.source_node_id],
                        self.node_axial[edge.target_node_id],
                    ),
                )
                for edge in self.edges
            ]
            self.reembedded_units.add(unit_id)
            self.action_history.append({
                "kind": str(contract.kind),
                "contract_id": str(contract.contract_id),
                "unit_id": unit_id,
                "topology_reembedded": True,
                "attempts": attempts,
                "solution_index": int(solution_index),
            })
            return True
        return False

    def _restore(self, snapshot: "HexAssemblyState") -> None:
        self.placements = dict(snapshot.placements)
        self.node_axial = dict(snapshot.node_axial)
        self.final_node_positions = dict(snapshot.final_node_positions)
        self.occupied_axial = dict(snapshot.occupied_axial)
        self.edges = list(snapshot.edges)
        self.used_ports = set(snapshot.used_ports)
        self.satisfied_contracts = set(snapshot.satisfied_contracts)
        self.allowed_unbonded_contacts = set(
            snapshot.allowed_unbonded_contacts
        )
        self.reembedded_units = set(snapshot.reembedded_units)
        self.current_su_types = dict(snapshot.current_su_types)
        self.stage = str(snapshot.stage)
        self.action_history = copy.deepcopy(snapshot.action_history)
        self.substitution_relaxed_edges = set(snapshot.substitution_relaxed_edges)
        self.substitution_relaxed_contracts = set(snapshot.substitution_relaxed_contracts)

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

    def audit_resource_partition_errors(self) -> List[Dict[str, Any]]:
        assembled = self.problem.node_map()
        ownership = self.problem.ownership_map()
        resources = tuple(self.problem.unassembled_closure_resources)
        errors: List[Dict[str, Any]] = []
        if set(assembled) != set(ownership):
            errors.append({
                "reason": "assembled_node_ownership_mismatch",
                "missing_ownership_ids": sorted(set(assembled) - set(ownership)),
                "unexpected_ownership_ids": sorted(set(ownership) - set(assembled)),
            })
        resource_ids = [int(resource.node_id) for resource in resources]
        if len(resource_ids) != len(set(resource_ids)):
            errors.append({
                "reason": "duplicate_unassembled_resource_id",
                "node_ids": sorted(resource_ids),
            })
        overlap = sorted(set(resource_ids) & set(assembled))
        if overlap:
            errors.append({
                "reason": "unassembled_resource_owned_by_graph",
                "node_ids": overlap,
            })
        if len(resources) > MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES:
            errors.append({
                "reason": "unassembled_resource_limit_exceeded",
                "count": len(resources),
                "limit": MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
            })
        for resource in resources:
            if str(resource.state) != "remaining_closure_resource":
                errors.append({
                    "reason": "unassembled_resource_state_invalid",
                    "node_id": int(resource.node_id),
                    "state": str(resource.state),
                })
            if int(resource.su_type) not in TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES:
                errors.append({
                    "reason": "unassembled_resource_su_invalid",
                    "node_id": int(resource.node_id),
                    "su_type": int(resource.su_type),
                })
        return errors

    def is_complete(self) -> bool:
        return (
            all(self.is_placed(unit.unit_id) for unit in self.problem.units)
            and not self.mandatory_contracts()
            and len(self.node_axial) == len(self.problem.ownership)
            and self.component_count() <= 1
            and len(set(self.node_axial.values())) == len(self.node_axial)
            and not self.audit_geometry_errors()
            and not self.audit_topology_conformation_errors()
            and not self.audit_port_direction_errors()
            and not self.audit_unintended_contacts()
            and not self.audit_substitution_errors()
            and not self.audit_resource_partition_errors()
        )

    def audit_substitution_errors(self) -> List[Dict[str, Any]]:
        """Validate the bookkeeping of the final SU26/SU30 pass.

        Five-membered replacements deliberately keep an integer staging
        coordinate for lattice audits and expose the physical mean coordinate
        through ``final_node_positions``.  This audit verifies that the two
        source sites were actually collapsed, the target label is present, and
        no final coordinates collide.
        """
        errors: List[Dict[str, Any]] = []
        current = self.current_su_types
        for action in self.action_history:
            if action.get("kind") != "substitution":
                continue
            target_su = int(action.get("target_su", -1))
            ring_size = int(action.get("ring_size", 0))
            if ring_size == 6:
                site_ids = [int(value) for value in action.get("site_ids", [])]
                if len(site_ids) != 1 or site_ids[0] not in self.node_axial:
                    errors.append({"reason": "six_member_site_missing", "site_ids": site_ids})
                elif int(current.get(site_ids[0], -1)) != target_su:
                    errors.append({"reason": "six_member_label_mismatch", "site_id": site_ids[0], "expected": target_su, "actual": current.get(site_ids[0])})
                continue
            if ring_size != 5:
                continue
            source_ids = [int(value) for value in action.get("site_ids", [])]
            representative = int(action.get("representative_node_id", -1))
            removed = int(action.get("removed_node_id", -1))
            if len(source_ids) != 2 or representative not in source_ids or removed not in source_ids:
                errors.append({"reason": "five_member_source_metadata_invalid", "site_ids": source_ids})
                continue
            if representative not in self.node_axial:
                errors.append({"reason": "five_member_representative_missing", "node_id": representative})
            if removed in self.node_axial or removed in current:
                errors.append({"reason": "five_member_removed_site_present", "node_id": removed})
            if int(current.get(representative, -1)) != target_su:
                errors.append({"reason": "five_member_label_mismatch", "node_id": representative, "expected": target_su, "actual": current.get(representative)})
            expected = action.get("mean_position")
            actual = self.final_node_positions.get(representative)
            if expected is None or actual is None or len(expected) < 2 or any(abs(float(expected[i]) - float(actual[i])) > 1e-9 for i in (0, 1)):
                errors.append({"reason": "five_member_mean_position_mismatch", "node_id": representative, "expected": expected, "actual": actual})
            source_positions = action.get("source_positions")
            if isinstance(source_positions, (list, tuple)) and len(source_positions) == 2 and all(isinstance(value, (list, tuple)) and len(value) >= 2 for value in source_positions):
                mean = (
                    (float(source_positions[0][0]) + float(source_positions[1][0])) / 2.0,
                    (float(source_positions[0][1]) + float(source_positions[1][1])) / 2.0,
                )
                if actual is None or any(abs(mean[i] - float(actual[i])) > 1e-9 for i in (0, 1)):
                    errors.append({"reason": "five_member_source_mean_mismatch", "node_id": representative, "expected": list(mean), "actual": actual})
            bonded = {
                frozenset((int(edge.source_node_id), int(edge.target_node_id)))
                for edge in self.edges
            }
            if frozenset(source_ids) in bonded:
                errors.append({"reason": "five_member_source_edge_present", "site_ids": source_ids})
        positions: Dict[Tuple[float, float], List[int]] = {}
        for node_id in self.node_axial:
            position = self.final_node_positions.get(int(node_id))
            if position is None:
                continue
            key = (round(float(position[0]), 9), round(float(position[1]), 9))
            positions.setdefault(key, []).append(int(node_id))
        for position, node_ids in positions.items():
            if len(node_ids) > 1:
                errors.append({"reason": "duplicate_final_positions", "position": list(position), "node_ids": sorted(node_ids)})
        for pair in self.substitution_relaxed_edges:
            if any(int(node_id) not in self.node_axial for node_id in pair):
                errors.append({"reason": "relaxed_edge_endpoint_missing", "edge": sorted(int(node_id) for node_id in pair)})
        return errors

    def audit_geometry_errors(self) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for edge in self.edges:
            source = self.node_axial.get(int(edge.source_node_id))
            target = self.node_axial.get(int(edge.target_node_id))
            if source is None or target is None:
                continue
            if not self._edge_geometry_valid(
                int(edge.source_node_id), int(edge.target_node_id), source, target
            ):
                errors.append({
                    "source_node_id": int(edge.source_node_id),
                    "target_node_id": int(edge.target_node_id),
                    "kind": str(edge.kind),
                    "contract_id": str(edge.contract_id),
                })
        return errors

    def audit_topology_conformation_errors(self) -> List[Dict[str, Any]]:
        """Verify that every placed unit is still its immutable local shape."""
        errors: List[Dict[str, Any]] = []
        for unit_id, placement in sorted(self.placements.items()):
            unit = self.unit(unit_id)
            if unit_id in self.reembedded_units:
                errors.append({
                    "unit_id": str(unit_id),
                    "reason": "topology_reembedded",
                })
            for node in unit.nodes:
                node_id = int(node.node_id)
                expected = axial_add(
                    transform_axial(
                        node.local_axial,
                        placement.rotation,
                        placement.reflection,
                    ),
                    placement.translation,
                )
                actual = self.node_axial.get(node_id)
                if actual != expected:
                    errors.append({
                        "unit_id": str(unit_id),
                        "node_id": node_id,
                        "reason": "local_geometry_changed",
                        "expected_axial": list(expected),
                        "actual_axial": list(actual) if actual is not None else None,
                    })
        return errors

    def audit_port_direction_errors(self) -> List[Dict[str, Any]]:
        """Check every satisfied contract against both transformed ports."""
        errors: List[Dict[str, Any]] = []
        for contract in self.problem.contracts:
            if str(contract.contract_id) not in self.satisfied_contracts:
                continue
            if str(contract.contract_id) in self.substitution_relaxed_contracts:
                continue
            source = self.node_axial.get(int(contract.source_node_id))
            target = self.node_axial.get(int(contract.target_node_id))
            if source is None or target is None:
                errors.append({
                    "contract_id": str(contract.contract_id),
                    "reason": "satisfied_contract_endpoint_missing",
                })
                continue
            direction = self._edge_direction(source, target)
            if direction is None:
                errors.append({
                    "contract_id": str(contract.contract_id),
                    "reason": "satisfied_contract_not_adjacent",
                })
                continue
            for unit_id, port_id, expected in (
                (
                    str(contract.source_unit_id),
                    str(contract.source_port_id),
                    direction,
                ),
                (
                    str(contract.target_unit_id),
                    str(contract.target_port_id),
                    axial_neg(direction),
                ),
            ):
                port = self._ports.get(f"{unit_id}:{port_id}")
                if port is None:
                    errors.append({
                        "contract_id": str(contract.contract_id),
                        "unit_id": unit_id,
                        "reason": "contract_port_missing",
                    })
                    continue
                actual = self.global_port_direction(unit_id, port)
                if actual is not None and actual != expected:
                    errors.append({
                        "contract_id": str(contract.contract_id),
                        "unit_id": unit_id,
                        "port_id": port_id,
                        "reason": "port_direction_mismatch",
                        "expected_direction": list(expected),
                        "actual_direction": list(actual),
                    })
        return errors

    def audit_unintended_contacts(self) -> List[Dict[str, Any]]:
        """Report honeycomb-neighbour contacts which have no graph edge."""
        ownership = self.problem.ownership_map()
        bonded = {
            frozenset((int(edge.source_node_id), int(edge.target_node_id)))
            for edge in self.edges
        }
        errors: List[Dict[str, Any]] = []
        seen: Set[frozenset[int]] = set()
        for source_id, source in sorted(self.node_axial.items()):
            for direction in HEX_DIRECTIONS:
                target = axial_add(source, direction)
                target_id = self.occupied_axial.get(target)
                if target_id is None or int(target_id) == int(source_id):
                    continue
                pair = frozenset((int(source_id), int(target_id)))
                if (
                    pair in seen
                    or pair in bonded
                    or pair in self.allowed_unbonded_contacts
                ):
                    continue
                seen.add(pair)
                # A topology unit owns its complete canonical projection.
                # Collision detection applies between independently placed
                # units; it must not reinterpret a fixed unit's non-edge
                # proximity as a new bond.
                if ownership.get(int(source_id)) == ownership.get(int(target_id)):
                    continue
                if self._potential_unintended_contact(
                    int(source_id), source, int(target_id), target
                ):
                    errors.append({
                        "source_node_id": min(pair),
                        "target_node_id": max(pair),
                        "reason": "unbonded_honeycomb_neighbors",
                    })
        return errors

    def state_signature(self) -> Tuple[Any, ...]:
        return (
            str(self.stage),
            tuple(sorted((
                unit_id,
                placement.rotation,
                placement.translation,
                placement.reflection,
            ) for unit_id, placement in self.placements.items())),
            tuple(sorted((node_id, axial) for node_id, axial in self.node_axial.items())),
            tuple(sorted((node_id, position) for node_id, position in self.final_node_positions.items())),
            tuple(sorted((edge.source_node_id, edge.target_node_id, edge.kind, edge.contract_id) for edge in self.edges)),
            tuple(sorted(self.used_ports)),
            tuple(sorted(self.satisfied_contracts)),
            tuple(sorted(
                tuple(sorted(pair)) for pair in self.allowed_unbonded_contacts
            )),
            tuple(sorted(self.reembedded_units)),
            tuple(sorted(self.current_su_types.items())),
            tuple(sorted(tuple(sorted(pair)) for pair in self.substitution_relaxed_edges)),
            tuple(sorted(self.substitution_relaxed_contracts)),
        )

    def audit(self) -> Dict[str, Any]:
        expected_ids = {int(node_id) for node_id, _unit_id in self.problem.ownership}
        placed_ids = set(int(node_id) for node_id in self.node_axial)
        duplicate_axials = {
            str(axial): sorted(node_id for node_id, value in self.node_axial.items() if value == axial)
            for axial in set(self.node_axial.values())
            if sum(1 for value in self.node_axial.values() if value == axial) > 1
        }
        non_hex_edges = self.audit_geometry_errors()
        conformation_errors = self.audit_topology_conformation_errors()
        port_direction_errors = self.audit_port_direction_errors()
        unintended_contacts = self.audit_unintended_contacts()
        substitution_errors = self.audit_substitution_errors()
        resource_partition_errors = self.audit_resource_partition_errors()
        unassembled_resources = tuple(
            self.problem.unassembled_closure_resources
        )
        # Audit the complete assembled molecule, including staging nodes that
        # may be hidden from the active collision index during flex search.
        shape_metrics = state_layout_metrics(self, full=True, include_all=True)
        shape_warnings = []
        if float(shape_metrics["principal_axis_ratio"]) > 2.5:
            shape_warnings.append("excessive_principal_axis_ratio")
        if float(shape_metrics.get("largest_void_radius", 0.0)) > 8.0:
            shape_warnings.append("large_internal_void")
        if float(shape_metrics.get("unit_density_cv", 0.0)) > 0.60:
            shape_warnings.append("uneven_unit_density")
        if float(shape_metrics.get("hull_fill", 1.0)) < 0.30:
            shape_warnings.append("low_convex_hull_fill")

        ownership = self.problem.ownership_map()
        unit_phase = {
            str(unit.unit_id): str(unit.phase) for unit in self.problem.units
        }
        side_unit_loads: Counter[str] = Counter()
        for edge in self.edges:
            if str(edge.kind) != "side":
                continue
            left = ownership.get(int(edge.source_node_id), "")
            right = ownership.get(int(edge.target_node_id), "")
            if left == right:
                continue
            if unit_phase.get(left) == "aromatic" and unit_phase.get(right) == "side":
                side_unit_loads[left] += 1
            elif unit_phase.get(right) == "aromatic" and unit_phase.get(left) == "side":
                side_unit_loads[right] += 1
        side_attachment_count = sum(side_unit_loads.values())
        side_coverage_ratio = (
            len(side_unit_loads) / side_attachment_count
            if side_attachment_count else 1.0
        )
        side_distribution_warnings = []
        if max(side_unit_loads.values(), default=0) > 2:
            side_distribution_warnings.append("side_unit_overloaded")
        if side_attachment_count >= 4 and side_coverage_ratio < 0.65:
            side_distribution_warnings.append("side_attachments_clustered")
        return {
            "complete": bool(self.is_complete()),
            "stage": str(self.stage),
            "placed_unit_count": len(self.placements),
            "unit_count": len(self.problem.units),
            "placed_node_count": len(placed_ids),
            "expected_node_count": len(expected_ids),
            "source_node_count": int(dict(self.problem.diagnostics).get(
                "source_node_count",
                len(expected_ids) + len(unassembled_resources),
            )),
            "unassembled_resource_count": len(unassembled_resources),
            "unassembled_resource_ids": sorted(
                int(resource.node_id) for resource in unassembled_resources
            ),
            "strict_all_nodes_assembled": not bool(unassembled_resources),
            "completion_mode": (
                "tolerated_unassembled_closure_resources"
                if unassembled_resources else "strict_all_nodes_assembled"
            ),
            "resource_partition_errors": resource_partition_errors,
            "resource_partition_valid": not resource_partition_errors,
            "missing_node_ids": sorted(expected_ids - placed_ids),
            "unexpected_node_ids": sorted(placed_ids - expected_ids),
            "pending_contract_ids": sorted(contract.contract_id for contract in self.mandatory_contracts()),
            "duplicate_axials": duplicate_axials,
            "non_hex_edges": non_hex_edges,
            "topology_conformation_errors": conformation_errors,
            "topology_conformation_valid": not conformation_errors,
            "port_direction_errors": port_direction_errors,
            "port_directions_valid": not port_direction_errors,
            "unintended_contacts": unintended_contacts,
            "nonbonded_contacts_valid": not unintended_contacts,
            "substitution_errors": substitution_errors,
            "substitution_valid": not substitution_errors,
            "shape_metrics": shape_metrics,
            "shape_valid": not shape_warnings,
            "shape_warnings": shape_warnings,
            "side_distribution": {
                "attachment_count": int(side_attachment_count),
                "covered_aromatic_unit_count": len(side_unit_loads),
                "coverage_ratio": float(side_coverage_ratio),
                "max_unit_load": max(side_unit_loads.values(), default=0),
                "unit_loads": dict(sorted(
                    side_unit_loads.items(), key=lambda row: unit_sort_key(row[0])
                )),
            },
            "side_distribution_valid": not side_distribution_warnings,
            "side_distribution_warnings": side_distribution_warnings,
            "allowed_unbonded_contacts": sorted(
                sorted(pair) for pair in self.allowed_unbonded_contacts
            ),
            "geometry_valid": not (
                duplicate_axials
                or non_hex_edges
                or conformation_errors
                or port_direction_errors
                or unintended_contacts
                or substitution_errors
            ),
            "component_count": self.component_count(),
            "edge_count": len(self.edges),
            "geometry_fallback_contract_ids": sorted(
                str(action.get("contract_id", ""))
                for action in self.action_history
                if action.get("geometry_fallback") and action.get("contract_id")
            ),
            "substitution_relaxed_edge_count": len(self.substitution_relaxed_edges),
            "substitution_relaxed_contract_ids": sorted(self.substitution_relaxed_contracts),
        }

    def to_dict(self) -> Dict[str, Any]:
        ownership = self.problem.ownership_map()
        units = {unit.unit_id: unit for unit in self.problem.units}

        def node_role(node_id: int) -> str:
            unit = units.get(ownership.get(int(node_id), ""))
            if unit is None:
                return ""
            return next(
                (str(node.role) for node in unit.nodes if int(node.node_id) == int(node_id)),
                "",
            )

        return {
            "placements": {
                unit_id: {
                    "rotation": int(placement.rotation),
                    "translation": [int(placement.translation[0]), int(placement.translation[1])],
                    "reflection": bool(placement.reflection),
                    "reembedded": unit_id in self.reembedded_units,
                }
                for unit_id, placement in sorted(self.placements.items())
            },
            "nodes": [
                {
                    "node_id": int(node_id),
                    "su_type": int(self.current_su_types.get(int(node_id), -1)),
                    "planned_su_type": int(
                        self.problem.node_map().get(int(node_id), -1)
                    ),
                    "axial": [int(axial[0]), int(axial[1])],
                    "final_position": list(self.final_node_positions.get(
                        int(node_id), (float(axial[0]), float(axial[1]))
                    )),
                    "unit_id": ownership.get(int(node_id), ""),
                    "role": node_role(int(node_id)),
                }
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
            "unassembled_resources": [
                {
                    "node_id": int(resource.node_id),
                    "su_type": int(resource.su_type),
                    "state": str(resource.state),
                    "hop1_ids": [int(value) for value in resource.hop1_ids],
                    "reason": str(resource.reason),
                }
                for resource in self.problem.unassembled_closure_resources
            ],
            "audit": self.audit(),
        }


def relabel_aromatic_sites(
    geometric_state: HexAssemblyState,
    boundary_site_ids: Mapping[int, int],
    actions: Sequence[Mapping[str, Any]],
    *,
    protected_aromatic_ids: Iterable[int] = (),
    replay_contract_kinds: Iterable[str] = ("rigid", "flex"),
    stage: str = "flex",
) -> HexAssemblyState:
    """Commit searched physical slots by one global aromatic GID permutation."""
    state = geometric_state
    protected_ids = {
        int(node_id)
        for contract in state.problem.contracts
        if contract.kind == "rigid"
        for node_id in (contract.source_node_id, contract.target_node_id)
    } | {
        int(node_id) for node_id in protected_aromatic_ids
    }
    aromatic_units = [
        unit for unit in state.problem.units if unit.phase == "aromatic"
    ]
    node_template = {
        int(node.node_id): node
        for unit in aromatic_units
        for node in unit.nodes
    }
    movable_ids = {
        int(node.node_id)
        for unit in aromatic_units
        for node in unit.nodes
        if int(node.node_id) not in protected_ids and int(node.su_type) != 12
    }
    assignments = {
        int(boundary_id): int(site_id)
        for boundary_id, site_id in boundary_site_ids.items()
    }
    if not set(assignments) <= movable_ids:
        raise InverseInputError(
            "boundary GIDs are not movable aromatic perimeter nodes: "
            f"{sorted(set(assignments)-movable_ids)[:12]}"
        )
    if len(set(assignments.values())) != len(assignments):
        raise InverseInputError("multiple boundaries selected the same SU13 site")
    if not set(assignments.values()) <= movable_ids:
        raise InverseInputError(
            "search selected non-perimeter sites: "
            f"{sorted(set(assignments.values())-movable_ids)[:12]}"
        )

    slot_rows = [
        (str(unit.unit_id), tuple(node.local_axial), int(node.node_id), node)
        for unit in aromatic_units
        for node in unit.nodes
        if int(node.node_id) in movable_ids
    ]
    slot_by_old_id = {row[2]: row for row in slot_rows}
    new_id_by_slot: Dict[Tuple[str, Tuple[int, int]], int] = {}
    for boundary_id, old_site_id in assignments.items():
        unit_id, coordinate, _old_id, _node = slot_by_old_id[old_site_id]
        new_id_by_slot[(unit_id, coordinate)] = int(boundary_id)

    remaining_ids = sorted(movable_ids - set(assignments))
    remaining_slots = sorted(
        (
            (unit_id, coordinate)
            for unit_id, coordinate, old_id, _node in slot_rows
            if old_id not in set(assignments.values())
        ),
        key=lambda row: (unit_sort_key(row[0]), row[1]),
    )
    if len(remaining_ids) != len(remaining_slots):
        raise InverseInputError("aromatic GID permutation is not bijective")
    new_id_by_slot.update(zip(remaining_slots, remaining_ids))

    provisional_units: Dict[str, TopologyUnit] = {}
    old_to_new_aromatic_id: Dict[int, int] = {}
    for unit in aromatic_units:
        old_id_by_coordinate = {
            tuple(node.local_axial): int(node.node_id) for node in unit.nodes
        }
        new_id_at_coordinate = {}
        for node in unit.nodes:
            coordinate = tuple(node.local_axial)
            old_id = int(node.node_id)
            new_id_at_coordinate[coordinate] = (
                old_id
                if old_id not in movable_ids
                else int(new_id_by_slot[(str(unit.unit_id), coordinate)])
            )
            old_to_new_aromatic_id[old_id] = int(
                new_id_at_coordinate[coordinate]
            )
        new_nodes = []
        for node in unit.nodes:
            coordinate = tuple(node.local_axial)
            new_id = new_id_at_coordinate[coordinate]
            template = node_template[new_id]
            new_nodes.append(replace(
                template,
                local_axial=coordinate,
                su_type=int(node.su_type),
                role=str(node.role),
            ))
        new_edges = tuple(
            replace(
                edge,
                source_node_id=new_id_at_coordinate[
                    tuple(unit.node_map()[int(edge.source_node_id)].local_axial)
                ],
                target_node_id=new_id_at_coordinate[
                    tuple(unit.node_map()[int(edge.target_node_id)].local_axial)
                ],
            )
            for edge in unit.internal_edges
        )
        provisional_units[str(unit.unit_id)] = replace(
            unit,
            nodes=tuple(new_nodes),
            internal_edges=new_edges,
            ports=(),
            source_node_ids=tuple(
                new_id_at_coordinate[tuple(node.local_axial)]
                for node in unit.nodes
            ),
        )

    old_ports_by_node: Dict[int, List[Any]] = {}
    for unit in aromatic_units:
        for port in unit.ports:
            old_ports_by_node.setdefault(int(port.node_id), []).append(port)
    owner: Dict[int, str] = {
        int(node.node_id): str(unit_id)
        for unit_id, unit in provisional_units.items()
        for node in unit.nodes
    }
    final_aromatic_units = []
    for unit_id in sorted(provisional_units, key=unit_sort_key):
        provisional = provisional_units[unit_id]
        ports = []
        for node in provisional.nodes:
            for port in old_ports_by_node.get(int(node.node_id), []):
                ports.append(replace(
                    port,
                    node_id=int(node.node_id),
                    direction=aromatic_slot_direction(
                        provisional, int(node.node_id)
                    ),
                ))
        final_aromatic_units.append(replace(provisional, ports=tuple(ports)))

    units = tuple(final_aromatic_units) + tuple(
        unit for unit in state.problem.units if unit.phase != "aromatic"
    )
    for unit in units:
        if unit.phase == "aromatic":
            continue
        for node in unit.nodes:
            owner[int(node.node_id)] = str(unit.unit_id)

    contracts = []
    for contract in state.problem.contracts:
        source_unit = state.unit(contract.source_unit_id)
        target_unit = state.unit(contract.target_unit_id)
        contracts.append(replace(
            contract,
            source_unit_id=(
                owner[int(contract.source_node_id)]
                if source_unit.phase == "aromatic"
                else str(contract.source_unit_id)
            ),
            target_unit_id=(
                owner[int(contract.target_node_id)]
                if target_unit.phase == "aromatic"
                else str(contract.target_unit_id)
            ),
        ))
    problem = replace(
        state.problem,
        units=units,
        contracts=tuple(contracts),
        ownership=tuple(sorted(owner.items())),
    )

    rebuilt = HexAssemblyState(problem)
    for unit_id, placement in sorted(state.placements.items()):
        unit = rebuilt.unit(unit_id)
        coordinates = rebuilt._placement_coordinates(
            unit,
            placement.rotation,
            placement.translation,
            placement.reflection,
        )
        if any(axial in rebuilt.occupied_axial for axial in coordinates.values()):
            raise InverseInputError(
                f"relabelled assembly collides while restoring {unit_id!r}"
            )
        rebuilt.placements[unit_id] = placement
        for node_id, axial in coordinates.items():
            rebuilt.node_axial[int(node_id)] = axial
            rebuilt.occupied_axial[axial] = int(node_id)
            rebuilt.final_node_positions[int(node_id)] = (float(axial[0]), float(axial[1]))
        for edge in unit.internal_edges:
            left, right = int(edge.source_node_id), int(edge.target_node_id)
            rebuilt._append_edge(
                left,
                right,
                str(edge.kind),
                direction=rebuilt._edge_direction(
                    coordinates[left], coordinates[right]
                ),
            )

    rebuilt.allowed_unbonded_contacts = {
        frozenset(
            old_to_new_aromatic_id.get(int(node_id), int(node_id))
            for node_id in pair
        )
        for pair in state.allowed_unbonded_contacts
    }

    replay_kinds = {str(kind) for kind in replay_contract_kinds}
    satisfied_ids = {
        str(contract.contract_id)
        for contract in problem.contracts
        if contract.kind in replay_kinds
        and str(contract.contract_id) in state.satisfied_contracts
    }
    for contract in problem.contracts:
        if str(contract.contract_id) not in satisfied_ids:
            continue
        if not rebuilt.force_connect_contract(
            contract,
            "fixed_site_geometry_replay",
            geometry_fallback=False,
        ):
            raise InverseInputError(
                f"cannot replay searched contract {contract.contract_id!r}"
            )
    rebuilt.action_history.extend(copy.deepcopy(list(actions)))
    rebuilt.stage = str(stage)
    if str(stage) == "side" and rebuilt.is_complete():
        # Aromatic clusters intentionally use only SU12/SU13 while sites are
        # being searched.  Once every prescribed connection is committed, the
        # final molecule must expose the complete inverse SU labels (including
        # non-port aromatic hetero atoms such as SU26) to data export and G2S.
        rebuilt.current_su_types = {
            int(node_id): int(su_type)
            for node_id, su_type in rebuilt.problem.node_map().items()
        }
    audit = rebuilt.audit()
    if not audit["geometry_valid"]:
        raise InverseInputError(
            "relabelled assembly failed geometry audit: "
            f"conformation={len(audit['topology_conformation_errors'])} "
            f"ports={len(audit['port_direction_errors'])} "
            f"contacts={audit['unintended_contacts'][:8]}"
        )
    return rebuilt
