"""Molecular placement state for immutable inverse topology units."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import copy
from RL_MTCS.validation import GeometryValidation
from RL_MTCS.mdp.transition import StateTransitions
from typing import Any, Dict, List, Optional, Set, Tuple

from RL_MTCS.geometry import (
    Axial,
    HEX_DIRECTIONS,
    axial_add,
    transform_axial,
)
from RL_MTCS.mdp.types import (
    ExternalContract,
    InverseAssemblyInput,
    TopologyUnit,
    UnitPort,
    UnitPlacement,
    AssemblyEdge,
)


@dataclass
class HexAssemblyState(StateTransitions, GeometryValidation):
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
    # Original CSV templates are retained independently of the selected pose.
    conformer_sources: Dict[str, TopologyUnit] = field(default_factory=dict)
    current_su_types: Dict[int, int] = field(default_factory=dict)
    main_anchor: Optional[str] = None
    last_rigid_axis: Optional[str] = None
    last_flex_axis: Optional[str] = None
    backbone_hull: tuple = ()
    excluded_unit_ids: Set[str] = field(default_factory=set)
    task_inventory: tuple = ()
    stage: str = "aromatic"
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    substitution_relaxed_edges: Set[frozenset[int]] = field(default_factory=set)
    source_problem: Optional[InverseAssemblyInput] = None

    def __post_init__(self) -> None:
        if self.source_problem is None:
            self.source_problem = self.problem
        self._units = {unit.unit_id: unit for unit in self.problem.units}
        self._ports = {
            f"{unit.unit_id}:{port.port_id}": port
            for unit in self.problem.units
            for port in unit.ports
        }
        self._su25_center_parents = {child: parent for unit in self.source_problem.units
                                    for parent, child in unit.su25_center_branches}
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
        # Avoid constructing complete default maps only to overwrite them.
        new = object.__new__(HexAssemblyState)
        new.problem = self.problem
        new.source_problem = self.source_problem
        new._units = dict(self._units)
        new._ports = dict(self._ports)
        new._su25_center_parents = self._su25_center_parents
        new.placements = dict(self.placements)
        new.node_axial = dict(self.node_axial)
        new.final_node_positions = dict(self.final_node_positions)
        new.occupied_axial = dict(self.occupied_axial)
        new.edges = list(self.edges)
        new.used_ports = set(self.used_ports)
        new.satisfied_contracts = set(self.satisfied_contracts)
        new.conformer_sources = dict(self.conformer_sources)
        new.current_su_types = dict(self.current_su_types)
        new.main_anchor = self.main_anchor
        new.last_rigid_axis = self.last_rigid_axis
        new.last_flex_axis = self.last_flex_axis
        new.backbone_hull = self.backbone_hull
        new.excluded_unit_ids = set(self.excluded_unit_ids)
        new.task_inventory = self.task_inventory
        new.stage = str(self.stage)
        new.action_history = copy.deepcopy(self.action_history)
        new.substitution_relaxed_edges = set(self.substitution_relaxed_edges)
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

    def unit(self, unit_id: str) -> TopologyUnit:
        return self._units[str(unit_id)]

    def is_placed(self, unit_id: str) -> bool:
        return str(unit_id) in self.placements


    def unplaced_units(self, phase: Optional[str] = None) -> List[TopologyUnit]:
        from RL_MTCS.mdp.components import phase_of
        values = [unit for unit in self.problem.units if not self.is_placed(unit.unit_id)]
        if phase is not None:
            values = [unit for unit in values if phase_of(unit) == phase]
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

    def current_node_map(self) -> Dict[int, int]:
        """Return mutable assembly-time SU labels for visualization/export."""
        return dict(self.current_su_types)

    @staticmethod
    def _edge_direction(source: Axial, target: Axial) -> Optional[Axial]:
        delta = (int(target[0]) - int(source[0]), int(target[1]) - int(source[1]))
        return delta if delta in HEX_DIRECTIONS else None

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
        from RL_MTCS.mdp.components import phase_of
        if phase not in ('rigid', 'flex', 'ring', 'side', 'substitution'):
            raise ValueError(f'unknown assembly phase: {phase}')
        if any(not self.is_placed(unit.unit_id) for unit in self.problem.units if phase_of(unit)==phase):
            return False
        if str(phase) == "rigid":
            rigid = [contract for contract in self.problem.contracts if contract.kind == "rigid"]
            su10 = {node_id for node_id, su in self.problem.node_su_types if su == 10}
            used = [node_id for contract in rigid for node_id in (
                contract.source_node_id, contract.target_node_id
            )]
            return (
                len(rigid) == len(su10) // 2
                and len(used) == len(su10)
                and set(used) == su10
                and all(contract.contract_id in self.satisfied_contracts for contract in rigid)
            )
        if phase in {'flex','ring','side'}:
            return not any(c.mandatory and c.contract_id not in self.satisfied_contracts
                           and any(phase_of(self.unit(uid))==phase for uid in (c.source_unit_id,c.target_unit_id))
                           for c in self.problem.contracts)
        return not self.mandatory_contracts(phase)

    def is_complete(self) -> bool:
        """Original task completion; geometric legality is audited separately."""
        from RL_MTCS.reporting.completion import completion_report
        return completion_report(self)['all_complete']

    def forward_signature(self):
        """A geometric checkpoint key, independent of its reporting stage."""
        from RL_MTCS.mdp.components import choose_anchor
        snapshot = self.clone()
        choose_anchor(snapshot)
        snapshot.stage = 'forward_checkpoint'
        return snapshot.state_signature()

    def state_signature(self) -> Tuple[Any, ...]:
        from RL_MTCS.actions.conformers import geometry_signature
        return (
            str(self.stage), self.main_anchor, self.backbone_hull,
            self.problem.node_su_types, self.problem.ownership, self.problem.contracts,
            self.problem.aromatic_residual_node_ids,
            self.problem.units,
            self.last_rigid_axis, self.last_flex_axis,
            tuple(sorted(self.excluded_unit_ids)),
            tuple(sorted(self.occupied_axial.items())),
            tuple((uid, geometry_signature(self.unit(uid)))
                  for uid in sorted(self.conformer_sources)),
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
            tuple(sorted(self.current_su_types.items())),
            tuple(sorted(tuple(sorted(pair)) for pair in self.substitution_relaxed_edges)),
        )

    def aromatic_site_bindings(self) -> Dict[str, Any]:
        return {
            str(node.node_id): {"unit_id": unit.unit_id, "slot_index": index,
                                "slot_id": node.slot_id or f"{unit.unit_id}:slot:{node.local_axial[0]},{node.local_axial[1]}",
                                "local_axial": list(node.local_axial)}
            for unit in self.problem.units if unit.phase == "aromatic"
            for index, node in enumerate(unit.nodes)
        }

    def to_dict(self) -> Dict[str, Any]:
        from RL_MTCS.mdp.components import phase_of
        ownership = self.problem.ownership_map()
        units = {unit.unit_id: unit for unit in self.problem.units}
        planned = self.problem.node_map()
        targets = self.problem.assembly_target_node_map()
        residual_ids = set(self.problem.aromatic_residual_node_ids)
        original = self.source_problem.node_map()
        source_owner = self.source_problem.ownership_map()
        nodes = {n.node_id: n for u in self.problem.units for n in u.nodes}
        degree = Counter(n for e in self.edges for n in (e.source_node_id, e.target_node_id))
        active_contracts = {n: [] for n in self.node_axial}
        for c in self.problem.contracts:
            if c.contract_id in self.satisfied_contracts:
                for n in (c.source_node_id, c.target_node_id):
                    active_contracts.setdefault(n, []).append(c.contract_id)

        def node_role(node_id: int) -> str:
            unit = units.get(ownership.get(int(node_id), ""))
            if unit is None:
                return ""
            return next(
                (str(node.role) for node in unit.nodes if int(node.node_id) == int(node_id)),
                "",
            )

        return {
            "main_anchor": self.main_anchor,
            "last_rigid_axis": self.last_rigid_axis,
            "last_flex_axis": self.last_flex_axis,
            "excluded_unit_ids": sorted(self.excluded_unit_ids),
            "task_inventory": list(self.task_inventory),
            "aromatic_residual_nodes": self.problem.aromatic_residual_rows(),
            "action_history": copy.deepcopy(self.action_history),
            "substitution_geometry_exceptions": [sorted(pair) for pair in self.substitution_relaxed_edges],
            "su25_center_branches": [
                {"parent_node_id": parent, "branch_node_id": child,
                 "placed": child in self.node_axial}
                for child, parent in sorted(self._su25_center_parents.items())
            ],
            "units": [{"unit_id":u.unit_id,"stage":phase_of(u),"placed":u.unit_id in self.placements,
                       "excluded":u.unit_id in self.excluded_unit_ids} for u in self.problem.units],
            "aromatic_site_bindings": self.aromatic_site_bindings(),
            "rigid_contracts": [
                {
                    "contract_id": str(contract.contract_id),
                    "source_unit_id": str(contract.source_unit_id),
                    "source_node_id": int(contract.source_node_id),
                    "target_unit_id": str(contract.target_unit_id),
                    "target_node_id": int(contract.target_node_id),
                    "satisfied": str(contract.contract_id) in self.satisfied_contracts,
                }
                for contract in self.problem.contracts if contract.kind == "rigid"
            ],
            "selected_conformers": {
                uid: {
                    "node_coordinates": {str(n.node_id): list(n.local_axial)
                                         for n in self.unit(uid).nodes},
                    "port_directions": {p.port_id: list(p.direction)
                                        for p in self.unit(uid).ports},
                    "geometry_source": "generated_from_connectivity",
                }
                for uid in sorted(self.conformer_sources)
            },
            "placements": {
                unit_id: {
                    "rotation": int(placement.rotation),
                    "translation": [int(placement.translation[0]), int(placement.translation[1])],
                    "reflection": bool(placement.reflection),
                }
                for unit_id, placement in sorted(self.placements.items())
            },
            "nodes": [
                {
                    "node_id": int(node_id),
                    "logical_gid": int(node_id),
                    "physical_slot_id": nodes[node_id].slot_id or f'{ownership[node_id]}:node:{node_id}',
                    "source_unit_id": source_owner.get(node_id, ""),
                    "original_su_type": original.get(node_id, -1),
                    "actual_degree": degree[node_id],
                    "su25_center_parent_id": self._su25_center_parents.get(node_id),
                    "satisfied_contract_ids": sorted(active_contracts[node_id]),
                    "label_state": ('substituted' if self.current_su_types.get(node_id) in {26, 30}
                                    else 'residual_converted' if node_id in residual_ids
                                    else 'connected' if active_contracts[node_id]
                                    else 'aromatic_pending' if units[ownership[node_id]].phase == 'aromatic'
                                    and self.current_su_types.get(node_id) == 13 else 'internal'),
                    "su_type": int(self.current_su_types.get(int(node_id), -1)),
                    "planned_su_type": int(
                        planned.get(int(node_id), -1)
                    ),
                    "assembly_target_su_type": targets.get(node_id, -1),
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
