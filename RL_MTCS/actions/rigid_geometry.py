"""Rigid port alignment and component poses; no search policy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from RL_MTCS.geometry import HEX_DIRECTIONS, axial_add, axial_distance, axial_neg, transform_axial
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import ExternalContract, TopologyUnit, UnitPort


def _port_compatible(source: UnitPort, target: UnitPort, source_su: int, target_su: int) -> bool:
    if source.required_su_types and int(target_su) not in set(source.required_su_types):
        return False
    if target.required_su_types and int(source_su) not in set(target.required_su_types):
        return False
    return True


class RigidPoseGenerator:
    """Rigid-only alignment of existing aromatic components."""

    def __init__(self, state: HexAssemblyState):
        self.state = state
        self.state.stage = 'rigid'

    def _candidate_for_pair(
        self,
        source_unit: TopologyUnit,
        source_port: UnitPort,
        target_unit: TopologyUnit,
        target_port: UnitPort,
        *,
        contract: Optional[ExternalContract] = None,
    ) -> List[Dict[str, Any]]:
        if source_unit.unit_id == target_unit.unit_id:
            return []
        if source_unit.phase != 'aromatic' or target_unit.phase != 'aromatic':
            return []
        if source_unit.unit_id in self.state.placements and target_unit.unit_id in self.state.placements:
            source_axial = self.state.node_axial.get(int(source_port.node_id))
            target_axial = self.state.node_axial.get(int(target_port.node_id))
            if source_axial is None or target_axial is None:
                return []
            if axial_distance(source_axial, target_axial) != 1:
                if target_unit.unit_id in self.state.component_unit_ids(source_unit.unit_id):
                    return []
                actions: List[Dict[str, Any]] = []
                actions.extend(self._component_transform_candidates(
                    source_unit,
                    source_port,
                    target_unit,
                    target_port,
                    contract=contract,
                ))
                actions.extend(self._component_transform_candidates(
                    target_unit,
                    target_port,
                    source_unit,
                    source_port,
                    contract=contract,
                ))
                return actions
            return [self._make_action(source_unit, source_port, target_unit, target_port, contract=contract)]

        if source_unit.unit_id in self.state.placements:
            source_unit, source_port, target_unit, target_port = target_unit, target_port, source_unit, source_port
        if target_unit.unit_id not in self.state.placements or source_unit.unit_id in self.state.placements:
            return []
        target_axial = self.state.node_axial.get(int(target_port.node_id))
        if target_axial is None:
            return []
        source_su = source_unit.node_map().get(int(source_port.node_id))
        target_su_node = target_unit.node_map().get(int(target_port.node_id))
        if source_su is None or target_su_node is None:
            return []
        if not _port_compatible(source_port, target_port, int(source_su.su_type), int(target_su_node.su_type)):
            return []

        actions: List[Dict[str, Any]] = []
        target_direction = self.state.global_port_direction(target_unit.unit_id, target_port)
        directions = [target_direction] if target_direction is not None else list(HEX_DIRECTIONS)
        for reflection in (False, True):
            for rotation in range(6):
                source_direction = self.state.global_port_direction(source_unit.unit_id, source_port, rotation=rotation, reflection=reflection)
                for direction in directions:
                    if direction is None:
                        continue
                    direction = tuple(int(value) for value in direction)
                    if source_direction is not None and source_direction != axial_neg(direction):
                        continue
                    source_target = axial_add(target_axial, direction)
                    source_local = self.state.global_node_axial(
                        source_unit.unit_id,
                        int(source_port.node_id),
                        rotation=rotation,
                        translation=(0, 0),
                        reflection=reflection,
                    )
                    translation = (source_target[0] - source_local[0], source_target[1] - source_local[1])
                    ok, _reason = self.state.can_place(
                        source_unit.unit_id,
                        rotation,
                        translation,
                        attachment_edge=(int(source_port.node_id), int(target_port.node_id)),
                        reflection=reflection,
                    )
                    if not ok:
                        continue
                    action = self._make_action(
                        source_unit,
                        source_port,
                        target_unit,
                        target_port,
                        contract=contract,
                        source_rotation=rotation,
                        source_translation=translation,
                    )
                    action["source_reflection"] = bool(reflection)
                    actions.append(action)
        return actions

    def _component_transform_candidates(
        self,
        movable_unit: TopologyUnit,
        movable_port: UnitPort,
        stationary_unit: TopologyUnit,
        stationary_port: UnitPort,
        *,
        contract: Optional[ExternalContract],
    ) -> List[Dict[str, Any]]:
        movable_axial = self.state.node_axial.get(int(movable_port.node_id))
        stationary_axial = self.state.node_axial.get(int(stationary_port.node_id))
        if movable_axial is None or stationary_axial is None:
            return []
        stationary_direction = self.state.global_port_direction(
            stationary_unit.unit_id, stationary_port
        )
        movable_direction = self.state.global_port_direction(
            movable_unit.unit_id, movable_port
        )
        directions = [stationary_direction] if stationary_direction is not None else list(HEX_DIRECTIONS)
        component_nodes = self.state.component_node_ids(movable_unit.unit_id)
        actions: List[Dict[str, Any]] = []
        for reflection in (False, True):
            for rotation in range(6):
                transformed_direction = (
                    transform_axial(movable_direction, rotation, reflection)
                    if movable_direction is not None else None
                )
                rotated_axial = transform_axial(
                    movable_axial, rotation, reflection
                )
                for direction in directions:
                    if direction is None:
                        continue
                    direction = tuple(int(value) for value in direction)
                    if transformed_direction is not None and transformed_direction != axial_neg(direction):
                        continue
                    desired = axial_add(stationary_axial, direction)
                    translation = (
                        desired[0] - rotated_axial[0],
                        desired[1] - rotated_axial[1],
                    )
                    ok, _reason = self.state.can_transform_component(
                        movable_unit.unit_id,
                        rotation,
                        translation,
                        attachment_edge=(int(movable_port.node_id), int(stationary_port.node_id)),
                        reflection=reflection,
                    )
                    if not ok:
                        continue
                    action = self._make_action(
                        movable_unit,
                        movable_port,
                        stationary_unit,
                        stationary_port,
                        contract=contract,
                    )
                    action.update({
                        "move_component_unit_id": movable_unit.unit_id,
                        "component_rotation": int(rotation),
                        "component_reflection": bool(reflection),
                        "component_translation": [int(translation[0]), int(translation[1])],
                        "moved_node_count": len(component_nodes),
                    })
                    actions.append(action)
        return actions

    @staticmethod
    def _make_action(
        source_unit: TopologyUnit,
        source_port: UnitPort,
        target_unit: TopologyUnit,
        target_port: UnitPort,
        *,
        contract: Optional[ExternalContract] = None,
        source_rotation: int = 0,
        source_translation: Tuple[int, int] = (0, 0),
    ) -> Dict[str, Any]:
        return {
            "kind": "rigid",
            "contract_id": str(contract.contract_id if contract is not None else ""),
            "source_unit_id": str(source_unit.unit_id),
            "source_node_id": int(source_port.node_id),
            "source_port_id": str(source_port.port_id),
            "target_unit_id": str(target_unit.unit_id),
            "target_node_id": int(target_port.node_id),
            "target_port_id": str(target_port.port_id),
            "source_rotation": int(source_rotation) % 6,
            "source_translation": [int(source_translation[0]), int(source_translation[1])],
        }


def _rigid_unit_components(
    state: HexAssemblyState,
) -> List[Tuple[set[str], List[Any]]]:
    """Return connected components of the prescribed aromatic rigid graph."""
    rigid_contracts = [
        contract for contract in state.problem.contracts
        if contract.kind == "rigid" and contract.contract_id in state.satisfied_contracts
    ]
    aromatic_ids = {
        str(unit.unit_id)
        for unit in state.problem.units
        if unit.phase == "aromatic"
    }
    adjacency: Dict[str, set[str]] = {
        unit_id: set() for unit_id in aromatic_ids
    }
    for contract in rigid_contracts:
        left, right = str(contract.source_unit_id), str(contract.target_unit_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    components: List[Tuple[set[str], List[Any]]] = []
    unseen = set(aromatic_ids)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        units = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    units.add(neighbor)
                    stack.append(neighbor)
        contracts = [
            contract for contract in rigid_contracts
            if str(contract.source_unit_id) in units
            and str(contract.target_unit_id) in units
        ]
        components.append((units, contracts))
    components.sort(key=lambda item: min(item[0]))
    return components
