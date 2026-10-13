"""Atomic molecular transitions shared by search and greedy continuation."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import copy
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from RL_MTCS.geometry import (Axial, HEX_DIRECTIONS, axial_add, axial_neg,
    transform_axial, aromatic_slot_direction, unit_sort_key, connection_axis)
from RL_MTCS.mdp.types import (ExternalContract, InverseInputError, ConnectionAction,
    TopologyUnit, UnitPlacement, AssemblyEdge)
from RL_MTCS.mdp.sites import aromatic_perimeter_ids, hetero_staging_ids
from RL_MTCS.reporting.diagnostics import note_audit_failure
from RL_MTCS.storage.state_delta import encode_transition, replay_transition


class StateTransitions:
    """Placement mutations used while constructing complete atomic actions."""

    def reshape_side(self, candidate, rotation, translation, reflection=False):
        uid = candidate.unit_id
        if not self.is_placed(uid) or self.unit(uid).phase != 'side':
            return False
        contracts = [c for c in self.problem.contracts
                     if uid in (c.source_unit_id, c.target_unit_id)]
        if len(contracts) != 1 or contracts[0].contract_id not in self.satisfied_contracts:
            return False
        contract = contracts[0]
        root = contract.source_node_id if contract.source_unit_id == uid else contract.target_node_id
        pid = contract.source_port_id if contract.source_unit_id == uid else contract.target_port_id
        old_port = self._ports[f'{uid}:{pid}']
        new_port = next((p for p in candidate.ports if p.port_id == pid), None)
        coordinates = self._placement_coordinates(candidate, rotation, translation, reflection)
        if (new_port is None or coordinates.get(root) != self.node_axial[root]
                or transform_axial(new_port.direction, rotation, reflection)
                   != self.global_port_direction(uid, old_port)):
            return False
        trial = self.clone()
        ids = set(trial.unit(uid).node_map())
        trial.placements.pop(uid)
        for node in ids:
            trial.occupied_axial.pop(trial.node_axial.pop(node))
            trial.final_node_positions.pop(node)
        trial.edges = [e for e in trial.edges
                       if not {e.source_node_id, e.target_node_id} <= ids]
        trial.select_conformer(candidate)
        if not trial.place_unit(uid, rotation, translation, reflection=reflection,
                attachment_edge=(contract.source_node_id, contract.target_node_id)):
            return False
        self._restore(trial)
        return True

    def select_conformer(self, candidate: TopologyUnit) -> None:
        from RL_MTCS.actions.conformers import validate_conformer

        if self.is_placed(candidate.unit_id):
            raise InverseInputError("cannot replace a placed unit's conformer")
        original = self.conformer_sources.get(candidate.unit_id, self.unit(candidate.unit_id))
        validate_conformer(original, candidate)
        if candidate == self.unit(candidate.unit_id):
            return
        self.conformer_sources[candidate.unit_id] = original
        self.problem = replace(self.problem, units=tuple(
            candidate if u.unit_id == candidate.unit_id else u
            for u in self.problem.units
        ))
        self._units[candidate.unit_id] = candidate
        for port in candidate.ports:
            self._ports[f"{candidate.unit_id}:{port.port_id}"] = port

    def transform_component(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        attachment_edge: Optional[Tuple[int, int]] = None,
        attachment_edges: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
    ) -> bool:
        ok, _reason = self.can_transform_component(
            unit_id,
            rotation,
            translation,
            attachment_edge=attachment_edge,
            attachment_edges=attachment_edges,
            reflection=reflection,
        )
        if not ok:
            return False
        component_units = self.component_unit_ids(unit_id)
        component_nodes = self.component_node_ids(unit_id)
        rigid_relevant_nodes: Optional[Set[int]] = None
        if str(self.stage) == "rigid" and attachment_edge is not None:
            contact_nodes = {int(value) for value in attachment_edge}
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
        return True

    def _activate_aromatic_endpoints(
        self,
        endpoints: Iterable[Tuple[str, int]],
    ) -> bool:
        planned = self.problem.node_map()
        ownership = self.problem.ownership_map()
        activated = {}
        for unit_id, node_id in endpoints:
            unit = self._units.get(str(unit_id))
            node_id = int(node_id)
            if unit is not None and unit.phase == "aromatic" and node_id in planned:
                activated[node_id] = int(planned[node_id])
        for node_id, su in activated.items():
            neighbors = {e.target_node_id if e.source_node_id == node_id else e.source_node_id
                         for e in self.edges if node_id in {e.source_node_id, e.target_node_id}}
            external = [n for n in neighbors if ownership.get(n) != ownership.get(node_id)]
            if len(neighbors) != 3 or len(external) != 1:
                return False
        self.current_su_types.update(activated)
        return True

    def place_unit(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        attachment_edge: Optional[Tuple[int, int]] = None,
        attachment_edges: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
    ) -> bool:
        ok, _reason = self.can_place(
            unit_id,
            rotation,
            translation,
            attachment_edge=attachment_edge,
            attachment_edges=attachment_edges,
            reflection=reflection,
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
        return True

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
        ownership = self.problem.ownership_map()
        if ownership.get(source_node_id) != source_unit_id or ownership.get(target_node_id) != target_unit_id:
            return False, "endpoint_owner_mismatch"
        labels = self.problem.node_map()
        if (source_port.required_su_types and labels.get(target_node_id) not in source_port.required_su_types
                or target_port.required_su_types and labels.get(source_node_id) not in target_port.required_su_types):
            return False, "port_partner_su_type_mismatch"
        if source_key in self.used_ports or target_key in self.used_ports:
            return False, "port_already_used"
        dynamic_rigid = str(action.get("kind", "")) == "rigid"
        if not dynamic_rigid:
            expected = next((c for c in self.problem.contracts if c.contract_id == action.get('contract_id')), None)
            if (expected is None or (expected.source_unit_id, expected.target_unit_id,
                    expected.source_node_id, expected.target_node_id, expected.source_port_id, expected.target_port_id) !=
                    (source_unit_id, target_unit_id, source_node_id, target_node_id, source_port_id, target_port_id)
                    or action.get('kind', expected.kind) != expected.kind):
                return False, "action_contract_identity_mismatch"
        if dynamic_rigid:
            if self.unit(source_unit_id).phase != "aromatic" or self.unit(target_unit_id).phase != "aromatic":
                return False, "rigid_requires_aromatic_units"
            if self.problem.node_map().get(source_node_id) != 10 or self.problem.node_map().get(target_node_id) != 10:
                return False, "rigid_requires_su10_ports"
            if source_port.phase != "rigid" or target_port.phase != "rigid":
                return False, "rigid_port_phase_mismatch"
            if (source_node_id not in aromatic_perimeter_ids(self.unit(source_unit_id))
                    or target_node_id not in aromatic_perimeter_ids(self.unit(target_unit_id))):
                return False, "rigid_endpoint_not_ring_perimeter"
            if target_unit_id in self.component_unit_ids(source_unit_id):
                return False, "rigid_cycle"
            expected_id = f"rigid:dynamic:{min(source_node_id, target_node_id)}-{max(source_node_id, target_node_id)}"
            if str(action.get("contract_id", "")) != expected_id:
                return False, "rigid_contract_id_mismatch"
        move_component = str(action.get("move_component_unit_id", ""))
        if move_component:
            if move_component not in self.placements:
                return False, "move_component_not_placed"
            if not self.transform_component(
                move_component,
                int(action.get("component_rotation", 0)),
                tuple(action.get("component_translation", (0, 0))),
                attachment_edge=(source_node_id, target_node_id),
                reflection=bool(action.get("component_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "component_transform_invalid"
        if not self.is_placed(source_unit_id):
            if not self.place_unit(
                source_unit_id,
                int(action.get("source_rotation", 0)),
                tuple(action.get("source_translation", (0, 0))),
                attachment_edge=(source_node_id, target_node_id),
                reflection=bool(action.get("source_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "source_placement_invalid"
        if not self.is_placed(target_unit_id):
            if not self.place_unit(
                target_unit_id,
                int(action.get("target_rotation", 0)),
                tuple(action.get("target_translation", (0, 0))),
                attachment_edge=(source_node_id, target_node_id),
                reflection=bool(action.get("target_reflection", False)),
            ):
                self._restore(snapshot)
                return False, "target_placement_invalid"
        if source_node_id not in self.node_axial or target_node_id not in self.node_axial:
            self._restore(snapshot)
            return False, "endpoint_not_placed"
        direction = self._edge_direction(self.node_axial[source_node_id], self.node_axial[target_node_id])
        kind = str(action.get("kind", self.stage))
        if direction is None or not self._edge_geometry_valid(
            source_node_id,
            target_node_id,
            self.node_axial[source_node_id],
            self.node_axial[target_node_id],
            kind,
        ):
            self._restore(snapshot)
            return False, "endpoints_not_honeycomb_neighbors"
        source_direction = self.global_port_direction(source_unit_id, source_port)
        target_direction = self.global_port_direction(target_unit_id, target_port)
        if ((source_direction is not None and source_direction != direction)
                or (target_direction is not None and target_direction != axial_neg(direction))):
            self._restore(snapshot)
            return False, "rigid_port_direction_mismatch" if dynamic_rigid else "port_direction_mismatch"
        contract_id = str(action.get("contract_id", ""))
        self._append_edge(source_node_id, target_node_id, kind, contract_id, direction)
        if dynamic_rigid and not any(
            contract.contract_id == contract_id for contract in self.problem.contracts
        ):
            contract = ExternalContract(
                contract_id=contract_id,
                source_unit_id=source_unit_id,
                source_node_id=source_node_id,
                target_unit_id=target_unit_id,
                target_node_id=target_node_id,
                kind="rigid",
                fixed=False,
                source_port_id=source_port_id,
                target_port_id=target_port_id,
            )
            self.problem = replace(
                self.problem, contracts=(*self.problem.contracts, contract)
            )
        self.used_ports.update({source_key, target_key})
        if contract_id:
            self.satisfied_contracts.add(contract_id)
        if not self._activate_aromatic_endpoints((
            (source_unit_id, source_node_id),
            (target_unit_id, target_node_id),
        )):
            self._restore(snapshot)
            return False, "actual_endpoint_label_mismatch"
        self.action_history.append(copy.deepcopy(action))
        return True, ""

    def connect_adjacent_contract(
        self, contract: ExternalContract, reason: str,
    ) -> bool:
        return self.connect_adjacent_contracts((contract,), reason)

    def connect_adjacent_contracts(self, contracts, reason: str) -> bool:
        """Commit every endpoint of a multi-port attachment before activation."""
        contracts = tuple(contracts)
        snapshot = self.clone()
        for contract in contracts:
            if not self._connect_adjacent_contract(contract, reason):
                self._restore(snapshot)
                return False
        if not self._activate_aromatic_endpoints(
            pair for c in contracts for pair in (
                (c.source_unit_id, c.source_node_id), (c.target_unit_id, c.target_node_id))
        ):
            self._restore(snapshot)
            return False
        return True

    def _connect_adjacent_contract(
        self,
        contract: ExternalContract,
        reason: str,
    ) -> bool:
        """Connect an already adjacent contract without inventing a long edge."""
        source_key = f"{contract.source_unit_id}:{contract.source_port_id}"
        target_key = f"{contract.target_unit_id}:{contract.target_port_id}"
        if contract not in self.problem.contracts or contract.contract_id in self.satisfied_contracts:
            return False
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
                str(contract.kind),
            )
        ):
            return False
        for uid, pid, expected in (
            (contract.source_unit_id, contract.source_port_id, direction),
            (contract.target_unit_id, contract.target_port_id, axial_neg(direction)),
        ):
            port = self._ports.get(f"{uid}:{pid}")
            gid = contract.source_node_id if uid == contract.source_unit_id else contract.target_node_id
            if port is None or port.node_id != gid or self.problem.ownership_map().get(gid) != uid:
                return False
            partner = contract.target_node_id if gid == contract.source_node_id else contract.source_node_id
            if port.required_su_types and self.problem.node_map().get(partner) not in port.required_su_types:
                return False
            actual = self.global_port_direction(uid, port)
            if actual is not None and actual != expected:
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
        self.action_history.append({
            "kind": str(contract.kind),
            "contract_id": str(contract.contract_id),
            "reason": str(reason),
        })
        return True

    def _restore(self, snapshot: "HexAssemblyState") -> None:
        self.main_anchor = snapshot.main_anchor
        self.backbone_hull = snapshot.backbone_hull
        self.excluded_unit_ids = set(snapshot.excluded_unit_ids)
        self.last_rigid_axis = snapshot.last_rigid_axis
        self.last_flex_axis = snapshot.last_flex_axis
        self.task_inventory = snapshot.task_inventory
        self.problem = snapshot.problem
        self.source_problem = snapshot.source_problem
        self._units = dict(snapshot._units)
        self._ports = dict(snapshot._ports)
        self.conformer_sources = dict(snapshot.conformer_sources)
        self.placements = dict(snapshot.placements)
        self.node_axial = dict(snapshot.node_axial)
        self.final_node_positions = dict(snapshot.final_node_positions)
        self.occupied_axial = dict(snapshot.occupied_axial)
        self.edges = list(snapshot.edges)
        self.used_ports = set(snapshot.used_ports)
        self.satisfied_contracts = set(snapshot.satisfied_contracts)
        self.current_su_types = dict(snapshot.current_su_types)
        self.stage = str(snapshot.stage)
        self.action_history = copy.deepcopy(snapshot.action_history)
        self.substitution_relaxed_edges = set(snapshot.substitution_relaxed_edges)


def relabel_aromatic_sites(
    geometric_state: HexAssemblyState,
    boundary_site_ids: Mapping[int, int],
    actions: Sequence[Mapping[str, Any]],
    *,
    protected_aromatic_ids: Iterable[int] = (),
    replay_contract_kinds: Iterable[str] = ("rigid", "flex", "ring"),
    stage: str = "flex",
    validate_geometry: bool = True,
    movable_hetero_ids: Iterable[int] = (),
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
    } | (hetero_staging_ids(state) - set(movable_hetero_ids))
    protected_ids.update(n for n, su in state.current_su_types.items() if su in {26, 30})
    aromatic_units = [
        unit for unit in state.problem.units if unit.phase == "aromatic"
    ]
    node_template = {
        int(node.node_id): node
        for unit in aromatic_units
        for node in unit.nodes
    }
    perimeter_ids = {n for unit in aromatic_units for n in aromatic_perimeter_ids(unit)}
    movable_ids = {
        int(node.node_id)
        for unit in aromatic_units
        for node in unit.nodes
        if int(node.node_id) not in protected_ids
        and int(node.node_id) in perimeter_ids
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

    # Preserve untouched occupants; only displaced IDs need another slot.
    assigned_slots = set(assignments.values())
    stationary = movable_ids - set(assignments) - assigned_slots
    for old_id in stationary:
        unit_id, coordinate, _old_id, _node = slot_by_old_id[old_id]
        new_id_by_slot[(unit_id, coordinate)] = old_id
    remaining_ids = sorted(movable_ids - set(assignments) - stationary)
    remaining_slots = sorted(
        (
            (unit_id, coordinate)
            for unit_id, coordinate, old_id, _node in slot_rows
            if old_id not in assigned_slots and old_id not in stationary
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
                slot_id=node.slot_id,
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
        )

    # ``current_su_types`` describes the searched physical state, so it must
    # follow the same coordinate/GID permutation as the aromatic templates.
    # Rebuilding from ``problem`` alone would silently turn physical flex
    # endpoint labels back into the aromatic staging SU13 labels.
    remapped_current_su_types = {
        int(old_to_new_aromatic_id.get(int(node_id), int(node_id))): int(su_type)
        for node_id, su_type in state.current_su_types.items()
    }

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

    from RL_MTCS.mdp.state import HexAssemblyState
    rebuilt = HexAssemblyState(problem, source_problem=state.source_problem)
    rebuilt.main_anchor = state.main_anchor
    rebuilt.last_rigid_axis = state.last_rigid_axis
    rebuilt.last_flex_axis = state.last_flex_axis
    rebuilt.backbone_hull = state.backbone_hull
    rebuilt.excluded_unit_ids = set(state.excluded_unit_ids)
    rebuilt.task_inventory = state.task_inventory
    rebuilt.conformer_sources = dict(state.conformer_sources)
    rebuilt.substitution_relaxed_edges = {
        frozenset(old_to_new_aromatic_id.get(n, n) for n in pair)
        for pair in state.substitution_relaxed_edges
    }

    # Preserve the coordinates actually found by the search.  Recomputing
    # them from UnitPlacement is unsafe here: rigid component moves can leave
    # staging placements in the metadata while ``node_axial`` is the
    # authoritative physical snapshot.  Relabelling changes aromatic GIDs at
    # fixed local slots; it must therefore be a pure ID permutation, not a
    # second geometric placement pass.
    source_to_rebuilt: Dict[int, int] = {
        int(old_id): int(old_to_new_aromatic_id.get(int(old_id), int(old_id)))
        for old_id in state.node_axial
    }
    for old_id, axial in state.node_axial.items():
        new_id = int(source_to_rebuilt[int(old_id)])
        if axial in rebuilt.occupied_axial:
            other = rebuilt.occupied_axial[axial]
            raise InverseInputError(
                "relabelled assembly collides while restoring "
                f"node {new_id!r} with node {other!r} at {axial!r}"
            )
        rebuilt.node_axial[new_id] = tuple(axial)
        rebuilt.occupied_axial[tuple(axial)] = new_id
        rebuilt.final_node_positions[new_id] = tuple(
            state.final_node_positions.get(int(old_id), (float(axial[0]), float(axial[1])))
        )

    # Reconstruct placement metadata from the authoritative coordinates too.
    # This keeps later component transforms correct even if an upstream
    # operation retained a stale placement record while updating node_axial.
    for unit_id in sorted(state.placements, key=unit_sort_key):
        unit = rebuilt.unit(unit_id)
        if not unit.nodes or any(int(node.node_id) not in rebuilt.node_axial for node in unit.nodes):
            continue
        stored = state.placements[unit_id]
        # Prefer the exact transform selected by search whenever it still
        # reproduces the snapshot.  Coordinate-only inference is ambiguous
        # for symmetric/collinear units (notably chains): several reflections
        # can place every node identically while sending a directional port
        # somewhere else.
        if all(
            rebuilt._placement_coordinates(
                unit, stored.rotation, stored.translation, stored.reflection
            ).get(int(node.node_id)) == rebuilt.node_axial[int(node.node_id)]
            for node in unit.nodes
        ):
            rebuilt.placements[str(unit_id)] = stored
            continue
        reference = unit.nodes[0]
        target = rebuilt.node_axial[int(reference.node_id)]
        matched = None
        for reflection in (False, True):
            for rotation in range(6):
                rotated = transform_axial(reference.local_axial, rotation, reflection)
                translation = (target[0] - rotated[0], target[1] - rotated[1])
                if all(
                    axial_add(
                        transform_axial(node.local_axial, rotation, reflection),
                        translation,
                    ) == rebuilt.node_axial[int(node.node_id)]
                    for node in unit.nodes
                ):
                    matched = UnitPlacement(
                        str(unit_id), rotation, translation, reflection
                    )
                    break
            if matched is not None:
                break
        if matched is None:
            raise InverseInputError(
                f"cannot reconcile placement metadata with searched coordinates for {unit_id!r}"
            )
        rebuilt.placements[str(unit_id)] = matched

    # Rebuild only internal edges from the relabelled templates.  External
    # contract edges are replayed below, using the same satisfied-contract
    # set as the searched state.
    for unit in problem.units:
        if not all(int(node.node_id) in rebuilt.node_axial for node in unit.nodes):
            continue
        for edge in unit.internal_edges:
            left, right = int(edge.source_node_id), int(edge.target_node_id)
            if left not in rebuilt.node_axial or right not in rebuilt.node_axial:
                continue
            rebuilt._append_edge(
                left,
                right,
                str(edge.kind),
                direction=rebuilt._edge_direction(
                    rebuilt.node_axial[left], rebuilt.node_axial[right]
                ),
            )

    replay_kinds = {str(kind) for kind in replay_contract_kinds}
    satisfied_ids = {
        str(contract.contract_id)
        for contract in problem.contracts
        if contract.kind in replay_kinds
        and str(contract.contract_id) in state.satisfied_contracts
    }
    if not rebuilt.connect_adjacent_contracts(
        (c for c in problem.contracts if c.contract_id in satisfied_ids),
        "fixed_site_geometry_replay",
    ):
        raise InverseInputError("cannot atomically replay searched contracts")
    # Contract replay restores geometry, not new search actions. Retaining its
    # synthetic log at every relabel would duplicate the whole prefix again.
    rebuilt.action_history = copy.deepcopy(list(actions))
    rebuilt.stage = str(stage)
    # Unconnected slots remain SU13 even when their immutable task expects
    # SU5/SU9/SU11. Replayed edges are the sole source of activated labels.
    for node_id, su in remapped_current_su_types.items():
        if su in {26, 30}:
            rebuilt.current_su_types[node_id] = su
    audit = rebuilt.audit(diagnostics=False) if validate_geometry else None
    if audit is not None and not audit["legal"]:
        raise InverseInputError(
            "relabelled assembly failed geometry audit: "
            f"conformation={len(audit['topology_conformation_errors'])} "
            f"ports={len(audit['port_direction_errors'])} "
            f"port_errors={audit['port_direction_errors'][:8]}",
            diagnostics={
                "geometry_audit": audit,
                "failure_scope": "relabel_commit",
            },
        )
    return rebuilt




def state_key(state):
    semantic = repr((state.problem, state.state_signature(), state.task_inventory))
    history = json.dumps(state.action_history, sort_keys=True)
    return sha256((semantic + history).encode('utf-8')).digest()


def validate_proposal(child, phase):
    checked = child.__dict__.pop('_candidate_audited', False)
    if not checked:
        audit = child.audit(diagnostics=False)
        if not audit['legal']:
            note_audit_failure(audit)
            return None
    child.stage = phase
    if phase in ('rigid', 'flex'):
        axis = child.action_history[-1].get('connection_axis')
        if axis is None:
            edge = child.edges[-1]
            axis = connection_axis(child._edge_direction(
                child.node_axial[edge.source_node_id], child.node_axial[edge.target_node_id]))
        setattr(child, f'last_{phase}_axis', axis)
        child.action_history[-1]['connection_axis'] = axis
    return child


def prepare_action(source, child, phase, source_key):
    description = child.action_history[-1]
    uid = description.get('topology_unit_id')
    poses = tuple((u, p.rotation, p.translation, p.reflection)
                  for u, p in sorted(child.placements.items()) if source.placements.get(u) != p)
    bindings = tuple(sorted((int(a), int(b)) for a, b in
                            description.get('boundary_site_assignments', {}).items()))
    return ConnectionAction(phase, source_key, state_key(child), encode_transition(source, child),
                            json.dumps(description, sort_keys=True),
                            child.unit(uid) if uid else None, poses, bindings)


def apply_action(state, action):
    """Apply only to the audited source, without rescanning or partial mutation.

    The payload contains the complete conformer, all component poses, labels,
    contracts and ports. The target fingerprint certifies exact replay of the
    preview that passed the shared audit; no second whole-state audit is needed.
    """
    if state_key(state) != action.source_key:
        raise InverseInputError('stale connection action: source state changed')
    child = replay_transition(state, action.payload)
    if state_key(child) != action.target_key:
        raise InverseInputError('connection action replay differs from its validated target')
    child._connection_action = action
    return child


def commit_attachment(candidate, source, phase):
    protected = {n for e in source.edges if e.contract_id
                 for n in (e.source_node_id,e.target_node_id)}
    # Geometry errors are ordinary failed proposals; replay inconsistencies are
    # programming/input errors and deliberately propagate to the caller.
    current = relabel_aromatic_sites(candidate.state, candidate.boundary_site_ids,
        [*source.action_history, candidate.actions[-1]], protected_aromatic_ids=protected,
        replay_contract_kinds=('rigid','flex','ring','side'), stage=phase, validate_geometry=False)
    current.main_anchor = source.main_anchor or candidate.state.main_anchor
    audit = current.audit(diagnostics=False)
    if not audit['legal']:
        note_audit_failure(audit)
        return None
    current._candidate_audited = True
    current.action_history[-1]['stage'] = phase
    return current
