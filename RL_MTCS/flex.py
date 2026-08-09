"""Geometry-preserving flexible assembly over rigid aromatic components.

The inverse CSV owns topology identity and connectivity.  Every topology unit
is given one immutable local hex-grid geometry during initialization.  Search
may apply only a dihedral rigid transform plus translation to that geometry;
it must never rebuild or fold individual topology nodes during connection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .geometry import (
    MAX_Q_R_ASPECT_RATIO,
    MIN_Q_R_ASPECT_RATIO,
    aromatic_slot_direction,
    axial_to_cartesian,
    coordinate_aspect,
    state_shape_rank,
    unit_sort_key,
)
from .state import (
    HexAssemblyState,
    axial_add,
    axial_distance,
    axial_neg,
    transform_axial,
    relabel_aromatic_sites,
)
from .models import (
    ExternalContract,
    HEX_DIRECTIONS,
    InverseAssemblyInput,
    InverseInputError,
    InputSourceKind,
    TopologyUnit,
)
from .rigid import SearchResult, TopologyPlacementStage


def _unit_sort_key(unit_id: str) -> Tuple[Any, ...]:
    return unit_sort_key(unit_id)


@dataclass(frozen=True)
class InteriorFlexStep:
    topology_unit_id: str
    attachment_contract_id: str = ""
    closure_contract_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_unit_id": str(self.topology_unit_id),
            "attachment_contract_id": str(self.attachment_contract_id),
            "closure_contract_ids": list(self.closure_contract_ids),
        }


@dataclass(frozen=True)
class BackboneFlexStep:
    topology_unit_id: str
    attachment_contract_id: str
    merge_contract_ids: Tuple[str, ...]

    @property
    def contract_ids(self) -> Tuple[str, ...]:
        return (self.attachment_contract_id,) + self.merge_contract_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_unit_id": str(self.topology_unit_id),
            "attachment_contract_id": str(self.attachment_contract_id),
            "merge_contract_ids": list(self.merge_contract_ids),
        }


@dataclass(frozen=True)
class FlexAssemblyPlan:
    anchor_rigid_component_unit_ids: Tuple[str, ...]
    rigid_components: Tuple[Tuple[str, ...], ...]
    backbone_topology_unit_ids: Tuple[str, ...]
    backbone_contract_ids: Tuple[str, ...]
    backbone_steps: Tuple[BackboneFlexStep, ...]
    interior_steps: Tuple[InteriorFlexStep, ...]

    @property
    def interior_topology_unit_ids(self) -> Tuple[str, ...]:
        return tuple(step.topology_unit_id for step in self.interior_steps)

    @property
    def interior_contract_ids(self) -> Tuple[str, ...]:
        return tuple(
            contract_id
            for step in self.interior_steps
            for contract_id in (
                ((step.attachment_contract_id,) if step.attachment_contract_id else ())
                + step.closure_contract_ids
            )
        )

    @property
    def all_contract_ids(self) -> Tuple[str, ...]:
        return self.backbone_contract_ids + self.interior_contract_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_rigid_component_unit_ids": list(
                self.anchor_rigid_component_unit_ids
            ),
            "rigid_components": [list(values) for values in self.rigid_components],
            "backbone_topology_unit_ids": list(self.backbone_topology_unit_ids),
            "backbone_contract_ids": list(self.backbone_contract_ids),
            "backbone_steps": [step.to_dict() for step in self.backbone_steps],
            "interior_topology_unit_ids": list(self.interior_topology_unit_ids),
            "interior_contract_ids": list(self.interior_contract_ids),
            "interior_steps": [step.to_dict() for step in self.interior_steps],
        }


def _aromatic_endpoint(
    state: HexAssemblyState,
    contract: ExternalContract,
) -> Tuple[str, int]:
    source = state.unit(contract.source_unit_id)
    target = state.unit(contract.target_unit_id)
    if source.phase == "aromatic" and target.phase == "flex":
        return str(source.unit_id), int(contract.source_node_id)
    if target.phase == "aromatic" and source.phase == "flex":
        return str(target.unit_id), int(contract.target_node_id)
    raise InverseInputError(
        f"flex contract {contract.contract_id!r} must connect one aromatic "
        "unit and one flexible topology unit"
    )


def _topology_endpoint(
    state: HexAssemblyState,
    contract: ExternalContract,
) -> Tuple[str, int]:
    source = state.unit(contract.source_unit_id)
    target = state.unit(contract.target_unit_id)
    if source.phase == "flex" and target.phase == "aromatic":
        return str(source.unit_id), int(contract.source_node_id)
    if target.phase == "flex" and source.phase == "aromatic":
        return str(target.unit_id), int(contract.target_node_id)
    raise InverseInputError(
        f"flex contract {contract.contract_id!r} has invalid unit phases"
    )


def _topology_unit_id(
    state: HexAssemblyState,
    contract: ExternalContract,
) -> str:
    return _topology_endpoint(state, contract)[0]


def _rigid_components(state: HexAssemblyState) -> Tuple[Tuple[str, ...], ...]:
    aromatic_ids = {
        str(unit.unit_id) for unit in state.problem.units
        if unit.phase == "aromatic"
    }
    remaining = set(aromatic_ids)
    components: List[Tuple[str, ...]] = []
    while remaining:
        seed = min(remaining, key=_unit_sort_key)
        component = tuple(sorted(
            state.component_unit_ids(seed) & aromatic_ids,
            key=_unit_sort_key,
        ))
        if not component:
            raise InverseInputError(f"aromatic unit {seed!r} is not placed")
        components.append(component)
        remaining.difference_update(component)
    return tuple(sorted(
        components,
        key=lambda values: tuple(map(_unit_sort_key, values)),
    ))


def build_flex_assembly_plan(state: HexAssemblyState) -> FlexAssemblyPlan:
    """Build a rooted spanning plan followed by all non-tree contracts."""
    pending_rigid = state.mandatory_contracts("rigid")
    if pending_rigid:
        raise InverseInputError(
            "flex planning requires a completed rigid stage; pending="
            + ",".join(contract.contract_id for contract in pending_rigid)
        )
    rigid_components = _rigid_components(state)
    component_by_unit = {
        unit_id: index
        for index, component in enumerate(rigid_components)
        for unit_id in component
    }
    unit_nodes = {
        unit.unit_id: len(unit.nodes) for unit in state.problem.units
    }

    flex_contracts = [
        contract for contract in state.problem.contracts
        if contract.kind == "flex"
    ]
    by_topology: Dict[str, List[ExternalContract]] = {}
    contract_component: Dict[str, int] = {}
    for contract in flex_contracts:
        topology_id = _topology_unit_id(state, contract)
        aromatic_id, _node_id = _aromatic_endpoint(state, contract)
        by_topology.setdefault(topology_id, []).append(contract)
        contract_component[str(contract.contract_id)] = component_by_unit[aromatic_id]
    for values in by_topology.values():
        values.sort(key=lambda contract: str(contract.contract_id))

    topology_components = {
        topology_id: {
            contract_component[str(contract.contract_id)]
            for contract in contracts
        }
        for topology_id, contracts in by_topology.items()
    }
    multi_component_hubs = [
        topology_id for topology_id, component_ids in topology_components.items()
        if len(component_ids) > 2
    ]
    if multi_component_hubs:
        root_topology_id = min(
            multi_component_hubs,
            key=lambda topology_id: (
                -len(topology_components[topology_id]),
                -len(state.unit(topology_id).nodes),
                _unit_sort_key(topology_id),
            ),
        )
        root_components = topology_components[root_topology_id]
        anchor_index = min(
            root_components,
            key=lambda index: (
                -sum(
                    unit_nodes[unit_id]
                    for unit_id in rigid_components[index]
                ),
                tuple(map(_unit_sort_key, rigid_components[index])),
            ),
        )
    else:
        anchor_index = min(
            range(len(rigid_components)),
            key=lambda index: (
                -sum(
                    unit_nodes[unit_id]
                    for unit_id in rigid_components[index]
                ),
                -len(rigid_components[index]),
                tuple(map(_unit_sort_key, rigid_components[index])),
            ),
        )

    component_flex_degree: Dict[int, int] = {
        component_index: sum(
            component_index in {
                contract_component[str(contract.contract_id)]
                for contract in contracts
            }
            for contracts in by_topology.values()
        )
        for component_index in range(len(rigid_components))
    }
    connected: Set[int] = {int(anchor_index)}
    component_depth: Dict[int, int] = {int(anchor_index): 0}
    remaining = set(by_topology)
    backbone_units: List[str] = []
    backbone_contracts: List[str] = []
    backbone_steps: List[BackboneFlexStep] = []
    deferred_by_unit: Dict[str, List[str]] = {}

    while len(connected) < len(rigid_components):
        candidates = []
        for topology_id in remaining:
            component_ids = {
                contract_component[str(contract.contract_id)]
                for contract in by_topology[topology_id]
            }
            inside = component_ids & connected
            outside = component_ids - connected
            if inside and outside:
                candidates.append((
                    min(component_depth[index] for index in inside),
                    min(component_flex_degree[index] for index in outside),
                    -len(outside),
                    len(state.unit(topology_id).nodes),
                    _unit_sort_key(topology_id),
                    topology_id,
                    inside,
                    outside,
                ))
        if not candidates:
            missing = sorted(set(range(len(rigid_components))) - connected)
            raise InverseInputError(
                f"flex topology cannot connect all rigid components; missing={missing}"
            )
        (
            parent_depth,
            _outside_degree,
            _neg_outside,
            _node_count,
            _key,
            topology_id,
            inside,
            outside,
        ) = min(candidates)
        contracts = by_topology[topology_id]
        anchor_contract = min(
            (
                contract for contract in contracts
                if contract_component[str(contract.contract_id)] in inside
            ),
            key=lambda contract: str(contract.contract_id),
        )
        merge_contracts = [
            min(
                (
                    contract for contract in contracts
                    if contract_component[str(contract.contract_id)] == component_id
                ),
                key=lambda contract: str(contract.contract_id),
            )
            for component_id in sorted(outside)
        ]
        selected = [anchor_contract] + merge_contracts
        selected_ids = {str(contract.contract_id) for contract in selected}
        deferred = [
            str(contract.contract_id) for contract in contracts
            if str(contract.contract_id) not in selected_ids
        ]
        if deferred:
            deferred_by_unit[topology_id] = deferred
        backbone_units.append(topology_id)
        backbone_contracts.extend(
            str(contract.contract_id) for contract in selected
        )
        backbone_steps.append(BackboneFlexStep(
            topology_unit_id=str(topology_id),
            attachment_contract_id=str(anchor_contract.contract_id),
            merge_contract_ids=tuple(
                str(contract.contract_id) for contract in merge_contracts
            ),
        ))
        connected.update(outside)
        for component_index in outside:
            component_depth[int(component_index)] = int(parent_depth) + 1
        remaining.remove(topology_id)

    interior_steps: List[InteriorFlexStep] = []
    for topology_id in backbone_units:
        deferred = deferred_by_unit.get(topology_id, [])
        if deferred:
            interior_steps.append(InteriorFlexStep(
                topology_unit_id=topology_id,
                closure_contract_ids=tuple(deferred),
            ))
    for topology_id in sorted(remaining, key=_unit_sort_key):
        contract_ids = [
            str(contract.contract_id) for contract in by_topology[topology_id]
        ]
        if contract_ids:
            interior_steps.append(InteriorFlexStep(
                topology_unit_id=topology_id,
                attachment_contract_id=contract_ids[0],
                closure_contract_ids=tuple(contract_ids[1:]),
            ))

    plan = FlexAssemblyPlan(
        anchor_rigid_component_unit_ids=rigid_components[anchor_index],
        rigid_components=rigid_components,
        backbone_topology_unit_ids=tuple(backbone_units),
        backbone_contract_ids=tuple(backbone_contracts),
        backbone_steps=tuple(backbone_steps),
        interior_steps=tuple(interior_steps),
    )
    expected_ids = {str(contract.contract_id) for contract in flex_contracts}
    if set(plan.all_contract_ids) != expected_ids:
        raise InverseInputError(
            "flex plan contract coverage mismatch; "
            f"missing={sorted(expected_ids-set(plan.all_contract_ids))} "
            f"unexpected={sorted(set(plan.all_contract_ids)-expected_ids)}"
        )
    return plan


def _contracts_by_topology(
    state: HexAssemblyState,
) -> Dict[str, Tuple[ExternalContract, ...]]:
    values: Dict[str, List[ExternalContract]] = {}
    for contract in state.problem.contracts:
        if contract.kind == "flex":
            values.setdefault(_topology_unit_id(state, contract), []).append(contract)
    return {
        topology_id: tuple(sorted(rows, key=lambda row: str(row.contract_id)))
        for topology_id, rows in values.items()
    }


def _port_for_contract(
    state: HexAssemblyState,
    contract: ExternalContract,
    unit_id: str,
):
    if str(contract.source_unit_id) == str(unit_id):
        port_id = str(contract.source_port_id)
    elif str(contract.target_unit_id) == str(unit_id):
        port_id = str(contract.target_port_id)
    else:
        raise InverseInputError(
            f"contract {contract.contract_id!r} does not touch {unit_id!r}"
        )
    port = state._ports.get(f"{unit_id}:{port_id}")
    if port is None:
        raise InverseInputError(
            f"contract {contract.contract_id!r} has no port on {unit_id!r}"
        )
    return port


def _component_map(plan: FlexAssemblyPlan) -> Dict[str, int]:
    return {
        unit_id: index
        for index, component in enumerate(plan.rigid_components)
        for unit_id in component
    }


def _contract_component(
    state: HexAssemblyState,
    plan: FlexAssemblyPlan,
    contract: ExternalContract,
) -> int:
    aromatic_id, _node_id = _aromatic_endpoint(state, contract)
    return _component_map(plan)[aromatic_id]


def _bridge_body_length(unit: TopologyUnit) -> int:
    if len(unit.ports) < 2:
        return len(unit.nodes)
    adjacency: Dict[int, Set[int]] = {
        int(node.node_id): set() for node in unit.nodes
    }
    for edge in unit.internal_edges:
        left, right = int(edge.source_node_id), int(edge.target_node_id)
        adjacency[left].add(right)
        adjacency[right].add(left)
    endpoints = [int(port.node_id) for port in unit.ports]
    longest = 1
    for start in endpoints:
        queue = [(start, 1)]
        seen = {start}
        for node_id, depth in queue:
            longest = max(longest, depth)
            for neighbor in adjacency.get(node_id, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return longest


def _pose_is_vertical(
    unit: TopologyUnit,
    rotation: int,
    reflection: bool,
) -> bool:
    virtual = []
    nodes = unit.node_map()
    for port in unit.ports:
        if port.direction is None:
            continue
        endpoint = axial_add(nodes[int(port.node_id)].local_axial, port.direction)
        virtual.append(transform_axial(endpoint, rotation, reflection))
    if len(virtual) < 2:
        return False
    pairs = [
        (left, right)
        for left_index, left in enumerate(virtual)
        for right in virtual[left_index + 1:]
    ]
    left, right = max(
        pairs,
        key=lambda pair: (
            pair[1][0] - pair[0][0]
        ) ** 2 + (
            pair[1][1] - pair[0][1]
        ) ** 2,
    )
    dq, dr = right[0] - left[0], right[1] - left[1]
    x = math.sqrt(3.0) * 0.5 * dq
    y = dr - 0.5 * dq
    return abs(y) > abs(x)


def _pose_allowed(
    unit: TopologyUnit,
    rotation: int,
    reflection: bool,
) -> bool:
    # Every immutable local topology may use the full dihedral symmetry of the
    # hex grid.  Molecular q/r aspect is a global search objective; filtering
    # an individual long bridge by its screen orientation can remove the only
    # collision-free placement at a fixed aromatic port.
    del unit, rotation, reflection
    return True


def _cart(axial: Tuple[int, int]) -> Tuple[float, float]:
    return axial_to_cartesian(axial)


def _aromatic_slot_direction(
    unit: TopologyUnit,
    node_id: int,
) -> Tuple[int, int]:
    return aromatic_slot_direction(unit, node_id)


def _slot_pose_options(
    state: HexAssemblyState,
    contract: ExternalContract,
    donor_node_id: int,
) -> List[Dict[str, Any]]:
    aromatic_id, _boundary_node_id = _aromatic_endpoint(state, contract)
    topology_id, topology_node_id = _topology_endpoint(state, contract)
    aromatic_unit = state.unit(aromatic_id)
    topology_unit = state.unit(topology_id)
    topology_node = topology_unit.node_map()[topology_node_id]
    topology_port = _port_for_contract(state, contract, topology_id)
    if topology_port.direction is None:
        _aromatic_id, boundary_node_id = _aromatic_endpoint(state, contract)
        return ([{
            "donor_node_id": int(donor_node_id),
            "rotation": 0,
            "reflection": False,
            "translation": (0, 0),
            "coordinates": frozenset(),
            "vertical": False,
        }] if int(donor_node_id) == int(boundary_node_id) else [])
    placement = state.placements[aromatic_id]
    local_outward = _aromatic_slot_direction(
        aromatic_unit, donor_node_id
    )
    outward = transform_axial(
        local_outward, placement.rotation, placement.reflection
    )
    boundary_axial = state.node_axial[int(donor_node_id)]
    outcomes = []
    for reflection in (False, True):
        for rotation in range(6):
            if not _pose_allowed(topology_unit, rotation, reflection):
                continue
            port_direction = transform_axial(
                topology_port.direction, rotation, reflection
            )
            if outward != axial_neg(port_direction):
                continue
            transformed_node = transform_axial(
                topology_node.local_axial, rotation, reflection
            )
            desired_node = axial_add(boundary_axial, outward)
            translation = (
                desired_node[0] - transformed_node[0],
                desired_node[1] - transformed_node[1],
            )
            ok, _reason = state.can_place(
                topology_id,
                rotation,
                translation,
                allowed_contacts=[(topology_node_id, donor_node_id)],
                reflection=reflection,
            )
            if not ok:
                continue
            coordinates = state._placement_coordinates(
                topology_unit, rotation, translation, reflection
            )
            outcomes.append({
                "donor_node_id": int(donor_node_id),
                "rotation": int(rotation),
                "reflection": bool(reflection),
                "translation": tuple(translation),
                "coordinates": frozenset(coordinates.values()),
                "vertical": _pose_is_vertical(
                    topology_unit, rotation, reflection
                ),
            })
    return outcomes


def _remap_aromatic_unit(
    unit: TopologyUnit,
    target_slot_by_boundary: Mapping[int, int],
    fixed_node_ids: Set[int],
) -> TopologyUnit:
    if not target_slot_by_boundary:
        return unit
    node_map = unit.node_map()
    movable_ids = [
        int(node.node_id) for node in unit.nodes
        if int(node.node_id) not in fixed_node_ids
        and int(node.su_type) != 12
    ]
    target_coordinates = {
        int(boundary_id): tuple(node_map[int(donor_id)].local_axial)
        for boundary_id, donor_id in target_slot_by_boundary.items()
    }
    if len(set(target_coordinates.values())) != len(target_coordinates):
        raise InverseInputError(
            f"duplicate optimized flex slots on {unit.unit_id!r}"
        )
    remaining_ids = sorted(
        set(movable_ids) - set(target_coordinates)
    )
    remaining_coordinates = sorted(
        {
            tuple(node_map[node_id].local_axial)
            for node_id in movable_ids
        } - set(target_coordinates.values())
    )
    if len(remaining_ids) != len(remaining_coordinates):
        raise InverseInputError(
            f"cannot complete aromatic slot permutation for {unit.unit_id!r}"
        )
    new_coordinate = dict(target_coordinates)
    new_coordinate.update(zip(remaining_ids, remaining_coordinates))
    for node in unit.nodes:
        new_coordinate.setdefault(int(node.node_id), tuple(node.local_axial))

    old_id_by_coordinate = {
        tuple(node.local_axial): int(node.node_id) for node in unit.nodes
    }
    new_id_by_coordinate = {
        coordinate: node_id for node_id, coordinate in new_coordinate.items()
    }
    new_edges = []
    for edge in unit.internal_edges:
        left_coord = tuple(
            node_map[int(edge.source_node_id)].local_axial
        )
        right_coord = tuple(
            node_map[int(edge.target_node_id)].local_axial
        )
        new_edges.append(replace(
            edge,
            source_node_id=int(new_id_by_coordinate[left_coord]),
            target_node_id=int(new_id_by_coordinate[right_coord]),
        ))
    new_nodes = tuple(
        replace(node, local_axial=new_coordinate[int(node.node_id)])
        for node in unit.nodes
    )
    provisional = replace(unit, nodes=new_nodes, internal_edges=tuple(new_edges))
    new_ports = tuple(
        replace(
            port,
            direction=_aromatic_slot_direction(
                provisional, int(port.node_id)
            ),
        )
        for port in unit.ports
    )
    del old_id_by_coordinate
    return replace(provisional, ports=new_ports)


def _rebuild_state_with_problem(
    state: HexAssemblyState,
    problem: InverseAssemblyInput,
) -> HexAssemblyState:
    rebuilt = HexAssemblyState(problem=problem)
    for unit_id, placement in sorted(state.placements.items()):
        allowed = []
        for contract in problem.contracts:
            if (
                str(contract.contract_id) not in state.satisfied_contracts
                or contract.kind != "rigid"
            ):
                continue
            if str(contract.source_unit_id) == unit_id:
                other_id = str(contract.target_unit_id)
            elif str(contract.target_unit_id) == unit_id:
                other_id = str(contract.source_unit_id)
            else:
                continue
            if other_id in rebuilt.placements:
                allowed.append((
                    int(contract.source_node_id), int(contract.target_node_id)
                ))
        if not rebuilt.place_unit(
            unit_id,
            placement.rotation,
            placement.translation,
            allowed_contacts=allowed,
            reflection=placement.reflection,
        ):
            raise InverseInputError(
                f"optimized aromatic geometry cannot rebuild {unit_id!r}"
            )
    for contract in problem.contracts:
        if (
            contract.kind == "rigid"
            and str(contract.contract_id) in state.satisfied_contracts
            and not rebuilt.force_connect_contract(
                contract, "rebuild_after_flex_slot_optimization", geometry_fallback=False
            )
        ):
            raise InverseInputError(
                f"cannot restore rigid contract {contract.contract_id!r}"
            )
    rebuilt.current_su_types = dict(state.current_su_types)
    rebuilt.stage = "flex"
    rebuilt.action_history = copy.deepcopy(state.action_history)
    return rebuilt


def _optimize_flex_boundary_slots(
    state: HexAssemblyState,
) -> Tuple[HexAssemblyState, Dict[str, Any]]:
    """Assign fixed flex boundary GIDs to usable slots in their own unit."""
    if state.problem.source_kind != InputSourceKind.CSV_PAIR:
        return state.clone(), {"aromatic_flex_slot_assignments": []}
    by_aromatic: Dict[str, List[Tuple[int, ExternalContract]]] = {}
    for contract in state.problem.contracts:
        if contract.kind != "flex":
            continue
        aromatic_id, boundary_node_id = _aromatic_endpoint(state, contract)
        by_aromatic.setdefault(aromatic_id, []).append((
            boundary_node_id, contract
        ))
    rigid_node_ids = {
        int(node_id)
        for contract in state.problem.contracts
        if contract.kind == "rigid"
        for node_id in (contract.source_node_id, contract.target_node_id)
    }
    mappings: Dict[str, Dict[int, int]] = {}
    diagnostics = []
    for aromatic_id, boundary_rows in sorted(by_aromatic.items()):
        unit = state.unit(aromatic_id)
        donor_ids = [
            int(node.node_id) for node in unit.nodes
            if int(state.current_node_map().get(int(node.node_id), -1)) == 13
            and int(node.node_id) not in rigid_node_ids
        ]
        options: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        contract_by_boundary = {
            int(node_id): contract for node_id, contract in boundary_rows
        }
        for boundary_id, contract in boundary_rows:
            rows = []
            for donor_id in donor_ids:
                for pose in _slot_pose_options(state, contract, donor_id):
                    rows.append((int(donor_id), pose))
            options[int(boundary_id)] = rows
        ordered = sorted(
            options,
            key=lambda node_id: (len(options[node_id]), node_id),
        )
        best: Tuple[Tuple[Any, ...], Dict[int, int]] | None = None

        def assign(
            index: int,
            selected: Dict[int, int],
            used_donors: Set[int],
            path_coordinates: List[frozenset[Tuple[int, int]]],
            pose_ranks: List[Tuple[Any, ...]],
        ) -> None:
            nonlocal best
            if index >= len(ordered):
                rank = tuple(pose_ranks) + (tuple(sorted(selected.items())),)
                if best is None or rank < best[0]:
                    best = rank, dict(selected)
                return
            boundary_id = ordered[index]
            seen_donors = set()
            for donor_id, pose in options[boundary_id]:
                if donor_id in used_donors or donor_id in seen_donors:
                    continue
                seen_donors.add(donor_id)
                coordinates = pose["coordinates"]
                if any(coordinates & existing for existing in path_coordinates):
                    continue
                selected[boundary_id] = donor_id
                assign(
                    index + 1,
                    selected,
                    used_donors | {donor_id},
                    path_coordinates + [coordinates],
                    pose_ranks + [(
                        int(bool(pose["vertical"])),
                        int(bool(pose["reflection"])),
                        int(pose["rotation"]),
                        int(donor_id),
                    )],
                )
                selected.pop(boundary_id, None)

        assign(0, {}, set(), [], [])
        if best is None:
            missing = {
                boundary_id: len(rows)
                for boundary_id, rows in options.items()
            }
            raise InverseInputError(
                f"no collision-free flex boundary slots on {aromatic_id!r}: {missing}"
            )
        mappings[aromatic_id] = best[1]
        diagnostics.append({
            "aromatic_unit_id": aromatic_id,
            "boundary_to_slot_node": dict(best[1]),
            "option_counts": {
                str(node_id): len(rows) for node_id, rows in options.items()
            },
        })

    units = tuple(
        _remap_aromatic_unit(
            unit,
            mappings.get(str(unit.unit_id), {}),
            rigid_node_ids,
        )
        if unit.phase == "aromatic"
        else unit
        for unit in state.problem.units
    )
    problem = replace(state.problem, units=units)
    return _rebuild_state_with_problem(state, problem), {
        "aromatic_flex_slot_assignments": diagnostics,
    }


def _force_contracts(
    state: HexAssemblyState,
    contracts: Sequence[ExternalContract],
    reason: str,
) -> bool:
    snapshot = state.clone()
    for contract in contracts:
        if str(contract.contract_id) in state.satisfied_contracts:
            continue
        if not state.force_connect_contract(
            contract, reason, geometry_fallback=False
        ):
            state._restore(snapshot)
            return False
    return True


def _place_topology_outcomes(
    state: HexAssemblyState,
    topology_unit_id: str,
    contracts: Sequence[ExternalContract],
    max_candidates: int,
) -> List[Tuple[HexAssemblyState, List[Dict[str, Any]]]]:
    """Place one immutable topology against every already-fixed endpoint."""
    if not contracts:
        return []
    if state.is_placed(topology_unit_id):
        trial = state.clone()
        if _force_contracts(
            trial, contracts, "fixed_flex_existing_topology"
        ):
            return [(trial, [])]
        return []

    unit = state.unit(topology_unit_id)
    outcomes = []
    seen = set()
    for reflection in (False, True):
        for rotation in range(6):
            if not _pose_allowed(unit, rotation, reflection):
                continue
            translation_options: Set[Tuple[int, int]] | None = None
            contacts = []
            valid = True
            for contract in contracts:
                aromatic_id, aromatic_node_id = _aromatic_endpoint(
                    state, contract
                )
                _topology_id, topology_node_id = _topology_endpoint(
                    state, contract
                )
                aromatic_axial = state.node_axial.get(aromatic_node_id)
                topology_node = unit.node_map().get(topology_node_id)
                topology_port = _port_for_contract(
                    state, contract, topology_unit_id
                )
                aromatic_port = _port_for_contract(
                    state, contract, aromatic_id
                )
                if aromatic_axial is None or topology_node is None:
                    valid = False
                    break
                transformed_node = transform_axial(
                    topology_node.local_axial, rotation, reflection
                )
                aromatic_direction = state.global_port_direction(
                    aromatic_id, aromatic_port
                )
                if topology_port.direction is not None:
                    transformed_port = transform_axial(
                        topology_port.direction, rotation, reflection
                    )
                    desired_nodes = {
                        axial_add(aromatic_axial, axial_neg(transformed_port))
                    }
                    if (
                        aromatic_direction is not None
                        and aromatic_direction != axial_neg(transformed_port)
                    ):
                        valid = False
                        break
                elif aromatic_direction is not None:
                    desired_nodes = {
                        axial_add(aromatic_axial, aromatic_direction)
                    }
                else:
                    desired_nodes = {
                        axial_add(aromatic_axial, direction)
                        for direction in HEX_DIRECTIONS
                    }
                options = {
                    (
                        desired_node[0] - transformed_node[0],
                        desired_node[1] - transformed_node[1],
                    )
                    for desired_node in desired_nodes
                }
                translation_options = (
                    options
                    if translation_options is None
                    else translation_options & options
                )
                contacts.append((topology_node_id, aromatic_node_id))
            if not valid or not translation_options:
                continue
            for translation in sorted(translation_options):
                trial = state.clone()
                if not trial.place_unit(
                    topology_unit_id,
                    rotation,
                    translation,
                    allowed_contacts=contacts,
                    reflection=reflection,
                    allow_unbonded_neighbors=True,
                ):
                    continue
                if not _force_contracts(
                    trial, contracts, "fixed_flex_topology_placement"
                ):
                    continue
                signature = trial.state_signature()
                if signature in seen:
                    continue
                seen.add(signature)
                action = {
                    "kind": "flex",
                    "topology_unit_id": str(topology_unit_id),
                    "contract_ids": [str(row.contract_id) for row in contracts],
                    "topology_rotation": int(rotation),
                    "topology_reflection": bool(reflection),
                    "topology_translation": [
                        int(translation[0]), int(translation[1])
                    ],
                    "topology_reembedded": False,
                    "fixed_local_geometry": True,
                    "vertical_pose": _pose_is_vertical(
                        unit, rotation, reflection
                    ),
                }
                rank = (
                    int(action["vertical_pose"]),
                    int(bool(reflection)),
                    int(rotation),
                    abs(int(translation[0])) + abs(int(translation[1])),
                )
                outcomes.append((rank, trial, [action]))
    outcomes.sort(key=lambda item: item[0])
    return [
        (trial, actions)
        for _rank, trial, actions in outcomes[:max(1, int(max_candidates))]
    ]


def _component_clearance(
    state: HexAssemblyState,
    component_unit_id: str,
) -> Tuple[int, int]:
    component_units = state.component_unit_ids(component_unit_id)
    free_ports = 0
    free_neighbors = 0
    for contract in state.problem.contracts:
        if (
            contract.kind != "flex"
            or str(contract.contract_id) in state.satisfied_contracts
        ):
            continue
        if str(contract.source_unit_id) in component_units:
            unit_id = str(contract.source_unit_id)
            node_id = int(contract.source_node_id)
            port_id = str(contract.source_port_id)
        elif str(contract.target_unit_id) in component_units:
            unit_id = str(contract.target_unit_id)
            node_id = int(contract.target_node_id)
            port_id = str(contract.target_port_id)
        else:
            continue
        axial = state.node_axial.get(node_id)
        port = state._ports.get(f"{unit_id}:{port_id}")
        if axial is None or port is None:
            continue
        direction = state.global_port_direction(unit_id, port)
        directions = (direction,) if direction is not None else ()
        available = sum(
            axial_add(axial, value) not in state.occupied_axial
            for value in directions
        )
        free_neighbors += int(available)
        free_ports += int(available > 0)
    return free_ports, free_neighbors


def _move_component_outcomes(
    state: HexAssemblyState,
    topology_unit_id: str,
    component_unit_id: str,
    contracts: Sequence[ExternalContract],
    max_candidates: int,
) -> List[Tuple[HexAssemblyState, List[Dict[str, Any]]]]:
    """Move one whole rigid component to all prescribed topology ports."""
    if not contracts or not state.is_placed(topology_unit_id):
        return []
    if component_unit_id in state.component_unit_ids(topology_unit_id):
        trial = state.clone()
        return (
            [(trial, [])]
            if _force_contracts(
                trial, contracts, "fixed_flex_existing_component"
            )
            else []
        )

    outcomes = []
    seen = set()
    for reflection in (False, True):
        for rotation in range(6):
            translation_options: Set[Tuple[int, int]] | None = None
            contacts = []
            valid = True
            for contract in contracts:
                aromatic_id, aromatic_node_id = _aromatic_endpoint(
                    state, contract
                )
                _topology_id, topology_node_id = _topology_endpoint(
                    state, contract
                )
                if aromatic_id not in state.component_unit_ids(component_unit_id):
                    valid = False
                    break
                topology_axial = state.node_axial.get(topology_node_id)
                aromatic_axial = state.node_axial.get(aromatic_node_id)
                topology_port = _port_for_contract(
                    state, contract, topology_unit_id
                )
                aromatic_port = _port_for_contract(
                    state, contract, aromatic_id
                )
                topology_direction = state.global_port_direction(
                    topology_unit_id, topology_port
                )
                aromatic_direction = state.global_port_direction(
                    aromatic_id, aromatic_port
                )
                if topology_axial is None or aromatic_axial is None:
                    valid = False
                    break
                transformed_aromatic = transform_axial(
                    aromatic_axial, rotation, reflection
                )
                transformed_aromatic_direction = (
                    transform_axial(
                        aromatic_direction, rotation, reflection
                    )
                    if aromatic_direction is not None
                    else None
                )
                if topology_direction is not None:
                    desired_aromatics = {
                        axial_add(topology_axial, topology_direction)
                    }
                    if (
                        transformed_aromatic_direction is not None
                        and transformed_aromatic_direction
                        != axial_neg(topology_direction)
                    ):
                        valid = False
                        break
                elif transformed_aromatic_direction is not None:
                    desired_aromatics = {
                        axial_add(
                            topology_axial,
                            axial_neg(transformed_aromatic_direction),
                        )
                    }
                else:
                    desired_aromatics = {
                        axial_add(topology_axial, direction)
                        for direction in HEX_DIRECTIONS
                    }
                options = {
                    (
                        desired_aromatic[0] - transformed_aromatic[0],
                        desired_aromatic[1] - transformed_aromatic[1],
                    )
                    for desired_aromatic in desired_aromatics
                }
                translation_options = (
                    options
                    if translation_options is None
                    else translation_options & options
                )
                contacts.append((topology_node_id, aromatic_node_id))
            if not valid or not translation_options:
                continue
            for translation in sorted(translation_options):
                trial = state.clone()
                if not trial.transform_component(
                    component_unit_id,
                    rotation,
                    translation,
                    allowed_contacts=contacts,
                    reflection=reflection,
                    allow_unbonded_neighbors=True,
                ):
                    continue
                if not _force_contracts(
                    trial, contracts, "fixed_flex_component_merge"
                ):
                    continue
                signature = trial.state_signature()
                if signature in seen:
                    continue
                seen.add(signature)
                free_ports, free_neighbors = _component_clearance(
                    trial, component_unit_id
                )
                all_axials = list(trial.node_axial.values())
                q_span = max(value[0] for value in all_axials) - min(
                    value[0] for value in all_axials
                )
                r_span = max(value[1] for value in all_axials) - min(
                    value[1] for value in all_axials
                )
                action = {
                    "kind": "flex",
                    "topology_unit_id": str(topology_unit_id),
                    "contract_ids": [str(row.contract_id) for row in contracts],
                    "move_component_unit_id": str(component_unit_id),
                    "component_rotation": int(rotation),
                    "component_reflection": bool(reflection),
                    "component_translation": [
                        int(translation[0]), int(translation[1])
                    ],
                    "topology_reembedded": False,
                    "fixed_local_geometry": True,
                }
                rank = (
                    -int(free_ports),
                    -int(free_neighbors),
                    int(q_span + r_span),
                    int(bool(reflection)),
                    int(rotation),
                )
                outcomes.append((rank, trial, [action]))
    outcomes.sort(key=lambda item: item[0])
    return [
        (trial, actions)
        for _rank, trial, actions in outcomes[:max(1, int(max_candidates))]
    ]


def _connected_component_indices(
    state: HexAssemblyState,
    plan: FlexAssemblyPlan,
) -> Set[int]:
    connected_units = state.component_unit_ids(
        plan.anchor_rigid_component_unit_ids[0]
    )
    return {
        index
        for index, component in enumerate(plan.rigid_components)
        if any(unit_id in connected_units for unit_id in component)
    }


def _topology_component_indices(
    state: HexAssemblyState,
    plan: FlexAssemblyPlan,
    contracts: Sequence[ExternalContract],
) -> Set[int]:
    return {
        _contract_component(state, plan, contract)
        for contract in contracts
    }


def _backbone_step_outcomes(
    state: HexAssemblyState,
    plan: FlexAssemblyPlan,
    topology_unit_id: str,
    contracts: Sequence[ExternalContract],
    max_candidates: int,
) -> List[Tuple[HexAssemblyState, List[Dict[str, Any]]]]:
    connected = _connected_component_indices(state, plan)
    grouped: Dict[int, List[ExternalContract]] = {}
    for contract in contracts:
        grouped.setdefault(
            _contract_component(state, plan, contract), []
        ).append(contract)
    inside = [
        contract
        for component_index in sorted(set(grouped) & connected)
        for contract in grouped[component_index]
    ]
    outside = [
        (component_index, tuple(grouped[component_index]))
        for component_index in sorted(set(grouped) - connected)
    ]
    if not outside:
        return _place_topology_outcomes(
            state, topology_unit_id, contracts, max_candidates
        )
    if not inside:
        return []

    attachment_outcomes = _place_topology_outcomes(
        state, topology_unit_id, inside, max_candidates
    )
    current_outcomes = attachment_outcomes
    for component_index, component_contracts in outside:
        next_outcomes = []
        representative = plan.rigid_components[component_index][0]
        for trial, actions in current_outcomes:
            for moved, moved_actions in _move_component_outcomes(
                trial,
                topology_unit_id,
                representative,
                component_contracts,
                max_candidates,
            ):
                next_outcomes.append((moved, actions + moved_actions))
        current_outcomes = next_outcomes[:max(1, int(max_candidates))]
        if not current_outcomes:
            return []

    complete = []
    for trial, actions in current_outcomes:
        if _force_contracts(
            trial, contracts, "fixed_flex_complete_topology"
        ):
            complete.append((trial, actions))
    return complete[:max(1, int(max_candidates))]


def _close_eligible_outcomes(
    state: HexAssemblyState,
    plan: FlexAssemblyPlan,
    by_topology: Mapping[str, Sequence[ExternalContract]],
    max_candidates: int,
) -> List[Tuple[HexAssemblyState, List[Dict[str, Any]]]]:
    """Place every bridge whose rigid endpoints are already in one component."""
    results: List[Tuple[HexAssemblyState, List[Dict[str, Any]]]] = []

    def solve(
        current: HexAssemblyState,
        actions: List[Dict[str, Any]],
    ) -> None:
        if len(results) >= max(1, int(max_candidates)):
            return
        connected = _connected_component_indices(current, plan)
        eligible = []
        for topology_id, contracts in by_topology.items():
            pending = [
                row for row in contracts
                if str(row.contract_id) not in current.satisfied_contracts
            ]
            if not pending:
                continue
            components = _topology_component_indices(
                current, plan, contracts
            )
            if components and components <= connected:
                placement_outcomes = _place_topology_outcomes(
                    current,
                    topology_id,
                    contracts,
                    max_candidates=max_candidates,
                )
                if not placement_outcomes:
                    # Every aromatic endpoint is already in the rooted rigid
                    # component, so no later backbone attachment can change
                    # their relative pose.  A fixed topology that cannot close
                    # now can never close in a descendant state.
                    return
                eligible.append((
                    len(placement_outcomes),
                    -len(contracts),
                    _unit_sort_key(topology_id),
                    topology_id,
                    placement_outcomes,
                ))
        if not eligible:
            results.append((current, actions))
            return
        _count, _neg_contracts, _key, _topology_id, outcomes = min(eligible)
        for trial, selected in outcomes:
            solve(trial, actions + selected)

    solve(state, [])
    return results


@dataclass
class _SiteSearchCandidate:
    state: HexAssemblyState
    used_site_ids: frozenset[int]
    boundary_site_ids: Dict[int, int]
    active_component_indices: frozenset[int]
    actions: List[Dict[str, Any]]
    nmr_score: Dict[str, Any] | None = None
    nmr_rank: float | None = None


def _rigid_endpoint_ids(state: HexAssemblyState) -> Set[int]:
    return {
        int(node_id)
        for contract in state.problem.contracts
        if contract.kind == "rigid"
        for node_id in (contract.source_node_id, contract.target_node_id)
    }


def _component_slot_records(
    state: HexAssemblyState,
    component: Sequence[str],
    used_site_ids: Iterable[int],
) -> List[Tuple[str, int, Tuple[int, int], Tuple[int, int]]]:
    """Return free physical SU13 sites with their global outward vectors."""
    used = {int(value) for value in used_site_ids}
    rigid_ids = _rigid_endpoint_ids(state)
    records = []
    for unit_id in sorted(component, key=_unit_sort_key):
        unit = state.unit(unit_id)
        placement = state.placements[unit_id]
        for node in unit.nodes:
            node_id = int(node.node_id)
            if int(node.su_type) != 13 or node_id in rigid_ids or node_id in used:
                continue
            outward = transform_axial(
                _aromatic_slot_direction(unit, node_id),
                placement.rotation,
                placement.reflection,
            )
            records.append((
                str(unit_id),
                node_id,
                tuple(state.node_axial[node_id]),
                tuple(outward),
            ))
    return sorted(records, key=lambda row: (_unit_sort_key(row[0]), row[1]))


def _flex_port_records(
    state: HexAssemblyState,
    topology_unit_id: str,
) -> List[Tuple[ExternalContract, int, int, Any]]:
    records = []
    for contract in state.problem.contracts:
        if contract.kind != "flex":
            continue
        if _topology_unit_id(state, contract) != str(topology_unit_id):
            continue
        _aromatic_id, boundary_id = _aromatic_endpoint(state, contract)
        _topology_id, topology_node_id = _topology_endpoint(state, contract)
        records.append((
            contract,
            int(topology_node_id),
            int(boundary_id),
            _port_for_contract(state, contract, topology_unit_id),
        ))
    return sorted(records, key=lambda row: str(row[0].contract_id))


def _flex_port_groups(
    state: HexAssemblyState,
    topology_unit_id: str,
) -> List[List[Tuple[ExternalContract, int, int, Any]]]:
    """Group adjacent virtual aromatic boundaries into one attachment side."""
    records = _flex_port_records(state, topology_unit_id)
    unit = state.unit(topology_unit_id)
    virtual = []
    for record in records:
        _contract, node_id, _boundary_id, port = record
        if port.direction is None:
            virtual.append(None)
        else:
            virtual.append(axial_add(
                unit.node_map()[node_id].local_axial,
                port.direction,
            ))
    adjacency = {index: set() for index in range(len(records))}
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if (
                virtual[left] is not None
                and virtual[right] is not None
                and axial_distance(virtual[left], virtual[right]) == 1
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    groups = []
    remaining = set(range(len(records)))
    while remaining:
        root = min(remaining)
        stack = [root]
        selected = {root}
        remaining.remove(root)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                selected.add(neighbor)
                stack.append(neighbor)
        groups.append([records[index] for index in sorted(selected)])
    return groups


def _append_site_contract(
    state: HexAssemblyState,
    contract: ExternalContract,
    topology_node_id: int,
    physical_site_id: int,
) -> None:
    direction = state._edge_direction(
        state.node_axial[int(topology_node_id)],
        state.node_axial[int(physical_site_id)],
    )
    if direction is None:
        raise InverseInputError(
            f"site contract {contract.contract_id!r} is not a hex neighbor"
        )
    state._append_edge(
        int(topology_node_id),
        int(physical_site_id),
        "flex",
        str(contract.contract_id),
        direction,
    )
    state.satisfied_contracts.add(str(contract.contract_id))


def _occupied_span(state: HexAssemblyState) -> Tuple[int, int, int]:
    shape = coordinate_aspect(state.occupied_axial)
    return (
        int(shape["total_span"]),
        int(shape["q_span"]),
        int(shape["r_span"]),
    )


def _coordinate_aspect(state: HexAssemblyState) -> Dict[str, Any]:
    return coordinate_aspect(state.occupied_axial)


def _state_shape_rank(state: HexAssemblyState) -> Tuple[Any, ...]:
    return state_shape_rank(state)


def _site_candidate_rank(candidate: _SiteSearchCandidate) -> Tuple[Any, ...]:
    nmr_prefix: Tuple[Any, ...] = ()
    if candidate.nmr_rank is not None and candidate.nmr_score is not None:
        nmr_prefix = (
            float(candidate.nmr_rank),
            float(candidate.nmr_score["loss"]),
        )
    return (
        *nmr_prefix,
        *_state_shape_rank(candidate.state),
        tuple(sorted(candidate.boundary_site_ids.items())),
    )


def _score_site_candidates(
    candidates: Sequence[_SiteSearchCandidate],
    *,
    nmr_scorer: Any,
    depth: int,
    nmr_weight: float,
    geometry_weight: float,
    contact_weight: float,
    protected_aromatic_ids: Iterable[int],
) -> None:
    if nmr_scorer is None or not candidates:
        return
    scores = nmr_scorer.score_candidates(
        candidates,
        stage="flex",
        depth=int(depth),
        assignments_getter=lambda candidate: candidate.boundary_site_ids,
        protected_getter=lambda _candidate: protected_aromatic_ids,
    )
    for candidate, score in zip(candidates, scores):
        shape = _coordinate_aspect(candidate.state)
        shape_penalty = (
            float(shape["violation"])
            + 0.10 * abs(float(shape["q_over_r"]) - 1.0)
            + 0.05 * abs(float(shape["width_over_height"]) - 1.0)
        )
        contact_penalty = len(candidate.state.allowed_unbonded_contacts)
        candidate.nmr_score = dict(score)
        candidate.nmr_rank = (
            float(nmr_weight) * float(score["loss"])
            + float(geometry_weight) * float(shape_penalty)
            + float(contact_weight) * float(contact_penalty)
        )


def _normalize_global_orientation(
    candidate: _SiteSearchCandidate,
) -> Tuple[_SiteSearchCandidate, int]:
    """Choose the equivalent whole-molecule hex rotation with best framing."""
    if not candidate.state.placements:
        return candidate, 0
    representative = min(candidate.state.placements, key=_unit_sort_key)
    options = []
    for rotation in range(6):
        trial = candidate.state.clone()
        if rotation and not trial.transform_component(
            representative,
            rotation,
            (0, 0),
        ):
            continue
        shape = _coordinate_aspect(trial)
        rank = (
            float(shape["violation"]),
            abs(float(shape["width_over_height"]) - 1.0),
            abs(float(shape["q_over_r"]) - 1.0),
            max(int(shape["q_span"]), int(shape["r_span"])),
            int(rotation),
        )
        options.append((rank, trial, int(rotation)))
    if not options:
        return candidate, 0
    _rank, state, rotation = min(options, key=lambda row: row[0])
    return _SiteSearchCandidate(
        state=state,
        used_site_ids=candidate.used_site_ids,
        boundary_site_ids=dict(candidate.boundary_site_ids),
        active_component_indices=candidate.active_component_indices,
        actions=list(candidate.actions),
        nmr_score=(dict(candidate.nmr_score) if candidate.nmr_score else None),
        nmr_rank=candidate.nmr_rank,
    ), rotation


def _su25_vinyl_terminal_ids(unit: TopologyUnit) -> Set[int]:
    """Return SU16 termini of the exceptional ``25-15-16`` branch."""
    node_su = {
        int(node.node_id): int(node.su_type)
        for node in unit.nodes
    }
    adjacency: Dict[int, Set[int]] = {
        node_id: set() for node_id in node_su
    }
    for edge in unit.internal_edges:
        left = int(edge.source_node_id)
        right = int(edge.target_node_id)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return {
        node_id
        for node_id, su_type in node_su.items()
        if su_type == 16
        and any(
            node_su.get(neighbor) == 15
            and any(
                node_su.get(grandparent) == 25
                for grandparent in adjacency.get(neighbor, set())
                if grandparent != node_id
            )
            for neighbor in adjacency.get(node_id, set())
        )
    }


def _su25_placement_contact_pairs(
    state: HexAssemblyState,
    unit: TopologyUnit,
    coordinates: Mapping[int, Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Allow only the inevitable close contacts of a bent SU25 vinyl arm."""
    pairs: Set[Tuple[int, int]] = set()
    for terminal_id in _su25_vinyl_terminal_ids(unit):
        terminal_axial = coordinates[int(terminal_id)]
        for direction in HEX_DIRECTIONS:
            occupant = state.occupied_axial.get(
                axial_add(terminal_axial, direction)
            )
            if occupant is not None:
                pairs.add((int(terminal_id), int(occupant)))
    return pairs


def _su25_component_contact_pairs(
    state: HexAssemblyState,
    topology_unit: TopologyUnit,
    component_unit_id: str,
    rotation: int,
    translation: Tuple[int, int],
    reflection: bool,
) -> Set[Tuple[int, int]]:
    """Find SU25-terminal contacts created by a proposed component pose."""
    terminals = {
        int(node_id): state.node_axial[int(node_id)]
        for node_id in _su25_vinyl_terminal_ids(topology_unit)
    }
    if not terminals:
        return set()
    pairs: Set[Tuple[int, int]] = set()
    for node_id in state.component_node_ids(component_unit_id):
        transformed = axial_add(
            transform_axial(
                state.node_axial[int(node_id)], rotation, reflection
            ),
            translation,
        )
        for terminal_id, terminal_axial in terminals.items():
            delta = (
                transformed[0] - terminal_axial[0],
                transformed[1] - terminal_axial[1],
            )
            if delta in HEX_DIRECTIONS:
                pairs.add((int(node_id), int(terminal_id)))
    return pairs


def _remember_exact_unbonded_contacts(
    state: HexAssemblyState,
    pairs: Iterable[Tuple[int, int]],
) -> None:
    bonded = {
        frozenset((int(edge.source_node_id), int(edge.target_node_id)))
        for edge in state.edges
    }
    state.allowed_unbonded_contacts.update(
        pair
        for pair in (
            frozenset((int(left), int(right)))
            for left, right in pairs
        )
        if pair not in bonded
    )


def _backbone_site_outcomes(
    candidate: _SiteSearchCandidate,
    topology_unit_id: str,
    outside_component_indices: Sequence[int],
    rigid_components: Sequence[Sequence[str]],
    max_candidates: int,
) -> List[_SiteSearchCandidate]:
    """Attach one topology; adjacent ports move the same rigid component."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    groups = _flex_port_groups(state, topology_unit_id)
    if len(groups) != len(outside_component_indices) + 1:
        return []
    allow_hub_neighbors = len(groups) > 2
    outcomes: List[_SiteSearchCandidate] = []
    seen = set()

    def match_group(
        group: Sequence[Tuple[ExternalContract, int, int, Any]],
        slots: Sequence[Tuple[str, int, Tuple[int, int], Tuple[int, int]]],
        topology_state: HexAssemblyState,
        seed: Mapping[int, int],
        selected: Set[int],
    ) -> List[Tuple[Dict[int, int], Set[int]]]:
        by_geometry: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]],
            List[int],
        ] = {}
        for _unit_id, site_id, axial, outward in slots:
            by_geometry.setdefault((axial, outward), []).append(int(site_id))
        choices = [(dict(seed), set(selected))]
        for _contract, node_id, boundary_id, port in group:
            if int(boundary_id) in seed:
                continue
            direction = topology_state.global_port_direction(
                topology_unit_id, port
            )
            if direction is None:
                return []
            required = axial_add(topology_state.node_axial[node_id], direction)
            matching = by_geometry.get((required, axial_neg(direction)), [])
            next_choices = []
            for assignments, used in choices:
                for site_id in matching:
                    if site_id in used:
                        continue
                    next_choices.append((
                        {**assignments, int(boundary_id): int(site_id)},
                        used | {int(site_id)},
                    ))
            choices = next_choices
            if not choices:
                break
        return choices

    for anchor_group_index, anchor_group in enumerate(groups):
        remaining_groups = [
            group for index, group in enumerate(groups)
            if index != anchor_group_index
        ]
        first_contract, first_node, first_boundary, first_port = anchor_group[0]
        if first_port.direction is None:
            continue
        for active_index in sorted(candidate.active_component_indices):
            anchor_slots = _component_slot_records(
                state,
                rigid_components[active_index],
                candidate.used_site_ids,
            )
            for _uid, first_site, first_axial, first_outward in anchor_slots:
                for reflection in (False, True):
                    for rotation in range(6):
                        if not _pose_allowed(unit, rotation, reflection):
                            continue
                        port_direction = transform_axial(
                            first_port.direction, rotation, reflection
                        )
                        if first_outward != axial_neg(port_direction):
                            continue
                        local = transform_axial(
                            unit.node_map()[first_node].local_axial,
                            rotation,
                            reflection,
                        )
                        desired = axial_add(first_axial, first_outward)
                        translation = (
                            desired[0] - local[0], desired[1] - local[1]
                        )
                        trial = state.clone()
                        # Temporarily place the topology with all possible
                        # group contacts allowed, then retain exact matches.
                        coordinates = trial._placement_coordinates(
                            unit, rotation, translation, reflection
                        )
                        provisional_contacts = [
                            (node_id, site_id)
                            for _c, node_id, _b, port in anchor_group
                            for _su, site_id, site_coord, site_out in anchor_slots
                            if port.direction is not None
                            and site_coord == axial_add(
                                coordinates[node_id],
                                transform_axial(port.direction, rotation, reflection),
                            )
                            and site_out == axial_neg(transform_axial(
                                port.direction, rotation, reflection
                            ))
                        ]
                        su25_contacts = _su25_placement_contact_pairs(
                            trial, unit, coordinates
                        )
                        if not trial.place_unit(
                            topology_unit_id,
                            rotation,
                            translation,
                            allowed_contacts=(
                                provisional_contacts + list(su25_contacts)
                            ),
                            reflection=reflection,
                            allow_unbonded_neighbors=allow_hub_neighbors,
                        ):
                            continue
                        _remember_exact_unbonded_contacts(
                            trial, su25_contacts
                        )
                        anchor_matches = match_group(
                            anchor_group,
                            anchor_slots,
                            trial,
                            {int(first_boundary): int(first_site)},
                            {int(first_site)},
                        )
                        partial = []
                        for assignments, selected_sites in anchor_matches:
                            anchored = trial.clone()
                            for row_contract, row_node, row_boundary, _port in anchor_group:
                                _append_site_contract(
                                    anchored,
                                    row_contract,
                                    row_node,
                                    assignments[row_boundary],
                                )
                            partial.append((anchored, assignments, selected_sites, []))

                        for group, component_index in zip(
                            remaining_groups, outside_component_indices
                        ):
                            next_partial = []
                            row_contract, row_node, row_boundary, row_port = group[0]
                            if row_port.direction is None:
                                partial = []
                                break
                            for partial_state, assignments, selected_sites, moves in partial:
                                topology_direction = partial_state.global_port_direction(
                                    topology_unit_id, row_port
                                )
                                assert topology_direction is not None
                                topology_axial = partial_state.node_axial[row_node]
                                slots = _component_slot_records(
                                    partial_state,
                                    rigid_components[component_index],
                                    set(candidate.used_site_ids) | selected_sites,
                                )
                                representative = rigid_components[component_index][0]
                                for _su, site_id, site_axial, site_outward in slots:
                                    for component_reflection in (False, True):
                                        for component_rotation in range(6):
                                            if transform_axial(
                                                site_outward,
                                                component_rotation,
                                                component_reflection,
                                            ) != axial_neg(topology_direction):
                                                continue
                                            transformed_site = transform_axial(
                                                site_axial,
                                                component_rotation,
                                                component_reflection,
                                            )
                                            desired_site = axial_add(
                                                topology_axial, topology_direction
                                            )
                                            component_translation = (
                                                desired_site[0] - transformed_site[0],
                                                desired_site[1] - transformed_site[1],
                                            )
                                            transformed_slots = [(
                                                slot_unit,
                                                other_site,
                                                axial_add(transform_axial(
                                                    other_axial,
                                                    component_rotation,
                                                    component_reflection,
                                                ), component_translation),
                                                transform_axial(
                                                    other_outward,
                                                    component_rotation,
                                                    component_reflection,
                                                ),
                                            ) for (
                                                slot_unit,
                                                other_site,
                                                other_axial,
                                                other_outward,
                                            ) in slots]
                                            matches = match_group(
                                                group,
                                                transformed_slots,
                                                partial_state,
                                                {int(row_boundary): int(site_id)},
                                                set(selected_sites) | {int(site_id)},
                                            )
                                            for group_assignments, group_sites in matches:
                                                contacts = [
                                                    (node_id, group_assignments[boundary_id])
                                                    for _c, node_id, boundary_id, _p in group
                                                ]
                                                su25_contacts = (
                                                    _su25_component_contact_pairs(
                                                        partial_state,
                                                        unit,
                                                        representative,
                                                        component_rotation,
                                                        component_translation,
                                                        component_reflection,
                                                    )
                                                )
                                                moved = partial_state.clone()
                                                if not moved.transform_component(
                                                    representative,
                                                    component_rotation,
                                                    component_translation,
                                                    allowed_contacts=(
                                                        contacts
                                                        + list(su25_contacts)
                                                    ),
                                                    reflection=component_reflection,
                                                    allow_unbonded_neighbors=allow_hub_neighbors,
                                                ):
                                                    continue
                                                _remember_exact_unbonded_contacts(
                                                    moved, su25_contacts
                                                )
                                                for item_contract, item_node, item_boundary, _p in group:
                                                    _append_site_contract(
                                                        moved,
                                                        item_contract,
                                                        item_node,
                                                        group_assignments[item_boundary],
                                                    )
                                                next_partial.append((
                                                    moved,
                                                    {**assignments, **group_assignments},
                                                    set(group_sites),
                                                    moves + [{
                                                        "component_index": int(component_index),
                                                        "component_unit_id": str(representative),
                                                        "rotation": int(component_rotation),
                                                        "reflection": bool(component_reflection),
                                                        "translation": list(component_translation),
                                                    }],
                                                ))
                            partial = sorted(
                                next_partial,
                                key=lambda row: _state_shape_rank(row[0]),
                            )[:max(1, int(max_candidates))]
                            if not partial:
                                break

                        for final_state, assignments, selected_sites, moves in partial:
                            action = {
                                "kind": "flex",
                                "topology_unit_id": str(topology_unit_id),
                                "contract_ids": [str(row[0].contract_id) for row in records],
                                "attachment_contract_id": str(first_contract.contract_id),
                                "merge_contract_ids": [
                                    str(row[0].contract_id) for row in records
                                    if str(row[0].contract_id) != str(first_contract.contract_id)
                                ],
                                "topology_rotation": int(rotation),
                                "topology_reflection": bool(reflection),
                                "topology_translation": list(translation),
                                "component_moves": moves,
                                "boundary_site_assignments": dict(assignments),
                                "attachment_group_count": len(groups),
                                "planned_flex_backbone_merge": True,
                                "topology_reembedded": False,
                                "fixed_local_geometry": True,
                                "coordinate_aspect": _coordinate_aspect(final_state),
                            }
                            result = _SiteSearchCandidate(
                                state=final_state,
                                used_site_ids=frozenset(
                                    set(candidate.used_site_ids) | selected_sites
                                ),
                                boundary_site_ids={
                                    **candidate.boundary_site_ids, **assignments
                                },
                                active_component_indices=frozenset(
                                    set(candidate.active_component_indices)
                                    | {int(value) for value in outside_component_indices}
                                ),
                                actions=candidate.actions + [action],
                            )
                            signature = (
                                result.state.state_signature(),
                                tuple(sorted(result.boundary_site_ids.items())),
                            )
                            if signature not in seen:
                                seen.add(signature)
                                outcomes.append(result)
                                if len(outcomes) >= max(1, int(max_candidates)):
                                    return sorted(
                                        outcomes, key=_site_candidate_rank
                                    )[:max(1, int(max_candidates))]
    return sorted(outcomes, key=_site_candidate_rank)[:max(1, int(max_candidates))]


def _interior_site_outcomes(
    candidate: _SiteSearchCandidate,
    topology_unit_id: str,
    rigid_components: Sequence[Sequence[str]],
    max_candidates: int,
) -> List[_SiteSearchCandidate]:
    """Place a remaining bridge between already fixed skeleton sites."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    active_units = tuple(
        unit_id
        for index in sorted(candidate.active_component_indices)
        for unit_id in rigid_components[index]
    )
    slots = _component_slot_records(state, active_units, candidate.used_site_ids)
    slot_index: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        List[Tuple[str, int, Tuple[int, int], Tuple[int, int]]],
    ] = {}
    for slot in slots:
        slot_index.setdefault((slot[2], slot[3]), []).append(slot)
    outcomes: List[_SiteSearchCandidate] = []
    seen = set()

    for anchor_index, anchor_record in enumerate(records):
        contract, topology_node_id, boundary_id, topology_port = anchor_record
        if topology_port.direction is None:
            continue
        for _slot_unit, physical_site_id, site_axial, outward in slots:
            for reflection in (False, True):
                for rotation in range(6):
                    if not _pose_allowed(unit, rotation, reflection):
                        continue
                    anchor_direction = transform_axial(
                        topology_port.direction, rotation, reflection
                    )
                    if outward != axial_neg(anchor_direction):
                        continue
                    transformed_anchor = transform_axial(
                        unit.node_map()[topology_node_id].local_axial,
                        rotation,
                        reflection,
                    )
                    desired_anchor = axial_add(site_axial, outward)
                    translation = (
                        desired_anchor[0] - transformed_anchor[0],
                        desired_anchor[1] - transformed_anchor[1],
                    )
                    choices = [({
                        int(boundary_id): int(physical_site_id)
                    }, {int(physical_site_id)})]
                    for row_index, row in enumerate(records):
                        if row_index == anchor_index:
                            continue
                        _row_contract, row_node_id, row_boundary_id, row_port = row
                        if row_port.direction is None:
                            choices = []
                            break
                        row_direction = transform_axial(
                            row_port.direction, rotation, reflection
                        )
                        row_node = axial_add(
                            transform_axial(
                                unit.node_map()[row_node_id].local_axial,
                                rotation,
                                reflection,
                            ),
                            translation,
                        )
                        required_site = axial_add(row_node, row_direction)
                        matching = slot_index.get((
                            required_site, axial_neg(row_direction)
                        ), [])
                        next_choices = []
                        for assignments, selected in choices:
                            for _uid, matching_id, _coord, _outward in matching:
                                if int(matching_id) in selected:
                                    continue
                                next_choices.append((
                                    {
                                        **assignments,
                                        int(row_boundary_id): int(matching_id),
                                    },
                                    selected | {int(matching_id)},
                                ))
                        choices = next_choices
                        if not choices:
                            break
                    for assignments, selected in choices:
                        contacts = [
                            (int(row[1]), int(assignments[int(row[2])]))
                            for row in records
                        ]
                        trial = state.clone()
                        if not trial.place_unit(
                            topology_unit_id,
                            rotation,
                            translation,
                            allowed_contacts=contacts,
                            reflection=reflection,
                        ):
                            continue
                        for row_contract, row_node, row_boundary, _row_port in records:
                            _append_site_contract(
                                trial,
                                row_contract,
                                row_node,
                                assignments[row_boundary],
                            )
                        action = {
                            "kind": "flex",
                            "topology_unit_id": str(topology_unit_id),
                            "contract_ids": [
                                str(row[0].contract_id) for row in records
                            ],
                            "topology_rotation": int(rotation),
                            "topology_reflection": bool(reflection),
                            "topology_translation": list(translation),
                            "boundary_site_assignments": dict(assignments),
                            "planned_internal_closure": True,
                            "topology_reembedded": False,
                            "fixed_local_geometry": True,
                            "coordinate_aspect": _coordinate_aspect(trial),
                        }
                        result = _SiteSearchCandidate(
                            state=trial,
                            used_site_ids=frozenset(
                                set(candidate.used_site_ids) | selected
                            ),
                            boundary_site_ids={
                                **candidate.boundary_site_ids,
                                **assignments,
                            },
                            active_component_indices=candidate.active_component_indices,
                            actions=candidate.actions + [action],
                        )
                        signature = (
                            result.state.state_signature(),
                            tuple(sorted(result.boundary_site_ids.items())),
                        )
                        if signature not in seen:
                            seen.add(signature)
                            outcomes.append(result)
                            if len(outcomes) >= max(1, int(max_candidates)):
                                return sorted(
                                    outcomes, key=_site_candidate_rank
                                )[:max(1, int(max_candidates))]
    return sorted(outcomes, key=_site_candidate_rank)[:max(1, int(max_candidates))]


def _select_backbone_topologies(
    state: HexAssemblyState,
    component_count: int,
) -> Tuple[List[str], List[str]]:
    topology_ids = sorted(
        {
            _topology_unit_id(state, contract)
            for contract in state.problem.contracts
            if contract.kind == "flex"
        },
        key=_unit_sort_key,
    )
    ranked = sorted(
        topology_ids,
        key=lambda unit_id: (
            -len(_flex_port_groups(state, unit_id)),
            -len(state.unit(unit_id).nodes),
            _unit_sort_key(unit_id),
        ),
    )
    required = max(0, int(component_count) - 1)
    selected = []
    capacity = 0
    for unit_id in ranked:
        expansion = max(0, len(_flex_port_groups(state, unit_id)) - 1)
        if expansion <= 0 or capacity + expansion > required:
            continue
        selected.append(unit_id)
        capacity += expansion
        if capacity == required:
            break
    if capacity != required:
        raise InverseInputError(
            f"flex topologies cannot span {component_count} rigid components; "
            f"required_expansion={required} available={capacity}"
        )
    remaining = [unit_id for unit_id in topology_ids if unit_id not in selected]
    return selected, sorted(
        remaining,
        key=lambda unit_id: (-len(state.unit(unit_id).nodes), _unit_sort_key(unit_id)),
    )


def _site_search_plan(
    state: HexAssemblyState,
    rigid_components: Sequence[Sequence[str]],
    anchor_index: int,
    backbone_ids: Sequence[str],
    interior_ids: Sequence[str],
    actions: Sequence[Mapping[str, Any]],
) -> FlexAssemblyPlan:
    action_by_topology = {
        str(action.get("topology_unit_id")): action for action in actions
    }
    backbone_steps = []
    for unit_id in backbone_ids:
        action = action_by_topology.get(unit_id, {})
        default_ids = [
            str(row[0].contract_id)
            for row in _flex_port_records(state, unit_id)
        ]
        attachment_id = str(
            action.get(
                "attachment_contract_id",
                default_ids[0] if default_ids else "",
            )
        )
        merge_ids = tuple(
            str(value)
            for value in action.get("merge_contract_ids", default_ids[1:])
        )
        backbone_steps.append(BackboneFlexStep(
            topology_unit_id=str(unit_id),
            attachment_contract_id=attachment_id,
            merge_contract_ids=merge_ids,
        ))
    interior_steps = []
    for unit_id in interior_ids:
        action = action_by_topology.get(unit_id, {})
        contract_ids = tuple(str(value) for value in action.get(
            "contract_ids",
            [
                str(row[0].contract_id)
                for row in _flex_port_records(state, unit_id)
            ],
        ))
        interior_steps.append(InteriorFlexStep(
            topology_unit_id=str(unit_id),
            attachment_contract_id=contract_ids[0] if contract_ids else "",
            closure_contract_ids=contract_ids[1:],
        ))
    return FlexAssemblyPlan(
        anchor_rigid_component_unit_ids=tuple(rigid_components[anchor_index]),
        rigid_components=tuple(tuple(values) for values in rigid_components),
        backbone_topology_unit_ids=tuple(backbone_ids),
        backbone_contract_ids=tuple(
            contract_id
            for step in backbone_steps
            for contract_id in step.contract_ids
        ),
        backbone_steps=tuple(backbone_steps),
        interior_steps=tuple(interior_steps),
    )


def _run_csv_site_flex_search(
    state: HexAssemblyState,
    *,
    candidate_k: int,
    max_steps: int,
    beam_width: int,
    strict: bool,
    nmr_scorer: Any = None,
    nmr_weight: float = 1.0,
    geometry_weight: float = 0.05,
    contact_weight: float = 0.02,
) -> Tuple[SearchResult, FlexAssemblyPlan]:
    """Search physical aromatic sites while topology resources stay fixed."""
    initial = state.clone()
    rigid_components = _rigid_components(initial)
    node_counts = {
        index: sum(len(initial.unit(unit_id).nodes) for unit_id in component)
        for index, component in enumerate(rigid_components)
    }
    backbone_ids, interior_ids = _select_backbone_topologies(
        initial, len(rigid_components)
    )
    first_expansion = (
        len(_flex_port_groups(initial, backbone_ids[0])) - 1
        if backbone_ids else 0
    )
    if first_expansion > 1:
        # The compact three-way hub must start from three small independent
        # aromatic rings.  A large fused/rigid component cannot fit around its
        # immutable arms without overlap.
        anchor_index = min(
            range(len(rigid_components)),
            key=lambda index: (
                node_counts[index],
                tuple(map(_unit_sort_key, rigid_components[index])),
            ),
        )
    else:
        anchor_index = min(
            range(len(rigid_components)),
            key=lambda index: (
                -node_counts[index],
                tuple(map(_unit_sort_key, rigid_components[index])),
            ),
        )
    total_contracts = sum(
        len(_flex_port_records(initial, unit_id))
        for unit_id in backbone_ids + interior_ids
    )
    errors: List[str] = []
    if total_contracts > int(max_steps):
        errors.append(
            f"flex_max_steps={max_steps} is smaller than the planned "
            f"contract count {total_contracts}"
        )

    anchor_nodes = {
        int(node.node_id)
        for unit_id in rigid_components[anchor_index]
        for node in initial.unit(unit_id).nodes
    }
    # Unconnected rigid components are staging objects, not obstacles.  Keep
    # their placements so a later action can rigidly transform the whole
    # component, but expose only the rooted skeleton to collision checks.
    initial.occupied_axial = {
        axial: int(node_id)
        for axial, node_id in initial.occupied_axial.items()
        if int(node_id) in anchor_nodes
    }
    beam = [_SiteSearchCandidate(
        state=initial,
        used_site_ids=frozenset(),
        boundary_site_ids={},
        active_component_indices=frozenset({int(anchor_index)}),
        actions=[],
    )]
    remaining_component_indices = [
        index for index in range(len(rigid_components)) if index != anchor_index
    ]
    # The first, highest-port-count topology is a compact hub.  Feed only its
    # close arms with the smallest rigid components, then restore the proven
    # large-to-small order for the longer two-ended backbone bridges.
    hub_components = sorted(
        remaining_component_indices,
        key=lambda index: (
            node_counts[index],
            tuple(map(_unit_sort_key, rigid_components[index])),
        ),
    )[:first_expansion]
    hub_component_set = set(hub_components)
    component_order = hub_components + sorted(
        (
            index for index in remaining_component_indices
            if index not in hub_component_set
        ),
        key=lambda index: (
            -node_counts[index],
            tuple(map(_unit_sort_key, rigid_components[index])),
        ),
    )
    component_cursor = 0
    search_width = (
        max(1, int(beam_width))
        if nmr_scorer is not None
        else max(4, min(16, max(int(beam_width), int(candidate_k))))
    )
    outcome_limit = (
        max(1, int(candidate_k))
        if nmr_scorer is not None
        else max(4, min(12, int(candidate_k)))
    )
    explored = 0
    nmr_protected_ids = _rigid_endpoint_ids(initial)

    pending_backbone_ids = list(backbone_ids)
    executed_backbone_ids: List[str] = []
    while pending_backbone_ids and beam:
        selected_topology_id: str | None = None
        selected_expansion = 0
        selected_beam: List[_SiteSearchCandidate] = []
        for topology_unit_id in pending_backbone_ids:
            expansion = len(_flex_port_groups(initial, topology_unit_id)) - 1
            outside = component_order[
                component_cursor:component_cursor + expansion
            ]
            if len(outside) != expansion:
                continue
            next_beam: List[_SiteSearchCandidate] = []
            for candidate in beam:
                outcomes = _backbone_site_outcomes(
                    candidate,
                    topology_unit_id,
                    outside,
                    rigid_components,
                    outcome_limit,
                )
                explored += len(outcomes)
                next_beam.extend(outcomes)
            _score_site_candidates(
                next_beam,
                nmr_scorer=nmr_scorer,
                depth=len(executed_backbone_ids) + 1,
                nmr_weight=nmr_weight,
                geometry_weight=geometry_weight,
                contact_weight=contact_weight,
                protected_aromatic_ids=nmr_protected_ids,
            )
            unique: Dict[Tuple[Any, ...], _SiteSearchCandidate] = {}
            for next_candidate in sorted(next_beam, key=_site_candidate_rank):
                signature = (
                    next_candidate.state.state_signature(),
                    tuple(sorted(next_candidate.boundary_site_ids.items())),
                )
                unique.setdefault(signature, next_candidate)
                if len(unique) >= search_width:
                    break
            if unique:
                selected_topology_id = str(topology_unit_id)
                selected_expansion = int(expansion)
                selected_beam = list(unique.values())
                break

        if selected_topology_id is None:
            errors.append(
                "no fixed-geometry backbone placement for any remaining "
                f"topology: {pending_backbone_ids}"
            )
            beam = []
            break
        beam = selected_beam
        component_cursor += selected_expansion
        pending_backbone_ids.remove(selected_topology_id)
        executed_backbone_ids.append(selected_topology_id)

    if beam and component_cursor != len(component_order):
        errors.append(
            f"flex backbone left {len(component_order)-component_cursor} "
            "rigid components disconnected"
        )
        beam = []

    for topology_unit_id in interior_ids if beam else ():
        next_beam = []
        for candidate in beam:
            outcomes = _interior_site_outcomes(
                candidate,
                topology_unit_id,
                rigid_components,
                outcome_limit,
            )
            explored += len(outcomes)
            next_beam.extend(outcomes)
        _score_site_candidates(
            next_beam,
            nmr_scorer=nmr_scorer,
            depth=len(executed_backbone_ids) + interior_ids.index(topology_unit_id) + 1,
            nmr_weight=nmr_weight,
            geometry_weight=geometry_weight,
            contact_weight=contact_weight,
            protected_aromatic_ids=nmr_protected_ids,
        )
        unique = {}
        for candidate in sorted(next_beam, key=_site_candidate_rank):
            signature = (
                candidate.state.state_signature(),
                tuple(sorted(candidate.boundary_site_ids.items())),
            )
            unique.setdefault(signature, candidate)
            if len(unique) >= search_width:
                break
        beam = list(unique.values())
        if not beam:
            errors.append(
                f"no fixed-geometry interior placement for {topology_unit_id!r}"
            )
            break

    orientation_rotations: Dict[Tuple[Any, ...], int] = {}
    if beam:
        oriented_beam = [
            _normalize_global_orientation(candidate)
            for candidate in beam
        ]
        aspect_valid_beam = []
        for candidate, rotation in oriented_beam:
            signature = candidate.state.state_signature()
            orientation_rotations[signature] = int(rotation)
            if _coordinate_aspect(candidate.state)["valid"]:
                aspect_valid_beam.append(candidate)
        if aspect_valid_beam:
            beam = aspect_valid_beam
        else:
            best_shape = _coordinate_aspect(
                min(beam, key=_site_candidate_rank).state
            )
            errors.append(
                "no complete flex placement satisfies q_span/r_span in "
                f"[{MIN_Q_R_ASPECT_RATIO}, {MAX_Q_R_ASPECT_RATIO}]; "
                f"best={best_shape['q_over_r']:.6f} "
                f"q_span={best_shape['q_span']} r_span={best_shape['r_span']}"
            )
            beam = []

    planned_backbone_ids = (
        executed_backbone_ids
        if not pending_backbone_ids
        else executed_backbone_ids + pending_backbone_ids
    )
    plan = _site_search_plan(
        initial,
        rigid_components,
        anchor_index,
        planned_backbone_ids,
        interior_ids,
        beam[0].actions if beam else (),
    )
    selected: _SiteSearchCandidate | None = None
    global_orientation_rotation = 0
    if beam:
        selected = min(
            beam,
            key=lambda candidate: (
                abs(
                    float(_coordinate_aspect(candidate.state)["width_over_height"])
                    - 1.0
                ),
                _site_candidate_rank(candidate),
            ),
        )
        global_orientation_rotation = int(
            orientation_rotations.get(selected.state.state_signature(), 0)
        )
        current = relabel_aromatic_sites(
            selected.state,
            selected.boundary_site_ids,
            selected.actions,
        )
        actions = selected.actions
    else:
        current = state.clone()
        current.stage = "flex"
        actions = []
        message = errors[-1] if errors else "no geometry-preserving flex assembly"
        if strict:
            raise InverseInputError(message)

    result_row = TopologyPlacementStage(current, "flex").get_result()
    backbone_done = sum(
        contract_id in current.satisfied_contracts
        for contract_id in plan.backbone_contract_ids
    )
    interior_done = sum(
        contract_id in current.satisfied_contracts
        for contract_id in plan.interior_contract_ids
    )
    result_row.update({
        "rigid_components_total": len(rigid_components),
        "backbone_topology_units": len(planned_backbone_ids),
        "backbone_contracts": len(plan.backbone_contract_ids),
        "interior_topology_units": len(interior_ids),
        "interior_contracts": len(plan.interior_contract_ids),
        "backbone_contracts_done": int(backbone_done),
        "backbone_complete": backbone_done == len(plan.backbone_contract_ids),
        "interior_contracts_done": int(interior_done),
        "interior_complete": interior_done == len(plan.interior_contract_ids),
        "planned_internal_closures": int(interior_done),
        "planning_errors": list(errors),
        "explored_geometry_states": int(explored),
        "topology_reembedding_used": False,
        "global_orientation_rotation": int(global_orientation_rotation),
        "coordinate_aspect": _coordinate_aspect(current),
        "slot_geometry": {
            "mode": "search_time_physical_su13_selection",
            "boundary_site_assignments": (
                dict(selected.boundary_site_ids) if selected is not None else {}
            ),
        },
        "plan": plan.to_dict(),
        "search_algorithm": (
            "nmr_guided_beam_search" if nmr_scorer is not None
            else "geometry_beam_search"
        ),
        "search_width": int(search_width),
        "candidate_k": int(outcome_limit),
        "nmr_guidance": (
            {
                "selected": (
                    dict(selected.nmr_score)
                    if selected is not None and selected.nmr_score is not None
                    else None
                ),
                "statistics": nmr_scorer.statistics("flex"),
            }
            if nmr_scorer is not None else {"enabled": False}
        ),
    })
    return SearchResult(
        state=current,
        score=float(TopologyPlacementStage(current, "flex").score()),
        action_sequence=list(actions),
        info={"phase": "flex", "result": result_row},
    ), plan


def run_planned_flex_search(
    state: HexAssemblyState,
    *,
    iterations: int,
    candidate_k: int,
    max_steps: int,
    beam_width: int = 1,
    c_puct: float = 1.414,
    seed: int | None = None,
    strict: bool = False,
    nmr_scorer: Any = None,
    nmr_weight: float = 1.0,
    geometry_weight: float = 0.05,
    contact_weight: float = 0.02,
) -> Tuple[SearchResult, FlexAssemblyPlan]:
    """Connect all prescribed flex units without changing local geometry."""
    csv_mode = state.problem.source_kind == InputSourceKind.CSV_PAIR
    if csv_mode:
        return _run_csv_site_flex_search(
            state,
            candidate_k=candidate_k,
            max_steps=max_steps,
            beam_width=beam_width,
            strict=strict,
            nmr_scorer=nmr_scorer,
            nmr_weight=nmr_weight,
            geometry_weight=geometry_weight,
            contact_weight=contact_weight,
        )
    del iterations, beam_width, c_puct, seed
    current, slot_diagnostics = _optimize_flex_boundary_slots(
        state.clone()
    )
    plan = build_flex_assembly_plan(current)
    by_topology = _contracts_by_topology(current)
    actions: List[Dict[str, Any]] = []
    errors: List[str] = []
    explored_states = 0
    outcome_limit = min(24, max(8, 2 * int(candidate_k)))
    failed_backbone_states: Set[Tuple[int, Tuple[Any, ...]]] = set()

    if len(plan.all_contract_ids) > int(max_steps):
        errors.append(
            f"flex_max_steps={max_steps} is smaller than the planned "
            f"contract count {len(plan.all_contract_ids)}"
        )

    def solve_backbone(
        step_index: int,
        source_state: HexAssemblyState,
    ) -> Tuple[HexAssemblyState, List[Dict[str, Any]]] | None:
        nonlocal explored_states
        search_key = (int(step_index), source_state.state_signature())
        if search_key in failed_backbone_states:
            return None
        if explored_states >= 20000:
            return None
        if step_index >= len(plan.backbone_steps):
            closure_outcomes = _close_eligible_outcomes(
                source_state, plan, by_topology, outcome_limit
            )
            for trial, closure_actions in closure_outcomes:
                if not trial.mandatory_contracts("flex"):
                    return trial, closure_actions
            failed_backbone_states.add(search_key)
            return None

        step = plan.backbone_steps[step_index]
        topology_id = step.topology_unit_id
        contracts = by_topology[topology_id]
        if all(
            str(contract.contract_id) in source_state.satisfied_contracts
            for contract in contracts
        ):
            return solve_backbone(step_index + 1, source_state)

        step_outcomes = _backbone_step_outcomes(
            source_state,
            plan,
            topology_id,
            contracts,
            outcome_limit,
        )
        for trial, step_actions in step_outcomes:
            if explored_states >= 20000:
                break
            explored_states += 1
            closure_outcomes = _close_eligible_outcomes(
                trial, plan, by_topology, outcome_limit
            )
            for closed, closure_actions in closure_outcomes:
                solved = solve_backbone(step_index + 1, closed)
                if solved is not None:
                    final_state, later_actions = solved
                    return (
                        final_state,
                        step_actions + closure_actions + later_actions,
                    )
        failed_backbone_states.add(search_key)
        return None

    initial_outcomes = _close_eligible_outcomes(
        current, plan, by_topology, outcome_limit
    )
    solved = None
    for initial_state, initial_actions in initial_outcomes:
        candidate = solve_backbone(0, initial_state)
        if candidate is not None:
            final_state, later_actions = candidate
            solved = final_state, initial_actions + later_actions
            break
    if solved is None:
        message = (
            "no geometry-preserving flex assembly satisfies every fixed "
            f"topology contract after exploring {explored_states} states"
        )
        errors.append(message)
        if strict:
            raise InverseInputError(message)
    else:
        current, actions = solved

    result_row = TopologyPlacementStage(current, "flex").get_result()
    backbone_done = sum(
        contract_id in current.satisfied_contracts
        for contract_id in plan.backbone_contract_ids
    )
    interior_done = sum(
        contract_id in current.satisfied_contracts
        for contract_id in plan.interior_contract_ids
    )
    result_row.update({
        "rigid_components_total": len(plan.rigid_components),
        "backbone_topology_units": len(plan.backbone_topology_unit_ids),
        "backbone_contracts": len(plan.backbone_contract_ids),
        "interior_topology_units": len(plan.interior_topology_unit_ids),
        "interior_contracts": len(plan.interior_contract_ids),
        "backbone_contracts_done": int(backbone_done),
        "backbone_complete": backbone_done == len(plan.backbone_contract_ids),
        "interior_contracts_done": int(interior_done),
        "interior_complete": interior_done == len(plan.interior_contract_ids),
        "planned_internal_closures": int(interior_done),
        "planning_errors": list(errors),
        "explored_geometry_states": int(explored_states),
        "topology_reembedding_used": False,
        "slot_geometry": slot_diagnostics,
        "plan": plan.to_dict(),
    })
    return SearchResult(
        state=current,
        score=float(TopologyPlacementStage(current, "flex").score()),
        action_sequence=list(actions),
        info={"phase": "flex", "result": result_row},
    ), plan


__all__ = [
    "BackboneFlexStep",
    "FlexAssemblyPlan",
    "InteriorFlexStep",
    "build_flex_assembly_plan",
    "relabel_aromatic_sites",
    "run_planned_flex_search",
]
