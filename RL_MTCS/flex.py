"""Geometry-preserving flexible assembly over rigid aromatic components.

The inverse CSV owns topology identity and connectivity. Generated tree
conformers may vary on the hex grid, while explicit fixed geometries and
all node/edge identities are preserved and audited.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .geometry import (
    HEX_DIRECTIONS,
    axial_distance,
    aromatic_slot_direction,
    state_layout_metrics,
    state_shape_rank,
    unit_sort_key,
)
from .state import (
    HexAssemblyState,
    axial_add,
    axial_neg,
    transform_axial,
    relabel_aromatic_sites,
    aromatic_perimeter_ids,
    stage_snapshot,
    hetero_staging_ids,
)
from .models import (
    ExternalContract,
    InverseInputError,
    TopologyUnit,
)
from .rigid import SearchResult, TopologyPlacementStage


def _unit_sort_key(unit_id: str) -> Tuple[Any, ...]:
    return unit_sort_key(unit_id)


def _source_product_type(unit: TopologyUnit) -> str:
    """Return the topology label supplied by the inverse CSV input."""
    metadata = dict(unit.metadata)
    return str(metadata.get("source_product_type", unit.product_type)).lower()


def _flex_stage_rank(unit: TopologyUnit) -> int:
    """Order flexible units as chains first, then aliphatic rings.

    CSV materialization may expose a generated tree as ``flex_chain`` even
    when its source row is an ``aliphatic_ring``.  The source label is the
    authoritative stage classification; the geometry representation must not
    change the scheduling order.
    """
    kind = _source_product_type(unit)
    if kind in {"aliphatic_ring", "fatty_ring", "fatty_ring_fused",
                "fatty_ring_side", "fatty_ring_upper_lower"}:
        return 1
    return 0


def _topology_complexity_key(unit: TopologyUnit) -> Tuple[int, ...]:
    """Sort complex graphs first, independently of coordinates and GID order.

    Compare attachment count, independent cycles, branch excess, then size.
    Ports contribute to node degree so a branch at an endpoint is counted too.
    Product labels such as ``aliphatic_ring`` do not establish an internal cycle.
    """
    adjacency = {node.node_id: set() for node in unit.nodes}
    for edge in unit.internal_edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            neighbors = adjacency[stack.pop()] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    edge_count = sum(map(len, adjacency.values())) // 2
    cycles = edge_count - len(adjacency) + components
    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    for port in unit.ports:
        degrees[port.node_id] += 1
    branches = sum(max(0, degree - 2) for degree in degrees.values())
    return (-len(unit.ports), -cycles, -branches, -len(unit.nodes), -edge_count)


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
    if not state.phase_done("rigid"):
        raise InverseInputError(
            "flex planning requires all SU10 ports connected; pending="
            + ",".join(contract.contract_id for contract in pending_rigid)
            + f" completed={sum(c.kind == 'rigid' and c.contract_id in state.satisfied_contracts for c in state.problem.contracts)}"
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

    chain_topology_ids = {
        topology_id
        for topology_id in by_topology
        if _flex_stage_rank(state.unit(topology_id)) == 0
    }
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
                    -len(outside),
                    _topology_complexity_key(state.unit(topology_id)),
                    min(component_flex_degree[index] for index in outside),
                    len(state.unit(topology_id).nodes),
                    _unit_sort_key(topology_id),
                    topology_id,
                    inside,
                    outside,
                ))
        # Chains are the only legal Flex units while rigid components remain
        # disconnected.  A ring must never bypass an unplaced chain: doing so
        # makes the planning layer disagree with the actual search order.
        chain_candidates = [
            candidate for candidate in candidates
            if candidate[6] in chain_topology_ids
        ]
        if remaining & chain_topology_ids:
            if not chain_candidates:
                missing = sorted(set(range(len(rigid_components))) - connected)
                raise InverseInputError(
                    "flex_chain_stage_blocked: no chain can connect the "
                    f"remaining rigid components; missing={missing}"
                )
            candidates = chain_candidates
        if not candidates:
            missing = sorted(set(range(len(rigid_components))) - connected)
            raise InverseInputError(
                f"flex topology cannot connect all rigid components; missing={missing}"
            )
        (
            parent_depth,
            _neg_outside,
            _complexity,
            _outside_degree,
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
    interior_topology_ids: List[str] = []
    for topology_id in backbone_units:
        deferred = deferred_by_unit.get(topology_id, [])
        if deferred:
            interior_topology_ids.append(topology_id)
    interior_topology_ids.extend(remaining)
    for topology_id in sorted(
        interior_topology_ids,
        key=lambda uid: (_flex_stage_rank(state.unit(uid)), _unit_sort_key(uid)),
    ):
        deferred = deferred_by_unit.get(topology_id, [])
        if deferred:
            interior_steps.append(InteriorFlexStep(
                topology_unit_id=topology_id,
                closure_contract_ids=tuple(deferred),
            ))
            continue
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


def _aromatic_slot_direction(unit: TopologyUnit, node_id: int) -> Tuple[int, int]:
    return aromatic_slot_direction(unit, node_id)


@dataclass
class _SiteSearchCandidate:
    state: HexAssemblyState
    used_site_ids: frozenset[int]
    boundary_site_ids: Dict[int, int]
    active_component_indices: frozenset[int]
    actions: List[Dict[str, Any]]
    nmr_score: Dict[str, Any] | None = None
    nmr_rank: float | None = None
    shape_rank: Tuple[Any, ...] = ()
    anchor_index: int = 0
    rigid_candidate_index: int = 0
    remaining_candidate_counts: Tuple[Tuple[str, int], ...] = ()
    remaining_candidate_sets: Tuple[Tuple[str, Tuple[Tuple[int, ...], ...]], ...] = ()
    domain_risk: float = 0.0
    domain_conflict: Dict[str, Any] | None = None


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
    rigid_ids = _rigid_endpoint_ids(state) | hetero_staging_ids(state)
    records = []
    for unit_id in sorted(component, key=_unit_sort_key):
        unit = state.unit(unit_id)
        placement = state.placements[unit_id]
        perimeter_ids = aromatic_perimeter_ids(unit)
        for node in unit.nodes:
            node_id = int(node.node_id)
            if int(node.su_type) != 13 or node_id in rigid_ids or node_id in used:
                continue
            if node_id not in perimeter_ids:
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


def _diverse_slot_records(
    records: Sequence[Tuple[str, int, Tuple[int, int], Tuple[int, int]]],
    limit: int,
) -> List[Tuple[str, int, Tuple[int, int], Tuple[int, int]]]:
    """Sample sites across aromatic units and outward directions."""
    if len(records) <= max(1, int(limit)):
        return list(records)
    grouped: Dict[
        Tuple[str, Tuple[int, int]],
        List[Tuple[str, int, Tuple[int, int], Tuple[int, int]]],
    ] = {}
    for record in records:
        grouped.setdefault((str(record[0]), tuple(record[3])), []).append(record)
    keys = sorted(grouped, key=lambda row: (_unit_sort_key(row[0]), row[1]))
    selected = []
    round_index = 0
    while keys and len(selected) < max(1, int(limit)):
        next_keys = []
        for key in keys:
            values = grouped[key]
            if round_index < len(values):
                selected.append(values[round_index])
                next_keys.append(key)
                if len(selected) >= max(1, int(limit)):
                    break
        keys = next_keys
        round_index += 1
    return selected


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


# The inverse topology uses the same axial convention as ``geometry.py``.
# These are the legal secondary 11-23...-11 paths supplied by the chemistry
# model: five horizontal lengths and two vertical lengths (with both signs
# and both mirrored caps generated below).  Keeping the table here makes the
# displacement rule explicit and, importantly, prevents a bounded site scan
# from inventing an arbitrary endpoint pairing.
_SECONDARY_HORIZONTAL_DIRECTIONS = ((1, 1), (1, 0), (-1, 0), (-1, -1))
_SECONDARY_VERTICAL_CAPS = {
    (0, 1): ((1, 1), (-1, 0)),
    (0, -1): ((1, 0), (-1, -1)),
}


def _secondary_horizontal_path(
    start: Tuple[int, int], main: Tuple[int, int], n23: int,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, int]]:
    pair = {
        (1, 1): (1, 0),
        (1, 0): (1, 1),
        (-1, 0): (-1, -1),
        (-1, -1): (-1, 0),
    }[tuple(main)]
    current = tuple(start)
    path = []
    for index in range(int(n23)):
        step = main if index % 2 == 0 else pair
        current = axial_add(current, step)
        path.append(current)
    last_step = main if int(n23) % 2 == 0 else pair
    return tuple(path), axial_add(current, last_step)


def _secondary_vertical_path(
    start: Tuple[int, int], vertical: Tuple[int, int], n23: int,
    cap: Tuple[int, int],
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, int]]:
    current = tuple(start)
    path = []
    directions = []
    for index in range(int(n23)):
        step = vertical if index % 2 == 0 else cap
        directions.append(step)
        current = axial_add(current, step)
        path.append(current)
    last_step = cap if directions[-1] == vertical else vertical
    return tuple(path), axial_add(current, last_step)


def _build_secondary_displacement_table() -> Dict[
    Tuple[int, int], Tuple[Tuple[str, Tuple[int, int], int, Tuple[int, int] | None], ...]
]:
    table: Dict[
        Tuple[int, int],
        List[Tuple[str, Tuple[int, int], int, Tuple[int, int] | None]],
    ] = {}
    origin = (0, 0)
    for main in _SECONDARY_HORIZONTAL_DIRECTIONS:
        for n23 in range(1, 6):
            _path, end = _secondary_horizontal_path(origin, main, n23)
            table.setdefault(end, []).append(("horizontal", main, n23, None))
    for vertical, caps in _SECONDARY_VERTICAL_CAPS.items():
        for cap in caps:
            for n23 in range(1, 3):
                _path, end = _secondary_vertical_path(origin, vertical, n23, cap)
                table.setdefault(end, []).append(("vertical", vertical, n23, cap))
    return {key: tuple(values) for key, values in table.items()}


_SECONDARY_DISPLACEMENT_TABLE = _build_secondary_displacement_table()


def secondary_flex_displacement_candidates(
    source: Tuple[int, int],
    target: Tuple[int, int],
    source_outward: Tuple[int, int] | None = None,
    target_outward: Tuple[int, int] | None = None,
) -> List[Dict[str, Any]]:
    """Return legal 13-13 secondary paths between two aromatic sites.

    This is a geometry audit/helper only.  It does not mutate SU labels or
    create contracts.  Direction checks are optional so callers can first
    count displacement-compatible pairs and then apply port orientation.
    """
    source = tuple(map(int, source))
    target = tuple(map(int, target))
    displacement = (target[0] - source[0], target[1] - source[1])
    results: List[Dict[str, Any]] = []
    for direction, main, n23, cap in _SECONDARY_DISPLACEMENT_TABLE.get(
        displacement, ()
    ):
        if source_outward is not None and tuple(source_outward) != tuple(main):
            continue
        if direction == "horizontal":
            path, end = _secondary_horizontal_path(source, main, n23)
        else:
            path, end = _secondary_vertical_path(source, main, n23, cap)
        if end != target:
            continue
        if target_outward is not None:
            arrival = (target[0] - path[-1][0], target[1] - path[-1][1])
            if tuple(target_outward) != axial_neg(arrival):
                continue
        results.append({
            "direction": direction,
            "source_site": source,
            "target_site": target,
            "displacement": displacement,
            "chain_length": int(n23),
            "intermediate_coordinates": [list(value) for value in path],
            "source_outward": tuple(main),
            "target_outward": (
                tuple(target_outward) if target_outward is not None
                else axial_neg((target[0] - path[-1][0], target[1] - path[-1][1]))
            ),
        })
    return results


def _secondary_flex_pose_candidates(
    candidate: _SiteSearchCandidate,
    topology_unit_id: str,
    rigid_components: Sequence[Sequence[str]],
    slot_cache: Mapping[Any, Any] | None,
    diagnostics: Dict[str, Any] | None,
    max_checks: int | None,
):
    """Enumerate fixed flex-chain poses by legal 13-13 paths first.

    The previous generic scan examined every slot/orientation combination and
    often hit ``domain_check_limit`` before reaching a legal pair.  This
    generator indexes the target slot by the exact displacement table, then
    performs the normal collision/contract validation on only those pairs.
    """
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    if len(records) != 2 or any(row[3].direction is None for row in records):
        return
    active_units = tuple(
        uid for index in sorted(candidate.active_component_indices)
        for uid in rigid_components[index]
    )
    slot_key = tuple((uid, state.placements[uid]) for uid in active_units)
    slots = slot_cache.get(slot_key) if slot_cache is not None else None
    if slots is None:
        slots = _component_slot_records(state, active_units, ())
        if slot_cache is not None:
            slot_cache[slot_key] = slots
    used = set(candidate.used_site_ids)
    slots = [row for row in slots if row[1] not in used]
    by_geometry = {(row[2], row[3]): row[1] for row in slots}
    first_contract, first_node, first_boundary, first_port = records[0]
    second_contract, second_node, second_boundary, second_port = records[1]
    nodes = unit.node_map()
    seen = set()
    checks = 0
    for reflection in (False, True):
        for rotation in range(6):
            first_direction = transform_axial(first_port.direction, rotation, reflection)
            second_direction = transform_axial(second_port.direction, rotation, reflection)
            first_outward = axial_neg(first_direction)
            second_outward = axial_neg(second_direction)
            first_local = transform_axial(nodes[first_node].local_axial, rotation, reflection)
            second_local = transform_axial(nodes[second_node].local_axial, rotation, reflection)
            endpoint_delta = (
                second_local[0] + second_direction[0] - first_local[0] - first_direction[0],
                second_local[1] + second_direction[1] - first_local[1] - first_direction[1],
            )
            if endpoint_delta not in _SECONDARY_DISPLACEMENT_TABLE:
                continue
            for _uid, first_site, first_axial, first_site_outward in slots:
                target_axial = axial_add(first_axial, endpoint_delta)
                second_site = by_geometry.get((target_axial, second_outward))
                if second_site is None or second_site == first_site:
                    continue
                if first_site_outward != first_outward:
                    continue
                path_options = secondary_flex_displacement_candidates(
                    first_axial, target_axial, first_outward, second_outward,
                )
                if not path_options:
                    continue
                checks += 1
                if max_checks is not None and checks > max(0, int(max_checks)):
                    if diagnostics is not None:
                        diagnostics["pose_enumeration_truncated"] = True
                        diagnostics["pose_checks"] = checks - 1
                    return
                translation = (
                    first_axial[0] + first_outward[0] - first_local[0],
                    first_axial[1] + first_outward[1] - first_local[1],
                )
                assignments = {
                    int(first_boundary): int(first_site),
                    int(second_boundary): int(second_site),
                }
                contacts = [(first_node, first_site), (second_node, second_site)]
                valid, reason = state.can_place(
                    topology_unit_id, rotation, translation,
                    allowed_contacts=contacts, reflection=reflection,
                    allow_unbonded_neighbors=False,
                )
                if not valid:
                    if diagnostics is not None:
                        counts = diagnostics.setdefault("rejection_counts", {})
                        category = reason.split(":", 1)[0]
                        counts[category] = counts.get(category, 0) + 1
                    continue
                coordinates = state._placement_coordinates(
                    unit, rotation, translation, reflection
                )
                path = path_options[0]["intermediate_coordinates"]
                path_set = {tuple(value) for value in path}
                node_coordinates = set(coordinates.values())
                if not path_set.issubset(node_coordinates):
                    if diagnostics is not None:
                        diagnostics["path_geometry_mismatches"] = diagnostics.get(
                            "path_geometry_mismatches", 0
                        ) + 1
                    continue
                signature = (
                    rotation, reflection, translation,
                    tuple(sorted(assignments.items())),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                if diagnostics is not None:
                    diagnostics["secondary_candidate_count"] = diagnostics.get(
                        "secondary_candidate_count", 0
                    ) + 1
                yield rotation, translation, reflection, assignments, frozenset(
                    (first_site, second_site)
                )
    if diagnostics is not None:
        diagnostics["pose_checks"] = checks



def _append_site_contract(
    state: HexAssemblyState,
    contract: ExternalContract,
    topology_node_id: int,
    physical_site_id: int,
) -> None:
    """Commit one flex endpoint against a physical aromatic SU13 site.

    During flex search the contract endpoint in the inverse CSV is a staging
    GID.  ``physical_site_id`` is the actual SU13 slot selected by the pose
    search.  The latter is the resource that must be consumed: it is assigned
    the planned SU type of the contract's aromatic endpoint and cannot be
    selected by another flex contract.  The immutable inverse node map is
    intentionally left unchanged until the final relabel pass.
    """
    topology_node_id = int(topology_node_id)
    physical_site_id = int(physical_site_id)
    ownership = state.problem.ownership_map()
    aromatic_unit_id, boundary_node_id = _aromatic_endpoint(state, contract)
    expected_su_type = int(state.problem.node_map().get(int(boundary_node_id), -1))
    if expected_su_type < 0:
        raise InverseInputError(
            f"flex contract {contract.contract_id!r} has no planned SU type "
            f"for aromatic boundary {boundary_node_id}"
        )
    physical_unit_id = ownership.get(physical_site_id)
    physical_unit = (
        state.unit(physical_unit_id) if physical_unit_id is not None else None
    )
    # The searched boundary GID is deliberately allowed to land on a site in
    # a different rigid cluster.  That is how a flex chain performs a
    # cross-component merge; only the physical site's phase is constrained.
    if physical_unit is None or physical_unit.phase != "aromatic":
        raise InverseInputError(
            f"flex contract {contract.contract_id!r} selected site "
            f"{physical_site_id}, which is not an aromatic site"
        )
    if physical_site_id not in aromatic_perimeter_ids(physical_unit) or physical_site_id in hetero_staging_ids(state):
        raise InverseInputError('flex endpoint is not an available aromatic ring perimeter site')
    if physical_site_id in _rigid_endpoint_ids(state):
        raise InverseInputError(
            f"flex contract {contract.contract_id!r} selected a rigid endpoint "
            f"{physical_site_id}"
        )
    if topology_node_id not in state.node_axial or physical_site_id not in state.node_axial:
        raise InverseInputError(
            f"flex contract {contract.contract_id!r} endpoint is not placed"
        )
    current_su = int(state.current_su_types.get(physical_site_id, -1))
    if current_su != 13:
        raise InverseInputError(
            f"flex site {physical_site_id} is unavailable (current SU{current_su})"
        )

    topology_unit_id, expected_topology_node_id = _topology_endpoint(state, contract)
    if int(expected_topology_node_id) != topology_node_id:
        raise InverseInputError(
            f"flex contract {contract.contract_id!r} topology endpoint mismatch"
        )
    topology_port_id = (
        contract.source_port_id
        if state.unit(contract.source_unit_id).phase == "flex"
        else contract.target_port_id
    )
    topology_key = f"{topology_unit_id}:{topology_port_id}"
    aromatic_key = (
        f"{contract.source_unit_id}:{contract.source_port_id}"
        if state.unit(contract.source_unit_id).phase == "aromatic"
        else f"{contract.target_unit_id}:{contract.target_port_id}"
    )
    physical_key = f"aromatic_site:{physical_site_id}"
    if any(key in state.used_ports for key in (topology_key, aromatic_key, physical_key)):
        raise InverseInputError(
            f"flex endpoint already used for contract {contract.contract_id!r}"
        )
    direction = state._edge_direction(
        state.node_axial[topology_node_id],
        state.node_axial[physical_site_id],
    )
    if direction is None:
        raise InverseInputError(
            f"site contract {contract.contract_id!r} is not a hex neighbor"
        )
    state._append_edge(
        topology_node_id,
        physical_site_id,
        "flex",
        str(contract.contract_id),
        direction,
    )
    state.used_ports.update({topology_key, aromatic_key, physical_key})
    state.current_su_types[physical_site_id] = expected_su_type
    state.satisfied_contracts.add(str(contract.contract_id))


def _coordinate_aspect(state: HexAssemblyState) -> Dict[str, Any]:
    return state_layout_metrics(state, full=False, include_void=True)


def _state_shape_rank(state: HexAssemblyState) -> Tuple[Any, ...]:
    return state_shape_rank(state)


def _cheap_site_candidate_rank(
    candidate: _SiteSearchCandidate,
) -> Tuple[Any, ...]:
    axials = list(candidate.state.occupied_axial)
    if not axials:
        return (0, 0, 0, tuple(sorted(candidate.boundary_site_ids.items())))
    q_span = max(value[0] for value in axials) - min(value[0] for value in axials)
    r_span = max(value[1] for value in axials) - min(value[1] for value in axials)
    return (
        q_span + r_span,
        abs(q_span - r_span),
        len(candidate.state.allowed_unbonded_contacts),
        tuple(sorted(candidate.boundary_site_ids.items())),
    )


def _site_candidate_rank(candidate: _SiteSearchCandidate) -> Tuple[Any, ...]:
    nmr_prefix: Tuple[Any, ...] = ()
    if candidate.nmr_rank is not None and candidate.nmr_score is not None:
        nmr_prefix = (
            float(candidate.nmr_rank),
            float(candidate.nmr_score["loss"]),
        )
    if not candidate.shape_rank:
        candidate.shape_rank = _state_shape_rank(candidate.state)
    shape_rank = candidate.shape_rank
    combined_rank = float(shape_rank[0])
    if nmr_prefix:
        # Preserve NMR discrimination while making compactness a meaningful
        # secondary objective instead of allowing graph score alone to choose
        # a hollow or highly elongated pose.
        combined_rank = float(nmr_prefix[0]) + 0.10 * float(shape_rank[0])
    nmr_tail = nmr_prefix[1:] if nmr_prefix else ()
    return (
        combined_rank,
        *nmr_tail,
        *shape_rank[1:],
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
            + 0.05 * float(shape.get("layout_energy", 0.0))
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
    return replace(candidate, state=state, shape_rank=()), rotation


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
    diagnostics: Dict[str, Any] | None = None,
) -> List[_SiteSearchCandidate]:
    """Attach one topology; adjacent ports move the same rigid component."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    groups = _flex_port_groups(state, topology_unit_id)
    if len(groups) != len(outside_component_indices) + 1:
        return []
    # Cross-unit contacts require explicit allowed node pairs.
    allow_hub_neighbors = False
    outcomes: List[_SiteSearchCandidate] = []
    seen = set()
    # A yielded backbone candidate is a full state snapshot. Retain only a
    # small ranked frontier; a wide temporary pool at every live PUCT
    # iterator causes a memory cliff.
    pool_limit = max(4, min(8, 2 * max(1, int(max_candidates))))

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
            if diagnostics is not None and len(anchor_slots) > 48:
                diagnostics['candidate_pool_limited'] = True
                diagnostics['candidate_space_complete'] = False
            anchor_slots = _diverse_slot_records(anchor_slots, 48)
            for _uid, first_site, first_axial, first_outward in anchor_slots:
                for reflection in (False, True):
                    for rotation in range(6):
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
                            if anchored.audit_unintended_contacts():
                                continue
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
                                slots = _diverse_slot_records(slots, 48)
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
                                                if moved.audit_unintended_contacts():
                                                    continue
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
                                                if len(next_partial) >= pool_limit:
                                                    break
                                            if len(next_partial) >= pool_limit:
                                                break
                                        if len(next_partial) >= pool_limit:
                                            break
                                    if len(next_partial) >= pool_limit:
                                        break
                                if len(next_partial) >= pool_limit:
                                    break
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
                                anchor_index=candidate.anchor_index,
                                rigid_candidate_index=candidate.rigid_candidate_index,
                            )
                            signature = (
                                result.state.state_signature(),
                                tuple(sorted(result.boundary_site_ids.items())),
                            )
                            if signature not in seen:
                                seen.add(signature)
                                outcomes.append(result)
                                if len(outcomes) >= pool_limit:
                                    return sorted(
                                        outcomes, key=_site_candidate_rank
                                    )[:max(1, int(max_candidates))]
    return sorted(outcomes, key=_site_candidate_rank)[:max(1, int(max_candidates))]


def _interior_site_poses(
    candidate, topology_unit_id, rigid_components, slot_cache=None, diagnostics=None,
    max_checks=None,
):
    """One source of truth for placement-domain accounting and placement."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    if not records or any(row[3].direction is None for row in records):
        return
    # Flex-chain endpoints are governed by the explicit secondary 13-13
    # displacement table.  Enumerate those pairs directly; the generic
    # placement scan remains the fallback for aliphatic rings and other
    # multi-port topologies.
    if (
        str(_source_product_type(unit)).lower() == "flex_chain"
        and len(records) == 2
    ):
        yield from _secondary_flex_pose_candidates(
            candidate, topology_unit_id, rigid_components, slot_cache,
            diagnostics, max_checks,
        )
        return
    active_units = tuple(uid for i in sorted(candidate.active_component_indices)
                         for uid in rigid_components[i])
    slot_key = tuple((uid, state.placements[uid]) for uid in active_units)
    slots = slot_cache.get(slot_key) if slot_cache is not None else None
    if slots is None:
        slots = _component_slot_records(state, active_units, ())
        if slot_cache is not None:
            slot_cache[slot_key] = slots
    slots = [s for s in slots if s[1] not in candidate.used_site_ids]
    by_geometry = {(s[2], s[3]): s[1] for s in slots}
    # Every complete pose must include this port. Enumerating other anchors
    # produces only duplicates of the same rigid transform.
    _contract, node_id, boundary_id, port = records[0]
    nodes = unit.node_map()
    seen = set()
    checks = 0
    for _uid, site_id, site_axial, outward in slots:
        for reflection in (False, True):
            for rotation in range(6):
                if max_checks is not None and checks >= max(0, int(max_checks)):
                    if diagnostics is not None:
                        diagnostics['pose_enumeration_truncated'] = True
                        diagnostics['pose_checks'] = checks
                    return
                checks += 1
                direction = transform_axial(port.direction, rotation, reflection)
                if outward != axial_neg(direction):
                    continue
                local = transform_axial(nodes[node_id].local_axial, rotation, reflection)
                translation = (site_axial[0] + outward[0] - local[0],
                               site_axial[1] + outward[1] - local[1])
                assigned = {boundary_id: site_id}
                selected = {site_id}
                for _c, row_node, row_boundary, row_port in records[1:]:
                    row_direction = transform_axial(row_port.direction, rotation, reflection)
                    required = axial_add(axial_add(transform_axial(
                        nodes[row_node].local_axial, rotation, reflection), translation), row_direction)
                    matching = by_geometry.get((required, axial_neg(row_direction)))
                    if matching is None or matching in selected:
                        break
                    assigned[row_boundary] = matching
                    selected.add(matching)
                if len(assigned) != len(records):
                    if diagnostics is not None:
                        diagnostics['endpoint_misses'] = diagnostics.get('endpoint_misses', 0) + 1
                    continue
                contacts = [(n, assigned[b]) for _c, n, b, _p in records]
                valid, reason = state.can_place(
                    topology_unit_id, rotation, translation, allowed_contacts=contacts,
                    reflection=reflection,
                    allow_unbonded_neighbors=False,
                )
                if not valid:
                    if diagnostics is not None:
                        counts = diagnostics.setdefault('rejection_counts', {})
                        category = reason.split(':', 1)[0]
                        counts[category] = counts.get(category, 0) + 1
                        samples = diagnostics.setdefault('rejection_samples', [])
                        if len(samples) < 16:
                            samples.append(dict(topology=topology_unit_id, reason=reason,
                                                rotation=rotation, translation=list(translation),
                                                reflection=reflection, sites=dict(assigned)))
                    continue
                signature = (rotation, reflection, translation, tuple(sorted(assigned.items())))
                if signature not in seen:
                    seen.add(signature)
                    yield rotation, translation, reflection, assigned, frozenset(selected)
    if diagnostics is not None:
        diagnostics['pose_checks'] = checks


def audit_flex_feasibility(
    state: HexAssemblyState,
    *,
    rigid_candidates: Sequence[HexAssemblyState] | None = None,
    conformer_limit: int = 12,
    pose_check_limit: int = 256,
    pose_limit: int = 32,
    candidate_limit: int | None = None,
) -> Dict[str, Any]:
    """Count legal endpoint/direction placements before joint search.

    The audit deliberately does not mutate the supplied states.  A zero count
    is reported per rigid candidate and topology unit, while
    ``candidate_space_complete`` records whether the bounded conformer pool
    was exhausted.  Consequently this report distinguishes a geometric dead
    end from an incomplete candidate pool without claiming a global proof.
    """
    from .conformers import topology_conformer_info
    from .search import domain_conflict_check

    initial = state.clone()
    components = _rigid_components(initial)
    topology_ids = tuple(sorted({
        _topology_unit_id(initial, contract)
        for contract in initial.mandatory_contracts("flex")
    }, key=lambda uid: (_topology_complexity_key(initial.unit(uid)), _unit_sort_key(uid))))
    candidates = list(rigid_candidates) if rigid_candidates is not None else [initial]
    if candidate_limit is not None:
        limit = max(1, int(candidate_limit))
        if len(candidates) > limit:
            candidates = candidates[:limit]
    rows = []
    all_complete = True
    pose_check_limit = max(1, int(pose_check_limit))
    pose_limit = max(1, int(pose_limit))
    for candidate_index, source in enumerate(candidates):
        trial = source.clone()
        domains = {}
        unit_rows = []
        candidate_complete = True
        active = frozenset(range(len(components)))
        for uid in topology_ids:
            original = trial.conformer_sources.get(uid, trial.unit(uid))
            variants, variants_complete = topology_conformer_info(original, conformer_limit)
            candidate_complete = candidate_complete and bool(variants_complete)
            pose_count = 0
            site_sets = set()
            rejection = []
            pose_checks = 0
            pose_enumeration_complete = True
            for variant in variants:
                remaining_checks = pose_check_limit - pose_checks
                if remaining_checks <= 0:
                    pose_enumeration_complete = False
                    break
                variant_state = trial.clone()
                try:
                    variant_state.select_conformer(variant)
                except InverseInputError as exc:
                    rejection.append(str(exc))
                    continue
                probe = _SiteSearchCandidate(
                    variant_state, frozenset(), {}, active, [],
                )
                pose_diagnostics = {}
                for pose in _interior_site_poses(
                    probe, uid, components, {}, pose_diagnostics,
                    max_checks=remaining_checks,
                ):
                    pose_count += 1
                    site_sets.add(tuple(sorted(pose[4])))
                    if pose_count >= pose_limit:
                        pose_enumeration_complete = False
                        break
                pose_checks += int(pose_diagnostics.get('pose_checks', 0))
                if pose_diagnostics.get('pose_enumeration_truncated', False):
                    pose_enumeration_complete = False
                    break
                if pose_count >= pose_limit:
                    break
            # A multi-component unit is also feasible when it can anchor one
            # component and move the remaining rigid components together.
            groups = _flex_port_groups(trial, uid)
            backbone_count = 0
            if len(groups) > 1 and len(groups) == len(components):
                for anchor in range(len(components)):
                    anchor_nodes = {
                        int(node.node_id)
                        for unit_id in components[anchor]
                        for node in trial.unit(unit_id).nodes
                    }
                    probe = _SiteSearchCandidate(
                        trial.clone(), frozenset(), {}, frozenset({anchor}), [],
                    )
                    probe.state.occupied_axial = {
                        probe.state.node_axial[node_id]: node_id
                        for node_id in anchor_nodes
                        if node_id in probe.state.node_axial
                    }
                    outside = [index for index in range(len(components)) if index != anchor]
                    backbone_count += len(_backbone_site_outcomes(
                        probe, uid, outside, components, max(12, int(conformer_limit)),
                    ))
            legal_count = pose_count + backbone_count
            unit_complete = bool(
                variants_complete and pose_enumeration_complete
            )
            candidate_complete = candidate_complete and unit_complete
            domains[uid] = sorted(site_sets)
            unit_rows.append(dict(
                topology_unit_id=uid,
                source_product_type=_source_product_type(trial.unit(uid)),
                ring_geometry_mode=dict(trial.unit(uid).metadata).get(
                    "ring_geometry_mode", "tree"
                ),
                conformer_count=len(variants),
                conformer_space_complete=bool(variants_complete),
                pose_enumeration_complete=bool(pose_enumeration_complete),
                pose_check_limit=pose_check_limit,
                pose_limit=pose_limit,
                pose_checks=int(pose_checks),
                interior_pose_count=int(pose_count),
                backbone_pose_count=int(backbone_count),
                legal_pose_count=int(legal_count),
                secondary_candidate_count=int(
                    pose_diagnostics.get("secondary_candidate_count", 0)
                ),
                secondary_rejection_counts=dict(
                    pose_diagnostics.get("rejection_counts", {})
                ),
                path_geometry_mismatches=int(
                    pose_diagnostics.get("path_geometry_mismatches", 0)
                ),
                distinct_site_set_count=len(site_sets),
                rejection_samples=rejection[:4],
            ))
        conflict_scope = "interior_domains"
        if len(components) > 1:
            conflict = None
            conflict_scope = "deferred_until_backbone_connection"
        else:
            conflict = domain_conflict_check(domains, complete=candidate_complete) if domains else None
        all_single_pose = all(row["legal_pose_count"] > 0 for row in unit_rows)
        verified_empty = any(
            row['legal_pose_count'] == 0
            and row['conformer_space_complete']
            and row['pose_enumeration_complete']
            for row in unit_rows
        )
        if verified_empty:
            classification = "single_unit_domain_empty"
        elif conflict is not None:
            classification = "joint_domain_conflict_in_retained_domains"
        elif not all_single_pose:
            classification = "audit_incomplete"
        else:
            classification = "joint_layout_search_required"
        rows.append(dict(
            rigid_candidate_index=int(candidate_index),
            topology_units=unit_rows,
            all_units_have_legal_pose=all_single_pose,
            classification=classification,
            candidate_space_complete=bool(candidate_complete),
            domain_conflict_scope=conflict_scope,
            domain_conflict=conflict,
        ))
        all_complete = all_complete and candidate_complete
    return dict(
        candidate_count=len(rows),
        pose_check_limit=pose_check_limit,
        topology_unit_ids=list(topology_ids),
        candidates=rows,
        candidate_space_complete=bool(all_complete),
        all_units_have_legal_pose=all(
            row["all_units_have_legal_pose"] for row in rows
        ) if rows else False,
    )


def _place_interior_pose(candidate, topology_unit_id, pose):
    rotation, translation, reflection, assignments, selected = pose
    records = _flex_port_records(candidate.state, topology_unit_id)
    trial = candidate.state.clone()
    contacts = [(n, assignments[b]) for _c, n, b, _p in records]
    if not trial.place_unit(topology_unit_id, rotation, translation,
                            allowed_contacts=contacts, reflection=reflection):
        return None
    for contract, node, boundary, _port in records:
        _append_site_contract(trial, contract, node, assignments[boundary])
    action = dict(
        kind='flex', topology_unit_id=topology_unit_id,
        contract_ids=[r[0].contract_id for r in records],
        topology_rotation=rotation, topology_reflection=reflection,
        topology_translation=list(translation), boundary_site_assignments=dict(assignments),
        planned_internal_closure=True, topology_reembedded=False, fixed_local_geometry=True,
    )
    return _SiteSearchCandidate(
        state=trial, used_site_ids=candidate.used_site_ids | selected,
        boundary_site_ids={**candidate.boundary_site_ids, **assignments},
        active_component_indices=candidate.active_component_indices,
        actions=candidate.actions + [action], anchor_index=candidate.anchor_index,
        rigid_candidate_index=candidate.rigid_candidate_index,
        remaining_candidate_counts=candidate.remaining_candidate_counts,
        remaining_candidate_sets=candidate.remaining_candidate_sets,
        domain_risk=candidate.domain_risk,
        domain_conflict=candidate.domain_conflict,
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
    state, *, candidate_k, max_steps, beam_width, strict,
    nmr_scorer=None, nmr_weight=1.0, geometry_weight=0.05, contact_weight=0.02,
    search_budget=2000, conformer_limit=12, c_puct=1.414, seed=None,
    rigid_candidates=None, search_guidance=None, max_tree_nodes=512,
    domain_check_limit=256, domain_pose_limit=32,
    max_domain_check_limit=2048, max_domain_pose_limit=512,
):
    """Jointly search backbone choices, component poses and interior domains."""
    from itertools import islice, permutations
    from .conformers import topology_conformer_info
    from .models import AssemblySearchError
    from .search import SearchBudget, PersistentMCTS, SearchObservation, domain_conflict_check

    initial = state.clone()
    components = _rigid_components(initial)
    if not components:
        raise InverseInputError('flex search requires placed aromatic components')
    if not initial.phase_done('rigid'):
        raise InverseInputError('flex search requires all rigid contracts to be satisfied')
    pending_ids = tuple({
        _topology_unit_id(initial, c) for c in initial.mandatory_contracts('flex')
    })
    # The graph-level plan identifies the cross-component backbone.  Keep it
    # ahead of internal units; once the backbone is complete, domains are
    # selected dynamically by minimum remaining legal placements.  A chain
    # blockage is a bounded input/search result, not an invitation to bypass
    # the chain stage with a ring.
    try:
        plan_hint = build_flex_assembly_plan(initial)
    except InverseInputError as exc:
        status = (
            "flex_chain_stage_blocked"
            if "flex_chain_stage_blocked" in str(exc)
            else "candidate_space_exhausted"
        )
        feasibility = audit_flex_feasibility(
            initial, rigid_candidates=rigid_candidates,
            conformer_limit=conformer_limit,
            pose_check_limit=domain_check_limit,
            pose_limit=domain_pose_limit,
            candidate_limit=max(1, int(candidate_k)),
        )
        plan = _site_search_plan(
            initial, components, 0, [], pending_ids, [],
        )
        diagnostics = {
            "status": status,
            "failure_scope": "flex_chain_stage" if status == "flex_chain_stage_blocked" else "planning",
            "candidate_space_complete": bool(
                feasibility.get("candidate_space_complete", False)
            ),
            "feasibility_audit": feasibility,
            "planning_errors": [str(exc)],
            "completed_contract_count": 0,
            "zero_domain_units": [],
            "best_partial_state": {
                "placed_unit_ids": sorted(
                    str(uid) for uid in initial.placements
                    if str(uid) in set(pending_ids)
                ),
                "satisfied_contract_ids": [],
                "completed_contract_count": 0,
                "remaining_candidate_counts": {},
                "zero_domain_units": [],
                "component_count": int(initial.component_count()),
                "action_count": 0,
            },
            "pending_topologies": list(pending_ids),
        }
        row = TopologyPlacementStage(initial, "flex").get_result()
        row.update(
            done=False,
            status=status,
            failure_scope=diagnostics["failure_scope"],
            candidate_space_complete=diagnostics["candidate_space_complete"],
            completed_contract_count=0,
            zero_domain_units=[],
            best_partial_state=diagnostics["best_partial_state"],
            planning_errors=[str(exc)],
            search_diagnostics=diagnostics,
            flex_stage_order=("flex_chain", "aliphatic_ring"),
            plan=plan.to_dict(),
        )
        result = SearchResult(
            initial, float(TopologyPlacementStage(initial, "flex").score()),
            [], {"phase": "flex", "result": row},
        )
        if strict:
            raise AssemblySearchError(str(exc), diagnostics)
        return result, plan
    backbone_rank = {
        str(uid): index for index, uid in enumerate(plan_hint.backbone_topology_unit_ids)
    }
    stage_rank = {
        str(uid): _flex_stage_rank(initial.unit(uid)) for uid in pending_ids
    }
    pending = tuple(sorted(
        pending_ids,
        key=lambda uid: (
            stage_rank.get(uid, 0),
            0 if uid in backbone_rank else 1,
            backbone_rank.get(uid, 10**9),
            _topology_complexity_key(initial.unit(uid)),
            _unit_sort_key(uid),
        ),
    ))
    complexity = {uid: _topology_complexity_key(initial.unit(uid)) for uid in pending}
    contract_count = len(initial.mandatory_contracts('flex'))
    if contract_count > max_steps:
        message = f'flex contract budget {max_steps} is smaller than {contract_count}'
        diagnostics = {'status': 'budget_exhausted', 'required_contracts': contract_count,
                       'visited_states': 0, 'failure_scope': 'search_budget',
                       'candidate_space_complete': False,
                       'completed_contract_count': 0,
                       'zero_domain_units': list(pending),
                       'best_partial_state': {
                           'placed_unit_ids': sorted(initial.placements),
                           'satisfied_contract_ids': [],
                           'completed_contract_count': 0,
                           'zero_domain_units': list(pending),
                       }}
        if strict:
            raise AssemblySearchError(message, diagnostics)
        plan = _site_search_plan(initial, components, 0, [], pending, [])
        row = TopologyPlacementStage(initial, 'flex').get_result()
        row.update(done=False, status='budget_exhausted', planning_errors=[message],
                   failure_scope='search_budget', candidate_space_complete=False,
                   completed_contract_count=0, zero_domain_units=list(pending),
                   best_partial_state=diagnostics['best_partial_state'],
                   search_diagnostics=diagnostics, plan=plan.to_dict())
        return SearchResult(initial, 0.0, [], {'phase': 'flex', 'result': row}), plan
    rigid_candidates = list(rigid_candidates) if rigid_candidates is not None else [initial]
    # The rigid beam is a retained candidate pool, not an invitation to keep
    # every full state in the Flex tree.  Only the configured proposal width
    # participates in the joint search; truncation is reported as bounded
    # search rather than treated as a proof of infeasibility.
    if len(rigid_candidates) > max(1, int(candidate_k)):
        rigid_candidates = rigid_candidates[:max(1, int(candidate_k))]
    root = _SiteSearchCandidate(initial, frozenset(), {}, frozenset(), [], rigid_candidate_index=-1)
    best = root
    committed = {}
    # Keep a useful retained pool even when callers use a small legacy beam;
    # the PUCT tree still controls how many of these alternatives are visited.
    width = max(8, int(candidate_k), int(beam_width))
    budget = SearchBudget(max(1, int(search_budget)))
    # Each retained node owns a full state snapshot.  Keep the tree bounded
    # independently of the simulation budget so a notebook kernel cannot be
    # killed by a long-lived PUCT frontier.
    # NMR scoring already keeps a sizeable G2S model and graph cache resident;
    # use a smaller retained frontier in that mode.  The simulation budget is
    # still reported separately, so this is a memory bound rather than an
    # infeasibility claim.
    max_tree_nodes = max(16, int(max_tree_nodes))
    if nmr_scorer is not None:
        max_tree_nodes = min(max_tree_nodes, 64)
    domain_pose_limit = max(1, int(domain_pose_limit))
    max_domain_pose_limit = max(domain_pose_limit, int(max_domain_pose_limit))
    domain_check_limit = max(1, int(domain_check_limit))
    max_domain_check_limit = max(domain_check_limit, int(max_domain_check_limit))
    domain_pose_limits: Dict[str, int] = {}
    domain_check_limits: Dict[str, int] = {}
    diagnostics = dict(candidate_limit=width, conformer_limit=conformer_limit,
                       search_budget=budget.limit, candidate_pool_limited=False,
                       last_dead_end=None, topology_ordering='backbone_then_mrv',
                       topology_priority_order=list(pending),
                       topology_stage_order='flex_chain_then_aliphatic_ring',
                       flex_chain_units_total=sum(
                           stage_rank.get(uid, 0) == 0 for uid in pending
                       ),
                       aliphatic_ring_units_total=sum(
                           stage_rank.get(uid, 0) == 1 for uid in pending
                       ),
                       candidate_space_complete=True, feasibility_audit=None,
                       domain_risks=0, domain_scan_calls=0,
                       domain_scan_pose_checks=0, domain_scan_truncations=0,
                       domain_scan_check_limit=domain_check_limit,
                       domain_scan_pose_limit=domain_pose_limit)
    slots = {}
    protected = _rigid_endpoint_ids(initial)

    feasibility = audit_flex_feasibility(
        initial, rigid_candidates=rigid_candidates, conformer_limit=conformer_limit,
        pose_check_limit=domain_check_limit,
        pose_limit=domain_pose_limit,
        candidate_limit=max(1, int(candidate_k)),
    )
    diagnostics['feasibility_audit'] = feasibility
    diagnostics['candidate_space_complete'] = bool(
        feasibility.get('candidate_space_complete', True)
    )
    base_conformer_limit = max(1, int(conformer_limit))
    max_conformer_limit = min(96, max(base_conformer_limit, 8 * base_conformer_limit))
    variant_limits: Dict[str, int] = {}

    def remaining(candidate):
        unplaced = [uid for uid in pending if not candidate.state.is_placed(uid)]
        chains = [uid for uid in unplaced if stage_rank.get(uid, 0) == 0]
        # Do not expose ring domains until every flexible chain has been
        # placed.  This is a scheduling barrier, not a topology mutation.
        return tuple(chains if chains else unplaced)

    def variants(candidate, uid):
        original = candidate.state.conformer_sources.get(uid, candidate.state.unit(uid))
        limit = variant_limits.get(uid, base_conformer_limit)
        conformers, complete = topology_conformer_info(original, limit)
        if not complete:
            diagnostics['candidate_space_complete'] = False
        diagnostics.setdefault('conformer_generation', {})[uid] = dict(
            count=len(conformers), complete=bool(complete), limit=int(limit),
        )
        for index, unit in enumerate(conformers):
            trial = candidate.state.clone()
            trial.select_conformer(unit)
            yield index, replace(candidate, state=trial, shape_rank=())

    def _mrv_probe(start_candidates):
        """Run a bounded MRV backtracking pass before widening PUCT.

        PUCT remains the persistent search used for difficult/incomplete
        domains.  The probe is deliberately small and uses the same outcome
        generators as PUCT, but follows the inverse stage barriers directly.
        This matters for the common case where a valid chain/ring layout is
        only a handful of backtracks away and a bounded tree would otherwise
        spend its node budget on equivalent sibling poses.
        """
        probe_limit = min(
            max(1, int(search_budget) // 4),
            max(1, int(max_tree_nodes) * 2),
        )
        visited = 0
        best_probe = None
        probe_contract_ids = {
            str(contract.contract_id)
            for contract in initial.problem.contracts
            if contract.kind == "flex"
        }
        seen = set()

        def probe_rank(candidate):
            completed = len(
                probe_contract_ids & set(candidate.state.satisfied_contracts)
            )
            placed = sum(
                candidate.state.is_placed(uid) for uid in pending
            )
            return (
                int(completed),
                int(placed),
                -len(remaining(candidate)) if candidate.rigid_candidate_index >= 0 else 0,
                -len(candidate.actions),
            )

        def annotate_action(candidate, stage, conformer_index):
            if not candidate.actions:
                return candidate
            action = candidate.actions[-1]
            action["conformer_index"] = int(conformer_index)
            action["flex_stage"] = str(stage)
            action["geometry_source"] = dict(
                candidate.state.unit(action["topology_unit_id"]).metadata
            ).get("geometry_source", "fixed_input")
            return candidate

        def backbone_children(candidate):
            outside = sorted(
                set(range(len(components)))
                - set(candidate.active_component_indices)
            )
            if not outside:
                return []
            children = []
            for uid in plan_hint.backbone_topology_unit_ids:
                if candidate.state.is_placed(uid):
                    continue
                component_ids = {
                    next(
                        index for index, component in enumerate(components)
                        if _aromatic_endpoint(candidate.state, contract)[0]
                        in component
                    )
                    for contract in initial.problem.contracts
                    if contract.kind == "flex"
                    and _topology_unit_id(initial, contract) == uid
                }
                outside_for_unit = sorted(set(outside) & component_ids)
                inside_for_unit = component_ids - set(outside)
                if not inside_for_unit or not outside_for_unit:
                    continue
                if len(_flex_port_groups(candidate.state, uid)) != len(outside_for_unit) + 1:
                    continue
                for index, variant in variants(candidate, uid):
                    outcomes = _backbone_site_outcomes(
                        variant,
                        uid,
                        outside_for_unit,
                        components,
                        max(1, min(width, int(candidate_k))),
                    )
                    for outcome in outcomes:
                        annotate_action(outcome, "flex_chain", index)
                        children.append(outcome)
                if children:
                    # The plan is a spanning order.  Do not mix a later
                    # backbone topology into this level of the probe.
                    break
            return children

        def interior_children(candidate, ids):
            slot_cache = {}
            domains = {}
            for uid in ids:
                options = []
                scan_limit = min(
                    max_domain_check_limit,
                    max(domain_check_limit, domain_check_limits.get(uid, domain_check_limit)),
                )
                option_limit = min(
                    max_domain_pose_limit,
                    max(domain_pose_limit, domain_pose_limits.get(uid, domain_pose_limit)),
                )
                checks = 0
                for index, variant in variants(candidate, uid):
                    if checks >= scan_limit or len(options) >= option_limit:
                        break
                    pose_diagnostics = {}
                    for pose in _interior_site_poses(
                        variant,
                        uid,
                        components,
                        slot_cache,
                        pose_diagnostics,
                        max_checks=max(1, scan_limit - checks),
                    ):
                        options.append((index, variant.state.unit(uid), pose))
                        if len(options) >= option_limit:
                            break
                    checks += int(pose_diagnostics.get("pose_checks", 0))
                    if pose_diagnostics.get("pose_enumeration_truncated", False):
                        break
                domains[uid] = options
            if not domains:
                return []
            uid = min(
                domains,
                key=lambda value: (
                    len(domains[value]),
                    _topology_complexity_key(candidate.state.unit(value)),
                    -len(_flex_port_records(candidate.state, value)),
                    _unit_sort_key(value),
                ),
            )
            children = []
            for index, unit, pose in domains[uid]:
                trial = candidate.state.clone()
                trial.select_conformer(unit)
                outcome = _place_interior_pose(
                    replace(candidate, state=trial), uid, pose
                )
                if outcome is None:
                    continue
                annotate_action(
                    outcome,
                    "flex_chain" if stage_rank.get(uid, 0) == 0 else "aliphatic_ring",
                    index,
                )
                children.append(outcome)
            return children

        def visit(candidate):
            nonlocal visited, best_probe
            if visited >= probe_limit:
                return None
            visited += 1
            if best_probe is None or probe_rank(candidate) > probe_rank(best_probe):
                best_probe = candidate
            signature = (
                candidate.state.state_signature(),
                tuple(sorted(candidate.active_component_indices)),
            )
            if signature in seen:
                return None
            seen.add(signature)
            outside = set(range(len(components))) - set(candidate.active_component_indices)
            if outside:
                children = backbone_children(candidate)
            else:
                unplaced = [uid for uid in pending if not candidate.state.is_placed(uid)]
                chains = [uid for uid in unplaced if stage_rank.get(uid, 0) == 0]
                children = interior_children(candidate, chains if chains else unplaced)
            if not children:
                if not outside and not [uid for uid in pending if not candidate.state.is_placed(uid)]:
                    return candidate
                return None
            # Keep all legal choices available for backtracking, while trying
            # compact states first to reduce the first-solution latency.
            children.sort(key=_cheap_site_candidate_rank)
            for child in children:
                solved_candidate = visit(child)
                if solved_candidate is not None:
                    return solved_candidate
            return None

        for source_index, source in enumerate(start_candidates):
            anchors = sorted(
                range(len(components)),
                key=lambda index: -sum(
                    len(initial.unit(uid).nodes) for uid in components[index]
                ),
            )
            for anchor in anchors:
                anchor_nodes = {
                    node.node_id
                    for uid in components[anchor]
                    for node in initial.unit(uid).nodes
                }
                trial = source.clone()
                trial.occupied_axial = {
                    axial: node_id
                    for node_id, axial in trial.node_axial.items()
                    if node_id in anchor_nodes
                }
                root_candidate = _SiteSearchCandidate(
                    trial,
                    frozenset(),
                    {},
                    frozenset({anchor}),
                    [],
                    anchor_index=anchor,
                    rigid_candidate_index=source_index,
                )
                solved_candidate = visit(root_candidate)
                if solved_candidate is not None:
                    return solved_candidate, best_probe, visited
        return None, best_probe, visited

    def annotate(outcomes, index):
        for outcome in outcomes:
            outcome.actions[-1]['conformer_index'] = index
            outcome.actions[-1]['flex_stage'] = (
                'flex_chain'
                if stage_rank.get(str(outcome.actions[-1].get('topology_unit_id')), 0) == 0
                else 'aliphatic_ring'
            )
            outcome.actions[-1]['geometry_source'] = dict(
                outcome.state.unit(outcome.actions[-1]['topology_unit_id']).metadata
            ).get('geometry_source', 'fixed_input')
        return outcomes

    def ranked(outcomes, depth):
        # Score in batches without dropping alternatives on NMR/shape score.
        _score_site_candidates(
            outcomes, nmr_scorer=nmr_scorer, depth=depth,
            nmr_weight=nmr_weight, geometry_weight=geometry_weight,
            contact_weight=contact_weight, protected_aromatic_ids=protected,
        )
        return sorted(outcomes, key=_site_candidate_rank if nmr_scorer else _cheap_site_candidate_rank)

    def expand(candidate):
        if candidate.rigid_candidate_index < 0:
            for index, rigid in enumerate(rigid_candidates):
                yield replace(candidate, state=rigid.clone(), rigid_candidate_index=index)
            return
        if not candidate.active_component_indices:
            anchors = sorted(range(len(components)), key=lambda i: -sum(
                len(initial.unit(uid).nodes) for uid in components[i]))
            for anchor in anchors:
                anchor_nodes = {n.node_id for uid in components[anchor]
                                for n in initial.unit(uid).nodes}
                trial = candidate.state.clone()
                trial.occupied_axial = {p: n for n, p in trial.node_axial.items() if n in anchor_nodes}
                yield replace(candidate, state=trial, anchor_index=anchor,
                              active_component_indices=frozenset({anchor}))
            return
        ids = remaining(candidate)
        if not ids:
            return
        outside = sorted(set(range(len(components))) - set(candidate.active_component_indices))
        if outside:
            # The identity of a backbone topology is a search decision, not a
            # preselected fixed spanning tree, but only chain units that can
            # actually merge a remaining rigid component are legal here. Do
            # not spend simulations probing every interior chain and then
            # let a ring or an unrelated chain consume the backbone budget.
            backbone_ids = tuple(
                uid for uid in ids
                if uid in set(plan_hint.backbone_topology_unit_ids)
            )
            for uid in backbone_ids:
                current_limit = variant_limits.get(uid, base_conformer_limit)
                while True:
                    if current_limit > base_conformer_limit:
                        variant_limits[uid] = current_limit
                    # One bounded proposal wave per topology keeps the tree
                    # deep enough to reach interior placement. A previous
                    # 8/16/32 wave sequence filled the node cap with shallow
                    # backbone alternatives and never gave later units a
                    # chance to drive backtracking.
                    proposal_width = max(1, min(4, width))
                    for index, variant in variants(candidate, uid):
                        expansion = len(_flex_port_groups(variant.state, uid)) - 1
                        if not 0 < expansion <= len(outside):
                            continue
                        permutation_count = math.factorial(len(outside)) // math.factorial(len(outside) - expansion)
                        if permutation_count > proposal_width:
                            diagnostics['candidate_pool_limited'] = True
                            diagnostics['candidate_space_complete'] = False
                        for chosen in islice(permutations(outside, expansion), proposal_width):
                            outcomes = annotate(_backbone_site_outcomes(
                                variant, uid, chosen, components, proposal_width,
                                diagnostics), index)
                            if len(outcomes) >= proposal_width:
                                diagnostics['candidate_pool_limited'] = True
                                diagnostics['candidate_space_complete'] = False
                            for outcome in ranked(outcomes, len(candidate.actions) + 1):
                                yield outcome
                    generation = diagnostics.get('conformer_generation', {}).get(uid, {})
                    current_limit = int(generation.get('limit', current_limit))
                    if generation.get('complete', True) or current_limit >= max_conformer_limit:
                        break
                    current_limit = min(max_conformer_limit, max(current_limit + 1, 2 * current_limit))
                    variant_limits[uid] = current_limit
                    diagnostics['candidate_pool_expansions'] = diagnostics.get('candidate_pool_expansions', 0) + 1
            return

        selected_domain = None
        site_pairs = {}
        domain_complete = {}

        def collect_domains():
            nonlocal selected_domain, site_pairs, domain_complete
            selected_domain = None
            site_pairs = {}
            domain_complete = {}
            for uid in ids:
                scan_limit = domain_check_limits.get(uid, domain_check_limit)
                option_limit = domain_pose_limits.get(uid, domain_pose_limit)
                while True:
                    options = []
                    checks = 0
                    scan_complete = True
                    for index, variant in variants(candidate, uid):
                        remaining_checks = scan_limit - checks
                        if remaining_checks <= 0:
                            scan_complete = False
                            break
                        pose_diagnostics = {}
                        for pose in _interior_site_poses(
                            variant, uid, components, slots, pose_diagnostics,
                            max_checks=remaining_checks,
                        ):
                            options.append((index, variant.state.unit(uid), pose))
                            if len(options) >= option_limit:
                                scan_complete = False
                                diagnostics['candidate_pool_limited'] = True
                                diagnostics['candidate_space_complete'] = False
                                break
                        checks += int(pose_diagnostics.get('pose_checks', 0))
                        if pose_diagnostics.get('pose_enumeration_truncated', False):
                            scan_complete = False
                        if not scan_complete:
                            break
                    generation = diagnostics.get('conformer_generation', {}).get(uid, {})
                    conformers_complete = bool(generation.get('complete', True))
                    domain_complete[uid] = bool(scan_complete and conformers_complete)
                    diagnostics['domain_scan_calls'] += 1
                    diagnostics['domain_scan_pose_checks'] += checks
                    if not domain_complete[uid]:
                        diagnostics['candidate_pool_limited'] = True
                        diagnostics['candidate_space_complete'] = False
                        diagnostics['domain_scan_truncations'] += 1

                    # An empty sample is not a dead end until the per-unit
                    # scan budget has been widened enough to either find a
                    # pose or exhaust the retained domain.
                    if not options and not domain_complete[uid]:
                        if scan_limit < max_domain_check_limit:
                            domain_check_limits[uid] = min(
                                max_domain_check_limit, scan_limit * 2,
                            )
                            diagnostics['candidate_pool_expansions'] = diagnostics.get('candidate_pool_expansions', 0) + 1
                    current_limit = int(generation.get('limit', base_conformer_limit))
                    if (
                        not options and not conformers_complete
                        and current_limit < max_conformer_limit
                    ):
                        variant_limits[uid] = min(
                            max_conformer_limit,
                            max(current_limit + 1, 2 * current_limit),
                        )
                        diagnostics['candidate_pool_expansions'] = diagnostics.get('candidate_pool_expansions', 0) + 1
                    break
                site_pairs[uid] = sorted({
                    tuple(sorted(pose[4])) for _i, _u, pose in options
                })
                if not options:
                    diagnostics['last_dead_end'] = dict(
                        topology=uid, placed=[a['topology_unit_id'] for a in candidate.actions],
                        used_site_ids=sorted(candidate.used_site_ids),
                        candidate_site_sets=site_pairs,
                        boundary_site_assignments=dict(candidate.boundary_site_ids),
                        active_component_indices=sorted(candidate.active_component_indices),
                        site_coordinates={str(n): list(candidate.state.node_axial[n])
                                          for n in sorted(set(candidate.used_site_ids) | {
                                              n for pairs in site_pairs.values() for pair in pairs for n in pair
                                          })},
                        placements={uid: dict(rotation=p.rotation, translation=list(p.translation),
                                              reflection=p.reflection)
                                    for uid, p in candidate.state.placements.items()},
                        candidate_space_complete=bool(domain_complete[uid]),
                        domain_scan_pose_checks=int(checks),
                        domain_scan_check_limit=int(scan_limit),
                    )
                    if not domain_complete[uid]:
                        diagnostics['domain_risks'] = diagnostics.get('domain_risks', 0) + 1
                    return False
                priority = (
                    len(site_pairs[uid]), complexity[uid], len(options), _unit_sort_key(uid)
                )
                if selected_domain is None or priority < selected_domain[0]:
                    selected_domain = priority, uid, options
            return True

        if not collect_domains():
            return
        conflict = None
        conflict_expanded = False
        while True:
            pool_complete = all(
                diagnostics.get('conformer_generation', {}).get(uid, {}).get('complete', True)
                for uid in ids
            ) and all(domain_complete.values())
            conflict = domain_conflict_check(site_pairs, complete=pool_complete)
            if conflict is None or conflict.get('hard', False):
                break
            if conflict_expanded:
                break
            expanded = False
            for uid in conflict.get('topologies', []):
                current_check_limit = domain_check_limits.get(uid, domain_check_limit)
                current_pose_limit = domain_pose_limits.get(uid, domain_pose_limit)
                if current_check_limit < max_domain_check_limit:
                    domain_check_limits[uid] = min(
                        max_domain_check_limit, current_check_limit * 2,
                    )
                    expanded = True
                if current_pose_limit < max_domain_pose_limit:
                    domain_pose_limits[uid] = min(
                        max_domain_pose_limit, current_pose_limit * 2,
                    )
                    expanded = True
                generation = diagnostics.get('conformer_generation', {}).get(uid, {})
                current_limit = int(generation.get('limit', base_conformer_limit))
                if generation.get('complete', True) or current_limit >= max_conformer_limit:
                    pass
                else:
                    variant_limits[uid] = min(
                        max_conformer_limit, max(current_limit + 1, 2 * current_limit),
                    )
                    expanded = True
            if not expanded:
                break
            conflict_expanded = True
            diagnostics['candidate_pool_expansions'] = diagnostics.get('candidate_pool_expansions', 0) + 1
            if not collect_domains():
                return
        domain_counts = tuple(sorted((uid, len(site_pairs[uid])) for uid in ids))
        if conflict is not None:
            diagnostics['last_domain_conflict'] = dict(
                **conflict, placed=[a['topology_unit_id'] for a in candidate.actions],
                rigid_candidate_index=candidate.rigid_candidate_index)
            if conflict.get('hard', False):
                diagnostics['domain_prunes'] = diagnostics.get('domain_prunes', 0) + 1
                return
            # An incomplete candidate pool can only make this a risk score;
            # retain the domains so a later conformer expansion/backtrack can
            # discover a missing placement-domain option.
            diagnostics['domain_risks'] = diagnostics.get('domain_risks', 0) + 1
        if selected_domain:
            _priority, uid, options = selected_domain
            if not domain_complete.get(uid, True):
                expanded = False
                current_check_limit = domain_check_limits.get(uid, domain_check_limit)
                if current_check_limit < max_domain_check_limit:
                    domain_check_limits[uid] = min(
                        max_domain_check_limit, current_check_limit * 2,
                    )
                    expanded = True
                current_pose_limit = domain_pose_limits.get(uid, domain_pose_limit)
                if current_pose_limit < max_domain_pose_limit:
                    domain_pose_limits[uid] = min(
                        max_domain_pose_limit, current_pose_limit * 2,
                    )
                    expanded = True
                generation = diagnostics.get('conformer_generation', {}).get(uid, {})
                current_variant_limit = int(generation.get('limit', base_conformer_limit))
                if (
                    not generation.get('complete', True)
                    and current_variant_limit < max_conformer_limit
                ):
                    variant_limits[uid] = min(
                        max_conformer_limit,
                        max(current_variant_limit + 1, 2 * current_variant_limit),
                    )
                    expanded = True
                if expanded:
                    diagnostics['candidate_pool_expansions'] = diagnostics.get('candidate_pool_expansions', 0) + 1
            # Only the chosen topology needs full state copies. Other domains
            # contain immutable units and lightweight pose descriptions.
            chosen = []
            for index, unit, pose in options:
                trial = candidate.state.clone()
                trial.select_conformer(unit)
                outcome = _place_interior_pose(replace(candidate, state=trial), uid, pose)
                if outcome is not None:
                    outcome.remaining_candidate_counts = tuple(
                        (domain_uid, count) for domain_uid, count in domain_counts
                        if domain_uid != uid
                    )
                    outcome.remaining_candidate_sets = tuple(
                        (domain_uid, tuple(
                            tuple(int(site_id) for site_id in option)
                            for option in options_for_uid
                        ))
                        for domain_uid, options_for_uid in sorted(site_pairs.items())
                        if domain_uid != uid
                    )
                    outcome.domain_conflict = conflict
                    outcome.domain_risk = 0.0 if conflict is None else 1.0
                    chosen.extend(annotate([outcome], index))
                if len(chosen) >= width:
                    yield from ranked(chosen, len(candidate.actions) + 1)
                    chosen = []
            for outcome in ranked(chosen, len(candidate.actions) + 1):
                yield outcome

    def accept(candidate):
        if candidate.rigid_candidate_index < 0:
            return False
        if remaining(candidate) or len(candidate.active_component_indices) != len(components):
            return False
        normalized, rotation = _normalize_global_orientation(candidate)
        try:
            final = relabel_aromatic_sites(normalized.state,
                                            normalized.boundary_site_ids, normalized.actions)
        except InverseInputError as exc:
            diagnostics['commit_rejection'] = str(exc)
            return False
        committed.update(state=final, candidate=normalized, rotation=rotation)
        return True

    flex_contract_ids = {
        str(contract.contract_id)
        for contract in initial.problem.contracts
        if contract.kind == 'flex'
    }

    def partial_rank(candidate):
        completed = len(flex_contract_ids & set(candidate.state.satisfied_contracts))
        if candidate.remaining_candidate_sets:
            used_sites = set(candidate.used_site_ids)
            counts = [
                sum(not (set(option) & used_sites) for option in options)
                for _uid, options in candidate.remaining_candidate_sets
            ]
        else:
            counts = [int(value) for _uid, value in candidate.remaining_candidate_counts]
        zero_domains = sum(value <= 0 for value in counts)
        min_remaining = min(counts) if counts else (10**9 if not remaining(candidate) else 0)
        total_remaining = sum(max(0, int(value)) for value in counts)
        components_left = len(components) - len(candidate.active_component_indices)
        progress = len(candidate.actions) / max(1, len(pending))
        return (
            completed,
            -zero_domains,
            min_remaining,
            total_remaining,
            -float(candidate.domain_risk),
            -components_left,
            progress,
        )

    def partial_state_summary(candidate):
        """Return bounded diagnostics for the best retained partial state.

        Do not serialize a second full HexAssemblyState here: stage outputs
        already contain the selected molecule, while this summary is enough
        to explain why a branch was retained or rejected.
        """
        if candidate.remaining_candidate_sets:
            used_sites = set(candidate.used_site_ids)
            counts = {
                str(uid): sum(
                    not (set(option) & used_sites)
                    for option in options
                )
                for uid, options in candidate.remaining_candidate_sets
            }
        else:
            counts = {
                str(uid): int(value)
                for uid, value in candidate.remaining_candidate_counts
            }
        zero = sorted(uid for uid, count in counts.items() if count <= 0)
        completed = len(
            flex_contract_ids & set(candidate.state.satisfied_contracts)
        )
        return {
            "placed_unit_ids": sorted(
                str(uid) for uid in candidate.state.placements
                if str(uid) in set(pending)
            ),
            "satisfied_contract_ids": sorted(
                str(cid) for cid in candidate.state.satisfied_contracts
                if str(cid) in flex_contract_ids
            ),
            "completed_contract_count": int(completed),
            "remaining_candidate_counts": counts,
            "zero_domain_units": zero,
            "minimum_remaining_candidates": (
                min(counts.values()) if counts else None
            ),
            "total_remaining_candidates": int(sum(max(0, value) for value in counts.values())),
            "component_count": int(candidate.state.component_count()),
            "active_component_indices": sorted(
                int(value) for value in candidate.active_component_indices
            ),
            "action_count": int(len(candidate.actions)),
            "rigid_candidate_index": int(candidate.rigid_candidate_index),
        }

    def observe(candidate):
        nonlocal best
        if partial_rank(candidate) > partial_rank(best):
            best = candidate

    def action(parent, child):
        if parent.rigid_candidate_index < 0:
            return dict(kind='select_rigid', rigid_candidate_index=child.rigid_candidate_index)
        if not parent.active_component_indices:
            return dict(kind='select_anchor', anchor_index=child.anchor_index)
        return child.actions[-1]

    def value(candidate):
        completed_contracts = len(
            flex_contract_ids & set(candidate.state.satisfied_contracts)
        )
        contract_progress = completed_contracts / max(1, len(flex_contract_ids))
        progress = len(candidate.actions) / max(1, len(pending))
        if candidate.remaining_candidate_sets:
            used_sites = set(candidate.used_site_ids)
            counts = [
                sum(not (set(option) & used_sites) for option in options)
                for _uid, options in candidate.remaining_candidate_sets
            ]
        else:
            counts = [int(value) for _uid, value in candidate.remaining_candidate_counts]
        zero_domains = sum(value <= 0 for value in counts)
        min_remaining = min(counts) if counts else (0 if remaining(candidate) else 1)
        total_remaining = sum(max(0, int(value)) for value in counts)
        # Logarithmic domain slack rewards states that leave every remaining
        # unit room, while zero domains receive an explicit penalty.
        slack = math.tanh(math.log1p(max(0, min_remaining)) / 2.0)
        total_slack = math.tanh(math.log1p(max(0, total_remaining)) / 4.0)
        component_progress = len(candidate.active_component_indices) / max(1, len(components))
        risk = min(1.0, max(0.0, float(candidate.domain_risk)))
        quality = 0.0 if candidate.nmr_rank is None else math.tanh(candidate.nmr_rank)
        value = (
            0.44 * contract_progress
            + 0.16 * component_progress
            + 0.12 * progress
            + 0.12 * slack
            + 0.08 * total_slack
            - 0.32 * min(1.0, float(zero_domains))
            - 0.10 * risk
            - 0.04 * quality
        )
        return max(-1.0, min(1.0, value))

    probe_solution, probe_best, probe_nodes = _mrv_probe(rigid_candidates)
    budget.simulations += probe_nodes
    budget.visited += probe_nodes
    diagnostics["mrv_probe_nodes"] = int(probe_nodes)
    diagnostics["mrv_probe_solved"] = bool(probe_solution is not None)
    if probe_best is not None and partial_rank(probe_best) > partial_rank(best):
        best = probe_best

    tree = PersistentMCTS(
        root, expand=expand, accept=accept, observe=observe, evaluate=value, action=action,
        guidance=(search_guidance.adapted(lambda c: SearchObservation('flex', c.state, dict(
            used_site_ids=c.used_site_ids, boundary_site_ids=dict(c.boundary_site_ids),
            active_component_indices=c.active_component_indices,
            rigid_candidate_index=c.rigid_candidate_index,
            rigid_candidates=tuple(rigid_candidates) if c.rigid_candidate_index < 0 else (),
        ))) if search_guidance is not None else None), c_puct=c_puct, seed=seed,
        signature=lambda c: (c.state.state_signature(), tuple(sorted(c.boundary_site_ids.items())),
                             tuple(sorted(c.active_component_indices)), c.rigid_candidate_index),
        max_nodes=max_tree_nodes,
        # The public adapter budgets each rigid layout independently.
        root_child_limit=None,
    )
    try:
        if probe_solution is not None and accept(probe_solution):
            solved = probe_solution
        else:
            solved = tree.run(budget)
        tree.record()
    finally:
        # The selected state owns its chosen immutable unit; the global
        # conformer cache does not need to retain every discarded variant
        # after this bounded search returns.
        topology_conformer_info.cache_clear()
    selected = committed.get('candidate', best)
    current = committed.get('state', best.state).clone()
    if 'state' not in committed and selected.boundary_site_ids:
        # A partial result must use the same actual endpoints as its picture.
        # Replay failure is an error, never a license to export stale contracts.
        current = relabel_aromatic_sites(
            selected.state, selected.boundary_site_ids, selected.actions,
            protected_aromatic_ids=protected,
        )
    current.stage = 'flex'
    backbone_ids = [a['topology_unit_id'] for a in selected.actions
                    if a.get('planned_flex_backbone_merge')]
    interior_ids = [uid for uid in pending if uid not in backbone_ids]
    plan = _site_search_plan(initial, components, selected.anchor_index, backbone_ids, interior_ids, selected.actions)
    if solved is not None:
        status = 'complete'
    elif budget.exhausted:
        status = 'budget_exhausted'
    elif diagnostics.get('candidate_space_complete', False):
        status = 'proven_infeasible'
    else:
        status = 'candidate_space_exhausted'
    diagnostics.update(status=status, visited_states=budget.visited, backtracks=budget.backtracks,
                       simulations=budget.simulations, mcts=tree.diagnostics(),
                       pending_topologies=list(remaining(best)) if solved is None else [],
                       bounded_search=(status != 'proven_infeasible'),
                       failure_scope=(
                           'none' if solved is not None else
                           'search_budget' if status == 'budget_exhausted' else
                           'bounded_candidate_pool' if status == 'candidate_space_exhausted' else
                           'exhausted_flex_domains_for_retained_rigid_candidates'
                       ),
                       rigid_candidate_scope='retained_rigid_candidates',
                       active_component_indices=sorted(selected.active_component_indices),
                       completed_contract_count=len(
                           flex_contract_ids & set(current.satisfied_contracts)
                       ),
                       zero_domain_units=partial_state_summary(selected)[
                           'zero_domain_units'
                       ],
                       best_partial_state=partial_state_summary(selected),
                       candidate_space_complete=bool(
                           diagnostics.get('candidate_space_complete', False)
                       ))
    message = (f'flex MCTS {status}: placed {len(selected.actions)}/{len(pending)} topology units; '
               f'simulations={budget.simulations}/{budget.limit}, tree_nodes={len(tree.nodes)}')
    row = TopologyPlacementStage(current, 'flex').get_result()
    backbone_done = sum(cid in current.satisfied_contracts for cid in plan.backbone_contract_ids)
    interior_done = sum(cid in current.satisfied_contracts for cid in plan.interior_contract_ids)
    row.update(
        done=solved is not None, status=status,
        failure_scope=diagnostics.get('failure_scope', 'none'),
        candidate_space_complete=bool(
            diagnostics.get('candidate_space_complete', False)
        ),
        completed_contract_count=int(
            diagnostics.get('completed_contract_count', 0)
        ),
        zero_domain_units=list(diagnostics.get('zero_domain_units', [])),
        best_partial_state=diagnostics.get('best_partial_state', {}),
        rigid_components_total=len(components), backbone_topology_units=len(backbone_ids),
        backbone_contracts=len(plan.backbone_contract_ids), backbone_contracts_done=backbone_done,
        backbone_complete=backbone_done == len(plan.backbone_contract_ids),
        interior_topology_units=len(interior_ids), interior_contracts=len(plan.interior_contract_ids),
        interior_contracts_done=interior_done,
        interior_complete=interior_done == len(plan.interior_contract_ids),
        flex_chain_units_total=sum(stage_rank.get(uid, 0) == 0 for uid in pending),
        flex_chain_units_placed=sum(
            stage_rank.get(uid, 0) == 0 and current.is_placed(uid)
            for uid in pending
        ),
        aliphatic_ring_units_total=sum(stage_rank.get(uid, 0) == 1 for uid in pending),
        aliphatic_ring_units_placed=sum(
            stage_rank.get(uid, 0) == 1 and current.is_placed(uid)
            for uid in pending
        ),
        flex_stage_order=("flex_chain", "aliphatic_ring"),
        planned_internal_closures=interior_done,
        planning_errors=[] if solved is not None else [message],
        search_diagnostics=diagnostics, explored_geometry_states=budget.visited,
        topology_reembedding_used=False, selected_conformer_count=len(current.conformer_sources),
        global_orientation_rotation=committed.get('rotation', 0),
        coordinate_aspect=_coordinate_aspect(current), plan=plan.to_dict(),
        slot_geometry=dict(mode='search_time_physical_su13_selection',
                           committed=True,
                           boundary_slot_bindings={gid: slot for gid, slot in current.aromatic_site_bindings().items()
                                                   if int(gid) in selected.boundary_site_ids},
                           boundary_site_assignments=dict(selected.boundary_site_ids)),
        search_algorithm='persistent_puct', search_width=width, candidate_k=width,
        selected_rigid_candidate_index=max(0, selected.rigid_candidate_index),
        rigid_search_attempts=[dict(rigid_candidate_index=child.state.rigid_candidate_index,
                                   diagnostics=dict(visits=child.visits, value=child.value,
                                                    exhausted=child.closed))
                               for child in tree.root.children],
        nmr_guidance=(
            {
                'enabled': True,
                'statistics': nmr_scorer.statistics('flex'),
            }
            if nmr_scorer is not None else {'enabled': False}
        ),
    )
    result = SearchResult(current, float(TopologyPlacementStage(current, 'flex').score()),
                          list(selected.actions), {'phase': 'flex', 'result': row})
    result.info['stage_states'] = {'flex_chain': stage_snapshot(current, 'flex_chain')}
    if row['flex_chain_units_placed'] == row['flex_chain_units_total']:
        result.info['stage_states']['aliphatic_ring'] = stage_snapshot(current, 'aliphatic_ring')
    tree._close_subtree(tree.root)
    if solved is None and strict:
        raise AssemblySearchError(message, {**diagnostics, 'partial_molecule': current.to_dict(),
                                            'actions': list(selected.actions)})
    return result, plan


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
    search_budget: int = 2000,
    conformer_limit: int = 12,
    rigid_candidates: Sequence[HexAssemblyState] | None = None,
    search_guidance: Any = None,
    max_tree_nodes: int = 512,
    domain_check_limit: int = 256,
    domain_pose_limit: int = 32,
    max_domain_check_limit: int = 2048,
    max_domain_pose_limit: int = 512,
) -> Tuple[SearchResult, FlexAssemblyPlan]:
    """Connect all prescribed flex units without changing local geometry."""
    # Rebuild planning, domains and component indices for EACH rigid graph.
    # A retained candidate may have completely different site ownership.
    roots = list(rigid_candidates or [state])[:max(1, min(candidate_k, search_budget))]
    attempts = []
    best_result = None
    best_plan = None
    best_rank = None
    total = max(1, int(search_budget))
    for index, source in enumerate(roots):
        allocation = total // len(roots) + int(index < total % len(roots))
        result, plan = _run_csv_site_flex_search(
            source, candidate_k=candidate_k, max_steps=max_steps,
            beam_width=beam_width, strict=False, nmr_scorer=nmr_scorer,
            nmr_weight=nmr_weight, geometry_weight=geometry_weight,
            contact_weight=contact_weight, search_budget=allocation,
            conformer_limit=conformer_limit, c_puct=c_puct,
            seed=None if seed is None else seed + index, rigid_candidates=[source],
            search_guidance=search_guidance, max_tree_nodes=max_tree_nodes,
            domain_check_limit=domain_check_limit, domain_pose_limit=domain_pose_limit,
            max_domain_check_limit=max_domain_check_limit,
            max_domain_pose_limit=max_domain_pose_limit,
        )
        row = result.info['result']
        row['selected_rigid_candidate_index'] = index
        attempts.append(dict(rigid_candidate_index=index, budget=allocation,
                             status=row.get('status'), completed_contract_count=row.get('contracts_done', 0)))
        rank = (bool(row.get('done')), row.get('contracts_done', 0),
                -len(row.get('zero_domain_units', [])))
        if best_rank is None or rank > best_rank:
            best_result, best_plan, best_rank = result, plan, rank
        if row.get('done'):
            break
    best_result.info['result']['rigid_search_attempts'] = attempts
    best_result.info['result']['total_search_budget'] = total
    best_result.info['result']['candidate_space_complete'] = False
    if best_result.info['result'].get('status') == 'proven_infeasible':
        # Exhausting a domain conditional on a retained rigid pose does not
        # exhaust all rigid site bindings or all molecular layouts.
        best_result.info['result']['status'] = 'candidate_space_exhausted'
        best_result.info['result']['failure_scope'] = 'retained_rigid_candidates'
    if strict and not best_result.info['result'].get('done'):
        raise AssemblySearchError('bounded Flex search incomplete', best_result.info['result'])
    return best_result, best_plan
__all__ = [
    "BackboneFlexStep",
    "FlexAssemblyPlan",
    "InteriorFlexStep",
    "build_flex_assembly_plan",
    "audit_flex_feasibility",
    "secondary_flex_displacement_candidates",
    "relabel_aromatic_sites",
    "run_planned_flex_search",
]
