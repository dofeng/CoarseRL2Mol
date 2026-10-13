"""Flexible action geometry over rigid aromatic components.

The inverse CSV owns topology identity and connectivity. Generated tree
conformers may vary on the hex grid, while explicit fixed geometries and
all node/edge identities are preserved and audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from RL_MTCS.geometry import HEX_DIRECTIONS, aromatic_slot_direction, unit_sort_key, axial_add, axial_neg, transform_axial
from RL_MTCS.actions.ordering import ordered
from RL_MTCS.geometry import connection_axis
from RL_MTCS.actions.conformers import iter_conformer_states
from RL_MTCS.mdp.components import choose_anchor
from RL_MTCS.mdp.transition import commit_attachment
from RL_MTCS.reporting.diagnostics import note_unit
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.sites import aromatic_perimeter_ids, hetero_staging_ids
from RL_MTCS.mdp.types import (
    ExternalContract,
    InverseInputError,
    TopologyUnit,
)


def _probe(state):
    components = _rigid_components(state)
    anchor = choose_anchor(state)
    active = next((i for i,g in enumerate(components) if anchor in g), None)
    if active is None:
        return None, components
    used = {n for e in state.edges if e.contract_id for n in (e.source_node_id,e.target_node_id)}
    return _SiteSearchCandidate(state, frozenset(used), {}, frozenset({active}), [], anchor_index=active), components


def _mandatory_aromatic_footprint(state):
    """Sites occupied by EVERY available aromatic attachment, relative to its port.

    Align the outward vector to (1, 0), including both reflection choices.
    Intersect actual unit templates: no assumption about benzene versus fused
    rings is needed. This is a necessary collision test, never an attachment.
    """
    from RL_MTCS.geometry import transform_axial
    active = [uid for uid in state.placements if state.unit(uid).phase == 'aromatic']
    used = {n for e in state.edges if e.contract_id for n in (e.source_node_id, e.target_node_id)}
    footprint = None
    for uid, _, point, outward in _component_slot_records(state, active, used):
        offsets = [(state.node_axial[n.node_id][0]-point[0],
                    state.node_axial[n.node_id][1]-point[1]) for n in state.unit(uid).nodes]
        for reflection in (False, True):
            for rotation in range(6):
                if transform_axial(outward, rotation, reflection) != (1, 0):
                    continue
                occupied = {transform_axial(p, rotation, reflection) for p in offsets}
                footprint = occupied if footprint is None else footprint & occupied
    return footprint or set()


def _attachment_footprints_clear(unit, footprint):
    """Reject only inevitable overlaps of distinct aromatic endpoint units."""
    from RL_MTCS.geometry import transform_axial, axial_add, axial_neg
    from RL_MTCS.actions.conformers import _embedding_phases
    coords = {n.node_id: n.local_axial for n in unit.nodes}
    occupied = set(coords.values())
    mandatory_vertices = set(coords.values())
    for port in unit.ports:
        if port.direction is None:
            continue
        site = axial_add(coords[port.node_id], port.direction)
        rotation = next(r for r in range(6)
                        if transform_axial((1, 0), r, False) == axial_neg(port.direction))
        required = {axial_add(site, transform_axial(p, rotation, False)) for p in footprint}
        if occupied & required:
            return False
        occupied.update(required)
        mandatory_vertices.update(required)
    # Ordinary body sites and the mandatory aromatic footprints must share a
    # global lattice phase. Coarse endpoint bonds alone do not establish that.
    return any(all((q+r+phase) % 3 for q, r in mandatory_vertices)
               for phase in _embedding_phases(unit, coords, [p.direction for p in unit.ports]))


def iter_flex_outcomes(state, unit, merge, *, axis=None, seed=None):
    uid = unit.unit_id
    note_unit(uid)
    if merge:
        probe, components = _probe(state)
        if (probe is None or len(components) == len(probe.active_component_indices)
                or len(_flex_port_groups(state, uid)) < 2):
            return
    # Merge outcomes attach independent ports to distinct aromatic units.
    # Prune impossible conformers BEFORE the component permutation product.
    footprint = _mandatory_aromatic_footprint(state) if merge else set()
    feasible = (lambda variant: _attachment_footprints_clear(variant, footprint)) if footprint else None
    for trial in iter_conformer_states(state, uid, feasible):
        candidate, components = _probe(trial)
        if candidate is None:
            return
        if merge:
            outside = [i for i in range(len(components)) if i not in candidate.active_component_indices]
            n = min(len(outside), len(_flex_port_groups(trial, uid))-1)
            for count in range(n, 0, -1):
                for selected in permutations(ordered(outside, seed, f'components:{uid}'), count):
                    for result in _backbone_site_outcomes(candidate,uid,selected,components,axis=axis,seed=seed):
                        committed = commit_attachment(result,state,'flex')
                        if committed is not None:
                            yield committed
        else:
            for pose in _interior_site_poses(candidate,uid,components,axis=axis,seed=seed):
                result = _place_interior_pose(candidate,uid,pose)
                if result is not None:
                    committed = commit_attachment(result,state,'flex')
                    if committed is not None:
                        yield committed


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


def _aromatic_endpoint(
    state: HexAssemblyState,
    contract: ExternalContract,
) -> Tuple[str, int]:
    source = state.unit(contract.source_unit_id)
    target = state.unit(contract.target_unit_id)
    if source.phase == "aromatic" and target.phase != "aromatic":
        return str(source.unit_id), int(contract.source_node_id)
    if target.phase == "aromatic" and source.phase != "aromatic":
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
    if source.phase != "aromatic" and target.phase == "aromatic":
        return str(source.unit_id), int(contract.source_node_id)
    if target.phase != "aromatic" and source.phase == "aromatic":
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
        if unit.phase == "aromatic" and unit.unit_id in state.placements
    }
    remaining = set(aromatic_ids)
    components: List[Tuple[str, ...]] = []
    while remaining:
        seed = min(remaining, key=unit_sort_key)
        component = tuple(sorted(
            state.component_unit_ids(seed) & aromatic_ids,
            key=unit_sort_key,
        ))
        if not component:
            raise InverseInputError(f"aromatic unit {seed!r} is not placed")
        components.append(component)
        remaining.difference_update(component)
    return tuple(sorted(
        components,
        key=lambda values: tuple(map(unit_sort_key, values)),
    ))


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


@dataclass
class _SiteSearchCandidate:
    state: HexAssemblyState
    used_site_ids: frozenset[int]
    boundary_site_ids: Dict[int, int]
    active_component_indices: frozenset[int]
    actions: List[Dict[str, Any]]
    anchor_index: int = 0


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
    for unit_id in sorted(component, key=unit_sort_key):
        unit = state.unit(unit_id)
        placement = state.placements[unit_id]
        perimeter_ids = aromatic_perimeter_ids(unit)
        for node in unit.nodes:
            node_id = int(node.node_id)
            if (state.current_su_types.get(node_id, node.su_type) != 13
                    or node_id in rigid_ids or node_id in used
                    or f"aromatic_site:{node_id}" in state.used_ports):
                continue
            if node_id not in perimeter_ids:
                continue
            outward = transform_axial(
                aromatic_slot_direction(unit, node_id),
                placement.rotation,
                placement.reflection,
            )
            records.append((
                str(unit_id),
                node_id,
                tuple(state.node_axial[node_id]),
                tuple(outward),
            ))
    return sorted(records, key=lambda row: (unit_sort_key(row[0]), row[1]))


def _flex_port_records(
    state: HexAssemblyState,
    topology_unit_id: str,
) -> List[Tuple[ExternalContract, int, int, Any]]:
    records = []
    for contract in state.problem.contracts:
        if contract.kind not in {"flex", "ring"}:
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
    """Attachment ownership comes from topology semantics, never distance."""
    records = _flex_port_records(state, topology_unit_id)
    unit = state.unit(topology_unit_id)
    if not unit.attaches_to_aromatic_edge:
        return [[record] for record in records]
    return [records] if records else []


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
        if state.unit(contract.source_unit_id).phase != "aromatic"
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
    if direction is None or not state._edge_geometry_valid(
        topology_node_id,
        physical_site_id,
        state.node_axial[topology_node_id],
        state.node_axial[physical_site_id],
        contract.kind,
    ):
        raise InverseInputError(
            f"site contract {contract.contract_id!r} is not a strict honeycomb neighbor"
        )
    port = state._ports[topology_key]
    placement = state.placements[physical_unit_id]
    outward = transform_axial(aromatic_slot_direction(physical_unit, physical_site_id),
                              placement.rotation, placement.reflection)
    if (state.global_port_direction(topology_unit_id, port) != direction
            or outward != axial_neg(direction)):
        raise InverseInputError(f"site contract {contract.contract_id!r} port direction mismatch")
    state._append_edge(
        topology_node_id,
        physical_site_id,
        contract.kind,
        str(contract.contract_id),
        direction,
    )
    state.used_ports.update({topology_key, aromatic_key, physical_key})
    state.current_su_types[physical_site_id] = expected_su_type
    state.satisfied_contracts.add(str(contract.contract_id))


def _backbone_site_outcomes(candidate, topology_unit_id, outside_component_indices,
                            rigid_components, diagnostics=None, *, axis=None, seed=None):
    """Enumerate whole multi-endpoint placements without failed-scan limits."""
    groups = _flex_port_groups(candidate.state, topology_unit_id)
    count = len(groups) - len(outside_component_indices)
    if (count < 1 or not outside_component_indices
            or len(set(outside_component_indices)) != len(outside_component_indices)
            or any(i < 0 or i >= len(rigid_components) for i in outside_component_indices)
            or set(outside_component_indices) & set(candidate.active_component_indices)):
        return
    for selected in combinations(range(len(groups)), count):
        yield from _backbone_site_outcomes_for_anchor(
            candidate, topology_unit_id, outside_component_indices, rigid_components,
            diagnostics, selected, axis=axis, seed=seed)


def _backbone_site_outcomes_for_anchor(
    candidate: _SiteSearchCandidate,
    topology_unit_id: str,
    outside_component_indices: Sequence[int],
    rigid_components: Sequence[Sequence[str]],
    diagnostics: Dict[str, Any] | None = None,
    anchor_indices=None,
    *, axis=None, seed=None,
) -> List[_SiteSearchCandidate]:
    """Attach all independent endpoints transactionally for one anchor choice."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    if unit.attaches_to_aromatic_edge:
        return []
    records = _flex_port_records(state, topology_unit_id)
    groups = _flex_port_groups(state, topology_unit_id)
    if anchor_indices is None and len(groups) != len(outside_component_indices) + 1:
        return []
    # Cross-unit contacts require explicit allowed node pairs.
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
            direction = topology_state.global_port_direction(topology_unit_id, port)
            if direction is None:
                return []
            required = axial_add(topology_state.node_axial[node_id], direction)
            matching = [site_id for site_id in by_geometry.get(
                (required, axial_neg(direction)), [])
                if topology_state._edge_geometry_valid(
                    node_id, site_id, topology_state.node_axial[node_id], required, "flex")]
            if int(boundary_id) in seed:
                if seed[int(boundary_id)] not in matching:
                    return []
                continue
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

    choices = ([(None, [record for index in anchor_indices for record in groups[index]])]
               if anchor_indices is not None else list(enumerate(groups)))
    for anchor_group_index, anchor_group in choices:
        remaining_groups = [
            group for index, group in enumerate(groups)
            if index not in (anchor_indices if anchor_indices is not None else (anchor_group_index,))
        ]
        first_contract, first_node, first_boundary, first_port = anchor_group[0]
        if first_port.direction is None:
            continue
        active_groups = ([tuple(uid for i in sorted(candidate.active_component_indices)
                                for uid in rigid_components[i])]
                         if len(anchor_group) > 1 else
                         [rigid_components[i] for i in sorted(candidate.active_component_indices)])
        for active_units in active_groups:
            anchor_slots = _component_slot_records(
                state,
                active_units,
                candidate.used_site_ids,
            )
            anchor_slots = ordered(anchor_slots, seed, f'flex-slots:{topology_unit_id}')
            for _uid, first_site, first_axial, first_outward in anchor_slots:
                if axis is not None and connection_axis(first_outward) != axis:
                    continue
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
                        if not trial.place_unit(
                            topology_unit_id,
                            rotation,
                            translation,
                            attachment_edges=provisional_contacts,
                            reflection=reflection,
                        ):
                            continue
                        anchor_matches = match_group(
                            anchor_group,
                            anchor_slots,
                            trial,
                            {int(first_boundary): int(first_site)},
                            {int(first_site)},
                        )
                        partial = []
                        for assignments, selected_sites in anchor_matches:
                            owner = state.problem.ownership_map()
                            if len({owner[n] for n in selected_sites}) != len(selected_sites):
                                continue
                            anchored = trial.clone()
                            for row_contract, row_node, row_boundary, _port in anchor_group:
                                _append_site_contract(
                                    anchored,
                                    row_contract,
                                    row_node,
                                    assignments[row_boundary],
                                )
                            partial.append((anchored, assignments, selected_sites, []))

                        def extend(partials, group, component_index):
                            row_contract, row_node, row_boundary, row_port = group[0]
                            if row_port.direction is None:
                                return
                            for partial_state, assignments, selected_sites, moves in partials:
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
                                                moved = partial_state.clone()
                                                if not moved.transform_component(
                                                    representative,
                                                    component_rotation,
                                                    component_translation,
                                                    attachment_edges=contacts,
                                                    reflection=component_reflection,
                                                ):
                                                    continue
                                                for item_contract, item_node, item_boundary, _p in group:
                                                    _append_site_contract(
                                                        moved,
                                                        item_contract,
                                                        item_node,
                                                        group_assignments[item_boundary],
                                                    )
                                                yield (
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
                                                )
                        for group, component_index in zip(remaining_groups, outside_component_indices):
                            partial = extend(partial, group, component_index)

                        for final_state, assignments, selected_sites, moves in partial:
                            if len(assignments) != len(records):
                                continue
                            action = {
                                "kind": "flex",
                                "connection_axis": connection_axis(first_outward),
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
                                "merged_component_count": 1 + len(outside_component_indices),
                                "anchor_port_indices": list(anchor_indices or (anchor_group_index,)),
                                "connection_mode": "merge",
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
                            )
                            yield result


def _endpoint_pose_specs(endpoints, slots, *, axis=None, seed=None):
    """Cache only geometry matches; occupancy is rechecked for every state."""
    by_geometry = {(axial, outward): gid for gid, axial, outward in slots}
    for gid, axial, outward in ordered(slots, seed, 'interior-slots'):
        if axis is not None and connection_axis(outward) != axis:
            continue
        for reflection in (False, True):
            for rotation in range(6):
                local, direction = endpoints[0]
                direction = transform_axial(direction, rotation, reflection)
                if outward != axial_neg(direction):
                    continue
                point = transform_axial(local, rotation, reflection)
                translation = (axial[0] + outward[0] - point[0],
                               axial[1] + outward[1] - point[1])
                selected = [gid]
                for local, direction in endpoints[1:]:
                    direction = transform_axial(direction, rotation, reflection)
                    required = axial_add(axial_add(transform_axial(
                        local, rotation, reflection), translation), direction)
                    match = by_geometry.get((required, axial_neg(direction)))
                    if match is None or match in selected:
                        break
                    selected.append(match)
                if len(selected) == len(endpoints):
                    yield rotation, translation, reflection, tuple(selected)


def _interior_site_poses(
    candidate, topology_unit_id, rigid_components, diagnostics=None,
    *, axis=None, seed=None,
):
    """One source of truth for placement-domain accounting and placement."""
    state = candidate.state
    unit = state.unit(topology_unit_id)
    records = _flex_port_records(state, topology_unit_id)
    if not records or any(row[3].direction is None for row in records):
        return
    active_units = tuple(uid for i in sorted(candidate.active_component_indices)
                         for uid in rigid_components[i])
    slots = _component_slot_records(state, active_units, ())
    nodes = unit.node_map()
    endpoints = tuple((nodes[n].local_axial, p.direction) for _c, n, _b, p in records)
    specs = _endpoint_pose_specs(endpoints, tuple((s[1], s[2], s[3]) for s in slots), axis=axis, seed=seed)
    checks = 0
    for rotation, translation, reflection, sites in specs:
        if candidate.used_site_ids.intersection(sites):
            continue
        if unit.attaches_to_aromatic_edge and not state.aromatic_edge_sites(sites):
            continue
        if not unit.attaches_to_aromatic_edge and len(records) >= 3:
            owners = state.problem.ownership_map()
            if len({owners[gid] for gid in sites}) != len(sites):
                continue
        checks += 1
        assigned = {row[2]: gid for row, gid in zip(records, sites)}
        contacts = [(n, assigned[b]) for _c, n, b, _p in records]
        valid, reason = state.can_place(
            topology_unit_id, rotation, translation, attachment_edges=contacts,
            reflection=reflection,
        )
        if not valid:
            if diagnostics is not None:
                counts = diagnostics.setdefault('rejection_counts', {})
                category = reason.split(':', 1)[0]
                counts[category] = counts.get(category, 0) + 1
            continue
        yield rotation, translation, reflection, assigned, frozenset(sites)
    if diagnostics is not None:
        diagnostics['pose_checks'] = checks


def _place_interior_pose(candidate, topology_unit_id, pose):
    rotation, translation, reflection, assignments, selected = pose
    records = _flex_port_records(candidate.state, topology_unit_id)
    unit = candidate.state.unit(topology_unit_id)
    trial = candidate.state.clone()
    contacts = [(n, assignments[b]) for _c, n, b, _p in records]
    if not trial.place_unit(topology_unit_id, rotation, translation,
                            attachment_edges=contacts, reflection=reflection):
        return None
    for contract, node, boundary, _port in records:
        _append_site_contract(trial, contract, node, assignments[boundary])
    action = dict(
        kind=unit.phase, topology_unit_id=topology_unit_id,
        contract_ids=[r[0].contract_id for r in records],
        topology_rotation=rotation, topology_reflection=reflection,
        topology_translation=list(translation), boundary_site_assignments=dict(assignments),
        connection_axis=connection_axis(trial._edge_direction(
            trial.node_axial[assignments[records[0][2]]], trial.node_axial[records[0][1]])),
        connection_mode='internal',
    )
    return _SiteSearchCandidate(
        state=trial, used_site_ids=candidate.used_site_ids | selected,
        boundary_site_ids={**candidate.boundary_site_ids, **assignments},
        active_component_indices=candidate.active_component_indices,
        actions=candidate.actions + [action], anchor_index=candidate.anchor_index,
    )
