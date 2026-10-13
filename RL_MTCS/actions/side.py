"""Side-chain topology classification and attachment actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from RL_MTCS.geometry import unit_sort_key as unit_sort_key
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import ExternalContract, InverseInputError, TopologyUnit, UnitPort
from RL_MTCS.actions.conformers import iter_attached_conformers
from RL_MTCS.actions.ordering import ordered, round_robin
from RL_MTCS.geometry import hull_region, outside_hull, axial_add
from RL_MTCS.mdp.components import main_units
from RL_MTCS.mdp.transition import commit_attachment
from RL_MTCS.reporting.diagnostics import note_unit, note_rejection


def iter_side_outcomes(state, unit, *, region=None, seed=None, blocked_units=None):
    note_unit(unit.unit_id)
    from RL_MTCS.actions.flex import _component_slot_records
    records = [r for r in _side_records(state) if r.topology_unit_id == unit.unit_id]
    if not records:
        return
    record = records[0]
    complex_side = is_complex_side(unit)
    active = [u for u in main_units(state) if state.unit(u).phase == 'aromatic']
    slots = _component_slot_records(state,active,())
    # Preference passes must exhaust ALL conformers before falling back.
    preference = ((region,) if region is not None else
                  ('outside','crossing','inside') if complex_side else ('inside','crossing','outside'))
    for region in preference:
        feasible_slots = []
        for slot in slots:
            _owner, site, coord, outward = slot
            desired = axial_add(coord, outward)
            if ((region == 'outside' and not outside_hull(state.backbone_hull, desired))
                    or (region == 'inside' and outside_hull(state.backbone_hull, desired))):
                continue
            if desired in state.occupied_axial or not state._node_site_valid(record.topology_node_id, desired):
                if blocked_units is not None and desired in state.occupied_axial:
                    blocked_units.update(state._blocking_units({record.topology_node_id:desired}))
                note_rejection('blocked_attachment_entry', unit.unit_id)
                continue
            feasible_slots.append(slot)
        if not feasible_slots:
            continue
        slots_in_order = ordered(feasible_slots, seed, f'side-slots:{unit.unit_id}')
        source = state.conformer_sources.get(unit.unit_id, unit)
        yield from round_robin((_side_at_slot(state, source, record, slot, region, blocked_units)
                                for slot in slots_in_order))


def _side_pose_key(unit, coordinates):
    """Equivalent occupied geometry must not consume another candidate slot."""
    centers = {child for _, child in unit.su25_center_branches}
    vertices = tuple(sorted((coordinates[n.node_id], n.su_type, n.metadata,
                             n.node_id in centers) for n in unit.nodes))
    edges = tuple(sorted((tuple(sorted((coordinates[e.source_node_id],
                                        coordinates[e.target_node_id]))), e.kind)
                         for e in unit.internal_edges))
    ports = tuple(sorted((coordinates[p.node_id], p.role, p.phase) for p in unit.ports))
    return vertices, edges, ports


def _side_at_slot(state, unit, record, slot, region, blocked_units=None):
    from RL_MTCS.actions.flex import _append_site_contract, _SiteSearchCandidate
    _owner, site, coord, outward = slot
    position = axial_add(coord, outward)
    seen = set()
    for variant, rotation, translation, reflection in iter_attached_conformers(
            state, unit, record.topology_port, position, site, blocked_units=blocked_units):
        coordinates = state._placement_coordinates(variant, rotation, translation, reflection)
        if hull_region(state.backbone_hull, coordinates.values()) != region:
            continue
        key = _side_pose_key(variant, coordinates)
        if key in seen:
            continue
        child = state.clone()
        child.select_conformer(variant)
        note_unit(unit.unit_id)
        if not child.place_unit(unit.unit_id, rotation, translation,
                                attachment_edge=(record.topology_node_id, site), reflection=reflection):
            if blocked_units is not None:
                errors = state._proposed_geometry_errors(coordinates, variant.internal_edges,
                    (frozenset((record.topology_node_id, site)),))
                blocked_units.update(state._blocking_units(coordinates, errors))
            continue
        _append_site_contract(child, record.contract, record.topology_node_id, site)
        boundary = (record.contract.source_node_id if child.unit(record.contract.source_unit_id).phase == 'aromatic'
                    else record.contract.target_node_id)
        action = dict(kind='side', topology_unit_id=unit.unit_id,
                      contract_ids=[record.contract.contract_id],
                      topology_rotation=rotation, topology_reflection=reflection,
                      topology_translation=list(translation),
                      boundary_site_assignments={boundary: site})
        result = _SiteSearchCandidate(child, frozenset({site}), {boundary: site}, frozenset(), [action])
        committed = commit_attachment(result, state, 'side')
        if committed is not None:
            seen.add(key)
            yield committed


def iter_side_repair_outcomes(state, unit, blockers, *, region=None, seed=None):
    """One fixed-anchor reshape plus a new attachment, committed together."""
    records = {r.topology_unit_id: r for r in _side_records(state)}
    owner = state.problem.ownership_map()
    for uid in ordered(sorted(blockers, key=unit_sort_key), seed, f'side-repair:{unit.unit_id}'):
        if uid not in state.placements or state.unit(uid).phase != 'side':
            continue
        record = records[uid]
        contract = record.contract
        site = (contract.source_node_id if owner[contract.source_node_id] != uid
                else contract.target_node_id)
        source = state.conformer_sources.get(uid, state.unit(uid))
        port = next(p for p in source.ports if p.port_id == record.topology_port.port_id)
        root_position = state.node_axial[record.topology_node_id]
        space = state.clone()
        ids = set(source.node_map())
        for node in ids:
            space.occupied_axial.pop(space.node_axial.pop(node))
            space.final_node_positions.pop(node)
        space.placements.pop(uid)
        space.edges = [e for e in space.edges if not ids.intersection((e.source_node_id, e.target_node_id))]
        for variant, rotation, translation, reflection in iter_attached_conformers(
                space, source, port, root_position, site):
            coordinates = state._placement_coordinates(variant, rotation, translation, reflection)
            if all(state.node_axial[n] == p for n, p in coordinates.items()):
                continue
            reshaped = state.clone()
            if not reshaped.reshape_side(variant, rotation, translation, reflection):
                continue
            for child in iter_side_outcomes(reshaped, unit, region=region, seed=seed):
                child.action_history[-1]['side_replacements'] = [dict(
                    topology_unit_id=uid, rotation=rotation, translation=list(translation),
                    reflection=reflection, node_ids=sorted(ids))]
                yield child


@dataclass(frozen=True)
class _SideRecord:
    contract: ExternalContract
    topology_unit_id: str
    topology_node_id: int
    boundary_node_id: int
    topology_port: UnitPort


def is_complex_side(unit: TopologyUnit) -> bool:
    """Classify by topology, independently of its selected conformer."""
    degree: Dict[int, int] = {int(node.node_id): 0 for node in unit.nodes}
    for edge in unit.internal_edges:
        degree[int(edge.source_node_id)] += 1
        degree[int(edge.target_node_id)] += 1
    return len(unit.nodes) >= 3 or max(degree.values(), default=0) >= 3


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
        f"side contract {contract.contract_id!r} must have one aromatic endpoint"
    )


def _topology_endpoint(
    state: HexAssemblyState,
    contract: ExternalContract,
) -> Tuple[str, int, str]:
    source = state.unit(contract.source_unit_id)
    target = state.unit(contract.target_unit_id)
    if source.phase == "side" and target.phase == "aromatic":
        return (
            str(source.unit_id),
            int(contract.source_node_id),
            str(contract.source_port_id),
        )
    if target.phase == "side" and source.phase == "aromatic":
        return (
            str(target.unit_id),
            int(contract.target_node_id),
            str(contract.target_port_id),
        )
    raise InverseInputError(
        f"side contract {contract.contract_id!r} must have one side endpoint"
    )


def _port_for_record(
    unit: TopologyUnit,
    node_id: int,
    preferred_port_id: str,
) -> UnitPort:
    port = next(
        (
            item for item in unit.ports
            if int(item.node_id) == int(node_id)
            and str(item.port_id) == str(preferred_port_id)
        ),
        None,
    )
    if port is None:
        port = next(
            (item for item in unit.ports if int(item.node_id) == int(node_id)),
            None,
        )
    if port is None or port.direction is None:
        raise InverseInputError(
            f"side topology {unit.unit_id!r} has no directed port at GID {node_id}"
        )
    return port


def _side_records(state: HexAssemblyState) -> List[_SideRecord]:
    records = []
    seen_units: Set[str] = set()
    for contract in state.problem.contracts:
        if contract.kind != "side":
            continue
        _aromatic_unit_id, boundary_id = _aromatic_endpoint(state, contract)
        topology_id, topology_node_id, port_id = _topology_endpoint(
            state, contract
        )
        if topology_id in seen_units:
            raise InverseInputError(
                f"side topology {topology_id!r} has more than one external contract"
            )
        seen_units.add(topology_id)
        unit = state.unit(topology_id)
        records.append(_SideRecord(
            contract=contract,
            topology_unit_id=topology_id,
            topology_node_id=int(topology_node_id),
            boundary_node_id=int(boundary_id),
            topology_port=_port_for_record(unit, topology_node_id, port_id),
        ))
    side_units = {unit.unit_id for unit in state.problem.units if unit.phase == "side"}
    if seen_units != side_units:
        raise InverseInputError(
            "side topology/contract mismatch; "
            f"missing={sorted(side_units-seen_units, key=unit_sort_key)[:12]}"
        )
    return records
