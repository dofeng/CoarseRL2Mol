"""Fixed-topology side-chain placement on a completed flex skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .geometry import (
    aromatic_slot_direction as _aromatic_slot_direction,
    coordinate_aspect,
    state_shape_rank as _state_shape_rank,
    unit_sort_key as _unit_sort_key,
)
from .state import (
    HexAssemblyState,
    axial_add,
    axial_neg,
    transform_axial,
    relabel_aromatic_sites,
)
from .models import (
    ExternalContract,
    HEX_DIRECTIONS,
    InverseInputError,
    InputSourceKind,
    TopologyUnit,
    UnitPort,
)
from .rigid import SearchResult, TopologyPlacementStage, search_phase


def _coordinate_aspect(state: HexAssemblyState) -> Dict[str, Any]:
    """Compatibility adapter around the shared pure geometry helper."""
    return coordinate_aspect(state.occupied_axial)


@dataclass(frozen=True)
class _SideRecord:
    contract: ExternalContract
    topology_unit_id: str
    topology_node_id: int
    boundary_node_id: int
    topology_port: UnitPort


@dataclass
class _SideCandidate:
    state: HexAssemblyState
    used_site_ids: frozenset[int]
    boundary_site_ids: Dict[int, int]
    actions: List[Dict[str, Any]]
    nmr_score: Dict[str, Any] | None = None
    nmr_rank: float | None = None


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
            f"missing={sorted(side_units-seen_units, key=_unit_sort_key)[:12]}"
        )
    return records


def _fixed_aromatic_endpoint_ids(state: HexAssemblyState) -> Set[int]:
    fixed: Set[int] = set()
    for contract in state.problem.contracts:
        if str(contract.contract_id) not in state.satisfied_contracts:
            continue
        for unit_id, node_id in (
            (contract.source_unit_id, contract.source_node_id),
            (contract.target_unit_id, contract.target_node_id),
        ):
            if state.unit(unit_id).phase == "aromatic":
                fixed.add(int(node_id))
    return fixed


def _site_records(
    state: HexAssemblyState,
    available_ids: Iterable[int],
) -> List[Tuple[str, int, Tuple[int, int], Tuple[int, int]]]:
    available = {int(node_id) for node_id in available_ids}
    rows = []
    for unit in state.problem.units:
        if unit.phase != "aromatic" or unit.unit_id not in state.placements:
            continue
        placement = state.placements[unit.unit_id]
        for node in unit.nodes:
            node_id = int(node.node_id)
            if node_id not in available or int(node.su_type) != 13:
                continue
            outward = transform_axial(
                _aromatic_slot_direction(unit, node_id),
                placement.rotation,
                placement.reflection,
            )
            rows.append((
                str(unit.unit_id),
                node_id,
                tuple(state.node_axial[node_id]),
                tuple(outward),
            ))
    return sorted(rows, key=lambda row: (_unit_sort_key(row[0]), row[1]))


def _record_hardness(
    state: HexAssemblyState,
    record: _SideRecord,
) -> Tuple[Any, ...]:
    unit = state.unit(record.topology_unit_id)
    q_values = [int(node.local_axial[0]) for node in unit.nodes]
    r_values = [int(node.local_axial[1]) for node in unit.nodes]
    span = (
        (max(q_values) - min(q_values))
        + (max(r_values) - min(r_values))
    )
    degree: Dict[int, int] = {int(node.node_id): 0 for node in unit.nodes}
    for edge in unit.internal_edges:
        degree[int(edge.source_node_id)] += 1
        degree[int(edge.target_node_id)] += 1
    return (
        -len(unit.nodes),
        -max(degree.values(), default=0),
        -int(span),
        _unit_sort_key(unit.unit_id),
    )


def _pose_specs(
    candidate: _SideCandidate,
    record: _SideRecord,
    site_rows: Sequence[
        Tuple[str, int, Tuple[int, int], Tuple[int, int]]
    ],
) -> List[Tuple[Tuple[Any, ...], int, int, Tuple[int, int], bool]]:
    state = candidate.state
    unit = state.unit(record.topology_unit_id)
    rows = []
    for _aromatic_unit, site_id, site_axial, outward in site_rows:
        if int(site_id) in candidate.used_site_ids:
            continue
        for reflection in (False, True):
            for rotation in range(6):
                port_direction = transform_axial(
                    record.topology_port.direction,
                    rotation,
                    reflection,
                )
                if port_direction != axial_neg(outward):
                    continue
                local = transform_axial(
                    unit.node_map()[record.topology_node_id].local_axial,
                    rotation,
                    reflection,
                )
                desired = axial_add(site_axial, outward)
                translation = (
                    desired[0] - local[0],
                    desired[1] - local[1],
                )
                ok, _reason = state.can_place(
                    record.topology_unit_id,
                    rotation,
                    translation,
                    allowed_contact=(record.topology_node_id, site_id),
                    reflection=reflection,
                    allow_unbonded_neighbors=True,
                )
                if not ok:
                    continue
                coordinates = state._placement_coordinates(
                    unit, rotation, translation, reflection
                )
                allowed_pair = frozenset((
                    int(record.topology_node_id), int(site_id)
                ))
                close_contacts = set()
                for node_id, axial in coordinates.items():
                    for direction in HEX_DIRECTIONS:
                        neighbor_axial = axial_add(axial, direction)
                        occupant = state.occupied_axial.get(neighbor_axial)
                        pair = (
                            frozenset((int(node_id), int(occupant)))
                            if occupant is not None else frozenset()
                        )
                        if (
                            occupant is not None
                            and pair != allowed_pair
                            and state._potential_unintended_contact(
                                int(node_id), axial,
                                int(occupant), neighbor_axial,
                            )
                        ):
                            close_contacts.add(pair)
                q_values = [coord[0] for coord in coordinates.values()]
                r_values = [coord[1] for coord in coordinates.values()]
                rank = (
                    len(close_contacts),
                    max(q_values) - min(q_values)
                    + max(r_values) - min(r_values),
                    int(site_id),
                    int(reflection),
                    int(rotation),
                )
                rows.append((
                    rank,
                    int(site_id),
                    int(rotation),
                    tuple(translation),
                    bool(reflection),
                ))
    return sorted(rows, key=lambda row: row[0])


def _place_outcomes(
    candidate: _SideCandidate,
    record: _SideRecord,
    site_rows: Sequence[
        Tuple[str, int, Tuple[int, int], Tuple[int, int]]
    ],
    limit: int,
) -> Tuple[List[_SideCandidate], int]:
    state = candidate.state
    outcomes = []
    seen = set()
    specs = _pose_specs(candidate, record, site_rows)
    for _rank, site_id, rotation, translation, reflection in specs:
        trial = state.clone()
        if not trial.place_unit(
            record.topology_unit_id,
            rotation,
            translation,
            allowed_contact=(record.topology_node_id, site_id),
            reflection=reflection,
            allow_unbonded_neighbors=True,
        ):
            continue
        direction = trial._edge_direction(
            trial.node_axial[record.topology_node_id],
            trial.node_axial[site_id],
        )
        if direction is None:
            continue
        trial._append_edge(
            record.topology_node_id,
            site_id,
            "side",
            str(record.contract.contract_id),
            direction,
        )
        trial.satisfied_contracts.add(str(record.contract.contract_id))
        action = {
            "kind": "side",
            "contract_id": str(record.contract.contract_id),
            "topology_unit_id": str(record.topology_unit_id),
            "topology_node_id": int(record.topology_node_id),
            "boundary_node_id": int(record.boundary_node_id),
            "physical_site_id": int(site_id),
            "topology_rotation": int(rotation),
            "topology_reflection": bool(reflection),
            "topology_translation": list(translation),
            "fixed_local_geometry": True,
            "allowed_nonbonded_contact_count": len(
                trial.allowed_unbonded_contacts
                - state.allowed_unbonded_contacts
            ),
        }
        result = _SideCandidate(
            state=trial,
            used_site_ids=frozenset(
                set(candidate.used_site_ids) | {int(site_id)}
            ),
            boundary_site_ids={
                **candidate.boundary_site_ids,
                int(record.boundary_node_id): int(site_id),
            },
            actions=candidate.actions + [action],
        )
        signature = (
            result.state.state_signature(),
            tuple(sorted(result.boundary_site_ids.items())),
        )
        if signature in seen:
            continue
        seen.add(signature)
        outcomes.append(result)
        if len(outcomes) >= max(1, int(limit)):
            break
    return outcomes, len(specs)


def _candidate_rank(candidate: _SideCandidate) -> Tuple[Any, ...]:
    nmr_prefix: Tuple[Any, ...] = ()
    if candidate.nmr_rank is not None and candidate.nmr_score is not None:
        nmr_prefix = (
            float(candidate.nmr_rank),
            float(candidate.nmr_score["loss"]),
        )
    return (
        *nmr_prefix,
        len(candidate.state.allowed_unbonded_contacts),
        *_state_shape_rank(candidate.state),
        tuple(sorted(candidate.boundary_site_ids.items())),
    )


def _score_side_candidates(
    candidates: Sequence[_SideCandidate],
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
        stage="side",
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


def run_planned_side_search(
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
) -> SearchResult:
    """Place every prescribed side unit without changing its local topology."""
    if state.problem.source_kind != InputSourceKind.CSV_PAIR:
        return search_phase(
            TopologyPlacementStage(state, "side"),
            iterations=iterations,
            candidate_k=candidate_k,
            max_steps=max_steps,
            beam_width=beam_width,
            c_puct=c_puct,
            seed=seed,
        )
    del iterations, c_puct, seed
    initial = state.clone()
    initial.stage = "side"
    records = _side_records(initial)
    if len(records) > int(max_steps):
        message = (
            f"side_max_steps={max_steps} is smaller than the prescribed "
            f"side topology count {len(records)}"
        )
        if strict:
            raise InverseInputError(message)
    fixed_ids = _fixed_aromatic_endpoint_ids(initial)
    boundary_ids = {int(record.boundary_node_id) for record in records}
    all_site_ids = {
        int(node.node_id)
        for unit in initial.problem.units
        if unit.phase == "aromatic"
        for node in unit.nodes
        if int(node.su_type) == 13 and int(node.node_id) not in fixed_ids
    }
    if len(all_site_ids) < len(records) or not boundary_ids <= all_site_ids:
        raise InverseInputError(
            "remaining aromatic sites cannot host all side boundaries; "
            f"sites={len(all_site_ids)} contracts={len(records)} "
            f"missing={sorted(boundary_ids-all_site_ids)[:12]} "
        )
    site_rows = _site_records(initial, all_site_ids)

    ordered = sorted(records, key=lambda row: _record_hardness(initial, row))
    search_width = (
        max(1, int(beam_width))
        if nmr_scorer is not None
        else max(2, min(12, max(int(beam_width), int(candidate_k))))
    )
    outcome_limit = (
        max(1, int(candidate_k))
        if nmr_scorer is not None
        else max(2, min(8, int(candidate_k)))
    )
    beam = [_SideCandidate(
        state=initial,
        used_site_ids=frozenset(),
        boundary_site_ids={},
        actions=[],
    )]
    explored = 0
    errors: List[str] = []
    for depth, record in enumerate(ordered, start=1):
        next_beam = []
        for candidate in beam:
            outcomes, inspected = _place_outcomes(
                candidate,
                record,
                site_rows,
                outcome_limit,
            )
            explored += int(inspected)
            next_beam.extend(outcomes)
        _score_side_candidates(
            next_beam,
            nmr_scorer=nmr_scorer,
            depth=depth,
            nmr_weight=nmr_weight,
            geometry_weight=geometry_weight,
            contact_weight=contact_weight,
            protected_aromatic_ids=fixed_ids,
        )
        unique: Dict[Tuple[Any, ...], _SideCandidate] = {}
        for candidate in sorted(next_beam, key=_candidate_rank):
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
                "no fixed-geometry side placement for "
                f"{record.topology_unit_id!r} ({record.contract.contract_id})"
            )
            break

    if beam:
        selected = min(beam, key=_candidate_rank)
        current = relabel_aromatic_sites(
            selected.state,
            selected.boundary_site_ids,
            selected.actions,
            protected_aromatic_ids=fixed_ids,
            replay_contract_kinds=("rigid", "flex", "side"),
            stage="side",
        )
        actions = selected.actions
        assignments = selected.boundary_site_ids
    else:
        current = initial
        actions = []
        assignments = {}
        if strict:
            raise InverseInputError(
                errors[-1] if errors else "no geometry-preserving side assembly"
            )

    result_row = TopologyPlacementStage(current, "side").get_result()
    result_row.update({
        "planning_errors": list(errors),
        "explored_geometry_poses": int(explored),
        "fixed_local_geometry": True,
        "resource_allocation_used": False,
        "force_completion_used": False,
        "allowed_nonbonded_contact_count": len(
            current.allowed_unbonded_contacts
        ),
        "coordinate_aspect": _coordinate_aspect(current),
        "slot_geometry": {
            "mode": "search_time_physical_su13_selection",
            "boundary_site_assignments": dict(assignments),
        },
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
                    if beam and selected.nmr_score is not None else None
                ),
                "statistics": nmr_scorer.statistics("side"),
            }
            if nmr_scorer is not None else {"enabled": False}
        ),
    })
    return SearchResult(
        state=current,
        score=float(TopologyPlacementStage(current, "side").score()),
        action_sequence=list(actions),
        info={"phase": "side", "result": result_row},
    )


__all__ = ["run_planned_side_search"]
