"""Fixed-topology side-chain placement on a completed flex skeleton."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .geometry import (
    aromatic_slot_direction as _aromatic_slot_direction,
    axial_distance,
    state_layout_metrics,
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
    return state_layout_metrics(state, full=False, include_void=True)


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
    distribution_rank: Tuple[Any, ...] = ()
    shape_rank: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _SideDistributionContext:
    site_to_unit: Mapping[int, str]
    site_axial: Mapping[int, Tuple[int, int]]
    unit_to_component: Mapping[str, int]
    unit_capacity: Mapping[str, int]
    component_capacity: Mapping[int, int]
    site_bin: Mapping[int, Tuple[int, int]]
    bin_capacity: Mapping[Tuple[int, int], int]


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


def _aromatic_component_map(state: HexAssemblyState) -> Dict[str, int]:
    aromatic_ids = {
        str(unit.unit_id)
        for unit in state.problem.units
        if unit.phase == "aromatic"
    }
    adjacency: Dict[str, Set[str]] = {
        unit_id: set() for unit_id in aromatic_ids
    }
    for contract in state.problem.contracts:
        if contract.kind != "rigid":
            continue
        left = str(contract.source_unit_id)
        right = str(contract.target_unit_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    result: Dict[str, int] = {}
    for component_index, root in enumerate(
        sorted(aromatic_ids, key=_unit_sort_key)
    ):
        if root in result:
            continue
        result[root] = component_index
        queue = [root]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in result:
                    result[neighbor] = component_index
                    queue.append(neighbor)
    # Compress sparse indices caused by roots already assigned by an earlier
    # traversal so component reports remain stable and easy to read.
    remap = {
        value: index for index, value in enumerate(sorted(set(result.values())))
    }
    return {unit_id: remap[index] for unit_id, index in result.items()}


def _side_distribution_context(
    state: HexAssemblyState,
    site_rows: Sequence[Tuple[str, int, Tuple[int, int], Tuple[int, int]]],
) -> _SideDistributionContext:
    site_to_unit = {int(row[1]): str(row[0]) for row in site_rows}
    site_axial = {int(row[1]): tuple(row[2]) for row in site_rows}
    unit_to_component = _aromatic_component_map(state)
    unit_capacity = Counter(site_to_unit.values())
    component_capacity = Counter(
        unit_to_component[unit_id] for unit_id in site_to_unit.values()
    )
    q_values = [axial[0] for axial in site_axial.values()]
    r_values = [axial[1] for axial in site_axial.values()]
    min_q, max_q = min(q_values), max(q_values)
    min_r, max_r = min(r_values), max(r_values)
    q_span = max(1, max_q - min_q)
    r_span = max(1, max_r - min_r)
    site_bin = {
        site_id: (
            min(2, int((axial[0] - min_q) / q_span * 3)),
            min(2, int((axial[1] - min_r) / r_span * 3)),
        )
        for site_id, axial in site_axial.items()
    }
    return _SideDistributionContext(
        site_to_unit=site_to_unit,
        site_axial=site_axial,
        unit_to_component=unit_to_component,
        unit_capacity=dict(unit_capacity),
        component_capacity=dict(component_capacity),
        site_bin=site_bin,
        bin_capacity=dict(Counter(site_bin.values())),
    )


def _side_distribution_metrics(
    assignments: Mapping[int, int],
    context: _SideDistributionContext,
) -> Dict[str, Any]:
    selected = [
        int(site_id) for site_id in assignments.values()
        if int(site_id) in context.site_to_unit
    ]
    total = len(selected)
    if not selected:
        return {
            "energy": 0.0,
            "unit_repeat_fraction": 0.0,
            "max_unit_load": 0,
            "component_load_rmse": 0.0,
            "spatial_load_rmse": 0.0,
            "crowding": 0.0,
            "unit_loads": {},
            "component_loads": {},
        }
    unit_loads = Counter(context.site_to_unit[site_id] for site_id in selected)
    component_loads = Counter(
        context.unit_to_component[context.site_to_unit[site_id]]
        for site_id in selected
    )
    bin_loads = Counter(context.site_bin[site_id] for site_id in selected)
    unit_repeat_fraction = sum(
        max(0, load - 1) ** 2 for load in unit_loads.values()
    ) / total

    def proportional_rmse(
        loads: Mapping[Any, int], capacities: Mapping[Any, int]
    ) -> float:
        capacity_total = max(1, sum(capacities.values()))
        scale = max(1.0, total / max(1, len(capacities)))
        return math.sqrt(sum(
            (
                float(loads.get(key, 0))
                - total * float(capacity) / capacity_total
            ) ** 2
            for key, capacity in capacities.items()
        ) / max(1, len(capacities))) / scale

    component_rmse = proportional_rmse(
        component_loads, context.component_capacity
    )
    spatial_rmse = proportional_rmse(bin_loads, context.bin_capacity)
    pair_penalty = 0.0
    pair_count = 0
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            distance = axial_distance(
                context.site_axial[left], context.site_axial[right]
            )
            pair_penalty += (max(0.0, 6.0 - float(distance)) / 6.0) ** 2
            pair_count += 1
    crowding = pair_penalty / max(1, pair_count)
    max_unit_load = max(unit_loads.values(), default=0)
    energy = (
        6.0 * unit_repeat_fraction
        + 1.5 * max(0, max_unit_load - 1)
        + 3.0 * component_rmse
        + 2.0 * spatial_rmse
        + 1.0 * crowding
    )
    return {
        "energy": float(energy),
        "unit_repeat_fraction": float(unit_repeat_fraction),
        "max_unit_load": int(max_unit_load),
        "component_load_rmse": float(component_rmse),
        "spatial_load_rmse": float(spatial_rmse),
        "crowding": float(crowding),
        "unit_loads": dict(sorted(unit_loads.items(), key=lambda row: _unit_sort_key(row[0]))),
        "component_loads": {
            str(key): int(value) for key, value in sorted(component_loads.items())
        },
    }


def _side_distribution_rank(
    assignments: Mapping[int, int],
    context: _SideDistributionContext,
) -> Tuple[Any, ...]:
    metrics = _side_distribution_metrics(assignments, context)
    return (
        float(metrics["energy"]),
        float(metrics["unit_repeat_fraction"]),
        int(metrics["max_unit_load"]),
        float(metrics["component_load_rmse"]),
        float(metrics["spatial_load_rmse"]),
        float(metrics["crowding"]),
    )


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
    context: _SideDistributionContext,
) -> List[Tuple[Tuple[Any, ...], str, int, int, Tuple[int, int], bool]]:
    state = candidate.state
    unit = state.unit(record.topology_unit_id)
    rows = []
    for aromatic_unit, site_id, site_axial, outward in site_rows:
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
                    *_side_distribution_rank(
                        {
                            **candidate.boundary_site_ids,
                            int(record.boundary_node_id): int(site_id),
                        },
                        context,
                    ),
                    max(q_values) - min(q_values)
                    + max(r_values) - min(r_values),
                    int(site_id),
                    int(reflection),
                    int(rotation),
                )
                rows.append((
                    rank,
                    str(aromatic_unit),
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
    context: _SideDistributionContext,
) -> Tuple[List[_SideCandidate], int]:
    state = candidate.state
    outcomes = []
    seen = set()
    specs = _pose_specs(candidate, record, site_rows, context)
    # Round-robin across aromatic units before the expensive state clones.
    # This prevents low site IDs from consuming the entire pose pool.
    grouped: Dict[str, List[Tuple[Any, ...]]] = defaultdict(list)
    for spec in specs:
        grouped[str(spec[1])].append(spec)
    groups = [
        grouped[unit_id]
        for unit_id in sorted(
            grouped,
            key=lambda value: (grouped[value][0][0], _unit_sort_key(value)),
        )
    ]
    pose_limit = max(64, 4 * max(1, int(limit)))
    diverse_specs = []
    round_index = 0
    while groups and len(diverse_specs) < pose_limit:
        next_groups = []
        for group in groups:
            if round_index < len(group):
                diverse_specs.append(group[round_index])
                next_groups.append(group)
                if len(diverse_specs) >= pose_limit:
                    break
        groups = next_groups
        round_index += 1
    for _rank, aromatic_unit, site_id, rotation, translation, reflection in diverse_specs:
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
            "aromatic_unit_id": str(aromatic_unit),
            "rigid_component_index": int(
                context.unit_to_component[str(aromatic_unit)]
            ),
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
            distribution_rank=_side_distribution_rank(
                {
                    **candidate.boundary_site_ids,
                    int(record.boundary_node_id): int(site_id),
                },
                context,
            ),
        )
        signature = (
            result.state.state_signature(),
            tuple(sorted(result.boundary_site_ids.items())),
        )
        if signature in seen:
            continue
        seen.add(signature)
        outcomes.append(result)
    # Distribution and local-contact terms are cheap and remove most poses
    # before the global void calculation.  The retained set still spans twice
    # the requested output width.
    outcomes.sort(key=lambda candidate: (
        candidate.distribution_rank,
        len(candidate.state.allowed_unbonded_contacts),
        tuple(sorted(candidate.boundary_site_ids.items())),
    ))
    outcomes = outcomes[:max(24, 2 * max(1, int(limit)))]
    outcomes.sort(key=_candidate_rank)
    return outcomes[: max(1, int(limit))], len(diverse_specs)


def _candidate_rank(candidate: _SideCandidate) -> Tuple[Any, ...]:
    nmr_prefix: Tuple[Any, ...] = ()
    if candidate.nmr_rank is not None and candidate.nmr_score is not None:
        nmr_prefix = (
            float(candidate.nmr_rank),
            float(candidate.nmr_score["loss"]),
        )
    if not candidate.shape_rank:
        candidate.shape_rank = _state_shape_rank(candidate.state)
    shape_rank = candidate.shape_rank
    distribution_rank = candidate.distribution_rank or (0.0,) * 6
    combined_rank = (
        0.08 * float(shape_rank[0])
        + 0.60 * float(distribution_rank[0])
    )
    if nmr_prefix:
        combined_rank += float(nmr_prefix[0])
    nmr_tail = nmr_prefix[1:] if nmr_prefix else ()
    return (
        combined_rank,
        *nmr_tail,
        *distribution_rank,
        len(candidate.state.allowed_unbonded_contacts),
        *shape_rank[1:],
        tuple(sorted(candidate.boundary_site_ids.items())),
    )


def _bounded_nmr_scoring_pool(
    candidates: Sequence[_SideCandidate],
    limit: int,
) -> List[_SideCandidate]:
    """Deduplicate and distribution-prefilter one expensive NMR layer."""
    unique: Dict[Tuple[Any, ...], _SideCandidate] = {}
    for candidate in sorted(candidates, key=_candidate_rank):
        signature = (
            candidate.state.state_signature(),
            tuple(sorted(candidate.boundary_site_ids.items())),
        )
        unique.setdefault(signature, candidate)
        if len(unique) >= max(1, int(limit)):
            break
    return list(unique.values())


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
            + 0.05 * float(shape.get("layout_energy", 0.0))
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
    distribution_context = _side_distribution_context(initial, site_rows)

    ordered = sorted(records, key=lambda row: _record_hardness(initial, row))
    search_width = max(
        8, min(16, max(2 * int(beam_width), int(candidate_k)))
    )
    outcome_limit = max(
        12, min(24, max(int(candidate_k), 2 * int(beam_width)))
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
                distribution_context,
            )
            explored += int(inspected)
            next_beam.extend(outcomes)
        if nmr_scorer is not None:
            next_beam = _bounded_nmr_scoring_pool(
                next_beam, max(8, search_width)
            )
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
        selected = min(
            beam,
            key=lambda candidate: (
                float(_candidate_rank(candidate)[0])
                + 0.05 * float(
                    state_layout_metrics(candidate.state, full=True)["layout_energy"]
                ),
                _candidate_rank(candidate),
            ),
        )
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
        "side_distribution": _side_distribution_metrics(
            assignments, distribution_context
        ),
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
